import hashlib
import logging
import os
import shutil
import tempfile
import urllib.parse

import boto3

from aws_secrets import load_secrets

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_s3 = None


def _client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _process_object(bucket: str, key: str) -> None:
    # Imported here, not at the top: these imports are slow and heavy
    from Pipeline.process_document import process_document
    from db.database import close_conn, get_upload_target, set_failed

    parts = key.split("/")                      # uploads/<workspace_id>/<doc_id>/<filename>
    if len(parts) != 4 or parts[0] != "uploads":
        logger.warning("Ignoring unexpected object key: %s", key)
        return
    _, workspace_id, doc_id, filename = parts

    s3 = _client()
    work_dir = tempfile.mkdtemp(dir="/tmp")
    try:
        doc = get_upload_target(doc_id)
        if not doc or str(doc["workspace_id"]) != workspace_id:
            logger.warning("No matching document for %s (workspace deleted?). Removing the upload.", key)
            s3.delete_object(Bucket=bucket, Key=key)
            return
        if doc["status"] == "done":              # S3 can deliver an event twice
            s3.delete_object(Bucket=bucket, Key=key)
            return

        local_path = os.path.join(work_dir, filename)
        s3.download_file(bucket, key, local_path)

        if doc["content_hash"] and _sha256(local_path) != doc["content_hash"]:
            set_failed(doc_id, "download", "The uploaded file did not match its checksum.", "hash mismatch")
            return

        ok = process_document(local_path, doc_id, max_workers=int(os.getenv("PARSE_WORKERS", "2")))
        if ok:
            s3.delete_object(Bucket=bucket, Key=key)   # keep the file if it failed, so it can be inspected
    except Exception as e:
        logger.exception("Worker failed for %s", key)
        try:
            set_failed(doc_id, "worker", str(e)[:2000], repr(e))
        except Exception:
            logger.exception("Could not record the failure")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        close_conn()                             # a frozen container must not keep a stale connection


def lambda_handler(event, context):
    load_secrets()
    records = event.get("Records", [])
    for record in records:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        _process_object(bucket, key)
    return {"processed": len(records)}