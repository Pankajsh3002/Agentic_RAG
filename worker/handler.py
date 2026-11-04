import hashlib
import logging
import os
import shutil
import tempfile
import threading
import urllib.parse

import boto3

from aws_secrets import load_secrets

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "60"))
WATCHDOG_MARGIN_S = 45          # report a timeout this long before Lambda would kill the run

_s3 = None


def _background_warmup():
    try:
        from Data_Ingestion.PDF_Markdown.Parse_PDF import _get_converter
        _get_converter()
    except Exception as e:
        logger.warning("Background converter pre-warmup failed: %s", e)


# Trigger pre-warmup during Lambda container init so model weights load into RAM before events arrive
threading.Thread(target=_background_warmup, daemon=True).start()


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


def _pdf_pages(path: str):
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        logger.exception("Could not count the pages of %s", path)
        return None


def _start_watchdog(doc_id: str, seconds: float) -> threading.Timer:
    """If the run is about to hit Lambda's time limit, tell the visitor instead of leaving them waiting."""
    def fire():
        from db.database import close_conn, set_failed
        try:
            set_failed(doc_id, "timeout",
                       "This file took too long to process (the limit is 15 minutes). Try a smaller file.",
                       "watchdog")
        except Exception:
            logger.exception("Watchdog could not record the timeout")
        finally:
            close_conn()

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def _process_object(bucket: str, key: str, remaining_ms: int) -> None:
    from db.database import (
        close_conn, get_upload_target, set_failed, set_stage, set_status,
    )

    parts = key.split("/")                      # uploads/<workspace_id>/<doc_id>/<filename>
    if len(parts) != 4 or parts[0] != "uploads":
        logger.warning("Ignoring unexpected object key: %s", key)
        return
    _, workspace_id, doc_id, filename = parts

    s3 = _client()
    work_dir = tempfile.mkdtemp(dir="/tmp")
    watchdog = None
    try:
        load_secrets()
        doc = get_upload_target(doc_id)
        if not doc or str(doc["workspace_id"]) != workspace_id:
            logger.warning("No matching document for %s (workspace deleted?). Removing the upload.", key)
            s3.delete_object(Bucket=bucket, Key=key)
            return
        if doc["status"] == "done":              # S3 can deliver an event twice
            s3.delete_object(Bucket=bucket, Key=key)
            return

        # Immediately update status so UI reflects processing without delay
        set_status(doc_id, "processing")
        set_stage(doc_id, "downloading")

        local_path = os.path.join(work_dir, filename)
        s3.download_file(bucket, key, local_path)

        if doc["content_hash"] and _sha256(local_path) != doc["content_hash"]:
            set_failed(doc_id, "download", "The uploaded file did not match its checksum.", "hash mismatch")
            return

        if filename.lower().endswith(".pdf"):
            pages = _pdf_pages(local_path)
            if pages and pages > MAX_PDF_PAGES:
                set_failed(doc_id, "validating",
                           f"This PDF has {pages} pages. The demo accepts up to {MAX_PDF_PAGES} pages.",
                           f"pages={pages}")
                s3.delete_object(Bucket=bucket, Key=key)
                return

        watchdog = _start_watchdog(doc_id, max(30.0, remaining_ms / 1000 - WATCHDOG_MARGIN_S))

        # Import heavy ML pipeline only after DB connection & download are validated
        from Pipeline.process_document import process_document
        ok = process_document(local_path, doc_id, max_workers=int(os.getenv("PARSE_WORKERS", "15")))
        if ok:
            s3.delete_object(Bucket=bucket, Key=key)   # keep the file if it failed, so it can be inspected
    except Exception as e:
        logger.exception("Worker failed for %s", key)
        try:
            set_failed(doc_id, "worker", str(e)[:2000], repr(e))
        except Exception:
            logger.exception("Could not record the failure")
    finally:
        if watchdog:
            watchdog.cancel()
        shutil.rmtree(work_dir, ignore_errors=True)
        close_conn()                             # a frozen container must not keep a stale connection


def lambda_handler(event, context):
    records = event.get("Records", [])
    for record in records:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        _process_object(bucket, key, context.get_remaining_time_in_millis())
    return {"processed": len(records)}