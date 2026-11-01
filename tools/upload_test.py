import argparse
import hashlib
import sys
import time
from pathlib import Path

import boto3

from db.database import create_workspace, get_document_status, list_workspaces, register_document

SESSION = "aws-test"


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a document and upload it to S3, like the API will do later")
    ap.add_argument("file")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    region = boto3.session.Session().region_name
    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = args.bucket or f"agentic-rag-uploads-{account}-{region}"

    existing = list_workspaces(SESSION)
    ws = existing[0] if existing else create_workspace(SESSION, "AWS test")
    ws_id = str(ws["workspace_id"])

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reg = register_document(ws_id, path.name, digest, force=True)
    key = f"uploads/{ws_id}/{reg['doc_id']}/{path.name}"
    print("workspace:", ws_id)
    print("document: ", reg["doc_id"])
    print(f"uploading to s3://{bucket}/{key}")
    boto3.client("s3").upload_file(str(path), bucket, key)

    started, last = time.time(), None
    while time.time() - started < args.timeout:
        st = get_document_status(reg["doc_id"], ws_id)
        steps = ", ".join(f"{k}:{v['status']}" for k, v in sorted((st["branches"] or {}).items()))
        line = f"{st['status']} / {st['stage']} / {steps}"
        if line != last:
            print(f"[{time.time() - started:6.1f}s] {line}")
            last = line
        if st["status"] in ("done", "failed"):
            if st["error"]:
                print("error:", st["error"])
            return
        time.sleep(5)
    print("Gave up waiting. Check the logs (see the test step).")


if __name__ == "__main__":
    main()