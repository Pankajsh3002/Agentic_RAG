import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Data_Ingestion.Markdown_Router import route_document
from Pipeline.branches import BRANCHES, ESSENTIAL
from db.database import (
    clear_branches, close_conn, failed_branches, get_pipeline_markdown, mark_done,
    save_markdown, set_branch, set_failed, set_stage, set_status,
)

logger = logging.getLogger(__name__)

# Docling uses a lot of memory, so only this many documents are parsed at the same time.
# The model-calling branches are not limited by this.
_PARSE_SLOTS = threading.Semaphore(int(os.getenv("PARSE_CONCURRENCY", "1")))


def _run_branch(name: str, doc_id: str, markdown: str) -> tuple[str, str | None]:
    """Run one branch in its own thread. Returns (branch name, error message or None)."""
    started = time.monotonic()
    try:
        set_branch(doc_id, name, "running")
        BRANCHES[name](doc_id, markdown)
        set_branch(doc_id, name, "done")
        return name, None
    except Exception as e:
        logger.exception("doc_id=%s branch '%s' failed", doc_id, name)
        set_branch(doc_id, name, "failed", str(e)[:2000], traceback.format_exc()[-8000:])
        return name, str(e)
    finally:
        logger.info("doc_id=%s branch '%s' took %.1f s", doc_id, name, time.monotonic() - started)
        close_conn()   # each worker thread has its own connection


def process_document(local_path: str, doc_id: str, max_workers: int = 5) -> bool:
    extracted_dir = tempfile.mkdtemp(prefix="extracted_")
    stage = "starting"
    try:
        set_status(doc_id, "processing")

        stage = "queued"
        set_stage(doc_id, stage)
        with _PARSE_SLOTS:                       # waits here if another document is being parsed
            stage = "parsing"
            set_stage(doc_id, stage)
            markdown = route_document(local_path, extracted_dir=extracted_dir, max_workers=max_workers)

        # Some parsers resolve relative links against the temp upload folder; strip that prefix
        base = str(Path(local_path).resolve().parent)
        if markdown and len(base) > 1:
            markdown = markdown.replace(base + "/", "")

        stage = "validating"
        set_stage(doc_id, stage)
        if not markdown or not markdown.strip():
            raise ValueError("Router returned empty markdown")

        stage = "saving"
        set_stage(doc_id, stage)
        save_markdown(doc_id, markdown, finish=False)   # store markdown, do NOT mark done yet

        stage = "analyzing"
        set_stage(doc_id, stage)
        clear_branches(doc_id)
        for name in BRANCHES:
            set_branch(doc_id, name, "pending")

        # All branches run at the same time, each in its own thread
        branches_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(BRANCHES)) as pool:
            results = list(pool.map(lambda n: _run_branch(n, doc_id, markdown), BRANCHES))
        logger.info("doc_id=%s all branches finished in %.1f s", doc_id, time.monotonic() - branches_started)

        failed = [name for name, err in results if err]
        essential_failed = [n for n in failed if n in ESSENTIAL]
        if essential_failed:
            raise RuntimeError("Required step(s) failed: " + ", ".join(essential_failed))
        if failed:
            logger.warning("doc_id=%s finished with optional step(s) failed: %s", doc_id, failed)

        mark_done(doc_id)
        return True
    except Exception as e:
        logger.exception("doc_id=%s failed at stage '%s'", doc_id, stage)
        set_failed(doc_id, stage, str(e)[:2000], traceback.format_exc()[-8000:])
        return False
    finally:
        shutil.rmtree(extracted_dir, ignore_errors=True)


def retry_failed_branches(doc_id: str) -> bool:
    """
    Re-run only the failed branches, using the markdown already stored.
    The caller must have claimed the document with begin_retry() first.
    """
    stage = "retrying"
    try:
        markdown = get_pipeline_markdown(doc_id)
        names = [n for n in failed_branches(doc_id) if n in BRANCHES]
        if not markdown or not names:
            raise ValueError("Nothing to retry")

        for name in names:
            set_branch(doc_id, name, "pending")
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            results = list(pool.map(lambda n: _run_branch(n, doc_id, markdown), names))

        essential_failed = [n for n, err in results if err and n in ESSENTIAL]
        if essential_failed:
            raise RuntimeError("Required step(s) failed: " + ", ".join(essential_failed))

        mark_done(doc_id)
        return True
    except Exception as e:
        logger.exception("doc_id=%s retry failed", doc_id)
        set_failed(doc_id, stage, str(e)[:2000], traceback.format_exc()[-8000:])
        return False


if __name__ == "__main__":
    # Debug run: python3 -m Pipeline.process_document <path-to-file>
    import hashlib
    import sys
    from db.database import create_workspace, get_conn, list_workspaces, register_document

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1]

    session = "debug-cli"
    existing = list_workspaces(session)
    ws_id = str((existing[0] if existing else create_workspace(session, "Debug"))["workspace_id"])

    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    reg = register_document(ws_id, path.split("/")[-1], digest, force=True)
    ok = process_document(path, reg["doc_id"])

    conn = get_conn()
    doc = conn.execute(
        "SELECT status, stage, left(error, 200) AS error FROM documents WHERE doc_id = %s",
        (reg["doc_id"],),
    ).fetchone()
    branches = conn.execute(
        "SELECT branch, status, left(error, 100) AS error FROM document_branches "
        "WHERE doc_id = %s ORDER BY branch",
        (reg["doc_id"],),
    ).fetchall()
    print("workspace:", ws_id)
    print("success:", ok, "|", doc)
    for b in branches:
        print("  branch:", b)