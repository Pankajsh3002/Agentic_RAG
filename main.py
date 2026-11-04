import asyncio
import hashlib
import logging
import os
import secrets
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import date
import time

from collections import defaultdict, deque

from Agent.agent import MAX_QUESTION_CHARS, answer_question

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db.database import (
    abort_pending_upload, begin_retry, create_workspace, delete_expired_workspaces, delete_workspace,
    fail_interrupted_documents, get_document_markdown, get_document_status, get_summary, get_workspace,
    list_documents, list_entities, list_events, list_relations, list_summaries,
    list_workspaces, register_document, set_failed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".png", ".jpeg"}
HTML_EXTENSIONS = {".html", ".htm"}
MAIL_EXTENSIONS = {".eml", ".msg", ".xml"}
TABULAR_EXTENSIONS = {".xlsx", ".xls", ".csv", ".tsv"}
DOCX_EXTENSIONS = {".docx", ".doc"}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = (
    IMAGE_EXTENSIONS | HTML_EXTENSIONS | MAIL_EXTENSIONS
    | TABULAR_EXTENSIONS | DOCX_EXTENSIONS | PDF_EXTENSIONS
)
MAX_UPLOAD_MB = 50
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "")     # set: browsers upload straight to S3
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
WORKSPACE_TTL_HOURS = int(os.getenv("WORKSPACE_TTL_HOURS", "24"))
SESSION_COOKIE = "rag_session"
SESSION_MAX_AGE = 7 * 24 * 3600
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"   # set to 1 when served over HTTPS

# Docling is heavy, so allow at most 2 documents to be parsed at once locally.
_PARSE_SLOTS = threading.Semaphore(int(os.getenv("PARSE_CONCURRENCY", "1")))


# ---------- background cleanup of expired workspaces ----------
async def _expiry_loop():
    while True:
        try:
            n = await asyncio.to_thread(delete_expired_workspaces, WORKSPACE_TTL_HOURS)
            if n:
                logger.info("Deleted %d expired workspace(s)", n)
        except Exception:
            logger.exception("Expiry cleanup failed")
        await asyncio.sleep(3600)


def _preload_local_parser():
    """In local mode, warm up the parser in the background so uploads parse without delay."""
    try:
        from Pipeline.process_document import process_document  # noqa: F401
        logger.info("Local document parser warmed up in background thread.")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    try:
        from aws_secrets import load_secrets
        load_secrets()
    except Exception:
        pass

    if not UPLOAD_BUCKET:    # with direct uploads, the cloud worker owns running jobs, not this server
        try:
            n = await asyncio.to_thread(fail_interrupted_documents)
            if n:
                logger.warning("Marked %d interrupted document(s) as failed", n)
        except Exception:
            logger.exception("Could not check for interrupted documents")
        threading.Thread(target=_preload_local_parser, daemon=True).start()

    task = asyncio.create_task(_expiry_loop())
    yield
    task.cancel()


app = FastAPI(title="Agentic RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ---------- database connection auto-cleanup (allows Aurora to auto-pause) ----------
@app.middleware("http")
async def db_cleanup_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    finally:
        from db.database import close_conn
        close_conn()


# ---------- anonymous session cookie ----------
@app.middleware("http")
async def session_cookie(request: Request, call_next):
    sid = request.cookies.get(SESSION_COOKIE)
    is_new = not sid or len(sid) < 20
    if is_new:
        sid = secrets.token_urlsafe(32)
    request.state.session_id = sid

    response = await call_next(request)
    if is_new:
        response.set_cookie(
            SESSION_COOKIE, sid, max_age=SESSION_MAX_AGE,
            httponly=True, samesite="lax", secure=COOKIE_SECURE,
        )
    return response


def get_session_id(request: Request) -> str:
    return request.state.session_id


def owned_workspace(workspace_id: uuid.UUID, sid: str = Depends(get_session_id)) -> str:
    """Resolve the workspace from the URL, only if it belongs to the caller's session."""
    if not get_workspace(str(workspace_id), sid):
        raise HTTPException(404, "Workspace not found")
    return str(workspace_id)


# ---------- pipeline job ----------
def _run_pipeline(local_path: str, upload_dir: str, doc_id: str) -> None:
    """Background job: parse, store, then delete the upload only on success."""
    from Pipeline.process_document import process_document
    ok = process_document(local_path, doc_id)
    if ok:
        shutil.rmtree(upload_dir, ignore_errors=True)
    else:
        logger.warning("Kept failed upload for debugging: %s", local_path)


# ---------- pages ----------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ---------- workspaces ----------
class WorkspaceIn(BaseModel):
    name: str = "My workspace"


@app.post("/workspaces", status_code=201)
def create_ws(body: WorkspaceIn, sid: str = Depends(get_session_id)):
    try:
        return create_workspace(sid, body.name)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/workspaces")
def list_ws(sid: str = Depends(get_session_id)):
    rows = list_workspaces(sid)
    if not rows:
        try:
            create_workspace(sid, "Default Workspace")
            rows = list_workspaces(sid)
        except Exception as e:
            logger.warning("Could not auto-create default workspace: %s", e)
    return rows


@app.delete("/workspaces/{workspace_id}")
def delete_ws(workspace_id: uuid.UUID, sid: str = Depends(get_session_id)):
    if not delete_workspace(str(workspace_id), sid):
        raise HTTPException(404, "Workspace not found")
    return {"deleted": True}


# ---------- documents inside a workspace ----------
@app.get("/workspaces/{workspace_id}/documents")
def list_docs(ws_id: str = Depends(owned_workspace)):
    return list_documents(ws_id)


@app.post("/workspaces/{workspace_id}/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    ws_id: str = Depends(owned_workspace),
    file: UploadFile = File(...),
    force: bool = Form(False),
):
    if UPLOAD_BUCKET:
        raise HTTPException(400, "This server uses direct uploads. Please reload the page.")
    filename = Path(file.filename or "").name
    ext = Path(filename).suffix.lower()
    if not filename or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'")

    upload_dir = tempfile.mkdtemp(prefix="upload_")
    local_path = str(Path(upload_dir) / filename)

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    sha = hashlib.sha256()
    try:
        with open(local_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                sha.update(chunk)
                out.write(chunk)
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    try:
        result = register_document(ws_id, filename, sha.hexdigest(), force)
    except ValueError as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(409, str(e))
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    if result["duplicate"]:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return JSONResponse(status_code=200, content=result)

    background_tasks.add_task(_run_pipeline, local_path, upload_dir, result["doc_id"])
    return JSONResponse(status_code=202, content=result)


@app.get("/workspaces/{workspace_id}/documents/{doc_id}")
def document_status(doc_id: uuid.UUID, ws_id: str = Depends(owned_workspace)):
    row = get_document_status(str(doc_id), ws_id)
    if not row:
        raise HTTPException(404, "Document not found")
    return row


@app.get("/workspaces/{workspace_id}/documents/{doc_id}/markdown")
def document_markdown(doc_id: uuid.UUID, ws_id: str = Depends(owned_workspace)):
    row = get_document_markdown(str(doc_id), ws_id)
    if not row:
        raise HTTPException(404, "Document not found")
    if row["status"] != "done":
        raise HTTPException(409, f"Document is {row['status']}")
    return {"doc_id": str(doc_id), "markdown": row["markdown"]}


@app.get("/workspaces/{workspace_id}/documents/{doc_id}/summary")
def document_summary(doc_id: uuid.UUID, ws_id: str = Depends(owned_workspace)):
    row = get_summary(str(doc_id), ws_id)
    if not row:
        raise HTTPException(404, "No summary for this document yet")
    return {"doc_id": str(doc_id), "summary": row["summary"], "model": row["model"]}


@app.post("/workspaces/{workspace_id}/documents/{doc_id}/retry", status_code=202)
def retry_document(
    doc_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    ws_id: str = Depends(owned_workspace),
):
    if not begin_retry(str(doc_id), ws_id):
        raise HTTPException(409, "Nothing to retry, or the document is still being processed")
    from Pipeline.process_document import retry_failed_branches
    background_tasks.add_task(retry_failed_branches, str(doc_id))
    return {"doc_id": str(doc_id), "status": "processing"}


@app.get("/workspaces/{workspace_id}/events")
def workspace_events(
    ws_id: str = Depends(owned_workspace),
    doc_id: uuid.UUID | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
):
    return list_events(ws_id, str(doc_id) if doc_id else None, date_from, date_to)


@app.get("/workspaces/{workspace_id}/graph")
def workspace_graph(
    ws_id: str = Depends(owned_workspace),
    doc_id: uuid.UUID | None = None,
    q: str | None = None,
):
    term = (q or "").strip()[:100] or None
    doc = str(doc_id) if doc_id else None
    return {
        "entities": list_entities(ws_id, doc, term),
        "relations": list_relations(ws_id, doc, term),
    }


@app.get("/workspaces/{workspace_id}/summaries")
def workspace_summaries(ws_id: str = Depends(owned_workspace)):
    return list_summaries(ws_id)


# ---------- chat (agent) ----------
CHAT_LIMIT_PER_HOUR = int(os.getenv("CHAT_LIMIT_PER_HOUR", "30"))
_chat_times: dict[str, deque] = defaultdict(deque)
_chat_lock = threading.Lock()


def _check_chat_rate(session_id: str) -> None:
    """At most CHAT_LIMIT_PER_HOUR questions per visitor per hour."""
    now = time.time()
    with _chat_lock:
        times = _chat_times[session_id]
        while times and now - times[0] > 3600:
            times.popleft()
        if len(times) >= CHAT_LIMIT_PER_HOUR:
            raise HTTPException(429, "Question limit reached for this hour. Please try again later.")
        times.append(now)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    question: str
    history: list[ChatTurn] = []


@app.post("/workspaces/{workspace_id}/chat")
def chat(body: ChatIn, ws_id: str = Depends(owned_workspace), sid: str = Depends(get_session_id)):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Questions can be at most {MAX_QUESTION_CHARS} characters.")
    if not any(d["status"] == "done" for d in list_documents(ws_id)):
        raise HTTPException(409, "No documents are ready yet. Upload a document and wait for processing to finish.")

    _check_chat_rate(sid)
    history = [{"role": t.role, "content": t.content} for t in body.history[-12:]]
    try:
        return answer_question(ws_id, question, history)
    except Exception:
        logger.exception("chat failed")
        raise HTTPException(502, "The assistant could not answer right now. Please try again.")



# ---------- direct uploads to S3 ----------
_s3 = None


def _s3_client():
    global _s3
    if _s3 is None:
        import boto3
        from botocore.config import Config
        region = os.getenv("AWS_REGION") or os.getenv("DB_REGION") or "ap-south-1"
        _s3 = boto3.client(
            "s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
    return _s3


@app.get("/config", include_in_schema=False)
def app_config():
    return {"upload_mode": "s3" if UPLOAD_BUCKET else "local", "max_upload_mb": MAX_UPLOAD_MB}


class UploadInit(BaseModel):
    filename: str
    size: int
    sha256: str
    force: bool = False


@app.post("/workspaces/{workspace_id}/uploads")
def init_upload(body: UploadInit, ws_id: str = Depends(owned_workspace)):
    if not UPLOAD_BUCKET:
        raise HTTPException(400, "Direct uploads are not enabled on this server")

    filename = Path(body.filename or "").name.strip()
    ext = Path(filename).suffix.lower()
    if not filename or len(filename) > 150 or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'")
    if not (0 < body.size <= MAX_UPLOAD_BYTES):
        raise HTTPException(413, f"Files must be between 1 byte and {MAX_UPLOAD_MB} MB")
    digest = body.sha256.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise HTTPException(400, "Invalid checksum")

    try:
        result = register_document(ws_id, filename, digest, body.force)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if result["duplicate"]:
        return JSONResponse(status_code=200, content=result)

    key = f"uploads/{ws_id}/{result['doc_id']}/{filename}"
    try:
        post = _s3_client().generate_presigned_post(
            Bucket=UPLOAD_BUCKET, Key=key,
            Conditions=[["content-length-range", 1, MAX_UPLOAD_BYTES]],   # S3 enforces the size cap
            ExpiresIn=600,
        )
    except Exception as e:
        logger.exception("could not create an upload form")
        set_failed(result["doc_id"], "upload", "Could not prepare the upload.", repr(e)[:500])
        raise HTTPException(502, "Could not prepare the upload. Please try again.")
    return JSONResponse(status_code=202, content={**result, "upload": post})


@app.delete("/workspaces/{workspace_id}/uploads/{doc_id}")
def abort_upload(doc_id: uuid.UUID, ws_id: str = Depends(owned_workspace)):
    return {"removed": abort_pending_upload(str(doc_id), ws_id)}


# ---------- AWS Lambda handler (ASGI adapter) ----------
from mangum import Mangum
handler = Mangum(app)