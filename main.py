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

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from Data_Ingestion.Markdown_Router import (  # adjust if your router lives elsewhere
    DOCX_EXTENSIONS, HTML_EXTENSIONS, IMAGE_EXTENSIONS,
    MAIL_EXTENSIONS, PDF_EXTENSIONS, TABULAR_EXTENSIONS,
)
from db.database import (
    begin_retry, create_workspace, delete_expired_workspaces, delete_workspace,
    get_document_markdown, get_document_status, get_summary, get_workspace,
    list_documents, list_entities, list_events, list_relations, list_workspaces,
    register_document,
)
from Pipeline.process_document import process_document, retry_failed_branches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = (
    IMAGE_EXTENSIONS | HTML_EXTENSIONS | MAIL_EXTENSIONS
    | TABULAR_EXTENSIONS | DOCX_EXTENSIONS | PDF_EXTENSIONS
)
MAX_UPLOAD_MB = 50
WORKSPACE_TTL_HOURS = int(os.getenv("WORKSPACE_TTL_HOURS", "24"))
SESSION_COOKIE = "rag_session"
SESSION_MAX_AGE = 7 * 24 * 3600
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"   # set to 1 when served over HTTPS

# Docling is heavy, so allow at most 2 documents to be parsed at once locally.
_parse_slots = threading.Semaphore(2)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_expiry_loop())
    yield
    task.cancel()


app = FastAPI(title="Agentic RAG", lifespan=lifespan)


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
    with _parse_slots:
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
    return list_workspaces(sid)


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