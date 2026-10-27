import os
import threading
import json

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

load_dotenv()

MAX_WORKSPACES_PER_SESSION = 3
MAX_DOCS_PER_WORKSPACE = 5

_local = threading.local()


_BRANCHES_SQL = """
    COALESCE((
        SELECT jsonb_object_agg(
                   b.branch,
                   jsonb_build_object('status', b.status, 'error', left(b.error, 150)))
        FROM document_branches b WHERE b.doc_id = d.doc_id
    ), '{}'::jsonb) AS branches
"""


def get_conn() -> psycopg.Connection:
    """One connection per thread, so parallel jobs never share a connection."""
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        conn = psycopg.connect(
            os.environ["DATABASE_URL"],
            autocommit=True,
            row_factory=dict_row,
        )
        _local.conn = conn
    return conn


def _vec(v: list[float]) -> str:
    """Format a Python list as pgvector text, e.g. '[0.1,0.2]'."""
    return "[" + ",".join(str(x) for x in v) + "]"


# ---------------- workspaces ----------------
def create_workspace(session_id: str, name: str) -> dict:
    name = (name or "").strip()[:80] or "Untitled workspace"
    conn = get_conn()
    with conn.transaction():
        count = conn.execute(
            "SELECT count(*) AS n FROM workspaces WHERE session_id = %s", (session_id,)
        ).fetchone()["n"]
        if count >= MAX_WORKSPACES_PER_SESSION:
            raise ValueError(
                f"You can have at most {MAX_WORKSPACES_PER_SESSION} workspaces. Delete one first."
            )
        return conn.execute(
            "INSERT INTO workspaces (session_id, name) VALUES (%s, %s) "
            "RETURNING workspace_id, name, created_at",
            (session_id, name),
        ).fetchone()


def list_workspaces(session_id: str) -> list[dict]:
    return get_conn().execute(
        """
        SELECT w.workspace_id, w.name, w.created_at,
               count(d.doc_id) AS documents,
               count(d.doc_id) FILTER (WHERE d.status = 'done') AS ready_documents
        FROM workspaces w LEFT JOIN documents d USING (workspace_id)
        WHERE w.session_id = %s
        GROUP BY w.workspace_id
        ORDER BY w.created_at
        """,
        (session_id,),
    ).fetchall()

def list_summaries(workspace_id: str) -> list[dict]:
    """Every document of a workspace with its summary (if any) and the summary step's state."""
    return get_conn().execute(
        """
        SELECT d.doc_id, d.filename, d.status, d.created_at,
               s.summary, s.model,
               b.status AS summary_status, left(b.error, 200) AS summary_error
        FROM documents d
        LEFT JOIN summaries s ON s.doc_id = d.doc_id
        LEFT JOIN document_branches b ON b.doc_id = d.doc_id AND b.branch = 'summary'
        WHERE d.workspace_id = %s
        ORDER BY d.created_at
        """,
        (workspace_id,),
    ).fetchall()
    
def get_workspace(workspace_id: str, session_id: str) -> dict | None:
    """Return the workspace only if it belongs to this session, else None."""
    return get_conn().execute(
        "SELECT workspace_id, name, created_at FROM workspaces "
        "WHERE workspace_id = %s AND session_id = %s",
        (workspace_id, session_id),
    ).fetchone()


def delete_workspace(workspace_id: str, session_id: str) -> bool:
    row = get_conn().execute(
        "DELETE FROM workspaces WHERE workspace_id = %s AND session_id = %s "
        "RETURNING workspace_id",
        (workspace_id, session_id),
    ).fetchone()
    return row is not None


def delete_expired_workspaces(hours: int = 24) -> int:
    """Remove workspaces (and everything inside them) older than `hours`."""
    rows = get_conn().execute(
        "DELETE FROM workspaces WHERE created_at < now() - make_interval(hours => %s) "
        "RETURNING workspace_id",
        (hours,),
    ).fetchall()
    return len(rows)


# ---------------- documents ----------------
def register_document(workspace_id: str, filename: str, content_hash: str, force: bool = False) -> dict:
    """
    Decide what to do with an incoming file, inside one workspace and one transaction.
    Returns {"doc_id", "status", "duplicate"}.
    """
    conn = get_conn()
    try:
        with conn.transaction():
            existing = conn.execute(
                """
                SELECT doc_id, status,
                       (status IN ('pending', 'processing')
                        AND updated_at < now() - interval '30 minutes') AS stale
                FROM documents WHERE workspace_id = %s AND content_hash = %s FOR UPDATE
                """,
                (workspace_id, content_hash),
            ).fetchone()

            if existing:
                duplicate = {"doc_id": str(existing["doc_id"]),
                             "status": existing["status"], "duplicate": True}
                in_progress = (existing["status"] in ("pending", "processing")
                               and not existing["stale"])

                if in_progress:
                    if force:
                        raise ValueError(
                            "This document is still being processed. Try again when it finishes."
                        )
                    return duplicate                    # a live job owns it

                if existing["status"] == "done" and not force:
                    return duplicate                    # already processed

                # failed, stale (dead job), or force: remove the old record
                conn.execute("DELETE FROM documents WHERE doc_id = %s", (existing["doc_id"],))

            count = conn.execute(
                "SELECT count(*) AS n FROM documents WHERE workspace_id = %s", (workspace_id,)
            ).fetchone()["n"]
            if count >= MAX_DOCS_PER_WORKSPACE:
                raise ValueError(
                    f"A workspace can hold at most {MAX_DOCS_PER_WORKSPACE} documents."
                )

            row = conn.execute(
                "INSERT INTO documents (workspace_id, filename, content_hash) "
                "VALUES (%s, %s, %s) RETURNING doc_id",
                (workspace_id, filename, content_hash),
            ).fetchone()
            return {"doc_id": str(row["doc_id"]), "status": "pending", "duplicate": False}
    except psycopg.errors.UniqueViolation:
        # two identical uploads arrived at the same moment; the other one won
        existing = conn.execute(
            "SELECT doc_id, status FROM documents WHERE workspace_id = %s AND content_hash = %s",
            (workspace_id, content_hash),
        ).fetchone()
        return {"doc_id": str(existing["doc_id"]),
                "status": existing["status"], "duplicate": True}


def list_documents(workspace_id: str) -> list[dict]:
    return get_conn().execute(
        f"""
        SELECT d.doc_id, d.filename, d.status, d.stage, d.error, d.created_at, d.updated_at,
               {_BRANCHES_SQL}
        FROM documents d WHERE d.workspace_id = %s ORDER BY d.created_at
        """,
        (workspace_id,),
    ).fetchall()


def get_document_status(doc_id: str, workspace_id: str) -> dict | None:
    """Status info for the polling endpoint (markdown and error_detail are not included)."""
    return get_conn().execute(
        f"""
        SELECT d.doc_id, d.filename, d.status, d.stage, d.error, d.created_at, d.updated_at,
               {_BRANCHES_SQL}
        FROM documents d WHERE d.doc_id = %s AND d.workspace_id = %s
        """,
        (doc_id, workspace_id),
    ).fetchone()


def get_document_markdown(doc_id: str, workspace_id: str) -> dict | None:
    return get_conn().execute(
        "SELECT status, markdown FROM documents WHERE doc_id = %s AND workspace_id = %s",
        (doc_id, workspace_id),
    ).fetchone()


# ---------------- pipeline helpers (called by the server with a known doc_id) ----------------
def set_status(doc_id: str, status: str, error: str | None = None) -> None:
    get_conn().execute(
        "UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE doc_id = %s",
        (status, error, doc_id),
    )


def set_stage(doc_id: str, stage: str) -> None:
    get_conn().execute(
        "UPDATE documents SET stage = %s, updated_at = now() WHERE doc_id = %s",
        (stage, doc_id),
    )


def set_failed(doc_id: str, stage: str, error: str, detail: str) -> None:
    get_conn().execute(
        """
        UPDATE documents
        SET status = 'failed', stage = %s, error = %s, error_detail = %s, updated_at = now()
        WHERE doc_id = %s
        """,
        (stage, error, detail, doc_id),
    )


def save_markdown(doc_id: str, markdown: str, finish: bool = True) -> None:
    """Store the markdown. With finish=True (default) also mark the document done."""
    if finish:
        get_conn().execute(
            """
            UPDATE documents
            SET markdown = %s, status = 'done', stage = 'completed',
                error = NULL, error_detail = NULL, updated_at = now()
            WHERE doc_id = %s
            """,
            (markdown, doc_id),
        )
    else:
        get_conn().execute(
            "UPDATE documents SET markdown = %s, updated_at = now() WHERE doc_id = %s",
            (markdown, doc_id),
        )

def save_summary(doc_id: str, summary: str, model: str | None = None) -> None:
    conn = get_conn()
    with conn.transaction():
        doc = conn.execute(
            "SELECT workspace_id FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        if not doc:
            raise ValueError("Document no longer exists (its workspace may have been deleted)")
        conn.execute(
            """
            INSERT INTO summaries (doc_id, workspace_id, summary, model)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE
            SET summary = EXCLUDED.summary, model = EXCLUDED.model, created_at = now()
            """,
            (doc_id, doc["workspace_id"], summary, model),
        )


def get_summary(doc_id: str, workspace_id: str) -> dict | None:
    return get_conn().execute(
        "SELECT summary, model, created_at FROM summaries WHERE doc_id = %s AND workspace_id = %s",
        (doc_id, workspace_id),
    ).fetchone()


def get_pipeline_markdown(doc_id: str) -> str | None:
    row = get_conn().execute(
        "SELECT markdown FROM documents WHERE doc_id = %s", (doc_id,)
    ).fetchone()
    return row["markdown"] if row else None


def failed_branches(doc_id: str) -> list[str]:
    rows = get_conn().execute(
        "SELECT branch FROM document_branches WHERE doc_id = %s AND status = 'failed'",
        (doc_id,),
    ).fetchall()
    return [r["branch"] for r in rows]


def begin_retry(doc_id: str, workspace_id: str) -> bool:
    """Claim a document for a retry. Returns False if there is nothing to retry or it is busy."""
    row = get_conn().execute(
        """
        UPDATE documents
        SET status = 'processing', stage = 'retrying', updated_at = now()
        WHERE doc_id = %s AND workspace_id = %s
          AND status IN ('done', 'failed') AND markdown IS NOT NULL
          AND EXISTS (SELECT 1 FROM document_branches b
                      WHERE b.doc_id = documents.doc_id AND b.status = 'failed')
        RETURNING doc_id
        """,
        (doc_id, workspace_id),
    ).fetchone()
    return row is not None

def mark_done(doc_id: str) -> None:
    get_conn().execute(
        """
        UPDATE documents
        SET status = 'done', stage = 'completed', error = NULL, error_detail = NULL,
            updated_at = now()
        WHERE doc_id = %s
        """,
        (doc_id,),
    )


# ---------------- chunks and search ----------------
def save_chunks(doc_id: str, chunks: list[dict], vectors: list[list[float]]) -> None:
    """Replace all chunks of a document in one transaction."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    conn = get_conn()
    with conn.transaction():
        doc = conn.execute(
            "SELECT workspace_id FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        if not doc:
            raise ValueError("Document no longer exists (its workspace may have been deleted)")

        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (workspace_id, doc_id, chunk_index, heading_path, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::halfvec)
                """,
                [
                    (doc["workspace_id"], doc_id, c["chunk_index"], c["heading_path"],
                     c["content"], _vec(v))
                    for c, v in zip(chunks, vectors)
                ],
            )


def search_chunks(query_vector: list[float], workspace_id: str,
                  limit: int = 5, doc_id: str | None = None) -> list[dict]:
    """Closest chunks inside ONE workspace (optionally inside one document)."""
    vec = _vec(query_vector)
    return get_conn().execute(
        """
        SELECT c.chunk_id, c.doc_id, d.filename, c.heading_path, c.content,
               1 - (c.embedding <=> %s::halfvec) AS similarity
        FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.workspace_id = %s AND (%s::uuid IS NULL OR c.doc_id = %s::uuid)
        ORDER BY c.embedding <=> %s::halfvec
        LIMIT %s
        """,
        (vec, workspace_id, doc_id, doc_id, vec, limit),
    ).fetchall()

def close_conn() -> None:
    """Close this thread's connection (called when a worker thread finishes)."""
    conn = getattr(_local, "conn", None)
    if conn is not None and not conn.closed:
        conn.close()
    _local.conn = None


def clear_branches(doc_id: str) -> None:
    get_conn().execute("DELETE FROM document_branches WHERE doc_id = %s", (doc_id,))


def set_branch(doc_id: str, branch: str, status: str,
               error: str | None = None, detail: str | None = None) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO document_branches (doc_id, branch, status, error, error_detail, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (doc_id, branch) DO UPDATE
        SET status = EXCLUDED.status, error = EXCLUDED.error,
            error_detail = EXCLUDED.error_detail, updated_at = now()
        """,
        (doc_id, branch, status, error, detail),
    )
    # keep the document "alive" so the 30-minute stale check never hits a running job
    conn.execute("UPDATE documents SET updated_at = now() WHERE doc_id = %s", (doc_id,))


# ---------------- branch raw output, events, graph ----------------
def save_branch_output(doc_id: str, branch: str, output: dict) -> None:
    """Keep the model's original JSON for debugging (capped at about 100 KB)."""
    text = json.dumps(output, ensure_ascii=False, default=str)
    if len(text) > 100_000:
        output = {"truncated": True, "size": len(text), "preview": text[:20_000]}
    get_conn().execute(
        "UPDATE document_branches SET raw_output = %s WHERE doc_id = %s AND branch = %s",
        (Jsonb(output), doc_id, branch),
    )


def save_events(doc_id: str, events: list[dict]) -> int:
    """Replace all events of a document in one transaction."""
    conn = get_conn()
    with conn.transaction():
        doc = conn.execute(
            "SELECT workspace_id FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        if not doc:
            raise ValueError("Document no longer exists (its workspace may have been deleted)")

        conn.execute("DELETE FROM events WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO events (workspace_id, doc_id, title, description, event_type,
                                    date_start, date_end, is_ongoing, date_precision,
                                    date_text, quote)
                VALUES (%s, %s, %s, %s, %s, %s::date, %s::date, %s, %s, %s, %s)
                """,
                [
                    (doc["workspace_id"], doc_id, e["title"], e["description"], e["event_type"],
                     e["date_start"], e["date_end"], e["is_ongoing"], e["date_precision"],
                     e["date_text"], e["quote"])
                    for e in events
                ],
            )
    return len(events)


def list_events(workspace_id: str, doc_id: str | None = None, date_from=None,
                date_to=None, limit: int = 300) -> list[dict]:
    """Events in ONE workspace. With a date filter, undated events are left out."""
    return get_conn().execute(
        """
        SELECT e.event_id, e.doc_id, d.filename, e.title, e.description, e.event_type,
               e.date_start, e.date_end, e.is_ongoing, e.date_precision, e.date_text, e.quote
        FROM events e JOIN documents d ON d.doc_id = e.doc_id
        WHERE e.workspace_id = %s
          AND (%s::uuid IS NULL OR e.doc_id = %s::uuid)
          AND (%s::date IS NULL OR COALESCE(e.date_end,
                 CASE WHEN e.is_ongoing THEN 'infinity'::date ELSE e.date_start END) >= %s::date)
          AND (%s::date IS NULL OR e.date_start <= %s::date)
        ORDER BY e.date_start NULLS LAST, e.event_id
        LIMIT %s
        """,
        (workspace_id, doc_id, doc_id, date_from, date_from, date_to, date_to, limit),
    ).fetchall()


def save_graph(doc_id: str, entities: list[dict], relations: list[dict]) -> tuple[int, int]:
    """Replace the whole graph of a document in one transaction. Returns (entities, relations)."""
    conn = get_conn()
    with conn.transaction():
        doc = conn.execute(
            "SELECT workspace_id FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        if not doc:
            raise ValueError("Document no longer exists (its workspace may have been deleted)")
        ws = doc["workspace_id"]

        conn.execute("DELETE FROM relations WHERE doc_id = %s", (doc_id,))
        conn.execute("DELETE FROM entities WHERE doc_id = %s", (doc_id,))

        ids: dict[str, int] = {}
        for e in entities:
            row = conn.execute(
                """
                INSERT INTO entities (workspace_id, doc_id, name, entity_key, entity_type, description)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, entity_key) DO UPDATE SET name = entities.name
                RETURNING entity_id
                """,
                (ws, doc_id, e["name"], e["key"], e["type"], e["description"]),
            ).fetchone()
            ids[e["key"]] = row["entity_id"]

        saved = 0
        for r in relations:
            source, target = ids.get(r["source_key"]), ids.get(r["target_key"])
            if source is None or target is None:
                continue
            conn.execute(
                """
                INSERT INTO relations (workspace_id, doc_id, source_id, target_id, relation, quote)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (ws, doc_id, source, target, r["relation"], r["quote"]),
            )
            saved += 1
    return len(ids), saved


def _like(term: str | None) -> str | None:
    """Turn a search word into a safe LIKE pattern."""
    if not term:
        return None
    t = term.lower().strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{t}%" if t else None


def list_entities(workspace_id: str, doc_id: str | None = None, term: str | None = None,
                  limit: int = 300) -> list[dict]:
    pattern = _like(term)
    return get_conn().execute(
        """
        SELECT e.entity_id, e.doc_id, d.filename, e.name, e.entity_type, e.description
        FROM entities e JOIN documents d ON d.doc_id = e.doc_id
        WHERE e.workspace_id = %s
          AND (%s::uuid IS NULL OR e.doc_id = %s::uuid)
          AND (%s::text IS NULL OR e.entity_key LIKE %s ESCAPE '\\')
        ORDER BY e.entity_type, e.name
        LIMIT %s
        """,
        (workspace_id, doc_id, doc_id, pattern, pattern, limit),
    ).fetchall()


def list_relations(workspace_id: str, doc_id: str | None = None, term: str | None = None,
                   limit: int = 300) -> list[dict]:
    """Links in ONE workspace. With a term, only links that touch a matching entity."""
    pattern = _like(term)
    return get_conn().execute(
        """
        SELECT r.relation_id, r.doc_id, d.filename,
               s.name AS source, s.entity_type AS source_type,
               r.relation,
               t.name AS target, t.entity_type AS target_type,
               r.quote
        FROM relations r
        JOIN entities s ON s.entity_id = r.source_id
        JOIN entities t ON t.entity_id = r.target_id
        JOIN documents d ON d.doc_id = r.doc_id
        WHERE r.workspace_id = %s
          AND (%s::uuid IS NULL OR r.doc_id = %s::uuid)
          AND (%s::text IS NULL OR s.entity_key LIKE %s ESCAPE '\\'
                                OR t.entity_key LIKE %s ESCAPE '\\')
        ORDER BY r.relation_id
        LIMIT %s
        """,
        (workspace_id, doc_id, doc_id, pattern, pattern, pattern, limit),
    ).fetchall()

if __name__ == "__main__":
    # Isolation self-test: python3 -m db.database   (run from the project root)
    a = create_workspace("session-A", "A workspace")
    b = create_workspace("session-B", "B workspace")
    try:
        wa, wb = str(a["workspace_id"]), str(b["workspace_id"])
        print("A opens its own workspace:", bool(get_workspace(wa, "session-A")))
        print("B opens A's workspace:", bool(get_workspace(wa, "session-B")), "(must be False)")

        same_hash = "samehash123"
        ra = register_document(wa, "sample.pdf", same_hash)
        rb = register_document(wb, "sample.pdf", same_hash)
        print("same file in two workspaces, both new:", not ra["duplicate"] and not rb["duplicate"])
        print("same file again in A is a duplicate:",
              register_document(wa, "sample.pdf", same_hash)["duplicate"])

        vec = [1.0] + [0.0] * 3071
        save_chunks(ra["doc_id"], [{"chunk_index": 0, "heading_path": "H", "content": "chunk of A"}], [vec])
        save_chunks(rb["doc_id"], [{"chunk_index": 0, "heading_path": "H", "content": "chunk of B"}], [vec])
        print("search in A:", [r["content"] for r in search_chunks(vec, wa)])
        print("search in B:", [r["content"] for r in search_chunks(vec, wb)])
    finally:
        delete_workspace(str(a["workspace_id"]), "session-A")
        delete_workspace(str(b["workspace_id"]), "session-B")
        print("test data removed")