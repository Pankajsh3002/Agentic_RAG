import logging
from datetime import date
from typing import Optional

from langchain_core.tools import tool

from Model_Calling.embedder import embed_query
from db.database import (
    close_conn, list_documents as db_list_documents, list_entities, list_events,
    list_relations, list_summaries, search_chunks,
)

logger = logging.getLogger(__name__)


def _clip(text, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _wrap(text: str) -> str:
    # Everything a tool returns is untrusted text taken from user documents
    return "<tool_data>\n" + text.replace("</tool_data>", "") + "\n</tool_data>"


def _run(body) -> str:
    """Run a tool body. Never raises, and closes this thread's DB connection afterwards,
    because the agent may run tools in short-lived worker threads."""
    try:
        return _wrap(body())
    except Exception as e:
        logger.exception("tool failed")
        return _wrap(f"Tool error: {e}")
    finally:
        close_conn()


def _parse_day(value: Optional[str]):
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        raise ValueError(f"'{value}' is not a valid date. Use the format YYYY-MM-DD.")


def _event_line(e: dict) -> str:
    if e["date_start"]:
        when = str(e["date_start"])
        if e["date_end"]:
            when += " to " + str(e["date_end"])
        elif e["is_ongoing"]:
            when += " (ongoing)"
        if e["date_text"]:
            when += f" [written as: {e['date_text']}]"
    else:
        when = f"no exact date [{e['date_text'] or 'unspecified'}]"
    return f'{when} | {e["title"]} ({e["event_type"]}) | {e["filename"]} | "{_clip(e["quote"], 220)}"'


def make_tools(workspace_id: str, evidence=None) -> list:
    """Build the tools for ONE workspace. The workspace is fixed here, on the server:
    no tool accepts a workspace argument, so the model can never search another one.
    If an Evidence object is given, every piece of text a tool returns is recorded in it."""

    def record(filename, text):
        if evidence is not None:
            evidence.add(filename, text)

    def resolve(filename: Optional[str]):
        """Match a filename (exact, then partial) to a document of this workspace."""
        docs = db_list_documents(workspace_id)
        name = (filename or "").strip().lower()
        found = [d for d in docs if d["filename"].lower() == name] or \
                [d for d in docs if name and name in d["filename"].lower()]
        if len(found) == 1:
            return str(found[0]["doc_id"]), None
        available = ", ".join(d["filename"] for d in docs) or "none"
        if not found:
            return None, f"No document named '{filename}'. Available documents: {available}"
        return None, f"'{filename}' matches several documents. Available documents: {available}"

    @tool
    def list_documents() -> str:
        """List the documents in this workspace with their processing status.
        Use it when you need to know which files exist."""
        def body():
            docs = db_list_documents(workspace_id)
            if not docs:
                return "The workspace has no documents."
            lines = []
            for d in docs:
                line = f"{d['filename']} (status: {d['status']})"
                record(d["filename"], line)
                lines.append(line)
            return "\n".join(lines)
        return _run(body)

    @tool
    def search_documents(query: str, filename: Optional[str] = None) -> str:
        """Semantic search over the full text of the documents. Use it for specific facts,
        clauses, numbers, names and definitions, and as a fallback when get_events or
        search_graph return nothing. Optionally restrict it to one document by filename."""
        def body():
            doc_id = None
            if filename:
                doc_id, err = resolve(filename)
                if err:
                    return err
            rows = search_chunks(embed_query(query[:500]), workspace_id, limit=8, doc_id=doc_id)
            if not rows:
                return "No matching text found."
            parts = []
            for i, r in enumerate(rows, 1):
                head = f"[{i}] {r['filename']}"
                if r["heading_path"]:
                    head += f" > {r['heading_path']}"
                head += f" (similarity {r['similarity']:.2f})"
                content = _clip(r["content"], 900)
                parts.append(head + "\n" + content)
                record(r["filename"], (r["heading_path"] + "\n" if r["heading_path"] else "") + content)
            return "\n\n".join(parts)
        return _run(body)

    @tool
    def get_summary(filename: Optional[str] = None) -> str:
        """Return the stored summary of one document (give its filename), or of every document
        in the workspace when no filename is given. Use it for overview questions such as
        'what is this document about'."""
        def body():
            rows = list_summaries(workspace_id)
            if filename:
                doc_id, err = resolve(filename)
                if err:
                    return err
                rows = [r for r in rows if str(r["doc_id"]) == doc_id]
            if not rows:
                return "The workspace has no documents."
            parts = []
            for r in rows:
                if r["summary"]:
                    text = _clip(r["summary"], 2500)
                    record(r["filename"], text)
                    parts.append(f"Summary of {r['filename']}:\n{text}")
                else:
                    parts.append(f"{r['filename']}: no summary available (document status: {r['status']}).")
            return "\n\n".join(parts)
        return _run(body)

    @tool
    def get_events(date_from: Optional[str] = None, date_to: Optional[str] = None,
                   filename: Optional[str] = None) -> str:
        """List dated events (deadlines, payments, milestones, lease terms, meetings) in
        chronological order. Optionally filter by period with date_from and date_to as ISO
        dates (YYYY-MM-DD), and restrict to one document with filename. It also lists events
        that have no exact calendar date (recurring or relative ones such as 'rent due on the
        5th of each month'), which may still apply to the question."""
        def lines_for(rows):
            out = []
            for e in rows:
                line = _event_line(e)
                record(e["filename"], line)
                out.append(line)
            return out

        def body():
            doc_id = None
            if filename:
                doc_id, err = resolve(filename)
                if err:
                    return err
            start, end = _parse_day(date_from), _parse_day(date_to)
            filtered = bool(start or end)
            events = list_events(workspace_id, doc_id, start, end, limit=60)
            dated = [e for e in events if e["date_start"]]
            if filtered:
                undated = [e for e in list_events(workspace_id, doc_id, None, None, limit=300)
                           if not e["date_start"]][:15]
            else:
                undated = [e for e in events if not e["date_start"]]
            if not dated and not undated:
                return "No events were found."

            parts = []
            if dated:
                parts.append("Dated events, oldest first:\n" + "\n".join(lines_for(dated)))
            elif filtered:
                parts.append("No dated events fall in that period.")
            if undated:
                parts.append("Events without an exact calendar date (they may still apply):\n"
                             + "\n".join(lines_for(undated)))
            return "\n\n".join(parts)
        return _run(body)

    @tool
    def search_graph(name: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Look up entities and the relationships between them (people, organizations,
        places, technologies, agreements, amounts). Give part of an entity name, for example
        'Sunrise Properties', to see everything connected to it. Use it for who / which /
        how-are-they-connected questions. Optionally restrict it to one document."""
        def body():
            doc_id = None
            if filename:
                doc_id, err = resolve(filename)
                if err:
                    return err
            term = (name or "").strip()[:100] or None
            relations = list_relations(workspace_id, doc_id, term, limit=40)
            if relations:
                lines = []
                for r in relations:
                    line = (f'{r["source"]} ({r["source_type"]}) -[{r["relation"]}]-> '
                            f'{r["target"]} ({r["target_type"]}) | {r["filename"]} | "{_clip(r["quote"], 220)}"')
                    record(r["filename"], line)
                    lines.append(line)
                return "Connections:\n" + "\n".join(lines)
            if term:
                entities = list_entities(workspace_id, doc_id, term, limit=20)
                if entities:
                    lines = []
                    for e in entities:
                        line = f'{e["name"]} ({e["entity_type"]}) | {e["filename"]} | {_clip(e["description"], 160)}'
                        record(e["filename"], line)
                        lines.append(line)
                    return "Entities found (no connections stored for them):\n" + "\n".join(lines)
            return "No matching entities or connections in the knowledge graph."
        return _run(body)

    return [list_documents, search_documents, get_summary, get_events, search_graph]