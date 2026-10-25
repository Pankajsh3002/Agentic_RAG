import logging
import os
from datetime import date

from Pipeline.extraction import (
    MAX_TOTAL_CHARS, RULES, ask_json, clean_text, norm, quote_in_doc, split_windows,
)
from db.database import save_branch_output, save_events

logger = logging.getLogger(__name__)

MAX_EVENTS = 150
MAX_TOKENS = 8000   # lower this if your model rejects the limit

EVENT_TYPES = {"milestone", "deadline", "payment", "meeting", "legal", "other"}
PRECISIONS = {"day", "month", "year", "range", "relative", "unknown"}

PROMPT = """You extract dated events from a document.
<<RULES>>

Return ONLY a JSON object with one key, "events", whose value is a list. Each event is an object with these keys:
- "title": short label, at most 12 words
- "description": one or two sentences, or null
- "event_type": one of milestone, deadline, payment, meeting, legal, other
- "date_text": the date or time expression exactly as written in the document
- "date_start": ISO date YYYY-MM-DD when the date is unambiguous, otherwise null
- "date_end": ISO date for the end of a range, otherwise null
- "is_ongoing": true if it is still continuing ("Present", "until further notice"), otherwise false
- "date_precision": precision of date_start: "day", "month" or "year"; "range" when date_end is also given; "relative" for expressions like "within 30 days of notice"; "unknown" if there is no date
- "quote": one sentence or line copied from the document that states this event, character for character, as one contiguous span with no ellipses

Rules:
- Only include events that have a time anchor: an exact date, a month or year, a range, or a relative expression.
- Never invent or calculate dates. For relative or recurring expressions ("30 days after signing", "the 5th of each month") set date_start to null.
- If a numeric date such as 03/04/2025 could be day/month or month/day, set date_start to null and keep it in date_text.
- A month-only date such as "June 2024" becomes date_start 2024-06-01 with precision "month". A year-only date becomes YYYY-01-01 with precision "year".
- For ranges such as "2021 - 2023" or "Jan 2020 - Present", set date_start, set date_end when an end is given, and set is_ongoing to true when it runs to the present.
- Do not include timeless facts (addresses, skills) or statements without any time anchor.
- At most 40 events. If there are none, return {"events": []}.

<document>
<<TEXT>>
</document>"""


def _parse_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        d = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return d if 1900 <= d.year <= 2100 else None


def _clean_event(raw, norm_doc: str) -> dict | None:
    """Validate one event from the model. Returns None if it must be dropped."""
    if not isinstance(raw, dict):
        return None

    title = clean_text(raw.get("title"), 200)
    quote = clean_text(raw.get("quote"), 600)
    if not title or not quote or not quote_in_doc(quote, norm_doc):
        return None          # no evidence in the document: drop it

    start, end = _parse_date(raw.get("date_start")), _parse_date(raw.get("date_end"))
    if end and (not start or end < start):
        end = None
    date_text = clean_text(raw.get("date_text"), 200)
    if not start and not date_text:
        return None          # no time anchor at all

    precision = raw.get("date_precision")
    if precision not in PRECISIONS:
        precision = "unknown"
    if end:
        precision = "range"
    elif not start and precision in ("day", "month", "year", "range"):
        precision = "unknown"
    elif start and precision == "relative":
        precision = "unknown"

    event_type = raw.get("event_type")
    return {
        "title": title,
        "description": clean_text(raw.get("description"), 500),
        "event_type": event_type if event_type in EVENT_TYPES else "other",
        "date_start": start.isoformat() if start else None,
        "date_end": end.isoformat() if end else None,
        "is_ongoing": raw.get("is_ongoing") is True,
        "date_precision": precision,
        "date_text": date_text,
        "quote": quote,
    }


def run(doc_id: str, markdown: str) -> None:
    model = os.getenv("EVENTS_MODEL") or None   # None -> MAIN_MODEL
    text = markdown[:MAX_TOTAL_CHARS]
    norm_doc = norm(text)

    raw_outputs: list[dict] = []
    events: list[dict] = []
    seen: set[tuple] = set()
    dropped = 0

    for window in split_windows(text):
        prompt = PROMPT.replace("<<RULES>>", RULES).replace("<<TEXT>>", window)
        data = ask_json(prompt, model, MAX_TOKENS)
        raw_outputs.append(data)

        items = data.get("events")
        if not isinstance(items, list):
            raise ValueError('The model output has no "events" list')

        for raw in items:
            event = _clean_event(raw, norm_doc)
            if event is None:
                dropped += 1
                continue
            key = (event["title"].lower(), event["date_start"], (event["date_text"] or "").lower())
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
        if len(events) >= MAX_EVENTS:
            events = events[:MAX_EVENTS]
            break

    save_branch_output(doc_id, "events", {"windows": raw_outputs, "kept": len(events), "dropped": dropped})
    save_events(doc_id, events)
    logger.info("doc_id=%s events: kept %d, dropped %d", doc_id, len(events), dropped)