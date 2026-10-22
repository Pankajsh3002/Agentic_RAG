import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import base64
import mimetypes

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage


import logging

logger = logging.getLogger(__name__)

# .env is in the project root (parent of Model_Calling/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = "https://openrouter.ai/api/v1"


def _build(model: str, temperature: float = 0):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=BASE_URL,
        timeout=30,
        max_retries=3,
    )


@lru_cache
def get_llm(role: str = "default",):
    main = os.getenv("MAIN_MODEL")
    small = os.getenv("SMALL_MODEL", main)

    configs = {
        "default":   (main, 0),
        "router":    (small, 0),
        "grader":    (small, 0),
        "generator": (main, 0.2),
    }
    model, temp = configs[role]
    return _build(model, temp)


def _parse_pages(pages: str | None, total: int) -> list[int]:
    """
    None / "" / blank -> all pages.
    '1,2,3', '2-4', '1,3-5' -> zero-based indexes.
    Pages outside the PDF's range are skipped with a warning.
    """
    if pages is None or not str(pages).strip():
        return list(range(total))

    requested = []
    for part in str(pages).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            requested.extend(range(start, end + 1))
        else:
            requested.append(int(part))

    valid = [p for p in dict.fromkeys(requested) if 1 <= p <= total]  # dedupes, keeps order
    skipped = [p for p in dict.fromkeys(requested) if p not in valid]

    if skipped:
        logger.warning("Skipping pages %s: PDF has only %d page(s)", skipped, total)
    if not valid:
        raise ValueError(f"None of the requested pages exist (PDF has {total} page(s))")

    return [p - 1 for p in valid]


def _to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode('utf-8')}"


def build_image_message(prompt: str, file_path: str, pages: str | None = None, dpi: int = 300) -> HumanMessage:
    """
    file_path: PDF or image file
    pages: e.g. "1,2,3" or "2,3,4" or "2-4" (1-based, PDFs only)
    dpi: render quality; higher = sharper but more tokens
    """
    content = [{"type": "text", "text": prompt}]

    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        try:
            for idx in _parse_pages(pages, len(doc)):
                pix = doc[idx].get_pixmap(dpi=dpi)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _to_data_url(pix.tobytes("png"))},
                })
        finally:
            doc.close()
    else:
        mime = mimetypes.guess_type(file_path)[0] or "image/png"
        with open(file_path, "rb") as f:
            content.append({
                "type": "image_url",
                "image_url": {"url": _to_data_url(f.read(), mime)},
            })

    return HumanMessage(content=content)



# llm = get_llm("default")

# msg = build_image_message(
#     prompt="Extract the key points from these resume.",
#     file_path="Pankaj_Sharma_AI_Engineer.pdf",
#     pages="1,2,3",
# )
# print(llm.invoke([msg]).content)


# llm = get_llm("generator")
# print(llm.invoke("Say hello").content)