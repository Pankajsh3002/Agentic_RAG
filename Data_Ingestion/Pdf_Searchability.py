"""
pdf_searchability.py
====================

Classify PDF pages (or a whole PDF) as:

    "searchable"            - real, trustworthy text layer that covers the page content
    "partially_searchable"  - some real text exists, but a big part of the page is
                              image-only (e.g. a searchable stamp/header on top of a
                              scanned body) or only part of the text layer is usable
    "unsearchable"          - no text layer, OR a text layer exists but it is garbage /
                              does not match the page image (bad OCR layer, broken font
                              encoding, outlined text, ...)
    "blank"                 - nothing on the page (ignored when judging a whole PDF)

Install:
    pip install pymupdf numpy
    # optional, for the strongest check (ocr_check=True):
    pip install pytesseract pillow   (+ the Tesseract binary)

Usage:
    from pdf_searchability import check_pdf_searchability

    check_pdf_searchability("file.pdf")                  # whole PDF
    check_pdf_searchability("file.pdf", 3)               # page 3 only (1-based)
    check_pdf_searchability("file.pdf", [1, 4, 7])       # selected pages
    check_pdf_searchability("file.pdf", "2-5,9")         # ranges
    check_pdf_searchability("file.pdf", ocr_check=True)  # verify against real OCR

CLI:
    python pdf_searchability.py file.pdf
    python pdf_searchability.py file.pdf --pages 1-3,8 --ocr-check --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pymupdf  # PyMuPDF

PageSpec = Union[None, int, str, Sequence[int]]

SEARCHABLE = "searchable"
PARTIAL = "partially_searchable"
UNSEARCHABLE = "unsearchable"
BLANK = "blank"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_pages(spec: PageSpec, total: int) -> List[int]:
    """Turn a user page spec (1-based) into a list of 0-based indexes."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return list(range(total))

    if isinstance(spec, int):
        nums = [spec]
    elif isinstance(spec, str):
        nums = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                nums.extend(range(int(a), int(b) + 1))
            else:
                nums.append(int(part))
    else:
        nums = [int(n) for n in spec]

    if not nums:  # empty list -> same as "no page given"
        return list(range(total))

    bad = [n for n in nums if n < 1 or n > total]
    if bad:
        raise ValueError(f"Page number(s) {bad} out of range. PDF has {total} page(s).")
    return [n - 1 for n in nums]


_BAD_CHARS = re.compile(r"[\ue000-\uf8ff\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
_VOWELS = set("aeiouyAEIOUY")


def _is_good_word(word: str) -> bool:
    """
    Cheap plausibility test for one extracted token.
    Catches the usual signatures of a broken text layer: (cid:12) codes, U+FFFD /
    private-use glyphs, punctuation soup ("|_.,"), consonant-only junk, "aaaa" runs,
    random case flips ("tHeCoMpAnY").
    Non-Latin scripts (Devanagari etc.) are only checked for the encoding-garbage markers.
    """
    if "(cid:" in word or _BAD_CHARS.search(word):
        return False

    core = re.sub(r"^\W+|\W+$", "", word)  # strip leading/trailing punctuation
    if not core:
        return False  # punctuation only

    alnum = sum(ch.isalnum() for ch in core)
    if alnum / len(core) < 0.5:
        return False

    if re.search(r"([^\W\d_])\1{3,}", core):  # same letter 4+ times
        return False

    letters = [c for c in core if c.isalpha()]
    if letters and all(c.isascii() for c in letters):
        if len(letters) >= 4 and not core.isupper() and not any(c in _VOWELS for c in letters):
            return False  # long lowercase/mixed-case word with no vowels
        if len(letters) >= 5:
            flips = sum(1 for a, b in zip(letters, letters[1:]) if a.islower() and b.isupper())
            if flips >= 2:
                return False
    return True


def _norm_tokens(words: Sequence[str], min_len: int = 3) -> List[str]:
    toks = []
    for w in words:
        for t in re.findall(r"\w+", w.lower()):
            if len(t) >= min_len:
                toks.append(t)
    return toks


def _mark(mask: np.ndarray, rect: "pymupdf.Rect", sx: float, sy: float,
          x_off: float, y_off: float, block: int, pad: int = 0) -> None:
    """Set mask cells (block grid) that overlap `rect` (page coords)."""
    gh, gw = mask.shape
    x0 = int(math.floor((rect.x0 - x_off) * sx / block)) - pad
    x1 = int(math.floor((rect.x1 - x_off) * sx / block)) + pad
    y0 = int(math.floor((rect.y0 - y_off) * sy / block)) - pad
    y1 = int(math.floor((rect.y1 - y_off) * sy / block)) + pad
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, gw - 1), min(y1, gh - 1)
    if x0 <= x1 and y0 <= y1:
        mask[y0:y1 + 1, x0:x1 + 1] = True


def _ink_grid(page: "pymupdf.Page", dpi: int, block: int, min_ink_px: int,
              edge_margin: float):
    """Render page, return (ink_block_mask, sx, sy)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]

    paper = float(np.percentile(gray, 95))
    ink = gray < max(paper - 60, 40)  # clearly darker than the paper

    h, w = ink.shape
    gh, gw = math.ceil(h / block), math.ceil(w / block)
    padded = np.zeros((gh * block, gw * block), dtype=np.uint8)
    padded[:h, :w] = ink
    counts = padded.reshape(gh, block, gw, block).sum(axis=(1, 3))
    grid = counts >= min_ink_px

    # ignore scanner borders / page-edge shadows
    my, mx = int(gh * edge_margin), int(gw * edge_margin)
    if my:
        grid[:my, :] = False
        grid[-my:, :] = False
    if mx:
        grid[:, :mx] = False
        grid[:, -mx:] = False

    sx, sy = pix.width / page.rect.width, pix.height / page.rect.height
    return grid, sx, sy


def _ocr_tokens(page: "pymupdf.Page", lang: str, dpi: int = 200) -> List[str]:
    """Run Tesseract on the rendered page and return confident word tokens."""
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    words = []
    for txt, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if txt and txt.strip() and c >= 60:
            words.append(txt)
    return _norm_tokens(words)


# --------------------------------------------------------------------------- #
# Single page analysis
# --------------------------------------------------------------------------- #
def _analyze_page(
    page: "pymupdf.Page",
    page_no: int,
    *,
    ocr_check: bool,
    ocr_lang: str,
    dpi: int,
    block: int,
    min_ink_px: int,
    min_image_area_ratio: float,
    gap_ok: float,
    gap_bad: float,
    stamp_max_words: int,
) -> Dict:
    rot = page.rotation_matrix
    rect = page.rect
    page_area = rect.width * rect.height

    # ---- text layer (visible AND invisible/OCR text is returned by PyMuPDF) ----
    raw_words = [w for w in page.get_text("words") if str(w[4]).strip()]
    total_words = len(raw_words)
    good, bad = [], 0
    for w in raw_words:
        if _is_good_word(w[4]):
            good.append(w)
        else:
            bad += 1
    good_words = len(good)
    good_ratio = good_words / total_words if total_words else 0.0

    # ---- images ----
    img_rects = []
    try:
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"]) * rot
            r.normalize()
            r = r & rect
            if not r.is_empty and (r.width * r.height) / page_area >= min_image_area_ratio:
                img_rects.append(r)
    except Exception:
        pass

    # ---- ink / coverage maps ----
    ink, sx, sy = _ink_grid(page, dpi, block, min_ink_px, edge_margin=0.02)
    total_ink = int(ink.sum())
    has_ink = total_ink >= 10

    text_mask = np.zeros_like(ink)
    for w in good:  # only TRUSTWORTHY words are allowed to "cover" ink
        r = pymupdf.Rect(w[:4]) * rot
        r.normalize()
        _mark(text_mask, r, sx, sy, rect.x0, rect.y0, block, pad=1)

    img_mask = np.zeros_like(ink)
    for r in img_rects:
        _mark(img_mask, r, sx, sy, rect.x0, rect.y0, block)
    image_cov = float(img_mask.sum()) / img_mask.size

    raster_gap = (
        int((ink & img_mask & ~text_mask).sum()) / total_ink if total_ink else 0.0
    )
    text_cov_of_ink = int((ink & text_mask).sum()) / total_ink if total_ink else 0.0
    has_scan_content = bool(img_rects) and int((ink & img_mask).sum()) >= 10

    metrics = {
        "total_words": total_words,
        "good_words": good_words,
        "good_word_ratio": round(good_ratio, 3),
        "has_raster_images": bool(img_rects),
        "image_coverage": round(image_cov, 3),
        "ink_covered_by_text": round(text_cov_of_ink, 3),
        "uncovered_image_ink": round(raster_gap, 3),
    }

    # ---------------------------------------------------------------- decide
    def result(status: str, reason: str) -> Dict:
        return {
            "page": page_no,
            "status": status,
            "needs_ocr": status in (PARTIAL, UNSEARCHABLE),
            "reason": reason,
            "metrics": metrics,
        }

    if total_words == 0:
        if has_ink:
            return result(
                UNSEARCHABLE,
                "No text layer at all - page content is image/graphics only (needs OCR).",
            )
        return result(BLANK, "No text and no visible content.")

    if good_words == 0:
        return result(
            UNSEARCHABLE,
            "A text layer exists but none of it is usable (garbled/(cid:) characters, "
            "symbol soup) - typical of a bad OCR layer or broken font encoding.",
        )

    # --- heuristic verdict ---
    if not has_scan_content:
        # pure digital page: judge only by text quality
        if good_ratio >= 0.6:
            status, reason = SEARCHABLE, "Digital text layer with good-quality text."
        elif good_ratio >= 0.35:
            status, reason = PARTIAL, "Part of the text layer is garbled / unreliable."
        else:
            status, reason = UNSEARCHABLE, "Text layer is mostly garbled / unreliable."
    else:
        if raster_gap <= gap_ok:
            if good_ratio >= 0.6:
                status, reason = SEARCHABLE, "Scanned image with a text layer that covers the content."
            else:
                status, reason = PARTIAL, "Text layer covers the page but a large part of it is garbled."
        elif raster_gap >= gap_bad:
            if good_ratio < 0.35:
                status, reason = (
                    UNSEARCHABLE,
                    "Text layer is mostly garbage and does not cover the scanned content (bad OCR layer).",
                )
            elif good_words <= stamp_max_words:
                status, reason = (
                    PARTIAL,
                    f"Only {good_words} word(s) are searchable (stamp/header/footer?) - "
                    "the scanned body has no usable text.",
                )
            else:
                status, reason = (
                    PARTIAL,
                    "Text layer exists but does not line up with the scanned content "
                    "(misaligned or poor OCR layer) - verify with ocr_check=True.",
                )
        else:
            status, reason = (
                PARTIAL,
                f"About {raster_gap:.0%} of the scanned content has no matching searchable text.",
            )

    # --- optional OCR cross-check (ground truth) ---
    if ocr_check:
        try:
            ocr_toks = _ocr_tokens(page, ocr_lang)
        except Exception as exc:  # tesseract / pytesseract missing etc.
            metrics["ocr_check"] = f"skipped ({type(exc).__name__}: {exc})"
        else:
            metrics["ocr_words"] = len(ocr_toks)
            if len(ocr_toks) >= 15:
                emb = Counter(_norm_tokens([w[4] for w in good]))
                ocr = Counter(ocr_toks)
                matched = sum(min(c, emb.get(t, 0)) for t, c in ocr.items())
                recall = matched / len(ocr_toks)
                metrics["ocr_text_recall"] = round(recall, 3)

                if recall >= 0.75 and good_ratio >= 0.6:
                    status, reason = SEARCHABLE, "Text layer matches what OCR reads from the page image."
                elif recall >= 0.25:
                    status, reason = (
                        PARTIAL,
                        f"Text layer only matches ~{recall:.0%} of what is visible on the page.",
                    )
                elif good_words <= stamp_max_words and good_ratio >= 0.6:
                    status, reason = (
                        PARTIAL,
                        f"Only a small amount of text ({good_words} words, e.g. a stamp) is "
                        f"searchable; body is image-only (text layer matches ~{recall:.0%}).",
                    )
                else:
                    status, reason = (
                        UNSEARCHABLE,
                        f"Text layer does not match the page image (only ~{recall:.0%} overlap "
                        "with OCR) - bad / wrong OCR layer.",
                    )
            else:
                metrics["ocr_check"] = "inconclusive (OCR found too few words)"

    return result(status, reason)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def check_pdf_searchability(
    pdf_path: str,
    page_numbers: PageSpec = None,
    ocr_check: bool = False,
    ocr_lang: str = "eng",
    *,
    dpi: int = 300,
    block: int = 8,
    min_ink_px: int = 3,
    min_image_area_ratio: float = 0.005,
    gap_ok: float = 0.20,
    gap_bad: float = 0.80,
    stamp_max_words: int = 40,
) -> Dict:
    """
    Check whether PDF pages are searchable.

    Parameters
    ----------
    pdf_path      : path to the PDF.
    page_numbers  : None / [] / ""  -> check the complete PDF (verdict for the whole file)
                    int             -> single page, 1-based           e.g. 3
                    list / tuple    -> selected pages                  e.g. [1, 4, 7]
                    str             -> ranges                          e.g. "2-5,9"
    ocr_check     : also OCR each page with Tesseract and compare with the embedded
                    text layer. Slower, but catches "looks searchable but isn't" layers.
    ocr_lang      : Tesseract language(s), e.g. "eng" or "eng+hin".

    Returns
    -------
    dict with:
        status       : "searchable" | "partially_searchable" | "unsearchable" | "blank"
                       (whole-PDF verdict when several pages are checked; blank pages ignored)
        needs_ocr    : True if OCR is needed for the checked page(s)
        summary      : {status: [page numbers]}
        pages        : per-page list with status, reason and metrics
    """
    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected.")
        total = doc.page_count
        indexes = _resolve_pages(page_numbers, total)

        pages = [
            _analyze_page(
                doc[i], i + 1,
                ocr_check=ocr_check, ocr_lang=ocr_lang, dpi=dpi, block=block,
                min_ink_px=min_ink_px, min_image_area_ratio=min_image_area_ratio,
                gap_ok=gap_ok, gap_bad=gap_bad, stamp_max_words=stamp_max_words,
            )
            for i in indexes
        ]

    summary: Dict[str, List[int]] = {SEARCHABLE: [], PARTIAL: [], UNSEARCHABLE: [], BLANK: []}
    for p in pages:
        summary[p["status"]].append(p["page"])

    real = {p["status"] for p in pages if p["status"] != BLANK}
    if not real:
        overall = BLANK
    elif real == {SEARCHABLE}:
        overall = SEARCHABLE
    elif real == {UNSEARCHABLE}:
        overall = UNSEARCHABLE
    else:
        overall = PARTIAL

    return {
        "file": pdf_path,
        "total_pages": total,
        "pages_checked": [p["page"] for p in pages],
        "status": overall,
        "needs_ocr": overall in (PARTIAL, UNSEARCHABLE),
        "summary": summary,
        "pages": pages,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
# def _main() -> None:
#     ap = argparse.ArgumentParser(description="Check whether PDF pages are searchable.")
#     ap.add_argument("pdf")
#     ap.add_argument("--pages", help='e.g. "3", "1,4,7" or "2-5,9" (1-based). Default: all pages')
#     ap.add_argument("--ocr-check", action="store_true", help="verify with Tesseract OCR (slower)")
#     ap.add_argument("--lang", default="eng", help="Tesseract language, e.g. eng+hin")
#     ap.add_argument("--json", action="store_true", help="print full JSON")
#     args = ap.parse_args()

#     res = check_pdf_searchability(args.pdf, args.pages, ocr_check=args.ocr_check, ocr_lang=args.lang)

#     if args.json:
#         json.dump(res, sys.stdout, indent=2)
#         print()
#         return

#     print(f"File   : {res['file']}  ({res['total_pages']} pages)")
#     print(f"Verdict: {res['status'].upper()}   (needs OCR: {res['needs_ocr']})")
#     for status, nums in res["summary"].items():
#         if nums:
#             print(f"  {status:<22}: {nums}")
#     print()
#     for p in res["pages"]:
#         print(f"  page {p['page']:>4} | {p['status']:<21} | {p['reason']}")


if __name__ == "__main__":
    result = check_pdf_searchability("/home/pankaj/projects/Agentic_Rag/20-1-43-016434.pdf",1)
    print(result)