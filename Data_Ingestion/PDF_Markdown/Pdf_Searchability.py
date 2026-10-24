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

    check_pdf_searchability("file.pdf", 1, 95)           # page 1: up to 5% may be unsearchable
    check_pdf_searchability("file.pdf", None, 95)        # whole PDF: up to 5% of ALL pages combined
    check_pdf_searchability("file.pdf", [1, 4, 7], 90)   # selected pages combined
    check_pdf_searchability("file.pdf", "2-5,9", 98)     # ranges
    check_pdf_searchability("file.pdf", 1, 95, ocr_check=True)  # verify against real OCR

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
    searchable_percent: float,
    ocr_check: bool,
    ocr_lang: str,
    dpi: int,
    block: int,
    min_ink_px: int,
    min_image_area_ratio: float,
    gap_bad: float,
    stamp_max_words: int,
) -> Dict:
    rot = page.rotation_matrix
    rect = page.rect
    page_area = rect.width * rect.height

    # block / min_ink_px are defined at 100 dpi and scaled, so changing dpi is safe
    s = dpi / 100.0
    block_px = max(2, round(block * s))
    min_ink = max(1, round(min_ink_px * s * s))

    # ---- text layer (visible AND invisible/OCR text is returned by PyMuPDF) ----
    raw_words = [w for w in page.get_text("words") if str(w[4]).strip()]
    total_words = len(raw_words)
    good, bad = [], []
    for w in raw_words:
        (good if _is_good_word(w[4]) else bad).append(w)
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
    ink, sx, sy = _ink_grid(page, dpi, block_px, min_ink, edge_margin=0.02)
    total_ink = int(ink.sum())
    has_ink = total_ink >= 10

    text_mask = np.zeros_like(ink)   # ink under TRUSTWORTHY words  -> searchable
    bad_mask = np.zeros_like(ink)    # ink under garbage words      -> NOT searchable
    img_mask = np.zeros_like(ink)    # ink inside raster images     -> searchable only if covered
    for w in good:
        r = pymupdf.Rect(w[:4]) * rot
        r.normalize()
        _mark(text_mask, r, sx, sy, rect.x0, rect.y0, block_px, pad=1)
    for w in bad:
        r = pymupdf.Rect(w[:4]) * rot
        r.normalize()
        _mark(bad_mask, r, sx, sy, rect.x0, rect.y0, block_px)
    for r in img_rects:
        _mark(img_mask, r, sx, sy, rect.x0, rect.y0, block_px)
    image_cov = float(img_mask.sum()) / img_mask.size

    # ---- searchable percentage of the page content --------------------------
    # content      = ink that is text or should be text (under good words, garbage
    #                words, or inside raster images). Plain vector lines/logos are ignored.
    # searchable   = the part of that content sitting under good words.
    covered_n = int((ink & text_mask).sum())
    uncovered_n = int((ink & (img_mask | bad_mask) & ~text_mask).sum())
    content_n = covered_n + uncovered_n

    if total_words == 0 or good_words == 0:
        content_n, covered_n = total_ink, 0          # nothing usable: all ink is unsearchable
        pct = 0.0 if has_ink else 100.0
    elif content_n == 0:
        pct = 100.0                                   # words but nothing visible to search
    else:
        pct = 100.0 * covered_n / content_n

    has_scan_content = bool(img_rects) and int((ink & img_mask).sum()) >= 10

    metrics = {
        "total_words": total_words,
        "good_words": good_words,
        "good_word_ratio": round(good_ratio, 3),
        "has_raster_images": bool(img_rects),
        "image_coverage": round(image_cov, 3),
        "searchable_percent": round(pct, 2),
        "content_blocks": content_n,          # weight used when pages are combined
        "searchable_blocks": covered_n,
    }

    def result(status: str, reason: str) -> Dict:
        p = metrics["searchable_percent"]
        return {
            "page": page_no,
            "status": status,
            "searchable_percent": p,
            "unsearchable_percent": round(100.0 - p, 2),
            "needs_ocr": status in (PARTIAL, UNSEARCHABLE),
            "reason": reason,
            "metrics": metrics,
        }

    # ---------------------------------------------------------------- decide
    if total_words == 0:
        if has_ink:
            return result(UNSEARCHABLE, "No text layer at all - page content is image/graphics only (needs OCR).")
        return result(BLANK, "No text and no visible content.")

    if good_words == 0:
        return result(
            UNSEARCHABLE,
            "A text layer exists but none of it is usable (garbled/(cid:) characters, "
            "symbol soup) - typical of a bad OCR layer or broken font encoding.",
        )

    low_cut = (1.0 - gap_bad) * 100.0   # "almost nothing is searchable" boundary (default 20 %)

    if pct >= searchable_percent:
        status = SEARCHABLE
        reason = f"{pct:.1f}% of the page content is searchable (required {searchable_percent:g}%)."
    elif good_ratio < 0.35:
        status = UNSEARCHABLE
        reason = (f"Text layer is mostly garbage ({pct:.1f}% searchable) - "
                  "bad OCR layer or broken font encoding.")
    elif pct <= low_cut and good_words <= stamp_max_words:
        status = PARTIAL
        reason = (f"Only {good_words} word(s) are searchable (stamp/header/footer?) - "
                  f"{pct:.1f}% searchable, the body has no usable text.")
    elif pct <= low_cut:
        status = PARTIAL
        reason = (f"Text layer exists but covers only {pct:.1f}% of the content "
                  "(misaligned or poor OCR layer) - verify with ocr_check=True.")
    else:
        status = PARTIAL
        reason = (f"{pct:.1f}% searchable, below the required {searchable_percent:g}% "
                  f"({100 - pct:.1f}% of the content has no usable text).")

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

                # OCR is the ground truth: its recall replaces the ink-based percentage
                pct = 100.0 * recall
                metrics["searchable_percent"] = round(pct, 2)
                metrics["searchable_blocks"] = int(round(content_n * recall))

                if pct >= searchable_percent:
                    status = SEARCHABLE
                    reason = f"Text layer matches {pct:.1f}% of what OCR reads from the page image."
                elif recall >= 0.25:
                    status = PARTIAL
                    reason = (f"Text layer only matches {pct:.1f}% of what is visible on the page "
                              f"(required {searchable_percent:g}%).")
                elif good_words <= stamp_max_words and good_ratio >= 0.6:
                    status = PARTIAL
                    reason = (f"Only a small amount of text ({good_words} words, e.g. a stamp) is "
                              f"searchable; body is image-only (text layer matches {pct:.1f}%).")
                else:
                    status = UNSEARCHABLE
                    reason = (f"Text layer does not match the page image (only {pct:.1f}% overlap "
                              "with OCR) - bad / wrong OCR layer.")
            else:
                metrics["ocr_check"] = "inconclusive (OCR found too few words)"

    return result(status, reason)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def check_pdf_searchability(
    pdf_path: str,
    page_numbers: PageSpec = None,
    searchable_percent: float = 95.0,
    ocr_check: bool = False,
    ocr_lang: str = "eng",
    *,
    dpi: int = 300,
    block: int = 8,
    min_ink_px: int = 3,
    min_image_area_ratio: float = 0.005,
    gap_bad: float = 0.80,
    stamp_max_words: int = 40,
) -> Dict:
    """
    Check whether PDF pages are searchable.

    Parameters
    ----------
    pdf_path           : path to the PDF.
    page_numbers       : None / [] / ""  -> complete PDF (all pages combined)
                         int             -> single page, 1-based           e.g. 1
                         list / tuple    -> selected pages                  e.g. [1, 4, 7]
                         str             -> ranges                          e.g. "2-5,9"
    searchable_percent : how much of the content must be searchable, e.g. 95 means
                         up to 5% may be unsearchable.
                         * single page       -> applied to that page
                         * None / many pages -> applied to ALL checked pages combined
                           (pages are weighted by how much content they have, so one
                           sparse page does not outweigh 50 dense ones)
    ocr_check          : also OCR each page with Tesseract and use how much of the
                         OCR text is found in the text layer as the percentage.
                         Slower; catches "looks searchable but isn't" layers.
                         (Tesseract itself is not perfect - with ocr_check you may
                         want a slightly lower searchable_percent, e.g. 90.)
    ocr_lang           : Tesseract language(s), e.g. "eng" or "eng+hin".

    Returns
    -------
    dict with:
        status              : "searchable" | "partially_searchable" | "unsearchable" | "blank"
        searchable_percent  : measured % for the checked page(s)
        unsearchable_percent: 100 - searchable_percent
        required_percent    : the threshold you passed
        needs_ocr           : True if status is partially_searchable / unsearchable
        pages_below_threshold: pages whose own % is under the threshold
        summary             : {status: [page numbers]}
        pages               : per-page list with status, percentages, reason and metrics
    """
    if not 0 < searchable_percent <= 100:
        raise ValueError("searchable_percent must be between 0 and 100 (e.g. 95 for 95%).")

    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected.")
        total = doc.page_count
        indexes = _resolve_pages(page_numbers, total)

        pages = [
            _analyze_page(
                doc[i], i + 1,
                searchable_percent=searchable_percent,
                ocr_check=ocr_check, ocr_lang=ocr_lang, dpi=dpi, block=block,
                min_ink_px=min_ink_px, min_image_area_ratio=min_image_area_ratio,
                gap_bad=gap_bad, stamp_max_words=stamp_max_words,
            )
            for i in indexes
        ]

    summary: Dict[str, List[int]] = {SEARCHABLE: [], PARTIAL: [], UNSEARCHABLE: [], BLANK: []}
    for p in pages:
        summary[p["status"]].append(p["page"])

    real = [p for p in pages if p["status"] != BLANK]

    # ---- combined percentage over every checked page (weighted by content) ----
    if not real:
        agg_pct = None
    else:
        content = sum(p["metrics"]["content_blocks"] for p in real)
        covered = sum(p["metrics"]["searchable_blocks"] for p in real)
        if content > 0:
            agg_pct = 100.0 * covered / content
        else:  # nothing measurable (e.g. only invisible text) -> plain average
            agg_pct = sum(p["searchable_percent"] for p in real) / len(real)

    if not real:
        overall = BLANK
    elif agg_pct >= searchable_percent:
        overall = SEARCHABLE
    elif all(p["status"] == UNSEARCHABLE for p in real):
        overall = UNSEARCHABLE
    else:
        overall = PARTIAL

    return {
        "file": pdf_path,
        "total_pages": total,
        "pages_checked": [p["page"] for p in pages],
        "status": overall,
        "searchable_percent": None if agg_pct is None else round(agg_pct, 2),
        "unsearchable_percent": None if agg_pct is None else round(100.0 - agg_pct, 2),
        "required_percent": searchable_percent,
        "needs_ocr": overall in (PARTIAL, UNSEARCHABLE),
        "pages_below_threshold": [p["page"] for p in real if p["searchable_percent"] < searchable_percent],
        "summary": summary,
        "pages": pages,
    }



if __name__ == "__main__":
    result = check_pdf_searchability("/home/pankaj/projects/Agentic_Rag/test.pdf", 1, 90)
    print(result["status"], f"- {result['searchable_percent']}% searchable "
          f"(required {result['required_percent']}%)")
    for p in result["pages"]:
        print(f"  page {p['page']}: {p['status']} | {p['searchable_percent']}% | {p['reason']}")