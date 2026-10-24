import logging
from concurrent.futures import ThreadPoolExecutor

import fitz

from Model_Calling.Model_Service import get_llm, invoke_llm

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 5


def _extract_page_text(pdf_path: str, page_number: int) -> str:
    """Runs the VLM on a single page and returns its extracted text."""
    try:
        vlm_response = invoke_llm(
            model_name="openai/gpt-6-luna",
            prompt="Extract the text exactly as it appears in the image and format as Markdown.",
            is_image="y",  # case-insensitive
            image_path=pdf_path,
            pages=page_number,
        )
        return vlm_response.get("response", "").strip()
    except Exception as e:
        # One failed page should not kill the whole document
        logger.error("Page %s failed: %s", page_number, e)
        return f"[Extraction failed for page {page_number}]"


def parse_image_with_llm(pdf_path: str, max_workers: int = DEFAULT_MAX_WORKERS) -> str:
    """
    Parses a PDF using a VLM and extracts OCR text from each page in parallel.
    Pages are processed concurrently but always assembled in ascending order.

    Args:
        pdf_path (str): Path to the PDF file.
        max_workers (int): Number of pages processed concurrently.

    Returns:
        str: The parsed content as a markdown string.
    """
    with fitz.open(pdf_path) as fitz_doc:
        total_pages = len(fitz_doc)

    page_numbers = list(range(1, total_pages + 1))

    # executor.map yields results in the same order as the input iterable,
    # regardless of which worker finishes first.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        page_texts = list(
            executor.map(lambda p: _extract_page_text(pdf_path, p), page_numbers)
        )

    return "".join(
        f"{text}\n\nPage {page_number} of {total_pages}\n\n"
        for page_number, text in zip(page_numbers, page_texts)
    )


if __name__ == "__main__":
    pdf_path = "/home/pankaj/projects/Agentic_Rag/test.pdf"
    parsed_content = parse_image_with_llm(pdf_path, max_workers=5)
    print(parsed_content)