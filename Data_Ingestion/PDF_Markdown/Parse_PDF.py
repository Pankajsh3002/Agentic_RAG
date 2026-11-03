import logging
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pymupdf as fitz
# import pytesseract
from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem

from Data_Ingestion.PDF_Markdown.Pdf_Searchability import check_pdf_searchability
from Model_Calling.Model_Service import get_llm, invoke_llm

# If running on Linux/WSL and tesseract is in PATH, you do not need tesseract_cmd.
# If needed explicitly on Linux: pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 5
VLM_MODEL = "openai/gpt-6-luna"
VLM_PROMPT = "Extract the text exactly as it appears in the image and format as Markdown."

# Docling's converter (and PyMuPDF) hold heavy / non-thread-safe state, so the local,
# CPU/GPU-bound steps are serialized with this lock. The slow network-bound VLM calls
# run outside the lock, which is where the parallel speedup comes from.
_LOCAL_LOCK = threading.Lock()


_CONVERTER = None
_CONVERTER_LOCK = threading.Lock()


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    pipeline_options.accelerator_options.num_threads = 2
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_classification = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        with _CONVERTER_LOCK:
            if _CONVERTER is None:
                _CONVERTER = _build_converter()
    return _CONVERTER


def _ocr_picture(pil_image) -> str:
    """Sends a single extracted picture to the VLM and returns its text."""
    # 1. Create a temporary image file on disk
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        pil_image.save(temp_image_path, format="PNG")

    try:
        # 2. Build message passing the temporary file path
        vlm_response = invoke_llm(
            model_name=VLM_MODEL,
            prompt=VLM_PROMPT,
            is_image="y",  # case-insensitive
            image_path=temp_image_path,
            pages=1,
        )
        return vlm_response.get("response", "").strip()
    finally:
        # 3. Ensure cleanup even if an exception occurs
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def _merge_ocr_into_markdown(markdown_output: str, ocr_texts: list) -> str:
    """Places OCR text from pictures into the markdown, after each image marker."""
    # Pattern definitions
    pattern1 = r"!\[.*?\]\(.*?\)"
    pattern2 = r"<img[^>]*>"
    pattern3 = r"<!--.*?image.*?-->"
    pattern4 = r"\[Figure \d+\]|\(Figure \d+\)"

    if re.findall(pattern1, markdown_output):
        image_pattern = pattern1
    elif re.findall(pattern2, markdown_output):
        image_pattern = pattern2
    elif re.findall(pattern3, markdown_output, re.IGNORECASE):
        image_pattern = pattern3
    elif re.findall(pattern4, markdown_output):
        image_pattern = pattern4
    else:
        image_pattern = None

    if image_pattern:
        parts = re.split(f"({image_pattern})", markdown_output)
        enhanced_markdown = ""
        ocr_index = 0

        for part in parts:
            enhanced_markdown += part
            if re.match(image_pattern, part):
                if ocr_index < len(ocr_texts):
                    enhanced_markdown += f"\n\n{ocr_texts[ocr_index]}\n\n"
                    ocr_index += 1
        return enhanced_markdown

    enhanced_markdown = markdown_output
    for idx, ocr_text in enumerate(ocr_texts, 1):
        enhanced_markdown += f"\n\n### Image {idx} OCR Text\n\n{ocr_text}\n"
    return enhanced_markdown


def parse_pdf_with_docling(pdf_path: str, max_workers: int = DEFAULT_MAX_WORKERS) -> str:
    """
    Parses a searchable PDF using Docling in a single optimized pass,
    preserving table structure and extracting picture OCR where available.

    Args:
        pdf_path (str): Path to the PDF file.
        max_workers (int): Number of concurrent workers for picture OCR.

    Returns:
        str: The parsed content as a markdown string.
    """
    converter = _get_converter()

    with _LOCAL_LOCK:
        result = converter.convert(pdf_path)
        docling_doc = result.document

        pil_images = []
        for node, level in docling_doc.iterate_items():
            if isinstance(node, PictureItem):
                try:
                    pil_image = node.get_image(docling_doc)
                    if pil_image:
                        pil_images.append(pil_image)
                except Exception as e:
                    logger.warning("Could not extract picture image: %s", e)

        markdown_output = docling_doc.export_to_markdown()

    # VLM calls for pictures run outside the lock concurrently
    if pil_images:
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                ocr_texts = list(executor.map(_ocr_picture, pil_images))
            ocr_texts = [t for t in ocr_texts if t]
            if ocr_texts:
                markdown_output = _merge_ocr_into_markdown(markdown_output, ocr_texts)
        except Exception as e:
            logger.warning("Picture OCR failed (%s); using base Docling markdown", e)

    return markdown_output.replace("&amp;", "&").strip()


if __name__ == "__main__":
    pdf_path = "/home/pankaj/projects/Agentic_Rag/test.pdf"
    markdown_content = parse_pdf_with_docling(pdf_path, max_workers=5)
    print("Final Markdown Content Preview")
    print(markdown_content)