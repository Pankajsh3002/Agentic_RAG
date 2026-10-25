import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

# Ensure project root is on sys.path so package imports resolve from any working directory
PROJECT_ROOT = "/home/pankaj/projects/Agentic_Rag"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Data_Ingestion.Image_Markdown.Parse_Image import parse_image_with_llm
from Data_Ingestion.HTML_Markdown.Parse_HTML import parse_html_with_docling
from Data_Ingestion.Mail_Markdown.Parse_Mail import parse_email_with_docling
from Data_Ingestion.Excel_Markdown.Parse_CSV_Excel import parse_tabular_file
from Data_Ingestion.PDF_Markdown.Docx_to_PDF_Conversion import convert_docx_to_pdf
from Data_Ingestion.PDF_Markdown.Parse_PDF import parse_pdf_with_docling

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 3
EXTRACTED_FILES_DIR = "/home/pankaj/projects/Agentic_Rag/Data_Ingestion/Extracted_Files"

IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".png", ".jpeg"}
HTML_EXTENSIONS = {".html", ".htm"}
MAIL_EXTENSIONS = {".eml", ".msg", ".xml"}
TABULAR_EXTENSIONS = {".xlsx", ".xls", ".csv", ".tsv"}
DOCX_EXTENSIONS = {".docx", ".doc"}
PDF_EXTENSIONS = {".pdf"}


def _resolve_converted_pdf_path(
    src_docx: str,
    out_folder: str,
    returned_value: Optional[str],
) -> str:
    """
    Determines the generated PDF path after calling convert_docx_to_pdf(src_docx, out_folder).
    Supports functions that either return the generated PDF file path directly or save
    '<docx_stem>.pdf' inside out_folder.
    """
    if isinstance(returned_value, (str, Path)):
        candidate = str(returned_value)
        if os.path.isfile(candidate) and candidate.lower().endswith(".pdf"):
            return candidate

    expected_pdf = os.path.join(out_folder, f"{Path(src_docx).stem}.pdf")
    if os.path.isfile(expected_pdf):
        return expected_pdf

    # Fallback: look for the most recently modified .pdf file inside out_folder
    pdf_files = [
        os.path.join(out_folder, f)
        for f in os.listdir(out_folder)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(out_folder, f))
    ]
    if pdf_files:
        return max(pdf_files, key=os.path.getmtime)

    raise FileNotFoundError(
        f"Converted PDF not found in '{out_folder}' for DOCX file '{src_docx}'."
    )


def _route_single_file(
    file_path: str,
    extracted_dir: str = EXTRACTED_FILES_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> str:
    """
    Routes a single file to its corresponding parser function based on file extension
    and returns the resulting Markdown string.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    os.makedirs(extracted_dir, exist_ok=True)
    ext = Path(file_path).suffix.lower()

    # 1. Image files (.tif, .tiff, .jpg, .png, .jpeg)
    if ext in IMAGE_EXTENSIONS:
        logger.info("Routing image file '%s' -> parse_image_with_llm", file_path)
        return parse_image_with_llm(file_path, max_workers=max_workers)

    # 2. HTML files (.html, .htm)
    if ext in HTML_EXTENSIONS:
        logger.info("Routing HTML file '%s' -> parse_html_with_docling", file_path)
        return parse_html_with_docling(file_path)

    # 3. Email files (.eml, .msg, .xml)
    if ext in MAIL_EXTENSIONS:
        logger.info("Routing mail file '%s' -> parse_email_with_docling", file_path)
        return parse_email_with_docling(file_path, extracted_dir)

    # 4. Excel / CSV files (.xlsx, .xls, .csv, .tsv)
    if ext in TABULAR_EXTENSIONS:
        logger.info("Routing tabular file '%s' -> parse_tabular_file", file_path)
        return parse_tabular_file(file_path)

    # 5. DOCX files (.docx, .doc) -> convert to PDF in extracted_dir -> parse_pdf_with_docling
    if ext in DOCX_EXTENSIONS:
        logger.info(
            "Routing DOCX file '%s' -> convert_docx_to_pdf -> parse_pdf_with_docling",
            file_path,
        )
        converted_ret = convert_docx_to_pdf(file_path, extracted_dir)
        pdf_path = _resolve_converted_pdf_path(file_path, extracted_dir, converted_ret)
        try:
            return parse_pdf_with_docling(pdf_path, max_workers=max_workers)
        finally:
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    logger.info("Removed temporary converted PDF: %s", pdf_path)
                except OSError as e:
                    logger.warning("Could not remove temporary PDF '%s': %s", pdf_path, e)

    # 6. PDF files (.pdf)
    if ext in PDF_EXTENSIONS:
        logger.info("Routing PDF file '%s' -> parse_pdf_with_docling", file_path)
        return parse_pdf_with_docling(file_path, max_workers=max_workers)

    raise ValueError(f"Unsupported file format '{ext}' for file: {file_path}")


def process_extracted_files_loop(
    extracted_dir: str = EXTRACTED_FILES_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> List[str]:
    """
    Continuously checks extracted_dir for any generated/extracted files (e.g., email
    attachments or converted artifacts), routes each file through the appropriate parser,
    and removes each file once its processing completes.

    Returns:
        List[str]: Markdown blocks for all processed files from extracted_dir.
    """
    os.makedirs(extracted_dir, exist_ok=True)
    extracted_markdowns: List[str] = []
    attempted_files: Set[str] = set()

    while True:
        pending_files = sorted(
            os.path.join(extracted_dir, name)
            for name in os.listdir(extracted_dir)
            if os.path.isfile(os.path.join(extracted_dir, name))
            and os.path.join(extracted_dir, name) not in attempted_files
        )

        if not pending_files:
            break

        for extracted_file_path in pending_files:
            attempted_files.add(extracted_file_path)
            file_name = os.path.basename(extracted_file_path)
            try:
                logger.info("Processing extracted file: %s", extracted_file_path)
                md_content = _route_single_file(
                    extracted_file_path,
                    extracted_dir=extracted_dir,
                    max_workers=max_workers,
                )
                if md_content and md_content.strip():
                    extracted_markdowns.append(
                        f"## Extracted File: {file_name}\n\n{md_content.strip()}"
                    )
            except Exception as e:
                logger.error(
                    "Failed to process extracted file '%s': %s",
                    extracted_file_path,
                    e,
                )
                extracted_markdowns.append(
                    f"## Extracted File: {file_name}\n\n[Extraction failed for {file_name}: {e}]"
                )
            finally:
                # Always remove the extracted file after processing completes
                if os.path.exists(extracted_file_path):
                    try:
                        os.remove(extracted_file_path)
                        logger.info("Removed processed extracted file: %s", extracted_file_path)
                    except OSError as e:
                        logger.warning(
                            "Failed to delete extracted file '%s': %s",
                            extracted_file_path,
                            e,
                        )

    return extracted_markdowns


def route_document(
    file_path: str,
    extracted_dir: str = EXTRACTED_FILES_DIR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> str:
    """
    Main entry point that routes an input file to its specific parser function,
    processes any files that appear inside extracted_dir (such as mail attachments
    or converted files), deletes those extracted files upon completion, and returns
    the combined Markdown string.

    Args:
        file_path (str): Path to the input document.
        extracted_dir (str): Directory for DOCX conversion and mail attachments.
        max_workers (int): Worker count for parallel PDF/Image parsers.

    Returns:
        str: Final combined Markdown content.
    """
    os.makedirs(extracted_dir, exist_ok=True)

    # 1. Route and parse the primary input file
    primary_markdown = _route_single_file(
        file_path=file_path,
        extracted_dir=extracted_dir,
        max_workers=max_workers,
    )

    # 2. Loop over extracted_dir to process and remove any newly appeared files
    extracted_markdowns = process_extracted_files_loop(
        extracted_dir=extracted_dir,
        max_workers=max_workers,
    )

    all_sections = [primary_markdown.strip()] if primary_markdown else []
    all_sections.extend(extracted_markdowns)

    return "\n\n".join(section for section in all_sections if section).strip()


if __name__ == "__main__":
    sample_input = "/home/pankaj/projects/Agentic_Rag/Pankaj_Sharma_Resume.docx"
    if os.path.exists(sample_input):
        final_md = route_document(sample_input, max_workers=5)
        print("Final Routed Markdown Content Preview")
        print(final_md)