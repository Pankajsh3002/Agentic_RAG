import os
import re
import tempfile

# os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pymupdf as fitz
# import pytesseract
from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem

from Data_Ingestion.Docling_PDF_Parser.Pdf_Searchability import check_pdf_searchability
from Model_Calling.Model_Service import get_llm, invoke_llm

# If running on Linux/WSL and tesseract is in PATH, you do not need tesseract_cmd.
# If needed explicitly on Linux: pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def parse_pdf_with_docling(pdf_path: str) -> str:
    """
    Parses a PDF using Docling and extracts OCR text from images.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The parsed content as a markdown string.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_classification = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    with fitz.open(pdf_path) as fitz_doc:
        total_pages = len(fitz_doc)

    complete_text = ""
    

    for page_number in range(1, total_pages + 1):
        image_searchable = check_pdf_searchability(pdf_path, page_number,90)
        # image_searchable= {"status": "searchable"}
        if image_searchable.get("status") == "searchable":
            print(f"Page {page_number} is already searchable.")

            result = converter.convert(pdf_path, page_range=(page_number, page_number))
            docling_doc = result.document

            ocr_texts = []
            for node, level in docling_doc.iterate_items():
                if isinstance(node, PictureItem):
                    pil_image = node.get_image(docling_doc)
                    if pil_image:
                        # 1. Create a temporary image file on disk
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                            temp_image_path = tmp_file.name
                            pil_image.save(temp_image_path, format="PNG")

                        try:
                            # 2. Build message passing the temporary file path
                            vlm_response = invoke_llm( model_name="openai/gpt-6-luna",
                                                    prompt="Extract the text exactly as it appears in the image and format as Markdown.",
                                                    is_image="y",                    # case-insensitive
                                                    image_path=temp_image_path,
                                                    pages=1,
                                                    )
                            ocr_text = vlm_response.get("response", "").strip()

                            if ocr_text:
                                ocr_texts.append(ocr_text)
                        finally:
                            # 4. Ensure cleanup even if an exception occurs
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)

            markdown_output = docling_doc.export_to_markdown()

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
            else:
                enhanced_markdown = markdown_output
                for idx, ocr_text in enumerate(ocr_texts, 1):
                    enhanced_markdown += f"\n\n### Image {idx} OCR Text\n\n{ocr_text}\n"

            complete_text += f"{enhanced_markdown}\n\nPage {page_number} of {total_pages}\n\n"

        else:
            print(f"Page {page_number} requires vision extraction.")

            vlm_response = invoke_llm( model_name="openai/gpt-6-luna",
                                        prompt="Extract the text exactly as it appears in the image and format as Markdown.",
                                        is_image="y",                    # case-insensitive
                                        image_path=pdf_path,
                                        pages=page_number,
                                        )
            complete_text += f"{vlm_response['response']}\n\nPage {page_number} of {total_pages}\n\n"

    return complete_text.replace("&amp;", "&").strip()


if __name__ == "__main__":
    pdf_path = "/home/pankaj/projects/Agentic_Rag/test.pdf"
    markdown_content = parse_pdf_with_docling(pdf_path)
    print("Final Markdown Content Preview (first 1000 chars):")
    print(markdown_content)