import logging
import os
import re
import tempfile
from pathlib import Path

from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, HTMLFormatOption
from docling_core.types.doc import PictureItem

from Model_Calling.Model_Service import get_llm, invoke_llm

logger = logging.getLogger(__name__)

VLM_MODEL = "openai/gpt-6-luna"
VLM_PROMPT = "Extract the text exactly as it appears in the image and format as Markdown."

# Images smaller than this (width or height, in px) are treated as icons / tracking
# pixels / spacers and are not sent to the VLM.
DEFAULT_MIN_IMAGE_SIZE = 50


def _build_converter(source_uri) -> DocumentConverter:
    """
    Builds a Docling converter for HTML with image fetching turned on.

    `source_uri` lets Docling resolve relative <img src="..."> paths. Only parse HTML
    you trust: fetching images makes the process read local files / make requests.
    """
    kwargs = dict(fetch_images=True, source_uri=source_uri)
    try:
        # Newer Docling versions gate fetching behind explicit flags as well
        backend_options = HTMLBackendOptions(
            **kwargs, enable_local_fetch=True, enable_remote_fetch=True
        )
    except (TypeError, ValueError):
        backend_options = HTMLBackendOptions(**kwargs)

    return DocumentConverter(
        format_options={
            InputFormat.HTML: HTMLFormatOption(backend_options=backend_options)
        }
    )


def _ocr_picture(pil_image) -> str:
    """Sends a single extracted picture to the VLM and returns its text."""
    # 1. Create a temporary image file on disk
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        pil_image.convert("RGB").save(temp_image_path, format="PNG")

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
    """
    Places OCR text after each image marker in the markdown.

    `ocr_texts` is aligned with the pictures in document order (empty string for
    pictures that were skipped or failed), so the n-th marker always gets the n-th
    picture's text.
    """
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
            if re.fullmatch(image_pattern, part):
                if ocr_index < len(ocr_texts) and ocr_texts[ocr_index]:
                    enhanced_markdown += f"\n\n{ocr_texts[ocr_index]}\n\n"
                ocr_index += 1
        return enhanced_markdown

    # No image markers found: append all OCR text at the end
    enhanced_markdown = markdown_output
    for idx, ocr_text in enumerate((t for t in ocr_texts if t), 1):
        enhanced_markdown += f"\n\n### Image {idx} OCR Text\n\n{ocr_text}\n"
    return enhanced_markdown


def parse_html_with_docling(
    html_path: str, min_image_size: int = DEFAULT_MIN_IMAGE_SIZE
) -> str:
    """
    Parses an HTML file (or URL) using Docling in a single pass and extracts OCR text
    from any images, inserting that text right after the image's position.

    Args:
        html_path (str): Path to a local HTML file, or an http(s) URL.
        min_image_size (int): Skip images whose width or height is below this (px).

    Returns:
        str: The parsed content as a markdown string.
    """
    is_url = html_path.startswith(("http://", "https://"))
    source_uri = html_path if is_url else Path(html_path).resolve()

    converter = _build_converter(source_uri)

    # Whole HTML document in one go (no multi-worker)
    result = converter.convert(html_path)
    docling_doc = result.document

    # Collect pictures in document order; keep a slot (None) for any that can't be used
    # so the OCR results stay aligned with the image markers in the markdown.
    pil_images = []
    for node, _level in docling_doc.iterate_items():
        if isinstance(node, PictureItem):
            try:
                pil_image = node.get_image(docling_doc)
            except Exception as e:
                logger.warning("Could not load image %s: %s", node.self_ref, e)
                pil_image = None

            if pil_image is not None and min(pil_image.size) >= min_image_size:
                pil_images.append(pil_image)
            else:
                pil_images.append(None)

    print(f"Found {len(pil_images)} image(s) in HTML, {sum(i is not None for i in pil_images)} to OCR.")

    ocr_texts = []
    for idx, pil_image in enumerate(pil_images, 1):
        if pil_image is None:
            ocr_texts.append("")
            continue
        try:
            ocr_texts.append(_ocr_picture(pil_image))
        except Exception as e:
            # One failed image should not kill the whole document
            logger.error("OCR failed for image %s: %s", idx, e)
            ocr_texts.append("")

    markdown_output = docling_doc.export_to_markdown()
    enhanced_markdown = _merge_ocr_into_markdown(markdown_output, ocr_texts)

    return enhanced_markdown.replace("&amp;", "&").strip()


if __name__ == "__main__":
    html_path = "/home/pankaj/projects/Agentic_Rag/sample2.html"
    markdown_content = parse_html_with_docling(html_path)
    print("Final Markdown Content Preview:")
    print(markdown_content)