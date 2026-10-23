import base64
import io
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF (PDF)
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from datetime import datetime

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PARAMS = {"temperature": 0.2}
MAX_SIDE = 2000  # downscale large images to control tokens and request size


# ---------- page parsing ----------
def _parse_pages(pages, total: int) -> list[int]:
    """None/''/blank -> all pages. '1,2,3', '2-4' -> zero-based indexes. Out-of-range pages skipped."""
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

    unique = list(dict.fromkeys(requested))
    valid = [p for p in unique if 1 <= p <= total]
    skipped = [p for p in unique if p not in valid]

    if skipped:
        logger.warning("Skipping pages %s: file has only %d page(s)", skipped, total)
    if not valid:
        raise ValueError(f"None of the requested pages exist (file has {total} page(s))")
    return [p - 1 for p in valid]


# ---------- image loading (pdf, tif, tiff, jpg, jpeg, png, bmp, gif, webp ...) ----------
def _data_url(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode('utf-8')}"


def _pil_to_data_url(img: Image.Image) -> str:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return _data_url(buf.getvalue(), "image/jpeg")


def _load_images(image_path: str, pages=None, dpi: int = 150) -> list[str]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    urls = []
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        try:
            for idx in _parse_pages(pages, len(doc)):
                pix = doc[idx].get_pixmap(dpi=dpi)
                urls.append(_data_url(pix.tobytes("png"), "image/png"))
        finally:
            doc.close()
    else:
        # Pillow handles single and multi-page images (tif/tiff/gif) and common formats
        with Image.open(path) as im:
            total = getattr(im, "n_frames", 1)
            for idx in _parse_pages(pages, total):
                im.seek(idx)
                urls.append(_pil_to_data_url(im))
    return urls


# ---------- message builder ----------
def build_message(prompt: str, is_image: str = "N", image_path: str = "", pages="") -> HumanMessage:
    flag = str(is_image).strip().upper()
    if flag not in ("Y", "N"):
        raise ValueError("is_image must be 'Y' or 'N' (case-insensitive)")

    if flag == "N":
        return HumanMessage(content=prompt)

    if not image_path:
        raise ValueError("image_path is required when is_image='Y'")

    content = [{"type": "text", "text": prompt}]
    for url in _load_images(image_path, pages):
        content.append({"type": "image_url", "image_url": {"url": url}})
    return HumanMessage(content=content)


# ---------- LLM wrapper ----------
class LLMService:
    def __init__(self, model_name: str | None = None, timeout: int = 60, model_parameter: dict | None = None):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

        model_name = model_name or os.getenv("MAIN_MODEL")
        if not model_name:
            raise ValueError("model_name not provided and MAIN_MODEL not found in .env")

        params = {**DEFAULT_PARAMS, **(model_parameter or {})}  # user values override defaults

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=BASE_URL,
            timeout=timeout,
            max_retries=3,
            **params,
        )

    def invoke(self, prompt, is_image: str = "N", image_path: str = "", pages=""):
        """prompt: a string, or a list of LangChain messages (for graph nodes)."""
        if isinstance(prompt, str):
            messages = [build_message(prompt, is_image, image_path, pages)]
        else:
            messages = prompt
        return self.llm.invoke(messages)

    def bind_tools(self, tools, **kwargs):
        return self.llm.bind_tools(tools, **kwargs)

    def __getattr__(self, name):
        # anything else (with_structured_output, stream, with_fallbacks, ...) goes to the real model
        return getattr(self.llm, name)


def get_llm(model_name: str | None = None, timeout: int = 60, model_parameter: dict | None = None) -> LLMService:
    return LLMService(model_name, timeout, model_parameter)

def _get_token_usage(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens")

    # fallback for providers that only fill response_metadata
    meta = getattr(response, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage", {})
    return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")

# ---------- one-shot function with all parameters ----------
def invoke_llm(
    model_name: str | None,
    prompt: str,
    is_image: str = "N",
    image_path: str = "",
    pages="",
    timeout: int = 60,
    model_parameter: dict | None = None,
) -> dict:
    llm = LLMService(model_name, timeout, model_parameter)

    start_time = datetime.now()
    response = llm.invoke(prompt, is_image, image_path, pages)
    end_time = datetime.now()

    token_in, token_out = _get_token_usage(response)

    return {
        "response": response.content,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "token_in": token_in,
        "token_out": token_out,
    }




# from Model_Calling.Model_Service import get_llm, invoke_llm

# # 1. Text only, one-shot
# r = invoke_llm("openai/gpt-4o-mini", "Say hello")
# print(r.content)

# # 2. Image/PDF/TIFF, one-shot (pages blank = all pages)
# r = invoke_llm(
#     model_name="openai/gpt-4o-mini",
#     prompt="Extract the key points",
#     is_image="y",                    # case-insensitive
#     image_path="docs/scan.tiff",
#     pages="1,2,3",
#     timeout=90,
#     model_parameter={"temperature": 0, "max_tokens": 1500},
# )

# # 3. Reusable object: invoke and bind_tools
# llm = get_llm("openai/gpt-4o-mini", timeout=60, model_parameter={"temperature": 0})
# llm.invoke("Hello")
# llm.invoke("Describe this", is_image="Y", image_path="photo.png")

# llm_with_tools = llm.bind_tools([search_case_docs])
# structured = llm.with_structured_output(MySchema)