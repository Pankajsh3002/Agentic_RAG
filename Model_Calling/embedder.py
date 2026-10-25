import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DIMENSIONS = 3072   # must match halfvec(3072) in the database
MAX_PARALLEL = 5            # simultaneous requests (one text per request)


# ---------- embedding wrapper ----------
class EmbeddingService:
    def __init__(self, model_name: str | None = None, dimensions: int | None = None, timeout: int = 60):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

        model_name = model_name or os.getenv("EMBED_MODEL")
        if not model_name:
            raise ValueError("model_name not provided and EMBED_MODEL not found in .env")

        self.model_name = model_name
        self.dimensions = dimensions or int(os.getenv("EMBED_DIM", DEFAULT_DIMENSIONS))
        self.client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=timeout, max_retries=3)

    def _embed_one(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=text,                  # a single string, not a list
            dimensions=self.dimensions,
        )
        return resp.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Embedding %d text(s) with %s (%d dims)", len(texts), self.model_name, self.dimensions)

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            vectors = list(pool.map(self._embed_one, texts))   # map keeps the input order

        if len(vectors[0]) != self.dimensions:
            raise ValueError(
                f"Model '{self.model_name}' returned {len(vectors[0])} dimensions but "
                f"{self.dimensions} were expected. Use a model that supports 'dimensions', "
                f"or change the dimension and the vector size in the database."
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embedder(model_name: str | None = None, dimensions: int | None = None, timeout: int = 60) -> EmbeddingService:
    return EmbeddingService(model_name, dimensions, timeout)


# ---------- one-shot functions ----------
def embed_texts(texts: list[str], model_name: str | None = None,
                dimensions: int | None = None, timeout: int = 60) -> list[list[float]]:
    return EmbeddingService(model_name, dimensions, timeout).embed_texts(texts)


def embed_query(text: str, model_name: str | None = None,
                dimensions: int | None = None, timeout: int = 60) -> list[float]:
    return EmbeddingService(model_name, dimensions, timeout).embed_query(text)