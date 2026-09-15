"""Local embedding model wrapper.

Uses sentence-transformers so the pipeline doesn't burn API quota just to
embed thousands of chunks, and works fully offline once the model is cached.
"""
from __future__ import annotations

from functools import lru_cache

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
