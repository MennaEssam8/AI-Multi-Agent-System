"""Embedding model wrapper, lazily loaded and cached."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(texts: list[str]):
    return _get_model().encode(texts)
