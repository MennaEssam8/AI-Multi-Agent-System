"""Similarity search over the FAISS index."""

from __future__ import annotations

import numpy as np


def retrieve(query_embedding, index, documents: list[str], k: int = 3) -> list[str]:
    query_embedding = np.array(query_embedding, dtype="float32")

    # k can't exceed the number of indexed documents
    k = min(k, index.ntotal)
    if k == 0:
        return []

    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]
