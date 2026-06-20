"""FAISS vector store helpers."""

from __future__ import annotations

import numpy as np


def create_index(embeddings):
    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    import faiss

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
