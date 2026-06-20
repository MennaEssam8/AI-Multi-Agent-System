"""
RAG (retrieval-augmented generation) agent.

Original bug: this module built the FAISS index at *import time*,
meaning every import re-embedded and re-indexed all documents -- slow,
and wasteful when this is one of several agents in a graph that might
not even need RAG for a given request.

Fixed by lazily building the index once, on first use, and caching it.
Call `reset_index()` if documents change at runtime (e.g. new file
uploaded) and you want a rebuild.
"""

from __future__ import annotations

from rag.embeddings import get_embeddings
from rag.vector_store import create_index
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.document_loader import load_documents
from rag.chunker import chunk_documents

from state import AgentState

_index = None
_documents: list[str] = []


def _ensure_index_built(data_path: str = "data") -> None:
    global _index, _documents

    if _index is not None:
        return

    raw_docs = load_documents(data_path)
    _documents = chunk_documents(raw_docs) if raw_docs else []

    if not _documents:
        _index = None
        return

    doc_embeddings = get_embeddings(_documents)
    _index = create_index(doc_embeddings)


def reset_index() -> None:
    """Force the index to rebuild on next call (e.g. after adding docs)."""
    global _index, _documents
    _index = None
    _documents = []


def rag_answer(question: str, data_path: str = "data") -> str:
    _ensure_index_built(data_path)

    if _index is None:
        return "No documents are loaded yet, so I can't answer that."

    query_embedding = get_embeddings([question])
    context_docs = retrieve(query_embedding, _index, _documents)
    context = "\n".join(context_docs)

    return generate_answer(question, context)


def rag_node(state: AgentState) -> dict:
    try:
        answer = rag_answer(state["working_text"])
        return {
            "results": [{"agent": "rag", "output": answer}],
            "step": state["step"] + 1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "results": [{"agent": "rag", "output": None, "error": str(exc)}],
            "step": state["step"] + 1,
        }
