"""Splits raw documents into overlapping chunks for retrieval."""

from __future__ import annotations


def chunk_documents(documents: list[str], chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text("\n".join(documents))
