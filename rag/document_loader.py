"""Loads .txt documents from a directory into memory."""

from __future__ import annotations

import os


def load_documents(data_path: str = "data") -> list[str]:
    documents = []

    if not os.path.isdir(data_path):
        return documents

    for file in sorted(os.listdir(data_path)):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(content)

    return documents
