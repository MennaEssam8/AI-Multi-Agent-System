"""
Answer generation for the RAG agent.

"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_llm():
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )

    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.2, max_tokens=500)


_SYSTEM_PROMPT = """You are a question-answering assistant. Answer the
user's question using ONLY the provided context. If the context does
not contain enough information to answer, say so plainly rather than
guessing."""


def generate_answer(question: str, context: str) -> str:
    if not context.strip():
        return "I don't have any relevant documents to answer that question."

    llm = _get_llm()
    human_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = llm.invoke(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", human_prompt),
        ]
    )
    return response.content
