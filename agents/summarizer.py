"""
Summarization agent.

"""

from __future__ import annotations

from functools import lru_cache

from state import AgentState


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline

    return pipeline("summarization", model="facebook/bart-large-cnn")


def summarize(text: str, max_length: int = 100, min_length: int = 30) -> str:
    # BART has a model max input; very short inputs also break
    # min_length > input length, so clamp min_length defensively.
    word_count = len(text.split())
    effective_min = min(min_length, max(5, word_count // 2))
    effective_max = max(max_length, effective_min + 10)

    result = _get_pipeline()(
        text,
        max_length=effective_max,
        min_length=effective_min,
        do_sample=False,
    )
    return result[0]["summary_text"]


def summarizer_node(state: AgentState) -> dict:
    try:
        summary = summarize(state["working_text"])
        return {
            "results": [{"agent": "summarizer", "output": summary}],
            "working_text": summary,  # downstream agents see the summary
            "step": state["step"] + 1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "results": [{"agent": "summarizer", "output": None, "error": str(exc)}],
            "step": state["step"] + 1,
        }
