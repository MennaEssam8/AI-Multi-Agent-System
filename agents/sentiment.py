"""
Sentiment analysis agent.

The HuggingFace pipeline is created lazily (on first use) and cached,
rather than at import time. This matters once this module is imported
as part of a graph that might route to *other* agents most of the
time -- we don't want to pay model-load cost for an agent that never
runs in a given request.
"""

from __future__ import annotations

from functools import lru_cache

from state import AgentState


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def analyze_sentiment(text: str) -> dict:
    result = _get_pipeline()(text)[0]
    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 4),
    }


def sentiment_node(state: AgentState) -> dict:
    """LangGraph node wrapper. Reads working_text, appends result."""
    try:
        output = analyze_sentiment(state["working_text"])
        return {
            "results": [{"agent": "sentiment", "output": output}],
            "step": state["step"] + 1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "results": [{"agent": "sentiment", "output": None, "error": str(exc)}],
            "step": state["step"] + 1,
        }
