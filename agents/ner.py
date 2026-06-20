"""
Named Entity Recognition agent.

Lazy-loaded HF pipeline, grouped/cleaned output (the raw HF NER
pipeline emits one entry per *token*, which fragments multi-word
entities like "New York" into "New" + "York" -- grouped_entities=True
merges those back into single spans).
"""

from __future__ import annotations

from functools import lru_cache

from state import AgentState


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline

    return pipeline("ner", grouped_entities=True)


def extract_entities(text: str) -> list[dict]:
    results = _get_pipeline()(text)

    entities = []
    for item in results:
        entities.append(
            {
                "entity": item.get("entity_group", item.get("entity")),
                "word": item["word"],
                "score": round(item["score"], 4),
            }
        )
    return entities


def ner_node(state: AgentState) -> dict:
    try:
        output = extract_entities(state["working_text"])
        return {
            "results": [{"agent": "ner", "output": output}],
            "step": state["step"] + 1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "results": [{"agent": "ner", "output": None, "error": str(exc)}],
            "step": state["step"] + 1,
        }
