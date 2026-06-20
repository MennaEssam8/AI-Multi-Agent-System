"""
LLM-based router.

Replaces the old HuggingFace zero-shot classifier. Instead of forcing
the request into one of two hardcoded labels, this asks an LLM to read
the request and return a structured plan: an ordered list of agents to
run. This means:

  - New agents are just a new line in the prompt + state.py Literal,
    not a retrain.
  - Multi-step requests work: "summarize this article and check the
    sentiment" -> plan = ["summarizer", "sentiment"].
  - Ambiguous/unsupported requests are caught explicitly instead of
    silently defaulting to the wrong label.

Structured output is enforced via tool-calling, so we get back a real
list of valid task names, not free text we have to regex.

Uses Groq (free tier) running Llama 3.3 70B instead of a paid API.
Groq's OpenAI-compatible tool-calling is reliable enough for this
small, fixed set of routing labels. Get a free key at
https://console.groq.com/keys
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from state import AgentState, TaskName

load_dotenv()

VALID_TASKS: list[TaskName] = ["sentiment", "ner", "summarizer", "rag"]

_ROUTE_TOOL = {
    "name": "route",
    "description": (
        "Decide which agent(s) should handle the user's request, in "
        "the order they should run."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": VALID_TASKS,
                },
                "description": (
                    "Ordered list of agents to invoke. Use multiple "
                    "entries only if the request genuinely needs "
                    "more than one step (e.g. summarize then analyze "
                    "sentiment of the summary)."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence on why this plan was chosen.",
            },
        },
        "required": ["plan", "reasoning"],
    },
}

_SYSTEM_PROMPT = f"""You are a routing component in a multi-agent NLP system.
Given a user request, decide which of the following agents should handle it,
in order:

- sentiment: detect sentiment/emotion/polarity of a piece of text.
- ner: extract named entities (people, places, organizations, dates, etc.).
- summarizer: condense a long piece of text into a shorter summary.
- rag: answer a question using the system's document knowledge base
  (use this whenever the user is *asking a question* that requires
  looked-up information, rather than analyzing a piece of text they gave you).

Rules:
- Only choose from: {VALID_TASKS}.
- Most requests need exactly one agent. Only return multiple if the
  request explicitly chains steps (e.g. "summarize this, then tell me
  the sentiment of the summary").
- If the request doesn't match any agent, return an empty plan.
Always call the `route` tool with your decision."""


def _get_router_llm() -> ChatGroq:
    """Lazily construct the router LLM so importing this module never
    requires an API key (useful for tests / offline agent unit tests)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
        max_tokens=300,
    ).bind_tools([_ROUTE_TOOL], tool_choice="route")


def route_node(state: AgentState) -> dict:
    """Entry node of the graph. Reads state['input'], asks Claude for a
    plan, and writes plan/step/working_text back into state."""
    try:
        llm = _get_router_llm()
        response = llm.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", state["input"]),
            ]
        )
        tool_calls = response.tool_calls
        if not tool_calls:
            raise ValueError("Router did not return a tool call.")

        args = tool_calls[0]["args"]
        plan = [t for t in args.get("plan", []) if t in VALID_TASKS]

        if not plan:
            return {
                "plan": [],
                "step": 0,
                "working_text": state["input"],
                "final_response": (
                    "I couldn't match that request to any available "
                    "agent (sentiment, ner, summarizer, rag). Could you "
                    "rephrase what you'd like done?"
                ),
                "error": None,
            }

        return {
            "plan": plan,
            "step": 0,
            "working_text": state["input"],
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001 - surface any failure into state
        return {
            "plan": [],
            "step": 0,
            "working_text": state["input"],
            "final_response": f"Routing failed: {exc}",
            "error": str(exc),
        }
