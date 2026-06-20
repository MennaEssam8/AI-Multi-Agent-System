"""
Shared state schema for the multi-agent graph.

Every node in the LangGraph reads from and writes to this single
state object. Keeping it centralized means any node can see what
previous nodes produced, and the router can make a multi-step plan
(e.g. "summarize, then run sentiment on the summary") without each
agent needing to know about the others.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
import operator

# The full set of tasks any agent node in this graph can perform.
TaskName = Literal["sentiment", "ner", "summarizer", "rag", "unsupported"]


class AgentState(TypedDict):
    # --- input ---
    input: str  # the raw user request, unchanged

    # --- routing ---
    # The router fills this in. It's a *list* of tasks because some
    # requests need more than one agent (e.g. "summarize this and tell
    # me the sentiment" -> ["summarizer", "sentiment"]).
    plan: list[TaskName]

    # Index into `plan` of the task we're about to run / just ran.
    # Lets conditional edges decide whether to loop back to the router
    # dispatch step or finish.
    step: int

    # --- working data ---
    # The text each agent should actually operate on. Usually equals
    # `input`, but if summarizer ran first, sentiment may want to run
    # on the summary instead of the raw input.
    working_text: str

    # --- results ---
    # Annotated with operator.add so that nodes running in parallel
    # branches (if you fan out conditional edges) merge cleanly instead
    # of overwriting each other. Each agent appends one dict like:
    # {"agent": "sentiment", "output": {...}}
    results: Annotated[list[dict[str, Any]], operator.add]

    # --- output ---
    final_response: str  # what the UI actually displays

    # --- diagnostics ---
    error: str | None
