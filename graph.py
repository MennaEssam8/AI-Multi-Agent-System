"""
LangGraph wiring.

Flow:

    START -> router -> dispatch -> [sentiment | ner | summarizer | rag] -+
                           ^                                             |
                           |_____________________________________________|
                           (loop back until plan is exhausted)
                                       |
                                       v
                                    finalize -> END

`dispatch` is a conditional edge (not a real node with logic) that
looks at state["plan"] and state["step"] to decide which agent node
to send execution to next, or whether to move on to `finalize`. This
is what makes multi-step plans like ["summarizer", "sentiment"] work:
after summarizer_node runs and increments step, dispatch sends
execution to sentiment_node, which now sees working_text == the
summary, not the original input.

The compiled graph is exposed as `build_graph()`. Each node module
(agents/*.py) lazy-loads its underlying model, so calling build_graph()
itself is cheap -- no model is loaded until a request actually needs
it.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from state import AgentState
from router import route_node
from agents.sentiment import sentiment_node
from agents.ner import ner_node
from agents.summarizer import summarizer_node
from agents.rag_agent import rag_node

_NODE_MAP = {
    "sentiment": "sentiment",
    "ner": "ner",
    "summarizer": "summarizer",
    "rag": "rag",
}


def _dispatch(state: AgentState) -> str:
    """Conditional edge: decide the next node based on plan/step."""
    plan = state.get("plan", [])
    step = state.get("step", 0)

    if step >= len(plan):
        return "finalize"

    next_task = plan[step]
    return _NODE_MAP.get(next_task, "finalize")


def _finalize_node(state: AgentState) -> dict:
    """Builds the human-readable final_response from accumulated results."""
    # route_node may have already set final_response (e.g. for an
    # unsupported / failed request) -- don't overwrite that.
    if state.get("final_response"):
        return {}

    results = state.get("results", [])
    if not results:
        return {"final_response": "No results were produced."}

    lines = []
    for item in results:
        agent = item["agent"]
        if item.get("error"):
            lines.append(f"[{agent}] failed: {item['error']}")
            continue

        output = item["output"]
        if agent == "sentiment":
            lines.append(
                f"[sentiment] {output['sentiment']} (confidence: {output['confidence']})"
            )
        elif agent == "ner":
            if not output:
                lines.append("[ner] No entities found.")
            else:
                entities = ", ".join(f"{e['word']} ({e['entity']})" for e in output)
                lines.append(f"[ner] {entities}")
        elif agent == "summarizer":
            lines.append(f"[summarizer] {output}")
        elif agent == "rag":
            lines.append(f"[rag] {output}")
        else:
            lines.append(f"[{agent}] {output}")

    return {"final_response": "\n\n".join(lines)}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", route_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("ner", ner_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("rag", rag_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("router")

    # After routing, dispatch to the first task (or straight to finalize
    # if the plan is empty / routing failed).
    graph.add_conditional_edges(
        "router",
        _dispatch,
        {
            "sentiment": "sentiment",
            "ner": "ner",
            "summarizer": "summarizer",
            "rag": "rag",
            "finalize": "finalize",
        },
    )

    # After each agent runs, dispatch again: either the next step in the
    # plan, or finalize if the plan is exhausted. This is what enables
    # chained multi-agent plans.
    for node_name in ["sentiment", "ner", "summarizer", "rag"]:
        graph.add_conditional_edges(
            node_name,
            _dispatch,
            {
                "sentiment": "sentiment",
                "ner": "ner",
                "summarizer": "summarizer",
                "rag": "rag",
                "finalize": "finalize",
            },
        )

    graph.add_edge("finalize", END)

    return graph.compile()


def run(user_input: str) -> AgentState:
    """Convenience entry point: run a single user request through the graph."""
    app = build_graph()
    initial_state: AgentState = {
        "input": user_input,
        "plan": [],
        "step": 0,
        "working_text": user_input,
        "results": [],
        "final_response": "",
        "error": None,
    }
    return app.invoke(initial_state)
