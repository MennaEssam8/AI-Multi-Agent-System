# AI Multi-Agent System

A multi-agent NLP system orchestrated with **LangGraph**, routed by an
LLM (**Llama 3.3 70B via Groq's free API**, using `langchain-groq`)
instead of a hardcoded if/elif chain or a brittle zero-shot classifier.
No paid API required.

## Architecture

```
START -> router -> dispatch -> [sentiment | ner | summarizer | rag] -> finalize -> END
                       ^                       |
                       |_______________________|
                     (loops until plan is exhausted)
```

- **`state.py`** — `AgentState`, the shared TypedDict passed between
  every node in the graph.
- **`router.py`** — a free Groq-hosted LLM reads the user's request
  and returns a structured *plan*: an ordered list of agents to run.
  Supports chained requests (e.g. "summarize this, then check its
  sentiment").
- **`graph.py`** — builds and compiles the LangGraph `StateGraph`,
  wiring the router, agent nodes, and a `finalize` node that turns
  accumulated results into a human-readable response.
- **`agents/`** — one module per capability. Each exposes a plain
  Python function (e.g. `analyze_sentiment`) and a `*_node` wrapper
  for the graph. Models are lazy-loaded and cached on first use.
- **`rag/`** — retrieval pipeline (chunking, embeddings via
  `sentence-transformers`, FAISS index, retrieval, and answer
  generation via the same free Groq LLM) used by the `rag` agent.
- **`app.py`** — CLI entry point.
- **`ui/streamlit_app.py`** — Streamlit chat-style UI.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your free GROQ_API_KEY (console.groq.com/keys)
```

## Usage

CLI:
```bash
python app.py "What's the sentiment of: I love this product!"
python app.py   # interactive mode
```

Streamlit:
```bash
streamlit run ui/streamlit_app.py
```

## Adding a new agent

1. Create `agents/your_agent.py` with a plain function plus a
   `your_agent_node(state) -> dict` wrapper (see `agents/sentiment.py`
   for the pattern — lazy model loading, append to `results`, bump
   `step`).
2. Add `"your_agent"` to `TaskName` in `state.py` and `VALID_TASKS` in
   `router.py`, and describe it in the router's system prompt.
3. Register the node and its conditional edges in `graph.py`.

Because routing is decided by an LLM reading a prompt (not a trained
classifier), this only requires a prompt update — no retraining.

## Notes

- Groq's free tier has generous but real rate limits (requests/tokens
  per minute). If you hit a rate-limit error, wait a moment and retry,
  or check current limits at https://console.groq.com/settings/limits
- Documents for the `rag` agent live in `data/*.txt`. The FAISS index
  is built lazily on first RAG query and cached; call
  `agents.rag_agent.reset_index()` if you add/change documents at
  runtime and want a rebuild.
- Individual agent nodes are designed so each can later be swapped for
  a compiled LangGraph subgraph without changing the rest of the
  system — they're plain functions returning state-shaped dicts.
