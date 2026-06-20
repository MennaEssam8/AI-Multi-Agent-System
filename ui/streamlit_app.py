"""
Streamlit UI for the multi-agent NLP system.

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Allow `import graph`, `import state`, etc. when run as
# `streamlit run ui/streamlit_app.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import run  # noqa: E402

st.set_page_config(page_title="Multi-Agent NLP System", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Agent NLP System")
st.caption(
    "Routed by a free Groq-hosted LLM. Supports sentiment analysis, "
    "named entity recognition, and document Q&A (RAG)."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: input, plan, response, error

with st.sidebar:
    st.subheader("Available agents")
    st.markdown(
        "- **sentiment** — polarity & confidence\n"
        "- **ner** — named entities\n"
        "- **rag** — Q&A over `data/*.txt`"
    )
    st.divider()
    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY is not set. Copy .env.example to .env first.")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

user_input = st.text_area(
    "What would you like done?",
    placeholder=(
        "e.g. 'What's the sentiment of: I love this product!' or "
        "'Check the sentiment of this sentence.'"
    ),
    height=100,
)

if st.button("Run", type="primary", disabled=not user_input.strip()):
    with st.spinner("Routing and running agents..."):
        try:
            final_state = run(user_input)
            st.session_state.history.insert(
                0,
                {
                    "input": user_input,
                    "plan": final_state.get("plan", []),
                    "response": final_state.get("final_response", ""),
                    "error": final_state.get("error"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")

st.divider()

for item in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**You:** {item['input']}")
        if item["plan"]:
            st.caption(f"Plan: {' → '.join(item['plan'])}")
        if item["error"]:
            st.error(item["response"])
        else:
            st.markdown(item["response"])
