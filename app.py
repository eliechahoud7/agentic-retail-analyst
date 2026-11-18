"""Streamlit chat UI for the Agentic Retail Data Analyst.

Run with:  streamlit run app.py

Users see only the business answer, or a short message when no reliable answer could be
produced. All safeguards and checks run inside the 3-agent LangGraph workflow; this page only
decides how their outcome is presented (see retail_analyst/ui_support.py). Diagnostics (agent
trace, checks, SQL) are written to the application log for developers, never to the page.
"""
from __future__ import annotations

import logging
import time

import streamlit as st

from retail_analyst.graph import format_trace
from retail_analyst.ui_support import (SERVICE_UNAVAILABLE, answer_status, escape_markdown, retry_summary,
                                       run_question, safe_error, service_available, user_answer)

log = logging.getLogger("retail_analyst.app")
if not log.handlers:  # Streamlit reruns the script; configure the logger only once
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

EXAMPLES = [
    "What are the top 5 product categories by sales revenue?",
    "What share of payment value comes from each payment type?",
    "Which customer states have the highest late delivery rate?",
]

st.set_page_config(page_title="Retail Data Analyst", page_icon="📊", layout="centered")


@st.cache_resource(show_spinner=False)
def load_graph():
    from retail_analyst.graph import build_graph  # local Ollama models from .env
    return build_graph()


def get_graph():
    # Tests can inject a graph (e.g. with scripted models) through session state.
    return st.session_state.get("graph") or load_graph()


@st.cache_data(ttl=60, show_spinner=False)
def check_service() -> bool:
    return service_available()


def log_diagnostics(question: str, state: dict, seconds: float) -> None:
    """Developer diagnostics: everything the user does not see."""
    log.info("Question: %s | status: %s | %.1f s | %s", question, answer_status(state), seconds,
             retry_summary(state.get("trace", [])))
    for issue in state.get("issues", []):
        log.info("Check [%s]: %s", issue.get("kind"), issue.get("message"))
    log.info("Agent trace:\n%s", format_trace(state.get("trace", [])))


def answer(question: str) -> dict:
    """Run the agents and return what the user should see."""
    start = time.perf_counter()
    try:
        state = run_question(get_graph(), question)
    except Exception as exc:  # infrastructure failure (database or Ollama down)
        log.error("Question failed: %s | %s", question, safe_error(exc))
        return {"reliable": False, "markdown": SERVICE_UNAVAILABLE}
    log_diagnostics(question, state, time.perf_counter() - start)
    reliable, markdown = user_answer(state)
    return {"reliable": reliable, "markdown": markdown}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Retail Data Analyst")
    st.markdown("Ask business questions about sales, products, payments, deliveries and customer "
                "reviews of the Olist e-commerce store (2016–2018).")
    if not check_service():
        st.warning(SERVICE_UNAVAILABLE)
    st.subheader("Try a question")
    example = None
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"example_{i}", width="stretch"):
            example = q
    if st.button("Clear conversation", key="clear"):
        st.session_state.messages = []

st.title("📊 Retail Data Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask a business question, e.g. 'Which payment types are most used?'") or example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(escape_markdown(message["content"]))

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))
    with st.chat_message("assistant"):
        with st.spinner("Analysing your question…"):
            result = answer(question)
        st.markdown(escape_markdown(result["markdown"]))
    st.session_state.messages.append({"role": "assistant", "content": result["markdown"],
                                      "reliable": result["reliable"]})
