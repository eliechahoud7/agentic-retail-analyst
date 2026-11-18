"""LangGraph workflow: the Supervisor routes between the Data Analyst and Analytics agents,
then runs its reporting stage.

    START -> supervisor -(next_agent)-> data_analyst -> supervisor
                                     -> analytics    -> supervisor
                                     -> report -> END

Run from the command line:  python -m retail_analyst.graph "Which categories sell the most?"
"""
from __future__ import annotations

import json
import sys
import time

from langgraph.graph import END, START, StateGraph

from retail_analyst.agents.analytics import make_analytics_node
from retail_analyst.agents.data_analyst import make_data_analyst_node
from retail_analyst.agents.supervisor import make_report_node, make_supervisor_node
from retail_analyst.config import get_llm
from retail_analyst.state import RECURSION_LIMIT, AgentState, initial_state


def default_llms() -> dict:
    """Same local model for every agent by default (no model swapping on a 2 GB GPU)."""
    return {
        "supervisor": get_llm("supervisor", json_mode=True, max_tokens=200),
        "data_analyst": get_llm("data_analyst", json_mode=True, max_tokens=300),
        # Only used when the model repeats an already-failed request.
        "data_analyst_retry": get_llm("data_analyst", json_mode=True, max_tokens=300, temperature=0.3),
        "analytics": get_llm("analytics", json_mode=True, max_tokens=200),
    }


def build_graph(llms: dict | None = None):
    llms = llms or default_llms()
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", make_supervisor_node(llms["supervisor"]))
    graph.add_node("data_analyst", make_data_analyst_node(llms["data_analyst"], llms.get("data_analyst_retry")))
    graph.add_node("analytics", make_analytics_node(llms["analytics"]))
    graph.add_node("report", make_report_node())  # Supervisor's final stage (deterministic)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next_agent"],
                                {"data_analyst": "data_analyst", "analytics": "analytics", "report": "report"})
    graph.add_edge("data_analyst", "supervisor")
    graph.add_edge("analytics", "supervisor")
    graph.add_edge("report", END)
    return graph.compile()


def ask(question: str, graph=None) -> AgentState:
    graph = graph or build_graph()
    return graph.invoke(initial_state(question), {"recursion_limit": RECURSION_LIMIT})


def format_trace(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        head = f"[{e['step']}] {e['agent']}.{e['action']}"
        if e["action"] == "route":
            lines.append(f"{head} -> {e['next']} ({e['reason']}) | task: {e['task']}")
        elif e["action"] == "query":
            status = f"{e['rows']} rows" if "rows" in e else f"ERROR: {e.get('error')}"
            sql = " ".join(e.get("sql", "").split())
            lines.append(f"{head} attempt {e['attempt']}: {status}\n      request: {json.dumps(e.get('spec'))}"
                         + "".join(f"\n      corrected: {c}" for c in e.get("corrections", []))
                         + (f"\n      sql: {sql[:300]}" if sql else ""))
        elif e["action"] == "analysis":
            status = e.get("result") or f"ERROR: {e.get('error')}"
            lines.append(f"{head} attempt {e['attempt']}: {json.dumps(e['operation'])}\n      -> {status}")
        elif e["action"] == "request_data":
            lines.append(f"{head}: {e['request']}")
        elif e["action"] == "duplicate_request":
            lines.append(f"{head} attempt {e['attempt']}: repeated an already-failed request (not run)")
        elif e["action"] == "duplicate_analysis":
            lines.append(f"{head}: {json.dumps(e['operation'])} already computed (skipped)")
        else:
            lines.append(f"{head}: {e.get('result') or e.get('error', '')}")
    return "\n".join(lines)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What were the top 5 product categories by sales revenue?"
    start = time.perf_counter()
    result = ask(question)
    print(f"QUESTION: {question}\n\n--- AGENT TRACE ---\n{format_trace(result['trace'])}")
    print(f"\n--- FINAL ANSWER ({time.perf_counter() - start:.0f}s) ---\n{result['final_answer']}")
