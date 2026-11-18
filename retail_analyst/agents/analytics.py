"""Analytics agent: chooses ONE allowlisted pandas operation (as JSON) on the fetched datasets,
or asks the Supervisor for more data. It never generates or executes Python code."""
from __future__ import annotations

import json

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from retail_analyst.agents import describe_datasets, event, parse_json
from retail_analyst.prompts import ANALYTICS_PROMPT
from retail_analyst.state import MAX_ANALYSIS_ATTEMPTS, MAX_DATA_CALLS, AgentState
from retail_analyst.tools.python_tool import dataframe_from_rows, run_analysis


def column_guide(frames: dict) -> str:
    """Numeric vs other columns per dataset, plus one valid example request (a template for a
    small model; the model still chooses the operation and every request is still validated)."""
    lines, example = [], None
    for name, df in frames.items():
        numeric = [c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
        other = [c for c in df.columns if c not in numeric]
        lines.append(f"{name}: numeric columns (for value/x/y) = {numeric}; other columns (for by/period) = {other}")
        if example is None and numeric:
            example = {"operation": "top_n", "dataset": name, "value": numeric[-1], "n": 5}
    if example:
        lines.append(f"Example of a valid request: {json.dumps(example)}")
    lines.append("Parameters value, x, y, by and period must be column NAMES (strings) from the lists above.")
    return "\n".join(lines)


def make_analytics_node(llm):
    def analytics(state: AgentState) -> dict:
        datasets = state.get("datasets", {})
        frames = {name: dataframe_from_rows(d["rows"]) for name, d in datasets.items()}
        task = state.get("task") or state["question"]
        base = (f"Business question: {state['question']}\nYour task: {task}\n\n"
                f"Datasets:\n{describe_datasets(datasets)}\n\n{column_guide(frames)}")
        prompt = base
        analyses = list(state.get("analyses", []))
        errors = list(state.get("errors", []))
        events = []
        update = {"analytics_calls": state.get("analytics_calls", 0) + 1}

        for attempt in range(1, MAX_ANALYSIS_ATTEMPTS + 1):
            try:
                reply = llm.invoke([SystemMessage(ANALYTICS_PROMPT), HumanMessage(prompt)]).content
            except Exception as exc:
                errors.append(f"Analytics could not reach the language model ({type(exc).__name__}).")
                events.append(event(state, "analytics", "llm_error", task=task, error=str(exc)[:200]))
                break

            request = parse_json(reply)
            if request is None:
                problem = "The reply was not a JSON object."
            elif request.get("need_data") and not request.get("operation"):
                if state.get("data_calls", 0) < MAX_DATA_CALLS:
                    events.append(event(state, "analytics", "request_data", task=task,
                                        request=str(request["need_data"])))
                    update["data_request"] = str(request["need_data"])
                    break
                problem = "No more data can be fetched. Use one operation on the existing datasets."
            else:
                request = {k: v for k, v in request.items() if v is not None and k != "need_data"}
                if any(a["request"] == request for a in analyses):
                    # Already computed successfully: do not repeat it.
                    events.append(event(state, "analytics", "duplicate_analysis", task=task, operation=request))
                    break
                result = run_analysis(request, frames)
                events.append(event(state, "analytics", "analysis", task=task, attempt=attempt,
                                    operation=request, result=result.summary, error=result.error))
                if result.ok:
                    analyses.append({"request": request, **result.to_dict()})
                    break
                problem = result.error
            prompt = f"{base}\n\nYour previous reply: {reply}\nProblem: {problem}\nReply with one corrected JSON object."
        else:
            errors.append(f"Analytics failed after {MAX_ANALYSIS_ATTEMPTS} attempts: {task}")

        return {**update, "analyses": analyses, "errors": errors, "trace": events}

    return analytics
