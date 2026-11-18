"""Data Analyst agent: chooses ONE business metric and grouping from the verified catalog
(retail_analyst/catalog.py). Code compiles the query, applying the year, "top N", sort order
and calculation read from the question (retail_analyst/question.py); the safe SQL tool runs it
read-only. The model never writes SQL or chooses tables and columns.

Retry policy (MAX_SQL_ATTEMPTS attempts per call):
- Failed requests and their exact errors are kept in state["sql_failures"], so later attempts
  AND later Data Analyst calls see what already failed.
- A request identical to an earlier failure is never run again. The model is asked once more
  (retry model, temperature 0.3); if it still repeats, the call stops.
- {"metric": null} means no catalog metric fits the question: the system refuses to answer.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from retail_analyst.agents import event, parse_json
from retail_analyst.catalog import UnsupportedRequest, catalog_prompt, check_fit, compile_query
from retail_analyst.prompts import DATA_ANALYST_PROMPT
from retail_analyst.question import parse_question
from retail_analyst.state import MAX_SQL_ATTEMPTS, AgentState
from retail_analyst.tools.sql_tool import run_sql

MAX_FAILURES_SHOWN = 2
OUT_OF_SCOPE = "The question does not match any supported business metric."


def canonical(spec) -> str:
    """Comparable form of a request: key order and whitespace ignored."""
    if isinstance(spec, dict):
        return json.dumps(spec, sort_keys=True, default=str).lower()
    return " ".join(str(spec).lower().split())


def correction_prompt(base: str, failures: list[dict]) -> str:
    """Failed requests with their exact errors (the catalog errors list the allowed choices)."""
    if not failures:
        return base
    shown = failures[-MAX_FAILURES_SHOWN:]
    listing = "\n".join(f"{i}. {f['request']}\n   Error: {f['error']}" for i, f in enumerate(shown, 1))
    return (f"{base}\n\nThese requests already FAILED. Do not repeat them:\n{listing}\n\n"
            "Write a different, corrected JSON request that fixes the last error.")


def make_data_analyst_node(llm, retry_llm=None):
    retry_llm = retry_llm or llm

    def data_analyst(state: AgentState) -> dict:
        question = state["question"]
        task = state.get("task") or question
        constraints = parse_question(question)
        system = SystemMessage(DATA_ANALYST_PROMPT.format(catalog=catalog_prompt()))
        # Until data exists, the request is based on the user's question only: Supervisor task
        # texts were observed adding details the user never asked for (periods, groupings).
        # Follow-up fetches (e.g. requested by Analytics) include the task.
        base = f"Business question: {question}"
        if state.get("datasets"):
            base += f"\nAdditional data needed: {task}"
        datasets = dict(state.get("datasets", {}))
        failures = list(state.get("sql_failures", []))
        errors = list(state.get("errors", []))
        events, update = [], {}

        def generate(model, prompt: str) -> tuple[dict | None, str]:
            reply = model.invoke([system, HumanMessage(prompt)]).content
            spec = parse_json(reply)
            return spec, (canonical(spec) if spec is not None else canonical(reply))

        for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
            prompt = correction_prompt(base, failures)
            failed = {f["key"] for f in failures}
            try:
                spec, key = generate(llm, prompt)
                if key in failed:
                    events.append(event(state, "data_analyst", "duplicate_request", task=task,
                                        attempt=attempt, spec=spec))
                    spec, key = generate(retry_llm, prompt + "\nYour last answer repeated a request that "
                                                              "already failed. Write a DIFFERENT request.")
            except Exception as exc:  # Ollama unreachable, timeout, ...
                errors.append(f"Data Analyst could not reach the language model ({type(exc).__name__}).")
                events.append(event(state, "data_analyst", "llm_error", task=task, error=str(exc)[:200]))
                break
            if key in failed:
                events.append(event(state, "data_analyst", "duplicate_request", task=task, attempt=attempt, spec=spec))
                errors.append("Data Analyst kept proposing a request that had already failed; stopped retrying.")
                update["data_stalled"] = True  # another call would repeat it too (seen in every run)
                break
            if isinstance(spec, dict) and "metric" in spec and spec["metric"] in (None, "", "null", "none"):
                events.append(event(state, "data_analyst", "out_of_scope", task=task, attempt=attempt, spec=spec))
                update["scope_reason"] = OUT_OF_SCOPE
                break

            sql, rows, columns, truncated, notes = "", [], [], False, []
            try:
                if spec is None:
                    raise UnsupportedRequest("The reply was not a JSON object.")
                compiled = compile_query(spec, constraints)
                check_fit(compiled.metric, compiled.grouping, question)  # the choice must fit the question
                sql, notes = compiled.sql, list(compiled.notes)
                result = run_sql(sql)
                problem = result.error if not result.ok else (
                    "The query returned no rows for this question." if result.row_count == 0 else None)
                if result.ok:
                    rows, columns, truncated = result.rows, result.columns, result.truncated
            except UnsupportedRequest as exc:
                problem = f"Invalid request: {exc}"
            events.append(event(state, "data_analyst", "query", task=task, attempt=attempt, spec=spec,
                                corrections=notes, sql=sql, rows=len(rows) if problem is None else None,
                                error=problem))
            if problem is None:
                name = f"d{len(datasets) + 1}"
                applied = {"metric": compiled.metric, "group_by": compiled.grouping,
                           "calculation": compiled.calculation, "years": list(constraints.years),
                           "ascending": constraints.ascending, "top_n": constraints.top_n}
                datasets[name] = {"task": task, "request": applied, "sql": sql, "columns": columns,
                                  "rows": rows, "row_count": len(rows), "truncated": truncated,
                                  "corrections": notes}
                break
            failures.append({"key": key, "request": json.dumps(spec) if spec is not None else "(not JSON)",
                             "error": problem})
        else:
            errors.append(f"Data Analyst failed after {MAX_SQL_ATTEMPTS} attempts: {task}")

        return {**update, "datasets": datasets, "data_calls": state.get("data_calls", 0) + 1,
                "sql_failures": failures, "errors": errors, "trace": events}

    return data_analyst
