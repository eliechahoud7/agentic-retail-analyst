"""Shared LangGraph state and the workflow limits that guarantee termination."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

MAX_ROUTING_STEPS = 6       # supervisor decisions before the report is forced
MAX_DATA_CALLS = 2          # Data Analyst runs per question (initial fetch + one extra request)
MAX_SQL_ATTEMPTS = 3        # per Data Analyst run: 1 attempt + 2 retries on SQL errors
MAX_ANALYTICS_CALLS = 2     # Analytics runs per question
MAX_ANALYSIS_ATTEMPTS = 2   # per Analytics run: 1 attempt + 1 retry on invalid requests
# Hard backstop: worst case is 2 nodes per routing step + report, well below this.
RECURSION_LIMIT = 2 * MAX_ROUTING_STEPS + 6


class AgentState(TypedDict, total=False):
    question: str                    # the user's business question
    next_agent: str                  # supervisor decision: data_analyst | analytics | report
    task: str                        # instruction handed to the next agent
    datasets: dict[str, dict]        # name -> {task, sql, columns, rows, row_count, truncated}
    analyses: list[dict]             # results of allowlisted operations (with the request)
    data_request: str | None         # set by Analytics when it needs more data
    scope_reason: str | None         # why the question cannot be answered from the data (-> refusal)
    data_stalled: bool               # the Data Analyst stopped because it kept repeating a failed request
    sql_failures: list[dict]         # [{sql, error}] of failed queries, shared across Data Analyst calls
    data_calls: int
    analytics_calls: int
    step: int                        # number of supervisor routing decisions so far
    errors: list[str]
    checks: list[str]                # caveats found in the final consistency checks
    issues: list[dict]               # the same findings classified: {kind: data|text|info, message}
    trace: Annotated[list[dict], operator.add]  # append-only execution trace
    answer_body: str                 # the written answer only (no evidence/caveat sections)
    final_answer: str                # answer + evidence + caveats, as one markdown text


def initial_state(question: str) -> AgentState:
    return AgentState(question=question.strip(), datasets={}, analyses=[], data_request=None, sql_failures=[],
                      data_calls=0, analytics_calls=0, step=0, errors=[], checks=[], trace=[])
