"""Supervisor agent: routes work between the specialists, then (as its final stage) checks the
evidence for consistency and writes the business answer. There is no separate reporting agent."""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from retail_analyst.agents import describe_datasets, event, parse_json
from retail_analyst.catalog import unexplained_words
from retail_analyst.question import parse_question, read_ranges
from retail_analyst.summary import answer_data, summary_text
from retail_analyst.prompts import SUPERVISOR_PROMPT
from retail_analyst.state import MAX_ANALYTICS_CALLS, MAX_DATA_CALLS, MAX_ROUTING_STEPS, AgentState

AGENTS = ("data_analyst", "analytics", "report")
COMPUTED_IN_QUERY = ("ranking", "share", "growth")


# ---------------------------------------------------------------------------
# Stage 1: routing
# ---------------------------------------------------------------------------
# Calculations the catalog query does not provide; only these justify the Analytics agent.
_NEEDS_ANALYTICS = re.compile(r"\b(correlat\w*|median|variance|standard deviation|spread)\b", re.IGNORECASE)


def allowed_next(state: AgentState) -> list[str]:
    """Which handoffs are legal now, given the evidence collected and the limits."""
    datasets = state.get("datasets", {})
    analysed = {a["request"].get("dataset") for a in state.get("analyses", [])}
    # Rankings, shares and growth are already computed (in SQL) by the catalog: analysing them
    # again is redundant and can produce nonsense (e.g. the growth of a growth percentage).
    open_datasets = [name for name, d in datasets.items() if name not in analysed
                     and (d.get("request") or {}).get("calculation") not in COMPUTED_IN_QUERY]
    options = []
    # A catalog dataset answers the question; fetch again only when Analytics asked for more data
    # (the benchmark showed identical second requests that only cost time).
    if state.get("data_calls", 0) < MAX_DATA_CALLS and (not datasets or state.get("data_request")):
        options.append("data_analyst")
    # Analytics only when the question asks for a calculation the catalog does not provide,
    # and only while some dataset still needs it (no repeated analyses).
    if (open_datasets and state.get("analytics_calls", 0) < MAX_ANALYTICS_CALLS
            and _NEEDS_ANALYTICS.search(state.get("question", ""))):
        options.append("analytics")
    if datasets or not options:
        options.append("report")
    return options


_STOPWORDS = {"what", "which", "have", "does", "from", "each", "with", "there", "that", "this",
              "between", "during", "data", "fetch", "much", "many", "were", "into", "their"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def task_is_specific(task: str, question: str) -> bool:
    """A task must be a real sentence that keeps at least two key words of the question."""
    overlap = _content_words(task) & _content_words(question)
    return len(task.split()) >= 6 and len(overlap) >= min(2, len(_content_words(question)))


def fallback_task(agent: str, question: str) -> str:
    if agent == "data_analyst":
        return f"Fetch the data, with the metric, grouping and time period it asks for, to answer: {question}"
    return f"Choose the one calculation on the fetched data that best answers: {question}"


def _status(state: AgentState) -> str:
    analyses = state.get("analyses", [])
    done = "\n".join(f"- {a['operation']}: {a['summary']}" for a in analyses) or "(none)"
    return (f"Datasets fetched:\n{describe_datasets(state.get('datasets', {}), sample_rows=2)}\n"
            f"Analyses done:\n{done}")


def make_supervisor_node(llm):
    def supervisor(state: AgentState) -> dict:
        step = state.get("step", 0) + 1
        view = {**state, "step": step}
        question = state["question"]
        options = allowed_next(state)
        update: dict = {"step": step}

        def decide(nxt: str, task: str, reason: str) -> dict:
            return {**update, "next_agent": nxt, "task": task,
                    "trace": [event(view, "supervisor", "route", next=nxt, task=task, reason=reason)]}

        # Hard rules first (no LLM call needed).
        if step > MAX_ROUTING_STEPS:
            return decide("report", "Write the final answer.", f"step limit ({MAX_ROUTING_STEPS}) reached")
        if not state.get("datasets"):
            if refusal := parse_question(question).refusal:  # causes, forecasts, missing periods
                update["scope_reason"] = refusal
                return decide("report", "Explain that the question cannot be answered from the data.",
                              "question cannot be answered from the data")
            if unknown := unexplained_words(question):  # closed world: no silently ignored conditions
                quoted = ", ".join(f'"{w}"' for w in unknown[:3])
                update["scope_reason"] = (
                    f"The question uses {quoted}, which I cannot match to a supported measure, grouping or "
                    "period, so an answer could ignore part of the question. Supported measures include revenue, "
                    "units sold, orders, customers, payments, delivery time, late deliveries and review scores.")
                return decide("report", "Explain that the question cannot be answered reliably.",
                              f"unexplained words: {', '.join(unknown)}")
            if state.get("scope_reason"):  # the Data Analyst found no matching metric
                return decide("report", "Explain that the question cannot be answered from the data.",
                              "no supported metric matches the question")
            if state.get("data_stalled"):  # it kept repeating a failed request; a new call would too
                return decide("report", "Explain that no reliable data could be retrieved.",
                              "the Data Analyst kept repeating a failed request")
            if "data_analyst" in options:  # nothing else is possible without data: no model call needed
                return decide("data_analyst", question, "no data yet")
        if options == ["report"]:
            return decide("report", "Write the final answer.", "no further handoffs allowed")
        if request := state.get("data_request"):
            update["data_request"] = None
            if "data_analyst" in options:
                return decide("data_analyst", request, "Analytics requested more data")

        # LLM decision, validated against the legal options.
        prompt = (f"Business question: {question}\n\n{_status(state)}\n\n"
                  f"Allowed next steps: {', '.join(options)}")
        try:
            reply = llm.invoke([SystemMessage(SUPERVISOR_PROMPT), HumanMessage(prompt)]).content
            decision = parse_json(reply) or {}
            reason = "LLM decision"
        except Exception as exc:
            decision, reason = {}, f"LLM unavailable ({type(exc).__name__})"

        nxt = decision.get("next")
        task = decision.get("task") if isinstance(decision.get("task"), str) and decision.get("task").strip() else question
        if not state.get("datasets"):
            if nxt != "data_analyst":
                reason = f"forced: no data yet (LLM chose {nxt!r})"
            nxt = "data_analyst"
        elif nxt not in options:
            fallback = "analytics" if not state.get("analyses") and "analytics" in options else "report"
            reason = f"fallback: invalid or disallowed choice {nxt!r}"
            nxt, task = fallback, (task if fallback != "report" else "Write the final answer.")
        if nxt != "report" and not task_is_specific(task, question):
            reason += f"; vague task {task!r} replaced"
            task = fallback_task(nxt, question)
        return decide(nxt, task, reason)

    return supervisor


# ---------------------------------------------------------------------------
# Stage 2: consistency checks + final report (deterministic: no language model)
# ---------------------------------------------------------------------------
def repeated_labels(dataset: dict) -> str:
    """Sanity check: in a result, the combination of label (non-numeric) columns should be
    unique per row. Repeats (e.g. the same category twice) usually mean rows are individual
    records, not groups. Returns a short description, or "" when labels are unique."""
    rows = dataset.get("rows") or []
    if len(rows) < 2:
        return ""
    labels = [c for c in dataset.get("columns", [])
              if all(not isinstance(r.get(c), (int, float)) or isinstance(r.get(c), bool) for r in rows)]
    if not labels:
        return ""
    seen, repeated = set(), []
    for r in rows:
        key = tuple(r.get(c) for c in labels)
        if key in seen and key not in repeated:
            repeated.append(key)
        seen.add(key)
    if not repeated:
        return ""
    examples = ", ".join("/".join(str(v) for v in key) for key in repeated[:3])
    return f"{', '.join(labels)}: {examples}"


_YEAR = re.compile(r"\b(20\d{2})\b")


def period_not_covered(question: str, dataset: dict) -> str:
    """If the question names a year, the data must be filtered to it or contain it.
    Returns the uncovered year(s), or "" when the dataset fits the question's period.
    Date ranges are read exactly as the question rules read them (question.read_ranges): a range
    covering the whole dataset requires no year; a whole-year range requires those years."""
    remaining, range_years, _ = read_ranges(question or "")
    years = set(_YEAR.findall(remaining)) | {str(y) for y in range_years or ()}
    if not years:
        return ""
    request = dataset.get("request") or {}
    filters = json.dumps([request.get("filters", []), request.get("years", [])], default=str)
    periods = [v for r in dataset.get("rows", []) for v in r.values()
               if isinstance(v, str) and re.match(r"^\d{4}", v)]
    covered = any(y in filters for y in years) or any(v.startswith(y) for v in periods for y in years)
    return "" if covered else ", ".join(sorted(years))


def _evidence_section(state: AgentState) -> str:
    lines = ["**Evidence:**"]
    for name, d in state.get("datasets", {}).items():
        lines.append(f"- Dataset `{name}` ({d['row_count']} rows) from:\n```sql\n{d['sql']}\n```")
    for a in state.get("analyses", []):
        lines.append(f"- Analysis `{json.dumps(a['request'])}`: {a['summary']}")
    return "\n".join(lines)


def make_report_node(llm=None):
    """The Supervisor's final stage. The answer text is generated by code from the verified data
    (retail_analyst/summary.py): in the benchmark, model-written summaries stated false superlatives
    and month attributions in 6 of 13 answers. `llm` is accepted for compatibility and not used."""
    def report(state: AgentState) -> dict:
        # Every finding is kept in `checks` (developer view) and classified in `issues`:
        #   scope: the question cannot be answered from the data -> refusal with this reason
        #   data:  the retrieved data itself is unreliable -> no answer may be shown
        #   info:  diagnostic only (problems that were fixed or did not affect the data)
        issues: list[dict] = []

        def add(kind: str, message: str) -> None:
            issues.append({"kind": kind, "message": message})

        if state.get("scope_reason"):
            add("scope", state["scope_reason"])
        if not state.get("datasets"):
            add("data", "No data could be retrieved for this question.")
        if state.get("step", 0) > MAX_ROUTING_STEPS:
            add("data", "The workflow stopped at its step limit; the analysis may be incomplete.")
        for name, d in state.get("datasets", {}).items():
            if d["truncated"]:
                add("data", f"Dataset {name} was cut at {d['row_count']} rows; totals from it may be incomplete.")
            for note in d.get("corrections", []):
                add("info", f"Dataset {name}: the data request was corrected automatically. {note}")
            if repeated := repeated_labels(d):
                add("data", f"Dataset {name} repeats labels that should be unique ({repeated}); its rows "
                            "may be individual records rather than one row per group, so rankings or "
                            "totals per group may be wrong.")
            if years := period_not_covered(state.get("question", ""), d):
                add("data", f"Dataset {name} is neither filtered to nor contains the period in the question "
                            f"({years}); it may describe a different period.")
        for error in state.get("errors", []):
            add("info", error)

        if state.get("datasets"):
            rows, meta = answer_data(state)
            body = f"**Answer:** {summary_text(rows, meta)}"
        else:
            body = ("**Answer:** I could not retrieve data from the database to answer this question, "
                    "so no answer is given rather than a guess.")

        issues = list({i["message"]: i for i in issues}.values())  # de-duplicate, keep order
        checks = [i["message"] for i in issues]
        caveats = "\n".join(f"- {c}" for c in checks) or "- No issues detected by the automatic checks."
        final = f"{body}\n\n{_evidence_section(state)}\n\n**Caveats:**\n{caveats}"
        return {"final_answer": final, "answer_body": body, "checks": checks, "issues": issues,
                "trace": [event(state, "supervisor", "report", result=f"{len(checks)} caveat(s)")]}

    return report
