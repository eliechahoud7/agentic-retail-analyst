"""The three agents (Supervisor, Data Analyst, Analytics) and small shared helpers."""
from __future__ import annotations

import json
import re

import pandas as pd


def parse_json(text: str) -> dict | None:
    """Parse the first JSON object in an LLM reply; None if there is none."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def extract_sql(text: str) -> str:
    """Take the SQL out of an LLM reply (fenced block preferred)."""
    text = text or ""
    if match := re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        return match.group(1).strip()
    if match := re.search(r"\b(with|select)\b.*", text, re.DOTALL | re.IGNORECASE):
        return match.group(0).strip()
    return text.strip()


def event(state: dict, agent: str, action: str, **fields) -> dict:
    """One trace entry; empty fields are dropped."""
    return {"step": state.get("step", 0), "agent": agent, "action": action,
            **{k: v for k, v in fields.items() if v not in (None, "", [], {})}}


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "text"


def describe_datasets(datasets: dict[str, dict], sample_rows: int = 3) -> str:
    """Dataset names, sizes, typed columns and a few sample rows, for prompts."""
    if not datasets:
        return "(no datasets fetched yet)"
    parts = []
    for name, d in datasets.items():
        df = pd.DataFrame.from_records(d["rows"], columns=d["columns"])
        cols = ", ".join(f"{c} ({_kind(df[c])})" for c in df.columns)
        sample = json.dumps(d["rows"][:sample_rows], default=str)
        parts.append(f"{name}: {d['row_count']} rows, fetched for: {d.get('task', '')}\n"
                     f"  columns: {cols}\n  first rows: {sample}")
    return "\n".join(parts)
