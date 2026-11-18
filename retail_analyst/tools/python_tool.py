"""run_analysis: a fixed allowlist of pandas operations on datasets returned by run_sql.

The Analytics agent sends a JSON-like request, e.g.
    {"operation": "group_sum", "dataset": "orders", "by": "customer_state_std", "value": "payment_total"}
Every field is validated and dispatched to a hand-written function. No LLM-generated
Python is ever executed (no eval/exec).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd

MAX_OUTPUT_ROWS = 50


class AnalysisError(ValueError):
    """Invalid request (unknown operation, dataset, column or parameter)."""


@dataclass
class AnalysisResult:
    ok: bool
    operation: str
    rows: list[dict] = field(default_factory=list)
    summary: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _column(df: pd.DataFrame, name: Any, param: str) -> str:
    if not isinstance(name, str) or name not in df.columns:
        raise AnalysisError(f"'{param}': unknown column {name!r}. Available: {list(df.columns)}")
    return name


def _columns(df: pd.DataFrame, names: Any, param: str) -> list[str]:
    names = [names] if isinstance(names, str) else names
    if not isinstance(names, list) or not names or len(names) > 3:
        raise AnalysisError(f"'{param}' must be a column name or a list of 1-3 column names.")
    return [_column(df, n, param) for n in names]


def _numeric(df: pd.DataFrame, name: Any, param: str) -> str:
    col = _column(df, name, param)
    if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
        raise AnalysisError(f"'{param}': column {col!r} is not numeric.")
    return col


DERIVED = ("change", "growth_pct", "share_pct")  # already-computed calculations


def _additive(df: pd.DataFrame, name: Any, param: str) -> str:
    """A numeric column that may be summed or compared over time. Percentages and changes that
    were already calculated are not: their sum, share or growth is meaningless."""
    col = _numeric(df, name, param)
    if col in DERIVED or col.endswith("_pct"):
        raise AnalysisError(f"'{param}': {col!r} is already a calculated percentage or change; "
                            "use the underlying value column instead.")
    return col


def _records(df: pd.DataFrame) -> list[dict]:
    out = df.head(MAX_OUTPUT_ROWS).round(4)
    return out.astype(object).where(out.notna(), None).to_dict("records")


# ---------------------------------------------------------------------------
# Operations (each receives a validated DataFrame plus raw params, validates its own params)
# ---------------------------------------------------------------------------
def group_sum(df: pd.DataFrame, by: Any, value: Any) -> tuple[pd.DataFrame, str]:
    by_cols, val = _columns(df, by, "by"), _additive(df, value, "value")
    out = df.groupby(by_cols, dropna=False)[val].sum().reset_index().sort_values(val, ascending=False)
    return out, f"Sum of {val} by {', '.join(by_cols)} ({len(out)} groups)."


def top_n(df: pd.DataFrame, value: Any, n: Any = 10, ascending: Any = False) -> tuple[pd.DataFrame, str]:
    val = _numeric(df, value, "value")
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_OUTPUT_ROWS:
        raise AnalysisError(f"'n' must be an integer between 1 and {MAX_OUTPUT_ROWS}.")
    if not isinstance(ascending, bool):
        raise AnalysisError("'ascending' must be true or false.")
    out = df.sort_values(val, ascending=ascending).head(n)
    return out, f"{'Bottom' if ascending else 'Top'} {len(out)} rows by {val}."


def mean_median(df: pd.DataFrame, value: Any, by: Any = None) -> tuple[pd.DataFrame, str]:
    val = _numeric(df, value, "value")
    stats = ["count", "mean", "median", "min", "max"]
    if by is None:
        out = df[val].agg(stats).to_frame().T.reset_index(drop=True)
        return out, f"Statistics of {val} over {int(out.loc[0, 'count'])} rows."
    by_cols = _columns(df, by, "by")
    out = df.groupby(by_cols, dropna=False)[val].agg(stats).reset_index().sort_values("mean", ascending=False)
    return out, f"Statistics of {val} by {', '.join(by_cols)} ({len(out)} groups)."


def share_of_total(df: pd.DataFrame, by: Any, value: Any) -> tuple[pd.DataFrame, str]:
    out, _ = group_sum(df, by, value)
    val = out.columns[-1]
    total = out[val].sum()
    if total == 0:
        raise AnalysisError(f"Total of {val!r} is zero; shares are undefined.")
    out["share_pct"] = out[val] / total * 100
    return out, f"Share of total {val} (total = {total:,.2f})."


def growth(df: pd.DataFrame, period: Any, value: Any) -> tuple[pd.DataFrame, str]:
    per, val = _column(df, period, "period"), _additive(df, value, "value")
    out = df.groupby(per)[val].sum().reset_index().sort_values(per)
    if len(out) < 2:
        raise AnalysisError("Growth needs at least two periods.")
    prev = out[val].shift(1)
    out["change"] = out[val] - prev
    out["growth_pct"] = (out["change"] / prev.where(prev != 0)) * 100
    first, last = out.iloc[0], out.iloc[-1]
    return out, (f"{val} per {per}: {len(out)} periods from {first[per]} ({first[val]:,.2f}) "
                 f"to {last[per]} ({last[val]:,.2f}).")


def correlation(df: pd.DataFrame, x: Any, y: Any, method: Any = "pearson") -> tuple[pd.DataFrame, str]:
    cx, cy = _numeric(df, x, "x"), _numeric(df, y, "y")
    if method not in ("pearson", "spearman"):
        raise AnalysisError("'method' must be 'pearson' or 'spearman'.")
    pair = df[[cx, cy]].dropna()
    if len(pair) < 3:
        raise AnalysisError("Correlation needs at least 3 complete rows.")
    if method == "spearman":  # Spearman = Pearson on ranks (avoids a scipy dependency)
        pair = pair.rank()
    r = pair[cx].corr(pair[cy])
    out = pd.DataFrame([{"x": cx, "y": cy, "method": method, "n": len(pair), "r": r}])
    return out, f"{method.title()} correlation between {cx} and {cy}: r = {r:.3f} (n = {len(pair)})."


OPERATIONS: dict[str, Callable[..., tuple[pd.DataFrame, str]]] = {
    "group_sum": group_sum,
    "top_n": top_n,
    "mean_median": mean_median,
    "share_of_total": share_of_total,
    "growth": growth,
    "correlation": correlation,
}

# Allowed parameters per operation (besides "operation" and "dataset"); used for validation
# and for describing the tool to the Analytics agent.
PARAMETERS: dict[str, dict[str, bool]] = {  # name -> required?
    "group_sum": {"by": True, "value": True},
    "top_n": {"value": True, "n": False, "ascending": False},
    "mean_median": {"value": True, "by": False},
    "share_of_total": {"by": True, "value": True},
    "growth": {"period": True, "value": True},
    "correlation": {"x": True, "y": True, "method": False},
}


def run_analysis(request: dict, datasets: dict[str, pd.DataFrame]) -> AnalysisResult:
    """Validate and execute one allowlisted operation. Never raises; errors are returned."""
    op = request.get("operation") if isinstance(request, dict) else None
    try:
        if op not in OPERATIONS:
            raise AnalysisError(f"Unknown operation {op!r}. Allowed: {list(OPERATIONS)}")
        name = request.get("dataset")
        if name not in datasets:
            raise AnalysisError(f"Unknown dataset {name!r}. Available: {list(datasets)}")
        df = datasets[name]
        if df.empty:
            raise AnalysisError(f"Dataset {name!r} is empty.")

        params = {k: v for k, v in request.items() if k not in ("operation", "dataset")}
        allowed = PARAMETERS[op]
        if unknown := set(params) - set(allowed):
            raise AnalysisError(f"Unknown parameter(s) for {op}: {sorted(unknown)}. Allowed: {list(allowed)}")
        if missing := [p for p, required in allowed.items() if required and p not in params]:
            raise AnalysisError(f"Missing parameter(s) for {op}: {missing}")

        out, summary = OPERATIONS[op](df, **params)
        return AnalysisResult(ok=True, operation=op, rows=_records(out), summary=summary)
    except AnalysisError as exc:
        return AnalysisResult(ok=False, operation=str(op), error=str(exc))
    except (TypeError, ValueError) as exc:  # e.g. incomparable period values
        return AnalysisResult(ok=False, operation=str(op), error=f"Analysis failed: {exc}")


def dataframe_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from SqlResult.rows."""
    return pd.DataFrame.from_records(rows)
