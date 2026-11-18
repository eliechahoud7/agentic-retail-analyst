"""Deterministic answer text and results table, generated from verified query results.

Every statement a user reads is computed here from the rows returned by the catalog query:
which item is highest or lowest, what follows, the largest increase or decrease, the overall
change, shares. No language model writes user-facing text. The benchmark showed why: the model
misread tables and stated false superlatives and month attributions in 6 of 13 answers.
"""
from __future__ import annotations

import math
import re

from retail_analyst.catalog import GROUPINGS, METRICS
from retail_analyst.labels import display_label

MAX_TABLE_ROWS = 20

METRIC_NAMES = {  # singular noun phrases, so every template reads naturally ("... was highest in ...")
    "revenue": "revenue", "units_sold": "number of units sold", "freight": "freight cost",
    "orders": "number of orders", "customers": "number of customers", "average_order_value": "average order value",
    "late_delivery_rate": "late delivery rate", "average_delivery_days": "average delivery time",
    "payment_value": "payment value", "average_review_score": "average review score",
    "reviews": "number of reviews",
}
METRIC_UNITS = {"average_delivery_days": " days"}  # units the number format alone does not show
GROUP_TITLES = {  # (singular, plural)
    "category": ("Product Category", "Product Categories"), "seller_state": ("Seller State", "Seller States"),
    "customer_state": ("Customer State", "Customer States"), "month": ("Month", "Months"),
    "payment_type": ("Payment Method", "Payment Methods"), "delivery_status": ("Delivery Status", "Delivery Statuses"),
}
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")

# ---------------------------------------------------------------------------
# Column labels and value formatting
# ---------------------------------------------------------------------------
_LABELS = {  # friendly column names; anything else is derived from the column name
    "category_reporting": "Category", "category_en_clean": "Category", "product_category_name": "Category",
    "payment_type": "Payment type", "customer_state_std": "State", "seller_state_std": "Seller state",
    "year_month": "Month", "purchase_year_month": "Month", "review_creation_month": "Month",
    "cohort_month": "Cohort", "purchase_date": "Date", "delivery_class": "Delivery",
    "items_price_total": "Revenue", "items_price_total_delivered": "Revenue (delivered)",
    "payment_value": "Payment value", "payment_value_total": "Payment value", "payment_total": "Payments",
    "items_freight_total": "Freight", "share_pct": "Share", "growth_pct": "Growth", "change": "Change",
    "late_rate": "Late delivery rate", "review_score": "Review score", "row_count": "Count",
    "n_orders": "Orders", "units": "Units", "n_late": "Late orders",
    # catalog result columns (retail_analyst/catalog.py)
    "revenue": "Revenue", "orders": "Orders", "customers": "Customers", "reviews": "Reviews",
    "freight_total": "Freight", "delivery_status": "Delivery", "avg_delivery_days": "Average delivery days",
}
_AGG_WORDS = {"sum": "", "avg": "Average", "min": "Minimum", "max": "Maximum", "count": "Number of"}
_MONEY_HINTS = ("price", "payment", "revenue", "freight", "value", "total")
_NOT_MONEY = ("pct", "share", "rate", "n_", "count", "units", "installments", "score", "days", "hours")
_SMALL_WORDS = {"a", "an", "and", "by", "for", "from", "in", "of", "on", "or", "per", "the", "to", "vs", "with"}
_PERIOD = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?([T ].*)?$")  # 2018, 2018-01, 2018-01-31, timestamps


def _bare(column: str) -> str:
    return re.sub(r"^(sum|avg|min|max|count)_", "", column)


def column_label(column: str) -> str:
    if column in _LABELS:
        return _LABELS[column]
    agg = column.split("_", 1)[0]
    base = _LABELS.get(_bare(column), _bare(column).replace("_", " ").strip().capitalize())
    prefix = _AGG_WORDS.get(agg, "") if column != _bare(column) else ""
    return f"{prefix} {base[0].lower() + base[1:]}".strip() if prefix else base


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not (
        isinstance(value, float) and math.isnan(value))


def _kind(column: str, values: list) -> str:
    """money | percent | rate | integer | number | label"""
    numbers = [v for v in values if v is not None]
    if not numbers or not all(_is_number(v) for v in numbers):
        return "label"
    bare = _bare(column)
    if bare.endswith("_pct") or bare in ("share_pct", "growth_pct"):
        return "percent"
    if "rate" in bare and all(0 <= v <= 1 for v in numbers):
        return "rate"
    if any(h in bare for h in _MONEY_HINTS) and not any(n in bare for n in _NOT_MONEY):
        return "money"
    return "integer" if all(float(v).is_integer() for v in numbers) else "number"


def humanize_label(value) -> str:
    """'health_beauty' -> 'Health & Beauty'; 'credit_card' -> 'Credit Card'; 'AL', '2018-01' unchanged."""
    return display_label(value)


def format_value(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if kind == "money":
        return f"R$ {value:,.0f}" if abs(value) >= 1000 else f"R$ {value:,.2f}"
    if kind == "percent":
        return f"{value:.2f}%"
    if kind == "rate":
        return f"{value * 100:.2f}%"
    if kind == "integer":
        return f"{int(value):,}"
    if kind == "number":
        return f"{value:,.2f}"
    return humanize_label(value)


def period_name(value) -> str:
    """'2017-11' -> 'November 2017'; other labels get their display name."""
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(value))
    return f"{_MONTHS[int(match.group(2)) - 1]} {match.group(1)}" if match else humanize_label(value)


def table_kinds(rows: list[dict]) -> dict[str, str]:
    columns = list(rows[0]) if rows else []
    kinds = {c: _kind(c, [r.get(c) for r in rows]) for c in columns}
    # A "change" column (growth) uses the same unit as the value column before it.
    if "change" in kinds:
        idx = columns.index("change")
        if idx > 0 and kinds[columns[idx - 1]] in ("money", "integer", "number"):
            kinds["change"] = kinds[columns[idx - 1]]
    return kinds


def is_ranking(rows: list[dict], kinds: dict[str, str]) -> bool:
    """Inferred ranking (used when no request metadata is available): labelled rows ordered from
    highest to lowest on the first numeric column, and not a time series."""
    labels = [c for c, k in kinds.items() if k == "label"]
    numeric = [c for c, k in kinds.items() if k != "label"]
    if not labels or not numeric or not 2 <= len(rows) <= MAX_TABLE_ROWS:
        return False
    if all(_PERIOD.match(str(r.get(labels[0]))) for r in rows):
        return False  # a time series that happens to decline is not a ranking
    values = [r.get(numeric[0]) for r in rows]
    return all(_is_number(v) for v in values) and all(a >= b for a, b in zip(values, values[1:]))


def results_table(rows: list[dict], ranked: bool | None = None) -> str:
    """Markdown table of the result rows with friendly headers, units and (for rankings) a rank column."""
    if not rows:
        return ""
    kinds = table_kinds(rows)
    shown = rows[:MAX_TABLE_ROWS]
    ranked = is_ranking(rows, kinds) if ranked is None else ranked
    headers = (["Rank"] if ranked else []) + [column_label(c) for c in kinds]
    aligns = (["---:"] if ranked else []) + ["---" if k == "label" else "---:" for k in kinds.values()]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    for i, r in enumerate(shown, 1):
        cells = ([str(i)] if ranked else []) + [format_value(r.get(c), k) for c, k in kinds.items()]
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
    if len(rows) > MAX_TABLE_ROWS:
        lines.append(f"\n*Showing the first {MAX_TABLE_ROWS} of {len(rows)} rows.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The answer: heading, factual summary, table
# ---------------------------------------------------------------------------
def answer_data(state: dict) -> tuple[list[dict], dict]:
    """Rows and applied request of the answer. The catalog dataset is the answer: rankings, shares
    and growth are computed in its query (analyses are diagnostics only)."""
    datasets = list(state.get("datasets", {}).values())
    if not datasets:
        return [], {}
    return datasets[-1].get("rows", []), datasets[-1].get("request") or {}


def _roles(rows: list[dict], meta: dict, kinds: dict[str, str]) -> tuple[str | None, str | None]:
    """(label column, value column) from the request metadata, else inferred from the rows."""
    metric, group = METRICS.get(meta.get("metric") or ""), GROUPINGS.get(meta.get("group_by") or "")
    labels = [c for c, k in kinds.items() if k == "label"]
    numeric = [c for c, k in kinds.items() if k != "label" and c not in ("change", "growth_pct", "share_pct")]
    label = group.column if group and group.column in kinds else (labels[0] if labels else None)
    value = metric.column if metric and metric.column in kinds else (numeric[0] if numeric else None)
    return label, value


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _years(meta: dict) -> str:
    years = [str(y) for y in meta.get("years") or []]
    return f" in {_join(years)}" if years else ""


def _calculation(rows: list[dict], meta: dict, kinds: dict[str, str], label: str | None) -> str:
    if meta.get("calculation"):
        return meta["calculation"]
    if "growth_pct" in kinds:
        return "growth"
    if "share_pct" in kinds:
        return "share"
    if not label:
        return "total"
    return "ranking" if is_ranking(rows, kinds) else "breakdown"


def summary_text(rows: list[dict], meta: dict) -> str:
    """One or two factual sentences, every comparison computed from the rows."""
    if not rows:
        return "No results were found."
    kinds = table_kinds(rows)
    label, value = _roles(rows, meta, kinds)
    if value is None:
        return "Here are the results."
    name = METRIC_NAMES.get(meta.get("metric") or "", column_label(value).lower())
    calc = _calculation(rows, meta, kinds, label)
    unit = METRIC_UNITS.get(meta.get("metric") or "", "")
    fmt = lambda r: format_value(r.get(value), kinds[value]) + unit  # noqa: E731
    lab = lambda r: period_name(r.get(label))  # noqa: E731
    valued = [r for r in rows if _is_number(r.get(value))]

    if calc == "total" or not label:
        return f"The {name}{_years(meta)} was {fmt(rows[0])}."

    if calc == "growth" and len(valued) >= 2:
        first, last = valued[0], valued[-1]
        text_ = f"{name[0].upper() + name[1:]} went from {fmt(first)} in {lab(first)} to {fmt(last)} in {lab(last)}"
        if first[value]:
            text_ += f", a change of {(last[value] - first[value]) / abs(first[value]) * 100:+.1f}%"
        text_ += "."
        changes = [r for r in rows if _is_number(r.get("growth_pct"))]
        parts = []
        if (up := max(changes, key=lambda r: r["growth_pct"], default=None)) and up["growth_pct"] > 0:
            parts.append(f"the largest increase was in {lab(up)} ({up['growth_pct']:+.2f}%)")
        if (down := min(changes, key=lambda r: r["growth_pct"], default=None)) and down["growth_pct"] < 0:
            parts.append(f"the largest decrease was in {lab(down)} ({down['growth_pct']:+.2f}%)")
        if parts:
            joined = " and ".join(parts)
            text_ += f" {joined[0].upper() + joined[1:]}."
        return text_

    if calc == "share" and "share_pct" in kinds:
        top = rows[0]
        text_ = f"{lab(top)} accounts for the largest share of {name}: {top['share_pct']:.2f}% ({fmt(top)})"
        if len(rows) > 1:
            text_ += ", followed by " + _join([f"{lab(r)} ({r['share_pct']:.2f}%)" for r in rows[1:3]])
        text_ += "."
        if len(rows) > 3:
            together = sum(r["share_pct"] for r in rows[:3] if _is_number(r.get("share_pct")))
            text_ += f" Together, these three account for {together:.2f}% of the total."
        return text_

    if calc == "ranking":
        word = "lowest" if meta.get("ascending") else "highest"
        text_ = f"{lab(rows[0])} has the {word} {name} ({fmt(rows[0])})"
        if len(rows) > 1:
            text_ += ", followed by " + _join([f"{lab(r)} ({fmt(r)})" for r in rows[1:3]])
        return text_ + "."

    # breakdown: every group is in the result, so highest and lowest are exact
    if not valued:
        return "Here are the results."
    high, low = max(valued, key=lambda r: r[value]), min(valued, key=lambda r: r[value])
    if len(valued) == 2:
        return f"The {name} is {fmt(high)} for {lab(high)}, compared with {fmt(low)} for {lab(low)}."
    if all(_PERIOD.match(str(r.get(label))) for r in valued):
        return (f"{name[0].upper() + name[1:]} was highest in {lab(high)} ({fmt(high)}) "
                f"and lowest in {lab(low)} ({fmt(low)}).")
    return f"{lab(high)} has the highest {name} ({fmt(high)}) and {lab(low)} the lowest ({fmt(low)})."


def heading_for(question: str) -> str:
    """Fallback heading from the question text (used when no request metadata is available)."""
    text_ = question.strip().rstrip("?.! ")
    text_ = re.sub(r"^(what|which|how|who|when|where|is|are|do|does|did|show me|list|give me)\b"
                   r"(\s+(is|are|was|were|did|does|do|the|there|me))*\s+", "", text_, flags=re.IGNORECASE)
    words = text_.split()
    titled = [w if (w.isupper() or any(ch.isdigit() for ch in w)) else
              (w.lower() if i and w.lower() in _SMALL_WORDS else w[:1].upper() + w[1:])
              for i, w in enumerate(words)]
    return " ".join(titled) or "Answer"


def title_for(rows: list[dict], meta: dict, question: str) -> str:
    """Heading from the metric and grouping, e.g. 'Top 5 Product Categories by Revenue'."""
    metric = meta.get("metric")
    if metric not in METRIC_NAMES:
        return heading_for(question)
    name = " ".join(w if w in _SMALL_WORDS else w.capitalize() for w in METRIC_NAMES[metric].split())
    single, plural = GROUP_TITLES.get(meta.get("group_by") or "", ("", ""))
    years = _years(meta)
    calc = meta.get("calculation")
    if calc == "ranking" and plural:
        return f"{'Bottom' if meta.get('ascending') else 'Top'} {len(rows)} {plural} by {name}{years}"
    if calc == "share" and single:
        return f"Share of {name} by {single}{years}"
    if calc == "growth":
        return f"Monthly {name} Growth{years}"
    if single:
        return f"{name} by {single}{years}"
    return f"{name}{years}"


def answer_markdown(state: dict) -> str:
    """Heading, factual summary and results table for a reliable answer."""
    rows, meta = answer_data(state)
    ranked = (meta.get("calculation") == "ranking") if meta.get("calculation") else None
    parts = [f"### {title_for(rows, meta, state.get('question', ''))}", summary_text(rows, meta),
             results_table(rows, ranked=ranked)]
    return "\n\n".join(p for p in parts if p)
