"""Structured query builder: the Data Analyst describes ONE query as JSON, this module validates
every part against the live schema metadata and generates the SQL. The model never writes SQL.

Request format (only "table" and "columns" or "metrics" are required):
    {"table": "analytics.seller_performance",
     "columns": ["seller_state_std"],              # shown; become GROUP BY when metrics are used
     "metrics": ["sum(units)", "count(*)"],         # sum | avg | min | max | count
     "filters": [["seller_state_std", "in", ["SP", "RJ"]]],   # = != > >= < <= in
     "order_by": "sum(units) desc",                 # a column or a metric, optional asc/desc
     "limit": 10,                                   # 1..500, default 100
     "join": "warehouse.dim_products"}              # optional, only along a real foreign key

The generated SQL is still executed through run_sql (validator + read-only role).
"""
from __future__ import annotations

import math
import re

from retail_analyst.config import TableInfo, load_schema, tables_with_column

AGGREGATES = ("sum", "avg", "min", "max", "count")
OPERATORS = ("=", "!=", ">", ">=", "<", "<=", "in")
NUMERIC_TYPES = {"smallint", "integer", "bigint", "numeric", "real", "float"}
KEYS = {"table", "join", "columns", "metrics", "filters", "order_by", "limit"}
DEFAULT_LIMIT, MAX_LIMIT, MAX_IN_VALUES = 100, 500, 50

_METRIC = re.compile(r"^\s*(\w+)\s*\(\s*(\*|[\w.]+)\s*\)\s*$")


class QuerySpecError(ValueError):
    """The request is invalid; the message explains how to fix it."""


def _usable_columns(info: TableInfo) -> list[str]:
    return [c for c, _ in info.columns if not c.startswith("dq_")]


def _table(name) -> TableInfo:
    tables = load_schema()
    if not isinstance(name, str) or name.strip().lower() not in tables:
        raise QuerySpecError(f"Unknown table {name!r}. Use a schema-qualified table, e.g. analytics.sales_daily.")
    return tables[name.strip().lower()]


def _join_condition(base: TableInfo, other: TableInfo) -> str:
    for cols, ref, ref_cols in base.foreign_keys:
        if ref == other.full_name:
            return " AND ".join(f"t.{c} = j.{r}" for c, r in zip(cols, ref_cols))
    for cols, ref, ref_cols in other.foreign_keys:
        if ref == base.full_name:
            return " AND ".join(f"j.{c} = t.{r}" for c, r in zip(cols, ref_cols))
    raise QuerySpecError(f"{base.full_name} and {other.full_name} have no foreign key between them; "
                         "use one table or a table pair from the listed joins.")


def _as_list(value, param: str) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise QuerySpecError(f"'{param}' must be a list.")
    return value


def _literal(value, ctype: str, column: str) -> str:
    """Render a filter value as a safe SQL literal matching the column type."""
    if ctype == "bool":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if not isinstance(value, bool):
            raise QuerySpecError(f"Filter on {column}: value must be true or false.")
        return "TRUE" if value else "FALSE"
    if ctype in NUMERIC_TYPES:
        if isinstance(value, bool):
            raise QuerySpecError(f"Filter on {column}: value must be a number.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise QuerySpecError(f"Filter on {column}: value {value!r} must be a number.") from None
        if not math.isfinite(number):
            raise QuerySpecError(f"Filter on {column}: value must be a finite number.")
        return str(int(number)) if number.is_integer() else repr(number)
    if isinstance(value, (dict, list)) or value is None:
        raise QuerySpecError(f"Filter on {column}: value must be text, a number or true/false.")
    text = str(value)
    if len(text) > 200:
        raise QuerySpecError(f"Filter on {column}: value is too long.")
    return "'" + text.replace("'", "''") + "'"


_SUMMARISING = ("sum", "avg", "min", "max")  # count(col) per group of col is legitimate


def _metric_columns(spec: dict) -> set[str]:
    """Bare column names used inside sum/avg/min/max metrics."""
    found = set()
    for item in spec.get("metrics") or []:
        match = _METRIC.match(item) if isinstance(item, str) else None
        if match and match.group(1).lower() in _SUMMARISING and match.group(2) != "*":
            found.add(match.group(2).split(".")[-1])
    return found


def _bare_name(value) -> str | None:
    return value.strip().split(".")[-1] if isinstance(value, str) and value.strip() else None


def _referenced_columns(spec: dict) -> set[str]:
    """Every column a request uses: columns, metric arguments, filters and order_by."""
    refs = {_bare_name(c) for c in spec.get("columns") or [] if isinstance(spec.get("columns"), list)}
    for item in spec.get("metrics") or []:
        match = _METRIC.match(item) if isinstance(item, str) else None
        if match and match.group(2) != "*":
            refs.add(match.group(2).split(".")[-1])
    for f in spec.get("filters") or []:
        if isinstance(f, list) and f:
            refs.add(_bare_name(f[0]))
        elif isinstance(f, dict):
            refs.add(_bare_name(f.get("column")))
    order = spec.get("order_by")
    if isinstance(order, str) and order.strip():
        target = order.strip().rsplit(None, 1)
        target = target[0] if len(target) == 2 and target[1].lower() in ("asc", "desc") else order.strip()
        match = _METRIC.match(target)
        refs.add(match.group(2).split(".")[-1] if match and match.group(2) != "*" else _bare_name(target))
    return {r for r in refs if r}


def auto_correct(spec):
    """Fix common mistakes that have exactly one possible meaning. Returns (spec, notes).

    1. A bare metric (no aggregate) that repeats a column already in "columns" is a duplicate
       of that column: it is dropped.
    2. A column listed in "columns" (a group) that is also summarised in "metrics" splits every
       group into one row per value: it is removed from "columns".
    3. If the chosen table lacks some requested columns and exactly ONE readable table has all
       of them, the request is moved to that table. Ambiguous cases are left to fail.
    """
    if not isinstance(spec, dict):
        return spec, []
    notes = []
    columns = spec.get("columns") if isinstance(spec.get("columns"), list) else []
    listed = {_bare_name(c) for c in columns}

    metrics = spec.get("metrics")
    if isinstance(metrics, list):
        duplicates = [m for m in metrics if isinstance(m, str) and not _METRIC.match(m) and _bare_name(m) in listed]
        if duplicates:
            kept = [m for m in metrics if m not in duplicates]
            spec = {k: v for k, v in {**spec, "metrics": kept}.items() if not (k == "metrics" and not kept)}
            notes.append(f"Removed {duplicates} from metrics because they repeat columns already requested.")

    summarised = _metric_columns(spec)
    if summarised and columns:
        kept = [c for c in columns if _bare_name(c) not in summarised]
        if len(kept) != len(columns):
            removed = [c for c in columns if c not in kept]
            spec = {**spec, "columns": kept}
            notes.append(f"Removed {removed} from columns because they are also summarised in metrics "
                         "(grouping by them would split each group into one row per value).")

    tables = load_schema()
    table = spec.get("table").strip().lower() if isinstance(spec.get("table"), str) else None
    refs = _referenced_columns(spec)
    if table in tables and refs and not spec.get("join"):
        have = {c for c, _ in tables[table].columns if not c.startswith("dq_")}
        if not refs <= have:
            candidates = [name for name, info in tables.items()
                          if refs <= {c for c, _ in info.columns if not c.startswith("dq_")}]
            if len(candidates) == 1:
                spec = {**spec, "table": candidates[0]}
                notes.append(f"Moved the request from {table} to {candidates[0]}, the only table that has all "
                             f"requested columns ({', '.join(sorted(refs))}).")
    return spec, notes


def build_query(spec: dict) -> str:
    """Validate a query request and return the generated SQL (raises QuerySpecError)."""
    if not isinstance(spec, dict):
        raise QuerySpecError("The query must be a JSON object.")
    if unknown := set(spec) - KEYS:
        raise QuerySpecError(f"Unknown key(s) {sorted(unknown)}. Allowed: {sorted(KEYS)}.")

    base = _table(spec.get("table"))
    sources = {"t": base}
    from_sql = f"{base.full_name} AS t"
    if spec.get("join"):
        other = _table(spec["join"])
        if other.full_name == base.full_name:
            raise QuerySpecError("'join' must be a different table.")
        from_sql += f" JOIN {other.full_name} AS j ON {_join_condition(base, other)}"
        sources["j"] = other
    names = " and ".join(i.full_name for i in sources.values())

    def column(name, param: str) -> tuple[str, str, str]:
        """(qualified expression, type, bare name) of a column in the chosen table(s)."""
        if not isinstance(name, str) or not name.strip():
            raise QuerySpecError(f"'{param}' must contain column names (text).")
        bare = name.strip().split(".")[-1]
        for alias, info in sources.items():
            for col, ctype in info.columns:
                if col == bare and not col.startswith("dq_"):
                    return f"{alias}.{col}", ctype, col
        owners = [t for t in tables_with_column(bare) if t not in {i.full_name for i in sources.values()}]
        hint = f" Tables that have it: {', '.join(owners)}." if owners else ""
        columns = "; ".join(f"{i.full_name}: {', '.join(_usable_columns(i))}" for i in sources.values())
        raise QuerySpecError(f"'{param}': column {bare!r} does not exist in {names}.{hint} "
                             f"Available columns - {columns}")

    # SELECT list: plain columns, then metrics.
    select, group_by, output = [], [], {}
    summarised = _metric_columns(spec)
    for name in _as_list(spec.get("columns"), "columns"):
        expr, _, bare = column(name, "columns")
        if bare in summarised:  # auto_correct() normally removes these first
            raise QuerySpecError(f"Column {bare!r} is both a group in 'columns' and summarised in 'metrics'; "
                                 "remove it from 'columns'.")
        if bare not in output:
            select.append(f"{expr} AS {bare}")
            group_by.append(expr)
            output[bare] = expr

    metrics = {}
    for item in _as_list(spec.get("metrics"), "metrics"):
        match = _METRIC.match(item) if isinstance(item, str) else None
        if not match:
            raise QuerySpecError(f"Metric {item!r} must look like sum(column), avg(column) or count(*).")
        agg, arg = match.group(1).lower(), match.group(2)
        if agg not in AGGREGATES:
            raise QuerySpecError(f"Aggregate {agg!r} is not allowed. Use one of {list(AGGREGATES)}.")
        if arg == "*":
            if agg != "count":
                raise QuerySpecError("Only count(*) may use *.")
            expr, alias = "*", "row_count"
        else:
            expr, ctype, bare = column(arg, "metrics")
            if agg in ("sum", "avg") and ctype not in NUMERIC_TYPES:
                raise QuerySpecError(f"{agg}({bare}) needs a numeric column; {bare} is {ctype}.")
            alias = f"{agg}_{bare}"
        if alias not in output:
            select.append(f"{agg.upper()}({expr}) AS {alias}")
            output[alias] = alias
            metrics[f"{agg}({arg.split('.')[-1]})"] = alias
    if not select:
        raise QuerySpecError("Give at least one entry in 'columns' or 'metrics'.")

    # WHERE: [column, operator, value] triples (dicts with column/op/value also accepted).
    where = []
    for f in _as_list(spec.get("filters"), "filters"):
        if isinstance(f, dict):
            f = [f.get("column"), f.get("op") or f.get("operator"), f.get("value")]
        if not isinstance(f, list) or len(f) != 3:
            raise QuerySpecError('Each filter must be ["column", "operator", value].')
        expr, ctype, bare = column(f[0], "filters")
        op = str(f[1]).strip().lower()
        if op not in OPERATORS:
            raise QuerySpecError(f"Operator {f[1]!r} is not allowed. Use one of {list(OPERATORS)}.")
        if op == "in":
            values = f[2] if isinstance(f[2], list) else None
            if not values or len(values) > MAX_IN_VALUES:
                raise QuerySpecError(f"'in' needs a list of 1-{MAX_IN_VALUES} values.")
            where.append(f"{expr} IN ({', '.join(_literal(v, ctype, bare) for v in values)})")
        else:
            where.append(f"{expr} {op} {_literal(f[2], ctype, bare)}")

    # ORDER BY: one column or metric, asc/desc (NULLs always last).
    order_sql = ""
    if order := spec.get("order_by"):
        if not isinstance(order, str):
            raise QuerySpecError("'order_by' must be text like \"sum(units) desc\".")
        parts = order.strip().rsplit(None, 1)
        direction = "ASC"
        if len(parts) == 2 and parts[1].lower() in ("asc", "desc"):
            order, direction = parts[0], parts[1].upper()
        key = order.strip().replace(" ", "").lower()
        if key in metrics:
            target = metrics[key]
        elif key.split(".")[-1] in output:
            target = output[key.split(".")[-1]]
        elif metrics and not group_by:
            raise QuerySpecError(f"'order_by' must be one of the metrics: {list(metrics)}.")
        elif metrics:
            raise QuerySpecError(f"'order_by' must be a selected column or metric: {list(output)}.")
        else:
            target = column(order, "order_by")[0]
        order_sql = f"\nORDER BY {target} {direction} NULLS LAST"

    limit = spec.get("limit", DEFAULT_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise QuerySpecError(f"'limit' must be an integer between 1 and {MAX_LIMIT}.")

    sql = f"SELECT {', '.join(select)}\nFROM {from_sql}"
    if where:
        sql += "\nWHERE " + " AND ".join(where)
    if metrics and group_by:
        sql += "\nGROUP BY " + ", ".join(group_by)
    return sql + order_sql + f"\nLIMIT {limit}"
