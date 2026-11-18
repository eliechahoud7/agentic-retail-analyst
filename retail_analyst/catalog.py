"""Business metric catalog (semantic layer).

The Data Analyst chooses ONE metric and ONE optional grouping from this closed list; code
compiles a verified query from the definitions below. Definitions follow the database's own
documentation (table comments and view definitions of retail_intelligence):

- revenue = item prices excluding freight (fact_order_lines.line_price_total); payments are a
  different measure and cannot be attributed to products, categories or sellers;
- delivery metrics use the documented eligible population: order_status = 'delivered' with a
  delivery date (same as analytics.delivery_by_state / delivery_monthly);
- monthly series are limited to the reliable window (in_candidate_window, 2017-01 to 2018-08).

Every metric/grouping combination is tested against independently computed values
(tests/test_catalog.py). The compiled SQL still runs through run_sql (validator + read-only role).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from retail_analyst.question import DEFAULT_TOP_N, Constraints

ELIGIBLE = "o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL"
MAX_GROUPS = 100


class UnsupportedRequest(ValueError):
    """The request cannot be answered reliably with the catalog; the message says why."""


@dataclass(frozen=True)
class Source:
    from_sql: str
    date: str                               # purchase date column used for year filters and months
    window: str                             # reliable-period flag
    joins: dict[str, str] = field(default_factory=dict)  # grouping -> JOIN clause it needs


@dataclass(frozen=True)
class Grouping:
    key: str
    description: str
    column: str                             # output column name (known to the presentation layer)
    expr: str = ""                          # SQL expression; "" = month (built from the source date)
    where: str = ""                         # extra condition the grouping needs


@dataclass(frozen=True)
class Metric:
    key: str
    description: str
    source: str
    expr: str
    column: str
    groupings: tuple[str, ...]
    where: str = ""


SOURCES = {
    "order_lines": Source("warehouse.fact_order_lines AS l", "l.purchase_date", "l.in_candidate_window", {
        "category": "JOIN warehouse.dim_products AS p ON p.product_id = l.product_id",
        "seller_state": "JOIN warehouse.dim_sellers AS s ON s.seller_id = l.seller_id",
        "customer_state": "JOIN warehouse.fact_orders AS o ON o.order_id = l.order_id"}),
    "orders": Source("warehouse.fact_orders AS o", "o.purchase_date", "o.in_candidate_window"),
    "payments": Source("warehouse.fact_payments AS y", "y.purchase_date", "y.in_candidate_window"),
    "reviews": Source("warehouse.fact_reviews AS r", "r.purchase_date", "r.in_candidate_window", {
        "customer_state": "JOIN warehouse.fact_orders AS o ON o.order_id = r.order_id",
        "delivery_status": "JOIN warehouse.fact_orders AS o ON o.order_id = r.order_id"}),
}

GROUPINGS = {g.key: g for g in [
    Grouping("category", "product category", "category_reporting", "p.category_reporting"),
    Grouping("seller_state", "state of the seller", "seller_state_std", "s.seller_state_std"),
    Grouping("customer_state", "state of the customer", "customer_state_std", "o.customer_state_std"),
    Grouping("month", "calendar month of purchase (for trends, growth or 'per month')", "year_month"),
    Grouping("payment_type", "payment method (credit card, boleto, voucher, debit card)", "payment_type",
             "y.payment_type"),
    Grouping("delivery_status", "late versus on-time delivered orders", "delivery_status",
             "CASE WHEN o.is_late THEN 'late' ELSE 'on_time' END", ELIGIBLE),
]}

_SALES_GROUPS = ("category", "seller_state", "customer_state", "month")
_ORDER_GROUPS = ("customer_state", "month")
_REVIEW_GROUPS = ("customer_state", "month", "delivery_status")

METRICS = {m.key: m for m in [
    Metric("revenue", "sales revenue: item prices excluding freight, in BRL (\"sales\", \"revenue\")",
           "order_lines", "SUM(l.line_price_total)", "revenue", _SALES_GROUPS),
    Metric("units_sold", "number of product units sold", "order_lines", "SUM(l.quantity)", "units", _SALES_GROUPS),
    Metric("freight", "freight (shipping) charges, in BRL", "order_lines", "SUM(l.freight_total)",
           "freight_total", _SALES_GROUPS),
    Metric("orders", "number of orders", "orders", "COUNT(*)", "orders", _ORDER_GROUPS),
    Metric("customers", "number of distinct customers", "orders", "COUNT(DISTINCT o.customer_unique_id)",
           "customers", _ORDER_GROUPS),
    Metric("average_order_value", "average item value per order, in BRL", "orders",
           "ROUND(AVG(o.items_price_total)::numeric, 2)", "avg_order_value", _ORDER_GROUPS, "o.has_items"),
    Metric("late_delivery_rate", "share of delivered orders that arrived after the estimated date",
           "orders", "ROUND(COUNT(*) FILTER (WHERE o.is_late)::numeric / NULLIF(COUNT(*), 0), 4)",
           "late_rate", _ORDER_GROUPS, ELIGIBLE),
    Metric("average_delivery_days", "average days from purchase to delivery", "orders",
           "ROUND(AVG(o.purchase_to_delivery_days)::numeric, 2)", "avg_delivery_days", _ORDER_GROUPS, ELIGIBLE),
    Metric("payment_value", "amount customers paid (includes freight), in BRL; not revenue", "payments",
           "SUM(y.payment_value)", "payment_value", ("payment_type", "month")),
    Metric("average_review_score", "average customer review score (1 to 5 stars)", "reviews",
           "ROUND(AVG(r.review_score)::numeric, 3)", "avg_review_score", _REVIEW_GROUPS),
    Metric("reviews", "number of customer reviews", "reviews", "COUNT(*)", "reviews", _REVIEW_GROUPS),
]}


# Business terms that name each metric and grouping in a question (lowercase, whole words).
# Used to check that the Data Analyst's choice fits the question (check_fit).
#
# Synonym groups added after the final regression run (approved): general business wording for
# selling, earning, waiting and lateness. Some of these words come from the independent question
# set ("takings", "earns", "wait", "promised date") and the special checks ("best-selling"), so
# those questions are no longer independent evidence. Ambiguous words ("selling", "buy") are listed
# under every metric they can mean, so either reasonable choice is accepted. Deliberately NOT added:
# "products" alone (would accept "products in stock" as units sold) and "spend".
_SELLING = ("sell", "sells", "selling", "best-selling", "best selling", "top-selling", "top selling",
            "bestseller", "bestsellers", "best seller", "best sellers", "buy", "buys", "bought")
METRIC_TERMS = {
    "revenue": ("revenue", "revenues", "sales", "sold", "turnover", "gmv", "earned", "earnings",
                "earn", "earns", "earning", "takings", "income", "sales value") + _SELLING,
    "units_sold": ("units", "units sold", "items sold", "products sold", "quantity", "quantities",
                   "number of items", "number of units", "volume", "sales volume") + _SELLING,
    "freight": ("freight", "freight cost", "freight costs", "shipping cost", "shipping costs",
                "shipping fee", "shipping fees", "delivery cost", "delivery costs"),
    "orders": ("orders", "order count", "number of orders", "purchases", "how many orders"),
    "customers": ("customers", "buyers", "clients", "shoppers", "number of customers", "unique customers",
                  "distinct customers"),
    "average_order_value": ("average order value", "order value", "order values", "aov", "basket size",
                            "average basket", "average ticket", "average order"),
    # "estimated (delivery) date" and "promised date" alone were removed: they let "average time to the
    # ESTIMATED delivery date" pass as actual delivery time (fresh question set). Only the lateness
    # phrases remain.
    "late_delivery_rate": ("late", "lateness", "late delivery", "late deliveries", "late delivery rate",
                           "delay", "delays", "delayed", "on time", "on-time", "overdue", "behind schedule",
                           "after the promised date", "later than promised", "after the estimated date",
                           "later than estimated"),
    "average_delivery_days": ("delivery time", "delivery times", "delivery days", "days to deliver",
                              "time to deliver", "shipping time", "delivery speed", "how long", "wait",
                              "waits", "waiting", "waiting time", "wait time", "lead time", "transit time",
                              "delivery duration", "to arrive", "take to arrive", "takes to arrive"),
    "payment_value": ("payment", "payments", "paid", "payment value", "payment values", "amount paid"),
    "average_review_score": ("review score", "review scores", "rating", "ratings", "stars", "satisfaction",
                             "score", "scores"),
    "reviews": ("reviews", "number of reviews", "review count"),
}
GROUPING_TERMS = {
    "category": ("category", "categories", "product category", "product categories", "product type",
                 "product types", "product line", "product lines", "department", "departments"),
    "seller_state": ("seller state", "seller states", "sellers' state", "sellers' states", "state of the seller",
                     "state of the sellers"),
    "customer_state": ("state", "states", "customer state", "customer states", "region", "regions",
                       "location", "locations"),
    "month": ("month", "months", "monthly", "per month", "each month", "by month", "trend", "trends",
              "over time", "grow", "grew", "growth", "changed", "change over time"),
    "payment_type": ("payment type", "payment types", "payment method", "payment methods", "credit card",
                     "credit cards", "boleto", "voucher", "vouchers", "debit card", "debit cards"),
    "delivery_status": ("late", "delay", "delays", "delayed", "on time", "on-time", "delivery status",
                        "late deliveries", "late orders"),
}


def _mentioned(question: str, terms: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    """Keys whose terms occur in the question as whole words. A shorter match inside a longer
    match of ANOTHER key does not count ('units sold' is units, not revenue; 'seller state' is
    seller_state, not customer_state)."""
    text_ = (question or "").lower()
    spans = [(m.start(), m.end(), key, term)
             for key, key_terms in terms.items() for term in key_terms
             for m in re.finditer(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text_)]
    found: dict[str, list[str]] = {}
    for start, end, key, term in spans:
        covered = any(s <= start and end <= e and (e - s) > (end - start) and k != key for s, e, k, _ in spans)
        if not covered:
            found.setdefault(key, []).append(term)
    return found


# Closed-world check: every word of a question must be accounted for by a metric or grouping term,
# a constraint the code applies (years, numbers, ranking, share, trend, date ranges) or one of these
# generic words. Any other content word may be a condition the catalog cannot represent
# ("estimated", "with more than one payment method", "unique products"), so the question is refused
# instead of answered with that condition silently ignored.
# Deliberately NOT generic: before/after/since/until/than (conditions), unique/distinct (only in
# metric terms), average/mean (only directly before an average metric).
_GENERIC_WORDS = frozenset("""
what which who how when where is are was were be been being do does did has have had can you me i we us our
my your their its it there this that these those the a an of in on at for to from by per each every all and
or vs versus as with within during between across over overall total totals s please show list give tell
display see saw get let know compare compared comparison breakdown break down rank ranked ranking ranks
top bottom highest lowest largest smallest biggest greatest longest shortest most least more less fewest
maximum minimum max min leading high low higher lower share percentage percent proportion distribution
contribution come comes came generated generate make made place placed purchased purchase year years yearly
annual annually period periods relationship related relation brazil brazilian platform marketplace store
company business customer product products item items order delivery deliveries delivered days often
number amount much many sum whole entire full trend change changes changing evolve evolved evolution
median correlation correlated variance standard deviation spread
""".split())
_NUMBER_WORDS = frozenset("one two three four five six seven eight nine ten twenty".split())
_AVERAGE_WORDS = frozenset(("average", "avg", "mean"))
_AVERAGE_METRICS = ("average_order_value", "average_delivery_days", "average_review_score")


def _term_spans(text_: str, terms: dict[str, tuple[str, ...]], keys=None) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for key, key_terms in terms.items() if keys is None or key in keys
            for term in key_terms
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text_)]


def unexplained_words(question: str) -> list[str]:
    """Words of the question that no metric, grouping, constraint or generic word accounts for."""
    from retail_analyst.question import read_ranges  # local import: question.py is imported by this module

    text_ = read_ranges(question or "")[0].lower()
    covered = [False] * len(text_)
    for start, end in _term_spans(text_, METRIC_TERMS) + _term_spans(text_, GROUPING_TERMS):
        covered[start:end] = [True] * (end - start)
    average_spans = _term_spans(text_, METRIC_TERMS, keys=_AVERAGE_METRICS)
    tokens = list(re.finditer(r"[a-z]+|\d+", text_))
    unexplained = []
    for i, tok in enumerate(tokens):
        word = tok.group(0)
        if all(covered[tok.start():tok.end()]) or word.isdigit() or word in _GENERIC_WORDS or word in _NUMBER_WORDS:
            continue
        if word in _AVERAGE_WORDS and any(s <= nxt.start() < e for nxt in tokens[i + 1:i + 4]
                                          for s, e in average_spans):
            continue  # "average" directly before an average metric ("average customer rating")
        unexplained.append(word)
    return list(dict.fromkeys(unexplained))


def check_fit(metric: str, grouping: str | None, question: str) -> None:
    """The chosen metric must be named in the question, and the grouping must match what the
    question asks for (a grouping it names, and no named grouping left out)."""
    metrics = _mentioned(question, METRIC_TERMS)
    if metric not in metrics:
        named = ", ".join(f"{k} ({', '.join(dict.fromkeys(v))})" for k, v in metrics.items())
        raise UnsupportedRequest(f"The metric {metric} does not match the question. "
                                 + (f"The question mentions: {named}." if named else
                                    "The question names none of the supported metrics."))
    asked = [g for g in _mentioned(question, GROUPING_TERMS) if g in METRICS[metric].groupings]
    if grouping is None and asked:
        raise UnsupportedRequest(f"The question asks to group by {', '.join(asked)}; set group_by accordingly.")
    if grouping is not None and grouping not in asked:
        raise UnsupportedRequest(f"The question does not ask to group by {grouping}. "
                                 + (f"It asks to group by: {', '.join(asked)}." if asked else
                                    "It names none of this metric's groupings; use group_by null if it asks "
                                    "for one overall total."))


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    metric: str
    grouping: str | None
    calculation: str
    notes: tuple[str, ...] = ()


def catalog_prompt() -> str:
    """The closed list shown to the Data Analyst."""
    metrics = "\n".join(f"- {m.key}: {m.description}. Groupings: {', '.join(m.groupings)}" for m in METRICS.values())
    groupings = "\n".join(f"- {g.key}: {g.description}" for g in GROUPINGS.values())
    return f"Metrics:\n{metrics}\n\nGroupings:\n{groupings}"


def compile_query(request: dict, constraints: Constraints) -> CompiledQuery:
    """Validate a {metric, group_by} request and build the SQL, applying the question's constraints."""
    if not isinstance(request, dict):
        raise UnsupportedRequest("The request must be a JSON object.")
    metric = METRICS.get(str(request.get("metric")).strip().lower()) if request.get("metric") else None
    if metric is None:
        raise UnsupportedRequest(f"Unknown metric {request.get('metric')!r}. Use one of: {', '.join(METRICS)}.")
    group = request.get("group_by")
    group = str(group).strip().lower() if group not in (None, "", "null", "none") else None
    notes = [f"Ignored unknown key(s) {sorted(set(request) - {'metric', 'group_by'})}."] \
        if set(request) - {"metric", "group_by"} else []

    calc = constraints.calculation
    if calc == "growth":
        if group is None and "month" in metric.groupings:
            group, notes = "month", notes + ["Growth is measured month by month."]
        elif group != "month":
            raise UnsupportedRequest("Growth can only be measured over months (group_by month).")
    if group is not None and group not in metric.groupings:
        raise UnsupportedRequest(f"{metric.key} cannot be grouped by {group}. "
                                 f"Allowed groupings: {', '.join(metric.groupings)}.")
    if calc in ("ranking", "share") and group is None:
        raise UnsupportedRequest(f"A {calc} needs a grouping. Allowed: {', '.join(metric.groupings)}.")
    if group is None:
        calc = "total"

    source = SOURCES[metric.source]
    grouping = GROUPINGS[group] if group else None
    select, joins, where = [], [], []
    if grouping:
        expr = f"to_char({source.date}, 'YYYY-MM')" if group == "month" else grouping.expr
        select.append(f"{expr} AS {grouping.column}")
        if group in source.joins:
            joins.append(source.joins[group])
        if grouping.where:
            where.append(grouping.where)
        if group == "month":
            where.append(source.window)  # monthly series only within the reliable period
    select.append(f"{metric.expr} AS {metric.column}")
    if metric.where:
        where.append(metric.where)
    if constraints.years:
        where.append(f"EXTRACT(YEAR FROM {source.date}) IN ({', '.join(map(str, constraints.years))})")

    sql = f"SELECT {', '.join(select)}\nFROM {source.from_sql}"
    sql += "".join(f"\n{j}" for j in dict.fromkeys(joins))
    if where:
        sql += "\nWHERE " + " AND ".join(dict.fromkeys(where))
    if grouping:
        sql += f"\nGROUP BY 1"

    value, label = metric.column, grouping.column if grouping else None
    direction = "ASC" if constraints.ascending else "DESC"
    if calc == "total":
        pass
    elif calc == "growth":
        sql = (f"SELECT {label}, {value}, {value} - LAG({value}) OVER w AS change,\n"
               f"  ROUND(100.0 * ({value} - LAG({value}) OVER w) / NULLIF(LAG({value}) OVER w, 0), 2) AS growth_pct\n"
               f"FROM (\n{sql}\n) AS base\nWINDOW w AS (ORDER BY {label})\nORDER BY {label}")
    elif calc == "share":
        limit = constraints.top_n or MAX_GROUPS
        sql = (f"SELECT {label}, {value},\n  ROUND(100.0 * {value} / NULLIF(SUM({value}) OVER (), 0), 2) AS share_pct\n"
               f"FROM (\n{sql}\n) AS base\nORDER BY {value} {direction} NULLS LAST\nLIMIT {limit}")
    elif calc == "ranking":
        sql += f"\nORDER BY {value} {direction} NULLS LAST\nLIMIT {constraints.top_n or DEFAULT_TOP_N}"
    elif group == "month":
        sql += f"\nORDER BY {label}"
    else:  # breakdown
        sql += f"\nORDER BY {value} {direction} NULLS LAST\nLIMIT {MAX_GROUPS}"
    return CompiledQuery(sql=sql, metric=metric.key, grouping=group, calculation=calc, notes=tuple(notes))
