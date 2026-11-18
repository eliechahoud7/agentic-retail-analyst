"""Deterministic reading of explicit constraints in a business question.

The language models never decide these: years, "top N", sort direction and the kind of
calculation are read from the question by code and enforced when the query is compiled.
Questions the historical data cannot answer (causes, forecasts, relative or missing periods)
are recognised here so the system refuses instead of guessing.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from retail_analyst.labels import CATEGORY_NAMES

DATA_YEARS = (2016, 2017, 2018)  # orders run from 2016-09 to 2018-10
DEFAULT_TOP_N = 10
MAX_TOP_N = 50

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "twenty": 20}
_RANK_WORDS = r"(top|bottom|highest|lowest|largest|smallest|biggest|best|worst|most|least|leading)"
_TOP_N = re.compile(rf"\b{_RANK_WORDS}\s+(\d{{1,3}}|{'|'.join(_NUMBER_WORDS)})\b"
                    rf"|\b(\d{{1,3}}|{'|'.join(_NUMBER_WORDS)})\s+{_RANK_WORDS}\b", re.IGNORECASE)
_COUNT_AFTER = re.compile(rf"\b(which|the|list|show|give me|name)\s+(\d{{1,2}}|{'|'.join(_NUMBER_WORDS)})\b",
                          re.IGNORECASE)  # "Which 3 categories ...", "the five states ..." (rankings only)
_ASCENDING = re.compile(r"\b(lowest|least|bottom|smallest|fewest|minimum|worst[- ]?sell\w*)\b", re.IGNORECASE)
_RANKING = re.compile(rf"\b{_RANK_WORDS}\b|\b(rank\w*|which|maximum|minimum|fewest)\b", re.IGNORECASE)
_SHARE = re.compile(r"\b(share|percentage|proportion|percent of|distribution|split)\b|%\s*of", re.IGNORECASE)
_GROWTH = re.compile(r"\b(grow\w*|growth|trend\w*|evolv\w*|evolution|over time|month[- ]over[- ]month|changed|"
                     r"increas\w*|decreas\w*|declin\w*)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

_CAUSAL = re.compile(r"\bwhy\b|\breasons?\s+(for|why|behind)\b|\bcaus\w*\b", re.IGNORECASE)
_FORECAST = re.compile(r"\b(will|forecast\w*|predict\w*|projection\w*|projected|future)\b"
                       r"|\bnext\s+(year|quarter|month|week)\b", re.IGNORECASE)
_RELATIVE = re.compile(r"\b(last|this|previous|past|current)\s+(year|quarter|month|week)\b"
                       r"|\b(today|yesterday|recently|lately|year[- ]to[- ]date|ytd)\b", re.IGNORECASE)

# Qualifiers and filters the metric catalog cannot represent. Ignoring them gives a confident
# wrong answer (independent question set: "cancelled orders" answered with all orders, "new
# customers" with active customers, "worst reviews" with the number of reviews, "received in
# 2018" by purchase date). Such questions are refused, with a reason.
_STATUS = re.compile(r"\b(cancel\w*|returned|returns|refund\w*|undelivered|not delivered|unavailable|failed)\b",
                     re.IGNORECASE)
# Customer lifecycle concepts, detected in context: "new" alone is not one ("new marketing
# strategies", "new categories" are not about customer types).
_CUSTOMER_TYPE = re.compile(
    r"\b(new|first[- ]time|returning|repeat|recurring|loyal|existing|one[- ]time)\s+"
    r"(customers?|buyers?|shoppers?|clients?|purchasers?)\b"
    r"|\brepeat\s+(purchases?|orders?|buying|business)\b"
    r"|\b(customer\s+)?(retention|churn\w*|loyalty|lifetime value)\b"
    r"|\b(come|comes|came|coming)\s+back\b|\bsecond\s+(purchase|order)\b|\bbought\s+again\b"
    r"|\b(acquir\w*|acquisition)\b[^.?!]*\bcustomers?\b|\bcustomers?\b[^.?!]*\b(acquir\w*|acquisition)\b",
    re.IGNORECASE)
# Events with their own date (not the purchase date). Receiving something as a destination or
# recipient ("Which states receive the most orders?") is not such an event.
_EVENT_VERB = r"(receive|received|receiving|written|wrote|submit|submitted|posted|answered|created)"
_PARTICIPLES = {"receive": "received", "receiving": "received", "wrote": "written", "submit": "submitted"}
_EVENT_DATE = re.compile(
    rf"\b(review\w*|ratings?|feedback|comments?)\b[^.?!]*\b{_EVENT_VERB}\b"      # reviews received/written
    rf"|\b{_EVENT_VERB}\b[^.?!]*\b(review\w*|ratings?|feedback|comments?)\b"      # received ... reviews
    rf"|\bcustomers?\s+(?:\w+\s+){{0,2}}(receive|received|receiving)\b",          # customers received = delivery
    re.IGNORECASE)
_POPULARITY = re.compile(r"\b(popular\w*|favou?rite|most common|least common|most used|least used|"
                         r"most frequent\w*|least frequent\w*)\b", re.IGNORECASE)
_BEST_WORST = re.compile(r"\b(best|worst)\b(?![- ]?sell)", re.IGNORECASE)  # "best-selling" is fine
_TIME_GRAIN = re.compile(r"\b(daily|per day|each day|by day|weekly|per week|each week|by week|weekdays?|weekends?|"
                         r"day of the week|quarterly|per quarter|each quarter|by quarter|hourly|per hour)\b",
                         re.IGNORECASE)
_ONE_PERIOD = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|"
                         r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Q[1-4])\b"
                         r"|\b(first|second|third|fourth|1st|2nd|3rd|4th) quarter\b")  # case-sensitive
_PAYMENT_VALUE = re.compile(r"\b(credit cards?|debit cards?|boletos?|vouchers?)\b", re.IGNORECASE)
_STATE_CODE = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")
_STATE_NAMES = ("acre", "alagoas", "amapa", "amazonas", "bahia", "ceara", "distrito federal", "espirito santo",
                "goias", "maranhao", "mato grosso", "minas gerais", "para", "paraiba", "parana", "pernambuco",
                "piaui", "rio de janeiro", "rio grande do norte", "rio grande do sul", "rondonia", "roraima",
                "santa catarina", "sao paulo", "sergipe", "tocantins")


# Date ranges ("from September 2016 to October 2018"). A range is not one specific period: it is
# read as no filter when it covers all the data, as a year filter when it spans whole calendar
# years, and is refused otherwise (a partial-year range cannot be represented).
_MONTH_NUMBER = {m: i for i, m in enumerate(("January", "February", "March", "April", "May", "June", "July",
                                             "August", "September", "October", "November", "December"), 1)}
_RANGE = re.compile(r"\b(?:from\s+|between\s+)?(" + "|".join(_MONTH_NUMBER) + r")\s+(\d{4})\s*"
                    r"(?:to|until|through|and|-)\s*(" + "|".join(_MONTH_NUMBER) + r")\s+(\d{4})\b")
_ENDPOINT_COMPARISON = re.compile(r"\b(percentage|percent|increase\w*|decrease\w*|difference|grew by|growth rate)\b",
                                  re.IGNORECASE)
DATA_START, DATA_END = (2016, 9), (2018, 10)


def read_ranges(question: str) -> tuple[str, tuple[int, ...] | None, str | None]:
    """(question without range text, years implied by the ranges or None, refusal or None)."""
    years: set[int] = set()
    found = False
    for m in _RANGE.finditer(question or ""):
        found = True
        start, end = (int(m.group(2)), _MONTH_NUMBER[m.group(1)]), (int(m.group(4)), _MONTH_NUMBER[m.group(3)])
        if _ENDPOINT_COMPARISON.search(question):
            return question, None, (f"The question compares two specific months (\"{m.group(0).strip()}\"), but the "
                                    "supported metrics cover whole years or every month, so an answer could describe "
                                    "a different period.")
        if start <= DATA_START and end >= DATA_END:
            continue  # the whole dataset: no filter
        if start[1] == 1 and end[1] == 12 and start[0] <= end[0]:
            years.update(range(start[0], end[0] + 1))  # whole calendar years
            continue
        return question, None, None  # partial range: left to the single-period rule (refused)
    if not found:
        return question, None, None
    return _RANGE.sub(" ", question), tuple(sorted(years)), None


def _plain(text_: str) -> str:
    """Lowercase, accents removed, punctuation as spaces: 'São Paulo' -> 'sao paulo'."""
    text_ = unicodedata.normalize("NFKD", text_).encode("ascii", "ignore").decode("ascii").lower()
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", text_).split()) + " "


def _category_names() -> list[str]:
    names = set()
    for key, display in CATEGORY_NAMES.items():
        names.add(_plain(key.replace("_", " ")).strip())
        names.add(_plain(display).strip())
    return sorted(names - {"uncategorized"}, key=len, reverse=True)


_CATEGORY_NAMES = _category_names()


def unsupported_qualifier(question: str) -> str | None:
    """A reason to refuse when the question depends on something the catalog cannot represent."""
    q, plain = question or "", _plain(question or "")
    if m := _STATUS.search(q):
        return (f"The question is about \"{m.group(0)}\" orders, but the supported metrics do not separate "
                "orders by status, so an answer would be misleading.")
    if m := _CUSTOMER_TYPE.search(q):
        return (f"The question depends on \"{m.group(0)}\", but the supported metrics do not separate new "
                "and returning customers, so an answer would be misleading.")
    if _EVENT_DATE.search(q):
        verb = re.search(_EVENT_VERB, q, re.IGNORECASE)
        participle = _PARTICIPLES.get(verb.group(0).lower(), verb.group(0).lower()) if verb else "received"
        return (f"The question refers to when something was {participle}, but the data is analysed by "
                "purchase date, so an answer could describe a different period.")
    if m := _POPULARITY.search(q):
        return (f"\"{m.group(0)}\" depends on how often something is used, which the supported metrics do "
                "not measure.")
    if m := _BEST_WORST.search(q):
        return (f"\"{m.group(0)}\" does not say which measure to use. Please ask for the highest or lowest "
                "value of a specific measure, for example the lowest average review score.")
    if m := _TIME_GRAIN.search(q):
        return (f"Grouping \"{m.group(0)}\" is not supported; the supported periods are months and years.")
    if m := _ONE_PERIOD.search(q):
        return (f"The question is about one specific period (\"{m.group(0)}\"), but the supported metrics "
                "cover whole years or every month, so an answer could describe a different period.")
    if m := _PAYMENT_VALUE.search(q):
        return (f"The question is about one payment method (\"{m.group(0)}\"). Answers can compare all "
                "payment methods, but not filter one.")
    state = _STATE_CODE.search(q)
    state = state.group(0) if state else next((s for s in _STATE_NAMES if f" {s} " in plain), None)
    if state:
        return (f"The question is about one state ({state.upper() if len(state) == 2 else state.title()}). "
                "Answers can compare all states, but not filter one.")
    if category := next((c for c in _CATEGORY_NAMES if f" {c} " in plain), None):
        return (f"The question is about one product category (\"{category}\"). Answers can compare all "
                "categories, but not filter one.")
    return None


@dataclass(frozen=True)
class Constraints:
    years: tuple[int, ...] = ()
    top_n: int | None = None
    ascending: bool = False
    calculation: str = "breakdown"   # ranking | share | growth | breakdown
    refusal: str | None = None       # why the question cannot be answered from the data


def parse_question(question: str) -> Constraints:
    q, range_years, range_refusal = read_ranges(question or "")
    if range_refusal:
        return Constraints(refusal=range_refusal)
    if _CAUSAL.search(q):
        return Constraints(refusal="The data shows what happened, but not why it happened, so causes "
                                   "cannot be determined from it.")
    if _FORECAST.search(q):
        return Constraints(refusal="The data only covers past orders (2016 to 2018), so it cannot be "
                                   "used to forecast future values.")
    if match := _RELATIVE.search(q):
        return Constraints(refusal=f"The question uses a relative period (\"{match.group(0)}\"). The data "
                                   "covers 2016 to 2018; please name the year, for example 2018.")
    if reason := unsupported_qualifier(q):
        return Constraints(refusal=reason)
    years = tuple(sorted({int(y) for y in _YEAR.findall(q)} | set(range_years or ())))
    if missing := [y for y in years if y not in DATA_YEARS]:
        return Constraints(refusal=f"The data covers September 2016 to October 2018, so there is no data "
                                   f"for {', '.join(map(str, missing))}.")

    raw = None
    if match := _TOP_N.search(q):
        raw = next(g for g in (match.group(2), match.group(3)) if g)
    elif _RANKING.search(q) and (match := _COUNT_AFTER.search(q)):
        raw = match.group(2)
    top_n = min(int(raw) if raw.isdigit() else _NUMBER_WORDS[raw.lower()], MAX_TOP_N) or None if raw else None

    if _GROWTH.search(q):
        calculation = "growth"
    elif _SHARE.search(q):
        calculation = "share"
    elif top_n or _RANKING.search(q):
        calculation = "ranking"
    else:
        calculation = "breakdown"
    return Constraints(years=years, top_n=top_n, ascending=bool(_ASCENDING.search(q)), calculation=calculation)
