"""run_sql: validated, read-only SELECT execution with a row cap.

The validator is a guardrail for LLM mistakes; the database role (SELECT-only grants,
read-only sessions, 30 s timeout) remains the real security boundary.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import psycopg
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from retail_analyst.config import get_engine

MAX_ROWS = 500

# Keywords that write or change schema/permissions. Statements must start with SELECT/WITH,
# so this targets what can hide inside one: data-modifying CTEs and SELECT ... INTO.
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|merge|truncate|drop|alter|create|grant|revoke|copy|call|"
    r"execute|into|lock|vacuum|refresh)\b",
    re.IGNORECASE,
)
# Functions with side effects or server/file access, callable from inside a SELECT.
_FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(set_config|pg_sleep\w*|pg_read\w*|pg_ls_\w*|pg_stat_file|lo_\w+|dblink\w*|"
    r"pg_terminate_backend|pg_cancel_backend|pg_reload_conf|pg_notify|pg_advisory\w*|"
    r"nextval|setval|query_to_xml\w*|table_to_xml\w*|cursor_to_xml\w*)\s*\(",
    re.IGNORECASE,
)
_ROW_LOCKS = re.compile(r"\bfor\s+(no\s+key\s+)?(key\s+)?share\b", re.IGNORECASE)

_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRINGS = re.compile(r"'(?:[^']|'')*'")
_QUOTED_IDENTIFIERS = re.compile(r'"(?:[^"]|"")*"')


class UnsafeQueryError(ValueError):
    """Raised when a query fails validation (never sent to the database)."""


@dataclass
class SqlResult:
    ok: bool
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False          # True if more than MAX_ROWS rows existed
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def preview(self, n: int = 10) -> str:
        """Short text rendering for an LLM prompt."""
        if not self.ok:
            return f"ERROR: {self.error}"
        head = [" | ".join(self.columns)]
        head += [" | ".join(str(r[c]) for c in self.columns) for r in self.rows[:n]]
        more = f"\n... {self.row_count} rows total" + (" (truncated at limit)" if self.truncated else "")
        return "\n".join(head) + (more if self.row_count > n or self.truncated else "")


def validate_sql(query: str) -> str:
    """Return the normalised query, or raise UnsafeQueryError explaining why it is rejected."""
    if not isinstance(query, str) or not query.strip():
        raise UnsafeQueryError("Query is empty.")
    if "$" in query or "\\" in query:
        raise UnsafeQueryError("Dollar-quoting, parameters and backslash escapes are not allowed.")

    # Inspect a copy without comments, strings and quoted identifiers so their contents
    # neither trigger false rejections nor hide forbidden keywords.
    stripped = _COMMENTS.sub(" ", query)
    stripped = _STRINGS.sub("''", stripped)
    stripped = _QUOTED_IDENTIFIERS.sub('""', stripped)
    if "'" in stripped.replace("''", "") or "/*" in stripped:
        raise UnsafeQueryError("Unbalanced quotes or comments.")

    body = stripped.strip().rstrip(";").strip()
    if ";" in body:
        raise UnsafeQueryError("Only a single statement is allowed.")
    first_word = re.match(r"\(*\s*(\w+)", body)
    if not first_word or first_word.group(1).lower() not in ("select", "with"):
        raise UnsafeQueryError("Only SELECT or WITH queries are allowed.")
    if match := _FORBIDDEN_KEYWORDS.search(body):
        raise UnsafeQueryError(f"Keyword not allowed: {match.group(1).upper()}.")
    if match := _FORBIDDEN_FUNCTIONS.search(body):
        raise UnsafeQueryError(f"Function not allowed: {match.group(1)}.")
    if _ROW_LOCKS.search(body):
        raise UnsafeQueryError("Row-locking clauses are not allowed.")

    # Remove the trailing semicolon from the original (comments/strings left intact).
    return _COMMENTS.sub(" ", query).strip().rstrip(";").strip()


def _db_message(exc: BaseException | None) -> str:
    """First line of the server's error message (never includes credentials)."""
    diag = getattr(exc, "diag", None)
    message = getattr(diag, "message_primary", None)
    return message or (str(exc).splitlines()[0] if exc else "unknown error")


def _to_jsonable(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    return value


def run_sql(query: str, max_rows: int = MAX_ROWS) -> SqlResult:
    """Validate and execute a read-only query. Never raises; errors are returned in SqlResult."""
    max_rows = max(1, min(int(max_rows), MAX_ROWS))
    try:
        clean = validate_sql(query)
    except UnsafeQueryError as exc:
        return SqlResult(ok=False, sql=str(query), error=f"Rejected: {exc}")

    # Wrapping lets the database stop after max_rows + 1 rows (the extra row detects truncation).
    wrapped = f"SELECT * FROM (\n{clean}\n) AS agent_query LIMIT {max_rows + 1}"
    try:
        with get_engine().connect() as conn:
            dbapi_conn = conn.connection
            # Raw DB-API cursor without parameters: ':' and '%' in the SQL are never
            # interpreted as placeholders.
            cur = dbapi_conn.cursor()
            try:
                cur.execute(wrapped)
                columns = [d.name for d in cur.description]
                raw = cur.fetchall()
            finally:
                cur.close()
                dbapi_conn.rollback()  # read-only anyway; also clears an aborted transaction
    except psycopg.Error as exc:
        return SqlResult(ok=False, sql=clean, error=f"Database error: {_db_message(exc)}")
    except DBAPIError as exc:  # e.g. connection failures raised through SQLAlchemy
        return SqlResult(ok=False, sql=clean, error=f"Database error: {_db_message(exc.orig)}")
    except SQLAlchemyError as exc:
        return SqlResult(ok=False, sql=clean, error=f"Database error: {type(exc).__name__}")

    truncated = len(raw) > max_rows
    rows = [{c: _to_jsonable(v) for c, v in zip(columns, r)} for r in raw[:max_rows]]
    return SqlResult(ok=True, sql=clean, columns=columns, rows=rows,
                     row_count=len(rows), truncated=truncated)
