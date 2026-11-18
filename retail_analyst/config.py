"""Settings from .env, the read-only database engine, and live schema metadata.

Read-only access is enforced in three independent layers:
1. The retail_agent_ro role only has SELECT grants (verified separately).
2. Every connection starts with default_transaction_read_only=on and a 30 s statement_timeout.
3. The SQL tool only accepts single SELECT/WITH statements (see tools/sql_tool.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, Engine, create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ALLOWED_SCHEMAS = ("warehouse", "analytics", "dbt_marts")
QUERY_TIMEOUT_MS = 30_000
NUM_CTX = 4096  # Ollama context window: fits the prompts, small enough for a 2 GB GPU


@dataclass(frozen=True)
class Settings:
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str = field(repr=False)  # never shown in repr/logs
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    supervisor_model: str = ""
    data_analyst_model: str = ""
    analytics_model: str = ""

    def model_for(self, agent: str) -> str:
        """Per-agent model override, falling back to OLLAMA_MODEL."""
        return getattr(self, f"{agent}_model", "") or self.ollama_model


@lru_cache
def get_settings() -> Settings:
    required = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing settings in .env: {', '.join(missing)}")
    return Settings(
        pg_host=os.environ["PGHOST"],
        pg_port=int(os.environ["PGPORT"]),
        pg_database=os.environ["PGDATABASE"],
        pg_user=os.environ["PGUSER"],
        pg_password=os.environ["PGPASSWORD"],
        ollama_base_url=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
        ollama_model=os.getenv("OLLAMA_MODEL") or "qwen2.5:1.5b",
        supervisor_model=os.getenv("SUPERVISOR_MODEL", ""),
        data_analyst_model=os.getenv("DATA_ANALYST_MODEL", ""),
        analytics_model=os.getenv("ANALYTICS_MODEL", ""),
    )


def get_llm(agent: str, json_mode: bool = False, max_tokens: int = 400, temperature: float = 0.0):
    """Local Qwen via Ollama: deterministic by default, 4096-token context, bounded output."""
    from langchain_ollama import ChatOllama  # imported lazily so DB-only code does not need Ollama

    s = get_settings()
    return ChatOllama(
        model=s.model_for(agent),
        base_url=s.ollama_base_url,
        temperature=temperature,
        num_ctx=NUM_CTX,
        num_predict=max_tokens,
        format="json" if json_mode else None,
    )


@lru_cache
def get_engine() -> Engine:
    """SQLAlchemy engine whose sessions are read-only with a 30 s statement timeout."""
    s = get_settings()
    url = URL.create(
        "postgresql+psycopg",
        username=s.pg_user,
        password=s.pg_password,
        host=s.pg_host,
        port=s.pg_port,
        database=s.pg_database,
    )
    return create_engine(
        url,
        hide_parameters=True,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        connect_args={
            "options": f"-c default_transaction_read_only=on -c statement_timeout={QUERY_TIMEOUT_MS}",
            "connect_timeout": 10,
        },
    )


# ---------------------------------------------------------------------------
# Schema metadata
# ---------------------------------------------------------------------------
@dataclass
class TableInfo:
    schema: str
    name: str
    kind: str                                  # "table" or "view"
    columns: list[tuple[str, str]]             # (column, type)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[tuple[list[str], str, list[str]]] = field(default_factory=list)
    est_rows: int | None = None                # planner estimate; None for views

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"


_TYPE_ALIASES = {
    "character varying": "text", "timestamp without time zone": "timestamp",
    "timestamp with time zone": "timestamptz", "double precision": "float",
    "ARRAY": "array", "boolean": "bool",
}

_RELATIONS_SQL = text("""
    SELECT n.nspname, c.relname, CASE WHEN c.relkind IN ('v', 'm') THEN 'view' ELSE 'table' END,
           CASE WHEN c.relkind IN ('r', 'p') AND c.reltuples >= 0 THEN c.reltuples::bigint END
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = ANY(:schemas) AND c.relkind IN ('r', 'p', 'v', 'm')
      AND has_table_privilege(c.oid, 'SELECT')
    ORDER BY 1, 2
""")

_COLUMNS_SQL = text("""
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = ANY(:schemas)
    ORDER BY table_schema, table_name, ordinal_position
""")

# pg_catalog is used because information_schema hides constraints of tables we do not own.
_CONSTRAINTS_SQL = text("""
    SELECT n.nspname, c.relname, con.contype,
           ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(num, ord)
                 JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.num
                 ORDER BY k.ord)::text[],
           fn.nspname || '.' || fc.relname,
           ARRAY(SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY k(num, ord)
                 JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.num
                 ORDER BY k.ord)::text[]
    FROM pg_constraint con
    JOIN pg_class c ON c.oid = con.conrelid JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_class fc ON fc.oid = con.confrelid
    LEFT JOIN pg_namespace fn ON fn.oid = fc.relnamespace
    WHERE con.contype IN ('p', 'f') AND n.nspname = ANY(:schemas)
    ORDER BY 1, 2, con.conname
""")


@lru_cache
def load_schema() -> dict[str, TableInfo]:
    """Live metadata for every relation the agent role can SELECT, keyed by 'schema.name'."""
    params = {"schemas": list(ALLOWED_SCHEMAS)}
    with get_engine().connect() as conn:
        tables = {
            f"{s}.{n}": TableInfo(schema=s, name=n, kind=k, columns=[], est_rows=rows)
            for s, n, k, rows in conn.execute(_RELATIONS_SQL, params)
        }
        for s, t, col, dtype in conn.execute(_COLUMNS_SQL, params):
            if f"{s}.{t}" in tables:
                tables[f"{s}.{t}"].columns.append((col, _TYPE_ALIASES.get(dtype, dtype)))
        for s, t, kind, cols, ref_table, ref_cols in conn.execute(_CONSTRAINTS_SQL, params):
            info = tables.get(f"{s}.{t}")
            if info is None:
                continue
            if kind == "p":
                info.primary_key = list(cols)
            else:
                info.foreign_keys.append((list(cols), ref_table, list(ref_cols)))
    return tables


def schema_summary(tables: dict[str, TableInfo] | None = None, include_dq: bool = True) -> str:
    """Compact, prompt-ready description of the readable schema."""
    tables = tables if tables is not None else load_schema()
    lines = []
    for info in tables.values():
        size = f", ~{info.est_rows:,} rows" if info.est_rows else ""
        cols = [(c, t) for c, t in info.columns if include_dq or not c.startswith("dq_")]
        lines.append(f"{info.full_name} ({info.kind}{size})")
        lines.append("  columns: " + ", ".join(f"{c} {t}" for c, t in cols))
        if info.primary_key:
            lines.append("  primary key: " + ", ".join(info.primary_key))
        for cols, ref, ref_cols in info.foreign_keys:
            lines.append(f"  {', '.join(cols)} -> {ref}({', '.join(ref_cols)})")
    return "\n".join(lines)


_PROMPT_SCHEMA_ORDER = ("analytics", "dbt_marts", "warehouse")  # pre-aggregated first


@lru_cache
def concise_schema(exclude: tuple[str, ...] = ()) -> str:
    """Short schema for prompts: names and columns only (no types, no dq_* flags), plus joins.
    Tables in `exclude` are listed by name only (their full columns are shown elsewhere)."""
    tables = load_schema()
    ordered = sorted(tables.values(), key=lambda t: (_PROMPT_SCHEMA_ORDER.index(t.schema), t.name))
    lines = []
    for info in ordered:
        if info.full_name in exclude:
            lines.append(f"{info.full_name} (see full columns above)")
            continue
        cols = [c for c, _ in info.columns if not c.startswith("dq_")]
        lines.append(f"{info.full_name} ({info.kind}): {', '.join(cols)}")
    joins = [
        f"{info.name}.{', '.join(cols)} -> {ref.split('.')[-1]}.{', '.join(ref_cols)}"
        for info in ordered for cols, ref, ref_cols in info.foreign_keys
    ]
    if joins:
        lines.append("Joins (warehouse): " + "; ".join(joins))
    return "\n".join(lines)


def tables_with_column(column: str) -> list[str]:
    """Readable relations that contain the given column."""
    return [name for name, info in load_schema().items() if any(c == column for c, _ in info.columns)]


def _stem(word: str) -> str:
    word = word.lower()
    if word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


# Business words that do not literally appear in table/column names.
_SYNONYMS = {"revenue": {"price", "sale"}, "monthly": {"month"}, "daily": {"date", "day"},
             "delay": {"late", "delivery"}, "delayed": {"late"}, "rating": {"review", "score"}}


def relevant_tables(text_: str, k: int = 3) -> list[str]:
    """Rank relations by word overlap with the question/task (table-name words weigh 3, columns 1)."""
    import re

    words = {_stem(w) for w in re.findall(r"[a-zA-Z]{3,}", text_)}
    words |= {s for w in list(words) for s in _SYNONYMS.get(w, ())}
    scores = {}
    for name, info in load_schema().items():
        table_words = {_stem(w) for w in info.name.split("_")}
        column_words = {_stem(w) for c, _ in info.columns if not c.startswith("dq_") for w in c.split("_")}
        scores[name] = 3 * len(words & table_words) + len(words & column_words)
    ranked = sorted(scores, key=lambda n: (-scores[n], _PROMPT_SCHEMA_ORDER.index(n.split(".")[0]), n))
    return [n for n in ranked[:k] if scores[n] > 0]


def table_details(names: list[str], include_dq: bool = True) -> str:
    """Full metadata (types, keys and optionally dq_* columns) for the given 'schema.table' names."""
    tables = load_schema()
    return schema_summary({n: tables[n] for n in names if n in tables}, include_dq=include_dq)
