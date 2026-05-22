"""Helpers for `index_health_checks` — search-index diagnostic history.

Stateless: each function opens its own psycopg connection. The table
is bootstrapped lazily by callers via `create_index_health_checks_table`
(also wired into `run.py`'s schema-bootstrap block).
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "index_health_checks.sql"


def create_index_health_checks_table(database_url: str) -> None:
    """Create the table + indices if they don't exist (idempotent)."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def record_index_check(
    database_url: str,
    user_id: int,
    index_name: str,
    status: str,
    *,
    num_documents: int | None = None,
    num_embeddings: int | None = None,
    metadata_count: int | None = None,
    avg_doclen: float | None = None,
    pg_total_docs: int | None = None,
    pg_indexed_docs: int | None = None,
    details: dict | None = None,
    error: str | None = None,
) -> None:
    """Insert one row. Best-effort — never raises (callers shouldn't crash
    on a logging failure)."""
    sql = (
        "INSERT INTO index_health_checks "
        "(user_id, index_name, status, num_documents, num_embeddings, "
        " metadata_count, avg_doclen, pg_total_docs, pg_indexed_docs, "
        " details, error) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)"
    )
    try:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    int(user_id),
                    index_name,
                    status,
                    num_documents,
                    num_embeddings,
                    metadata_count,
                    float(avg_doclen) if avg_doclen is not None else None,
                    pg_total_docs,
                    pg_indexed_docs,
                    json.dumps(details or {}),
                    error,
                ),
            )
    except Exception as exc:
        print(f"  (index_health_checks insert failed: {exc})")


def users_by_check_priority(
    database_url: str,
    *,
    vip_only: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Return users ordered by check-staleness for the next sweep.

    Sort:
      1. VIPs first (so a healthy VIP signal is always recent).
      2. Within each tier, oldest last-check first; never-checked at
         the very front (`NULL` < anything).
      3. Name as a deterministic tiebreaker.

    Each dict: ``id``, ``slug``, ``name``, ``index_name``, ``vip``,
    ``last_check_at`` (timestamptz | None), ``last_status`` (str | None).
    """
    sql = (
        "SELECT u.id, u.username, u.name, u.index_name, u.vip, "
        "       latest.checked_at, latest.status "
        "  FROM users u "
        "  LEFT JOIN LATERAL ("
        "    SELECT checked_at, status "
        "      FROM index_health_checks "
        "     WHERE user_id = u.id "
        "     ORDER BY checked_at DESC "
        "     LIMIT 1"
        "  ) AS latest ON TRUE "
    )
    if vip_only:
        sql += " WHERE u.vip = TRUE "
    sql += " ORDER BY u.vip DESC, " "          latest.checked_at ASC NULLS FIRST, " "          u.name "
    if limit is not None:
        sql += f" LIMIT {int(limit)} "

    out: list[dict] = []
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        for id_, slug, name, idx, vip, last_at, last_status in cur.fetchall():
            out.append(
                {
                    "id": id_,
                    "slug": slug,
                    "name": name or slug,
                    "index_name": idx or slug,
                    "vip": bool(vip),
                    "last_check_at": last_at,
                    "last_status": last_status,
                }
            )
    return out
