"""Functional helpers for the `dead_urls` table.

Persist URLs that failed the link probe so subsequent pipeline runs
don't re-fetch and re-kill them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "dead_urls.sql"


def create_dead_urls_table(database_url: str) -> None:
    """Create the `dead_urls` table if it doesn't exist."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def load_dead_urls(database_url: str) -> set[str]:
    """Return the set of all URLs known to be dead."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT url FROM dead_urls")
        return {row[0] for row in cur.fetchall()}


def mark_urls_dead(database_url: str, urls: Iterable[str]) -> None:
    """Insert (or refresh checked_at for) the given dead URLs."""
    rows = [(u,) for u in urls]
    if not rows:
        return
    sql = "INSERT INTO dead_urls (url) VALUES (%s) ON CONFLICT (url) DO UPDATE SET checked_at = now()"
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
