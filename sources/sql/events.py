"""Functional helper for the `events` table.

All functions are stateless: they take a `database_url` and perform a
side-effect on Postgres. No classes, no module state, no hidden connections.

Requires the `users` and `sessions` tables to exist first (FKs).
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "events.sql"


def create_events_table(database_url: str) -> None:
    """Create the `events` table and its indices if they don't exist."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
