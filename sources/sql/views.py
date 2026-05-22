"""Functional helper for SQL views.

All functions are stateless: they take a `database_url` and perform a
side-effect on Postgres. No classes, no module state, no hidden connections.

Views depend on the `documents` and `users` tables — call after the
create_*_table helpers.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "views.sql"


def create_views(database_url: str) -> None:
    """Create / refresh the SQL views (`user_source_counts`, …)."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
