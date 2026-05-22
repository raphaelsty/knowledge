"""Functional helpers for the `auth_sessions` table.

The Python side only needs to create the table during bootstrap; the
Rust API owns reads/writes at request time.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "auth_sessions.sql"


def create_auth_sessions_table(database_url: str) -> None:
    """Create the `auth_sessions` table and its indices if absent."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
