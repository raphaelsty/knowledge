"""Functional helpers for the `favorites` table.

Only the bootstrap creator lives here; the Rust API owns all runtime
reads and writes (one row per toggle, cheap).
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "favorites.sql"
DOC_SQL_PATH = Path(__file__).parent / "favorite_documents.sql"


def create_favorites_table(database_url: str) -> None:
    """Create the `favorites` + `favorite_documents` tables if absent.

    Both schemas are per-user and independent — personalities vs
    document URLs — but we bootstrap them together so a fresh DB
    comes up with both tables ready.
    """
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
        conn.execute(DOC_SQL_PATH.read_text(encoding="utf-8"))
