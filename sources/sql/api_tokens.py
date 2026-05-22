"""Helpers for `api_tokens` — user-scoped bearer tokens for upload auth.

The Rust API owns the create/list/revoke flows; this Python module
exists so `run.py` can bootstrap the schema alongside every other
table.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "api_tokens.sql"


def create_api_tokens_table(database_url: str) -> None:
    """Create the table + indices if they don't exist (idempotent)."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
