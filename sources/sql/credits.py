"""Schema bootstrap for the credit-billing tables.

Kept separate from `sources/sql/users.py` so the credits feature is a
clean drop-in / drop-out. `run.py` imports `create_credits_tables`
and calls it alongside the other `create_*_table` helpers — no other
file should depend on this module.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "credits.sql"


def create_credits_tables(database_url: str) -> None:
    """Create the credit_events + polar_customers tables and the
    atomic-ledger helper functions if they don't exist yet.

    Idempotent — every statement in credits.sql is `IF NOT EXISTS`
    or `CREATE OR REPLACE`.
    """
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
