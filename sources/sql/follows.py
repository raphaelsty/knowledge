"""Idempotent bootstrap for the ``follows`` table."""

from __future__ import annotations

from pathlib import Path

import psycopg

_SQL_PATH = Path(__file__).parent / "follows.sql"


def create_follows_table(database_url: str) -> None:
    """Create the directed follow graph table.

    Idempotent — the SQL is all ``CREATE TABLE IF NOT EXISTS`` /
    ``CREATE INDEX IF NOT EXISTS``. Safe to call on every pipeline
    bootstrap (run.py invokes it alongside the other ``create_*_table``
    helpers).
    """
    sql = _SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
