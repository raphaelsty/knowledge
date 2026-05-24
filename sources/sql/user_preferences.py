"""Idempotent bootstrap for the per-viewer preference tables.

`create_user_preferences_tables(database_url)` is wired into the
API's boot migrations (api/src/main.rs) and into run.py so a fresh
deployment / pipeline run lifts both tables in place.

The recompute logic lives in `sources.utils.user_preferences` and
is called separately — this module is schema-only.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

_SQL_PATH = Path(__file__).parent / "user_preferences.sql"


def create_user_preferences_tables(database_url: str) -> None:
    """Create the `user_personality_weight` + `user_category_weight`
    tables and their indices. Idempotent — runs CREATE TABLE IF
    NOT EXISTS / CREATE INDEX IF NOT EXISTS."""
    sql = _SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
