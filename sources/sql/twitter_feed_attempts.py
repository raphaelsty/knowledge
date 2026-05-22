"""Functional helper for the `twitter_feed_attempts` table.

Mirrors the shape of `sources.sql.twitter_feed_status` — one stateless
``create_*`` function that executes the .sql file on the supplied
database URL.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "twitter_feed_attempts.sql"


def create_twitter_feed_attempts_table(database_url: str) -> None:
    """Create the `twitter_feed_attempts` table if missing."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
