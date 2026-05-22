"""Functional helper for the `twitter_feed_status` singleton table.

Mirrors the shape of `sources.sql.events` etc. — stateless `create_*`
function that executes the .sql file on the supplied database URL.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "twitter_feed_status.sql"


def create_twitter_feed_status_table(database_url: str) -> None:
    """Create the single-row `twitter_feed_status` table if missing.
    Also seeds the sentinel id=1 row so the upsert path takes the
    cheap UPDATE branch on every heartbeat."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
