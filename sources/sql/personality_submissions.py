"""Functional helpers for the `personality_submissions` table.

User-submitted suggestions for new VIP personalities. Reviewed
manually by an admin before any `users` row is created.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "personality_submissions.sql"


def create_personality_submissions_table(database_url: str) -> None:
    """Create the `personality_submissions` table + indices if absent."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
