"""Functional helpers for the `oauth_identities` table.

Provider-verified third-party identities (currently: GitHub). Lets
us tell apart user-typed `sources.github` strings (unverifiable)
from OAuth-confirmed account ownership.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "oauth_identities.sql"


def create_oauth_identities_table(database_url: str) -> None:
    """Create the `oauth_identities` table + indices if absent."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
