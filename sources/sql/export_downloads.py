"""Bootstrap helper for the `export_downloads` audit table.

Only the table creator lives here; the Rust API owns all runtime
inserts (one row per served export request) and the read paths
(account history, owner audit view).
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "export_downloads.sql"


def create_export_downloads_table(database_url: str) -> None:
    """Create the `export_downloads` audit table if absent."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
