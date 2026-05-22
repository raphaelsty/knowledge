"""Schema bootstrap for `user_storage`.

Dedicated module so the storage-billing feature stays decoupled from
the rest of the credit infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "user_storage.sql"


def create_user_storage_table(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
