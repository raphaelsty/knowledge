"""Schema bootstrap for the sponsor-a-VIP queue.

Lives in its own module so the feature is a clean drop-in / drop-out
on top of the credits ledger.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "vip_sponsorships.sql"


def create_vip_sponsorships_table(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))
