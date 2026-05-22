"""Helpers for the HN front-page tables.

Stateless: each function opens its own psycopg connection. Schema is
bootstrapped lazily by ``create_hn_frontpage_tables`` (also wired into
``run.py`` so a fresh DB gets the tables on first pipeline run).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "hn_frontpage.sql"


def create_hn_frontpage_tables(database_url: str) -> None:
    """Create the three HN front-page tables (idempotent)."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def insert_run(database_url: str, items: list[dict]) -> int:
    """Insert one front-page snapshot + its items in a single transaction.

    Returns the new ``run_id``. ``items`` must be the list of dicts
    produced by ``sources.hackernews.Frontpage``.
    """
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO hn_frontpage_runs (n_items) VALUES (%s) RETURNING id",
            (len(items),),
        )
        row = cur.fetchone()
        assert row is not None
        run_id = int(row[0])
        for it in items:
            submitted_at = it.get("submitted_at") or 0
            submitted_dt = datetime.fromtimestamp(int(submitted_at), tz=timezone.utc) if submitted_at else None
            cur.execute(
                "INSERT INTO hn_frontpage_items "
                "(run_id, hn_id, rank, url, title, summary, points, "
                " num_comments, submitted_at, author) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (run_id, hn_id) DO NOTHING",
                (
                    run_id,
                    int(it["hn_id"]),
                    int(it["rank"]),
                    it["url"],
                    it["title"],
                    it.get("summary", ""),
                    int(it.get("points") or 0),
                    int(it.get("num_comments") or 0),
                    submitted_dt,
                    it.get("by", ""),
                ),
            )
    return run_id


def get_run_items(database_url: str, run_id: int) -> list[dict]:
    """Return all items for one run, ordered by rank."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT hn_id, rank, url, title, summary, points, num_comments, submitted_at, author "
            "  FROM hn_frontpage_items WHERE run_id = %s ORDER BY rank",
            (run_id,),
        )
        rows = cur.fetchall()
    out = []
    for hn_id, rank, url, title, summary, points, num_comments, submitted_at, author in rows:
        out.append(
            {
                "hn_id": int(hn_id),
                "rank": int(rank),
                "url": url,
                "title": title,
                "summary": summary,
                "points": int(points),
                "num_comments": int(num_comments),
                "submitted_at": submitted_at,
                "by": author,
            }
        )
    return out


def replace_user_picks(
    database_url: str,
    user_id: int,
    run_id: int,
    picks: list[tuple[int, float]],
) -> None:
    """Wipe and re-insert the caller's picks for a single (user, run).

    ``picks`` is a list of ``(hn_id, score)``, ordered by score
    descending. Rank is assigned by position. The replace is done in
    one transaction so the feed never sees a partial state.
    """
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM hn_user_picks WHERE user_id = %s AND run_id = %s",
            (int(user_id), int(run_id)),
        )
        for rank, (hn_id, score) in enumerate(picks, start=1):
            cur.execute(
                "INSERT INTO hn_user_picks (user_id, run_id, hn_id, score, rank) VALUES (%s, %s, %s, %s, %s)",
                (int(user_id), int(run_id), int(hn_id), float(score), rank),
            )


def latest_run_id(database_url: str) -> int | None:
    """Return the id of the most recent front-page snapshot, or None."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT MAX(id) FROM hn_frontpage_runs")
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None
