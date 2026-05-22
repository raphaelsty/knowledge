"""One-off: re-date Wikipedia references + HuggingFace likes.

Why this script exists
----------------------
Wikipedia references used to inherit the article's `Last-Modified`
date — but wiki articles get edited constantly, so every ref ended
up dated "today" and flooded the feed. The new extractor pulls a
publication year out of the citation context instead.

HuggingFace likes used to inherit the repo's `last_modified`
timestamp, which is also recent for most popular repos and pushed
likes to the top of the feed. The new extractor uses the like's
own `createdAt` (when the user clicked the heart) from the public
likes endpoint.

This script back-applies both new policies to rows that were
ingested before the code change:

  * Wikipedia: extract a year from the existing `summary` text (no
    external calls). Fall back to today − 5 years.
  * HuggingFace: refetch `/api/users/{hf_username}/likes` per
    personality (unauth, free) and map URL → createdAt. Fall back to
    leaving the row alone if the like is no longer there.

PG `documents.date` and the per-personality SQLite
`indexes/{slug}/metadata.db` METADATA.date column are updated in
lockstep so the feed and search results stay coherent without a
full reindex.

Usage::

    DATABASE_URL=... uv run python scripts/backfill_wiki_hf_dates.py
    DATABASE_URL=... uv run python scripts/backfill_wiki_hf_dates.py --dry
    DATABASE_URL=... uv run python scripts/backfill_wiki_hf_dates.py --only wiki
    DATABASE_URL=... uv run python scripts/backfill_wiki_hf_dates.py --only hf
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path

import psycopg
import requests

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
INDEXES_DIR = Path(__file__).resolve().parent.parent / "indexes"

# Same regex shape as `sources.wikipedia.references._YEAR_RE`.
_YEAR_RE = re.compile(r"(?:^|[^0-9])((?:19|20)\d{2})(?:[^0-9]|$)")
# Same default backshift the live extractor uses.
_WIKI_DATE_BACKSHIFT_YEARS = 5


def _extract_year(text: str) -> int | None:
    """Most-plausible 4-digit year in `text`, or None.

    Picks the year that appears most often; ties broken by earliest
    occurrence. Years > current year are ignored (typo guard).
    """
    if not text:
        return None
    this_year = datetime.date.today().year
    counts: dict[int, int] = {}
    first: dict[int, int] = {}
    for i, m in enumerate(_YEAR_RE.finditer(text)):
        y = int(m.group(1))
        if 1900 <= y <= this_year:
            counts[y] = counts.get(y, 0) + 1
            first.setdefault(y, i)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0][0]


def _wiki_fallback_date() -> str:
    """today − `_WIKI_DATE_BACKSHIFT_YEARS`, as `YYYY-MM-DD`."""
    today = datetime.date.today()
    return today.replace(year=today.year - _WIKI_DATE_BACKSHIFT_YEARS).isoformat()


# ────────────────────────────────────────────────────────────────────
# SQLite mirror helpers
# ────────────────────────────────────────────────────────────────────


def _sqlite_path(slug: str) -> Path | None:
    p = INDEXES_DIR / slug / "metadata.db"
    return p if p.exists() else None


def _update_sqlite_dates(slug: str, rows: list[tuple[str, str]]) -> int:
    """Bulk UPDATE METADATA.date by URL. Returns rows touched."""
    path = _sqlite_path(slug)
    if path is None or not rows:
        return 0
    n = 0
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.executemany(
            "UPDATE METADATA SET date = ? WHERE url = ?",
            [(d, u) for (u, d) in rows],
        )
        n = cur.rowcount
        conn.commit()
    return n


# ────────────────────────────────────────────────────────────────────
# Wikipedia backfill
# ────────────────────────────────────────────────────────────────────


def backfill_wikipedia(conn, dry: bool) -> None:
    """Re-date every wikipedia-tagged doc from its summary."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT d.url, d.summary, d.date, u.username
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE d.source = 'wikipedia'
           AND d.deleted = false
        """
    )
    rows = cur.fetchall()
    print(f"wikipedia: {len(rows)} doc(s) to consider")

    fallback = _wiki_fallback_date()
    per_slug: dict[str, list[tuple[str, str]]] = {}
    pg_updates: list[tuple[str, str, str]] = []  # (slug, url, new_date)
    extracted = 0
    fellback = 0
    skipped = 0

    for url, summary, old_date, slug in rows:
        year = _extract_year(summary or "")
        new_date = f"{year}-01-01" if year else fallback
        if year:
            extracted += 1
        else:
            fellback += 1
        if str(old_date) == new_date:
            skipped += 1
            continue
        pg_updates.append((slug, url, new_date))
        per_slug.setdefault(slug, []).append((url, new_date))

    print(f"  extracted_year={extracted} fellback={fellback} no_change={skipped} to_update={len(pg_updates)}")

    if dry or not pg_updates:
        return

    cur.executemany(
        "UPDATE documents SET date = %s::date, updated_at = now() WHERE url = %s",
        [(d, u) for (_, u, d) in pg_updates],
    )
    conn.commit()
    print(f"  pg: updated {cur.rowcount} row(s)")

    sqlite_touched = 0
    for slug, batch in per_slug.items():
        sqlite_touched += _update_sqlite_dates(slug, batch)
    print(f"  sqlite: updated {sqlite_touched} row(s) across {len(per_slug)} index(es)")


# ────────────────────────────────────────────────────────────────────
# HuggingFace likes backfill
# ────────────────────────────────────────────────────────────────────


def _fetch_hf_likes(hf_username: str) -> list[dict]:
    """Public `/api/users/{u}/likes` — returns each entry with
    `createdAt` + `repo.{name,type}`. Paced with a small sleep to
    stay under HF's ~480 rpm public ceiling."""
    url = f"https://huggingface.co/api/users/{urllib.parse.quote(hf_username)}/likes"
    r = requests.get(
        url,
        headers={"User-Agent": "Knowledge/1.0 backfill"},
        timeout=30,
    )
    time.sleep(0.15)
    if r.status_code != 200:
        print(f"    HF likes for @{hf_username}: HTTP {r.status_code}")
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def _hf_url_for(repo_id: str, kind: str) -> str:
    if kind == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    if kind == "space":
        return f"https://huggingface.co/spaces/{repo_id}"
    return f"https://huggingface.co/{repo_id}"


def backfill_huggingface(conn, dry: bool) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.username, u.sources->'huggingface'
          FROM users u
         WHERE u.sources ? 'huggingface'
           AND EXISTS (
             SELECT 1 FROM documents d
              WHERE d.user_id = u.id AND d.source = 'huggingface'
                AND d.deleted = false
           )
         ORDER BY u.username
        """
    )
    targets = []
    for slug, hf_cfg in cur.fetchall():
        if isinstance(hf_cfg, str) and hf_cfg:
            targets.append((slug, hf_cfg))
    print(f"huggingface: {len(targets)} personality/personalities to refetch")

    total_pg = 0
    total_sqlite = 0
    for slug, hf_user in targets:
        # Pull current docs for this slug so we know which URLs to map.
        cur.execute(
            """
            SELECT d.url, d.date
              FROM documents d
              JOIN users u ON u.id = d.user_id
             WHERE u.username = %s
               AND d.source = 'huggingface'
               AND d.deleted = false
            """,
            (slug,),
        )
        current = dict(cur.fetchall())
        if not current:
            continue

        items = _fetch_hf_likes(hf_user)
        print(f"  @{slug} (hf={hf_user}): {len(items)} like(s) returned, {len(current)} stored doc(s)")

        url_to_date: dict[str, str] = {}
        for it in items:
            repo = it.get("repo") or {}
            name = repo.get("name") or ""
            kind = repo.get("type") or "model"
            if not name:
                continue
            built = _hf_url_for(name, kind)
            stamp = it.get("createdAt") or it.get("likedAt") or ""
            if isinstance(stamp, str) and len(stamp) >= 10:
                url_to_date[built] = stamp[:10]

        pg_updates: list[tuple[str, str]] = []
        for u_, old in current.items():
            new = url_to_date.get(u_)
            if not new:
                # Like is gone from HF (unliked); leave row alone.
                continue
            if str(old) == new:
                continue
            pg_updates.append((u_, new))

        print(f"    matched_new={len(pg_updates)} (rest unchanged or unliked)")
        if dry or not pg_updates:
            continue

        cur.executemany(
            "UPDATE documents SET date = %s::date, updated_at = now() WHERE url = %s",
            [(d, u_) for (u_, d) in pg_updates],
        )
        conn.commit()
        total_pg += cur.rowcount

        n = _update_sqlite_dates(slug, pg_updates)
        total_sqlite += n
        print(f"    pg={cur.rowcount} sqlite={n}")

    print(f"huggingface: pg={total_pg} sqlite={total_sqlite}")


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=("wiki", "hf"),
        help="Run only one side of the backfill.",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print what would change; don't write to PG or SQLite.",
    )
    args = p.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(database_url) as conn:
        if args.only != "hf":
            backfill_wikipedia(conn, args.dry)
        if args.only != "wiki":
            backfill_huggingface(conn, args.dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
