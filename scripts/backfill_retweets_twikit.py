#!/usr/bin/env python3
"""Rebuild retweet doc summaries + linked_urls via twikit (no API cost).

What this script does — and why
-------------------------------
A retweet's wrapper text on twitter.com is only ~140 chars of the
original tweet and carries no media of its own. When the pipeline
first ingested those tweets, the original full text + photos +
videos lived only on the inner ``retweeted_tweet`` payload, which we
didn't recurse into. The result: retweet docs read like
``"RT @user: I feel what these results…"`` and never showed the
actual body or images of the source tweet.

The fix is the same idea as ``backfill_twitter_urls.py`` — look up
the wrapper tweet, dive into ``retweeted_tweet``, and rewrite
``summary`` + ``linked_urls`` + ``link_hosts`` from the inner
payload through the existing pipeline helpers. The difference is the
*lookup mechanism*: this script hits twikit (cookie-authenticated
x.com) instead of twitterapi.io, so no twitterapi.io credit is
consumed.

For each tweet doc whose summary still starts with ``RT @`` (the
old, truncated form) we:

  1. Pull the wrapper tweet ID out of the document URL
     (``/status/<id>`` segment).
  2. Batch-fetch the wrappers via :meth:`twitter.Bookmarks.lookup`,
     which routes them through :func:`_twikit_to_dict` and returns
     the twitterapi.io-shaped dicts the pipeline helpers expect.
  3. Re-render through :func:`_tweet_self_sufficient_summary` (which
     recurses into ``retweeted_tweet``) and :func:`_build_linked_urls`
     (which OG-previews every external URL the inner tweet cites).
  4. Write the new ``summary`` / ``linked_urls`` / ``link_hosts``
     to PG and mirror them into the slug's SQLite metadata so the
     search index reflects the change without a reindex.

Cookies
-------
Twitter cookies come from Safari by default (via
:func:`sources.twitter.cookies.get_safari_cookies`). To use a
different session, set ``TWITTER_AUTH_TOKEN`` + ``TWITTER_CT0``
env vars.

Usage
-----
::

    DATABASE_URL=... uv run python scripts/backfill_retweets_twikit.py --slug tony-wu
    DATABASE_URL=... uv run python scripts/backfill_retweets_twikit.py --slug tony-wu --limit 50 --dry
    DATABASE_URL=... uv run python scripts/backfill_retweets_twikit.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

import psycopg

from sources.twitter.bookmarks import Bookmarks
from sources.twitter.tweets import (
    _build_linked_urls,
    _link_source_for,
    _parse_date,
    _retweet_extra_tags,
    _tweet_display_title,
    _tweet_self_sufficient_summary,
)
from sources.utils.cleaning import clean_summary, clean_title

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
INDEXES_DIR = Path(__file__).resolve().parent.parent / "indexes"
TWEET_ID_RE = re.compile(r"/status/(\d+)")


# ────────────────────────────────────────────────────────────────────
# Cookie loading
# ────────────────────────────────────────────────────────────────────


def _load_cookies() -> tuple[str, str]:
    """Return (auth_token, ct0). Env beats Safari extraction."""
    tok = os.environ.get("TWITTER_AUTH_TOKEN")
    ct0 = os.environ.get("TWITTER_CT0")
    if tok and ct0:
        return tok, ct0
    from sources.twitter.cookies import get_safari_cookies

    creds = get_safari_cookies()
    return creds["auth_token"], creds["ct0"]


# ────────────────────────────────────────────────────────────────────
# Doc selection
# ────────────────────────────────────────────────────────────────────


def _candidate_rows(conn, slug: str | None, force: bool = False) -> list[tuple[str, str]]:
    """Return ``[(slug, url), …]`` for tweet docs that look unenriched.

    Three heuristics catch the legacy formats:

      * ``summary LIKE 'RT @%%'``       — old truncated retweet, body
        missing entirely.
      * ``summary LIKE '%%↪ Quoting %%'`` — old quote-tweet one-liner;
        the quoted body / media never made it into the doc.
      * ``summary ~ '[📷🎬](\\s*$|\\s+\\|)'`` — bare media marker with
        no URL after it. The legacy `clean_summary` helper stripped
        every URL globally, including our intentional `📷 <url>`
        markers, so any photo/video the frontend card renderer
        would have shown is gone. Twikit can re-hydrate the source
        tweet and let us re-emit the marker with its URL.

    Already-enriched docs (``Retweet @user`` / ``Quoting @user`` on
    its own line, and tweets whose media markers carry a URL) are
    skipped — they've been through this codepath before.
    """
    # `--force` widens the filter to every twitter doc for the slug —
    # used after a parser change (e.g. deeper retweet-of-quote
    # recursion) that affects rows whose current shape would
    # otherwise satisfy the "already enriched" heuristic.
    if force:
        sql = r"""
            SELECT u.username, d.url
              FROM documents d
              JOIN users u ON u.id = d.user_id
             WHERE d.source = 'twitter'
               AND d.deleted = false
        """
    else:
        # `Retweet @…` covers the previous-generation backfill that
        # baked the attribution into the summary — those need
        # re-rendering so the prefix moves to `extra_tags`. The
        # other clauses keep the older heuristics (legacy `RT @`,
        # legacy `↪ Quoting`, bare media markers).
        sql = r"""
            SELECT u.username, d.url
              FROM documents d
              JOIN users u ON u.id = d.user_id
             WHERE d.source = 'twitter'
               AND d.deleted = false
               AND (
                     d.summary LIKE 'RT @%%'
                  OR d.summary LIKE 'Retweet @%%'
                  OR d.summary LIKE '%%↪ Quoting %%'
                  OR d.summary ~ '[📷🎬](\s*$|\s+\|)'
               )
        """
    params: tuple = ()
    if slug:
        sql += " AND u.username = %s"
        params = (slug,)
    sql += " ORDER BY u.username, d.date DESC NULLS LAST"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(r[0], r[1]) for r in cur.fetchall()]


def _extract_id(url: str) -> str | None:
    m = TWEET_ID_RE.search(url or "")
    return m.group(1) if m else None


# ────────────────────────────────────────────────────────────────────
# Writers
# ────────────────────────────────────────────────────────────────────


def _update_pg(
    conn,
    slug: str,
    url: str,
    summary: str,
    linked_urls: list[dict],
    link_hosts: list[str],
    title: str,
    date: str,
    extra_tags: list[str] | None = None,
) -> None:
    """Write the rebuilt fields for one doc. Match on (user_id, url).

    `extra_tags` (set to e.g. `['retweet @cshorten30']` for retweet
    docs) lands in `documents.extra_tags` so the frontend chip strip
    can surface the retweet attribution that used to live as a
    `Retweet @x\\n\\n` prefix on the summary.
    """
    sql = """
        UPDATE documents
           SET summary     = %s,
               title       = %s,
               linked_urls = %s::jsonb,
               link_hosts  = %s,
               extra_tags  = %s,
               date        = COALESCE(NULLIF(%s, '')::date, date),
               updated_at  = now()
          FROM users u
         WHERE u.username = %s
           AND documents.user_id = u.id
           AND documents.url = %s
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                summary,
                title,
                json.dumps(linked_urls),
                link_hosts,
                list(extra_tags or []),
                date,
                slug,
                url,
            ),
        )
        conn.commit()


# Cache of slugs whose `metadata.db` has been confirmed to carry the
# `linked_urls` / `link_hosts` columns. Older indexes were built
# before that schema was added; we ALTER them on first touch so each
# slug only pays the cost once per run.
_SCHEMA_ENSURED: set[str] = set()


def _ensure_sqlite_schema(slug: str) -> bool:
    """Add `linked_urls` / `link_hosts` / `extra_tags` columns to
    METADATA if missing.

    Returns True when the index exists on disk (columns now present),
    False when the slug has no index yet.
    """
    if slug in _SCHEMA_ENSURED:
        return True
    path = INDEXES_DIR / slug / "metadata.db"
    if not path.exists():
        return False
    with sqlite3.connect(path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(METADATA)").fetchall()}
        if "linked_urls" not in cols:
            conn.execute("ALTER TABLE METADATA ADD COLUMN linked_urls TEXT DEFAULT '[]'")
        if "link_hosts" not in cols:
            conn.execute("ALTER TABLE METADATA ADD COLUMN link_hosts TEXT DEFAULT ''")
        if "extra_tags" not in cols:
            conn.execute("ALTER TABLE METADATA ADD COLUMN extra_tags TEXT DEFAULT ''")
        conn.commit()
    _SCHEMA_ENSURED.add(slug)
    return True


def _update_sqlite(
    slug: str,
    url: str,
    summary: str,
    linked_urls: list[dict],
    link_hosts: list[str],
    title: str,
    date: str,
    extra_tags: list[str] | None = None,
) -> bool:
    """Mirror the rewrite into the slug's SQLite metadata. Returns True
    when METADATA had a row for this URL, False otherwise (the index
    may be older than the row — the next rebuild will sync it)."""
    if not _ensure_sqlite_schema(slug):
        return False
    path = INDEXES_DIR / slug / "metadata.db"
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE METADATA "
            "   SET summary = ?, "
            "       title = ?, "
            "       linked_urls = ?, "
            "       link_hosts = ?, "
            "       extra_tags = ?, "
            "       date = COALESCE(NULLIF(?, ''), date) "
            " WHERE url = ?",
            (
                summary,
                title,
                json.dumps(linked_urls),
                ",".join(link_hosts),
                ",".join(extra_tags or []),
                date,
                url,
            ),
        )
        touched = cur.rowcount
        conn.commit()
    return touched > 0


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--slug", help="Personality username to backfill.")
    grp.add_argument("--all", action="store_true", help="Backfill every personality's retweets.")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on docs to process this run (0 = all).",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print what would change; don't write to PG or SQLite.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Skip the 'already enriched' heuristic and re-render every "
        "twitter doc (most useful after a parser change — e.g. deeper "
        "retweet-of-quote recursion — where existing rows look fine "
        "by surface inspection but are missing inner content).",
    )
    args = p.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    auth_tok, ct0 = _load_cookies()

    with psycopg.connect(database_url) as conn:
        rows = _candidate_rows(
            conn,
            args.slug if not args.all else None,
            force=args.force,
        )
        if args.limit:
            rows = rows[: args.limit]
        print(f"backfill: {len(rows)} retweet doc(s) flagged as old-format")
        if not rows:
            return 0

        # Group ids by slug so we still keep the (slug, url) link.
        by_id: dict[str, tuple[str, str]] = {}
        for slug, url in rows:
            tid = _extract_id(url)
            if not tid:
                continue
            by_id.setdefault(tid, (slug, url))
        print(f"  {len(by_id)} extractable tweet id(s)")

        # Chunk the work so progress lands in PG + the log as we go.
        # `Bookmarks.lookup` itself sends 50-id batches to twikit;
        # processing 200 ids per chunk gives ~4 twikit calls per
        # write-flush, which is small enough to recover from a crash
        # without losing >1 minute of work and large enough to keep
        # the wall-clock overhead from psycopg setup negligible.
        CHUNK = 200
        ids = list(by_id.keys())
        b = Bookmarks(auth_token=auth_tok, ct0=ct0)

        pg_writes = 0
        sqlite_writes = 0
        skipped = 0
        for i in range(0, len(ids), CHUNK):
            chunk_ids = ids[i : i + CHUNK]
            chunk_idx = i // CHUNK + 1
            chunk_total = (len(ids) + CHUNK - 1) // CHUNK
            print(
                f"chunk {chunk_idx}/{chunk_total} "
                f"({i}-{i + len(chunk_ids)}/{len(ids)}): "
                f"twikit lookup of {len(chunk_ids)} id(s)…",
                flush=True,
            )
            results = b.lookup(chunk_ids)
            print(f"  hydrated {len(results)}/{len(chunk_ids)}", flush=True)

            chunk_pg = 0
            chunk_sql = 0
            chunk_skipped = 0
            for tid in chunk_ids:
                slug, url = by_id[tid]
                tw = results.get(tid)
                if not tw:
                    chunk_skipped += 1
                    continue

                # `_link_source_for` follows the same recursion as
                # the summary builder — retweets pull from the inner
                # tweet, quotes pull from both wrapper and quoted
                # side.
                linked_urls, link_hosts = _build_linked_urls(_link_source_for(tw))
                # Apply the same cleaning the regular pipeline runs
                # before write — strips bare t.co URLs from the body
                # while keeping `📷 <url>` / `🎬 <url>` media markers
                # intact (see `sources/utils/cleaning.py:clean_summary`).
                summary = clean_summary(_tweet_self_sufficient_summary(tw))
                title = clean_title(_tweet_display_title(tw, slug))
                date = _parse_date(tw)
                # Retweet attribution moved off the summary text and
                # into a clickable chip via `extra_tags`. Quote-only
                # tweets resolve to `[]`.
                extra_tags = _retweet_extra_tags(tw)

                if args.dry:
                    handles = link_hosts or ["(no links)"]
                    print(
                        f"    → @{slug} {url}  "
                        f"summary={summary[:60]!r}…  "
                        f"links={','.join(handles)}  "
                        f"extra={','.join(extra_tags) or '-'}",
                        flush=True,
                    )
                    continue

                _update_pg(
                    conn,
                    slug,
                    url,
                    summary,
                    linked_urls,
                    link_hosts,
                    title,
                    date,
                    extra_tags=extra_tags,
                )
                chunk_pg += 1
                if _update_sqlite(
                    slug,
                    url,
                    summary,
                    linked_urls,
                    link_hosts,
                    title,
                    date,
                    extra_tags=extra_tags,
                ):
                    chunk_sql += 1

            pg_writes += chunk_pg
            sqlite_writes += chunk_sql
            skipped += chunk_skipped
            print(
                f"  wrote pg={chunk_pg} sqlite={chunk_sql} "
                f"skipped={chunk_skipped}  "
                f"total pg={pg_writes} sqlite={sqlite_writes}",
                flush=True,
            )

        print(
            f"done: pg={pg_writes} sqlite={sqlite_writes} skipped={skipped}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
