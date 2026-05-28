#!/usr/bin/env python3
"""Backfill `documents.referenced_author` for every twitter doc.

What this does and why
----------------------
The `referenced_author` column (added 2026-05-28) carries the @handle
of the person a tweet refers to — a retweet target, a quoted tweet's
author, or the user being replied to. It's the raw signal we'll use
to find candidate accounts worth adding to the personality roster
("user X consistently retweets Y; who is Y?").

This script walks every `source = 'twitter'` row whose
`referenced_author IS NULL`, newest-first, batch-hydrates via
cookie-authenticated twikit, picks the @handle from the tweet payload
with precedence ``retweet > quote > reply``, and UPDATEs the column
in place. Tweets that turn out to be plain originals (no retweet,
quote, or reply) get the empty-string sentinel so a re-run doesn't
re-hit twikit for them.

Engagement (twitter_likes / twitter_retweets / …) is refreshed at
the same time — we already paid for the twikit lookup, so it's free
to also stamp the engagement columns. `backfill_twitter_engagement.py`
remains the canonical script for engagement-only sweeps.

Be gentle: chunks are smaller than the engagement script's (100 vs
200) and the script sleeps between chunks so we don't burst twikit's
GraphQL rate limit. Cookies: Safari extraction by default, or set
`TWITTER_AUTH_TOKEN` + `TWITTER_CT0` env vars.

Usage
-----
::

    DATABASE_URL=... uv run python scripts/backfill_twitter_referenced_author.py
    DATABASE_URL=... uv run python scripts/backfill_twitter_referenced_author.py --slug tony-wu
    DATABASE_URL=... uv run python scripts/backfill_twitter_referenced_author.py --limit 500 --dry
"""

from __future__ import annotations

import argparse
import os
import re
import time

import psycopg

from sources.twitter.bookmarks import Bookmarks
from sources.twitter.tweets import _referenced_author, _tweet_engagement

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
TWEET_ID_RE = re.compile(r"/status/(\d+)")

# Between the engagement script (which uses 200) and "deeply
# cautious": `Bookmarks.lookup` already chunks into 50-id GraphQL
# calls with its own polite delay, so 150 here is roughly three
# API calls per write-flush — enough throughput for a real
# backfill, still under the engagement script's chunk size.
CHUNK = 150

# Extra sleep between chunks on top of `Bookmarks.lookup`'s internal
# pacing. The internal `_POLITE_DELAY` covers per-batch spacing;
# this is the headroom we leave on top of it. 1.5 s × ~thousand
# chunks adds ~25 minutes to a full sweep — visible but tolerable,
# and well under the GraphQL rate-limit reset window.
INTER_CHUNK_SLEEP_SEC = 1.5


# ────────────────────────────────────────────────────────────────────
# Cookie loading
# ────────────────────────────────────────────────────────────────────


def _load_cookies() -> tuple[str, str]:
    """Return ``(auth_token, ct0)``. Env beats Safari extraction."""
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


def _candidate_rows(conn, slug: str | None) -> list[tuple[str, str]]:
    """Return ``[(slug, url), …]`` for twitter docs with NULL referenced_author.

    Ordered newest-first across the whole table (not grouped by user)
    so a stop / restart picks up roughly where the previous run left
    off — the recent stuff stays freshest regardless of slug. Only
    rows with a parseable `/status/<id>` URL are eligible; resource
    docs surfaced through a tweet (e.g. an arxiv paper) live under
    their own source and aren't tweets to look up.
    """
    sql = """
        SELECT u.username, d.url
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE d.source = 'twitter'
           AND d.deleted = false
           AND d.url ~ '/status/[0-9]+'
           AND d.referenced_author IS NULL
    """
    params: list = []
    if slug:
        sql += "           AND u.username = %s\n"
        params.append(slug)
    sql += " ORDER BY d.date DESC NULLS LAST, d.url"
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return [(r[0], r[1]) for r in cur.fetchall()]


def _extract_id(url: str) -> str | None:
    m = TWEET_ID_RE.search(url or "")
    return m.group(1) if m else None


# ────────────────────────────────────────────────────────────────────
# Writers
# ────────────────────────────────────────────────────────────────────


def _update_doc(
    conn,
    slug: str,
    url: str,
    referenced_author: str,
    metrics: dict[str, int | None],
) -> bool:
    """UPDATE referenced_author + engagement columns for one doc.

    Always stamps `referenced_author` (even when it's the empty
    sentinel) so the row drops out of the candidate set on the next
    run. Engagement columns are written with COALESCE so a payload
    that didn't include, say, `viewCount` doesn't overwrite a real
    prior value with NULL.

    Deadlocks: prod daemons (indexer, categorize, clean) also write
    to `documents`. If they touch the same row in opposite order to
    us, Postgres aborts the loser — we retry a couple of times with
    a short sleep so the transient race doesn't tank the whole
    backfill. After 3 attempts we give up on the row (it stays
    NULL → picked up on the next sweep).
    """
    sql = """
        UPDATE documents
           SET referenced_author = %s,
               twitter_likes     = COALESCE(%s, twitter_likes),
               twitter_retweets  = COALESCE(%s, twitter_retweets),
               twitter_replies   = COALESCE(%s, twitter_replies),
               twitter_quotes    = COALESCE(%s, twitter_quotes),
               twitter_views     = COALESCE(%s, twitter_views),
               twitter_bookmarks = COALESCE(%s, twitter_bookmarks),
               engagement_updated_at = CASE
                   WHEN %s OR %s OR %s OR %s OR %s OR %s THEN now()
                   ELSE engagement_updated_at
               END,
               updated_at = now()
          FROM users u
         WHERE u.username = %s
           AND documents.user_id = u.id
           AND documents.url = %s
    """
    likes = metrics.get("twitter_likes")
    retweets = metrics.get("twitter_retweets")
    replies = metrics.get("twitter_replies")
    quotes = metrics.get("twitter_quotes")
    views = metrics.get("twitter_views")
    bookmarks = metrics.get("twitter_bookmarks")
    params = (
        referenced_author,
        likes,
        retweets,
        replies,
        quotes,
        views,
        bookmarks,
        likes is not None,
        retweets is not None,
        replies is not None,
        quotes is not None,
        views is not None,
        bookmarks is not None,
        slug,
        url,
    )
    import psycopg.errors

    for attempt in range(3):
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                conn.commit()
                return cur.rowcount > 0
        except psycopg.errors.DeadlockDetected:
            conn.rollback()
            if attempt == 2:
                print(f"    deadlock x3 on {slug} {url} — skipping")
                return False
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            conn.rollback()
            raise
    return False


def _mark_unhit(conn, slug: str, url: str) -> bool:
    """Stamp the empty sentinel on a row whose tweet twikit couldn't fetch.

    Deleted / protected / 404 tweets will never come back, so writing
    ``''`` lets them drop out of the candidate set instead of
    re-burning a slot on every run. Engagement columns are left
    untouched.
    """
    import psycopg.errors

    sql = """
        UPDATE documents
           SET referenced_author = '',
               updated_at = now()
          FROM users u
         WHERE u.username = %s
           AND documents.user_id = u.id
           AND documents.url = %s
           AND documents.referenced_author IS NULL
    """
    for attempt in range(3):
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (slug, url))
                conn.commit()
                return cur.rowcount > 0
        except psycopg.errors.DeadlockDetected:
            conn.rollback()
            if attempt == 2:
                return False
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            conn.rollback()
            raise
    return False


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slug",
        default=None,
        help="Optional personality username; default = every personality.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on docs to process this run (0 = all).",
    )
    p.add_argument(
        "--mark-missing",
        action="store_true",
        help=(
            "Stamp the empty sentinel on tweets twikit can't return "
            "(deleted / protected / 404) so they drop out of future runs."
        ),
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print what would change; don't write to PG.",
    )
    args = p.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    auth_tok, ct0 = _load_cookies()

    with psycopg.connect(database_url) as conn:
        rows = _candidate_rows(conn, args.slug)
        if args.limit:
            rows = rows[: args.limit]
        print(f"backfill: {len(rows)} twitter doc(s) need referenced_author")
        if not rows:
            return 0

        # Same fan-out trick the engagement script uses: a single
        # tweet id can live under several personalities (A retweeted
        # tweet X that C also bookmarked). Hydrate once, update each
        # (slug, url) it appears under.
        by_id: dict[str, list[tuple[str, str]]] = {}
        for slug, url in rows:
            tid = _extract_id(url)
            if not tid:
                continue
            by_id.setdefault(tid, []).append((slug, url))
        print(f"  {len(by_id)} unique tweet id(s) to hydrate")

        ids = list(by_id.keys())
        b = Bookmarks(auth_token=auth_tok, ct0=ct0)

        updated = 0
        marked_missing = 0
        skipped_no_hit = 0
        total_chunks = (len(ids) + CHUNK - 1) // CHUNK
        for i in range(0, len(ids), CHUNK):
            chunk_ids = ids[i : i + CHUNK]
            chunk_idx = i // CHUNK + 1
            print(
                f"chunk {chunk_idx}/{total_chunks} "
                f"({i}-{i + len(chunk_ids)}/{len(ids)}): "
                f"twikit lookup of {len(chunk_ids)} id(s)…",
                flush=True,
            )
            results = b.lookup(chunk_ids)
            print(f"  hydrated {len(results)}/{len(chunk_ids)}", flush=True)

            for tid in chunk_ids:
                tweet = results.get(tid)
                if tweet is None:
                    skipped_no_hit += 1
                    if args.mark_missing and not args.dry:
                        for slug, url in by_id[tid]:
                            if _mark_unhit(conn, slug, url):
                                marked_missing += 1
                    continue
                ref = _referenced_author(tweet)
                metrics = _tweet_engagement(tweet)
                for slug, url in by_id[tid]:
                    # Drop the owner's own handle so a self-reply
                    # (a thread continuation) doesn't show up as
                    # "user X references user X". Mirrors the same
                    # guard in `compose_thread_doc`.
                    owner_ref = "" if ref and ref == slug.lower() else ref
                    if args.dry:
                        print(f"    [dry] {slug} {url}: ref={owner_ref!r} metrics={metrics}")
                        updated += 1
                        continue
                    if _update_doc(conn, slug, url, owner_ref, metrics):
                        updated += 1
            # Per-row commits inside _update_doc kept lock-holding
            # time short to avoid deadlocks with prod daemons; no
            # chunk-level commit needed.
            print(
                f"  progress: updated={updated} marked-missing={marked_missing} miss(twikit)={skipped_no_hit}",
                flush=True,
            )
            # Be gentle on twikit between chunks. Last chunk has
            # nothing to wait for.
            if i + CHUNK < len(ids):
                time.sleep(INTER_CHUNK_SLEEP_SEC)

        print(
            f"done: updated {updated} doc(s); "
            f"{skipped_no_hit} not returned by twikit "
            f"({'marked ' + str(marked_missing) + ' as missing' if args.mark_missing else 'left NULL — pass --mark-missing to short-circuit'})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
