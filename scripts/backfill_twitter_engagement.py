#!/usr/bin/env python3
"""Backfill twitter engagement metrics (likes / retweets / replies /
quotes / views / bookmarks) via twikit — no twitterapi.io credit.

What this does and why
----------------------
The behavioural columns added on 2026-05-22 (`twitter_likes`,
`twitter_retweets`, `twitter_replies`, `twitter_quotes`, `twitter_views`,
`twitter_bookmarks`, `engagement_updated_at`) only start filling as
`make twitter-feed` re-ingests each tweet. Existing rows stay NULL
until their next sync — which for old / inactive personalities may
never happen. This script forces the lookup: walks every
`source = 'twitter'` row whose `engagement_updated_at IS NULL`,
extracts the tweet id from the URL, batch-hydrates via
:meth:`twitter.Bookmarks.lookup` (cookie-authenticated x.com), and
UPDATEs the engagement columns in place.

For thread docs, the URL points at the *root* tweet — we use the
root's engagement only. That matches what a fresh ingestion of a
single-tweet doc records and avoids ballooning the script into a
conversation-walker. If you want thread-summed engagement, re-run
`make twitter-feed` for the affected slugs.

The script never POSTs through the ingest endpoint; it writes
directly to PG. Cookies: Safari extraction by default, or set
`TWITTER_AUTH_TOKEN` + `TWITTER_CT0` env vars.

Usage
-----
::

    DATABASE_URL=... uv run python scripts/backfill_twitter_engagement.py --slug tony-wu
    DATABASE_URL=... uv run python scripts/backfill_twitter_engagement.py --slug tony-wu --limit 200 --dry
    DATABASE_URL=... uv run python scripts/backfill_twitter_engagement.py --all
    DATABASE_URL=... uv run python scripts/backfill_twitter_engagement.py --all --refresh-older-than 14d
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timedelta, timezone

import psycopg

from sources.twitter.bookmarks import Bookmarks
from sources.twitter.tweets import _tweet_engagement

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
TWEET_ID_RE = re.compile(r"/status/(\d+)")

# Backfill batch size. `Bookmarks.lookup` itself splits into 50-id
# twikit calls, so 200 here means ~4 GraphQL calls per write-flush
# — small enough to recover from a crash without losing more than
# ~1 minute of work, large enough to amortise psycopg setup cost.
CHUNK = 200


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
# Duration parsing
# ────────────────────────────────────────────────────────────────────


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([dhm])\s*$", re.IGNORECASE)


def _parse_duration(s: str) -> timedelta:
    """Accept ``"14d"`` / ``"24h"`` / ``"90m"``; default unit = days."""
    m = _DURATION_RE.match(s)
    if not m:
        # Plain integer → days.
        try:
            return timedelta(days=int(s))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid duration {s!r}") from exc
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit == "d":
        return timedelta(days=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(minutes=n)


# ────────────────────────────────────────────────────────────────────
# Doc selection
# ────────────────────────────────────────────────────────────────────


def _candidate_rows(
    conn,
    slug: str | None,
    refresh_older_than: timedelta | None,
) -> list[tuple[str, str]]:
    """Return ``[(slug, url), …]`` for twitter docs that need a metric refresh.

    Selection:
      * Always: rows with `engagement_updated_at IS NULL`. Those have
        never been measured (legacy data, or ingested before the
        columns existed).
      * Optionally: rows with `engagement_updated_at < now() - X`. The
        feed-ranking signal goes stale as engagement keeps accumulating
        after publication; the same operator that ran the first
        backfill can re-run with `--refresh-older-than 14d` once a
        fortnight to keep popular tweets ranked correctly.

    Only `source = 'twitter'` rows with an `x.com/<user>/status/<id>`
    URL are candidates — resource docs surfaced through a tweet (e.g.
    an arxiv paper) live under their own source bucket and aren't
    eligible.
    """
    sql = """
        SELECT u.username, d.url
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE d.source = 'twitter'
           AND d.deleted = false
           AND d.url ~ '/status/[0-9]+'
           AND (
                 d.engagement_updated_at IS NULL
    """
    params: list = []
    if refresh_older_than is not None:
        sql += "             OR d.engagement_updated_at < %s\n"
        params.append(datetime.now(timezone.utc) - refresh_older_than)
    sql += "           )\n"
    if slug:
        sql += "           AND u.username = %s\n"
        params.append(slug)
    sql += " ORDER BY u.username, d.date DESC NULLS LAST"
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return [(r[0], r[1]) for r in cur.fetchall()]


def _extract_id(url: str) -> str | None:
    m = TWEET_ID_RE.search(url or "")
    return m.group(1) if m else None


# ────────────────────────────────────────────────────────────────────
# Writer
# ────────────────────────────────────────────────────────────────────


def _update_engagement(
    conn,
    slug: str,
    url: str,
    metrics: dict[str, int | None],
) -> bool:
    """UPDATE engagement columns for one doc. Match on (username, url).

    Skips the write entirely when every metric is ``None`` — that means
    twikit didn't return a usable payload (deleted / protected / 404)
    and we'd otherwise just stamp `engagement_updated_at` on a row we
    didn't actually measure, which would hide it from the "still
    NULL" candidate set on the next run.
    """
    if all(v is None for v in metrics.values()):
        return False
    sql = """
        UPDATE documents
           SET twitter_likes     = COALESCE(%s, twitter_likes),
               twitter_retweets  = COALESCE(%s, twitter_retweets),
               twitter_replies   = COALESCE(%s, twitter_replies),
               twitter_quotes    = COALESCE(%s, twitter_quotes),
               twitter_views     = COALESCE(%s, twitter_views),
               twitter_bookmarks = COALESCE(%s, twitter_bookmarks),
               engagement_updated_at = now(),
               updated_at = now()
          FROM users u
         WHERE u.username = %s
           AND documents.user_id = u.id
           AND documents.url = %s
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                metrics.get("twitter_likes"),
                metrics.get("twitter_retweets"),
                metrics.get("twitter_replies"),
                metrics.get("twitter_quotes"),
                metrics.get("twitter_views"),
                metrics.get("twitter_bookmarks"),
                slug,
                url,
            ),
        )
        return cur.rowcount > 0


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--slug", help="Personality username to backfill.")
    grp.add_argument("--all", action="store_true", help="Backfill every personality.")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on docs to process this run (0 = all).",
    )
    p.add_argument(
        "--refresh-older-than",
        type=_parse_duration,
        default=None,
        help=(
            "Also refresh docs whose engagement_updated_at is older than this "
            "duration (e.g. '14d', '24h'). Default: only NULL-engagement rows."
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
        rows = _candidate_rows(
            conn,
            args.slug if not args.all else None,
            refresh_older_than=args.refresh_older_than,
        )
        if args.limit:
            rows = rows[: args.limit]
        print(f"backfill: {len(rows)} twitter doc(s) need engagement metrics")
        if not rows:
            return 0

        # Map tweet_id → (slug, url). A tweet id is globally unique on
        # x.com, but the same URL can live under multiple users (a
        # personality A retweets a tweet from B that personality C also
        # bookmarked). Keep every (slug, url) pair so the UPDATE fans
        # out across all owners.
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
        skipped_no_metrics = 0
        skipped_no_hit = 0
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

            for tid in chunk_ids:
                tweet = results.get(tid)
                if tweet is None:
                    skipped_no_hit += 1
                    continue
                metrics = _tweet_engagement(tweet)
                if all(v is None for v in metrics.values()):
                    skipped_no_metrics += 1
                    continue
                for slug, url in by_id[tid]:
                    if args.dry:
                        print(f"    [dry] {slug} {url}: {metrics}")
                        updated += 1
                        continue
                    if _update_engagement(conn, slug, url, metrics):
                        updated += 1
            if not args.dry:
                conn.commit()
            print(
                f"  progress: updated={updated} miss(twikit)={skipped_no_hit} miss(metrics)={skipped_no_metrics}",
                flush=True,
            )

        print(
            f"done: updated {updated} doc(s); "
            f"{skipped_no_hit} not returned by twikit (deleted / protected / 404), "
            f"{skipped_no_metrics} returned but had no measurable engagement"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
