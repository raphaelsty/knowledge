#!/usr/bin/env python3
"""Backfill `linked_urls` / `link_hosts` for tweet documents.

When `sources/twitter/tweets.py` learnt to put external URLs onto
the parent tweet doc (instead of minting a separate companion doc
per linked page), every tweet ingested *before* the change kept its
old shape: empty `linked_urls`, empty `link_hosts`, plus a
free-floating sibling doc whose `source_url` points back at the
tweet. The frontend's link-preview cluster reads only the new
columns, and the source filter does `source IN (...) OR
link_hosts && ARRAY[...]`, so without this backfill old tweets are
silent.

What this script does for one slug per run:
  1. Find every twitter-sourced doc whose `link_hosts` is empty.
  2. Pair URL → tweet_id (`/status/<id>` segment).
  3. Batch-hydrate the payloads from twitterapi.io (free for
     cached ids; the regular pipeline budget guard doesn't apply
     here since we're not minting new rows).
  4. Re-render the doc with `_tweet_self_sufficient_summary` and
     `_build_linked_urls` (which fetches OG title/summary/image
     for each link — `MAX_LINKED_URLS_PER_DOC` capped).
  5. Write summary, `linked_urls`, `link_hosts` to Postgres.
  6. Mirror the same fields into the slug's SQLite metadata via
     the API's `/indices/{slug}/metadata/update` endpoint so
     search results reflect the new payload without a full
     reindex.
  7. Flag any companion docs (rows whose `source_url` points at one
     of the tweets we just enriched AND whose `source != 'twitter'`)
     with `to_delete = TRUE` so the pipeline's purge worker drops
     them from PG + the index on its next run.

Required env:
   TWITTERAPIIO_API_KEY  — same key the regular pipeline uses.
   ADMIN_API_KEY         — only if the running API sets it; the
                           metadata-update endpoint sits behind
                           that header.
   DATABASE_URL          — PG connection string.
   API_URL               — local API base (default http://localhost:8080).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from collections.abc import Iterator

import psycopg
import requests

from sources.twitter.tweets import (
    _build_linked_urls,
    _link_source_for,
    _tweet_self_sufficient_summary,
)

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
TWITTERAPIIO_BASE = "https://api.twitterapi.io"
TWEET_ID_RE = re.compile(r"/status/(\d+)")


def list_tweet_docs(database_url: str, slug: str) -> list[str]:
    """Return tweet doc URLs (newest first) whose `link_hosts` is empty.

    The empty-hosts check is what makes the backfill idempotent —
    re-running it is free for rows we've already touched.
    """
    sql = """
        SELECT d.url
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE u.username = %s
           AND d.source = 'twitter'
           AND d.deleted = false
           AND cardinality(d.link_hosts) = 0
         ORDER BY d.date DESC NULLS LAST
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (slug,))
            return [r[0] for r in cur.fetchall()]


def extract_tweet_id(url: str) -> str | None:
    m = TWEET_ID_RE.search(url or "")
    return m.group(1) if m else None


def chunks(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def fetch_tweets(tweet_ids: list[str], api_key: str) -> dict[str, dict]:
    """Batch GET /twitter/tweets. Returns `{id: tweet_payload}`."""
    out: dict[str, dict] = {}
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    for batch in chunks(tweet_ids, 100):
        params = {"tweet_ids": ",".join(batch)}
        url = f"{TWITTERAPIIO_BASE}/twitter/tweets?" + urllib.parse.urlencode(params)
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"  [warn] twitterapi.io {r.status_code} on chunk ({len(batch)} ids): {r.text[:200]}")
            continue
        data = r.json()
        items = []
        if isinstance(data, dict):
            for k in ("tweets", "data", "items"):
                v = data.get(k)
                if isinstance(v, list):
                    items = v
                    break
        elif isinstance(data, list):
            items = data
        for tw in items:
            tid = str(tw.get("id") or tw.get("id_str") or "")
            if tid:
                out[tid] = tw
        # Be polite — same min interval the pipeline uses.
        time.sleep(0.25)
    return out


def update_pg(
    database_url: str,
    user_id: int,
    url: str,
    summary: str,
    linked_urls: list[dict],
    link_hosts: list[str],
) -> None:
    """Write the new summary + linked_urls + link_hosts to PG."""
    sql = """
        UPDATE documents
           SET summary    = %s,
               linked_urls = %s::jsonb,
               link_hosts = %s,
               updated_at = now()
         WHERE user_id = %s AND url = %s
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    summary,
                    json.dumps(linked_urls),
                    link_hosts,
                    user_id,
                    url,
                ),
            )
            conn.commit()


def update_sqlite_metadata(
    api_url: str,
    api_key: str | None,
    slug: str,
    url: str,
    summary: str,
    linked_urls: list[dict],
    link_hosts: list[str],
) -> bool:
    """Push the same fields into the slug's SQLite metadata so the
    next search result reflects the change without a full reindex.

    `link_hosts` is comma-encoded on the index side (same convention
    as `tags`); `linked_urls` is JSON-stringified.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {
        "condition": "url = ?",
        "parameters": [url],
        "updates": {
            "summary": summary,
            "linked_urls": json.dumps(linked_urls),
            "link_hosts": ",".join(link_hosts),
        },
    }
    r = requests.post(
        f"{api_url}/indices/{slug}/metadata/update",
        json=payload,
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        print(f"  [warn] metadata.update {r.status_code} for {url}: {r.text[:200]}")
        return False
    return True


def mark_companions_to_delete(database_url: str, user_id: int, tweet_urls: list[str]) -> int:
    """Flag companion docs surfaced via these tweet URLs for purge.

    A companion is any document whose `source_url` points at a tweet
    we just enriched AND whose `source != 'twitter'`. We never
    physically delete here — the pipeline's purge step handles that
    so the index drop happens in lockstep with the PG drop.
    """
    if not tweet_urls:
        return 0
    sql = """
        UPDATE documents
           SET to_delete = TRUE,
               updated_at = now()
         WHERE user_id = %s
           AND source <> 'twitter'
           AND deleted = false
           AND to_delete = false
           AND source_url = ANY(%s)
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, tweet_urls))
            n = cur.rowcount
            conn.commit()
            return n


def user_id_for_slug(database_url: str, slug: str) -> int | None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (slug,))
            row = cur.fetchone()
            return int(row[0]) if row else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slug",
        required=True,
        help="Personality username to backfill (e.g. jobergum).",
    )
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
        "--skip-companion-cleanup",
        action="store_true",
        help="Don't flag companion (non-twitter) docs for purge.",
    )
    args = p.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_url = os.environ.get("API_URL", "http://localhost:8080")
    api_key = os.environ.get("ADMIN_API_KEY") or None
    twitterapi_key = os.environ.get("TWITTERAPIIO_API_KEY") or ""
    if not twitterapi_key:
        print("error: TWITTERAPIIO_API_KEY is required.", file=sys.stderr)
        return 2

    uid = user_id_for_slug(database_url, args.slug)
    if uid is None:
        print(f"error: no user with username {args.slug!r}", file=sys.stderr)
        return 2

    urls = list_tweet_docs(database_url, args.slug)
    if args.limit:
        urls = urls[: args.limit]
    print(f"@{args.slug}: {len(urls)} tweet doc(s) need backfill")
    if not urls:
        return 0

    by_id: dict[str, str] = {}
    for url in urls:
        tid = extract_tweet_id(url)
        if tid:
            by_id[tid] = url
    print(f"  {len(by_id)} extractable tweet id(s)")

    payloads = fetch_tweets(list(by_id.keys()), twitterapi_key)
    print(f"  hydrated {len(payloads)} payload(s) from twitterapi.io")

    enriched_urls: list[str] = []
    pg_writes = 0
    sqlite_writes = 0
    no_links = 0
    for tid, url in by_id.items():
        tweet = payloads.get(tid)
        if not tweet:
            continue
        # Mirror the pipeline's choice of link source via the shared
        # `_link_source_for` helper: retweets pull from the inner
        # tweet, quote tweets from both wrapper + quoted side, plain
        # tweets from themselves.
        linked_urls, link_hosts = _build_linked_urls(_link_source_for(tweet))
        summary = _tweet_self_sufficient_summary(tweet)
        if not linked_urls:
            # Text-only tweet — nothing to enrich. Skip the writes
            # to keep updated_at clean.
            no_links += 1
            continue
        print(f"  → {url}  ({len(linked_urls)} link(s): {', '.join(link_hosts)})")
        if args.dry:
            for link in linked_urls:
                title_preview = (link.get("title") or "")[:80]
                print(f"      • {link.get('host')}  {title_preview}")
            continue
        update_pg(database_url, uid, url, summary, linked_urls, link_hosts)
        pg_writes += 1
        if update_sqlite_metadata(
            api_url,
            api_key,
            args.slug,
            url,
            summary,
            linked_urls,
            link_hosts,
        ):
            sqlite_writes += 1
        enriched_urls.append(url)

    flagged = 0
    if not args.dry and not args.skip_companion_cleanup and enriched_urls:
        flagged = mark_companions_to_delete(database_url, uid, enriched_urls)

    print(f"done: pg={pg_writes} sqlite={sqlite_writes} no-links={no_links} companions_flagged={flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
