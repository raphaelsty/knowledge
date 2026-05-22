"""Hydrate a single Twitter thread that's already partially stored in
PG and re-emit it as one merged document.

Why this script exists
----------------------
Twitter's "Tweets" tab hides self-replies — which is how every
multi-part thread is technically posted — so the twikit fetcher only
captures the root. This backfiller takes a slug + thread root URL,
runs Twitter's ``conversation_id:`` search to enumerate every reply
in the conversation, filters to same-author parts, runs the merged
doc through ``compose_thread_doc`` (the shared formatter that both
pipeline paths already use), and UPSERTs the result.

Usage::

    uv run python scripts/backfill_twitter_thread.py \\
        --slug manuel-faysse \\
        --url https://x.com/ManuelFaysse/status/2055214689613664303

It's deliberately scoped to one thread per run so we can validate the
flow before turning it into a sweeping backfill across every user.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import psycopg

# Allow `from sources...` imports when invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.twitter.bookmarks import _rate_limit_aware, _twikit_to_dict  # noqa: E402
from sources.twitter.cookies import get_safari_cookies  # noqa: E402
from sources.twitter.tweets import compose_thread_doc  # noqa: E402

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

_URL_RE = re.compile(r"^https?://(?:x|twitter)\.com/([^/]+)/status/(\d+)")


def parse_url(url: str) -> tuple[str, str]:
    m = _URL_RE.match(url.strip())
    if not m:
        raise SystemExit(f"URL doesn't look like a tweet permalink: {url}")
    return m.group(1), m.group(2)


async def fetch_conversation(handle: str, tweet_id: str) -> list[dict]:
    """Return every part of the conversation rooted at tweet_id,
    written by ``handle``. Sorted oldest first."""
    from twikit import Client

    creds = get_safari_cookies()
    client = Client("en-US")
    client.set_cookies({"auth_token": creds["auth_token"], "ct0": creds["ct0"]})
    print(f"  searching: conversation_id:{tweet_id} …")
    try:
        res = await _rate_limit_aware(
            lambda: client.search_tweet(f"conversation_id:{tweet_id}", "Latest"),
            label="conversation search",
        )
    except Exception as e:
        # 404 here is often Twitter showing a degraded session view
        # after back-to-back queries from the same cookies. Surface
        # it so the operator can retry rather than silently empty.
        print(f"  ! search failed: {e}")
        return []
    raws = list(res) if res else []
    print(f"  got {len(raws)} tweets in conversation; filtering to @{handle}")
    parts: list[dict] = []
    seen_ids: set[str] = set()
    for tw in raws:
        screen = (getattr(getattr(tw, "user", None), "screen_name", "") or "").lower()
        if screen != handle.lower():
            continue
        d = _twikit_to_dict(tw)
        if not d or not d.get("id"):
            continue
        if d["id"] in seen_ids:
            continue
        seen_ids.add(d["id"])
        parts.append(d)
    # The search excludes the root in some Twitter responses; fetch it
    # explicitly so the merged doc has [1/N] at the top.
    if tweet_id not in seen_ids:
        try:
            root = await client.get_tweet_by_id(tweet_id)
            d = _twikit_to_dict(root)
            if d and d.get("id"):
                parts.append(d)
        except Exception as e:
            print(f"  ! could not fetch root tweet ({e}); using oldest part as anchor")
    parts.sort(key=lambda t: t.get("createdAt") or "")
    return parts


def upsert_merged_thread(
    database_url: str,
    user_slug: str,
    parts: list[dict],
    handle: str,
) -> None:
    """UPSERT the merged thread doc into documents, soft-deleting any
    pre-existing per-part rows that aren't the new anchor URL."""
    url, doc = compose_thread_doc(parts, username=handle)
    part_urls = {f"https://x.com/{handle}/status/{p['id']}" for p in parts}
    # We always tag bookmark/like/etc. independently; thread on its own
    # is what `compose_thread_doc` set.
    tags = doc["tags"]
    title = doc["title"]
    summary = doc["summary"]
    date = doc["date"] or None

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM users WHERE username = %s",
                (user_slug,),
            )
            row = cur.fetchone()
            if not row:
                raise SystemExit(f"no user with slug {user_slug!r}")
            user_id = row[0]

            # Soft-delete any existing per-part docs (other than what
            # will become the merged anchor) so the merged thread is
            # the only visible row.
            cur.execute(
                """
                UPDATE documents
                   SET deleted = true, updated_at = now()
                 WHERE user_id = %s
                   AND url = ANY(%s)
                   AND url <> %s
                """,
                (user_id, list(part_urls), url),
            )
            removed = cur.rowcount

            # UPSERT the merged doc. Reset `indexed=false` so the
            # next reindex pass pushes the merged version into the
            # ColBERT index — without that, the feed keeps serving
            # the stale per-part doc from the cached index.
            cur.execute(
                """
                INSERT INTO documents
                    (user_id, url, title, summary, date, tags, source, deleted, indexed)
                VALUES (%s, %s, %s, %s, %s, %s, 'twitter', false, false)
                ON CONFLICT (user_id, url) DO UPDATE
                   SET title      = EXCLUDED.title,
                       summary    = EXCLUDED.summary,
                       date       = COALESCE(EXCLUDED.date, documents.date),
                       tags       = EXCLUDED.tags,
                       deleted    = false,
                       indexed    = false,
                       updated_at = now()
                """,
                (user_id, url, title, summary, date, tags),
            )
            conn.commit()
    print(f"  ✓ upserted merged thread at {url} (soft-deleted {removed} per-part docs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True, help="Knowledge user slug (e.g. manuel-faysse)")
    ap.add_argument("--url", required=True, help="Thread-root tweet URL (https://x.com/<handle>/status/<id>)")
    args = ap.parse_args()

    handle, tweet_id = parse_url(args.url)
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    parts = asyncio.run(fetch_conversation(handle, tweet_id))
    if not parts:
        print("No parts returned by Twitter — bailing.")
        return 1
    print(f"  collected {len(parts)} parts for @{handle}:")
    for p in parts:
        print(f"    id={p['id']}  text: {(p.get('text') or '')[:80]}")

    upsert_merged_thread(database_url, args.slug, parts, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
