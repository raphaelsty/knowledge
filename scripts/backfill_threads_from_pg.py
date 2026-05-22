"""Merge same-author Twitter thread parts in PG into single thread docs,
using only existing data — no Twitter API calls.

Why
---
The pipeline now produces merged thread documents at fetch time
(`Bookmarks._merge_threads`, `Tweets._collect_own_tweets`), but every
historical run is full of per-part rows that never got merged. We
recover them here by clustering on tweet-id proximity.

Heuristic
---------
Tweet ids are snowflake-based: numerically close ids mean tweets
posted within a few seconds of each other by the same account. A
thread is exactly that: a burst of self-replies by one author. So we:

  1. For every user, walk `documents` rows where source = 'twitter'.
  2. Parse `(author_handle, tweet_id)` out of the
     ``https://x.com/<handle>/status/<id>`` URL.
  3. Bucket by ``(user_id, author_handle)``, sort by ``tweet_id``.
  4. Cluster consecutive ids where the gap < ``--id-gap`` (default
     100 billion ≈ Twitter's "posted within seconds" window).
  5. Merge clusters of size ≥ 2 into one doc anchored at the oldest
     part. Soft-delete the rest. Set ``indexed=false`` so the next
     reindex pushes the merged version into the ColBERT index.

The same-author + tight-id-window rule is conservative — it merges
the central core of any thread (the "1/N" burst posted in one sit-down)
without dragging in later "thanks" replies posted minutes/hours later.

Idempotent: clusters where the anchor already carries
``twitter-thread`` AND has the box-drawing separator in the summary
are skipped.

Usage::

    uv run python scripts/backfill_threads_from_pg.py --dry
    uv run python scripts/backfill_threads_from_pg.py --slug manuel-faysse
    uv run python scripts/backfill_threads_from_pg.py            # everyone

"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable

import psycopg
from psycopg.types.json import Jsonb

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

# Same separator the frontend `_TWEET_SEPARATOR` splits on. Both
# pipeline paths emit this exact string between parts.
_THREAD_SEPARATOR = "\n\n────────\n\n"

# URL → (handle, tweet_id). Accept both x.com and twitter.com hosts.
_TWEET_URL_RE = re.compile(r"^https?://(?:www\.)?(?:x|twitter)\.com/([^/]+)/status/(\d+)")


def parse_tweet_url(url: str) -> tuple[str, int] | None:
    m = _TWEET_URL_RE.match(url.strip())
    if not m:
        return None
    try:
        return m.group(1).lower(), int(m.group(2))
    except (ValueError, TypeError):
        return None


def fetch_user_ids(conn, slug: str | None) -> list[tuple[int, str]]:
    """Return (id, username) for the slug, or every user when slug is None."""
    sql = "SELECT id, username FROM users"
    params: tuple = ()
    if slug:
        sql += " WHERE username = %s"
        params = (slug,)
    sql += " ORDER BY id"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def fetch_twitter_docs(conn, user_id: int) -> list[dict]:
    """Pull every non-deleted twitter doc for one user."""
    sql = (
        "SELECT url, title, summary, date, tags, linked_urls, link_hosts "
        "  FROM documents "
        " WHERE user_id = %s AND source = 'twitter' AND deleted = false"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()
    out: list[dict] = []
    for url, title, summary, date, tags, linked_urls, link_hosts in rows:
        parsed = parse_tweet_url(url)
        if not parsed:
            continue
        handle, tweet_id = parsed
        out.append(
            {
                "url": url,
                "title": title or "",
                "summary": summary or "",
                "date": date,
                "tags": list(tags or []),
                "linked_urls": linked_urls or [],
                "link_hosts": list(link_hosts or []),
                "handle": handle,
                "tweet_id": tweet_id,
            }
        )
    return out


def cluster(docs: list[dict], id_gap: int) -> Iterable[list[dict]]:
    """Yield clusters of ≥ 2 docs from the same author whose tweet
    ids are within ``id_gap`` of the previous one in sorted order."""
    by_handle: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_handle[d["handle"]].append(d)
    for _handle, group in by_handle.items():
        group.sort(key=lambda d: d["tweet_id"])
        bucket: list[dict] = []
        for d in group:
            if not bucket:
                bucket.append(d)
                continue
            if d["tweet_id"] - bucket[-1]["tweet_id"] <= id_gap:
                bucket.append(d)
            else:
                if len(bucket) >= 2:
                    yield bucket
                bucket = [d]
        if len(bucket) >= 2:
            yield bucket


def already_merged(anchor: dict) -> bool:
    """A previously-merged anchor has both the twitter-thread tag and
    the box-drawing separator in its summary."""
    return "twitter-thread" in (anchor.get("tags") or []) and _THREAD_SEPARATOR in (anchor.get("summary") or "")


def compose_merged(parts: list[dict]) -> dict:
    """Produce the merged-doc payload for one cluster.

    Anchor = oldest part. Title pattern matches the pipeline-side
    `compose_thread_doc` so the on-disk shape is identical.
    """
    n = len(parts)
    body = [f"[{i + 1}/{n}] {p['summary'].strip()}" for i, p in enumerate(parts)]
    summary = _THREAD_SEPARATOR.join(s for s in body if s)

    # Title — strip any pre-existing "— thread (...)" suffix on the
    # anchor's title before re-applying, so reruns don't compound.
    anchor = parts[0]
    base_title = re.sub(r"\s*—\s*thread\s*\(.*?\)\s*$", "", anchor["title"]).strip()
    title = f"{base_title} — thread ({n} tweets)" if base_title else f"@{anchor['handle']} — thread ({n} tweets)"

    # Tag union — preserve per-part kind tags (twitter-like / -tweet /
    # -retweet / -bookmark) plus the canonical twitter / twitter-thread.
    tag_set: set[str] = set()
    for p in parts:
        for t in p["tags"]:
            tag_set.add(t)
    tag_set.add("twitter")
    tag_set.add("twitter-thread")
    # Drop anti-tags that no longer apply: a merged thread isn't
    # "just" a tweet anymore. Keep granular kind tags so source
    # filters still work.
    tag_set.discard("twitter-tweet")
    tags = sorted(tag_set)

    # Linked URLs / hosts — union with dedupe by url.
    seen_link_urls: set[str] = set()
    linked_urls: list = []
    for p in parts:
        for entry in p["linked_urls"]:
            u = (entry.get("url") if isinstance(entry, dict) else None) or ""
            if not u or u in seen_link_urls:
                continue
            seen_link_urls.add(u)
            linked_urls.append(entry)
    host_set: set[str] = set()
    for p in parts:
        for h in p["link_hosts"]:
            if h:
                host_set.add(h)
    link_hosts = sorted(host_set)

    return {
        "anchor_url": anchor["url"],
        "title": title,
        "summary": summary,
        "date": anchor["date"],
        "tags": tags,
        "linked_urls": linked_urls,
        "link_hosts": link_hosts,
    }


def apply_merge(
    conn,
    user_id: int,
    cluster_docs: list[dict],
    merged: dict,
    dry: bool,
) -> None:
    """UPSERT the anchor doc with merged fields and soft-delete the
    rest of the cluster. Marks ``indexed=false`` so the next reindex
    pass rewrites the embedding."""
    if dry:
        return
    part_urls = [p["url"] for p in cluster_docs]
    with conn.cursor() as cur:
        # Update the anchor in place.
        cur.execute(
            """
            UPDATE documents
               SET title       = %s,
                   summary     = %s,
                   date        = %s,
                   tags        = %s,
                   linked_urls = %s::jsonb,
                   link_hosts  = %s,
                   deleted     = false,
                   indexed     = false,
                   updated_at  = now()
             WHERE user_id = %s AND url = %s
            """,
            (
                merged["title"],
                merged["summary"],
                merged["date"],
                merged["tags"],
                Jsonb(merged["linked_urls"]),
                merged["link_hosts"],
                user_id,
                merged["anchor_url"],
            ),
        )
        # Soft-delete the non-anchor parts.
        cur.execute(
            """
            UPDATE documents
               SET deleted    = true,
                   indexed    = false,
                   updated_at = now()
             WHERE user_id = %s
               AND url = ANY(%s)
               AND url <> %s
            """,
            (user_id, part_urls, merged["anchor_url"]),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default=None, help="restrict to one user (slug)")
    ap.add_argument(
        "--id-gap",
        type=int,
        default=100_000_000_000,
        help="Maximum tweet-id gap between consecutive parts of the "
        "same thread (default 100B). Tighter = fewer false-positive "
        "merges, smaller cluster sizes.",
    )
    ap.add_argument("--dry", action="store_true", help="report what would change, don't write")
    args = ap.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(database_url) as conn:
        users = fetch_user_ids(conn, args.slug)
        if not users:
            print("No users matched.")
            return 1

        total_clusters = 0
        total_merged_parts = 0
        total_users_touched = 0
        for uid, slug in users:
            docs = fetch_twitter_docs(conn, uid)
            if not docs:
                continue
            user_clusters = 0
            user_merged_parts = 0
            for cl in cluster(docs, args.id_gap):
                anchor = cl[0]
                if already_merged(anchor):
                    continue
                merged = compose_merged(cl)
                user_clusters += 1
                user_merged_parts += len(cl)
                apply_merge(conn, uid, cl, merged, args.dry)
                if args.dry:
                    print(f"  [{slug}] cluster of {len(cl)} @{anchor['handle']} → {merged['anchor_url']}")
            if user_clusters:
                print(f"{slug:<28} {user_clusters:>3} cluster(s), {user_merged_parts:>4} part(s) merged")
                total_clusters += user_clusters
                total_merged_parts += user_merged_parts
                total_users_touched += 1
        if not args.dry:
            conn.commit()

    verb = "Would merge" if args.dry else "Merged"
    print(
        f"\n{verb} {total_clusters} cluster(s) across {total_users_touched} user(s) — {total_merged_parts} per-part docs collapsed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
