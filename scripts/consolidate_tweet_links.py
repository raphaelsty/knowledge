#!/usr/bin/env python3
"""Consolidate existing companion docs onto their parent tweet rows.

The pipeline used to mint one "companion" document per URL embedded
in a tweet (one row at `source_url = tweet.url`, `source = host`).
The new shape stores those previews inline on the tweet doc itself:
`linked_urls JSONB` (list of `{url, host, title, summary, image}`)
plus `link_hosts TEXT[]` (GIN-indexed). The companion rows then
become redundant and want to be tombstoned.

This script does the consolidation **without re-fetching anything
from twitterapi.io**. Every field we need is already on the
companion rows in Postgres:

   url        ← companion.url
   host       ← companion.source          (the pipeline already
                                            normalised it to a host
                                            via `_source_tag`)
   title      ← companion.title
   summary    ← companion.summary
   image      ← `""`   (companions never stored an OG image)

Image stays empty by default — the frontend card falls back to the
destination's favicon. Pass `--fetch-images` to enable a slow second
pass that pulls `og:image` directly from each destination URL via
plain HTTP (no twitterapi.io quota involved). With ~23k distinct
destinations that pass takes hours, hence opt-in.

What runs per personality:

  1. ALTER `indexes/{slug}/metadata.db` to add the two columns if
     they're missing (the SQLite METADATA schema is derived at
     index-build time, so pre-existing indexes need a one-off
     ALTER each).
  2. Group companions by tweet URL in a single SQL pass.
  3. UPDATE the parent tweet rows with `linked_urls` +
     `link_hosts` (only if the parent tweet doc exists and isn't
     already enriched).
  4. UPDATE the companion rows to `to_delete = TRUE` so the
     pipeline's purge step drops them on its next run.
  5. Mirror `linked_urls` + `link_hosts` into the slug's SQLite
     METADATA via one UPDATE per tweet (direct, no API hop —
     much faster than `/indices/{slug}/metadata/update` for 20k+
     rows).

Idempotent: re-running skips tweets whose `link_hosts` is already
populated. Adding new companions later won't re-trigger because the
join only fires for tweets with `link_hosts = '{}'`.

Required env:
   DATABASE_URL          PG connection string.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
INDEXES_DIR = Path("indexes")
TWEET_ID_RE = re.compile(r"/status/\d+")
# Polite OG fetch budget — only consulted when --fetch-images is set.
OG_TIMEOUT_S = 6.0
OG_MAX_BYTES = 65_536
_OG_IMAGE_RE = re.compile(
    rb"""<meta[^>]*(?:property|name)\s*=\s*["']?(?:og:image|twitter:image(?::src)?)["']?[^>]*\bcontent\s*=\s*["']([^"']+)""",
    re.IGNORECASE,
)


def list_personalities_with_backlog(database_url: str) -> list[tuple[int, str]]:
    """Return `(user_id, username)` for every personality with at
    least one tweet doc still missing `link_hosts`."""
    sql = """
        SELECT DISTINCT u.id, u.username
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE d.source = 'twitter'
           AND d.deleted = false
           AND cardinality(d.link_hosts) = 0
        ORDER BY u.username
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def list_personalities_with_missing_images(
    database_url: str,
) -> list[tuple[int, str]]:
    """Return `(user_id, username)` for every personality that has at
    least one `linked_urls` entry with an empty `image` field. The
    image-only pass walks this set and parallel-fetches og:image
    for each unique destination URL without touching the
    consolidation join."""
    sql = """
        SELECT DISTINCT u.id, u.username
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE jsonb_array_length(d.linked_urls) > 0
           AND d.deleted = false
           AND EXISTS (
               SELECT 1
                 FROM jsonb_array_elements(d.linked_urls) AS e
                WHERE COALESCE(e->>'image', '') = ''
           )
        ORDER BY u.username
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def gather_existing_payloads(database_url: str, user_id: int) -> dict[str, list[dict]]:
    """For the `--images-only` pass: pull every `(tweet_url,
    linked_urls)` row that still has at least one entry without an
    image. The returned payloads carry the existing url / host /
    title / summary as-is; only `image` is missing and waiting to
    be filled in."""
    sql = """
        SELECT d.url, d.linked_urls
          FROM documents d
         WHERE d.user_id = %s
           AND d.deleted = false
           AND jsonb_array_length(d.linked_urls) > 0
           AND EXISTS (
               SELECT 1
                 FROM jsonb_array_elements(d.linked_urls) AS e
                WHERE COALESCE(e->>'image', '') = ''
           )
    """
    out: dict[str, list[dict]] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            for url, links in cur.fetchall():
                if isinstance(links, list):
                    out[url] = links
    return out


def gather_links_for_user(database_url: str, user_id: int) -> dict[str, list[dict]]:
    """Return `{tweet_url: [linked_url_entry, ...]}` derived from
    every companion document for *user_id*.

    Only tweets that still need backfilling (empty `link_hosts`) are
    keyed in the result. Companion rows that don't map to a
    surviving tweet doc are skipped silently — they'll be tombstoned
    in step 4 anyway.
    """
    sql = """
        WITH backlog AS (
            SELECT url
              FROM documents
             WHERE user_id = %s
               AND source = 'twitter'
               AND deleted = false
               AND cardinality(link_hosts) = 0
        )
        SELECT c.source_url AS tweet_url,
               c.url        AS dest_url,
               c.source     AS host,
               c.title      AS title,
               c.summary    AS summary
          FROM documents c
          JOIN backlog b ON b.url = c.source_url
         WHERE c.user_id = %s
           AND c.source <> 'twitter'
           AND c.deleted = false
           AND c.source_url IS NOT NULL
         ORDER BY c.source_url, c.created_at
    """
    grouped: dict[str, list[dict]] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, user_id))
            for tweet_url, dest_url, host, title, summary in cur.fetchall():
                bucket = grouped.setdefault(tweet_url, [])
                if len(bucket) >= 5:
                    # MAX_LINKED_URLS_PER_DOC in the pipeline; keep
                    # the parity here so the on-disk row never
                    # exceeds what the pipeline would have produced.
                    continue
                # Dedupe by destination URL within a single tweet
                # (rare, but possible if two companions accidentally
                # share a URL).
                if any(e["url"] == dest_url for e in bucket):
                    continue
                bucket.append(
                    {
                        "url": dest_url,
                        "host": (host or "").lower(),
                        "title": title or "",
                        "summary": summary or "",
                        "image": "",
                    }
                )
    return grouped


def fetch_og_image(url: str) -> str | None:
    """Best-effort og:image extraction from the destination's HTML.

    Direct HTTP, ~6s timeout, 64 KB body cap. Returns the absolute
    URL on hit, `None` on any failure — callers must treat the
    result as optional.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": ("Mozilla/5.0 (compatible; KnowledgeBackfill/1.0; " "+https://knowledge-web.org)"),
                "Accept": "text/html,*/*;q=0.5",
            },
        )
        with urllib.request.urlopen(req, timeout=OG_TIMEOUT_S) as resp:
            raw = resp.read(OG_MAX_BYTES)
    except Exception:
        return None
    m = _OG_IMAGE_RE.search(raw)
    if not m:
        return None
    cand = m.group(1).decode("utf-8", errors="replace").strip()
    if not cand:
        return None
    return urllib.parse.urljoin(url, cand)


def ensure_sqlite_columns(slug: str) -> bool:
    """ALTER `indexes/{slug}/metadata.db` to add `linked_urls` /
    `link_hosts` columns if missing. Returns True on success / no-op,
    False if the index doesn't exist on disk (some pre-VIP slugs
    don't carry one)."""
    path = INDEXES_DIR / slug / "metadata.db"
    if not path.exists():
        return False
    conn = sqlite3.connect(str(path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(METADATA)").fetchall()}
        if "linked_urls" not in cols:
            conn.execute("ALTER TABLE METADATA ADD COLUMN linked_urls TEXT DEFAULT '[]'")
        if "link_hosts" not in cols:
            conn.execute("ALTER TABLE METADATA ADD COLUMN link_hosts TEXT DEFAULT ''")
        conn.commit()
    finally:
        conn.close()
    return True


def update_postgres(
    database_url: str,
    user_id: int,
    payloads: dict[str, list[dict]],
) -> tuple[int, int]:
    """Apply the consolidated payload to PG. Returns
    `(tweets_updated, companions_flagged)`.

    Companion flagging covers every companion of every enriched
    tweet — even the ones we dropped from the cap-5 truncation —
    because the tweet doc carries the canonical view now and the
    extras are still redundant.
    """
    if not payloads:
        return (0, 0)
    tweet_urls = list(payloads.keys())
    tweet_writes = 0
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            # 1. Bulk UPDATE the tweet rows. UNNEST keeps it to one
            #    round-trip even for users with thousands of tweets.
            urls_arr = []
            linked_arr = []
            hosts_arr = []
            for url, entries in payloads.items():
                urls_arr.append(url)
                linked_arr.append(json.dumps(entries))
                # link_hosts is a TEXT[] column; we pass a list per
                # tweet (de-duped while preserving encounter order).
                seen: set[str] = set()
                hosts: list[str] = []
                for e in entries:
                    h = e.get("host", "")
                    if h and h not in seen:
                        seen.add(h)
                        hosts.append(h)
                hosts_arr.append(hosts)
            # 2-D TEXT[] is tricky to bind, so we encode each row's
            # hosts as comma-joined and split server-side. Same
            # trick the bulk-save endpoint uses for `tags`.
            hosts_csv = [",".join(h) for h in hosts_arr]
            sql = """
                UPDATE documents d
                   SET linked_urls = u.linked_urls::jsonb,
                       link_hosts = CASE WHEN u.hosts_csv = '' THEN '{}'::text[]
                                         ELSE string_to_array(u.hosts_csv, ',') END,
                       updated_at = now()
                  FROM UNNEST(%s::text[], %s::text[], %s::text[])
                       AS u(url, linked_urls, hosts_csv)
                 WHERE d.user_id = %s
                   AND d.url = u.url
                   AND d.source = 'twitter'
                   AND cardinality(d.link_hosts) = 0
            """
            cur.execute(sql, (urls_arr, linked_arr, hosts_csv, user_id))
            tweet_writes = cur.rowcount
            # 2. Tombstone the companion rows so the pipeline's
            #    purge worker drops them next run.
            companion_sql = """
                UPDATE documents
                   SET to_delete = TRUE,
                       updated_at = now()
                 WHERE user_id = %s
                   AND source <> 'twitter'
                   AND deleted = false
                   AND to_delete = false
                   AND source_url = ANY(%s)
            """
            cur.execute(companion_sql, (user_id, tweet_urls))
            companion_writes = cur.rowcount
            conn.commit()
    return (tweet_writes, companion_writes)


def update_postgres_images(
    database_url: str,
    user_id: int,
    payloads: dict[str, list[dict]],
) -> int:
    """Image-only PG writer. Overwrites `linked_urls` in place for
    the supplied docs; never touches `link_hosts` or companion
    rows. `payloads` carries the fully-resolved list (every entry
    still has the host / title / summary the consolidate step
    produced; only `image` was updated by the fetch pass)."""
    if not payloads:
        return 0
    urls = list(payloads.keys())
    linked_json = [json.dumps(payloads[u]) for u in urls]
    sql = """
        UPDATE documents d
           SET linked_urls = u.linked_urls::jsonb,
               updated_at = now()
          FROM UNNEST(%s::text[], %s::text[]) AS u(url, linked_urls)
         WHERE d.user_id = %s
           AND d.url = u.url
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (urls, linked_json, user_id))
            n = cur.rowcount
            conn.commit()
            return n


def update_sqlite(slug: str, payloads: dict[str, list[dict]]) -> int:
    """Mirror `linked_urls` + `link_hosts` into the slug's index
    metadata sidecar. Returns the number of rows updated.

    Direct sqlite3 UPDATE — at ~20k rows the per-row API hop would
    be the bottleneck. SQLite handles single-writer concurrency on
    its own, and the running API is read-only against METADATA
    outside its own /metadata/update endpoint.
    """
    path = INDEXES_DIR / slug / "metadata.db"
    if not path.exists() or not payloads:
        return 0
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        n = 0
        # One executemany — SQLite buffers + commits at the end.
        rows = []
        for url, entries in payloads.items():
            hosts = []
            seen: set[str] = set()
            for e in entries:
                h = e.get("host", "")
                if h and h not in seen:
                    seen.add(h)
                    hosts.append(h)
            rows.append(
                (json.dumps(entries), ",".join(hosts), url),
            )
        cur.executemany(
            "UPDATE METADATA " "   SET linked_urls = ?, link_hosts = ? " " WHERE url = ?",
            rows,
        )
        n = cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def maybe_fetch_images(
    payloads: dict[str, list[dict]],
    image_cache: dict[str, str],
    workers: int = 24,
) -> int:
    """Walk every entry across all payloads and fill in `image`
    when we don't already have it cached. Returns the number of
    HTTP fetches actually issued (cache hits don't count). Updates
    the per-entry dict in place.

    Fetches run on a thread pool because each call is I/O-bound
    (network + slow HTML parse) and 23k sequential 6-second
    timeouts would burn the better part of a day. The cache is
    shared across personalities — a URL linked from 30 tweets only
    pays for one fetch.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Phase 1: collect every distinct URL that still needs an
    # image. Cache hits are settled inline so we don't queue them
    # for the pool.
    to_fetch: set[str] = set()
    for entries in payloads.values():
        for entry in entries:
            url = entry.get("url", "")
            if not url or entry.get("image"):
                continue
            cached = image_cache.get(url)
            if cached is not None:
                entry["image"] = cached
                continue
            to_fetch.add(url)
    if not to_fetch:
        return 0

    # Phase 2: parallel fetch. The pool size is a network knob —
    # 24 workers keeps the script moving without overwhelming the
    # local DNS resolver or upstream rate limits.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_og_image, u): u for u in to_fetch}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                image_cache[url] = fut.result() or ""
            except Exception:
                image_cache[url] = ""

    # Phase 3: write the resolved images back into every payload
    # entry that referenced one of the fetched URLs.
    for entries in payloads.values():
        for entry in entries:
            url = entry.get("url", "")
            if url and not entry.get("image"):
                entry["image"] = image_cache.get(url, "")
    return len(to_fetch)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slug",
        default=None,
        help="Restrict to one personality (e.g. jobergum). Default = "
        "all personalities with at least one un-backfilled tweet.",
    )
    p.add_argument(
        "--fetch-images",
        action="store_true",
        help="During the consolidation join, also issue one HTTP "
        "request per unique destination URL to pull `og:image`. "
        "Slow even when parallelised — flip on once you've decided "
        "you want previews on the first run rather than a separate "
        "`--images-only` pass.",
    )
    p.add_argument(
        "--images-only",
        action="store_true",
        help="Skip the consolidation join and only fetch og:image "
        "for already-enriched docs whose `linked_urls` carries an "
        "entry with an empty `image`. Useful for backfilling "
        "previews after a first run that ran without --fetch-images.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on personalities to process this run (0 = all).",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print what would change; don't write to PG or SQLite.",
    )
    args = p.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    # Target list. The two backfill modes select different
    # working sets: the consolidation join walks personalities with
    # un-enriched tweets (empty `link_hosts`); the images-only pass
    # walks personalities whose existing payloads still have empty
    # `image` fields.
    if args.slug:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username FROM users WHERE username = %s",
                    (args.slug,),
                )
                row = cur.fetchone()
                if not row:
                    print(
                        f"error: no user with username {args.slug!r}",
                        file=sys.stderr,
                    )
                    return 2
                targets = [(int(row[0]), str(row[1]))]
    elif args.images_only:
        targets = list_personalities_with_missing_images(database_url)
        if args.limit:
            targets = targets[: args.limit]
    else:
        targets = list_personalities_with_backlog(database_url)
        if args.limit:
            targets = targets[: args.limit]

    mode_tag = "images-only" if args.images_only else "consolidate"
    print(f"{len(targets)} personality(ies) to process ({mode_tag})")
    if args.fetch_images and not args.images_only:
        print("  --fetch-images: ON (slow; ~6s per destination URL)")

    total_tweets = 0
    total_companions = 0
    total_sqlite = 0
    total_imgs = 0
    image_cache: dict[str, str] = {}
    for user_id, slug in targets:
        if args.images_only:
            payloads = gather_existing_payloads(database_url, user_id)
            if not payloads:
                print(f"  @{slug}: nothing to do")
                continue
            # Always fetch in this mode — that's the whole point.
            n_imgs = maybe_fetch_images(payloads, image_cache)
            total_imgs += n_imgs
            if args.dry:
                print(f"  @{slug}: would update {len(payloads)} doc(s); " f"fetched_images={n_imgs}")
                continue
            # Schema is guaranteed present here — every doc in the
            # work set already has linked_urls populated, so the
            # consolidate run that produced them already ALTERed
            # the SQLite METADATA table.
            sqlite_writes = update_postgres_images(database_url, user_id, payloads)
            sqlite_meta_writes = update_sqlite(slug, payloads)
            total_tweets += sqlite_writes
            total_sqlite += sqlite_meta_writes
            print(
                f"  @{slug}: pg_tweets={sqlite_writes} "
                f"sqlite_tweets={sqlite_meta_writes} "
                f"og_images_fetched={n_imgs}"
            )
            continue

        payloads = gather_links_for_user(database_url, user_id)
        if not payloads:
            print(f"  @{slug}: nothing to do")
            continue

        if args.fetch_images:
            n_imgs = maybe_fetch_images(payloads, image_cache)
            total_imgs += n_imgs
        else:
            n_imgs = 0

        if args.dry:
            n_tweets = len(payloads)
            n_companions_est = sum(len(v) for v in payloads.values())
            print(
                f"  @{slug}: would update {n_tweets} tweet(s), "
                f"flag ≥{n_companions_est} companion(s); "
                f"fetched_images={n_imgs}"
            )
            continue

        # ALTER first so the SQLite UPDATE below has somewhere to
        # write. Index may not exist for non-VIP slugs that never
        # got built — skip the SQLite step if so.
        index_present = ensure_sqlite_columns(slug)
        tweet_writes, companion_writes = update_postgres(database_url, user_id, payloads)
        sqlite_writes = update_sqlite(slug, payloads) if index_present else 0
        total_tweets += tweet_writes
        total_companions += companion_writes
        total_sqlite += sqlite_writes
        print(
            f"  @{slug}: pg_tweets={tweet_writes} "
            f"sqlite_tweets={sqlite_writes} "
            f"companions_flagged={companion_writes} "
            f"og_images_fetched={n_imgs}"
        )

    print(
        f"\ndone: "
        f"pg_tweets={total_tweets} "
        f"sqlite_tweets={total_sqlite} "
        f"companions_flagged={total_companions} "
        f"og_images_fetched={total_imgs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
