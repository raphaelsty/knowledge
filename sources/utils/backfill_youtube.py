"""CLI: backfill title + summary on YouTube docs that landed with garbage metadata.

When a tweet links a youtu.be / youtube.com URL the twitter pipeline
used to try a plain HTML scrape; YouTube serves a JS-rendered shell to
non-browser clients, so the scrape returned nothing and the fallback
slug-derived title kicked in (e.g. ``K4i C5YYvr Qk`` for ``k4iC5YYvrQk``).

This script walks the `documents` table, picks every YouTube row that
either has an empty summary, a junky slug-style title, or both, and
calls YouTube's oEmbed endpoint (no API key required) to fetch the real
title + channel name. Writes results back via plain UPDATEs — no row
is deleted, no row is added.

Usage::

    make backfill-youtube           # apply to every user
    make backfill-youtube DRY=1     # plan-only
    make backfill-youtube SLUG=alice
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

# A "slug-style" title is one that has no spaces and looks like a URL
# path segment — i.e. the fallback `_title_from_url_slug` output. We
# treat any row whose title is empty, equals the URL, or is at most a
# couple of words derived from the video id, as a backfill candidate.
_SLUG_LIKE = re.compile(r"^[A-Za-z0-9 ]{1,40}$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="backfill_youtube.py")
    p.add_argument("--slug", default=None, help="Restrict to one user.")
    p.add_argument(
        "--dry",
        action="store_true",
        help="Plan only — print candidates, no DB write.",
    )
    return p.parse_args()


def _oembed(url: str, timeout: float = 5.0) -> dict | None:
    oe = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url, safe='')}&format=json"
    req = urllib.request.Request(
        oe,
        headers={"User-Agent": "Knowledge/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read(16_384).decode("utf-8", errors="replace"))
    except Exception:
        # 404 (unlisted/private/dead), 403 (region/age-gated, embed
        # disabled), 429 (rate-limited), JSON parse errors, timeouts —
        # all silently skip. The row stays as-is and a later run can
        # retry once the upstream cooperates.
        return None


def _candidate(title: str, summary: str, url: str) -> bool:
    """A row needs backfilling if the title looks slug-ish OR there is no summary."""
    if not summary or not summary.strip():
        return True
    t = (title or "").strip()
    if not t:
        return True
    if t == url:
        return True
    # Slug-style titles look like "K4i C5YYvr Qk" — short, no real words.
    if _SLUG_LIKE.match(t) and " " in t and not any(c.islower() for c in t.split()):
        return True
    return False


def main() -> None:
    args = parse_args()
    db_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    where = [
        # psycopg parses `%` as a parameter placeholder unless escaped
        # as `%%` — the LIKE wildcards live in the SQL literal so we
        # double them up.
        "(d.url ILIKE 'https://youtu.be/%%' "
        "OR d.url ILIKE 'https://www.youtube.com/%%' "
        "OR d.url ILIKE 'https://youtube.com/%%')",
        "d.deleted = FALSE",
    ]
    params: list = []
    if args.slug:
        where.append("u.username = %s")
        params.append(args.slug)

    sql = (
        "SELECT u.id, u.username, d.url, d.title, d.summary "
        "  FROM documents d JOIN users u ON u.id = d.user_id "
        " WHERE " + " AND ".join(where) + " ORDER BY u.username, d.url"
    )

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        candidates = [
            (uid, user, url, title, summary)
            for (uid, user, url, title, summary) in rows
            if _candidate(title, summary, url)
        ]
        print(f"Inspected {len(rows)} YouTube docs, {len(candidates)} need backfill.")
        if args.dry:
            for _uid, user, url, title, summary in candidates[:20]:
                print(f"  - {user}  {url}  title={title!r:30}  sum_len={len(summary or '')}")
            if len(candidates) > 20:
                print(f"  … (+{len(candidates) - 20} more)")
            print("\n--dry: no DB writes.")
            return

        fixed = skipped = 0
        for uid, user, url, _title, _summary in candidates:
            meta = _oembed(url)
            if not meta or not meta.get("title"):
                skipped += 1
                continue
            new_title = meta["title"].strip()
            author = (meta.get("author_name") or "").strip()
            new_summary = f"Video by {author}" if author else "YouTube video"
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET title = %s, summary = %s, updated_at = now()"
                    " WHERE user_id = %s AND url = %s",
                    (new_title[:500], new_summary[:1000], uid, url),
                )
            conn.commit()
            fixed += 1
            # Be polite to YouTube's edge — oEmbed is cheap but not free.
            time.sleep(0.15)
            print(f"  ✓ {user:<28} {url}  → {new_title[:60]}")

        print(f"\nDone. fixed={fixed} skipped={skipped} (no oEmbed result)")


if __name__ == "__main__":
    main()
