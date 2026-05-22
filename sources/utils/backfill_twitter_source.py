"""CLI: derive `sources.twitter.username` from `links.twitter`.

Many personalities only carry their Twitter URL in ``users.links.twitter``
but have no ``sources.twitter`` entry, so the Python pipeline skips
Twitter fetching for them entirely.

This helper walks every user where:
  * ``links.twitter`` is set
  * ``sources.twitter`` is **not** set (we never overwrite an explicit
    config — humans may have tuned `min_likes` / `max_pages`)
  * a usable @handle can be extracted from the URL

and writes::

    sources.twitter = {"username": <handle>, "max_pages": 5,
                       "max_age_years": 2}

so the next ``make run`` invokes the Twitter fetcher with the API key
already in ``.env``. Idempotent — re-runs only touch rows still missing
the source.

Run: ``uv run python -m sources.utils.backfill_twitter_source``
"""

from __future__ import annotations

import json
import os
import re
import sys

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_MAX_AGE_YEARS = 2

# Match a handle in `https://x.com/<handle>` or `https://twitter.com/<handle>`,
# tolerating an optional trailing path / query / fragment. Drops `@`, `?...`
# and anything after the first slash.
_HANDLE_RE = re.compile(r"^https?://(?:www\.)?(?:x|twitter)\.com/(?:#!/)?@?([A-Za-z0-9_]{1,15})(?:[/?#].*)?$")


def extract_handle(url: str) -> str | None:
    if not url:
        return None
    m = _HANDLE_RE.match(url.strip())
    if not m:
        return None
    handle = m.group(1)
    # Twitter system paths that aren't real handles.
    if handle.lower() in {"home", "explore", "search", "i", "intent", "settings"}:
        return None
    return handle


def backfill(database_url: str, *, dry_run: bool = False) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Walk users that need a Twitter source derived and update them.

    Returns ``(updated, skipped_no_handle)``. Each tuple is
    ``(slug, handle_or_url)``. Safe to call from the pipeline boot —
    doesn't print anything; the caller decides what to log.
    """
    updated: list[tuple[str, str]] = []
    skipped_no_handle: list[tuple[str, str]] = []

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, links->>'twitter' AS tw "
                "  FROM users "
                " WHERE links ? 'twitter' "
                "   AND NOT (sources ? 'twitter')"
            )
            candidates = cur.fetchall()

        for slug, twitter_url in candidates:
            handle = extract_handle(twitter_url or "")
            if not handle:
                skipped_no_handle.append((slug, twitter_url or ""))
                continue
            # No `max_pages` — let the fetcher's own default apply so we
            # don't pin behaviour at backfill time. Keep `max_age_years`
            # because the fetcher requires a hard date fence.
            twitter_block = {
                "username": handle,
                "max_age_years": DEFAULT_MAX_AGE_YEARS,
            }
            if dry_run:
                updated.append((slug, handle))
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users "
                    "   SET sources = sources || jsonb_build_object('twitter', %s::jsonb), "
                    "       updated_at = now() "
                    " WHERE username = %s "
                    "   AND NOT (sources ? 'twitter')",
                    (json.dumps(twitter_block), slug),
                )
            updated.append((slug, handle))

        if not dry_run:
            conn.commit()

    return updated, skipped_no_handle


def main() -> None:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    dry_run = "--dry" in sys.argv
    updated, skipped = backfill(url, dry_run=dry_run)

    print(f"{'(dry-run) ' if dry_run else ''}updated   {len(updated):>3}")
    print(f"skipped (unparseable URL) {len(skipped):>3}")
    for slug, handle in updated:
        print(f"  + {slug:<28} → @{handle}")
    for slug, raw in skipped:
        print(f"  ? {slug:<28} → {raw}  (no handle extracted)")


if __name__ == "__main__":
    main()
