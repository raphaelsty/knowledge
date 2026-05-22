"""Suggest Twitter/X handles for personalities missing one.

Iterates over every user where ``sources.twitter`` is unset, runs a
DuckDuckGo lite search ``"<name> twitter"``, and extracts plausible
``x.com/<handle>`` candidates from the result HTML.

By default the script only PRINTS suggestions — review them and apply
manually with SQL or pass ``--apply`` to auto-set the top suggestion
(use with caution; DDG is good but not perfect, especially for common
names).

Usage::

    uv run python -m sources.utils.find_twitter_handles
    uv run python -m sources.utils.find_twitter_handles --apply
    uv run python -m sources.utils.find_twitter_handles --slug peter-norvig
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Iterable

import psycopg
import requests

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
LITE_URL = "https://lite.duckduckgo.com/lite/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Knowledge/1.0"

# Handles that aren't real personality accounts — Twitter UI paths,
# generic brand accounts, etc. Anything in this set is filtered out
# of the candidate list.
JUNK_HANDLES = {
    "intent",
    "share",
    "home",
    "search",
    "i",
    "explore",
    "settings",
    "messages",
    "compose",
    "login",
    "signup",
    "logout",
    "tos",
    "privacy",
    "rules",
    "support",
    "help",
    "twitter",
    "x",
    "verified",
    "premium",
    "twitterapi",
    "TwitterDev",
    "Data_AI_Summit",
}

HANDLE_RE = re.compile(r"(?:twitter\.com|x\.com)/([A-Za-z0-9_]{1,15})\b")


def search_handles(query: str, *, max_results: int = 10) -> list[str]:
    """Return up to ``max_results`` distinct handle candidates for *query*.

    Uses DDG's lite endpoint, which renders results as plain HTML
    (no JS, no anti-scraping headers). Order is preserved — the first
    handle in the list is the top-ranked DDG result.
    """
    try:
        r = requests.post(
            LITE_URL,
            data={"q": query},
            headers={"User-Agent": UA},
            timeout=20,
        )
    except requests.RequestException as exc:
        print(f"  ⚠ request failed: {exc}")
        return []

    if r.status_code != 200:
        print(f"  ⚠ HTTP {r.status_code}")
        return []

    seen: set[str] = set()
    out: list[str] = []
    for m in HANDLE_RE.finditer(r.text):
        h = m.group(1)
        if h in seen:
            continue
        if h in JUNK_HANDLES or h.lower() in {x.lower() for x in JUNK_HANDLES}:
            continue
        # Strip handles that are pure digits — they're tweet IDs, not user handles.
        if h.isdigit():
            continue
        seen.add(h)
        out.append(h)
        if len(out) >= max_results:
            break
    return out


def list_missing(database_url: str, *, slug: str | None = None) -> list[dict]:
    """Return users with no ``sources.twitter`` (optionally a single slug)."""
    sql = "SELECT id, username, name, vip   FROM users  WHERE NOT (sources ? 'twitter')"
    params: tuple = ()
    if slug:
        sql += " AND username = %s"
        params = (slug,)
    sql += " ORDER BY vip DESC, name"

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [{"id": r[0], "username": r[1], "name": r[2], "vip": r[3]} for r in cur.fetchall()]


def apply_handle(database_url: str, slug: str, handle: str) -> None:
    """Write ``links.twitter`` and ``sources.twitter.username`` for *slug*."""
    sql = (
        "UPDATE users SET "
        "  links   = jsonb_set(COALESCE(links,'{}'::jsonb),   '{twitter}',          %s::jsonb), "
        "  sources = jsonb_set(COALESCE(sources,'{}'::jsonb), '{twitter}',          %s::jsonb), "
        "  updated_at = now() "
        "WHERE username = %s"
    )
    link_json = json.dumps(f"https://x.com/{handle}")
    src_json = json.dumps({"username": handle})
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, (link_json, src_json, slug))
        conn.commit()


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="find_twitter_handles",
        description="Suggest Twitter handles via DuckDuckGo for users with no twitter source.",
    )
    p.add_argument(
        "--apply", action="store_true", help="Auto-apply the top candidate (review the printed suggestions first!)."
    )
    p.add_argument("--slug", default=None, help="Run for a single personality slug instead of every missing one.")
    p.add_argument("--max", type=int, default=5, help="Show up to N candidates per user (default: 5).")
    p.add_argument(
        "--sleep", type=float, default=1.5, help="Seconds between DDG queries to stay polite (default: 1.5)."
    )
    args = p.parse_args(argv if argv is None else list(argv))

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    users = list_missing(database_url, slug=args.slug)
    if not users:
        print("No users to process — every personality already has sources.twitter set.")
        return 0

    print(f"Found {len(users)} personalities without sources.twitter:\n")

    suggestions: list[tuple[str, str, list[str]]] = []
    for u in users:
        query = f"{u['name']} twitter"
        cands = search_handles(query, max_results=args.max)
        suggestions.append((u["username"], u["name"], cands))

        flag = " ★" if u["vip"] else "  "
        if cands:
            print(f"{flag} {u['username']:<28} {u['name']}")
            for i, c in enumerate(cands):
                marker = "→" if i == 0 else " "
                print(f"        {marker} https://x.com/{c}")
        else:
            print(f"{flag} {u['username']:<28} {u['name']}  (no candidates)")
        time.sleep(args.sleep)

    if not args.apply:
        print("\nReview the suggestions above. Re-run with --apply to set the TOP candidate")
        print("for each user, or apply selectively via SQL.")
        return 0

    # Auto-apply mode — set top candidate for everyone who got one.
    print("\n--apply: writing top candidate for each user...")
    n_set = 0
    for slug, _name, cands in suggestions:
        if not cands:
            continue
        handle = cands[0]
        apply_handle(database_url, slug, handle)
        print(f"  ✓ {slug:<28} → @{handle}")
        n_set += 1
    print(f"\nApplied {n_set} handle{'s' if n_set != 1 else ''}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
