"""Map the "AI Twitter inner circle" via the follow-graph of existing
VIPs.

Idea
----
SOTA-chasers tend to follow the same handful of accounts: paper-spotters
(Karpathy, Lilian Weng, swyx), newsletter authors (Nathan Lambert,
Simon Willison, Sebastian Raschka), and the rest of the "AI Twitter"
inner circle. We surface those accounts by computing the *overlap* in
who our existing VIPs follow on X — the more VIPs that follow someone,
the more central they are to that circle by construction.

Flow
----
1. Pull the top-N VIPs from PG that have a Twitter handle configured
   (sources.twitter.username), sorted by their own twitter_followers
   so the seed set is the densest hubs we already have.
2. For each VIP, walk `/twitter/user/followings` up to `--max-follows`
   accounts (default 1000, ~5 pages of 200).
3. Aggregate: per followed account, count distinct VIPs that follow it
   and remember which ones.
4. Drop accounts that are already on Knowledge (case-insensitive match
   against `users.username` or `sources.twitter.username`).
5. Write CSV ordered by VIP-overlap desc, then follower count desc.

Output columns:

    handle, name, followers, following, bio, verified,
    vip_overlap_count, vip_followers (semicolon-separated VIP slugs)

The CSV is purely a *candidate list* — no DB writes. The next step is
manually triaging it (or feeding the top-K into the add-personality
flow you already have).

Usage::

    uv run python scripts/fetch_twitter_inner_circle.py
    uv run python scripts/fetch_twitter_inner_circle.py --vips 30 --max-follows 1000
    uv run python scripts/fetch_twitter_inner_circle.py --out data/people/twitter_inner_circle.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import psycopg

_API_BASE = "https://api.twitterapi.io"
_USER_AGENT = "knowledge-twitter-inner-circle/1.0"
# twitterapi.io tolerates fast calls but be polite — followings is
# paginated 200/page, so even at 0.5s/page we walk 1000 follows per
# VIP in ~3 seconds.
_REQ_DELAY_S = 0.4
# Per-page cap on the followings endpoint (server-side max).
_PAGE_SIZE = 200

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"


def _get_json(url: str, api_key: str) -> dict | None:
    req = urllib.request.Request(
        url,
        headers={"X-API-Key": api_key, "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print(f"    ! HTTP {resp.status} for {url}")
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"    ! fetch error: {e}")
        return None


def fetch_followings(handle: str, api_key: str, max_count: int) -> list[dict]:
    """Walk the followings endpoint until we hit `max_count` or the
    server says `has_next_page=false`."""
    out: list[dict] = []
    cursor: str | None = None
    while len(out) < max_count:
        params = {"userName": handle, "pageSize": _PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        url = f"{_API_BASE}/twitter/user/followings?{urllib.parse.urlencode(params)}"
        page = _get_json(url, api_key)
        if not page:
            break
        batch = page.get("followings") or []
        if not batch:
            break
        out.extend(batch)
        if not page.get("has_next_page"):
            break
        cursor = page.get("next_cursor")
        if not cursor:
            break
        time.sleep(_REQ_DELAY_S)
    return out[:max_count]


def load_seed_vips(database_url: str, limit: int) -> list[dict]:
    """Top-N VIPs with a Twitter handle, ranked by follower count desc."""
    sql = """
        SELECT u.username                              AS slug,
               u.name,
               LOWER(u.sources->'twitter'->>'username') AS twitter_handle,
               u.twitter_followers
          FROM users u
         WHERE u.vip = TRUE
           AND u.sources ? 'twitter'
           AND COALESCE(u.sources->'twitter'->>'username', '') <> ''
         ORDER BY u.twitter_followers DESC NULLS LAST, u.id
         LIMIT %s
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
    return [{"slug": s, "name": n, "twitter": t.lstrip("@") if t else "", "followers": f} for s, n, t, f in rows if t]


def load_existing_handles(database_url: str) -> set[str]:
    """Every Twitter handle Knowledge already knows about (lower-cased)."""
    sql = """
        SELECT LOWER(username)                              AS slug,
               LOWER(sources->'twitter'->>'username')       AS handle
          FROM users
    """
    out: set[str] = set()
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for slug, handle in cur.fetchall():
                if slug:
                    out.add(slug)
                if handle:
                    out.add(handle.lstrip("@"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vips", type=int, default=30, help="Seed VIP count (default 30)")
    ap.add_argument(
        "--max-follows",
        type=int,
        default=1000,
        help="Cap on followings fetched per seed VIP (default 1000)",
    )
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Drop candidates followed by fewer than this many seed VIPs (default 2)",
    )
    ap.add_argument(
        "--out",
        default="data/people/twitter_inner_circle.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    api_key = os.environ.get("TWITTERAPIIO_API_KEY")
    if not api_key:
        print("TWITTERAPIIO_API_KEY missing in env — bailing.")
        return 1
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    seed = load_seed_vips(database_url, args.vips)
    if not seed:
        print("No VIPs with a Twitter handle in PG — nothing to crawl.")
        return 1
    existing = load_existing_handles(database_url)
    print(f"Seed: {len(seed)} VIPs; will skip {len(existing)} handles already on Knowledge.\n")

    # candidate_handle → {
    #   "name": str, "followers": int, "following": int,
    #   "bio": str, "verified": bool,
    #   "vips": set[slug]
    # }
    cand: dict[str, dict] = {}

    for i, vip in enumerate(seed, 1):
        print(f"[{i}/{len(seed)}] {vip['slug']:<28} (@{vip['twitter']})")
        follows = fetch_followings(vip["twitter"], api_key, args.max_follows)
        print(f"    {len(follows)} follows")
        for u in follows:
            handle = (u.get("userName") or u.get("screen_name") or "").lstrip("@").lower()
            if not handle or handle in existing:
                continue
            entry = cand.get(handle)
            if entry is None:
                entry = {
                    "name": (u.get("name") or "").strip(),
                    "followers": int(u.get("followers_count") or 0),
                    "following": int(u.get("following_count") or u.get("friends_count") or 0),
                    "bio": (u.get("description") or "").strip(),
                    "verified": bool(u.get("verified") or u.get("isBlueVerified") or False),
                    "vips": set(),
                }
                cand[handle] = entry
            entry["vips"].add(vip["slug"])
        time.sleep(_REQ_DELAY_S)

    # Filter + sort. Primary: vip_overlap desc. Tiebreak: followers desc.
    filtered = [(h, c) for h, c in cand.items() if len(c["vips"]) >= args.min_overlap]
    filtered.sort(key=lambda hc: (-len(hc[1]["vips"]), -hc[1]["followers"]))

    repo_root = Path(__file__).resolve().parent.parent
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "handle",
                "name",
                "followers",
                "following",
                "bio",
                "verified",
                "vip_overlap_count",
                "vip_followers",
            ]
        )
        for handle, c in filtered:
            # Trim bio so the CSV stays readable — full bio is one
            # twitterapi.io call away (`/twitter/user/info`) when
            # enriching individual rows later.
            bio = c["bio"].replace("\n", " ")[:240]
            w.writerow(
                [
                    handle,
                    c["name"],
                    c["followers"],
                    c["following"],
                    bio,
                    "true" if c["verified"] else "false",
                    len(c["vips"]),
                    ";".join(sorted(c["vips"])),
                ]
            )

    print(
        f"\n✓ wrote {out_path} — {len(filtered):,} candidates "
        f"(filtered from {len(cand):,} unique follows; min_overlap={args.min_overlap})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
