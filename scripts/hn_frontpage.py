#!/usr/bin/env python3
"""Daily HackerNews front-page → per-user picks job.

Runs ONCE per day to keep every user's feed fresh with HN stories that
match what they actually read about. Independent of `make run`: this
job only needs the search API to be up (no source fetching, no
embedding rebuild). Schedule it on its own cron so feed freshness is
decoupled from the much slower per-personality pipeline.

Flow:
    1. Snapshot the current HN front page (Firebase API).
    2. Insert one `hn_frontpage_runs` row + its ~30
       `hn_frontpage_items` rows.
    3. For each user, encode every article's title via
       ``/indices/{name}/search_with_encoding`` and read back the
       top-K ColBERT scores against their library.
    4. Keep the top-N articles by *mean* score, then **reorder by HN
       upvote count** so the feed surfaces the most-discussed
       relevant items first.
    5. REPLACE the user's picks for this run (atomic per-user); the
       feed query joins on `run_id = MAX(id)` so it always sees a
       coherent set.

Usage:
    DATABASE_URL=postgresql://... API_URL=http://localhost:8080 \\
      uv run python scripts/hn_frontpage.py [flags]

Flags:
    --slug NAME      score for one user only (debug)
    --top-per-user N keep at most N picks per user (default 10)
    --top N          fetch only the top-N front-page items
                     (default 30 — matches HN's own front page)
    --limit N        stop after N users (cron-friendly)
    --dry            fetch + score, but don't write picks to PG.
                     (The run + items ARE still inserted.)
    --no-snapshot    skip step 1/2 and score against the most recent
                     existing run (useful if you re-run after fixing
                     a bug in the scorer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make the package importable when run as `python scripts/hn_frontpage.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg  # noqa: E402

from sources.hackernews import Frontpage  # noqa: E402
from sources.sql import (  # noqa: E402
    create_hn_frontpage_tables,
    get_run_items,
    insert_run,
    latest_run_id,
    list_personalities,
    replace_user_picks,
)

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_TOP_PER_USER = 10
DEFAULT_TOP = 30
# Window over which we average the ColBERT scores. Large enough to
# smooth out single-token noise, small enough to stay relevant.
SEARCH_TOP_K = 10


# ── HTTP helpers ────────────────────────────────────────────────────────


def _post_json(url: str, payload: dict, timeout: int = 60) -> tuple[int, dict | None, str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Knowledge/hn-frontpage"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        try:
            txt = e.read().decode("utf-8", "replace")
        except Exception:
            txt = ""
        return e.code, None, txt[:300]
    except Exception as exc:
        return 0, None, str(exc)


def _search_index(api_url: str, index_name: str, queries: list[str], top_k: int) -> list[list[float]] | None:
    """Batched search call. Returns one score list per query, or None on error."""
    status, body, err = _post_json(
        f"{api_url}/indices/{index_name}/search_with_encoding",
        {"queries": queries, "params": {"top_k": top_k}},
    )
    if status != 200 or not body:
        if status != 404:
            print(f"      ! search_with_encoding({index_name}): status={status} err={err}")
        return None
    return [(r.get("scores") or []) for r in body.get("results") or []]


def _score(api_url: str, user_index: str, items: list[dict]) -> list[tuple[int, float]]:
    """Score every article by the mean ColBERT similarity against the user.

    For each article, we send the title as the query and average the
    top-K returned scores. That's it — no tag graph, no global IDF,
    no specialization weights. The single number tells us how much
    the user's library "agrees" that this article is on-topic.

    Returns ``[(hn_id, mean_score), ...]`` ordered by score desc.
    """
    queries = [it["title"].strip() for it in items]
    score_lists = _search_index(api_url, user_index, queries, SEARCH_TOP_K)
    if score_lists is None:
        return []
    out: list[tuple[int, float]] = []
    for it, scores in zip(items, score_lists, strict=False):
        if not scores:
            continue
        mean_score = sum(scores) / len(scores)
        out.append((int(it["hn_id"]), float(mean_score)))
    out.sort(key=lambda r: r[1], reverse=True)
    return out


# ── Main ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hn_frontpage.py", description=__doc__)
    p.add_argument("--slug", default=None, help="Score only this personality.")
    p.add_argument("--top-per-user", type=int, default=DEFAULT_TOP_PER_USER)
    p.add_argument("--top", type=int, default=DEFAULT_TOP)
    p.add_argument("--limit", type=int, default=0, help="Stop after N users.")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--no-snapshot", action="store_true")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print the per-article ColBERT mean for each user.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_url = os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")

    create_hn_frontpage_tables(database_url)

    # ── Step 1+2: snapshot the front page ────────────────────────────
    if args.no_snapshot:
        run_id = latest_run_id(database_url)
        if run_id is None:
            print("--no-snapshot requested but no existing run found. Bailing.")
            return 1
        items = get_run_items(database_url, run_id)
        print(f"Re-scoring against existing run #{run_id} ({len(items)} items)")
    else:
        print(f"Fetching HN front page (top={args.top}) ...")
        items = Frontpage(top=args.top)()
        if not items:
            print("No items fetched; aborting.")
            return 1
        run_id = insert_run(database_url, items)
        print(f"Inserted run #{run_id} with {len(items)} items")

    # Cache for the post-score upvote reorder. `points` is set by the
    # Frontpage scraper; `get_run_items` carries the column through
    # as well so this works for both code paths.
    points_by_id = {int(it["hn_id"]): int(it.get("points") or 0) for it in items}
    title_by_id = {int(it["hn_id"]): it["title"] for it in items}

    # ── Step 3+4: score each user ────────────────────────────────────
    personalities = list_personalities(database_url)
    if args.slug:
        personalities = [p for p in personalities if p["slug"] == args.slug]
        if not personalities:
            print(f"slug '{args.slug}' not in users table")
            return 1
    if args.limit:
        personalities = personalities[: args.limit]

    total_picks = 0
    started = time.perf_counter()
    for i, p in enumerate(personalities, start=1):
        index_name = p["indexName"]
        print(f"  [{i}/{len(personalities)}] {p['slug']:<28} (index={index_name})")
        scored = _score(api_url, index_name, items)
        if not scored:
            print("      no scores returned (empty / missing index?)")
            continue
        # 1. Keep top-N by ColBERT mean score.
        top = scored[: args.top_per_user]
        # 2. Re-order by HN upvote count — most-discussed first.
        top_by_upvotes = sorted(top, key=lambda r: -points_by_id.get(r[0], 0))
        if args.debug:
            print("      ┌─ picks (after upvote reorder) ──────────────────────")
            for hn_id, sc in top_by_upvotes:
                pts = points_by_id.get(hn_id, 0)
                print(f"      │ {pts:5d} pts  score={sc:5.2f}  {title_by_id.get(hn_id, '')[:60]}")
            print("      └──────────────────────────────────────────────────────")
        print(f"      {len(top_by_upvotes)} picks  (best score {top[0][1]:.2f}, worst {top[-1][1]:.2f})")
        if not args.dry:
            try:
                replace_user_picks(database_url, p["id"], run_id, top_by_upvotes)
            except psycopg.Error as exc:
                print(f"      ! pg write failed: {exc}")
                continue
        total_picks += len(top_by_upvotes)

    dur = time.perf_counter() - started
    verb = "would write" if args.dry else "wrote"
    print(f"\nDone. {verb} {total_picks} picks across {len(personalities)} users in {dur:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
