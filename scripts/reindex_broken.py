#!/usr/bin/env python3
"""Spot broken per-user search indices and rebuild them from PG.

A broken index is one where the on-disk shape disagrees with PG:

  - **broken**:  API GET /indices/{name} reports
                   ``num_documents > 0 AND num_embeddings == 0`` —
                 the chunk scaffolding exists but the embedding arrays
                 were never written. Search returns
                 ``HTTP 500 "No data to merge"``. This is the failure
                 mode produced by a pipeline that crashed mid-write.
  - **error**:   API GET /indices/{name} returned 5xx (e.g. next-plaid
                 fails to load the index). Disk is corrupt.
  - **missing**: API returned 404 BUT PG has documents for the user —
                 a per-user index that should exist but doesn't.

`pg_drift` (index num_documents disagrees with PG count beyond the noise
floor) is reported but NOT auto-rebuilt by default — it's frequently
caused by an in-flight pipeline pass and resolves on its own. Pass
``--include-drift`` to rebuild those too.

Rebuild path: we call ``sources.utils.run_pipeline`` with an empty
``sources_config`` so no fetchers run. The pipeline's own
``_heal_broken_index`` deletes the bad on-disk index, marks every doc
``indexed=false`` in PG, and the indexing stage at the tail of the
pipeline re-embeds the entire library from scratch. Cheap relative to
``make run SLUG=…`` because the network legs (Twitter, GitHub, blog
crawls) are skipped — only the embedder runs.

Usage::

    DATABASE_URL=postgresql://... API_URL=http://localhost:8080 \\
        uv run python scripts/reindex_broken.py [--dry] [--include-drift] \\
                                                [--vip-only] [--slug SLUG]

Or via the Makefile::

    make reindex-broken          # detect and rebuild
    make reindex-broken DRY=1    # report what would be rebuilt, no work
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

# Allow `uv run python scripts/...` to import the `sources` package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg  # noqa: E402

from sources.sql import get_user_tags, get_vip_tags  # noqa: E402
from sources.utils import run_pipeline  # noqa: E402

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_URL = "http://localhost:8080"

# Mirrors the threshold in scripts/check_indexes.py — index counts are
# allowed to lag PG by max(5 docs, 5%) without us calling it drift.
DRIFT_ABS = 5
DRIFT_FRAC = 0.05


def classify(api_url: str, slug: str, pg_total: int, pg_indexed: int) -> tuple[str, str]:
    """Return (verdict, reason) for one user's index.

    Verdicts:
      healthy      — counts agree, num_embeddings > 0
      broken       — num_documents > 0 but num_embeddings == 0
      error        — API 5xx (index file corrupt / fails to load)
      missing      — API 404 and PG has docs (an index that should exist)
      empty        — API 404 and PG has 0 docs (clean state, ignore)
      pg_drift     — index loads but disagrees with PG beyond the noise floor
    """
    try:
        with urllib.request.urlopen(f"{api_url}/indices/{slug}", timeout=10) as resp:
            info = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code == 404:
            return ("missing", "API 404") if pg_total > 0 else ("empty", "no docs")
        if e.code >= 500 or "NEXT_PLAID_ERROR" in body or "No data to merge" in body:
            return ("error", f"HTTP {e.code} {body[:80]}")
        return ("error", f"HTTP {e.code}")
    except Exception as e:
        return ("error", f"transport: {e}")

    n_docs = int(info.get("num_documents") or 0)
    n_emb = int(info.get("num_embeddings") or 0)
    if n_docs > 0 and n_emb == 0:
        return ("broken", f"num_documents={n_docs}, num_embeddings=0")
    # Drift = the PG view of how many docs are indexed disagrees with
    # what the index actually holds. The denominator is the larger of
    # `pg_indexed` and `pg_total` so we still catch the case where a
    # heal-then-failed-rebuild left PG at 0 indexed but the on-disk
    # index still carries hundreds of docs (geoffrey-hinton 0/637).
    pg_baseline = max(pg_indexed, pg_total)
    if pg_baseline > 0:
        drift = abs(pg_indexed - n_docs)
        threshold = max(DRIFT_ABS, int(pg_baseline * DRIFT_FRAC))
        if drift > threshold:
            return ("pg_drift", f"pg_indexed={pg_indexed} api={n_docs} drift={drift}")
    return ("healthy", "")


def fetch_users(database_url: str, *, vip_only: bool, slug: str | None) -> list[dict]:
    """Pull (id, slug, name, index_name, vip, total, indexed) for each candidate."""
    where = ["1=1"]
    params: list = []
    if slug:
        where.append("u.username = %s")
        params.append(slug)
    elif vip_only:
        where.append("u.vip = TRUE")
    sql = f"""
        SELECT u.id, u.username, u.name, u.index_name, u.vip,
               COUNT(d.url) FILTER (WHERE TRUE)::bigint              AS pg_total,
               COUNT(d.url) FILTER (WHERE d.indexed = TRUE)::bigint  AS pg_indexed
          FROM users u
          LEFT JOIN documents d ON d.user_id = u.id
         WHERE {" AND ".join(where)}
         GROUP BY u.id
         ORDER BY u.vip DESC, u.username
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "slug": r[1],
            "name": r[2],
            "index_name": r[3],
            "vip": r[4],
            "pg_total": r[5],
            "pg_indexed": r[6],
        }
        for r in rows
    ]


def force_heal(api_url: str, slug: str, user_id: int, database_url: str) -> None:
    """Pre-pipeline heal for pg_drift cases.

    The pipeline's own heal path (`_heal_broken_index`) only fires when
    the index FAILS TO LOAD — it doesn't trigger for pg_drift because
    the index loads fine; the drift is only visible by comparing PG
    against the index's metadata count. So `run_pipeline` would skip
    the rebuild entirely for those.

    For drift cases we manually do the same two steps the in-pipeline
    healer does: DELETE the index, then reset every `indexed=true` row
    in PG to `indexed=false`. The pipeline then takes the fresh-build
    path naturally.
    """
    try:
        req = urllib.request.Request(f"{api_url}/indices/{slug}", method="DELETE")
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  warn: delete index failed: HTTP {e.code}")
    except Exception as e:
        print(f"  warn: delete index failed: {e}")
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET indexed=false, updated_at=now() " "WHERE user_id = %s AND indexed = true",
                    (user_id,),
                )
    except Exception as e:
        print(f"  warn: reset indexed flags failed: {e}")


def rebuild(user: dict, database_url: str, api_url: str, vip_tag_set: set[str], verdict: str) -> bool:
    """Trigger the in-pipeline heal+rebuild for one user."""
    # pg_drift / healthy need a manual heal — the pipeline's auto-heal
    # only fires on load-time failures, not when the index loads fine.
    # broken/error cases load with bad shape and the pipeline heals
    # them itself.
    if verdict in {"pg_drift", "healthy"}:
        force_heal(api_url, user["index_name"], user["id"], database_url)

    own_tags = get_user_tags(database_url, user["id"])
    shared_tags = sorted(vip_tag_set | set(own_tags))
    try:
        run_pipeline(
            slug=user["slug"],
            name=user["name"],
            index_name=user["index_name"],
            sources_config={},  # no fetchers — we only want to reindex from PG
            user_id=user["id"],
            database_url=database_url,
            shared_tags=shared_tags,
            n_workers=1,
            vip=bool(user["vip"]),
            do_index=True,  # this script exists to reindex
        )
        return True
    except Exception as e:
        print(f"  [!] {user['slug']}: rebuild failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="report only, don't rebuild")
    ap.add_argument("--include-drift", action="store_true", help="also rebuild pg_drift indexes")
    ap.add_argument("--vip-only", action="store_true", help="only consider VIP users")
    ap.add_argument("--slug", default=None, help="restrict to one slug")
    ap.add_argument(
        "--all",
        action="store_true",
        help=(
            "rebuild every slug regardless of verdict (use when stored "
            "document text has changed and embeddings need to be regenerated "
            "from PG — no source fetchers, just the embedder)"
        ),
    )
    args = ap.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_url = os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")

    users = fetch_users(database_url, vip_only=args.vip_only, slug=args.slug)
    print(f"Scanning {len(users)} user(s) against {api_url} ...\n")

    by_verdict: dict[str, list[tuple[dict, str]]] = {}
    for u in users:
        verdict, reason = classify(api_url, u["index_name"], u["pg_total"], u["pg_indexed"])
        by_verdict.setdefault(verdict, []).append((u, reason))

    targets: list[tuple[dict, str, str]] = []
    targetable = {"broken", "error", "missing"}
    if args.include_drift:
        targetable.add("pg_drift")
    if args.all:
        # Force every non-empty user into the rebuild list. Empty
        # users (no PG docs) are pointless to embed.
        targetable.update({"healthy", "pg_drift", "broken", "error", "missing"})
    for verdict, entries in sorted(by_verdict.items()):
        marker = "→ rebuild" if verdict in targetable else "  skip"
        print(f"  [{verdict:<9}] {len(entries):>3}  {marker}")
        for u, reason in entries:
            if verdict == "empty":
                # Nothing to embed.
                continue
            if not args.all and verdict == "healthy":
                continue
            print(f"      {u['slug']:<28} pg={u['pg_indexed']}/{u['pg_total']:<6} {reason}")
            if verdict in targetable:
                targets.append((u, verdict, reason))

    if not targets:
        print("\nNothing to rebuild.")
        return 0

    print(f"\n{'DRY RUN — ' if args.dry else ''}Rebuilding {len(targets)} index(es)...\n")
    if args.dry:
        return 0

    vip_tag_set = set(get_vip_tags(database_url))
    ok = 0
    t0 = time.perf_counter()
    for i, (u, verdict, reason) in enumerate(targets, 1):
        print(f"\n{'=' * 60}\n  [{i}/{len(targets)}] {u['slug']}  ({verdict}: {reason})\n{'=' * 60}")
        if rebuild(u, database_url, api_url, vip_tag_set, verdict):
            ok += 1
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s — {ok}/{len(targets)} rebuilt successfully.")
    return 0 if ok == len(targets) else 2


if __name__ == "__main__":
    raise SystemExit(main())
