#!/usr/bin/env python3
"""Scan per-user search indices for health and persist verdicts to PG.

For each target user we:
  1. GET /indices/{name}                — fetch shape (num_documents,
     num_embeddings, metadata_count, avg_doclen).
  2. SELECT counts from `documents`     — total rows + rows flagged
     `indexed = TRUE`. The latter is the API's expected num_documents.
  3. Classify the verdict (see status enum below) and INSERT one
     row into `index_health_checks` so we keep history across runs.

Run modes:
  - default            — sweep all users in staleness order (oldest
                         last-check first; never-checked at the front;
                         VIPs prioritized within each tier).
  - --slug X           — check just one user.
  - --vip-only         — restrict to VIPs.
  - --limit N          — stop after N users (handy for periodic sweeps
                         that touch ~10 users per cron tick).

Status enum:
  healthy        — every check passed.
  missing        — API 404 on GET /indices/{name}.
  broken         — num_documents > 0 BUT num_embeddings == 0
                   (the corruption mode we hit on Amelie).
  meta_mismatch  — num_documents != metadata_count.
  pg_drift       — abs(pg_indexed_docs - num_documents) is more than
                   max(5, 5% of pg_indexed_docs). Smaller drift is
                   normal (in-flight indexing, async deletes).
  error          — request failed in a way we couldn't classify;
                   `error` carries the message.

Usage::

    DATABASE_URL=postgresql://... API_URL=http://localhost:8080 \
      uv run python scripts/check_indexes.py [--slug NAME] [--vip-only] [--limit N]

Or via the Makefile (sets envs + sane defaults)::

    make index-check                        # sweep everyone, oldest first
    make index-check SLUG=max-halford       # one user
    make index-check VIP=1 LIMIT=20         # next 20 stale VIPs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

# So `uv run python scripts/...` can find the package without -m.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg  # noqa: E402

from sources.sql import (  # noqa: E402
    create_index_health_checks_table,
    record_index_check,
    users_by_check_priority,
)

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_URL = "http://localhost:8080"
# Drift tolerance: docs present in PG but not yet pushed to the index
# (or vice-versa) up to this magnitude is normal — pipelines write
# async, deletes drain on a worker, etc. Anything above flips
# pg_drift.
DRIFT_TOLERANCE_ABS = 5
DRIFT_TOLERANCE_PCT = 0.05


def _api_index_info(api_base: str, index_name: str, timeout: int = 10) -> tuple[str | None, dict | None, str | None]:
    """Return (status, body_dict, raw_error).

    status is one of: 'ok', 'missing', 'error'.
    """
    try:
        with urllib.request.urlopen(f"{api_base}/indices/{index_name}", timeout=timeout) as resp:
            if resp.status != 200:
                return "error", None, f"HTTP {resp.status}"
            return "ok", json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing", None, None
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        return "error", None, f"HTTP {e.code} {body[:200]}"
    except Exception as exc:
        return "error", None, str(exc)


def _pg_doc_counts(database_url: str, user_id: int) -> tuple[int, int]:
    """Return (total_docs, indexed_docs) for `user_id` from PG."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE indexed = TRUE) " "  FROM documents WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    return (int(row[0]), int(row[1])) if row else (0, 0)


def _classify(
    api_status: str,
    api_body: dict | None,
    pg_total: int,
    pg_indexed: int,
) -> tuple[str, dict]:
    """Return (verdict, details_dict). `details` carries diagnostic numbers."""
    if api_status == "missing":
        details = {"reason": "API 404 — index file does not exist"}
        if pg_indexed > 0:
            details["pg_orphan_indexed_count"] = pg_indexed
        return "missing", details

    if api_status != "ok" or not api_body:
        return "error", {"reason": "API call failed"}

    n_docs = int(api_body.get("num_documents") or 0)
    n_emb = int(api_body.get("num_embeddings") or 0)
    n_meta = int(api_body.get("metadata_count") or 0)

    # Broken — index file exists, claims docs, but no embeddings on disk.
    if n_docs > 0 and n_emb == 0:
        return "broken", {"reason": "num_documents > 0 but num_embeddings == 0"}

    # Metadata mismatch — every doc should have a metadata row.
    if n_docs != n_meta:
        return "meta_mismatch", {
            "reason": "num_documents != metadata_count",
            "delta": n_docs - n_meta,
        }

    # PG ↔ API drift: how many docs PG thinks are indexed vs how many
    # the API actually has. Tolerance: 5 docs absolute or 5% of the
    # PG count, whichever is larger.
    drift = abs(pg_indexed - n_docs)
    threshold = max(DRIFT_TOLERANCE_ABS, int(pg_indexed * DRIFT_TOLERANCE_PCT))
    if drift > threshold:
        return "pg_drift", {
            "reason": "PG indexed-count vs API doc-count drift exceeds tolerance",
            "pg_indexed": pg_indexed,
            "api_documents": n_docs,
            "drift": drift,
            "threshold": threshold,
        }

    return "healthy", {"drift": drift, "threshold": threshold}


def _fmt_status(status: str) -> str:
    icons = {"healthy": "✓", "missing": "∅", "broken": "✗", "meta_mismatch": "⚠", "pg_drift": "≠", "error": "!"}
    return f"{icons.get(status, '?')} {status}"


def check_one(database_url: str, api_base: str, user: dict) -> dict:
    """Run one check + record it. Returns the verdict row as a dict."""
    user_id = int(user["id"])
    index_name = user["index_name"]

    api_status, api_body, api_error = _api_index_info(api_base, index_name)
    pg_total, pg_indexed = _pg_doc_counts(database_url, user_id)

    verdict, details = _classify(api_status, api_body, pg_total, pg_indexed)
    if api_status == "error":
        # Promote the raw error into the verdict's error field.
        verdict = "error"
        details["api_error"] = api_error

    record_index_check(
        database_url,
        user_id=user_id,
        index_name=index_name,
        status=verdict,
        num_documents=int(api_body["num_documents"])
        if api_body and api_body.get("num_documents") is not None
        else None,
        num_embeddings=int(api_body["num_embeddings"])
        if api_body and api_body.get("num_embeddings") is not None
        else None,
        metadata_count=int(api_body["metadata_count"])
        if api_body and api_body.get("metadata_count") is not None
        else None,
        avg_doclen=float(api_body["avg_doclen"]) if api_body and api_body.get("avg_doclen") is not None else None,
        pg_total_docs=pg_total,
        pg_indexed_docs=pg_indexed,
        details=details,
        error=api_error if api_status == "error" else None,
    )

    return {
        "slug": user["slug"],
        "index_name": index_name,
        "status": verdict,
        "vip": user.get("vip"),
        "pg_total": pg_total,
        "pg_indexed": pg_indexed,
        "api_documents": (api_body or {}).get("num_documents"),
        "api_embeddings": (api_body or {}).get("num_embeddings"),
        "api_metadata": (api_body or {}).get("metadata_count"),
        "details": details,
    }


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="check_indexes",
        description="Scan per-user search indices and record health verdicts to PG.",
    )
    p.add_argument("--slug", default=None, help="Check a single personality slug.")
    p.add_argument("--vip-only", action="store_true", help="Restrict the sweep to VIPs.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N users (default: no limit when scanning everyone).",
    )
    args = p.parse_args(argv if argv is None else list(argv))

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_base = os.environ.get("API_URL", DEFAULT_API_URL)

    create_index_health_checks_table(database_url)

    if args.slug:
        # Single-user mode: still pull through the priority helper so
        # we get the same row shape (id, name, vip, last_check_at).
        users = [u for u in users_by_check_priority(database_url) if u["slug"] == args.slug]
        if not users:
            print(f"Error: personality '{args.slug}' not found")
            return 1
    else:
        users = users_by_check_priority(database_url, vip_only=args.vip_only, limit=args.limit)

    if not users:
        print("No users to check.")
        return 0

    label = "VIP only" if args.vip_only else "all"
    print(f"Checking {len(users)} user(s) [{label}], oldest-checked first.\n")

    by_status: dict[str, int] = {}
    t0 = time.perf_counter()
    for u in users:
        last = u["last_check_at"]
        last_str = last.isoformat(timespec="minutes") if last else "never"
        result = check_one(database_url, api_base, u)
        by_status[result["status"]] = by_status.get(result["status"], 0) + 1

        tag = "★" if u["vip"] else " "
        # Compact one-line summary; details vary per status.
        api_docs = result["api_documents"]
        api_docs_str = f"api={api_docs:>5}" if api_docs is not None else "api=  -  "
        print(
            f"  {tag} {u['slug']:<28}  {_fmt_status(result['status']):<18}  "
            f"pg={result['pg_indexed']:>5}/{result['pg_total']:<5}  {api_docs_str}  "
            f"(prev: {last_str})"
        )
        if result["status"] != "healthy":
            for k, v in result["details"].items():
                print(f"        ↳ {k}: {v}")

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s — {len(users)} checked: "
        + ", ".join(f"{n} {st}" for st, n in sorted(by_status.items(), key=lambda kv: -kv[1]))
    )
    # Non-zero exit when anything was unhealthy, so the makefile target
    # surfaces a problem to cron / CI.
    return 0 if all(st == "healthy" for st in by_status) else 2


if __name__ == "__main__":
    raise SystemExit(main())
