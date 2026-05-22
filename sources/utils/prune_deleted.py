"""CLI: evict soft-deleted documents from the ColBERT search index.

Soft-delete lives in PG (`documents.deleted = TRUE`) so a re-running
pipeline never resurrects the row. The search index, however, can still
hold the doc's vectors until something purges it. This script bridges
that gap: walk every user whose library has soft-deleted rows, call
``DELETE /indices/{index_name}/documents`` with the matching URLs in
batches, and we're back in sync.

Designed to run periodically (cron, nightly, etc.). Idempotent — if the
index batch already removed an URL on a prior run, the second call is a
no-op.

Usage::

    make prune-deleted                # default: dry run not enabled, applies removals
    make prune-deleted DRY=1          # plan-only, no API calls
    make prune-deleted SLUG=alice     # restrict to one user

The API endpoint needs ``ADMIN_API_KEY``; the script reads it from the
environment and forwards as ``X-API-Key``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_BASE = "http://localhost:8080"
URL_BATCH = 200  # must match MAX_DELETE_BATCH_CONDITIONS in the Rust API


def _api_request(
    api_base: str,
    api_key: str | None,
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = 30.0,
) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{api_base}{path}",
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace") if e.fp else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prune_deleted.py",
        description="Remove soft-deleted documents from the ColBERT index.",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Restrict to a single user (by username). Defaults to all users.",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Plan only — list what would be removed, don't call the API.",
    )
    return p.parse_args()


def _gather(conn: psycopg.Connection, slug: str | None) -> list[tuple[str, str, list[str]]]:
    """Return [(username, index_name, urls)] for every user with soft-deleted rows."""
    where = ["d.deleted = TRUE"]
    params: list = []
    if slug:
        where.append("u.username = %s")
        params.append(slug)
    sql = (
        "SELECT u.username, COALESCE(NULLIF(u.index_name, ''), u.username) AS idx, d.url "
        "  FROM documents d JOIN users u ON u.id = d.user_id "
        " WHERE " + " AND ".join(where) + " ORDER BY u.username, d.url"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    by_user: dict[str, tuple[str, list[str]]] = {}
    for username, idx, url in rows:
        if username not in by_user:
            by_user[username] = (idx, [])
        by_user[username][1].append(url)
    return [(u, idx, urls) for u, (idx, urls) in by_user.items()]


def _delete_index_urls(api_base: str, api_key: str | None, index_name: str, urls: list[str]) -> tuple[int, int]:
    """Batched DELETE /indices/{name}/documents. Returns (ok, bad) batch counts."""
    ok = bad = 0
    for i in range(0, len(urls), URL_BATCH):
        chunk = urls[i : i + URL_BATCH]
        placeholders = ",".join("?" for _ in chunk)
        body = {"condition": f"url IN ({placeholders})", "parameters": list(chunk)}
        status, text = _api_request(api_base, api_key, "DELETE", f"/indices/{index_name}/documents", body=body)
        if status in (200, 202):
            ok += 1
        else:
            bad += 1
            print(f"  ! batch {i // URL_BATCH + 1}: HTTP {status} — {text[:200]}")
    if ok:
        # Let the server worker drain queued deletes so the next search
        # is consistent with what the script reported.
        time.sleep(1.5)
    return ok, bad


def main() -> None:
    args = parse_args()
    db_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_base = os.environ.get("API_URL", DEFAULT_API_BASE).rstrip("/")
    api_key = os.environ.get("ADMIN_API_KEY")
    if not api_key and not args.dry:
        print(
            "warning: ADMIN_API_KEY is not set — the index DELETE endpoint will 401.",
            file=sys.stderr,
        )

    with psycopg.connect(db_url) as conn:
        targets = _gather(conn, args.slug)
        if not targets:
            print("Nothing to prune — no soft-deleted rows.")
            return

        total = sum(len(urls) for _, _, urls in targets)
        print(f"Plan: prune {total} doc(s) across {len(targets)} user(s)")
        for username, idx, urls in targets:
            print(f"  {username:<28} index='{idx}'  {len(urls)} URLs")

        if args.dry:
            print("\n--dry: no API calls made.")
            return

        for username, idx, urls in targets:
            ok, bad = _delete_index_urls(api_base, api_key, idx, urls)
            print(f"  ✓ {username:<28} index batches ok={ok} bad={bad}")

    print("\nDone.")


if __name__ == "__main__":
    main()
