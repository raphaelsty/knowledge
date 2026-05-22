"""CLI: delete documents (and clean the matching index entries).

Usage examples (via the Makefile wrapper, but the CLI works standalone)::

    make delete SLUG=simon-willison                  # everything for one user
    make delete SOURCE=twitter                       # everything tagged 'twitter' across all users
    make delete SLUG=simon-willison SOURCE=twitter   # one user, one source
    make delete SOURCE=reddit,twitter                # multiple sources, all users

Refuses to run with neither filter — there's no good reason to wipe the
whole library and a typo shouldn't be how that happens. Add ``--yes`` to
skip the confirmation prompt; otherwise the CLI prints a summary and
asks for explicit ``yes`` before touching anything.

Strategy:
  * **SLUG only** (no SOURCE): wipe the user's whole index via
    ``DELETE /indices/{name}`` and ``DELETE FROM documents WHERE user_id=X``.
    The next ``make run`` rebuilds the index from PG cleanly thanks to the
    health-check guard in ``sources/utils/client.py``.
  * **SLUG + SOURCE** or **SOURCE only**: surgical per-URL deletes.
    Read the matching URLs from PG, send them to
    ``DELETE /indices/{name}/documents`` in batches of
    ``MAX_DELETE_BATCH_CONDITIONS`` (default 200), then
    ``DELETE FROM documents WHERE …``.

The API endpoint requires ``ADMIN_API_KEY``; the CLI reads it from the
environment and forwards it as ``X-API-Key``.
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
        prog="delete_documents.py",
        description="Delete documents (and matching index entries) for a user, a source, or the intersection of both.",
    )
    p.add_argument("--slug", default=None, help="Personality username (e.g. 'simon-willison').")
    p.add_argument(
        "--source",
        default=None,
        help="Comma-separated source keys to delete (e.g. 'twitter,reddit').",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print what would be deleted, don't touch anything.",
    )
    return p.parse_args()


def _resolve_targets(
    conn: psycopg.Connection, slug: str | None, sources: list[str] | None
) -> list[tuple[int, str, str, list[str]]]:
    """Return rows of (user_id, username, index_name, urls_to_delete).

    `urls_to_delete` is empty when we're wiping the whole user (SLUG only,
    no SOURCE) — that case takes the "delete index + truncate user docs"
    fast path.
    """
    out: list[tuple[int, str, str, list[str]]] = []
    where = []
    params: list = []
    if slug:
        where.append("u.username = %s")
        params.append(slug)
    if sources:
        # `source = ANY(%s)` lets us pass the list as one parameter.
        where.append("d.source = ANY(%s)")
        params.append(sources)
    where_sql = " AND ".join(where) if where else "TRUE"

    if slug and not sources:
        # Whole-user wipe — short path that doesn't enumerate URLs.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, index_name FROM users WHERE username = %s",
                (slug,),
            )
            row = cur.fetchone()
        if row:
            out.append((row[0], row[1], row[2] or row[1], []))
        return out

    # Source-scoped delete: enumerate exact URLs we'll remove from each
    # affected user's index.
    sql = (
        "SELECT u.id, u.username, COALESCE(NULLIF(u.index_name, ''), u.username), d.url "
        "  FROM documents d JOIN users u ON u.id = d.user_id "
        f" WHERE {where_sql} "
        " ORDER BY u.username, d.url"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    by_user: dict[int, tuple[str, str, list[str]]] = {}
    for uid, username, index_name, url in rows:
        if uid not in by_user:
            by_user[uid] = (username, index_name, [])
        by_user[uid][2].append(url)
    for uid, (username, index_name, urls) in by_user.items():
        out.append((uid, username, index_name, urls))
    return out


def _delete_index_urls(
    api_base: str,
    api_key: str | None,
    index_name: str,
    urls: list[str],
) -> tuple[int, int]:
    """Send batched DELETE /indices/{name}/documents calls.

    Returns (batches_ok, batches_failed). Errors are logged but don't
    raise — the PG row deletion still proceeds because letting docs
    drift in the index is recoverable (drift purge during the next
    `make run` cleans them up), but leaving them in PG-only would be
    silent corruption.
    """
    ok = bad = 0
    for i in range(0, len(urls), URL_BATCH):
        chunk = urls[i : i + URL_BATCH]
        placeholders = ",".join("?" for _ in chunk)
        body = {
            "condition": f"url IN ({placeholders})",
            "parameters": list(chunk),
        }
        status, text = _api_request(api_base, api_key, "DELETE", f"/indices/{index_name}/documents", body=body)
        if status in (200, 202):
            ok += 1
        else:
            bad += 1
            print(f"  ! index batch {i // URL_BATCH + 1}: HTTP {status} — {text[:200]}")
    # Server queues deletes; give the worker a moment to drain so search
    # results are consistent right after the CLI returns.
    if ok:
        time.sleep(2)
    return ok, bad


def main() -> None:
    args = parse_args()
    sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else []
    if not args.slug and not sources:
        print(
            "error: refusing to delete without a filter — pass --slug or --source",
            file=sys.stderr,
        )
        sys.exit(2)

    db_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_base = os.environ.get("API_URL", DEFAULT_API_BASE).rstrip("/")
    api_key = os.environ.get("ADMIN_API_KEY")
    if not api_key:
        print(
            "warning: ADMIN_API_KEY is not set. The API requires it for index "
            "delete endpoints — index cleanup will likely 401.",
            file=sys.stderr,
        )

    with psycopg.connect(db_url) as conn:
        targets = _resolve_targets(conn, args.slug, sources)
        if not targets:
            print("Nothing to delete (no matching documents).")
            return

        # Print a summary and confirm.
        print("Plan:")
        total_docs = 0
        for uid, username, index_name, urls in targets:
            if urls:
                print(f"  {username:<28} index='{index_name}'  {len(urls)} URLs (filtered by source)")
                total_docs += len(urls)
            else:
                # Whole-user wipe.
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM documents WHERE user_id = %s", (uid,))
                    n = cur.fetchone()[0]
                print(f"  {username:<28} index='{index_name}'  {n} docs (entire user)")
                total_docs += n
        print(f"\nTotal: {total_docs} documents across {len(targets)} user(s)")
        if sources:
            print(f"Source filter: {', '.join(sources)}")

        if args.dry:
            print("\n--dry: no changes applied.")
            return

        if not args.yes:
            answer = input("\nProceed? type 'yes' to confirm: ").strip().lower()
            if answer != "yes":
                print("Aborted.")
                return

        # Apply.
        for uid, username, index_name, urls in targets:
            if urls:
                # Partial: delete from index by URL, then PG by URL.
                ok, bad = _delete_index_urls(api_base, api_key, index_name, urls)
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM documents  WHERE user_id = %s AND source = ANY(%s) AND url = ANY(%s)",
                        (uid, sources, urls),
                    )
                    deleted = cur.rowcount
                conn.commit()
                print(f"  ✓ {username:<28} index batches ok={ok} bad={bad}  PG deleted={deleted}")
            else:
                # Whole-user wipe.
                status, text = _api_request(api_base, api_key, "DELETE", f"/indices/{index_name}")
                index_msg = "removed" if status in (200, 202, 204, 404) else f"HTTP {status} ({text[:80]})"
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE user_id = %s", (uid,))
                    deleted = cur.rowcount
                conn.commit()
                print(f"  ✓ {username:<28} index {index_msg}, PG deleted={deleted}")

    print("\nDone.")


if __name__ == "__main__":
    main()
