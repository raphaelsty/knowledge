"""CLI: purge documents flagged ``to_delete = TRUE``.

Soft-deletes are written by the API when a user removes a source from
their profile (e.g. drops a website from `sources.websites`). This job
walks the tombstones, removes the matching rows from each user's
ColBERT index, then deletes them from PostgreSQL.

Usage::

    make purge                  # purge every tombstone in the DB
    make purge SLUG=simon-willison      # only that user
    make purge DRY=1                    # preview without touching anything
    make purge YES=1                    # skip confirmation prompt

The two-step (index first, PG second) ordering matches what the existing
``delete_documents`` CLI does — letting docs drift in the index is
recoverable (the next ``make run`` reconciles), but PG-only deletion
without index cleanup would mean the search returns rows the API can't
hydrate.

Index cleanup needs ``ADMIN_API_KEY`` (passed as ``X-API-Key``); the
Makefile target forwards it from the environment. The PG side reads
``DATABASE_URL`` directly.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg

from sources.utils.delete_documents import (
    DEFAULT_API_BASE,
    DEFAULT_DATABASE_URL,
    _delete_index_urls,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="purge_tombstones.py",
        description="Hard-delete documents flagged to_delete=TRUE and " "remove their entries from the ColBERT index.",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Limit the purge to one personality (username). Default: every user.",
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
        help="Print the plan and exit; touch nothing.",
    )
    return p.parse_args()


def _gather(
    conn: psycopg.Connection,
    slug: str | None,
) -> list[tuple[int, str, str, list[str]]]:
    """Return ``[(user_id, username, index_name, urls)]`` for tombstoned rows."""
    sql = (
        "SELECT u.id, u.username, COALESCE(NULLIF(u.index_name, ''), u.username), d.url "
        "  FROM documents d JOIN users u ON u.id = d.user_id "
        " WHERE d.to_delete = TRUE "
    )
    params: list = []
    if slug:
        sql += " AND u.username = %s"
        params.append(slug)
    sql += " ORDER BY u.username, d.url"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    grouped: dict[int, tuple[str, str, list[str]]] = {}
    for uid, username, index_name, url in rows:
        if uid not in grouped:
            grouped[uid] = (username, index_name, [])
        grouped[uid][2].append(url)
    return [(uid, name, idx, urls) for uid, (name, idx, urls) in grouped.items()]


def main() -> None:
    args = parse_args()

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
        targets = _gather(conn, args.slug)
        if not targets:
            print("Nothing to purge — no rows with to_delete = TRUE.")
            return

        # Print a summary.
        print("Plan:")
        total = 0
        for _uid, username, index_name, urls in targets:
            print(f"  {username:<28} index='{index_name}'  {len(urls)} URLs")
            total += len(urls)
        print(f"\nTotal: {total} document(s) across {len(targets)} user(s)")
        if args.slug:
            print(f"Slug filter: {args.slug}")

        if args.dry:
            print("\n--dry: no changes applied.")
            return

        if not args.yes:
            answer = input("\nProceed? type 'yes' to confirm: ").strip().lower()
            if answer != "yes":
                print("Aborted.")
                return

        # Apply: per user, drop URLs from the index, then DELETE FROM PG.
        purged = 0
        for uid, username, index_name, urls in targets:
            ok, bad = _delete_index_urls(api_base, api_key, index_name, urls)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents " " WHERE user_id = %s AND to_delete = TRUE AND url = ANY(%s)",
                    (uid, urls),
                )
                deleted = cur.rowcount
            conn.commit()
            purged += deleted
            print(f"  ✓ {username:<28} index batches ok={ok} bad={bad}  " f"PG deleted={deleted}")

    print(f"\nDone — {purged} row(s) purged.")


if __name__ == "__main__":
    main()
