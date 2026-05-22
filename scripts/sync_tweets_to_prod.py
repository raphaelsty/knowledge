"""Push local twitter docs into prod PG, mapping by username (slug).

Design goals (in priority order):

1. **Never erase or overwrite prod data.** Every row insertion goes
   through ``INSERT ... ON CONFLICT (user_id, url) DO NOTHING``, so
   anything prod already has is left exactly as-is.
2. **Map users by slug, not id.** `users.id` is a local sequence
   that diverges between the dev box and Hetzner. The only stable
   identifier shared by both is `users.username` (the kebab-case
   slug). We resolve every local user to its prod id via this map
   and refuse to fabricate a prod user when one is missing.
3. **Per-user early stopping.** Before streaming inserts we compare
   local and prod twitter-doc counts. If prod already has ≥ local,
   the user is short-circuited and reported as "covered". Saves the
   per-row INSERT round-trip when there's nothing to do.
4. **Streaming, not "load both DBs into memory".** Local docs are
   fetched per-user, batched into chunks of ``BATCH`` rows, and
   pushed with ``cur.executemany``. Memory is bounded by the
   biggest single user's batch — fine even for the heaviest VIPs.
5. **Reportable.** A summary line per user (`+N inserted / S
   skipped`) and a grand total at the end, so the operator can spot
   weird outliers.

Out of scope:
  - Inserting users that exist locally but not in prod. The script
    skips them with a clear log line; creating accounts in prod
    needs a deliberate decision, not an "automatic side effect" of
    a tweet sync.
  - Pushing non-twitter docs. The filter ``source='twitter'`` keeps
    the blast radius small and the count comprehensible.

CLI:

    python scripts/sync_tweets_to_prod.py \\
        --local-url postgresql://knowledge:knowledge@localhost:5433/knowledge \\
        --prod-url  postgresql://knowledge:PASS@localhost:15433/knowledge \\
        [--dry-run] [--limit N] [--batch 500]

Both URLs default to the dev / tunnelled-prod conventions used by the
Makefile (`make sync-tweets-to-prod`). ``--dry-run`` reports what
*would* happen without writing a single row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import psycopg

DEFAULT_LOCAL_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
# 15433 is the convention used by `scripts/twitter_feed.sh` for the
# SSH tunnel into Hetzner's loopback-only PG. Set $PROD_DATABASE_URL
# from the Makefile target so the password isn't hard-coded here.
DEFAULT_PROD_URL = os.environ.get("PROD_DATABASE_URL", "postgresql://knowledge:knowledge@localhost:15433/knowledge")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts}  {msg}", flush=True)


# Columns we copy. Same set as `upsert_documents` writes, minus the
# pipeline-internal gates (`cleaned`, `tagged`, `indexed`, …) and the
# timestamps — those should fall back to their column defaults so the
# new prod row queues for re-embedding the next time the indexer runs.
_COLS = (
    "user_id",
    "url",
    "title",
    "summary",
    "date",
    "tags",
    "extra_tags",
    "source",
    "source_url",
    "linked_urls",
    "link_hosts",
)
_SELECT = (
    "SELECT url, title, summary, date, tags, extra_tags, "
    "source, source_url, linked_urls, link_hosts "
    "FROM documents "
    "WHERE user_id = %s AND source = 'twitter' AND deleted = FALSE"
)
_INSERT = (
    "INSERT INTO documents "
    f"  ({', '.join(_COLS)}) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s) "
    # ON CONFLICT DO NOTHING — never touch a row prod already has.
    "ON CONFLICT (user_id, url) DO NOTHING"
)


def _slug_to_id(conn) -> dict[str, int]:
    """Return ``{username: id}`` for every user in the connection's DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT username, id FROM users")
        return dict(cur.fetchall())


def _twitter_doc_count(conn, user_id: int) -> int:
    """How many live twitter docs does ``user_id`` have right now?"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM documents WHERE user_id = %s AND source = 'twitter' AND deleted = FALSE",
            (user_id,),
        )
        return int(cur.fetchone()[0])


def _local_users_with_twitter(conn) -> list[tuple[int, str, int]]:
    """Return ``[(local_id, slug, doc_count)]`` for every local user
    that has at least one twitter doc. Ordered by slug for
    deterministic resumption between restarts."""
    sql = (
        "SELECT u.id, u.username, COUNT(d.*) "
        "  FROM users u "
        "  JOIN documents d ON d.user_id = u.id "
        " WHERE d.source = 'twitter' AND d.deleted = FALSE "
        " GROUP BY u.id, u.username "
        " ORDER BY u.username"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        return [(uid, slug, int(n)) for uid, slug, n in cur.fetchall()]


def _stream_user(local_conn, prod_conn, local_uid: int, prod_uid: int, batch: int, dry_run: bool) -> tuple[int, int]:
    """Copy one user's twitter docs from local to prod.

    Returns ``(inserted, scanned)``. ``inserted`` is what
    ``cur.rowcount`` reports across all batches — the number of new
    prod rows actually created. ``scanned`` is the total local rows
    we read, regardless of insert outcome.
    """
    scanned = 0
    inserted = 0
    rows_buf: list[tuple] = []

    # Regular client-side cursor: the heaviest single VIP has a few
    # thousand tweet rows (a few MB at most), so materialising is
    # cheaper than the autocommit-vs-server-cursor dance.
    with local_conn.cursor() as lcur:
        lcur.execute(_SELECT, (local_uid,))
        for url, title, summary, dt, tags, extra_tags, source, source_url, linked_urls, link_hosts in lcur.fetchall():
            scanned += 1
            # `linked_urls` is JSONB on the read side → psycopg gave us a
            # native dict/list. Re-serialise so the INSERT can cast it
            # via ``%s::jsonb`` (matches `upsert_documents`).
            lu_str = json.dumps(linked_urls) if linked_urls is not None else "[]"
            rows_buf.append(
                (
                    prod_uid,
                    url,
                    title or "",
                    summary or "",
                    dt,
                    list(tags or []),
                    list(extra_tags or []),
                    source or "twitter",
                    source_url,
                    lu_str,
                    list(link_hosts or []),
                )
            )
            if len(rows_buf) >= batch:
                inserted += _flush_batch(prod_conn, rows_buf, dry_run)
                rows_buf.clear()
    if rows_buf:
        inserted += _flush_batch(prod_conn, rows_buf, dry_run)
    return inserted, scanned


def _flush_batch(prod_conn, rows: list[tuple], dry_run: bool) -> int:
    """Push one chunk of rows, return how many actually inserted."""
    if dry_run:
        # We can't measure the would-insert count without round-trip-
        # ping, so report the batch size and trust the ON CONFLICT
        # estimate. This is fine: --dry-run is for sanity, not
        # accounting.
        return len(rows)
    with prod_conn.cursor() as pcur:
        pcur.executemany(_INSERT, rows)
        # `cur.rowcount` after `executemany` is the sum of per-row
        # rowcounts; on `ON CONFLICT DO NOTHING` that's the count of
        # actually-new rows — exactly what we want to report.
        return int(pcur.rowcount or 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sync_tweets_to_prod", description=__doc__)
    p.add_argument("--local-url", default=DEFAULT_LOCAL_URL)
    p.add_argument("--prod-url", default=DEFAULT_PROD_URL)
    p.add_argument(
        "--batch",
        type=int,
        default=500,
        help="Rows per INSERT. 500 fits comfortably in PG's default "
        "max_locks_per_transaction. Lower if you see lock storms.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after syncing this many users (0 = all). Useful for a smoke test before the long run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the per-user verdict without writing to prod.",
    )
    args = p.parse_args(argv)

    _log(f"local: {args.local_url}")
    _log(f"prod : {args.prod_url.split('@')[-1]}  (password hidden)")
    if args.dry_run:
        _log("DRY-RUN — no inserts will be issued")

    with psycopg.connect(args.local_url) as local_conn, psycopg.connect(args.prod_url) as prod_conn:
        local_conn.autocommit = True
        # Each batch is its own implicit transaction; safer than
        # holding one giant transaction open across 200k inserts
        # (would block VACUUM on the live prod box).
        prod_conn.autocommit = True

        prod_slug_to_id = _slug_to_id(prod_conn)
        _log(f"prod has {len(prod_slug_to_id)} users")

        local_users = _local_users_with_twitter(local_conn)
        _log(f"local has {len(local_users)} users with twitter docs")

        t0 = time.perf_counter()
        total_inserted = 0
        total_scanned = 0
        missing_in_prod: list[str] = []
        covered: list[str] = []
        synced: list[tuple[str, int, int]] = []  # (slug, inserted, scanned)

        for i, (local_uid, slug, local_n) in enumerate(local_users, start=1):
            if args.limit and i > args.limit:
                _log(f"--limit {args.limit} reached, stopping")
                break

            prod_uid = prod_slug_to_id.get(slug)
            if prod_uid is None:
                _log(f"[{i:>3}/{len(local_users)}] {slug}  ! not in prod, skipping")
                missing_in_prod.append(slug)
                continue

            prod_n = _twitter_doc_count(prod_conn, prod_uid)
            # Early stop: prod already has at least as many twitter
            # docs as local → assume it's caught up. Cheap (a single
            # COUNT on the partial index used by `_twitter_doc_count`),
            # and the dominant case for the second pass of the script.
            if prod_n >= local_n:
                _log(f"[{i:>3}/{len(local_users)}] {slug}  covered (prod={prod_n} ≥ local={local_n})")
                covered.append(slug)
                continue

            _log(f"[{i:>3}/{len(local_users)}] {slug}  local={local_n} prod={prod_n} → syncing")
            inserted, scanned = _stream_user(local_conn, prod_conn, local_uid, prod_uid, args.batch, args.dry_run)
            verdict = "would insert" if args.dry_run else "inserted"
            _log(f"        {verdict} {inserted:>5d} of {scanned:>5d} local rows ({scanned - inserted} already in prod)")
            synced.append((slug, inserted, scanned))
            total_inserted += inserted
            total_scanned += scanned

        dur = time.perf_counter() - t0
        _log("")
        _log("=" * 60)
        _log(f"done in {int(dur)}s")
        _log(f"  users synced     : {len(synced)}")
        _log(f"  users covered    : {len(covered)}")
        _log(f"  users not in prod: {len(missing_in_prod)}")
        if missing_in_prod:
            _log("  missing slugs    : " + ", ".join(missing_in_prod[:10]) + ("…" if len(missing_in_prod) > 10 else ""))
        _log(f"  rows scanned     : {total_scanned}")
        _log(f"  rows inserted    : {total_inserted}  ({'dry-run' if args.dry_run else 'live'})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
