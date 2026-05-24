"""Hourly refresher for `feed_snapshot` + every VIP's `personal_snapshot`.

A tiny loop that, once an hour:
  1. Rebuilds the global `feed_snapshot` table.
  2. Iterates every VIP user and rebuilds their per-user
     `personal_snapshot` rows.

Both passes share the same anchor-collapse + sci/recency/link-bonus
shape so the personal page and the global feed read consistently.
The per-VIP refresh is bounded (the largest VIP has ~10 k docs and
the CTE runs in well under a second) so the whole cycle still fits
inside a single hour with margin to spare on the smallest CPU caps.

Lives in its own Dokploy compose service (`knowledge-feed-snapshot`)
so its CPU budget is bounded independently of the heavier daemons —
the global refresh is a 2-5 s CTE; the per-VIP pass is N × <1 s
once an hour, otherwise the process sleeps.

On startup the daemon does one refresh immediately so a fresh
deploy doesn't have to wait an hour to populate the tables.

Failures log + back off:
  * 60 s retry on the first failure of the global refresh
  * 5 min retry on consecutive failures
Per-VIP personal_snapshot failures are logged but don't abort the
sweep — one bad user shouldn't starve the rest. The API's snapshot
freshness check (>3 h stale → live-query fallback) covers the worst
case where the daemon stays down.
"""

from __future__ import annotations

import logging
import os
import time

import psycopg

from sources.sql.feed_snapshot import (
    create_feed_snapshot_table,
    feed_snapshot_age_seconds,
    refresh_feed_snapshot,
)
from sources.sql.personal_snapshot import (
    create_personal_snapshot_table,
    refresh_personal_snapshot,
)

# Loop cadence + back-off intervals.
REFRESH_INTERVAL_SECS = 60 * 60  # 1 h
RETRY_BACKOFF_SECS = 60  # 1 min after a single failure
EXTENDED_BACKOFF_SECS = 5 * 60  # 5 min after consecutive failures


def _log() -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("knowledge.feed_snapshot")


def _vip_user_ids(database_url: str) -> list[tuple[int, str]]:
    """Return every VIP user that has at least one non-deleted doc,
    as `(id, username)` tuples ordered by username for stable
    iteration. The HAVING clause skips brand-new VIPs with no docs
    yet — refreshing an empty corpus would just write zero rows."""
    sql = """
        SELECT u.id, u.username
          FROM users u
          JOIN documents d ON d.user_id = u.id
         WHERE u.vip = TRUE
           AND d.deleted = FALSE
         GROUP BY u.id, u.username
         ORDER BY u.username
    """
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def _refresh_all_personal_snapshots(database_url: str, log: logging.Logger) -> None:
    """Sweep every VIP, rebuilding their personal_snapshot rows.

    Best-effort per user — one failed VIP doesn't abort the sweep. We
    use a fresh PG connection per call (inside `refresh_personal_snapshot`)
    so a borked transaction can't poison the next user.
    """
    vips = _vip_user_ids(database_url)
    log.info("personal_snapshot.sweep.start vip_count=%d", len(vips))
    t0 = time.monotonic()
    ok = 0
    fail = 0
    total_rows = 0
    for uid, slug in vips:
        try:
            n = refresh_personal_snapshot(database_url, uid)
            total_rows += n
            ok += 1
        except Exception as exc:  # noqa: BLE001 — one user can't kill the sweep
            fail += 1
            log.warning("personal_snapshot.refresh.failed user=%s err=%s", slug, exc)
    log.info(
        "personal_snapshot.sweep.complete ok=%d fail=%d rows=%d elapsed=%.2fs",
        ok,
        fail,
        total_rows,
        time.monotonic() - t0,
    )
    # VACUUM ANALYZE once after the full sweep. Each per-user DELETE
    # + INSERT generates dead tuples; over 450 VIPs an hourly sweep
    # produces ~50k dead tuples even with autovac, slowly bloating
    # the on-disk size. One VACUUM per sweep keeps the table close
    # to its working size (~120k live rows) instead of drifting.
    try:
        with psycopg.connect(database_url, autocommit=True) as conn:
            # PARALLEL 0 keeps the worker count to 1 so the VACUUM
            # works on hosts with the default 64 MB /dev/shm (parallel
            # vacuum fails there with "no space left on device").
            conn.execute("VACUUM (ANALYZE, PARALLEL 0) personal_snapshot")
    except Exception as exc:  # noqa: BLE001
        log.warning("personal_snapshot.vacuum.failed err=%s", exc)


def main() -> None:
    log = _log()
    database_url = os.environ["DATABASE_URL"]

    # Schema bootstrap. Idempotent — the API container also calls
    # this at boot via include_str! migrations, but running it here
    # too lets the daemon stand alone in dev (no API container
    # required to populate the tables).
    log.info("feed_snapshot.bootstrap.start")
    try:
        create_feed_snapshot_table(database_url)
        create_personal_snapshot_table(database_url)
    except Exception as exc:
        # Don't crash the daemon — the API may have already created
        # the tables. Log and keep going.
        log.warning("feed_snapshot.bootstrap.failed err=%s", exc)
    else:
        log.info("feed_snapshot.bootstrap.complete")

    consecutive_failures = 0
    while True:
        start = time.monotonic()
        try:
            rows = refresh_feed_snapshot(database_url)
            elapsed = time.monotonic() - start
            age = feed_snapshot_age_seconds(database_url)
            log.info(
                "feed_snapshot.refresh.complete rows=%d elapsed=%.2fs age_after=%s",
                rows,
                elapsed,
                age,
            )
            # Once the global feed is refreshed, sweep every VIP's
            # personal_snapshot. Best-effort — a failure inside the
            # sweep is logged but doesn't restart the back-off (which
            # tracks the global refresh, the load-bearing piece).
            try:
                _refresh_all_personal_snapshots(database_url, log)
            except Exception as exc:  # noqa: BLE001
                log.warning("personal_snapshot.sweep.aborted err=%s", exc)
            consecutive_failures = 0
            sleep_secs = REFRESH_INTERVAL_SECS
        except Exception as exc:  # noqa: BLE001 — daemon stays alive
            consecutive_failures += 1
            log.error(
                "feed_snapshot.refresh.failed attempt=%d err=%s",
                consecutive_failures,
                exc,
            )
            # First failure: short retry — could be transient (PG
            # restart, network blip). Consecutive failures: back off
            # so we're not spinning on a structural problem.
            sleep_secs = RETRY_BACKOFF_SECS if consecutive_failures == 1 else EXTENDED_BACKOFF_SECS

        # Subtract elapsed so the schedule stays on the hour boundary
        # rather than drifting forward by `elapsed` each cycle.
        remaining = max(1, sleep_secs - int(time.monotonic() - start))
        time.sleep(remaining)


if __name__ == "__main__":
    main()
