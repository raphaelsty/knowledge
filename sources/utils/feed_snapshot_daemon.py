"""Hourly refresher for the `feed_snapshot` table.

A tiny loop that calls `refresh_feed_snapshot` once an hour. Lives
in its own Dokploy compose service (`knowledge-feed-snapshot`) so
its CPU budget is bounded independently of the heavier daemons —
the refresh is a 2-5 s CTE once per hour, otherwise the process
sleeps.

On startup the daemon does one refresh immediately so a fresh
deploy doesn't have to wait an hour to populate the table.

Failures log + back off:
  * 60 s retry on the first failure
  * 5 min retry on consecutive failures
The API's snapshot freshness check (>3 h stale → live-query
fallback) covers the worst case where the daemon stays down.
"""

from __future__ import annotations

import logging
import os
import time

from sources.sql.feed_snapshot import (
    create_feed_snapshot_table,
    feed_snapshot_age_seconds,
    refresh_feed_snapshot,
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


def main() -> None:
    log = _log()
    database_url = os.environ["DATABASE_URL"]

    # Schema bootstrap. Idempotent — the API container also calls
    # this at boot via include_str! migrations, but running it here
    # too lets the daemon stand alone in dev (no API container
    # required to populate the table).
    log.info("feed_snapshot.bootstrap.start")
    try:
        create_feed_snapshot_table(database_url)
    except Exception as exc:
        # Don't crash the daemon — the API may have already created
        # the table. Log and keep going.
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
