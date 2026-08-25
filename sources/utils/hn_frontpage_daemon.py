"""Daily HackerNews front-page → per-user picks daemon.

Why this exists
---------------
The HN injection feature shipped with its logic in `scripts/`, driven
by `make hn-frontpage`, and nothing ever ran it. `scripts/` isn't even
copied into `Dockerfile.daemons`, so no prod container could have
invoked it. The result: exactly one run row ever written (2026-05-14,
by hand, with `--slug`), carrying picks for a single user. Every other
user's feed has shown zero HN cards since the feature landed.

This module is the missing piece — the same shape as the other
daemons (`feed_snapshot_daemon`, `clean_daemon`, `categorize_daemon`)
so it deploys as one more Dokploy compose service instead of a cron
entry nobody remembers to add.

The scoring lives in `sources.hackernews.picks`, shared with the debug
CLI so the thing running in prod is the thing you can reproduce
locally. See that module for why picks are z-scored per article rather
than ranked by the raw ColBERT mean.

Cadence
-------
Daily by default, immediately on startup so a deploy doesn't wait a
day to populate the feed. `HN_REFRESH_INTERVAL_SECS` overrides it.
Each run replaces the previous run's picks; the feed reads only
`MAX(hn_frontpage_runs.id)`, so there's nothing to clean up.

Failures log and back off — 5 min on the first, an hour on repeats.
A failed run leaves the previous run intact rather than publishing an
empty one (see `refresh_picks` for the ordering that guarantees this),
so the worst case is a stale front page, not a feed with no HN cards.
"""

from __future__ import annotations

import logging
import os
import time

from sources.hackernews.picks import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP,
    DEFAULT_TOP_PER_USER,
    refresh_picks,
)

REFRESH_INTERVAL_SECS = int(os.environ.get("HN_REFRESH_INTERVAL_SECS", 24 * 60 * 60))
RETRY_BACKOFF_SECS = 5 * 60
EXTENDED_BACKOFF_SECS = 60 * 60


def _log() -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("knowledge.hn_frontpage")


def main() -> None:
    log = _log()
    database_url = os.environ["DATABASE_URL"]
    api_url = os.environ.get("API_URL", "http://knowledge-api:8080")

    top = int(os.environ.get("HN_TOP", DEFAULT_TOP))
    top_per_user = int(os.environ.get("HN_TOP_PER_USER", DEFAULT_TOP_PER_USER))
    threshold = float(os.environ.get("HN_THRESHOLD", DEFAULT_THRESHOLD))

    log.info(
        "hn_frontpage.daemon.start interval=%ds top=%d top_per_user=%d threshold=%.2f api=%s",
        REFRESH_INTERVAL_SECS,
        top,
        top_per_user,
        threshold,
        api_url,
    )

    consecutive_failures = 0
    while True:
        start = time.monotonic()
        try:
            stats = refresh_picks(
                database_url,
                api_url,
                top=top,
                top_per_user=top_per_user,
                threshold=threshold,
                log=log.info,
            )
            log.info(
                "hn_frontpage.refresh.complete run=%s users=%d picks=%d elapsed=%.1fs",
                stats.run_id,
                stats.users_with_picks,
                stats.total_picks,
                time.monotonic() - start,
            )
            consecutive_failures = 0
            sleep_secs = REFRESH_INTERVAL_SECS
        except Exception as exc:  # noqa: BLE001 — daemon stays alive
            consecutive_failures += 1
            log.error(
                "hn_frontpage.refresh.failed attempt=%d err=%s",
                consecutive_failures,
                exc,
            )
            sleep_secs = RETRY_BACKOFF_SECS if consecutive_failures == 1 else EXTENDED_BACKOFF_SECS

        # Subtract elapsed so the schedule doesn't drift forward by the
        # run duration every cycle.
        remaining = max(1, sleep_secs - int(time.monotonic() - start))
        time.sleep(remaining)


if __name__ == "__main__":
    main()
