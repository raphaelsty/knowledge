"""Indexer daemon — keeps the single `__all__` search index current.

Architecture
------------
There is ONE ColBERT index, `__all__`. The fetch pipeline (`run.py`)
writes documents to Postgres with `indexed_all = FALSE`; this daemon is
the sole writer of the index. Every sweep it calls
`build_all_index.sync_all_index`, which:

  1. ensures `__all__` LOADS — recreating it empty + resetting the
     `indexed_all` flags ONLY when it's structurally broken (404/5xx);
  2. removes soft-deleted docs still in the index;
  3. streams not-yet-synced docs (newest first) into the live index and
     flips `indexed_all = TRUE`.

The `indexed_all` flag is the resumable cursor: a sync killed mid-way
(deploy / OOM / crash) resumes exactly where it stopped, so `__all__`
always converges to the full corpus. A from-scratch rebuild happens
ONLY on a structural break — a merely partial/behind index is topped up
incrementally, never dropped.

There are no per-personality indices any more; search serves everything
from `__all__` (the frontend pre-filters by `owner` for personality
pages). The daemon does no per-user work, so the encoder is dedicated to
the `__all__` sync.

Coordination
~~~~~~~~~~~~
A break is detected within ~`ALL_CHECK_INTERVAL_SECS` (the sync's health
GET). Multiple daemon instances are harmless: pushes are idempotent for
our purposes (the staging/recreate path rebuilds fresh) and the flag
cursor keeps them from re-doing each other's work.

CLI
~~~
::

    python -m sources.indexer_daemon                   # loop forever
    python -m sources.indexer_daemon --once            # one sync, exit
    python -m sources.indexer_daemon --dry             # no writes, exit
    python -m sources.indexer_daemon --sleep 5         # loop wake-up gap
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime

from sources.utils.index_health import DEFAULT_API_URL, DEFAULT_DATABASE_URL

# How often to run the incremental `__all__` sync (which also does the
# cheap health GET that detects a break). Short, so a broken index is
# spotted within seconds and new docs flow in promptly.
ALL_CHECK_INTERVAL_SECS = 20.0
# Cap docs synced per sweep so one pass stays bounded and the loop keeps
# re-checking health frequently. The `indexed_all` flag makes the sync
# resumable, so a large backlog just takes several sweeps to drain.
ALL_SYNC_MAX_DOCS = 6000


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts}  {msg}", flush=True)


_stop_requested = False


def _install_signal_handlers() -> None:
    """Cooperative SIGINT/SIGTERM — finish the in-flight sweep, then exit."""

    def _handler(signum, _frame):
        global _stop_requested
        if _stop_requested:
            _log(f"signal {signum} received twice, aborting now")
            sys.exit(130)
        _stop_requested = True
        _log(f"signal {signum} received — finishing current sweep then exiting")

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            pass


def maybe_sync_all_index(database_url: str, api_url: str, state: dict) -> None:
    """Keep the single `__all__` index current — incrementally.

    Runs at the top of each sweep, at most once per
    ``ALL_CHECK_INTERVAL_SECS``. One pass:

      1. ensures `__all__` LOADS — recreating it empty + resetting the
         `indexed_all` flags ONLY when it's structurally broken (404/5xx);
      2. removes soft-deleted docs still in it;
      3. streams up to ``ALL_SYNC_MAX_DOCS`` not-yet-synced docs (newest
         first) into the live index, flipping `indexed_all = TRUE`.

    The flag is the resumable cursor, so a backlog drains over several
    sweeps and a sync killed mid-way resumes where it stopped. A full
    from-scratch rebuild happens only via step 1's recreate-on-break.

    Best-effort: any failure logs and returns so the daemon keeps going.
    """
    now = time.monotonic()
    if state["checked_at"] and (now - state["checked_at"]) < ALL_CHECK_INTERVAL_SECS:
        return
    state["checked_at"] = now

    try:
        from sources.utils.build_all_index import sync_all_index
    except Exception as exc:  # noqa: BLE001
        _log(f"  __all__ sync: import failed: {exc!r}")
        return

    t0 = time.perf_counter()
    try:
        summary = sync_all_index(database_url, api_url, os.environ.get("ADMIN_API_KEY"), max_docs=ALL_SYNC_MAX_DOCS)
    except Exception as exc:  # noqa: BLE001
        _log(f"  [!] __all__ sync failed after {int(time.perf_counter() - t0)}s: {exc!r}")
        return

    if summary["rebuilt"] or summary["added"] or summary["removed"]:
        _log(
            f"  __all__ sync: rebuilt={summary['rebuilt']} +{summary['added']:,} "
            f"-{summary['removed']:,} in {int(time.perf_counter() - t0)}s"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="indexer_daemon",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get("API_URL", DEFAULT_API_URL),
        help="ColBERT API base. Default $API_URL or http://localhost:8080.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds between loop wake-ups. The sync self-gates on "
        "ALL_CHECK_INTERVAL_SECS, so these wake-ups are cheap no-ops "
        "when there's nothing to do.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run one sync pass and exit. Useful as a cron tick.",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Exit without writing anything (no sync).",
    )
    args = p.parse_args(argv)

    _install_signal_handlers()
    _log("indexer-daemon starting")
    _log(f"  database_url={args.database_url}")
    _log(f"  api_url     ={args.api_url}")
    _log(f"  sleep={args.sleep}s")

    # In-memory timer for the __all__ incremental sync (see
    # maybe_sync_all_index). Reset on every process start so a
    # deploy-interrupted sync resumes immediately on boot.
    all_index_state = {"checked_at": 0.0}

    while not _stop_requested:
        if args.dry:
            _log("--dry: would sync __all__; exiting without writing")
            return 0
        maybe_sync_all_index(args.database_url, args.api_url, all_index_state)
        if _stop_requested:
            break
        if args.once:
            _log("synced once, exiting (--once)")
            return 0
        # Wake often so SIGINT stays responsive and a freed backlog drains
        # promptly; the sync self-gates on ALL_CHECK_INTERVAL_SECS.
        slept = 0.0
        while slept < args.sleep and not _stop_requested:
            time.sleep(min(5.0, args.sleep - slept))
            slept += 5.0

    _log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
