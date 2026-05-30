"""Indexer daemon — long-running process that owns the ColBERT index.

Architecture
------------
The fetch pipeline (``run.py`` / ``run_pipeline(do_index=False)``)
writes documents to Postgres. **Nothing else** touches the on-disk
index. This daemon is the sole writer for every per-user index:

  * it scans the live index API + PG to classify every user's index
    as ``broken``, ``error``, ``missing``, ``pg_drift`` (PG has rows
    that haven't been embedded yet), ``empty`` (no PG docs), or
    ``healthy``;
  * it builds a priority queue and processes one user at a time;
  * for each picked user it invokes ``run_pipeline`` with
    ``sources_config={}`` (no fetchers) and ``do_index=True`` — the
    same in-process embedder + heal logic that ``make run`` used to
    call inline, just gated on this daemon being the only caller.

Priority (highest first)
~~~~~~~~~~~~~~~~~~~~~~~~
1. **Broken / error** — index returns HTTP 5xx ("No data to merge"),
   or PG flags everything indexed but the API has 0 embeddings. The
   user library is currently unsearchable; fix first.
2. **Missing** — PG has docs but the API returns 404 for the index.
3. **Pg_drift** — index loads but ``indexed=false`` rows are
   pending. Ordered by the size of the backlog so users with
   thousands of fresh tweets get embedded before users with three.
4. (Anything else is healthy → skipped.)

Within each tier we tie-break on VIP first, then alphabetical slug
for determinism.

``__all__`` upkeep
~~~~~~~~~~~~~~~~~~
Two mechanisms, both owned here so the cross-personality index never
needs a human:

1. **Self-heal (prioritised).** At the top of every sweep the daemon
   classifies ``__all__`` (``maybe_rebuild_all_index``). If it's
   genuinely unusable — ``broken`` / ``error`` / ``missing`` (e.g. a
   deploy SIGTERMed a rebuild and left 0-byte centroids, so every read
   500s) — it rebuilds from PG via the staging+promote path *before*
   any per-user work. Probed at most once a minute; a failed rebuild
   backs off for ``ALL_REBUILD_COOLDOWN_SECS``, but a fresh process
   start retries immediately (timers are in-memory), so the very
   restart that killed a rebuild also triggers its recovery.

2. **Incremental mirror.** After a VIP user's per-user index is rebuilt
   we mirror their docs into ``__all__`` via
   ``index_health.update_all_index_for_slugs`` — delete the user's
   existing chunks, then re-push — so newly promoted VIPs don't fall
   out of feed search between full rebuilds. Best-effort: a failure
   logs and moves on.

``stale`` / ``pg_drift`` verdicts do NOT trigger the full rebuild —
they're minor and reconciled by the incremental mirror + the hourly
job; only the unusable verdicts above are worth the multi-minute cost.

Coordination
~~~~~~~~~~~~
Even though the daemon is the only writer in steady state, multiple
daemon instances may be started by accident (laptop + prod, two
operators). Before processing a user we acquire a PG advisory lock
keyed by ``user_id`` (see ``sources.utils.index_locks``) — a second
daemon sees the lock held and skips that user. Locks release
automatically when the holding daemon's PG session ends, so a kill
-9 doesn't wedge the queue.

CLI
~~~
::

    python -m sources.indexer_daemon                   # loop forever
    python -m sources.indexer_daemon --once            # one user, exit
    python -m sources.indexer_daemon --dry             # print queue, do nothing
    python -m sources.indexer_daemon --vip-only        # ignore non-VIPs
    python -m sources.indexer_daemon --include-drift   # also process pg_drift
    python -m sources.indexer_daemon --sleep 5         # gap between users
    python -m sources.indexer_daemon --idle-sleep 600  # wait when queue empty
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections.abc import Iterable
from datetime import datetime

import psycopg

from sources.utils.index_health import (
    ALL_INDEX_NAME,
    DEFAULT_API_URL,
    DEFAULT_DATABASE_URL,
    classify_index,
)
from sources.utils.index_locks import IndexBusy, acquire_index_lock

# Priority tiers (lower number = higher priority).
PRI_BROKEN = 0
PRI_ERROR = 1
PRI_MISSING = 2
PRI_DRIFT = 3
# `backlog` covers users whose index health checks out (pg_indexed ≈
# api.num_documents) but who still have indexed=false rows sitting in
# PG — typically tweets synced in from a dev box that landed after
# the last embed pass. classify_index reports `healthy` for them
# because its drift formula only compares pg_indexed vs api, so
# without this tier they'd never be queued and the unindexed rows
# would pile up forever. Ranked below drift on purpose: a small
# backlog is benign and shouldn't pre-empt a genuinely broken index.
PRI_BACKLOG = 4
PRI_HEALTHY = 99  # skipped

# After this many consecutive failed attempts on a single user, the
# daemon force-deletes the on-disk index + resets indexed=FALSE in
# PG and lets the next iteration rebuild from scratch. Catches the
# "index file loads OK but every batch fails with 'No data to merge'"
# state that `run_pipeline`'s in-process healer misses — it only
# fires on a load-time failure, not on per-batch failures.
HARD_HEAL_THRESHOLD = 3

# Cool-down after a failed iteration. Longer than `--sleep` (which
# governs the success path) so a stuck user doesn't get hammered
# at the same cadence as healthy users.
FAILURE_SLEEP_SECS = 10.0

# ── __all__ self-heal ────────────────────────────────────────────────
# The cross-personality `__all__` index powers logged-out / bare-search
# and the search-time cross-personality score join. It's a derivative
# (rebuildable entirely from `documents`), and historically only a
# manual `make all-rebuild` recreated it — so a mid-build interruption
# (e.g. a deploy SIGTERMing the rebuild) left it 0-byte and every
# read 500'd until someone noticed. The daemon now owns its recovery:
# at the top of each sweep it classifies `__all__`, and if the index
# is genuinely unusable it rebuilds it FIRST, before any per-user work.
#
# Only the unloadable/empty verdicts trigger the (expensive, ~minutes)
# full rebuild — `stale` / `pg_drift` are minor and already reconciled
# by the per-user incremental push hook + the hourly daemon, so we
# don't full-rebuild for those.
ALL_BROKEN_VERDICTS = frozenset({"broken", "error", "missing"})
# How often to run the incremental `__all__` sync (which also does the
# cheap health GET that detects a break). Short, so a broken index is
# spotted within seconds and new docs flow in promptly.
ALL_CHECK_INTERVAL_SECS = 20.0
# Cap docs synced per sweep so one pass stays bounded and the loop keeps
# re-checking health frequently. The `indexed_all` flag makes the sync
# resumable, so a large backlog just takes several sweeps to drain.
ALL_SYNC_MAX_DOCS = 6000

_VERDICT_TO_PRIORITY = {
    "broken": PRI_BROKEN,
    "error": PRI_ERROR,
    "missing": PRI_MISSING,
    "pg_drift": PRI_DRIFT,
    "backlog": PRI_BACKLOG,
    "healthy": PRI_HEALTHY,
    "empty": PRI_HEALTHY,
}


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts}  {msg}", flush=True)


def _fetch_user_rows(conn) -> list[dict]:
    """Pull every VIP user that has at least one document in PG, with
    per-user document counts. Users without any documents are skipped
    — there's nothing to index, and the API would return 404
    forever otherwise.

    Non-VIPs are filtered out entirely: Plaid indexing is a VIP-only
    feature, and non-VIP search falls back to the SQL keyword path in
    ``api/src/handlers/personalities.rs::fallback_search``. Without
    this filter the daemon would happily build per-user Plaid indices
    for every signed-up account, which is the cost shape we explicitly
    want to avoid as the user base grows.
    """
    sql = """
        SELECT u.id, u.username, u.username AS index_name, u.vip,
               COUNT(d.*) FILTER (WHERE d.deleted = FALSE) AS pg_total,
               COUNT(d.*) FILTER (
                   WHERE d.deleted = FALSE AND d.indexed = TRUE
               ) AS pg_indexed
          FROM users u
          LEFT JOIN documents d ON d.user_id = u.id
         WHERE u.vip = TRUE
         GROUP BY u.id
        HAVING COUNT(d.*) FILTER (WHERE d.deleted = FALSE) > 0
         ORDER BY u.vip DESC, u.username
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def build_priority_queue(
    database_url: str,
    api_url: str,
    *,
    vip_only: bool = False,
    include_drift: bool = True,
) -> list[dict]:
    """Audit every user's index and return the sorted work queue.

    Each entry is a dict carrying:
      ``id``, ``username``, ``index_name``, ``vip``, ``pg_total``,
      ``pg_indexed``, ``verdict``, ``reason``, ``priority``, ``backlog``.

    The list is sorted so that ``queue[0]`` is the user the daemon
    should process next.
    """
    with psycopg.connect(database_url) as conn:
        rows = _fetch_user_rows(conn)
    if vip_only:
        rows = [r for r in rows if r["vip"]]

    out: list[dict] = []
    for r in rows:
        if r["index_name"] == ALL_INDEX_NAME:
            # Defensive — `__all__` should never be a user's
            # `index_name`, but if a row ever sneaks in we ignore it
            # so this daemon can't accidentally rebuild the cross-
            # personality index. That's owned by ``make all-rebuild``.
            continue
        verdict, reason = classify_index(api_url, r["index_name"], r["pg_total"], r["pg_indexed"])
        backlog = max(0, int(r["pg_total"]) - int(r["pg_indexed"]))
        # Reclassify "healthy with unindexed backlog" as `backlog` so
        # the daemon embeds the pending rows. classify_index's drift
        # formula only fires when pg_indexed disagrees with the index;
        # a user can carry hundreds of indexed=false rows (e.g. tweets
        # synced from a dev box) and still look healthy by that
        # measure. Without this, those rows never get embedded.
        if verdict == "healthy" and backlog > 0:
            verdict = "backlog"
            reason = f"backlog={backlog} indexed=false in PG"
        prio = _VERDICT_TO_PRIORITY.get(verdict, PRI_HEALTHY)
        # `pg_drift` is opt-in: in practice it usually heals itself
        # the next time the daemon comes through, so the operator
        # can choose whether to include it in the queue.
        if verdict == "pg_drift" and not include_drift:
            continue
        if prio == PRI_HEALTHY:
            continue
        out.append(
            {
                **r,
                "verdict": verdict,
                "reason": reason,
                "priority": prio,
                "backlog": backlog,
            }
        )

    # Multi-key sort: priority tier first, then largest backlog
    # within tier (so the noisiest broken users heal first), then
    # VIPs (already pre-sorted but re-applied for stability), then
    # slug for a deterministic tiebreaker.
    out.sort(key=lambda r: (r["priority"], -r["backlog"], 0 if r["vip"] else 1, r["username"]))
    return out


def _print_queue(queue: Iterable[dict], limit: int = 30) -> None:
    """Pretty-print the top of the queue (for `--dry` / dashboard)."""
    queue_list = list(queue)
    _log(f"queue: {len(queue_list)} user(s) need indexing")
    if not queue_list:
        return
    for i, r in enumerate(queue_list[:limit], start=1):
        vip_tag = "★" if r["vip"] else " "
        _log(
            f"  [{i:>3}] {vip_tag} {r['username']:<32} "
            f"{r['verdict']:<8} pg={r['pg_indexed']}/{r['pg_total']:<6} "
            f"backlog={r['backlog']:<6} {r['reason'][:60]}"
        )
    if len(queue_list) > limit:
        _log(f"  … and {len(queue_list) - limit} more")


_stop_requested = False


def _install_signal_handlers() -> None:
    """Cooperative SIGINT/SIGTERM — finish the in-flight user, then exit."""

    def _handler(signum, _frame):
        global _stop_requested
        if _stop_requested:
            _log(f"signal {signum} received twice, aborting now")
            sys.exit(130)
        _stop_requested = True
        _log(f"signal {signum} received — finishing current user then exiting")

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            pass


def _process_one(row: dict, database_url: str, api_url: str) -> bool:
    """Reindex one user. Returns True on success, False on failure.

    Acquires a per-user advisory lock (non-blocking). If another
    daemon owns it, we report and move on without waiting — the
    other daemon will handle this user, no work is lost.
    """
    user_id = int(row["id"])
    slug = row["username"]

    # Lazy import — pulling `run_pipeline` drags in the whole
    # fetcher graph, which is heavy on import. Keep the daemon's
    # boot fast for `--dry` runs that never call this.
    from sources.sql import get_user_tags, get_vip_tags
    from sources.utils import run_pipeline

    own_tags = get_user_tags(database_url, user_id)
    shared_tags = sorted(set(get_vip_tags(database_url)) | set(own_tags))

    try:
        with acquire_index_lock(database_url, user_id, blocking=False):
            os.environ["API_URL"] = api_url
            run_pipeline(
                slug=slug,
                name=slug,  # display name == slug here; cosmetic
                index_name=row["index_name"],
                sources_config={},  # NO fetchers — embedder only
                user_id=user_id,
                database_url=database_url,
                shared_tags=shared_tags,
                n_workers=1,
                vip=bool(row["vip"]),
                do_index=True,
            )
    except IndexBusy:
        _log(f"  skip {slug}: another writer holds the index lock")
        return False
    except Exception as exc:
        _log(f"  [!] {slug}: {exc!r}")
        return False

    # `__all__` is no longer mirrored per-slug here. It's a single index
    # maintained by `maybe_sync_all_index`, which streams every VIP doc
    # in via the `indexed_all` flag — new docs land in `__all__` on the
    # next sync sweep regardless of the per-user index. Pushing per-slug
    # here too would double-add (the sync doesn't know this path ran), so
    # the hook is removed.
    return True


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
        help="Seconds between users when the queue is non-empty.",
    )
    p.add_argument(
        "--idle-sleep",
        type=float,
        default=600.0,
        help="Seconds to wait between full sweeps when nothing needs work. "
        "Default 10 minutes — long enough that an idle daemon stops "
        "competing with the API for CPU, short enough that a freshly-"
        "uploaded bookmark gets embedded within ~one coffee.",
    )
    p.add_argument(
        "--vip-only",
        action="store_true",
        help="Restrict the queue to VIP users.",
    )
    p.add_argument(
        "--include-drift",
        action="store_true",
        default=True,
        help="Also process users with pg_drift (default true).",
    )
    p.add_argument(
        "--exclude-drift",
        dest="include_drift",
        action="store_false",
        help="Skip pg_drift users (handy for a tight broken-only sweep).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Process the top user (if any) and exit. Useful as a cron tick.",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print the queue and exit without writing anything.",
    )
    args = p.parse_args(argv)

    _install_signal_handlers()
    _log("indexer-daemon starting")
    _log(f"  database_url={args.database_url}")

    # In-memory timer for the __all__ incremental sync (see
    # maybe_sync_all_index). Reset on every process start so a
    # deploy-interrupted sync resumes immediately on boot.
    all_index_state = {"checked_at": 0.0}
    _log(f"  api_url     ={args.api_url}")
    _log(
        f"  vip_only={args.vip_only}  include_drift={args.include_drift}  "
        f"sleep={args.sleep}s  idle_sleep={args.idle_sleep}s"
    )

    while not _stop_requested:
        # Prioritised: keep the single `__all__` index current
        # (incremental sync + recreate-on-break) before any per-user
        # indexing. Skipped in --dry (audit-only) mode.
        if not args.dry:
            maybe_sync_all_index(args.database_url, args.api_url, all_index_state)
            if _stop_requested:
                break

        # Single-index world: there is no per-user indexing any more.
        # The daemon's only job is to keep `__all__` current (done above
        # by maybe_sync_all_index). Per-personality indices were retired,
        # so the priority-queue / per-user reindex path is gone — which
        # also frees the encoder entirely for the `__all__` sync.
        if args.dry:
            _log("--dry: __all__ sync only; nothing else to do")
            return 0
        if args.once:
            _log("synced once, exiting (--once)")
            return 0
        # Wake often so SIGINT stays responsive and a freed backlog drains
        # promptly; the sync self-gates on ALL_CHECK_INTERVAL_SECS, so
        # these wake-ups are cheap no-ops when there's nothing to do.
        slept = 0.0
        while slept < args.sleep and not _stop_requested:
            time.sleep(min(5.0, args.sleep - slept))
            slept += 5.0

    _log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
