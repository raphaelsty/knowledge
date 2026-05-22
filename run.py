#!/usr/bin/env python3
"""
Knowledge Pipeline Runner

Flow (all personalities or one via CLI arg):
  1. Ensure schema exists (idempotent CREATE IF NOT EXISTS).
  2. Read the working list of personalities from the ``users`` table
     and iterate. PG is the runtime source of truth — new personalities
     must be inserted into ``users`` via SQL (or a separate seed tool).

Usage:
    python run.py                              # all personalities, all sources
    python run.py max-halford                  # one personality, all sources
    python run.py --source twitter             # all personalities, twitter only
    python run.py max-halford --source twitter # one personality, twitter only
    python run.py --source twitter,reddit      # multiple sources, comma-sep

Restricting to a subset of sources keeps the rest of the pipeline
identical (cleanup + tagging still run on the new docs the selected
fetchers produced), so a single-source run leaves the user's
library in a coherent state.

`run.py` is the **data plane**: fetch sources, clean, tag, write to
Postgres. The ColBERT index is owned by ``sources.indexer_daemon``,
a separate process that watches PG and embeds rows on its own
schedule. `run.py` no longer touches any index (per-user or
``__all__``).
"""

import argparse
import os
import time
import traceback

from sources.sql import (
    create_api_tokens_table,
    create_auth_sessions_table,
    create_dead_urls_table,
    create_documents_table,
    create_events_table,
    create_export_downloads_table,
    create_favorites_table,
    create_follows_table,
    create_hn_frontpage_tables,
    create_index_health_checks_table,
    create_oauth_identities_table,
    create_personality_submissions_table,
    create_pipeline_runs_table,
    create_pipeline_source_runs_table,
    create_sessions_table,
    create_twitter_feed_status_table,
    create_users_table,
    create_views,
    get_user_tags,
    get_vip_tags,
    list_personalities,
)

# Credit-billing schema lives in its own module so the feature is a
# clean drop-in. See sources/sql/credits.sql for the table layout.
from sources.sql.credits import create_credits_tables
from sources.sql.user_storage import create_user_storage_table
from sources.sql.vip_sponsorships import create_vip_sponsorships_table
from sources.utils import run_pipeline
from sources.utils.backfill_all_sources import backfill as backfill_all_sources
from sources.utils.backfill_twitter_source import backfill as backfill_twitter_source

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Run the knowledge pipeline for one or all personalities, "
        "optionally restricted to a single source connector.",
    )
    p.add_argument(
        "slug",
        nargs="?",
        default=None,
        help="Personality slug (e.g. 'max-halford'). Omit to run every personality.",
    )
    p.add_argument(
        "--source",
        "-s",
        default=None,
        help="Comma-separated source keys to keep (e.g. 'twitter' or 'twitter,reddit'). "
        "All other sources are skipped for this run.",
    )
    p.add_argument(
        "--workers",
        "-w",
        type=int,
        default=-1,
        help="Parallelism for the source-fetch phase (threads, IO-bound). "
        "-1 (default) = min(cpu_count(), n_tasks). 1 = fully sequential. "
        "Cleaning and tagging always run sequentially after fan-in.",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Run this many personalities in parallel as separate processes. "
        "Default 1 = sequential. Each job is one full pipeline (fetch → "
        "clean → tag → embed). Bumping this is mainly useful when the "
        "embedder isn't the bottleneck — i.e. for source-heavy runs.",
    )
    p.add_argument(
        "--twikit",
        action="store_true",
        help="Route the Twitter source through cookie-authenticated twikit "
        "(Safari cookies) instead of api.twitterapi.io. Reads each VIP's "
        "public tweets + likes for free; bookmarks aren't accessible this "
        "way and are skipped. Useful for burst backfills without consuming "
        "twitterapi.io credit.",
    )
    return p.parse_args()


def filter_sources(sources: dict, only: set[str] | None) -> dict:
    """Return a copy of `sources` keeping only keys in `only` (if set)."""
    if not only:
        return sources
    return {k: v for k, v in sources.items() if k in only}


def main():
    args = parse_args()
    only_sources: set[str] | None = {s.strip() for s in args.source.split(",") if s.strip()} if args.source else None

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    # Fail-fast cookie check when `--twikit` is on. Without this we
    # used to walk 450 personalities completing each one with a
    # silent twitter-skip ("disabled (no Safari cookies)") and
    # bumping their `last_success_at` — so 328 VIPs ended the run
    # with their twitter source untouched but no obvious sign of
    # failure. Abort loudly instead, with the exact recovery
    # instructions.
    if args.twikit:
        try:
            from sources.twitter.cookies import get_safari_cookies

            creds = get_safari_cookies()
        except Exception as e:
            print(
                "\n[!] --twikit requires Safari cookies for x.com but they are unavailable:\n"
                f"    {e}\n"
                "    Open Safari, navigate to https://x.com, and confirm you're signed in.\n"
                "    Then retry the same command.",
                flush=True,
            )
            raise SystemExit(2) from None
        token_tail = (creds.get("auth_token") or "")[-6:] or "?"
        ct0_tail = (creds.get("ct0") or "")[-6:] or "?"
        print(
            f"twikit cookies OK at boot — auth_token …{token_tail}, ct0 …{ct0_tail}",
            flush=True,
        )

    # Bootstrap schema (idempotent — IF NOT EXISTS / CREATE OR REPLACE).
    create_users_table(database_url)
    create_documents_table(database_url)
    create_dead_urls_table(database_url)
    create_sessions_table(database_url)
    create_twitter_feed_status_table(database_url)
    create_auth_sessions_table(database_url)
    create_api_tokens_table(database_url)
    create_favorites_table(database_url)
    create_follows_table(database_url)
    create_events_table(database_url)
    create_export_downloads_table(database_url)
    create_pipeline_runs_table(database_url)
    create_pipeline_source_runs_table(database_url)
    create_index_health_checks_table(database_url)
    create_oauth_identities_table(database_url)
    create_personality_submissions_table(database_url)
    create_hn_frontpage_tables(database_url)
    create_credits_tables(database_url)
    create_user_storage_table(database_url)
    create_vip_sponsorships_table(database_url)
    create_views(database_url)

    # Twitter auto-enable: derive `sources.twitter.username` from
    # `links.twitter` for every user that has a Twitter URL but no
    # `sources.twitter` block. Idempotent — only touches rows still
    # missing the source. The pipeline orchestrator below then sees the
    # filled-in source and runs the Twitter fetcher with the API key
    # from .env. No `max_pages` is set here; the fetcher's own default
    # applies.
    derived, skipped = backfill_twitter_source(database_url)
    if derived:
        print(f"Twitter source auto-enabled for {len(derived)} users")
        for slug, handle in derived:
            print(f"  + {slug:<28} → @{handle}")
    if skipped:
        print(f"Skipped {len(skipped)} users with un-parseable links.twitter URL")

    # General source backfill: derives `youtube_search` from display
    # name, `websites` from links.website (probes RSS/sitemap candidates),
    # `github_repos` / `github_gists` from links.github, and validates
    # `huggingface` against the HF API. Idempotent — only fills missing
    # keys, never overwrites human-curated values. Skips wikipedia/
    # reddit/SO/arxiv-by-name because those need per-person reasoning.
    plan = backfill_all_sources(database_url)
    if plan:
        print(f"Source backfill: {len(plan)} users got new keys")
        for slug, adds in plan:
            print(f"  + {slug:<28} {', '.join(sorted(adds.keys()))}")

    # Working list comes from PG.
    all_personalities = list_personalities(database_url)

    # Optional CLI filter by slug.
    if args.slug:
        to_process = [p for p in all_personalities if p["slug"] == args.slug]
        if not to_process:
            print(f"Error: personality '{args.slug}' not found in users table")
            raise SystemExit(1)
    else:
        to_process = all_personalities

    # Tag-vocabulary policy: the cross-user pool is restricted to VIP
    # users so a non-VIP's idiosyncratic tags can't seed everyone
    # else's libraries. Per-personality vocab = VIP pool ∪ that
    # user's own tags (computed inside the loop). For VIP users the
    # union collapses to the VIP pool, since their tags are already
    # in there.
    vip_tags = get_vip_tags(database_url)
    vip_tag_set = set(vip_tags)
    print(f"\nVIP tag vocabulary: {len(vip_tags):,} tags (cross-user pool, VIPs only)")
    if only_sources:
        print(f"Source filter: only running {sorted(only_sources)}")

    # Surface the queue order so the operator can sanity-check it
    # before a long run. `list_personalities` already sorts by:
    # vip DESC, last_success_at ASC NULLS FIRST, name — so VIPs
    # never-run land at the very top, then VIPs by staleness, then
    # everyone else in the same priority shape.
    if not args.slug:
        n_vip = sum(1 for p in to_process if p.get("vip"))
        n_never = sum(1 for p in to_process if p.get("last_success_at") is None)
        print(f"Queue: {len(to_process)} personalities — {n_vip} VIP, " f"{n_never} never-run-yet (front of queue)")
        head = to_process[:5]
        for p in head:
            ts = p.get("last_success_at")
            stamp = ts.isoformat(timespec="minutes") if ts else "never"
            tag = "★" if p.get("vip") else " "
            print(f"  {tag} {p['slug']:<28} last success: {stamp}")
        if len(to_process) > 5:
            print(f"  … {len(to_process) - 5} more")

    total_start = time.perf_counter()

    # Pre-build (personality, sources, shared_tags) tuples so the
    # parallel branch doesn't open PG connections in the parent
    # process for every job. The PG hit for `get_user_tags` is cheap
    # (one indexed query per user) so we do it serially up-front.
    jobs: list[dict] = []
    for personality in to_process:
        slug = personality["slug"]
        name = personality["name"]
        sources = filter_sources(personality["sources"], only_sources)
        if not sources:
            reason = (
                f"no '{', '.join(sorted(only_sources))}' source configured" if only_sources else "no sources configured"
            )
            print(f"Skipping {name} ({reason})")
            continue
        is_vip = bool(personality.get("vip"))
        own_tags = get_user_tags(database_url, personality["id"])
        shared_tags = sorted(vip_tag_set | set(own_tags))
        jobs.append(
            {
                "slug": slug,
                "name": name,
                "index_name": personality["indexName"],
                "sources_config": sources,
                "user_id": personality["id"],
                "database_url": database_url,
                "shared_tags": shared_tags,
                "n_workers": args.workers,
                "vip": is_vip,
                "twikit": bool(args.twikit),
            }
        )

    if not jobs:
        print("\nNo personalities to process.")
        return

    parallel = max(1, args.jobs)
    if parallel == 1 or len(jobs) == 1:
        for i, job in enumerate(jobs, 1):
            _run_one(job, i, len(jobs))
    else:
        # Process pool — each worker forks/spawns a fresh Python and
        # runs `run_pipeline` to completion for its assigned slug.
        # Stdout from concurrent jobs will interleave; the pipeline's
        # `@@stage|msg@@` markers make that legible enough for an
        # operator tailing the log.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"\nFanning out: {len(jobs)} personalities across {parallel} workers")
        completed = 0
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run_one_in_subprocess, job): job for job in jobs}
            for fut in as_completed(futures):
                completed += 1
                job = futures[fut]
                try:
                    fut.result()
                    print(f"\n[done {completed}/{len(jobs)}] {job['slug']}")
                except Exception as exc:
                    print(f"\n[!] Pipeline failed for {job['name']} ({job['slug']}): {exc}")

    total = time.perf_counter() - total_start
    m, s = divmod(total, 60)
    print(f"\nAll personalities processed in {int(m)}m {s:.1f}s")
    # NOTE: document categorization runs as its own systemd service
    # (`knowledge-categorize-daemon` → sources/utils/categorize_daemon.py),
    # not inline in run.py. The daemon keeps the data plane decoupled
    # from the categorization sweep and respects a strict 10 % CPU
    # quota / 384 MB memory cap that wouldn't fit a foreground step.


def _run_one(job: dict, idx: int, total: int) -> None:
    """Sequential single-personality run with the banner the legacy
    flow used to print."""
    print(f"\n{'=' * 60}")
    print(f"  [{idx}/{total}] {job['name']} ({job['slug']})")
    print(f"{'=' * 60}\n")
    # Data-only pass. The ColBERT index is the indexer daemon's job
    # (`sources.indexer_daemon`) — `run.py` just fetches + cleans +
    # writes to Postgres. The daemon picks up freshly-inserted rows
    # (`indexed=false`) on its next sweep and embeds them.
    try:
        run_pipeline(
            slug=job["slug"],
            name=job["name"],
            index_name=job["index_name"],
            sources_config=job["sources_config"],
            user_id=job["user_id"],
            database_url=job["database_url"],
            shared_tags=job["shared_tags"],
            n_workers=job["n_workers"],
            vip=job["vip"],
            twitter_via_twikit=job.get("twikit", False),
            do_index=False,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"\n[!] Pipeline failed for {job['name']} ({job['slug']}): {exc}")
        traceback.print_exc()
        print("[!] Continuing with remaining personalities…")


def _run_one_in_subprocess(job: dict) -> None:
    """Process-pool worker. Returns nothing; exceptions propagate to
    the parent via the future result. Data-only; index work is owned
    by the separate indexer daemon."""
    print(f"\n[start] {job['slug']}")
    run_pipeline(
        slug=job["slug"],
        name=job["name"],
        index_name=job["index_name"],
        sources_config=job["sources_config"],
        user_id=job["user_id"],
        database_url=job["database_url"],
        shared_tags=job["shared_tags"],
        n_workers=job["n_workers"],
        vip=job["vip"],
        twitter_via_twikit=job.get("twikit", False),
        do_index=False,
    )


if __name__ == "__main__":
    main()
