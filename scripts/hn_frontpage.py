#!/usr/bin/env python3
"""Debug CLI for the HackerNews front-page → per-user picks job.

**In production this job runs as a daemon**, not from here: the
`knowledge-hn-frontpage` compose service runs
`sources.utils.hn_frontpage_daemon` once a day. This script exists so
you can run the same code by hand against one user and see what it
picks — it shares every line of logic with the daemon via
`sources.hackernews.picks`.

(It could never have run in prod anyway: `Dockerfile.daemons` copies
`sources/`, not `scripts/`.)

Flow, in `sources.hackernews.picks.refresh_picks`:
    1. Snapshot the current HN front page (Firebase API).
    2. Score every article against each user's library through
       ``/indices/{name}/search_with_encoding``.
    3. Convert to per-article z-scores across the cohort, so ranking
       reflects *this* user's affinity instead of title length. (Read
       the module docstring — the raw ColBERT mean correlates +0.79
       with title token count and gives every user the same picks.)
    4. Keep the top-N above the z threshold, re-order by HN upvotes.
    5. Insert the run + REPLACE each user's picks atomically.

Usage:
    DATABASE_URL=postgresql://... API_URL=http://localhost:8080 \\
      uv run python scripts/hn_frontpage.py [flags]

Flags:
    --slug NAME      score for one user only (still scores a small
                     reference cohort, since z-scores need a baseline)
    --top-per-user N keep at most N picks per user (default 10)
    --top N          fetch only the top-N front-page items
                     (default 30 — matches HN's own front page)
    --threshold Z    z floor a pick must clear (default 0.5)
    --limit N        stop after N users
    --dry            fetch + score, write nothing at all (no run row
                     either, so the live feed is untouched)
    --no-snapshot    skip the fetch and score against the most recent
                     existing run (useful when iterating on scoring)
    --debug          print each user's picks with z-score and upvotes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the package importable when run as `python scripts/hn_frontpage.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sources.hackernews.picks import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_TOP,
    DEFAULT_TOP_PER_USER,
    refresh_picks,
)

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_URL = "http://localhost:8080"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hn_frontpage.py", description=__doc__)
    p.add_argument("--slug", default=None, help="Score only this personality.")
    p.add_argument("--top-per-user", type=int, default=DEFAULT_TOP_PER_USER)
    p.add_argument("--top", type=int, default=DEFAULT_TOP)
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum z-score for a pick (default %(default)s).",
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N users.")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--no-snapshot", action="store_true")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print each user's picks with z-score and upvote count.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        refresh_picks(
            os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            os.environ.get("API_URL", DEFAULT_API_URL),
            top=args.top,
            top_per_user=args.top_per_user,
            threshold=args.threshold,
            slug=args.slug,
            limit=args.limit,
            dry=args.dry,
            no_snapshot=args.no_snapshot,
            debug=args.debug,
        )
    except Exception as exc:  # noqa: BLE001 — CLI surface, print not traceback
        print(f"failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
