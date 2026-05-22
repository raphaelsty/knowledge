"""Filter the raw twitter_inner_circle.csv down to "AI curator" rows.

Two passes:

1. Drop macro-celebrity accounts. The follow-graph aggregation pulls
   in handles like @elonmusk, @jack, @openai because *everyone*
   follows them — that's social gravity, not signal. Heuristics that
   catch these without over-pruning:
     • followers ≥ 3,000,000   (gravity-tier)
     • OR following ≤ 10       (broadcast-only accounts)

2. Keep only rows whose bio shows a newsletter / writing signal.
   Real bios for the people the operator wants always advertise the
   publication — `substack.com/...`, "Author of … newsletter",
   "Subscribe at …", "Daily digest", etc. We look for those tokens
   directly because bio URLs are t.co-shortened and the original
   destination isn't in the CSV.

Reads:  data/people/twitter_inner_circle.csv
Writes: data/people/twitter_inner_circle_newsletter.csv  (subset)

Usage::

    uv run python scripts/filter_twitter_inner_circle.py
    uv run python scripts/filter_twitter_inner_circle.py --keep-macro
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Hard tokens. Match case-insensitively. These are publication-signal
# words we trust on their own:
_NEWSLETTER_TOKENS = (
    "newsletter",
    "substack",
    "beehiiv",
    "ghost.io",
    "ck.page",
    "convertkit",
    "buttondown",
    "subscribe to",
    "subscribe at",
    "daily digest",
    "weekly digest",
    "weekly roundup",
    "weekly newsletter",
    "the latent space",
    "import ai",
    "interconnects",
    "last week in ai",
    "ai snake oil",
)

# Soft tokens that ONLY count when paired with a publication noun. We
# don't want bare "writes" / "writer" matching every journalist on the
# planet, but "writes the X newsletter" is exactly the signal we want.
_SOFT_VERB_RE = re.compile(
    r"\b(writes?|writing|publishes?|publishing|author of)\b[^\n]{0,40}"
    r"\b(newsletter|substack|blog|digest|column|the\s+[A-Z][^\s]+)",
    re.IGNORECASE,
)

# Macro-celebrity gates.
_MACRO_FOLLOWERS = 3_000_000
_MACRO_FOLLOWING_FLOOR = 10


def _is_macro(followers: int, following: int) -> bool:
    """Return True for accounts that look like social-gravity sinks
    rather than AI Twitter signal."""
    if followers >= _MACRO_FOLLOWERS:
        return True
    if following <= _MACRO_FOLLOWING_FLOOR:
        # Brand / broadcast account — follows almost nobody.
        return True
    return False


def _has_newsletter_signal(bio: str) -> bool:
    bio_l = bio.lower()
    for tok in _NEWSLETTER_TOKENS:
        if tok in bio_l:
            return True
    if _SOFT_VERB_RE.search(bio):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/people/twitter_inner_circle.csv",
        help="input CSV (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        default="data/people/twitter_inner_circle_newsletter.csv",
        help="output CSV (default: %(default)s)",
    )
    ap.add_argument(
        "--keep-macro",
        action="store_true",
        help="skip the macro-celebrity prune (debugging)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    in_path = (repo_root / args.inp).resolve()
    out_path = (repo_root / args.out).resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    dropped_macro = 0
    dropped_no_newsletter = 0
    kept = 0

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        fieldnames = list(r.fieldnames or [])
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            total += 1
            followers = int(row.get("followers") or 0)
            following = int(row.get("following") or 0)
            bio = row.get("bio") or ""
            if not args.keep_macro and _is_macro(followers, following):
                dropped_macro += 1
                continue
            if not _has_newsletter_signal(bio):
                dropped_no_newsletter += 1
                continue
            w.writerow(row)
            kept += 1

    print(
        f"\nRead {total:,} rows from {in_path.name}\n"
        f"  dropped {dropped_macro:,} macro-celebrity\n"
        f"  dropped {dropped_no_newsletter:,} no newsletter signal\n"
        f"  kept    {kept:,}\n"
        f"→ {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
