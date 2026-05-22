"""Pull the CSRankings author registry + per-area publication counts
and write a single joined CSV.

Sources (raw GitHub):
  • csrankings.csv             — name, affiliation, homepage, scholar_id
  • generated-author-info.csv  — name, dept, area, sub-area, count, year

We join on `name` and pivot the area counts into one row per author so
the resulting CSV is operator-friendly:

    name,affiliation,homepage,scholar_id,
    total_count,top_area,top_area_count,areas_json

`areas_json` is a small JSON blob with the per-area pub counts so the
later "match against PG / enrich" step has every signal it needs.

Usage::

    uv run python scripts/fetch_csrankings.py
    uv run python scripts/fetch_csrankings.py --out data/people/csrankings.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# Raw CSV endpoints. Both files live in the master branch of the
# CSRankings repo and are updated nightly by their CI.
_AUTHORS_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/master/csrankings.csv"
_AREA_COUNTS_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/master/generated-author-info.csv"


def _fetch_csv(url: str) -> list[dict]:
    """GET a CSV and return a list of dicts (DictReader). Surfaces any
    network or parse error to the caller."""
    req = urllib.request.Request(url, headers={"User-Agent": "knowledge-csrankings/1.0"})
    print(f"  ↓ {url}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(body)))
    print(f"    parsed {len(rows):,} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/people/csrankings.csv", help="output CSV path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    authors = _fetch_csv(_AUTHORS_URL)
    areas = _fetch_csv(_AREA_COUNTS_URL)

    # Aggregate area counts per author.
    # `generated-author-info.csv` has one row per (name, area, year), so
    # we sum across years and pivot to {area: count}.
    by_author_area: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in areas:
        name = (r.get("name") or "").strip()
        area = (r.get("area") or "").strip()
        try:
            cnt = float(r.get("count") or 0)
        except ValueError:
            cnt = 0
        if not name or not area:
            continue
        by_author_area[name][area] += cnt

    # Emit one row per author from the canonical registry; left-join area
    # counts. Authors with no publication-info row (rare) still appear,
    # just with empty area data.
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "affiliation",
                "homepage",
                "scholar_id",
                "dblp_id",
                "total_count",
                "top_area",
                "top_area_count",
                "areas_json",
            ]
        )
        for r in authors:
            name = (r.get("name") or "").strip()
            if not name:
                continue
            area_counts = by_author_area.get(name, {})
            total = sum(area_counts.values())
            top_area, top_count = ("", 0.0)
            if area_counts:
                top_area, top_count = max(area_counts.items(), key=lambda kv: kv[1])
            w.writerow(
                [
                    name,
                    (r.get("affiliation") or "").strip(),
                    (r.get("homepage") or "").strip(),
                    (r.get("scholarid") or "").strip(),
                    # The registry uses each author's name as their DBLP
                    # disambiguation key, which is how CSRankings links
                    # to DBLP. Persist it so we can hit the DBLP fetcher
                    # without re-scraping later.
                    name,
                    round(total, 2),
                    top_area,
                    round(top_count, 2),
                    json.dumps({k: round(v, 2) for k, v in sorted(area_counts.items())}),
                ]
            )
    print(f"\n✓ wrote {out_path} ({len(authors):,} authors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
