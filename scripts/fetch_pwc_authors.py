"""Pull author signals from Papers with Code, ranked per task.

What this fetches
-----------------
Walks the Papers with Code REST API (`paperswithcode.com/api/v1/`):

  1. List tasks  (`/tasks/`)         — paginated, capped by `--max-tasks`.
  2. Top papers  (`/tasks/{id}/papers/?ordering=-stars`) — per task,
     capped by `--papers-per-task`.
  3. Aggregate authors across (task × paper) pairs.

We don't have author IDs at the API level — papers expose authors as
plain name strings — so the output keys on author *name*. Same caveat
as for the CSRankings dump: matching against PG users / arXiv / DBLP
happens in a later step.

Output CSV columns:

    name, papers_seen, tasks_seen, top_task, top_task_papers,
    sample_paper_title, sample_paper_url, tasks_json

`tasks_json` is `{task_name: papers_count}` so the matcher has the
full per-area breakdown when we get to enrichment.

Usage::

    uv run python scripts/fetch_pwc_authors.py
    uv run python scripts/fetch_pwc_authors.py --max-tasks 100 --papers-per-task 50
    uv run python scripts/fetch_pwc_authors.py --out data/people/pwc_authors.csv

The defaults pull ~50 tasks × 30 papers ≈ 1,500 papers and 5–8k
author-task rows in ~3–5 minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

_API_BASE = "https://paperswithcode.com/api/v1"
_USER_AGENT = "knowledge-pwc/1.0 (+https://knowledge-web.org)"
# Be polite — sleep this many seconds between API calls so we don't
# accidentally hammer the public API. PwC has historically been
# tolerant but unspecified about limits.
_REQ_DELAY_S = 0.4


def _get_json(url: str) -> dict | None:
    """One GET; returns parsed JSON or None on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print(f"    ! HTTP {resp.status} for {url}")
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"    ! fetch error: {e}  ({url})")
        return None


def _paginate(start_url: str, max_pages: int) -> list[dict]:
    """Walk PwC's DRF-style pagination via the `next` link until empty
    or `max_pages` reached. Returns the concatenated `results`."""
    out: list[dict] = []
    url: str | None = start_url
    pages = 0
    while url and pages < max_pages:
        page = _get_json(url)
        if not page:
            break
        results = page.get("results") or []
        out.extend(results)
        pages += 1
        url = page.get("next")
        time.sleep(_REQ_DELAY_S)
    return out


def fetch_tasks(max_tasks: int) -> list[dict]:
    print(f"\n[1/2] Fetching tasks (cap {max_tasks})")
    # PwC paginates at 50/page by default — we just walk until we have
    # enough or pagination runs out.
    pages_cap = max(1, (max_tasks + 49) // 50)
    tasks = _paginate(f"{_API_BASE}/tasks/", max_pages=pages_cap)[:max_tasks]
    print(f"    {len(tasks)} tasks")
    return tasks


def fetch_papers_for_task(task_id: str, papers_per_task: int) -> list[dict]:
    """Top papers for a task, ranked by GitHub stars desc."""
    url = f"{_API_BASE}/tasks/{urllib.parse.quote(task_id, safe='')}/papers/" f"?ordering=-stars"
    pages_cap = max(1, (papers_per_task + 49) // 50)
    return _paginate(url, max_pages=pages_cap)[:papers_per_task]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/people/pwc_authors.csv", help="output CSV path")
    ap.add_argument(
        "--max-tasks",
        type=int,
        default=80,
        help="Cap on the number of tasks to walk (default 80).",
    )
    ap.add_argument(
        "--papers-per-task",
        type=int,
        default=30,
        help="Cap on papers per task (default 30).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = fetch_tasks(args.max_tasks)
    if not tasks:
        print("No tasks returned — bailing.")
        return 1

    # author_name → {
    #     "papers": int,
    #     "tasks": Counter(task_name -> papers),
    #     "sample": (title, url),  # first paper we saw
    # }
    agg: dict[str, dict] = defaultdict(lambda: {"papers": 0, "tasks": defaultdict(int), "sample": None})

    print(f"\n[2/2] Walking papers (≤{args.papers_per_task} per task)")
    for i, task in enumerate(tasks, 1):
        task_id = task.get("id") or task.get("slug") or ""
        task_name = (task.get("name") or task_id).strip()
        if not task_id:
            continue
        papers = fetch_papers_for_task(task_id, args.papers_per_task)
        print(f"  [{i}/{len(tasks)}] {task_name:<40} {len(papers)} papers")
        for paper in papers:
            authors = paper.get("authors") or []
            title = (paper.get("title") or "").strip()
            url = (paper.get("url_abs") or paper.get("url_pdf") or "").strip()
            for a in authors:
                # Authors come back as strings in this endpoint.
                name = (a if isinstance(a, str) else (a.get("name") or "")).strip()
                if not name:
                    continue
                row = agg[name]
                row["papers"] += 1
                row["tasks"][task_name] += 1
                if row["sample"] is None and title:
                    row["sample"] = (title, url)

    # Emit CSV. One row per author; per-task pivot in tasks_json.
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "papers_seen",
                "tasks_seen",
                "top_task",
                "top_task_papers",
                "sample_paper_title",
                "sample_paper_url",
                "tasks_json",
            ]
        )
        # Sort by total papers desc so the most-prolific authors are at
        # the top of the file — convenient for spot-checks.
        for name, row in sorted(agg.items(), key=lambda kv: kv[1]["papers"], reverse=True):
            tasks_dict = dict(row["tasks"])
            top_task, top_count = ("", 0)
            if tasks_dict:
                top_task, top_count = max(tasks_dict.items(), key=lambda kv: kv[1])
            sample = row["sample"] or ("", "")
            w.writerow(
                [
                    name,
                    row["papers"],
                    len(tasks_dict),
                    top_task,
                    top_count,
                    sample[0],
                    sample[1],
                    json.dumps(tasks_dict, sort_keys=True),
                ]
            )

    print(f"\n✓ wrote {out_path} ({len(agg):,} authors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
