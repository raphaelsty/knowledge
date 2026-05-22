"""Pull author signals from Semantic Scholar's bulk paper-search API,
per AI subfield, and write one joined CSV.

Strategy
--------
Semantic Scholar's `fieldsOfStudy` is too coarse ("Computer Science")
to give us per-area people lists. So we drive the bulk search with a
small dictionary of subfield → query string, ranked by citation count
desc, and aggregate authors across the result set.

Endpoint:
    https://api.semanticscholar.org/graph/v1/paper/search/bulk
    ?query=<subfield_query>
    &fieldsOfStudy=Computer Science
    &minCitationCount=<...>
    &sort=citationCount:desc
    &fields=paperId,title,year,citationCount,authors,url,externalIds

The free tier is keyless but rate-limited (~100 req/5 min). We pace
ourselves with `_REQ_DELAY_S` and a small max-pages cap per subfield.

Output CSV columns:

    name, author_id, papers_seen, total_citations, subfields_seen,
    top_subfield, top_subfield_papers, sample_paper_title,
    sample_paper_url, subfields_json

`subfields_json` = `{subfield: papers}` so the matcher has every
per-area signal.

Usage::

    uv run python scripts/fetch_semantic_scholar_authors.py
    uv run python scripts/fetch_semantic_scholar_authors.py --max-papers-per-subfield 300
    uv run python scripts/fetch_semantic_scholar_authors.py --out data/people/sem_scholar_authors.csv
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

_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_USER_AGENT = "knowledge-semanticscholar/1.0 (+https://knowledge-web.org)"
# S2 free tier: be conservative. 1 req per 2s easily stays under any
# documented limit and lets the script run alongside other workloads.
_REQ_DELAY_S = 1.5

# Subfield → search query. Queries are deliberately broad ("OR"-ish)
# so the citation-count sort surfaces seminal work rather than narrow
# niche papers. The keys become the per-subfield column in the
# aggregated CSV.
_SUBFIELDS: dict[str, str] = {
    "Natural Language Processing": "natural language processing OR language model OR machine translation",
    "Computer Vision": "computer vision OR image recognition OR object detection",
    "Reinforcement Learning": "reinforcement learning",
    "Machine Learning Theory": "generalization OR optimization OR neural network theory",
    "Generative Models": "diffusion model OR generative adversarial OR variational autoencoder",
    "Speech & Audio": "speech recognition OR audio model OR text-to-speech",
    "Robotics & Embodied AI": "robot learning OR manipulation OR embodied agent",
    "Recommender Systems": "recommender system OR collaborative filtering",
    "Information Retrieval": "neural retrieval OR dense retrieval OR ranking",
    "Graph Learning": "graph neural network OR graph representation",
    "Multimodal Learning": "vision language OR multimodal model OR CLIP",
    "Efficiency & Systems for ML": "efficient inference OR model distillation OR quantization OR mixture of experts",
    "AI Safety & Alignment": "AI alignment OR reward model OR RLHF OR red teaming",
}


def _get_json(url: str) -> dict | None:
    """One GET; parses JSON or returns None on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            if resp.status != 200:
                print(f"    ! HTTP {resp.status} for {url}")
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"    ! fetch error: {e}")
        return None


def _fetch_subfield(query: str, max_papers: int, min_citations: int) -> list[dict]:
    """Bulk-search S2 for one subfield. Walks `token`-based pagination
    until we hit `max_papers` or the server runs out."""
    fields = "paperId,title,year,citationCount,authors.authorId,authors.name,url,externalIds"
    base = {
        "query": query,
        "fieldsOfStudy": "Computer Science",
        "minCitationCount": str(min_citations),
        "sort": "citationCount:desc",
        "fields": fields,
    }
    out: list[dict] = []
    token: str | None = None
    while len(out) < max_papers:
        params = dict(base)
        if token:
            params["token"] = token
        url = f"{_API}?{urllib.parse.urlencode(params)}"
        page = _get_json(url)
        if not page:
            break
        data = page.get("data") or []
        if not data:
            break
        out.extend(data)
        token = page.get("token")
        if not token:
            break
        time.sleep(_REQ_DELAY_S)
    return out[:max_papers]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="data/people/sem_scholar_authors.csv",
        help="output CSV path",
    )
    ap.add_argument(
        "--max-papers-per-subfield",
        type=int,
        default=400,
        help="Cap on top-cited papers fetched per subfield (default 400).",
    )
    ap.add_argument(
        "--min-citations",
        type=int,
        default=20,
        help="Server-side filter — drop papers with fewer citations (default 20).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # author_key → {
    #   "name": str,
    #   "author_id": str | None,
    #   "papers": int,
    #   "citations": int,           # citations contributed by papers we saw
    #   "subfields": Counter,
    #   "sample": (title, url),
    # }
    #
    # Key on author_id when present (stable), else fall back to name.
    agg: dict[str, dict] = {}

    def _key(a: dict) -> str:
        return a.get("authorId") or f"name::{(a.get('name') or '').strip().lower()}"

    for i, (subfield, query) in enumerate(_SUBFIELDS.items(), 1):
        print(f"\n[{i}/{len(_SUBFIELDS)}] {subfield}")
        papers = _fetch_subfield(query, args.max_papers_per_subfield, args.min_citations)
        print(f"    {len(papers)} papers")
        for p in papers:
            title = (p.get("title") or "").strip()
            url = (p.get("url") or "").strip()
            cit = int(p.get("citationCount") or 0)
            for a in p.get("authors") or []:
                name = (a.get("name") or "").strip()
                if not name:
                    continue
                k = _key(a)
                row = agg.get(k)
                if row is None:
                    row = {
                        "name": name,
                        "author_id": a.get("authorId") or "",
                        "papers": 0,
                        "citations": 0,
                        "subfields": defaultdict(int),
                        "sample": None,
                    }
                    agg[k] = row
                row["papers"] += 1
                row["citations"] += cit
                row["subfields"][subfield] += 1
                if row["sample"] is None and title:
                    row["sample"] = (title, url)
        time.sleep(_REQ_DELAY_S)

    # Emit CSV — one row per author, sorted by citations contributed
    # so the most-influential authors land at the top of the file.
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "author_id",
                "papers_seen",
                "total_citations",
                "subfields_seen",
                "top_subfield",
                "top_subfield_papers",
                "sample_paper_title",
                "sample_paper_url",
                "subfields_json",
            ]
        )
        for row in sorted(agg.values(), key=lambda r: r["citations"], reverse=True):
            sf = dict(row["subfields"])
            top_sf, top_count = ("", 0)
            if sf:
                top_sf, top_count = max(sf.items(), key=lambda kv: kv[1])
            sample = row["sample"] or ("", "")
            w.writerow(
                [
                    row["name"],
                    row["author_id"],
                    row["papers"],
                    row["citations"],
                    len(sf),
                    top_sf,
                    top_count,
                    sample[0],
                    sample[1],
                    json.dumps(sf, sort_keys=True),
                ]
            )

    print(f"\n✓ wrote {out_path} ({len(agg):,} authors across {len(_SUBFIELDS)} subfields)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
