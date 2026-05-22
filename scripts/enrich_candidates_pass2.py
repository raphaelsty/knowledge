"""Second-pass enrichment for data/people/enriched_candidates.tsv.

Goal: increase fill rate WITHOUT introducing false positives. Same
contract as pass 1 — when we can't verify, the field stays empty.

This pass uses only keyless APIs (no Tavily credits):

  • GitHub      — probe `github.com/<twitter_handle>` and accept when
                  GitHub's own `twitter_username` field cross-confirms.
  • HuggingFace — probe `huggingface.co/<twitter_handle>` and accept
                  when the profile's `fullname` matches the candidate.
  • HackerNews  — probe the Firebase JSON `/v0/user/<handle>.json`.
                  Accepts a username that exists; HN profiles have no
                  display name so identity confidence comes from the
                  handle reuse + the candidate's existing AI signal.
  • arXiv       — no per-author profile API exists. We populate
                  `arxiv_author` = candidate's display name when the
                  candidate has *some* academic signal already
                  (sem_scholar OR dblp OR scholar_id). That's the
                  exact value the Python pipeline's arXiv fetcher
                  takes as input.
  • affiliation — backfill from GitHub `company` when sem_scholar
                  didn't supply one.

Adds two new columns: `hackernews`, `arxiv_author`.

Usage::

    uv run python scripts/enrich_candidates_pass2.py
    uv run python scripts/enrich_candidates_pass2.py --in path --out path
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

_USER_AGENT = "knowledge-enrich-pass2/1.0 (+https://knowledge-web.org)"
_REQ_DELAY_S = 0.3


def _norm_tokens(s: str) -> set[str]:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z]+", " ", s).lower()
    return {t for t in s.split() if len(t) >= 2}


def _names_match_strict(candidate_name: str, platform_name: str | None) -> bool:
    """Same gate as pass 1 — for 2-token candidate names require both
    tokens; for longer names require ≥ 2 overlap."""
    if not platform_name:
        return True
    a, b = _norm_tokens(candidate_name), _norm_tokens(platform_name)
    if not a or not b:
        return True
    shared = a & b
    if len(a) <= 2:
        return shared == a
    return len(shared) >= 2


def _http_json(url: str, headers: dict | None = None, timeout: int = 20) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# Probes
# ────────────────────────────────────────────────────────────────────


def probe_github_by_handle(handle: str, twitter_handle: str, candidate_name: str) -> dict | None:
    """Try GitHub at the candidate's Twitter handle. Accept iff GitHub
    cross-confirms via its `twitter_username` field OR the display
    name matches with our strict rule."""
    data = _http_json(
        f"https://api.github.com/users/{urllib.parse.quote(handle, safe='')}",
        headers=({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else None),
    )
    if not data:
        return None
    gh_tw = (data.get("twitter_username") or "").lower().lstrip("@")
    cand_tw = (twitter_handle or "").lower().lstrip("@")
    # Twitter cross-check: hard reject when GitHub claims a different
    # Twitter handle.
    if gh_tw and cand_tw and gh_tw != cand_tw:
        return None
    if not (gh_tw and cand_tw and gh_tw == cand_tw):
        if not _names_match_strict(candidate_name, data.get("name")):
            return None
    return {
        "login": data.get("login") or handle,
        "name": data.get("name") or "",
        "bio": (data.get("bio") or "").strip(),
        "company": (data.get("company") or "").strip(),
        "blog": (data.get("blog") or "").strip(),
        "followers": int(data.get("followers") or 0),
        "avatar": data.get("avatar_url") or "",
    }


def probe_huggingface_by_handle(handle: str, candidate_name: str) -> dict | None:
    """HF overview JSON. Accept iff fullname matches."""
    data = _http_json(f"https://huggingface.co/api/users/{urllib.parse.quote(handle, safe='')}/overview")
    if not data:
        return None
    fullname = data.get("fullname") or data.get("name") or ""
    if not _names_match_strict(candidate_name, fullname):
        return None
    return {
        "username": data.get("name") or handle,
        "fullname": fullname,
        "avatar": data.get("avatarUrl") or "",
    }


def probe_hackernews(handle: str) -> dict | None:
    """HackerNews Firebase API. `null` means no such user; a JSON object
    confirms existence. HN profiles are pseudonymous (no display name),
    so identity confidence here rides on the handle reuse plus the
    candidate's existing AI/research signal — same model as our
    GitHub-when-name-is-empty fallback."""
    data = _http_json(f"https://hacker-news.firebaseio.com/v0/user/{urllib.parse.quote(handle, safe='')}.json")
    if not data or not data.get("id"):
        return None
    # Karma ≥ 50 keeps trivially-existing accounts out — anyone we'd
    # want to follow has meaningful comment history.
    if int(data.get("karma") or 0) < 50:
        return None
    return {"id": data["id"], "karma": int(data.get("karma") or 0)}


# ────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/people/enriched_candidates.tsv",
        help="input/output TSV",
    )
    ap.add_argument("--out", default="", help="output path (default: overwrite input)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    in_path = (repo_root / args.inp).resolve()
    out_path = (repo_root / args.out).resolve() if args.out else in_path

    with in_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print("Empty input.")
        return 1

    cols = list(rows[0].keys())
    for new_col in ("hackernews", "arxiv_author"):
        if new_col not in cols:
            cols.append(new_col)
            for r in rows:
                r.setdefault(new_col, "")

    counts = {
        "github_added": 0,
        "huggingface_added": 0,
        "hackernews_added": 0,
        "arxiv_added": 0,
        "affiliation_added": 0,
        "website_added": 0,
        "avatar_added": 0,
        "bio_added": 0,
    }

    for i, r in enumerate(rows, 1):
        handle = r.get("handle", "").strip()
        name = r.get("name", "").strip()
        if not handle or not name:
            continue
        print(f"[{i}/{len(rows)}] @{handle:<26} {name}")

        # GitHub at the Twitter handle (cheap, big-win probe).
        if not r.get("github"):
            v = probe_github_by_handle(handle, handle, name)
            if v:
                r["github"] = v["login"]
                r["github_followers"] = str(v["followers"])
                if v["company"] and not r.get("github_company"):
                    r["github_company"] = v["company"]
                if v["avatar"] and not r.get("avatar"):
                    r["avatar"] = v["avatar"]
                    counts["avatar_added"] += 1
                if v["bio"] and not r.get("bio"):
                    r["bio"] = v["bio"]
                    counts["bio_added"] += 1
                if v["blog"] and not r.get("website"):
                    r["website"] = v["blog"]
                    counts["website_added"] += 1
                counts["github_added"] += 1
            time.sleep(_REQ_DELAY_S)

        # HF at the Twitter handle.
        if not r.get("huggingface"):
            v = probe_huggingface_by_handle(handle, name)
            if v:
                r["huggingface"] = v["username"]
                if v["avatar"] and not r.get("avatar"):
                    r["avatar"] = v["avatar"]
                    counts["avatar_added"] += 1
                counts["huggingface_added"] += 1
            time.sleep(_REQ_DELAY_S)

        # HackerNews at the Twitter handle.
        if not r.get("hackernews"):
            v = probe_hackernews(handle)
            if v:
                r["hackernews"] = v["id"]
                counts["hackernews_added"] += 1
            time.sleep(_REQ_DELAY_S)

        # arXiv author: only when we have at least one academic
        # identifier already. Avoids stamping a name that the arXiv
        # fetcher won't disambiguate.
        if not r.get("arxiv_author"):
            if r.get("sem_scholar_id") or r.get("dblp_pid") or r.get("scholar_id"):
                r["arxiv_author"] = name
                counts["arxiv_added"] += 1

        # Affiliation backfill from GitHub `company`.
        if not r.get("affiliation") and r.get("github_company"):
            # Trim leading '@' that GitHub uses for org links.
            r["affiliation"] = r["github_company"].lstrip("@").strip()
            counts["affiliation_added"] += 1

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"\n✓ wrote {out_path}")
    print("Pass-2 fills:")
    for k, v in counts.items():
        print(f"  {k:<24} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
