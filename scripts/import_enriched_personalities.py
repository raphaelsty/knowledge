"""Import enriched candidates into `users` as VIP personalities.

Reads data/people/enriched_candidates.tsv and INSERTs one row per
candidate. Each row is marked `vip=true public=true`. Sources and
links are reconstructed from the enrichment columns so the existing
Python pipeline can pick the personality up on its next run.

Maps enrichment column → `users` column:

    handle              → sources.twitter.username, links.twitter
    name                → users.name
    avatar              → users.avatar
    bio                 → users.description (when present)
    github              → sources.github.username, links.github
    huggingface         → sources.huggingface.username, links.huggingface
    hackernews          → sources.hackernews.username, links.hackernews
    dblp_pid            → sources.dblp.author (= candidate name)
    scholar_id          → sources.scholar.user_id, links.scholar
    arxiv_author        → sources.arxiv.author
    website             → sources.websites.urls[0], links.website
    sem_scholar_*       → users.citations (when present)
    affiliation         → discarded (no native column; pipeline mines
                          it from sources at run-time)
    specialty           → kept in `description` as a brief blurb when
                          the candidate has no GitHub bio

Slug is derived from the display name with the same rules the Rust
`personalities::create` endpoint uses (lowercase ASCII alphanumerics,
non-ASCII collapsed to dashes, leading/trailing dashes trimmed).

`ON CONFLICT (username) DO NOTHING` means re-running is safe; only
new rows land. We print a summary at the end.

Usage::

    DATABASE_URL=... uv run python scripts/import_enriched_personalities.py
    DATABASE_URL=... uv run python scripts/import_enriched_personalities.py --dry
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import unicodedata
from pathlib import Path

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"


def slugify(s: str) -> str:
    """Mirror of `personalities::create::slugify` in the Rust API."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    out: list[str] = []
    last_dash = True
    for c in s:
        if c.isalnum() and c.isascii():
            out.append(c.lower())
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    while out and out[-1] == "-":
        out.pop()
    return "".join(out)


def build_sources(r: dict) -> dict:
    s: dict = {}
    handle = (r.get("handle") or "").strip().lstrip("@")
    if handle:
        s["twitter"] = {"username": handle}
    if r.get("github"):
        s["github"] = {"username": r["github"].strip()}
    if r.get("huggingface"):
        s["huggingface"] = {"username": r["huggingface"].strip()}
    if r.get("hackernews"):
        s["hackernews"] = {"username": r["hackernews"].strip()}
    if r.get("scholar_id"):
        s["scholar"] = {"user_id": r["scholar_id"].strip()}
    if r.get("dblp_pid"):
        # Pipeline's DBLP fetcher uses the display name; the pid is
        # stored as a stable disambiguator (we keep both).
        s["dblp"] = {"author": (r.get("name") or "").strip(), "pid": r["dblp_pid"].strip()}
    if r.get("arxiv_author"):
        s["arxiv"] = {"author": r["arxiv_author"].strip()}
    if r.get("website"):
        s["websites"] = {"urls": [r["website"].strip()]}
    return s


def build_links(r: dict) -> dict:
    links: dict = {}
    handle = (r.get("handle") or "").strip().lstrip("@")
    if handle:
        links["twitter"] = f"https://x.com/{handle}"
    if r.get("github"):
        links["github"] = f"https://github.com/{r['github'].strip()}"
    if r.get("huggingface"):
        links["huggingface"] = f"https://huggingface.co/{r['huggingface'].strip()}"
    if r.get("hackernews"):
        links["hackernews"] = f"https://news.ycombinator.com/user?id={r['hackernews'].strip()}"
    if r.get("scholar_id"):
        links["scholar"] = f"https://scholar.google.com/citations?user={r['scholar_id'].strip()}"
    if r.get("website"):
        links["website"] = r["website"].strip()
    return links


def build_description(r: dict) -> str:
    """Prefer the GitHub bio; fall back to a one-liner that records
    the curated specialty so the profile isn't blank."""
    bio = (r.get("bio") or "").strip()
    if bio:
        return bio
    specialty = (r.get("specialty") or "").strip()
    if specialty:
        return f"AI researcher / engineer — {specialty}."
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/people/enriched_candidates.tsv",
        help="input TSV",
    )
    ap.add_argument("--dry", action="store_true", help="print plan, no DB writes")
    args = ap.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    repo_root = Path(__file__).resolve().parent.parent
    in_path = (repo_root / args.inp).resolve()
    with in_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print("Empty input.")
        return 1
    print(f"Read {len(rows)} candidates from {in_path}\n")

    # Pre-load existing slugs/handles so we can predict collisions and
    # print a clean summary before hitting the unique constraint.
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT LOWER(username), LOWER(sources->'twitter'->>'username') FROM users")
            existing_slugs = set()
            existing_handles = set()
            for slug, handle in cur.fetchall():
                if slug:
                    existing_slugs.add(slug)
                if handle:
                    existing_handles.add(handle.lstrip("@"))

    seen_slugs_this_batch: set[str] = set()
    plan: list[tuple[str, dict]] = []

    skipped_dup_db = 0
    skipped_dup_batch = 0
    for r in rows:
        name = (r.get("name") or "").strip()
        handle = (r.get("handle") or "").strip().lstrip("@").lower()
        if not name or not handle:
            continue

        slug = slugify(name)
        if not slug:
            continue

        # Skip when slug or twitter handle is already in DB.
        if slug in existing_slugs or handle in existing_handles:
            skipped_dup_db += 1
            continue

        # Skip duplicates within this batch (rare — same display name
        # producing same slug for two handles).
        if slug in seen_slugs_this_batch:
            skipped_dup_batch += 1
            continue
        seen_slugs_this_batch.add(slug)

        plan.append((slug, r))

    print(f"Plan: {len(plan)} to import; {skipped_dup_db} already in DB; {skipped_dup_batch} intra-batch dups.\n")

    if args.dry:
        for slug, r in plan[:10]:
            print(f"  {slug:<32} @{r['handle']:<24} {r['name']}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more")
        return 0

    inserted = 0
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for slug, r in plan:
                sources = build_sources(r)
                links = build_links(r)
                description = build_description(r)
                avatar = (r.get("avatar") or "").strip() or None
                try:
                    citations: int | None = int(r["sem_scholar_citations"]) if r.get("sem_scholar_citations") else None
                except ValueError:
                    citations = None
                cur.execute(
                    """
                    INSERT INTO users (
                        username, name, description, index_name,
                        public, vip, sources, links,
                        avatar, citations
                    ) VALUES (
                        %s, %s, %s, %s,
                        TRUE, TRUE, %s::jsonb, %s::jsonb,
                        %s, %s
                    )
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (
                        slug,
                        r["name"],
                        description,
                        slug,
                        json.dumps(sources),
                        json.dumps(links),
                        avatar,
                        citations,
                    ),
                )
                if cur.rowcount:
                    inserted += 1
            conn.commit()

    print(f"\n✓ inserted {inserted} new VIPs (of {len(plan)} planned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
