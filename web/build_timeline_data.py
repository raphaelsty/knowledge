#!/usr/bin/env python3
"""Build the curated "impactful works" dataset for the chronological timeline.

Signal: a document is "impactful / agreed-upon" when it is referenced inside
the document collections of MANY different VIP people in the AI community.
We group near-identical documents by (normalised title, year), union the set
of people that reference them, and keep works referenced by at least
MIN_PEOPLE distinct people.

Output: web/data.json  (consumed by web/index.html)
"""
import json
import glob
import os
import re
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS_GLOB = os.path.join(ROOT, "data", "documents", "*.jsonl")
USERS = os.path.join(ROOT, "data", "users.jsonl")
OUT = os.path.join(HERE, "data.json")

MIN_PEOPLE = 5          # consensus threshold
MAX_WORKS = 600         # safety cap
YEAR_FLOOR = 2012       # deep-learning era; drops arXiv-id date artifacts

# Titles from blocked / error / login pages that leak into scrapes.
JUNK = re.compile(
    r"please wait|verification|just a moment|attention required|"
    r"are you a robot|captcha|page not found|404|not found|sign in|log ?in|"
    r"access denied|enable javascript|redirecting|loading\.\.\.",
    re.I,
)

# Citation / discovery infrastructure that is not itself an "impactful work".
INFRA = {
    "dblp", "scopus", "google colab", "semantic scholar", "crossref", "orcid",
    "researchgate", "association for computing machinery", "openalex",
    "google scholar", "arxiv.org", "papers with code", "connected papers",
}


def norm(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def categorize(doc) -> str:
    """Coarse type used for colour-coding in the visualization."""
    url = (doc.get("canonical_url") or doc.get("url") or "").lower()
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    title = (doc.get("title") or "")
    tl = title.lower()
    tags = " ".join(doc.get("tags") or []).lower()
    source = (doc.get("source") or "").lower()

    if "github.com" in url or "github" in source or "huggingface.co" in url:
        return "tool"
    if any(k in tl for k in ("system card", "technical report", "model card")):
        return "model"
    if "arxiv.org" in url or "aclanthology.org" in url or "openreview" in url:
        return "paper"
    if "wikipedia" in source or "wikipedia" in tags:
        return "reference"
    if any(k in host for k in ("youtube", "youtu.be")):
        return "talk"
    if "x.com" in host or "twitter" in host:
        return "post"
    if "blog" in url or "blog" in tags or source in ("substack",):
        return "blog"
    return "paper"


def display_title(doc) -> str:
    t = doc.get("clean_title") or doc.get("title") or ""
    t = re.sub(r"\s*[·|\-–—]\s*(GitHub|arXiv|Hugging Face|YouTube).*$", "", t)
    t = t.strip()
    # Drop trailing site noise like "· Change.org"
    return t[:140]


def main():
    # Optional: map slug -> display name for tooltips
    names = {}
    if os.path.exists(USERS):
        with open(USERS) as f:
            for line in f:
                try:
                    u = json.loads(line)
                    names[u["slug"]] = u.get("name") or u["slug"]
                except Exception:
                    pass

    people = collections.defaultdict(set)
    meta = {}
    best_len = collections.defaultdict(int)

    for fp in glob.glob(DOCS_GLOB):
        person = os.path.basename(fp)[:-6]
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                date = d.get("date") or ""
                year = date[:4]
                if not year.isdigit():
                    continue
                title = d.get("title") or ""
                nt = norm(title)
                if len(nt) < 3:
                    continue
                key = (nt, year)
                people[key].add(person)
                # keep the doc with the richest summary as representative
                summ = d.get("summary") or ""
                if len(summ) > best_len[key] or key not in meta:
                    meta[key] = d
                    best_len[key] = len(summ)

    works = []
    for key, ppl in people.items():
        if len(ppl) < MIN_PEOPLE:
            continue
        if int(key[1]) < YEAR_FLOOR:
            continue
        d = meta[key]
        date = d.get("date") or ""
        title = display_title(d)
        if not title or JUNK.search(title):
            continue
        if norm(title) in INFRA or categorize(d) == "reference":
            continue
        summary = (d.get("summary") or "").strip()
        summary = re.sub(r"\s+", " ", summary)[:320]
        works.append({
            "title": title,
            "date": date,
            "year": int(key[1]),
            "people": len(ppl),
            "names": sorted(names.get(s, s) for s in ppl)[:24],
            "type": categorize(d),
            "url": d.get("canonical_url") or d.get("url"),
            "source": d.get("source") or "",
            "summary": summary,
            "tags": (d.get("tags") or [])[:6],
        })

    works.sort(key=lambda w: (-w["people"], w["date"]))
    works = works[:MAX_WORKS]
    works.sort(key=lambda w: (w["date"], -w["people"]))

    payload = {
        "generated_from": "knowledge VIP data snapshot",
        "min_people": MIN_PEOPLE,
        "count": len(works),
        "works": works,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {OUT}: {len(works)} works "
          f"({works[0]['year']}–{works[-1]['year']})")


if __name__ == "__main__":
    main()
