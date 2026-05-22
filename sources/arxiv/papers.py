"""
arXiv author search.

Searches arXiv for papers by a specific author name.
Uses canonical arXiv abs/ URLs to deduplicate with Scholar sources.
Free API, returns full abstracts and categories.

After the arXiv API call returns the paper list, we make one or more
batch calls to Semantic Scholar's `/graph/v1/paper/batch` endpoint to
fetch citation counts for the same arXiv IDs — the arXiv API itself
doesn't expose citation data. The sidecar is best-effort: a failed S2
lookup leaves `citation_count` absent on the doc (so the column stays
NULL and a later run retries).
"""

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

__all__ = ["Papers"]

_API = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_DELAY = 3.0  # arXiv asks for 3s between requests
_ARXIV_ID_RE = re.compile(r"/abs/(\d{4}\.\d{4,5})")

# Semantic Scholar bulk endpoint. Free tier accepts up to 500 IDs per
# call and is rate-limited at ~1 req/sec — well within what an
# author's catalogue needs (most authors have <500 papers, so one
# call covers the lot).
_S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_BATCH_SIZE = 500
_S2_TIMEOUT = 30


def _fetch_arxiv_citations(arxiv_ids: list[str]) -> dict[str, int]:
    """Return ``{arxiv_id: citationCount}`` for the supplied IDs via S2.

    Batches up to 500 IDs per call. Failures (network, 4xx, 5xx, rate
    limit) return an empty dict for that batch — the caller treats
    missing IDs as "not measured" so the column stays NULL and a later
    run can retry.

    Caller responsibility: pass plain arXiv IDs (e.g. ``"2106.09685"``),
    not full URLs. We prefix with ``ARXIV:`` here as the S2 batch
    endpoint requires.
    """
    out: dict[str, int] = {}
    if not arxiv_ids:
        return out
    for start in range(0, len(arxiv_ids), _S2_BATCH_SIZE):
        chunk = arxiv_ids[start : start + _S2_BATCH_SIZE]
        body = json.dumps({"ids": [f"ARXIV:{aid}" for aid in chunk]}).encode("utf-8")
        # `fields` is a query parameter, not a body field. Keep the
        # request lean: just the two fields we need.
        url = f"{_S2_BATCH}?fields=externalIds,citationCount"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "User-Agent": "Knowledge/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_S2_TIMEOUT) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            print(f"    S2 citation lookup failed ({len(chunk)} ids): {exc}")
            continue
        if not isinstance(payload, list):
            continue
        for entry in payload:
            # S2 returns null entries for IDs it doesn't know — skip
            # them silently; the column stays NULL on those docs.
            if not isinstance(entry, dict):
                continue
            ext = entry.get("externalIds") or {}
            aid = ext.get("ArXiv")
            cc = entry.get("citationCount")
            if aid and cc is not None:
                try:
                    out[str(aid)] = int(cc)
                except (TypeError, ValueError):
                    pass
        # 1 req/sec is the documented free-tier ceiling; the batch
        # itself is cheap server-side so a single second between
        # chunks keeps us comfortably under the limit.
        if start + _S2_BATCH_SIZE < len(arxiv_ids):
            time.sleep(1.0)
    return out


class Papers:
    """Fetch papers from arXiv by author name."""

    def __init__(self, author: str, max_results: int = 200):
        self.author = author
        self.max_results = max_results

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        print(f"    Fetching arXiv papers for '{self.author}'...")
        data: dict[str, dict] = {}
        # Side-table keyed on canonical URL → arxiv id (the 10-digit
        # form `2106.09685`). Built as we walk arXiv's response so the
        # S2 citation sidecar at the bottom of this method can map
        # results back to docs without re-parsing URLs.
        arxiv_ids_by_url: dict[str, str] = {}
        start = 0
        page_size = 100

        while start < self.max_results:
            # Quote the author name for exact match (avoid "Max Halford" matching "Max Ronecker")
            query = urllib.parse.quote(f'au:"{self.author}"')
            url = f"{_API}?search_query={query}&start={start}&max_results={page_size}&sortBy=submittedDate&sortOrder=descending"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    root = ET.fromstring(resp.read())
            except Exception as exc:
                print(f"    arXiv API error: {exc}")
                break

            entries = root.findall("atom:entry", _NS)
            if not entries:
                break

            # Page-level early exit: arXiv is sorted by submittedDate desc, so
            # once every paper on a page is known we've caught up.
            if existing_urls is not None:
                new_in_page = 0
                for e in entries:
                    lid = e.find("atom:id", _NS)
                    if lid is None or not lid.text:
                        continue
                    raw_url = lid.text.strip()
                    m = _ARXIV_ID_RE.search(raw_url)
                    canonical = f"https://arxiv.org/abs/{m.group(1)}" if m else raw_url.replace("http://", "https://")
                    if canonical not in existing_urls:
                        new_in_page += 1
                if new_in_page == 0:
                    print(f"    No new papers on page starting {start}, stopping early.")
                    break
                print(f"    Page start={start}: {new_in_page} new paper(s).")

            for entry in entries:
                # Get canonical arXiv URL
                link_el = entry.find("atom:id", _NS)
                if link_el is None or not link_el.text:
                    continue
                raw_url = link_el.text.strip()
                m = _ARXIV_ID_RE.search(raw_url)
                if m:
                    canonical = f"https://arxiv.org/abs/{m.group(1)}"
                    aid = m.group(1)
                else:
                    canonical = raw_url.replace("http://", "https://")
                    aid = ""

                if existing_urls and canonical in existing_urls:
                    continue
                if canonical in data:
                    continue
                if aid:
                    arxiv_ids_by_url[canonical] = aid

                title_el = entry.find("atom:title", _NS)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

                abstract_el = entry.find("atom:summary", _NS)
                abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""
                if len(abstract) > 250:
                    abstract = abstract[:247] + "..."

                published_el = entry.find("atom:published", _NS)
                date = (published_el.text or "")[:10] if published_el is not None else ""

                # Categories
                categories = []
                for cat in entry.findall("arxiv:primary_category", _NS):
                    term = cat.get("term", "")
                    if term:
                        categories.append(term)

                data[canonical] = {
                    "title": title,
                    "summary": abstract,
                    "date": date,
                    "tags": ["arxiv"] + categories,
                }

            start += len(entries)
            if len(entries) < page_size:
                break
            time.sleep(_DELAY)

        # S2 citation sidecar. One batch covers up to 500 papers; an
        # author with more than that pays the (~1s/batch) penalty
        # once per backfill run. Best-effort — partial / failed
        # lookups leave `citation_count` absent on the doc, which the
        # upsert turns into a NULL column, which a later run retries.
        if arxiv_ids_by_url:
            print(f"    Fetching citation counts for {len(arxiv_ids_by_url)} paper(s) via Semantic Scholar...")
            citations = _fetch_arxiv_citations(list(arxiv_ids_by_url.values()))
            if citations:
                hits = 0
                for url, aid in arxiv_ids_by_url.items():
                    cc = citations.get(aid)
                    if cc is not None and url in data:
                        data[url]["citation_count"] = cc
                        hits += 1
                print(f"    Got citations for {hits}/{len(arxiv_ids_by_url)} paper(s)")
            else:
                print("    No citations returned (S2 unreachable or all unknown)")

        print(f"    {len(data)} papers (deduplicated)")
        return data
