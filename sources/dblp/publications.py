"""
DBLP computer science bibliography fetcher.

Fetches publications from DBLP, the definitive CS publication database.
Uses canonical URLs (arXiv > DOI > DBLP page) to deduplicate with
Semantic Scholar and Google Scholar.

API docs: https://dblp.org/faq/How+to+use+the+dblp+search+API.html
"""

import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

__all__ = ["Publications"]

_API = "https://dblp.org/search/publ/api"
_DELAY = 1.0


def _fetch_xml(url: str) -> ET.Element:
    req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return ET.fromstring(resp.read())


def _canonical_url(info: dict) -> str:
    """Pick best URL: arXiv > DOI > DBLP page."""
    ee = info.get("ee", "")
    if "arxiv.org" in ee:
        m = re.search(r"(\d{4}\.\d{4,5})", ee)
        if m:
            return f"https://arxiv.org/abs/{m.group(1)}"
    if "doi.org" in ee:
        return ee
    return info.get("url", ee)


class Publications:
    """Fetch publications from DBLP by author name."""

    def __init__(self, author: str, max_results: int = 200):
        self.author = author
        self.max_results = max_results

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        print(f"    Fetching DBLP publications for '{self.author}'...")
        data: dict[str, dict] = {}
        offset = 0
        page_size = 100

        while offset < self.max_results:
            url = f"{_API}?q=author%3A{urllib.parse.quote(self.author)}&h={page_size}&f={offset}&format=json"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read())
            except Exception as e:
                print(f"    DBLP API error: {e}")
                break

            hits = result.get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                break

            # Page-level early exit: DBLP returns by relevance, not pure date, so
            # don't hard-break on first all-known page. Instead stop when we've
            # seen two consecutive all-known pages.
            if existing_urls is not None:
                new_in_page = 0
                for h in hits:
                    info = h.get("info", {})
                    canonical = _canonical_url(info)
                    if canonical and canonical not in existing_urls:
                        new_in_page += 1
                if new_in_page == 0:
                    print(f"    No new publications on page offset={offset}, stopping early.")
                    break
                print(f"    Offset {offset}: {new_in_page} new publication(s).")

            for hit in hits:
                info = hit.get("info", {})
                title = info.get("title", "").rstrip(".")
                if not title:
                    continue

                canonical = _canonical_url(info)
                if not canonical:
                    continue
                if existing_urls and canonical in existing_urls:
                    continue
                if canonical in data:
                    continue

                year = info.get("year", "")
                # `venue` is normally a string but DBLP returns a list
                # for cross-referenced publications (workshops inside
                # bigger venues, common for prolific authors like
                # Bengio). Flatten to a single comma-separated string
                # so the downstream `". ".join` doesn't choke on a
                # nested list.
                venue_raw = info.get("venue", "")
                if isinstance(venue_raw, list):
                    venue = ", ".join(str(v) for v in venue_raw if v)
                else:
                    venue = str(venue_raw or "")
                authors_raw = info.get("authors", {}).get("author", [])
                if isinstance(authors_raw, dict):
                    authors_raw = [authors_raw]

                # Each author may be a string OR a {"text": str, "@pid": ...} dict.
                # Coerce defensively so any unexpected shape (list-of-strings,
                # nested dict) becomes a flat string.
                def _author_str(a):
                    if isinstance(a, dict):
                        t = a.get("text", "")
                        return str(t) if not isinstance(t, list) else ", ".join(str(x) for x in t)
                    return str(a)

                authors = ", ".join(_author_str(a) for a in authors_raw[:5])

                summary_parts = []
                if authors:
                    summary_parts.append(authors)
                if venue:
                    summary_parts.append(venue)
                summary = ". ".join(summary_parts)
                if len(summary) > 250:
                    summary = summary[:247] + "..."

                data[canonical] = {
                    "title": title,
                    "summary": summary,
                    "date": f"{year}-01-01" if year else "",
                    "tags": ["scholar"],
                }

            offset += len(hits)
            if len(hits) < page_size:
                break
            time.sleep(_DELAY)

        print(f"    {len(data)} publications (deduplicated)")
        return data
