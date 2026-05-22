"""
arXiv author search.

Searches arXiv for papers by a specific author name.
Uses canonical arXiv abs/ URLs to deduplicate with Scholar sources.
Free API, returns full abstracts and categories.
"""

import re
import time
import urllib.request
import xml.etree.ElementTree as ET

__all__ = ["Papers"]

_API = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_DELAY = 3.0  # arXiv asks for 3s between requests
_ARXIV_ID_RE = re.compile(r"/abs/(\d{4}\.\d{4,5})")


class Papers:
    """Fetch papers from arXiv by author name."""

    def __init__(self, author: str, max_results: int = 200):
        self.author = author
        self.max_results = max_results

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        print(f"    Fetching arXiv papers for '{self.author}'...")
        data: dict[str, dict] = {}
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
                else:
                    canonical = raw_url.replace("http://", "https://")

                if existing_urls and canonical in existing_urls:
                    continue
                if canonical in data:
                    continue

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

        print(f"    {len(data)} papers (deduplicated)")
        return data
