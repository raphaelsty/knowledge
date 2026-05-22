"""
Google Scholar profile scraper.

Fetches publications from a public Google Scholar profile page.
Extracts title, authors, year, and citation count for each paper.

Be very gentle: Google Scholar aggressively rate-limits scrapers.
We fetch at most a few pages with long delays between requests.
"""

import re
import time
import urllib.request
from html import unescape

__all__ = ["Publications"]

_BASE = "https://scholar.google.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Delay between pagination requests (seconds) — be very polite
_DELAY = 5.0
_PAGE_SIZE = 100

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


_TITLE_YEAR_RE = re.compile(r"(?:^|[^\d])(19[5-9]\d|20[0-3]\d)(?:[^\d]|$)")


def _year_from_title(title: str) -> str:
    """Pull a 4-digit publication year out of a title string.

    Scholar embeds the year in citations like
    "creme, a Python library for online machine learning, 2019"
    or "BERT: Pre-training… (2018)". We accept any 4-digit year in
    the range 1950–2039 surrounded by non-digit context, returning
    the FIRST match (titles rarely contain more than one and the
    publication year is almost always the leading one). Returns ""
    when no plausible year is found.
    """
    if not title:
        return ""
    m = _TITLE_YEAR_RE.search(title)
    return m.group(1) if m else ""


def _clean(text: str) -> str:
    text = unescape(text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _fetch_page(user_id: str, start: int = 0) -> str:
    """Fetch a Google Scholar profile page."""
    url = f"{_BASE}/citations?user={user_id}&hl=en&cstart={start}&pagesize={_PAGE_SIZE}&sortby=pubdate"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_publications(html: str, user_id: str) -> list[dict]:
    """Extract publications from a Scholar profile page."""
    pubs = []
    rows = re.findall(r"<tr class=\"gsc_a_tr\">(.*?)</tr>", html, re.DOTALL)

    for row in rows:
        # Title
        title_m = re.search(r"class=\"gsc_a_at\"[^>]*>(.*?)</a>", row)
        if not title_m:
            continue
        title = _clean(title_m.group(1))

        # Citation page link (relative)
        href_m = re.search(r"href=\"(/citations\?[^\"]+)\"[^>]*class=\"gsc_a_at\"", row)
        if not href_m:
            href_m = re.search(r"<a\s+href=\"(/citations\?[^\"]+)\"", row)
        citation_url = f"{_BASE}{unescape(href_m.group(1))}" if href_m else ""

        # Authors and venue (two gs_gray divs)
        gray = re.findall(r"class=\"gs_gray\">(.*?)</div>", row)
        authors = _clean(gray[0]) if len(gray) > 0 else ""
        venue = _clean(gray[1]) if len(gray) > 1 else ""

        # Year
        year_m = re.search(r"class=\"gsc_a_y\">.*?<span[^>]*>(\d{4})</span>", row, re.DOTALL)
        year = year_m.group(1) if year_m else ""

        # Citation count
        cite_m = re.search(r"class=\"gsc_a_ac.*?\">\s*(\d+)", row)
        citations = int(cite_m.group(1)) if cite_m else 0

        pubs.append(
            {
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": year,
                "citations": citations,
                "url": citation_url,
            }
        )

    return pubs


class Publications:
    """
    Fetch publications from a Google Scholar profile.

    Parameters
    ----------
    user_id : str
        Google Scholar user ID (from the profile URL, e.g. "WLN3QrAAAAAJ").
    max_pages : int, default=3
        Maximum number of pages to fetch (100 pubs per page).
        Keep low to avoid being blocked.
    min_citations : int, default=0
        Only include papers with at least this many citations.
    """

    def __init__(self, user_id: str, max_pages: int = 3, min_citations: int = 0):
        self.user_id = user_id
        self.max_pages = max_pages
        self.min_citations = min_citations

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """Fetch publications and return document metadata."""
        print(f"    Fetching Google Scholar profile: {self.user_id}")
        data: dict[str, dict] = {}

        for page in range(self.max_pages):
            start = page * _PAGE_SIZE

            try:
                html = _fetch_page(self.user_id, start)
            except Exception as e:
                print(f"    Scholar fetch error (page {page}): {e}")
                break

            pubs = _parse_publications(html, self.user_id)
            if not pubs:
                break

            # Page-level early exit: Scholar returns by citation count by default,
            # but when sorted by date (the default when no explicit sort specified
            # in user profiles) the most-recent first ordering means once all
            # pubs on a page are known, older pages are too.
            if existing_urls is not None:
                new_in_page = sum(1 for p in pubs if p.get("url") and p["url"] not in existing_urls)
                if new_in_page == 0:
                    print(f"    No new publications on page {page + 1}, stopping early.")
                    break

            for pub in pubs:
                if self.min_citations > 0 and pub["citations"] < self.min_citations:
                    continue

                url = pub["url"]
                if not url:
                    continue
                if existing_urls and url in existing_urls:
                    continue

                summary_parts = []
                if pub["authors"]:
                    summary_parts.append(pub["authors"])
                if pub["venue"]:
                    summary_parts.append(pub["venue"])
                if pub["citations"]:
                    summary_parts.append(f"{pub['citations']} citations")

                # Scholar's main column gives us `pub['year']` directly, but
                # ~1% of entries (older DOI / Zenodo citations) ship without
                # it. The year is almost always in the title itself
                # (e.g. "…, 2019" or "…(2024)"), so fall back to a regex
                # rather than dropping the doc to NULL.
                year = pub["year"] or _year_from_title(pub["title"])
                data[url] = {
                    "title": pub["title"],
                    "summary": ". ".join(summary_parts),
                    "date": f"{year}-01-01" if year else "",
                    "tags": ["scholar"],
                }

            print(f"    Page {page + 1}: {len(pubs)} publications")

            if len(pubs) < _PAGE_SIZE:
                break  # last page

            if page < self.max_pages - 1:
                time.sleep(_DELAY)

        print(f"    Total: {len(data)} publications")
        return data
