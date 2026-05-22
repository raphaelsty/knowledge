"""
Semantic Scholar paper fetcher.

Uses the public Semantic Scholar API to fetch an author's papers.
Produces canonical URLs (arXiv > DOI > S2 page) to avoid duplicates
with papers already in the database from other sources (Google Scholar,
blog links to arXiv, etc.).

API docs: https://api.semanticscholar.org/api-docs/graph
Rate limit: 100 req/5min without API key.
"""

import json
import re
import time
import urllib.request

__all__ = ["Papers"]

_API_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,year,citationCount,externalIds,abstract,url,venue,publicationTypes"
_DELAY = 1.0  # seconds between requests
_PAGE_SIZE = 100

_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")

# The Semantic Scholar author API over-attributes non-publication artifacts
# (software releases, datasets). Two principled signals we use to reject them
# — works for every personality, not just the one where we noticed it:
#
#   1. `publicationTypes` explicitly marks non-papers as Dataset/Software/Book.
#   2. Titles matching ``<org>/<repo>: <version>`` are the canonical shape of
#      GitHub→Zenodo software releases (e.g. ``nipy/nipype: 1.8.3``). No
#      real paper title has this structure.
_NON_PUBLICATION_TYPES = frozenset({"Dataset", "Software", "Book"})
_SOFTWARE_RELEASE_TITLE_RE = re.compile(r"^[\w.-]+/[\w.-]+:\s*v?\d+(?:\.\d+)+")


def _is_noise(paper: dict) -> bool:
    """Return True for entries we should NOT index."""
    for t in paper.get("publicationTypes") or []:
        if t in _NON_PUBLICATION_TYPES:
            return True
    title = (paper.get("title") or "").strip()
    if _SOFTWARE_RELEASE_TITLE_RE.match(title):
        return True
    return False


def _fetch_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _canonical_url(paper: dict) -> str:
    """Pick the best URL for a paper: arXiv > DOI > Semantic Scholar page."""
    ext = paper.get("externalIds") or {}

    # Prefer arXiv (most useful for ML/CS papers)
    arxiv_id = ext.get("ArXiv", "")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"

    # DOI link
    doi = ext.get("DOI", "")
    if doi:
        return f"https://doi.org/{doi}"

    # Fallback to Semantic Scholar page
    return paper.get("url") or ""


class Papers:
    """
    Fetch papers from Semantic Scholar by author ID or name search.

    Uses canonical URLs (arXiv preferred) to avoid duplicates with
    papers already in the database from Google Scholar, blog links, etc.

    Parameters
    ----------
    author_id : str | None
        Semantic Scholar author ID (numeric string). If provided, fetches directly.
    author_name : str | None
        Author name to search for. Uses the top result's ID.
    max_papers : int, default=300
        Maximum number of papers to fetch.
    min_citations : int, default=0
        Only include papers with at least this many citations.
    """

    def __init__(
        self,
        author_id: str | None = None,
        author_name: str | None = None,
        max_papers: int = 300,
        min_citations: int = 0,
    ):
        self.author_id = author_id
        self.author_name = author_name
        self.max_papers = max_papers
        self.min_citations = min_citations

    def _resolve_author_id(self) -> str | None:
        """Resolve author name to Semantic Scholar author ID."""
        if self.author_id:
            return self.author_id
        if not self.author_name:
            return None
        url = f"{_API_BASE}/author/search?query={urllib.parse.quote(self.author_name)}&limit=1"
        try:
            data = _fetch_json(url)
            results = data.get("data", [])
            if results:
                return str(results[0]["authorId"])
        except Exception as e:
            print(f"    Author search failed: {e}")
        return None

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """Fetch papers and return document metadata."""

        author_id = self._resolve_author_id()
        if not author_id:
            print("    Could not resolve Semantic Scholar author")
            return {}

        print(f"    Fetching Semantic Scholar papers for author {author_id}...")
        data: dict[str, dict] = {}
        offset = 0

        while offset < self.max_papers:
            url = f"{_API_BASE}/author/{author_id}/papers?fields={_FIELDS}&limit={_PAGE_SIZE}&offset={offset}"

            try:
                result = _fetch_json(url)
            except Exception as e:
                print(f"    S2 API error at offset {offset}: {e}")
                break

            papers = result.get("data", [])
            if not papers:
                break

            for paper in papers:
                if _is_noise(paper):
                    continue

                citations = paper.get("citationCount") or 0
                if self.min_citations > 0 and citations < self.min_citations:
                    continue

                canonical = _canonical_url(paper)
                if not canonical:
                    continue

                # Skip if already in database (handles arXiv/DOI overlap)
                if existing_urls and canonical in existing_urls:
                    continue
                if canonical in data:
                    continue

                title = paper.get("title") or ""
                if not title:
                    continue

                year = paper.get("year")
                venue = paper.get("venue") or ""
                abstract = (paper.get("abstract") or "").strip()
                if len(abstract) > 250:
                    abstract = abstract[:247] + "..."

                # Build summary from abstract or venue + citations
                summary = abstract
                if not summary:
                    parts = []
                    if venue:
                        parts.append(venue)
                    if citations:
                        parts.append(f"{citations} citations")
                    summary = ". ".join(parts)

                data[canonical] = {
                    "title": title,
                    "summary": summary,
                    "date": f"{year}-01-01" if year else "",
                    "tags": ["scholar"],
                }

            offset += len(papers)
            if len(papers) < _PAGE_SIZE:
                break

            time.sleep(_DELAY)

        print(f"    {len(data)} papers (deduplicated)")
        return data
