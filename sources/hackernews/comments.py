"""
HackerNews comments extractor.

Fetches a user's comments via the public Algolia API and extracts
the parent story URL for each comment. No authentication required.

If a person commented on a HN story, that story is worth indexing.
"""

import json
import time
import urllib.request

__all__ = ["Comments"]

_ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
_ALGOLIA_ITEM = "https://hn.algolia.com/api/v1/items"

# Delay between Algolia API requests (seconds)
_DELAY = 0.5


def _fetch_json(url: str, timeout: int = 15) -> dict:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class Comments:
    """
    Extract stories that a HackerNews user has commented on.

    Uses the Algolia HN Search API (public, no auth needed) to find
    all comments by a user, then resolves each to its parent story URL.

    Parameters
    ----------
    username : str
        HackerNews username.
    max_items : int, default=500
        Maximum number of comments to process.
    """

    def __init__(self, username: str, max_items: int = 500):
        self.username = username
        self.max_items = max_items

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """Fetch HN comments and extract parent story URLs."""
        print(f"    Fetching HN comments for @{self.username}...")

        data: dict[str, dict] = {}
        seen_stories: set[int] = set()
        page = 0
        hits_per_page = 100
        total_fetched = 0

        while total_fetched < self.max_items:
            url = f"{_ALGOLIA_SEARCH}?tags=comment,author_{self.username}&hitsPerPage={hits_per_page}&page={page}"

            try:
                result = _fetch_json(url)
            except Exception as e:
                print(f"    Algolia API error: {e}")
                break

            hits = result.get("hits", [])
            if not hits:
                break

            # Page-level early exit: if every story referenced on this page is
            # either already in the database or has no external URL to fetch,
            # older pages will be the same or worse — stop.
            if existing_urls is not None:
                new_in_page = sum(1 for h in hits if h.get("story_url") and h["story_url"] not in existing_urls)
                if new_in_page == 0:
                    print(f"    No new story URLs on page {page}, stopping early.")
                    break
                print(f"    Page {page}: {new_in_page} new story URL(s).")

            for hit in hits:
                story_id = hit.get("story_id")
                if not story_id or story_id in seen_stories:
                    continue
                seen_stories.add(story_id)

                story_url = hit.get("story_url")
                story_title = hit.get("story_title") or ""

                # Skip items without an external URL (Ask HN, Show HN text posts)
                if not story_url:
                    continue

                # Skip if already known
                if existing_urls and story_url in existing_urls:
                    continue

                # Use the comment creation date
                date = ""
                created_at = hit.get("created_at", "")
                if created_at and len(created_at) >= 10:
                    date = created_at[:10]

                # Use the comment text as summary (truncated)
                comment_text = hit.get("comment_text") or ""
                # Strip HTML tags from comment
                import re

                summary = re.sub(r"<[^>]+>", " ", comment_text)
                summary = re.sub(r"\s+", " ", summary).strip()
                if len(summary) > 200:
                    summary = summary[:197] + "..."

                data[story_url] = {
                    "title": story_title,
                    "summary": summary,
                    "date": date,
                    "tags": ["hackernews"],
                }

            total_fetched += len(hits)
            page += 1

            nb_pages = result.get("nbPages", 0)
            if page >= nb_pages:
                break

            time.sleep(_DELAY)

        print(f"    Found {len(data)} unique stories from {total_fetched} comments")
        return data
