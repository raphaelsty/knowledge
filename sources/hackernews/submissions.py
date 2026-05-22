"""
HackerNews submissions extractor.

Fetches stories a user submitted to HN. These are often their own
blog posts or projects they find important enough to share.
Uses the public Algolia API — no auth needed.
"""

import json
import time
import urllib.request

__all__ = ["Submissions"]

_ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"
_DELAY = 0.5


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


class Submissions:
    """Fetch stories submitted by a HN user."""

    def __init__(self, username: str, max_items: int = 500):
        self.username = username
        self.max_items = max_items

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        print(f"    Fetching HN submissions for @{self.username}...")
        data: dict[str, dict] = {}
        page = 0
        total = 0

        while total < self.max_items:
            url = f"{_ALGOLIA}?tags=story,author_{self.username}&hitsPerPage=100&page={page}"
            try:
                result = _fetch_json(url)
            except Exception as e:
                print(f"    Algolia error: {e}")
                break

            hits = result.get("hits", [])
            if not hits:
                break

            # Page-level early exit: Algolia returns most-recent-first. If every
            # story on this page is already known and has a URL, we are caught up.
            if existing_urls is not None:
                new_in_page = sum(1 for h in hits if h.get("url") and h["url"] not in existing_urls)
                if new_in_page == 0:
                    print(f"    No new submissions on page {page}, stopping early.")
                    break
                print(f"    Page {page}: {new_in_page} new submission(s).")

            for hit in hits:
                story_url = hit.get("url")
                if not story_url:
                    continue
                if existing_urls and story_url in existing_urls:
                    continue
                if story_url in data:
                    continue
                title = hit.get("title") or ""
                created = (hit.get("created_at") or "")[:10]
                points = hit.get("points") or 0
                num_comments = hit.get("num_comments") or 0

                summary = ""
                if points or num_comments:
                    summary = f"{points} points, {num_comments} comments on HN"

                data[story_url] = {
                    "title": title,
                    "summary": summary,
                    "date": created,
                    "tags": ["hackernews"],
                }

            total += len(hits)
            page += 1
            if page >= result.get("nbPages", 0):
                break
            time.sleep(_DELAY)

        print(f"    {len(data)} stories submitted")
        return data
