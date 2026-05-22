"""
Reddit user activity extractor.

Fetches a user's submissions (link posts → external URLs) and comments
(→ parent story URL) from the public Reddit JSON API. No auth required.

Submissions are the primary value: link posts point to real content.
Comments extract the parent post's external URL (if it's a link post).
"""

import json
import time
import urllib.request

__all__ = ["Posts"]

_HEADERS = {"User-Agent": "Knowledge/1.0 (research project; https://github.com/raphaelsty/knowledge)"}
_REDDIT_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "i.reddit.com",
    "v.redd.it",
    "i.redd.it",
    "preview.redd.it",
}
_DELAY = 2.0  # Reddit asks for ≤1 req/s; be extra polite


def _fetch_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _is_reddit_url(url: str) -> bool:
    from urllib.parse import urlparse

    return urlparse(url).netloc.lower().rstrip(".") in _REDDIT_DOMAINS


class Posts:
    """
    Extract links from a Reddit user's submissions and comments.

    **Submissions**: link posts give real external URLs (blog posts, papers, etc.).
    **Comments**: the parent post's external URL is extracted (if link post).

    Parameters
    ----------
    username : str
        Reddit username.
    max_pages : int, default=5
        Max pages to fetch (100 items per page). 5 pages = 500 items.
    include_comments : bool, default=True
        Also extract parent story URLs from comments.
    """

    def __init__(self, username: str, max_pages: int = 5, include_comments: bool = True):
        self.username = username
        self.max_pages = max_pages
        self.include_comments = include_comments

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}

        # Fetch submissions (link posts)
        print(f"    Fetching Reddit submissions for u/{self.username}...")
        submissions = self._paginate(
            "submitted",
            existing_urls=existing_urls,
            url_of=lambda item: item.get("url", ""),
        )
        for item in submissions:
            url = item.get("url", "")
            if not url or _is_reddit_url(url):
                continue
            if existing_urls and url in existing_urls:
                continue
            if url in data:
                continue

            title = item.get("title", "")
            subreddit = item.get("subreddit", "")
            created = item.get("created_utc", 0)
            date = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else ""

            # Use selftext or title as summary
            selftext = (item.get("selftext") or "").strip()
            if selftext and len(selftext) > 200:
                selftext = selftext[:197] + "..."
            summary = selftext or (f"r/{subreddit}: {title}" if subreddit else title)

            data[url] = {
                "title": title,
                "summary": summary,
                "date": date,
                "tags": ["reddit"],
            }

        print(f"    {len(data)} URLs from submissions")

        # Fetch comments (parent post URLs)
        if self.include_comments:
            print(f"    Fetching Reddit comments for u/{self.username}...")
            comments = self._paginate(
                "comments",
                existing_urls=existing_urls,
                url_of=lambda item: item.get("link_url", ""),
            )
            comment_urls = 0
            seen_links = set()

            for item in comments:
                link_url = item.get("link_url", "")
                if not link_url or _is_reddit_url(link_url):
                    continue
                if link_url in seen_links or link_url in data:
                    continue
                if existing_urls and link_url in existing_urls:
                    continue
                seen_links.add(link_url)

                link_title = item.get("link_title", "")
                subreddit = item.get("subreddit", "")
                created = item.get("created_utc", 0)
                date = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else ""

                # Use the comment body as summary
                comment_body = (item.get("body") or "").strip()
                if len(comment_body) > 200:
                    comment_body = comment_body[:197] + "..."
                summary = comment_body or (f"r/{subreddit}: {link_title}" if subreddit else link_title)

                data[link_url] = {
                    "title": link_title,
                    "summary": summary,
                    "date": date,
                    "tags": ["reddit"],
                }
                comment_urls += 1

            print(f"    {comment_urls} URLs from comments")

        print(f"    Total: {len(data)} Reddit URLs")
        return data

    def _paginate(
        self,
        endpoint: str,
        existing_urls: set[str] | None = None,
        url_of=None,
    ) -> list[dict]:
        """Paginate through a user's submissions or comments.

        Reddit returns newest-first, so once an entire page contains
        only URLs we've already ingested, every page after it is also
        all-known. We break early in that case — same shape as the
        GitHub-stars fetcher's page-level early-exit. New URLs are
        still discovered: as long as a page surfaces *one* unknown
        URL, the walker keeps going and we stop only when a fresh
        page yields zero new ones.
        """
        items = []
        after = None

        for page_idx in range(self.max_pages):
            url = f"https://www.reddit.com/user/{self.username}/{endpoint}.json?limit=100&sort=new&raw_json=1"
            if after:
                url += f"&after={after}"

            try:
                result = _fetch_json(url)
            except Exception as e:
                print(f"    Reddit API error: {e}")
                break

            children = result.get("data", {}).get("children", [])
            if not children:
                break

            page_data = [c["data"] for c in children]
            items.extend(page_data)

            # Early-exit: every URL on this page is one we've already
            # got in the database. Skip the remaining pages — they're
            # strictly older and therefore also all-known.
            if existing_urls and url_of is not None:
                page_urls = [u for u in (url_of(it) for it in page_data) if u and not _is_reddit_url(u)]
                if page_urls and all(u in existing_urls for u in page_urls):
                    print(f"    Reddit: page {page_idx + 1} all-known ({len(page_urls)} URLs), stopping early")
                    break

            after = result.get("data", {}).get("after")
            if not after:
                break

            time.sleep(_DELAY)

        return items
