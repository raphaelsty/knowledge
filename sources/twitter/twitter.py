"""
Twitter module for extracting bookmarked and liked tweets.

This module uses Twikit to fetch a user's bookmarked and liked tweets.
Bookmarks are filtered by minimum like count. Authentication uses
browser cookies (auth_token and ct0) to bypass Cloudflare protection.
"""

import asyncio
import re
import time
from urllib.parse import urlparse

from twikit import Client

__all__ = ["Twitter"]

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)")
_SKIP_DOMAINS = {"x.com", "twitter.com", "t.co", "pic.twitter.com"}


def _extract_links(tweet) -> list[str]:
    """Extract non-Twitter URLs from a tweet, normalizing arxiv links."""
    urls = []
    try:
        entities = tweet.urls or []
    except Exception:
        return urls
    for entity in entities:
        raw = entity.get("expanded_url") or entity.get("url") or ""
        if not raw:
            continue
        # Skip twitter/x.com and image links
        domain = urlparse(raw).netloc.lower()
        if any(domain.endswith(d) for d in _SKIP_DOMAINS):
            continue
        # Normalize arxiv links to canonical abs/ form
        m = _ARXIV_RE.search(raw)
        if m:
            urls.append(f"https://arxiv.org/abs/{m.group(1)}")
        else:
            urls.append(raw)
    return urls


class Twitter:
    """
    Extract knowledge from Twitter bookmarked and liked tweets using Twikit.

    Authenticates with Twitter via browser cookies and fetches bookmarked
    tweets (filtered by like count) and all liked tweets.

    Parameters
    ----------
    auth_token : str
        The ``auth_token`` cookie from a logged-in Twitter/X browser session.
    ct0 : str
        The ``ct0`` cookie (CSRF token) from a logged-in Twitter/X session.
    username : str
        Twitter/X screen name (used to fetch liked tweets).
    min_likes : int, default=10
        Minimum number of likes (favorite_count) a bookmarked tweet must
        have to be included. Does not apply to liked tweets.

    Example
    -------
    >>> from sources import twitter
    >>>
    >>> tw = twitter.Twitter(
    ...     auth_token="your_auth_token",
    ...     ct0="your_ct0_token",
    ...     username="raphaelsrty",
    ...     min_likes=10,
    ... )
    >>> documents = tw()

    Notes
    -----
    To get the cookies, log into x.com in your browser, open DevTools
    (F12) > Application > Cookies > https://x.com, and copy the values
    for ``auth_token`` and ``ct0``.
    """

    def __init__(self, auth_token: str, ct0: str, username: str, min_likes: int = 10):
        self.auth_token = auth_token
        self.ct0 = ct0
        self.username = username
        self.min_likes = min_likes

    def __call__(self, max_pages: int = 200, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """
        Fetch bookmarked and liked tweets and extract document metadata.

        Parameters
        ----------
        max_pages : int, default=200
            Maximum number of pagination pages to fetch per source.
        existing_urls : set[str] | None, default=None
            URLs already in the database. When provided, pagination stops
            early as soon as a full page yields zero new tweets.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping tweet URLs to document metadata containing:
            - title: "Twitter @username" format
            - summary: Full tweet text
            - date: Tweet creation date
            - tags: ["twitter"]
        """
        return asyncio.run(self._fetch_all(max_pages=max_pages, existing_urls=existing_urls))

    async def _fetch_all(self, max_pages: int, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """Fetch both bookmarks and likes."""
        client = Client("en-US")
        client.set_cookies(
            {
                "auth_token": self.auth_token,
                "ct0": self.ct0,
            }
        )

        data: dict[str, dict] = {}

        # Fetch bookmarks (filtered by min_likes)
        print("  Fetching bookmarks...")
        bookmarks = await self._paginate(
            await client.get_bookmarks(count=100),
            max_pages=max_pages,
            min_likes=self.min_likes,
            existing_urls=existing_urls,
        )
        data.update(bookmarks)
        print(f"  Found {len(bookmarks)} bookmarks with {self.min_likes}+ likes.")

        # Fetch liked tweets (no like filter)
        print("  Fetching liked tweets...")
        user = await client.get_user_by_screen_name(self.username)
        likes = await self._paginate(
            await client.get_user_tweets(user.id, tweet_type="Likes", count=100),
            max_pages=max_pages,
            min_likes=0,
            existing_urls=existing_urls,
        )
        # Don't overwrite bookmarks (they take priority)
        for url, doc in likes.items():
            if url not in data:
                data[url] = doc
        print(f"  Found {len(likes)} liked tweets.")

        return data

    @staticmethod
    async def _paginate(
        results, max_pages: int, min_likes: int, existing_urls: set[str] | None = None
    ) -> dict[str, dict]:
        """Paginate through a Result[Tweet] and extract documents."""
        data: dict[str, dict] = {}

        for _page in range(max_pages):
            if not results:
                break

            new_in_page = 0
            for tweet in results:
                if min_likes > 0 and tweet.favorite_count is not None and tweet.favorite_count < min_likes:
                    continue

                date = ""
                if tweet.created_at_datetime is not None:
                    date = tweet.created_at_datetime.strftime("%Y-%m-%d")

                author = tweet.user.screen_name if tweet.user else "unknown"
                tweet_url = f"https://x.com/{author}/status/{tweet.id}"

                if existing_urls is None or tweet_url not in existing_urls:
                    new_in_page += 1

                data[tweet_url] = {
                    "date": date,
                    "title": f"Twitter @{author}",
                    "summary": tweet.text or "",
                    "tags": ["twitter"],
                }

                # Extract linked URLs and create dedicated documents
                for link_url in _extract_links(tweet):
                    if link_url not in data:
                        data[link_url] = {
                            "date": date,
                            "title": f"Twitter @{author}",
                            "summary": tweet.text or "",
                            "tags": ["twitter"],
                        }

            if existing_urls is not None and new_in_page == 0:
                print("    No new tweets in page, stopping early.")
                break

            try:
                results = await results.next()
            except Exception:
                break

            time.sleep(1)

        return data
