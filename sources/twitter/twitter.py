"""
Twitter module for extracting bookmarked and liked tweets.

This module uses Twikit to fetch a user's bookmarked and liked tweets.
Bookmarks are filtered by minimum like count. Authentication uses
browser cookies (auth_token and ct0) to bypass Cloudflare protection.
"""

import asyncio
import time

from twikit import Client

__all__ = ["Twitter"]


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

    def __call__(self, max_pages: int = 200) -> dict[str, dict]:
        """
        Fetch bookmarked and liked tweets and extract document metadata.

        Parameters
        ----------
        max_pages : int, default=200
            Maximum number of pagination pages to fetch per source.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping tweet URLs to document metadata containing:
            - title: "Twitter @username" format
            - summary: Full tweet text
            - date: Tweet creation date
            - tags: ["twitter"]
        """
        return asyncio.run(self._fetch_all(max_pages=max_pages))

    async def _fetch_all(self, max_pages: int) -> dict[str, dict]:
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
        )
        # Don't overwrite bookmarks (they take priority)
        for url, doc in likes.items():
            if url not in data:
                data[url] = doc
        print(f"  Found {len(likes)} liked tweets.")

        return data

    @staticmethod
    async def _paginate(results, max_pages: int, min_likes: int) -> dict[str, dict]:
        """Paginate through a Result[Tweet] and extract documents."""
        data: dict[str, dict] = {}

        for _page in range(max_pages):
            if not results:
                break

            for tweet in results:
                if min_likes > 0 and tweet.favorite_count is not None and tweet.favorite_count < min_likes:
                    continue

                date = ""
                if tweet.created_at_datetime is not None:
                    date = tweet.created_at_datetime.strftime("%Y-%m-%d")

                author = tweet.user.screen_name if tweet.user else "unknown"
                tweet_url = f"https://x.com/{author}/status/{tweet.id}"

                data[tweet_url] = {
                    "date": date,
                    "title": f"Twitter @{author}",
                    "summary": tweet.text or "",
                    "tags": ["twitter"],
                }

            try:
                results = await results.next()
            except Exception:
                break

            time.sleep(1)

        return data
