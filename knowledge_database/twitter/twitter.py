"""
Twitter module for extracting liked tweets.

This module uses the Twitter API v2 to fetch a user's liked tweets
and extract relevant metadata including annotations and entities.
"""

import datetime
import time

import requests

__all__ = ["Twitter"]


class Twitter:
    """
    Extract knowledge from Twitter liked tweets.

    Uses Twitter API v2 to fetch liked tweets and extract text content,
    publication dates, and named entity annotations.

    Parameters
    ----------
    username : str
        Twitter username (for constructing tweet URLs).
    user_id : int
        Numeric Twitter user ID (required for API calls).
    token : str
        Twitter API Bearer token with read permissions.

    Attributes
    ----------
    url : str
        Constructed API endpoint URL for fetching liked tweets.

    Example
    -------
    >>> from knowledge_database import twitter
    >>>
    >>> tw = twitter.Twitter(
    ...     username="your_username",
    ...     user_id=123456789,
    ...     token="your_bearer_token",
    ... )
    >>> documents = tw(limit=10)
    >>>
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}: {doc['summary'][:50]}...")

    Notes
    -----
    To find your Twitter user ID, use https://tweeterid.com/
    """

    def __init__(self, username: str, user_id: int, token: str):
        self.username = username
        self.user_id = user_id
        self.token = token

        # Construct API URL with all required fields
        self.url = (
            f"https://api.twitter.com/2/users/{self.user_id}/liked_tweets"
            f"?max_results=100"
            f"&expansions=author_id"
            f"&user.fields=name"
            f"&tweet.fields=attachments,author_id,context_annotations,"
            f"created_at,entities,id,referenced_tweets,source,text,withheld"
        )

    def __call__(self, limit: int = 100) -> dict[str, dict]:
        """
        Fetch liked tweets and extract document metadata.

        Paginates through the user's liked tweets, extracting tweet content
        and any named entity annotations.

        Parameters
        ----------
        limit : int, default=100
            Maximum number of API pages to fetch (100 tweets per page).

        Returns
        -------
        dict[str, dict]
            Dictionary mapping tweet URLs to document metadata containing:
            - title: "Twitter @username" format
            - summary: Full tweet text
            - date: Tweet creation date
            - tags: ["twitter"] + extracted entity annotations
        """
        data: dict[str, dict] = {}
        next_token = ""

        for _ in range(limit):
            # Fetch page of liked tweets
            response = requests.get(
                self.url + next_token,
                headers={"Authorization": f"Bearer {self.token}"},
            )
            tweets = response.json()

            if "data" not in tweets:
                break

            # Build user ID to username mapping
            users = {user["id"]: user["username"] for user in tweets["includes"]["users"]}

            # Process each tweet in the response
            for tweet in tweets["data"]:
                author_username = users[tweet["author_id"]]
                tweet_url = f"https://twitter.com/{author_username}/status/{tweet['id']}"

                # Parse tweet timestamp
                date = datetime.datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")

                # Extract entity annotations as tags
                annotations = tweet.get("entities", {}).get("annotations", [])
                entity_tags = [annotation["normalized_text"].lower() for annotation in annotations]

                data[tweet_url] = {
                    "date": date,
                    "title": f"Twitter @{author_username}",
                    "summary": tweet["text"],
                    "tags": list(set(["twitter"] + entity_tags)),
                }

            # Check for more pages
            if "next_token" not in tweets["meta"]:
                break

            next_token = "&pagination_token=" + tweets["meta"]["next_token"]
            time.sleep(1)  # Rate limiting

        return data
