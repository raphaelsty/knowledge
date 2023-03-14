import time

import datetime

import requests

__all__ = ["Twitter"]


class Twitter:
    """Twitter knowledge.

    Parameters
    ----------
    username
        Twitter username.
    user_id
        Twitter user id.
    token
        Twitter API token.

    Example:
    --------

    >>> from knowledge_database import twitter

    >>> news = twitter.Twitter(
    ...     username = "username",
    ...     user_id = "user_id",
    ...     token = "token"
    ... )

    >>> news(limit=1)

    References
    ----------
    1. [Get Tweeter ID](https://tweeterid.com/)

    """

    def __init__(self, username: str, user_id: int, token: str):
        self.username = username
        self.user_id = user_id
        self.token = token
        self.url = f"https://api.twitter.com/2/users/{self.user_id}/liked_tweets?max_results=100&expansions=author_id&user.fields=name&tweet.fields=attachments,author_id,context_annotations,created_at,entities,id,referenced_tweets,source,text,withheld"

    def __call__(self, limit: int = 100):
        data = {}
        next_token = ""

        for _ in range(limit):

            tweets = requests.get(
                self.url + next_token, headers={"Authorization": f"Bearer {self.token}"}
            ).json()

            if "data" not in tweets:
                break

            # Get user names
            users = {
                user["id"]: user["username"] for user in tweets["includes"]["users"]
            }

            data = {
                **data,
                **{
                    f"https://twitter.com/{users[tweet['author_id']]}/status/{tweet['id']}": {
                        "date": datetime.datetime.strptime(
                            tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        ).strftime("%Y-%m-%d"),
                        "title": f"Twitter @{users[tweet['author_id']]}",
                        "summary": tweet["text"],
                        "tags": list(
                            set(
                                ["twitter"]
                                + [
                                    annotation["normalized_text"].lower()
                                    for annotation in tweet.get("entities", {}).get(
                                        "annotations", []
                                    )
                                ]
                            )
                        ),
                    }
                    for tweet in tweets["data"]
                },
            }

            if "next_token" not in tweets["meta"]:
                break

            next_token = "&pagination_token=" + tweets["meta"]["next_token"]
            time.sleep(1)

        return data
