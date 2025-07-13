import datetime

import requests
from bs4 import BeautifulSoup

__all__ = ["HackerNews"]


class HackerNews:
    """HackerNews upvoted posts.

    Parameters
    ----------
    username
        HackerNews username.
    password
        HackerNews password.

    Examples
    --------

    >>> from knowledge_database import hackernews

    >>> news = hackernews.HackerNews(
    ...     username="username",
    ...     password="password",
    ... )

    >>> documents = news()

    """

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def __call__(self):
        """Get upvoted content on HackerNews."""
        data = {}

        with requests.Session() as session:

            p = session.post(
                "https://news.ycombinator.com/login?goto=news",
                data={"acct": self.username, "pw": self.password},
            )

            if ("user?id=" + self.username) in p.text:
                print("Hackernews - login successful")

            html = session.get(
                f"https://news.ycombinator.com/upvoted?id={self.username}"
            ).text

            soup = BeautifulSoup(html, "html.parser")

            for entry in soup.find_all("td", class_="title"):

                record = entry.find("a")
                if record is None:
                    continue

                record = record.__dict__
                attributes = record.get("attrs")

                if attributes is None:
                    continue

                if self.username in attributes["href"]:
                    continue

                data[attributes["href"]] = {
                    "title": f"Hackernews {entry.text}",
                    "tags": ["hackernews"],
                    "summary": "",
                    "date": datetime.datetime.today().strftime("%Y-%m-%d"),
                }

        return data
