import datetime
import json
import re

import requests
import tqdm
import trafilatura
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
                    "summary": self.get_summary(attributes["href"]),
                    "date": datetime.datetime.today().strftime("%Y-%m-%d"),
                }

        return data

    @staticmethod
    def get_summary(url, num_tokens=50):
        """
        Fetches the core article content from a URL, cleans it,
        and returns the first N tokens.

        Args:
            url (str): The URL of the webpage to process.
            num_tokens (int): The number of tokens to return from the core body.

        Returns:
            str: The first N words of the core article text,
                or None if the content cannot be fetched or extracted.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html_content = response.text

            core_text = trafilatura.extract(html_content)

            if not core_text:
                return ""

            cleaned_text = re.sub(r"\s+", " ", core_text).strip()

            tokens = cleaned_text.split()
            first_n_tokens = tokens[:num_tokens]

            return " ".join(first_n_tokens)

        except requests.exceptions.RequestException as e:
            print(f"Could not fetch {url}: {e}")
            return ""
        except Exception as e:
            print(f"An error occurred while processing {url}: {e}")
            return ""
