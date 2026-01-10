"""
HackerNews module for extracting upvoted posts.

This module fetches upvoted posts from a HackerNews user's profile and
extracts article content using web scraping.
"""

import datetime
import re

import requests
import trafilatura
from bs4 import BeautifulSoup

__all__ = ["HackerNews"]


class HackerNews:
    """
    Extract knowledge from HackerNews upvoted posts.

    Authenticates with HackerNews and scrapes the user's upvoted posts,
    extracting article titles and summarized content from linked URLs.

    Parameters
    ----------
    username : str
        HackerNews username.
    password : str
        HackerNews password for authentication.

    Example
    -------
    >>> from knowledge_database import hackernews
    >>>
    >>> hn = hackernews.HackerNews(
    ...     username="your_username",
    ...     password="your_password",
    ... )
    >>> documents = hn()
    >>>
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}")
    """

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def __call__(self) -> dict[str, dict]:
        """
        Fetch upvoted posts and extract document metadata.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping article URLs to document metadata containing:
            - title: "Hackernews" prefix + article title
            - summary: First ~50 tokens of article content
            - date: Current date
            - tags: ["hackernews"]
        """
        data: dict[str, dict] = {}

        with requests.Session() as session:
            # Authenticate with HackerNews
            login_response = session.post(
                "https://news.ycombinator.com/login?goto=news",
                data={"acct": self.username, "pw": self.password},
            )

            if ("user?id=" + self.username) in login_response.text:
                print("Hackernews - login successful")

            # Fetch upvoted posts page
            html = session.get(f"https://news.ycombinator.com/upvoted?id={self.username}").text

            soup = BeautifulSoup(html, "html.parser")

            # Extract links from upvoted posts
            for entry in soup.find_all("td", class_="title"):
                record = entry.find("a")
                if record is None:
                    continue

                record = record.__dict__
                attributes = record.get("attrs")

                if attributes is None:
                    continue

                # Skip self-referential links
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
    def get_summary(url: str, num_tokens: int = 50) -> str:
        """
        Extract article summary from a URL.

        Uses trafilatura to extract main content from web pages,
        returning the first N tokens as a summary.

        Parameters
        ----------
        url : str
            URL of the article to summarize.
        num_tokens : int, default=50
            Number of words to include in the summary.

        Returns
        -------
        str
            First N words of the article content, or empty string on failure.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html_content = response.text

            # Extract main article content
            core_text = trafilatura.extract(html_content)

            if not core_text:
                return ""

            # Normalize whitespace and truncate
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
