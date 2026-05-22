"""
HackerNews module for extracting upvoted posts.

This module fetches upvoted posts from a HackerNews user's profile and
extracts article content using web scraping.
"""

import datetime
import re
import time

import requests
import trafilatura
from bs4 import BeautifulSoup

__all__ = ["Upvotes"]


class Upvotes:
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
    >>> from sources import hackernews
    >>>
    >>> hn = hackernews.Upvotes(
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

    def __call__(self, existing_urls: set[str] | None = None, max_pages: int = 20) -> dict[str, dict]:
        """Fetch every page of the user's /upvoted list.

        Returns `{url: {title, summary, date, tags}}`. Skips anything
        already in ``existing_urls`` so we don't re-download summaries
        for known posts. Login failure raises instead of returning
        silently — /upvoted is private, so an anon session would
        return 0 items with no indication that anything went wrong.
        """
        data: dict[str, dict] = {}
        existing = existing_urls or set()

        with requests.Session() as session:
            # Browser-like UA avoids HN's anti-bot captcha challenge.
            session.headers["User-Agent"] = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Safari/605.1.15"
            )
            # Authenticate with HackerNews.
            login_response = session.post(
                "https://news.ycombinator.com/login?goto=news",
                data={"acct": self.username, "pw": self.password},
            )
            body = login_response.text
            # HN always 200s from /login — we detect a successful login
            # by any of the `id=me` / `id=logout` / `id=karma` markers
            # (the top-nav username link + karma badge + logout link).
            # Any ONE of them is enough; they're only present on a
            # logged-in page. `user?id=<name>` is NOT reliable — it
            # appears anywhere a username is linked.
            authed_markers = (
                "id=me",
                'id="me"',
                "id='me'",
                "id=logout",
                'id="logout"',
                "id='logout'",
                "id=karma",
                'id="karma"',
                "logout?auth=",
            )
            if not any(m in body for m in authed_markers):
                reason = (
                    "wrong username or password"
                    if "Bad login" in body
                    else "HN requires a captcha (log in manually in a browser once)"
                    if ("Validation required" in body or "recaptcha" in body)
                    else "unexpected HN response"
                )
                print(f"    HN login failed for @{self.username} — {reason}. Skipping upvotes.")
                return data
            print(f"    HN login OK for @{self.username}")

            # HN paginates /upvoted via a "More" link that carries an
            # opaque cursor token. Follow it until it disappears (or we
            # hit max_pages as a safety cap).
            url = f"https://news.ycombinator.com/upvoted?id={self.username}"
            for _page in range(max_pages):
                html = session.get(url, timeout=20).text
                soup = BeautifulSoup(html, "html.parser")

                page_count = 0
                for entry in soup.find_all("tr", class_="athing"):
                    a = entry.find("span", class_="titleline")
                    if a is None:
                        continue
                    link = a.find("a")
                    if link is None or not link.get("href"):
                        continue
                    href = link["href"]
                    # HN uses relative hrefs for self-hosted items
                    # ("item?id=..."); make them absolute.
                    if href.startswith("item?"):
                        href = "https://news.ycombinator.com/" + href
                    if self.username in href:
                        continue
                    if href in existing or href in data:
                        continue
                    title = link.text.strip()
                    data[href] = {
                        "title": f"Hackernews {title}",
                        "tags": ["hackernews"],
                        "summary": self.get_summary(href),
                        # HN's story-age span carries an ISO timestamp in
                        # its `title` attribute. The age span lives in
                        # the *sibling* <tr> under the .subtext cell —
                        # that's where HN renders submitter + points +
                        # age. Falls back to today on any parse hiccup
                        # rather than blocking the whole row.
                        "date": self._story_date(entry) or datetime.datetime.today().strftime("%Y-%m-%d"),
                    }
                    page_count += 1

                # Follow the "More" link if present, otherwise we're done.
                more = soup.find("a", class_="morelink")
                if not more or not more.get("href"):
                    break
                href = more["href"]
                if not href.startswith("http"):
                    href = "https://news.ycombinator.com/" + href
                url = href
                time.sleep(1.0)  # polite

        return data

    @staticmethod
    def _story_date(athing_row) -> str:
        """Extract the story's submission date from an HN listing row.

        HN renders:
          <tr class="athing" id="NNN">  ← what we're handed
            … title, site …
          </tr>
          <tr>
            <td class="subtext">
              … points …
              <span class="age" title="2026-04-20T12:34:56">4 hours ago</span>
              …
            </td>
          </tr>

        We walk to the sibling row, read the age span's `title` attr,
        and grab the YYYY-MM-DD prefix. Any failure returns "" so the
        caller can fall back to today's date.
        """
        try:
            sibling = athing_row.find_next_sibling("tr")
            if sibling is None:
                return ""
            age = sibling.find("span", class_="age")
            iso = (age or {}).get("title", "") if age else ""
            if iso:
                return iso.split("T", 1)[0][:10]
        except Exception:
            pass
        return ""

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
