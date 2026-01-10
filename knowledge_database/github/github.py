"""
GitHub module for extracting starred repositories.

This module fetches starred repositories from a GitHub user's profile and
extracts relevant metadata including topics, descriptions, and README content.
"""

import collections
import datetime
import re
import time

import requests

__all__ = ["Github"]


class Github:
    """
    Extract knowledge from GitHub starred repositories.

    Fetches starred repositories and extracts metadata including repository
    topics, descriptions, and a clean text summary from README files.

    Parameters
    ----------
    user : str
        GitHub username whose starred repositories to fetch.

    Attributes
    ----------
    user : str
        The GitHub username.

    Example
    -------
    >>> from knowledge_database import github
    >>>
    >>> gh = github.Github(user="raphaelsty")
    >>> documents = gh(per_page=100, limit=10)
    >>>
    >>> # Each document contains: title, summary, date, tags
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}: {len(doc['tags'])} tags")
    """

    def __init__(self, user: str):
        self.user = user

    def __call__(
        self,
        per_page: int = 100,
        limit: int = 100,
    ) -> dict[str, dict]:
        """
        Fetch starred repositories and extract document metadata.

        Parameters
        ----------
        per_page : int, default=100
            Number of results per API page (max 100).
        limit : int, default=100
            Maximum number of pages to fetch.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping repository URLs to document metadata containing:
            - title: Repository name
            - summary: Description + README excerpt
            - date: Current date (when starred info was fetched)
            - tags: Repository topics + programming language
        """
        stars = []

        # Paginate through starred repositories
        for page in range(limit):
            response = requests.get(f"https://api.github.com/users/{self.user}/starred?{per_page}=10&page={page}")

            if response.status_code != 200:
                print("Github request failed.")
                break

            page_data = response.json()
            if len(page_data) == 0:
                break

            stars += page_data
            time.sleep(0.1)  # Rate limiting

        data: dict[str, dict] = collections.defaultdict(dict)

        for repository in stars:
            if "url" not in repository:
                continue

            url = repository["html_url"]

            # Collect tags from topics and language
            tags = [tag.lower() for tag in repository["topics"]]
            if repository.get("language") is not None:
                tags += [repository["language"].lower()]
            tags = list(set(tags))

            # Extract clean text from README
            readme_text = self.get_readme_text_by_token_count(
                repository["html_url"],
                min_tokens=50,
            )

            description = repository.get("description") or ""

            data[url] = {
                "date": datetime.datetime.today().strftime("%Y-%m-%d"),
                "title": f"{repository['name']}",
                "summary": f"{description} \n {readme_text}" if readme_text else description,
                "tags": tags,
            }

        return data

    @staticmethod
    def get_readme_text_by_token_count(
        github_url: str,
        min_tokens: int = 50,
    ) -> str | None:
        """
        Extract clean plain text from a repository's README.

        Fetches the README.md file and extracts readable paragraph text,
        filtering out markdown artifacts, headings, and code blocks.

        Parameters
        ----------
        github_url : str
            URL of the GitHub repository.
        min_tokens : int, default=50
            Minimum number of words to collect before stopping.

        Returns
        -------
        str | None
            Clean text excerpt from the README, or None if not found.
        """
        match = re.search(r"github\.com/([^/]+)/([^/]+)", github_url)
        if not match:
            return None

        user, repo = match.groups()

        # Try common default branch names
        branches_to_try = ["main", "master"]
        readme_content = None

        for branch in branches_to_try:
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/README.md"
            try:
                response = requests.get(raw_url)
                response.raise_for_status()
                readme_content = response.text
                break
            except requests.exceptions.RequestException:
                continue

        if not readme_content:
            return None

        # Strip HTML tags
        text_only_content = re.sub(r"<[^>]+>", "", readme_content)

        collected_text = ""
        lines = text_only_content.splitlines()

        # Extract only paragraph text, skipping markdown artifacts
        for line in lines:
            stripped_line = line.strip()

            is_heading = stripped_line.startswith("#")
            is_list_item = stripped_line.startswith(("* ", "- ", "+ "))
            is_blockquote = stripped_line.startswith(">")
            is_just_an_image_or_link = stripped_line.startswith("[") and stripped_line.endswith(")")
            is_horizontal_rule = re.match(r"^[-*_]{3,}$", stripped_line) is not None

            if (
                not stripped_line
                or is_heading
                or is_list_item
                or is_blockquote
                or is_just_an_image_or_link
                or is_horizontal_rule
            ):
                continue

            collected_text += stripped_line + " "

            current_tokens = len(collected_text.split())
            if current_tokens >= min_tokens:
                break

        if not collected_text:
            return None

        # Clean remaining special characters
        clean_text = re.sub(r"[^a-zA-Z0-9\s.,?!'-]", "", collected_text)
        normalized_text = re.sub(r"\s+", " ", clean_text)

        return normalized_text.strip()
