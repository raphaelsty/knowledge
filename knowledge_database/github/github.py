import collections
import datetime
import json
import os
import re
import time

import requests

__all__ = ["Github"]


class Github:
    """Github Knowledge.

    Parameters
    ----------
    username
        Github username.

    Examples
    --------

    >>> from knowledge_database import github

    >>> github_raph = github.Github(user="raphaelsty")

    >>> github_raph(per_page=3, limit=3)

    """

    def __init__(self, user: str):
        self.user = user

    def __call__(self, per_page: int = 100, limit: int = 100):

        stars = []

        for page in range(limit):

            r = requests.get(
                f"https://api.github.com/users/{self.user}/starred?{per_page}=10&page={page}"
            )

            if r.status_code != 200:
                print("Github request failed.")
                break

            r = r.json()
            if len(r) == 0:
                break

            stars += r

            time.sleep(0.1)

        data = collections.defaultdict(dict)

        for repository in stars:

            if "url" not in repository:
                continue

            url = repository["html_url"]

            tags = [tag.lower() for tag in repository["topics"]]
            if repository.get("language", None) is not None:
                tags += [repository["language"].lower()]
            tags = list(set(tags))

            readme_text = self.get_readme_text_by_token_count(
                repository["html_url"],
                min_tokens=100,
            )

            description = repository.get("description") or ""

            data[url] = {
                "date": datetime.datetime.today().strftime("%Y-%m-%d"),
                "title": f"{repository['name']}",
                "summary": (
                    f"{description} \n {readme_text}" if readme_text else description
                ),
                "tags": tags,
            }

        return data

    @staticmethod
    def get_readme_text_by_token_count(
        github_url: str,
        min_tokens: int = 100,
    ) -> str | None:
        """
        Fetches a README, removes HTML, markdown artifacts, and special tokens,
        and extracts clean plain text until a minimum token count is reached.

        Args:
            github_url: The URL of the GitHub repository.
            min_tokens: The minimum number of tokens (words) to collect.

        Returns:
            A clean string of plain text from the README, or None if it cannot be found.
        """
        match = re.search(r"github\.com/([^/]+)/([^/]+)", github_url)
        if not match:
            return None

        user, repo = match.groups()

        branches_to_try = ["main", "master"]
        readme_content = None

        for branch in branches_to_try:
            raw_url = (
                f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/README.md"
            )
            try:
                response = requests.get(raw_url)
                response.raise_for_status()
                readme_content = response.text
                break
            except requests.exceptions.RequestException:
                continue

        if not readme_content:
            return None

        text_only_content = re.sub(r"<[^>]+>", "", readme_content)

        collected_text = ""
        lines = text_only_content.splitlines()

        for line in lines:
            stripped_line = line.strip()

            is_heading = stripped_line.startswith("#")
            is_list_item = stripped_line.startswith(("* ", "- ", "+ "))
            is_blockquote = stripped_line.startswith(">")
            is_just_an_image_or_link = stripped_line.startswith(
                "["
            ) and stripped_line.endswith(")")
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

        clean_text = re.sub(r"[^a-zA-Z0-9\s.,?!'-]", "", collected_text)
        normalized_text = re.sub(r"\s+", " ", clean_text)

        return normalized_text.strip()
