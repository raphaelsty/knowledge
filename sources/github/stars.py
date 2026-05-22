"""
GitHub module for extracting starred repositories.

This module fetches starred repositories from a GitHub user's profile and
extracts relevant metadata including topics, descriptions, and README content.
"""

import collections
import datetime
import os
import re
import time

import requests

__all__ = ["Stars"]


def _unwrap_star_item(item: dict) -> dict:
    """Normalise a /starred response row to ``{starred_at, repo}``.

    Two response shapes exist depending on the Accept header:
      * legacy (`vnd.github.v3+json`)  → ``repo`` itself, no timestamp
      * star (`vnd.github.star+json`)  → ``{starred_at, repo}``
    We default to the star variant, but if the server falls back to
    the legacy shape (e.g. the header is dropped by an edge cache)
    we still want to ingest the repo, just without a per-star date.
    """
    if isinstance(item, dict) and "repo" in item and isinstance(item["repo"], dict):
        return item
    return {"starred_at": None, "repo": item}


def _gh_headers() -> dict[str, str]:
    # `application/vnd.github.star+json` flips the /starred response
    # shape from `[repo, …]` to `[{starred_at, repo}, …]`, giving us
    # an ISO-8601 timestamp per star. Without this we'd have to fall
    # back to "all stars dated today", which buries 1k+ docs in a
    # single same-day cluster on the search page.
    headers = {"Accept": "application/vnd.github.star+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


class Stars:
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
    >>> from sources import github
    >>>
    >>> gh = github.Stars(user="raphaelsty")
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
        existing_urls: set[str] | None = None,
    ) -> dict[str, dict]:
        """
        Fetch starred repositories and extract document metadata.

        Uses an early-exit strategy: fetches a small probe page first, and
        stops paginating as soon as a page contains no new repositories
        (all URLs already in existing_urls). README content is only fetched
        for new repositories.

        Parameters
        ----------
        per_page : int, default=100
            Number of results per API page (max 100).
        limit : int, default=100
            Maximum number of pages to fetch.
        existing_urls : set[str] | None, default=None
            URLs already in the database. When provided, enables early exit
            and skips README fetching for known repos.

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

        # Paginate through starred repositories (most recent first)
        for page in range(limit):
            try:
                response = requests.get(
                    f"https://api.github.com/users/{self.user}/starred?per_page={per_page}&page={page + 1}",
                    headers=_gh_headers(),
                    timeout=20,
                )
            except requests.RequestException as e:
                print(f"    GitHub request failed: {e}")
                break

            if response.status_code != 200:
                # 403 + `X-RateLimit-Remaining: 0` (or 429) is GitHub's
                # rate-limit signal. Log it explicitly so a continuous
                # runner can spot the throttling in its tail; for any
                # non-200 we break early regardless — the partial page
                # set already in `stars` is still safe to return.
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset = response.headers.get("X-RateLimit-Reset")
                if response.status_code in (403, 429) and remaining == "0":
                    print(
                        f"    GitHub rate-limited (HTTP {response.status_code}) — "
                        f"reset at epoch {reset}. Early-stopping."
                    )
                else:
                    print(f"    GitHub request failed (status {response.status_code}).")
                break

            page_data = response.json()
            if len(page_data) == 0:
                break

            # Items now arrive as `{starred_at, repo}` thanks to the
            # `star+json` Accept header. Earlier code assumed the
            # legacy `[repo, …]` shape — keep backward-compat for
            # the rare case the header is rejected (returns the old
            # shape) by detecting and unwrapping.
            normalized = [_unwrap_star_item(item) for item in page_data]

            # Early exit: if all repos on this page are already known, stop
            if existing_urls is not None:
                new_in_page = sum(
                    1
                    for item in normalized
                    if item.get("repo")
                    and item["repo"].get("html_url")
                    and item["repo"]["html_url"] not in existing_urls
                )
                stars += normalized
                if new_in_page == 0:
                    print(f"    No new stars on page {page + 1}, stopping early.")
                    break
                print(f"    Page {page + 1}: {new_in_page} new stars.")
            else:
                stars += normalized

            time.sleep(0.1)  # Rate limiting

        data: dict[str, dict] = collections.defaultdict(dict)
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        # `stars` is in the order GitHub returned them — newest-first.
        # Fall-back date math (only used when an item is missing
        # `starred_at`, which shouldn't happen with the new Accept
        # header but defensive code is cheap): assume one star per
        # day, walking backwards from today, so even synthesized
        # dates preserve order.
        for rank, item in enumerate(stars):
            repository = item.get("repo") or {}
            if not repository or "url" not in repository:
                continue

            url = repository["html_url"]

            # Per-star date — prefer the API's `starred_at`, fall
            # back to a synthesized date that at least preserves the
            # newest-first ordering. We don't care about absolute
            # accuracy; we care that two stars don't collide on the
            # same day and that "newest" sorts correctly.
            starred_at = item.get("starred_at")
            if isinstance(starred_at, str) and len(starred_at) >= 10:
                date = starred_at[:10]
            else:
                date = (datetime.datetime.today() - datetime.timedelta(days=rank)).strftime("%Y-%m-%d")

            # Collect tags from topics and language
            repo_tags = [tag.lower() for tag in repository["topics"]]
            if repository.get("language") is not None:
                repo_tags += [repository["language"].lower()]
            repo_tags = list(set(repo_tags))

            description = repository.get("description") or ""

            # Only fetch README for new repositories
            if existing_urls is not None and url in existing_urls:
                readme_text = None
            else:
                readme_text = self.get_readme_text_by_token_count(
                    repository["html_url"],
                    min_tokens=50,
                )

            data[url] = {
                "date": date,
                "title": repository["name"],
                "summary": f"{description} \n {readme_text}" if readme_text else description,
                "tags": repo_tags,
            }

        # Note: `today` only used as the fallback baseline above —
        # real dates come from the API.
        del today

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
                response = requests.get(raw_url, timeout=15)
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
