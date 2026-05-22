"""
GitHub own repositories fetcher.

Fetches a user's own public repositories (not stars).
These represent their actual projects and work.
"""

import json
import time
import urllib.request

__all__ = ["Repositories"]

_DELAY = 1.0


def _build_headers() -> dict[str, str]:
    import os

    headers = {"User-Agent": "Knowledge/1.0", "Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers=_build_headers())
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


class Repositories:
    """Fetch a user's own public repositories."""

    def __init__(self, users: list[str]):
        self.users = users

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}
        for user in self.users:
            print(f"    Fetching repos for @{user}...")
            page = 1
            while True:
                url = f"https://api.github.com/users/{user}/repos?per_page=100&page={page}&sort=updated&type=owner"
                try:
                    repos = _fetch_json(url)
                except Exception as e:
                    print(f"    API error: {e}")
                    break
                if not repos:
                    break

                # Page-level early exit: sorted by `updated` desc, so once a full
                # page contains no new repos we're caught up; older pages would be
                # even older.
                if existing_urls is not None:
                    new_in_page = sum(
                        1
                        for r in repos
                        if not r.get("fork") and r.get("html_url") and r["html_url"] not in existing_urls
                    )
                    if new_in_page == 0:
                        print(f"    No new repos on page {page}, stopping early.")
                        break
                    print(f"    Page {page}: {new_in_page} new repo(s).")

                for repo in repos:
                    if repo.get("fork"):
                        continue
                    html_url = repo.get("html_url", "")
                    if not html_url or (existing_urls and html_url in existing_urls):
                        continue
                    desc = repo.get("description") or ""
                    topics = repo.get("topics") or []
                    lang = repo.get("language") or ""
                    stars = repo.get("stargazers_count", 0)
                    updated = (repo.get("pushed_at") or "")[:10]

                    summary = desc
                    if stars > 0:
                        summary += f" ({stars} stars)" if summary else f"{stars} stars"
                    if len(summary) > 250:
                        summary = summary[:247] + "..."

                    tags = [t.lower() for t in topics]
                    if lang and lang.lower() not in tags:
                        tags.append(lang.lower())

                    data[html_url] = {
                        "title": repo.get("name", ""),
                        "summary": summary,
                        "date": updated,
                        "tags": tags,
                    }
                page += 1
                time.sleep(_DELAY)
            print(f"    {user}: {sum(1 for u in data if user.lower() in u.lower())} repos")
        print(f"    Total: {len(data)} own repos")
        return data
