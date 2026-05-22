"""
GitHub Gists fetcher.

Fetches a user's public gists — code snippets, config files, mini-tutorials.
"""

import json
import time
import urllib.request

__all__ = ["Gists"]

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


class Gists:
    """Fetch a user's public gists."""

    def __init__(self, users: list[str]):
        self.users = users

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}
        for user in self.users:
            print(f"    Fetching gists for @{user}...")
            page = 1
            while True:
                url = f"https://api.github.com/users/{user}/gists?per_page=100&page={page}"
                try:
                    gists = _fetch_json(url)
                except Exception as e:
                    print(f"    API error: {e}")
                    break
                if not gists:
                    break

                # Page-level early exit: if all gists on this page are known we
                # are caught up (GitHub returns most-recently-created first).
                if existing_urls is not None:
                    new_in_page = sum(1 for g in gists if g.get("html_url") and g["html_url"] not in existing_urls)
                    if new_in_page == 0:
                        print(f"    No new gists on page {page}, stopping early.")
                        break
                    print(f"    Page {page}: {new_in_page} new gist(s).")

                for gist in gists:
                    html_url = gist.get("html_url", "")
                    if not html_url or (existing_urls and html_url in existing_urls):
                        continue
                    desc = (gist.get("description") or "").strip()
                    files = gist.get("files") or {}
                    filenames = list(files.keys())
                    title = desc or (filenames[0] if filenames else f"Gist {gist.get('id', '')[:8]}")
                    if len(title) > 100:
                        title = title[:97] + "..."
                    created = (gist.get("created_at") or "")[:10]
                    langs = {(f.get("language") or "").lower() for f in files.values()} - {""}
                    data[html_url] = {
                        "title": title,
                        "summary": desc if desc != title else ", ".join(filenames[:3]),
                        "date": created,
                        "tags": list(langs),
                    }
                page += 1
                time.sleep(_DELAY)
        print(f"    Total: {len(data)} gists")
        return data
