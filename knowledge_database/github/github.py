import collections
import time
import datetime

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

    >>> github_raph = github.Github(user="user")

    >>> github_raph.stars(per_page=3, limit=3)

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

            date = datetime.datetime.strptime(
                repository["created_at"], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y-%m-%d")

            data[url] = {
                "date": date,
                "title": f"{repository['name']}",
                "summary": repository["description"],
                "tags": tags,
            }

        return data
