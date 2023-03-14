import requests
from bs4 import BeautifulSoup

__all__ = ["Reddit"]


class Reddit:
    """Reddit knowledge. WIP.

    Example:
    --------

    >>> from knowledge_database import reddit

    >>> reddit_knowledge = reddit.Reddit(user="")

    >>> reddit_knowledge()

    """

    def __init__(self, user: str):
        self.user = user

    def __call__(self):
        pass
