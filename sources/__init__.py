"""
Sources - Data fetchers for the knowledge base.

Each subpackage collects documents from one platform (GitHub, HackerNews,
Zotero, HuggingFace, Twitter) and returns them as dicts keyed by URL.
The ``tags`` module enriches documents with extra topic labels.

Example
-------
>>> from sources import github
>>> gh = github.Github(user="raphaelsty")
>>> documents = gh()
"""

__all__ = [
    "github",
    "hackernews",
    "huggingface",
    "zotero",
    "twitter",
    "tags",
]
