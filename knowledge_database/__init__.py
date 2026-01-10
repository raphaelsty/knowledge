"""
Knowledge Database - A personal knowledge management system.

This package provides tools for collecting, organizing, and searching through
personal knowledge from various sources including GitHub, HackerNews, Zotero,
Semanlink, and HuggingFace.

Modules
-------
github
    Extract starred repositories from GitHub.
hackernews
    Extract upvoted posts from HackerNews.
huggingface
    Extract liked models, datasets, and spaces from HuggingFace.
zotero
    Extract bookmarks from Zotero libraries.
semanlink
    Parse knowledge from Semanlink RDF/Turtle files.
twitter
    Extract liked tweets from Twitter.
tags
    Generate tag relationships and extra tags for documents.
retriever
    BM25-based document and tag retrieval.
graph
    Knowledge graph construction and traversal.
pipeline
    Unified pipeline combining retrieval and graph visualization.

Example
-------
>>> from knowledge_database import github, pipeline
>>>
>>> # Collect GitHub stars
>>> gh = github.Github(user="raphaelsty")
>>> documents = gh()
>>>
>>> # Search through documents
>>> pipe = pipeline.Pipeline(documents=documents, triples=triples)
>>> results = pipe.search("neural networks")
"""

__all__ = [
    "github",
    "hackernews",
    "huggingface",
    "zotero",
    "semanlink",
    "twitter",
    "tags",
    "retriever",
    "graph",
    "pipeline",
]
