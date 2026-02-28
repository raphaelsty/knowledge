"""
Tags module for generating tag relationships and automatic tagging.

This module provides utilities for building a tag co-occurrence graph and
automatically discovering additional relevant tags for documents based on
their content.
"""

import collections
import itertools

from cherche import retrieve
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["get_extra_tags", "get_tags_triples"]


def get_tags_triples(
    data: dict[str, dict],
    excluded_tags: dict[str, bool] | None = None,
) -> list[dict]:
    """
    Build a graph of tag co-occurrence relationships.

    Creates edges between tags that appear together in the same document,
    enabling visualization of knowledge domain relationships.

    Parameters
    ----------
    data : dict[str, dict]
        Dictionary mapping URLs to document metadata. Each document should
        contain 'tags' and 'extra-tags' lists.
    excluded_tags : dict[str, bool], optional
        Tags to exclude from the graph (e.g., generic source tags like 'github').

    Returns
    -------
    list[dict]
        List of edge dictionaries with 'head' and 'tail' keys representing
        connected tags. Each edge appears only once (undirected graph).

    Example
    -------
    >>> documents = {
    ...     "url1": {"tags": ["python", "ml"], "extra-tags": ["pytorch"]},
    ...     "url2": {"tags": ["python", "web"], "extra-tags": []},
    ... }
    >>> triples = get_tags_triples(documents)
    >>> # Creates edges: python-ml, python-pytorch, ml-pytorch, python-web
    """
    excluded_tags = {} if excluded_tags is None else excluded_tags
    triples = []

    # Track seen edges to avoid duplicates (undirected graph)
    seen: dict[str, dict[str, bool]] = collections.defaultdict(dict)

    for _, document in data.items():
        all_tags = document["tags"] + document["extra-tags"]

        # Create edges for all tag pairs within the document
        for head, tail in itertools.combinations(all_tags, 2):
            # Skip excluded tags
            if head in excluded_tags or tail in excluded_tags:
                continue

            # Skip if edge already exists (either direction)
            if head in seen[tail] or tail in seen[head]:
                continue

            triples.append({"head": head, "tail": tail})
            seen[head][tail] = True
            seen[tail][head] = True

    return triples


def get_extra_tags(data: dict[str, dict]) -> dict[str, dict]:
    """
    Automatically discover additional relevant tags for documents.

    Uses TF-IDF similarity between document content (title + summary) and
    existing tags to suggest new tags that weren't manually assigned.

    Parameters
    ----------
    data : dict[str, dict]
        Dictionary mapping URLs to document metadata. Each document should
        contain 'title', 'summary', and 'tags'.

    Returns
    -------
    dict[str, dict]
        Updated document dictionary with 'extra-tags' added to each document.
        Extra tags are existing tags from other documents that match the
        document's content with similarity > 0.2.

    Example
    -------
    >>> documents = {
    ...     "url1": {"title": "PyTorch Tutorial", "summary": "Deep learning...", "tags": ["pytorch"]},
    ...     "url2": {"title": "TensorFlow Guide", "summary": "Neural networks...", "tags": ["tensorflow"]},
    ... }
    >>> enriched = get_extra_tags(documents)
    >>> # Documents may now have cross-referenced tags based on content similarity
    """
    # Build index of existing document-tag assignments
    documents_dict: dict[str, bool] = {}
    tagged: dict[str, dict[str, bool]] = collections.defaultdict(dict)

    for url, document in data.items():
        for tag in document["tags"]:
            documents_dict[tag] = True
            tagged[url][tag] = True

    # Create tag retriever using character n-gram TF-IDF
    documents_list = [{"tag": tag} for tag in documents_dict]
    retriever = (
        retrieve.Flash(
            key="tag",
            on="tag",
            k=10,
        )
        | retrieve.TfIdf(
            key="tag",
            on="tag",
            documents=documents_list,
            k=3,
            tfidf=TfidfVectorizer(
                lowercase=True,
                ngram_range=(4, 7),
                analyzer="char_wb",
            ),
        )
    ).add(documents_list)

    # Find extra tags for each document based on content similarity
    extra_tags: dict[str, list[str]] = {}
    for url, document in data.items():
        content = document.get("title", "") + " " + document.get("summary", "")
        extra_tags[url] = [
            tag["tag"] for tag in retriever(content) if tag["similarity"] > 0.2 and tag["tag"] not in tagged[url]
        ]

    # Merge extra tags into document data
    enriched_data = {url: {**{"extra-tags": extra_tags[url]}, **document} for url, document in data.items()}

    return enriched_data
