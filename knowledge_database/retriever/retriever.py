"""
Retriever module for BM25-based document and tag search.

This module provides sparse retrieval capabilities using character-level
BM25 vectorization for typo-tolerant search across documents and tags.
"""

import copy

from cherche import retrieve
from lenlp import sparse

__all__ = ["Retriever"]


class Retriever:
    """
    BM25-based retriever for documents and tags.

    Uses character n-gram BM25 vectorization for fuzzy matching that handles
    typos and partial matches. Maintains separate indices for document search
    and tag search.

    Parameters
    ----------
    documents : dict[str, dict]
        Dictionary mapping URLs to document metadata. Each document should
        contain 'title', 'summary', 'date', 'tags', and optionally 'extra-tags'.

    Attributes
    ----------
    retriever : cherche.retrieve.TfIdf
        Document retriever with BM25 scoring over title, summary, tags, and date.
    retriever_tags : cherche.retrieve.TfIdf
        Tag retriever for matching query terms to known tags.

    Example
    -------
    >>> import json
    >>> from knowledge_database import retriever
    >>>
    >>> with open("database/database.json") as f:
    ...     documents = json.load(f)
    >>>
    >>> r = retriever.Retriever(documents=documents)
    >>>
    >>> # Search documents
    >>> results = r.documents("neural search")
    >>>
    >>> # Find matching tags
    >>> tags = r.tags("deep learning")
    """

    def __init__(self, documents: dict):
        updated_documents = copy.deepcopy(documents)

        # Convert dict to list format with URL included
        documents_list = [{"url": url, **document} for url, document in documents.items()]

        # Flatten tags into searchable text field
        updated_documents_list = [
            {
                **{
                    "url": url,
                    "tags": " ".join(document.pop("tags") + document.pop("extra-tags")),
                },
                **document,
            }
            for url, document in updated_documents.items()
        ]

        # Build document retriever with character n-gram BM25
        # Using char_wb analyzer for word-boundary-aware character n-grams
        self.retriever = (
            retrieve.TfIdf(
                key="url",
                on=["title", "tags", "summary", "date", "extra-tags"],
                k=30,
                tfidf=sparse.BM25Vectorizer(
                    normalize=True,
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    b=0,  # Disable length normalization for consistent scoring
                ),
                documents=updated_documents_list,
            )
        ) + documents_list

        # Build unique tag index
        tags_dict: dict[str, bool] = {}
        for document in documents_list:
            for tag in document["tags"] + document["extra-tags"]:
                tags_dict[tag] = True
        tags_list = [{"tag": tag} for tag in tags_dict]

        # Build tag retriever with character n-gram BM25
        self.retriever_tags = (
            retrieve.TfIdf(
                key="tag",
                on=["tag"],
                k=10,
                tfidf=sparse.BM25Vectorizer(
                    normalize=True,
                    ngram_range=(2, 5),
                    analyzer="char",
                ),
                documents=tags_list,
            )
            + tags_list
        )

    def documents(self, q: str) -> list[dict]:
        """
        Search for documents matching the query.

        Parameters
        ----------
        q : str
            Search query string.

        Returns
        -------
        list[dict]
            List of matching documents ranked by BM25 score, each containing
            'url', 'title', 'summary', 'date', 'tags', 'extra-tags', and 'similarity'.
        """
        return self.retriever(q)

    def tags(self, q: str) -> list[str]:
        """
        Find tags matching the query.

        Parameters
        ----------
        q : str
            Search query string.

        Returns
        -------
        list[str]
            List of matching tag names ranked by BM25 score.
        """
        return [tag["tag"] for tag in self.retriever_tags(q)]

    def documents_tags(self, q: str) -> list[dict]:
        """
        Search for documents with tag-based filtering.

        Currently identical to documents() but can be extended for
        tag-specific filtering logic.

        Parameters
        ----------
        q : str
            Search query string.

        Returns
        -------
        list[dict]
            List of matching documents ranked by BM25 score.
        """
        return self.retriever(q)
