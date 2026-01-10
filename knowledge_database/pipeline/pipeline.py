"""
Pipeline module for unified knowledge search and visualization.

This module combines document retrieval with graph-based tag visualization,
providing a single interface for searching and exploring the knowledge base.
"""

import datetime

from ..graph import Graph
from ..retriever import Retriever

__all__ = ["Pipeline"]


class Pipeline:
    """
    Unified pipeline for knowledge search and graph visualization.

    Combines BM25-based document retrieval with a knowledge graph for
    interactive exploration of tag relationships. Supports both text
    search and chronological browsing.

    Parameters
    ----------
    documents : dict[str, dict]
        Dictionary mapping URLs to document metadata. Each document should
        contain 'title', 'summary', 'date', 'tags', and optionally 'extra-tags'.
    triples : list[dict]
        List of tag relationship edges for building the knowledge graph.
    excluded_tags : dict[str, bool], optional
        Tags to exclude from graph visualization (e.g., generic tags).
    k_latest_documents : int, default=200
        Number of recent documents to cache for the "latest" view.

    Attributes
    ----------
    retriever : Retriever
        BM25-based document and tag search engine.
    graph : Graph
        Knowledge graph for tag relationship visualization.
    latest_documents : list[dict]
        Cached list of most recently added documents.

    Example
    -------
    >>> import json
    >>> from knowledge_database import pipeline
    >>>
    >>> with open("database/database.json") as f:
    ...     documents = json.load(f)
    >>> with open("database/triples.json") as f:
    ...     triples = json.load(f)
    >>>
    >>> pipe = pipeline.Pipeline(documents=documents, triples=triples)
    >>>
    >>> # Search for documents
    >>> results = pipe.search("neural networks")
    >>>
    >>> # Get documents with graph visualization
    >>> docs, nodes, links = pipe("machine learning")
    >>>
    >>> # Get only graph data for visualization
    >>> nodes, links = pipe.plot("deep learning")
    """

    def __init__(
        self,
        documents: dict,
        triples: list,
        excluded_tags: dict | None = None,
        k_latest_documents: int = 200,
    ):
        self.retriever = Retriever(documents=documents)
        self.graph = Graph(triples=triples)
        self.excluded_tags = {} if excluded_tags is None else excluded_tags

        # Pre-compute document counts per tag for graph node sizing
        document_counts: dict[str, int] = {}
        for _url, document in documents.items():
            for tag in document.get("tags", []) + document.get("extra-tags", []):
                document_counts[tag] = document_counts.get(tag, 0) + 1
        self.graph.set_document_counts(document_counts)

        # Cache recent documents sorted by date (newest first)
        self.latest_documents = sorted(
            [{"url": url, **document} for url, document in documents.items()],
            key=lambda doc: datetime.datetime.strptime(doc["date"], "%Y-%m-%d"),
            reverse=True,
        )[:k_latest_documents]

    def get_latest_documents(self, count: int) -> list[dict]:
        """
        Get the most recently added documents.

        Parameters
        ----------
        count : int
            Number of documents to return.

        Returns
        -------
        list[dict]
            List of document dictionaries sorted by date (newest first).
        """
        return self.latest_documents[:count]

    def search(self, q: str, tags: bool = False) -> list[dict]:
        """
        Search for documents matching a query.

        Parameters
        ----------
        q : str
            Search query string.
        tags : bool, default=False
            If True, filter results to documents whose tags match the query.
            If False, search across all document fields.

        Returns
        -------
        list[dict]
            List of matching documents ranked by relevance.
        """
        if tags:
            return self.retriever.documents_tags(q)
        return self.retriever.documents(q)

    def __call__(
        self,
        q: str,
        k_tags: int = 20,
        k_yens: int = 3,
        k_walk: int = 3,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Search for documents and generate a knowledge graph visualization.

        Performs document retrieval and extracts tags from matching documents
        to build an interactive graph showing tag relationships.

        Parameters
        ----------
        q : str
            Search query string.
        k_tags : int, default=20
            Maximum number of document tags to include in the graph.
        k_yens : int, default=3
            Maximum number of shortest paths between tag pairs.
        k_walk : int, default=3
            Number of neighbors for random walk exploration.

        Returns
        -------
        documents : list[dict]
            Matching documents ranked by relevance.
        nodes : list[dict]
            Graph nodes representing tags with metadata.
        links : list[dict]
            Graph edges connecting related tags.
        """
        documents = self.retriever.documents(q)
        retrieved_tags = self.retriever.tags(q)

        # Collect unique tags from matching documents
        tags: dict[str, bool] = {}
        for document in documents:
            for tag in document["tags"] + document["extra-tags"]:
                if tag not in self.excluded_tags:
                    tags[tag] = True

        # Build graph from document tags and query-matched tags
        nodes, links = self.graph(
            tags=list(tags)[:k_tags],
            retrieved_tags=retrieved_tags,
            k_yens=k_yens,
            k_walk=k_walk,
        )
        return documents, nodes, links

    def plot(
        self,
        q: str,
        k_tags: int = 20,
        k_yens: int = 3,
        k_walk: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate a knowledge graph visualization for a query.

        Similar to __call__ but returns only the graph data without documents.
        Useful for graph-only visualizations.

        Parameters
        ----------
        q : str
            Search query string.
        k_tags : int, default=20
            Maximum number of document tags to include in the graph.
        k_yens : int, default=3
            Maximum number of shortest paths between tag pairs.
        k_walk : int, default=3
            Number of neighbors for random walk exploration.

        Returns
        -------
        nodes : list[dict]
            Graph nodes representing tags with metadata.
        links : list[dict]
            Graph edges connecting related tags.
        """
        _, nodes, links = self(q=q, k_tags=k_tags, k_yens=k_yens, k_walk=k_walk)
        return nodes, links
