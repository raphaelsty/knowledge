"""
Semanlink module for parsing RDF/Turtle knowledge bases.

This module parses Semanlink's semantic web data format to extract
document metadata from arXiv papers and other linked resources.
"""

import collections
import datetime

import rdflib

__all__ = ["Semanlink"]


class Semanlink:
    """
    Extract knowledge from Semanlink RDF/Turtle files.

    Parses Semanlink's semantic knowledge base format, which uses RDF
    triples to represent documents, tags, and their relationships.
    Focuses on extracting arXiv paper metadata.

    Parameters
    ----------
    urls : list[str]
        List of URLs pointing to Turtle (.ttl) files to parse.

    Example
    -------
    >>> from knowledge_database import semanlink
    >>>
    >>> sl = semanlink.Semanlink(
    ...     urls=[
    ...         "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sldocs-2023-01-26.ttl",
    ...         "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sltags-2020-11-18.ttl",
    ...     ]
    ... )
    >>> documents = sl()
    >>>
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}: {len(doc['tags'])} tags")
    """

    def __init__(self, urls: list[str]):
        self.urls = urls

    def __call__(self) -> dict[str, dict]:
        """
        Parse Turtle files and extract document metadata.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping URLs to document metadata containing:
            - title: Paper title
            - summary: arXiv abstract
            - date: Publication date
            - tags: Semantic tags (cleaned and normalized)
        """
        graph = rdflib.Graph()
        triples = []

        # Parse all Turtle files and collect triples
        for url in self.urls:
            turtle = graph.parse(url, format="turtle")
            triples += [(head.toPython(), relation.toPython(), tail.toPython()) for head, relation, tail in turtle]

        # Group triples by subject into a nested dictionary
        data: dict = collections.defaultdict(lambda: collections.defaultdict(list))
        for head, relation, tail in triples:
            # Extract relation name from full URI
            relation = relation.split("/")[-1].split("#")[-1]
            data[head][relation].append(tail)

        # Extract documents with required arXiv metadata
        clean: dict[str, dict] = collections.defaultdict(dict)

        required_relations = [
            "bookmarkOf",
            "arxiv_summary",
            "title",
            "arxiv_published",
            "arxiv_author",
        ]

        for _, metadata in data.items():
            # Check all required fields are present
            valid = all(relation in metadata for relation in required_relations)

            if not valid:
                continue

            url = metadata["bookmarkOf"][0]
            summary = metadata["arxiv_summary"][0]

            # Parse arXiv timestamp
            date = datetime.datetime.strptime(metadata["arxiv_published"][0], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

            title = metadata["title"][0]

            # Clean and normalize tags (convert underscores to spaces, lowercase)
            tags = list({" ".join(tag.split("/")[-1].lower().split("_")) for tag in metadata["tag"]})

            clean[url] = {
                "tags": tags,
                "title": title,
                "summary": summary,
                "date": date,
            }

        return clean
