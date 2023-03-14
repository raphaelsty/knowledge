import collections
import itertools
import typing

from cherche import retrieve
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["get_extra_tags", "get_tags_triples"]


def get_tags_triples(data: typing.List, excluded_tags=None):
    """Create a graph of interconnected tags."""
    excluded_tags = {} if excluded_tags is None else excluded_tags

    triples = []

    seen = collections.defaultdict(dict)

    for _, document in data.items():

        tags = document["tags"] + document["extra-tags"]
        for head, tail in itertools.combinations(tags, 2):

            if head in excluded_tags or tail in excluded_tags:
                continue

            if head in seen[tail] or tail in seen[head]:
                continue

            triples.append({"head": head, "tail": tail})
            seen[head][tail] = True
            seen[tail][head] = True

    return triples


def get_extra_tags(data: typing.List):
    """Create extra-tags for each document."""

    documents = {}
    tagged = collections.defaultdict(dict)

    for url, document in data.items():
        for tag in document["tags"]:
            documents[tag] = True
            tagged[url][tag] = True

    documents = [{"tag": tag} for tag in documents]
    retriever = (
        retrieve.Flash(
            key="tag",
            on="tag",
            k=10,
        )
        | retrieve.TfIdf(
            key="tag",
            on="tag",
            documents=documents,
            k=3,
            tfidf=TfidfVectorizer(
                lowercase=True, ngram_range=(4, 7), analyzer="char_wb"
            ),
        )
    ).add(documents)

    extra_tags = {}
    for url, document in data.items():
        extra_tags[url] = [
            tag["tag"]
            for tag in retriever(
                document.get("title", "") + " " + document.get("summary", "")
            )
            if tag["similarity"] > 0.2 and tag["tag"] not in tagged[url]
        ]

    data = {
        url: {**{"extra-tags": extra_tags[url]}, **document}
        for url, document in data.items()
    }

    return data
