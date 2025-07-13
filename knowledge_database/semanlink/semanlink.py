import collections
import datetime
import typing

import rdflib

__all__ = ["Semanlink"]


class Semanlink:
    """Semanlink Knowledge base.

    Parameters
    ----------
    urls
        List of urls to the Semanlink knowledge base.

    Example
    -------

    >>> from knowledge_database import semanlink

    >>> semanlink_knowledge = semanlink.Semanlink(
    ...     urls = [
    ...         "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sldocs-2023-01-26.ttl",
    ...         "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sltags-2020-11-18.ttl",
    ...     ]
    ... )

    >>> semanlink_knowledge()

    """

    def __init__(self, urls: typing.List):
        self.urls = urls

    def __call__(self):

        graph = rdflib.Graph()
        triples = []
        for url in self.urls:

            turtle = graph.parse(url, format="turtle")

            triples += [
                (head.toPython(), relation.toPython(), tail.toPython())
                for head, relation, tail in turtle
            ]

        data = collections.defaultdict(lambda: collections.defaultdict(list))
        for head, relation, tail in triples:
            relation = relation.split("/")[-1].split("#")[-1]
            data[head][relation].append(tail)

        clean = collections.defaultdict(dict)

        for _, metadata in data.items():

            valid = True

            for relation in [
                "bookmarkOf",
                "arxiv_summary",
                "title",
                "arxiv_published",
                "arxiv_author",
            ]:
                if relation not in metadata:

                    valid = False
                    break

            if not valid:
                continue

            url = metadata["bookmarkOf"][0]

            summary = metadata["arxiv_summary"][0]

            date = datetime.datetime.strptime(
                metadata["arxiv_published"][0], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y-%m-%d")

            title = metadata["title"][0]

            tags = list(
                set(
                    [
                        " ".join(tag.split("/")[-1].lower().split("_"))
                        for tag in metadata["tag"]
                    ]
                )
            )

            clean[url] = {
                "tags": tags,
                "title": title,
                "summary": summary,
                "date": date,
            }

        return clean
