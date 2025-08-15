import copy
import typing

from cherche import retrieve
from lenlp import sparse

__all__ = ["Retriever"]


class Retriever:
    """Knowledge retriever.

    Parameters
    ----------
    documents
        List of documents.

    Examples:
    ---------

    >>> import json
    >>> from knowledge_database import retriever

    >>> with open("database/database.json", "r") as f:
    ...     documents = json.load(f)

    >>> knowledge_retriever = retriever.Retriever(documents=documents)

    >>> candidates = knowledge_retriever.documents("neural search")
    >>> candidates = knowledge_retriever.tags("neural search")
    >>> candidates = knowledge_retriever.documents_tags("neural search")

    """

    def __init__(self, documents: typing.Dict):
        updated_documents = copy.deepcopy(documents)
        documents = [{"url": url, **document} for url, document in documents.items()]

        updated_documents = [
            {
                **{
                    "url": url,
                    "tags": " ".join(document.pop("tags") + document.pop("extra-tags")),
                },
                **document,
            }
            for url, document in updated_documents.items()
        ]

        self.retriever = (
            retrieve.TfIdf(
                key="url",
                on=["title", "tags", "summary", "date"],
                k=30,
                tfidf=sparse.BM25Vectorizer(
                    normalize=True,
                    ngram_range=(3, 6),
                    analyzer="char",
                    b=0,
                ),
                documents=updated_documents,
            )
        ) + documents

        tags = {}
        for document in documents:
            for tag in document["tags"] + document["extra-tags"]:
                tags[tag] = True
        tags = [{"tag": tag} for tag in tags]

        self.retriever_tags = (
            retrieve.TfIdf(
                key="tag",
                on=["tag"],
                k=5,
                tfidf=sparse.BM25Vectorizer(
                    normalize=True,
                    ngram_range=(2, 5),
                    analyzer="char_wb",
                ),
                documents=tags,
            )
            + tags
        )

    def documents(self, q: str):
        """Match documents."""
        return self.retriever(q)

    def tags(self, q: str):
        """Match tags."""
        return [tag["tag"] for tag in self.retriever_tags(q)]

    def documents_tags(self, q: str):
        """Match documents and tags."""
        return self.retriever(q)
