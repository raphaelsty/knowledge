"""
Zotero module for extracting bookmarks from Zotero libraries.

This module interfaces with the Zotero API to fetch saved items
and extract document metadata for the knowledge base.
"""

import datetime

from pyzotero import zotero

__all__ = ["Zotero"]


class Zotero:
    """
    Extract knowledge from Zotero reference libraries.

    Connects to a Zotero library (user or group) and fetches saved items,
    extracting titles, abstracts, dates, and tags.

    Parameters
    ----------
    library_id : str
        The numeric ID of the Zotero library.
    library_type : str
        Type of library: "user" for personal libraries or "group" for shared.
    api_key : str
        Zotero API key with read permissions for the library.

    Attributes
    ----------
    client : pyzotero.zotero.Zotero
        Authenticated Zotero API client.

    Example
    -------
    >>> from knowledge_database import zotero
    >>>
    >>> z = zotero.Zotero(
    ...     library_id="12345",
    ...     library_type="group",
    ...     api_key="your_api_key",
    ... )
    >>> documents = z(limit=100)
    >>>
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}: {doc['date']}")
    """

    def __init__(self, library_id: str, library_type: str, api_key: str):
        self.client = zotero.Zotero(library_id, library_type, api_key, preserve_json_order=True)

    def __call__(self, limit: int = 10000) -> dict[str, dict]:
        """
        Fetch items from Zotero and extract document metadata.

        Parameters
        ----------
        limit : int, default=10000
            Maximum number of items to fetch from the library.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping URLs to document metadata containing:
            - title: Item title
            - summary: Abstract/note content
            - date: Date the item was added to Zotero
            - tags: Lowercase tags assigned to the item
        """
        data: dict[str, dict] = {}

        for document in self.client.top(limit=limit):
            # Parse the date added timestamp
            date = datetime.datetime.strptime(document["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

            url = document["data"]["url"]
            title = document["data"]["title"]
            summary = document["data"]["abstractNote"]

            # Extract and normalize tags
            tags = [tag["tag"].lower() for tag in document["data"]["tags"]]

            data[url] = {
                "title": title,
                "summary": summary,
                "date": date,
                "tags": tags,
            }

        return data
