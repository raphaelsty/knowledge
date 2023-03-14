from pyzotero import zotero

import datetime

__all__ = ["Zotero"]


class Zotero:
    """Class for interacting with Zotero API

    Parameters
    ----------
    library_id
        The ID of the library to access.
    library_type
        The type of library to access. Must be one of "user" or "group".
    api_key
        The API key for the library.

    Example:
    --------

    >>> from knowledge_database import zotero

    >>> knowledge = zotero.Zotero(
    ...     library_id="library_id",
    ...     library_type="group",
    ...     api_key="api_key",
    ... )

    >>> knowledge(limit=10)

    """

    def __init__(self, library_id: str, library_type: str, api_key: str):
        self.client = zotero.Zotero(
            library_id, library_type, api_key, preserve_json_order=True
        )

    def __call__(self, limit: int = 10000):
        """Get bookmarks from Zotero."""
        data = {}

        for idx, document in enumerate(self.client.top(limit=limit)):

            date = datetime.datetime.strptime(
                document["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y-%m-%d")

            url = document["data"]["url"]

            title = document["data"]["title"]

            summary = document["data"]["abstractNote"]

            tags = [tag["tag"].lower() for tag in document["data"]["tags"]]

            data[url] = {
                "title": title,
                "summary": summary,
                "date": date,
                "tags": tags,
            }

        return data
