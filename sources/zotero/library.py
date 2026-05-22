"""
Zotero module for extracting bookmarks from Zotero libraries.

This module interfaces with the Zotero API to fetch saved items
and extract document metadata for the knowledge base.
"""

import datetime

from pyzotero import zotero

__all__ = ["Library"]


class Library:
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
    >>> from sources import zotero
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

    def __call__(self, limit: int | None = None) -> dict[str, dict]:
        """
        Fetch every top-level item from the library.

        The Zotero API caps ``limit`` at 100 per request, so we wrap the
        ``top()`` call with ``client.everything()`` which paginates
        through all pages. Missing URLs are skipped (can't de-dup or
        index books / offline PDFs meaningfully), and missing fields
        fall back to empty strings so one malformed item doesn't crash
        the whole crawl.

        Parameters
        ----------
        limit : int or None
            Optional safety cap. None (the default) means no cap — we
            crawl the whole library.
        """
        data: dict[str, dict] = {}

        # everything() auto-paginates; 100 is the API max per page.
        items = self.client.everything(self.client.top(limit=100))

        for document in items:
            d = document.get("data") or {}
            url = (d.get("url") or "").strip()
            if not url:
                continue

            raw_date = d.get("dateAdded") or ""
            try:
                date = datetime.datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                date = ""

            tags = [(t.get("tag") or "").lower() for t in (d.get("tags") or []) if t.get("tag")]

            data[url] = {
                "title": d.get("title") or "",
                "summary": d.get("abstractNote") or "",
                "date": date,
                "tags": tags,
            }

            if limit is not None and len(data) >= limit:
                break

        return data
