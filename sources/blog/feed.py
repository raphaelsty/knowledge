"""
RSS / Atom / RSS 1.0 (RDF) / JSON Feed fetcher.
"""

from ._helpers import _decode_xml, _dispatch_feed, _http_fetch

__all__ = ["Feed"]

# Per-feed entry cap. Most blogs publish < 100 entries per feed, but
# aggregators and `feed.json` full-archive exports can ship thousands
# that we neither want in the index nor in the dead-link probe.
_FEED_MAX_ENTRIES = 1_000


class Feed:
    """
    Fetch blog posts from an RSS, Atom, RSS 1.0 (RDF), or JSON Feed.

    Parameters
    ----------
    feed_url : str
        URL of the feed.
    tags : list[str]
        Base tags to apply to all entries from this feed.
    max_entries : int | None
        Cap on entries returned. Default ``_FEED_MAX_ENTRIES`` (1000).
    """

    def __init__(
        self,
        feed_url: str,
        tags: list[str] | None = None,
        max_entries: int | None = None,
    ):
        self.feed_url = feed_url
        self.tags = tags or []
        self.max_entries = max_entries if max_entries is not None else _FEED_MAX_ENTRIES

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        print(f"    Fetching feed: {self.feed_url}")
        try:
            raw, final_url, ct = _http_fetch(self.feed_url)
        except Exception as e:
            print(f"    Failed to fetch feed: {e}")
            return {}

        text = _decode_xml(raw, ct)
        data = _dispatch_feed(text, self.tags, final_url)
        if not data:
            print(f"    Unknown or empty feed: {self.feed_url}")
            return {}

        before_dedup = len(data)
        if existing_urls:
            data = {url: doc for url, doc in data.items() if url not in existing_urls}

        if len(data) > self.max_entries:
            # Python dicts are insertion-ordered, and feed parsers emit
            # newest-first, so slicing keeps the most recent entries.
            data = dict(list(data.items())[: self.max_entries])

        # Make the no-new-entries case obvious in the log instead of a
        # silent "Parsed 0 entries". Same listing fetch was performed,
        # so any new URL the feed publishes still gets discovered.
        if existing_urls and before_dedup > 0 and not data:
            print(f"    Feed up-to-date: {before_dedup} entries all known")
        else:
            print(f"    Parsed {len(data)} entries from feed")
        return data
