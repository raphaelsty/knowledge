"""
YouTube channel video fetcher.

Fetches videos from a YouTube channel using the public RSS feed.
Accepts either a channel ID (UC...) or a handle (@name).

The RSS feed returns the 15 most recent videos with titles and descriptions.
No API key required.
"""

import datetime
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

__all__ = ["Channels"]


def _fallback_date() -> str:
    """Today − 3 years — used when the channel RSS doesn't carry an
    `<atom:published>` element. Matches
    :func:`sources.youtube.search._fallback_date` so videos from
    either fetcher land in the same "old enough to not impersonate
    today" tier."""
    return (datetime.date.today() - datetime.timedelta(days=365 * 3)).isoformat()


_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _resolve_handle(handle: str) -> str | None:
    """Resolve a YouTube @handle to a channel ID."""
    if handle.startswith("UC") and len(handle) == 24:
        return handle  # already a channel ID

    if not handle.startswith("@"):
        handle = f"@{handle}"

    url = f"https://www.youtube.com/{handle}/videos"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        all_ids = re.findall(r'"(UC[A-Za-z0-9_-]{22})"', html)
        if all_ids:
            return Counter(all_ids).most_common(1)[0][0]
    except Exception:
        pass
    return None


def _fetch_channel_feed(channel_id: str) -> list[dict]:
    """Fetch videos from a channel's RSS feed."""
    feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    req = urllib.request.Request(feed_url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        root = ET.fromstring(resp.read())

    videos = []
    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        vid_el = entry.find("yt:videoId", _NS)
        published_el = entry.find("atom:published", _NS)
        desc_el = entry.find("media:group/media:description", _NS)

        if title_el is None or vid_el is None:
            continue

        title = title_el.text or ""
        vid_id = vid_el.text or ""
        date = (published_el.text or "")[:10] if published_el is not None else ""
        if not date:
            # RSS occasionally drops the `<published>` element for
            # older videos. Fall back to today − 3y rather than
            # leaving the row undated.
            date = _fallback_date()
        desc = (desc_el.text or "")[:300] if desc_el is not None else ""

        videos.append(
            {
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": title.strip(),
                "summary": desc.strip(),
                "date": date,
            }
        )

    return videos


class Channels:
    """
    Fetch videos from YouTube channels.

    Parameters
    ----------
    channels : list[str]
        List of channel IDs (UC...) or handles (@name or just "name").
    """

    def __init__(self, channels: list[str]):
        self.channels = channels

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """Fetch videos from all channels."""
        data: dict[str, dict] = {}

        for channel in self.channels:
            print(f"    Fetching YouTube: {channel}")

            channel_id = _resolve_handle(channel)
            if not channel_id:
                print(f"    Could not resolve channel: {channel}")
                continue

            try:
                videos = _fetch_channel_feed(channel_id)
            except Exception as e:
                print(f"    Feed fetch error: {e}")
                continue

            added = 0
            for video in videos:
                url = video["url"]
                if existing_urls and url in existing_urls:
                    continue
                if url in data:
                    continue
                data[url] = {
                    "title": video["title"],
                    "summary": video["summary"],
                    "date": video["date"],
                    "tags": ["youtube"],
                }
                added += 1

            print(f"    {added} videos from {channel}")

        print(f"    Total: {len(data)} videos")
        return data
