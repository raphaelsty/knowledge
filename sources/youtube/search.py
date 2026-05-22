"""
YouTube search — finds talks, interviews, and presentations featuring a person.

Searches YouTube for a person's name and extracts video metadata.
Captures talks on OTHER channels (conferences, podcasts, etc.)
that the person's own channel wouldn't have.

Uses YouTube's public search page — no API key needed.
"""

import datetime
import json
import re
import urllib.parse
import urllib.request

__all__ = ["Search"]


def _fallback_date() -> str:
    """Date stamp for videos whose real upload date we can't extract.

    YouTube now serves an anti-bot consent wall to anonymous
    server-side requests, so the watch-page fetch in
    :func:`_fetch_video_date` regularly returns nothing. We used
    to leave such rows undated, which the legacy upsert path then
    quietly defaulted to "today" — every failed fetch ended up
    masquerading as a brand-new video at the top of the feed.

    Instead, when we can't recover the real date, stamp the video
    as today − 3 years. The doc still surfaces in the
    timeline (which excludes NULL dates) but lands far enough
    back that it doesn't impersonate fresh content. This mirrors
    the Wikipedia-references fallback in
    `sources/wikipedia/references.py`.
    """
    return (datetime.date.today() - datetime.timedelta(days=365 * 3)).isoformat()


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# YouTube's watch page embeds the precise upload date in two places:
#   1. <meta itemprop="datePublished" content="YYYY-MM-DD">
#   2. A JSON-LD script with `"uploadDate":"YYYY-MM-DDTHH:MM:SS..."`.
# The search results payload only gives a fuzzy "2 years ago" text, so we
# hit the watch page once per result to pull the real date. Bounded by
# Search.max_results, fast enough for a sync fetcher.
_DATE_META_RE = re.compile(r'itemprop="datePublished"\s+content="(\d{4}-\d{2}-\d{2})"')
_DATE_JSON_RE = re.compile(r'"uploadDate"\s*:\s*"(\d{4}-\d{2}-\d{2})')


def _fetch_video_date(video_url: str) -> str:
    """Return the video's upload date as `YYYY-MM-DD`, or `""` on any error."""
    try:
        req = urllib.request.Request(video_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        m = _DATE_META_RE.search(body) or _DATE_JSON_RE.search(body)
        return m.group(1) if m else ""
    except Exception:
        return ""


class Search:
    """Search YouTube for videos featuring a person.

    Parameters
    ----------
    queries : list[str]
        Search queries (e.g. ["Max Halford talk", "Max Halford interview"]).
    max_results : int
        Maximum results per query.
    must_contain : list[str] | None
        If set, at least one of these strings must appear in the video title
        (case-insensitive). Filters out false positives like "Helford River".
    """

    def __init__(self, queries: list[str], max_results: int = 30, must_contain: list[str] | None = None):
        self.queries = queries
        self.max_results = max_results
        self.must_contain = [s.lower() for s in must_contain] if must_contain else None

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}

        for query in self.queries:
            print(f"    Searching YouTube: {query}")
            try:
                encoded = urllib.parse.quote(query)
                url = f"https://www.youtube.com/results?search_query={encoded}"
                req = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="replace")

                # Extract video data from ytInitialData JSON blob
                m = re.search(r"var ytInitialData = ({.*?});</script>", html, re.DOTALL)
                if not m:
                    print("    Could not parse YouTube search results")
                    continue

                yt_data = json.loads(m.group(1))
                contents = (
                    yt_data.get("contents", {})
                    .get("twoColumnSearchResultsRenderer", {})
                    .get("primaryContents", {})
                    .get("sectionListRenderer", {})
                    .get("contents", [])
                )

                count = 0
                for section in contents:
                    items = section.get("itemSectionRenderer", {}).get("contents", [])
                    for item in items:
                        video = item.get("videoRenderer")
                        if not video:
                            continue
                        vid_id = video.get("videoId", "")
                        if not vid_id:
                            continue

                        video_url = f"https://www.youtube.com/watch?v={vid_id}"
                        if existing_urls and video_url in existing_urls:
                            continue
                        if video_url in data:
                            continue

                        title_runs = video.get("title", {}).get("runs", [])
                        title = "".join(r.get("text", "") for r in title_runs)

                        desc_runs = video.get("detailedMetadataSnippets", [{}])
                        desc = ""
                        if desc_runs:
                            snippet_runs = desc_runs[0].get("snippetText", {}).get("runs", [])
                            desc = "".join(r.get("text", "") for r in snippet_runs)
                        if len(desc) > 200:
                            desc = desc[:197] + "..."

                        # `publishedTimeText` from search results is fuzzy ("2 years
                        # ago") — useless for our timeline-by-date ordering. The
                        # watch page carries the exact upload date in its meta /
                        # JSON-LD, so we fetch it once per result. One extra round
                        # trip per video, bounded by `max_results`. When YouTube
                        # blocks the fetch (consent wall, anti-bot), fall back
                        # to `today - 3y` so the doc surfaces in the timeline
                        # but doesn't impersonate fresh content.
                        published = _fetch_video_date(video_url) or _fallback_date()

                        channel = video.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
                        if channel and not desc:
                            desc = f"Video by {channel}"

                        # Filter: title, description, or channel must contain a required string
                        if self.must_contain:
                            haystack = f"{title} {desc} {channel}".lower()
                            if not any(s in haystack for s in self.must_contain):
                                continue

                        data[video_url] = {
                            "title": title,
                            "summary": desc,
                            "date": published,
                            "tags": ["youtube"],
                        }
                        count += 1
                        if count >= self.max_results:
                            break
                    if count >= self.max_results:
                        break

                print(f"    Found {count} videos for '{query}'")

            except Exception as e:
                print(f"    YouTube search error: {e}")

        print(f"    Total: {len(data)} videos")
        return data
