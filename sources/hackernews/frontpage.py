"""
HackerNews front-page snapshot.

Pulls the current top-30 stories straight from the official Firebase
API (`topstories.json` + per-item lookups). Unlike the per-user
`Submissions` / `Upvotes` / `Comments` fetchers, this one is **global
and stateless**: it just returns whatever HN is showing the world right
now. The driver script (`scripts/hn_frontpage.py`) is what bolts the
per-user relevance scoring on top.
"""

from __future__ import annotations

import json
import time
import urllib.request

__all__ = ["Frontpage"]

_TOPSTORIES = "https://hacker-news.firebaseio.com/v0/topstories.json"
_ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
_HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"
_DELAY = 0.05
_TIMEOUT = 15


def _fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read())


class Frontpage:
    """Fetch the current HN front page as a list of dicts.

    Each entry: ``{hn_id, rank, url, title, summary, points,
    num_comments, submitted_at, by}``. ``url`` falls back to the HN
    discussion page for Ask/Show/text posts that have no external link.
    Order follows HN's own ranking, so list index = rank-1.
    """

    def __init__(self, top: int = 30):
        self.top = top

    def __call__(self) -> list[dict]:
        try:
            ids = _fetch_json(_TOPSTORIES)
        except Exception as exc:
            print(f"    HN frontpage: topstories fetch failed: {exc}")
            return []
        if not isinstance(ids, list):
            return []
        ids = ids[: self.top]

        out: list[dict] = []
        for rank, hn_id in enumerate(ids, start=1):
            try:
                item = _fetch_json(_ITEM.format(id=hn_id))
            except Exception as exc:
                print(f"    HN frontpage: item {hn_id} failed: {exc}")
                continue
            if not isinstance(item, dict) or item.get("type") != "story":
                continue
            url = item.get("url") or _HN_ITEM_URL.format(id=hn_id)
            title = (item.get("title") or "").strip()
            if not title:
                continue
            points = int(item.get("score") or 0)
            num_comments = int(item.get("descendants") or 0)
            submitted_at = int(item.get("time") or 0)
            out.append(
                {
                    "hn_id": int(hn_id),
                    "rank": rank,
                    "url": url,
                    "title": title,
                    # Score + comment count is a useful proxy summary
                    # — keeps the card non-empty when we have no other
                    # description handy.
                    "summary": f"{points} points, {num_comments} comments on HN",
                    "points": points,
                    "num_comments": num_comments,
                    "submitted_at": submitted_at,
                    "by": item.get("by") or "",
                }
            )
            time.sleep(_DELAY)
        print(f"    HN frontpage: {len(out)} stories")
        return out
