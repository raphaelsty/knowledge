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
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

# Reused rather than re-implemented: `_sample_pages` is the hardened
# parallel page fetcher the blog sitemap crawler uses (bounded read,
# non-HTML skip, gzip/deflate, per-host concurrency cap, empty on any
# error), and `_strip_html` / `_clean_summary` are the same tag+entity
# stripper and word-boundary truncator every other source summary goes
# through. Duplicating either here would just mean two of them drifting.
from sources.blog._helpers import _clean_summary, _strip_html
from sources.blog.sitemap import _sample_pages

__all__ = ["Frontpage"]

_TOPSTORIES = "https://hacker-news.firebaseio.com/v0/topstories.json"
_ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
_HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"
_DELAY = 0.05
_TIMEOUT = 15

_URL_RE = re.compile(r"https?://\S+")
# Minimum characters of non-URL text before we'll call something a
# summary. Plenty of front-page self-posts are just a pile of links
# ("Apple introduces M6" → three press-release URLs and nothing else);
# stripping the tags leaves bare URLs, which reads worse on a card than
# the vote-count fallback does.
_MIN_PROSE_CHARS = 40

# arXiv abs/pdf pages advertise a meta description of "Abstract page for
# arXiv paper 2608.21590: <title>", which just repeats the title we
# already have. The Atom API returns the real abstract, and `id_list`
# takes the whole batch in one request — so no need for the 3 s
# inter-request courtesy delay `sources/arxiv/papers.py` observes.
_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


def _is_prose(text: str) -> bool:
    """True if `text` carries enough non-URL prose to be a summary."""
    return len(_URL_RE.sub(" ", text).strip()) >= _MIN_PROSE_CHARS


def _arxiv_abstracts(urls: list[str]) -> dict[str, str]:
    """``{url: abstract}`` for any arXiv links among `urls`.

    One batched API call for the lot. Returns ``{}`` on any failure —
    callers fall through to the generic page scrape.
    """
    by_id: dict[str, str] = {}
    for url in urls:
        match = _ARXIV_ID_RE.search(url)
        if match:
            by_id.setdefault(match.group(1), url)
    if not by_id:
        return {}
    try:
        query = f"{_ARXIV_API}?id_list={','.join(sorted(by_id))}&max_results={len(by_id)}"
        req = urllib.request.Request(query, headers={"User-Agent": "Knowledge/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            root = ET.fromstring(resp.read())
    except Exception as exc:
        print(f"    HN frontpage: arXiv abstracts failed: {exc}")
        return {}

    out: dict[str, str] = {}
    for entry in root.findall("atom:entry", _ARXIV_NS):
        id_el = entry.find("atom:id", _ARXIV_NS)
        summary_el = entry.find("atom:summary", _ARXIV_NS)
        if id_el is None or summary_el is None:
            continue
        match = _ARXIV_ID_RE.search(id_el.text or "")
        if not match:
            continue
        url = by_id.get(match.group(1))
        if url:
            out[url] = (summary_el.text or "").strip()
    return out


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
                    # Filled in by `_fill_summaries` below.
                    "summary": "",
                    "points": points,
                    "num_comments": num_comments,
                    "submitted_at": submitted_at,
                    "by": item.get("by") or "",
                    # Self-post body (Ask HN / Show HN). Scratch key —
                    # consumed and removed by `_fill_summaries`.
                    "_text": item.get("text") or "",
                }
            )
            time.sleep(_DELAY)
        self._fill_summaries(out)
        print(f"    HN frontpage: {len(out)} stories")
        return out

    @staticmethod
    def _fill_summaries(items: list[dict]) -> None:
        """Give every item a real description, in place.

        The summary used to be the literal string "N points, M comments
        on HN" for every story, so the feed card showed vote counts
        where its abstract belongs. Now:

          1. Self-posts (Ask HN / Show HN) carry their body in the
             Firebase item's `text` — no extra request needed.
          2. arXiv links get their real abstract from the Atom API,
             which the abs page's own meta description doesn't carry.
          3. Everything else gets one bounded fetch of the linked
             page's description / og:description / twitter:description,
             fanned out in parallel.
          4. The points/comments line survives only as a last resort,
             for pages that block scrapers (twitter.com, say) or expose
             no description at all — the original intent was to keep a
             card from rendering empty, and that still holds.
        """
        needs_fetch: list[dict] = []
        for it in items:
            body = _clean_summary(_strip_html(it.pop("_text", "")))
            if _is_prose(body):
                it["summary"] = body
            else:
                needs_fetch.append(it)

        pending = sorted({it["url"] for it in needs_fetch})
        abstracts = _arxiv_abstracts(pending)
        metas = _sample_pages([u for u in pending if u not in abstracts]) if pending else {}
        described = 0
        for it in needs_fetch:
            if it["url"] in abstracts:
                description = abstracts[it["url"]]
            else:
                _title, description, _date = metas.get(it["url"], ("", "", ""))
            description = _clean_summary(_strip_html(description))
            if _is_prose(description):
                it["summary"] = description
                described += 1
            else:
                it["summary"] = f"{it['points']} points, {it['num_comments']} comments on HN"
        if needs_fetch:
            print(f"    HN frontpage: described {described}/{len(needs_fetch)} linked pages")
