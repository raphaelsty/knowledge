"""
Twitter/X source — fetches tweets and thread-root resources via TwitterAPI.io.

Produces two kinds of documents:

1. **Own tweets** — the user's original posts.
   ``{url: {title, summary, date, tags: ["twitter"]}}``

2. **Thread-root resources** — external URLs (arxiv, github, …) found in
   the root tweet of conversations the user replied in.
   ``{url: {title, summary, date, tags: ["twitter-thread", …], source_url}}``

The companion ``filter_tweets()`` function runs a zero-shot model2vec
classifier over tweet-only documents and drops mood / social noise while
letting all resource documents pass through untouched.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import requests

__all__ = ["Tweets", "filter_tweets"]


# ═══════════════════════════════════════════════════════════════════════════
# Compiled patterns
# ═══════════════════════════════════════════════════════════════════════════

_SKIP_DOMAINS = frozenset(
    {
        "x.com",
        "twitter.com",
        "t.co",
        "pic.twitter.com",
        "pbs.twimg.com",
        "imgur.com",
        "i.imgur.com",
        "i.redd.it",
        "preview.redd.it",
        "giphy.com",
        "media.giphy.com",
        "imgbb.com",
        "i.ibb.co",
        "postimg.cc",
        "i.postimg.cc",
        "flickr.com",
        "flic.kr",
    }
)

_IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".ico",
        ".tif",
        ".tiff",
        ".heic",
        ".avif",
    }
)

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)")
_RAW_URL_RE = re.compile(r"https?://[^\s)\]'\">]+", re.IGNORECASE)
_TWEET_URL_RE = re.compile(r"^https?://x\.com/[^/]+/status/\d+/?$")
# X's long-form "Article" paths — these are standalone content
# rather than navigation back into another tweet, so the
# skip-domain rule in `_extract_urls` makes an exception for
# them.
_IS_X_ARTICLE_PATH = re.compile(r"^/i/article/\d+/?$")

_INTERESTING_TOKENS = re.compile(
    r"\b(paper|preprint|arxiv|github|repo|code|gist|notebook|"
    r"blog|article|post|video|talk|lecture|demo|hugging|hf\.co|"
    r"check|see|read|via|here|link|thread)\b",
    re.IGNORECASE,
)

_API_BASE = "https://api.twitterapi.io"


# ═══════════════════════════════════════════════════════════════════════════
# URL helpers
# ═══════════════════════════════════════════════════════════════════════════


def _source_tag(url: str) -> str | None:
    """Map a URL to a knowledge-base source label.

    Known platforms collapse to a brand label (``arxiv``, ``github``,
    ``huggingface`` …) so the filter chip groups every paper / repo /
    model under one bucket regardless of subdomain.

    Anything else falls back to the bare hostname (e.g.
    ``mixedbread.com``, ``lighton.ai``) so each website surfaces as its
    own filter chip on the search page — matching what the website
    fetcher does for first-party feeds and sitemaps. Without this, every
    tweeted blog link bucketed into a generic ``blog`` chip and the user
    couldn't tell mixedbread from lighton in the sources panel.

    Returns ``None`` only when the URL is unparseable.
    """
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return None
    if not host:
        return None
    _map = {
        "arxiv.org": "arxiv",
        "gist.github.com": "github",
        "github.com": "github",
        "news.ycombinator.com": "hackernews",
        "huggingface.co": "huggingface",
        "youtube.com": "youtube",
        "youtu.be": "youtube",
        "scholar.google.com": "scholar",
    }
    for domain, tag in _map.items():
        if host == domain or host.endswith(f".{domain}"):
            return tag
    # Fall back to the hostname so every website gets its own chip.
    return host


# ── Page-meta extraction ────────────────────────────────────────────────
# When a tweet surfaces an external URL, the document stored for that URL
# should describe the *linked page*, not the tweet. These helpers pull
# <title> and <meta description> with a short timeout; failures fall back
# to a URL-slug title.

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RES = [
    re.compile(
        rf'<meta[^>]*\b{attr}=["\']?{val}["\']?[^>]*\bcontent=["\']([^"\']+)',
        re.IGNORECASE,
    )
    for attr, val in (
        ("name", "description"),
        ("property", "og:description"),
        ("name", "twitter:description"),
    )
]
# Image-equivalent of `_META_DESC_RES`. Order matters — og:image is
# the canonical and gives us the upstream-curated preview; the
# twitter:image variants exist for sites that ship only Twitter cards
# (no Open Graph), and the bare `image_src` <link> is a last resort.
_META_IMAGE_RES = [
    re.compile(
        rf'<meta[^>]*\b{attr}=["\']?{val}["\']?[^>]*\bcontent=["\']([^"\']+)',
        re.IGNORECASE,
    )
    for attr, val in (
        ("property", "og:image"),
        ("name", "twitter:image"),
        ("name", "twitter:image:src"),
    )
] + [
    re.compile(
        r'<link[^>]*\brel=["\']?image_src["\']?[^>]*\bhref=["\']([^"\']+)',
        re.IGNORECASE,
    )
]


def _strip_html_text(s: str) -> str:
    from html import unescape

    s = _HTML_TAG_RE.sub(" ", s)
    s = unescape(s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def _fetch_youtube_oembed(url: str, timeout: float = 5.0) -> tuple[str | None, str | None]:
    """Pull ``(title, description)`` for a YouTube video via oEmbed.

    YouTube serves a JS-rendered shell to plain HTTP clients, so the
    generic <title> / og:description scrape returns nothing useful and
    we fall back to the URL slug — that's how "k4iC5YYvrQk" ends up as
    a title. oEmbed is a 1-byte JSON endpoint that returns the real
    video title + channel name without an API key.

    Description is synthesised as ``Video by <channel>`` since oEmbed
    doesn't carry the actual transcript / blurb.
    """
    import json as _json

    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return None, None
    if not (host == "youtu.be" or host.endswith("youtube.com") or host == "www.youtube.com"):
        return None, None
    try:
        oe_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url, safe='')}&format=json"
        req = urllib.request.Request(
            oe_url,
            headers={"User-Agent": "Knowledge/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read(16_384).decode("utf-8", errors="replace"))
    except Exception:
        return None, None
    title = (data.get("title") or "").strip() or None
    author = (data.get("author_name") or "").strip()
    description = f"Video by {author}" if author else None
    return title, description


def _fetch_page_meta(url: str, timeout: float = 5.0) -> tuple[str | None, str | None]:
    """Return ``(title, description)`` for a URL, or ``(None, None)`` on failure.

    Thin wrapper around :func:`_fetch_page_preview` for callers that
    don't need the OG image — kept so the existing 2-tuple call sites
    don't have to change.
    """
    title, description, _ = _fetch_page_preview(url, timeout=timeout)
    return title, description


def _fetch_page_preview(url: str, timeout: float = 5.0) -> tuple[str | None, str | None, str | None]:
    """Return ``(title, description, og_image)`` for a URL.

    YouTube oEmbed gives us title + author but no preview URL, so we
    still parse the HTML head for the image; we just keep oEmbed's
    title/description if the HTML scrape doesn't have something
    better. Any failure (bad URL, network, non-200, non-HTML) returns
    a tuple of ``None``s — callers must treat each field as optional.
    """
    yt_title, yt_desc = _fetch_youtube_oembed(url, timeout=timeout)
    # Wrap BOTH Request() construction and urlopen() in the try.
    # `Request(url, …)` raises ValueError when `url` has no scheme
    # (e.g. a malformed t.co expansion like 'replays.ht'), and that
    # exception used to escape the per-tweet meta fetch and tank the
    # whole twitter task. Any URL we can't fetch should silently fall
    # back to "no metadata".
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Knowledge/1.0"
                ),
                "Accept": "text/html,*/*;q=0.5",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(65_536)
    except Exception:
        return yt_title, yt_desc, None

    html = raw.decode("utf-8", errors="replace")

    title = None
    m = _TITLE_RE.search(html)
    if m:
        title = _strip_html_text(m.group(1)) or None

    description = None
    for regex in _META_DESC_RES:
        m = regex.search(html)
        if m:
            description = _strip_html_text(m.group(1)) or None
            if description:
                break

    image = None
    for regex in _META_IMAGE_RES:
        m = regex.search(html)
        if m:
            cand = (m.group(1) or "").strip()
            if cand:
                # Resolve protocol-relative + relative URLs against
                # the page itself so the stored link is absolute.
                image = urllib.parse.urljoin(url, cand)
                break

    return (title or yt_title, description or yt_desc, image)


def _title_from_url_slug(url: str) -> str:
    """Fallback title: convert the last URL path segment into a readable string."""
    try:
        path = urlparse(url).path.rstrip("/")
    except Exception:
        return url
    slug = path.rsplit("/", 1)[-1] if "/" in path else path
    if not slug:
        return url
    cleaned = slug.replace("-", " ").replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else url


def _extract_urls(tweet: dict) -> list[str]:
    """Return deduplicated external URLs from a tweet payload.

    Checks three locations (in order of reliability):
    1. ``entities.urls[*].expanded_url``   — standard Twitter API shape
    2. ``tweet.urls[*]``                   — TwitterAPI.io top-level variant
    3. Regex over the raw tweet text       — last resort
    """
    out: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        raw = raw.strip().rstrip(".,;:)\"'")
        if not raw:
            return
        # `urlparse` can raise `ValueError: Invalid IPv6 URL` on
        # malformed inputs (mostly from regex-extracted text — t.co
        # expansions occasionally land with bracketed garbage).
        # Anything we can't parse, we drop on the floor — there's no
        # signal in a URL we can't even normalize.
        try:
            parsed = urlparse(raw)
        except ValueError:
            return
        host = parsed.netloc.lower()
        if any(host.endswith(d) for d in _SKIP_DOMAINS):
            # Twitter long-form Articles (e.g.
            # `x.com/i/article/<id>`) live on x.com but are
            # actual standalone content the author posted, not
            # navigation back into another tweet. Keep them so
            # they surface as a link-preview card; the skip-domain
            # rule otherwise treats every x.com URL as "yet
            # another tweet" and drops them.
            if not _IS_X_ARTICLE_PATH.match(parsed.path or ""):
                return
        # Schemeless strings (e.g. "open.ai", "stat.ML") sneak through
        # the regex fallback. Without a scheme they'd later choke
        # `urllib.request.Request()`. Drop them here at the source.
        if not parsed.scheme or parsed.scheme not in {"http", "https"}:
            return
        # Skip bare image files — no searchable knowledge.
        path_lower = parsed.path.lower().split("?")[0]
        if any(path_lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
            return
        m = _ARXIV_RE.search(raw)
        if m:
            raw = f"https://arxiv.org/abs/{m.group(1)}"
        if raw not in seen:
            seen.add(raw)
            out.append(raw)

    # 1. entities.urls
    for u in (tweet.get("entities") or {}).get("urls") or []:
        _push(u.get("expanded_url") or u.get("url") or "")

    # 2. top-level urls
    for u in tweet.get("urls") or []:
        _push(u if isinstance(u, str) else (u.get("expanded_url") or u.get("url") or ""))

    # 3. regex fallback
    for raw in _RAW_URL_RE.findall(tweet.get("text") or tweet.get("full_text") or ""):
        _push(raw)

    return out


def _canonical_host(url: str) -> str | None:
    """Return the canonical bare host of *url* (or None on parse fail).

    Strips `www.` and a single leading `blog.` so a `blog.foo.com`
    link and a `foo.com` link land under the same source chip.
    `_SKIP_DOMAINS` already filtered t.co / x.com noise upstream of
    every call to this helper.
    """
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return None
    if not host:
        return None
    host = re.sub(r"^(?:www|blog)\.", "", host)
    return host or None


# Hard cap on `linked_urls` per document. Some threads cite a long
# list ("see also: A, B, C, D, E, F, G, H"); keep the JSONB payload
# bounded so the row stays cheap to read and the index card doesn't
# turn into a wall of preview tiles.
MAX_LINKED_URLS_PER_DOC = 5


# Joiner the frontend's `renderTweetSummary` (web/search/page.js)
# splits on to render each part as its own visual block with media.
# Both Twitter sources MUST use this exact separator so a thread
# produced by either path renders identically.
_THREAD_SEPARATOR = "\n\n────────\n\n"


def compose_thread_doc(parts: list[dict], *, username: str) -> tuple[str, dict]:
    """Build one document from one or more tweets that share a
    ``conversationId``.

    Both Twitter source paths (``tweets.Tweets._collect_own_tweets`` on
    the twitterapi.io side, ``bookmarks.Bookmarks._merge_threads`` on
    the twikit side) go through this function so the on-disk shape is
    identical regardless of where the tweets came from. Same keys,
    same separator, same title format, same root-selection logic.

    Parameters
    ----------
    parts
        Raw tweet dicts in the twitterapi.io shape. Must be non-empty.
    username
        Author handle used to build the doc URL and the display title
        when the tweet's own ``user.screen_name`` is missing.

    Returns
    -------
    ``(url, doc)`` — the doc URL is the canonical "thread root" URL
    (the tweet whose ``id == conversationId`` when present, else the
    oldest tweet in the group). The doc dict has six keys:
    ``title, summary, date, tags, linked_urls, link_hosts``.
    """
    if not parts:
        raise ValueError("compose_thread_doc: empty parts list")
    parts = sorted(parts, key=lambda t: t.get("createdAt") or "")
    # Prefer the actual thread root (id == conversationId). When we
    # don't have it (likes/bookmarks rarely include the first tweet
    # of a thread), fall back to the oldest part — that's the safe
    # canonical entry point.
    root = next(
        (t for t in parts if str(t.get("id") or "") == str(t.get("conversationId") or "")),
        parts[0],
    )
    n = len(parts)
    if n > 1:
        body = [f"[{i + 1}/{n}] {_tweet_self_sufficient_summary(t)}" for i, t in enumerate(parts)]
        summary = _THREAD_SEPARATOR.join(b for b in body if b)
        title = f"{_tweet_display_title(root, username)} — thread ({n} tweets)"
        tags = ["twitter", "twitter-thread"]
    else:
        summary = _tweet_self_sufficient_summary(parts[0])
        title = _tweet_display_title(parts[0], username)
        tags = ["twitter"]
    link_source: list[dict] = []
    for p in parts:
        link_source.extend(_link_source_for(p))
    linked_urls, link_hosts = _build_linked_urls(link_source)
    url = _tweet_url(root, username)
    doc = {
        "title": title,
        "summary": summary,
        "date": _parse_date(root),
        "tags": tags,
        "linked_urls": linked_urls,
        "link_hosts": link_hosts,
    }
    # Engagement = sum across thread parts. A 10-tweet thread where
    # every part got 100 likes shows as 1000 likes on the card; that's
    # the cumulative attention the thread captured, which is what feed
    # ranking should sort on.
    doc.update(_sum_engagement(parts))
    # Referenced author — taken from the thread root. Self-replies
    # (chain continuations) get filtered out so the column reflects
    # the thread's *external* reference, not the user's own
    # in-thread plumbing.
    ref = _referenced_author(root)
    if ref and username and ref == username.lower():
        ref = ""
    doc["referenced_author"] = ref
    return url, doc


def _build_linked_urls(
    tweets: list[dict],
    *,
    fetch_meta: bool = True,
) -> tuple[list[dict], list[str]]:
    """Compose the `(linked_urls, link_hosts)` pair for a tweet doc.

    For each external URL across the supplied tweets (a single tweet
    or every part of a thread), produce one dict shaped
    ``{url, host, title, summary, image}``. Title / summary / image
    come from a server-side OG fetch via :func:`_fetch_page_preview`
    — the same call site we used to spend on the now-retired
    companion-document path, so the per-ingest HTTP cost is
    unchanged; only the destination of the data moves into the
    parent doc.

    Dedupes by URL across the whole thread, capped at
    `MAX_LINKED_URLS_PER_DOC` entries. Result order is "first time
    the URL appears, oldest tweet first" so a thread that introduces
    a paper in its second tweet keeps that paper in the preview
    cluster even when a later tweet hits the cap.

    Set `fetch_meta=False` for cheap tests / dry runs — the call
    site is the entire OG-fetch budget, and the function is otherwise
    a few dict assignments.
    """
    seen: set[str] = set()
    out: list[dict] = []
    hosts: list[str] = []
    hosts_seen: set[str] = set()
    for tw in tweets:
        # `article_card` (stamped by `bookmarks._twikit_to_dict`)
        # carries the title + preview text X serves for a long-form
        # Article attached to this tweet. We match it to the t.co
        # expansion via `rest_id` and preload the entry so we don't
        # have to hit `_fetch_page_preview` (which is blocked by X's
        # consent wall anyway and would yield an empty card).
        article_card = tw.get("article_card") if isinstance(tw, dict) else None
        for raw in _extract_urls(tw):
            if raw in seen:
                continue
            seen.add(raw)
            host = _canonical_host(raw)
            entry: dict = {"url": raw, "host": host or ""}
            article_match = (
                article_card and host == "x.com" and article_card.get("rest_id") and article_card["rest_id"] in raw
            )
            if article_match and article_card:
                entry["title"] = article_card.get("title", "")
                entry["summary"] = article_card.get("summary", "")
                entry["image"] = ""
            elif fetch_meta:
                title, description, image = _fetch_page_preview(raw)
                entry["title"] = title or ""
                entry["summary"] = description or ""
                entry["image"] = image or ""
            else:
                entry["title"] = ""
                entry["summary"] = ""
                entry["image"] = ""
            out.append(entry)
            if host and host not in hosts_seen:
                hosts_seen.add(host)
                hosts.append(host)
            if len(out) >= MAX_LINKED_URLS_PER_DOC:
                return out, hosts
    return out, hosts


def _is_substantive_reply(tweet: dict) -> bool:
    """Heuristic: is this reply likely to lead to an interesting thread root?

    Cheap signals (no API call required):
    - Reply already contains an external URL.
    - Reply text is long (>= 80 chars) — likely substantive commentary.
    - Reply text contains resource-related vocabulary (paper, github, …).

    Short reactions ("yes", "lol", "this!") return ``False`` so we skip
    hydrating their conversation root.
    """
    text = tweet.get("text") or tweet.get("full_text") or ""
    if not text:
        return False
    if _extract_urls(tweet):
        return True
    if len(text) >= 80:
        return True
    return bool(_INTERESTING_TOKENS.search(text))


# ═══════════════════════════════════════════════════════════════════════════
# TwitterAPI.io response parsing
# ═══════════════════════════════════════════════════════════════════════════


def _items(data: dict) -> list[dict]:
    """Extract the tweet list from a TwitterAPI.io response.

    The API wraps results in several possible shapes depending on the
    endpoint — this handles all of them.
    """
    if isinstance(data.get("tweets"), list):
        return data["tweets"]
    inner = data.get("data")
    if isinstance(inner, dict) and isinstance(inner.get("tweets"), list):
        return inner["tweets"]
    if isinstance(inner, list):
        return inner
    return []


def _next_cursor(data: dict) -> str | None:
    """Return the pagination cursor for the next page, if any."""
    return data.get("next_cursor") or (data.get("data") or {}).get("next_cursor") or data.get("nextCursor") or None


def _is_reply(tweet: dict) -> bool:
    """Return ``True`` if the tweet is a reply to another tweet."""
    return bool(tweet.get("isReply") or tweet.get("in_reply_to_status_id") or tweet.get("inReplyToId"))


def _conversation_id(tweet: dict) -> str | None:
    """Return the thread-root tweet ID (provided directly by TwitterAPI.io)."""
    for key in ("conversationId", "conversation_id"):
        val = tweet.get(key)
        if val:
            return str(val)
    return None


def _author(tweet: dict) -> str:
    """Return the screen name of the tweet's author."""
    user = tweet.get("user") or tweet.get("author") or {}
    return user.get("screen_name") or user.get("userName") or user.get("username") or ""


def _engagement_int(tweet: dict, *keys: str) -> int | None:
    """Pull the first non-None int from the tweet under any of ``keys``.

    Twitter's API has shipped a handful of casings across the years:
    ``favorite_count`` (v1.1), ``favoriteCount`` (twitterapi.io camelCase),
    ``like_count`` / ``likeCount`` (v2). We accept all spellings so the
    same helper works for both fetch paths.

    Returns ``None`` (not ``0``) when the field is absent so we don't
    overwrite a real prior count with a zero from a payload that simply
    didn't include the metric.
    """
    for k in keys:
        v = tweet.get(k)
        if v is None:
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n >= 0:
            return n
    return None


# Engagement-field name candidates, ordered most→least likely. Twitter
# / TwitterAPI.io / twikit (which uses `_legacy`) collectively use all
# of these spellings; `_engagement_int` walks them in order and returns
# the first int it finds.
_LIKES_KEYS = ("likeCount", "favorite_count", "favoriteCount", "like_count", "favourite_count")
_RETWEETS_KEYS = ("retweetCount", "retweet_count")
_REPLIES_KEYS = ("replyCount", "reply_count")
_QUOTES_KEYS = ("quoteCount", "quote_count")
_VIEWS_KEYS = ("viewCount", "view_count", "views", "impression_count", "impressionCount")
_BOOKMARKS_KEYS = ("bookmarkCount", "bookmark_count")


def _tweet_engagement(tweet: dict) -> dict[str, int | None]:
    """Return the engagement dict for one tweet, ``None``-valued for
    metrics the payload didn't include.

    Retweets carry no engagement of their own on the wrapper — every
    like / view counts on the source tweet — so we recurse into
    ``retweeted_tweet`` when the wrapper has all-None counts. The
    counts on the *inner* tweet are then what the card should show
    (and what gets indexed for popularity ranking).
    """
    out = {
        "twitter_likes": _engagement_int(tweet, *_LIKES_KEYS),
        "twitter_retweets": _engagement_int(tweet, *_RETWEETS_KEYS),
        "twitter_replies": _engagement_int(tweet, *_REPLIES_KEYS),
        "twitter_quotes": _engagement_int(tweet, *_QUOTES_KEYS),
        "twitter_views": _engagement_int(tweet, *_VIEWS_KEYS),
        "twitter_bookmarks": _engagement_int(tweet, *_BOOKMARKS_KEYS),
    }
    if all(v is None for v in out.values()):
        rt = tweet.get("retweeted_tweet")
        if isinstance(rt, dict):
            return _tweet_engagement(rt)
    return out


def _sum_engagement(parts: list[dict]) -> dict[str, int | None]:
    """Aggregate engagement across a thread's tweets.

    Per-part metrics are added; if every part is missing a given metric
    we leave it ``None`` rather than reporting a misleading 0. A single
    measured part is enough — the rest are treated as 0 for the sum
    (so a thread with one viral tweet still surfaces).
    """
    keys = (
        "twitter_likes",
        "twitter_retweets",
        "twitter_replies",
        "twitter_quotes",
        "twitter_views",
        "twitter_bookmarks",
    )
    totals: dict[str, int | None] = dict.fromkeys(keys)
    for p in parts:
        eng = _tweet_engagement(p)
        for k in keys:
            v = eng.get(k)
            if v is None:
                continue
            totals[k] = (totals[k] or 0) + v
    return totals


def _referenced_author(tweet: dict) -> str:
    """Return the lower-cased @handle this tweet refers to, or ``""``.

    Used to populate ``documents.referenced_author`` — the raw signal
    for spotting accounts that show up often in our users' libraries
    (heavily retweeted, frequently quoted, regularly replied to) and
    are worth pulling into the personality roster.

    Precedence — first match wins, so a retweet of a quote counts as
    the retweet target:

      1. ``retweeted_tweet.user.screen_name``
      2. ``quoted_tweet.user.screen_name``
      3. ``in_reply_to_screen_name`` (carried over from the v1.1
         legacy payload by ``bookmarks._twikit_to_dict``;
         twitterapi.io ships it as ``inReplyToUsername``)

    Returns ``""`` when none apply. The DB layer uses ``NULL`` for
    "not yet inspected" and ``''`` for "inspected, no reference",
    so callers can pass the return value straight through.
    """
    rt = tweet.get("retweeted_tweet")
    if isinstance(rt, dict):
        handle = _author(rt)
        if handle:
            return handle.lower()
    q = tweet.get("quoted_tweet")
    if isinstance(q, dict):
        handle = _author(q)
        if handle:
            return handle.lower()
    reply = (tweet.get("in_reply_to_screen_name") or tweet.get("inReplyToUsername") or "").strip()
    if reply:
        return reply.lower()
    return ""


def _retweet_extra_tags(tweet: dict) -> list[str]:
    """Extra tags carrying the retweet attribution.

    The summary used to lead with ``Retweet @<handle>\\n\\n…`` —
    handy but it polluted the text body and ended up reading as a
    page title in the card chrome. We moved the attribution to a
    filterable chip: ``retweet @<handle>`` lands on every retweet
    doc's ``extra_tags``, the card chip strip shows it inline, and
    a click filters the result list to everything that retweet-of
    that author.

    Returns ``[]`` for non-retweet tweets (so callers can pass the
    return value through `set(...)`-style merges without guarding).
    """
    rt = tweet.get("retweeted_tweet")
    if not rt:
        return []
    handle = _author(rt)
    if not handle:
        return []
    return [f"retweet @{handle.lower()}"]


def _parse_date(tweet: dict) -> str:
    """Parse TwitterAPI.io's timestamp into ``YYYY-MM-DD``."""
    created = tweet.get("createdAt") or tweet.get("created_at") or ""
    if created:
        try:
            from datetime import datetime as _dt

            return _dt.strptime(created, "%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def _tweet_url(tweet: dict, fallback_author: str = "unknown") -> str:
    """Build the canonical ``https://x.com/<author>/status/<id>`` URL."""
    tid = tweet.get("id") or tweet.get("id_str") or ""
    author = _author(tweet) or fallback_author
    return f"https://x.com/{author}/status/{tid}" if tid else ""


def _author_display(tweet: dict) -> str:
    """Author's display name (e.g. ``Andrej Karpathy``), falling back to handle."""
    user = tweet.get("user") or tweet.get("author") or {}
    return user.get("name") or user.get("displayName") or _author(tweet) or ""


def _tweet_display_title(tweet: dict, fallback_handle: str = "") -> str:
    """Human-friendly card title: ``Name (@handle)``.

    Falls back to ``@handle`` if no display name is available, matching the
    previous behaviour for tweets whose author block didn't carry a real name.
    """
    handle = _author(tweet) or fallback_handle
    display = _author_display(tweet)
    if display and handle and display.lower() != handle.lower():
        return f"{display} (@{handle})"
    return f"@{handle}" if handle else "Tweet"


def _tweet_media_urls(tweet: dict) -> list[tuple[str, str]]:
    """Return ``[(kind, url)]`` for every image / video / gif in a tweet.

    Reads ``extendedEntities.media`` (TwitterAPI.io's variant of the public
    Twitter API shape). Each `media` entry has a `type` (photo / video /
    animated_gif) plus a ``media_url_https`` for photos and a list of
    ``video_info.variants`` for videos — we pick the smallest progressive
    MP4 to keep the inline preview light.
    """
    out: list[tuple[str, str]] = []
    media = (tweet.get("extendedEntities") or {}).get("media") or []
    for m in media:
        kind = m.get("type") or ""
        if kind == "photo":
            u = m.get("media_url_https") or m.get("media_url") or ""
            if u:
                out.append(("photo", u))
        elif kind in ("video", "animated_gif"):
            variants = (m.get("video_info") or {}).get("variants") or []
            # Filter to playable MP4s; pick the lowest bitrate so the
            # card preview stays light.
            mp4s = [v for v in variants if v.get("content_type") == "video/mp4"]
            mp4s.sort(key=lambda v: int(v.get("bitrate") or 0))
            mp4_url = (mp4s[0].get("url") if mp4s else "") or ""
            poster = m.get("media_url_https") or ""
            # Encode as "<poster> | <mp4>" so the frontend can render
            # the poster inline (which loads fine — pbs.twimg.com
            # doesn't care about Referer) and link the mp4 out to
            # x.com on click. Twitter's video CDN rejects requests
            # whose Referer isn't x.com, so we can't embed the .mp4
            # directly. If only one of the two is available, we keep
            # whatever we have.
            if poster and mp4_url:
                out.append((kind, f"{poster} | {mp4_url}"))
            elif mp4_url or poster:
                out.append((kind, mp4_url or poster))
    return out


def _tweet_self_sufficient_summary(tweet: dict, depth: int = 0) -> str:
    """Compose a text summary that reads on its own — full text + media
    markers + retweet/quoted content inline. Zero extra API calls;
    all data is in the payload.

    Recursion is asymmetric so the common shapes unspool fully
    without letting pathological chains run away:

      * **Retweet wrapper** (retweeted_tweet) — recurses only at
        depth 0. Twitter doesn't allow retweet-of-retweet, so one
        hop is always enough.
      * **Quoted tweet** (quoted_tweet) — recurses at depth ≤ 1,
        i.e. one level inside a retweet. That covers the
        "Amelie retweets CShorten's quote of antoine" case, where
        the inner quote (and its image) would otherwise be
        dropped.

    Marker convention (matching what `web/search/page.js` parses):
       `📷 <url>`              — photo, rendered as an inline tile.
       `🎬 <poster> | <mp4>`   — video, poster tile + tweet click-out.

    External URLs the tweet points at are *not* serialised into the
    summary text. They flow through the document's `linked_urls` /
    `link_hosts` columns instead, and the card renderer reads them
    directly from the result row.

    Retweets: the wrapper is just a "RT @user: …" pointer with the
    truncated source text and no media of its own. We recurse into
    `retweeted_tweet` so the card surfaces the original's full
    text, photos, and videos — what the user would see on
    twitter.com — prefixed with a small "Retweet @handle"
    attribution.
    """
    # Retweet short-circuit. Everything worth showing lives on the
    # inner tweet, so recurse there and surface the inner body
    # verbatim — no "Retweet @handle" prefix. The retweeted handle
    # is communicated via the doc's `extra_tags`
    # (`retweet @<handle>`) at emit time, which the card chips
    # render as a filterable pill; the summary text stays focused
    # on the content the user actually wants to read.
    rt = tweet.get("retweeted_tweet")
    if rt and depth < 1:
        return _tweet_self_sufficient_summary(rt, depth=depth + 1)

    parts: list[str] = []
    text = tweet.get("text") or tweet.get("full_text") or ""
    # The "article" field is TwitterAPI.io's note-tweet long-form. If the
    # API ships a longer text body there, prefer it over the truncated
    # `text`.
    article = tweet.get("article") or {}
    if isinstance(article, dict):
        art_text = article.get("text") or article.get("body") or ""
        if art_text and len(art_text) > len(text):
            text = art_text
    if text:
        parts.append(text.strip())
    # Media markers — kept as plain text so the existing summary
    # rendering shows them without a schema change. Format: a short
    # tag plus the URL so the renderer (or the user) can preview.
    for kind, url in _tweet_media_urls(tweet):
        tag = "📷" if kind == "photo" else "🎬"
        parts.append(f"{tag} {url}")
    # Quoted tweet — recurse the same way retweets do so the doc
    # surfaces the quoted body, media markers (📷 / 🎬), and any
    # nested attribution. The guard allows one extra hop relative
    # to retweets so a "retweet of a quote of X" still surfaces X.
    quoted = tweet.get("quoted_tweet")
    if quoted and depth < 2:
        q_handle = _author(quoted)
        q_summary = _tweet_self_sufficient_summary(quoted, depth=depth + 1)
        if q_summary:
            attribution = f"Quoting @{q_handle}" if q_handle else "Quoting"
            parts.append(f"{attribution}\n\n{q_summary}")
    return "\n\n".join(p for p in parts if p)


def _link_source_for(tweet: dict, depth: int = 0) -> list[dict]:
    """List of tweets whose URLs should feed into the doc's ``linked_urls``.

    Mirrors :func:`_tweet_self_sufficient_summary`'s recursion so the
    text + previews stay in lockstep:

      * **Retweet** — only the inner tweet (the wrapper is a stub).
        Recurses so "RT of a quote" still picks up the original's
        quoted-tweet URLs.
      * **Quote**  — wrapper plus the quoted tweet (both can carry
        their own URLs that the reader expects to see previewed).
        Recurses one level deeper than retweets to match
        :func:`_tweet_self_sufficient_summary`.
      * **Plain**  — just the wrapper.
    """
    rt = tweet.get("retweeted_tweet")
    if rt:
        return _link_source_for(rt, depth=depth + 1)
    out = [tweet]
    q = tweet.get("quoted_tweet")
    if q and depth < 2:
        out.extend(_link_source_for(q, depth=depth + 1))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# On-disk caches (shared across personalities — tweets are immutable)
# ═══════════════════════════════════════════════════════════════════════════


def _cache_path(data_dir: str) -> Path:
    """``data/.twitterapi_cache.json`` — one level above personality dir."""
    return Path(data_dir).parent / ".twitterapi_cache.json"


def _cache_load(data_dir: str) -> dict[str, dict]:
    """Load the tweet-object cache from disk."""
    p = _cache_path(data_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _last_seen_path(data_dir: str) -> Path:
    """``data/.twitterapi_last_seen.json`` — per-username incremental cursors."""
    return Path(data_dir).parent / ".twitterapi_last_seen.json"


def _last_seen_load(data_dir: str) -> dict[str, str]:
    """Load ``{username: newest_tweet_id}`` from disk.

    Backward-compatible: migrates ``{username: {id: ..., date: ...}}``
    from the previous format to plain ``{username: tweet_id}``.
    """
    p = _last_seen_path(data_dir)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = v.get("id", "")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Twitter class
# ═══════════════════════════════════════════════════════════════════════════


class Tweets:
    """Fetch tweets and thread-root resources for a single X/Twitter user.

    Parameters
    ----------
    username : str
        Target X screen name (e.g. ``"karpathy"``).
    api_key : str
        TwitterAPI.io API key (required).
    data_dir : str
        Personality data directory (e.g. ``"data/karpathy"``).
        Caches are stored one level up (``data/``).
    include_tweets : bool
        Store the user's own tweets as documents.
    include_replies : bool
        Hydrate conversation roots from the user's replies and extract
        external URLs as resource documents.
    filter_replies : bool
        Drop short/reaction replies before hydrating their roots.
    max_parents : int
        Hard cap on conversation roots to hydrate per run.
    min_interval : float
        Minimum seconds between API calls (rate-limit pacing).
    user_id : str or None
        Optional TwitterAPI.io ``X-User-Id`` header for billing.

    Examples
    --------
    >>> tw = Twitter(username="karpathy", api_key="sk-...")
    >>> documents = tw(max_pages=10, existing_urls=set())
    >>> # Filter noise
    >>> documents = filter_tweets(documents)
    """

    def __init__(
        self,
        username: str,
        api_key: str,
        data_dir: str = "data",
        include_tweets: bool = True,
        include_replies: bool = True,
        filter_replies: bool = True,
        max_parents: int = 5000,
        min_interval: float = 0.34,
        user_id: str | None = None,
    ):
        self.username = username
        self.api_key = api_key
        self.data_dir = data_dir
        self.include_tweets = include_tweets
        self.include_replies = include_replies
        self.filter_replies = filter_replies
        self.max_parents = max_parents

        self._session = requests.Session()
        self._session.headers["X-API-Key"] = self.api_key
        if user_id:
            self._session.headers["X-User-Id"] = user_id

        self._min_interval = min_interval
        self._last_call_at = 0.0

    # ── Public API ─────────────────────────────────────────────────────

    def __call__(
        self,
        max_pages: int = 5,
        existing_urls: set[str] | None = None,
        stop_date: str = "",
        max_tweets: int | None = None,
        budget: object | None = None,
    ) -> dict[str, dict]:
        """Fetch documents for the target user.

        Parameters
        ----------
        max_pages : int
            Maximum timeline pages to fetch (~20 tweets each).
        existing_urls : set[str] or None
            URLs already in the database. Enables early-exit when a full
            page contains nothing new, and skips known URLs in output.
        stop_date : str
            ``YYYY-MM-DD`` date fence. Pagination stops when a tweet is
            older than this date. Prevents re-fetching tweets that were
            previously parsed but filtered out by ``filter_tweets()``.
        max_tweets : int or None
            Hard cap on total tweets (own + replies) collected during
            pagination. ``None`` disables the cap.

        Returns
        -------
        dict[str, dict]
            ``{url: {title, summary, date, tags, …}}`` ready for
            ``_merge_and_track()`` in the pipeline.
        """
        return self._fetch_all(
            max_pages=max_pages,
            budget=budget,
            existing_urls=existing_urls,
            stop_date=stop_date,
            max_tweets=max_tweets,
        )

    # ── HTTP layer ─────────────────────────────────────────────────────

    def _pace(self) -> None:
        """Sleep if needed to respect the per-second rate limit."""
        elapsed = time.time() - self._last_call_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_at = time.time()

    def _get(self, path: str, *, retries: int = 2, **params) -> dict:
        """Issue a GET request with pacing and automatic 429/network retry."""
        url = f"{_API_BASE}{path}"
        for attempt in range(retries + 1):
            self._pace()
            try:
                r = self._session.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException as exc:
                if attempt < retries:
                    time.sleep(self._min_interval * (attempt + 2))
                    continue
                raise RuntimeError(f"network error on {url}: {exc}") from exc
            if r.status_code == 429 and attempt < retries:
                time.sleep(self._min_interval * (attempt + 2))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {url} → {r.text[:300]}")
            try:
                return r.json()
            except ValueError as exc:
                raise RuntimeError(f"non-JSON from {url}: {exc}") from exc
        raise RuntimeError(f"429 after {retries} retries: {url}")

    def _fetch_timeline_page(self, cursor: str | None = None) -> dict:
        """Fetch one page of the user's timeline (tweets + replies)."""
        params = {"userName": self.username, "includeReplies": "true"}
        if cursor:
            params["cursor"] = cursor
        return self._get("/twitter/user/last_tweets", **params)

    def _hydrate_tweets(self, tweet_ids: list[str], cache: dict[str, dict]) -> list[dict]:
        """Batch-fetch tweet objects by ID, serving cache hits for free.

        IDs present in *cache* are returned directly. Misses are fetched
        in chunks of 100 via ``/twitter/tweets`` and added to *cache*
        in-place so subsequent runs benefit.
        """
        if not tweet_ids:
            return []

        out: list[dict] = []
        misses: list[str] = []
        for tid in tweet_ids:
            hit = cache.get(tid)
            if hit:
                out.append(hit)
            else:
                misses.append(tid)

        # Per-chunk failure isolation. `_get` raises after retries on
        # network/timeout errors. Without this guard a single bad chunk
        # would tank the whole twitter task and we'd lose every tweet
        # already collected in Phase 1. Skip the failing chunk, keep
        # going — partial hydration is much better than zero.
        n_failed = 0
        for i in range(0, len(misses), 100):
            chunk = misses[i : i + 100]
            try:
                data = self._get("/twitter/tweets", tweet_ids=",".join(chunk))
            except Exception as exc:
                n_failed += 1
                print(f"    Hydrate chunk {i // 100 + 1} failed ({len(chunk)} ids): {exc}")
                continue
            for tw in _items(data):
                tid = str(tw.get("id") or tw.get("id_str") or "")
                if tid:
                    out.append(tw)
                    cache[tid] = tw
        if n_failed:
            print(f"    Hydrate: {n_failed} chunk(s) skipped after retries; partial result returned.")
        return out

    # ── Core pipeline ──────────────────────────────────────────────────

    def _fetch_all(
        self,
        max_pages: int,
        existing_urls: set[str] | None = None,
        stop_date: str = "",
        max_tweets: int | None = None,
        budget: object | None = None,
    ) -> dict[str, dict]:
        """Orchestrate timeline fetch → reply hydration → document assembly.

        ``budget`` is an optional `sources.credits.Budget` (or any
        object with an ``allow(meta) -> bool`` method). When set, each
        outbound /twitter API request consults it: pagination stops
        the first time `allow()` returns False, so a user who runs
        out of credits stops paying mid-run instead of mid-flight.
        VIPs pass a free budget (always-True, no debits). When
        ``budget`` is None the fetcher behaves as before (free).
        """
        cache = _cache_load(self.data_dir)
        last_seen = _last_seen_load(self.data_dir)

        stop_id = last_seen.get(self.username)
        if stop_id:
            print(f"    Incremental: will stop at tweet {stop_id}")
        if stop_date:
            print(f"    Date fence: will stop at tweets older than {stop_date}")
        if max_tweets is not None:
            print(f"    Max tweets: {max_tweets}")

        # ── Phase 1: paginate the timeline ─────────────────────────────
        own_tweets, replies = self._paginate_timeline(
            max_pages=max_pages,
            existing_urls=existing_urls,
            stop_id=stop_id,
            stop_date=stop_date,
            cache=cache,
            max_tweets=max_tweets,
            budget=budget,
        )

        # ── Phase 1b: detach self-thread continuations from `replies` ──
        # When the user is replying to themselves we treat the chain
        # as part of their own timeline (a thread), not a reply to
        # somebody else. The tweets are already in memory — zero
        # extra API calls. _collect_own_tweets will group them by
        # conversationId into single-document threads.
        if replies:
            self_thread_parts: list[dict] = []
            other_replies: list[dict] = []
            uname = (self.username or "").lower()
            for tw in replies:
                target = (tw.get("inReplyToUsername") or "").lower()
                if target and target == uname:
                    self_thread_parts.append(tw)
                else:
                    other_replies.append(tw)
            if self_thread_parts:
                print(
                    f"    Detached {len(self_thread_parts)} self-thread replies "
                    f"from {len(replies)} replies; will merge into threads."
                )
                own_tweets.extend(self_thread_parts)
                replies = other_replies

        # ── Phase 2: filter junk replies ───────────────────────────────
        if self.filter_replies and replies:
            before = len(replies)
            replies = [tw for tw in replies if _is_substantive_reply(tw)]
            skipped = before - len(replies)
            if skipped:
                print(f"    Reply filter: kept {len(replies)}, dropped {skipped}")

        # ── Phase 3: assemble documents ────────────────────────────────
        documents: dict[str, dict] = {}

        if self.include_tweets:
            self._collect_own_tweets(own_tweets, documents, existing_urls)

        if self.include_replies:
            self._collect_reply_resources(replies, documents, existing_urls, cache, budget=budget)

        print(f"    Twitter done: {len(documents)} documents")
        return documents

    # ── Timeline pagination ────────────────────────────────────────────

    def _paginate_timeline(
        self,
        max_pages: int,
        existing_urls: set[str] | None,
        stop_id: str | None,
        stop_date: str,
        cache: dict[str, dict],
        max_tweets: int | None = None,
        budget: object | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Walk the timeline newest-first, splitting tweets vs replies.

        Stops early when:
        - We hit *stop_id* (incremental cursor from last run).
        - A tweet is older than *stop_date* (date fence from last run).
        - We've collected at least *max_tweets* tweets (hard cap).
        - A full page has zero new tweets (all in *existing_urls*).
        - No more pages (cursor exhausted).

        The date fence is the primary guard against re-fetching tweets
        that were previously fetched but then filtered out by
        ``filter_tweets()`` — those URLs are absent from *existing_urls*
        but the date fence still stops pagination.
        """
        own_tweets: list[dict] = []
        replies: list[dict] = []
        cursor: str | None = None

        from sources.credits import twitter_page_cost, twitter_worst_case_cost

        for page in range(max_pages):
            # Pre-flight: refuse to fetch when the balance can't cover
            # the worst case (a full 100-tweet page). Avoids paying
            # twitterapi.io for a response we then can't bill the user
            # for. VIP budgets short-circuit precheck() to True.
            if budget is not None and not budget.precheck(twitter_worst_case_cost()):
                print(f"    page {page + 1}: insufficient credits — stopping pagination")
                break
            try:
                data = self._fetch_timeline_page(cursor)
            except RuntimeError as exc:
                print(f"    page {page + 1}: {exc}")
                break

            items = _items(data)
            if not items:
                break

            # Bill the ACTUAL cost based on tweets returned (with margin).
            if budget is not None:
                cost = twitter_page_cost(len(items))
                if not budget.charge(
                    cost,
                    {"page": page + 1, "username": self.username, "tweets": len(items)},
                ):
                    print(f"    page {page + 1}: debit failed ({cost} credits) — stopping pagination")
                    break

            new_in_page = 0
            hit_stop = False
            hit_date = False
            hit_cap = False

            for tw in items:
                tid = str(tw.get("id") or tw.get("id_str") or "")

                # Incremental stop: we've caught up to where we left off.
                if stop_id and tid == stop_id:
                    hit_stop = True
                    break

                # Date fence: tweet is older than our last full parse.
                if stop_date:
                    tweet_date = _parse_date(tw)
                    if tweet_date and tweet_date < stop_date:
                        hit_date = True
                        break

                url = _tweet_url(tw, self.username)
                is_known = existing_urls is not None and url in existing_urls

                if _is_reply(tw):
                    replies.append(tw)
                else:
                    own_tweets.append(tw)

                if not is_known:
                    new_in_page += 1

                # Pre-populate cache so root hydration can skip these.
                if tid:
                    cache[tid] = tw

                # Hard cap — stop as soon as the target count is reached.
                if max_tweets is not None and len(own_tweets) + len(replies) >= max_tweets:
                    hit_cap = True
                    break

            cursor = _next_cursor(data)
            all_seen = existing_urls is not None and new_in_page == 0 and not hit_stop and not hit_date

            suffix = ""
            if hit_stop:
                suffix = "  STOP (incremental)"
            elif hit_date:
                suffix = f"  STOP (date fence: {stop_date})"
            elif hit_cap:
                suffix = f"  STOP (max_tweets: {max_tweets})"
            elif all_seen:
                suffix = "  STOP (all seen)"

            print(
                f"    page {page + 1}: +{len(items)} tweets "
                f"(own={len(own_tweets)} replies={len(replies)} new={new_in_page})"
                f"{suffix}"
            )

            if hit_stop or hit_date or hit_cap or all_seen or not cursor:
                break

        print(f"    Fetched: {len(own_tweets)} own tweets, {len(replies)} replies")
        return own_tweets, replies

    # ── Document assembly ──────────────────────────────────────────────

    def _collect_own_tweets(
        self,
        tweets: list[dict],
        documents: dict[str, dict],
        existing_urls: set[str] | None,
    ) -> None:
        """Store each own tweet as a document + extract inline resource URLs.

        Tweets that share a ``conversationId`` (i.e. parts of one of the
        author's threads) are merged into a single document — keyed on
        the root tweet's URL — so the search page shows the whole
        thread as one card instead of N stitched fragments. Costs zero
        extra API calls; every part was already in the paginated
        timeline response.
        """
        # Group by conversationId. A single tweet gets its own bucket.
        from collections import defaultdict

        # Retweets share the source tweet's conversationId, so naive
        # grouping would lump N retweets-of-the-same-source into a
        # phantom "thread of 105 tweets" all reading identically. We
        # split: retweets are routed to a flat per-source-tweet bucket
        # (dedup by retweeted_tweet.id), originals + self-thread parts
        # go through the conversation-grouping logic.
        def _is_rt(tw: dict) -> bool:
            if tw.get("retweeted_tweet"):
                return True
            txt = tw.get("text") or tw.get("full_text") or ""
            return txt.startswith("RT @")

        retweets: dict[str, dict] = {}
        by_conv: dict[str, list[dict]] = defaultdict(list)
        # Dedupe BY TWEET ID first — the timeline can hand the same
        # tweet across pages and we've also seen the API echo a single
        # tweet many times under the same conversation. Either way,
        # we never want one tweet to multiply into N thread parts.
        seen_ids: set[str] = set()
        # And dedupe BY (conversationId, normalised text) — different
        # tweet ids occasionally carry identical text (e.g. when the
        # API returns the same reply under multiple list contexts),
        # which would produce a thread of 103 carbon-copy parts.
        seen_content: set[tuple[str, str]] = set()
        for tw in tweets:
            tid = str(tw.get("id") or "")
            if tid and tid in seen_ids:
                continue
            if tid:
                seen_ids.add(tid)
            if _is_rt(tw):
                rt = tw.get("retweeted_tweet") or {}
                src_id = str(rt.get("id") or "")
                if src_id and src_id not in retweets:
                    retweets[src_id] = tw
                continue
            cid = str(tw.get("conversationId") or tw.get("id") or "")
            if not cid:
                continue
            text_key = (tw.get("text") or tw.get("full_text") or "").strip()
            sig = (cid, text_key)
            if text_key and sig in seen_content:
                continue
            if text_key:
                seen_content.add(sig)
            by_conv[cid].append(tw)

        # Emit each unique retweet as its own (deduped) document.
        # The wrapper carries no media / linked URLs of its own, so we
        # pull `linked_urls` from the *original* tweet — matching what
        # `_tweet_self_sufficient_summary` already does for the text /
        # media side. Without this the retweet would render with no
        # link-preview cluster even when the source tweet pointed at a
        # paper / blog post.
        for tw in retweets.values():
            url = _tweet_url(tw, self.username)
            if not url:
                continue
            if existing_urls is not None and url in existing_urls:
                continue
            linked_urls, link_hosts = _build_linked_urls(_link_source_for(tw))
            documents[url] = {
                "title": _tweet_display_title(tw, self.username),
                "summary": _tweet_self_sufficient_summary(tw),
                "date": _parse_date(tw),
                "tags": ["twitter", "retweet"],
                # `extra_tags` carries the retweet's original-author
                # attribution as a clickable chip — e.g.
                # `retweet @cshorten30` — replacing the legacy
                # "Retweet @x" prefix on the summary body.
                "extra-tags": _retweet_extra_tags(tw),
                "linked_urls": linked_urls,
                "link_hosts": link_hosts,
                # The retweet wrapper carries no engagement of its own;
                # `_tweet_engagement` already recurses into
                # `retweeted_tweet` so this lands the SOURCE tweet's
                # like/retweet/view counts on the retweet doc — which is
                # what the feed ranking should care about.
                **_tweet_engagement(tw),
            }

        for _cid, group in by_conv.items():
            # All thread-doc assembly (root selection, numbered summary,
            # link-source union) lives in `compose_thread_doc` so the
            # twikit bookmarks path can produce byte-identical output
            # by calling the same function.
            url, doc = compose_thread_doc(group, username=self.username)
            if not url:
                continue
            if existing_urls is not None and url in existing_urls:
                continue
            documents[url] = doc

    def _collect_reply_resources(
        self,
        replies: list[dict],
        documents: dict[str, dict],
        existing_urls: set[str] | None,
        cache: dict[str, dict],
        budget: object | None = None,
    ) -> None:
        """Hydrate conversation roots of replies and extract resource URLs.

        Each `_hydrate_tweets` call hits `/twitter/tweets` and is a
        paid API call. We gate it on the budget so a user with low
        credits doesn't pay for reply-hydration after their main
        timeline pagination already stopped early.
        """
        # ── Pass 1: harvest URLs from the REPLY body itself ───────────
        # Symmetric with the own-tweet path: when the user replies with
        # a link in their own text (e.g. "great paper https://arxiv.org/...")
        # the URL lives on the reply, not on the parent we're about to
        # hydrate. Free pickup — these tweet objects are already in
        # memory from `_paginate_timeline`, no extra API call.
        for tw in replies:
            for link in _extract_urls(tw):
                if link in documents:
                    continue
                if existing_urls is not None and link in existing_urls:
                    continue
                self._record_resource(documents, link, tw)

        # ── Pass 2: hydrate parent thread-roots (existing behaviour) ──
        # Deduplicate by conversation ID.
        convo_ids: list[str] = []
        seen_convos: set[str] = set()
        cid_to_reply: dict[str, dict] = {}

        for tw in replies:
            cid = _conversation_id(tw)
            if cid and cid not in seen_convos and cid != str(tw.get("id") or ""):
                seen_convos.add(cid)
                convo_ids.append(cid)
                cid_to_reply.setdefault(cid, tw)

        convo_ids = convo_ids[: self.max_parents]
        if not convo_ids:
            return

        misses = [cid for cid in convo_ids if cid not in cache]
        hits = len(convo_ids) - len(misses)
        print(f"    Hydrating {len(convo_ids)} thread roots: {hits} cached, {len(misses)} new...")

        # Credit gate: only the `misses` count costs money (cached
        # tweets are returned without an API call). Each chunk of 100
        # IDs is one paid API call billed at twitter_page_cost(N) for
        # the N tweets actually returned in that chunk.
        from sources.credits import twitter_page_cost, twitter_worst_case_cost

        roots: list[dict] = []
        # Serve cache hits for free up-front.
        cached_roots = [cache[cid] for cid in convo_ids if cid in cache]
        roots.extend(cached_roots)

        if misses and budget is not None and not budget.precheck(twitter_worst_case_cost()):
            print("    hydrate: insufficient credits — skipping thread-root fetch")
            return

        for i in range(0, len(misses), 100):
            chunk = misses[i : i + 100]
            if budget is not None and not budget.precheck(twitter_worst_case_cost()):
                print("    hydrate: insufficient credits — stopping mid-hydration")
                break
            try:
                data = self._get("/twitter/tweets", tweet_ids=",".join(chunk))
            except Exception as exc:
                print(f"    Hydrate chunk {i // 100 + 1} failed ({len(chunk)} ids): {exc}")
                continue
            chunk_roots: list[dict] = []
            for tw in _items(data):
                tid = str(tw.get("id") or tw.get("id_str") or "")
                if tid:
                    chunk_roots.append(tw)
                    cache[tid] = tw
            if budget is not None:
                cost = twitter_page_cost(len(chunk_roots))
                if not budget.charge(
                    cost,
                    {"endpoint": "/twitter/tweets", "chunk": i // 100 + 1, "tweets": len(chunk_roots)},
                ):
                    print(f"    hydrate: debit failed ({cost} credits) — stopping")
                    break
            roots.extend(chunk_roots)

        for root in roots:
            rid = str(root.get("id") or root.get("id_str") or "")
            reply = cid_to_reply.get(rid)

            via = _tweet_url(reply, self.username) if reply else None

            for link in _extract_urls(root):
                if link not in documents and (existing_urls is None or link not in existing_urls):
                    self._record_resource(documents, link, root, surfaced_via=via)

    def _record_resource(
        self,
        documents: dict[str, dict],
        url: str,
        root_tweet: dict,
        *,
        surfaced_via: str | None = None,
    ) -> None:
        """Create a resource document from an external URL in a tweet.

        The document describes the **linked page** (fetched title + meta
        description), NOT the tweet. The tweet URL is kept only in
        ``source_url`` so the UI can show an attribution ribbon.

        ``source`` is derived from the URL's domain (github / youtube /
        scholar / …) or defaults to ``"blog"`` — never ``"twitter"``,
        because this doc isn't a tweet.
        """
        # `_source_tag` returns the brand label for known platforms or
        # the bare hostname otherwise — never falls back to a generic
        # bucket. The `or ""` here is just a None-guard.
        src = _source_tag(url) or ""
        root_url = _tweet_url(root_tweet)

        page_title, page_desc = _fetch_page_meta(url)
        title = page_title or _title_from_url_slug(url)
        summary = page_desc or ""

        doc: dict = {
            "title": title,
            "summary": summary,
            "date": _parse_date(root_tweet),
            "tags": ["twitter-thread", src],
            "source": src,
            # Engagement is the *root tweet's* — i.e. how viral the
            # post that surfaced this resource was. A paper linked
            # from a 10k-like tweet ranks higher than one buried in a
            # 3-like reply, which is what feed customization wants.
            **_tweet_engagement(root_tweet),
        }
        if root_url:
            doc["source_url"] = root_url
        if surfaced_via and surfaced_via != root_url:
            doc["surfaced_via_reply"] = surfaced_via

        documents[url] = doc

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _newest_id(own_tweets: list[dict], replies: list[dict]) -> str | None:
        """Return the newest tweet ID across both lists (for the cursor).

        The timeline is newest-first, so the first tweet in either list
        is the most recent.
        """
        for tw in own_tweets[:1] + replies[:1]:
            tid = str(tw.get("id") or tw.get("id_str") or "")
            if tid:
                return tid
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Zero-shot tweet classifier
# ═══════════════════════════════════════════════════════════════════════════
#
# Each category has a set of prototype sentences whose centroid defines
# the class anchor in model2vec's embedding space. At inference we pick
# the argmax category for each tweet and keep only the informational ones.

_KEEP_LABELS = frozenset(
    {
        "research_paper",
        "technical_explanation",
        "tool_resource",
        "tutorial_education",
        "opinion_insight",
        "meta_commentary",
    }
)

_DROP_LABELS = frozenset(
    {
        "mood_social",
        "announcement",
    }
)

_CATEGORIES: dict[str, list[str]] = {
    # ── Informational (kept by default) ────────────────────────────────
    # Sharing or discussing published research, studies, or findings.
    "research_paper": [
        "Fascinating new paper on how coral reefs adapt to rising ocean temperatures — the data from the Pacific transects is compelling.",
        "This study in The Lancet shows that early intervention reduces long-term hospitalisation rates by 40%. Worth reading for anyone in public health.",
        "New preprint: the authors demonstrate that their algorithm outperforms existing baselines on all six benchmark datasets.",
        "Interesting economics paper arguing that minimum wage increases in mid-size cities had minimal negative employment effects when phased in gradually.",
        "Great longitudinal study tracking student outcomes across 15 years — the effect sizes on early reading programs are striking.",
        "Must-read paper on CRISPR delivery mechanisms. The lipid nanoparticle approach solves the tissue specificity problem that plagued earlier methods.",
        "This Nature paper maps the complete neural circuit of a fruit fly brain. 130,000 neurons, every synapse catalogued. A landmark dataset.",
        "New systematic review on sleep and cognitive performance — the meta-analysis across 34 studies makes a strong case for 7-8 hours.",
        "Just read this paper on carbon capture costs — the levelised cost analysis finally accounts for transportation and storage properly.",
        "Solid replication study: the original findings on priming effects did not hold when tested across eight independent labs.",
    ],
    # Detailed explanation of how something works — any domain.
    "technical_explanation": [
        "Here is how mRNA vaccines work: the lipid nanoparticle delivers synthetic messenger RNA into your cells, which then produce the target protein and trigger an immune response.",
        "The reason bridges use expansion joints is thermal expansion. Steel expands roughly 12mm per 10 metres for every 100 degree change, so without gaps the structure would buckle.",
        "A good mental model for how compilers work: source code is parsed into an abstract syntax tree, transformed through optimisation passes, then lowered to machine instructions.",
        "The way sourdough fermentation works is that wild yeast and lactobacillus bacteria form a symbiotic culture. The bacteria produce lactic acid which gives flavour, while yeast produces CO2 for rise.",
        "Why do planes fly? The key mechanism is not just Bernoulli's principle — it is the angle of attack. The wing deflects air downward, and by Newton's third law, the air pushes the wing up.",
        "How double-entry bookkeeping works: every transaction is recorded in two accounts — a debit and a credit — so the books always balance. This is the foundation of all modern accounting.",
        "The reason salt lowers the freezing point of water is a colligative property: dissolved ions disrupt the crystal lattice formation that ice requires, so you need colder temperatures to freeze.",
        "How ranked-choice voting works: voters rank candidates in order of preference. If no one wins a majority, the last-place candidate is eliminated and their votes redistribute to each voter's next choice.",
        "Thread: here is a step-by-step breakdown of how containerisation works in shipping logistics and why standardised container sizes revolutionised global trade.",
        "The way noise-cancelling headphones work: a microphone picks up ambient sound, the chip inverts the waveform, and a speaker plays the inverted signal — destructive interference cancels the noise.",
    ],
    # Sharing tools, code, libraries, products, datasets, useful links.
    "tool_resource": [
        "Just released an open-source tool that automates data migration between SQL databases. Clean API, well tested, MIT licence.",
        "This is an excellent free resource for learning to draw — structured lessons from basic shapes to full figure drawing, all Creative Commons.",
        "New dataset released: 50 million annotated satellite images covering every continent, free for research use.",
        "Built a small command-line utility that converts between dozens of file formats. Single binary, no dependencies, works offline.",
        "Highly recommend this library — it handles PDF parsing, table extraction, and OCR in one clean interface. Saved me hours of work.",
        "Open-sourcing our internal dashboard template. It handles authentication, role-based access, and real-time data streaming out of the box.",
        "This browser extension highlights manipulative design patterns on e-commerce sites. Really eye-opening to see how many are on a typical checkout page.",
        "New free API for historical weather data going back to 1940. High resolution, global coverage, well-documented endpoints.",
        "Shipped a tool that analyses your codebase and generates architecture diagrams automatically. Supports Python, Go, Java, and TypeScript.",
        "This spreadsheet template for personal finances is genuinely excellent. Tracks spending, investments, and tax obligations in one place.",
    ],
    # Teaching content, how-tos, walkthroughs, courses, lectures.
    "tutorial_education": [
        "New video tutorial: I walk through the entire process of setting up a home server from scratch, including networking, storage, and backups.",
        "Wrote a beginner's guide to understanding financial statements. Covers balance sheets, income statements, and cash flow — with worked examples from real companies.",
        "Step-by-step thread on how to read a scientific paper effectively: abstract first, then figures, then methods. Skip the introduction until you need context.",
        "Recorded a lecture explaining the fundamentals of music theory — scales, chords, progressions — using only a piano and simple diagrams.",
        "Posted a detailed cooking tutorial: how to make fresh pasta from scratch, including the science behind gluten development and why resting the dough matters.",
        "Here is a guide I wrote on how to set up automated testing for your project. Covers unit tests, integration tests, and continuous integration pipelines.",
        "Educational thread on how to read an electrocardiogram. I explain each wave, what it represents electrically, and the common abnormalities to look for.",
        "My latest workshop recording: designing experiments with proper controls, sample sizes, and statistical analysis — aimed at early-career researchers.",
        "A clear explanation of how to read legal contracts. I break down the most common clauses, what they actually mean, and the red flags to watch for.",
        "This interactive course teaches data visualisation principles from Tufte to modern dashboards. Completely free, project-based, with real datasets.",
    ],
    # Substantive opinions, analyses, predictions, commentary on a topic.
    "opinion_insight": [
        "I think the biggest underrated problem in healthcare right now is not diagnosis but follow-up. Patients fall through the cracks between appointments.",
        "Hot take: remote work is not going away. The companies forcing return-to-office will lose their best talent to competitors who offer flexibility.",
        "The most important trend in urban planning is the shift from car-centric design to mixed-use walkable neighbourhoods. The data on quality of life is overwhelming.",
        "Prediction: within five years, most routine legal document review will be automated. The economics are too compelling for large firms to ignore.",
        "Unpopular opinion: code reviews are more valuable for knowledge sharing than for catching bugs. The real benefit is that two people understand every change.",
        "The reason this policy failed is not that the idea was wrong, but that implementation ignored local context. Top-down mandates without community input rarely work.",
        "I strongly believe that teaching critical thinking should start in primary school. We teach children what to think, not how to evaluate evidence.",
        "Controversial view: most productivity advice is procrastination in disguise. The actual bottleneck for most people is fear of starting, not lack of systems.",
        "The biggest lesson from the supply chain crisis is that efficiency and resilience are in tension. Just-in-time works until it doesn't.",
        "After spending a decade in this field, I'm convinced that the fundamental bottleneck is not technology but institutional incentives.",
    ],
    # ── Non-informational (dropped by default) ─────────────────────────
    # Personal life updates, job changes, milestone celebrations — no
    # transferable knowledge, just "this happened to me" news.
    "announcement": [
        "Some personal news: after five great years, I am leaving my current role to start something new.",
        "Excited to announce that I am joining a new team. Can't wait to get started!",
        "I got accepted into the program! Moving to a new city next month. Nervous but excited.",
        "We just closed our funding round. Grateful for all the support.",
        "Wrapping up my time here this month. Grateful for everything I learned. On to the next chapter.",
        "Big news: we are hiring for several new positions on the team. Details in the thread.",
        "Just signed the lease on our first office space. It's actually happening.",
        "I got married last weekend. Best day of my life. Back to regular posting soon.",
        "Very grateful to receive this award. Thank you to the committee and everyone who nominated me.",
        "First day at the new job. The team is great, the mission is exciting. Let's go.",
        "Officially done with my degree! Four years of hard work. Time to celebrate.",
        "Today is my last day. Bittersweet. Thank you to everyone who made this journey special.",
    ],
    # Commentary about an industry, community, platform, or discourse.
    "meta_commentary": [
        "This platform continues to be the fastest channel for news in this space, even as the signal-to-noise ratio keeps degrading.",
        "The discourse online has entered a phase where everyone claims expertise but few show their work. Makes it hard to know who to trust.",
        "Observation: the conference review process scales poorly with submission volume, and the incentives are misaligned with quality.",
        "A lot of the current debate on this topic is performative. People are optimising for engagement, not understanding.",
        "The community has a culture of overpromising in public and underdelivering in practice. We need more honest post-mortems.",
        "Growing gap between how insiders and outsiders understand this field. The public narrative is years behind the actual state of things.",
        "Every year we have the same debate about credentials vs experience. Both matter. The framing is wrong.",
        "The newsletter ecosystem is getting noisy. There are three or four excellent ones and hundreds that just repackage the same information.",
        "Social media rewards confident predictions and punishes nuance. The people who hedge appropriately get less engagement than the ones who don't.",
        "Interesting how quickly the consensus shifted on this. Six months ago this was a fringe position, now everyone acts like they always agreed.",
    ],
    # Reactions, thanks, jokes, greetings, small talk, vibes, selfies.
    # These carry no transferable knowledge — pure social signal.
    "mood_social": [
        "lol",
        "haha yes exactly",
        "this is amazing!",
        "love it!",
        "so true",
        "happy birthday!",
        "congrats, well deserved!",
        "thanks everyone for the kind words",
        "good morning",
        "coffee time",
        "travelling today, see you soon",
        "great chat last night",
        "that was a rough day",
        "just landed, exhausted",
        "who else is watching the game tonight",
        "so many notifications, I can't keep up",
        "thank you so much, really appreciate it",
        "oh no",
        "this is hilarious",
        "mood",
        "same",
        "had a great time, thanks for having me",
        "my pleasure, happy to do it again anytime",
        "so proud of the team today",
        "friday vibes",
        "can't believe it's already December",
        "needed this today",
        "100%",
        "nailed it",
        "yep",
        "big if true",
        "crying laughing at this",
        "okay that's actually funny",
        "RIP my mentions",
        "the vibes are immaculate",
        "I can neither confirm nor deny",
        "weekend mode activated",
        "brb",
        "that was fun, let's do it again",
        "thanks for all the birthday messages",
    ],
}


def _centroid(model, sentences: list[str]):
    """Compute the L2-normalised mean embedding of a sentence list."""
    import numpy as np

    embs = model.encode(sentences)
    c = embs.mean(axis=0)
    c /= np.linalg.norm(c) + 1e-12
    return c


def filter_tweets(
    documents: dict[str, dict],
    keep_labels: set[str] | None = None,
    model_name: str = "minishlab/potion-base-8M",
) -> dict[str, dict]:
    """Classify tweet documents and keep only informational ones.

    Non-tweet documents (resources with ``source_url``, or URLs that
    don't match ``x.com/<user>/status/<id>``) always pass through
    untouched. Only own-tweet docs are classified and potentially dropped.

    Parameters
    ----------
    documents : dict[str, dict]
        Full document dict as returned by ``Twitter.__call__()``.
    keep_labels : set[str] or None
        Category labels to keep. Defaults to ``_KEEP_LABELS``:
        research_paper, technical_explanation, tool_resource,
        tutorial_education, opinion_insight.
    model_name : str
        model2vec model for embedding (default: ``potion-base-8M``).

    Returns
    -------
    dict[str, dict]
        Filtered documents. Each kept tweet gains a ``"label"`` field.
    """
    import numpy as np
    from model2vec import StaticModel

    labels_set: set[str] = keep_labels if keep_labels is not None else _KEEP_LABELS

    # Partition: tweets to classify vs everything else (passthrough).
    tweet_urls: list[str] = []
    tweet_texts: list[str] = []
    result: dict[str, dict] = {}

    for url, doc in documents.items():
        if _TWEET_URL_RE.match(url) and "source_url" not in doc:
            tweet_urls.append(url)
            tweet_texts.append(doc.get("summary") or "")
        else:
            result[url] = doc

    if not tweet_urls:
        return documents

    # Embed tweets and compute similarity to each category centroid.
    print(f"    Classifying {len(tweet_urls)} tweets...")
    model = StaticModel.from_pretrained(model_name)
    labels = list(_CATEGORIES.keys())
    centroids = np.stack([_centroid(model, _CATEGORIES[k]) for k in labels])

    embs = model.encode(tweet_texts)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms

    sims = embs @ centroids.T  # (N, C)
    best = sims.argmax(axis=1)

    kept = 0
    dropped = 0
    for i, url in enumerate(tweet_urls):
        label = labels[best[i]]
        if label in labels_set:
            doc = documents[url].copy()
            doc["label"] = label
            result[url] = doc
            kept += 1
        else:
            dropped += 1

    print(f"    Filter: kept {kept}, dropped {dropped} ({', '.join(sorted(_DROP_LABELS))})")
    return result
