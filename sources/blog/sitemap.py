"""
sitemap.xml / sitemapindex.xml fetcher.

Walks a sitemap (recursing into nested sitemapindex trees up to
``_SITEMAP_INDEX_MAX_DEPTH``) and yields one document per URL.
"""

import collections
import concurrent.futures
import re as _re
import threading
import urllib.parse
import urllib.request

from ._helpers import (
    _NS,
    _UA,
    _coerce_date,
    _decode_xml,
    _http_fetch,
    _parse_date,
    _safe_xml_parse,
    _title_from_url,
)

__all__ = ["Sitemap"]

# When a sitemap is small enough, we afford one HTTP GET per URL to
# pull the real <title> + <meta description> instead of synthesising
# both from the URL slug. The 100-URL ceiling keeps the worst-case
# fan-out under a minute on personal blogs while still completely
# avoiding scrape work for catalog-style sites.
_SAMPLE_PAGES_THRESHOLD = 100

# Global concurrency cap for the per-page sampler. 32 workers fans
# out enough across many small sites without overwhelming any single
# host (per-host throttling below caps that to 4 in-flight).
_SAMPLE_GLOBAL_WORKERS = 32
_SAMPLE_PER_HOST_LIMIT = 4
# Hard timeout per page; on miss we fall back to URL-slug derivation
# so a single slow host can't stall the pipeline.
_SAMPLE_FETCH_TIMEOUT = 8

_TITLE_RE = _re.compile(r"<title[^>]*>(.*?)</title>", _re.IGNORECASE | _re.DOTALL)
_META_RE = _re.compile(
    r"""<meta\s+[^>]*?(?:name|property)\s*=\s*["']?(description|og:description|twitter:description)["']?[^>]*?content\s*=\s*["']([^"']+)["']""",
    _re.IGNORECASE,
)
_META_RE_REVERSE = _re.compile(
    r"""<meta\s+[^>]*?content\s*=\s*["']([^"']+)["'][^>]*?(?:name|property)\s*=\s*["']?(description|og:description|twitter:description)["']?""",
    _re.IGNORECASE,
)
_HTML_TAG_RE = _re.compile(r"<[^>]+>")
_WS_RE = _re.compile(r"\s+")

# Default per-site URL cap. Kept deliberately tight (1000) because
# even well-curated personal blogs very rarely have more than this
# many posts worth indexing — sites that do are usually aggregators,
# CMS-heavy marketing sites, or SEO farms whose bulk pages we don't
# want in the library. Callers can raise this explicitly via
# ``max_urls`` when they know a site warrants deeper crawl.
_SITEMAP_MAX_URLS = 1_000
# How deep to recurse into nested sitemapindex trees.
_SITEMAP_INDEX_MAX_DEPTH = 3
# Fan-out cap per sitemap-index. Some sites (thenounproject.com,
# large e-commerce) publish hundreds of numbered child sitemaps;
# without a cap the pipeline spends minutes hammering a single host
# that we don't even care about. 50 children × 1k URLs each is
# still plenty of coverage for normal blogs.
_SITEMAP_INDEX_MAX_CHILDREN = 50

# Path fragments that almost always point at list/index pages rather
# than content. Skipping these removes the bulk of SEO noise from
# big blogging platforms (WordPress tags, Jekyll categories,
# paginated archives) without needing per-site knowledge. Any URL
# whose PATH contains one of these segments — as a discrete segment,
# not a substring — is dropped.
_IRRELEVANT_PATH_SEGMENTS: tuple[str, ...] = (
    "tag",
    "tags",
    "category",
    "categories",
    "author",
    "authors",
    "page",  # /page/2, /page/3, …
    "archive",
    "archives",
    "label",
    "labels",
    "topic",
    "topics",
    "feed",
    "rss",
    "atom",
    "amp",
    "search",
)


def _extract_meta(body: bytes) -> tuple[str, str, str]:
    """Pull (title, description, date) from an HTML body, regex-only.

    Skips full HTML parsing — the head section we care about is
    typically in the first few KB and we don't need DOM precision
    for these attributes. ``description`` falls back to OG / Twitter
    variants. HTML entities (``&#8211;``, ``&amp;``, …) are
    unescaped so titles read naturally.

    Date extraction (best-effort, ranked by confidence):
      1. ``<meta property="article:published_time" content="…">``
      2. ``<meta name="datePublished" content="…">`` / OG variants
      3. JSON-LD ``"datePublished":"…"``
      4. ``<time datetime="…">`` first occurrence
      5. Inline date strings near labels like "Published on" / "Posted"

    Empty string when nothing parseable is found.
    """
    import html as _html

    # Larger window for date extraction — page templates often put the
    # publication date in a body-level <time> or inline span, not the
    # head. 96 KB is enough for the first article block on most blogs.
    body_text = body[:96_000].decode("utf-8", errors="replace")
    head = body_text[:32_768]
    title = ""
    m = _TITLE_RE.search(head)
    if m:
        title = _HTML_TAG_RE.sub(" ", m.group(1))
        title = _html.unescape(title)
        title = _WS_RE.sub(" ", title).strip()
    desc = ""
    for pattern in (_META_RE, _META_RE_REVERSE):
        m = pattern.search(head)
        if m:
            # Group ordering depends on which pattern matched;
            # description is whichever group looks like prose.
            candidates = [g for g in m.groups() if g and len(g) > 4]
            for c in candidates:
                if c.lower() not in {"description", "og:description", "twitter:description"}:
                    desc = _html.unescape(_WS_RE.sub(" ", c)).strip()
                    break
        if desc:
            break
    date = _extract_date_from_html(body_text)
    return title, desc, date


# ── Date extraction from page body ─────────────────────────────────────
#
# Order matters: try the most authoritative signals first
# (article:published_time, JSON-LD datePublished), fall back to
# <time> tags and finally to inline-text scanning.

_DATE_META_PATTERNS = (
    # property="article:published_time" content="..."
    _re.compile(
        r"""<meta[^>]+(?:property|name)=["'](?:article:published_time|article:published|og:published_time|datePublished|date|publish[-_]?date|pubdate|publication_date|date\.published)["'][^>]+content=["']([^"']+)["']""",
        _re.IGNORECASE,
    ),
    # content="..." comes first
    _re.compile(
        r"""<meta[^>]+content=["']([^"']+)["'][^>]+(?:property|name)=["'](?:article:published_time|article:published|og:published_time|datePublished|date|publish[-_]?date|pubdate|publication_date|date\.published)["']""",
        _re.IGNORECASE,
    ),
    # JSON-LD: "datePublished": "2023-01-15T..."
    _re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),
    # <time datetime="2023-01-15">…</time> — first hit only
    _re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', _re.IGNORECASE),
    # itemprop="datePublished" content="..."
    _re.compile(
        r"""<[^>]+itemprop=["']datePublished["'][^>]*(?:content|datetime)=["']([^"']+)["']""",
        _re.IGNORECASE,
    ),
)
# Inline-text fallback. Scans for labelled date strings like
# "Published on January 1, 2023", "Posted January 1, 2023",
# "Date: 2023-01-15". Uses the existing `_parse_date` to normalise.
_INLINE_DATE_LABEL_RE = _re.compile(
    # Strict label whitelist: "Published [on] …", "Posted [on] …",
    # "Publié(e) [le] …", "Date: …". We deliberately do NOT match a
    # bare "on" prefix — it false-positives on template footers like
    # "Released on: November 28, 2023" (Timothy Ricks Webflow template
    # at the bottom of every lighton.ai blog post).
    r"(?:published|posted|publi[ée](?:e|ed)?|date)\s*"
    r"(?:on\s+|le\s+|:\s*)?"
    r"((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:January|February|March|"
    r"April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|"
    r"Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
    _re.IGNORECASE,
)
# Bare date strings — last-ditch when no label is present. Common on
# minimal templates (the Webflow blog at lighton.ai is a real example:
# the date sits in a plain <div class="text-size-small text-color-dark">
# with no microdata).
_BARE_DATE_RE = _re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    _re.IGNORECASE,
)


def _extract_date_from_html(text: str) -> str:
    """Return a YYYY-MM-DD date from an HTML body, or ``""``."""
    from ._helpers import _parse_date

    # 1-3. Structured signals first.
    for pat in _DATE_META_PATTERNS:
        m = pat.search(text)
        if m:
            d = _parse_date(m.group(1))
            if d:
                return d
    # 4. Labelled inline strings.
    m = _INLINE_DATE_LABEL_RE.search(text)
    if m:
        d = _parse_date(m.group(1))
        if d:
            return d
    # 5. Bare "Month Day, Year" anywhere in body. We pick the FIRST hit;
    # publication dates are typically near the top of the article.
    m = _BARE_DATE_RE.search(text)
    if m:
        d = _parse_date(m.group(0))
        if d:
            return d
    return ""


def _fetch_page_meta(url: str, timeout: int = _SAMPLE_FETCH_TIMEOUT) -> tuple[str, str, str]:
    """HTTP GET ``url`` and extract (title, description, date).

    Empty strings on error or if a particular field couldn't be
    parsed. The date field is the publication date inferred from
    common metadata (article:published_time, JSON-LD, <time>, …)
    or, as a last resort, inline-text scanning of the page body.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = (resp.headers.get("Content-Type") or "").lower()
            # Skip non-HTML responses (PDF, images) — they have no
            # <title>/<meta> we'd recognise, and we don't want to
            # download large binaries.
            if ct and "html" not in ct and "xml" not in ct:
                return "", "", ""
            raw = resp.read(512_000)
            enc = (resp.headers.get("Content-Encoding") or "").lower()
            if "gzip" in enc:
                import gzip as _gz

                raw = _gz.decompress(raw)
            elif "deflate" in enc:
                import zlib

                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    raw = zlib.decompress(raw, -zlib.MAX_WBITS)
            return _extract_meta(raw)
    except Exception:
        return "", "", ""


def _sample_pages(urls: list[str]) -> dict[str, tuple[str, str, str]]:
    """Parallel fetch. Returns ``{url: (title, description, date)}``.

    URLs whose fetch fails or returns nothing land with ``("", "", "")``;
    callers should fall back to URL-slug derivation in that case.

    Concurrency: a global thread pool fans out 32 fetches at once,
    but a per-host semaphore caps to ``_SAMPLE_PER_HOST_LIMIT`` so a
    single slow site can't soak the pool and so we don't trip
    rate-limit responses on any one host.
    """
    out: dict[str, tuple[str, str, str]] = {}
    if not urls:
        return out
    host_sems: dict[str, threading.Semaphore] = collections.defaultdict(
        lambda: threading.Semaphore(_SAMPLE_PER_HOST_LIMIT)
    )

    def _job(url: str) -> tuple[str, tuple[str, str, str]]:
        host = urllib.parse.urlparse(url).netloc.lower()
        sem = host_sems[host]
        with sem:
            return url, _fetch_page_meta(url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_SAMPLE_GLOBAL_WORKERS) as ex:
        for url, meta in ex.map(_job, urls):
            out[url] = meta
    return out


def _looks_like_list_page(url: str) -> bool:
    """True when the URL's path contains an index/listing segment.

    Conservative: checks *segments* (e.g. drop `/tag/ml`, keep
    `/posts/tagging-nlp`) so legitimate slugs that happen to contain
    "tag" or "page" aren't dropped.
    """
    try:
        from urllib.parse import urlparse

        path = urlparse(url).path.strip("/").lower()
    except Exception:
        return False
    if not path:
        return False
    segments = path.split("/")
    return any(seg in _IRRELEVANT_PATH_SEGMENTS for seg in segments)


def _parse_sitemap_xml(
    raw: bytes,
    content_type: str,
    base_url: str,
    depth: int = 0,
    seen_indexes: set[str] | None = None,
    remaining: int | None = None,
) -> list[tuple[str, str]]:
    """Parse a sitemap.xml or sitemapindex. Recurses into nested indexes.

    Returns a flat list of (url, lastmod) tuples, capped at `remaining`.
    """
    seen_indexes = seen_indexes if seen_indexes is not None else set()
    text = _decode_xml(raw, content_type)
    root = _safe_xml_parse(text)
    if root is None:
        return []

    results: list[tuple[str, str]] = []
    tag = root.tag.lower()
    # sitemapindex → recurse into each child <sitemap><loc>...</loc></sitemap>
    if tag.endswith("}sitemapindex") or tag == "sitemapindex":
        if depth >= _SITEMAP_INDEX_MAX_DEPTH:
            return []
        children_visited = 0
        for sm_el in root.findall("sm:sitemap", _NS) or root.findall("sitemap"):
            if children_visited >= _SITEMAP_INDEX_MAX_CHILDREN:
                print(f"    Skipping rest of sitemap-index at {base_url} (> {_SITEMAP_INDEX_MAX_CHILDREN} children)")
                break
            loc = sm_el.find("sm:loc", _NS)
            if loc is None:
                loc = sm_el.find("loc")
            child_url = (loc.text or "").strip() if loc is not None else ""
            if not child_url or child_url in seen_indexes:
                continue
            seen_indexes.add(child_url)
            children_visited += 1
            try:
                child_raw, child_final, child_ct = _http_fetch(child_url, timeout=30)
            except Exception as e:
                print(f"    Skipping child sitemap {child_url}: {e}")
                continue
            results.extend(
                _parse_sitemap_xml(
                    child_raw,
                    child_ct,
                    child_final,
                    depth=depth + 1,
                    seen_indexes=seen_indexes,
                    remaining=(None if remaining is None else max(0, remaining - len(results))),
                )
            )
            if remaining is not None and len(results) >= remaining:
                break
        return results

    # Regular urlset
    urls = root.findall("sm:url", _NS) or root.findall("url")
    for url_el in urls:
        loc = url_el.find("sm:loc", _NS)
        if loc is None:
            loc = url_el.find("loc")
        if loc is None or not (loc.text or "").strip():
            continue
        lastmod = url_el.find("sm:lastmod", _NS)
        if lastmod is None:
            lastmod = url_el.find("lastmod")
        date = _parse_date(lastmod.text) if lastmod is not None and lastmod.text else ""
        results.append(((loc.text or "").strip(), date))
        if remaining is not None and len(results) >= remaining:
            break
    return results


class Sitemap:
    """
    Fetch blog posts from a sitemap.xml (or sitemapindex).

    Parameters
    ----------
    sitemap_url : str
    tags : list[str]
    url_filter : str | None
        Only include URLs containing this substring.
    max_urls : int | None
        Hard cap on URLs returned (defaults to `_SITEMAP_MAX_URLS`).
    """

    def __init__(
        self,
        sitemap_url: str,
        tags: list[str] | None = None,
        url_filter: str | None = None,
        max_urls: int | None = None,
    ):
        self.sitemap_url = sitemap_url
        self.tags = tags or []
        self.url_filter = url_filter
        self.max_urls = max_urls if max_urls is not None else _SITEMAP_MAX_URLS

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        print(f"    Fetching sitemap: {self.sitemap_url}")
        try:
            raw, final_url, ct = _http_fetch(self.sitemap_url)
        except Exception as e:
            print(f"    Failed to fetch sitemap: {e}")
            return {}

        # Parse one item past the cap so we can distinguish
        # "site has exactly max_urls articles" (keep) from
        # "site has more than max_urls articles" (catalog → skip).
        try:
            entries = _parse_sitemap_xml(raw, ct, final_url, remaining=self.max_urls + 1)
        except Exception as e:
            print(f"    Failed to parse sitemap: {e}")
            return {}

        # First pass: apply the structural filters (url_filter, list/
        # index-page skip) and count the remaining "content" URLs.
        candidates: list[tuple[str, str]] = []
        skipped_list_pages = 0
        for url, date in entries:
            if self.url_filter and self.url_filter not in url:
                continue
            if _looks_like_list_page(url):
                skipped_list_pages += 1
                continue
            candidates.append((url, date))

        # Catalog/directory guard: if a site has more content URLs
        # than ``max_urls``, it's almost certainly not a personal blog
        # (it's a parts catalog, art gallery, dataset host, …). We'd
        # rather return zero than a truncated slice — partial indexes
        # confuse search results and bloat the dead-link probe.
        if len(candidates) > self.max_urls:
            print(
                f"    Skipping {self.sitemap_url}: {len(candidates)}+ content URLs "
                f"exceeds max_urls={self.max_urls} (looks like a catalog, not a blog)"
            )
            return {}

        # Decide once per sitemap whether to scrape each page for
        # real metadata. Small sitemaps (< _SAMPLE_PAGES_THRESHOLD)
        # afford one HTTP per URL — that's typically a personal blog
        # where the slug-derived title would be poor anyway. Large
        # sitemaps fall back to the URL-slug derivation; junk titles
        # get filtered downstream by the title/summary cleanup.
        sampled_meta: dict[str, tuple[str, str, str]] = {}
        new_candidates = [(url, date) for url, date in candidates if not (existing_urls and url in existing_urls)]
        # Early-return: every URL the sitemap lists is already in the DB.
        # Avoid the per-page metadata scrape (the only expensive step
        # here) and the empty-data return-path noise. We still got the
        # listing fetch — that's how we *discovered* the no-change state.
        # Re-runs on a stable site collapse to a single HTTP roundtrip.
        if existing_urls and candidates and not new_candidates:
            print(f"    Sitemap up-to-date: {len(candidates)} URLs all known ({self.sitemap_url})")
            return {}
        if 0 < len(new_candidates) <= _SAMPLE_PAGES_THRESHOLD:
            print(
                f"    Sampling {len(new_candidates)} pages for real titles "
                f"(under {_SAMPLE_PAGES_THRESHOLD}-URL threshold)"
            )
            sampled_meta = _sample_pages([url for url, _ in new_candidates])

        data: dict[str, dict] = {}
        sampled_used = 0
        for rank, (url, date) in enumerate(new_candidates):
            slug_title = _title_from_url(url)
            scraped_title, scraped_desc, scraped_date = sampled_meta.get(url, ("", "", ""))
            # Date precedence: sitemap lastmod (if non-empty) > scraped
            # body date > URL-embedded date > rank-based fallback. The
            # scraped date overrides the URL year-only fallback (which
            # would land on Jan 1 of the year embedded in the slug),
            # giving us real publication dates from `<meta
            # article:published_time>` / JSON-LD / `<time datetime=…>`
            # whenever the page exposes them. When sitemap lastmod is
            # present, we trust it — sitemaps that bother emitting
            # lastmod usually do so accurately.
            if not date and scraped_date:
                date = scraped_date
            date = _coerce_date(date, fallback_url=url, rank=rank)
            if scraped_title:
                sampled_used += 1
                title = scraped_title
                # If the scraped description is missing, leave summary
                # empty rather than echoing the title (the cleanup
                # filter penalises title==summary single-token docs).
                summary = scraped_desc
            else:
                title = slug_title
                summary = slug_title if slug_title != url else ""
            data[url] = {
                "title": title,
                "summary": summary,
                "date": date,
                "tags": list(self.tags),
            }

        tail_parts = []
        if skipped_list_pages:
            tail_parts.append(f"skipped {skipped_list_pages} list/index pages")
        if sampled_used:
            tail_parts.append(f"{sampled_used} titles from page scrape")
        tail = (" · " + " · ".join(tail_parts)) if tail_parts else ""
        print(f"    Parsed {len(data)} URLs from sitemap{tail}")
        return data
