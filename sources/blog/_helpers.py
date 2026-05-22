"""
Shared helpers for the blog package — feed/sitemap parsing primitives.

HTML/XML utilities, date parsing, URL cleanup, and per-format feed parsers
(Atom, RSS 2.0, RDF, JSON Feed). Used by `Feed` and `Sitemap`. No public API.
"""

import gzip
import json as _json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape

# Namespace prefixes used across feed formats
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rss1": "http://purl.org/rss/1.0/",
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
}

# Browser-ish UA — some sites (Cloudflare-fronted, academic servers) 403 plain tool UAs.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15 Knowledge/1.0"
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Script/style/noscript blocks — remove tag AND content before tag-stripping so
# CSS/JS doesn't leak into summaries (common on Jekyll + WordPress themes).
_SCRIPT_STYLE_RE = re.compile(
    r"<(?:script|style|noscript|template)\b[^>]*>.*?</(?:script|style|noscript|template)>",
    re.IGNORECASE | re.DOTALL,
)
# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Summary prefixes we want to strip: leading RFC-style dates ("Thu, 09 Apr 2026 ...")
# that some feeds emit as the first paragraph of the description.
_LEADING_RFC_DATE_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
    r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[+\-]\d{4}|\s*[A-Z]{2,4})?)?",
    re.IGNORECASE,
)
# Common "Continue reading" / "Read more" / "The post ... appeared first on ..." footers.
_TRAILING_NOISE_RE = re.compile(
    r"\s*(?:Continue reading.*|Read (?:more|the rest).*|"
    r"The post .*? appeared first on .+?\.?|"
    r"\[\s*…\s*\]|"
    r"\(more…\))\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Weekday abbreviations longer than 3 chars — some non-compliant feeds use
# "Thurs", "Tues", "Wednes", "Sunda". We normalize to the 3-letter standard.
_WEEKDAY_NORMALIZE = [
    (re.compile(r"^\s*(Monday|Mondy)", re.I), "Mon"),
    (re.compile(r"^\s*Tues(?:day)?", re.I), "Tue"),
    (re.compile(r"^\s*Wednes(?:day)?", re.I), "Wed"),
    (re.compile(r"^\s*Thurs(?:day)?", re.I), "Thu"),
    (re.compile(r"^\s*Frid(?:ay)?", re.I), "Fri"),
    (re.compile(r"^\s*Saturd(?:ay)?", re.I), "Sat"),
    (re.compile(r"^\s*Sund(?:ay)?", re.I), "Sun"),
]
# Some feeds emit "UTC" or "GMT+0" as timezone name — strptime only knows a small set.
_TZ_NORMALIZE = [
    (re.compile(r"\bUTC\b"), "+0000"),
    (re.compile(r"\bGMT\s*\+?0+\b"), "+0000"),
    (re.compile(r"\bEST\b"), "-0500"),
    (re.compile(r"\bEDT\b"), "-0400"),
    (re.compile(r"\bPST\b"), "-0800"),
    (re.compile(r"\bPDT\b"), "-0700"),
]


def _strip_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace.

    First strips <script>/<style>/<noscript>/<template> elements *with their content*
    so CSS and JavaScript don't leak into summaries (Jekyll themes and WordPress
    plugins routinely inject these inline). Then unescapes entities and removes the
    remaining markup. The order is important: inner markup must be stripped BEFORE
    entity unescape so that escaped `&lt;script&gt;` in CDATA doesn't suddenly look
    like a real tag.
    """
    if text is None:
        return ""
    # Work on escaped form first — CDATA description fields are often double-escaped.
    # Strip visible script/style/comment blocks whether tag-escaped or not.
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = unescape(text)
    # After unescape, a second pass catches script/style that were &lt;script&gt; encoded.
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _clean_summary(text: str, max_len: int = 300) -> str:
    """Strip leading/trailing noise from a summary and truncate cleanly."""
    if not text:
        return ""
    s = text.strip()
    # Remove a leading embedded RFC-date paragraph that some feeds prefix
    s = _LEADING_RFC_DATE_RE.sub("", s, count=1).lstrip(" \t:-|—")
    # Trailing "Continue reading..." / "The post X appeared first on Y"
    s = _TRAILING_NOISE_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    if len(s) > max_len:
        # Cut at last word boundary within the limit for nicer truncation
        cut = s[: max_len - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        s = cut.rstrip(" ,.;:-—") + "…"
    return s


def _clean_title(title: str, link: str | None = None) -> str:
    """Clean a feed-entry title.

    - Strip whitespace and HTML.
    - Title-case slug-style all-lowercase titles ("microgpt" → "Microgpt" → from URL slug).
    - Strip trailing " | Site Name" / " - Blog Name" suffixes that duplicate the feed title.
    """
    if not title:
        return ""
    t = _strip_html(title)
    # Strip trailing site-name suffixes. We only trim when:
    #   - separator is " | " or " – " or " — " (pipe / en-dash / em-dash), or
    #   - separator is " - " AND the suffix is a short (<=3 words) non-sentence tail
    #     AND the prefix is meaningfully longer (>= 4 words)
    # This avoids eating real subtitles like "Running faster - How I optimized this".
    m = re.search(r"^(.*?)\s+([|–—])\s+([^|–—]{2,40})$", t)
    if m:
        t = m.group(1).strip()
    else:
        m = re.search(r"^(.*?)\s+-\s+([^\-]{2,40})$", t)
        if m:
            prefix, suffix = m.group(1).strip(), m.group(2).strip()
            if (
                len(prefix.split()) >= 4
                and len(suffix.split()) <= 3
                and suffix.lower() not in {"the sequel", "part 2", "part ii", "part one", "part two"}
            ):
                t = prefix
    # If the title is a single lowercase token (Jekyll slug-as-title), pretty it up
    if link and t and t == t.lower() and " " not in t and "-" not in t and "_" not in t:
        # Try a nicer title from the URL slug
        slug_title = _title_from_url(link)
        if slug_title and slug_title.lower() != t.lower():
            # Prefer the URL-derived title if it contains info (multi-word)
            if " " in slug_title:
                return slug_title
        return t[:1].upper() + t[1:]
    return t


def _tags_from_url(url: str) -> list[str]:
    """Derive crude tag hints from URL path segments.

    e.g. https://blog.langchain.com/tag/agents/ → ["agents"]
         https://example.com/blog/category/llms/foo → ["llms"]
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return []
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return []
    hints: list[str] = []
    BORING = {
        "posts",
        "post",
        "blog",
        "blogs",
        "tag",
        "tags",
        "category",
        "categories",
        "articles",
        "article",
        "p",
        "entry",
        "essays",
        "essay",
        "notes",
        "note",
        "news",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
    for p in parts[:-1]:  # exclude the final slug
        slug = p.lower()
        if slug in BORING:
            continue
        if slug.isdigit() or re.match(r"^\d{4}$", slug) or re.match(r"^\d{1,2}$", slug):
            continue  # year/month segments
        # Skip date-prefixed post slugs (e.g. "2019-05-collaboration", "2024-03-15-title")
        if re.match(r"^\d{4}[-_]\d{1,2}([-_]\d{1,2})?([-_].+)?$", slug):
            continue
        # A segment like "large-language-models" → "large language models"
        hint = slug.replace("-", " ").replace("_", " ").strip()
        # Drop if it has any digit (likely a numeric id or dated slug) or is too long
        if any(c.isdigit() for c in hint):
            continue
        if 2 < len(hint) < 40:
            hints.append(hint)
    return hints


_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a, %d %b %Y %H:%M:%S GMT",
    "%d %b %Y %H:%M:%S %z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _normalize_date_string(s: str) -> str:
    """Normalize non-compliant weekday/timezone tokens so strptime can parse."""
    for pat, repl in _WEEKDAY_NORMALIZE:
        s = pat.sub(repl, s, count=1)
    for pat, repl in _TZ_NORMALIZE:
        s = pat.sub(repl, s)
    return s


def _parse_date(date_str: str) -> str:
    """Parse various date formats into YYYY-MM-DD. Returns "" if unparseable."""
    if not date_str:
        return ""
    s = _normalize_date_string(date_str.strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Final resort: any YYYY-MM-DD substring
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Or YYYY/MM/DD
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def _coerce_date(
    date_str: str,
    fallback_url: str | None = None,
    rank: int | None = None,
) -> str:
    """Best-effort date: parse date_str, else extract from URL, else
    walk-backwards-from-today by ``rank`` days, else today.

    The ``rank`` fallback is what stops a sitemap from stamping all
    420 of its URLs with today's date — call sites pass an
    enumerate index so each unknown-date URL within the same fetch
    gets a distinct day, preserving the page's order. If ``rank`` is
    None (legacy callers), the behaviour is the original "today".
    """
    from datetime import timedelta

    d = _parse_date(date_str)
    if d:
        return d
    if fallback_url:
        d = _date_from_url(fallback_url)
        if d:
            return d
    if rank is not None and rank > 0:
        return (datetime.now() - timedelta(days=rank)).strftime("%Y-%m-%d")
    return _today()


def _http_fetch(url: str, timeout: int = 30) -> tuple[bytes, str, str]:
    """Fetch a URL. Returns (body_bytes, final_url, content_type).

    Handles gzipped HTTP responses transparently (sites often return
    Content-Encoding: gzip even when not asked).
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "application/atom+xml, application/rss+xml, application/xml, "
            "application/json, text/xml, */*;q=0.5",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        enc = (resp.headers.get("Content-Encoding") or "").lower()
        if "gzip" in enc:
            raw = gzip.decompress(raw)
        elif "deflate" in enc:
            import zlib

            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                raw = zlib.decompress(raw, -zlib.MAX_WBITS)
        return raw, resp.geturl(), resp.headers.get("Content-Type", "")


def _decode_xml(raw: bytes, content_type: str = "") -> str:
    """Decode XML bytes to str, respecting BOM / XML declaration / HTTP charset."""
    # If the bytes are gzip-magic (0x1f 0x8b), decompress. This handles .xml.gz files
    # that weren't flagged by Content-Encoding.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    # Charset from Content-Type header
    ct_charset = None
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if m:
        ct_charset = m.group(1)
    # XML declaration
    xml_charset = None
    m = re.match(rb"<\?xml[^>]*encoding=[\"']([\w\-]+)[\"']", raw)
    if m:
        try:
            xml_charset = bytes(m.group(1)).decode("ascii")
        except Exception:
            xml_charset = None
    for enc in (xml_charset, ct_charset, "utf-8", "utf-8-sig", "latin-1"):
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _safe_xml_parse(text: str) -> ET.Element | None:
    """Parse XML, stripping problematic xml-stylesheet PIs if the first parse fails."""
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        # Strip processing instructions and comments that sometimes break ET
        stripped = re.sub(r"<\?xml-stylesheet[^>]*\?>", "", text)
        stripped = re.sub(r"<!--.*?-->", "", stripped, flags=re.DOTALL)
        try:
            return ET.fromstring(stripped)
        except ET.ParseError:
            return None


# ---------- feed parsers ---------------------------------------------------


_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_name",
    "utm_id",
    "utm_reader",
    "utm_referrer",
    "mc_cid",
    "mc_eid",
    "fbclid",
    "gclid",
    "gbraid",
    "wbraid",
    "igshid",
    "ref",
    "ref_src",
    "ref_url",
    "spm",
    "yclid",
    "_hsenc",
    "_hsmi",
}


def _strip_tracking(url: str) -> str:
    """Remove common tracking query params so equivalent URLs dedup cleanly."""
    try:
        pr = urllib.parse.urlparse(url)
    except Exception:
        return url
    if not pr.query:
        return url
    # parse_qsl keeps duplicates; keeping all of them is fine, we just filter keys.
    kept = [
        (k, v) for k, v in urllib.parse.parse_qsl(pr.query, keep_blank_values=True) if k.lower() not in _TRACKING_PARAMS
    ]
    new_query = urllib.parse.urlencode(kept)
    cleaned = pr._replace(query=new_query, fragment="")
    return urllib.parse.urlunparse(cleaned).rstrip("?")


def _resolve_url(base: str, href: str) -> str:
    """Resolve a feed-entry href against the feed's final URL and drop trackers."""
    if not href:
        return ""
    absolute = urllib.parse.urljoin(base, href.strip())
    return _strip_tracking(absolute)


def _fallback_summary(summary: str, title: str, category_hints: list[str] | None = None) -> str:
    """If the feed exposed no usable summary, fall back to the title itself.

    Rationale: many personal feeds (Sophie Alpert, Mitchell Hashimoto, bare
    academic Jekyll feeds) omit <summary>/<description> entirely. Search quality
    is better with the title echoed into summary than with an empty field.

    Also: some feeds (e.g. research.google) stuff a single category string into
    the description. If `summary` is short and identical to one of the category
    tags, prefer the title instead — the title carries the real article info.
    """
    if summary:
        low = summary.strip().lower()
        if category_hints and len(low) < 30:
            cats = {(c or "").strip().lower() for c in category_hints}
            if low in cats:
                return (title or "").strip()
        return summary
    return (title or "").strip()


def _parse_atom(root: ET.Element, tags: list[str], base_url: str) -> dict[str, dict]:
    data = {}
    for rank, entry in enumerate(root.findall("atom:entry", _NS)):
        title_el = entry.find("atom:title", _NS)
        raw_title = title_el.text if title_el is not None else ""

        # Pick the best alternate link
        link = ""
        for link_el in entry.findall("atom:link", _NS):
            rel = link_el.get("rel", "alternate")
            href = link_el.get("href", "")
            if rel == "alternate" and href:
                link = _resolve_url(base_url, href)
                break
            if not link and href and rel not in {"self", "enclosure", "edit"}:
                link = _resolve_url(base_url, href)

        title = _clean_title(raw_title, link=link)
        if not link or not title:
            continue

        published = entry.find("atom:published", _NS)
        updated = entry.find("atom:updated", _NS)
        date_str = ""
        if published is not None and published.text:
            date_str = published.text
        elif updated is not None and updated.text:
            date_str = updated.text

        # Content / summary: handle type=html, type=xhtml, plain text
        summary = ""
        content_el = entry.find("atom:content", _NS)
        summary_el = entry.find("atom:summary", _NS)
        if content_el is not None:
            ctype = (content_el.get("type") or "text").lower()
            if ctype == "xhtml":
                # Content is a child <div> with XHTML inside
                parts: list[str] = []
                for child in content_el:
                    parts.append(ET.tostring(child, encoding="unicode", method="text") or "")
                summary = _strip_html(" ".join(parts))
            elif content_el.text:
                summary = _strip_html(content_el.text)
        if not summary and summary_el is not None and summary_el.text:
            summary = _strip_html(summary_el.text)
        summary = _fallback_summary(_clean_summary(summary), title)

        entry_tags = list(tags)
        for cat in entry.findall("atom:category", _NS):
            term = (cat.get("term") or cat.get("label") or "").strip().lower()
            if term and term not in entry_tags:
                entry_tags.append(term)
        # Fall back to URL path hints if the feed didn't expose categories
        if len(entry_tags) == len(tags):
            for hint in _tags_from_url(link):
                if hint not in entry_tags:
                    entry_tags.append(hint)

        data[link] = {
            "title": title,
            "summary": summary,
            "date": _coerce_date(date_str, fallback_url=link, rank=rank),
            "tags": entry_tags,
        }
    return data


def _parse_rss2(root: ET.Element, tags: list[str], base_url: str) -> dict[str, dict]:
    data = {}
    channel = root.find("channel")
    if channel is None:
        return data

    for rank, item in enumerate(channel.findall("item")):
        title_el = item.find("title")
        raw_title = title_el.text if title_el is not None else ""

        link_el = item.find("link")
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        if not link:
            # Atom-namespaced link inside an RSS 2.0 item (sometimes used)
            atom_link = item.find("atom:link", _NS)
            if atom_link is not None:
                link = atom_link.get("href", "")
        # Some feeds (Blogger) put the canonical link in <guid isPermaLink="true">
        if not link:
            guid = item.find("guid")
            if guid is not None and guid.text and (guid.get("isPermaLink", "true").lower() != "false"):
                link = guid.text.strip()
        link = _resolve_url(base_url, link)
        title = _clean_title(raw_title, link=link)
        if not link or not title:
            continue

        pub_date = item.find("pubDate")
        dc_date = item.find("dc:date", _NS)
        date_str = ""
        if pub_date is not None and pub_date.text:
            date_str = pub_date.text
        elif dc_date is not None and dc_date.text:
            date_str = dc_date.text

        summary = ""
        content_encoded = item.find("content:encoded", _NS)
        description = item.find("description")
        if content_encoded is not None and content_encoded.text:
            summary = _strip_html(content_encoded.text)
        elif description is not None and description.text:
            summary = _strip_html(description.text)
        # Gather categories first so _fallback_summary can detect category-only
        # descriptions (research.google emits only the category into <description>).
        entry_tags = list(tags)
        raw_cats: list[str] = []
        for cat in item.findall("category"):
            term = (cat.text or "").strip()
            if term:
                raw_cats.append(term)
                if term.lower() not in entry_tags:
                    entry_tags.append(term.lower())
        for subj in item.findall("dc:subject", _NS):
            term = (subj.text or "").strip()
            if term:
                raw_cats.append(term)
                if term.lower() not in entry_tags:
                    entry_tags.append(term.lower())
        summary = _fallback_summary(_clean_summary(summary), title, category_hints=raw_cats)
        if len(entry_tags) == len(tags):
            for hint in _tags_from_url(link):
                if hint not in entry_tags:
                    entry_tags.append(hint)

        data[link] = {
            "title": title,
            "summary": summary,
            "date": _coerce_date(date_str, fallback_url=link, rank=rank),
            "tags": entry_tags,
        }
    return data


def _parse_rdf(root: ET.Element, tags: list[str], base_url: str) -> dict[str, dict]:
    """RSS 1.0 (RDF) — items are siblings of channel, linked via rdf:about."""
    data = {}
    for rank, item in enumerate(root.findall("rss1:item", _NS)):
        title_el = item.find("rss1:title", _NS)
        raw_title = title_el.text if title_el is not None else ""
        link_el = item.find("rss1:link", _NS)
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        if not link:
            link = item.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
        link = _resolve_url(base_url, link)
        title = _clean_title(raw_title, link=link)
        if not link or not title:
            continue

        dc_date = item.find("dc:date", _NS)
        date_str = dc_date.text if dc_date is not None and dc_date.text else ""

        desc = item.find("rss1:description", _NS)
        content_encoded = item.find("content:encoded", _NS)
        summary = ""
        if content_encoded is not None and content_encoded.text:
            summary = _strip_html(content_encoded.text)
        elif desc is not None and desc.text:
            summary = _strip_html(desc.text)
        summary = _fallback_summary(_clean_summary(summary), title)

        entry_tags = list(tags)
        for subj in item.findall("dc:subject", _NS):
            term = (subj.text or "").strip().lower()
            if term and term not in entry_tags:
                entry_tags.append(term)
        if len(entry_tags) == len(tags):
            for hint in _tags_from_url(link):
                if hint not in entry_tags:
                    entry_tags.append(hint)

        data[link] = {
            "title": title,
            "summary": summary,
            "date": _coerce_date(date_str, fallback_url=link, rank=rank),
            "tags": entry_tags,
        }
    return data


def _parse_json_feed(text: str, tags: list[str], base_url: str) -> dict[str, dict]:
    """JSON Feed 1.1 — https://www.jsonfeed.org/version/1.1/"""
    try:
        doc = _json.loads(text)
    except ValueError:
        return {}
    if not isinstance(doc, dict):
        return {}
    data = {}
    for rank, item in enumerate(doc.get("items") or []):
        if not isinstance(item, dict):
            continue
        link = (item.get("url") or item.get("external_url") or "").strip()
        link = _resolve_url(base_url, link)
        title = _clean_title(item.get("title") or "", link=link)
        if not link or not title:
            continue
        date_str = item.get("date_published") or item.get("date_modified") or ""
        summary = ""
        for key in ("summary", "content_text", "content_html"):
            val = item.get(key)
            if val:
                summary = _strip_html(val) if key == "content_html" else val.strip()
                break
        summary = _fallback_summary(_clean_summary(summary), title)
        entry_tags = list(tags)
        for t in item.get("tags") or []:
            t = (t or "").strip().lower()
            if t and t not in entry_tags:
                entry_tags.append(t)
        if len(entry_tags) == len(tags):
            for hint in _tags_from_url(link):
                if hint not in entry_tags:
                    entry_tags.append(hint)
        data[link] = {
            "title": title,
            "summary": summary,
            "date": _coerce_date(date_str, fallback_url=link, rank=rank),
            "tags": entry_tags,
        }
    return data


def _dispatch_feed(text: str, tags: list[str], base_url: str) -> dict[str, dict]:
    """Detect feed format and parse. Returns {} on unknown format."""
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith("{"):
        return _parse_json_feed(text, tags, base_url)

    root = _safe_xml_parse(text)
    if root is None:
        return {}
    tag = root.tag.lower()
    if tag == "{http://www.w3.org/2005/atom}feed" or tag == "feed":
        return _parse_atom(root, tags, base_url)
    if tag == "rss":
        return _parse_rss2(root, tags, base_url)
    if tag.endswith("}rdf") or tag == "rdf:rdf":
        return _parse_rdf(root, tags, base_url)
    return {}


_MONTH_MAP = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

_URL_DATE_RE = re.compile(r"/(\d{4})/([A-Za-z]{3}|\d{1,2})/(\d{1,2})/")
# `/YYYY-MM-DD<sep>` anywhere in the path. Separator class is liberal
# (`-`, `/`, `_`, `.`) because real-world blog URLs use all of them:
#   /2023-05-31-slug/         (Hugo / Jekyll)
#   /2023-05-31/slug          (Hugo / Jekyll without trailing dash)
#   /posts/2023-05-31_slug/   (Distill / djnavarro.net underscore style)
#   /2023-05-31.html          (some flat layouts)
_URL_DATE_DASH_RE = re.compile(r"/(\d{4})-(\d{2})-(\d{2})(?:[-/_.]|$)")
# Year alone at the start of a path segment, e.g. `/blog/2023-rise-of-llms`.
# Restrict to last-segment slug starts so we don't mis-fire on container
# directories like `/posts/2024-archive/`. Years 1990-2099 only — leaves
# room for fictional or historical posts without catching e.g. "2401" in
# product slugs.
_URL_YEAR_SLUG_RE = re.compile(
    r"/(19[9]\d|20\d{2})(?:-|$)",
)


def _date_from_url(url: str) -> str:
    """Best-effort date from a URL's path.

    Tries three patterns, in order of confidence:
      1. ``/YYYY/MM/DD/...``  (canonical blog permalink) → full date
      2. ``/YYYY-MM-DD-...``  (Hugo/Jekyll-style slug)    → full date
      3. ``/YYYY-...``         (year-prefixed slug)        → ``YYYY-01-01``

    Pattern (3) is a soft fallback: it gets the *year* right (which
    is what callers really need to bucket "old" vs "new" posts) and
    pins month/day to Jan 1, which is fine for the date-sort and
    the ranked-fallback logic downstream.
    """
    m = _URL_DATE_RE.search(url)
    if m:
        year, month_str, day = m.group(1), m.group(2).lower(), m.group(3)
        month = _MONTH_MAP.get(month_str, month_str.zfill(2))
        if month.isdigit():
            return f"{year}-{month}-{day.zfill(2)}"
    m = _URL_DATE_DASH_RE.search(url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Year-only fallback. Search the whole path so e.g.
    # `/lighton-blogs/2023-the-rise-of-on-prem-llms` matches.
    m = _URL_YEAR_SLUG_RE.search(url)
    if m:
        return f"{m.group(1)}-01-01"
    return ""


def _title_from_url(url: str) -> str:
    from urllib.parse import unquote
    from urllib.parse import urlparse as _up

    path = unquote(_up(url).path).rstrip("/")
    slug = path.rsplit("/", 1)[-1] if "/" in path else path
    # We intentionally KEEP web-page extensions (.html, .php, …) so
    # the cleanup step downstream can use them as a quality signal:
    # a slug that's only `index.html` produces a useless title, and
    # the cleaner drops those docs (and, by extension, any source
    # whose entire sitemap is .html-style URLs).
    title = slug.replace("-", " ").replace("_", " ").strip()
    return title[:1].upper() + title[1:] if title else url
