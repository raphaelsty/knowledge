"""
Wikipedia references extractor.

Pulls the curated outbound links from a person/topic's Wikipedia
page and turns each one into a real document — using the link's
anchor text as the title and the surrounding paragraph or list-item
(plus the nearest section heading) as the summary.

Dating policy
-------------
We do NOT use the article's `Last-Modified` header as the ref date —
Wikipedia articles get edited constantly, so every reference would
end up dated "today" and flood the feed. Instead, for each ref we:

  1. Try to extract a publication year from the citation context
     (e.g. "(2019)", "Retrieved 21 March 2020"). On hit, use
     `YYYY-01-01`.
  2. Fall back to the page's last-modified date minus 5 years so
     refs still appear in the feed but sit below recently-added
     real bookmarks.

Why a richer extractor is needed
--------------------------------
The previous implementation called the MediaWiki `extlinks` API,
which returns raw URLs only — no anchor text, no context. Every
document then ended up with the SAME title (`"Wikipedia: X"`) and
SAME summary (`"Referenced on the Wikipedia page for X"`), which
duplicated across hundreds of links and gave ColBERT nothing
discriminative to embed.

The new flow uses the Wikimedia REST HTML endpoint:

    https://en.wikipedia.org/api/rest_v1/page/html/{title}

which returns the rendered article. For each `<a class="external">`
or `[rel*=mw-external]` link we capture:

  * the link's own anchor text (citation title, link label, …)
  * the closest container (`<li>`, `<p>`, `<dd>`) — the citation
    or paragraph that hosts the link
  * the nearest preceding `<h2>` / `<h3>` / `<h4>` heading

Each link becomes:

  * title   — the cleaned anchor text (or the full citation when
              the anchor is just a quoted publication name)
  * summary — `[Section] anchor — context paragraph` cleaned of
              the citation-number noise MediaWiki sprinkles in
  * tags    — `["wikipedia", "<subject>", "<section>"]`
  * date    — the host article's last-modified date (the Wikimedia
              `Last-Modified` header on the rendered HTML response).
              Individual refs don't carry their own publication date,
              but the page they were curated into does, and that's a
              meaningful "when this entered the library" signal.
"""

import datetime
import re
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

__all__ = ["References"]

_HTML_API = "https://en.wikipedia.org/api/rest_v1/page/html/{}"
_USER_AGENT = "Knowledge/1.0 (+https://knowledge-web.org) BeautifulSoup"

# Domains we don't want to surface — they're either internal
# Wikimedia infrastructure, archive snapshots of links we already
# capture, or DOIs (which deserve their own dedicated source if we
# want them).
_SKIP_DOMAINS = {
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "wikiquote.org",
    "wikisource.org",
    "commons.wikimedia.org",
    "web.archive.org",
    "archive.org",
    "doi.org",
    "dx.doi.org",
}

# Recognise an arxiv `abs` or `pdf` URL and capture the paper id
# (digits.digits, possibly with a version suffix). Used to derive a
# meaningful placeholder title like "arXiv:1606.06565" when the
# citation context offered nothing usable.
_ARXIV_ID_RE = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def _url_placeholder_title(url: str) -> str:
    """
    Compose an honest placeholder title from a URL when neither the
    anchor nor the surrounding citation context gave us anything
    descriptive. The previous fallback was a flat `"Wikipedia: X"`
    using the host article's subject, which read misleadingly when
    the linked resource was a paper that wasn't actually about — or
    by — that subject. Now we lean on the URL itself:

      * arXiv `abs`/`pdf` URLs → `arXiv:<id>`
      * other URLs → `<host> — <path-tail>` (tail trimmed of file
        extensions and common slugs).

    Returns `""` when the URL is empty or we can't parse it.
    """
    if not url:
        return ""
    m = _ARXIV_ID_RE.search(url)
    if m:
        return f"arXiv:{m.group(1)}"
    try:
        parsed = urllib.parse.urlparse(url)
    except (ValueError, TypeError):
        return ""
    # `lstrip` would strip ANY of the characters in "www." (so a host
    # like "wn.com" would lose its leading "w") — use a single
    # explicit prefix strip instead.
    host = (parsed.hostname or "").strip()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    path_tail = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    # Strip a trailing file extension and replace separators with
    # spaces so the placeholder reads like a label, not a slug.
    path_tail = re.sub(r"\.(html?|pdf|aspx?|php)$", "", path_tail, flags=re.IGNORECASE)
    path_tail = path_tail.replace("-", " ").replace("_", " ").strip()
    if path_tail:
        return f"{host} — {path_tail}"
    return host


# Mediawiki sprinkles "[edit]" buttons into rendered headings.
_EDIT_RE = re.compile(r"\[\s*edit\s*\]\s*$", re.IGNORECASE)
# References blocks are prefixed with backref glyphs like "1 2"
# (a b c d ↑ ↑) before the citation text — cosmetic noise we strip.
# Three patterns, all anchored to start of string:
#   1. Leading digit groups: "1 ", "1 2 ", "12 ".
#   2. Letter backrefs: "a b c ", "ab ^ ".
#   3. The up-arrow MediaWiki uses for single-use refs ("↑ Pióro, …").
_REF_NUMBERING_RE = re.compile(
    r"^(?:\d+\s+){1,5}|^[a-z\s]{1,12}\^\s+|^[↑↑]+\s*",
    re.IGNORECASE,
)
# Citation tails Wikipedia bakes in. Each fragment is anchored to the
# end of the text and stripped iteratively so a string ending with
# multiple of them (e.g. "Archived... Retrieved...") gets fully cleaned.
_REF_TAIL_REGEXES = (
    re.compile(r"\s*Retrieved\s+\d{1,2}\s+\w+\s+\d{4}\.?\s*$"),
    re.compile(r"\s*Archived\s+from\s+the\s+original.*?\.?\s*$"),
    re.compile(r"\s*\.\s*$"),
)


def _fetch_html(title: str) -> tuple[str, str]:
    """Fetch the rendered HTML + `YYYY-MM-DD` last-modified date for a
    Wikipedia page. Returns `("", "")` on any error.

    The Wikimedia REST HTML endpoint returns a `Last-Modified` HTTP
    header with the article's most-recent revision timestamp. We use
    that as the date for every reference extracted from the page —
    Wikipedia refs don't carry their own publication date, but the
    article they were curated into does, and "when this list was last
    edited" is the most accurate signal we can attach to each ref.
    """
    url = _HTML_API.format(urllib.parse.quote(title, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            last_modified = _parse_http_date(resp.headers.get("Last-Modified"))
            return body, last_modified
    except Exception as e:
        print(f"    Wikipedia HTML error for {title}: {e}")
        return "", ""


def _parse_http_date(raw: str | None) -> str:
    """Parse an RFC 7231 IMF-fixdate (`Wed, 21 Oct 2015 07:28:00 GMT`)
    into a `YYYY-MM-DD` string. Returns "" if `raw` is missing or
    malformed."""
    if not raw:
        return ""
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
        return dt.date().isoformat()
    except Exception:
        return ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_section_heading(text: str) -> str:
    return _normalize(_EDIT_RE.sub("", text or ""))


def _strip_ref_noise(text: str) -> str:
    """Remove backref numbering / '↑' glyphs / citation tails."""
    text = _normalize(text)
    text = _REF_NUMBERING_RE.sub("", text).strip()
    # Iterate so chains of "Archived...Retrieved..." get peeled off.
    for _ in range(3):
        before = text
        for rx in _REF_TAIL_REGEXES:
            text = rx.sub("", text).strip()
        if text == before:
            break
    # Wikipedia wraps citation titles in straight/curly double
    # quotes (e.g. `"Title" . Publisher`). Once the surrounding
    # citation has been peeled off these quotes only ever read as
    # noise — drop them globally. Apostrophes are kept because
    # they appear in genuine prose.
    text = re.sub(r'["“”]', "", text)
    # Collapse double-spacing introduced by the quote removal and
    # strip any orphan punctuation now leading or trailing.
    text = _normalize(text).strip(" .,;:")
    return text


def _clean_anchor(text: str) -> str:
    text = _normalize(text)
    # Strip surrounding straight or curly quotes that Wikipedia
    # uses for publication titles, plus any leading "↑" backref
    # glyph the MediaWiki references renderer prepends.
    text = text.strip("\"'“”‘’")
    text = re.sub(r"^[↑\s]+", "", text)
    return text


def _truncate_at_sentence(text: str, budget: int) -> str:
    """Cut at a sentence boundary near `budget`, ellipsize otherwise."""
    text = _normalize(text)
    if len(text) <= budget:
        return text
    cut = text[:budget]
    for stop in (". ", "! ", "? "):
        i = cut.rfind(stop)
        if i >= budget * 0.5:
            return cut[: i + 1].rstrip()
    return cut.rstrip(" ,;:-—") + "…"


def _is_external(a) -> bool:
    """True for `<a>` tags Wikipedia marks as outbound."""
    cls = a.get("class") or []
    if isinstance(cls, str):
        cls = [cls]
    if "external" in cls:
        return True
    rel = a.get("rel") or []
    if isinstance(rel, str):
        rel = [rel]
    return any("external" in r for r in rel) or any("mw-external" in r for r in rel)


def _container(a):
    """Walk up to the nearest paragraph / list-item / definition."""
    node = a.parent
    while node is not None and getattr(node, "name", None) not in (
        "li",
        "p",
        "dd",
        "dt",
        "td",
    ):
        node = node.parent
    return node


def _section_of(node):
    """Nearest preceding h2/h3/h4 heading text."""
    cur = node
    while cur is not None:
        prev = cur.find_previous(["h2", "h3", "h4"])
        if prev is None:
            return ""
        return _strip_section_heading(prev.get_text(" ", strip=True))
    return ""


def _container_text(container, link_anchor: str) -> str:
    """
    Pull the container's text minus the verbose 'Retrieved YYYY'
    tail and the citation-number prefix that Wikipedia bakes into
    rendered references. If the resulting context is just the
    anchor itself (link-only line), return ''.
    """
    if container is None:
        return ""
    raw = _strip_ref_noise(container.get_text(" ", strip=True))
    # Drop surrounding quotes Wikipedia wraps publication names in.
    raw = raw.strip("\"'“”‘’")
    if not raw:
        return ""
    # If the container is *just* the anchor text we already used as
    # the title, keep the anchor as the summary so the row isn't
    # empty — but signal that there's nothing more to say.
    if _clean_anchor(raw).lower() == _clean_anchor(link_anchor).lower():
        return ""
    return raw


# Years we'll accept from a citation context. Capped to "current year"
# at runtime so a typo'd 2099 doesn't masquerade as a recent ref.
_YEAR_RE = re.compile(r"(?:^|[^0-9])((?:19|20)\d{2})(?:[^0-9]|$)")
# Default backshift when no year can be extracted: how many years to
# subtract from the article's last-modified date so wiki refs don't
# dominate "today" in the feed.
_WIKI_DATE_BACKSHIFT_YEARS = 5


def _extract_ref_year(*texts: str) -> int | None:
    """Return the most plausible publication year from citation text.

    Picks the *latest* 4-digit 19xx/20xx year that's ≤ current year —
    citation templates often include the publisher's footer copyright
    (e.g. "© 2026") alongside the paper's year, so we prefer the
    smaller of the two by anchoring on whichever year appears *most
    frequently*. Falls back to the first match when frequencies tie.
    """
    this_year = datetime.date.today().year
    counts: dict[int, int] = {}
    first: dict[int, int] = {}
    for idx, text in enumerate(texts):
        if not text:
            continue
        for m in _YEAR_RE.finditer(text):
            y = int(m.group(1))
            if 1900 <= y <= this_year:
                counts[y] = counts.get(y, 0) + 1
                first.setdefault(y, idx)
    if not counts:
        return None
    # Most-frequent wins; on ties pick the earliest text occurrence.
    return sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0][0]


def _ref_date(container_text: str, anchor: str, page_date: str) -> str:
    """Pick a date for a Wikipedia reference.

    Order of preference:
      1. Year extracted from the citation container / anchor.
      2. Page last-modified date minus `_WIKI_DATE_BACKSHIFT_YEARS`
         (so wiki refs sit below recently-added real bookmarks).
      3. Empty string when even the page date is missing.
    """
    year = _extract_ref_year(container_text, anchor)
    if year is not None:
        return f"{year}-01-01"
    if page_date:
        try:
            dt = datetime.date.fromisoformat(page_date)
            return dt.replace(year=dt.year - _WIKI_DATE_BACKSHIFT_YEARS).isoformat()
        except ValueError:
            pass
    return ""


# Wikipedia citation templates wrap the paper / article title in
# straight or curly double quotes, e.g.
#     Authors (Year). "Paper Title". Journal, 1(2): 3.
# We look at the RAW container text (before `_strip_ref_noise`'s
# global quote-strip) so we can lift the title out before it's gone.
# Match a non-empty quoted phrase of 1–400 chars, requiring at least
# two whitespace-separated tokens so we don't grab one-word labels
# like "PDF" or "Preprint".
_QUOTED_TITLE_RE = re.compile(r'["“]([^"”\n]{4,400})["”]')


def _extract_quoted_title(container) -> str:
    """
    Lift the first plausible paper title out of a citation container.

    Wikipedia's cs1/cs2 citation templates render the title between
    quotes, so the first substantive quoted phrase in the container
    is the title. We work off the raw text (not `_strip_ref_noise`'d)
    because the noise stripper drops every quote globally.
    """
    if container is None:
        return ""
    raw = _normalize(container.get_text(" ", strip=True))
    for m in _QUOTED_TITLE_RE.finditer(raw):
        cand = _normalize(m.group(1)).strip(" .,;:")
        # Require ≥ 2 tokens so we skip terse quoted labels.
        if len(cand.split()) >= 2:
            return cand
    return ""


class References:
    """
    Extract external links from Wikipedia pages.

    Each link becomes a separate document with:
      * `title`   — the link's anchor text (the citation / source name)
      * `summary` — surrounding paragraph or list-item, prefixed by
                    the section heading the link sits in
      * `tags`    — `["wikipedia", "<subject>", "<section>"]`
    """

    SUMMARY_BUDGET = 700

    def __init__(self, pages: list[str]):
        """
        Parameters
        ----------
        pages : list[str]
            Wikipedia page titles (e.g. ["Demis_Hassabis", "AlphaFold"]).
            Underscores and spaces both work.
        """
        self.pages = pages

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}

        for page_title in self.pages:
            print(f"    Fetching Wikipedia refs: {page_title}")
            subject = page_title.replace("_", " ")
            html, page_date = _fetch_html(page_title)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            anchors = [a for a in soup.find_all("a", href=True) if _is_external(a)]

            kept = 0
            for a in anchors:
                href = a.get("href", "").strip()
                if not href.startswith(("http://", "https://")):
                    continue
                # Domain blacklist (mirrored from the legacy version).
                parsed = urllib.parse.urlparse(href)
                domain = parsed.netloc.lower()
                if any(domain.endswith(d) for d in _SKIP_DOMAINS):
                    continue
                # arXiv category/archive listings (e.g.
                # `arxiv.org/archive/cs.CY`, `arxiv.org/list/cs.LG/...`,
                # `arxiv.org/category/...`) get rendered as external
                # links right next to the real paper citation. They
                # aren't documents — they're navigation. Drop them so
                # we only keep `/abs/...` / `/pdf/...` paper URLs.
                if domain.endswith("arxiv.org") and not parsed.path.startswith(("/abs/", "/pdf/")):
                    continue
                if existing_urls and href in existing_urls:
                    continue
                if href in data:
                    continue

                anchor_text = _clean_anchor(a.get_text(" ", strip=True))
                container = _container(a)
                section = _section_of(container or a)
                context = _container_text(container, anchor_text)
                quoted_title = _extract_quoted_title(container)

                title = self._compose_title(anchor_text, quoted_title, context, subject, href)
                summary = self._compose_summary(section, anchor_text, context, subject)
                tags = ["wikipedia", subject.lower()]
                if section:
                    tags.append(section.lower())

                ref_date = _ref_date(context, anchor_text, page_date)

                data[href] = {
                    "title": title,
                    "summary": summary,
                    # Prefer a year extracted from the citation
                    # context; otherwise the page's last-mod date
                    # minus a few years so wiki refs don't all flood
                    # "today" in the feed.
                    "date": ref_date,
                    "tags": tags,
                }
                kept += 1

            print(f"    {page_title}: {kept} reference(s) kept of {len(anchors)}")

        print(f"    Total: {len(data)} Wikipedia references")
        return data

    # ────────────────────────────────────────────────────────────────
    # Title / summary composition
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compose_title(anchor: str, quoted_title: str, context: str, subject: str, url: str = "") -> str:
        """
        Pick the most descriptive title we have:
          1. The anchor text when it's substantive (≥ 2 words and not
             just the subject's name).
          2. The quoted-title we lifted from the citation template
             (e.g. `"MoE-Mamba: Efficient Selective…"`). Beats the
             context fallback because on a citation list the first
             12 context words are the author block, not the title.
          3. Fall back to the first 12 words of the surrounding
             context if the anchor is just a number / single token.
          4. URL-derived placeholder so the row reads honestly when
             the citation contained no extractable label (arxiv IDs
             surface as `arXiv:1606.06565`; other hosts fall back to
             "<host> — <path-tail>"). This used to be a flat
             `"Wikipedia: <subject>"`, which read misleadingly when
             the linked resource was, say, an arxiv paper that
             wasn't actually authored by the Wikipedia page's
             subject.
          5. Last resort (URL also empty): `Wikipedia: <subject>`.
        """
        if anchor and len(anchor.split()) >= 2 and anchor.lower() != subject.lower():
            return anchor[:140]
        if quoted_title:
            return quoted_title[:140]
        if context:
            words = context.split()
            head = " ".join(words[:12])
            if head:
                return head[:140]
        if anchor:
            return anchor[:140]
        url_title = _url_placeholder_title(url)
        if url_title:
            return url_title[:140]
        return f"Wikipedia: {subject}"

    @classmethod
    def _compose_summary(cls, section: str, anchor: str, context: str, subject: str) -> str:
        """
        Compose a discriminative summary. Layout:

          [{Section} — {Subject}] {context}

        — section + subject as a small breadcrumb so a single
        ColBERT query like "AlphaFold references" can match docs
        from the AlphaFold page even when the URL host alone has no
        signal. The anchor text is included only when the context
        doesn't already contain it (avoids "X — X. X is …").
        """
        breadcrumb_parts = []
        if section:
            breadcrumb_parts.append(section)
        breadcrumb_parts.append(subject)
        breadcrumb = " — ".join(breadcrumb_parts)

        body = context or ""
        if anchor and anchor.lower() not in (body or "").lower():
            body = f"{anchor}. {body}".strip().rstrip(".") + "."
        if not body:
            # Genuinely nothing to add: keep the breadcrumb so each
            # doc still has something distinctive vs. its siblings.
            body = anchor or subject

        composed = f"[{breadcrumb}] {body}".strip()
        return _truncate_at_sentence(composed, cls.SUMMARY_BUDGET)
