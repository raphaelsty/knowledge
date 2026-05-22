"""Clean titles and summaries for display on document cards.

Two public helpers:
  - ``clean_title(s)`` — strip URLs + reshape slug-style titles
    ("light-river" → "Light River", "Interactive_Tools" → "Interactive Tools",
    "mySuperRepo" → "My Super Repo"). Well-formed prose is left alone so we
    don't title-case actual sentences ("Reducing the memory footprint of …").
  - ``clean_summary(s)`` — strip URLs and normalize whitespace. Never
    changes casing.

The cleaning is intentionally conservative: we only reshape titles that
look slug-ish (no spaces AND carry a hyphen / underscore / camelCase
boundary). Acronyms (e.g. ``HKML``, ``SQL``) and already-mixed-case tokens
are preserved.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
# Trailing "(via …)" or "(via …" (no closing paren) left behind after the
# URL inside got stripped. Covers the legacy Twitter-fetcher summaries.
_TRAILING_VIA_RE = re.compile(r"\s*\(\s*via\b[^)]*\)?\s*$", re.IGNORECASE)
# Notebook / plain-text extensions we want gone from the visible title.
# Code-file extensions like .js / .py are NOT in this list — they're often
# part of the project name ("sigma.js").
_TRAILING_EXT_RE = re.compile(r"\.(ipynb|md|rst|txt)$", re.IGNORECASE)
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")
_SLUG_SEP_RE = re.compile(r"[-_]")


def strip_urls(text: str) -> str:
    """Remove any ``http(s)://…`` or ``www.…`` tokens from a string."""
    if not text:
        return text
    return _URL_RE.sub("", text)


def clean_title(title: str | None) -> str:
    """Return a display-friendly title."""
    if not title:
        return ""
    t = strip_urls(title).strip()
    t = _TRAILING_EXT_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    if _looks_like_slug(t):
        t = _prettify_slug(t)
    return t


def clean_summary(summary: str | None) -> str:
    """Return a display-friendly summary.

    Strips URLs and collapses whitespace, with one carve-out: lines
    that begin with a media marker (`📷 <url>` for photos,
    `🎬 <poster> | <mp4>` for videos) are kept verbatim and on their
    own line. The frontend card renderer in `web/search/page.js`
    parses those exact patterns (`/^📷\\s+(https?:\\/\\/…)/`) to
    render inline tiles, so stripping the URL would silently drop
    every attached photo/video from the card.

    Non-marker lines get the legacy treatment: URLs gone, whitespace
    collapsed. The output uses `\\n` between any surviving marker
    lines so the frontend regex still hits at line start; content
    inside one paragraph stays on a single line.
    """
    if not summary:
        return ""
    pieces: list[str] = []
    for raw in summary.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("📷", "🎬")):
            # Media marker — preserve URLs verbatim; the renderer
            # depends on them.
            pieces.append(line)
        else:
            cleaned = strip_urls(line)
            cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
            if cleaned:
                pieces.append(cleaned)
    s = "\n".join(pieces)
    s = _TRAILING_VIA_RE.sub("", s)
    return s.strip()


# ── Internals ───────────────────────────────────────────────────────────


def _looks_like_slug(s: str) -> bool:
    """True when the string has no spaces but carries slug/camel markers."""
    if not s or " " in s:
        return False
    return bool(_SLUG_SEP_RE.search(s) or _CAMEL_BOUNDARY_RE.search(s))


def _prettify_slug(s: str) -> str:
    """Split a slug-style token and title-case it, preserving acronyms."""
    # camelCase → camel Case
    s = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", s)
    # kebab / snake → spaces
    s = _SLUG_SEP_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()

    words: list[str] = []
    for word in s.split(" "):
        if not word:
            continue
        if len(word) > 1 and word.isupper():
            words.append(word)  # acronym — keep
        elif len(word) > 1 and any(c.isupper() for c in word[1:]):
            words.append(word)  # already mixed — keep (PyTorch, GitHub)
        else:
            words.append(word.capitalize())
    return " ".join(words)
