"""Populate raw social-follower counts on the `users` table.

One entry point: ``populate_social_counts(database_url, user_id)``.

The function probes each social count that is still NULL and writes a
value when the upstream API succeeds. It is idempotent: once a column is
filled, later pipeline runs skip the lookup.

Upstream APIs:
  * Twitter/X — ``api.twitterapi.io/twitter/user/info`` (needs
    ``TWITTERAPIIO_API_KEY``)
  * GitHub   — ``api.github.com/users/{user}`` (optional token via
    ``GITHUB_TOKEN`` / ``GH_TOKEN`` for higher rate limits)
  * Citations — Semantic Scholar ``graph/v1/author/search`` (keyless)
"""

from __future__ import annotations

import os
import re
import urllib.parse

import requests

from sources.sql.users import get_social_counts, set_social_counts

# ── Handle extraction ─────────────────────────────────────────────────


def _handle_from_url(url: str, host_suffixes: tuple[str, ...]) -> str | None:
    """Pull a single-segment handle out of ``https://<host>/<handle>``."""
    if not url:
        return None
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return None
    if not any(p.netloc.lower().endswith(s) for s in host_suffixes):
        return None
    handle = p.path.strip("/").split("/", 1)[0]
    return handle or None


def _twitter_handle(links: dict, sources: dict) -> str | None:
    cfg = sources.get("twitter") if isinstance(sources.get("twitter"), dict) else None
    if cfg and cfg.get("username"):
        return str(cfg["username"]).lstrip("@")
    return _handle_from_url(links.get("twitter", ""), ("x.com", "twitter.com"))


def _github_handle(links: dict, sources: dict) -> str | None:
    gh = sources.get("github")
    if isinstance(gh, list) and gh:
        return str(gh[0])
    if isinstance(gh, dict) and gh.get("user"):
        return str(gh["user"])
    return _handle_from_url(links.get("github", ""), ("github.com",))


# ── Upstream lookups ──────────────────────────────────────────────────


def _twitter_user_payload(handle: str, api_key: str) -> dict | None:
    """Single fetch of a Twitter profile from twitterapi.io.

    Returned dict is the inner profile object (unwrapped from the
    `data` envelope when present). Callers pull whichever field
    they need — follower count, avatar URL, etc. Returns None on
    network failure or non-200.
    """
    if not handle or not api_key:
        return None
    try:
        r = requests.get(
            "https://api.twitterapi.io/twitter/user/info",
            params={"userName": handle},
            headers={"X-API-Key": api_key},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        return data
    return payload if isinstance(payload, dict) else None


def fetch_twitter_followers(handle: str, api_key: str) -> int | None:
    """Fetch follower count from api.twitterapi.io. Returns None on failure."""
    data = _twitter_user_payload(handle, api_key)
    if not data:
        return None
    for key in ("followers", "followers_count", "followersCount"):
        v = data.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def fetch_twitter_avatar(handle: str, api_key: str) -> str | None:
    """Return the profile picture URL for a Twitter handle, or None.

    The current twitterapi.io response surfaces the avatar at the
    `profilePicture` key (camelCase, full-resolution URL). Older
    payload variants used `profile_image_url_https` with a `_normal`
    thumbnail suffix — we keep those as fallbacks so historical
    responses (e.g. cached, replayed) still work, and strip the
    `_normal` suffix on the way out for the original-size image.
    """
    data = _twitter_user_payload(handle, api_key)
    if not data:
        return None
    for key in (
        "profilePicture",
        "profile_image_url_https",
        "profileImageUrlHttps",
        "profile_image_url",
        "profileImageUrl",
    ):
        v = data.get(key)
        if isinstance(v, str) and v:
            # `_normal.jpg` is Twitter's 48x48 thumbnail. Drop the
            # `_normal` infix so we get the full-resolution image.
            return v.replace("_normal.", ".")
    return None


def fetch_github_avatar(handle: str, token: str | None = None) -> str | None:
    """Return a GitHub user's avatar URL, or None.

    Two fast paths:
      • The avatar URL we get from the API is a stable CDN URL,
        ideal because it's already sized and versioned.
      • Failing that, `https://github.com/<handle>.png` 302-redirects
        to the same CDN URL; we use it as a fallback so we still
        return *something* even without a token.

    Returns None when the handle 404s or the request fails.
    """
    if not handle:
        return None
    headers: dict[str, str] = {"User-Agent": "knowledge/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(
            f"https://api.github.com/users/{handle}",
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            url = r.json().get("avatar_url")
            if isinstance(url, str) and url:
                return url
        elif r.status_code == 404:
            return None
    except Exception:
        pass
    # Last-resort permanent redirect — verify the handle exists with
    # a HEAD so we don't hand back a 404-bound URL.
    try:
        r = requests.head(
            f"https://github.com/{handle}.png",
            headers={"User-Agent": "knowledge/1.0"},
            timeout=10,
            allow_redirects=False,
        )
        if r.status_code in (200, 301, 302):
            return f"https://github.com/{handle}.png"
    except Exception:
        pass
    return None


def fetch_github_followers(handle: str, token: str | None = None) -> int | None:
    """Fetch follower count from api.github.com. Returns None on failure."""
    if not handle:
        return None
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        r = requests.get(
            f"https://api.github.com/users/{handle}",
            headers=headers,
            timeout=15,
        )
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception:
        return None
    v = payload.get("followers") if isinstance(payload, dict) else None
    return int(v) if isinstance(v, int) else None


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ'\-]+")


def _fetch_citations_from_scholar_id(scholar_id: str) -> int | None:
    """Scrape the total citation count from a Google Scholar profile.

    Preferred over the Semantic Scholar name-search fallback because a
    scholar_id is a stable unique key — no risk of picking up a
    namesake's stats. Mirrors the logic in `probe_scholar` (Rust):
    the stats table on a profile page renders as a sequence of
    `<td class="gsc_rsb_std">N</td>` cells and the first cell is the
    total citation count.
    """
    if not scholar_id:
        return None
    try:
        r = requests.get(
            f"https://scholar.google.com/citations?user={scholar_id}&hl=en",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; knowledge-popularity/1.0)"},
        )
        if r.status_code != 200:
            return None
        html = r.text
    except Exception:
        return None
    # Bail early when the profile isn't real — `gsc_prf_in` is the
    # <div id="gsc_prf_in"> that wraps the author's name on valid
    # profile pages.
    if "gsc_prf_in" not in html:
        return None
    # The first stats cell is `<td class="gsc_rsb_std">123456</td>`.
    # Splitting on the class name alone is NOT enough — the class is
    # defined earlier in the page's inline CSS, so parts[1] is a giant
    # stylesheet blob, not the cell we want. Match with a regex that
    # pins the `">"` boundary + the numeric payload.
    import re as _re

    m = _re.search(r'gsc_rsb_std"[^>]*>\s*([\d,]+)\s*</td>', html)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def fetch_citations(name: str, scholar_id: str | None = None) -> int | None:
    """Best-effort total citation count.

    Resolution order:
      1. If `scholar_id` is provided, scrape Google Scholar directly
         (authoritative — unique per author).
      2. Otherwise fall back to the Semantic Scholar name-search, which
         is noisier but doesn't require configuration. Tokens-subset
         filtering keeps obvious namesakes out of the top match.
    """
    if scholar_id:
        v = _fetch_citations_from_scholar_id(scholar_id)
        if v is not None:
            return v
    if not name or not _WORD_RE.search(name):
        return None
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/author/search",
            params={"query": name, "limit": 3, "fields": "name,citationCount"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json().get("data") or []
    except Exception:
        return None
    q_tokens = {t.lower() for t in _WORD_RE.findall(name)}
    best: int | None = None
    for item in data:
        nm = (item.get("name") or "").lower()
        if not q_tokens.issubset(set(_WORD_RE.findall(nm))):
            continue
        c = item.get("citationCount")
        if isinstance(c, int) and (best is None or c > best):
            best = c
    return best


# ── Orchestrator ──────────────────────────────────────────────────────


def populate_social_counts(
    database_url: str,
    user_id: int,
    *,
    display_name: str | None = None,
) -> dict:
    """Fill any NULL social count on the user row. Returns the written values.

    Only columns currently NULL are probed. Fetch failures leave the
    column NULL so a later run can retry.
    """
    current = get_social_counts(database_url, user_id)
    links = current["links"]
    sources = current["sources"]

    tw_handle = _twitter_handle(links, sources)
    gh_handle = _github_handle(links, sources)

    twitter_api_key = os.environ.get("TWITTERAPIIO_API_KEY", "")
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    written: dict[str, int | str] = {}

    if current["twitter_followers"] is None and tw_handle:
        v = fetch_twitter_followers(tw_handle, twitter_api_key)
        if v is not None:
            written["twitter_followers"] = v

    if current["github_followers"] is None and gh_handle:
        v = fetch_github_followers(gh_handle, github_token)
        if v is not None:
            written["github_followers"] = v

    if current["citations"] is None:
        scholar_id = ""
        if isinstance(sources.get("scholar"), dict):
            scholar_id = str(sources["scholar"].get("user_id") or "")
        v = fetch_citations(display_name or "", scholar_id=scholar_id)
        if v is not None:
            written["citations"] = v

    # Avatar: GitHub first (stable CDN URL, no API key needed for
    # the .png redirect), Twitter second (twitterapi.io profile
    # image). The frontend already falls back to initials when
    # `avatar` is NULL, so a failed fetch here is harmless — the
    # next pipeline tick will retry.
    if current["avatar"] is None:
        v = None
        if gh_handle:
            v = fetch_github_avatar(gh_handle, github_token)
        if v is None and tw_handle:
            v = fetch_twitter_avatar(tw_handle, twitter_api_key)
        if v is not None:
            written["avatar"] = v

    if written:
        set_social_counts(database_url, user_id, **written)
    return written
