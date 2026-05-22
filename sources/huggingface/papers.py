"""Activity-based HuggingFace fetcher: papers + articles a user has
upvoted, submitted, or authored on HuggingFace, parsed from the
embedded ``UserProfile`` JSON island on the
``/<username>/activity/<kind>`` pages.

Three activity feeds, same JSON shape:

  * ``upvotes``  — papers + articles the user has upvoted
  * ``papers``   — papers the user has submitted or authored on HF
  * ``articles`` — blog posts the user has authored on HF

Why scrape the HTML island and not an API endpoint
--------------------------------------------------
HuggingFace does NOT expose a public REST endpoint for a user's
upvoted papers (we probed ``/api/users/<u>/upvotes``,
``/api/users/<u>/papers``, ``/api/papers?upvoted_by=<u>`` — all 404).
The ``/<u>/activity/<kind>`` page server-renders the recent ~20
activities into ``<div data-target="UserProfile" data-props="…">``
where ``data-props`` is HTML-escaped JSON. We extract that, parse it,
and walk the ``activities`` array.

Pagination caveat
-----------------
HF doesn't paginate the activity HTML server-side (``?p=N``,
``?skip=N``, ``?cursor=...`` all return the same first 20 items —
the on-page infinite-scroll fires an authenticated XHR we can't
reach unauthenticated). So a single fetch yields the user's 20 most
recent paper-related activities. Subsequent pipeline runs pick up
whatever's accumulated since the last run. Heavy-upvote accounts
get a slow incremental backfill rather than a one-shot dump.

Output shape
------------
Each emitted doc is keyed on a stable canonical URL:

  * **Papers** → ``https://arxiv.org/abs/<id>`` (same canonical as
    the dedicated arXiv fetcher, so the canonical_url generated
    column collapses the two paths to one card).
  * **Articles** → ``https://huggingface.co/blog/<author>/<slug>``.

For papers the abstract is hydrated via ``/api/papers/{id}`` (the
activity payload only carries title + thumbnail + upvotes).
"""

from __future__ import annotations

import html as _html
import json as _json
import re

import requests

from .likes import _request_with_retry

__all__ = ["Activity"]

_BASE = "https://huggingface.co"
_ARXIV_URL = "https://arxiv.org/abs/{}"
_ISLAND_RE = re.compile(r'data-target="UserProfile" data-props="([^"]+)"')
_VALID_KINDS = frozenset({"upvotes", "papers", "articles"})


class Activity:
    """Fetch a user's paper-related HuggingFace activity feed.

    Parameters
    ----------
    username : str
        Public HuggingFace username. The activity pages are public so
        no token is required.
    kinds : list[str], optional
        Subset of ``{"upvotes", "papers", "articles"}``. Defaults to
        all three. Order matters only when two kinds emit the same
        canonical URL (earlier wins) — typically ``upvotes`` first
        because that's what the user is most explicitly opting into.

    Example
    -------
    >>> docs = Activity(username="clem")()
    >>> for url in docs:
    ...     print(url)
    """

    # Per-paper abstract budget. The full HF API ``summary`` field is
    # the verbatim arXiv abstract (300–2000 chars). Most cards never
    # need more than ~500; the indexer truncates at 200 anyway.
    _ABSTRACT_BUDGET = 600

    def __init__(self, username: str, kinds: list[str] | None = None):
        if not username:
            raise ValueError("Activity requires a username")
        self.username = username
        if kinds is None:
            self.kinds = ["upvotes", "papers", "articles"]
        else:
            invalid = set(kinds) - _VALID_KINDS
            if invalid:
                raise ValueError(f"unknown activity kind(s): {sorted(invalid)}")
            self.kinds = list(kinds)

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        existing = existing_urls or set()
        for kind in self.kinds:
            try:
                self._fetch_kind(out, kind, existing)
            except Exception as exc:
                # Best-effort: a single failing kind shouldn't tank
                # the other two (or the wider pipeline).
                print(f"    HF activity {kind} for {self.username} failed: {exc}")
        print(f"    HF activity for {self.username}: {len(out)} doc(s)")
        return out

    # ────────────────────────────────────────────────────────────────
    # Per-kind page fetch + parse
    # ────────────────────────────────────────────────────────────────

    def _fetch_kind(self, out: dict, kind: str, existing: set[str]) -> None:
        url = f"{_BASE}/{self.username}/activity/{kind}"
        resp = _request_with_retry(
            requests.get,
            url,
            label=f"activity/{kind}({self.username})",
            headers={"User-Agent": "Knowledge/1.0"},
            timeout=20,
        )
        if resp is None or resp.status_code >= 400:
            return
        m = _ISLAND_RE.search(resp.text)
        if not m:
            return
        try:
            payload = _json.loads(_html.unescape(m.group(1)))
        except Exception:
            return
        for activity in payload.get("activities") or []:
            target_type = activity.get("targetType")
            if target_type == "paper":
                self._emit_paper(out, activity, existing)
            elif target_type == "article":
                self._emit_article(out, activity, existing)
            # silently skip targetType in {changelog, collection, ...}

    # ────────────────────────────────────────────────────────────────
    # Per-target emission
    # ────────────────────────────────────────────────────────────────

    def _emit_paper(self, out: dict, activity: dict, existing: set[str]) -> None:
        """Emit one paper doc, canonical URL = arXiv abs/.

        The activity payload only carries title + upvotes + publish
        date; the actual abstract lives on ``/api/papers/{id}``. We
        fetch it once per paper since the abstract is what makes the
        doc searchable in ColBERT — without it the card just shows
        the title and the indexer has nothing distinctive to embed.
        """
        target = activity.get("target") or {}
        paper_id = (target.get("id") or "").strip()
        if not paper_id:
            return
        url = _ARXIV_URL.format(paper_id)
        if url in existing or url in out:
            return
        title = (target.get("title") or "").strip()
        date = (target.get("publishedAt") or "")[:10]
        abstract = self._fetch_paper_summary(paper_id) or ""
        abstract = abstract[: self._ABSTRACT_BUDGET]
        # The activity HF URL is preserved as the source_url so the
        # card can show a small "via HF" attribution chip and the
        # user can click through to the HF paper page.
        hf_url = f"{_BASE}{activity.get('targetUrl', '')}" if activity.get("targetUrl") else None
        out[url] = {
            "title": title,
            "summary": abstract,
            "date": date,
            "tags": ["arxiv", "huggingface"],
            "source": "arxiv",
            "source_url": hf_url,
        }

    def _fetch_paper_summary(self, paper_id: str) -> str | None:
        """Hit ``/api/papers/{id}`` for the abstract.

        Falls back to ``None`` on any error so the doc still emits
        with title-only — the row will get a richer summary later if
        the dedicated arXiv fetcher (or its Semantic Scholar sidecar)
        picks it up under the same canonical URL.
        """
        url = f"{_BASE}/api/papers/{paper_id}"
        resp = _request_with_retry(
            requests.get,
            url,
            label=f"papers/{paper_id}",
            headers={"User-Agent": "Knowledge/1.0"},
            timeout=15,
        )
        if resp is None or resp.status_code >= 400:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        # Prefer the human-readable abstract; fall back to the LLM
        # TL;DR (``ai_summary``) when the abstract is missing.
        return (data.get("summary") or data.get("ai_summary") or "").strip() or None

    def _emit_article(self, out: dict, activity: dict, existing: set[str]) -> None:
        """Emit one article doc, canonical URL = HF blog URL.

        We don't hit a second endpoint for the article body — HF's
        ``/api/blog/<author>/<slug>`` returns 404 and the og:description
        on the page itself is generic boilerplate. We surface the
        title + author block from the activity payload directly; if
        the user wants the full article they click through.
        """
        target = activity.get("target") or {}
        target_url = (activity.get("targetUrl") or "").strip()
        if not target_url:
            return
        url = f"{_BASE}{target_url}"
        if url in existing or url in out:
            return
        title = (target.get("title") or "").strip()
        date = (target.get("publishedAt") or "")[:10]
        # Compose a one-line summary from the author list so the
        # card has SOMETHING below the title to differentiate it from
        # other articles by the same person.
        authors = [(a.get("fullname") or a.get("name") or "").strip() for a in (target.get("authorsData") or [])]
        authors = [a for a in authors if a]
        if authors:
            summary = "HuggingFace article by " + ", ".join(authors[:3])
            if len(authors) > 3:
                summary += f" (+{len(authors) - 3})"
        else:
            summary = "HuggingFace article"
        out[url] = {
            "title": title,
            "summary": summary,
            "date": date,
            "tags": ["huggingface", "article"],
            "source": "huggingface",
        }
