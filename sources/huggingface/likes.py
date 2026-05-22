"""
HuggingFace module for extracting liked models, datasets, and spaces.

Fetches liked items from a HuggingFace user's profile, then builds a
descriptive document per repo. Metadata comes from the Hub API
(`HfApi.repo_info`) — the README is only parsed when the API has
nothing useful to offer.

Why a richer extractor is needed
--------------------------------
The previous implementation stripped YAML frontmatter and kept the
first 50 tokens of the rest. For most model READMEs that means a
boilerplate header (`Model Card for X`, badges, base-model links)
because authors put their actual description further down the page.
Many cards never put any prose at the top at all — they link to a
project site instead. The result was a near-identical summary across
hundreds of liked repos.

The new flow:
  1. Pull repo metadata via the Hub API (cardData = parsed YAML,
     tags, pipeline_tag, library_name, downloads, last_modified).
  2. Build a structured "fact sheet" line — pipeline / library /
     license / base model — that's already useful even with no prose.
  3. Append the first *substantive* prose paragraph from the README
     (skip badges, images, HTML, code fences, blockquotes, and
     headings). Capped to a sentence-aware budget.
  4. Surface the real `last_modified` timestamp instead of "today".
  5. Promote pipeline_tag / library_name into the document tags so
     ColBERT can match queries like "text-generation" or "diffusers".
"""

import datetime
import os
import re
import threading
import time

import requests
import trafilatura
import yaml
from huggingface_hub import HfApi

__all__ = ["Likes"]


# ──────────────────────────────────────────────────────────────────────────
# Rate-limit handling
# ──────────────────────────────────────────────────────────────────────────
#
# HuggingFace caps unauthenticated `huggingface.co/api/*` traffic at a
# few hundred requests per minute. Power-user accounts (akhaliq with
# 2.6k likes ⇒ ~2 requests per repo for repo_info + README) blow past
# that and start receiving 429s for everyone else in the same run.
#
# We mitigate at two layers:
#
# 1. **Process-wide token-bucket pacer** (`_acquire_slot`). Every
#    outbound HTTP request through this module sleeps just enough to
#    keep the rolling window under `_MAX_RPS`. Conservative default
#    of 8 req/s = 480 req/min, well below the public ceiling and slow
#    enough to coexist with the rest of the pipeline. Override via
#    `HF_MAX_RPS` env var for users who hold a token.
#
# 2. **Exponential-backoff retry on 429** (`_request_with_retry`).
#    Three attempts, doubling delay each time, with full jitter. If
#    HuggingFace returns a `Retry-After` header we honour it; else we
#    fall back to the next backoff slot. After all retries the caller
#    sees an empty result and the document gets a fact-sheet-only
#    summary (no README) — a partial degradation rather than a crash.

_MAX_RPS = float(os.environ.get("HF_MAX_RPS", "8"))
_BACKOFF_BASE_SEC = float(os.environ.get("HF_BACKOFF_BASE", "1.5"))
_BACKOFF_MAX_RETRIES = int(os.environ.get("HF_BACKOFF_RETRIES", "3"))

_pacer_lock = threading.Lock()
_pacer_next_slot = 0.0


def _acquire_slot() -> None:
    """Sleep until the next pacer slot. Thread-safe."""
    global _pacer_next_slot
    interval = 1.0 / _MAX_RPS if _MAX_RPS > 0 else 0.0
    with _pacer_lock:
        now = time.monotonic()
        wait_until = max(now, _pacer_next_slot)
        _pacer_next_slot = wait_until + interval
        sleep_for = wait_until - now
    if sleep_for > 0:
        time.sleep(sleep_for)


def _retry_after_seconds(resp) -> float | None:
    """Parse a `Retry-After` header (seconds or HTTP-date)."""
    if resp is None:
        return None
    val = getattr(resp, "headers", {}).get("Retry-After") if hasattr(resp, "headers") else None
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return None


def _request_with_retry(method, url, *, label, **kwargs):
    """
    Run a `requests.request` with global pacing + 429 retry.

    Returns a `Response` on success (any 2xx-4xx that isn't 429), or
    `None` after exhausting retries. Pure I/O — callers parse the
    body themselves.
    """
    last_resp = None
    for attempt in range(_BACKOFF_MAX_RETRIES + 1):
        _acquire_slot()
        try:
            resp = method(url, **kwargs)
        except requests.RequestException as e:
            print(f"    HF {label} network error: {e}")
            return None
        if resp.status_code != 429:
            return resp
        last_resp = resp
        if attempt == _BACKOFF_MAX_RETRIES:
            break
        # Honour Retry-After when present; otherwise back off exponentially
        # with full jitter so concurrent runs don't synchronise their next
        # attempt and re-trigger the limiter.
        wait = _retry_after_seconds(resp) or (_BACKOFF_BASE_SEC * (2**attempt) + (0.25 * attempt))
        print(f"    HF {label} 429 (attempt {attempt + 1}/{_BACKOFF_MAX_RETRIES}); sleeping {wait:.1f}s")
        time.sleep(wait)
    return last_resp


# Badge/image lines and other obvious noise that prefixes most cards.
_BADGE_LINE_RE = re.compile(
    r"""^\s*(
        \[!\[                # markdown image-link badge: [![alt](img)](href)
        | !\[                # markdown image: ![alt](src)
        | <[a-z!/]            # raw HTML / comment
        | https?://           # bare URL
        | <!--                # HTML comment
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Boilerplate-only headings authors stamp on otherwise-empty cards.
_BOILERPLATE_HEADING_RE = re.compile(
    r"^model\s+card(\s+for\s+.+)?$|^dataset\s+card(\s+for\s+.+)?$|^card\s+for\s+.+$",
    re.IGNORECASE,
)

# Inline markdown we strip from prose for the summary text.
_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_INLINE_HTML_RE = re.compile(r"<[^>]+>")


class Likes:
    """
    Extract knowledge from HuggingFace liked repositories.

    Fetches liked models, datasets, and spaces from a user's HuggingFace
    profile, then builds a per-repo document by combining Hub-API
    metadata with a cleaned first-paragraph excerpt of the README.

    Parameters
    ----------
    token : str, optional
        HuggingFace User Access Token. If not provided, relies on
        local authentication via `huggingface-cli login`.

    Example
    -------
    >>> from sources import huggingface
    >>>
    >>> hf = huggingface.HuggingFace(token="hf_xxxxx")
    >>> documents = hf()
    >>>
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}: {doc['tags']}")
    """

    # Character budget for the user-facing summary. Sentence-aware
    # truncation below — we cut at the last sentence boundary that
    # falls within budget rather than mid-word. The pipeline's
    # `summary[:200]` slice in `client.py:1208` is what feeds the
    # ColBERT embedding; the rest of the budget is purely for the
    # card UI, so two-three sentences of context is the sweet spot.
    SUMMARY_BUDGET = 700

    def __init__(self, token: str = None, username: str = None):
        """
        Two paths, same per-repo enrichment:

        * `token` set → call `HfApi.list_liked_repos()` to get the
          *authenticated* user's likes (private likes included).
        * `username` set (no token, or in addition to it) → call the
          public `GET /api/users/{u}/likes` endpoint to list any
          public profile's likes.

        Both paths converge on `_process_entry`, which fetches
        repo_info + README and builds the rich title/tags/summary.
        """
        self.token = token
        self.username = username
        self.api = HfApi(token=self.token)

    # ──────────────────────────────────────────────────────────────────
    # Top-level entry
    # ──────────────────────────────────────────────────────────────────

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}

        def is_known(url: str) -> bool:
            return existing_urls is not None and url in existing_urls

        # Public-username path: list likes via the unauthenticated
        # endpoint, dispatch each one through `_process_entry`.
        if self.username and not self.token:
            try:
                likes = self._fetch_public_likes(self.username)
            except Exception as e:
                print(f"    HuggingFace API error: {e}")
                return data
            kept = 0
            for item in likes:
                repo = item.get("repo", {})
                repo_id = repo.get("name", "")
                kind = repo.get("type", "model")  # "model" | "dataset" | "space"
                if not repo_id:
                    continue
                url = self._url_for(repo_id, kind)
                if is_known(url):
                    continue
                # `GET /api/users/{u}/likes` returns `createdAt` per
                # like — the moment the user clicked the heart. That
                # beats the repo's last-modified for "when did this
                # enter MY library", so we surface it as the doc
                # date.
                liked_at = item.get("createdAt") or item.get("likedAt") or ""
                try:
                    self._process_entry(data, url, repo_id, kind, liked_at=liked_at)
                    kept += 1
                except Exception as e:
                    print(f"    HuggingFace {kind} {repo_id} failed: {e}")
            print(f"    {kept} new liked repos for {self.username} (of {len(likes)} total)")
            return data

        # Authenticated path (legacy default).
        try:
            likes = self.api.list_liked_repos()
        except Exception as e:
            print(f"Error fetching likes: {e}")
            return data

        for kind, attr in (
            ("model", "models"),
            ("dataset", "datasets"),
            ("space", "spaces"),
        ):
            for entry in getattr(likes, attr, []) or []:
                repo_id = entry.repo_id if hasattr(entry, "repo_id") else str(entry)
                url = self._url_for(repo_id, kind)
                if is_known(url):
                    continue
                self._process_entry(data, url, repo_id, kind)

        return data

    @staticmethod
    def _url_for(repo_id: str, kind: str) -> str:
        if kind == "dataset":
            return f"https://huggingface.co/datasets/{repo_id}"
        if kind == "space":
            return f"https://huggingface.co/spaces/{repo_id}"
        return f"https://huggingface.co/{repo_id}"

    @staticmethod
    def _fetch_public_likes(username: str) -> list[dict]:
        """`GET /api/users/{u}/likes` — public, no auth, paced + retried."""
        import urllib.parse

        url = f"https://huggingface.co/api/users/{urllib.parse.quote(username)}/likes"
        resp = _request_with_retry(
            requests.get,
            url,
            label=f"likes({username})",
            headers={"User-Agent": "Knowledge/1.0"},
            timeout=20,
        )
        if resp is None or resp.status_code >= 400:
            code = resp.status_code if resp is not None else "ERR"
            raise RuntimeError(f"HTTP {code}")
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    # ──────────────────────────────────────────────────────────────────
    # Per-entry extraction
    # ──────────────────────────────────────────────────────────────────

    def _process_entry(
        self,
        data: dict,
        url: str,
        repo_id: str,
        kind: str,  # "model" | "dataset" | "space"
        liked_at: str = "",
    ) -> None:
        print(f"Processing {kind}: {repo_id}")

        info = self._fetch_repo_info(repo_id, kind)
        card_data = self._card_data(info)
        readme = self._fetch_readme(repo_id, kind, info)

        # Title — the bare repo name (org/name reads weird in titles;
        # the org sits in the description and the URL).
        repo_short = repo_id.split("/", 1)[-1]
        title = f"HuggingFace {kind}: {repo_short}"

        summary = self._build_summary(repo_id, kind, info, card_data, readme)
        tags = self._build_tags(kind, info, card_data)
        date = self._extract_date(info, liked_at=liked_at)

        data[url] = {
            "title": title,
            "tags": tags,
            "summary": summary,
            "date": date,
        }

    # ──────────────────────────────────────────────────────────────────
    # Hub API helpers
    # ──────────────────────────────────────────────────────────────────

    def _fetch_repo_info(self, repo_id: str, kind: str):
        """
        Fetch the public Hub API JSON for a repo (paced + 429-retried).

        Returns the parsed dict directly — the Python pipeline used to
        go through `HfApi.repo_info` here, but `huggingface_hub` does
        its own request and we'd lose the global pacer. Talking to the
        same `/api/{models|datasets|spaces}/{id}?full=true` endpoint
        ourselves keeps every outbound call going through the
        rate-limited path above. `?full=true` surfaces `cardData`
        (parsed YAML frontmatter) just like `repo_info`.
        """
        segment = "datasets" if kind == "dataset" else "spaces" if kind == "space" else "models"
        api_url = f"https://huggingface.co/api/{segment}/{repo_id}?full=true"
        headers = {"User-Agent": "Knowledge/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = _request_with_retry(
            requests.get,
            api_url,
            label=f"info({repo_id})",
            headers=headers,
            timeout=20,
        )
        if resp is None or resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _card_data(info) -> dict:
        """Parsed YAML frontmatter from the repo card, or `{}`.

        Now that `_fetch_repo_info` returns a plain dict, the field
        is just a dictionary key — but we keep the legacy CardData
        / object branches for paranoia in case the shape ever
        changes again.
        """
        if info is None:
            return {}
        if isinstance(info, dict):
            cd = info.get("cardData") or info.get("card_data") or {}
            return cd if isinstance(cd, dict) else {}
        cd = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        if cd is None:
            return {}
        if isinstance(cd, dict):
            return cd
        if hasattr(cd, "to_dict"):
            try:
                return cd.to_dict() or {}
            except Exception:
                return {}
        return {}

    def _default_branch(self, repo_id: str, kind: str, info=None) -> str:
        """Default branch from the repo info, or `"main"` as fallback."""
        if info is None:
            return "main"
        if isinstance(info, dict):
            return info.get("default_branch") or info.get("defaultBranch") or "main"
        for attr in ("default_branch", "defaultBranch"):
            value = getattr(info, attr, None)
            if value:
                return value
        return "main"

    # ──────────────────────────────────────────────────────────────────
    # README handling
    # ──────────────────────────────────────────────────────────────────

    def _fetch_readme(self, repo_id: str, kind: str, info) -> str:
        """Raw README.md text; empty string when unavailable."""
        branch = self._default_branch(repo_id, kind, info)
        if kind == "dataset":
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/README.md"
        elif kind == "space":
            url = f"https://huggingface.co/spaces/{repo_id}/resolve/{branch}/README.md"
        else:
            url = f"https://huggingface.co/{repo_id}/resolve/{branch}/README.md"
        resp = _request_with_retry(
            requests.get,
            url,
            label=f"readme({repo_id})",
            headers={"User-Agent": "knowledge-bot/1.0 (+https://github.com)"},
            timeout=15,
        )
        if resp is None or resp.status_code >= 400:
            return ""
        return resp.text

    @staticmethod
    def _strip_frontmatter(content: str) -> tuple[str, dict]:
        """
        Split a README into `(body, parsed_yaml)`.

        When the README opens with `---\n...\n---` we parse the YAML
        block separately so callers can mine it for fields the API
        didn't surface (rare, but not unheard of).
        """
        if not content.strip().startswith("---"):
            return content, {}
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
        if not match:
            return content, {}
        try:
            parsed = yaml.safe_load(match.group(1)) or {}
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}
        return content[match.end() :], parsed

    @classmethod
    def _first_paragraph(cls, body: str) -> str:
        """
        Return the first prose paragraph that's substantive enough to
        be a description.

        Skips badges, raw HTML, code fences, blockquotes, and headings
        whose entire text is generic boilerplate ("Model Card for X").
        Falls back to trafilatura on the rendered HTML if the markdown
        path produces nothing usable.
        """
        if not body:
            return ""

        lines = body.split("\n")
        in_code = False
        paragraphs: list[list[str]] = []
        current: list[str] = []

        for raw in lines:
            line = raw.rstrip()
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code = not in_code
                if current:
                    paragraphs.append(current)
                    current = []
                continue
            if in_code:
                continue
            if not stripped:
                if current:
                    paragraphs.append(current)
                    current = []
                continue
            if stripped.startswith(">"):
                continue
            if _BADGE_LINE_RE.match(stripped):
                continue
            if stripped.startswith("#"):
                # Standalone heading: drop only when the heading text
                # itself is boilerplate. Otherwise keep it as a topic
                # cue without the leading ##.
                heading_text = stripped.lstrip("# ").strip()
                if not heading_text or _BOILERPLATE_HEADING_RE.match(heading_text):
                    continue
                stripped = heading_text
            current.append(stripped)
        if current:
            paragraphs.append(current)

        for para in paragraphs:
            text = " ".join(para)
            text = _INLINE_IMG_RE.sub("", text)
            text = _INLINE_LINK_RE.sub(r"\1", text)
            text = _INLINE_HTML_RE.sub("", text)
            text = re.sub(r"\s+", " ", text).strip(" \t-—–·")
            if cls._is_substantive(text):
                return text

        # Markdown path was unhelpful — try trafilatura on whatever's
        # left (handles HTML embedded inside the README cleanly).
        try:
            core = trafilatura.extract(body) or ""
        except Exception:
            core = ""
        core = re.sub(r"\s+", " ", core).strip()
        return core if cls._is_substantive(core) else ""

    @staticmethod
    def _is_substantive(text: str) -> bool:
        """At least 6 words and not a one-liner like a single URL."""
        if not text:
            return False
        # Drop strings that are nearly all uppercase (likely a banner).
        letters = [c for c in text if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
            return False
        return len(text.split()) >= 6

    # ──────────────────────────────────────────────────────────────────
    # Summary / tag composition
    # ──────────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        repo_id: str,
        kind: str,
        info,
        card_data: dict,
        readme: str,
    ) -> str:
        """
        Compose a description from API metadata + first README paragraph.

        Layout:
          "{Pipeline} {kind} by {org}{license note}. {prose...}"

        Each piece is optional — a repo with neither a pipeline tag
        nor a README still yields a sensible "{kind} by {org}" line.
        """
        org, _, _ = repo_id.partition("/")
        body, frontmatter_yaml = self._strip_frontmatter(readme)
        prose = self._first_paragraph(body)

        # Prefer Hub-API fields, fall back to YAML frontmatter values.
        pipeline = (
            getattr(info, "pipeline_tag", None) or card_data.get("pipeline_tag") or frontmatter_yaml.get("pipeline_tag")
        )
        library = (
            getattr(info, "library_name", None) or card_data.get("library_name") or frontmatter_yaml.get("library_name")
        )
        license_id = card_data.get("license") or frontmatter_yaml.get("license")
        base_models = card_data.get("base_model") or frontmatter_yaml.get("base_model") or []
        if isinstance(base_models, str):
            base_models = [base_models]

        descriptor_bits: list[str] = []
        if pipeline:
            descriptor_bits.append(str(pipeline).replace("-", " "))
        descriptor_bits.append(kind)
        descriptor = " ".join(descriptor_bits)

        meta_clauses: list[str] = []
        if org:
            meta_clauses.append(f"by {org}")
        if library:
            meta_clauses.append(f"built with {library}")
        if base_models:
            base = base_models[0]
            if isinstance(base, str) and base and base != repo_id:
                meta_clauses.append(f"derived from {base}")
        if license_id:
            meta_clauses.append(f"license: {license_id}")

        head = (
            f"{descriptor.capitalize()} {', '.join(meta_clauses)}".strip() if meta_clauses else descriptor.capitalize()
        )
        if not head.endswith("."):
            head += "."

        if prose:
            joined = f"{head} {prose}"
        else:
            joined = head

        return self._sentence_truncate(joined, self.SUMMARY_BUDGET)

    @staticmethod
    def _sentence_truncate(text: str, budget: int) -> str:
        """Cut at a sentence boundary near `budget`; ellipsize otherwise."""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= budget:
            return text
        cut = text[:budget]
        for stop in (". ", "! ", "? "):
            i = cut.rfind(stop)
            if i >= budget * 0.5:
                return cut[: i + 1].rstrip()
        return cut.rstrip(" ,;:-—") + "…"

    @staticmethod
    def _build_tags(kind: str, info, card_data: dict) -> list[str]:
        """
        Combine the static `["huggingface", kind]` pair with the
        repo's own pipeline/library/topic tags. Lowercased, deduped,
        capped at a reasonable length so the indexed text doesn't
        bloat.
        """
        tags: list[str] = ["huggingface", kind]

        pipeline = getattr(info, "pipeline_tag", None) or card_data.get("pipeline_tag")
        if pipeline:
            tags.append(str(pipeline))

        library = getattr(info, "library_name", None) or card_data.get("library_name")
        if library:
            tags.append(str(library))

        # Hub tags + YAML tags — both surface different things
        # (Hub tags include things like "license:mit", "language:en").
        api_tags = list(getattr(info, "tags", []) or [])
        yaml_tags = card_data.get("tags") or []
        if isinstance(yaml_tags, str):
            yaml_tags = [yaml_tags]
        elif not isinstance(yaml_tags, list):
            yaml_tags = []

        for raw in (*api_tags, *yaml_tags):
            if not isinstance(raw, str):
                continue
            t = raw.strip().lower()
            # Drop clutter prefixes that aren't searchable as topics.
            if not t or t.startswith(("license:", "region:", "arxiv:", "dataset:")):
                continue
            if t.startswith("base_model:"):
                continue
            tags.append(t)

        # Dedup while preserving order, cap at 12 to bound index size.
        seen: set[str] = set()
        out: list[str] = []
        for t in tags:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            out.append(tl)
            if len(out) >= 12:
                break
        return out

    @staticmethod
    def _extract_date(info, liked_at: str = "") -> str:
        """Pick a date for a liked HuggingFace repo.

        Preference:
          1. `liked_at` (when the user actually clicked the heart) —
             passed in from the public-likes endpoint's `createdAt`.
             This is the strongest "entered the library" signal.
          2. The repo's last-modified timestamp (Hub API
             `last_modified` / `lastModified`).
          3. Today, as a last resort, so the doc still surfaces in
             the timeline feed (which excludes NULL dates).
        """
        if isinstance(liked_at, str) and len(liked_at) >= 10:
            # ISO-8601 ("2026-05-08T16:24:53.000Z") — date prefix is enough.
            return liked_at[:10]
        if info is not None:
            # Plain dict (our `_fetch_repo_info` returns this) or a
            # legacy `ModelInfo`-style object — handle both.
            def _read(key_or_attr: str):
                if isinstance(info, dict):
                    return info.get(key_or_attr)
                return getattr(info, key_or_attr, None)

            for attr in ("last_modified", "lastModified"):
                value = _read(attr)
                if value is None:
                    continue
                if isinstance(value, datetime.datetime):
                    return value.strftime("%Y-%m-%d")
                if isinstance(value, str) and value:
                    # Hub returns ISO-8601; the date prefix is enough.
                    return value[:10]
        return datetime.datetime.today().strftime("%Y-%m-%d")
