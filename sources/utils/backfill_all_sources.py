"""CLI: deterministic source-key backfill from `users.links`.

Codifies the patterns the per-user enrichment agents reliably found, so
new personalities get automatic coverage without an LLM in the loop.

Adds (only if missing):
  - `youtube_search` — derived from display name, low false-positive risk
    because of `must_contain` filter on the lowercase name fragment.
  - `websites` — `links.website` URL, validated by probing any of
    `/sitemap.xml`, `/feed.xml`, `/feed`, `/atom.xml`, `/index.xml`,
    `/rss.xml`. The first that returns 200 with XML-shaped content wins
    (kind=feed for atom/rss, kind=sitemap for sitemap.xml).
  - `huggingface` — derive a handle from `links.github` (`github.com/X`
    → `X`), probe `huggingface.co/X`. Add only when the profile exists
    AND has at least one of: 1+ followers, 1+ models/datasets, OR a
    bio that links back to the same github / website. Skips empty
    name-squat profiles.
  - `github_repos`, `github_gists` — derive from `links.github`. Cheap
    to add (the GitHub fetcher early-exits on 404), so we don't probe.

Deliberately SKIPS (too risky without per-person reasoning):
  - `wikipedia`     — most personalities don't have an article; verifying
                      requires checking the page title disambiguates them.
  - `reddit`/`stackoverflow`/`hn_*` — handle squatting is endemic.
  - `arxiv`/`scholar`/`dblp`/`semantic_scholar` — common-name collisions
                      need per-person disambiguation.

Idempotent: only fills `sources` keys that are currently missing. Never
overwrites a human-curated value.

Usage::

    DATABASE_URL=postgresql://... uv run python -m sources.utils.backfill_all_sources
    # …or dry-run:
    DATABASE_URL=postgresql://... uv run python -m sources.utils.backfill_all_sources --dry
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
HTTP_TIMEOUT = 6.0
PROBE_WORKERS = 12

# ── Display-name → youtube_search defaults ──────────────────────────────

_YT_QUERY_TEMPLATES = ('"{name}" talk', '"{name}" interview', '"{name}" lecture', '"{name}" podcast')


def youtube_search_block(display_name: str) -> dict | None:
    name = (display_name or "").strip()
    if not name or len(name.split()) < 2:
        # Single-word personalities (e.g. "antirez", "akhaliq") are too
        # ambiguous for a generic search; let humans curate those.
        return None
    must = name.lower()
    return {
        "queries": [tpl.format(name=name) for tpl in _YT_QUERY_TEMPLATES],
        "must_contain": [must],
        "max_results": 30,
    }


# ── Websites probing ────────────────────────────────────────────────────
# Order matters — we prefer feed-shaped sources first because the pipeline
# can ingest them whole, while sitemap fallback handles static sites.

_WEBSITE_CANDIDATES = (
    ("/feed.xml", "feed"),
    ("/feed", "feed"),
    ("/atom.xml", "feed"),
    ("/rss.xml", "feed"),
    ("/index.xml", "feed"),
    ("/sitemap.xml", "sitemap"),
)


def _probe_url(client: httpx.Client, url: str) -> bool:
    """True iff `url` returns 200 with body that smells like XML/RSS/sitemap."""
    try:
        r = client.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError:
        return False
    if r.status_code != 200:
        return False
    head = (r.text or "")[:512].lower()
    # Permissive on content-type — many sites serve feeds as text/html.
    return any(tok in head for tok in ("<rss", "<feed", "<sitemap", "<urlset", "<?xml"))


def find_website_block(website_url: str | None, client: httpx.Client) -> list[dict] | None:
    if not website_url:
        return None
    base = website_url.strip().rstrip("/")
    if not re.match(r"^https?://", base):
        return None
    for path, kind in _WEBSITE_CANDIDATES:
        candidate = base + path
        if _probe_url(client, candidate):
            entry = {"input": candidate, "url": candidate, "kind": kind, "tags": ["blog"]}
            return [entry]
    # Last-resort: many academic homepages have no feed but list links to
    # papers/talks. We DON'T add the bare root here because the sitemap
    # fetcher will 404 on it; the agent-level check is what catches this.
    return None


# ── Hugging Face profile validation ─────────────────────────────────────

_GITHUB_HANDLE_RE = re.compile(r"^https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/?$")


def _github_handle(github_url: str | None) -> str | None:
    if not github_url:
        return None
    m = _GITHUB_HANDLE_RE.match(github_url.strip())
    if not m:
        return None
    handle = m.group(1)
    if handle.lower() in {"orgs", "settings", "marketplace"}:
        return None
    return handle


def find_huggingface_handle(handle: str | None, github_url: str | None, client: httpx.Client) -> str | None:
    """Return `<handle>` only when `huggingface.co/<handle>` is non-empty.

    "Non-empty" = response body contains a back-link to the user's GitHub
    or website, OR the JSON API reports ≥1 follower / model / dataset.
    Empty placeholder profiles (a lot of well-known names have one) are
    filtered out so we don't add a dead source.
    """
    if not handle:
        return None
    api = f"https://huggingface.co/api/users/{handle}/overview"
    try:
        r = client.get(api, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    # Profile exists. Check it's not a name-squat: we want some signal.
    followers = int(data.get("numFollowers") or 0)
    models = int(data.get("numModels") or 0)
    datasets = int(data.get("numDatasets") or 0)
    spaces = int(data.get("numSpaces") or 0)
    if followers + models + datasets + spaces > 0:
        return handle
    # Otherwise check if profile body cross-links back to our GitHub URL.
    if github_url:
        try:
            page = client.get(f"https://huggingface.co/{handle}", timeout=HTTP_TIMEOUT)
            if page.status_code == 200 and (github_url in page.text or handle in page.text):
                # Still want at least *one* signal — the github backlink.
                if github_url in page.text:
                    return handle
        except httpx.HTTPError:
            pass
    return None


# ── Per-user enrichment ─────────────────────────────────────────────────


def derive_additions(user: dict, client: httpx.Client) -> dict:
    """Return a dict of NEW source-keys to add (never modifies existing)."""
    sources = user.get("sources") or {}
    links = user.get("links") or {}
    name = (user.get("name") or "").strip()

    out: dict = {}

    # 1. youtube_search ----------------------------------------------------
    if "youtube_search" not in sources:
        block = youtube_search_block(name)
        if block:
            out["youtube_search"] = block

    # 2. websites ----------------------------------------------------------
    if "websites" not in sources:
        wb = find_website_block(links.get("website"), client)
        if wb:
            out["websites"] = wb

    # 3. github_repos / github_gists --------------------------------------
    gh = _github_handle(links.get("github"))
    if gh:
        if "github_repos" not in sources:
            out["github_repos"] = [gh]
        if "github_gists" not in sources:
            out["github_gists"] = [gh]

    # 4. huggingface -------------------------------------------------------
    if "huggingface" not in sources and gh:
        confirmed = find_huggingface_handle(gh, links.get("github"), client)
        if confirmed:
            out["huggingface"] = confirmed

    return out


def backfill(database_url: str, *, dry_run: bool = False) -> list[tuple[str, dict]]:
    """Probe every user, return the per-user additions plan, optionally apply.

    Plan shape: ``[(slug, {key: value, ...}), ...]``. Idempotent — only
    fills keys that aren't already in `sources`.
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, name, links, sources "
                "  FROM users "
                " WHERE sources IS NOT NULL AND sources <> '{}'::jsonb"
            )
            rows = [
                {"username": u, "name": n, "links": lk or {}, "sources": s or {}} for (u, n, lk, s) in cur.fetchall()
            ]

    plans: list[tuple[str, dict]] = []
    with httpx.Client(headers={"User-Agent": "knowledge-backfill/1.0"}) as client:
        with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as ex:
            futures = {ex.submit(derive_additions, u, client): u["username"] for u in rows}
            for f in as_completed(futures):
                slug = futures[f]
                try:
                    additions = f.result()
                except Exception:
                    continue
                if additions:
                    plans.append((slug, additions))
    plans.sort(key=lambda x: x[0])

    if dry_run or not plans:
        return plans

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for slug, adds in plans:
                cur.execute(
                    "UPDATE users SET sources = sources || %s::jsonb, updated_at = now() WHERE username = %s",
                    (json.dumps(adds), slug),
                )
        conn.commit()

    return plans


def main() -> None:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    dry_run = "--dry" in sys.argv

    plans = backfill(url, dry_run=dry_run)
    print(f"[backfill] {len(plans)} users get at least one new key\n")
    for slug, adds in plans:
        print(f"  + {slug:<28} {', '.join(sorted(adds.keys()))}")

    if dry_run:
        print("\n[backfill] dry-run, no DB writes")
        return

    counts: dict[str, int] = {}
    for _, adds in plans:
        for k in adds.keys():
            counts[k] = counts.get(k, 0) + 1
    print("\n[backfill] keys added:")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {k:<20} {counts[k]:>3}")


if __name__ == "__main__":
    main()
