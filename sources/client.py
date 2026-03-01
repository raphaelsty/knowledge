#!/usr/bin/env python3
"""
Knowledge Database Builder

Aggregates knowledge from multiple sources (GitHub, HackerNews, Zotero,
HuggingFace, Twitter/X) into a unified database and builds the search pipeline.

Environment Variables
---------------------
HACKERNEWS_USERNAME : str
    HackerNews username for fetching upvoted posts.
HACKERNEWS_PASSWORD : str
    HackerNews password for authentication.
ZOTERO_LIBRARY_ID : str
    Zotero library ID for fetching bookmarks.
ZOTERO_API_KEY : str
    Zotero API key with read permissions.
HUGGINGFACE_TOKEN : str
    HuggingFace token for fetching liked items.
TWITTER_AUTH_TOKEN : str
    The auth_token cookie from a logged-in Twitter/X browser session.
TWITTER_CT0 : str
    The ct0 cookie (CSRF token) from a logged-in Twitter/X session.

Configuration
-------------
sources.yml : file
    YAML configuration specifying which sources to enable.

Output Files
------------
web/data/database.json : JSON
    Aggregated document database.
"""

import json
import os
import time
from collections import Counter
from urllib.parse import urlparse

import yaml

from . import github, hackernews, huggingface, tags, twitter, zotero
from .database import ensure_schema, load_all_documents, load_generated, save_all_documents, save_generated
from .taxonomy import build as build_taxonomy


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def merge_new_documents(existing: dict, new: dict) -> tuple[dict, set[str]]:
    """Merge new documents, skipping URLs already in the database.

    Returns the merged dict and the set of newly added URLs.
    """
    new_only = {url: doc for url, doc in new.items() if url not in existing}
    print(f"Found {len(new_only)} new documents.")
    return {**existing, **new_only}, set(new_only.keys())


def main():
    pipeline_start = time.perf_counter()
    timings: list[tuple[str, float]] = []

    # =============================================================================
    # Database Setup
    # =============================================================================

    use_pg = os.environ.get("DATABASE_URL") is not None
    if use_pg:
        print("Using PostgreSQL database...")
        ensure_schema()
    else:
        print("No DATABASE_URL set, using JSON file storage.")

    # =============================================================================
    # Configuration
    # =============================================================================

    # Load source configuration
    with open("sources.yml") as f:
        sources = yaml.load(f, Loader=yaml.FullLoader)

    # Load credentials from environment
    hackernews_username = os.environ.get("HACKERNEWS_USERNAME")
    hackernews_password = os.environ.get("HACKERNEWS_PASSWORD")
    zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID")
    zotero_api_key = os.environ.get("ZOTERO_API_KEY")
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    twitter_auth_token = os.environ.get("TWITTER_AUTH_TOKEN")
    twitter_ct0 = os.environ.get("TWITTER_CT0")

    # =============================================================================
    # Load Existing Database
    # =============================================================================

    data: dict = {}

    t0 = time.perf_counter()
    if use_pg:
        data = load_all_documents()
    elif os.path.exists("web/data/database.json"):
        with open("web/data/database.json", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    timings.append(("Load database", time.perf_counter() - t0))

    # =============================================================================
    # Fetch Data from Sources
    # =============================================================================

    new_urls: set[str] = set()

    # GitHub starred repositories
    if sources.get("github") is not None:
        t0 = time.perf_counter()
        print("Fetching GitHub stars...")
        for user in sources["github"]:
            fetcher = github.Github(user=user)
            data, added = merge_new_documents(data, fetcher())
            new_urls |= added
        timings.append(("Fetch GitHub", time.perf_counter() - t0))

    # HackerNews upvoted posts
    if hackernews_username is not None and hackernews_password is not None:
        t0 = time.perf_counter()
        print("Fetching HackerNews upvotes...")
        fetcher = hackernews.HackerNews(
            username=hackernews_username,
            password=hackernews_password,
        )
        data, added = merge_new_documents(data, fetcher())
        new_urls |= added
        timings.append(("Fetch HackerNews", time.perf_counter() - t0))
    else:
        print("Skipping HackerNews (no credentials).")

    # Zotero library
    if zotero_library_id is not None and zotero_api_key is not None:
        t0 = time.perf_counter()
        print("Fetching Zotero library...")
        fetcher = zotero.Zotero(
            library_id=zotero_library_id,
            library_type="group",
            api_key=zotero_api_key,
        )
        data, added = merge_new_documents(data, fetcher())
        new_urls |= added
        timings.append(("Fetch Zotero", time.perf_counter() - t0))
    else:
        print("Skipping Zotero (no credentials).")

    # HuggingFace liked items
    if huggingface_token is not None and sources.get("huggingface") is not None:
        t0 = time.perf_counter()
        print("Fetching HuggingFace likes...")
        fetcher = huggingface.HuggingFace(token=huggingface_token)
        data, added = merge_new_documents(data, fetcher())
        new_urls |= added
        timings.append(("Fetch HuggingFace", time.perf_counter() - t0))
    else:
        print("Skipping HuggingFace (no token).")

    # Twitter/X bookmarked tweets
    if sources.get("twitter") is not None:
        t0 = time.perf_counter()
        # Use env vars (CI) or auto-extract from Safari (local)
        if twitter_auth_token is None or twitter_ct0 is None:
            try:
                print("No Twitter env vars found, extracting cookies from Safari...")
                safari_cookies = twitter.get_safari_cookies()
                twitter_auth_token = safari_cookies["auth_token"]
                twitter_ct0 = safari_cookies["ct0"]
            except Exception as e:
                print(f"Skipping Twitter/X (could not get cookies: {e})")
                twitter_auth_token = None

        if twitter_auth_token is not None and twitter_ct0 is not None:
            print("Fetching Twitter/X bookmarks...")
            twitter_config = sources["twitter"]
            twitter_fetcher = twitter.Twitter(
                auth_token=twitter_auth_token,
                ct0=twitter_ct0,
                username=twitter_config.get("username", ""),
                min_likes=twitter_config.get("min_likes", 10),
            )
            max_pages = twitter_config.get("max_pages", 5)
            data, added = merge_new_documents(
                data,
                twitter_fetcher(
                    max_pages=max_pages,
                    existing_urls=set(data.keys()),
                ),
            )
            new_urls |= added
        timings.append(("Fetch Twitter/X", time.perf_counter() - t0))
    else:
        print("Skipping Twitter/X (disabled).")

    # =============================================================================
    # Data Cleaning
    # =============================================================================

    t0 = time.perf_counter()
    print("Cleaning document data...")

    for _url, document in data.items():
        # Ensure all required fields exist
        for field in ["title", "tags", "summary", "date"]:
            if document.get(field) is None:
                document[field] = "" if field != "tags" else []

        # Clean invalid Unicode characters (e.g., lone surrogates)
        for field in ["title", "summary"]:
            if isinstance(document.get(field), str):
                document[field] = document[field].encode("utf-8", "replace").decode("utf-8")

    # Remove documents with empty title or URL
    before = len(data)
    data = {url: doc for url, doc in data.items() if url.strip() and doc.get("title", "").strip()}
    removed = before - len(data)
    if removed:
        print(f"  Removed {removed} documents with empty title or URL")

    # Deduplicate by normalized URL (strip trailing slash, www., force https)
    def _normalize_url(u: str) -> str:
        u = u.strip().rstrip("/")
        if u.startswith("http://"):
            u = "https://" + u[7:]
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return f"{parsed.scheme}://{host}{parsed.path}{parsed.query}"

    seen: dict[str, str] = {}  # normalized -> original url
    duplicates: list[str] = []
    for url in list(data.keys()):
        norm = _normalize_url(url)
        if norm in seen:
            duplicates.append(url)
        else:
            seen[norm] = url
    for url in duplicates:
        del data[url]
    if duplicates:
        print(f"  Removed {len(duplicates)} duplicate URLs (after normalization)")

    timings.append(("Clean data", time.perf_counter() - t0))

    # =============================================================================
    # Generate Extra Tags
    # =============================================================================

    t0 = time.perf_counter()
    print("Generating extra tags from document content...")
    data = tags.get_extra_tags(data=data)
    timings.append(("Generate extra tags", time.perf_counter() - t0))

    # =============================================================================
    # Save Database
    # =============================================================================

    t0 = time.perf_counter()
    print("Saving database...")
    if use_pg:
        save_all_documents(data)
    else:
        with open("web/data/database.json", "w") as f:
            json.dump(data, f, indent=4)
    timings.append(("Save database", time.perf_counter() - t0))

    # =============================================================================
    # Extract Source Filters
    # =============================================================================

    t0 = time.perf_counter()

    # Domain aliases: merge variants into a single source key
    DOMAIN_ALIASES = {
        "x.com": "twitter.com",
        "www.github.com": "github.com",
        "www.arxiv.org": "arxiv.org",
        "www.twitter.com": "twitter.com",
        "mobile.twitter.com": "twitter.com",
        "mobile.x.com": "twitter.com",
    }

    # Friendly labels for known domains
    DOMAIN_LABELS = {
        "github.com": "GitHub",
        "twitter.com": "X",
        "arxiv.org": "arXiv",
        "huggingface.co": "HuggingFace",
    }

    MIN_SOURCE_COUNT = 5  # Minimum documents to be a named source
    MAX_SOURCE_FILTERS = 8  # Cap the number of filters to keep UI clean

    print("Extracting source filters from URLs...")
    domain_counter: Counter = Counter()
    for url in data:
        try:
            domain = urlparse(url).netloc.lower()
            domain = DOMAIN_ALIASES.get(domain, domain)
            # Strip www. prefix for consistency
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domain_counter[domain] += 1
        except Exception:
            continue

    # Build sources list ordered by frequency, only domains above threshold
    sources_list = []
    for domain, count in domain_counter.most_common():
        if count >= MIN_SOURCE_COUNT:
            label = DOMAIN_LABELS.get(domain, domain.split(".")[0].capitalize())
            sources_list.append({"key": domain, "label": label, "count": count})

    # Also check tags/titles for sources without distinct URLs (e.g. hackernews)
    hn_count = sum(
        1
        for doc in data.values()
        if "hackernews" in (doc.get("title") or "").lower()
        or any("hackernews" in (t or "").lower() for t in doc.get("tags", []))
        or any("hackernews" in (t or "").lower() for t in doc.get("extra-tags", []))
    )
    if hn_count >= MIN_SOURCE_COUNT:
        # Insert hackernews if not already covered by a domain
        if not any(s["key"] == "hackernews" for s in sources_list):
            sources_list.append({"key": "hackernews", "label": "HackerNews", "count": hn_count})
            # Re-sort by count
            sources_list.sort(key=lambda s: s["count"], reverse=True)

    # Keep only the top N sources to avoid UI clutter
    sources_list = sources_list[:MAX_SOURCE_FILTERS]

    print(
        f"Found {len(sources_list)} source filters: {', '.join(s['label'] + ' (' + str(s['count']) + ')' for s in sources_list)}"
    )
    if use_pg:
        save_generated("sources", sources_list)
    else:
        with open("web/data/sources.json", "w") as f:
            json.dump(sources_list, f, indent=2)
    timings.append(("Extract source filters", time.perf_counter() - t0))

    # =============================================================================
    # Build Knowledge Graph
    # =============================================================================

    # Tags to exclude from graph visualization (too generic)
    EXCLUDED_TAGS = {
        "twitter": True,
        "github": True,
        "hackernews": True,
        "arxiv doc": True,
    }

    t0 = time.perf_counter()
    print("Building tag co-occurrence graph...")
    triples = tags.get_tags_triples(data=data, excluded_tags=EXCLUDED_TAGS)
    timings.append(("Build tag triples", time.perf_counter() - t0))

    # =============================================================================
    # Build Tag Tree
    # =============================================================================

    t0 = time.perf_counter()
    print("Building tag tree...")
    build_taxonomy(triples=triples, database=data, use_pg=use_pg)
    timings.append(("Build taxonomy", time.perf_counter() - t0))

    # =============================================================================
    # Write Buffer for Incremental Indexing
    # =============================================================================

    t0 = time.perf_counter()
    buffer_dir = os.environ.get("BUFFER_DIR", "buffer")
    # Filter new_urls to only those that survived cleaning/dedup
    new_urls = {url for url in new_urls if url in data}
    if new_urls:
        os.makedirs(buffer_dir, exist_ok=True)
        batch = [
            {
                "url": url,
                "title": data[url].get("title", ""),
                "summary": data[url].get("summary", ""),
                "date": data[url].get("date", ""),
                "tags": data[url].get("tags", []),
                "extra_tags": data[url].get("extra-tags", []),
            }
            for url in new_urls
        ]
        filename = f"{int(time.time() * 1000)}_{os.getpid()}.json"
        filepath = os.path.join(buffer_dir, filename)
        with open(filepath, "w") as f:
            json.dump(batch, f)
        print(f"Wrote {len(batch)} new documents to {filepath}")
    else:
        print("No new documents to buffer.")
    timings.append(("Write buffer", time.perf_counter() - t0))

    # =============================================================================
    # Rescue Placement
    # =============================================================================

    if new_urls and use_pg:
        t0 = time.perf_counter()
        folder_tree = load_generated("folder_tree")
        if folder_tree:
            from .taxonomy import load_model, rescue_unplaced_docs

            model = load_model("minishlab/potion-base-8M")
            folder_tree, n = rescue_unplaced_docs(folder_tree, data, model)
            if n:
                save_generated("folder_tree", folder_tree)
                print(f"Rescued {n} documents into folder tree.")
        timings.append(("Rescue placement", time.perf_counter() - t0))

    # =============================================================================
    # Summary
    # =============================================================================

    total = time.perf_counter() - pipeline_start
    print("\n" + "=" * 60)
    print("  Pipeline timing summary")
    print("=" * 60)
    for label, elapsed in timings:
        pct = elapsed / total * 100
        print(f"  {label:<25s} {_fmt(elapsed):>8s}  ({pct:4.1f}%)")
    print(f"  {'─' * 41}")
    print(f"  {'Total':<25s} {_fmt(total):>8s}")
    print("=" * 60)
