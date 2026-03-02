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


def step(pct: int, label: str, detail: str = ""):
    """Emit a structured progress line for the frontend to parse."""
    print(f"@@{pct}|{label}|{detail}@@", flush=True)


def merge_new_documents(existing: dict, new: dict) -> tuple[dict, set[str]]:
    """Merge new documents, skipping URLs already in the database.

    Returns the merged dict and the set of newly added URLs.
    """
    new_only = {url: doc for url, doc in new.items() if url not in existing}
    return {**existing, **new_only}, set(new_only.keys())


def main():
    pipeline_start = time.perf_counter()
    timings: list[tuple[str, float]] = []

    # =============================================================================
    # Database Setup
    # =============================================================================

    step(2, "Connecting", "Initializing database")
    use_pg = os.environ.get("DATABASE_URL") is not None
    if use_pg:
        ensure_schema()
    else:
        print("No DATABASE_URL set, using JSON file storage.")

    # =============================================================================
    # Configuration
    # =============================================================================

    with open("sources.yml") as f:
        sources = yaml.load(f, Loader=yaml.FullLoader)

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
    step(4, "Loading database", "Reading existing documents")
    if use_pg:
        data = load_all_documents()
    elif os.path.exists("web/data/database.json"):
        with open("web/data/database.json", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    step(6, "Database loaded", f"{len(data):,} documents")
    timings.append(("Load database", time.perf_counter() - t0))

    # =============================================================================
    # Fetch Data from Sources
    # =============================================================================

    new_urls: set[str] = set()
    github_users = sources.get("github") or []
    n_sources = len(github_users)
    has_hn = hackernews_username is not None and hackernews_password is not None
    has_zotero = zotero_library_id is not None and zotero_api_key is not None
    has_hf = huggingface_token is not None and sources.get("huggingface") is not None
    has_twitter = sources.get("twitter") is not None
    n_sources += sum([has_hn, has_zotero, has_hf, has_twitter])

    # Allocate 8–40% for source fetching, split evenly across sources
    src_start, src_end = 8, 40
    src_step = (src_end - src_start) // max(n_sources, 1)
    src_pct = src_start

    # GitHub starred repositories
    if github_users:
        t0 = time.perf_counter()
        for user in github_users:
            step(src_pct, "Fetching GitHub", f"Stars from @{user}")
            fetcher = github.Github(user=user)
            data, added = merge_new_documents(data, fetcher(existing_urls=set(data.keys())))
            new_urls |= added
            detail = f"@{user}: +{len(added)} new" if added else f"@{user}: up to date"
            src_pct = min(src_pct + src_step, src_end)
            step(src_pct, "GitHub", detail)
        timings.append(("Fetch GitHub", time.perf_counter() - t0))

    # HackerNews upvoted posts
    if has_hn:
        t0 = time.perf_counter()
        step(src_pct, "Fetching HackerNews", f"Upvotes from @{hackernews_username}")
        fetcher = hackernews.HackerNews(
            username=hackernews_username,
            password=hackernews_password,
        )
        data, added = merge_new_documents(data, fetcher())
        new_urls |= added
        src_pct = min(src_pct + src_step, src_end)
        step(src_pct, "HackerNews", f"+{len(added)} new" if added else "Up to date")
        timings.append(("Fetch HackerNews", time.perf_counter() - t0))
    else:
        step(src_pct, "Sources", "HackerNews skipped (no credentials)")

    # Zotero library
    if has_zotero:
        t0 = time.perf_counter()
        step(src_pct, "Fetching Zotero", "Reading library")
        fetcher = zotero.Zotero(
            library_id=zotero_library_id,
            library_type="group",
            api_key=zotero_api_key,
        )
        data, added = merge_new_documents(data, fetcher())
        new_urls |= added
        src_pct = min(src_pct + src_step, src_end)
        step(src_pct, "Zotero", f"+{len(added)} new" if added else "Up to date")
        timings.append(("Fetch Zotero", time.perf_counter() - t0))
    else:
        step(src_pct, "Sources", "Zotero skipped (no credentials)")

    # HuggingFace liked items
    if has_hf:
        t0 = time.perf_counter()
        step(src_pct, "Fetching HuggingFace", "Liked models and datasets")
        fetcher = huggingface.HuggingFace(token=huggingface_token)
        data, added = merge_new_documents(data, fetcher())
        new_urls |= added
        src_pct = min(src_pct + src_step, src_end)
        step(src_pct, "HuggingFace", f"+{len(added)} new" if added else "Up to date")
        timings.append(("Fetch HuggingFace", time.perf_counter() - t0))
    else:
        step(src_pct, "Sources", "HuggingFace skipped (no token)")

    # Twitter/X bookmarked tweets
    if has_twitter:
        t0 = time.perf_counter()
        if twitter_auth_token is None or twitter_ct0 is None:
            try:
                safari_cookies = twitter.get_safari_cookies()
                twitter_auth_token = safari_cookies["auth_token"]
                twitter_ct0 = safari_cookies["ct0"]
            except Exception:
                twitter_auth_token = None

        if twitter_auth_token is not None and twitter_ct0 is not None:
            twitter_config = sources["twitter"]
            username = twitter_config.get("username", "")
            step(src_pct, "Fetching Twitter/X", f"Bookmarks from @{username}" if username else "Bookmarks")
            twitter_fetcher = twitter.Twitter(
                auth_token=twitter_auth_token,
                ct0=twitter_ct0,
                username=username,
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
            src_pct = min(src_pct + src_step, src_end)
            step(src_pct, "Twitter/X", f"+{len(added)} new" if added else "Up to date")
        else:
            step(src_pct, "Sources", "Twitter/X skipped (no cookies)")
        timings.append(("Fetch Twitter/X", time.perf_counter() - t0))

    total_new = len(new_urls)
    step(42, "Sources complete", f"{total_new} new document{'s' if total_new != 1 else ''} found")

    # =============================================================================
    # Data Cleaning
    # =============================================================================

    t0 = time.perf_counter()
    step(44, "Cleaning", f"Validating {len(data):,} documents")

    for _url, document in data.items():
        for field in ["title", "tags", "summary", "date"]:
            if document.get(field) is None:
                document[field] = "" if field != "tags" else []
        for field in ["title", "summary"]:
            if isinstance(document.get(field), str):
                document[field] = document[field].encode("utf-8", "replace").decode("utf-8")

    before = len(data)
    data = {url: doc for url, doc in data.items() if url.strip() and doc.get("title", "").strip()}
    removed = before - len(data)

    def _normalize_url(u: str) -> str:
        u = u.strip().rstrip("/")
        if u.startswith("http://"):
            u = "https://" + u[7:]
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return f"{parsed.scheme}://{host}{parsed.path}{parsed.query}"

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for url in list(data.keys()):
        norm = _normalize_url(url)
        if norm in seen:
            duplicates.append(url)
        else:
            seen[norm] = url
    for url in duplicates:
        del data[url]

    cleaned = removed + len(duplicates)
    step(48, "Cleaned", f"{len(data):,} documents" + (f" ({cleaned} removed)" if cleaned else ""))
    timings.append(("Clean data", time.perf_counter() - t0))

    # =============================================================================
    # Generate Extra Tags
    # =============================================================================

    t0 = time.perf_counter()
    if new_urls:
        step(50, "Generating tags", f"Extracting keywords from {len(data):,} documents")
        data = tags.get_extra_tags(data=data)
        step(62, "Tags generated", f"{len(data):,} documents tagged")
    else:
        step(62, "Tags", "Skipped (no new documents)")
    timings.append(("Generate extra tags", time.perf_counter() - t0))

    # =============================================================================
    # Save Database
    # =============================================================================

    t0 = time.perf_counter()
    step(64, "Saving", f"Writing {len(data):,} documents" + (" to PostgreSQL" if use_pg else ""))
    if use_pg:
        save_all_documents(data)
    else:
        with open("web/data/database.json", "w") as f:
            json.dump(data, f, indent=4)
    step(70, "Saved", f"{len(data):,} documents")
    timings.append(("Save database", time.perf_counter() - t0))

    # =============================================================================
    # Extract Source Filters
    # =============================================================================

    t0 = time.perf_counter()
    step(72, "Source filters", "Analyzing URL domains")

    DOMAIN_ALIASES = {
        "x.com": "twitter.com",
        "www.github.com": "github.com",
        "www.arxiv.org": "arxiv.org",
        "www.twitter.com": "twitter.com",
        "mobile.twitter.com": "twitter.com",
        "mobile.x.com": "twitter.com",
    }

    DOMAIN_LABELS = {
        "github.com": "GitHub",
        "twitter.com": "X",
        "arxiv.org": "arXiv",
        "huggingface.co": "HuggingFace",
    }

    MIN_SOURCE_COUNT = 5
    MAX_SOURCE_FILTERS = 8

    domain_counter: Counter = Counter()
    for url in data:
        try:
            domain = urlparse(url).netloc.lower()
            domain = DOMAIN_ALIASES.get(domain, domain)
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domain_counter[domain] += 1
        except Exception:
            continue

    sources_list = []
    for domain, count in domain_counter.most_common():
        if count >= MIN_SOURCE_COUNT:
            label = DOMAIN_LABELS.get(domain, domain.split(".")[0].capitalize())
            sources_list.append({"key": domain, "label": label, "count": count})

    hn_count = sum(
        1
        for doc in data.values()
        if "hackernews" in (doc.get("title") or "").lower()
        or any("hackernews" in (t or "").lower() for t in doc.get("tags", []))
        or any("hackernews" in (t or "").lower() for t in doc.get("extra-tags", []))
    )
    if hn_count >= MIN_SOURCE_COUNT:
        if not any(s["key"] == "hackernews" for s in sources_list):
            sources_list.append({"key": "hackernews", "label": "HackerNews", "count": hn_count})
            sources_list.sort(key=lambda s: s["count"], reverse=True)

    sources_list = sources_list[:MAX_SOURCE_FILTERS]
    filter_summary = ", ".join(s["label"] for s in sources_list[:4])
    if len(sources_list) > 4:
        filter_summary += f" +{len(sources_list) - 4} more"

    if use_pg:
        save_generated("sources", sources_list)
    else:
        with open("web/data/sources.json", "w") as f:
            json.dump(sources_list, f, indent=2)
    step(76, "Filters ready", filter_summary)
    timings.append(("Extract source filters", time.perf_counter() - t0))

    # =============================================================================
    # Build Knowledge Graph
    # =============================================================================

    EXCLUDED_TAGS = {
        "twitter": True,
        "github": True,
        "hackernews": True,
        "arxiv doc": True,
    }

    if new_urls:
        t0 = time.perf_counter()
        step(78, "Building graph", "Computing tag co-occurrences")
        triples = tags.get_tags_triples(data=data, excluded_tags=EXCLUDED_TAGS)
        step(82, "Graph built", f"{len(triples):,} tag connections")
        timings.append(("Build tag triples", time.perf_counter() - t0))

        # =============================================================================
        # Build Tag Tree
        # =============================================================================

        t0 = time.perf_counter()
        step(84, "Building taxonomy", "Clustering tags into folders")
        build_taxonomy(triples=triples, database=data, use_pg=use_pg)
        step(94, "Taxonomy built", "Folder tree updated")
        timings.append(("Build taxonomy", time.perf_counter() - t0))
    else:
        step(94, "Taxonomy", "Skipped (no new documents)")

    # =============================================================================
    # Index Documents via API
    # =============================================================================

    t0 = time.perf_counter()
    import urllib.error
    import urllib.request

    api_base = os.environ.get("API_URL", "http://localhost:8080")
    BATCH = 300

    # Check if the search index already exists
    index_exists = False
    try:
        with urllib.request.urlopen(f"{api_base}/indices/knowledge", timeout=5) as resp:
            index_exists = resp.status == 200
    except Exception:
        pass

    # Decide what to index: all docs if fresh, only new if incremental
    new_urls = {url for url in new_urls if url in data}
    urls_to_index = list(data.keys()) if not index_exists else list(new_urls)

    if urls_to_index:
        # Ensure index is declared
        if not index_exists:
            step(96, "Creating index", "Declaring search index")
            payload = json.dumps({"name": "knowledge", "config": {"nbits": 2}}).encode()
            req = urllib.request.Request(
                f"{api_base}/indices",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10):
                    pass
            except urllib.error.HTTPError as e:
                if e.code != 409:  # 409 = already exists, that's fine
                    print(f"  Warning: create index failed: {e}")

        # Build document texts and metadata
        docs_to_index = []
        metadata_to_index = []
        for url in urls_to_index:
            doc = data[url]
            title = doc.get("title", "")
            doc_tags = doc.get("tags", [])
            extra = doc.get("extra-tags", [])
            summary = doc.get("summary", "")
            text = f"{title} {' '.join(doc_tags)} {' '.join(extra)} {summary[:200]}".strip()
            if not text:
                continue
            docs_to_index.append(text)
            metadata_to_index.append(
                {
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "date": doc.get("date", ""),
                    "tags": ",".join(doc_tags),
                    "extra_tags": ",".join(extra),
                }
            )

        if docs_to_index:
            n_batches = (len(docs_to_index) + BATCH - 1) // BATCH
            label = "Building index" if not index_exists else "Indexing"
            step(96, label, f"{len(docs_to_index)} documents in {n_batches} batches")
            indexed = 0
            for i in range(0, len(docs_to_index), BATCH):
                batch_docs = docs_to_index[i : i + BATCH]
                batch_meta = metadata_to_index[i : i + BATCH]
                batch_num = i // BATCH + 1
                payload = json.dumps(
                    {
                        "documents": batch_docs,
                        "metadata": batch_meta,
                        "pool_factor": 2,
                    }
                ).encode()
                req = urllib.request.Request(
                    f"{api_base}/indices/knowledge/update_with_encoding",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=300):
                        indexed += len(batch_docs)
                        pct = 96 + (batch_num * 2) // n_batches
                        step(min(pct, 98), label, f"Batch {batch_num}/{n_batches} ({indexed:,} docs)")
                except Exception as e:
                    print(f"  Warning: batch {batch_num} failed: {e}")
            step(98, "Indexed", f"{indexed:,} documents in search engine")
        else:
            step(98, "Index", "No indexable documents")
    else:
        step(98, "Index", "No new documents to index")
    timings.append(("Index documents", time.perf_counter() - t0))

    # =============================================================================
    # Rescue Placement
    # =============================================================================

    if new_urls and use_pg:
        t0 = time.perf_counter()
        step(97, "Placing documents", "Finding best folders for new documents")
        folder_tree = load_generated("folder_tree")
        if folder_tree:
            from .taxonomy import load_model, rescue_unplaced_docs

            model = load_model("minishlab/potion-base-8M")
            folder_tree, n = rescue_unplaced_docs(folder_tree, data, model)
            if n:
                save_generated("folder_tree", folder_tree)
                step(99, "Documents placed", f"{n} documents added to folders")
            else:
                step(99, "Placement", "All documents already in folders")
        timings.append(("Rescue placement", time.perf_counter() - t0))

    # =============================================================================
    # Summary
    # =============================================================================

    total = time.perf_counter() - pipeline_start
    step(100, "Sources updated", f"{len(data):,} documents, {total_new} new ({_fmt(total)})")

    # =============================================================================
    # Save Run Metadata
    # =============================================================================

    from datetime import datetime, timezone

    run_data = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_secs": round(total, 2),
        "new_documents": total_new,
        "total_documents": len(data),
        "success": True,
        "timings": [{"step": label, "duration_secs": round(elapsed, 2)} for label, elapsed in timings],
    }
    if use_pg:
        save_generated("pipeline_run", run_data)

    print("\n" + "=" * 60)
    print("  Pipeline timing summary")
    print("=" * 60)
    for label, elapsed in timings:
        pct = elapsed / total * 100
        print(f"  {label:<25s} {_fmt(elapsed):>8s}  ({pct:4.1f}%)")
    print(f"  {'─' * 41}")
    print(f"  {'Total':<25s} {_fmt(total):>8s}")
    print("=" * 60)
