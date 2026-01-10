#!/usr/bin/env python3
"""
Knowledge Database Builder

This script aggregates knowledge from multiple sources (GitHub, HackerNews,
Zotero, Semanlink, HuggingFace) into a unified database and builds the
search pipeline.

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

Configuration
-------------
sources.yml : file
    YAML configuration specifying which sources to enable.

Output Files
------------
database/database.json : JSON
    Aggregated document database.
database/triples.json : JSON
    Tag co-occurrence graph edges.
database/pipeline.pkl : Pickle
    Serialized search pipeline for the API.
"""

import json
import os
import pickle

import yaml

from knowledge_database import (
    github,
    hackernews,
    huggingface,
    pipeline,
    semanlink,
    tags,
    zotero,
)

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

# =============================================================================
# Load Existing Database
# =============================================================================

data: dict = {}

if os.path.exists("database/database.json"):
    with open("database/database.json", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

# =============================================================================
# Fetch Data from Sources
# =============================================================================


def merge_new_documents(existing: dict, new: dict) -> dict:
    """Merge new documents, skipping URLs already in the database."""
    new_only = {url: doc for url, doc in new.items() if url not in existing}
    print(f"Found {len(new_only)} new documents.")
    return {**existing, **new_only}


# GitHub starred repositories
if sources.get("github") is not None:
    print("Fetching GitHub stars...")
    for user in sources["github"]:
        fetcher = github.Github(user=user)
        data = merge_new_documents(data, fetcher())

# HackerNews upvoted posts
if hackernews_username is not None and hackernews_password is not None:
    print("Fetching HackerNews upvotes...")
    fetcher = hackernews.HackerNews(
        username=hackernews_username,
        password=hackernews_password,
    )
    data = merge_new_documents(data, fetcher())
else:
    print("Skipping HackerNews (no credentials).")

# Zotero library
if zotero_library_id is not None and zotero_api_key is not None:
    print("Fetching Zotero library...")
    fetcher = zotero.Zotero(
        library_id=zotero_library_id,
        library_type="group",
        api_key=zotero_api_key,
    )
    data = merge_new_documents(data, fetcher())
else:
    print("Skipping Zotero (no credentials).")

# Semanlink knowledge base
if sources.get("semanlink"):
    print("Fetching Semanlink data...")
    fetcher = semanlink.Semanlink(
        urls=[
            "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sldocs-2023-01-26.ttl",
            "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sltags-2020-11-18.ttl",
        ]
    )
    data = merge_new_documents(data, fetcher())
else:
    print("Skipping Semanlink (disabled).")

# HuggingFace liked items
if huggingface_token is not None and sources.get("huggingface") is not None:
    print("Fetching HuggingFace likes...")
    fetcher = huggingface.HuggingFace(token=huggingface_token)
    data = merge_new_documents(data, fetcher())
else:
    print("Skipping HuggingFace (no token).")

# =============================================================================
# Data Cleaning
# =============================================================================

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

# =============================================================================
# Generate Extra Tags
# =============================================================================

print("Generating extra tags from document content...")
data = tags.get_extra_tags(data=data)

# =============================================================================
# Save Database
# =============================================================================

print("Saving database...")
with open("database/database.json", "w") as f:
    json.dump(data, f, indent=4)

# =============================================================================
# Build Knowledge Graph
# =============================================================================

# Tags to exclude from graph visualization (too generic)
EXCLUDED_TAGS = {
    "twitter": True,
    "github": True,
    "semanlink": True,
    "hackernews": True,
    "arxiv doc": True,
}

print("Building tag co-occurrence graph...")
triples = tags.get_tags_triples(data=data, excluded_tags=EXCLUDED_TAGS)
with open("database/triples.json", "w") as f:
    json.dump(triples, f, indent=4)

# =============================================================================
# Build Search Pipeline
# =============================================================================

print("Building search pipeline...")
knowledge_pipeline = pipeline.Pipeline(
    documents=data,
    triples=triples,
    excluded_tags=EXCLUDED_TAGS,
)

print("Serializing pipeline...")
with open("database/pipeline.pkl", "wb") as f:
    pickle.dump(knowledge_pipeline, f)

print("Done!")
