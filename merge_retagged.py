#!/usr/bin/env python3
"""Merge retagged batch files back into database.json and show stats."""

import json
import os
import sys
from collections import Counter

DATABASE_PATH = "database/database.json"
BATCH_DIR = "database"
NUM_BATCHES = 15


def compute_stats(db):
    all_tags = []
    tag_counts_per_doc = []
    docs_with_0 = 0
    docs_with_1 = 0

    for url, doc in db.items():
        tags = doc.get("tags", []) + doc.get("extra-tags", [])
        tag_counts_per_doc.append(len(tags))
        all_tags.extend(tags)
        if len(tags) == 0:
            docs_with_0 += 1
        elif len(tags) == 1:
            docs_with_1 += 1

    tag_freq = Counter(all_tags)
    unique_tags = len(tag_freq)
    avg_tags = sum(tag_counts_per_doc) / len(tag_counts_per_doc) if tag_counts_per_doc else 0

    return {
        "total_docs": len(db),
        "unique_tags": unique_tags,
        "avg_tags_per_doc": round(avg_tags, 2),
        "docs_with_0_tags": docs_with_0,
        "docs_with_1_tag": docs_with_1,
        "top_30_tags": tag_freq.most_common(30),
        "tags_with_spaces": sum(1 for t in tag_freq if " " in t),
        "noise_tags": {t: tag_freq[t] for t in ["was", "now", "time", "model", "models", "data", "code", "library", "paper", "ai", "twitter", "hackernews", "raphaelsty"] if t in tag_freq},
    }


def print_stats(label, stats):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total documents:    {stats['total_docs']}")
    print(f"  Unique tags:        {stats['unique_tags']}")
    print(f"  Avg tags/doc:       {stats['avg_tags_per_doc']}")
    print(f"  Docs with 0 tags:   {stats['docs_with_0_tags']}")
    print(f"  Docs with 1 tag:    {stats['docs_with_1_tag']}")
    print(f"  Tags with spaces:   {stats['tags_with_spaces']}")
    if stats['noise_tags']:
        print(f"  Noise tags remaining: {stats['noise_tags']}")
    else:
        print(f"  Noise tags remaining: NONE (clean!)")
    print(f"\n  Top 30 tags:")
    for tag, count in stats['top_30_tags']:
        print(f"    {tag:40s} {count}")


def merge():
    # Load original database
    with open(DATABASE_PATH, "r") as f:
        db = json.load(f)

    print(f"Original database: {len(db)} documents")
    before_stats = compute_stats(db)
    print_stats("BEFORE retagging", before_stats)

    # Check which retagged batch files exist
    missing = []
    retagged_files = []
    for i in range(NUM_BATCHES):
        path = os.path.join(BATCH_DIR, f"batch_{i:02d}_retagged.json")
        if os.path.exists(path):
            retagged_files.append((i, path))
        else:
            missing.append(i)

    if missing:
        print(f"\nWARNING: Missing retagged batches: {missing}")
        print("Proceeding with available batches...")

    # Merge retagged data
    updated_count = 0
    for batch_idx, path in retagged_files:
        with open(path, "r") as f:
            batch = json.load(f)

        for url, retagged_doc in batch.items():
            if url in db:
                db[url]["tags"] = retagged_doc.get("tags", [])
                db[url]["extra-tags"] = retagged_doc.get("extra-tags", [])
                updated_count += 1
            else:
                print(f"  WARNING: URL not in database: {url[:80]}...")

        print(f"  Merged batch {batch_idx:02d}: {len(batch)} docs")

    print(f"\nTotal documents updated: {updated_count}")

    after_stats = compute_stats(db)
    print_stats("AFTER retagging", after_stats)

    # Write updated database
    with open(DATABASE_PATH, "w") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

    print(f"\nDatabase written to {DATABASE_PATH}")
    print(f"Document count preserved: {len(db)}")

    # Cleanup batch files
    if "--cleanup" in sys.argv:
        for i in range(NUM_BATCHES):
            for suffix in ["", "_retagged"]:
                path = os.path.join(BATCH_DIR, f"batch_{i:02d}{suffix}.json")
                if os.path.exists(path):
                    os.remove(path)
                    print(f"  Removed {path}")
        print("Batch files cleaned up.")


if __name__ == "__main__":
    merge()
