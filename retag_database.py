#!/usr/bin/env python3
"""Split database.json into batches for parallel retagging."""

import json
import math
import os

DATABASE_PATH = "database/database.json"
BATCH_DIR = "database"
NUM_BATCHES = 15


def split_into_batches():
    with open(DATABASE_PATH, "r") as f:
        db = json.load(f)

    urls = list(db.keys())
    total = len(urls)
    batch_size = math.ceil(total / NUM_BATCHES)

    print(f"Total documents: {total}")
    print(f"Batch size: ~{batch_size}")
    print(f"Number of batches: {NUM_BATCHES}")

    for i in range(NUM_BATCHES):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_urls = urls[start:end]

        batch_data = {}
        for url in batch_urls:
            doc = db[url]
            batch_data[url] = {
                "title": doc.get("title", ""),
                "summary": doc.get("summary", ""),
                "tags": doc.get("tags", []),
                "extra-tags": doc.get("extra-tags", []),
            }

        batch_path = os.path.join(BATCH_DIR, f"batch_{i:02d}.json")
        with open(batch_path, "w") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        print(f"  Batch {i:02d}: {len(batch_data)} docs → {batch_path}")

    print("\nDone! Batch files created.")


if __name__ == "__main__":
    split_into_batches()
