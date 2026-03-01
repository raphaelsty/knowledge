#!/usr/bin/env python3
"""Place new documents into existing folder tree via embedding similarity."""

import time

from sources.database import load_all_documents, load_generated, save_generated
from sources.taxonomy import load_model, rescue_unplaced_docs


def main():
    folder_tree = load_generated("folder_tree")
    if not folder_tree:
        print("No folder_tree in DB, skipping.")
        return
    database = load_all_documents()
    if not database:
        return
    t0 = time.perf_counter()
    model = load_model("minishlab/potion-base-8M")
    folder_tree, n = rescue_unplaced_docs(folder_tree, database, model)
    if n == 0:
        print(f"No unplaced documents. ({time.perf_counter() - t0:.1f}s)")
        return
    save_generated("folder_tree", folder_tree)
    print(f"Rescued {n} docs into folder tree. ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
