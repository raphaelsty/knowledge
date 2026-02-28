#!/usr/bin/env python3
"""
One-time migration: import existing JSON files into PostgreSQL.

Usage:
    DATABASE_URL=postgresql://knowledge:knowledge@localhost:5432/knowledge \
        python scripts/migrate_json_to_pg.py

Reads from web/data/ and writes to the documents and generated_data tables.
"""

import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.database import ensure_schema, save_all_documents, save_generated

DATA_DIR = Path("web/data")


def main():
    print("Ensuring schema exists...")
    ensure_schema()

    # Migrate database.json → documents table
    db_path = DATA_DIR / "database.json"
    if db_path.exists():
        with open(db_path) as f:
            data = json.load(f)
        print(f"Migrating {len(data)} documents from database.json...")
        save_all_documents(data)
        print("  Done.")
    else:
        print(f"  Skipping: {db_path} not found")

    # Migrate generated JSON files → generated_data table
    for key, filename in [
        ("folder_tree", "folder_tree.json"),
        ("sources", "sources.json"),
        ("tree", "tree.json"),
    ]:
        path = DATA_DIR / filename
        if path.exists():
            with open(path) as f:
                blob = json.load(f)
            print(f"Migrating {filename} → generated_data['{key}']...")
            save_generated(key, blob)
            print("  Done.")
        else:
            print(f"  Skipping: {path} not found")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
