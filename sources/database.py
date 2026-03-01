"""
PostgreSQL database abstraction for the knowledge base.

Provides functions to load/save documents and generated data blobs,
replacing the previous JSON file storage.

Environment Variables
---------------------
DATABASE_URL : str
    PostgreSQL connection string.
    Default: postgresql://knowledge:knowledge@localhost:5432/knowledge
"""

import json
import os
from datetime import date

import psycopg
from psycopg.rows import dict_row

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    url         TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    date        DATE,
    tags        TEXT[] NOT NULL DEFAULT '{}',
    extra_tags  TEXT[] NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_date ON documents (date DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN (tags);

CREATE TABLE IF NOT EXISTS generated_data (
    key         TEXT PRIMARY KEY,
    data        JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS favorites (
    url         TEXT PRIMARY KEY REFERENCES documents(url) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _get_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def _connect():
    return psycopg.connect(_get_url(), row_factory=dict_row)


def ensure_schema() -> None:
    """Create tables and indices if they don't exist."""
    with _connect() as conn:
        conn.execute(_SCHEMA_SQL)


def _parse_date(d: str) -> date | None:
    """Parse a date string (YYYY-MM-DD or similar) into a date object."""
    if not d:
        return None
    try:
        # Handle common formats
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%B %d, %Y"):
            try:
                from datetime import datetime

                return datetime.strptime(d, fmt).date()
            except ValueError:
                continue
        return None
    except Exception:
        return None


def load_all_documents() -> dict[str, dict]:
    """Load all documents from PG, returning the same shape as the JSON file.

    Returns a dict keyed by URL, each value having:
      title, summary, date, tags, extra-tags
    """
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY date DESC NULLS LAST").fetchall()

    data = {}
    for row in rows:
        data[row["url"]] = {
            "title": row["title"],
            "summary": row["summary"],
            "date": row["date"].isoformat() if row["date"] else "",
            "tags": list(row["tags"]) if row["tags"] else [],
            "extra-tags": list(row["extra_tags"]) if row["extra_tags"] else [],
        }
    return data


def save_all_documents(data: dict[str, dict]) -> None:
    """Batch upsert all documents into PG."""
    with _connect() as conn:
        with conn.cursor() as cur:
            for url, doc in data.items():
                tags = doc.get("tags", [])
                extra_tags = doc.get("extra-tags", [])
                parsed_date = _parse_date(doc.get("date", ""))

                cur.execute(
                    """
                    INSERT INTO documents (url, title, summary, date, tags, extra_tags)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                        title = EXCLUDED.title,
                        summary = EXCLUDED.summary,
                        date = EXCLUDED.date,
                        tags = EXCLUDED.tags,
                        extra_tags = EXCLUDED.extra_tags,
                        updated_at = now()
                    """,
                    (
                        url,
                        doc.get("title", ""),
                        doc.get("summary", ""),
                        parsed_date,
                        tags,
                        extra_tags,
                    ),
                )


def save_generated(key: str, data) -> None:
    """Save a generated data blob (folder_tree, sources, tree) as JSONB."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO generated_data (key, data)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET
                data = EXCLUDED.data,
                updated_at = now()
            """,
            (key, json.dumps(data)),
        )


def load_generated(key: str):
    """Load a generated data blob by key. Returns None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT data FROM generated_data WHERE key = %s",
            (key,),
        ).fetchone()

    if row is None:
        return None
    return row["data"]


def load_favorites() -> list[str]:
    """Return list of favorited document URLs."""
    with _connect() as conn:
        rows = conn.execute("SELECT url FROM favorites ORDER BY created_at DESC").fetchall()
    return [row["url"] for row in rows]


def toggle_favorite(url: str) -> bool:
    """Toggle favorite status for a URL. Returns True if now favorited, False if removed."""
    with _connect() as conn:
        existing = conn.execute("SELECT url FROM favorites WHERE url = %s", (url,)).fetchone()
        if existing:
            conn.execute("DELETE FROM favorites WHERE url = %s", (url,))
            return False
        else:
            conn.execute("INSERT INTO favorites (url) VALUES (%s)", (url,))
            return True
