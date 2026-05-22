"""Ad-hoc: re-index every `indexed = FALSE` doc for a given user,
skipping the timeline-fetch phase of the full pipeline.

Used after a metadata-only PG update (e.g. media-URL backfill) where
we need the ColBERT sidecar to pick up the new summary without the
~5 min Twitter pagination cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_BASE = "http://localhost:8080"
BATCH = 300
DELETE_CHUNK = 200


def website_name(url: str) -> str:
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc.lower().removeprefix("www.")
        return host or ""
    except Exception:
        return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True)
    args = p.parse_args()

    db_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_base = os.environ.get("API_URL", DEFAULT_API_BASE).rstrip("/")
    admin_key = os.environ.get("ADMIN_API_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["X-API-Key"] = admin_key

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT u.id, COALESCE(NULLIF(u.index_name, ''), u.username), "
                "       d.url, d.title, d.summary, "
                "       COALESCE(to_char(d.date, 'YYYY-MM-DD'), ''), "
                "       d.tags, d.extra_tags, d.source, d.source_url "
                "  FROM documents d JOIN users u ON u.id = d.user_id "
                " WHERE u.username = %s AND d.indexed = FALSE AND d.deleted = FALSE",
                (args.slug,),
            )
            rows = cur.fetchall()

    if not rows:
        print(f"No unindexed docs for {args.slug}.")
        return

    uid = rows[0][0]
    index_name = rows[0][1]
    print(f"Re-indexing {len(rows)} doc(s) for {args.slug} → index='{index_name}'")

    # 1) Delete existing chunks for these URLs.
    urls = [r[2] for r in rows]
    for i in range(0, len(urls), DELETE_CHUNK):
        chunk = urls[i : i + DELETE_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        payload = json.dumps({"condition": f"url IN ({placeholders})", "parameters": chunk}).encode()
        req = urllib.request.Request(
            f"{api_base}/indices/{index_name}/documents",
            data=payload,
            headers=headers,
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=60):
                pass
        except Exception as e:
            print(f"  pre-delete batch {i // DELETE_CHUNK + 1}: {e}", file=sys.stderr)
    print("  delete-then-upsert: waiting 5s for index merge queue…")
    time.sleep(5)

    # 2) Build batches + POST update_with_encoding.
    docs: list[str] = []
    meta: list[dict] = []
    for _uid, _idx, url, title, summary, date, tags, extra, source, source_url in rows:
        tags = tags or []
        extra = extra or []
        text = (
            f"{title or ''} {' '.join(tags)} {' '.join(extra)} "
            f"{(summary or '')[:200]} {source or ''} {website_name(url)}"
        ).strip()
        if not text:
            continue
        docs.append(text)
        meta.append(
            {
                "url": url,
                "title": title or "",
                "summary": summary or "",
                "date": date or "",
                "tags": ",".join(tags),
                "extra_tags": ",".join(extra),
                "source": source or "",
                "source_url": source_url or "",
            }
        )

    posted: list[str] = []
    n_batches = (len(docs) + BATCH - 1) // BATCH
    for i in range(0, len(docs), BATCH):
        body = json.dumps(
            {
                "documents": docs[i : i + BATCH],
                "metadata": meta[i : i + BATCH],
                "pool_factor": 2,
            }
        ).encode()
        req = urllib.request.Request(
            f"{api_base}/indices/{index_name}/update_with_encoding",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp.read()
            posted.extend(m["url"] for m in meta[i : i + BATCH])
            print(f"  batch {i // BATCH + 1}/{n_batches}: {len(posted)} posted")
        except Exception as e:
            print(f"  batch {i // BATCH + 1}: {e}", file=sys.stderr)

    # 3) Flip indexed=TRUE for everything that landed.
    if posted:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET indexed = TRUE, updated_at = now()" "  WHERE user_id = %s AND url = ANY(%s)",
                    (uid, posted),
                )
            conn.commit()
        print(f"  PG: indexed=TRUE for {len(posted)} rows")
    print("Done.")


if __name__ == "__main__":
    main()
