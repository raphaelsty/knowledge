#!/usr/bin/env python3
"""Build the cross-personality `__all__` search index — zero-downtime.

Pulls every document owned by a VIP user from PG, formats it the same
way `sources.utils.client.run_pipeline` does for per-user indices,
and pushes the result into a staging index. Once the build is fully
populated, calls the API's `promote` endpoint to atomically swap the
live `__all__` slot to point at the new corpus.

Flow:

  1. DELETE `__all__staging__` (clean up leftovers from a prior crash).
  2. CREATE `__all__staging__` (fresh, empty).
  3. POST `update_with_encoding` × N batches into `__all__staging__`.
  4. POST `/indices/__all__/promote {"from": "__all__staging__"}`.

The promote handler renames the staging directory into `__all__`'s
slot on disk and calls `state.register_index` which is an ArcSwap on
the live `IndexSlot` — concurrent readers transition between the old
and new index in a single atomic instruction, with no empty window.

If the push fails partway, `__all__` is left untouched and the next
run will retry from scratch by cleaning up `__all__staging__` first.

Usage::

    DATABASE_URL=postgresql://...  ADMIN_API_KEY=secret  \
    API_URL=http://localhost:8080  uv run python -m sources.utils.build_all_index

Or via the Makefile::

    make index-all
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import psycopg

from sources.utils.client import website_name

INDEX_NAME = "__all__"
# Build into the staging name first, then atomically promote into
# INDEX_NAME at the end. Old `__all__` keeps serving the live site
# during the entire rebuild — readers transition in one CPU
# instruction when the promote handler swaps ArcSwap slots.
STAGING_NAME = "__all__staging__"
BATCH = 300

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"


# ── DB ────────────────────────────────────────────────────────────────


def _vip_documents(database_url: str) -> list[tuple[str, dict]]:
    """Return every (url, doc) for users where vip = TRUE.

    `doc` is shaped to match `client._merge_and_track`'s output:
    title, summary, date, tags (list), extra-tags (list),
    source (str), source_url (str).
    """
    sql = (
        "SELECT d.url, d.title, d.summary, d.date, d.tags, d.extra_tags, "
        "       d.source, d.source_url, u.username "
        "  FROM documents d "
        "  JOIN users u ON u.id = d.user_id "
        " WHERE u.vip = TRUE "
        " ORDER BY u.username, d.url"
    )
    out: list[tuple[str, dict]] = []
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        for url, title, summary, date, tags, extra_tags, source, source_url, slug in cur.fetchall():
            doc = {
                "title": title or "",
                "summary": summary or "",
                "date": date or "",
                "tags": list(tags) if tags else [],
                "extra-tags": list(extra_tags) if extra_tags else [],
                "source": source or "",
                "source_url": source_url or "",
                # `_owner` is just informational — the API doesn't read
                # it. Useful when grepping the index payload to figure
                # out which personality contributed a given URL.
                "_owner": slug,
            }
            out.append((url, doc))
    return out


# ── HTTP helpers ─────────────────────────────────────────────────────


def _post(api_base: str, path: str, payload: dict, headers: dict, timeout: int) -> tuple[int, str]:
    """POST JSON, return (status, body). Never raises on HTTP errors."""
    req = urllib.request.Request(
        f"{api_base}{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
        return e.code, body


def _reset_staging(api_base: str, headers: dict) -> None:
    """Delete any leftover staging index from a previous failed run,
    then declare a fresh empty staging index.

    The live `__all__` index is NEVER touched here — that's the whole
    point. Old search keeps working until we promote at the end.
    """
    # Drop any leftover staging from a previous crashed rebuild.
    req = urllib.request.Request(
        f"{api_base}/indices/{STAGING_NAME}",
        headers=headers,
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=60):
            print(f"  Cleaned up leftover '{STAGING_NAME}' from a prior run.")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  No leftover '{STAGING_NAME}' to clean up.")
        else:
            body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            print(f"  ⚠ staging drop returned HTTP {e.code}: {body[:200]}")
    except Exception as exc:
        print(f"  ⚠ staging drop failed: {exc}")

    # Brief pause so any mmap handles release before the next create.
    time.sleep(2.0)

    print(f"  Declaring fresh staging index '{STAGING_NAME}'...")
    status, body = _post(
        api_base,
        "/indices",
        {"name": STAGING_NAME, "config": {"nbits": 2}},
        headers,
        timeout=30,
    )
    if status not in (200, 201):
        raise RuntimeError(f"create staging index failed: HTTP {status} {body[:200]}")


def _promote_staging(api_base: str, headers: dict) -> None:
    """Atomically replace `__all__` with the contents of the freshly-
    built staging index. Implemented server-side via an ArcSwap on the
    in-memory `IndexSlot`, so concurrent readers transition in a
    single atomic instruction with no empty window."""
    print(f"  Promoting '{STAGING_NAME}' → '{INDEX_NAME}'...")
    status, body = _post(
        api_base,
        f"/indices/{INDEX_NAME}/promote",
        {"from": STAGING_NAME},
        headers,
        timeout=120,
    )
    if status != 200:
        raise RuntimeError(f"promote failed: HTTP {status} {body[:200]}")
    print(f"  ✓ Promote complete — {INDEX_NAME} now serves the new corpus.")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_base = os.environ.get("API_URL", "http://localhost:8080")
    api_key = os.environ.get("ADMIN_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    t0 = time.perf_counter()
    print(f"Loading VIP documents from {database_url}")
    docs = _vip_documents(database_url)
    print(f"  → {len(docs):,} VIP documents found")

    if not docs:
        print("Nothing to index.")
        return 0

    _reset_staging(api_base, headers)

    # Build text + metadata exactly like run_pipeline does — same
    # ColBERT input shape so search behaves identically.
    texts: list[str] = []
    metas: list[dict] = []
    urls_aligned: list[str] = []
    for url, doc in docs:
        title = doc.get("title", "")
        doc_tags = doc.get("tags") or []
        extra = doc.get("extra-tags") or []
        summary = doc.get("summary", "") or ""
        source = doc.get("source", "") or ""
        website = website_name(url)

        text = f"{title} {' '.join(doc_tags)} {' '.join(extra)} {summary[:200]} {source} {website}".strip()
        if not text:
            continue
        texts.append(text)
        urls_aligned.append(url)
        metas.append(
            {
                "url": url,
                "title": title,
                "summary": summary,
                "date": str(doc.get("date") or ""),
                "tags": ",".join(doc_tags),
                "extra_tags": ",".join(extra),
                "source": source,
                "source_url": doc.get("source_url") or "",
                # carry the owner so the frontend can attribute results
                "owner": doc.get("_owner") or "",
            }
        )

    n = len(texts)
    if n == 0:
        print("No indexable text after filtering.")
        return 0

    # Index is freshly created above — no per-URL pre-purge needed.
    # Push in batches of `BATCH`.
    n_batches = (n + BATCH - 1) // BATCH
    print(f"  Indexing {n:,} documents in {n_batches} batch(es) of {BATCH}...", flush=True)
    pushed = 0
    # The API's per-index update queue is capped at 100 pending items
    # server-side. Push too fast and we get 503 SERVICE_UNAVAILABLE
    # ("Update queue full"). The fix is to back off and RETRY the same
    # batch — never advance on a 503, since the data wasn't accepted.
    # Exponential backoff avoids hammering: 2s → 4s → 8s → 16s → 30s
    # (cap), with unlimited retries for queue-full.
    QUEUE_FULL_BACKOFF = (2, 4, 8, 16, 30)
    for i in range(0, n, BATCH):
        batch_texts = texts[i : i + BATCH]
        batch_meta = metas[i : i + BATCH]
        attempt = 0
        while True:
            status, body = _post(
                api_base,
                f"/indices/{STAGING_NAME}/update_with_encoding",
                {"documents": batch_texts, "metadata": batch_meta, "pool_factor": 2},
                headers,
                timeout=600,
            )
            # Treat "queue full" as backpressure, not a hard failure.
            if status == 503 and "queue full" in body.lower():
                wait = QUEUE_FULL_BACKOFF[min(attempt, len(QUEUE_FULL_BACKOFF) - 1)]
                attempt += 1
                if attempt == 1 or attempt % 5 == 0:
                    print(
                        f"    batch {i // BATCH + 1}/{n_batches} ⏸ queue full (attempt {attempt}, sleeping {wait}s)...",
                        flush=True,
                    )
                time.sleep(wait)
                continue
            break

        if status >= 400:
            print(
                f"    ⚠ batch {i // BATCH + 1}/{n_batches} failed: {status} {body[:200]}",
                flush=True,
            )
            continue
        pushed += len(batch_texts)
        print(
            f"    batch {i // BATCH + 1}/{n_batches} ✓ ({pushed:,}/{n:,} pushed)",
            flush=True,
        )

    # Only promote when the entire push succeeded. A partial staging
    # index must not replace the live `__all__` — better to keep the
    # old one and let the next run try again.
    if pushed != n:
        print(f"\n[!] Skipping promote — only {pushed:,}/{n:,} pushed. '{INDEX_NAME}' left untouched.")
        return 1

    _promote_staging(api_base, headers)

    elapsed = time.perf_counter() - t0
    print(f"\nDone — {pushed:,} documents indexed into '{INDEX_NAME}' (via promote) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
