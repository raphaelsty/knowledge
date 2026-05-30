#!/usr/bin/env python3
"""Keep the single cross-personality `__all__` search index current.

There is ONE search index, `__all__`. It is maintained INCREMENTALLY and
in place — never rebuilt from scratch unless it is structurally broken:

  • The per-doc `documents.indexed_all` flag is the cursor. Docs with
    `indexed_all = FALSE` are not yet in `__all__`; the sync streams
    them (newest first) into the live index via `update_with_encoding`
    and flips them TRUE. New docs default FALSE, so they flow in on the
    next sweep automatically. The flag makes the sync RESUMABLE: a sync
    killed mid-way (deploy / OOM / crash) picks up exactly where it
    stopped, so `__all__` always converges to the full corpus.

  • Soft-deleted docs that are still in the index (`deleted = TRUE AND
    indexed_all = TRUE`) are removed from `__all__` and their flag
    cleared.

  • REBUILD FROM SCRATCH happens ONLY when the live index won't load
    (HTTP 404/5xx) — we drop it, create a fresh empty one, reset every
    VIP doc to `indexed_all = FALSE`, and let the incremental sync refill
    it (newest first, so the feed is useful within minutes). A merely
    partial or behind index is never dropped — it's just topped up.

Memory is bounded: the sync pulls a batch at a time (keyset/cursor via
the shrinking `idx_documents_unsynced_all` partial index) and never
loads the corpus into RAM. Pushes survive transient API failures (an API
container restarting under a deploy) by retrying, so a sync is not lost
to an interruption.

Usage::

    DATABASE_URL=postgresql://...  ADMIN_API_KEY=secret  \
    API_URL=http://localhost:8080  uv run python -m sources.utils.build_all_index

The daemon calls `sync_all_index(...)` every sweep with a bounded
`max_docs`; the CLI / `make index-all` runs it unbounded to drain fully.
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
BATCH = 300
DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

# Columns + join shared by the sync queries. `d.user_id` trails so we can
# mark (user_id, url) after a batch is pushed.
_VIP_COLS = "d.url, d.title, d.summary, d.date, d.tags, d.extra_tags, d.source, d.source_url, u.username, d.user_id"
_VIP_JOIN = "FROM documents d JOIN users u ON u.id = d.user_id WHERE u.vip = TRUE"


# ── shaping ───────────────────────────────────────────────────────────


def _row_to_text_meta(row: tuple) -> tuple[str, dict] | None:
    """Shape one DB row (first 9 cols of `_VIP_COLS`) into the
    (text, metadata) pair the index ingests. Returns None when the row
    has no indexable text."""
    url, title, summary, date, tags, extra_tags, source, source_url, slug = row[:9]
    doc_tags = list(tags) if tags else []
    extra = list(extra_tags) if extra_tags else []
    summary = summary or ""
    title = title or ""
    source = source or ""
    website = website_name(url)
    text = f"{title} {' '.join(doc_tags)} {' '.join(extra)} {summary[:200]} {source} {website}".strip()
    if not text:
        return None
    meta = {
        "url": url,
        "title": title,
        "summary": summary,
        "date": str(date or ""),
        "tags": ",".join(doc_tags),
        "extra_tags": ",".join(extra),
        "source": source,
        "source_url": source_url or "",
        # owner drives the per-personality pre-filter on `__all__`
        "owner": slug or "",
    }
    return text, meta


# ── HTTP ──────────────────────────────────────────────────────────────


def _post(api_base: str, path: str, payload: dict, headers: dict, timeout: int) -> tuple[int, str]:
    """POST JSON, return (status, body). Never raises — connection-level
    failures (API restarting under a deploy, timeout, DNS blip) come back
    as status ``0`` so callers can treat them as transient and retry."""
    req = urllib.request.Request(f"{api_base}{path}", data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
        return e.code, body
    except Exception as exc:  # noqa: BLE001 — URLError, socket.timeout, ConnectionReset, …
        return 0, f"connection error: {exc}"


def _index_status(api_base: str, name: str, timeout: int = 10) -> int:
    """GET /indices/{name} → HTTP status (0 on connection error). 200 =
    loads fine; 404 = missing; 5xx = won't load (corrupt)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{api_base}/indices/{name}"), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return 0


def _push_one_batch(api_base: str, headers: dict, batch_texts: list, batch_meta: list, *, label_idx: str) -> bool:
    """Push one batch into the LIVE `__all__`, RETRYING through transient
    failures (connection error / queue-full 503 / 429 / 502 / 504) so a
    sync survives a deploy restarting the API instead of aborting. A real
    4xx is not retried. Returns True on success."""
    BACKOFF = (2, 4, 8, 16, 30)
    MAX_RETRY_SECONDS = 900.0
    attempt = 0
    t0 = time.perf_counter()
    while True:
        status, body = _post(
            api_base,
            f"/indices/{INDEX_NAME}/update_with_encoding",
            {"documents": batch_texts, "metadata": batch_meta, "pool_factor": 2},
            headers,
            timeout=600,
        )
        transient = status == 0 or status in (429, 502, 503, 504)
        if transient and (time.perf_counter() - t0) < MAX_RETRY_SECONDS:
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            attempt += 1
            if attempt == 1 or attempt % 5 == 0:
                reason = "connection error" if status == 0 else f"HTTP {status}"
                print(
                    f"    batch {label_idx} ⏸ {reason} (attempt {attempt}, sleeping {wait}s)... {body[:80]}", flush=True
                )
            time.sleep(wait)
            continue
        break
    if status < 200 or status >= 300:
        print(f"    ⚠ batch {label_idx} failed after {attempt} retries: {status} {body[:200]}", flush=True)
        return False
    return True


# ── index lifecycle (rebuild-on-break only) ──────────────────────────


def _recreate_empty(api_base: str, headers: dict, database_url: str) -> None:
    """Drop the broken `__all__`, create a fresh empty one, and reset
    every VIP doc to `indexed_all = FALSE` so the incremental sync
    refills it from scratch. Only ever called when the index won't load."""
    print(f"  [REBUILD] '{INDEX_NAME}' won't load — recreating empty + resetting sync flags...", flush=True)
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{api_base}/indices/{INDEX_NAME}", headers=headers, method="DELETE"), timeout=60
        )
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠ delete '{INDEX_NAME}' returned HTTP {e.code}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ delete '{INDEX_NAME}' failed: {exc}", flush=True)
    time.sleep(2.0)  # let mmap handles release before recreate
    status, body = _post(api_base, "/indices", {"name": INDEX_NAME, "config": {"nbits": 2}}, headers, timeout=30)
    if status not in (200, 201):
        raise RuntimeError(f"create '{INDEX_NAME}' failed: HTTP {status} {body[:200]}")
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE documents AS d SET indexed_all = FALSE "
            "FROM users u WHERE d.user_id = u.id AND u.vip AND d.indexed_all"
        )
        conn.commit()
    print(f"  [REBUILD] empty '{INDEX_NAME}' ready; sync will refill it newest-first.", flush=True)


def _ensure_loads(api_base: str, headers: dict, database_url: str) -> bool:
    """If `__all__` is missing (404) or corrupt (5xx/connection), recreate
    it empty and reset flags. Returns True iff it had to rebuild. A
    loads-fine-but-partial index is left alone (the sync tops it up)."""
    status = _index_status(api_base, INDEX_NAME)
    if status == 200:
        return False
    if status == 0:
        # API itself unreachable — not an index break; skip this sweep.
        print(f"  '{INDEX_NAME}' status check: API unreachable — skipping sync this sweep.", flush=True)
        return False
    _recreate_empty(api_base, headers, database_url)
    return True


# ── incremental sync ──────────────────────────────────────────────────


def _mark_synced(database_url: str, keys: list[tuple[int, str]], synced: bool) -> None:
    """Flip `indexed_all` for a batch of (user_id, url) pairs."""
    if not keys:
        return
    uids = [k[0] for k in keys]
    urls = [k[1] for k in keys]
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE documents AS d SET indexed_all = %s "
            "FROM (SELECT unnest(%s::bigint[]) AS uid, unnest(%s::text[]) AS url) v "
            "WHERE d.user_id = v.uid AND d.url = v.url",
            (synced, uids, urls),
        )
        conn.commit()


def _clear_deleted(database_url: str, api_base: str, headers: dict, max_docs: int | None) -> int:
    """Remove soft-deleted docs that are still in `__all__`, newest-irrelevant
    order, and clear their flag. Returns the count removed."""
    removed = 0
    sql = (
        "SELECT d.url, d.user_id FROM documents d JOIN users u ON u.id = d.user_id "
        "WHERE u.vip AND d.deleted AND d.indexed_all LIMIT %s"
    )
    while max_docs is None or removed < max_docs:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(sql, (BATCH,))
            rows = cur.fetchall()
        if not rows:
            break
        urls = [r[0] for r in rows]
        placeholders = ", ".join(["?"] * len(urls))
        req = urllib.request.Request(
            f"{api_base}/indices/{INDEX_NAME}/documents",
            data=json.dumps({"condition": f"url IN ({placeholders})", "parameters": urls}).encode(),
            headers=headers,
            method="DELETE",
        )
        try:
            urllib.request.urlopen(req, timeout=120)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ clear_deleted batch failed: {exc}", flush=True)
            break
        _mark_synced(database_url, [(r[1], r[0]) for r in rows], synced=False)
        removed += len(rows)
    if removed:
        print(f"  removed {removed:,} deleted doc(s) from '{INDEX_NAME}'.", flush=True)
    return removed


def _add_missing(database_url: str, api_base: str, headers: dict, max_docs: int | None) -> int:
    """Stream VIP docs not yet in `__all__` (newest first) into the live
    index, marking each batch synced. The `indexed_all` flag is the
    cursor, so this is resumable and bounded in memory. Returns the count
    added."""
    added = 0
    sql = (
        f"SELECT {_VIP_COLS} {_VIP_JOIN} "
        "AND NOT d.indexed_all AND NOT d.deleted "
        "ORDER BY d.date DESC NULLS LAST LIMIT %s"
    )
    batch_no = 0
    while max_docs is None or added < max_docs:
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(sql, (BATCH,))
            rows = cur.fetchall()
        if not rows:
            break
        texts: list[str] = []
        metas: list[dict] = []
        keys: list[tuple[int, str]] = []
        for row in rows:
            tm = _row_to_text_meta(row)
            keys.append((row[9], row[0]))  # (user_id, url) — marked regardless of text
            if tm:
                texts.append(tm[0])
                metas.append(tm[1])
        batch_no += 1
        if texts:
            ok = _push_one_batch(api_base, headers, texts, metas, label_idx=str(batch_no))
            if not ok:
                # transient budget exhausted — leave flags FALSE, retry next sweep
                break
        # Mark the whole SELECTed batch synced (incl. empty-text rows, which
        # we intentionally skip indexing but never want to re-scan forever).
        _mark_synced(database_url, keys, synced=True)
        added += len(rows)
        if batch_no == 1 or batch_no % 20 == 0:
            print(f"    synced batch {batch_no} (+{added:,} docs into '{INDEX_NAME}')", flush=True)
    return added


def sync_all_index(
    database_url: str, api_base: str, admin_key: str | None = None, *, max_docs: int | None = None
) -> dict:
    """One incremental maintenance pass for `__all__`:

      1. ensure it loads (recreate empty + reset flags only if broken),
      2. remove soft-deleted docs still in it,
      3. stream in up to `max_docs` not-yet-synced docs (newest first).

    `max_docs=None` drains everything (CLI / manual). The daemon passes a
    bound so each sweep stays responsive and re-checks health often.
    Returns a small summary dict."""
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["X-API-Key"] = admin_key

    rebuilt = _ensure_loads(api_base, headers, database_url)
    removed = _clear_deleted(database_url, api_base, headers, max_docs)
    added = _add_missing(database_url, api_base, headers, max_docs)
    return {"rebuilt": rebuilt, "removed": removed, "added": added}


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_base = os.environ.get("API_URL", "http://localhost:8080")
    api_key = os.environ.get("ADMIN_API_KEY", "")
    t0 = time.perf_counter()
    print(f"Syncing '{INDEX_NAME}' from {database_url}", flush=True)
    summary = sync_all_index(database_url, api_base, api_key or None, max_docs=None)
    print(
        f"\nDone — rebuilt={summary['rebuilt']} removed={summary['removed']:,} "
        f"added={summary['added']:,} in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
