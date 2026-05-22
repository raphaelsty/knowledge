"""Long-lived document-categorization daemon.

Runs on prod under systemd (`knowledge-categorize-daemon`). Pulls
batches of uncategorized documents from Postgres, embeds them with
a small static model (Potion), and writes one to three category
slugs per document into `document_category_assignments`.

Why a daemon rather than a step inside run.py
---------------------------------------------
- Categorization should not block fetch / clean / index when those
  finish — keeping it in its own systemd unit means the data plane
  is never gated on it.
- Resource posture: niced to 19, CPU quota capped at 10 % of one
  core, memory capped at 384 MB. Categorization is cheap per doc
  but the corpus is hundreds of thousands of rows, so we want it
  to spread out over many minutes / hours rather than burst.
- The daemon also owns the periodic refresh of the catalogue's
  refined prototypes (see `_refresh_prototypes_if_stale`), which is
  the only step that's not pure streaming.

Memory profile
--------------
Streaming throughout. Never holds more than `BATCH_SIZE` docs in
memory at once. The prototype-refresh step keeps a bounded
per-category min-heap of the top `ANCHOR_MAX_PER_CAT` anchors —
178 cats × 30 anchors × 512 dims × 4 bytes = ~11 MB. The Potion
model itself is ~150 MB RSS. Total expected RSS is < 300 MB even
during the refresh phase.

Ordering
--------
Most-recent-first by `documents.date DESC NULLS LAST, url DESC`,
so a freshly-ingested cohort of VIP tweets gets categorized before
the daemon walks deeper into the historical tail.

Idempotency
-----------
The SQL fetch filters out any (user_id, url) that already has at
least one row in `document_category_assignments`. Re-runs and
restarts never double-write.

Refresh cadence
---------------
Refined category prototypes are blended on the first iteration if
the on-disk cache is missing or older than `REFRESH_INTERVAL_S`
(default 24 h). Refreshing walks the entire corpus once via
streaming top-K anchor heaps — no full-corpus matrix is ever
materialised in memory.

Environment variables
---------------------
  DATABASE_URL                       required, Postgres DSN
  CATEGORIZE_EMBED_MODEL             default minishlab/potion-base-32M
  CATEGORIZE_BATCH_SIZE              default 20
  CATEGORIZE_IDLE_SLEEP_S            default 600
  CATEGORIZE_REFRESH_INTERVAL_S      default 86400  (24 h)
  CATEGORIZE_INTER_BATCH_SLEEP_S     default 0.0    (CPUQuota does the throttling)
"""

from __future__ import annotations

import heapq
import logging
import os
import sys
import time
from itertools import count
from pathlib import Path

import numpy as np
import psycopg

from sources.utils.categorize import (
    ANCHOR_BLEND,
    ANCHOR_MAX_PER_CAT,
    ANCHOR_MIN_COUNT,
    ANCHOR_MIN_GAP,
    ANCHOR_MIN_SCORE,
    CACHE_DIR,
    EMBED_MODEL,
    _load_model,
    _ModelAdapter,
    category_embeddings,
    doc_input_text,
    ensure_schema,
    fetch_categories,
    persist_batch,
)

LOG = logging.getLogger("categorize-daemon")

DATABASE_URL = os.environ.get("DATABASE_URL")

# Higher-confidence thresholds for the daemon than for the
# offline/CLI flow in `sources.utils.categorize`. The user's bar
# here is "don't assign at all if not sure", so we raise the
# absolute floor to 0.55 (was 0.42) and tighten KEEP_RATIO so a
# secondary slug needs to be essentially tied with the primary to
# survive. The daemon overrides categorize.py's module-level
# thresholds via these constants — pick_categories() in the
# library still consults the library's values, so the daemon owns
# its own loop here rather than relying on a re-import.
# Confidence thresholds, env-overridable. Defaults relaxed from the
# v1 0.55 / 0.97 settings because the backfill goal is now "process
# every doc and surface a category when one fits at all". The
# `score` column on document_category_assignments still records
# the raw cosine sim so the UI can later filter out low-confidence
# rows without re-running the daemon. Docs below the floor still
# get marked categorized = TRUE (no assignment row written) so the
# daemon never re-fetches the same low-confidence batch — the
# infinite-loop bug from the v1 0.55 floor.
DAEMON_ABS_FLOOR = float(os.environ.get("CATEGORIZE_ABS_FLOOR", "0.30"))
DAEMON_KEEP_RATIO = float(os.environ.get("CATEGORIZE_KEEP_RATIO", "0.92"))
DAEMON_MAX_CATS = int(os.environ.get("CATEGORIZE_MAX_CATS", "3"))

BATCH_SIZE = int(os.environ.get("CATEGORIZE_BATCH_SIZE", "20"))
IDLE_SLEEP_S = float(os.environ.get("CATEGORIZE_IDLE_SLEEP_S", "600"))
INTER_BATCH_SLEEP_S = float(os.environ.get("CATEGORIZE_INTER_BATCH_SLEEP_S", "0.0"))
REFRESH_INTERVAL_S = float(os.environ.get("CATEGORIZE_REFRESH_INTERVAL_S", str(24 * 3600)))

# Refinement streams the whole corpus in larger batches than the
# assignment loop to keep total wall-clock reasonable. Still small
# enough that peak memory is bounded.
REFRESH_BATCH_SIZE = int(os.environ.get("CATEGORIZE_REFRESH_BATCH_SIZE", "200"))

# Filter: docs whose effective embedding input is shorter than this
# get skipped. Bare `@handle` tweets that the clean daemon couldn't
# extract a body from collapse to a few characters here and produce
# noisy embeddings.
DOC_INPUT_MIN_LEN = 25


# Where the refined prototypes get cached on disk so the daemon
# doesn't have to walk the whole corpus on every restart.
def _proto_cache_path() -> Path:
    return CACHE_DIR / f"refined_protos_{EMBED_MODEL.replace('/', '_')}.npz"


# ── Doc input + filtering ───────────────────────────────────────────


def _doc_text_or_skip(doc: dict) -> str | None:
    text = doc_input_text(doc)
    if len(text) < DOC_INPUT_MIN_LEN:
        return None
    return text


# ── Streaming SQL ───────────────────────────────────────────────────


_DOC_COLS = "user_id, url, source, title, summary, " "clean_title, clean_summary, date"


def _fetch_uncategorized_batch(conn: psycopg.Connection, batch_size: int) -> list[dict]:
    """Pull the next batch of un-categorized documents, newest first.

    Filters on the `categorized` flag (twin of `cleaned`) rather
    than on the presence of an assignment row. That guarantees
    forward progress even when the daemon processes a doc and
    decides not to assign anything (low-confidence match, empty
    cleaned text, etc.) — without the flag the same low-confidence
    doc gets re-fetched forever as long as it has no rows in
    document_category_assignments.

    Newest-first ordering matches the user's priority: a freshly-
    ingested cohort gets categorized before the daemon walks the
    historical tail.
    """
    # JOIN users so VIPs drain first — the daemon is free (local
    # Potion model, no API cost) so it's fine to process non-VIPs
    # too, but the explicit ordering matches the "VIPs first, then
    # non-VIPs" rule the rest of the pipeline (indexer, clean, twitter
    # feeder) already observes.
    sql = f"""
    SELECT {_DOC_COLS}
    FROM documents d
    JOIN users u ON u.id = d.user_id
    WHERE COALESCE(NULLIF(d.clean_summary, ''), NULLIF(d.summary, '')) IS NOT NULL
      AND d.categorized = FALSE
    ORDER BY u.vip DESC, d.date DESC NULLS LAST, d.url DESC
    LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (batch_size,))
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def _mark_categorized(conn: psycopg.Connection, keys: list[tuple[int, str]]) -> None:
    """Flip `documents.categorized = TRUE` for every (user_id, url)
    the daemon just processed — regardless of whether any
    assignment row landed. Idempotent: re-running an already-TRUE
    row is a no-op."""
    if not keys:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE documents SET categorized = TRUE WHERE user_id = %s AND url = %s",
            keys,
        )
    conn.commit()


def _iter_all_documents(conn: psycopg.Connection, batch_size: int):
    """Stream every assignable document in keyset-paginated batches
    (date DESC NULLS LAST, url DESC). Used by the prototype-refresh
    step; never materialises the full corpus in memory.
    """
    last_date = None
    last_url = None
    while True:
        if last_date is None:
            sql = (
                f"SELECT {_DOC_COLS}\n"
                f"FROM documents d\n"
                f"WHERE COALESCE(NULLIF(d.clean_summary, ''), NULLIF(d.summary, '')) IS NOT NULL\n"
                f"ORDER BY d.date DESC NULLS LAST, d.url DESC\n"
                f"LIMIT %s"
            )
            params: tuple = (batch_size,)
        else:
            # Keyset pagination. NULL dates land at the tail of the
            # ordering, so once we've crossed the boundary we keep
            # paginating with COALESCE to a sentinel.
            sql = (
                f"SELECT {_DOC_COLS}\n"
                f"FROM documents d\n"
                f"WHERE COALESCE(NULLIF(d.clean_summary, ''), NULLIF(d.summary, '')) IS NOT NULL\n"
                f"  AND (d.date, d.url) < (%s, %s)\n"
                f"ORDER BY d.date DESC NULLS LAST, d.url DESC\n"
                f"LIMIT %s"
            )
            params = (last_date, last_url, batch_size)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c.name for c in cur.description]
            rows = cur.fetchall()
        if not rows:
            return
        batch = [dict(zip(cols, r, strict=False)) for r in rows]
        yield batch
        last_date = batch[-1]["date"]
        last_url = batch[-1]["url"]


# ── Prototype refresh (streaming, top-K heaps) ──────────────────────


def _refresh_prototypes_streaming(
    conn: psycopg.Connection,
    model: _ModelAdapter,
    cats: list[dict],
    cat_embeds: np.ndarray,
) -> np.ndarray:
    """Walk the whole corpus once in batches, keep the best-anchored
    documents per category in a bounded min-heap, and blend the
    final centroids into the description-based prototypes.

    The min-heap is keyed by anchor score, so when a new candidate
    arrives and the heap is full the lowest-scoring incumbent is
    evicted. Memory stays bounded at ANCHOR_MAX_PER_CAT entries per
    category.

    Returns the refined (n_cats, dim) matrix, L2-normalised.
    """
    LOG.info("refreshing refined prototypes (streaming over corpus)")
    n_cats, dim = cat_embeds.shape
    # heaps[ci] holds (score, sequence_id, embedding_vector) tuples.
    # sequence_id breaks ties deterministically without requiring
    # ndarray comparison (which numpy doesn't define).
    heaps: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(n_cats)]
    seq = count()

    docs_seen = 0
    docs_used = 0
    for batch in _iter_all_documents(conn, REFRESH_BATCH_SIZE):
        # Filter empty inputs.
        kept_docs: list[dict] = []
        kept_inputs: list[str] = []
        for d in batch:
            t = _doc_text_or_skip(d)
            if t is not None:
                kept_docs.append(d)
                kept_inputs.append(t)
        docs_seen += len(batch)
        if not kept_inputs:
            continue
        embeds = model.encode(kept_inputs, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        sims = embeds @ cat_embeds.T  # (B, n_cats)
        top1_idx = np.argmax(sims, axis=1)
        rows_idx = np.arange(len(sims))
        top1_score = sims[rows_idx, top1_idx]
        # top2 via masking.
        masked = sims.copy()
        masked[rows_idx, top1_idx] = -np.inf
        top2_score = masked.max(axis=1)

        for i in range(len(sims)):
            s1 = float(top1_score[i])
            if s1 < ANCHOR_MIN_SCORE:
                continue
            if (s1 - float(top2_score[i])) < ANCHOR_MIN_GAP:
                continue
            ci = int(top1_idx[i])
            heap = heaps[ci]
            entry = (s1, next(seq), embeds[i].copy())
            if len(heap) < ANCHOR_MAX_PER_CAT:
                heapq.heappush(heap, entry)
            else:
                heapq.heappushpop(heap, entry)
            docs_used += 1

        if docs_seen and docs_seen % (REFRESH_BATCH_SIZE * 20) == 0:
            LOG.info(
                "refresh progress: %d docs seen, %d anchors retained",
                docs_seen,
                docs_used,
            )

    LOG.info(
        "refresh: walked %d docs, retained %d anchor candidates",
        docs_seen,
        docs_used,
    )

    # Build refined prototypes from the accumulated heaps.
    refined = cat_embeds.copy()
    refined_cats = 0
    for ci in range(n_cats):
        if len(heaps[ci]) < ANCHOR_MIN_COUNT:
            continue
        centroid = np.mean(np.stack([emb for _, _, emb in heaps[ci]]), axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        blended = ANCHOR_BLEND * cat_embeds[ci] + (1.0 - ANCHOR_BLEND) * centroid
        refined[ci] = blended / (np.linalg.norm(blended) + 1e-9)
        refined_cats += 1
    LOG.info("refresh: refined %d/%d categories", refined_cats, n_cats)

    _proto_cache_path().parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _proto_cache_path(),
        protos=refined,
        ts=np.float64(time.time()),
    )
    return refined


def _load_or_refresh_prototypes(
    conn: psycopg.Connection,
    model: _ModelAdapter,
    cats: list[dict],
    cat_embeds: np.ndarray,
) -> np.ndarray:
    cache = _proto_cache_path()
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            age = time.time() - float(z["ts"])
            if age < REFRESH_INTERVAL_S and z["protos"].shape == cat_embeds.shape:
                LOG.info(
                    "loaded refined prototypes from cache (age %.0f h)",
                    age / 3600.0,
                )
                return z["protos"]
            LOG.info("prototype cache stale (age %.1f h), refreshing", age / 3600.0)
        except Exception as e:
            LOG.warning("prototype cache unreadable, refreshing: %s", e)
    return _refresh_prototypes_streaming(conn, model, cats, cat_embeds)


# ── Assignment loop ─────────────────────────────────────────────────


def _pick_with_daemon_thresholds(sims_row: np.ndarray) -> list[tuple[int, float]]:
    """Same algorithm as `categorize.pick_categories`, but uses the
    daemon's tighter ABS_FLOOR / KEEP_RATIO constants so we can dial
    confidence without editing the library."""
    top = np.argpartition(-sims_row, DAEMON_MAX_CATS)[:DAEMON_MAX_CATS]
    top = top[np.argsort(-sims_row[top])]
    chosen: list[tuple[int, float]] = []
    top1 = float(sims_row[top[0]])
    if top1 < DAEMON_ABS_FLOOR:
        return []
    for rank, idx in enumerate(top):
        score = float(sims_row[idx])
        if rank == 0:
            chosen.append((int(idx), score))
            continue
        if score >= DAEMON_ABS_FLOOR and score >= DAEMON_KEEP_RATIO * top1:
            chosen.append((int(idx), score))
        else:
            break
    return chosen


def _assign_batch(
    conn: psycopg.Connection,
    model: _ModelAdapter,
    cats: list[dict],
    protos: np.ndarray,
    batch: list[dict],
) -> tuple[int, int, int]:
    """Encode, score, persist assignments, and mark every input doc
    `categorized = TRUE` regardless of whether it ended up with an
    assignment row. The flag flip is what guarantees forward
    progress on the backfill — without it, low-confidence /
    short-text docs would be re-fetched on every batch and the
    daemon would never reach the historical tail.

    Returns (assigned_docs, assignments_written, skipped_weak)."""
    kept_docs: list[dict] = []
    kept_inputs: list[str] = []
    skipped_short = 0
    for d in batch:
        t = _doc_text_or_skip(d)
        if t is None:
            skipped_short += 1
            continue
        kept_docs.append(d)
        kept_inputs.append(t)
    rows_out: list[tuple[int, str, int, float, bool]] = []
    assigned = 0
    skipped_weak = skipped_short
    if kept_inputs:
        embeds = model.encode(kept_inputs, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        sims = embeds @ protos.T  # (B, n_cats)
        for doc, sims_row in zip(kept_docs, sims, strict=False):
            picks = _pick_with_daemon_thresholds(sims_row)
            if not picks:
                skipped_weak += 1
                continue
            for rank, (idx, score) in enumerate(picks):
                rows_out.append((doc["user_id"], doc["url"], cats[idx]["id"], score, rank == 0))
            assigned += 1
    if rows_out:
        persist_batch(conn, rows_out)
    # Mark every doc the daemon saw as categorized — assigned OR
    # skipped (either by length guard or weak match). Critical for
    # forward progress, otherwise the same batch comes back forever.
    _mark_categorized(conn, [(d["user_id"], d["url"]) for d in batch])
    return (assigned, len(rows_out), skipped_weak)


# ── Main loop ───────────────────────────────────────────────────────


def run_forever() -> None:
    if not DATABASE_URL:
        LOG.error("DATABASE_URL not set")
        sys.exit(2)
    LOG.info(
        "categorize-daemon up: model=%s batch=%d idle_sleep=%.0fs floor=%.2f",
        EMBED_MODEL,
        BATCH_SIZE,
        IDLE_SLEEP_S,
        DAEMON_ABS_FLOOR,
    )
    model = _load_model()

    # Outer loop runs once per "session": fetch the catalogue + load
    # / refresh prototypes (each in their own short-lived
    # connection, so we never hold the documents table while the
    # CPU-bound work happens), then drop into a tight inner loop
    # that opens one connection per batch. Sleeps always happen
    # outside any `with psycopg.connect()` block — a long-held
    # idle-in-transaction connection blocks ALTER TABLE / UPDATE
    # auth_sessions / every endpoint that touches users, which is
    # what wedged prod earlier this afternoon.
    while True:
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                ensure_schema(conn)
                cats = fetch_categories(conn)
            if not cats:
                LOG.warning("document_categories empty; sleeping")
                time.sleep(IDLE_SLEEP_S)
                continue
            cat_embeds = category_embeddings(model, cats)
            # Prototype refresh: streams the whole corpus and is
            # I/O-light per query but takes minutes total. Use a
            # dedicated autocommit connection so each SELECT closes
            # its transaction as soon as the rows come back —
            # otherwise the encoding step between fetches sat
            # holding a documents-table transaction.
            with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
                protos = _load_or_refresh_prototypes(conn, model, cats, cat_embeds)
        except KeyboardInterrupt:
            LOG.info("interrupted, exiting")
            return
        except Exception as e:
            LOG.exception("daemon setup crashed: %s — retrying in 60s", e)
            time.sleep(60.0)
            continue

        # Inner loop — one PG connection per batch, opened lazily
        # right before we fetch and closed right after we mark the
        # batch categorized. The connection never outlives the
        # ~1 second of CPU work we do on the batch.
        while True:
            no_work = False
            try:
                with psycopg.connect(DATABASE_URL) as conn:
                    batch = _fetch_uncategorized_batch(conn, BATCH_SIZE)
                    if not batch:
                        no_work = True
                    else:
                        assigned, written, weak = _assign_batch(conn, model, cats, protos, batch)
                        LOG.info(
                            "batch: %d docs in, %d assigned, %d rows, %d skipped",
                            len(batch),
                            assigned,
                            written,
                            weak,
                        )
            except KeyboardInterrupt:
                LOG.info("interrupted, exiting")
                return
            except Exception as e:
                LOG.exception("batch loop crashed: %s — retrying in 60s", e)
                time.sleep(60.0)
                # Restart outer loop so we re-check the catalogue +
                # refresh interval after a connection-level error.
                break
            # Connection released. Now it's safe to sleep.
            if no_work:
                LOG.info("no uncategorized docs; sleeping %.0fs", IDLE_SLEEP_S)
                time.sleep(IDLE_SLEEP_S)
                # Restart outer loop so the prototype cache /
                # catalogue re-check fires periodically.
                break
            if INTER_BATCH_SLEEP_S > 0:
                time.sleep(INTER_BATCH_SLEEP_S)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_forever()


if __name__ == "__main__":
    main()
