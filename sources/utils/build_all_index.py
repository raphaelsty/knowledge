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

Broken-index fast path: if the live `__all__` is classified unusable
(broken / error / empty / missing) at the start, the build first
pushes a small RECENT-first seed (``BUILD_ALL_SEED_DOCS``, default
1000) and promotes it as soon as it loads + searches — restoring the
feed within seconds — then proceeds to the full rebuild and promotes
the complete corpus. When the live index is healthy this phase is
skipped and the classic promote-at-end (true zero-downtime) contract
holds.

Every promote is guarded: we drain the async write queue and run a
real semantic query against the staging index, and only swap it into
the live slot if it actually loads and returns hits. A truncated /
half-written staging (the failure that bricked the live index with
"NPY file too small") can therefore never replace a working slot.

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


def _date_sort_key(meta: dict) -> str:
    """Sort key for recency. DB `date` values are ISO strings
    (``YYYY-MM-DD``) which compare correctly as plain strings; anything
    empty or non-ISO (no leading digit) collapses to ``""`` so it sorts
    LAST under a descending sort and never crowds out real dates in the
    seed."""
    d = meta.get("date") or ""
    return d if d[:1].isdigit() else ""


def _wait_for_index(api_base: str, headers: dict, name: str, expected_docs: int, timeout: float = 180.0) -> bool:
    """Block until `name` reports >= `expected_docs` docs AND embeddings.

    `update_with_encoding` is async — the server queues each batch and a
    background worker flushes it to disk, so a freshly-pushed staging
    index can still 404 (or report a stale count / zero embeddings) the
    instant the last batch POST returns. We must let the queue fully
    drain before promote, otherwise the rename moves a half-written
    index into the live slot — which is how `__all__` ended up with a
    truncated `.npy` ("NPY file too small") and 500-ing every read.

    Requires ``num_embeddings > 0`` as well as the doc count: a non-zero
    doc count with zero embeddings means the residual/centroid files
    aren't on disk yet, and promoting then yields a load-broken index.
    Returns True once both are satisfied, False on timeout.
    """
    deadline = time.perf_counter() + timeout
    last_docs = last_emb = -1
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{api_base}/indices/{name}"), timeout=15) as r:
                payload = json.loads(r.read().decode())
            last_docs = int(payload.get("num_documents") or 0)
            last_emb = int(payload.get("num_embeddings") or 0)
            if last_docs >= expected_docs and last_emb > 0:
                return True
        except urllib.error.HTTPError:
            pass  # 404 until the first batch flushes
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    print(
        f"  ⚠ '{name}' reached only {last_docs}/{expected_docs} docs, "
        f"{last_emb} embeddings within {timeout:.0f}s before promote."
    )
    return False


def _verify_searchable(api_base: str, headers: dict, name: str) -> bool:
    """Final gate before promote: prove the index actually LOADS and
    answers a semantic query end-to-end.

    A healthy doc/embedding count isn't enough — the on-disk `.npy`
    residuals can still be truncated (the exact failure that bricked the
    live index: ``Index load failed: NPY file too small``). The only way
    to be sure is to make the API load + search it. We hit the embedding
    path (`search_with_encoding`) — the same one the feed UI uses — and
    require a 200 with at least one hit. If this fails we REFUSE to
    promote, so a corrupt staging can never replace the live slot.
    """
    try:
        status, body = _post(
            api_base,
            f"/indices/{name}/search_with_encoding",
            {"queries": ["transformer language model"], "k": 5},
            headers,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ verify '{name}': search raised {exc!r}")
        return False
    if status != 200:
        print(f"  ✗ verify '{name}': search HTTP {status} {body[:160]}")
        return False
    try:
        hits = json.loads(body)["results"][0]["document_ids"]
    except Exception:  # noqa: BLE001
        print(f"  ✗ verify '{name}': unparseable search response {body[:160]}")
        return False
    if not hits:
        print(f"  ✗ verify '{name}': search returned zero hits")
        return False
    print(f"  ✓ verify '{name}': loads + searches ({len(hits)} hits)")
    return True


def _guarded_promote(api_base: str, headers: dict, expected_docs: int, label: str) -> bool:
    """Wait for the staging queue to drain, prove it loads + searches,
    and only THEN promote. Refuses to swap a non-searchable staging into
    the live slot — this is the invariant that stops a truncated/partial
    build from ever bricking `__all__` again. Returns True iff promoted.
    """
    _wait_for_index(api_base, headers, STAGING_NAME, expected_docs)
    if not _verify_searchable(api_base, headers, STAGING_NAME):
        print(f"  [{label}] ✗ staging failed verification — NOT promoting; live slot left untouched.")
        return False
    _promote_staging(api_base, headers)
    return True


def _push_batches(
    api_base: str,
    headers: dict,
    target: str,
    texts: list[str],
    metas: list[dict],
    *,
    nice_ratio: float,
    label: str = "",
) -> int:
    """Encode + push `texts`/`metas` into `target` in batches of `BATCH`.

    Returns the number of documents successfully pushed. `nice_ratio`
    controls the post-batch encoder-yield pause (the rebuild sleeps for
    ``last_batch_seconds × nice_ratio`` so live `/encode` traffic isn't
    starved); pass ``0.0`` to push back-to-back (used for the urgent
    seed, where restoring search fast matters more than politeness).
    """
    n = len(texts)
    n_batches = (n + BATCH - 1) // BATCH
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Indexing {n:,} documents in {n_batches} batch(es) of {BATCH}...", flush=True)
    pushed = 0
    # The API's per-index update queue is capped at 100 pending items
    # server-side. Push too fast and we get 503 SERVICE_UNAVAILABLE
    # ("Update queue full"). The fix is to back off and RETRY the same
    # batch — never advance on a 503, since the data wasn't accepted.
    QUEUE_FULL_BACKOFF = (2, 4, 8, 16, 30)
    MIN_PAUSE_S = 0.2
    MAX_PAUSE_S = 30.0
    for i in range(0, n, BATCH):
        batch_texts = texts[i : i + BATCH]
        batch_meta = metas[i : i + BATCH]
        attempt = 0
        batch_t0 = time.perf_counter()
        while True:
            status, body = _post(
                api_base,
                f"/indices/{target}/update_with_encoding",
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
                        f"    {tag}batch {i // BATCH + 1}/{n_batches} ⏸ queue full (attempt {attempt}, sleeping {wait}s)...",
                        flush=True,
                    )
                time.sleep(wait)
                continue
            break
        batch_elapsed = time.perf_counter() - batch_t0

        if status >= 400:
            print(
                f"    ⚠ {tag}batch {i // BATCH + 1}/{n_batches} failed: {status} {body[:200]}",
                flush=True,
            )
            continue
        pushed += len(batch_texts)
        print(
            f"    {tag}batch {i // BATCH + 1}/{n_batches} ✓ ({pushed:,}/{n:,} pushed)",
            flush=True,
        )
        # Be nice — yield the encoder. Skip the pause after the last batch.
        if nice_ratio > 0 and i + BATCH < n:
            pause = max(MIN_PAUSE_S, min(MAX_PAUSE_S, batch_elapsed * nice_ratio))
            time.sleep(pause)
    return pushed


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

    # ── Decide: zero-downtime refresh, or urgent broken-index heal? ──
    # HEALTHY live index → classic contract: build the full staging
    # index and promote ONCE at the end, so the old corpus serves
    # uninterrupted and readers swap atomically (true zero-downtime).
    #
    # UNUSABLE live index (broken / error / missing / empty) → search is
    # already returning nothing, so don't wait ~40 min for the full
    # rebuild. Push a small RECENT-first SEED and promote it the moment
    # it's searchable — restoring search in seconds — then continue to
    # the full rebuild and promote the complete corpus. A small working
    # index beats a broken one, and if the full pass is later killed the
    # live slot keeps the seed instead of falling back to broken.
    #
    # `BUILD_ALL_SEED_DOCS` is intentionally small (recent-first) so the
    # broken→working swap happens almost immediately, per the "replace
    # the broken index as soon as documents are available" requirement.
    SEED_DOCS = int(os.environ.get("BUILD_ALL_SEED_DOCS", "1000"))
    NICE_RATIO = float(os.environ.get("BUILD_ALL_NICE_RATIO", "1.0"))

    urgent = False
    try:
        from sources.utils.index_health import classify_index

        verdict, reason = classify_index(api_base, INDEX_NAME, n, n)
        urgent = verdict in {"broken", "error", "missing", "empty"}
        print(
            f"  live '{INDEX_NAME}' verdict: {verdict} ({reason}) → "
            f"{'SEED-NOW then full rebuild' if urgent else 'standard promote-at-end'}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not classify live '{INDEX_NAME}': {exc!r}; standard rebuild)")

    # ── Phase 0: urgent seed (broken live index only) ───────────────
    # Recent-first so the partial index is maximally useful; pushed
    # back-to-back (nice_ratio=0) since there's no live traffic to spare
    # when search is already dead. Promote is guarded — we only swap the
    # seed in once it provably loads + searches.
    if urgent:
        order = sorted(range(n), key=lambda k: _date_sort_key(metas[k]), reverse=True)
        seed = order[: min(SEED_DOCS, n)]
        seed_texts = [texts[k] for k in seed]
        seed_metas = [metas[k] for k in seed]
        print(
            f"\n  [SEED] live index unusable — building a {len(seed):,}-doc recent-first seed to restore search ASAP..."
        )
        _reset_staging(api_base, headers)
        seed_pushed = _push_batches(
            api_base,
            headers,
            STAGING_NAME,
            seed_texts,
            seed_metas,
            nice_ratio=0.0,
            label="seed",
        )
        if seed_pushed == len(seed) and _guarded_promote(api_base, headers, seed_pushed, "seed"):
            print(f"  [SEED] ✓ search restored with {seed_pushed:,} recent docs; full rebuild continues below.")
        else:
            print(
                f"  [SEED] ⚠ seed not promoted ({seed_pushed:,}/{len(seed):,} pushed) — proceeding to the full rebuild."
            )

    # ── Phase 1: full rebuild ───────────────────────────────────────
    # Fresh staging (a seed promote, if any, consumed the previous one).
    # Throttled push so live search keeps a fair share of the encoder
    # while the full corpus is rebuilt.
    _reset_staging(api_base, headers)
    pushed = _push_batches(
        api_base,
        headers,
        STAGING_NAME,
        texts,
        metas,
        nice_ratio=NICE_RATIO,
        label="full",
    )

    # Never promote a partial staging — the live slot keeps whatever it
    # has (the seed when urgent, else the previous corpus).
    if pushed != n:
        kept = "seed" if urgent else "previous corpus"
        print(f"\n[!] Skipping final promote — only {pushed:,}/{n:,} pushed. '{INDEX_NAME}' keeps the {kept}.")
        return 1

    # Guarded promote: drain + verify load/search before swapping. If
    # the freshly-built staging fails verification we REFUSE to promote
    # so a corrupt build can't replace the live slot — the daemon will
    # retry on its next cycle.
    if not _guarded_promote(api_base, headers, pushed, "full"):
        kept = "seed" if urgent else "previous corpus"
        print(f"\n[!] Full staging failed verification — '{INDEX_NAME}' keeps the {kept}; will retry next run.")
        return 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone — {pushed:,} documents indexed into '{INDEX_NAME}' (via promote) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
