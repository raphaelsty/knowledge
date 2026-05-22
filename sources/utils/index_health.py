"""Index health: detect broken per-user indices, audit `__all__`, repair.

The goal is "every index always works". Three concerns live here:

1. **Per-user indices** — one ColBERT/Plaid index per personality.
   A pipeline that crashes mid-write, a corrupt on-disk file, or a
   missing index whose owner still has documents in PG all leave the
   library un-searchable. `scan_user_indexes` classifies each, and
   `repair_user_indexes` rebuilds the offenders from PG using the
   same in-pipeline heal+re-embed path. The `__all__` index is *never*
   in scope here — it's a derivative, rebuilt by a different path.

2. **The `__all__` index** — the cross-personality discovery index
   over every VIP's documents. `all_index_status` compares its doc
   count against the sum of every VIP's PG document count; if the
   index is behind, the index is stale and needs a full rebuild via
   `rebuild_all_index`.

3. **Repair execution** — `repair_user_indexes` invokes
   `sources.utils.run_pipeline` with an empty ``sources_config`` so
   no source fetchers run. The pipeline's existing healer deletes
   the bad on-disk index, marks every doc indexed=false, and the
   indexing stage re-embeds the user's library from scratch. The
   repair budget is just the embedder.

Used by:
  * ``run.py`` — startup audit of `__all__`, end-of-run rebuild after
    any VIP has been processed.
  * ``make repair-indexes`` — operator-driven repair pass for the
    per-user indices (`__all__` is excluded by construction).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import psycopg

# `__all__` is a SPECIAL index name (cross-personality). The repair
# entry-points below MUST refuse to touch it — that's an explicit
# product requirement. Its rebuild lives in `rebuild_all_index`.
ALL_INDEX_NAME = "__all__"

DRIFT_ABS = 5
DRIFT_FRAC = 0.05

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_URL = "http://localhost:8080"


# ── classification ────────────────────────────────────────────────────


def _get_index_info(api_url: str, name: str, timeout: int = 10) -> tuple[int | None, dict | str]:
    """GET /indices/{name}. Return (status, payload) where payload is
    the decoded JSON on 200, the response body on error."""
    try:
        with urllib.request.urlopen(f"{api_url}/indices/{name}", timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return None, str(e)


def classify_index(api_url: str, index_name: str, pg_total: int, pg_indexed: int) -> tuple[str, str]:
    """Same verdicts as ``scripts/reindex_broken.py``.

    Verdicts: healthy, broken, error, missing, empty, pg_drift.
    """
    status, payload = _get_index_info(api_url, index_name)
    if status == 200 and isinstance(payload, dict):
        n_docs = int(payload.get("num_documents") or 0)
        n_emb = int(payload.get("num_embeddings") or 0)
        # Two flavours of broken — search returns nothing in both:
        #   a) num_documents > 0 but num_embeddings == 0 — embedder
        #      crashed mid-write (the original broken case).
        #   b) num_documents == 0 while PG has docs — index file
        #      exists and loads, but it's empty. Used to land under
        #      pg_drift (opt-in repair); promote to broken so the
        #      heal_if_broken hook in run.py rebuilds it on the
        #      user's next pipeline iteration.
        if n_docs > 0 and n_emb == 0:
            return ("broken", f"num_documents={n_docs}, num_embeddings=0")
        if n_docs == 0 and pg_total > 0:
            return ("broken", f"api=0, pg has {pg_total} doc(s)")
        pg_baseline = max(pg_indexed, pg_total)
        if pg_baseline > 0:
            drift = abs(pg_indexed - n_docs)
            threshold = max(DRIFT_ABS, int(pg_baseline * DRIFT_FRAC))
            if drift > threshold:
                return ("pg_drift", f"pg_indexed={pg_indexed} api={n_docs} drift={drift}")
        return ("healthy", "")
    if status == 404:
        return ("missing", "API 404") if pg_total > 0 else ("empty", "no docs")
    body = payload if isinstance(payload, str) else json.dumps(payload)[:80]
    if status is None:
        return ("error", f"transport: {body}")
    if status >= 500 or "NEXT_PLAID_ERROR" in body or "No data to merge" in body:
        return ("error", f"HTTP {status} {body[:80]}")
    return ("error", f"HTTP {status}")


# ── scanning per-user indices ────────────────────────────────────────


def _fetch_user_index_rows(database_url: str, vip_only: bool, slug: str | None) -> list[dict]:
    where = ["1=1"]
    params: list = []
    if slug:
        where.append("u.username = %s")
        params.append(slug)
    elif vip_only:
        where.append("u.vip = TRUE")
    sql = f"""
        SELECT u.id, u.username, u.name, u.index_name, u.vip,
               COUNT(d.url) FILTER (WHERE TRUE)::bigint              AS pg_total,
               COUNT(d.url) FILTER (WHERE d.indexed = TRUE)::bigint  AS pg_indexed
          FROM users u
          LEFT JOIN documents d ON d.user_id = u.id
         WHERE {" AND ".join(where)}
         GROUP BY u.id
         ORDER BY u.vip DESC, u.username
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "slug": r[1],
            "name": r[2],
            "index_name": r[3],
            "vip": r[4],
            "pg_total": r[5],
            "pg_indexed": r[6],
        }
        for r in rows
    ]


def scan_user_indexes(
    database_url: str,
    api_url: str,
    *,
    vip_only: bool = False,
    slug: str | None = None,
) -> dict[str, list[tuple[dict, str]]]:
    """Classify every per-user index. `__all__` is never in this set
    because we iterate over the `users` table."""
    api_url = api_url.rstrip("/")
    users = _fetch_user_index_rows(database_url, vip_only, slug)
    by_verdict: dict[str, list[tuple[dict, str]]] = {}
    for u in users:
        # Defensive: even if someone created a personality literally
        # called __all__, refuse to classify it as a user index.
        if u["index_name"] == ALL_INDEX_NAME:
            continue
        verdict, reason = classify_index(api_url, u["index_name"], u["pg_total"], u["pg_indexed"])
        by_verdict.setdefault(verdict, []).append((u, reason))
    return by_verdict


# ── per-user repair ──────────────────────────────────────────────────


def force_heal_index(api_url: str, index_name: str, user_id: int, database_url: str) -> None:
    """Drop the index server-side, mark every doc ``indexed=false`` in PG.

    Used in two situations:
      * ``pg_drift`` — the index loads but disagrees with PG. The
        pipeline's auto-healer won't fire because the index loads OK,
        so we trigger the heal manually.
      * ``broken``/``error``/``missing`` — the index is unloadable, but
        every doc in PG is flagged ``indexed=true`` so the next
        ``run_pipeline`` would re-index zero docs. By dropping the index
        AND resetting the indexed flags we force a full rebuild on the
        next pipeline pass.
    """
    try:
        req = urllib.request.Request(f"{api_url}/indices/{index_name}", method="DELETE")
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  warn: delete index failed: HTTP {e.code}")
    except Exception as e:
        print(f"  warn: delete index failed: {e}")
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET indexed=false, updated_at=now() WHERE user_id = %s AND indexed = true",
                    (user_id,),
                )
    except Exception as e:
        print(f"  warn: reset indexed flags failed: {e}")


def _rebuild_one(user: dict, database_url: str, api_url: str, verdict: str) -> bool:
    """Rebuild one user's index. Hard-refuses to touch ``__all__``."""
    if user["index_name"] == ALL_INDEX_NAME:
        print(f"  refusing to rebuild '{ALL_INDEX_NAME}' via the per-user path")
        return False

    # Lazy imports — these modules are heavy (sklearn, torch via the
    # indexer), and `index_health` is also imported from run.py at
    # boot for a cheap stats check. Defer the cost until we actually
    # need to rebuild something.
    from sources.sql import get_user_tags, get_vip_tags
    from sources.utils import run_pipeline
    from sources.utils.index_locks import IndexBusy, acquire_index_lock

    own_tags = get_user_tags(database_url, user["id"])
    shared_tags = sorted(set(get_vip_tags(database_url)) | set(own_tags))
    try:
        # Non-blocking advisory lock so this older repair entry-point
        # (`make repair-indexes`, `continuous_pipeline.sh` idle slot)
        # never collides with the new `sources.indexer_daemon`
        # when both are alive in the same deployment.
        with acquire_index_lock(database_url, user["id"], blocking=False):
            if verdict in {"pg_drift", "healthy"}:
                force_heal_index(api_url, user["index_name"], user["id"], database_url)
            run_pipeline(
                slug=user["slug"],
                name=user["name"],
                index_name=user["index_name"],
                sources_config={},  # no fetchers — embedder only
                user_id=user["id"],
                database_url=database_url,
                shared_tags=shared_tags,
                n_workers=1,
                vip=bool(user["vip"]),
                do_index=True,  # this entry-point exists to embed
            )
        return True
    except IndexBusy:
        print(f"  skip {user['slug']}: index busy (held by another writer)")
        return False
    except Exception as e:
        print(f"  [!] {user['slug']}: rebuild failed: {e}")
        return False


def heal_if_broken(
    database_url: str,
    api_url: str,
    user_id: int,
    index_name: str,
    *,
    pg_total: int | None = None,
    pg_indexed: int | None = None,
) -> str | None:
    """Self-heal hook for the continuous pipeline.

    Called once per user before ``run_pipeline`` fires. If the user's
    on-disk index disagrees with PG in a way the regular pipeline
    *cannot* fix on its own (i.e. the index is unloadable but every
    PG row is flagged ``indexed=true``), drop the index on disk and
    reset the indexed flags — the indexing stage at the end of
    ``run_pipeline`` then re-embeds the entire library from scratch.

    The auto-heal targets exactly the failure mode the continuous
    runner would otherwise re-skip every 12 hours forever:

      * ``broken`` — ``num_documents>0 num_embeddings==0`` (mid-write
        crash; "No data to merge" on search).
      * ``error``  — HTTP 5xx on GET (corrupt on-disk index).
      * ``missing``— 404 but PG has docs (lost index directory).

    ``healthy`` and ``pg_drift`` are *not* targeted: healthy needs no
    work, and pg_drift usually resolves on its own as an in-flight
    pipeline pass finishes. Operators can still force them via
    ``make repair-indexes INCLUDE_DRIFT=1``.

    Returns the verdict if a heal was performed, otherwise ``None``.
    """
    api_url = api_url.rstrip("/")
    if index_name == ALL_INDEX_NAME:
        # `__all__` is rebuilt via staging+promote, never via heal.
        return None

    # Look up PG counts only if the caller didn't pass them in.
    if pg_total is None or pg_indexed is None:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FILTER (WHERE TRUE), "
                    "       COUNT(*) FILTER (WHERE indexed = TRUE) "
                    "  FROM documents WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone() or (0, 0)
                pg_total = int(row[0])
                pg_indexed = int(row[1])

    verdict, reason = classify_index(api_url, index_name, pg_total, pg_indexed)
    if verdict in {"broken", "error", "missing"}:
        print(f"  [heal] {index_name}: {verdict} ({reason}) — dropping index + resetting indexed flags")
        force_heal_index(api_url, index_name, user_id, database_url)
        return verdict
    return None


def repair_user_indexes(
    database_url: str,
    api_url: str,
    *,
    vip_only: bool = False,
    slug: str | None = None,
    include_drift: bool = False,
    dry: bool = False,
    only_one: bool = False,
) -> tuple[int, int]:
    """Detect and rebuild broken per-user indices.

    Returns ``(targeted, succeeded)``. The ``__all__`` index is never
    rebuilt here — that path is in ``rebuild_all_index``.

    When ``only_one`` is True, picks the first broken index and stops
    after that single rebuild. Used by the continuous pipeline to
    drain the broken-index backlog one-at-a-time during otherwise-idle
    intervals, without holding the runner for a 60-minute bulk pass.
    """
    api_url = api_url.rstrip("/")
    by_verdict = scan_user_indexes(database_url, api_url, vip_only=vip_only, slug=slug)

    targetable = {"broken", "error", "missing"}
    if include_drift:
        targetable.add("pg_drift")

    targets: list[tuple[dict, str, str]] = []
    for verdict, entries in sorted(by_verdict.items()):
        marker = "→ rebuild" if verdict in targetable else "  skip"
        print(f"  [{verdict:<9}] {len(entries):>3}  {marker}")
        for u, reason in entries:
            if verdict == "empty" or verdict == "healthy":
                continue
            print(f"      {u['slug']:<28} pg={u['pg_indexed']}/{u['pg_total']:<6} {reason}")
            if verdict in targetable:
                targets.append((u, verdict, reason))

    if not targets:
        print("\nNothing to rebuild.")
        return (0, 0)
    if only_one:
        targets = targets[:1]
    if dry:
        print(f"\nDRY RUN — would rebuild {len(targets)} index(es).")
        return (len(targets), 0)

    print(f"\nRebuilding {len(targets)} index(es)...\n")
    ok = 0
    t0 = time.perf_counter()
    for i, (u, verdict, reason) in enumerate(targets, 1):
        print(f"\n{'=' * 60}\n  [{i}/{len(targets)}] {u['slug']}  ({verdict}: {reason})\n{'=' * 60}")
        if _rebuild_one(u, database_url, api_url, verdict):
            ok += 1
    print(f"\nDone in {time.perf_counter() - t0:.1f}s — {ok}/{len(targets)} rebuilt.")
    return (len(targets), ok)


# ── `__all__` audit + rebuild ────────────────────────────────────────


def vip_document_total(database_url: str) -> int:
    """Sum of `documents` rows owned by VIP users. The source-of-truth
    target for `__all__`'s document count."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents d JOIN users u ON u.id = d.user_id WHERE u.vip = TRUE")
            (n,) = cur.fetchone()
    return int(n)


def all_index_doc_count(api_url: str) -> int | None:
    """Return `num_documents` of `__all__`, or ``None`` if missing/error."""
    status, payload = _get_index_info(api_url.rstrip("/"), ALL_INDEX_NAME)
    if status == 200 and isinstance(payload, dict):
        return int(payload.get("num_documents") or 0)
    return None


def all_index_status(database_url: str, api_url: str) -> dict:
    """Snapshot for the startup audit and end-of-run decision.

    Returns ``{"vip_total", "all_count", "stale", "reason"}`` where
    ``stale`` is True when the index has *fewer* docs than the VIP
    total (the user's spec: "if it contains less documents that the
    number of total documents"). A missing index counts as stale.
    """
    vip_total = vip_document_total(database_url)
    all_count = all_index_doc_count(api_url)
    if all_count is None:
        return {
            "vip_total": vip_total,
            "all_count": None,
            "stale": vip_total > 0,
            "reason": "index missing or unreachable",
        }
    stale = all_count < vip_total
    return {
        "vip_total": vip_total,
        "all_count": all_count,
        "stale": stale,
        "reason": (f"all={all_count} < vip_total={vip_total}" if stale else f"all={all_count} ≥ vip_total={vip_total}"),
    }


def update_all_index_for_slugs(
    database_url: str,
    api_url: str,
    slugs: list[str],
    admin_key: str | None = None,
) -> int:
    """Incrementally refresh `__all__` for the given `slugs` — no
    staging, no full drop+rebuild.

    Flow per call:
      1. DELETE existing chunks for each slug via
         ``/indices/__all__/documents`` with ``condition = owner = ?``.
         The API queues a batched delete; the slug's metadata rows
         are removed and the matching doc_ids are pulled from the
         vector index.
      2. SELECT each slug's current (url, doc) rows from PG and
         re-push them to ``/indices/__all__/update_with_encoding``.
         The push APPENDS (assigns fresh doc_ids), so step 1 is
         essential — without it, every continuous-pipeline iteration
         would double the slug's chunks in ``__all__``.

    Returns 0 on success, non-zero on any unrecoverable HTTP error.
    Empty `slugs` is a no-op. A missing ``__all__`` index is a no-op
    too — the next run.py startup audit will detect it and trigger
    the staging+promote full rebuild path.

    Doc text is built the same way ``sources/utils/build_all_index.py``
    does it so search behaves identically across the per-user and
    cross-personality paths.
    """
    if not slugs:
        return 0

    # Imported lazily so `run.py` startup doesn't pay the psycopg +
    # HTTP-machinery cost until an update actually fires. Mirrors the
    # contract `rebuild_all_index` uses below.
    import json as _json
    import time as _time
    import urllib.request as _urllib_request

    import psycopg as _psycopg

    from sources.utils.build_all_index import BATCH as _BATCH
    from sources.utils.client import website_name as _website_name

    api = api_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["X-API-Key"] = admin_key

    def _request(path: str, method: str, payload: dict | None, timeout: int = 600):
        data = _json.dumps(payload).encode() if payload is not None else None
        req = _urllib_request.Request(
            f"{api}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with _urllib_request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except _urllib_request.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")

    def _post(path: str, payload: dict, timeout: int = 600):
        return _request(path, "POST", payload, timeout)

    # Step 1: remove each slug's existing chunks from `__all__` first.
    # `update_with_encoding` APPENDS to the index (it assigns fresh
    # doc_ids on every push), so without a pre-delete the same slug
    # would accumulate duplicate chunks on every continuous-pipeline
    # iteration. We use the API's batched delete-by-condition endpoint
    # with the `owner` metadata key the build helper writes for every
    # `__all__` document. The DELETE returns 202 immediately and the
    # batch worker processes within ~2s; we sleep briefly so the push
    # below doesn't race with a not-yet-applied delete.
    print(f"__all__ update: clearing existing chunks for {len(slugs)} slug(s)...", flush=True)
    for s in slugs:
        status, body = _request(
            "/indices/__all__/documents",
            "DELETE",
            {"condition": "owner = ?", "parameters": [s]},
            timeout=120,
        )
        # 202 = queued. 404 = `__all__` not declared yet — fine, the
        # caller will trigger a full rebuild on the next pass when the
        # startup audit sees a missing index.
        if status == 404:
            print(f"  __all__ missing — skipping delete for '{s}' (will be repaired next run)")
            return 0
        if status >= 400:
            print(f"  [!] delete-by-owner for '{s}' returned HTTP {status}: {body[:200]}")
    # Wait for the batch delete worker to flush before pushing. The
    # max batch wait is ~2s; give it a small buffer for processing.
    _time.sleep(5)

    QUEUE_FULL_BACKOFF = (2, 4, 8, 16, 30)
    sql = (
        "SELECT d.url, d.title, d.summary, d.date, d.tags, d.extra_tags, "
        "       d.source, d.source_url, u.username "
        "  FROM documents d "
        "  JOIN users u ON u.id = d.user_id "
        " WHERE u.username = ANY(%s::text[]) "
        "   AND d.deleted = FALSE "
        " ORDER BY u.username, d.url"
    )

    total_pushed = 0
    texts: list[str] = []
    metas: list[dict] = []
    with _psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, (list(slugs),))
        for url, title, summary, date, tags, extra_tags, source, source_url, slug in cur.fetchall():
            doc_tags = list(tags) if tags else []
            extra = list(extra_tags) if extra_tags else []
            text = (
                f"{title or ''} {' '.join(doc_tags)} {' '.join(extra)} "
                f"{(summary or '')[:200]} {source or ''} {_website_name(url)}"
            ).strip()
            if not text:
                continue
            texts.append(text)
            metas.append(
                {
                    "url": url,
                    "title": title or "",
                    "summary": summary or "",
                    "date": str(date or ""),
                    "tags": ",".join(doc_tags),
                    "extra_tags": ",".join(extra),
                    "source": source or "",
                    "source_url": source_url or "",
                    "owner": slug,
                }
            )

    n = len(texts)
    if n == 0:
        print(f"__all__ update: nothing to push for slugs={slugs!r}")
        return 0
    n_batches = (n + _BATCH - 1) // _BATCH
    print(f"__all__ update: pushing {n:,} doc(s) for {len(slugs)} slug(s) in {n_batches} batch(es)...", flush=True)
    for i in range(0, n, _BATCH):
        batch_texts = texts[i : i + _BATCH]
        batch_meta = metas[i : i + _BATCH]
        attempt = 0
        while True:
            status, body = _post(
                "/indices/__all__/update_with_encoding",
                {"documents": batch_texts, "metadata": batch_meta, "pool_factor": 2},
                timeout=600,
            )
            if status == 503 and "queue full" in body.lower():
                wait = QUEUE_FULL_BACKOFF[min(attempt, len(QUEUE_FULL_BACKOFF) - 1)]
                attempt += 1
                _time.sleep(wait)
                continue
            break
        if status >= 400:
            print(f"  batch {i // _BATCH + 1}/{n_batches} HTTP {status}: {body[:200]}")
            return 1
        total_pushed += len(batch_texts)
        if (i // _BATCH) % 5 == 0 or (i + _BATCH) >= n:
            print(f"  batch {i // _BATCH + 1}/{n_batches} ✓ ({total_pushed:,}/{n:,})", flush=True)
    print(f"__all__ update: {total_pushed:,} doc(s) pushed", flush=True)
    return 0


def rebuild_all_index(database_url: str, api_url: str, admin_key: str | None = None) -> int:
    """Rebuild `__all__` via the staging+promote path.

    The cross-personality index is a *derivative*: every chunk it
    holds can be re-derived from `documents`, so a full rebuild is
    the simplest correct path. The build helper writes into a
    staging index and promotes via ArcSwap at the end, so search
    against `__all__` keeps working through the whole rebuild.

    Imported lazily so `run.py` startup doesn't pay the psycopg +
    HTTP-machinery cost until a rebuild actually fires.
    """
    # Re-export the build helper's env contract. Keep current process
    # env intact by restoring on the way out.
    prev = {k: os.environ.get(k) for k in ("DATABASE_URL", "API_URL", "ADMIN_API_KEY")}
    os.environ["DATABASE_URL"] = database_url
    os.environ["API_URL"] = api_url
    if admin_key is not None:
        os.environ["ADMIN_API_KEY"] = admin_key

    try:
        from sources.utils.build_all_index import main as build_all_main

        return int(build_all_main())
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── CLI ──────────────────────────────────────────────────────────────


def _cli() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sources.utils.index_health",
        description=(
            "Index health: scan/repair per-user indices, audit `__all__`. "
            "The `__all__` index is never repaired by the user-index paths."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="report verdicts for every per-user index")
    s.add_argument("--vip-only", action="store_true")
    s.add_argument("--slug", default=None)

    r = sub.add_parser("repair", help="rebuild broken per-user indices (NOT __all__)")
    r.add_argument("--vip-only", action="store_true")
    r.add_argument("--slug", default=None)
    r.add_argument("--include-drift", action="store_true")
    r.add_argument("--dry", action="store_true")
    r.add_argument(
        "--one",
        action="store_true",
        help="Repair at most one broken index then exit. "
        "Used by the continuous pipeline during idle intervals "
        "(cool-down window fully populated) so the runner spends "
        "its spare time draining the broken-index backlog instead "
        "of sleeping. Exit code 3 = nothing was broken.",
    )

    sub.add_parser("all-status", help="print the `__all__` audit snapshot")

    a = sub.add_parser("all-rebuild", help="drop + rebuild the `__all__` index")
    a.add_argument("--if-stale", action="store_true", help="no-op when not stale")

    args = ap.parse_args()
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_url = os.environ.get("API_URL", DEFAULT_API_URL)
    admin_key = os.environ.get("ADMIN_API_KEY") or None

    if args.cmd == "scan":
        out = scan_user_indexes(database_url, api_url, vip_only=args.vip_only, slug=args.slug)
        for verdict, entries in sorted(out.items()):
            print(f"  [{verdict:<9}] {len(entries):>3}")
            for u, reason in entries:
                print(f"      {u['slug']:<28} pg={u['pg_indexed']}/{u['pg_total']:<6} {reason}")
        return 0

    if args.cmd == "repair":
        targeted, ok = repair_user_indexes(
            database_url,
            api_url,
            vip_only=args.vip_only,
            slug=args.slug,
            include_drift=args.include_drift,
            dry=args.dry,
            only_one=args.one,
        )
        # Exit code contract for the continuous-pipeline caller:
        #   0 — repaired at least one index successfully, or dry-run
        #   2 — repair attempt failed
        #   3 — nothing was broken (lets the shell back off and sleep)
        if targeted == 0:
            return 3
        return 0 if (args.dry or targeted == ok) else 2

    if args.cmd == "all-status":
        snap = all_index_status(database_url, api_url)
        flag = "STALE" if snap["stale"] else "ok"
        all_count = snap["all_count"] if snap["all_count"] is not None else "missing"
        print(f"  __all__ {flag}  ({snap['reason']})  vip_total={snap['vip_total']}  all={all_count}")
        return 1 if snap["stale"] else 0

    if args.cmd == "all-rebuild":
        if args.if_stale:
            snap = all_index_status(database_url, api_url)
            if not snap["stale"]:
                print(f"  __all__ up-to-date ({snap['reason']}) — skipping rebuild")
                return 0
            print(f"  __all__ is stale ({snap['reason']}) — rebuilding")
        return rebuild_all_index(database_url, api_url, admin_key)

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
