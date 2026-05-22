#!/usr/bin/env python3
"""
Knowledge Database Builder

Runs the fetch → clean → tag → tree → index pipeline for a single personality.
Called by run.py which iterates over all personalities.

Each personality gets:
- Its own data directory (data/{slug}/)
- Its own search index (via indexName)
- Its own sources configuration (read from the `users.sources` JSONB column)
"""

import json
import os
import re
import time
from urllib.parse import urlparse

from .. import github, hackernews, huggingface, tags, twitter, youtube, zotero


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def step(pct: int, label: str, detail: str = ""):
    """Emit a structured progress line for the frontend to parse."""
    print(f"@@{pct}|{label}|{detail}@@", flush=True)


def website_name(url: str) -> str:
    """Extract a clean website name from a URL for indexing.

    Returns the hostname stripped of `www.`, plus the bare brand
    label (the registrable name before the public suffix) so a
    query like `github` matches both `github.com` and
    `docs.github.com`.

    Examples:
        https://www.techcrunch.com/...     → "techcrunch.com techcrunch"
        https://news.ycombinator.com/x     → "news.ycombinator.com ycombinator"
        https://github.com/user/repo       → "github.com github"
        https://blog.openai.com/post       → "blog.openai.com openai"
    """
    if not url:
        return ""
    try:
        host = (urlparse(url).hostname or "").lower()
    except (ValueError, TypeError):
        return ""
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 2:
        brand = parts[-2]
    else:
        brand = parts[0]
    if brand == host:
        return host
    return f"{host} {brand}"


def merge_new_documents(existing: dict, new: dict) -> tuple[dict, set[str]]:
    """Merge new documents, skipping URLs already in the database."""
    new_only = {url: doc for url, doc in new.items() if url not in existing}
    return {**existing, **new_only}, set(new_only.keys())


def hostname_source_key(url: str) -> str:
    """``'https://www.mixedbread.com/blog/foo' → 'mixedbread.com'``.

    Strips the leading ``www.`` so a single hostname maps to one chip.
    Returns ``''`` for unparseable inputs — the empty bucket is filtered
    out of the user-facing source panel by `user_source_counts`. Module-
    level mirror of the JS ``hostnameSourceKey`` helper; both must give
    the same answer or browser-side and pipeline-side syncs would
    disagree on which chip a doc belongs to.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    if not host:
        return ""
    return host.removeprefix("www.")


def _verify_indexed_urls(
    api_base: str,
    index_name: str,
    candidate_urls: list[str],
    auth_headers: dict,
    chunk: int = 200,
) -> list[str]:
    """Return the subset of `candidate_urls` actually present in the index.

    Used right after `update_with_encoding` to defend against the silent
    drop mode where the API returns 2xx for the batch as a whole but
    drops individual docs server-side. We POST to `/metadata/get` with
    a `url IN (...)` condition and intersect the response against the
    input set. Anything missing stays `indexed=false` in PG and gets
    retried next run.

    Chunked to stay under the server's `MAX_CONDITION_PARAMETERS`
    (~200) — same threshold the pipeline already uses for the
    pre-upsert delete.
    """
    import urllib.request

    if not candidate_urls:
        return []
    seen: set[str] = set()
    for j in range(0, len(candidate_urls), chunk):
        batch = candidate_urls[j : j + chunk]
        placeholders = ",".join("?" for _ in batch)
        payload = json.dumps({"condition": f"url IN ({placeholders})", "parameters": list(batch)}).encode()
        req = urllib.request.Request(
            f"{api_base}/indices/{index_name}/metadata/get",
            data=payload,
            headers=auth_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # `metadata/get` returns {"metadata": [...]} — accept
                # `results` too in case the API contract drifts.
                body = json.loads(resp.read())
                rows = body.get("metadata") or body.get("results") or []
        except Exception as e:
            # If verification itself fails we conservatively treat the
            # whole chunk as "unknown" — i.e. don't mark indexed. The
            # next run will retry rather than silently lose docs.
            print(f"  Warning: index verify chunk {j // chunk + 1} failed: {e}")
            continue
        for m in rows:
            u = m.get("url")
            if isinstance(u, str) and u:
                seen.add(u)
    # Preserve original order so subsequent SQL-side ops are stable.
    return [u for u in candidate_urls if u in seen]


def run_pipeline(
    slug: str,
    name: str,
    index_name: str,
    sources_config: dict,
    user_id: int,
    database_url: str,
    shared_tags: list[str] | None = None,
    n_workers: int = -1,
    vip: bool = False,
    twitter_via_twikit: bool = False,
    do_index: bool = False,
):
    """Run the fetch → clean → tag → (optionally) index pipeline.

    By default (``do_index=False``) the pipeline stops once the
    cleaned + tagged documents are persisted to Postgres. The ColBERT
    index is now built and repaired by ``sources.indexer_daemon``,
    a separate process that owns the index lifecycle end-to-end —
    detecting broken indexes, prioritising backfill of un-indexed
    documents, and re-embedding from PG. Decoupling makes the
    fetcher fast and quota-bounded (no API dep, no on-disk index
    contention), and lets the daemon run on its own schedule with
    a global priority view that ``make run SLUG=…`` never has.

    Pass ``do_index=True`` to opt back into the legacy single-process
    flow — only used by the daemon itself (one-shot reindex of one
    user) and by operators debugging.

    All persistence is to PostgreSQL — no JSON dumps. The caller is
    responsible for ensuring the user row exists before calling
    (FK on `documents.user_id`).

    Parameters
    ----------
    slug : str
        URL-safe identifier (e.g. "raphael-sourty").
    name : str
        Display name (e.g. "Raphael Sourty").
    index_name : str
        Search index name (e.g. "raphael-sourty").
    sources_config : dict
        Source configuration from the `users.sources` JSONB column (github, twitter, etc.).
    user_id : int
        users.id for this personality (FK target for documents/pipeline_runs).
    database_url : str
        Postgres URL.
    shared_tags : list[str] | None
        Tag vocabulary used to seed `get_extra_tags`. By contract,
        the caller composes this as (this user's own tags) ∪ (the
        union of VIP users' tags). Non-VIP users never contribute
        tags to anyone else's vocabulary. The contract is enforced
        with assertions at the start of the tag stage.
    n_workers : int
        Parallelism for the source-fetch phase. ``-1`` (default) uses
        ``min(cpu_count(), n_tasks)``. ``1`` runs every fetcher serially
        (matches the pre-parallel behavior). Anything > 1 dispatches
        each tracked fetcher block onto a thread pool. Threads are the
        right tool here: every fetcher is HTTP/IO-bound, so the GIL is
        released on socket reads and we get real concurrency without the
        process-spawn overhead. Cleaning, tagging, and indexing still
        run sequentially after the fan-in.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from sources.sql import (
        cleanup_stale_runs,
        finish_pipeline_run,
        load_dead_urls,
        load_documents,
        load_unindexed_documents,
        mark_documents_indexed,
        mark_urls_dead,
        start_pipeline_run,
        track_source,
        update_pipeline_run_stage,
        upsert_documents,
    )

    pipeline_start = time.perf_counter()
    timings: list[tuple[str, float]] = []

    # Live run tracker — insert a `running` row now so a dashboard
    # can see "Simon Willison's run started 12s ago, currently in
    # the tag stage". Sealed at the end of this function (success or
    # failure). Best-effort: if the INSERT fails we still run the
    # pipeline, we just lose visibility.
    #
    # Sweep stale `running` rows first — a prior process that crashed
    # hard (OOM, SIGKILL) would leave its row pending forever. Anything
    # older than 2h is orphaned by definition.
    try:
        swept = cleanup_stale_runs(database_url, max_age_hours=2.0)
        if swept:
            print(f"  (pipeline_runs: swept {swept} stale run{'s' if swept != 1 else ''})")
    except Exception as _exc:
        print(f"  (pipeline_runs cleanup failed: {_exc})")
    try:
        run_id = start_pipeline_run(database_url, user_id, trigger="python")
    except Exception as _exc:
        run_id = 0
        print(f"  (pipeline_runs start failed: {_exc})")

    # Resolve who pays for this personality's ongoing costs.
    # Rules:
    #   • Sponsored personality (`users.sponsored_by IS NOT NULL`)
    #     → bill the SPONSOR. The sponsor brought this library to
    #       the platform via the paid-add path; they own its
    #       ongoing parsing. We override `billing_vip = False` even
    #       when the sponsor is a VIP, because the VIP exemption
    #       applies to a user's *own* content, not to libraries
    #       they chose to introduce. Sponsors with $0 balance hit
    #       the same insufficient-credits path as anyone else.
    #   • Plain VIP (no sponsor — grandfathered cohort)
    #     → free; platform absorbs.
    #   • Non-VIP (legacy)
    #     → personality bills itself (legacy self-funding path).
    billing_user_id = user_id
    billing_vip = vip
    sponsored = False
    try:
        import psycopg  # local import to avoid top-level dependency

        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT sponsored_by FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                billing_user_id = int(row[0])
                billing_vip = False  # sponsor pays regardless of their own VIP flag
                sponsored = True
                print(f"  Sponsored personality — billing user_id={billing_user_id}")
    except Exception as _exc:
        print(f"  (sponsor lookup failed, falling back to self-billing: {_exc})")

    # Storage tick. Doc count is always for the personality; the
    # debit lands on the billing account resolved above.
    try:
        from sources.storage import charge_storage_if_due as _charge_storage

        _r = _charge_storage(
            database_url,
            user_id,
            is_vip=billing_vip,
            billing_user_id=billing_user_id,
        )
        if _r["charged"]:
            who = "" if billing_user_id == user_id else f" → user_id={billing_user_id}"
            print(
                f"  Storage: billed {_r['credits']} credit(s){who} "
                f"({_r['docs']} docs, new balance {_r.get('new_balance', '?')})"
            )
        elif _r["reason"] == "insufficient_credits":
            who = "sponsor" if sponsored else "user"
            print(f"  Storage: NOT BILLED — {who} short {_r['credits']} credit(s) for {_r['docs']} docs.")
        elif _r["reason"] not in ("vip", "under_free_quota") and not _r["reason"].startswith("too_early"):
            print(f"  Storage: skipped ({_r['reason']})")
    except Exception as _exc:
        print(f"  (storage tick failed: {_exc})")

    def _mark_stage(stage: str) -> None:
        """Best-effort stage update — never let a DB hiccup abort the run."""
        if run_id <= 0:
            return
        try:
            update_pipeline_run_stage(database_url, run_id, stage)
        except Exception as _exc:
            print(f"  (pipeline_runs stage update failed: {_exc})")

    def _ts(source: str, detail: str = "", timing_label: str | None = None):
        """Shorthand: bind db/run_id/user_id/timings into `track_source`.

        Each fetcher block wraps itself in `with _ts("github") as ts:`
        so a single source crashing records a `failed` row in
        `pipeline_source_runs` AND lets the next source still run.
        """
        return track_source(
            database_url,
            run_id,
            user_id,
            source,
            detail,
            timings,
            timing_label,
        )

    # =========================================================================
    # Fresh-fetch mode
    # =========================================================================
    # When ``KNOWLEDGE_FRESH`` is truthy, fetchers are called with an empty
    # existing_urls set — page-level early-exit is disabled.
    _fresh = os.environ.get("KNOWLEDGE_FRESH", "").strip().lower() in {"1", "true", "yes"}

    # Global set of URLs the dead-link probe has rejected on a previous
    # run. Unioned into existing_urls so fetchers short-circuit instead of
    # re-yielding URLs we'll just kill again. Refreshed once per run.
    _dead_urls: set[str] = set() if _fresh else load_dead_urls(database_url)

    def _existing():
        # Thread-safe snapshot. Under `_merge_lock` so a parallel
        # `_merge_and_track` can't mutate `data` mid-iteration. Returns
        # a fresh set (not a view) so callers can use it without
        # holding the lock.
        if _fresh:
            return set()
        with _merge_lock:
            return set(data.keys()) | _dead_urls

    # Per-website fetch cooldown. Personalities like Max Halford track
    # 100+ websites; refetching every one on every run is the bulk of
    # the runtime. ``WEBSITE_FETCH_TTL_HOURS`` (env, default 0 = no
    # cooldown) skips a website's fetch step when the most recent doc
    # of that source was inserted less than the TTL ago. Tagging and
    # indexing still run on the existing docs, so re-running a
    # tagging-only pass is just `WEBSITE_FETCH_TTL_HOURS=24 make run`.
    _website_ttl_hours = float(os.environ.get("WEBSITE_FETCH_TTL_HOURS", "0") or "0")

    def _website_is_fresh(source_key: str) -> bool:
        if _website_ttl_hours <= 0 or not source_key:
            return False
        from datetime import datetime, timedelta, timezone

        import psycopg

        try:
            with psycopg.connect(database_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(updated_at) FROM documents  WHERE user_id = %s AND source = %s",
                    (user_id, source_key),
                )
                row = cur.fetchone()
        except Exception:
            return False
        if not row or not row[0]:
            return False
        last = row[0]
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=_website_ttl_hours)
        return last >= cutoff

    # =========================================================================
    # Credentials from Environment
    # =========================================================================

    hackernews_username = os.environ.get("HACKERNEWS_USERNAME")
    hackernews_password = os.environ.get("HACKERNEWS_PASSWORD")
    zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID")
    zotero_api_key = os.environ.get("ZOTERO_API_KEY")
    huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
    twitterapiio_api_key = os.environ.get("TWITTERAPIIO_API_KEY")

    # =========================================================================
    # Social counts — lazy backfill
    # =========================================================================
    # twitter_followers / github_followers / citations are stored on the
    # users row and only fetched when still NULL. Failures stay NULL so a
    # later run retries. Uses Twitter + GitHub handles from users.links /
    # users.sources (already read from PG).
    try:
        from sources.utils.popularity import populate_social_counts

        filled = populate_social_counts(database_url, user_id, display_name=name)
        if filled:
            parts = ", ".join(f"{k}={v}" for k, v in filled.items())
            print(f"    Social counts: {parts}")
    except Exception as e:
        print(f"    Social counts lookup failed: {e}")

    # =========================================================================
    # Load Existing Database
    # =========================================================================

    t0 = time.perf_counter()
    step(4, "Loading database", "Reading existing documents from PG")
    data: dict = load_documents(database_url, user_id)
    step(6, "Database loaded", f"{len(data):,} documents")
    timings.append(("Load database", time.perf_counter() - t0))

    # =========================================================================
    # Fetch Data from Sources
    # =========================================================================

    _mark_stage("fetch")

    new_urls: set[str] = set()
    # With per-user documents in PG, ownership is scoped by user_id at the
    # schema level — `personality_urls` is just the set of URLs this run
    # touched (seed from existing data + fetcher returns).
    personality_urls: set[str] = set(data.keys())

    # Mutex around `data` / `personality_urls` / `new_urls`. With
    # `n_workers > 1` multiple fetcher threads call `_merge_and_track`
    # concurrently; without this lock the dict-rebuild + set updates
    # would race. Sequential mode (`n_workers == 1`) is also safe — the
    # uncontended lock is essentially free.
    _merge_lock = threading.Lock()

    def _merge_and_track(fetched: dict[str, dict], source_key: str = "") -> set[str]:
        """Tag each doc with its source key, merge into `data`, track new URLs.

        ``source_key`` populates ``documents.source`` on upsert (twitter,
        youtube, scholar, …). Docs that already declare a source keep theirs
        (e.g. the twitter fetcher sets per-doc sources based on linked-page
        domain).

        URL-based short-circuits: when a URL lives on a known brand domain
        we route it there regardless of how the pipeline discovered it.
        A Zotero-saved arXiv paper, a tweeted HuggingFace model, and a
        scholar-cited paper hosted on arxiv.org should all bucket the
        same way in the filter chips.
        """
        nonlocal data
        for url, d in fetched.items():
            if "source" not in d:
                d["source"] = source_key
            if "arxiv.org" in url:
                d["source"] = "arxiv"
            elif "huggingface.co" in url:
                d["source"] = "huggingface"
            elif "github.com" in url:
                d["source"] = "github"
            elif "youtube.com" in url or "youtu.be" in url:
                d["source"] = "youtube"
            elif d.get("source") == "zotero":
                # Zotero is a generic save-anything bookmark service —
                # if we land here, the URL didn't match any branded
                # bucket above. Fall back to the URL's hostname so an
                # ACL paper, an OpenReview submission, and a personal
                # blog each get their own filter chip instead of all
                # piling into a single "zotero" bucket.
                d["source"] = hostname_source_key(url) or "zotero"
        with _merge_lock:
            personality_urls.update(fetched.keys())
            merged, added = merge_new_documents(data, fetched)
            data = merged
            new_urls.update(added)
        return added

    github_users = sources_config.get("github") or []
    twitter_config = sources_config.get("twitter")

    # Resolve HN creds from sources.hackernews. Username alone is enough
    # for the public fetchers (Comments, Submissions); the password is
    # only required for the authenticated Upvotes fetcher.
    hn_user, hn_pass = hackernews_username, hackernews_password
    hn_cfg = sources_config.get("hackernews")
    if isinstance(hn_cfg, dict):
        if hn_cfg.get("username"):
            hn_user = hn_cfg["username"]
        enc = hn_cfg.get("password_enc")
        if enc:
            from sources.utils.secrets import decrypt as _decrypt

            dec = _decrypt(enc)
            if dec is not None:
                hn_pass = dec
    # `has_hn` now means "can run the authenticated Upvotes fetcher".
    has_hn = bool(hn_user and hn_pass)

    # Zotero: new shape after the API-key-only onboarding flow.
    # sources.zotero = { api_key_enc, user_id, groups: [{id, name}, ...] }
    # We decrypt the key once and build an (library_type, library_id)
    # list to iterate. Falls back to legacy env vars when the per-user
    # config is absent.
    zot_api_key = zotero_api_key
    zot_libraries: list[tuple[str, str]] = []
    zot_cfg = sources_config.get("zotero")
    if isinstance(zot_cfg, dict):
        enc = zot_cfg.get("api_key_enc")
        if enc:
            from sources.utils.secrets import decrypt as _decrypt

            dec = _decrypt(enc)
            if dec is not None:
                zot_api_key = dec
        if zot_cfg.get("user_id"):
            zot_libraries.append(("user", str(zot_cfg["user_id"])))
        for g in zot_cfg.get("groups") or []:
            if isinstance(g, dict) and g.get("id"):
                zot_libraries.append(("group", str(g["id"])))
        # Legacy shape: library_id/library_type from the old form.
        if not zot_libraries and zot_cfg.get("library_id"):
            zot_libraries.append((zot_cfg.get("library_type", "user"), str(zot_cfg["library_id"])))
    # Legacy env fallback — treat as a single group library (old default).
    if not zot_libraries and zotero_library_id:
        zot_libraries.append(("group", zotero_library_id))
    has_zotero = bool(zot_libraries and zot_api_key)
    hf_cfg = sources_config.get("huggingface")
    has_hf = hf_cfg is not None and (huggingface_token is not None or isinstance(hf_cfg, str))
    has_twitter = twitter_config is not None

    src_start, src_end = 8, 40

    # ── Fetcher tasks ────────────────────────────────────────────────
    #
    # Each `with _ts(...)` block below is registered as a closure on
    # `fetch_tasks` instead of being executed inline. Once every block
    # has had its chance to register, the dispatcher at the bottom of
    # the fetch section runs the list — sequentially when `n_workers
    # == 1`, or via a `ThreadPoolExecutor` otherwise. Fetchers are
    # IO-bound (HTTP), so threads buy real concurrency without the
    # process-spawn overhead of multiprocessing.
    #
    # Loop variables are captured via default-arg tricks (`u=user`)
    # so each closure binds the iteration's value, not a shared cell.
    fetch_tasks: list = []

    # GitHub starred repositories — one task per github handle so a
    # single 404 only fails its own row in pipeline_source_runs.
    for user in github_users:

        def _t1(u=user):
            with _ts("github", f"@{u}", f"Fetch GitHub @{u}") as ts:
                step(src_start, "Fetching GitHub", f"Stars from @{u}")
                fetcher = github.Stars(user=u)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="github")
                ts.add(len(added))
                detail = f"@{u}: +{len(added)} new" if added else f"@{u}: up to date"
                step(src_start, "GitHub", detail)

        fetch_tasks.append(_t1)

    # HackerNews — three fetchers, all keyed off the canonical
    # sources.hackernews.username:
    #   • Upvotes (needs password_enc, so gated on `has_hn`)
    #   • Comments (public, no auth)
    #   • Submissions (public, no auth)
    hn_has_user = bool(hn_user)
    if hn_has_user:
        if has_hn:

            def _t2():
                with _ts("hackernews", "upvotes", "Fetch HN upvotes") as ts:
                    step(src_start, "Fetching HackerNews", f"Upvotes from @{hn_user}")
                    fetcher = hackernews.Upvotes(username=hn_user, password=hn_pass)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="hackernews")
                    ts.add(len(added))
                    step(src_start, "HackerNews", f"+{len(added)} new upvotes" if added else "Up to date")

            fetch_tasks.append(_t2)

        def _t3():
            with _ts("hackernews", "comments", "Fetch HN comments") as ts:
                step(src_start, "Fetching HackerNews", f"Comments from @{hn_user}")
                fetcher = hackernews.Comments(username=hn_user, max_items=500)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="hackernews")
                ts.add(len(added))
                step(src_start, "HackerNews", f"+{len(added)} new comments" if added else "Up to date")

        fetch_tasks.append(_t3)

        def _t4():
            with _ts("hackernews", "submissions", "Fetch HN submissions") as ts:
                step(src_start, "Fetching HackerNews", f"Submissions from @{hn_user}")
                fetcher = hackernews.Submissions(username=hn_user, max_items=500)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="hackernews")
                ts.add(len(added))
                step(src_start, "HackerNews", f"+{len(added)} new submissions" if added else "Up to date")

        fetch_tasks.append(_t4)

    # Zotero libraries (personal + every accessible group) — one task
    # per library so a single library 403 doesn't hide under a generic
    # "Zotero" failure.
    if has_zotero:
        for lib_type_, lib_id_ in zot_libraries:

            def _t5(lt=lib_type_, lid=lib_id_):
                with _ts("zotero", f"{lt}/{lid}", f"Fetch Zotero {lt}/{lid}") as ts:
                    step(src_start, "Fetching Zotero", f"{lt}/{lid}")
                    fetcher = zotero.Library(
                        library_id=lid,
                        library_type=lt,
                        api_key=zot_api_key,
                    )
                    added = _merge_and_track(fetcher(), source_key="zotero")
                    ts.add(len(added))
                    step(src_start, "Zotero", f"{lt}/{lid}: +{len(added)}" if added else f"{lt}/{lid}: up to date")

            fetch_tasks.append(_t5)

    # HuggingFace liked items — both paths route through the rich
    # `huggingface.Likes` extractor so the per-repo enrichment
    # (repo_info + README → fact-sheet head + first prose paragraph)
    # is identical regardless of how we got the username.
    if has_hf:

        def _t6():
            with _ts("huggingface", hf_cfg if isinstance(hf_cfg, str) else "token", "Fetch HuggingFace") as ts:
                if isinstance(hf_cfg, str):
                    step(src_start, "Fetching HuggingFace", f"Likes by {hf_cfg}")
                    fetcher = huggingface.Likes(username=hf_cfg)
                else:
                    step(src_start, "Fetching HuggingFace", "Liked models and datasets")
                    fetcher = huggingface.Likes(token=huggingface_token)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="huggingface")
                ts.add(len(added))
                step(src_start, "HuggingFace", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t6)

    # Twitter/X (via TwitterAPI.io OR — if `--twikit` was passed —
    # via cookie-authenticated twikit using Safari cookies)
    if has_twitter:
        tw_cfg = twitter_config if isinstance(twitter_config, dict) else {}
        tw_username = tw_cfg.get("username", "")
        tw_api_key = tw_cfg.get("api_key") or twitterapiio_api_key
        tw_include_tweets = bool(tw_cfg.get("include_tweets", True))
        tw_include_replies = bool(tw_cfg.get("include_replies", True))
        tw_filter_replies = bool(tw_cfg.get("filter_replies", False))
        tw_max_parents = tw_cfg.get("max_parents", 5000)

        tw_label_parts = []
        if tw_include_tweets:
            tw_label_parts.append(f"tweets @{tw_username}" if tw_username else "tweets")
        if tw_include_replies:
            tw_label_parts.append(f"reply URLs @{tw_username}" if tw_username else "reply URLs")

        def _t7():
            with _ts("twitter", f"@{tw_username}" if tw_username else "tweets", "Fetch Twitter/X") as ts:
                suffix = "" if tw_api_key else " [no API key]"
                step(src_start, "Fetching Twitter/X", (", ".join(tw_label_parts) or "nothing enabled") + suffix)

                if tw_label_parts and tw_api_key:
                    from sources.sql import update_twitter_cursor

                    twitter_fetcher = twitter.Tweets(
                        username=tw_username,
                        api_key=tw_api_key,
                        include_tweets=tw_include_tweets,
                        include_replies=tw_include_replies,
                        filter_replies=tw_filter_replies,
                        max_parents=tw_max_parents,
                        min_interval=float(tw_cfg.get("min_interval", 0.34)),
                        user_id=tw_cfg.get("user_id"),
                    )
                    # VIP: 3000 tweet hard cap, no date fence (we want
                    # the full backfill). Non-VIP: 300 tweets, also no
                    # date fence — the cap alone bounds the work.
                    # Per-personality `tw_cfg.max_tweets` still wins
                    # when explicitly set, so a power user can override.
                    default_cap = 3000 if vip else 300
                    max_tweets = int(tw_cfg.get("max_tweets", default_cap))
                    # max_pages = safety ceiling; max_tweets is the
                    # precise cap. ~20 tweets/page → 200 pages covers
                    # 3000 tweets with headroom for replies.
                    max_pages = int(tw_cfg.get("max_pages", 200))

                    # Credit gate. The budget is bound to the
                    # *paying* account resolved earlier:
                    #   • Plain VIP (no sponsor)   → free (platform absorbs)
                    #   • Sponsored VIP            → bill the sponsor (1–4¢ per page)
                    #   • Non-VIP (legacy)         → bill the personality itself
                    # `billing_vip` is the payer's VIP status; if it's
                    # True the budget short-circuits and no rows land
                    # in credit_events.
                    from sources.credits import twitter_budget as _twitter_budget

                    tw_budget = _twitter_budget(
                        database_url,
                        billing_user_id,
                        billing_vip,
                        personality_user_id=user_id,
                    )
                    raw_docs = twitter_fetcher(
                        max_pages=max_pages,
                        existing_urls=_existing(),
                        stop_date="",  # no date fence — bound only by max_tweets
                        max_tweets=max_tweets,
                        budget=tw_budget,
                    )
                    if not billing_vip:
                        who = "" if billing_user_id == user_id else f" → user_id={billing_user_id}"
                        print(
                            f"    Twitter: {tw_budget.spent}¢ spent across {tw_budget.calls} calls{who}"
                            + (" (stopped early — insufficient balance)" if tw_budget.refused_at_call else "")
                        )
                    batch_dates = sorted({str(d["date"]) for d in raw_docs.values() if d.get("date")})
                    if batch_dates:
                        update_twitter_cursor(database_url, user_id, newest=batch_dates[-1], oldest=batch_dates[0])
                    if bool(tw_cfg.get("filter_tweets", False)):
                        raw_docs = twitter.filter_tweets(raw_docs)
                    added = _merge_and_track(raw_docs, source_key="twitter")
                    ts.add(len(added))
                    step(src_start, "Twitter/X", f"+{len(added)} new" if added else "Up to date")
                else:
                    ts.skip("disabled (no API key)" if not tw_api_key else "all modes disabled")
                    step(src_start, "Twitter/X", "disabled (no API key)" if not tw_api_key else "all modes disabled")

        # twikit-replacement task: pulls the same shape of docs as the
        # twitterapi.io path but goes through cookie-authenticated
        # twikit, reading the operator's Safari session. Useful for
        # bulk backfills (`make run TWIKIT=1`) so we don't consume
        # twitterapi.io credit. When this is on, both the
        # twitterapi.io task and the per-personality `cookies_enc`
        # task below are skipped — twikit alone provides tweets +
        # likes for the target handle.
        def _t7_twikit():
            with _ts("twitter", f"@{tw_username}" if tw_username else "twikit", "Fetch Twitter/X (twikit)") as ts:
                if not tw_username:
                    ts.skip("no twitter username configured")
                    step(src_start, "Twitter/X (twikit)", "no username")
                    return
                try:
                    from sources.twitter.cookies import get_safari_cookies

                    creds = get_safari_cookies()
                except Exception as e:
                    ts.skip(f"no Safari cookies: {e}")
                    step(src_start, "Twitter/X (twikit)", f"disabled ({e})")
                    return
                # Smear the per-personality fetches across the run so
                # we don't blast the session's GraphQL quota in a
                # single sprint. The single-tweet-thread Bookmarks
                # walk already paces page-to-page; this is the
                # personality-to-personality pacer. Override with
                # `TWIKIT_PERSONALITY_DELAY` (seconds) if you need
                # to tune it.
                inter_delay = float(os.environ.get("TWIKIT_PERSONALITY_DELAY", "8"))
                if inter_delay > 0:
                    time.sleep(inter_delay)
                step(
                    src_start,
                    "Fetching Twitter/X (twikit)",
                    f"tweets + likes @{tw_username}",
                )

                # Per-page flush — durably save tweets to PG as
                # each twikit page lands so a crash mid-pagination
                # doesn't throw away the pages already fetched.
                # Applies the same minimal cleaning (`clean_summary`
                # / `clean_title`) the regular pipeline cleanup
                # step runs at end-of-fetch; the final cleanup +
                # tag pass at end of `run_pipeline` is idempotent
                # over rows we already wrote here. Rows are inserted
                # with `indexed = FALSE` (the default) so the next
                # embed pass picks them up.
                from sources.sql import upsert_documents as _upsert
                from sources.utils.cleaning import (
                    clean_summary as _clean_summary,
                )
                from sources.utils.cleaning import (
                    clean_title as _clean_title,
                )

                def _flush_twikit_page(page_docs: dict) -> None:
                    if not page_docs:
                        return
                    cleaned: dict = {}
                    for u, d in page_docs.items():
                        cleaned[u] = {
                            **d,
                            "title": _clean_title(d.get("title") or ""),
                            "summary": _clean_summary(d.get("summary") or ""),
                            "source": d.get("source") or "twitter",
                        }
                    _upsert(database_url, user_id, cleaned)
                    # Mirror into the in-memory `data` dict so the
                    # end-of-pipeline cleanup pass sees these rows
                    # as part of the master set (the indexer reads
                    # off `data`, not PG). `data` is the enclosing
                    # `run_pipeline` scope's working dict.
                    data.update(cleaned)

                fetcher = twitter.Bookmarks(
                    auth_token=creds["auth_token"],
                    ct0=creds["ct0"],
                    target_username=tw_username,
                    on_page_flush=_flush_twikit_page,
                )
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="twitter")
                ts.add(len(added))
                step(
                    src_start,
                    "Twitter/X (twikit)",
                    f"+{len(added)} new" if added else "Up to date",
                )

        if twitter_via_twikit:
            fetch_tasks.append(_t7_twikit)
        else:
            fetch_tasks.append(_t7)

            # Cookie-authenticated sync (tweets + likes + bookmarks). Fires
            # only when the user pasted auth_token + ct0 through the profile
            # modal; those get stored encrypted as sources.twitter.cookies_enc.
            cookies_enc = tw_cfg.get("cookies_enc") if isinstance(tw_cfg, dict) else None
            if cookies_enc:

                def _t8(enc=cookies_enc):
                    with _ts("twitter", "cookies", "Fetch Twitter/X (cookies)") as ts:
                        from sources.utils.secrets import decrypt as _decrypt

                        raw = _decrypt(enc)
                        if raw:
                            try:
                                import json as _json

                                creds = _json.loads(raw)
                            except Exception:
                                creds = {}
                            auth_tok = creds.get("auth_token")
                            ct0 = creds.get("ct0")
                            if auth_tok and ct0:
                                step(src_start, "Fetching Twitter/X", "Cookies (tweets + likes + bookmarks)")
                                added = _merge_and_track(
                                    twitter.Bookmarks(auth_token=auth_tok, ct0=ct0)(existing_urls=_existing()),
                                    source_key="twitter",
                                )
                                ts.add(len(added))
                                step(src_start, "Twitter/X", f"+{len(added)} new (cookies)" if added else "Up to date")
                            else:
                                ts.skip("missing auth_token or ct0")
                        else:
                            ts.skip("could not decrypt cookies")

                fetch_tasks.append(_t8)

    # Websites = unified web-source list. Each entry is already
    # resolved by the Rust probe into one of:
    #   • { kind: "feed",    url, tags }   → routed through blog.Feed
    #   • { kind: "sitemap", url, url_filter, tags } → blog.Sitemap
    # Entries without a `kind` are treated as sitemap for backward
    # compatibility.
    #
    # Per-website source key: we tag each doc with the site's
    # hostname (e.g. `mixedbread.com`) instead of a shared `blog`
    # bucket. That surfaces one filter chip per website on the search
    # page, and the chip can render the site's favicon so users
    # recognize it at a glance.
    # Local alias for the module-level helper so the rest of
    # `run_pipeline` keeps reading naturally. Behaviour is identical;
    # see the docstring on `hostname_source_key` above.
    _hostname_source_key = hostname_source_key

    feed_configs: list[dict] = []
    sitemap_configs: list[dict] = []
    for ws in sources_config.get("websites") or []:
        if not isinstance(ws, dict) or not ws.get("url"):
            continue
        kind = ws.get("kind") or "sitemap"
        # `input` is the URL the user originally pasted. We prefer it
        # for hostname derivation because the resolved `url` often
        # points at a sub-path like /sitemap-index.xml that hides the
        # canonical host.
        entry = {
            "url": ws["url"],
            "tags": ws.get("tags") or ["blog"],
            "input": ws.get("input") or ws["url"],
        }
        if kind == "feed":
            feed_configs.append(entry)
        else:
            entry["url_filter"] = ws.get("url_filter") or None
            sitemap_configs.append(entry)

    if feed_configs:
        from .. import blog as _blog_feed

        for f_cfg in feed_configs:

            def _t9(cfg=f_cfg, blog=_blog_feed):
                f_url = cfg["url"]
                src_key = _hostname_source_key(cfg.get("input") or f_url)
                with _ts(src_key or "feed", "feed", f"Fetch feed {src_key or f_url}") as ts:
                    if _website_is_fresh(src_key):
                        ts.skip(f"cooldown < {_website_ttl_hours:g}h")
                        step(src_start, "Feed", f"{src_key}: skipped (< {_website_ttl_hours:g}h since last fetch)")
                        return
                    step(src_start, "Fetching feed", f_url.split("/")[2] if "/" in f_url else f_url)
                    fetcher = blog.Feed(feed_url=f_url, tags=cfg.get("tags") or ["blog"])
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key=src_key)
                    ts.add(len(added))
                    step(src_start, "Feed", f"+{len(added)} new" if added else "Up to date")

            fetch_tasks.append(_t9)

    if sitemap_configs:
        from .. import blog as _blog_sitemap

        for sm_cfg in sitemap_configs:

            def _t10(cfg=sm_cfg, blog=_blog_sitemap):
                sm_url = cfg["url"]
                sm_tags = cfg.get("tags") or ["blog"]
                sm_filter = cfg.get("url_filter") or None
                src_key = _hostname_source_key(cfg.get("input") or sm_url)
                with _ts(src_key or "sitemap", "sitemap", f"Fetch sitemap {src_key or sm_url}") as ts:
                    if _website_is_fresh(src_key):
                        ts.skip(f"cooldown < {_website_ttl_hours:g}h")
                        step(src_start, "Sitemap", f"{src_key}: skipped (< {_website_ttl_hours:g}h since last fetch)")
                        return
                    step(src_start, "Fetching sitemap", sm_url.split("/")[2] if "/" in sm_url else sm_url)
                    fetcher = blog.Sitemap(sitemap_url=sm_url, tags=sm_tags, url_filter=sm_filter)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key=src_key)
                    ts.add(len(added))
                    step(src_start, "Sitemap", f"+{len(added)} new" if added else "Up to date")

            fetch_tasks.append(_t10)

    # Google Scholar publications
    scholar_cfg = sources_config.get("scholar")
    if scholar_cfg:
        scholar_uid = scholar_cfg if isinstance(scholar_cfg, str) else scholar_cfg.get("user_id", "")
        sc_max_pages = 3 if isinstance(scholar_cfg, str) else scholar_cfg.get("max_pages", 3)
        sc_min_cites = 0 if isinstance(scholar_cfg, str) else scholar_cfg.get("min_citations", 0)
        if scholar_uid:

            def _t11(uid=scholar_uid, mp=sc_max_pages, mc=sc_min_cites):
                with _ts("scholar", uid, "Fetch Scholar") as ts:
                    from .. import scholar

                    step(src_start, "Fetching Scholar", f"Publications for {uid}")
                    fetcher = scholar.Publications(user_id=uid, max_pages=mp, min_citations=mc)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="scholar")
                    ts.add(len(added))
                    step(src_start, "Scholar", f"+{len(added)} new" if added else "Up to date")

            fetch_tasks.append(_t11)

    # Semantic Scholar (runs after Google Scholar — deduplicates via canonical arXiv/DOI URLs)
    s2_cfg = sources_config.get("semantic_scholar")
    if s2_cfg:
        s2_id = s2_cfg if isinstance(s2_cfg, str) else s2_cfg.get("author_id")
        s2_name = None if isinstance(s2_cfg, str) else s2_cfg.get("author_name")
        s2_max_papers = 300 if isinstance(s2_cfg, str) else s2_cfg.get("max_papers", 300)
        s2_min_cites = 0 if isinstance(s2_cfg, str) else s2_cfg.get("min_citations", 0)

        def _t12(sid=s2_id, sname=s2_name, mp=s2_max_papers, mc=s2_min_cites):
            with _ts("semantic_scholar", str(sid or sname or ""), "Fetch Semantic Scholar") as ts:
                from .. import semantic_scholar

                step(src_start, "Fetching Semantic Scholar", f"Papers for {sid or sname}")
                fetcher = semantic_scholar.Papers(
                    author_id=sid,
                    author_name=sname,
                    max_papers=mp,
                    min_citations=mc,
                )
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="scholar")
                ts.add(len(added))
                step(src_start, "Semantic Scholar", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t12)

    # Stack Exchange network (Stack Overflow + sister sites).
    #
    # One `sources.stackoverflow` config covers the whole network. When
    # the user OAuthed, we stored `associated_sites = [{
    # api_site_parameter, user_id, site_name }, ...]` so we can iterate
    # every site their account is active on. Public-only users (no
    # OAuth) fall back to a single-site crawl against Stack Overflow
    # using the user_id they supplied in the form.
    so_cfg = sources_config.get("stackoverflow")
    if so_cfg:
        from .. import stackoverflow as _stackoverflow

        so_max_pages = 5 if isinstance(so_cfg, str) else so_cfg.get("max_pages", 5)
        so_min_score = 1 if isinstance(so_cfg, str) else so_cfg.get("min_score", 1)

        # Resolve (site, user_id, name) tuples to crawl publicly.
        so_crawls: list[tuple[str, int | None, str]] = []
        if isinstance(so_cfg, dict):
            assoc = so_cfg.get("associated_sites") or []
            for s in assoc:
                if not isinstance(s, dict):
                    continue
                site = s.get("api_site_parameter")
                uid = s.get("user_id")
                label = s.get("site_name") or site
                if site and uid:
                    so_crawls.append((site, int(uid), label))
        if not so_crawls:
            so_uid = None if isinstance(so_cfg, str) else so_cfg.get("user_id")
            so_name = so_cfg if isinstance(so_cfg, str) else so_cfg.get("username")
            if so_uid or so_name:
                so_crawls.append(("stackoverflow", int(so_uid) if so_uid else None, "Stack Overflow"))

        for site_, uid_, label_ in so_crawls:
            # 1. Answers — topics the user has answered on this site.
            def _t13(site=site_, uid=uid_, label=label_, so=_stackoverflow, mp=so_max_pages, ms=so_min_score):
                with _ts("stackoverflow", f"{site}: answers", f"Fetch StackExchange {site} answers") as ts:
                    step(src_start, "Fetching StackExchange", f"{label}: answers")
                    fetcher = so.Answers(user_id=uid, site=site, max_pages=mp, min_score=ms)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="stackoverflow")
                    ts.add(len(added))
                    step(src_start, "StackExchange", f"{label}: +{len(added)}" if added else f"{label}: up to date")

            fetch_tasks.append(_t13)

            # 2. Questions — topics the user has asked on this site.
            def _t14(site=site_, uid=uid_, label=label_, so=_stackoverflow, mp=so_max_pages):
                with _ts("stackoverflow", f"{site}: questions", f"Fetch StackExchange {site} questions") as ts:
                    step(src_start, "Fetching StackExchange", f"{label}: questions")
                    q_fetcher = so.Questions(user_id=uid, site=site, max_pages=mp, min_score=0)
                    added = _merge_and_track(q_fetcher(existing_urls=_existing()), source_key="stackoverflow")
                    ts.add(len(added))
                    step(src_start, "StackExchange", f"{label}: +{len(added)}" if added else f"{label}: up to date")

            fetch_tasks.append(_t14)

        # 3. Favorites (OAuth) — one call per site since /me/favorites
        # is always scoped to a single `site` param. Only fires when
        # access_token_enc is present AND the app key is configured.
        token_enc = so_cfg.get("access_token_enc") if isinstance(so_cfg, dict) else None
        so_key = os.environ.get("STACKOVERFLOW_KEY")
        if token_enc and so_key:
            from sources.utils.secrets import decrypt as _decrypt

            access = _decrypt(token_enc)
            if access:
                fav_sites = [c[0] for c in so_crawls] or ["stackoverflow"]
                for site_ in fav_sites:

                    def _t15(site=site_, access=access, key=so_key, so=_stackoverflow, mp=so_max_pages):
                        with _ts("stackoverflow", f"{site}: favorites", f"Fetch StackExchange {site} favorites") as ts:
                            step(src_start, "Fetching StackExchange", f"{site}: favorites")
                            fav_fetcher = so.Favorites(
                                access_token=access,
                                key=key,
                                max_pages=mp,
                                site=site,
                            )
                            added = _merge_and_track(
                                fav_fetcher(existing_urls=_existing()),
                                source_key="stackoverflow",
                            )
                            ts.add(len(added))
                            step(
                                src_start,
                                "StackExchange",
                                f"{site}: +{len(added)} favorites" if added else f"{site}: up to date",
                            )

                    fetch_tasks.append(_t15)

    # YouTube channels
    youtube_cfg = sources_config.get("youtube")
    if youtube_cfg:
        yt_channels = youtube_cfg if isinstance(youtube_cfg, list) else [youtube_cfg]

        def _t16(channels=yt_channels):
            with _ts("youtube", f"{len(channels)} channel(s)", "Fetch YouTube") as ts:
                step(src_start, "Fetching YouTube", f"{len(channels)} channel(s)")
                fetcher = youtube.Channels(channels=channels)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="youtube")
                ts.add(len(added))
                step(src_start, "YouTube", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t16)

    # Reddit — public submissions + comments (no auth needed). Upvoted
    # posts require OAuth, which Reddit's 2024 policy makes painful to
    # register; skipping for now.
    reddit_cfg = sources_config.get("reddit")
    if reddit_cfg:
        reddit_user = reddit_cfg if isinstance(reddit_cfg, str) else reddit_cfg.get("username", "")
        rd_max_pages = 5 if isinstance(reddit_cfg, str) else reddit_cfg.get("max_pages", 5)
        if reddit_user:

            def _t17(user=reddit_user, mp=rd_max_pages):
                with _ts("reddit", f"u/{user}", "Fetch Reddit") as ts:
                    from .. import reddit

                    step(src_start, "Fetching Reddit", f"Activity from u/{user}")
                    fetcher = reddit.Posts(username=user, max_pages=mp)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="reddit")
                    ts.add(len(added))
                    step(src_start, "Reddit", f"+{len(added)} new" if added else "Up to date")

            fetch_tasks.append(_t17)

    # GitHub own repos (not stars)
    github_repos_cfg = sources_config.get("github_repos")
    if github_repos_cfg:
        gh_repo_users = github_repos_cfg if isinstance(github_repos_cfg, list) else [github_repos_cfg]

        def _t18(users=gh_repo_users):
            with _ts("github", f"repos {', '.join(users)}", "Fetch GitHub repos") as ts:
                step(src_start, "Fetching GitHub repos", f"Own repos for {', '.join(users)}")
                fetcher = github.Repositories(users=users)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="github")
                ts.add(len(added))
                step(src_start, "GitHub repos", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t18)

    # GitHub gists
    github_gists_cfg = sources_config.get("github_gists")
    if github_gists_cfg:
        gh_gist_users = github_gists_cfg if isinstance(github_gists_cfg, list) else [github_gists_cfg]

        def _t19(users=gh_gist_users):
            with _ts("github", f"gists {', '.join(users)}", "Fetch GitHub gists") as ts:
                step(src_start, "Fetching GitHub gists", f"Gists for {', '.join(users)}")
                fetcher = github.Gists(users=users)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="github")
                ts.add(len(added))
                step(src_start, "GitHub gists", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t19)

    # YouTube search (talks/interviews on other channels)
    yt_search_cfg = sources_config.get("youtube_search")
    if yt_search_cfg:
        if isinstance(yt_search_cfg, dict):
            yts_queries = yt_search_cfg.get("queries", [])
            yts_must_contain = yt_search_cfg.get("must_contain")
            yts_max_results = yt_search_cfg.get("max_results", 30)
        elif isinstance(yt_search_cfg, list):
            yts_queries = yt_search_cfg
            yts_must_contain = None
            yts_max_results = 30
        else:
            yts_queries = [yt_search_cfg]
            yts_must_contain = None
            yts_max_results = 30

        def _t20(qs=yts_queries, mc=yts_must_contain, mr=yts_max_results):
            with _ts("youtube", f"search: {qs[0] if qs else ''}", "Fetch YouTube search") as ts:
                step(src_start, "Searching YouTube", f"Talks featuring {qs[0] if qs else '?'}")
                fetcher = youtube.Search(queries=qs, max_results=mr, must_contain=mc)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="youtube")
                ts.add(len(added))
                step(src_start, "YouTube search", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t20)

    # Wikipedia references
    wiki_cfg = sources_config.get("wikipedia")
    if wiki_cfg:
        wiki_pages = wiki_cfg if isinstance(wiki_cfg, list) else [wiki_cfg]

        def _t21(pages=wiki_pages):
            with _ts("wikipedia", pages[0] if pages else "", "Fetch Wikipedia") as ts:
                from .. import wikipedia

                step(src_start, "Fetching Wikipedia", f"References from {pages[0]}")
                fetcher = wikipedia.References(pages=pages)
                added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="wikipedia")
                ts.add(len(added))
                step(src_start, "Wikipedia", f"+{len(added)} new" if added else "Up to date")

        fetch_tasks.append(_t21)

    # DBLP publications
    dblp_cfg = sources_config.get("dblp")
    if dblp_cfg:
        dblp_author = dblp_cfg if isinstance(dblp_cfg, str) else dblp_cfg.get("author", "")
        dblp_max = 200 if isinstance(dblp_cfg, str) else dblp_cfg.get("max_results", 200)
        if dblp_author:

            def _t22(author=dblp_author, mr=dblp_max):
                with _ts("dblp", author, "Fetch DBLP") as ts:
                    from .. import dblp

                    step(src_start, "Fetching DBLP", f"Publications by {author}")
                    fetcher = dblp.Publications(author=author, max_results=mr)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="scholar")
                    ts.add(len(added))
                    step(src_start, "DBLP", f"+{len(added)} new" if added else "Up to date")

            fetch_tasks.append(_t22)

    # arXiv author papers
    arxiv_cfg = sources_config.get("arxiv")
    if arxiv_cfg:
        arxiv_author = arxiv_cfg if isinstance(arxiv_cfg, str) else arxiv_cfg.get("author", "")
        arxiv_max = 200 if isinstance(arxiv_cfg, str) else arxiv_cfg.get("max_results", 200)
        if arxiv_author:

            def _t23(author=arxiv_author, mr=arxiv_max):
                with _ts("arxiv", author, "Fetch arXiv") as ts:
                    from .. import arxiv

                    step(src_start, "Fetching arXiv", f"Papers by {author}")
                    fetcher = arxiv.Papers(author=author, max_results=mr)
                    added = _merge_and_track(fetcher(existing_urls=_existing()), source_key="arxiv")
                    ts.add(len(added))
                    step(src_start, "arXiv", f"+{len(added)} new" if added else "Up to date")

            fetch_tasks.append(_t23)

    # ── Dispatch ─────────────────────────────────────────────────────
    #
    # All blocks above pushed their fetcher closures onto `fetch_tasks`.
    # Now run them. With `n_workers > 1` we use a `ThreadPoolExecutor`
    # so the IO-bound fetchers overlap (Python releases the GIL on
    # socket I/O). Each task body is wrapped in `_ts(...)`, which
    # already swallows exceptions and records a `failed` row, so the
    # pool never sees a raised exception — but we still call
    # `.result()` to surface any unrelated bug from the harness.
    n_tasks = len(fetch_tasks)
    if n_tasks == 0:
        step(42, "Sources", "No fetchers configured for this user")
    else:
        if n_workers <= 0:
            n_effective = min(os.cpu_count() or 1, n_tasks)
        else:
            n_effective = max(1, min(n_workers, n_tasks))

        step(src_start, "Sources", f"{n_tasks} fetcher(s) on {n_effective} worker(s)")

        # Progress emitter — bumps the bar from src_start towards
        # src_end as tasks finish, regardless of completion order.
        _completed = [0]
        _completed_lock = threading.Lock()

        def _wrap(fn):
            def _w():
                try:
                    fn()
                finally:
                    with _completed_lock:
                        _completed[0] += 1
                        pct = src_start + (src_end - src_start) * _completed[0] // n_tasks
                        done = _completed[0]
                    step(pct, "Sources", f"{done}/{n_tasks} fetchers complete")

            return _w

        if n_effective == 1:
            for fn in fetch_tasks:
                _wrap(fn)()
        else:
            with ThreadPoolExecutor(max_workers=n_effective, thread_name_prefix="fetch") as ex:
                futures = [ex.submit(_wrap(fn)) for fn in fetch_tasks]
                for f in futures:
                    # `_ts` already absorbed every fetcher exception. A
                    # raised here would be a bug in the harness — let
                    # it surface so we don't silently lose work.
                    f.result()

    # ── Pull in browser-synced docs ────────────────────────────────────
    #
    # The "Sync now" button on the search page POSTs new docs straight
    # into Postgres via /auth/me/documents/bulk. Those rows land with
    # `indexed = false` but never go through the Python fetchers, so
    # `new_urls` (which is populated only by `_merge_and_track`) misses
    # them. Without this step the dead-link check + the auto-tagger
    # would silently skip them on this run, and they'd be indexed with
    # raw fetcher metadata only.
    #
    # We treat any pre-existing `indexed = false` row as "needs full
    # processing this run": the cleaning loop already iterates `data`
    # so it covers them; this fold ensures the link-check and tagger
    # also include them.
    try:
        prerun_unindexed = load_unindexed_documents(database_url, user_id)
        js_synced = set(prerun_unindexed.keys()) - new_urls
        if js_synced:
            new_urls.update(js_synced)
            personality_urls.update(js_synced)
            # Make sure any JS-only metadata (title/summary/etc. the
            # browser fetcher attached) is present in `data`. If a row
            # was inserted by the bulk endpoint after `load_documents`
            # ran above, it would otherwise be missing here.
            for u, d in prerun_unindexed.items():
                if u not in data:
                    data[u] = d
            print(f"  Browser-synced backlog: {len(js_synced)} doc(s) added to this run")
    except Exception as _exc:
        # Soft-fail: an isolated DB hiccup shouldn't tank the rest of
        # the pipeline. The next run will pick the docs back up.
        print(f"  Browser-sync fold-in skipped: {_exc}")

    total_new = len(new_urls)
    step(42, "Sources complete", f"{total_new} new document{'s' if total_new != 1 else ''} found")

    # =========================================================================
    # Data Cleaning
    # =========================================================================

    _mark_stage("clean")
    t0 = time.perf_counter()
    step(44, "Cleaning", f"Validating {len(data):,} documents")

    from .cleaning import clean_summary, clean_title

    for _url, document in data.items():
        for field in ["title", "tags", "summary", "date"]:
            if document.get(field) is None:
                document[field] = "" if field != "tags" else []
        for field in ["title", "summary"]:
            if isinstance(document.get(field), str):
                document[field] = document[field].encode("utf-8", "replace").decode("utf-8")
        document["title"] = clean_title(document.get("title", ""))
        document["summary"] = clean_summary(document.get("summary", ""))

    before = len(data)
    data = {url: doc for url, doc in data.items() if url.strip() and doc.get("title", "").strip()}

    # Drop low-information docs that slip past per-source filters:
    #   • title AND summary each contain a single token (e.g. one-word
    #     repo names with no description) — nothing for the index to
    #     match on.
    #   • title AND summary are both purely numeric (year-only entries,
    #     auto-generated catalog stubs).
    # We require BOTH conditions to fail so legitimate one-word titles
    # with rich summaries (or vice versa) are kept.
    _NUMERIC_ONLY = re.compile(r"^[\d\W_]+$")

    def _token_count(s: str) -> int:
        return len(re.findall(r"\w+", s or ""))

    def _is_numeric_only(s: str) -> bool:
        s = (s or "").strip()
        return bool(s) and bool(_NUMERIC_ONLY.match(s))

    # Web-page extensions show up in the title only when we couldn't
    # derive a real title for the URL — i.e. the page wasn't sampled
    # (sitemap too large) AND the URL slug ended with `.html` /
    # `.php` / etc. Those titles read as filenames in search
    # results, so we drop them. Source-level fallout is intentional:
    # if every slug on a site has `.html`, the whole source vanishes.
    _BAD_TITLE_EXTS = (".html", ".htm", ".php", ".aspx", ".asp", ".jsp", ".json")

    def _has_page_ext(s: str) -> bool:
        s = (s or "").lower()
        return any(ext in s for ext in _BAD_TITLE_EXTS)

    def _keep(doc: dict) -> bool:
        title = doc.get("title", "") or ""
        summary = doc.get("summary", "") or ""
        if _token_count(title) <= 1 and _token_count(summary) <= 1:
            return False
        if _is_numeric_only(title) and _is_numeric_only(summary):
            return False
        if _has_page_ext(title):
            return False
        return True

    data = {url: doc for url, doc in data.items() if _keep(doc)}
    removed = before - len(data)

    def _normalize_url(u: str) -> str:
        u = u.strip().rstrip("/")
        if u.startswith("http://"):
            u = "https://" + u[7:]
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return f"{parsed.scheme}://{host}{parsed.path}{parsed.query}"

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for url in list(data.keys()):
        norm = _normalize_url(url)
        if norm in seen:
            duplicates.append(url)
        else:
            seen[norm] = url
    for url in duplicates:
        del data[url]

    cleaned = removed + len(duplicates)
    step(48, "Cleaned", f"{len(data):,} documents" + (f" ({cleaned} removed)" if cleaned else ""))
    timings.append(("Clean data", time.perf_counter() - t0))

    # =========================================================================
    # Dead Link Check (new URLs only)
    # =========================================================================

    if new_urls:
        _mark_stage("link_check")
        t0 = time.perf_counter()
        from .dead_links import DeadLinks

        step(49, "Checking links", f"Probing {len(new_urls)} new URLs")
        checker = DeadLinks()
        dead = checker.check(new_urls)
        if dead:
            for url in dead:
                data.pop(url, None)
                new_urls.discard(url)
                personality_urls.discard(url)
            # Remember these so subsequent runs short-circuit them in
            # `_existing()` instead of re-fetching → re-killing them.
            mark_urls_dead(database_url, dead)
            _dead_urls.update(dead)
            step(50, "Dead links", f"Removed {len(dead)}")
        else:
            step(50, "Links OK", "No dead links found")
        timings.append(("Dead link check", time.perf_counter() - t0))

    # =========================================================================
    # Generate Extra Tags
    # =========================================================================

    _mark_stage("tag")
    # Force a re-tag of every document, even when no new URLs were
    # fetched. Useful when the tagger logic itself changed (e.g.
    # switching to flashtext): bumping ``KNOWLEDGE_RETAG=1`` re-runs
    # `get_extra_tags` over the full library and persists the fresh
    # `extra-tags` lists into PG.
    _retag = os.environ.get("KNOWLEDGE_RETAG", "").strip().lower() in {"1", "true", "yes"}

    # Tag-vocabulary policy: the auto-tagger sees only (this user's
    # own tags) ∪ (VIP users' tags). Non-VIP users do NOT contribute to
    # the cross-user pool — that prevents a noisy library's per-user
    # tags from leaking into everyone else's vocabulary.
    #
    # We always recompute this from PG at the start of the tag stage
    # rather than trusting the `shared_tags` snapshot the caller passed
    # in. With concurrent pipeline runs (per-user reruns + the long
    # sequential job both write to `documents` mid-flight) the snapshot
    # captured at run.py startup can lag behind PG by minutes — and
    # that's fine, since the live floor is the right one to tag against.
    # The leaked-tags check stays: it catches the ORIGINAL regression
    # (non-VIP tags polluting the global pool) which never depends on
    # timing.
    from sources.sql import get_user_tags as _get_user_tags
    from sources.sql import get_vip_tags as _get_vip_tags

    _vip_pool = set(_get_vip_tags(database_url))
    _own_pool = set(_get_user_tags(database_url, user_id))
    _expected_floor = _vip_pool | _own_pool
    if shared_tags is not None:
        _vocab = set(shared_tags)
        _leaked = _vocab - _expected_floor
        assert not _leaked, (
            f"shared_tags contains {len(_leaked)} tags that aren't in "
            "the VIP pool or this user's own tags "
            f"(sample: {sorted(_leaked)[:5]}). "
            "Non-VIP users must not contribute to the cross-user vocabulary."
        )
    # Adopt the live floor for tagging, regardless of what the caller passed.
    shared_tags = sorted(_expected_floor)

    t0 = time.perf_counter()
    if new_urls or _retag:
        reason = "all docs (KNOWLEDGE_RETAG)" if _retag and not new_urls else f"{len(data):,} documents"
        step(50, "Generating tags", f"Extracting keywords from {reason}")
        data = tags.get_extra_tags(data=data, shared_tags=shared_tags)
        step(62, "Tags generated", f"{len(data):,} documents tagged")
    else:
        step(62, "Tags", "Skipped (no new documents — set KNOWLEDGE_RETAG=1 to force)")
    timings.append(("Generate extra tags", time.perf_counter() - t0))

    # =========================================================================
    # Save Database (to PostgreSQL)
    # =========================================================================

    t0 = time.perf_counter()
    # Scope to URLs this personality touched (via fetchers or prior runs).
    scoped_db = {url: data[url] for url in personality_urls if url in data}
    step(64, "Saving", f"Upserting {len(scoped_db):,} documents to PG")
    upsert_documents(database_url, user_id, scoped_db)
    step(70, "Saved", f"{len(scoped_db):,} documents in PG")
    timings.append(("Save database", time.perf_counter() - t0))

    # Source filters are now derived by the `user_source_counts` view and
    # served via GET /api/users/{slug}/sources — no client-side computation
    # or disk dump is needed here.

    # =========================================================================
    # Index Documents via API
    # =========================================================================

    # Indexing is now a separate concern, owned by the indexer
    # daemon (`sources.indexer_daemon`). `run_pipeline` defaults
    # to skipping the index stage entirely so a `make run` only
    # touches Postgres — fast, no API dependency, no race with the
    # daemon over the same on-disk index. Callers that explicitly
    # want the inline indexer (the daemon itself, or one-shot
    # `make run` invocations that pass `--index`) opt in via the
    # `do_index` keyword.
    if not do_index:
        finish_pipeline_run(
            database_url,
            run_id,
            success=True,
            new_documents=total_new,
            total_documents=len(data),
            duration_secs=round(time.perf_counter() - pipeline_start, 2),
            timings=[{"step": label, "duration_secs": round(elapsed, 2)} for label, elapsed in timings],
        )
        total = time.perf_counter() - pipeline_start
        print(f"\n  {name}: {len(data):,} documents, {total_new} new ({_fmt(total)}, no index)")
        for label, elapsed in timings:
            pct = elapsed / total * 100 if total > 0 else 0
            print(f"    {label:<25s} {_fmt(elapsed):>8s}  ({pct:4.1f}%)")
        return

    _mark_stage("index")
    t0 = time.perf_counter()
    import urllib.error
    import urllib.request

    api_base = os.environ.get("API_URL", "http://localhost:8080")
    api_key = os.environ.get("ADMIN_API_KEY", "")
    BATCH = 300

    auth_headers = {"Content-Type": "application/json"}
    if api_key:
        auth_headers["X-API-Key"] = api_key

    # Check if the search index already exists AND is healthy. We've
    # seen broken indices on disk (empty chunk files, num_embeddings=0
    # despite num_documents > 0) silently masquerade as "exists" because
    # the directory is on disk and the API responds 200. The pipeline
    # would then take the incremental path, only embed the ~handful of
    # rows currently flagged `indexed=false`, and leave the broken state
    # in place forever.
    #
    # Treat the index as "exists" only when the API reports it AND
    # `num_embeddings > 0` (or `num_documents == 0`, i.e. a clean empty
    # state). Any other shape — directory present but empty, num_docs
    # set but num_embeddings 0 — counts as broken: delete the index,
    # reset every doc's `indexed=false` in PG, and fall through to the
    # fresh-rebuild path below. `make run` is then self-healing without
    # paying the from-scratch cost in the common case.
    def _heal_broken_index(reason: str) -> None:
        """Wipe a broken index + mark docs un-indexed so we rebuild fresh."""
        print(f"  Warning: index '{index_name}' is broken ({reason}) — repairing")
        try:
            del_req = urllib.request.Request(
                f"{api_base}/indices/{index_name}",
                headers=auth_headers,
                method="DELETE",
            )
            with urllib.request.urlopen(del_req, timeout=30):
                pass
        except Exception as e:
            print(f"  Warning: failed to delete broken index: {e}")
        try:
            import psycopg as _heal_psycopg

            with _heal_psycopg.connect(database_url) as _heal_conn:
                with _heal_conn.cursor() as _heal_cur:
                    _heal_cur.execute(
                        "UPDATE documents SET indexed=false, updated_at=now() WHERE user_id = %s AND indexed = true",
                        (user_id,),
                    )
        except Exception as e:
            print(f"  Warning: failed to reset indexed flags: {e}")

    index_exists = False
    try:
        with urllib.request.urlopen(f"{api_base}/indices/{index_name}", timeout=5) as resp:
            if resp.status == 200:
                info = json.loads(resp.read())
                n_docs = int(info.get("num_documents") or 0)
                n_emb = int(info.get("num_embeddings") or 0)
                if n_docs == 0 or n_emb > 0:
                    index_exists = True
                else:
                    # Broken: docs claimed but no embeddings on disk.
                    _heal_broken_index(f"num_documents={n_docs}, num_embeddings={n_emb}")
    except urllib.error.HTTPError as e:
        # The API distinguishes "not found" (404 INDEX_NOT_FOUND) from
        # "exists but failed to load" (500 NEXT_PLAID_ERROR / "No data
        # to merge"). The latter signals the same disk-corruption mode
        # we hit with chunks of 128 bytes — clean it up here so the
        # rebuild path below produces a healthy index. 404 just falls
        # through with index_exists=False and the fresh-build runs.
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code >= 500 or "NEXT_PLAID_ERROR" in body or "No data to merge" in body:
            _heal_broken_index(f"HTTP {e.code} {body[:100]}")
    except Exception:
        # API down or transient — leave index_exists=False; the
        # fresh-rebuild path handles it.
        pass

    # ── Drift purge ──────────────────────────────────────────────
    # The ColBERT index and Postgres `documents` can drift apart:
    # docs deleted from PG (source pruned, slug renamed, manual
    # cleanup) stay in the index until something explicitly removes
    # them. Search returns those ghosts, but the sources panel —
    # backed by the `user_source_counts` view over PG — doesn't list
    # them, so users see unfilterable results.
    #
    # Fix: before incremental indexing, fetch every URL the index
    # currently holds and DELETE anything not present-and-live in PG.
    # `data` here was loaded via `load_documents`, which returns ALL
    # rows including soft-deleted ones (its other caller, the
    # fetcher-merge dedup, needs them so a previously-deleted URL
    # doesn't get silently re-imported). For the ghost-purge we want
    # the strict live set, so re-query PG directly with
    # `deleted = FALSE`. Without this filter, soft-deleted URLs stay
    # in `data.keys()`, the purge finds zero ghosts, and the index
    # carries those rows forever — manifesting as a permanent
    # `pg_drift` verdict equal to the user's deleted count.
    if index_exists:
        try:
            list_payload = json.dumps({"condition": "url != ?", "parameters": [""]}).encode()
            list_req = urllib.request.Request(
                f"{api_base}/indices/{index_name}/metadata/get",
                data=list_payload,
                headers=auth_headers,
                method="POST",
            )
            with urllib.request.urlopen(list_req, timeout=60) as resp:
                indexed_meta = json.loads(resp.read()).get("metadata", [])
            index_urls = {m.get("url") for m in indexed_meta if m.get("url")}
            import psycopg as _drift_psycopg

            with _drift_psycopg.connect(database_url) as _drift_conn:
                with _drift_conn.cursor() as _drift_cur:
                    _drift_cur.execute(
                        "SELECT url FROM documents WHERE user_id = %s AND deleted = FALSE",
                        (user_id,),
                    )
                    live_urls = {row[0] for row in _drift_cur.fetchall()}
            ghost_urls = sorted(index_urls - live_urls)
            if ghost_urls:
                step(
                    94,
                    "Purging stale index docs",
                    f"{len(ghost_urls)} URLs no longer in DB",
                )
                # Server caps DELETE conditions at MAX_DELETE_BATCH_CONDITIONS=200.
                DELETE_CHUNK = 200
                for j in range(0, len(ghost_urls), DELETE_CHUNK):
                    chunk = ghost_urls[j : j + DELETE_CHUNK]
                    placeholders = ",".join("?" for _ in chunk)
                    purge_payload = json.dumps(
                        {
                            "condition": f"url IN ({placeholders})",
                            "parameters": list(chunk),
                        }
                    ).encode()
                    purge_req = urllib.request.Request(
                        f"{api_base}/indices/{index_name}/documents",
                        data=purge_payload,
                        headers=auth_headers,
                        method="DELETE",
                    )
                    try:
                        with urllib.request.urlopen(purge_req, timeout=120):
                            pass
                    except Exception as e:
                        # Non-fatal — leaking a ghost is preferable
                        # to bailing out of the run.
                        print(f"  Warning: ghost-purge chunk {j // DELETE_CHUNK + 1} failed: {e}")
                # Give the server's batched-delete worker time to drain.
                time.sleep(3)
        except Exception as e:
            print(f"  Warning: drift-detection skipped ({e})")

    # Source of truth for "what still needs embedding" is the DB: every
    # row with `indexed = FALSE`. That includes (a) docs the pipeline
    # just inserted this run, (b) docs the user saved via the in-app
    # "Save" button since the last run, and (c) docs whose content was
    # just updated (upsert_documents resets indexed=false when title or
    # summary or tags changed). When the ColBERT index doesn't exist
    # yet, we widen the set to everything the user owns so the fresh
    # index isn't starved.
    unindexed = load_unindexed_documents(database_url, user_id)
    if not index_exists:
        # Fresh index — embed the full personality snapshot.
        to_index_docs = {u: d for u, d in data.items() if u in personality_urls}
    else:
        to_index_docs = unindexed
    urls_to_index = list(to_index_docs.keys())

    if urls_to_index:
        # Ensure index is declared
        if not index_exists:
            step(96, "Creating index", f"Declaring search index '{index_name}'")
            payload = json.dumps({"name": index_name, "config": {"nbits": 2}}).encode()
            req = urllib.request.Request(
                f"{api_base}/indices",
                data=payload,
                headers=auth_headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10):
                    pass
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    print(f"  Warning: create index failed: {e}")

        # Build document texts and metadata
        docs_to_index = []
        metadata_to_index = []
        # Keep URLs in lock-step with their batch position so we can mark
        # exactly the rows we successfully pushed.
        urls_aligned: list[str] = []
        for url in urls_to_index:
            doc = to_index_docs.get(url) or data.get(url)
            if not doc:
                continue
            title = doc.get("title", "")
            doc_tags = doc.get("tags", [])
            extra = doc.get("extra-tags", [])
            summary = doc.get("summary", "")
            source = doc.get("source", "") or ""
            website = website_name(url)
            # Append the source bucket + URL-derived website name at
            # the end of the indexed text. ColBERT can then match a
            # query like "lighton" or "huggingface" against the
            # source chip name OR the URL host, so the source filter
            # surfaces the right chips even when the user hasn't
            # typed any of the document's title or summary words.
            text = f"{title} {' '.join(doc_tags)} {' '.join(extra)} {summary[:200]} {source} {website}".strip()
            if not text:
                continue
            docs_to_index.append(text)
            urls_aligned.append(url)
            # `linked_urls` is JSON-stringified; the index treats it as
            # an opaque text blob and the frontend `JSON.parse`s it
            # when rendering. `link_hosts` mirrors the PG array column,
            # comma-encoded for the same reason `tags` is — PyLate's
            # index filter can't reach into a SQL array, so we keep
            # the whole-word comma-aware LIKE pattern compatible
            # across both columns.
            linked_urls_raw = doc.get("linked_urls") or []
            link_hosts_list = doc.get("link_hosts") or []
            metadata_to_index.append(
                {
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "date": doc.get("date", ""),
                    "tags": ",".join(doc_tags),
                    "extra_tags": ",".join(extra),
                    # Source must travel with the index entry so search
                    # results land in the right filter bucket without a
                    # round-trip to PG.
                    "source": doc.get("source", ""),
                    "source_url": doc.get("source_url") or "",
                    "linked_urls": json.dumps(linked_urls_raw) if isinstance(linked_urls_raw, list) else "[]",
                    "link_hosts": ",".join(link_hosts_list),
                }
            )

        if docs_to_index:
            n_batches = (len(docs_to_index) + BATCH - 1) // BATCH
            label = "Building index" if not index_exists else "Indexing"
            step(96, label, f"{len(docs_to_index)} documents in {n_batches} batches")
            indexed = 0
            # Purge any existing chunks for these URLs BEFORE we start
            # the upsert loop. /update_with_encoding APPENDS rather
            # than upserts, so a retag without prior deletion leaves
            # the stale metadata floating in the index and search
            # results show old tags. The delete endpoint is async and
            # batched (DELETE_BATCH_MAX_WAIT defaults to 2 s on the
            # server), so we send all deletes first, then wait for
            # the queue to drain before pushing fresh chunks.
            if index_exists and urls_aligned:
                # MAX_DELETE_BATCH_CONDITIONS = 200 server-side; chunk
                # the URL list to stay under that even on huge runs.
                DELETE_CHUNK = 200
                for j in range(0, len(urls_aligned), DELETE_CHUNK):
                    chunk = urls_aligned[j : j + DELETE_CHUNK]
                    placeholders = ",".join("?" for _ in chunk)
                    delete_payload = json.dumps(
                        {
                            "condition": f"url IN ({placeholders})",
                            "parameters": list(chunk),
                        }
                    ).encode()
                    delete_req = urllib.request.Request(
                        f"{api_base}/indices/{index_name}/documents",
                        data=delete_payload,
                        headers=auth_headers,
                        method="DELETE",
                    )
                    try:
                        with urllib.request.urlopen(delete_req, timeout=120):
                            pass
                    except Exception as e:
                        # Non-fatal: if the delete fails the upsert
                        # still adds the new chunks; we'll just leak
                        # duplicates on the failed URLs.
                        print(f"  Warning: pre-upsert delete chunk {j // DELETE_CHUNK + 1} failed: {e}")
                # Give the server's batched delete worker time to
                # drain. ~3× the server's max-wait covers the worst
                # case (queue full + max_wait fired) plus a buffer.
                time.sleep(7)

            # URLs we successfully POSTed (per-batch HTTP 2xx). The
            # actual server-side persistence is verified in one sweep
            # below, since `update_with_encoding` returns before the
            # docs are merged into the searchable index — per-batch
            # verification would always race the merge and report 0.
            posted_urls: list[str] = []

            for i in range(0, len(docs_to_index), BATCH):
                batch_docs = docs_to_index[i : i + BATCH]
                batch_meta = metadata_to_index[i : i + BATCH]
                batch_urls = urls_aligned[i : i + BATCH]
                batch_num = i // BATCH + 1

                payload = json.dumps(
                    {
                        "documents": batch_docs,
                        "metadata": batch_meta,
                        "pool_factor": 2,
                    }
                ).encode()
                req = urllib.request.Request(
                    f"{api_base}/indices/{index_name}/update_with_encoding",
                    data=payload,
                    headers=auth_headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=300):
                        posted_urls.extend(batch_urls)
                        pct = 96 + (batch_num * 2) // n_batches
                        step(
                            min(pct, 98),
                            label,
                            f"Batch {batch_num}/{n_batches} ({len(posted_urls):,} posted)",
                        )
                except Exception as e:
                    print(f"  Warning: batch {batch_num} failed: {e}")

            # ── Per-URL verification & PG flag flip ─────────────────
            #
            # `update_with_encoding` returns 2xx even when it silently
            # drops docs mid-batch (encoding errors, OOM-truncated
            # batches, etc.) — see the geoffrey-hinton drift incident:
            # 666 docs PG-flagged indexed but only 413 actually in the
            # index. Trusting the HTTP status alone is what produced
            # that 253-doc gap, and once `indexed=true` is set the
            # docs are skipped on every subsequent run, so drift goes
            # silent forever.
            #
            # Defense: after all batches POST, ask the index which of
            # the URLs we just sent are actually present, then flip
            # `indexed=TRUE` ONLY for those. Anything dropped stays
            # FALSE and gets retried on the next pass — eventually
            # consistent rather than optimistically-wrong.
            #
            # We sleep first so the server's batched merge has time
            # to drain (same rationale as the pre-upsert delete path
            # above). This is a one-shot wait per pipeline run, not
            # per batch.
            if posted_urls:
                time.sleep(7)
                verified = _verify_indexed_urls(api_base, index_name, posted_urls, auth_headers)
                indexed = len(verified)
                if verified:
                    mark_documents_indexed(database_url, user_id, verified)
                dropped = len(posted_urls) - len(verified)
                if dropped:
                    print(
                        f"  Warning: index silently dropped {dropped}/{len(posted_urls)}"
                        f" docs server-side — left at indexed=false for retry."
                    )
            step(98, "Indexed", f"{indexed:,} documents in search engine")
        else:
            step(98, "Index", "No indexable documents")
    else:
        step(98, "Index", "No new documents to index")
    timings.append(("Index documents", time.perf_counter() - t0))

    # =========================================================================
    # Summary
    # =========================================================================

    total = time.perf_counter() - pipeline_start
    step(100, "Sources updated", f"{len(data):,} documents, {total_new} new ({_fmt(total)})")

    # =========================================================================
    # Save Run Metadata
    # =========================================================================

    # Seal the live tracker row inserted at the top of this function.
    # When run_id is 0 the initial INSERT failed — skip silently; the
    # pipeline still reports success on stdout and via the return.
    finish_pipeline_run(
        database_url,
        run_id,
        success=True,
        new_documents=total_new,
        total_documents=len(data),
        duration_secs=round(total, 2),
        timings=[{"step": label, "duration_secs": round(elapsed, 2)} for label, elapsed in timings],
    )

    print(f"\n  {name}: {len(data):,} documents, {total_new} new ({_fmt(total)})")
    for label, elapsed in timings:
        pct = elapsed / total * 100 if total > 0 else 0
        print(f"    {label:<25s} {_fmt(elapsed):>8s}  ({pct:4.1f}%)")
