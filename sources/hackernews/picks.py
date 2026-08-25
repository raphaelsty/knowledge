"""Per-user HackerNews front-page picks: scoring + selection.

Shared by the daemon (`sources.utils.hn_frontpage_daemon`, the thing
that actually runs in prod) and the debug CLI (`scripts/hn_frontpage.py`).
Both call :func:`refresh_picks`; neither owns any of the logic.

Why the raw ColBERT mean is not a usable ranking signal
------------------------------------------------------
The obvious scorer — send the article title as the query, average the
top-K scores against the user's library — does not measure relevance.
ColBERT's MaxSim *sums* over query tokens, so the score grows with the
title's token count no matter what the library contains. Measured on
prod against a 30-story front page:

    corr(mean score, title token count) = +0.79   (every user)
    top-10 pick overlap between unrelated users  = 9-10 / 10

Every user got the same picks, ordered by title length. "Personalised"
in name only.

The fix is to score *relatively*. For each article we compute the mean
and spread across the whole scored cohort, then rank each user by their
z-score:

    z[u][a] = (S[u][a] - mean_a) / std_a

Anything intrinsic to the article — title length, generic phrasing,
how widely interesting it is — lives in `mean_a` and cancels out. What
survives is the only thing we care about: does *this* library agree
with this article more than the average library does. Length bias
disappears without special-casing it, and so does the "everyone gets
the popular story" collapse.

The trade-off is that z-scores are relative, so some user is always
top-ranked for every article, however dull. `threshold` (a z floor) is
what keeps genuinely irrelevant stories out: a user with nothing above
the floor simply gets fewer picks, which is the honest outcome.

Ordering note: picks are selected by z, then re-sorted by HN upvotes
before they are written, because `rank` is what the feed reads. So the
z-score decides *which* stories a user sees and the upvote count
decides what they see *first*.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from sources.hackernews.frontpage import Frontpage
from sources.sql import (
    create_hn_frontpage_tables,
    get_run_items,
    insert_run,
    latest_run_id,
    list_personalities,
    replace_user_picks,
)

__all__ = ["PickStats", "refresh_picks"]

# Window over which we average the returned ColBERT scores. Large
# enough to smooth single-token noise, small enough to stay relevant.
SEARCH_TOP_K = 10

DEFAULT_TOP = 30
DEFAULT_TOP_PER_USER = 10
# z floor. 0.5 keeps roughly the upper third of each article's
# cross-user distribution, which lands near DEFAULT_TOP_PER_USER picks
# on a 30-story front page without padding the feed with filler.
DEFAULT_THRESHOLD = 0.5
# Centering needs a cohort to center against. Below this we have no
# usable baseline and fall back to length-normalised raw scores.
MIN_COHORT = 8
# In single-user debug mode we still score this many extra users, so
# `--slug` exercises the same z-score path as a real run instead of a
# different one.
REFERENCE_COHORT = 12

Logger = Callable[[str], None]


class PickStats:
    """What one refresh did — for logging and for the CLI's summary."""

    def __init__(self) -> None:
        self.run_id: int | None = None
        self.items = 0
        self.eligible = 0
        self.scored = 0
        self.users_with_picks = 0
        self.total_picks = 0
        self.centered = False
        self.skipped_no_index = 0
        self.write_failures = 0

    def __str__(self) -> str:
        mode = "z-scored" if self.centered else "length-normalised (cohort too small)"
        return (
            f"run={self.run_id} items={self.items} eligible={self.eligible} "
            f"scored={self.scored} users_with_picks={self.users_with_picks} "
            f"picks={self.total_picks} ranking={mode} "
            f"skipped_no_index={self.skipped_no_index} write_failures={self.write_failures}"
        )


# ── HTTP ────────────────────────────────────────────────────────────


def _get_json(url: str, timeout: int = 60) -> object | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Knowledge/hn-frontpage"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _post_json(url: str, payload: dict, timeout: int = 120) -> tuple[int, dict | None, str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Knowledge/hn-frontpage"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        try:
            txt = e.read().decode("utf-8", "replace")
        except Exception:
            txt = ""
        return e.code, None, txt[:300]
    except Exception as exc:
        return 0, None, str(exc)


def live_index_names(api_url: str) -> set[str] | None:
    """Names of indices the search service actually holds, or None.

    Worth one request: on prod only 57 of 649 user rows have an index
    (the rest have never had a successful pipeline run), so without
    this filter a run spends 592 round trips collecting 404s that the
    scorer then silently discards.

    None means "couldn't tell" — callers should fall back to trying
    every user rather than skipping everybody.
    """
    payload = _get_json(f"{api_url}/indices")
    if payload is None:
        return None
    raw = payload if isinstance(payload, list) else (payload.get("indices") or payload.get("names") or [])
    names = {(x if isinstance(x, str) else x.get("name")) for x in raw}
    names.discard(None)
    return {str(n) for n in names}


def _score_user(
    api_url: str,
    index_name: str,
    queries: list[str],
    log: Logger = print,
) -> list[float] | None:
    """Mean top-K ColBERT score per query, or None if the index is unusable.

    Returns one float per query, aligned with `queries` — 0.0 where the
    search returned nothing for that title, so the caller can rely on
    the alignment.
    """
    status, body, err = _post_json(
        f"{api_url}/indices/{index_name}/search_with_encoding",
        {"queries": queries, "params": {"top_k": SEARCH_TOP_K}},
    )
    if status != 200 or not body:
        # 404 just means the index isn't built yet — expected for any
        # user whose pipeline has never completed, so not worth a line.
        # Anything else is a real problem and should be visible.
        if status != 404:
            log(f"hn.picks.score.failed index={index_name} status={status} err={err}")
        return None
    results = body.get("results") or []
    out: list[float] = []
    for i in range(len(queries)):
        scores = (results[i].get("scores") or []) if i < len(results) else []
        out.append(sum(scores) / len(scores) if scores else 0.0)
    return out


# ── Ranking ─────────────────────────────────────────────────────────


def _center_by_article(raw: dict[int, list[float]], n_articles: int) -> tuple[dict[int, list[float]], bool]:
    """Turn raw per-user scores into per-article z-scores.

    `raw` maps user_id -> one score per article. Returns the same shape
    holding z-scores, plus whether centering actually happened. With
    fewer than MIN_COHORT users there is no meaningful baseline: this
    returns the input unchanged and leaves the fallback to the caller.
    """
    if len(raw) < MIN_COHORT:
        return raw, False

    user_ids = list(raw)
    out: dict[int, list[float]] = {uid: [0.0] * n_articles for uid in user_ids}
    for a in range(n_articles):
        column = [raw[uid][a] for uid in user_ids]
        mean = sum(column) / len(column)
        var = sum((v - mean) ** 2 for v in column) / len(column)
        std = var**0.5
        if std <= 1e-9:
            # Every library agrees exactly — no signal to extract, and
            # dividing would manufacture one out of float noise.
            continue
        for uid in user_ids:
            out[uid][a] = (raw[uid][a] - mean) / std
    return out, True


def _length_normalise(raw: dict[int, list[float]], queries: list[str]) -> dict[int, list[float]]:
    """Fallback ranking when the cohort is too small to center.

    Dividing by the title's token count removes the bulk of the MaxSim
    length bias. Weaker than centering — it only cancels length, not
    the article's general-interest component — but far better than the
    raw mean.
    """
    ntok = [max(1, len(q.split())) for q in queries]
    return {uid: [s / ntok[a] for a, s in enumerate(scores)] for uid, scores in raw.items()}


# ── Entry point ─────────────────────────────────────────────────────


def refresh_picks(
    database_url: str,
    api_url: str,
    *,
    top: int = DEFAULT_TOP,
    top_per_user: int = DEFAULT_TOP_PER_USER,
    threshold: float = DEFAULT_THRESHOLD,
    slug: str | None = None,
    limit: int = 0,
    dry: bool = False,
    no_snapshot: bool = False,
    debug: bool = False,
    log: Logger = print,
) -> PickStats:
    """Snapshot the HN front page and rewrite every user's picks.

    Order of operations matters: we fetch and score *before* inserting
    the run row. The feed reads picks from `MAX(hn_frontpage_runs.id)`,
    so a run inserted while scoring is broken would become the latest
    run with no picks attached and the HN cards would vanish entirely.
    Scoring first means a failure leaves the previous run — stale, but
    intact — in place.
    """
    api_url = api_url.rstrip("/")
    stats = PickStats()

    create_hn_frontpage_tables(database_url)

    # ── Front-page items ────────────────────────────────────────────
    reuse_run_id: int | None = None
    if no_snapshot:
        reuse_run_id = latest_run_id(database_url)
        if reuse_run_id is None:
            raise RuntimeError("--no-snapshot requested but no existing run found")
        items = get_run_items(database_url, reuse_run_id)
        log(f"hn.picks.reuse_run run={reuse_run_id} items={len(items)}")
    else:
        items = Frontpage(top=top)()
        if not items:
            raise RuntimeError("HN front page returned no items")
        log(f"hn.picks.fetched items={len(items)}")
    stats.items = len(items)

    queries = [it["title"].strip() for it in items]
    points_by_id = {int(it["hn_id"]): int(it.get("points") or 0) for it in items}
    title_by_id = {int(it["hn_id"]): it["title"] for it in items}
    hn_ids = [int(it["hn_id"]) for it in items]

    # ── Who to score ────────────────────────────────────────────────
    everyone = list_personalities(database_url)
    live = live_index_names(api_url)
    if live is None:
        log("hn.picks.indices.unavailable falling back to scoring every user")
        eligible = everyone
    else:
        eligible = [p for p in everyone if p["indexName"] in live]
        stats.skipped_no_index = len(everyone) - len(eligible)
        log(f"hn.picks.eligible users={len(eligible)} skipped_no_index={stats.skipped_no_index}")

    if slug:
        target = [p for p in eligible if p["slug"] == slug]
        if not target:
            raise RuntimeError(f"slug '{slug}' has no live index (or is not a user)")
        # Keep the z-score path alive for one-user debugging by scoring
        # a reference cohort alongside the requested user.
        others = [p for p in eligible if p["slug"] != slug][:REFERENCE_COHORT]
        cohort = target + others
        keep_only = {target[0]["id"]}
    else:
        cohort = eligible[:limit] if limit else eligible
        keep_only = None
    stats.eligible = len(cohort)

    # ── Score ───────────────────────────────────────────────────────
    raw: dict[int, list[float]] = {}
    by_id: dict[int, dict] = {}
    for i, p in enumerate(cohort, start=1):
        scores = _score_user(api_url, p["indexName"], queries, log)
        if scores is None:
            continue
        raw[int(p["id"])] = scores
        by_id[int(p["id"])] = p
        if debug or i % 25 == 0:
            log(f"hn.picks.scored {i}/{len(cohort)} {p['slug']}")
    stats.scored = len(raw)
    if not raw:
        raise RuntimeError("no user could be scored (search API down or every index missing?)")

    ranked, centered = _center_by_article(raw, len(queries))
    if not centered:
        ranked = _length_normalise(raw, queries)
        log(f"hn.picks.cohort_too_small n={len(raw)} using length-normalised scores")
    stats.centered = centered

    # ── Select + write ──────────────────────────────────────────────
    run_id = reuse_run_id
    if run_id is None:
        if dry:
            log("hn.picks.dry no run inserted, no picks written")
        else:
            run_id = insert_run(database_url, items)
            log(f"hn.picks.run_inserted run={run_id}")
    stats.run_id = run_id

    for uid, scores in ranked.items():
        if keep_only is not None and uid not in keep_only:
            continue
        p = by_id[uid]
        order = sorted(range(len(scores)), key=lambda a: -scores[a])
        chosen = [a for a in order if scores[a] >= threshold][:top_per_user]
        if not chosen:
            if debug:
                log(f"hn.picks.none {p['slug']} best_z={scores[order[0]]:.2f} < {threshold}")
            continue
        # Re-sort by upvotes: z picks *what*, HN traffic picks *first*.
        chosen.sort(key=lambda a: -points_by_id.get(hn_ids[a], 0))
        picks = [(hn_ids[a], float(scores[a])) for a in chosen]

        if debug:
            log(f"hn.picks.for {p['slug']}")
            for a in chosen:
                log(
                    f"    {points_by_id.get(hn_ids[a], 0):5d} pts  z={scores[a]:+5.2f}  "
                    f"{title_by_id.get(hn_ids[a], '')[:64]}"
                )

        if not dry and run_id is not None:
            try:
                replace_user_picks(database_url, uid, run_id, picks)
            except Exception as exc:  # noqa: BLE001 — one user can't kill the sweep
                stats.write_failures += 1
                log(f"hn.picks.write.failed user={p['slug']} err={exc}")
                continue
        stats.users_with_picks += 1
        stats.total_picks += len(picks)

    # A run row with no picks attached is worse than no run at all: the
    # feed reads MAX(run_id), so it would show zero HN cards to
    # everybody until the next refresh. Raising here makes the daemon
    # treat it as a failure and retry on its short back-off (minutes)
    # instead of sitting on an empty run for a day.
    if not dry and stats.users_with_picks == 0:
        raise RuntimeError(
            f"run {run_id} published no picks "
            f"(scored={stats.scored}, write_failures={stats.write_failures}, threshold={threshold})"
        )

    log(f"hn.picks.complete {stats}")
    return stats
