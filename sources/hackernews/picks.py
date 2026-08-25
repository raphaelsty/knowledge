"""Per-user HackerNews front-page picks: scoring + selection.

Shared by the daemon (`sources.utils.hn_frontpage_daemon`, the thing
that actually runs in prod) and the debug CLI (`scripts/hn_frontpage.py`).
Both call :func:`refresh_picks`; neither owns any of the logic.

Scoring against the index that still exists
-------------------------------------------
The original job scored each user against a per-personality ColBERT
index named after their username. That architecture is gone — see
`sources/indexer_daemon.py`: "There are no per-personality indices any
more; search serves everything from `__all__`". The 59 per-user index
directories still on the prod disk are leftovers in the early alphabet,
which is why the job could only ever have scored users whose names
start with 'a' — and never `raphael-sourty`, the account that reads the
feed.

So we score the way the frontend does: one `__all__` query per user,
scoped with `owner = ?`. `__all__` covers every VIP, and a user with
nothing in it simply returns no hits and gets no picks.

Why the raw ColBERT mean is not a usable ranking signal
------------------------------------------------------
The obvious scorer — send the article title as the query, average the
top-K scores against the user's library — does not measure relevance.
ColBERT's MaxSim *sums* over query tokens, so the score grows with the
title's token count no matter what the library contains. Measured on
prod against a 30-story front page and a 20-user cohort:

    corr(mean score, title token count) = +0.79   (every user)
    top-10 pick overlap between unrelated users  = 9.4 / 10
    distinct articles picked across the cohort   = 11 / 30

Every user got the same picks, ordered by title length. "Personalised"
in name only.

The fix is to score *relatively*, by double-centering the matrix of
(user × article) scores:

    1. per article, across users:  removes everything intrinsic to the
       article — title length, generic phrasing, broad appeal.
    2. per user, across articles:  removes the user's own offset, so a
       3968-document library and a 40-document one are held to the same
       bar instead of the big library clearing every threshold.

What survives both passes is the only thing worth ranking on: this
article is unusually good *for this user*. Same cohort, after:

    corr(score, title token count)               +0.00
    top-10 pick overlap between unrelated users  3.1 / 10
    distinct articles picked across the cohort   30 / 30

The trade-off is that centered scores are relative, so some user is
always top-ranked for every article, however dull. `threshold` (a z
floor) is what keeps genuinely irrelevant stories out: a user with
nothing above the floor gets fewer picks, not filler.

Simply dividing by the title's token count was tried first and
rejected: it inverts the bias rather than removing it
(corr = -0.69, two-word titles first) and still hands every user the
same ten stories.

Audience vs cohort
------------------
The feed only renders picks for a logged-in viewer, so picks are
written for accounts that can actually log in. Centering needs more
libraries than that to form a baseline, so we *score* a larger
reference cohort of VIPs and *write* only for the audience. Around 50
searches a day, against 637 if we wrote picks for every personality
that nobody can log in as.

Ordering note: picks are selected by score, then re-sorted by HN
upvotes before they are written, because `rank` is what the feed reads.
So the score decides *which* stories a user sees and the upvote count
decides what they see *first*.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import psycopg

from sources.hackernews.frontpage import Frontpage
from sources.sql import (
    create_hn_frontpage_tables,
    get_run_items,
    insert_run,
    latest_run_id,
    replace_user_picks,
)

__all__ = ["PickStats", "refresh_picks"]

# The one index that still exists. Everything else on disk is a
# leftover from the retired per-personality architecture.
ALL_INDEX = "__all__"

# Window over which we average the returned ColBERT scores. Large
# enough to smooth single-token noise, small enough to stay relevant.
SEARCH_TOP_K = 10

DEFAULT_TOP = 30
DEFAULT_TOP_PER_USER = 10
# z floor, in per-user standard deviations. 0.5 keeps roughly the upper
# third of a user's distribution, which lands near DEFAULT_TOP_PER_USER
# on a 30-story front page without padding the feed with filler.
DEFAULT_THRESHOLD = 0.5
# How many extra VIP libraries to score purely to establish the
# per-article baseline. 48 is comfortably past the point where the
# per-article mean stops moving, and still only ~48 searches.
DEFAULT_REFERENCE_COHORT = 48
# Fewer scored libraries than this and the per-article baseline is
# noise. We abort the run rather than publish picks we don't trust —
# see the note in `refresh_picks`.
MIN_COHORT = 8

# The search router is rate-limited per client IP (RATE_LIMIT_ENABLED
# in prod: burst 100, then one slot back every RATE_LIMIT_PER_SECOND
# seconds — tower_governor's `per_second` is a replenish interval, not
# a rate). A full bucket covers a whole run, but a drained one turns
# every search into a 429, and without these retries the run would
# quietly score a handful of libraries and centre against noise.
# The API's `retry_after_seconds` is hardcoded to 2 and doesn't
# reflect the real wait, so we back off on our own schedule.
RATE_LIMIT_ATTEMPTS = 3
RATE_LIMIT_BACKOFF_SECS = (5, 20, 60)
# Once this many libraries in a row have exhausted their retries, the
# bucket is empty rather than briefly contended — every remaining
# search would fail the same way. Abort and let the daemon retry on
# its back-off, instead of grinding through the cohort for an hour.
RATE_LIMIT_GIVE_UP = 5
# Small gap between libraries so a run sips from the burst bucket
# instead of draining it in one go.
PACE_SECS = 0.2

Logger = Callable[[str], None]


class PickStats:
    """What one refresh did — for logging and for the CLI's summary."""

    def __init__(self) -> None:
        self.run_id: int | None = None
        self.items = 0
        self.audience = 0
        self.cohort = 0
        self.scored = 0
        self.empty_library = 0
        self.users_with_picks = 0
        self.total_picks = 0
        self.centered = False
        self.write_failures = 0

    def __str__(self) -> str:
        mode = "double-centered" if self.centered else "uncentered"
        return (
            f"run={self.run_id} items={self.items} audience={self.audience} "
            f"cohort={self.cohort} scored={self.scored} empty_library={self.empty_library} "
            f"users_with_picks={self.users_with_picks} picks={self.total_picks} "
            f"ranking={mode} write_failures={self.write_failures}"
        )


# ── Who to score ────────────────────────────────────────────────────


def audience_users(database_url: str) -> list[tuple[int, str]]:
    """Accounts that can log in, as `(id, username)`.

    Two ways in, so both are covered: password signup sets
    `password_hash` (see the INSERT in handlers/auth.rs), GitHub OAuth
    sets `email_verified = TRUE` and no password. Personalities created
    by the pipeline have neither.

    These are the only rows worth writing picks for — handlers/follows.rs
    gates the whole HN block on a logged-in viewer.
    """
    sql = """
        SELECT id, username
          FROM users
         WHERE password_hash IS NOT NULL
            OR email_verified = TRUE
         ORDER BY id
    """
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def reference_cohort(database_url: str, exclude: set[int], limit: int) -> list[tuple[int, str]]:
    """VIPs to score purely as a baseline for centering.

    Ordered by library size: the biggest libraries give the steadiest
    per-article mean, and the ordering is deterministic so consecutive
    runs center against the same reference.
    """
    sql = """
        SELECT id, username
          FROM users
         WHERE vip = TRUE
           AND COALESCE(document_count, 0) > 0
         ORDER BY document_count DESC, id
         LIMIT %s
    """
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, (limit + len(exclude),))
        rows = [(int(r[0]), str(r[1])) for r in cur.fetchall()]
    return [r for r in rows if r[0] not in exclude][:limit]


# ── Scoring ─────────────────────────────────────────────────────────


def _post_json(url: str, payload: dict, timeout: int = 180) -> tuple[int, dict | None, str | None]:
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


class RateLimited(RuntimeError):
    """The search router's bucket is empty, not merely contended."""


def score_owner(
    api_url: str,
    owner: str,
    queries: list[str],
    log: Logger = print,
) -> list[float] | None:
    """Mean top-K ColBERT score per query for one owner's library.

    Returns one float per query, aligned with `queries` (0.0 where the
    search returned nothing), or None if the request itself failed.
    An owner with nothing in `__all__` gets all zeros, not None —
    that's a real answer ("no signal"), not an error.

    Raises RateLimited if every attempt was throttled, so the caller
    can tell "this library has nothing" from "the API won't talk to
    us right now".
    """
    payload = {
        "queries": queries,
        "params": {"top_k": SEARCH_TOP_K},
        "filter_condition": "owner = ?",
        "filter_parameters": [owner],
    }
    url = f"{api_url}/indices/{ALL_INDEX}/search/filtered_with_encoding"

    status, body, err = 0, None, None
    for attempt in range(RATE_LIMIT_ATTEMPTS):
        status, body, err = _post_json(url, payload)
        if status != 429:
            break
        if attempt == RATE_LIMIT_ATTEMPTS - 1:
            break
        wait = RATE_LIMIT_BACKOFF_SECS[attempt]
        log(f"hn.picks.score.rate_limited owner={owner} attempt={attempt + 1} waiting={wait}s")
        time.sleep(wait)

    if status == 429:
        raise RateLimited(f"owner={owner} throttled after {RATE_LIMIT_ATTEMPTS} attempts")

    if status != 200 or not body:
        log(f"hn.picks.score.failed owner={owner} status={status} err={err}")
        return None
    results = body.get("results") or []
    out: list[float] = []
    for i in range(len(queries)):
        scores = (results[i].get("scores") or []) if i < len(results) else []
        out.append(sum(scores) / len(scores) if scores else 0.0)
    return out


# ── Ranking ─────────────────────────────────────────────────────────


def _double_center(raw: dict[int, list[float]], n_articles: int) -> dict[int, list[float]]:
    """Article-center then user-center the (user × article) score matrix.

    Callers must pass at least MIN_COHORT libraries; centering against
    fewer is centering against noise.
    """
    user_ids = list(raw)
    out: dict[int, list[float]] = {uid: [0.0] * n_articles for uid in user_ids}

    # Pass 1 — per article, across users. Kills title length and any
    # other article-intrinsic component.
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

    # Pass 2 — per user, across articles. Kills the user's own offset
    # so `threshold` means the same thing for a 4000-document library
    # and a 40-document one.
    for uid in user_ids:
        row = out[uid]
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        std = var**0.5
        if std <= 1e-9:
            out[uid] = [0.0] * n_articles
            continue
        out[uid] = [(v - mean) / std for v in row]

    return out


# ── Entry point ─────────────────────────────────────────────────────


def refresh_picks(
    database_url: str,
    api_url: str,
    *,
    top: int = DEFAULT_TOP,
    top_per_user: int = DEFAULT_TOP_PER_USER,
    threshold: float = DEFAULT_THRESHOLD,
    reference: int = DEFAULT_REFERENCE_COHORT,
    slug: str | None = None,
    limit: int = 0,
    dry: bool = False,
    no_snapshot: bool = False,
    debug: bool = False,
    log: Logger = print,
) -> PickStats:
    """Snapshot the HN front page and rewrite every audience user's picks.

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

    # ── Audience + reference cohort ─────────────────────────────────
    audience = audience_users(database_url)
    if slug:
        audience = [u for u in audience if u[1] == slug]
        if not audience:
            raise RuntimeError(f"slug '{slug}' is not an account that can log in")
    if limit:
        audience = audience[:limit]
    stats.audience = len(audience)

    audience_ids = {uid for uid, _ in audience}
    cohort = audience + reference_cohort(database_url, audience_ids, reference)
    stats.cohort = len(cohort)
    log(f"hn.picks.cohort audience={stats.audience} reference={stats.cohort - stats.audience}")

    # ── Score ───────────────────────────────────────────────────────
    raw: dict[int, list[float]] = {}
    name_by_id: dict[int, str] = {}
    throttled_in_a_row = 0
    for i, (uid, username) in enumerate(cohort, start=1):
        try:
            scores = score_owner(api_url, username, queries, log)
        except RateLimited as exc:
            throttled_in_a_row += 1
            if throttled_in_a_row >= RATE_LIMIT_GIVE_UP:
                raise RateLimited(
                    f"{throttled_in_a_row} libraries throttled in a row at {i}/{len(cohort)} "
                    f"— search bucket is empty, retrying later ({exc})"
                ) from exc
            continue
        throttled_in_a_row = 0
        if scores is None:
            continue
        if not any(scores):
            # Nothing of theirs is in `__all__` — no relevance signal
            # to rank with. Common for non-VIP signups, whose libraries
            # aren't indexed anywhere since the per-user indices went
            # away.
            stats.empty_library += 1
            if debug:
                log(f"hn.picks.empty_library {username}")
            continue
        raw[uid] = scores
        name_by_id[uid] = username
        if debug or i % 25 == 0:
            log(f"hn.picks.scored {i}/{len(cohort)} {username}")
        time.sleep(PACE_SECS)
    stats.scored = len(raw)

    # Too few libraries and the per-article mean is noise, which would
    # produce confident-looking nonsense. Better to abort: the daemon
    # retries on its back-off and the previous run keeps serving.
    # (Length-normalising instead was measured and rejected — it just
    # inverts the bias, ranking the shortest titles first.)
    if len(raw) < MIN_COHORT:
        raise RuntimeError(
            f"only {len(raw)} of {len(cohort)} libraries scored, need {MIN_COHORT} "
            f"to centre (rate-limited, or __all__ is empty?)"
        )

    ranked = _double_center(raw, len(queries))
    stats.centered = True

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
        # Reference-cohort libraries exist only to center the matrix.
        if uid not in audience_ids:
            continue
        username = name_by_id[uid]
        order = sorted(range(len(scores)), key=lambda a: -scores[a])
        chosen = [a for a in order if scores[a] >= threshold][:top_per_user]
        if not chosen:
            log(f"hn.picks.none {username} best={scores[order[0]]:+.2f} < {threshold}")
            continue
        # Re-sort by upvotes: score picks *what*, HN traffic picks *first*.
        chosen.sort(key=lambda a: -points_by_id.get(hn_ids[a], 0))
        picks = [(hn_ids[a], float(scores[a])) for a in chosen]

        if debug:
            log(f"hn.picks.for {username}")
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
                log(f"hn.picks.write.failed user={username} err={exc}")
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
