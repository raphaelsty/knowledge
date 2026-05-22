"""knowledge-twitter-feed — long-running client that feeds a Knowledge
production PG with tweet data, using the operator's local Safari
session as the twikit auth.

Why this exists
---------------
The pipeline's `make run TWIKIT=1 SOURCE=twitter` is great for a
one-shot backfill, but the operator wants something to leave running
on their Mac for days at a time:

  * always reads fresh cookies from Safari (the only browser
    `cookies.py` can extract from), so a Twitter session rotation
    or re-login is picked up automatically;
  * pulls each personality's tweets via twikit (no twitterapi.io
    credit);
  * writes to PG **as each twikit page lands** (per-page flush) so
    a laptop sleep, crash, or kill -9 mid-stream costs at most the
    handful of tweets in the page currently in flight;
  * loops indefinitely, prioritising **stalest first**: each pass
    queues every VIP with a twitter handle, ordered by how long ago
    their library was last touched (never-touched users at the very
    top). A successful fetch sets `updated_at = now()` on the new
    rows, so the next pass naturally demotes them to the back. Use
    `--min-age N` to skip users touched in the last N hours.

Usage
-----
::

    # Default: hits the prod PG via DATABASE_URL from .env or env.
    knowledge-twitter-feed

    # One pass only, then exit (handy in cron).
    knowledge-twitter-feed --one-shot

    # Explicit PG, skip the per-page index push, custom rest period.
    knowledge-twitter-feed --database-url postgresql://prod-host/knowledge \\
                           --rest 3600 \\
                           --personality-delay 4

    # Restart cookies mid-loop by re-opening Safari → the next
    # iteration silently picks up the rotated `auth_token`.

If you don't have Safari logged in, the client prints a clear
"open x.com in Safari" message and refuses to start. No silent
empty fetches.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone

from twikit.errors import (
    AccountSuspended,
    BadRequest,
    Forbidden,
    NotFound,
    TweetNotAvailable,
    UserNotFound,
    UserUnavailable,
)

from sources.twitter.bookmarks import Bookmarks, _refresh_safari_cookies
from sources.twitter.cookies import get_safari_cookies
from sources.utils.cleaning import clean_summary, clean_title

# Heartbeat target — the prod admin panel endpoint that records the
# feeder's last-known state into the single-row
# `twitter_feed_status` table. Authentication is a shared secret
# (KNOWLEDGE_ADMIN_TOKEN env, same one the server checks). When
# either the URL or the token is empty the sender is a silent
# no-op — so running the feeder against a local DB or in a dev
# context doesn't try to POST to an internet endpoint.
HEARTBEAT_URL = os.environ.get(
    "KNOWLEDGE_HEARTBEAT_URL",
    "https://knowledge-web.org/api/admin/twitter-feed/heartbeat",
)
HEARTBEAT_TOKEN = os.environ.get("KNOWLEDGE_ADMIN_TOKEN", "")


def _utc_iso() -> str:
    """RFC3339 / ISO-8601 UTC timestamp — the format the server's
    `to_char` round-trips through."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _heartbeat(**fields) -> None:
    """Fire-and-forget heartbeat POST. Failures are swallowed — the
    feeder's primary job is fetching tweets, not bookkeeping. A
    network blip mustn't crash the loop or block the next slug."""
    if not HEARTBEAT_URL or not HEARTBEAT_TOKEN:
        return
    body = json.dumps(fields).encode("utf-8")
    req = urllib.request.Request(
        HEARTBEAT_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Admin-Token": HEARTBEAT_TOKEN,
            "User-Agent": "knowledge-twitter-feed/1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            _ = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        # Heartbeats are best-effort; never crash the feeder if the
        # admin endpoint is unreachable.
        return


# Twikit exceptions that mean "this specific user's data can't be
# fetched" — wrong handle, suspended account, blocked us, deleted
# profile. These are *user-side* faults, not Twitter telling us off.
# They must NOT contribute to the consecutive-zero-yield stall
# counter, otherwise a small cluster of bad handles at the top of
# the staleness queue (where never-touched users always land first)
# kills the entire sweep before any healthy VIP gets fetched.
_USER_FAULT_EXC: tuple[type[BaseException], ...] = (
    UserNotFound,  # @handle resolved to nothing
    UserUnavailable,  # protected, locked, withheld
    AccountSuspended,  # account suspended by Twitter
    NotFound,  # generic 404 on the user resource
    Forbidden,  # user has blocked us or made the timeline private
    TweetNotAvailable,  # tweet-level 404 / withheld that propagated up
    BadRequest,  # malformed handle string
)


# Note on rate-limit handling: we don't track "consecutive zero-yield
# slugs" anymore. The earlier `_STALL_THRESHOLD` mechanism abandoned
# the pass after five consecutive empties to detect a rate-limit
# wall, but the upstream `_rate_limit_aware` wrapper already
# absorbs 429s with sleep-to-reset (and on a second 429 returns
# silently with no docs). Most "empty" responses turn out to be
# per-user faults — handles that don't exist, suspended accounts,
# parser bugs on weird tweet bodies — none of which justify pausing
# the sweep. We always walk the queue to completion; if Twitter is
# truly upset, the inline sleeps inside `_rate_limit_aware` pace
# things, and any remaining no-op calls finish fast and harmlessly.


# Colour support. ANSI escapes are emitted unless `NO_COLOR=1` is set
# (https://no-color.org). The log file tee'd by scripts/twitter_feed.sh
# also ends up with escape codes — view with `less -R`, or run with
# `NO_COLOR=1 make twitter-feed` for a clean log.
_USE_COLOR = os.environ.get("NO_COLOR", "").strip() == ""
_C_RESET = "\x1b[0m" if _USE_COLOR else ""
_C_DIM = "\x1b[2m" if _USE_COLOR else ""
_C_BOLD = "\x1b[1m" if _USE_COLOR else ""
_C_INFO = "\x1b[36m" if _USE_COLOR else ""  # cyan
_C_OK = "\x1b[32m" if _USE_COLOR else ""  # green
_C_WARN = "\x1b[33m" if _USE_COLOR else ""  # yellow
_C_ERR = "\x1b[31m" if _USE_COLOR else ""  # red
_C_HEAD = "\x1b[35m" if _USE_COLOR else ""  # magenta (per-slug header)
_C_BANNER = "\x1b[1;34m" if _USE_COLOR else ""  # bold blue (pass banner)


def _log(*parts) -> None:
    """Timestamped stdout line. `print` with explicit flush so the
    client looks alive when its stdout is being tailed via `tail -f`.
    The timestamp itself is dimmed so the eye lands on the content."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = " ".join(str(p) for p in parts)
    print(f"{_C_DIM}{ts}{_C_RESET}  {body}", flush=True)


# Log-level helpers — coloured prefix so the eye finds warnings and
# errors at a glance, and `grep` against the symbol still works in a
# plain-text log file. Each helper colours JUST the symbol; the body
# stays default to keep the bulk of the log readable.
def _log_info(*parts) -> None:
    _log(f"{_C_INFO}·{_C_RESET}", *parts)


def _log_ok(*parts) -> None:
    _log(f"{_C_OK}✓{_C_RESET}", *parts)


def _log_warn(*parts) -> None:
    _log(f"{_C_WARN}⚠{_C_RESET}  {_C_WARN}" + " ".join(str(p) for p in parts) + _C_RESET)


def _log_err(*parts) -> None:
    _log(f"{_C_ERR}✗{_C_RESET}  {_C_ERR}{_C_BOLD}" + " ".join(str(p) for p in parts) + _C_RESET)


def _log_banner(line: str) -> None:
    """Visual separator for pass start/end. Width matches the existing
    indented per-page lines so the whole log reads as one shape. The
    rule itself is dimmed and the banner text is bold blue so the
    boundaries pop without dominating the rest of the log."""
    bar = "═" * 72
    print(f"{_C_DIM}{bar}{_C_RESET}", flush=True)
    _log(f"{_C_BANNER}{line}{_C_RESET}")
    print(f"{_C_DIM}{bar}{_C_RESET}", flush=True)


def _slug_header(line: str) -> None:
    """The `── [N/M] slug @handle · pop · last_touch · ETA` line.
    Coloured magenta so each new personality reads as a section break
    without needing its own banner."""
    _log(f"{_C_HEAD}{line}{_C_RESET}")


def _human_count(n: int) -> str:
    """Compact follower / count rendering — 4_780_000 → '4.7M'."""
    if n is None:
        return "?"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _human_duration(secs: float) -> str:
    secs = max(0.0, float(secs))
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _vips_by_staleness(
    min_age_hours: float = 0.0,
) -> list[tuple[int, str, str, object, int]]:
    """Return ``[(user_id, slug, handle, last_touch, followers), …]``
    for every VIP that has a `sources.twitter.username` configured,
    in the canonical "touched-today demoted, popularity-desc" order.

    The work now happens server-side behind
    `GET /api/admin/twitter-queue` so the feeder no longer needs PG
    access. The Rust handler runs the same SQL the Python version
    used to run inline — see ``handlers::admin::admin_twitter_queue``.
    """
    api_base = os.environ.get("API_URL", "https://knowledge-web.org").rstrip("/")
    token = os.environ.get("KNOWLEDGE_ADMIN_TOKEN", "").strip()
    if not token:
        raise IngestAuthError(
            "KNOWLEDGE_ADMIN_TOKEN is not set in the environment; the "
            "admin queue endpoint refuses unauthenticated calls."
        )
    req = urllib.request.Request(
        f"{api_base}/api/admin/twitter-queue?min_age_hours={float(min_age_hours)}",
        method="GET",
        headers={"X-Admin-Token": token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        if e.code == 401:
            raise IngestAuthError(f"401 unauthorized: {body}") from e
        raise IngestUnavailable(f"queue endpoint {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise IngestNetworkError(f"queue endpoint unreachable: {e.reason}") from e
    except TimeoutError as e:
        raise IngestNetworkError(f"queue endpoint timeout: {e}") from e

    out: list[tuple[int, str, str, object, int]] = []
    if not isinstance(payload, list):
        return out
    for row in payload:
        if not isinstance(row, dict):
            continue
        handle = str(row.get("handle") or "")
        if not handle:
            continue
        last_touch = row.get("last_touch")
        if isinstance(last_touch, str) and last_touch:
            try:
                last_touch_obj: object = datetime.fromisoformat(last_touch.replace("Z", "+00:00"))
            except ValueError:
                last_touch_obj = last_touch
        else:
            last_touch_obj = None
        out.append(
            (
                int(row.get("user_id") or 0),
                str(row.get("slug") or ""),
                handle,
                last_touch_obj,
                int(row.get("twitter_followers") or 0),
            )
        )
    return out


def _format_age(last_touch) -> str:
    """Render `last_touch` as a short human age — `never`, `3d`, `12h`."""
    if last_touch is None:
        return "never"
    now = datetime.now(last_touch.tzinfo) if last_touch.tzinfo else datetime.now()
    delta = now - last_touch
    secs = max(0, int(delta.total_seconds()))
    if secs >= 86400:
        return f"{secs // 86400}d"
    if secs >= 3600:
        return f"{secs // 3600}h"
    if secs >= 60:
        return f"{secs // 60}m"
    return f"{secs}s"


def _ensure_cookies() -> tuple[str, str]:
    """Pull `(auth_token, ct0)` from Safari. Raises a clear error if
    x.com isn't signed in."""
    creds = get_safari_cookies()
    return creds["auth_token"], creds["ct0"]


# Typed exceptions so the caller can pick the right backoff for each
# kind of API failure.
class IngestAuthError(Exception):
    """Admin token missing / wrong (401). Fatal — pass should abort."""


class IngestRateLimited(Exception):
    """Server pushed back (429). Caller should sleep `retry_after` seconds."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class IngestUnavailable(Exception):
    """Server returned 5xx / DB unavailable (503). Retry after a short sleep."""


class IngestNetworkError(Exception):
    """Local network failure (timeout, DNS, connection reset)."""


class IngestError(Exception):
    """Catch-all for 4xx responses that aren't 401/429. Non-retryable."""


def _ingest_via_api(slug: str, cleaned: dict) -> tuple[int, int]:
    """POST a page of cleaned tweets to the Rust admin ingest endpoint.

    Returns ``(n_inserted, n_existed)`` so the caller (and the
    paginator) can early-stop the moment a page produces zero new
    rows — the API tells us "this URL was already in PG" via the
    `RETURNING (xmax = 0)` flag.

    Raises one of the typed `Ingest*` exceptions on failure so the
    caller can choose backoff strategy:
      * IngestAuthError    → fatal (bad / missing token)
      * IngestRateLimited  → sleep retry_after seconds then retry
      * IngestUnavailable  → sleep ~10s, retry
      * IngestNetworkError → sleep ~5s, retry
    Anything else propagates as the original exception.
    """
    if not cleaned:
        return 0, 0
    api_base = os.environ.get("API_URL", "https://knowledge-web.org").rstrip("/")
    token = os.environ.get("KNOWLEDGE_ADMIN_TOKEN", "").strip()
    if not token:
        raise IngestAuthError(
            "KNOWLEDGE_ADMIN_TOKEN is not set in the environment; the API "
            "ingest endpoint refuses unauthenticated calls."
        )
    payload = {
        "slug": slug,
        "documents": [
            {
                "url": url,
                "title": d.get("title") or "",
                "summary": d.get("summary") or "",
                "date": d.get("date") or "",
                "source": d.get("source") or "twitter",
                "source_url": d.get("source_url"),
                "tags": list(d.get("tags") or []),
                "extra_tags": list(d.get("extra_tags") or d.get("extra-tags") or []),
                "linked_urls": d.get("linked_urls") or [],
                "link_hosts": list(d.get("link_hosts") or []),
            }
            for url, d in cleaned.items()
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base}/api/admin/tweets/ingest",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Admin-Token": token,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Read whatever the server told us so we can echo it in the
        # log — the Rust handler returns plain-text error bodies for
        # the 4xx/5xx paths.
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        status = e.code
        if status == 401:
            raise IngestAuthError(f"401 unauthorized: {body_text}") from e
        if status == 429:
            retry = 60
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra:
                try:
                    retry = max(1, int(ra))
                except ValueError:
                    pass
            raise IngestRateLimited(
                f"429 rate limited (retry in {retry}s): {body_text}",
                retry_after=retry,
            ) from e
        if 500 <= status < 600:
            raise IngestUnavailable(f"{status} server error: {body_text}") from e
        # 4xx other than 401/429 — probably a malformed payload or
        # missing slug. Surface but don't pretend it's transient.
        raise IngestError(f"{status} {body_text}") from e
    except urllib.error.URLError as e:
        raise IngestNetworkError(f"network: {e.reason}") from e
    except TimeoutError as e:
        raise IngestNetworkError(f"timeout: {e}") from e
    return int(data.get("n_inserted") or 0), int(data.get("n_existed") or 0)


# Retry budget — how many times the flush will sleep + retry before
# giving up on a single page. With the 60s base sleep, 3 retries
# means up to ~3 minutes of patience per page before we move on to
# the next personality.
_INGEST_RETRY_BUDGET = 3


def _flush_factory(slug: str, counters: dict):
    """Build a per-page flush callback bound to one personality.

    Each flush POSTs the page to the Rust ingest endpoint and either:
      - records the (n_inserted, n_existed) counters on success;
      - sleeps the suggested backoff and retries on transient errors
        (rate limit, server-side 5xx, network blip);
      - re-raises auth/permanent errors so the caller can abort.

    `counters` is shared with the outer loop so the per-slug summary
    can report cumulative results AND so the paginator's early-stop
    in `bookmarks._paginate_stream` sees the dedup state.
    """

    def _do_post(cleaned: dict) -> tuple[int, int]:
        attempts = 0
        while True:
            attempts += 1
            try:
                return _ingest_via_api(slug, cleaned)
            except IngestRateLimited as e:
                counters["api_errors"] = counters.get("api_errors", 0) + 1
                wait = max(1, int(e.retry_after))
                _log_warn(
                    f"   ↳ API rate-limited (429); sleeping {wait}s then "
                    f"retrying (attempt {attempts}/{_INGEST_RETRY_BUDGET + 1})"
                )
                if attempts > _INGEST_RETRY_BUDGET:
                    _log_err(f"   ↳ giving up on this page after {attempts} attempts: {e}")
                    raise
                time.sleep(wait)
            except IngestUnavailable as e:
                counters["api_errors"] = counters.get("api_errors", 0) + 1
                wait = 10
                _log_warn(
                    f"   ↳ API server error (5xx): {e}; sleeping {wait}s then "
                    f"retrying (attempt {attempts}/{_INGEST_RETRY_BUDGET + 1})"
                )
                if attempts > _INGEST_RETRY_BUDGET:
                    _log_err(f"   ↳ giving up on this page after {attempts} attempts")
                    raise
                time.sleep(wait)
            except IngestNetworkError as e:
                counters["api_errors"] = counters.get("api_errors", 0) + 1
                wait = 5
                _log_warn(
                    f"   ↳ network error ({e}); sleeping {wait}s then "
                    f"retrying (attempt {attempts}/{_INGEST_RETRY_BUDGET + 1})"
                )
                if attempts > _INGEST_RETRY_BUDGET:
                    _log_err(f"   ↳ giving up on this page after {attempts} attempts")
                    raise
                time.sleep(wait)
            except IngestAuthError as e:
                counters["api_errors"] = counters.get("api_errors", 0) + 1
                _log_err(f"   ↳ auth error: {e}")
                raise
            except IngestError as e:
                counters["api_errors"] = counters.get("api_errors", 0) + 1
                _log_err(f"   ↳ API rejected the page: {e}")
                raise

    def _flush(page_docs: dict) -> None:
        if not page_docs:
            return
        cleaned: dict = {}
        for url, d in page_docs.items():
            cleaned[url] = {
                **d,
                "title": clean_title(d.get("title") or ""),
                "summary": clean_summary(d.get("summary") or ""),
                "source": d.get("source") or "twitter",
            }
        try:
            n_inserted, n_existed = _do_post(cleaned)
        except (IngestAuthError, IngestError):
            # Auth + non-retryable errors abort the personality. Re-raise
            # so the outer `_process_one` catches and records the failure.
            raise
        except Exception as e:
            _log_err(f"   ↳ ingest API call failed after retries ({e!r}); page lost")
            counters["pages_lost"] = counters.get("pages_lost", 0) + 1
            return
        counters["inserted_total"] = counters.get("inserted_total", 0) + n_inserted
        counters["existed_total"] = counters.get("existed_total", 0) + n_existed
        counters["pages_flushed"] = counters.get("pages_flushed", 0) + 1
        counters["last_inserted"] = n_inserted
        counters["last_existed"] = n_existed
        # Two-line summary: page count then a compact verdict.
        _log_info(f"   ↳ flushed {len(cleaned):>3d} doc(s) → API +{n_inserted} new, {n_existed} already there")

    return _flush


def _load_existing_twitter_urls(slug: str) -> set[str]:
    """Return every twitter URL we already have stored for ``slug``.

    Used to feed `Bookmarks.existing_urls` so the fetcher can skip
    tweets at source AND so the paginator can bail early the moment
    a whole page turns up nothing new. Fetched through the Rust
    admin endpoint `GET /api/admin/users/{slug}/twitter-urls` — no
    direct PG access needed.
    """
    api_base = os.environ.get("API_URL", "https://knowledge-web.org").rstrip("/")
    token = os.environ.get("KNOWLEDGE_ADMIN_TOKEN", "").strip()
    if not token:
        raise IngestAuthError(
            "KNOWLEDGE_ADMIN_TOKEN is not set in the environment; the admin endpoints refuse unauthenticated calls."
        )
    req = urllib.request.Request(
        f"{api_base}/api/admin/users/{urllib.parse.quote(slug)}/twitter-urls",
        method="GET",
        headers={"X-Admin-Token": token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        if e.code == 401:
            raise IngestAuthError(f"401 unauthorized: {body}") from e
        raise IngestUnavailable(f"urls endpoint {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise IngestNetworkError(f"urls endpoint unreachable: {e.reason}") from e
    except TimeoutError as e:
        raise IngestNetworkError(f"urls endpoint timeout: {e}") from e
    if isinstance(payload, list):
        return {str(u) for u in payload if isinstance(u, str)}
    return set()


def _process_one(slug: str, handle: str, auth_token: str, ct0: str) -> dict:
    """Fetch tweets for one personality. Returns the per-slug counters
    dict so the pass-level summary can aggregate them.

    No PG access from this process — both the existing-URL read and
    the per-page flush flow through the Rust admin API.
    """
    counters: dict = {
        "inserted_total": 0,
        "existed_total": 0,
        "pages_flushed": 0,
        "pages_lost": 0,
        "api_errors": 0,
    }
    fetcher = Bookmarks(
        auth_token=auth_token,
        ct0=ct0,
        target_username=handle,
        on_page_flush=_flush_factory(slug, counters),
    )
    # Pre-load this user's already-stored twitter URLs so the
    # fetcher can dedup at source AND so the paginator's early-stop
    # (in `bookmarks.py:_paginate_stream`) kicks in the moment a
    # whole page turns up nothing new.
    existing = _load_existing_twitter_urls(slug)
    counters["existing_count"] = len(existing)
    fetcher(existing_urls=existing)
    return counters


def _one_pass(args) -> tuple[int, int]:
    """Walk every missing VIP once. Returns `(processed, total)`.

    Reads queue + existing-URL state through the Rust admin API —
    no direct PG access. The legacy `database_url` arg is dropped.
    """
    pass_started_at = _utc_iso()
    pass_t0 = time.perf_counter()
    _heartbeat(state="starting", pass_started_at=pass_started_at)
    try:
        auth_token, ct0 = _ensure_cookies()
    except Exception as e:
        _log_err(
            f"Safari cookies unavailable: {e} — open Safari, sign in at "
            "https://x.com, and the next pass will pick the rotation up."
        )
        _heartbeat(state="error", last_error=f"safari cookies unavailable: {e}")
        return 0, 0
    _log_ok(f"twikit cookies OK — auth_token …{auth_token[-6:]}, ct0 …{ct0[-6:]}")

    try:
        targets = _vips_by_staleness(min_age_hours=args.min_age)
    except IngestAuthError as e:
        _log_err(f"queue endpoint refused: {e}")
        _heartbeat(state="error", last_error=str(e))
        return 0, 0
    except (IngestUnavailable, IngestNetworkError) as e:
        _log_err(f"queue endpoint unreachable: {e}; the next launchd cycle will retry")
        _heartbeat(state="error", last_error=str(e))
        return 0, 0
    total = len(targets)
    if not total:
        _log_info("queue empty — nothing to do this pass")
        _heartbeat(
            state="idle",
            pass_started_at=pass_started_at,
            pass_finished_at=_utc_iso(),
            pass_processed=0,
            pass_total=0,
            pass_completed=True,
        )
        return 0, 0
    queue_desc = f"queue: {total} VIP(s), popularity-desc{f', stale > {args.min_age}h' if args.min_age > 0 else ''}"
    _log_banner(queue_desc)

    # Pass-level aggregates so the rollup at the end has real numbers.
    pass_stats = {
        "inserted_total": 0,
        "existed_total": 0,
        "api_errors": 0,
        "early_stops": 0,  # slugs that bailed because all-known
        "with_new": 0,  # slugs that ingested >0 new tweets
        "skipped_user_faults": 0,  # suspended/private/missing
        "skipped_other": 0,  # twikit / parser blowups
        "slugs_done": 0,
    }
    processed = 0
    # API-down detector. The feeder's only remote dependency is the
    # Rust admin API (queue / existing-URLs / ingest). When it's
    # unreachable every slug will fail the same way, so abort after
    # a few back-to-back failures instead of logging hundreds of
    # identical errors. The launchd KeepAlive watcher restarts us
    # on the next cycle when the API comes back.
    API_FAIL_THRESHOLD = 3
    consecutive_api_failures = 0
    for i, (_uid, slug, handle, last_touch, followers) in enumerate(targets, start=1):
        # Cooperative shutdown check.
        if _stop_requested:
            _log_warn("stop requested, exiting pass early")
            break
        slug_t0 = time.perf_counter()
        # ETA: extrapolate per-slug time so far over the remaining
        # queue. Cheap, slightly optimistic (some slugs bail in 2s,
        # others take 30s+) but useful as an order-of-magnitude
        # estimate when the queue is long.
        elapsed_so_far = time.perf_counter() - pass_t0
        eta_str = ""
        if processed > 0:
            avg = elapsed_so_far / processed
            remaining = avg * (total - i + 1)
            eta_str = f" · ETA {_human_duration(remaining)}"
        _slug_header(
            f"── [{i:>3}/{total}] {slug}  @{handle}  "
            f"· pop {_human_count(followers)}  "
            f"· last_touch {_format_age(last_touch)}"
            f"{eta_str}"
        )
        _heartbeat(
            state="running",
            pass_started_at=pass_started_at,
            pass_processed=processed,
            pass_total=total,
            current_slug=slug,
            current_handle=handle,
        )
        fresh = _refresh_safari_cookies()
        if fresh:
            auth_token, ct0 = fresh
        slug_counters: dict = {}
        try:
            slug_counters = _process_one(slug, handle, auth_token, ct0)
        except IngestAuthError as e:
            # Bad / missing token — every subsequent slug will fail
            # the same way, so abort the whole pass.
            _log_err(f"   ↳ aborting pass: {e}")
            _heartbeat(
                state="error",
                pass_started_at=pass_started_at,
                last_error=f"IngestAuthError: {e}",
            )
            return processed, total
        except _USER_FAULT_EXC as e:
            _log_warn(f"   ↳ skip (@{handle}): {type(e).__name__}: {str(e)[:120]}")
            pass_stats["skipped_user_faults"] += 1
            consecutive_api_failures = 0
        except (IngestUnavailable, IngestNetworkError) as e:
            # The admin API is the only remote dependency now (no
            # more SSH tunnel). When it's down EVERY slug will fail
            # the same way, so abort after a few back-to-back errors
            # and let the launchd watcher restart us when the server
            # comes back.
            consecutive_api_failures += 1
            _log_err(f"   ↳ API unreachable: {str(e)[:140]}")
            if consecutive_api_failures >= API_FAIL_THRESHOLD:
                _log_err(
                    f"   ↳ {consecutive_api_failures} consecutive API failures "
                    "— aborting pass; the next launchd cycle will retry."
                )
                _heartbeat(
                    state="error",
                    pass_started_at=pass_started_at,
                    last_error=f"api unavailable after {consecutive_api_failures} attempts",
                )
                return processed, total
            pass_stats["skipped_other"] += 1
        except Exception as e:
            _log_err(f"   ↳ {type(e).__name__}: {str(e)[:160]}  — moving on")
            pass_stats["skipped_other"] += 1
            consecutive_api_failures = 0
            _heartbeat(
                state="running",
                pass_started_at=pass_started_at,
                pass_processed=processed,
                pass_total=total,
                current_slug=slug,
                current_handle=handle,
                last_error=f"{type(e).__name__}: {e}",
            )
        else:
            consecutive_api_failures = 0
            # Per-slug summary line: time + outcome.
            n_new = slug_counters.get("inserted_total", 0)
            n_existed = slug_counters.get("existed_total", 0)
            slug_secs = time.perf_counter() - slug_t0
            if n_new > 0:
                _log_ok(f"   ↳ {slug}: +{n_new} new, {n_existed} already known ({_human_duration(slug_secs)})")
                pass_stats["with_new"] += 1
                pass_stats["inserted_total"] += n_new
                pass_stats["existed_total"] += n_existed
            else:
                _log_info(f"   ↳ {slug}: already up-to-date ({_human_duration(slug_secs)})")
                pass_stats["early_stops"] += 1
            pass_stats["api_errors"] += slug_counters.get("api_errors", 0)
            pass_stats["slugs_done"] += 1
        processed += 1
        if args.personality_delay > 0 and i < total:
            time.sleep(args.personality_delay)

    duration = time.perf_counter() - pass_t0
    _log_banner(
        f"pass complete · processed {processed}/{total} · "
        f"duration {_human_duration(duration)} · "
        f"+{pass_stats['inserted_total']} new tweets"
    )
    _log_info(
        f"   summary: {pass_stats['with_new']} slug(s) ingested new content, "
        f"{pass_stats['early_stops']} already up-to-date, "
        f"{pass_stats['skipped_user_faults']} user-faults, "
        f"{pass_stats['skipped_other']} other errors, "
        f"{pass_stats['api_errors']} API errors"
    )
    rate = pass_stats["inserted_total"] / max(duration, 0.001) * 60.0
    if pass_stats["inserted_total"] > 0:
        _log_info(f"   throughput: {rate:.1f} new tweets / min")
    _heartbeat(
        state="idle",
        pass_started_at=pass_started_at,
        pass_finished_at=_utc_iso(),
        pass_processed=processed,
        pass_total=total,
        pass_completed=True,
    )
    return processed, total


_stop_requested = False


def _install_signal_handlers() -> None:
    """SIGINT / SIGTERM → cooperative shutdown after the in-flight
    slug finishes (no half-saved state — the per-page flush
    guarantees durability)."""

    def _handler(signum, _frame):
        global _stop_requested
        if _stop_requested:
            # Second signal — abort hard.
            _log(f"signal {signum} received twice, aborting now")
            sys.exit(130)
        _stop_requested = True
        _log(
            f"signal {signum} received — finishing current slug then "
            "exiting cleanly (send another signal to force-quit)"
        )

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            # We might be running inside a thread that doesn't own
            # the main signal context — fine, the parent process's
            # handlers will fire.
            pass


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="knowledge-twitter-feed",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get("API_URL", "https://knowledge-web.org"),
        help=(
            "Base URL of the Knowledge API (defaults to $API_URL or "
            "https://knowledge-web.org). Queue, existing-URL and "
            "ingest calls go through this — no direct PG access."
        ),
    )
    p.add_argument(
        "--rest",
        type=int,
        default=3600,
        help=(
            "Seconds to sleep between full passes once every VIP "
            "missing twitter docs has been touched (or a rate-limit "
            "stall happens). Default 3600 (one hour). Set to 0 to "
            "exit after a single pass."
        ),
    )
    p.add_argument(
        "--personality-delay",
        type=int,
        default=4,
        help=(
            "Seconds to wait between personalities inside a single "
            "pass. Keeps the per-cookie twikit quota healthy. "
            "Default 4."
        ),
    )
    p.add_argument(
        "--one-shot",
        action="store_true",
        help="Do a single pass and exit (no rest loop).",
    )
    p.add_argument(
        "--min-age",
        type=float,
        default=0.0,
        help=(
            "Only process users whose latest twitter doc is older "
            "than this many hours (or who have never been touched). "
            "Default 0 — every VIP is fair game on every pass, "
            "ordered oldest-touched first."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    _install_signal_handlers()

    _log("knowledge-twitter-feed starting")
    # Forward the CLI api-url to the env so the _admin_* helpers
    # below see it without having to pass it through every layer.
    if args.api_url:
        os.environ["API_URL"] = args.api_url
    _log(f"  api_url={args.api_url}")
    _log(f"  rest={args.rest}s  personality_delay={args.personality_delay}s")

    # Bail out loudly *before* the main loop if Safari isn't logged in.
    # The pass body retries on its own once cookies come back, but the
    # boot-time check makes the failure mode obvious to a fresh
    # operator.
    try:
        tok, ct0 = _ensure_cookies()
        _log(f"  initial cookies OK (auth_token …{tok[-6:]}, ct0 …{ct0[-6:]})")
    except Exception as e:
        _log(f"[!] {e}")
        _log("    Open Safari, navigate to https://x.com, and confirm you're signed in. Then re-run.")
        return 2

    pass_n = 0
    while not _stop_requested:
        pass_n += 1
        _log(f"=== pass {pass_n} ===")
        t0 = time.perf_counter()
        processed, total = _one_pass(args)
        dur = time.perf_counter() - t0
        _log(f"pass {pass_n} done: processed {processed}/{total} in {int(dur)}s")
        if args.one_shot or _stop_requested:
            break
        if total == 0:
            _log(f"nothing to do — sleeping {args.rest}s before next sweep")
        else:
            _log(f"sleeping {args.rest}s before next sweep")
        # Sleep in 30 s chunks so a signal during rest exits quickly.
        slept = 0
        while slept < args.rest and not _stop_requested:
            time.sleep(min(30, args.rest - slept))
            slept += 30
    _log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
