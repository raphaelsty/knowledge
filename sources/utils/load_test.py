"""Concurrent-connection saturation probe for the knowledge-api.

Ramps a thread pool from 1 → N concurrent workers, issues a
representative mix of requests at each level, and reports the
highest concurrency the host sustains within an SLO budget (p95
latency + error rate). Pure stdlib + the existing `requests` dep
— no new packages.

Why this shape rather than `hey` / `wrk` / `oha`:
  * Stays in the repo so `make` can drive it without bundling a
    Go/Rust binary in CI or asking the operator to brew install.
  * The endpoint mix lives in code, which means we always test
    the real shape the welcome page uses (feed + users + a
    representative search) rather than a single synthetic URL.
  * Per-level stats land in stdout in a format that's trivial to
    paste into an issue — operator-friendly.

Safety:
  * Ramps slowly and STOPS on the first SLO breach. The default
    ladder maxes at 200 — explicitly opt-in to go higher.
  * 30 s socket timeout on every request so a hung backend
    doesn't lock the runner.
  * Operator must pass `--yes` for any URL containing a
    production domain (`knowledge-web.org` by default), since
    this saturates real hardware.

Usage::

    # Local API (no consent needed):
    make load-test URL=http://localhost:8080

    # Prod (requires --yes):
    make load-test URL=https://knowledge-web.org YES=1

    # Custom ladder / duration:
    make load-test URL=... LEVELS=1,5,10,20,50,100 DURATION=15 YES=1
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_URL = "http://localhost:8080"

# Representative endpoint mix, weighted to mirror real welcome-page
# traffic. Every entry is repeated N times; the round-robin picker
# then samples uniformly from the expanded list, so the realised
# distribution matches the weights below:
#   feed  60% — every page load
#   users 30% — picker hydration + sidebar
#   search 10% — search bar invocation, much rarer per session
# Search alone takes 5-10s server-side (encoding + ColBERT) so over-
# representing it would conflate "feed scaling" with "embedder
# scaling", which are independent capacity questions.
ENDPOINT_MIX: list[tuple[str, str, dict | None]] = (
    [("GET", "/api/feed?limit=20", None)] * 6
    + [("GET", "/api/users", None)] * 3
    + [
        (
            "POST",
            "/indices/__all__/search_with_encoding",
            {"queries": ["transformer"], "params": {"top_k": 10}},
        )
    ]
    * 1
)

# Unique endpoints — only used for the "what are we hitting" banner.
ENDPOINT_PATHS = sorted({path for _, path, _ in ENDPOINT_MIX})

DEFAULT_LADDER = [1, 2, 5, 10, 20, 50, 100, 200]

# Stop conditions for the ramp. We DON'T halt on slow p95 — the
# operator wants to see the latency curve, not "ramp aborted at the
# first slow tail request". The hard stops are:
#   * error rate > threshold (something actually broke)
#   * p99 > 30s (timeout territory — the server isn't responding)
# p95 is still reported for every level so you can read the
# degradation as concurrency climbs.
SLO_ERROR_RATE = 0.05
SLO_P99_MS = 30_000.0


@dataclass
class LevelStats:
    concurrency: int
    n_requests: int
    rps: float
    p50: float
    p95: float
    p99: float
    err_rate: float
    sample_errors: list[Any] = field(default_factory=list)


def _worker(
    base: str,
    deadline: float,
    counter: dict,
    counter_lock: threading.Lock,
    latencies: list[float],
    errors: list[Any],
    lat_lock: threading.Lock,
) -> None:
    """One thread: pull request shapes round-robin off the mix until
    `deadline` elapses, push timings + error markers into shared lists."""
    sess = requests.Session()
    local_lat: list[float] = []
    local_err: list[Any] = []
    while time.monotonic() < deadline:
        with counter_lock:
            idx = counter["i"]
            counter["i"] += 1
        method, path, body = ENDPOINT_MIX[idx % len(ENDPOINT_MIX)]
        t0 = time.perf_counter()
        try:
            if method == "GET":
                r = sess.get(base + path, timeout=30)
            else:
                r = sess.post(base + path, json=body, timeout=30)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            local_lat.append(dt_ms)
            if r.status_code >= 500:
                local_err.append(r.status_code)
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            local_lat.append(dt_ms)
            local_err.append(type(e).__name__)
    # Single bulk-append per worker — minimises contention on the
    # shared lists. The order of timings doesn't matter for the
    # percentile calc downstream.
    with lat_lock:
        latencies.extend(local_lat)
        errors.extend(local_err)


def _pct(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, int(len(sorted_xs) * p))
    return sorted_xs[idx]


def run_level(base: str, concurrency: int, duration_secs: float) -> LevelStats | None:
    print(f"  concurrency={concurrency:>4} … ", end="", flush=True)
    deadline = time.monotonic() + duration_secs
    latencies: list[float] = []
    errors: list[Any] = []
    lat_lock = threading.Lock()
    counter = {"i": 0}
    counter_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(
                _worker,
                base,
                deadline,
                counter,
                counter_lock,
                latencies,
                errors,
                lat_lock,
            )
            for _ in range(concurrency)
        ]
        for f in as_completed(futures):
            f.result()
    n = len(latencies)
    if n == 0:
        print("(no responses)")
        return None
    sorted_lat = sorted(latencies)
    rps = n / duration_secs
    p50 = _pct(sorted_lat, 0.50)
    p95 = _pct(sorted_lat, 0.95)
    p99 = _pct(sorted_lat, 0.99)
    err_rate = len(errors) / n
    print(
        f"{n:>5} reqs · {rps:>6.0f} rps · "
        f"p50 {p50:>5.0f}ms · p95 {p95:>5.0f}ms · p99 {p99:>5.0f}ms · "
        f"errors {err_rate:>5.1%}"
    )
    # Cap the sample errors so a noisy run doesn't dump thousands.
    sample_errors = errors[:5]
    return LevelStats(
        concurrency=concurrency,
        n_requests=n,
        rps=rps,
        p50=p50,
        p95=p95,
        p99=p99,
        err_rate=err_rate,
        sample_errors=sample_errors,
    )


def _is_prod(url: str) -> bool:
    """Heuristic — anything outside localhost is treated as prod and
    requires --yes. Keeps an accidental `make load-test` from
    hammering the deployed API."""
    u = url.lower()
    return not (u.startswith("http://localhost") or u.startswith("http://127.0.0.1"))


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sources.utils.load_test",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--url",
        default=os.environ.get("LOAD_TEST_URL", DEFAULT_URL),
        help="Base URL (default: %(default)s).",
    )
    ap.add_argument(
        "--levels",
        default=os.environ.get("LOAD_TEST_LEVELS"),
        help=f"Comma-separated concurrency ladder. Default: {','.join(str(x) for x in DEFAULT_LADDER)}",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=float(os.environ.get("LOAD_TEST_DURATION", "10")),
        help="Seconds per concurrency level (default: 10).",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=int(os.environ.get("LOAD_TEST_MAX", "200")),
        help="Hard cap on concurrency — saturates the ladder above this (default: 200).",
    )
    ap.add_argument(
        "--p99-budget-ms",
        type=float,
        default=SLO_P99_MS,
        help=f"Stop ramp when p99 exceeds this (default: {SLO_P99_MS:.0f}ms — "
        "timeout territory). p95 / p50 are reported but never halt the run.",
    )
    ap.add_argument(
        "--err-budget",
        type=float,
        default=SLO_ERROR_RATE,
        help=f"Stop ramp when error rate exceeds this fraction (default: {SLO_ERROR_RATE:.2f}).",
    )
    ap.add_argument(
        "--responsive-p95-ms",
        type=float,
        default=2000.0,
        help="Latency threshold used ONLY to label the 'responsive "
        "concurrency' in the summary (default: 2000ms). The ramp "
        "doesn't halt on this.",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        default=os.environ.get("YES") == "1",
        help="Acknowledge that you are about to hammer the target. Required when URL is not localhost.",
    )
    args = ap.parse_args()

    if _is_prod(args.url) and not args.yes:
        print(
            f"\n[!] {args.url} looks like a production target.\n"
            f"    Re-run with `--yes` (or YES=1) to confirm you intend to load-test it.\n",
            file=sys.stderr,
        )
        return 2

    levels: list[int]
    if args.levels:
        levels = sorted({int(x.strip()) for x in args.levels.split(",") if x.strip()})
    else:
        levels = list(DEFAULT_LADDER)
    levels = [c for c in levels if c <= args.max]
    if not levels:
        print("[!] empty concurrency ladder after applying --max")
        return 2

    print(f"\nLoad test against {args.url}")
    print(f"Endpoint mix: {ENDPOINT_PATHS}")
    print(f"Per-level duration: {args.duration:.0f}s")
    print(f"Hard stop: errors > {args.err_budget:.0%} or p99 > {args.p99_budget_ms:.0f}ms (timeout territory)")
    print(f"'Responsive' label : p95 ≤ {args.responsive_p95_ms:.0f}ms (reporting threshold, not a stop signal)")
    print(f"Concurrency ladder: {levels}\n")

    results: list[LevelStats] = []
    breach: LevelStats | None = None
    for c in levels:
        r = run_level(args.url, c, args.duration)
        if not r:
            break
        results.append(r)
        if r.err_rate > args.err_budget or r.p99 > args.p99_budget_ms:
            breach = r
            reason = []
            if r.err_rate > args.err_budget:
                reason.append(f"errors {r.err_rate:.1%}")
            if r.p99 > args.p99_budget_ms:
                reason.append(f"p99 {r.p99:.0f}ms")
            print(f"\n>> Hard stop at concurrency={c} ({', '.join(reason)}) — ramp aborted\n")
            break

    if not results:
        return 2

    print("\n=== summary ===")
    print(f"{'conc':>6} {'rps':>7} {'p50':>9} {'p95':>9} {'p99':>9} {'err%':>7}")
    for r in results:
        print(
            f"{r.concurrency:>6} {r.rps:>7.0f} "
            f"{r.p50:>6.0f}ms {r.p95:>6.0f}ms {r.p99:>6.0f}ms "
            f"{r.err_rate * 100:>6.1f}%"
        )

    responsive = [r for r in results if r.err_rate <= args.err_budget and r.p95 <= args.responsive_p95_ms]
    if responsive:
        best = responsive[-1]
        print(
            f"\nResponsive ceiling : {best.concurrency} "
            f"({best.rps:.0f} rps, p95 {best.p95:.0f}ms, p99 {best.p99:.0f}ms)"
        )
    else:
        print(
            "\nResponsive ceiling : 0 — even the first level breached the "
            f"{args.responsive_p95_ms:.0f}ms p95 threshold "
            "(one of the endpoints in the mix is slow on its own)."
        )

    if breach:
        print(
            f"Hard-stop level    : {breach.concurrency} "
            f"({breach.rps:.0f} rps, p99 {breach.p99:.0f}ms, errors {breach.err_rate:.1%})"
        )
        if breach.sample_errors:
            print(f"Sample errors      : {breach.sample_errors!r}")
    else:
        last = results[-1]
        # Saturated capacity = the highest level we measured without
        # hitting a timeout or error wall. That's the best estimate
        # of "concurrent connections the host can sustain".
        print(
            f"Saturated capacity : {last.concurrency} concurrent "
            f"({last.rps:.0f} rps, p95 {last.p95:.0f}ms). "
            "Raise --max to keep climbing."
        )

    # Median across levels — cheap sanity check that nothing's totally broken.
    if len(results) >= 2:
        med_rps = statistics.median(r.rps for r in results)
        print(f"Median rps       : {med_rps:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
