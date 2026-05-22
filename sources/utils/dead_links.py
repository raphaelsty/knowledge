"""
Dead-link detector — checks URLs with a HEAD request and removes unreachable ones.

Only checks newly added URLs (passed explicitly by the caller) so the cost
is proportional to the number of new documents per run, not the total database.

Usage from the pipeline::

    from sources.utils import DeadLinks

    checker = DeadLinks()
    dead = checker.check(urls)  # → set of dead URLs
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from urllib.parse import urlparse

import requests

__all__ = ["DeadLinks"]

# Domains where a HEAD/GET probe is pointless or misleading.
# These always require authentication, return 403 to bots, or are
# known-good infrastructure that never has "dead" pages.
_SKIP_DOMAINS = frozenset(
    {
        "x.com",
        "twitter.com",
        "t.co",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "scholar.google.com",
        "scholar.google.co.uk",
    }
)

# Patterns for URLs that are always valid by construction (API-generated).
_SKIP_PATTERNS = (
    re.compile(r"^https?://arxiv\.org/abs/\d+\.\d+"),
    re.compile(r"^https?://news\.ycombinator\.com/item\?id=\d+"),
    re.compile(r"^https?://huggingface\.co/"),
)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)

# HTTP status codes that mean the page is genuinely gone.
# 401/403 are intentionally excluded — the page exists but is restricted.
_DEAD_STATUSES = frozenset({404, 410})


def _should_skip(url: str) -> bool:
    """Return True if this URL should not be probed."""
    try:
        host = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return True
    if any(host.endswith(d) for d in _SKIP_DOMAINS):
        return True
    return any(p.match(url) for p in _SKIP_PATTERNS)


def _check_one(url: str, timeout: float) -> tuple[str, bool]:
    """Probe a single URL.  Returns ``(url, is_dead)``.

    Tries HEAD first (cheap). Falls back to a streaming GET if the
    server returns 405 (Method Not Allowed).
    """
    try:
        r = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        if r.status_code == 405:
            r = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
                stream=True,
            )
            r.close()
        return (url, r.status_code in _DEAD_STATUSES)
    except requests.ConnectionError:
        return (url, True)
    except requests.Timeout:
        return (url, False)
    except Exception:
        return (url, True)


class DeadLinks:
    """Check a batch of URLs for dead links.

    Parameters
    ----------
    per_request_timeout : float
        Timeout in seconds for each individual HEAD/GET probe.
    total_timeout : float
        Wall-clock cap for the entire batch.  Any URLs still in-flight
        when this expires are assumed alive (not dead).
    workers : int
        Number of concurrent threads for probing.
    """

    def __init__(
        self,
        per_request_timeout: float = 5,
        total_timeout: float = 60,
        workers: int = 16,
    ):
        self.per_request_timeout = per_request_timeout
        self.total_timeout = total_timeout
        self.workers = workers

    def check(self, urls: set[str] | list[str]) -> set[str]:
        """Return the subset of *urls* that are dead (404, 410, DNS fail, …).

        URLs matching ``_SKIP_DOMAINS`` or ``_SKIP_PATTERNS`` are never
        probed and never returned as dead.  If the batch exceeds
        ``total_timeout``, remaining in-flight probes are cancelled and
        their URLs are assumed alive.
        """
        to_probe = [u for u in urls if not _should_skip(u)]
        if not to_probe:
            return set()

        skipped = len(urls) - len(to_probe)
        print(f"    Checking {len(to_probe)} links ({skipped} skipped)...")

        dead: set[str] = set()
        t0 = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_check_one, url, self.per_request_timeout): url for url in to_probe}

            remaining = self.total_timeout
            try:
                for future in concurrent.futures.as_completed(futures, timeout=remaining):
                    url, is_dead = future.result()
                    if is_dead:
                        dead.add(url)
                    # Shrink the remaining budget so as_completed's next
                    # iteration uses the correct deadline.
                    elapsed = time.monotonic() - t0
                    remaining = max(0, self.total_timeout - elapsed)
                    if remaining == 0:
                        break
            except concurrent.futures.TimeoutError:
                # Budget exhausted with requests still in flight — treat
                # pending URLs as alive (conservative) and move on.
                pending = sum(1 for f in futures if not f.done())
                print(f"    Dead-link check deadline reached, {pending} URLs unchecked (assumed alive)")

        # Cancel anything still running after the deadline.
        for future in futures:
            future.cancel()

        elapsed = time.monotonic() - t0
        if dead:
            print(f"    Found {len(dead)} dead links ({elapsed:.1f}s)")
        else:
            print(f"    All {len(to_probe)} links alive ({elapsed:.1f}s)")

        return dead
