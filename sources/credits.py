"""Python-side credit helpers used by the pipeline.

The web API has its own credit handlers in `api/src/handlers/credits.rs`;
this module is the matching surface for Python callers (pipeline +
batch jobs). It wraps the `credits_debit()` / `credits_balance()` SQL
functions defined in `sources/sql/credits.sql`.

The primary export is a `Budget` class that decides whether the next
paid external API call is allowed. Two factory helpers build a
correctly-configured budget for each cost type:

    twitter_budget(...) — Twitter API page fetches (1 credit / page)

VIPs (the grandfathered cohort + sponsored additions) pay nothing —
their budgets short-circuit `allow()` to True and never touch the
ledger. Everyone else pays per page; pagination stops when their
balance hits zero.

Schema: see sources/sql/credits.sql. All movements go through the
locked-row SQL helpers so concurrent pipeline runs can't over-debit.
"""

from __future__ import annotations

import json as _json
import math
from typing import Any

import psycopg

# ── Reference rates ─────────────────────────────────────────────────
# 1 credit is worth USD_PER_CREDIT. The bill for a paid call is:
#
#     billed_usd = (USD_PER_REQUEST + tweets * USD_PER_TWEET) * MARGIN
#     credits    = max(1, ceil(billed_usd / USD_PER_CREDIT))
#
# All four constants are tuned for twitterapi.io's published rate
# ($0.15 / 1 000 tweets) plus a 2× margin so the platform pockets
# enough to cover bandwidth, Postgres, and price drift without
# bleeding cash on a heavy fetcher run.
#
# IMPORTANT: keep `MARGIN >= 1.0`. The integer-ceiling on the credit
# count already gives a small fractional cushion, but the margin
# multiplier is the real safety net — bumping it to 2.5 or 3.0 is
# the right move if Twitter's price moves against us.

USD_PER_TWEET = 0.00015  # twitterapi.io: $0.15 / 1 000 tweets
USD_PER_REQUEST = 0.0005  # rough per-call overhead (their CDN + ours)
USD_PER_CREDIT = 0.01  # 1 credit = 1 ¢ on the user's invoice
MARGIN = 2.0  # billing multiplier
MIN_CREDITS_PER_CALL = 1  # floor (no "free" paid calls)


def twitter_page_cost(tweets_returned: int) -> int:
    """Credits to debit for one Twitter API page or hydrate-chunk.

    `tweets_returned` is the COUNT OF TWEETS in the API response, not
    the page number. Empty pages still cost the floor.
    """
    raw_usd = USD_PER_REQUEST + max(0, int(tweets_returned)) * USD_PER_TWEET
    billed_usd = raw_usd * MARGIN
    credits = math.ceil(billed_usd / USD_PER_CREDIT)
    return max(MIN_CREDITS_PER_CALL, credits)


# Pre-flight guardrail: we want to refuse a page fetch when the
# balance can't even cover the worst-case 100-tweet page. Avoids
# paying for an API call we then can't bill for.
def twitter_worst_case_cost() -> int:
    return twitter_page_cost(100)


# ── Export pricing ──────────────────────────────────────────────────
# When a user downloads the documents of a personality they don't own,
# they pay one credit per EXPORT_DOCS_PER_CREDIT documents. Math:
#
#   raw_unit_cost_per_doc ≈ $0.00018  (ingest + storage amortized)
#   billed_per_doc        = $0.0002   (= 1 credit / 50 docs)
#   margin                ≈ 33% over unit cost
#   buyer pays            < cost of re-fetching the same data via the
#                           ingest path ($0.0003/doc), so the export
#                           route stays attractive without giving the
#                           library away for free.
#
# Special cases — the export endpoint enforces these, the bill helper
# just returns the headline number:
#   • owner of the personality                → free
#   • VIPs (any personality)                  → free
#   • everyone else (public personality only) → paid at the rate below
EXPORT_DOCS_PER_CREDIT = 50
MIN_EXPORT_CREDITS = 1


def export_cost(doc_count: int) -> int:
    """Credits owed to export `doc_count` documents (non-owner, non-VIP).

    Floor of MIN_EXPORT_CREDITS so a 1-document export still costs the
    1¢ minimum — keeps the billing path honest without surprising the
    buyer for trivial fetches.
    """
    if doc_count <= 0:
        return MIN_EXPORT_CREDITS
    return max(
        MIN_EXPORT_CREDITS,
        math.ceil(int(doc_count) / EXPORT_DOCS_PER_CREDIT),
    )


def _dump_meta(meta: dict[str, Any] | None) -> str:
    """Serialise meta JSON for the SQL `jsonb` cast. None → '{}'."""
    return _json.dumps(meta or {}, default=str)


class Budget:
    """Gate on a sequence of paid external API calls with variable cost.

    Two-step protocol per call:
      1. ``precheck(worst_case)`` — refuse the next API call up-front
         when the balance can't even cover the worst-case bill. This
         avoids paying twitterapi.io for a fetch we then can't bill to
         the user.
      2. ``charge(amount, meta)`` — debit the *actual* cost once the
         response is in hand and we know how many tweets came back.

    VIPs get ``free=True`` and both methods short-circuit to True
    without touching the database.
    """

    def __init__(
        self,
        *,
        free: bool,
        database_url: str | None,
        user_id: int | None,
        kind: str,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        self.free = free
        self._database_url = database_url
        self._user_id = user_id
        self._kind = kind
        # `meta_extra` is merged into every charge() row. Lets the
        # caller stamp persistent context (e.g. the personality_user_id
        # being parsed) without each charge site repeating itself.
        self._meta_extra = meta_extra or {}
        self.calls = 0
        self.spent = 0
        self.refused_at_call: int | None = None

    def __repr__(self) -> str:
        suffix = " (VIP)" if self.free else ""
        return f"<Budget {self._kind}: {self.calls} calls, {self.spent} credits{suffix}>"

    def _balance(self) -> int | None:
        if self._database_url is None or self._user_id is None:
            return None
        try:
            with psycopg.connect(self._database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT credits_balance(%s)", (self._user_id,))
                    row = cur.fetchone()
        except Exception as exc:
            print(f"    credits: balance read failed for user={self._user_id}: {exc}")
            return None
        return int(row[0]) if row and row[0] is not None else 0

    def precheck(self, worst_case: int) -> bool:
        """Return True when the user can afford at least one more worst-case call."""
        if self.free:
            return True
        bal = self._balance()
        if bal is None:
            self.refused_at_call = self.calls + 1
            return False
        if bal < max(1, int(worst_case)):
            self.refused_at_call = self.calls + 1
            return False
        return True

    def charge(self, amount: int, meta: dict[str, Any] | None = None) -> bool:
        """Debit the actual cost. Returns True on success, False on insufficient/error."""
        amount = max(MIN_CREDITS_PER_CALL, int(amount))
        if self.free:
            self.calls += 1
            return True
        if self._database_url is None or self._user_id is None:
            self.refused_at_call = self.calls + 1
            return False
        # Merge persistent context (e.g. personality_user_id) into
        # each row's meta so the "cost per personality" report can
        # group debits without joining on fragile string keys.
        merged_meta = {**self._meta_extra, **(meta or {})}
        try:
            with psycopg.connect(self._database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT credits_debit(%s, %s, %s, %s::jsonb)",
                        (self._user_id, amount, self._kind, _dump_meta(merged_meta)),
                    )
                    row = cur.fetchone()
        except Exception as exc:
            print(f"    credits: debit failed for user={self._user_id}: {exc}")
            self.refused_at_call = self.calls + 1
            return False
        new_balance = row[0] if row else None
        if new_balance is None:
            self.refused_at_call = self.calls + 1
            return False
        self.calls += 1
        self.spent += amount
        return True


def twitter_budget(
    database_url: str,
    user_id: int,
    is_vip: bool,
    *,
    personality_user_id: int | None = None,
) -> Budget:
    """Build a Twitter-fetch budget for one pipeline run.

    VIPs get a free budget — no rows in `credit_events`. Non-VIPs are
    billed per page based on the number of tweets actually returned,
    via ``twitter_page_cost()``.

    `personality_user_id` is the user whose timeline is being parsed
    (which may differ from `user_id` — the payer — when a sponsor is
    funding the run). When set, every debit row's meta carries this
    id so the "cost per personality" report can group cleanly.
    """
    return Budget(
        free=is_vip,
        database_url=None if is_vip else database_url,
        user_id=None if is_vip else user_id,
        kind="debit:twitter-api",
        meta_extra={"personality_user_id": personality_user_id} if personality_user_id is not None else None,
    )


def balance(database_url: str, user_id: int) -> int:
    """Convenience accessor — current balance for a user."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT credits_balance(%s)", (user_id,))
            row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0
