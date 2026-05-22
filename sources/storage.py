"""Storage credit billing.

Companion to `sources/credits.py`. Twitter billing charges per API
call; storage billing charges per (document × month) for the bytes
the user occupies on our disk.

Pricing model
-------------
Linear per-document rate above a free quota. No tiers, no GB units
surfaced to the user — they think in documents, so we bill in
documents.

  ≤ FREE_DOCS              → free
  > FREE_DOCS              → USD_PER_DOC_PER_MONTH × surplus

The rate is anchored to real cost:

  • 20 KB on disk per non-VIP document (Postgres row + that user's
    next-plaid index slice). Measured against the live DB at 200k+ docs.
  • Hetzner Volume cost ≈ $0.005/GB-month (raw, what we pay them).
  • Per-doc raw infra cost = 20 KB × $0.005/GB ≈ $1e-7/doc/month.
  • Margin = 100×, to cover RAM (ColBERT indices stay warm),
    CPU (re-embedding on index rebuilds), bandwidth, ongoing
    operations, and the fact that real-world libraries top out
    around 100k docs — the rate is calibrated so a power user at
    that ceiling pays a meaningful $1/month rather than fifty cents.
  • Billed rate ≈ $1e-5/doc/month = $10 per million doc-months
    = $1 per 100k doc-months.

VIPs are free — they feed `__all__` (the heavy global index) but
pay nothing; the platform absorbs their cost.

A user is billed once per BILLING_PERIOD_DAYS. The "last charged"
timestamp is read from the most recent `kind='debit:storage'` row in
`credit_events`; no extra column is needed.
"""

from __future__ import annotations

import json as _json
import math
from typing import Any

import psycopg

# ── Rate constants ──────────────────────────────────────────────────
# Recipe to recompute USD_PER_DOC_PER_MONTH when the infra mix shifts:
#
#     raw  = BYTES_PER_DOC / 1024^3 * HETZNER_USD_PER_GB_MONTH
#     bill = raw * MARGIN
#
# As of 2026-05 that gives 20 KB × $0.005/GB-month × 50 ≈ $5e-6.
# Numbers below are pinned (not computed) so the rate doesn't move
# silently if a constant changes — the comment is the audit trail.

FREE_DOCS = 1_000  # first 1 000 documents are free
BYTES_PER_DOC = 20_000  # ~20 KB measured (1M docs ≈ 20 GB on disk)
HETZNER_USD_PER_GB_MONTH = 0.005  # Hetzner Volume cost (reference only)
STORAGE_MARGIN = 100  # over Hetzner raw (covers RAM, CPU, bandwidth)
USD_PER_DOC_PER_MONTH = 0.00001  # = 20 KB × $0.005/GB × 100 = $1 / 100k docs / mo
USD_PER_CREDIT = 0.01  # 1 credit = 1¢ — must match sources/credits.py
BILLING_PERIOD_DAYS = 30  # how often a paying user is ticked


def storage_credits(doc_count: int, *, free_docs: int = FREE_DOCS) -> int:
    """Credits owed for one BILLING_PERIOD_DAYS at this document count.

    Linear: the first `free_docs` are free; every document above
    that costs USD_PER_DOC_PER_MONTH. Result is rounded *up* to the
    nearest credit so we never under-bill — a user one document
    over the free quota still pays the 1-credit (1¢) floor.
    """
    n = int(doc_count)
    if n <= free_docs:
        return 0
    surplus = n - free_docs
    usd = surplus * USD_PER_DOC_PER_MONTH
    credits = math.ceil(usd / USD_PER_CREDIT)
    return max(1, credits)


def _dump_meta(meta: dict[str, Any] | None) -> str:
    return _json.dumps(meta or {}, default=str)


def _days_since_last_charge(cur: psycopg.Cursor, user_id: int) -> float | None:
    """Days elapsed since the user's most recent storage debit (None if never)."""
    cur.execute(
        """
        SELECT EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0
          FROM credit_events
         WHERE user_id = %s AND kind = 'debit:storage'
         ORDER BY id DESC
         LIMIT 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def _doc_count(cur: psycopg.Cursor, user_id: int) -> int:
    """Total documents owned by the user (across all their personalities)."""
    # documents.user_id maps a doc to its owner; this is the canonical
    # count for billing. The personality dimension isn't relevant —
    # one user can have N personalities, but the storage cost is per
    # row regardless.
    cur.execute("SELECT count(*) FROM documents WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    return int(row[0]) if row else 0


def charge_storage_if_due(
    database_url: str,
    user_id: int,
    *,
    is_vip: bool,
    billing_user_id: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Bill storage for one billing period, if a period has elapsed.

    `user_id` is the personality the documents belong to — we count
    documents under that account.

    `billing_user_id` is who pays. Defaults to `user_id` (the
    personality bills itself, the legacy behaviour). For
    sponsor-funded personalities (`users.sponsored_by IS NOT NULL`)
    the pipeline passes the sponsor's id here so the debit lands on
    their balance, not the personality's (which sits at $0).

    `is_vip` refers to the BILLING account — if the payer is VIP
    they pay nothing for any library they're billed for, including
    ones they sponsor.

    Behaviour matrix (against the billing account):
        VIP                       → skipped (free)
        under free quota          → skipped (0 credits owed)
        < BILLING_PERIOD_DAYS     → skipped (too early)
        insufficient credits      → debit failed; returns ok=False
    """
    bill_to = billing_user_id if billing_user_id is not None else user_id
    result: dict[str, Any] = {
        "user_id": user_id,
        "billing_user_id": bill_to,
        "charged": False,
        "ok": True,
        "reason": "",
        "docs": 0,
        "credits": 0,
    }

    if is_vip:
        result["reason"] = "vip"
        return result

    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                # Doc count is for the *personality*, not the payer.
                docs = _doc_count(cur, user_id)
                result["docs"] = docs

                owed = storage_credits(docs)
                result["credits"] = owed
                if owed == 0:
                    result["reason"] = "under_free_quota"
                    return result

                if not force:
                    # Billing period is tracked against the payer so
                    # back-to-back pipeline runs of multiple sponsored
                    # libraries on the same day don't all bill the
                    # sponsor at once.
                    days = _days_since_last_charge(cur, bill_to)
                    if days is not None and days < BILLING_PERIOD_DAYS:
                        result["reason"] = f"too_early ({days:.1f}d < {BILLING_PERIOD_DAYS}d)"
                        return result

                meta = {
                    "docs": docs,
                    "personality_user_id": user_id,
                    "free_docs": FREE_DOCS,
                    "bytes_per_doc": BYTES_PER_DOC,
                    "usd_per_doc_per_month": USD_PER_DOC_PER_MONTH,
                    "period_days": BILLING_PERIOD_DAYS,
                }
                cur.execute(
                    "SELECT credits_debit(%s, %s, %s, %s::jsonb)",
                    (bill_to, owed, "debit:storage", _dump_meta(meta)),
                )
                row = cur.fetchone()
                new_balance = row[0] if row else None
                if new_balance is None:
                    result["ok"] = False
                    result["reason"] = "insufficient_credits"
                    return result

                result["charged"] = True
                result["reason"] = "billed"
                result["new_balance"] = int(new_balance)
                return result
    except Exception as exc:
        result["ok"] = False
        result["reason"] = f"db_error: {exc}"
        return result
