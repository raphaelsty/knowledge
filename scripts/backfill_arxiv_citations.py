#!/usr/bin/env python3
"""Backfill arXiv `citation_count` via Semantic Scholar — no auth required.

What this does and why
----------------------
`sources/arxiv/papers.py` now batches a Semantic Scholar lookup at the
end of every author crawl to attach `citation_count` to fresh arxiv
docs. Rows ingested before that change — or arxiv URLs that came in
through other paths (Zotero, a tweeted paper, a blog citation) —
still have `citation_count IS NULL`. This script walks the docs
table, gathers every arxiv id whose row hasn't been measured, and
calls Semantic Scholar's `/graph/v1/paper/batch` endpoint to fill the
column in place.

Same lookup helper is reused for both jobs:
:func:`sources.arxiv.papers._fetch_arxiv_citations` (500 ids per call,
1 second between batches — comfortably under S2's free-tier ceiling).

Citation counts drift slowly compared to twitter engagement, but
they're not static: a year-old paper that suddenly gets cited in a
viral preprint can jump from 5 → 500 citations in weeks. Re-run with
`--refresh-older-than 90d` quarterly to keep the academic feed
ranked correctly.

Usage
-----
::

    DATABASE_URL=... uv run python scripts/backfill_arxiv_citations.py
    DATABASE_URL=... uv run python scripts/backfill_arxiv_citations.py --slug max-halford
    DATABASE_URL=... uv run python scripts/backfill_arxiv_citations.py --limit 1000 --dry
    DATABASE_URL=... uv run python scripts/backfill_arxiv_citations.py --refresh-older-than 90d
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timedelta, timezone

import psycopg

from sources.arxiv.papers import _fetch_arxiv_citations

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

# Recognise both the abs/ and pdf/ shapes; arxiv URLs in the wild
# come in as `arxiv.org/abs/2106.09685`, `arxiv.org/pdf/2106.09685`,
# and `arxiv.org/pdf/2106.09685v2.pdf` — we strip versions and the
# trailing `.pdf` so a single canonical id reaches Semantic Scholar.
ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


# ────────────────────────────────────────────────────────────────────
# Duration parsing — same syntax as the twitter backfill
# ────────────────────────────────────────────────────────────────────


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([dhm])\s*$", re.IGNORECASE)


def _parse_duration(s: str) -> timedelta:
    m = _DURATION_RE.match(s)
    if not m:
        try:
            return timedelta(days=int(s))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid duration {s!r}") from exc
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit == "d":
        return timedelta(days=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(minutes=n)


# ────────────────────────────────────────────────────────────────────
# Doc selection
# ────────────────────────────────────────────────────────────────────


def _candidate_rows(
    conn,
    slug: str | None,
    refresh_older_than: timedelta | None,
) -> list[tuple[str, str]]:
    """Return ``[(slug, url), …]`` for arxiv docs that need a citation lookup.

    Filter is URL-shape-based (``arxiv.org/abs/...`` or
    ``arxiv.org/pdf/...``) rather than `source = 'arxiv'`, because a
    paper ingested via Zotero / blog / twitter still has its
    canonical arxiv URL but lives under a different source bucket. We
    want citation counts on every arxiv paper regardless of how the
    pipeline found it.

    Selection:
      * Always: `citation_count IS NULL` — never measured.
      * Optionally: `engagement_updated_at < now() - X` — periodic
        refresh of stale counts.
    """
    sql = """
        SELECT u.username, d.url
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE d.deleted = false
           AND d.url ~* 'arxiv\\.org/(abs|pdf)/'
           AND (
                 d.citation_count IS NULL
    """
    params: list = []
    if refresh_older_than is not None:
        sql += "             OR d.engagement_updated_at < %s\n"
        params.append(datetime.now(timezone.utc) - refresh_older_than)
    sql += "           )\n"
    if slug:
        sql += "           AND u.username = %s\n"
        params.append(slug)
    sql += " ORDER BY u.username, d.date DESC NULLS LAST"
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return [(r[0], r[1]) for r in cur.fetchall()]


def _extract_arxiv_id(url: str) -> str | None:
    m = ARXIV_ID_RE.search(url or "")
    return m.group(1) if m else None


# ────────────────────────────────────────────────────────────────────
# Writer
# ────────────────────────────────────────────────────────────────────


def _update_citation(conn, slug: str, url: str, count: int) -> bool:
    """UPDATE `citation_count` for one doc.

    `engagement_updated_at` is stamped too — the column has always been
    documented as covering "any of the engagement signals", so a
    citation refresh counts.
    """
    sql = """
        UPDATE documents
           SET citation_count = %s,
               engagement_updated_at = now(),
               updated_at = now()
          FROM users u
         WHERE u.username = %s
           AND documents.user_id = u.id
           AND documents.url = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (count, slug, url))
        return cur.rowcount > 0


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slug",
        default=None,
        help="Restrict to one personality (default: all).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on docs to process this run (0 = all).",
    )
    p.add_argument(
        "--refresh-older-than",
        type=_parse_duration,
        default=None,
        help=(
            "Also refresh docs whose engagement_updated_at is older than this "
            "duration (e.g. '90d', '14d'). Default: only NULL-citation rows."
        ),
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Print what would change; don't write to PG.",
    )
    args = p.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(database_url) as conn:
        rows = _candidate_rows(
            conn,
            args.slug,
            refresh_older_than=args.refresh_older_than,
        )
        if args.limit:
            rows = rows[: args.limit]
        print(f"backfill: {len(rows)} arxiv doc(s) need citation counts")
        if not rows:
            return 0

        # Map arxiv_id → list of (slug, url). Same paper appears in
        # multiple libraries (different personalities all bookmark the
        # same preprint), and the same arxiv id can show up under both
        # an `abs/` and a `pdf/` URL in one library — fan the UPDATE
        # over every (slug, url) so a single S2 lookup pays off
        # everywhere.
        by_id: dict[str, list[tuple[str, str]]] = {}
        for slug, url in rows:
            aid = _extract_arxiv_id(url)
            if not aid:
                continue
            by_id.setdefault(aid, []).append((slug, url))
        print(f"  {len(by_id)} unique arxiv id(s) to look up")

        if args.dry:
            # In dry mode skip the S2 call entirely — we already know
            # what would be touched (every (slug, url) of every id).
            for aid, pairs in by_id.items():
                for slug, url in pairs:
                    print(f"  [dry] would look up {aid} → {slug} {url}")
            return 0

        # Single helper does the batching + pacing + error handling.
        # Returns `{arxiv_id: citation_count}` for ids S2 knows; missing
        # ids stay missing (NULL column, next run retries).
        citations = _fetch_arxiv_citations(list(by_id.keys()))
        print(f"  got citation counts for {len(citations)}/{len(by_id)} paper(s)")

        updated = 0
        skipped_no_hit = 0
        for aid, pairs in by_id.items():
            cc = citations.get(aid)
            if cc is None:
                skipped_no_hit += 1
                continue
            for slug, url in pairs:
                if _update_citation(conn, slug, url, cc):
                    updated += 1
        conn.commit()

        print(f"done: updated {updated} doc(s); {skipped_no_hit} arxiv id(s) unknown to Semantic Scholar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
