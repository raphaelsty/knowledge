"""Recompute the per-viewer preference weights.

Two tables get rewritten in one pass: `user_personality_weight`
and `user_category_weight`. Both are derived from the raw signal
streams (favorites, library, events) and squashed through TANH so
no single signal can dominate the score.

Call it from a daemon, a cron, or manually:

    python -m sources.utils.user_preferences

The whole job runs as ONE transaction so the read-side never sees
a half-rewritten state. ~5 s on the current corpus; scales with
the size of the events table.

Signals + coefficients (chosen by hand, will tune once we have
behavioural data on prod):

    starred       weight 1.0   — favorite_documents (very strong)
    library       weight 0.7   — viewer's own documents (strong)
    click         weight 0.4   — events.event_type = 3
    find_similar  weight 0.4   — events.event_type = 4
    long_dwell    weight 0.15  — card_seen with dwell ≥ 5s
    short_dwell   weight -0.05 — card_seen with dwell < 2s (mild −)
    is_follower   weight 0.30  — favorites table (the explicit star
                                  the viewer placed on the personality)

The signal is broadcast across every sharer of a doc — if 5
personalities co-share a URL the viewer engaged with, all 5 get
the credit. This is intentional: "the viewer engaged with content
this person shares" is the signal, even when shared with others.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import psycopg

# Signal coefficients. Tweakable; squashed through TANH below so
# absolute scale doesn't matter, only the relative ratios.
COEF_STARRED = 1.0
COEF_LIBRARY = 0.7
COEF_CLICK = 0.4
COEF_FIND_SIMILAR = 0.4
COEF_LONG_DWELL = 0.15
COEF_SHORT_DWELL = -0.05
COEF_IS_FOLLOWER = 0.30

# Dwell thresholds (ms) for the two card_seen buckets.
LONG_DWELL_MS = 5000
SHORT_DWELL_MS = 2000

# Output range cap: weights live in [-OUTPUT_SCALE, +OUTPUT_SCALE].
# TANH gives [-1, +1]; multiplying by 2 lets a strongly-engaged
# personality contribute up to +2 to a doc's score, equivalent to
# the average VIP-share bonus.
OUTPUT_SCALE = 2.0


def _log() -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("knowledge.user_preferences")


def recompute_all(database_url: str) -> dict:
    """Rebuild both weight tables in one transaction.

    Returns counts written so callers / tests can sanity-check.
    """
    log = _log()
    t0 = time.monotonic()
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            personality_n = _recompute_personality(cur)
            category_n = _recompute_category(cur)
        conn.commit()
    elapsed = time.monotonic() - t0
    log.info(
        "user_preferences.recompute.complete personality=%d category=%d elapsed=%.2fs",
        personality_n,
        category_n,
        elapsed,
    )
    return {
        "personality_rows": personality_n,
        "category_rows": category_n,
        "elapsed_secs": elapsed,
    }


# ── Per-personality recompute ───────────────────────────────────────


def _recompute_personality(cur: psycopg.Cursor) -> int:
    """Recompute `user_personality_weight` from scratch.

    Strategy: for every (viewer, personality) pair that has at least
    one signal, sum the weighted signal counts and TANH-squash. We
    don't bother with rows that have zero weight; missing rows in
    the read path default to 0 already.

    Mapping URL → sharer set is done via the `documents` table, NOT
    `feed_snapshot` — the snapshot only contains the top-60k anchors,
    but a viewer's engagement with a long-tail URL should still
    train their personality weights. The events table stores
    `doc_url`, so the join is `events.doc_url = documents.url`.
    """
    # `TRUNCATE` + INSERT is simpler than UPSERT here because the
    # recompute is from-scratch — no delta logic needed. The
    # transaction wrapping the whole job means readers see either
    # the old snapshot or the new one, never a half state.
    cur.execute("TRUNCATE user_personality_weight")
    cur.execute(
        """
        WITH starred AS (
            -- Per (viewer, personality): how many of the viewer's
            -- starred docs has this personality also saved?
            SELECT fd.user_id      AS viewer_id,
                   d.user_id       AS personality_id,
                   COUNT(*)::float AS n
              FROM favorite_documents fd
              JOIN documents          d ON d.url = fd.url AND d.deleted = FALSE
             GROUP BY fd.user_id, d.user_id
        ),
        library AS (
            -- Per (viewer, personality): how many docs the viewer
            -- has in their own library are also in this personality's
            -- library? Excludes self-overlap (the viewer = personality
            -- case adds no signal).
            SELECT v.user_id       AS viewer_id,
                   p.user_id       AS personality_id,
                   COUNT(*)::float AS n
              FROM documents v
              JOIN documents p
                ON p.url     = v.url
               AND p.user_id <> v.user_id
               AND p.deleted = FALSE
             WHERE v.deleted = FALSE
             GROUP BY v.user_id, p.user_id
        ),
        clicks AS (
            -- Per (viewer, personality): event_type = 3 (click).
            -- Maps doc_url to documents to find sharers.
            SELECT e.viewer_user_id AS viewer_id,
                   d.user_id        AS personality_id,
                   COUNT(*)::float  AS n
              FROM events    e
              JOIN documents d ON d.url = e.doc_url AND d.deleted = FALSE
             WHERE e.viewer_user_id IS NOT NULL
               AND e.event_type    = 3
             GROUP BY e.viewer_user_id, d.user_id
        ),
        find_similar AS (
            -- event_type = 4
            SELECT e.viewer_user_id AS viewer_id,
                   d.user_id        AS personality_id,
                   COUNT(*)::float  AS n
              FROM events    e
              JOIN documents d ON d.url = e.doc_url AND d.deleted = FALSE
             WHERE e.viewer_user_id IS NOT NULL
               AND e.event_type    = 4
             GROUP BY e.viewer_user_id, d.user_id
        ),
        dwell AS (
            -- card_seen (event_type=7) split by dwell length.
            SELECT e.viewer_user_id AS viewer_id,
                   d.user_id        AS personality_id,
                   SUM(CASE WHEN e.dwell_ms >= %s THEN 1 ELSE 0 END)::float AS long_n,
                   SUM(CASE WHEN e.dwell_ms <  %s THEN 1 ELSE 0 END)::float AS short_n
              FROM events    e
              JOIN documents d ON d.url = e.doc_url AND d.deleted = FALSE
             WHERE e.viewer_user_id IS NOT NULL
               AND e.event_type    = 7
               AND e.dwell_ms     IS NOT NULL
             GROUP BY e.viewer_user_id, d.user_id
        ),
        is_follower AS (
            -- The viewer explicitly starred the personality. One row
            -- means the prior is set; the COUNT here is always 1.
            SELECT user_id AS viewer_id,
                   favorite_id AS personality_id,
                   1.0::float AS n
              FROM favorites
        ),
        scored AS (
            SELECT COALESCE(s.viewer_id, l.viewer_id, c.viewer_id, fs.viewer_id, dw.viewer_id, f.viewer_id) AS viewer_id,
                   COALESCE(s.personality_id, l.personality_id, c.personality_id, fs.personality_id, dw.personality_id, f.personality_id) AS personality_id,
                   COALESCE(s.n,  0) * %s   -- COEF_STARRED
                 + COALESCE(l.n,  0) * %s   -- COEF_LIBRARY
                 + COALESCE(c.n,  0) * %s   -- COEF_CLICK
                 + COALESCE(fs.n, 0) * %s   -- COEF_FIND_SIMILAR
                 + COALESCE(dw.long_n,  0) * %s   -- COEF_LONG_DWELL
                 + COALESCE(dw.short_n, 0) * %s   -- COEF_SHORT_DWELL
                 + COALESCE(f.n,  0) * %s         -- COEF_IS_FOLLOWER
                   AS raw_score
              FROM        starred       s
              FULL JOIN library          l ON l.viewer_id = s.viewer_id AND l.personality_id = s.personality_id
              FULL JOIN clicks           c ON c.viewer_id = COALESCE(s.viewer_id,l.viewer_id) AND c.personality_id = COALESCE(s.personality_id,l.personality_id)
              FULL JOIN find_similar     fs ON fs.viewer_id = COALESCE(s.viewer_id,l.viewer_id,c.viewer_id) AND fs.personality_id = COALESCE(s.personality_id,l.personality_id,c.personality_id)
              FULL JOIN dwell            dw ON dw.viewer_id = COALESCE(s.viewer_id,l.viewer_id,c.viewer_id,fs.viewer_id) AND dw.personality_id = COALESCE(s.personality_id,l.personality_id,c.personality_id,fs.personality_id)
              FULL JOIN is_follower      f  ON f.viewer_id  = COALESCE(s.viewer_id,l.viewer_id,c.viewer_id,fs.viewer_id,dw.viewer_id) AND f.personality_id  = COALESCE(s.personality_id,l.personality_id,c.personality_id,fs.personality_id,dw.personality_id)
        )
        INSERT INTO user_personality_weight (viewer_id, personality_id, weight, refreshed_at)
        SELECT viewer_id,
               personality_id,
               -- TANH-squash so any single signal can't dominate.
               -- Output capped in [-OUTPUT_SCALE, +OUTPUT_SCALE].
               (TANH(raw_score / 5.0) * %s)::real AS weight,
               now()
          FROM scored
         WHERE viewer_id IS NOT NULL
           AND personality_id IS NOT NULL
           AND viewer_id <> personality_id   -- ignore self-engagement
           AND raw_score <> 0
    """,
        (
            LONG_DWELL_MS,
            SHORT_DWELL_MS,
            COEF_STARRED,
            COEF_LIBRARY,
            COEF_CLICK,
            COEF_FIND_SIMILAR,
            COEF_LONG_DWELL,
            COEF_SHORT_DWELL,
            COEF_IS_FOLLOWER,
            OUTPUT_SCALE,
        ),
    )
    return cur.rowcount or 0


# ── Per-category recompute ──────────────────────────────────────────


def _recompute_category(cur: psycopg.Cursor) -> int:
    """Recompute `user_category_weight`.

    Same shape as the personality recompute but the broadcast key
    is `document_categories.slug` instead of `documents.user_id`.
    We re-use `document_category_assignments` to map URL → category
    slugs, then aggregate signals.
    """
    cur.execute("TRUNCATE user_category_weight")
    cur.execute(
        """
        WITH cat_of_url AS (
            -- Distinct (url, slug) pairs across every doc-category
            -- assignment. A URL saved by two users in two different
            -- categories contributes to both.
            SELECT DISTINCT a.url, dc.slug
              FROM document_category_assignments a
              JOIN document_categories dc ON dc.id = a.category_id
        ),
        starred AS (
            SELECT fd.user_id AS viewer_id, c.slug, COUNT(*)::float AS n
              FROM favorite_documents fd
              JOIN cat_of_url c ON c.url = fd.url
             GROUP BY fd.user_id, c.slug
        ),
        library AS (
            SELECT d.user_id AS viewer_id, c.slug, COUNT(*)::float AS n
              FROM documents d
              JOIN cat_of_url c ON c.url = d.url
             WHERE d.deleted = FALSE
             GROUP BY d.user_id, c.slug
        ),
        clicks AS (
            SELECT e.viewer_user_id AS viewer_id, c.slug, COUNT(*)::float AS n
              FROM events e
              JOIN cat_of_url c ON c.url = e.doc_url
             WHERE e.viewer_user_id IS NOT NULL AND e.event_type = 3
             GROUP BY e.viewer_user_id, c.slug
        ),
        find_similar AS (
            SELECT e.viewer_user_id AS viewer_id, c.slug, COUNT(*)::float AS n
              FROM events e
              JOIN cat_of_url c ON c.url = e.doc_url
             WHERE e.viewer_user_id IS NOT NULL AND e.event_type = 4
             GROUP BY e.viewer_user_id, c.slug
        ),
        dwell AS (
            SELECT e.viewer_user_id AS viewer_id, c.slug,
                   SUM(CASE WHEN e.dwell_ms >= %s THEN 1 ELSE 0 END)::float AS long_n,
                   SUM(CASE WHEN e.dwell_ms <  %s THEN 1 ELSE 0 END)::float AS short_n
              FROM events e
              JOIN cat_of_url c ON c.url = e.doc_url
             WHERE e.viewer_user_id IS NOT NULL AND e.event_type = 7
               AND e.dwell_ms IS NOT NULL
             GROUP BY e.viewer_user_id, c.slug
        ),
        scored AS (
            SELECT COALESCE(s.viewer_id, l.viewer_id, c.viewer_id, fs.viewer_id, dw.viewer_id) AS viewer_id,
                   COALESCE(s.slug, l.slug, c.slug, fs.slug, dw.slug) AS slug,
                   COALESCE(s.n,  0) * %s
                 + COALESCE(l.n,  0) * %s
                 + COALESCE(c.n,  0) * %s
                 + COALESCE(fs.n, 0) * %s
                 + COALESCE(dw.long_n,  0) * %s
                 + COALESCE(dw.short_n, 0) * %s
                   AS raw_score
              FROM     starred       s
              FULL JOIN library      l ON l.viewer_id = s.viewer_id AND l.slug = s.slug
              FULL JOIN clicks       c ON c.viewer_id = COALESCE(s.viewer_id,l.viewer_id) AND c.slug = COALESCE(s.slug,l.slug)
              FULL JOIN find_similar fs ON fs.viewer_id = COALESCE(s.viewer_id,l.viewer_id,c.viewer_id) AND fs.slug = COALESCE(s.slug,l.slug,c.slug)
              FULL JOIN dwell        dw ON dw.viewer_id = COALESCE(s.viewer_id,l.viewer_id,c.viewer_id,fs.viewer_id) AND dw.slug = COALESCE(s.slug,l.slug,c.slug,fs.slug)
        )
        INSERT INTO user_category_weight (viewer_id, category_slug, weight, refreshed_at)
        SELECT viewer_id,
               slug,
               (TANH(raw_score / 5.0) * %s)::real,
               now()
          FROM scored
         WHERE viewer_id IS NOT NULL
           AND slug      IS NOT NULL
           AND raw_score <> 0
    """,
        (
            LONG_DWELL_MS,
            SHORT_DWELL_MS,
            COEF_STARRED,
            COEF_LIBRARY,
            COEF_CLICK,
            COEF_FIND_SIMILAR,
            COEF_LONG_DWELL,
            COEF_SHORT_DWELL,
            OUTPUT_SCALE,
        ),
    )
    return cur.rowcount or 0


# ── CLI ─────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres connection string (defaults to $DATABASE_URL)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL not set (pass --database-url or export it)")
    recompute_all(args.database_url)


if __name__ == "__main__":
    main()
