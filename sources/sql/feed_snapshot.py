"""Helpers for the `feed_snapshot` table.

Two responsibilities:

* `create_feed_snapshot_table` — idempotent schema bootstrap, called
  from `run.py` (and from the API's boot migrations via include_str!).
* `refresh_feed_snapshot` — atomic TRUNCATE+INSERT that re-scores the
  last `window_days` of documents and writes the result back. Invoked
  hourly by `knowledge-feed-snapshot-daemon`.

The score formula matches the viewer-agnostic part of the live
timeline query in api/src/handlers/follows.rs — keep them in sync.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "feed_snapshot.sql"


def create_feed_snapshot_table(database_url: str) -> None:
    """Apply `feed_snapshot.sql` against the DB. Idempotent."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


# Refresh window in days. The score's recency bonus caps at 5 weeks
# anyway, so beyond that point docs compete purely on sharer count;
# 180 days lets older deep cuts surface when multiple followees
# co-own them.
DEFAULT_WINDOW_DAYS = 180

# Hard cap on snapshot size. ~20k rows comfortably handles every
# request (50–200 rows requested) with a tail for cold paginations;
# the score-DESC index then keeps scans tight. Raising this is cheap
# storage-wise but extends the refresh cost linearly.
DEFAULT_MAX_ROWS = 20_000


def refresh_feed_snapshot(
    database_url: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> int:
    """Rebuild `feed_snapshot` atomically. Returns rows written.

    Strategy:
      * Score every URL inside the `window_days` window using the
        same formula the live handler used pre-snapshot (minus the
        viewer-specific terms).
      * Pick one representative per `anchor_url` (the visually-richest
        doc — most preview images, then most referenced URLs).
      * Aggregate sharers across all docs that share the anchor.
      * Take the top `max_rows` by score.
      * TRUNCATE + INSERT inside one transaction — readers see either
        the old snapshot in full or the new one in full, never a
        partial state.

    The TRUNCATE briefly takes an ACCESS EXCLUSIVE lock on the table.
    Readers queue for the duration of the INSERT (typically a few
    seconds on 20 k rows); the handler's `acquire_timeout=5s` then
    fails them over to the live-query fallback if the lock holds
    longer than expected.
    """
    insert_sql = _build_refresh_sql(window_days=window_days, max_rows=max_rows)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            # psycopg opens a transaction implicitly on first
            # statement; `commit` happens at context exit.
            cur.execute("TRUNCATE feed_snapshot")
            cur.execute(insert_sql)
            written = cur.rowcount or 0
        conn.commit()
    return written


def feed_snapshot_age_seconds(database_url: str) -> int | None:
    """How old is the freshest row, in seconds? `None` if empty."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT EXTRACT(EPOCH FROM (now() - MAX(refreshed_at)))::bigint FROM feed_snapshot")
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


# ── SQL builder ──────────────────────────────────────────────────────


def _build_refresh_sql(window_days: int, max_rows: int) -> str:
    """The big scoring CTE. Returns a `INSERT INTO feed_snapshot ...`
    string. Kept as a function (not a static string) so the daemon
    can tune window/max without code edits."""
    # NB: this query is heavy by design. Worth a few seconds once an
    # hour to keep every read instant. We compute against `documents`
    # (the source of truth) — the cached anchor / canonical URL fields
    # are GENERATED STORED so this scan is index-friendly.
    return f"""
        INSERT INTO feed_snapshot (
            url, canonical_url, anchor_url,
            title, date, summary, clean_title, clean_summary, urls, tags,
            source, source_url, linked_urls, link_hosts,
            primary_user_id, sharer_user_ids, sharers, sharer_count,
            any_vip_sharer, score, refreshed_at
        )
        WITH window_docs AS (
            -- Every non-deleted doc inside the window. We pre-compute
            -- per-doc anchor + image/url richness here so the
            -- representative pick below is a simple ORDER BY.
            SELECT
                d.user_id, d.url, d.title, d.date, d.summary,
                d.clean_title, d.clean_summary, d.urls, d.tags,
                d.extra_tags, d.source, d.source_url, d.created_at,
                d.linked_urls, d.link_hosts,
                d.canonical_url,
                d.canonical_referenced_urls,
                COALESCE(
                    (SELECT ref
                       FROM unnest(d.canonical_referenced_urls) ref
                      ORDER BY CASE
                          WHEN ref LIKE 'https://arxiv.org/abs/%'       THEN 1
                          WHEN ref LIKE 'https://huggingface.co/%'      THEN 2
                          WHEN ref LIKE 'https://github.com/%'          THEN 3
                          WHEN ref LIKE 'https://openreview.net/%'      THEN 4
                          WHEN ref LIKE 'https://doi.org/%'             THEN 5
                          WHEN ref LIKE 'https://paperswithcode.com/%'  THEN 6
                          WHEN ref LIKE 'https://aclanthology.org/%'    THEN 7
                          WHEN ref LIKE 'https://semanticscholar.org/%' THEN 8
                          WHEN ref LIKE 'https://distill.pub/%'         THEN 9
                          WHEN ref LIKE 'https://biorxiv.org/%'         THEN 10
                          WHEN ref LIKE 'https://medrxiv.org/%'         THEN 11
                          ELSE 99
                      END, ref
                      LIMIT 1),
                    d.canonical_url
                ) AS anchor_url,
                COALESCE((
                    SELECT count(*)::int
                      FROM jsonb_array_elements(d.linked_urls) e
                     WHERE COALESCE(e->>'image', '') <> ''
                ), 0) AS image_count,
                cardinality(d.canonical_referenced_urls) AS url_count,
                -- Resource-type signal — same enum as the live query.
                CASE
                    WHEN d.source IN ('arxiv', 'scholar', 'huggingface')
                      THEN 3
                    WHEN d.source = 'twitter'
                         AND d.link_hosts && ARRAY[
                           'arxiv', 'arxiv.org',
                           'huggingface', 'huggingface.co', 'hf.co',
                           'paperswithcode.com', 'openreview.net',
                           'aclanthology.org', 'distill.pub',
                           'jmlr.org', 'biorxiv.org', 'medrxiv.org',
                           'semanticscholar.org', 'scholar.google.com',
                           'neurips.cc', 'icml.cc'
                         ]::text[]
                      THEN 3
                    WHEN d.source IN ('github', 'github_repos')
                      THEN 1
                    ELSE 0
                END AS sci_score
              FROM documents d
             WHERE d.deleted = FALSE
               AND d.date IS NOT NULL
               AND d.date >= now() - interval '{int(window_days)} days'
        ),
        -- One row per anchor: pick the visually-richest representative.
        representative AS (
            SELECT DISTINCT ON (anchor_url)
                   anchor_url,
                   user_id, url, title, date, summary,
                   clean_title, clean_summary, urls, tags, source,
                   source_url, linked_urls, link_hosts, canonical_url,
                   sci_score
              FROM window_docs
             ORDER BY anchor_url,
                      image_count DESC,
                      url_count DESC,
                      date DESC,
                      created_at DESC
        ),
        -- Sharer rollup: every user whose doc maps to this anchor.
        anchor_sharers AS (
            SELECT w.anchor_url,
                   array_agg(DISTINCT w.user_id)                  AS sharer_user_ids,
                   bool_or(u.vip)                                 AS any_vip_sharer,
                   count(DISTINCT w.user_id)::int                 AS sharer_count,
                   jsonb_agg(DISTINCT jsonb_build_object(
                       'slug',             u.username,
                       'name',             u.name,
                       'avatar',           u.avatar,
                       'twitterFollowers', u.twitter_followers
                   ))                                             AS sharers,
                   -- Most notable sharer drives the followers bonus.
                   MAX(COALESCE(u.twitter_followers, 0))::bigint  AS top_followers
              FROM window_docs w
              JOIN users        u ON u.id = w.user_id
             GROUP BY w.anchor_url
        ),
        -- Final scoring. Mirrors the live-query formula minus the
        -- per-viewer terms (followee_share, fresh-self).
        scored AS (
            SELECT r.url, r.canonical_url, r.anchor_url,
                   r.title, r.date, r.summary, r.clean_title,
                   r.clean_summary, r.urls, r.tags, r.source,
                   r.source_url, r.linked_urls, r.link_hosts,
                   r.user_id                                       AS primary_user_id,
                   s.sharer_user_ids, s.sharers, s.sharer_count,
                   s.any_vip_sharer,
                   (
                       r.sci_score::float * 6
                     + CASE
                           WHEN r.date >= current_date - 7   THEN 5
                           WHEN r.date >= current_date - 14  THEN 4
                           WHEN r.date >= current_date - 21  THEN 3
                           WHEN r.date >= current_date - 28  THEN 2
                           WHEN r.date >= current_date - 35  THEN 1
                           ELSE 0
                       END
                     + LEAST(2, LN(GREATEST(1, s.sharer_count::float))) * 0.7
                     + CASE WHEN s.any_vip_sharer THEN 0.8 ELSE 0 END
                     + LEAST(
                         1.5,
                         LN(GREATEST(1, s.top_followers / 10000.0))
                       )
                     + CASE
                         WHEN r.source = 'twitter'
                              AND EXISTS (
                                  SELECT 1
                                    FROM jsonb_array_elements(r.linked_urls) e
                                   WHERE COALESCE(e->>'image', '') <> ''
                              )
                           THEN 1.5
                         ELSE 0
                       END
                   ) AS score
              FROM representative r
              JOIN anchor_sharers s ON s.anchor_url = r.anchor_url
        )
        ,
        -- Belt-and-braces dedup by URL. The DISTINCT ON in
        -- `representative` already keeps one row per anchor, but
        -- two anchors can share a representative URL when different
        -- user-docs of the same URL emit different `linked_urls`
        -- (and therefore different canonical_referenced_urls →
        -- different anchor_url). Without this step the INSERT trips
        -- the (url) PK. We keep the higher-scoring anchor.
        deduped AS (
            SELECT DISTINCT ON (url) *
              FROM scored
             ORDER BY url, score DESC, date DESC NULLS LAST
        )
        SELECT url, canonical_url, anchor_url,
               title, date, summary, clean_title, clean_summary, urls, tags,
               source, source_url, linked_urls, link_hosts,
               primary_user_id, sharer_user_ids, sharers, sharer_count,
               any_vip_sharer, score, now()
          FROM deduped
         ORDER BY score DESC, date DESC NULLS LAST
         LIMIT {int(max_rows)}
    """
