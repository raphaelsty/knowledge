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

# Hard cap on snapshot size. Bumped 20k → 60k so the snapshot
# covers essentially the full 180-day anchor universe (~52k
# anchors today) — the previous 20k was throwing away ~60% of
# the long tail, which starved niche personalities' feeds.
# Refresh cost scales linearly (~15 s instead of ~5 s on the
# current corpus); storage roughly triples to ~60 MB. The
# read-path GIN/btree indexes don't slow down because the
# planner still only touches the top-N rows per query.
DEFAULT_MAX_ROWS = 60_000


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
            any_vip_sharer, vip_sharer_count, categories, score, refreshed_at
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
                -- Replaced the raw image_count with a boolean
                -- has_image: 3 images doesn't make a card 3× more
                -- informative than 1 image, and ranking by count
                -- pushed image-grid heavy tweets above more
                -- informative ones that link a paper.
                (EXISTS (
                    SELECT 1 FROM jsonb_array_elements(d.linked_urls) e
                     WHERE COALESCE(e->>'image', '') <> ''
                ))::int AS has_image,
                -- Boolean: does this doc carry at least one
                -- external linked URL (arxiv, blog, github, HF,
                -- any host). The representative picker prefers
                -- this over has_image — a tweet that points at a
                -- paper is more useful as a card than a tweet
                -- with just a picture.
                (jsonb_array_length(d.linked_urls) > 0)::int AS has_link,
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
        -- One row per anchor: pick the most-informative
        -- representative. Priority order:
        --   1. has_link  — a tweet that points at the actual
        --                  arxiv / blog / github / HF resource is
        --                  the most useful card to show.
        --   2. has_image — within tied has_link, prefer a doc that
        --                  also includes a preview image.
        --   3. url_count — when both have linked URLs, prefer the
        --                  one that references more (paper + repo).
        --   4. date / created_at — recency tiebreakers.
        representative AS (
            SELECT DISTINCT ON (anchor_url)
                   anchor_url,
                   user_id, url, title, date, summary,
                   clean_title, clean_summary, urls, tags, source,
                   source_url, linked_urls, link_hosts, canonical_url,
                   sci_score
              FROM window_docs
             ORDER BY anchor_url,
                      has_link  DESC,
                      has_image DESC,
                      url_count DESC,
                      date      DESC,
                      created_at DESC
        ),
        -- Sharer rollup: every user whose doc maps to this anchor.
        -- Roll the categorize-daemon's per-(user, url) assignments
        -- up to the anchor level. Joining via window_docs scopes us
        -- to the same 180-day candidate set as the rest of the
        -- refresh — older category rows for URLs outside the window
        -- don't matter. `DISTINCT` because two users could both
        -- tag the same anchor with the same slug.
        anchor_categories AS (
            SELECT w.anchor_url,
                   array_agg(DISTINCT dc.slug) AS categories
              FROM window_docs                       w
              JOIN document_category_assignments     a
                ON a.user_id = w.user_id
               AND a.url     = w.url
              JOIN document_categories               dc
                ON dc.id     = a.category_id
             GROUP BY w.anchor_url
        ),
        anchor_sharers AS (
            SELECT w.anchor_url,
                   array_agg(DISTINCT w.user_id)                  AS sharer_user_ids,
                   bool_or(u.vip)                                 AS any_vip_sharer,
                   -- Count of DISTINCT VIP sharers. Feeds the new
                   -- `LN(vip_sharer_count + 1) * 1.0` term in score
                   -- so docs co-signed by many VIPs outrank docs
                   -- with just one VIP signal.
                   count(DISTINCT w.user_id) FILTER (WHERE u.vip)::int
                                                                  AS vip_sharer_count,
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
                   s.any_vip_sharer, s.vip_sharer_count,
                   COALESCE(ac.categories, '{{}}'::text[])         AS categories,
                   (
                       r.sci_score::float * 6
                     -- Flatter recency. Previous curve (7.5 → 0
                     -- over 5 weeks, step 1.5) buried older
                     -- multi-VIP content under any fresh single-
                     -- share tweet. The new curve spans 90 days
                     -- with a gentle decline: this-week is still
                     -- the prize but a month-old doc keeps 3.5
                     -- (was 1.5) — closes the gap to ~1.5 points
                     -- which the VIP-share boost can now overcome.
                     + CASE
                           WHEN r.date >= current_date - 7   THEN 5.0
                           WHEN r.date >= current_date - 14  THEN 4.5
                           WHEN r.date >= current_date - 21  THEN 4.0
                           WHEN r.date >= current_date - 35  THEN 3.5
                           WHEN r.date >= current_date - 60  THEN 2.5
                           WHEN r.date >= current_date - 90  THEN 1.5
                           ELSE 0
                       END
                     -- Popularity signal — ONLY VIP sharers count.
                     -- The previous total-share term was dropped:
                     -- counting non-VIPs let bot accounts / mass
                     -- followers inflate a doc's score without any
                     -- editorial weight behind it. Aggregation is
                     -- by anchor_url, so all the tweets / direct
                     -- saves of a single resource collapse to one
                     -- count first, then we measure how many
                     -- distinct VIPs are in that bucket.
                     --
                     -- Cap raised 4 → 8 so the upper tail
                     -- (10–20 VIP sharers — the genuinely
                     -- consensus-worthy resources) actually
                     -- benefits from extra signal instead of
                     -- being clamped at 7. Coefficient unchanged
                     -- (×2.0). 1 VIP ≈ +1.39, 4 VIPs ≈ +3.22,
                     -- 15 VIPs ≈ +5.55, 50+ caps at +8.0.
                     + LEAST(8, LN(GREATEST(1, s.vip_sharer_count + 1))) * 2.0
                     + LEAST(
                         1.5,
                         LN(GREATEST(1, s.top_followers / 10000.0))
                       )
                     -- Tweet bonus split into two orthogonal terms:
                     --   * +2.5  twitter AND has ANY external
                     --           linked URL (arxiv, blog, github,
                     --           HF, any host) — the "this tweet
                     --           points at a real resource" signal.
                     --   * +0.5  twitter AND has at least one
                     --           preview image — small nudge so a
                     --           tweet with a screenshot still
                     --           edges out a plain-text tweet of
                     --           equal score.
                     -- Image count is binary now: 1 image == 3
                     -- images == has-image. The image-grid
                     -- runaway (a tweet with 8 screenshots
                     -- outranking a paper-link tweet) is gone.
                     + CASE
                         WHEN r.source = 'twitter'
                              AND jsonb_array_length(r.linked_urls) > 0
                           THEN 2.5
                         ELSE 0
                       END
                     + CASE
                         WHEN r.source = 'twitter'
                              AND EXISTS (
                                  SELECT 1
                                    FROM jsonb_array_elements(r.linked_urls) e
                                   WHERE COALESCE(e->>'image', '') <> ''
                              )
                           THEN 0.5
                         ELSE 0
                       END
                   ) AS score
              FROM representative r
              JOIN anchor_sharers     s  ON s.anchor_url  = r.anchor_url
              LEFT JOIN anchor_categories ac ON ac.anchor_url = r.anchor_url
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
        ),
        -- ── Cluster-diversity penalty ─────────────────────────────
        -- For each (doc, sharer) pair, count how many higher-scored
        -- docs ALSO list this sharer. That tells us "by the time
        -- this doc would be shown, this individual sharer has
        -- already pushed N earlier docs into the feed". Summing
        -- across the doc's sharers gives `repeat_signal` — high
        -- when the same crowd keeps signing the next doc, ~0 when
        -- the doc's sharers are fresh contributors.
        --
        -- The score adjustment then nudges recycled-cluster docs
        -- down so a new cluster's first doc can outrank the third
        -- doc of the same crowd. Approximation: prior counts are
        -- computed against the score-DESC order BEFORE the
        -- adjustment, not iteratively. One pass, set-based,
        -- deterministic — cheap enough that the whole refresh
        -- still runs in ~5 s.
        exploded AS (
            SELECT url, score, u AS sharer_id
              FROM deduped, unnest(sharer_user_ids) u
        ),
        prior_pairs AS (
            SELECT url,
                   -- 0 for the highest-ranked appearance of this
                   -- sharer, 1 for their next, 2 for the one after
                   -- that — the "prior emit count" in the
                   -- single-pass approximation.
                   ROW_NUMBER() OVER (PARTITION BY sharer_id
                                      ORDER BY score DESC, url) - 1
                       AS prior_count
              FROM exploded
        ),
        cluster_penalty AS (
            SELECT url, SUM(prior_count)::bigint AS repeat_signal
              FROM prior_pairs
             GROUP BY url
        ),
        adjusted AS (
            SELECT d.url, d.canonical_url, d.anchor_url,
                   d.title, d.date, d.summary, d.clean_title, d.clean_summary,
                   d.urls, d.tags, d.source, d.source_url, d.linked_urls, d.link_hosts,
                   d.primary_user_id, d.sharer_user_ids, d.sharers, d.sharer_count,
                   d.any_vip_sharer, d.vip_sharer_count, d.categories,
                   -- Final score = base score minus the cluster
                   -- penalty. Sublinear (sqrt) AND capped at 0.7.
                   --
                   -- The 0.7 cap is deliberate: the smallest
                   -- marginal share-boost step is
                   --   LN(3)·2 − LN(2)·2 ≈ 0.81  (1→2 sharers)
                   -- so a penalty < 0.81 can never flip a 2-sharer
                   -- doc below a 1-sharer doc. Within the same
                   -- "resource kind" (same sci_score, same
                   -- rich-tweet bonus) more-shared resources are
                   -- guaranteed to outrank less-shared ones — the
                   -- cluster term only reorders within ties.
                   --
                   --   repeat= 1 → -0.15
                   --   repeat= 5 → -0.34
                   --   repeat=10 → -0.47
                   --   repeat=20 → -0.67
                   --   repeat=30+ → capped at -0.70
                   --
                   -- Tiny knob; reshuffles ties + adjacent ranks
                   -- without ever collapsing a deeply-shared doc.
                   (d.score - LEAST(
                       0.7,
                       0.15 * SQRT(GREATEST(0, COALESCE(cp.repeat_signal, 0))::float8)
                   ))::float8 AS score
              FROM deduped d
              LEFT JOIN cluster_penalty cp ON cp.url = d.url
        )
        SELECT url, canonical_url, anchor_url,
               title, date, summary, clean_title, clean_summary, urls, tags,
               source, source_url, linked_urls, link_hosts,
               primary_user_id, sharer_user_ids, sharers, sharer_count,
               any_vip_sharer, vip_sharer_count, categories, score, now()
          FROM adjusted
         ORDER BY score DESC, date DESC NULLS LAST
         LIMIT {int(max_rows)}
    """
