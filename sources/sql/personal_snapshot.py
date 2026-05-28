"""Helpers for the `personal_snapshot` table.

A per-VIP twin of `feed_snapshot`. The shape is identical (anchor
collapse, sci/recency/link/image bonus) but scoped to a single user's
own documents — no cross-user sharer roll-up, no cluster diversity
penalty. The point is to lift the "resource posts" (arxiv, HF, github,
tweets-with-paper) of a VIP's personal page above their plain-text
tweets, mirroring how the global feed presents them.

Two entry points:

* `create_personal_snapshot_table` — idempotent schema bootstrap,
  invoked from `run.py` (and the API boot migration).
* `refresh_personal_snapshot(database_url, user_id)` — DELETE+INSERT
  the rows for one user inside a single transaction. Called by the
  indexer daemon after a VIP's per-user index is rebuilt, and by the
  sweep daemon for VIPs whose snapshot is stale.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "personal_snapshot.sql"


def create_personal_snapshot_table(database_url: str) -> None:
    """Apply `personal_snapshot.sql` against the DB. Idempotent."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


# Window matches the feed snapshot — 180 days covers the long tail
# while keeping refresh cost bounded for the biggest VIPs (~10k docs).
DEFAULT_WINDOW_DAYS = 180


def refresh_personal_snapshot(
    database_url: str,
    user_id: int,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> int:
    """Rebuild `personal_snapshot` for one VIP. Returns rows written.

    Wraps DELETE + INSERT in a single transaction so readers see
    either the previous snapshot in full or the new one in full.
    The DELETE is bounded to `WHERE user_id = $1` — no global lock.
    """
    # `user_id` is interpolated directly (it's a typed int — no
    # injection risk) so psycopg won't try to format the LIKE
    # patterns (`%'arxiv...'`) as parameter placeholders.
    insert_sql = _build_refresh_sql(window_days=window_days, user_id=int(user_id))
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM personal_snapshot WHERE user_id = %s", (user_id,))
            cur.execute(insert_sql)
            written = cur.rowcount or 0
        conn.commit()
    return written


def personal_snapshot_age_seconds(database_url: str, user_id: int) -> int | None:
    """Seconds since the last refresh for this user. `None` if empty."""
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT EXTRACT(EPOCH FROM (now() - MAX(refreshed_at)))::bigint  FROM personal_snapshot WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


# ── SQL builder ──────────────────────────────────────────────────────


def _build_refresh_sql(window_days: int, user_id: int) -> str:
    """The per-VIP scoring CTE. `user_id` is interpolated as a literal
    int (typed, not bound) so the surrounding LIKE patterns don't
    confuse psycopg's `%` placeholder detection.
    """
    # Mirrors `feed_snapshot.py::_build_refresh_sql` minus:
    #   * sharer aggregation (single owner per anchor here)
    #   * VIP-share term and follower bonus (constant per user)
    #   * cluster-diversity penalty (only one sharer, would apply uniformly)
    # The recency curve and link/image bonus are copied verbatim so
    # the personal page reads the same way the feed does — a recent
    # tweet linking arxiv still beats a 6-month-old plain-text tweet.
    return f"""
        INSERT INTO personal_snapshot (
            user_id, url, canonical_url, anchor_url,
            title, date, summary, clean_title, clean_summary,
            urls, tags, extra_tags, source, source_url,
            linked_urls, link_hosts, categories,
            anchor_doc_count, indexed,
            sharer_user_ids, sharers, sharer_count,
            vip_sharer_count,
            score, refreshed_at
        )
        WITH window_docs AS (
            -- Every non-deleted doc owned by this user in the window.
            -- Same anchor logic as feed_snapshot: prefer arxiv > hf >
            -- github > openreview > doi > ... as the resource identity.
            SELECT
                d.user_id, d.url, d.title, d.date, d.summary,
                d.clean_title, d.clean_summary, d.urls, d.tags,
                d.extra_tags, d.source, d.source_url, d.created_at,
                d.linked_urls, d.link_hosts, d.canonical_url,
                d.canonical_referenced_urls, d.indexed,
                -- Behavioural counts for the engagement term below,
                -- mirroring feed_snapshot. NULL (un-backfilled) → 0.
                d.twitter_likes, d.twitter_retweets,
                d.twitter_replies, d.twitter_quotes,
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
                (EXISTS (
                    SELECT 1 FROM jsonb_array_elements(d.linked_urls) e
                     WHERE COALESCE(e->>'image', '') <> ''
                ))::int AS has_image,
                (jsonb_array_length(d.linked_urls) > 0)::int AS has_link,
                cardinality(d.canonical_referenced_urls) AS url_count,
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
                    -- github == huggingface weight (3). The popular-
                    -- repo case is what the VIP-share boost below is
                    -- meant to surface; a 1-sharer obscure runbook
                    -- only earns ~1.4 of the boost, a widely-saved
                    -- repo earns several points, so the share signal
                    -- does the disambiguation rather than a flat
                    -- per-source penalty.
                    WHEN d.source IN ('github', 'github_repos')
                      THEN 3
                    ELSE 0
                END AS sci_score
              FROM documents d
             WHERE d.deleted = FALSE
               AND d.user_id = {int(user_id)}
               AND d.date IS NOT NULL
               -- No rolling window here: a personal page should show
               -- the user's full library, not just a 180-day slice.
               -- Ilya Sutskever has 1.5k docs but only 5 in the last
               -- 180 days — his page would otherwise look broken.
               -- The score still favours recent content via the
               -- recency tier + age-damped sci_score, so old papers
               -- naturally cluster at the bottom rather than vanish.
        ),
        representative AS (
            SELECT DISTINCT ON (anchor_url)
                   anchor_url,
                   user_id, url, title, date, summary,
                   clean_title, clean_summary, urls, tags, extra_tags,
                   source, source_url, linked_urls, link_hosts,
                   canonical_url, sci_score, has_link, has_image,
                   url_count, created_at, indexed
              FROM window_docs
             ORDER BY anchor_url,
                      has_link  DESC,
                      has_image DESC,
                      url_count DESC,
                      date      DESC,
                      created_at DESC
        ),
        -- Per-anchor aggregates within this user's own corpus.
        anchor_rollup AS (
            SELECT w.anchor_url,
                   count(*)::int       AS anchor_doc_count,
                   bool_and(w.indexed) AS all_indexed
              FROM window_docs w
             GROUP BY w.anchor_url
        ),
        -- Categories assigned to any of the user's docs in the anchor
        -- group, unioned. Scoped to this user's assignments — keeps
        -- the per-user page faithful to how the user actually filed
        -- their library.
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
        -- Per-anchor behavioural roll-up over THIS user's own docs.
        -- MAX (not SUM) for the same reason as feed_snapshot: a
        -- resource the owner tweeted + retweeted shouldn't double
        -- count. On a personal page the owner's own tweet engagement
        -- is the honest "how much did this land" signal.
        anchor_engagement AS (
            SELECT anchor_url,
                   MAX(COALESCE(twitter_likes,    0)) AS max_likes,
                   MAX(COALESCE(twitter_retweets, 0)) AS max_retweets,
                   MAX(COALESCE(twitter_replies,  0)) AS max_replies,
                   MAX(COALESCE(twitter_quotes,   0)) AS max_quotes
              FROM window_docs
             GROUP BY anchor_url
        ),
        scored AS (
            SELECT r.user_id,
                   -- When the anchor IS in feed_snapshot, surface
                   -- feed_snapshot's representative metadata so the
                   -- personal page renders the SAME card the feed
                   -- shows (with all the aggregated linked URLs +
                   -- the rich summary the global crowd contributed).
                   -- Falls back to the user's own doc when the
                   -- anchor is long-tail / outside feed_snapshot.
                   COALESCE(fs.url,           r.url)           AS url,
                   COALESCE(fs.canonical_url, r.canonical_url) AS canonical_url,
                   r.anchor_url,
                   COALESCE(NULLIF(fs.title, ''),         r.title)         AS title,
                   COALESCE(fs.date,                      r.date)          AS date,
                   COALESCE(NULLIF(fs.summary, ''),       r.summary)       AS summary,
                   COALESCE(NULLIF(fs.clean_title, ''),   r.clean_title)   AS clean_title,
                   COALESCE(NULLIF(fs.clean_summary, ''), r.clean_summary) AS clean_summary,
                   r.urls, r.tags, r.extra_tags,
                   COALESCE(NULLIF(fs.source, ''),  r.source)               AS source,
                   COALESCE(fs.source_url,          r.source_url)           AS source_url,
                   -- Cap linked_urls: dedup by host, at most 3 entries.
                   -- `fs.linked_urls` (from feed_snapshot) is already
                   -- capped, so this is idempotent on the fs path; the
                   -- fallback `r.linked_urls` path needs the cap too
                   -- (anchors without a feed_snapshot row land here).
                   -- See feed_snapshot.py for the rationale.
                   COALESCE((
                       SELECT jsonb_agg(e ORDER BY rn)
                         FROM (
                             SELECT e, rn
                               FROM (
                                   SELECT e, rn,
                                          row_number() OVER (
                                              PARTITION BY COALESCE(
                                                  NULLIF(e->>'host',''),
                                                  e->>'url'
                                              )
                                              ORDER BY (
                                                  CASE WHEN COALESCE(e->>'image','') <> ''
                                                       THEN 0 ELSE 1 END
                                              ), rn
                                          ) AS host_rank
                                     FROM (
                                         SELECT e,
                                                row_number() OVER () AS rn
                                           FROM jsonb_array_elements(
                                               COALESCE(fs.linked_urls, r.linked_urls, '[]'::jsonb)
                                           ) e
                                     ) numbered
                               ) ranked
                              WHERE host_rank = 1
                              ORDER BY rn
                              LIMIT 3
                         ) capped
                   ), '[]'::jsonb)                                          AS linked_urls,
                   COALESCE(NULLIF(fs.link_hosts, '{{}}'::text[]),
                            r.link_hosts)                                   AS link_hosts,
                   COALESCE(ac.categories, '{{}}'::text[])         AS categories,
                   ar.anchor_doc_count,
                   ar.all_indexed                                  AS indexed,
                   -- Cross-personality sharer roll-up — read from the
                   -- already-computed global feed_snapshot so the
                   -- personal-page score reflects breadth even
                   -- though we're scoping rows to one owner.
                   COALESCE(fs.sharer_user_ids,
                            ARRAY[r.user_id]::bigint[])             AS sharer_user_ids,
                   COALESCE(fs.sharers,
                            jsonb_build_array(jsonb_build_object(
                                'slug',             u.username,
                                'name',             u.name,
                                'avatar',           u.avatar,
                                'twitterFollowers', u.twitter_followers
                            )))                                     AS sharers,
                   COALESCE(fs.sharer_count, 1)                     AS sharer_count,
                   COALESCE(fs.vip_sharer_count,
                            CASE WHEN u.vip THEN 1 ELSE 0 END)      AS vip_sharer_count,
                   (
                       -- sci_score×6 scaled by THREE factors for bare
                       -- sources (arxiv/scholar/HF/github):
                       --   * age_factor — old papers decay
                       --   * substance  — title-only / empty entries
                       --                  earn less
                       --   * share_factor — a 1-VIP "saved by the
                       --                  owner only" HF resource
                       --                  earns 30% of the baseline;
                       --                  3+ co-saving VIPs earn full
                       --                  credit. Together with the
                       --                  doubled VIP-share boost
                       --                  below, a 3-VIP-shared
                       --                  recent tweet now out-ranks
                       --                  a 1-VIP HF on the personal
                       --                  page, matching the user's
                       --                  intent that broad sharing
                       --                  beats lonely posts.
                       -- Multiplier trimmed 6 → 5 to match
                       -- feed_snapshot: less flat academic weight so
                       -- broadly-shared resources can outrank a lone
                       -- fresh arxiv link.
                       r.sci_score::float * 5 *
                       -- Age damping applies to ALL sources, not just
                       -- bare papers. Without this, a 7-year-old
                       -- Jan Leike tweet linking arxiv kept the full
                       -- ×18 sci bonus and out-ranked recent posts.
                       CASE
                           WHEN GREATEST(r.date, fs.date) >= current_date - 30  THEN 1.0
                           WHEN GREATEST(r.date, fs.date) >= current_date - 90  THEN 0.6
                           WHEN GREATEST(r.date, fs.date) >= current_date - 180 THEN 0.3
                           WHEN GREATEST(r.date, fs.date) >= current_date - 365 THEN 0.15
                           ELSE                                   0.05
                       END *
                       CASE
                           WHEN r.source NOT IN ('arxiv','scholar','huggingface','github','github_repos')
                               THEN 1.0
                           ELSE LEAST(
                               1.0,
                               GREATEST(
                                   0.2,
                                   length(COALESCE(r.summary, ''))::float / 120.0
                               )
                           )
                       END *
                       CASE
                           WHEN r.source NOT IN ('arxiv','scholar','huggingface','github','github_repos')
                               THEN 1.0
                           WHEN COALESCE(fs.vip_sharer_count, 0) >= 3 THEN 1.0
                           WHEN COALESCE(fs.vip_sharer_count, 0) = 2  THEN 0.7
                           WHEN COALESCE(fs.vip_sharer_count, 0) = 1  THEN 0.3
                           ELSE                                          0.1
                       END
                     -- Recency tier — peak bumped 8 → 12 to match
                     -- feed_snapshot. Personal pages now lean strongly
                     -- fresh too: the "this week" premium is on the
                     -- order of a full sci-bonus.
                     + CASE
                           WHEN GREATEST(r.date, fs.date) >= current_date - 7   THEN 12.0
                           WHEN GREATEST(r.date, fs.date) >= current_date - 14  THEN 9.5
                           WHEN GREATEST(r.date, fs.date) >= current_date - 21  THEN 7.0
                           WHEN GREATEST(r.date, fs.date) >= current_date - 35  THEN 5.0
                           WHEN GREATEST(r.date, fs.date) >= current_date - 60  THEN 2.5
                           WHEN GREATEST(r.date, fs.date) >= current_date - 90  THEN 1.0
                           ELSE 0
                       END
                     -- Tweet-with-resource bonus.
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
                     -- Resource-consensus bonus (mirrors feed_snapshot).
                     -- A non-paper resource (sci_score 0) that many
                     -- VIPs co-signed earns a consensus-scaled lift so
                     -- launches / products / blogs aren't buried under
                     -- the academic sci cliff. Reads cross-personality
                     -- vip_sharer_count from feed_snapshot.
                     + CASE
                         WHEN r.source = 'twitter'
                              AND r.sci_score = 0
                              AND jsonb_array_length(r.linked_urls) > 0
                           THEN LEAST(9.0, LN(GREATEST(1,
                                    COALESCE(fs.vip_sharer_count, 0) + 1)) * 3.0)
                         ELSE 0
                       END
                     -- Content-quality bonus. Long substantive tweets
                     -- (Karpathy threads, multi-paragraph writeups)
                     -- earn up to +2; title-only scholar entries earn
                     -- 0. Symmetric across sources so a paper with a
                     -- real abstract benefits too.
                     + LEAST(
                         2.0,
                         GREATEST(0.0,
                             length(COALESCE(r.summary, ''))::float - 60.0
                         ) / 300.0
                       )
                     -- Behavioural engagement — identical shape to
                     -- feed_snapshot ("heavily upvoted by plenty of
                     -- people"), kept secondary to the VIP-share boost
                     -- below. likes + retweets + 1.5·replies + quotes,
                     -- log-scaled /150, ×1.3, capped +6.0. NULL→0.
                     + LEAST(6.0, LN(1 + (
                           COALESCE(eng.max_likes,    0)
                         + COALESCE(eng.max_retweets, 0) * 1.0
                         + COALESCE(eng.max_replies,  0) * 1.5
                         + COALESCE(eng.max_quotes,   0) * 1.0
                       ) / 150.0) * 1.3)
                     -- Cross-personality VIP-share boost. Weighted
                     -- ×4 (vs the global feed's ×2) so the share
                     -- signal does the heavy lifting on personal
                     -- pages where most rows are 1-VIP by default.
                     -- Cap raised to 16 so a megastar paper
                     -- (50+ VIPs) still earns the full ceiling.
                     -- 1 VIP earns ~+2.8, 3 VIPs ~+5.5, 12 VIPs
                     -- ~+10, 50 caps at 16.
                     + LEAST(
                         16,
                         LN(GREATEST(1, COALESCE(fs.vip_sharer_count, 0) + 1))
                       ) * 4.0
                     -- Followers term — the most-notable sharer drives
                     -- the bonus. Capped at +1.5 so a megastar saving
                     -- something doesn't dominate (the VIP-share log
                     -- term is the primary breadth signal).
                     + LEAST(
                         1.5,
                         LN(GREATEST(1,
                             (SELECT MAX(COALESCE(u2.twitter_followers, 0))
                                FROM users u2
                               WHERE u2.id = ANY(
                                   COALESCE(fs.sharer_user_ids,
                                            ARRAY[r.user_id]::bigint[])
                               )) / 10000.0
                         ))
                       )
                   )
                   -- Hard total-score age multiplier, SOFTENED by
                   -- consensus (mirrors feed_snapshot #5). The raw
                   -- curve is steep (≤14d full … >2y 1.5%) so the page
                   -- leans fresh, but a resource dozens of VIPs saved
                   -- is canonical, not stale — blend toward 1.0 by
                   -- LEAST(0.5, vip_sharer_count/40): a 1-VIP doc is
                   -- unchanged, a 20-VIP resource keeps halfway to full
                   -- credit regardless of age. Cap 0.5 → consensus can
                   -- slow age decay but never switch it off.
                   * (
                       CASE
                           WHEN GREATEST(r.date, fs.date) >= current_date - 14   THEN 1.0
                           WHEN GREATEST(r.date, fs.date) >= current_date - 30   THEN 0.80
                           WHEN GREATEST(r.date, fs.date) >= current_date - 60   THEN 0.55
                           WHEN GREATEST(r.date, fs.date) >= current_date - 90   THEN 0.35
                           WHEN GREATEST(r.date, fs.date) >= current_date - 180  THEN 0.18
                           WHEN GREATEST(r.date, fs.date) >= current_date - 365  THEN 0.08
                           WHEN GREATEST(r.date, fs.date) >= current_date - 730  THEN 0.03
                           ELSE                                                    0.015
                       END
                       + (1.0 - CASE
                           WHEN GREATEST(r.date, fs.date) >= current_date - 14   THEN 1.0
                           WHEN GREATEST(r.date, fs.date) >= current_date - 30   THEN 0.80
                           WHEN GREATEST(r.date, fs.date) >= current_date - 60   THEN 0.55
                           WHEN GREATEST(r.date, fs.date) >= current_date - 90   THEN 0.35
                           WHEN GREATEST(r.date, fs.date) >= current_date - 180  THEN 0.18
                           WHEN GREATEST(r.date, fs.date) >= current_date - 365  THEN 0.08
                           WHEN GREATEST(r.date, fs.date) >= current_date - 730  THEN 0.03
                           ELSE                                                    0.015
                         END)
                         * LEAST(0.5, COALESCE(fs.vip_sharer_count, 0) / 40.0)
                     )
                                                                AS score
              FROM representative r
              JOIN users         u  ON u.id = r.user_id
              JOIN anchor_rollup ar ON ar.anchor_url = r.anchor_url
              LEFT JOIN anchor_categories ac  ON ac.anchor_url  = r.anchor_url
              LEFT JOIN anchor_engagement eng ON eng.anchor_url = r.anchor_url
              LEFT JOIN feed_snapshot fs      ON fs.anchor_url  = r.anchor_url
        ),
        -- Belt-and-braces dedup on (user_id, url). Mirrors the
        -- feed_snapshot's dedup step for the same reason: two
        -- anchors can occasionally share a representative URL.
        deduped AS (
            SELECT DISTINCT ON (user_id, url) *
              FROM scored
             ORDER BY user_id, url, score DESC, date DESC NULLS LAST
        )
        SELECT d.user_id, d.url, d.canonical_url, d.anchor_url,
               d.title, d.date, d.summary, d.clean_title, d.clean_summary,
               d.urls, d.tags, d.extra_tags, d.source, d.source_url,
               d.linked_urls, d.link_hosts, d.categories,
               d.anchor_doc_count, d.indexed,
               d.sharer_user_ids, d.sharers, d.sharer_count,
               d.vip_sharer_count,
               d.score, now()
          FROM deduped d
         ORDER BY d.score DESC, d.date DESC NULLS LAST
    """
