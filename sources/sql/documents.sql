-- Schema for the `documents` table.
--
-- One row per (user_id, url) pair. `user_id` references `users(id)` —
-- deleting a user cascades to their documents. `url` is the resource
-- itself. `source` classifies where we learned about it (twitter,
-- youtube, scholar, …); `source_url` is the specific mention URL
-- (e.g. the tweet that linked it). `tags` are user/curated; `extra_tags`
-- are generated.

CREATE TABLE IF NOT EXISTS documents (
    user_id       BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url           TEXT        NOT NULL,
    title         TEXT        NOT NULL DEFAULT '',
    summary       TEXT        NOT NULL DEFAULT '',
    -- Cleaner / normalised variants of `title` and `summary` produced
    -- by the cleaning stage (whose completion is tracked by the
    -- `cleaned` boolean below). Kept separately so the raw,
    -- as-scraped values stay available for re-runs / diffing.
    clean_title   TEXT        NOT NULL DEFAULT '',
    clean_summary TEXT        NOT NULL DEFAULT '',
    date          DATE,
    tags          TEXT[]      NOT NULL DEFAULT '{}',
    extra_tags    TEXT[]      NOT NULL DEFAULT '{}',
    source        TEXT        NOT NULL DEFAULT '',
    source_url    TEXT,
    -- Pipeline-stage gates. Each flag is flipped to TRUE by the
    -- worker that completes its stage, so rows inserted outside the
    -- pipeline (in-app "Save" button, browser-side /auth/me/documents
    -- bulk sync) are picked up the next time the pipeline runs.
    -- Conventions:
    --   • All four default FALSE so new rows queue for every stage.
    --   • Partial indexes below make "what still needs stage X?"
    --     queries O(1) regardless of total doc count.
    --   • Stages are independent — a failed one doesn't block the
    --     others. `indexed` depends on `tagged` in practice (the
    --     index text includes extra_tags) but nothing enforces it.
    cleaned       BOOLEAN     NOT NULL DEFAULT FALSE,  -- clean_title / clean_summary normalized the fields
    link_checked  BOOLEAN     NOT NULL DEFAULT FALSE,  -- dead-link probe saw a live URL
    tagged        BOOLEAN     NOT NULL DEFAULT FALSE,  -- extra_tags populated by the tagger
    indexed       BOOLEAN     NOT NULL DEFAULT FALSE,  -- ColBERT embeddings in the search index
    -- Soft-delete tombstone. Flipped to TRUE when the user removes the
    -- originating source from their profile (e.g. drops a website feed)
    -- so an offline job can later purge the row from PG and the ColBERT
    -- index. Re-adding the same source flips it back to FALSE so we
    -- don't double-fetch — the row simply rejoins the live set.
    to_delete     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, url)
);

-- Idempotent migrations for pre-stage-column DBs. Each flag back-fills
-- existing rows to TRUE on the assumption that the current pipeline
-- has already processed them — individual users can force a re-run by
-- updating the flag back to FALSE.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'indexed'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN indexed BOOLEAN NOT NULL DEFAULT FALSE;
        UPDATE documents SET indexed = TRUE;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'cleaned'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN cleaned BOOLEAN NOT NULL DEFAULT FALSE;
        UPDATE documents SET cleaned = TRUE;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'link_checked'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN link_checked BOOLEAN NOT NULL DEFAULT FALSE;
        UPDATE documents SET link_checked = TRUE;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'tagged'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN tagged BOOLEAN NOT NULL DEFAULT FALSE;
        -- A row counts as "tagged" if extra_tags was populated (the
        -- tagger is the only writer). Rows with an empty array are
        -- either freshly inserted or came from a source where the
        -- tagger hasn't run yet — in both cases leave them FALSE.
        UPDATE documents SET tagged = TRUE WHERE cardinality(extra_tags) > 0;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'to_delete'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN to_delete BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
    -- Audience flag. Default TRUE = visible to followers of the
    -- owner. A user can flip it to FALSE on compose to keep a doc
    -- private (visible only in their own library). The feed/timeline
    -- queries don't yet honour this column — that's a follow-up.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'public'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN public BOOLEAN NOT NULL DEFAULT TRUE;
    END IF;
    -- User-initiated soft delete. Set TRUE when the owner clicks the
    -- trash icon on a card; stays TRUE through subsequent pipeline
    -- runs (ON CONFLICT DO NOTHING in the bulk insert preserves the
    -- flag), so re-syncing a source doesn't resurrect the row. Read
    -- paths (timeline, sources, per-user documents) filter on
    -- `deleted = FALSE`. Distinct from `to_delete` which is the
    -- pipeline's own source-pruning tombstone.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'deleted'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN deleted BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
    -- created_via_favorite: TRUE iff the row was synthesised by an
    -- upvote from the feed (handlers::favorite_docs::add copies
    -- metadata from another library's row when the user has no
    -- existing documents row for the URL). The remove path uses this
    -- flag to decide whether un-upvoting should also delete the docs
    -- row — pure favorite-created rows go away on un-upvote, rows
    -- justified by a real sync stay put. Sync upserts (Python
    -- pipeline + the browser-side /auth/me/documents/bulk endpoint)
    -- clear the flag on conflict so a doc is promoted to "real"
    -- the moment any other source confirms it.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'created_via_favorite'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN created_via_favorite BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
    -- Inline link previews. Replaces the old "create a companion
    -- document per URL embedded in a tweet" pattern: a tweet with N
    -- external links now stays a single row whose `linked_urls`
    -- carries N entries shaped
    --   {url, host, title, summary, image}
    -- so the card renderer can show preview tiles inline. Capped at
    -- 5 entries per doc in the pipeline to keep payload bounded.
    -- `link_hosts` is the flat, GIN-indexed projection of the host
    -- field across every entry — the source-filter SQL does
    -- `link_hosts && ARRAY[...]` so a tweet linking hornet.dev shows
    -- under both the `twitter` chip AND the `hornet.dev` chip
    -- without a second row.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'linked_urls'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN linked_urls JSONB NOT NULL DEFAULT '[]'::jsonb;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'link_hosts'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN link_hosts TEXT[] NOT NULL DEFAULT '{}'::text[];
    END IF;
    -- Cleaned title / summary fields. The cleaning stage of the
    -- pipeline normalises the raw values (strips marketing
    -- suffixes, collapses whitespace, trims emoji noise on tweets,
    -- etc.) and writes the result here. Keeps the raw `title` /
    -- `summary` intact so the cleaner can be re-run with different
    -- rules without losing the original.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'clean_title'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN clean_title TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'clean_summary'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN clean_summary TEXT NOT NULL DEFAULT '';
    END IF;
    -- Flat URL list extracted from the raw `summary` text. The
    -- clean daemon rewrites bodies and the pedagogical prompt is
    -- free to drop labels like 'Paper:' / 'Project:' — but the
    -- frontend still needs to be able to surface every URL the
    -- original post referenced. `urls` is the authoritative
    -- record: a back-fill populates it for existing rows and the
    -- pipeline (or the cleaning daemon as a side-effect) keeps it
    -- in sync going forward. Duplicates the URLs already in
    -- `linked_urls.url` for the subset that got OG-previewed —
    -- carries the additional URLs (bare links, gist URLs, etc.)
    -- that the OG cluster missed.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'urls'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN urls TEXT[] NOT NULL DEFAULT '{}'::text[];
    END IF;

    -- `categorized` (added 2026-05-20). Twin of `cleaned`. Set TRUE by
    -- the categorize daemon after every doc it processes — whether or
    -- not the model produced a confident-enough match to write an
    -- assignment row. Critical: without this flag the daemon was
    -- infinite-looping over the same low-confidence docs because
    -- `NOT EXISTS (assignments)` re-fetched them every batch. The
    -- column gives us a separate "processed" signal that survives
    -- the (intentional) "no assignment when below threshold" path.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'categorized'
    ) THEN
        ALTER TABLE documents
            ADD COLUMN categorized BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;

    -- Behavioural / engagement columns (added 2026-05-22). One column
    -- per signal so the feed-ranking SQL stays index-friendly and the
    -- column meaning is unambiguous per source. NULL means "we never
    -- looked this up for this doc" (so a future backfill can target
    -- NULLs); zero is a real "we looked and the count is 0".
    --   • citation_count   — arXiv (via Semantic Scholar sidecar) and
    --                        scholar / semantic_scholar fetchers.
    --   • twitter_likes    — TwitterAPI.io `likeCount`, twikit
    --                        `favorite_count`. Summed across thread
    --                        parts so a 10-tweet thread carries the
    --                        thread-wide engagement.
    --   • twitter_retweets — `retweetCount` / `retweet_count`.
    --   • twitter_replies  — `replyCount` / `reply_count`.
    --   • twitter_quotes   — `quoteCount` / `quote_count`.
    --   • twitter_views    — `viewCount` / `view_count`. BIGINT
    --                        because popular tweets routinely cross
    --                        the 2.1B INT4 ceiling.
    --   • twitter_bookmarks — `bookmarkCount` / `bookmark_count`.
    --   • engagement_updated_at — when we last refreshed any of the
    --                        above. Drives the eventual "refresh
    --                        engagement" daemon that re-fetches stale
    --                        counts so a new tweet's likes climb in
    --                        the index as it goes viral.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'citation_count'
    ) THEN
        ALTER TABLE documents ADD COLUMN citation_count INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'twitter_likes'
    ) THEN
        ALTER TABLE documents ADD COLUMN twitter_likes INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'twitter_retweets'
    ) THEN
        ALTER TABLE documents ADD COLUMN twitter_retweets INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'twitter_replies'
    ) THEN
        ALTER TABLE documents ADD COLUMN twitter_replies INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'twitter_quotes'
    ) THEN
        ALTER TABLE documents ADD COLUMN twitter_quotes INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'twitter_views'
    ) THEN
        ALTER TABLE documents ADD COLUMN twitter_views BIGINT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'twitter_bookmarks'
    ) THEN
        ALTER TABLE documents ADD COLUMN twitter_bookmarks INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'documents' AND column_name = 'engagement_updated_at'
    ) THEN
        ALTER TABLE documents ADD COLUMN engagement_updated_at TIMESTAMPTZ;
    END IF;
END$$;

-- ────────────────────────────────────────────────────────────────────
-- canonicalize_url(text) — normalize a URL so cross-user share
-- aggregation can JOIN on logical-equivalent URLs without missing
-- variants. Same rules applied (in order) for every caller:
--
--   1. Strip fragment (#anchor)
--   2. Force https scheme
--   3. Lowercase host, strip leading "www."
--   4. Strip trailing slash from path (unless path == "/")
--   5. Drop tracking params: utm_*, fbclid, gclid, mc_eid, mc_cid,
--      ref, ref_src, ref_url, igshid, ncid, share_id, taid, bftwnews,
--      spm. Conservative blocklist — generic short keys ("s", "t") are
--      excluded because some sites use them as content IDs.
--   6. arxiv.org: pdf↔abs unification + strip version (vN)(.pdf)?,
--      drop query string (arxiv ignores it anyway).
--   7. youtu.be → youtube.com/watch?v=<id>
--
-- IMMUTABLE PARALLEL SAFE STRICT — required so it can drive the
-- STORED generated column below. Non-http schemes (mailto:, data:)
-- fall through to a lowercased copy of the input so the call sites
-- can be unconditional.
-- ────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION canonicalize_url(u TEXT)
RETURNS TEXT
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT
AS $$
DECLARE
  m       TEXT[];
  host    TEXT;
  path    TEXT;
  qs      TEXT;
BEGIN
  IF u IS NULL OR length(u) = 0 THEN
    RETURN '';
  END IF;
  u := regexp_replace(u, '#.*$', '');
  m := regexp_match(u, '^(https?)://([^/?#]+)([^?]*)(?:\?(.*))?$');
  IF m IS NULL THEN
    RETURN lower(u);
  END IF;
  host := lower(m[2]);
  IF host LIKE 'www.%' THEN host := substring(host FROM 5); END IF;
  path := COALESCE(m[3], '');
  qs   := COALESCE(m[4], '');
  IF length(path) > 1 AND right(path, 1) = '/' THEN
    path := left(path, length(path) - 1);
  END IF;
  IF qs <> '' THEN
    qs := regexp_replace(
      '&' || qs,
      '&(utm_[a-z_]+|fbclid|gclid|mc_eid|mc_cid|ref|ref_src|ref_url|igshid|ncid|share_id|taid|bftwnews|spm)=[^&]*',
      '',
      'gi'
    );
    qs := regexp_replace(qs, '^&+', '', '');
    qs := regexp_replace(qs, '&+', '&', 'g');
    qs := rtrim(qs, '&');
  END IF;
  IF host = 'arxiv.org' THEN
    m := regexp_match(path, '^/(?:pdf|abs)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$');
    IF m IS NOT NULL THEN
      path := '/abs/' || m[1];
      qs := '';
    END IF;
  END IF;
  IF host = 'youtu.be' THEN
    m := regexp_match(path, '^/([A-Za-z0-9_-]{6,})');
    IF m IS NOT NULL THEN
      host := 'youtube.com';
      path := '/watch';
      qs := 'v=' || m[1];
    END IF;
  END IF;
  RETURN 'https://' || host || path || CASE WHEN qs <> '' THEN '?' || qs ELSE '' END;
END;
$$;

-- compute_canonical_referenced_urls(url, urls, linked_urls)
--
-- Union of canonical forms of:
--   • the doc's own `url`
--   • every entry in `urls` (the cleaner's flat URL list)
--   • every `url` in the `linked_urls` JSONB (inline preview cards)
--
-- Filtered to entries that carry actual content: at least 13 chars,
-- not a bare host, and not on a noise blocklist (t.co, bit.ly). This
-- powers the LATERAL sharer expansion: a tweet linking paper X and a
-- blog linking paper X both end up in the same group as the arxiv
-- paper itself, so the avatar stack on any of them surfaces all
-- three sharers.
CREATE OR REPLACE FUNCTION compute_canonical_referenced_urls(
  doc_url   TEXT,
  doc_urls  TEXT[],
  doc_links JSONB
) RETURNS TEXT[]
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
  SELECT COALESCE(array_agg(DISTINCT canon), '{}'::text[])
    FROM (
      SELECT canonicalize_url(doc_url) AS canon
      UNION ALL
      SELECT canonicalize_url(u)
        FROM unnest(COALESCE(doc_urls, '{}'::text[])) u
      UNION ALL
      SELECT canonicalize_url(elem->>'url')
        FROM jsonb_array_elements(COALESCE(doc_links, '[]'::jsonb)) elem
       WHERE jsonb_typeof(elem) = 'object'
         AND COALESCE(elem->>'url', '') <> ''
    ) sub
   WHERE canon IS NOT NULL
     AND length(canon) > 12
     AND canon !~ '^https?://[^/]+/?$'
     AND canon NOT LIKE 'https://t.co/%'
     AND canon NOT LIKE 'https://bit.ly/%';
$$;

-- Canonical URL columns. STORED generated columns: PG computes them
-- at insert/update time, so app code never has to write canonical_url
-- — it falls out of `url`, `urls`, `linked_urls`. The first migration
-- on a populated table triggers a one-time rewrite (~10 min on
-- ~500k rows under ACCESS EXCLUSIVE lock; subsequent deploys are
-- a no-op because of IF NOT EXISTS).
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
     WHERE table_name = 'documents' AND column_name = 'canonical_url'
  ) THEN
    ALTER TABLE documents
      ADD COLUMN canonical_url TEXT
      GENERATED ALWAYS AS (canonicalize_url(url)) STORED;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
     WHERE table_name = 'documents' AND column_name = 'canonical_referenced_urls'
  ) THEN
    ALTER TABLE documents
      ADD COLUMN canonical_referenced_urls TEXT[]
      GENERATED ALWAYS AS (compute_canonical_referenced_urls(url, urls, linked_urls)) STORED;
  END IF;
END$$;

-- GIN index on `link_hosts` — the source-filter SQL does
-- `link_hosts && ARRAY[...]` so a chip click is a single index probe
-- regardless of how many rows reference that host.
CREATE INDEX IF NOT EXISTS idx_documents_link_hosts
    ON documents USING GIN (link_hosts);

-- Canonical URL indexes — drive the timeline's sharer aggregation.
-- The btree powers `d.canonical_url = m.canonical_url` (the strict
-- equality join in the LATERAL); the GIN powers the array-overlap
-- prong that picks up docs which merely *reference* this URL.
CREATE INDEX IF NOT EXISTS idx_documents_canonical_url
    ON documents (canonical_url) WHERE deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_documents_canonical_referenced_urls
    ON documents USING GIN (canonical_referenced_urls);

-- Index covering the anonymous /api/feed query
-- (`build_feed_payload`). Both halves of that query (the
-- DISTINCT ON for `latest_meta` and the GROUP BY for
-- `sharers_per_url`) filter on (`deleted = FALSE`, `date IS NOT
-- NULL`) and order/group by `url`. With (url, date DESC, user_id)
-- the planner switches both passes to Index Only Scan and the
-- external-merge sort that previously spilled 27 MB to disk goes
-- away — cold latency drops from ~9 s to ~1.6 s.
CREATE INDEX IF NOT EXISTS idx_documents_url_date_live
    ON documents (url, date DESC NULLS LAST, user_id)
 WHERE deleted = FALSE AND date IS NOT NULL;

-- Composite index on (user_id, source) — serves both per-user lookups
-- (leftmost prefix) AND the GROUP BY in the `user_source_counts` view,
-- which otherwise would need a full scan.
CREATE INDEX IF NOT EXISTS idx_documents_user_source ON documents (user_id, source);
-- Partial indexes: one per pipeline stage, so the "what still needs
-- stage X for this user?" queries stay instant even when the total
-- doc count grows large. Index only covers rows where the flag is
-- FALSE, so the index shrinks as the pipeline catches up.
CREATE INDEX IF NOT EXISTS idx_documents_user_uncleaned
    ON documents (user_id) WHERE cleaned = FALSE;
CREATE INDEX IF NOT EXISTS idx_documents_user_unchecked
    ON documents (user_id) WHERE link_checked = FALSE;
CREATE INDEX IF NOT EXISTS idx_documents_user_untagged
    ON documents (user_id) WHERE tagged = FALSE;
CREATE INDEX IF NOT EXISTS idx_documents_user_unindexed
    ON documents (user_id) WHERE indexed = FALSE;
-- Partial index on uncategorized rows, ordered by date DESC so the
-- categorize daemon's "newest-first" fetch stays index-only across
-- the full 400k+ doc corpus.
CREATE INDEX IF NOT EXISTS idx_documents_uncategorized_date
    ON documents (date DESC NULLS LAST, url DESC) WHERE categorized = FALSE;
-- Partial index on the to_delete tombstone — the offline purge job
-- needs to find these fast even when they're a tiny fraction of rows.
CREATE INDEX IF NOT EXISTS idx_documents_user_to_delete
    ON documents (user_id) WHERE to_delete = TRUE;
CREATE INDEX IF NOT EXISTS idx_documents_date        ON documents (date DESC NULLS LAST);
-- Per-user date-desc index: lets the feed timeline query stream the
-- newest rows per followee and early-terminate. Without it,
-- /api/timeline plans a full scan + sort across every doc of every
-- followee, which is ~1s once the corpus crosses a few thousand rows.
CREATE INDEX IF NOT EXISTS idx_documents_user_date
    ON documents (user_id, date DESC NULLS LAST);
-- Partial index — read paths filter `deleted = FALSE`, so the index
-- only covers live rows. Lets PG skip the column entirely for the
-- vast majority of queries.
CREATE INDEX IF NOT EXISTS idx_documents_user_date_live
    ON documents (user_id, date DESC NULLS LAST)
    WHERE deleted = FALSE;
-- Cross-user URL lookups: the timeline and any per-URL sharer
-- aggregation join `documents` on `url` alone (no user_id filter),
-- so the (user_id, url) primary key can't be used. The partial
-- index covers only live rows — same trick as `_user_date_live`.
CREATE INDEX IF NOT EXISTS idx_documents_url_live
    ON documents (url) WHERE deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_documents_tags        ON documents USING GIN (tags);

-- Per-user engagement-ranked indices. Partial on `deleted = FALSE`
-- because the timeline / popular-feed queries always filter live rows;
-- DESC NULLS LAST so docs we haven't fetched engagement for yet
-- naturally sink past the ones we have. `idx_documents_user_citations`
-- powers an "academic feed" ordering (citation_count DESC), the
-- twitter pair powers a "viral tweets" ordering.
CREATE INDEX IF NOT EXISTS idx_documents_user_citations
    ON documents (user_id, citation_count DESC NULLS LAST)
    WHERE deleted = FALSE AND citation_count IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_user_tw_likes
    ON documents (user_id, twitter_likes DESC NULLS LAST)
    WHERE deleted = FALSE AND twitter_likes IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_user_tw_views
    ON documents (user_id, twitter_views DESC NULLS LAST)
    WHERE deleted = FALSE AND twitter_views IS NOT NULL;

-- ── DB-level documentation (visible via \d+ and pg_description) ─────────
COMMENT ON TABLE documents IS
    'Bookmarks collected for each user. One row per (user_id, url). FK to users cascades on delete.';

COMMENT ON COLUMN documents.user_id    IS 'Owner. Foreign key → users(id). ON DELETE CASCADE.';
COMMENT ON COLUMN documents.url        IS 'Canonical URL of the resource. Part of the composite primary key.';
COMMENT ON COLUMN documents.title         IS 'Document title as displayed (raw, as-scraped).';
COMMENT ON COLUMN documents.summary       IS 'Short description / abstract shown on the card (raw, as-scraped).';
COMMENT ON COLUMN documents.clean_title   IS 'Normalised variant of `title` produced by the cleaning stage. Empty until the row is cleaned.';
COMMENT ON COLUMN documents.clean_summary IS 'Normalised variant of `summary` produced by the cleaning stage. Empty until the row is cleaned.';
COMMENT ON COLUMN documents.date       IS 'Publication date of the resource (not the ingestion date).';
COMMENT ON COLUMN documents.tags       IS 'User/curated tags. Indexed with GIN for overlap queries.';
COMMENT ON COLUMN documents.extra_tags IS 'Auto-generated tags (TF-IDF, model2vec) — kept separate from user tags.';
COMMENT ON COLUMN documents.source     IS 'Source type: twitter, youtube, scholar, github, hackernews, blog, … Empty string if unknown.';
COMMENT ON COLUMN documents.source_url IS 'URL of the specific mention that surfaced this resource (e.g. the tweet that linked it). Optional.';
COMMENT ON COLUMN documents.cleaned      IS 'TRUE once clean_title / clean_summary have normalised the title/summary fields.';
COMMENT ON COLUMN documents.link_checked IS 'TRUE once the dead-link probe saw a live URL. Dead URLs are deleted from the table, not marked.';
COMMENT ON COLUMN documents.tagged       IS 'TRUE once the tagger (LLM + flashtext) has populated extra_tags. Cheap to reset to force a retag.';
COMMENT ON COLUMN documents.indexed      IS 'TRUE once the row is in the ColBERT search index. The pipeline picks up FALSE rows and embeds them on the next run.';
COMMENT ON COLUMN documents.to_delete    IS 'Soft-delete tombstone. Flipped TRUE when the user removes the originating source; an offline job purges the row + its index entry. Re-adding the source flips it back FALSE so the doc rejoins without re-fetch.';
COMMENT ON COLUMN documents.created_at   IS 'Row creation (ingestion) timestamp.';
COMMENT ON COLUMN documents.updated_at   IS 'Last mutation timestamp. Updated by the application, not by a trigger.';

COMMENT ON COLUMN documents.citation_count        IS 'Citation count for academic papers. arXiv docs filled via Semantic Scholar sidecar; scholar/semantic_scholar fetchers fill directly. NULL = never looked up.';
COMMENT ON COLUMN documents.twitter_likes         IS 'Twitter like count (sum over thread parts). NULL = never fetched.';
COMMENT ON COLUMN documents.twitter_retweets      IS 'Twitter retweet count (sum over thread parts). NULL = never fetched.';
COMMENT ON COLUMN documents.twitter_replies       IS 'Twitter reply count (sum over thread parts). NULL = never fetched.';
COMMENT ON COLUMN documents.twitter_quotes        IS 'Twitter quote-tweet count (sum over thread parts). NULL = never fetched.';
COMMENT ON COLUMN documents.twitter_views         IS 'Twitter impression / view count (sum over thread parts). BIGINT — popular tweets cross 2.1B.';
COMMENT ON COLUMN documents.twitter_bookmarks     IS 'Twitter bookmark count (sum over thread parts). NULL = never fetched.';
COMMENT ON COLUMN documents.engagement_updated_at IS 'When any of the engagement columns above was last refreshed. Drives the eventual stale-engagement re-fetch daemon.';
