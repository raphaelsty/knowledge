-- Pre-computed feed table.
--
-- One row per "resource" (anchor URL) — a paper, a repo, a model,
-- a blog post. Two tweets that both link the same arxiv paper, the
-- arxiv abs page itself, and a /pdf/X.v2 variant all collapse to a
-- single row. The visually-richest doc (most preview images, then
-- most referenced URLs) is the representative.
--
-- The score column is viewer-agnostic: it folds sci-bonus,
-- weekly-bucketed recency over a 180-day window, log-of-total-share,
-- VIP-primary, log-of-followers, and a rich-tweet bonus. The
-- per-viewer additions (followee-share, fresh-self) live in the
-- handler and ride on top at read time. Reading is then a single
-- indexed scan: `WHERE sharer_user_ids && followees ORDER BY score
-- DESC LIMIT N` — typically <100 ms even with 50 k rows.
--
-- Refreshed atomically (TRUNCATE+INSERT in one txn) by the
-- `knowledge-feed-snapshot` daemon every hour. The handler checks
-- `refreshed_at` and falls back to the live query if the snapshot
-- is stale (>3 h) or empty.

CREATE TABLE IF NOT EXISTS feed_snapshot (
    -- Natural key. `url` is the representative doc's url (anchor
    -- URL would lose data — multiple anchor matches per anchor
    -- happen for legitimate dedup). One row per representative.
    url               TEXT        PRIMARY KEY,
    -- Canonical key the timeline uses to compare/sort.
    canonical_url     TEXT        NOT NULL,
    -- The dedup anchor — every doc whose anchor_url matches this
    -- row is rolled up into the sharer arrays below. Two tweets
    -- linking the same arxiv paper share this value, so the
    -- precomputed feed already presents one row per resource.
    anchor_url        TEXT        NOT NULL,

    -- Representative metadata, picked from the visually-richest doc.
    title             TEXT        NOT NULL DEFAULT '',
    date              DATE,
    summary           TEXT        NOT NULL DEFAULT '',
    clean_title       TEXT        NOT NULL DEFAULT '',
    clean_summary     TEXT        NOT NULL DEFAULT '',
    urls              TEXT[]      NOT NULL DEFAULT '{}',
    tags              TEXT[]      NOT NULL DEFAULT '{}',
    source            TEXT        NOT NULL DEFAULT '',
    source_url        TEXT,
    linked_urls       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    link_hosts        TEXT[]      NOT NULL DEFAULT '{}',

    -- Sharer aggregates. `sharer_user_ids` is the
    -- followee-intersection filter key — the GIN index below covers
    -- the `&&` lookup. `sharers` is the pre-rendered avatar stack
    -- the frontend reads as-is (no LATERAL JOIN at request time).
    primary_user_id   BIGINT      NOT NULL,
    sharer_user_ids   BIGINT[]    NOT NULL DEFAULT '{}',
    sharers           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    sharer_count      INTEGER     NOT NULL DEFAULT 0,
    -- True iff at least one sharer is a VIP. Drives the anon path:
    -- logged-out callers want the global-VIP-feed and this turns it
    -- into a single indexed scan instead of an array intersect.
    any_vip_sharer    BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Viewer-agnostic score. Computed at refresh time. The handler
    -- adds a follow-share and fresh-self bonus on read.
    score             DOUBLE PRECISION NOT NULL,

    -- Wall-clock of the refresh that wrote this row. The handler
    -- compares the MAX(refreshed_at) against a freshness window
    -- (3 h) and bypasses the snapshot path if it's stale.
    refreshed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migrations — any column added later goes here so a
-- redeploy lifts the schema in place.
ALTER TABLE feed_snapshot ADD COLUMN IF NOT EXISTS any_vip_sharer BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE feed_snapshot ADD COLUMN IF NOT EXISTS anchor_url     TEXT;
-- Backfill the new anchor_url column for pre-existing rows by
-- defaulting it to the canonical URL — the anchor degenerates to
-- the canonical when no priority host (arxiv/huggingface/…) appears
-- in the referenced URL set.
UPDATE feed_snapshot SET anchor_url = canonical_url WHERE anchor_url IS NULL;
ALTER TABLE feed_snapshot ALTER COLUMN anchor_url SET NOT NULL;

-- Primary read path on the anon timeline: top-N by score where
-- there's at least one VIP sharer. Partial index keeps it tight.
CREATE INDEX IF NOT EXISTS idx_feed_snapshot_vip_score
    ON feed_snapshot (score DESC, date DESC NULLS LAST)
    WHERE any_vip_sharer = TRUE;

-- Logged-in path: `sharer_user_ids && $followees` — GIN handles the
-- array overlap in microseconds. Combined with an ORDER BY score we
-- typically scan <500 rows to fill a 50-row page.
CREATE INDEX IF NOT EXISTS idx_feed_snapshot_sharers
    ON feed_snapshot USING gin (sharer_user_ids);

-- Score-only index for the general top-N case (no follow filter,
-- e.g. operator probes). Date desc as tiebreaker.
CREATE INDEX IF NOT EXISTS idx_feed_snapshot_score
    ON feed_snapshot (score DESC, date DESC NULLS LAST);

-- Allows the handler to cheaply check "is the snapshot fresh?"
-- without scanning all rows. `LIMIT 1` of the index hits one page.
CREATE INDEX IF NOT EXISTS idx_feed_snapshot_refreshed
    ON feed_snapshot (refreshed_at DESC);

COMMENT ON TABLE feed_snapshot IS
    'Hourly snapshot of the scored feed over a 180-day window. One row per anchor URL with a viewer-agnostic score; the timeline handler adds per-viewer bonuses on read. Refreshed atomically by the knowledge-feed-snapshot daemon.';

COMMENT ON COLUMN feed_snapshot.url             IS 'Representative URL (visually richest among the docs sharing the anchor).';
COMMENT ON COLUMN feed_snapshot.canonical_url   IS 'Canonical form of the representative URL — used for client-side dedup hints.';
COMMENT ON COLUMN feed_snapshot.anchor_url      IS 'Resource-identity key. Multiple URLs (paper + tweets-about-paper) collapse to one anchor and one row in this table.';
COMMENT ON COLUMN feed_snapshot.score           IS 'Viewer-agnostic score: sci×6 + weekly recency bucket + LN(total_share)·0.7 + VIP×0.8 + LN(followers/10k) + rich-tweet 1.5.';
COMMENT ON COLUMN feed_snapshot.sharer_user_ids IS 'All user_ids whose docs map to this anchor. Followee-overlap filter joins on this.';
COMMENT ON COLUMN feed_snapshot.any_vip_sharer  IS 'TRUE iff at least one sharer is a VIP. Powers the anon-timeline indexed scan.';
COMMENT ON COLUMN feed_snapshot.refreshed_at    IS 'Wall-clock of the refresh that wrote this row. Handler bypasses snapshot when MAX(refreshed_at) is too stale.';
