-- Pre-computed per-VIP personal page snapshot.
--
-- Same anchor-collapse + feed-style scoring as `feed_snapshot`, but
-- scoped to a single user's own docs. No cross-user sharer roll-up —
-- the per-anchor "sharer" is always the page owner — so there's no
-- VIP-share term and no cluster-diversity penalty. What remains is
-- the resource-quality shape: arxiv / HF / github > everything else,
-- tweets with a linked paper > plain tweets, recency tier.
--
-- One row per (user_id, anchor's representative URL). Two tweets the
-- same VIP posted linking the same paper collapse to one row; the
-- arxiv abs page + a tweet linking it collapse too.
--
-- Refreshed per-user on demand by `refresh_personal_snapshot` —
-- triggered both by the per-user pipeline (after a VIP's docs change)
-- and by a sweep daemon that catches stale rows.

CREATE TABLE IF NOT EXISTS personal_snapshot (
    user_id           BIGINT      NOT NULL,
    -- Representative URL (visually richest doc in the anchor group).
    url               TEXT        NOT NULL,
    canonical_url     TEXT        NOT NULL,
    anchor_url        TEXT        NOT NULL,

    title             TEXT        NOT NULL DEFAULT '',
    date              DATE,
    summary           TEXT        NOT NULL DEFAULT '',
    clean_title       TEXT        NOT NULL DEFAULT '',
    clean_summary     TEXT        NOT NULL DEFAULT '',
    urls              TEXT[]      NOT NULL DEFAULT '{}',
    tags              TEXT[]      NOT NULL DEFAULT '{}',
    extra_tags        TEXT[]      NOT NULL DEFAULT '{}',
    source            TEXT        NOT NULL DEFAULT '',
    source_url        TEXT,
    linked_urls       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    link_hosts        TEXT[]      NOT NULL DEFAULT '{}',
    -- Categories assigned to any doc in the anchor group, unioned.
    categories        TEXT[]      NOT NULL DEFAULT '{}',
    -- How many of the user's own docs collapsed into this anchor.
    -- Useful for the "user posted N times about this paper" hint.
    anchor_doc_count  INTEGER     NOT NULL DEFAULT 1,
    -- Cross-user sharer roll-up — populated from `feed_snapshot` so
    -- the personal page can render the same "people who also have
    -- this in their library" avatar stack the global feed shows.
    -- Reads as the page owner's slug when the anchor isn't in
    -- feed_snapshot (e.g. private/long-tail resources).
    sharer_user_ids   BIGINT[]    NOT NULL DEFAULT '{}',
    sharers           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    sharer_count      INTEGER     NOT NULL DEFAULT 0,
    -- Count of distinct VIP sharers across all libraries that own
    -- this anchor — mirrored from `feed_snapshot.vip_sharer_count`.
    -- Drives the per-anchor "many VIPs co-signed this" boost that
    -- lifts widely-saved resources on the personal page (same shape
    -- as the global feed's boost, scoped to single-user reads).
    vip_sharer_count  INTEGER     NOT NULL DEFAULT 0,
    -- `indexed` aggregate — TRUE iff *every* doc in the anchor group
    -- is indexed. The personal page surfaces an "embedding pending"
    -- pill when this is FALSE.
    indexed           BOOLEAN     NOT NULL DEFAULT TRUE,

    -- Feed-style score. See `personal_snapshot.py::_build_refresh_sql`
    -- for the breakdown. Bigger = more resource-y / more recent.
    score             DOUBLE PRECISION NOT NULL,
    refreshed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (user_id, url)
);

-- Idempotent migrations.
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS categories       TEXT[]  NOT NULL DEFAULT '{}';
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS anchor_doc_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS indexed          BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS extra_tags       TEXT[]  NOT NULL DEFAULT '{}';
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS sharer_user_ids  BIGINT[] NOT NULL DEFAULT '{}';
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS sharers          JSONB    NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS sharer_count     INTEGER  NOT NULL DEFAULT 0;
ALTER TABLE personal_snapshot ADD COLUMN IF NOT EXISTS vip_sharer_count INTEGER  NOT NULL DEFAULT 0;

-- Primary read path: top-N for one user by score.
CREATE INDEX IF NOT EXISTS idx_personal_snapshot_user_score
    ON personal_snapshot (user_id, score DESC, date DESC NULLS LAST);

-- Category overlap filter on a single user's snapshot.
CREATE INDEX IF NOT EXISTS idx_personal_snapshot_user_categories
    ON personal_snapshot USING gin (categories);

-- "How stale is THIS user's snapshot?" — cheap lookup for the daemon.
CREATE INDEX IF NOT EXISTS idx_personal_snapshot_refreshed
    ON personal_snapshot (user_id, refreshed_at DESC);

COMMENT ON TABLE personal_snapshot IS
    'Per-VIP personal-page snapshot. One row per anchor URL the VIP owns. Score = feed-style resource-quality shape (sci/recency/link/image bonus) without the cross-user sharer terms. Refreshed per-user by the indexer hook + a sweep daemon.';

COMMENT ON COLUMN personal_snapshot.anchor_url       IS 'Resource-identity key — multiple of the VIP''s docs (tweet + arxiv + variant) collapse to one anchor and one row here.';
COMMENT ON COLUMN personal_snapshot.score            IS 'Feed-style resource score: sci×6 + recency tier + link/image bonus. No cross-user VIP-share term (single owner per row).';
COMMENT ON COLUMN personal_snapshot.anchor_doc_count IS 'How many of this user''s docs map to this anchor.';
