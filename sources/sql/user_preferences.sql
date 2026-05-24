-- Per-viewer learned-preference tables.
--
-- Two derived tables, recomputed periodically from the raw signal
-- streams (favorites, documents, events). Each table answers a
-- single per-viewer question:
--
--   user_personality_weight  — "how much does this viewer engage
--                               with content shared by personality
--                               X?"   range ≈ [-2, +2]
--   user_category_weight     — "how much does this viewer engage
--                               with content tagged Y?"
--                               range ≈ [-2, +2]
--
-- The timeline handler reads these as additive terms on top of the
-- viewer-agnostic `feed_snapshot.score` to produce a personalised
-- score, then interleaves the top personalised picks into the
-- standard feed at 1-in-5 positions.
--
-- Both tables are append-replace per (viewer, target). The
-- recompute job rewrites rows in place via `INSERT … ON CONFLICT
-- DO UPDATE`; rows for inactive (viewer, target) pairs decay to
-- zero on the next refresh.
--
-- Storage: at ~400 VIPs × ~5 categories per active viewer the
-- footprint is a few rows per user. Even at 100k users that's
-- well under 100 MB.

CREATE TABLE IF NOT EXISTS user_personality_weight (
    viewer_id      BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    personality_id BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    weight         REAL        NOT NULL DEFAULT 0,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (viewer_id, personality_id)
);

-- Reverse lookup used by the recommendation query: "for this
-- viewer, give me the personalities with the highest weights".
CREATE INDEX IF NOT EXISTS idx_user_personality_weight_top
    ON user_personality_weight (viewer_id, weight DESC);

CREATE TABLE IF NOT EXISTS user_category_weight (
    viewer_id      BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category_slug  TEXT        NOT NULL,
    weight         REAL        NOT NULL DEFAULT 0,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (viewer_id, category_slug)
);

CREATE INDEX IF NOT EXISTS idx_user_category_weight_top
    ON user_category_weight (viewer_id, weight DESC);

COMMENT ON TABLE user_personality_weight IS
    'Learned per-viewer engagement weight for each personality. Recomputed periodically from favorites, library saves, clicks, find_similar, and card_seen dwell.';
COMMENT ON COLUMN user_personality_weight.weight IS
    'Squashed to [-2, +2] via TANH(...) so a single hot personality cannot dominate the score.';

COMMENT ON TABLE user_category_weight IS
    'Learned per-viewer engagement weight for each category slug. Same shape as user_personality_weight but indexed by the categorize daemon''s slug taxonomy.';
COMMENT ON COLUMN user_category_weight.weight IS
    'Same [-2, +2] squashed range.';
