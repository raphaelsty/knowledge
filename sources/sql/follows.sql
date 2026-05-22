-- Follow graph.
--
-- (follower_id, followed_id) is a directed edge: follower follows
-- followed. Both sides FK into users(id) with ON DELETE CASCADE so
-- account deletion cleans up the graph without orphans.
--
-- The PK enforces no duplicate follow rows. A CHECK prevents
-- self-follows (semantically nonsense; a "show me my followees"
-- query would otherwise have to special-case the caller's own row).
--
-- Two indices serve the two natural query axes:
--   • idx_follows_follower  → "who do I follow?"   (timeline build)
--   • idx_follows_followed  → "who follows X?"     (profile pages)

CREATE TABLE IF NOT EXISTS follows (
    follower_id  BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    followed_id  BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (follower_id, followed_id),
    CHECK (follower_id <> followed_id)
);

CREATE INDEX IF NOT EXISTS idx_follows_follower
    ON follows (follower_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_follows_followed
    ON follows (followed_id, created_at DESC);

COMMENT ON TABLE  follows IS
    'Directed follow graph: follower_id follows followed_id.';
COMMENT ON COLUMN follows.follower_id IS 'The user doing the following.';
COMMENT ON COLUMN follows.followed_id IS 'The user being followed.';
COMMENT ON COLUMN follows.created_at  IS 'When the follow edge was created.';
