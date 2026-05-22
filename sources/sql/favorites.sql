-- Schema for the `favorites` table.
--
-- Many-to-many: each row means "user X has favorited user Y". A user
-- cannot favorite themselves (CHECK), and either deletion cascades
-- cleanly (ON DELETE CASCADE on both FKs).
--
-- Primary key is the pair so a user can't double-favorite the same
-- personality; `created_at` is kept for an eventual "recent favorites"
-- ordering without needing a separate position column.

CREATE TABLE IF NOT EXISTS favorites (
    user_id     BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    favorite_id BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, favorite_id),
    CHECK (user_id <> favorite_id)
);

-- Reverse lookup: "who has starred X?" for eventual social features.
CREATE INDEX IF NOT EXISTS idx_favorites_favorite
    ON favorites (favorite_id);

COMMENT ON TABLE favorites IS
    'Per-user favorites list. One row per (user, favorited personality) pair.';
COMMENT ON COLUMN favorites.user_id     IS 'The user doing the favoriting.';
COMMENT ON COLUMN favorites.favorite_id IS 'The personality being favorited.';
COMMENT ON COLUMN favorites.created_at  IS 'When the star was added. Use for "recently favorited" ordering.';
