-- ── personality_bookmarks ───────────────────────────────────────────────
--
-- Cross-user "follow" / "bookmark" graph. One row per (owner, target) pair.
-- Read by the library picker on the search page to surface a dedicated
-- "Bookmarks" section above the by-category list, so a user's saved
-- people are one click away from being added as an active library.
--
-- Why a table (not a JSONB column on users): the relation is symmetric
-- in shape (FK → users on both sides) and bookmark counts per target
-- are useful for popularity tiles. A normalized table is cheaper to
-- query in either direction.
--
-- Cascade deletes: dropping a user purges any bookmark involving them.

CREATE TABLE IF NOT EXISTS personality_bookmarks (
    user_id            BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    bookmarked_user_id BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, bookmarked_user_id),
    CHECK (user_id <> bookmarked_user_id)
);

-- Owner-side lookup: "what has user X bookmarked?". The composite PK
-- already covers (user_id, ...) as its leftmost prefix, so a separate
-- index isn't strictly necessary — but we keep one anyway because the
-- frequent pattern is `WHERE user_id = $1` + ORDER BY created_at.
CREATE INDEX IF NOT EXISTS idx_personality_bookmarks_user
    ON personality_bookmarks (user_id, created_at DESC);

-- Target-side lookup: "who has bookmarked user Y?". Useful for future
-- popularity / inbound-link counts.
CREATE INDEX IF NOT EXISTS idx_personality_bookmarks_target
    ON personality_bookmarks (bookmarked_user_id);

COMMENT ON TABLE personality_bookmarks IS
    'User-to-user bookmark graph. Surfaces in the library picker''s "Bookmarks" section.';
