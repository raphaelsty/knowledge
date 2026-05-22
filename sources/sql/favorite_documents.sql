-- Schema for the `favorite_documents` table.
--
-- One row per (user, starred URL) pair. Documents are identified by
-- URL — they may live in any personality's library (or a future
-- anonymous submission flow), so we intentionally *don't* foreign-key
-- to `documents`. Favorites are private: never exposed on another
-- user's profile or in public APIs.

CREATE TABLE IF NOT EXISTS favorite_documents (
    user_id    BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url        TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, url)
);

-- `ORDER BY created_at DESC` is the list query's hot path.
CREATE INDEX IF NOT EXISTS idx_favorite_documents_recent
    ON favorite_documents (user_id, created_at DESC);

COMMENT ON TABLE favorite_documents IS
    'Per-user private favorites list (starred documents across every library).';
COMMENT ON COLUMN favorite_documents.user_id IS 'Owner of the favorite — never exposed to others.';
COMMENT ON COLUMN favorite_documents.url     IS 'Canonical URL of the starred document.';
