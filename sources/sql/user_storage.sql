-- Per-user storage stats.
--
-- Snapshot of how much disk a single user occupies, refreshed by the
-- API (POST /api/me/storage/refresh). We materialise this rather
-- than computing on every read because:
--   • `pg_column_size(d.*)` SUM is O(rows) and gets expensive past
--     a few hundred thousand documents.
--   • The per-user next-plaid index lives on disk, not in Postgres
--     — sizing it means a filesystem walk inside the API process.
--
-- We do NOT track the global `__all__` index here. Only VIPs feed
-- `__all__` and the platform absorbs that cost; this table is
-- strictly "what does the user owe storage credits for?".

CREATE TABLE IF NOT EXISTS user_storage (
    user_id      BIGINT      PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    -- Number of rows the user owns in `documents`.
    doc_count    INTEGER     NOT NULL DEFAULT 0 CHECK (doc_count >= 0),
    -- Sum of `pg_column_size(d.*)` across the user's documents.
    -- Approximate Postgres row payload (excludes index pages, TOAST
    -- pointer overhead, MVCC bloat).
    db_bytes     BIGINT      NOT NULL DEFAULT 0 CHECK (db_bytes >= 0),
    -- Size of the user's personal next-plaid index directory on
    -- disk (`indexes/{users.index_name}/`). Zero when the index
    -- doesn't exist yet (new account before first run).
    index_bytes  BIGINT      NOT NULL DEFAULT 0 CHECK (index_bytes >= 0),
    -- Last successful refresh. Stale rows can be re-computed by the
    -- API on demand or by the pipeline at run start.
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  user_storage IS
    'Snapshot of per-user storage footprint (postgres bytes + personal index bytes). Refreshed by POST /api/me/storage/refresh.';
COMMENT ON COLUMN user_storage.db_bytes    IS 'Sum of pg_column_size(d.*) over the user''s documents. Approximate row payload, excludes index pages and bloat.';
COMMENT ON COLUMN user_storage.index_bytes IS 'Size of indexes/{users.index_name}/ on disk in bytes. Personal index only — does not include __all__.';
