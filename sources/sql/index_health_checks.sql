-- Per-user search-index health checks.
--
-- One row per (user × check). Stores the diagnostic shape the API
-- returned at check time + a `status` verdict. The script `scripts/
-- check_indexes.py` writes here; `make index-check` reads the most
-- recent row per user to decide who to check next (oldest first).
--
-- Status verdicts:
--   • healthy        — index exists, num_documents == metadata_count,
--                      embeddings present, PG ≈ API count.
--   • missing        — API 404 on GET /indices/{name}.
--   • broken         — num_documents > 0 BUT num_embeddings == 0
--                      (or any other "loaded but empty payload" shape).
--   • meta_mismatch  — num_documents != metadata_count.
--   • pg_drift       — PG `indexed=true` count diverges from API count
--                      by more than 5% (or 5 docs absolute).
--   • error          — request failed in a way we couldn't classify;
--                      `error` carries the message.

CREATE TABLE IF NOT EXISTS index_health_checks (
    id                BIGSERIAL   PRIMARY KEY,
    user_id           BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    index_name        TEXT        NOT NULL,
    checked_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    status            TEXT        NOT NULL,
    -- API shape at check time (NULL when index is missing).
    num_documents     INTEGER,
    num_embeddings    INTEGER,
    metadata_count    INTEGER,
    avg_doclen        REAL,
    -- PG side, for drift detection.
    pg_total_docs     INTEGER,
    pg_indexed_docs   INTEGER,
    -- Free-form bag for whatever other fields the API returned, plus
    -- any computed diagnostics (e.g. drift ratio). Kept as JSONB so
    -- adding a new metric is schema-safe.
    details           JSONB       NOT NULL DEFAULT '{}',
    -- Populated when status = 'error' (or any unexpected code path).
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_ihc_user_checked
    ON index_health_checks (user_id, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_ihc_status_checked
    ON index_health_checks (status, checked_at DESC)
    WHERE status <> 'healthy';

COMMENT ON TABLE index_health_checks IS
    'Per-user search-index diagnostics, one row per check. Latest row per user-id drives the staleness queue.';
