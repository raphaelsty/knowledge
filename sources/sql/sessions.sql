-- Schema for the `sessions` table.
--
-- One row per anonymous browser session on a library. `id` is the
-- client-generated UUID (crypto.randomUUID in the frontend), so the
-- server can upsert without a round-trip.
--
-- Storage shape is intentionally tight: device + referrer live here
-- (one write per session) instead of on every event.

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID        PRIMARY KEY,
    user_id         BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Compact enums — see COMMENT ON COLUMN below for the mapping.
    device          SMALLINT    NOT NULL DEFAULT 0,
    referrer_domain TEXT,

    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-library session list, newest first (dashboard).
CREATE INDEX IF NOT EXISTS idx_sessions_user_started
    ON sessions (user_id, started_at DESC);

-- ── DB-level documentation ──────────────────────────────────────────────
COMMENT ON TABLE sessions IS
    'Anonymous browser sessions on a library page. One row per (browser tab × library).';

COMMENT ON COLUMN sessions.id              IS 'Client-generated UUID (crypto.randomUUID in analytics.js).';
COMMENT ON COLUMN sessions.user_id         IS 'Library being browsed. FK → users(id) ON DELETE CASCADE.';
COMMENT ON COLUMN sessions.device          IS 'Device enum: 0=desktop, 1=mobile.';
COMMENT ON COLUMN sessions.referrer_domain IS 'Referrer host (empty/NULL if direct navigation).';
COMMENT ON COLUMN sessions.started_at      IS 'First event in this session.';
COMMENT ON COLUMN sessions.last_seen_at    IS 'Latest event in this session — updated on every event ingest.';
