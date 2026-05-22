-- Schema for the `events` table.
--
-- One row per user interaction on a library page. Typed columns
-- (not JSONB) — each event costs roughly 70-170 bytes vs 150-300 for
-- the previous JSONB shape.
--
-- `user_id` is denormalized from sessions.user_id so the dashboard's
-- most common query ("all events for library X") can be served by an
-- index-only scan.

CREATE TABLE IF NOT EXISTS events (
    id             BIGSERIAL   PRIMARY KEY,
    session_id     UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id        BIGINT      NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    event_type     SMALLINT    NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- ── Payload columns (nullable, only set for the relevant event_type) ──
    -- Search & click context
    query          TEXT,
    result_count   SMALLINT,
    latency_ms     INTEGER,
    source_filter  TEXT,
    sort_mode      SMALLINT,   -- 0=relevance, 1=date

    -- Click & find_similar context
    doc_url        TEXT,
    position       SMALLINT,
    score          REAL,

    -- Recommendation-training signals (added later; idempotent migrations
    -- below add the same columns to existing tables).
    personality_slug TEXT,
    viewer_user_id   BIGINT,
    client_ts        TIMESTAMPTZ,

    -- Guardrail: event_type must be a known enum value (see comment below).
    CONSTRAINT chk_event_type CHECK (event_type BETWEEN 1 AND 6)
);

-- Idempotent migrations: bring an already-deployed events table up to
-- the latest column set. ALTER TABLE IF NOT EXISTS is no-op when the
-- column is already present, so this is safe to re-run on every boot.
ALTER TABLE events ADD COLUMN IF NOT EXISTS personality_slug TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS viewer_user_id   BIGINT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS client_ts        TIMESTAMPTZ;

-- viewer_user_id is FK to users(id) but we keep it nullable + ON DELETE
-- SET NULL so anonymous events and deleted users don't break the table.
-- Apply the constraint defensively in case the column was added in a
-- previous schema bump without the FK.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'events_viewer_user_id_fkey'
    ) THEN
        ALTER TABLE events
            ADD CONSTRAINT events_viewer_user_id_fkey
            FOREIGN KEY (viewer_user_id)
            REFERENCES users(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Composite index: primary access path is "events for library X filtered
-- by type, most recent first". Leftmost prefix (user_id) alone also
-- serves unfiltered time-series queries.
CREATE INDEX IF NOT EXISTS idx_events_user_type_created
    ON events (user_id, event_type, created_at DESC);

-- Session reconstruction (rare but useful for debugging).
CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id);

-- Recommendation-training access paths.
-- "What did this viewer do?" — feeds the per-user training set.
CREATE INDEX IF NOT EXISTS idx_events_viewer_created
    ON events (viewer_user_id, created_at DESC)
    WHERE viewer_user_id IS NOT NULL;

-- "Which queries led to clicks on doc X?" — feeds learning-to-rank.
CREATE INDEX IF NOT EXISTS idx_events_doc_query
    ON events (doc_url, event_type, created_at DESC)
    WHERE doc_url IS NOT NULL;

-- ── DB-level documentation ──────────────────────────────────────────────
COMMENT ON TABLE events IS
    'User interactions on library pages. One row per click / search / view. Typed columns for compact storage.';

COMMENT ON COLUMN events.id            IS 'Surrogate primary key.';
COMMENT ON COLUMN events.session_id    IS 'FK → sessions(id). Chain of events from the same browser tab.';
COMMENT ON COLUMN events.user_id       IS 'Library being browsed. Denormalized from sessions for index-only scans.';
COMMENT ON COLUMN events.event_type    IS 'Event enum: 1=view, 2=search, 3=click, 4=find_similar, 5=filter_apply, 6=folder_browse.';
COMMENT ON COLUMN events.created_at    IS 'Event timestamp (server-assigned).';
COMMENT ON COLUMN events.query         IS 'Search query text. Set for event_type=2 (search) and optionally on click (source query).';
COMMENT ON COLUMN events.result_count  IS 'Number of results returned for a search.';
COMMENT ON COLUMN events.latency_ms    IS 'Search latency in milliseconds.';
COMMENT ON COLUMN events.source_filter IS 'Active source filter(s) when the event fired (comma-separated).';
COMMENT ON COLUMN events.sort_mode     IS 'Sort mode enum: 0=relevance, 1=date.';
COMMENT ON COLUMN events.doc_url       IS 'Clicked/similar document URL.';
COMMENT ON COLUMN events.position      IS 'Click rank on the result list (0-based).';
COMMENT ON COLUMN events.score         IS 'Model/rerank score of the clicked document.';
COMMENT ON COLUMN events.personality_slug IS 'Slug of the personality being browsed (denormalised from users.username for easy joins on training-set extracts).';
COMMENT ON COLUMN events.viewer_user_id   IS 'Logged-in user actually triggering the event. NULL for anonymous browsing. Distinct from events.user_id, which identifies the library being browsed.';
COMMENT ON COLUMN events.client_ts        IS 'Client-side timestamp at event-fire time. Lets us reconstruct true order even when sendBeacon batches delay delivery.';
