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

    -- Guardrail: event_type must be a known enum value (see comment below).
    CONSTRAINT chk_event_type CHECK (event_type BETWEEN 1 AND 6)
);

-- Composite index: primary access path is "events for library X filtered
-- by type, most recent first". Leftmost prefix (user_id) alone also
-- serves unfiltered time-series queries.
CREATE INDEX IF NOT EXISTS idx_events_user_type_created
    ON events (user_id, event_type, created_at DESC);

-- Session reconstruction (rare but useful for debugging).
CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id);

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
