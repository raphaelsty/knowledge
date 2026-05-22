-- Single-row health surface for the launchd-managed twitter
-- feeder running on the operator's Mac.
--
-- The feeder POSTs a heartbeat to the prod API on every meaningful
-- transition (pass start, per-personality progress, pass end,
-- error) and the admin panel reads back the current row. We
-- intentionally keep just ONE row — old state is overwritten —
-- because the goal is "is the process running as expected right
-- now", not a long history. Recent failures live on
-- last_error / last_error_at; a richer audit log can be added
-- later if it turns out to be useful.

CREATE TABLE IF NOT EXISTS twitter_feed_status (
    -- Sentinel PK: forces a single row. Every upsert targets id=1.
    id                INTEGER     PRIMARY KEY CHECK (id = 1),
    -- Wall-clock of the most recent heartbeat from the client.
    -- Used by the panel to flag the daemon as "stale" if older
    -- than ~15 min.
    heartbeat_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Client-side state machine.
    --   starting  → pass kicking off, queue not yet known
    --   running   → mid-pass (current_slug populated)
    --   idle      → between passes, waiting for `--rest` interval
    --   sleeping  → sleeping after rate-limit stall
    --   error     → fatal error in the last attempt
    state             TEXT        NOT NULL DEFAULT 'unknown',
    -- Stats from the most recent pass (overwritten every pass).
    pass_started_at   TIMESTAMPTZ,
    pass_finished_at  TIMESTAMPTZ,
    pass_processed    INTEGER     NOT NULL DEFAULT 0,
    pass_total        INTEGER     NOT NULL DEFAULT 0,
    -- Personality the feeder is currently working on, if any.
    current_slug      TEXT,
    current_handle    TEXT,
    -- Most recent error message, truncated to 500 chars by the
    -- write path so a runaway stacktrace can't bloat the row.
    last_error        TEXT,
    last_error_at     TIMESTAMPTZ,
    -- How many passes the feeder has completed since its last
    -- start. Useful to spot "it ran once then died" patterns.
    pass_count        INTEGER     NOT NULL DEFAULT 0,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  twitter_feed_status IS 'Single-row health surface for the launchd-managed twitter feeder on the operator''s Mac. Heartbeat receiver; admin panel reads it back.';
COMMENT ON COLUMN twitter_feed_status.heartbeat_at IS 'Wall-clock of the most recent heartbeat; admin panel reads this to colour the tile (fresh / stale).';
COMMENT ON COLUMN twitter_feed_status.state        IS 'Client-side state machine: starting | running | idle | sleeping | error | unknown.';

-- Seed the singleton row so the upsert can do an UPDATE-only path
-- on every subsequent heartbeat (cheaper than an INSERT ... ON
-- CONFLICT plan re-resolve).
INSERT INTO twitter_feed_status (id, state) VALUES (1, 'unknown')
    ON CONFLICT (id) DO NOTHING;
