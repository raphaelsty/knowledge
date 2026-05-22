-- Schema for the `pipeline_runs` table.
--
-- Live tracker of every parsing run — one row per (user × invocation),
-- mutated as the run progresses. The row is INSERTed when the pipeline
-- starts (`status = 'running'`), UPDATEd at each stage, and sealed at
-- the end with `status = 'success' | 'failed'`.
--
-- Callers:
--   • Python pipeline (sources/utils/client.py::run_pipeline)
--   • JS browser sync (web/source/sync.js, via the /auth/me/sync/start
--     and /auth/me/sync/end endpoints)
--
-- Queries:
--   • "What's running right now?"   → WHERE status = 'running'
--   • "Latest run for each user"    → DISTINCT ON (user_id) ORDER BY started_at DESC
--   • "Last 24h failures"           → WHERE status = 'failed' AND started_at > now() - '1d'

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL   PRIMARY KEY,
    user_id         BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 'python' (server-side pipeline) or 'js-sync' (browser Sync button).
    -- Future triggers (cron, webhook) land here without a schema change.
    trigger         TEXT        NOT NULL DEFAULT 'python',
    -- running | success | failed. Default 'running' so the row stays
    -- pending until someone finishes or fails it.
    status          TEXT        NOT NULL DEFAULT 'running',
    -- Current step name (fetch / clean / link_check / tag / index).
    -- Lets a dashboard show "where is this run stuck?".
    stage           TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,                -- NULL while running
    duration_secs   REAL,                       -- NULL while running, set at completion
    new_documents   INTEGER     NOT NULL DEFAULT 0,
    total_documents INTEGER     NOT NULL DEFAULT 0,
    -- Populated when status = 'failed'. Free-form so we can carry a
    -- traceback / exception str without committing to a structured shape.
    error           TEXT,
    timings         JSONB       NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Latest-run-per-user query path (dashboard "last run: X min ago").
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_user_started
    ON pipeline_runs (user_id, started_at DESC);
-- Partial index: only rows currently in-flight, so the "what's running
-- right now" dashboard query stays O(1) no matter how large the table
-- grows.
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_running
    ON pipeline_runs (started_at DESC)
    WHERE status = 'running';

-- ── Idempotent migration for pre-tracker DBs ────────────────────────────
-- Adds the tracker columns, back-fills from the historical `success`
-- bool, then drops it. Safe to re-run.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'pipeline_runs' AND column_name = 'trigger'
    ) THEN
        ALTER TABLE pipeline_runs
            ADD COLUMN trigger TEXT NOT NULL DEFAULT 'python';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'pipeline_runs' AND column_name = 'status'
    ) THEN
        ALTER TABLE pipeline_runs
            ADD COLUMN status TEXT NOT NULL DEFAULT 'success';
        -- Back-fill using the legacy `success` column before we drop
        -- it below: success=TRUE → 'success', success=FALSE → 'failed'.
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_name = 'pipeline_runs' AND column_name = 'success'
        ) THEN
            UPDATE pipeline_runs
               SET status = CASE WHEN success THEN 'success' ELSE 'failed' END;
        END IF;
        -- Future inserts should default to 'running', not 'success' —
        -- flip the default now that the back-fill is done.
        ALTER TABLE pipeline_runs
            ALTER COLUMN status SET DEFAULT 'running';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'pipeline_runs' AND column_name = 'stage'
    ) THEN
        ALTER TABLE pipeline_runs ADD COLUMN stage TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'pipeline_runs' AND column_name = 'error'
    ) THEN
        ALTER TABLE pipeline_runs ADD COLUMN error TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'pipeline_runs' AND column_name = 'started_at'
    ) THEN
        ALTER TABLE pipeline_runs
            ADD COLUMN started_at TIMESTAMPTZ NOT NULL DEFAULT now();
        UPDATE pipeline_runs SET started_at = created_at;
    END IF;
    -- Legacy column: finished_at was NOT NULL. Relax it so rows can
    -- live in the table while still running.
    BEGIN
        ALTER TABLE pipeline_runs ALTER COLUMN finished_at DROP NOT NULL;
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
    BEGIN
        ALTER TABLE pipeline_runs ALTER COLUMN duration_secs DROP NOT NULL;
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
    -- Drop the legacy `success` boolean — `status` is now the only
    -- source of truth.
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'pipeline_runs' AND column_name = 'success'
    ) THEN
        ALTER TABLE pipeline_runs DROP COLUMN success;
    END IF;
END$$;

-- ── DB-level documentation ──────────────────────────────────────────────
COMMENT ON TABLE pipeline_runs IS
    'Live tracker of parsing runs. One row per invocation per user. INSERTed at start, mutated as stages complete, sealed at end.';

COMMENT ON COLUMN pipeline_runs.user_id         IS 'FK → users(id). ON DELETE CASCADE.';
COMMENT ON COLUMN pipeline_runs.trigger         IS 'Origin of the run: python (server pipeline) or js-sync (browser Sync button).';
COMMENT ON COLUMN pipeline_runs.status          IS 'running | success | failed. Default running; flipped by the worker that owns the row.';
COMMENT ON COLUMN pipeline_runs.stage           IS 'Current step name (fetch / clean / link_check / tag / index). NULL once the run is sealed.';
COMMENT ON COLUMN pipeline_runs.started_at      IS 'Wall-clock time the pipeline started.';
COMMENT ON COLUMN pipeline_runs.finished_at     IS 'Wall-clock time the pipeline finished. NULL while status = running.';
COMMENT ON COLUMN pipeline_runs.duration_secs   IS 'End-to-end duration in seconds. NULL while status = running.';
COMMENT ON COLUMN pipeline_runs.new_documents   IS 'New documents discovered this run.';
COMMENT ON COLUMN pipeline_runs.total_documents IS 'Total documents stored for this user after the run.';
COMMENT ON COLUMN pipeline_runs.error           IS 'Exception message when status = failed. NULL otherwise.';
COMMENT ON COLUMN pipeline_runs.timings         IS 'Per-step timings as a JSONB array of {step, duration_secs}.';
