-- Per-source breakdown of a pipeline run.
--
-- One parent row in `pipeline_runs` represents the whole invocation
-- for a user; one child row in `pipeline_source_runs` represents each
-- fetcher block (github stars, hackernews comments, mixedbread.com
-- sitemap, …). The split lets the admin panel answer:
--
--   • Which source failed last week and how often?
--   • For source X, what's the success rate / median duration?
--   • For user Y, how does each individual fetcher do over time?
--
-- Status:
--   • success — fetcher returned without raising; new_documents may be 0.
--   • failed  — fetcher raised; the exception message is in `error`.
--   • skipped — fetcher was bypassed deliberately (no creds, fresh
--               cooldown, all modes disabled). `error` carries the reason.

CREATE TABLE IF NOT EXISTS pipeline_source_runs (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          BIGINT      NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    user_id         BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Stable bucket — matches `documents.source` where applicable
    -- ('github', 'twitter', 'hackernews', 'huggingface', 'reddit',
    -- 'scholar', 'stackoverflow', 'youtube', 'arxiv', 'wikipedia',
    -- 'zotero') plus per-website hostnames for feeds/sitemaps.
    source          TEXT        NOT NULL,
    -- Optional sub-label: '@user', 'comments', 'submissions',
    -- 'bookmarks', 'group/12345', etc. Lets one source bucket
    -- carry multiple distinguishable rows per run.
    detail          TEXT,
    status          TEXT        NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_secs   REAL        NOT NULL DEFAULT 0,
    new_documents   INTEGER     NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_psr_user_started
    ON pipeline_source_runs (user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_psr_source_started
    ON pipeline_source_runs (source, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_psr_run
    ON pipeline_source_runs (run_id);
-- Hot path for "recent failures" admin view.
CREATE INDEX IF NOT EXISTS idx_psr_failed
    ON pipeline_source_runs (started_at DESC)
    WHERE status = 'failed';

COMMENT ON TABLE pipeline_source_runs IS
    'Per-source breakdown of a pipeline run. Child of pipeline_runs(id).';
