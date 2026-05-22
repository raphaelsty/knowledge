-- HackerNews front-page picks.
--
-- Three tables, all global (not user-scoped at the row level for the
-- first two — only `hn_user_picks` joins on user_id):
--
--   hn_frontpage_runs   one row per snapshot of the front page
--   hn_frontpage_items  the ~30 articles in each snapshot
--   hn_user_picks       per-user relevance scores, scoped to a run
--
-- Read path for the feed: pick the most-recent `hn_frontpage_runs.id`,
-- then SELECT items joined to `hn_user_picks` filtered on that run +
-- the caller's user_id. The "only the latest front page" requirement
-- falls out naturally from `run_id = (SELECT MAX(id) FROM
-- hn_frontpage_runs)`.

CREATE TABLE IF NOT EXISTS hn_frontpage_runs (
    id         BIGSERIAL   PRIMARY KEY,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_items    INTEGER     NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hn_frontpage_runs_fetched_at
    ON hn_frontpage_runs (fetched_at DESC);

CREATE TABLE IF NOT EXISTS hn_frontpage_items (
    run_id        BIGINT      NOT NULL REFERENCES hn_frontpage_runs(id) ON DELETE CASCADE,
    hn_id         BIGINT      NOT NULL,
    rank          INTEGER     NOT NULL,
    url           TEXT        NOT NULL,
    title         TEXT        NOT NULL,
    summary       TEXT        NOT NULL DEFAULT '',
    points        INTEGER     NOT NULL DEFAULT 0,
    num_comments  INTEGER     NOT NULL DEFAULT 0,
    submitted_at  TIMESTAMPTZ,
    author        TEXT        NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, hn_id)
);

CREATE INDEX IF NOT EXISTS idx_hn_frontpage_items_url ON hn_frontpage_items (url);

CREATE TABLE IF NOT EXISTS hn_user_picks (
    user_id    BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    run_id     BIGINT      NOT NULL REFERENCES hn_frontpage_runs(id) ON DELETE CASCADE,
    hn_id      BIGINT      NOT NULL,
    score      REAL        NOT NULL,
    rank       INTEGER     NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, run_id, hn_id),
    FOREIGN KEY (run_id, hn_id) REFERENCES hn_frontpage_items(run_id, hn_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_hn_user_picks_user_run
    ON hn_user_picks (user_id, run_id, rank);

COMMENT ON TABLE hn_frontpage_runs IS
    'One snapshot of the HackerNews front page (run once per day by scripts/hn_frontpage.py).';
COMMENT ON TABLE hn_frontpage_items IS
    'The ~30 articles in a given front-page snapshot. Global, not per-user.';
COMMENT ON TABLE hn_user_picks IS
    'Per-user relevance scores for HN front-page items. Feed shows rows where run_id = latest run.';
