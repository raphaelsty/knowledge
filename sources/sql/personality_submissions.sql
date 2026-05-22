-- ── personality_submissions ─────────────────────────────────────────────
--
-- User-submitted suggestions for new VIP personalities. The owner of
-- the project reviews each row and either promotes it to a real
-- `users` row (manually, off-line) or rejects it. We deliberately do
-- *not* create a user / spawn a pipeline run from this table — that
-- would let anyone provision compute by filling a form.
--
-- Layout:
--   • All the original handles from the "Add a personality" form
--     are stored as plain strings. Empty handles stay empty (we don't
--     enforce presence — the admin reviews + completes them).
--   • `status` tracks the lifecycle. `pending` is the default; an
--     admin flips it to `approved` after integrating the suggestion,
--     or `rejected` with a `notes` reason for the audit log.
--   • `submitter_id` records who sent the suggestion in. NULL is
--     allowed in case we later accept anonymous submissions; the
--     POST handler requires auth today.
--
-- This table is independent from `users` so deleting a user keeps
-- the historical record of what they submitted.

CREATE TABLE IF NOT EXISTS personality_submissions (
    id            BIGSERIAL   PRIMARY KEY,
    submitter_id  BIGINT      REFERENCES users(id) ON DELETE SET NULL,

    name          TEXT        NOT NULL,
    slug          TEXT        NOT NULL,
    description   TEXT        NOT NULL DEFAULT '',

    twitter_handle      TEXT NOT NULL DEFAULT '',
    github_handle       TEXT NOT NULL DEFAULT '',
    huggingface_handle  TEXT NOT NULL DEFAULT '',
    reddit_handle       TEXT NOT NULL DEFAULT '',
    hackernews_handle   TEXT NOT NULL DEFAULT '',
    stackoverflow_user_id TEXT NOT NULL DEFAULT '',
    arxiv_author        TEXT NOT NULL DEFAULT '',
    dblp_author         TEXT NOT NULL DEFAULT '',
    scholar_user_id     TEXT NOT NULL DEFAULT '',
    websites            TEXT NOT NULL DEFAULT '',

    status        TEXT        NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'approved', 'rejected')),
    notes         TEXT        NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at   TIMESTAMPTZ
);

-- Default review queue: oldest pending first.
CREATE INDEX IF NOT EXISTS idx_personality_submissions_pending
    ON personality_submissions (created_at)
    WHERE status = 'pending';

-- Per-submitter lookup ("what have I sent in?"). Most users will only
-- submit a handful, so an index over `submitter_id` alone is enough.
CREATE INDEX IF NOT EXISTS idx_personality_submissions_submitter
    ON personality_submissions (submitter_id, created_at DESC);

COMMENT ON TABLE personality_submissions IS
    'User-submitted personality suggestions. Reviewed manually by an admin before any VIP user row is created.';
