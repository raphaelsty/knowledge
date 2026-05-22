-- Per-user record of the twitter feeder's last attempt. Used by the
-- queue endpoint to demote slugs we just touched and to back off on
-- accounts that consistently fail (deleted handle, locked, etc.) so
-- we don't retry them on every pass.
--
-- One row per user (PK on user_id). Updated in place — no audit
-- log, no per-attempt history. The fields are deliberately minimal:
--
--   • last_attempt_at  → drives both the today-demote in the queue
--                        and the cooldown math below.
--   • last_status      → 'ok' | 'up_to_date' | 'user_fault' | 'error'
--                        | 'api_unavailable'. Short string, no enum.
--   • consecutive_failures → reset to 0 on any 'ok' / 'up_to_date'
--                        outcome; incremented on user_fault / error.
--                        Drives exponential backoff: a slug with N
--                        consecutive failures is hidden from the
--                        queue until last_attempt_at + min(30d,
--                        24h × 2^N).
--
-- FK cascade: deleting a user drops their attempt row too.
CREATE TABLE IF NOT EXISTS twitter_feed_attempts (
    user_id              BIGINT      PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    last_attempt_at      TIMESTAMPTZ NOT NULL,
    last_status          TEXT        NOT NULL DEFAULT 'unknown',
    consecutive_failures INTEGER     NOT NULL DEFAULT 0
);

COMMENT ON TABLE  twitter_feed_attempts          IS 'One row per VIP twitter handle; the feeder writes here after every slug attempt. Drives queue priority + failure backoff.';
COMMENT ON COLUMN twitter_feed_attempts.last_attempt_at      IS 'When the feeder last tried this slug (success or failure).';
COMMENT ON COLUMN twitter_feed_attempts.last_status          IS 'Outcome of the last attempt: ok | up_to_date | user_fault | error | api_unavailable.';
COMMENT ON COLUMN twitter_feed_attempts.consecutive_failures IS 'Count of consecutive non-success attempts. Resets to 0 on the next ok/up_to_date. Drives 24h × 2^N exponential backoff (capped at 30 days).';
