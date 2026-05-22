-- Sponsor-a-VIP queue.
--
-- Anyone with credits can submit a request to onboard a new person
-- as a VIP. The request debits credits at submission time. The
-- operator (or eventually an automated pipeline) reviews and either
-- approves (the candidate becomes a VIP) or rejects (credits are
-- refunded via a positive credit_event with a `refund` kind).
--
-- Kept in its own file because it's a credit-spending feature
-- distinct from the ledger itself.

CREATE TABLE IF NOT EXISTS vip_sponsorships (
    id                BIGSERIAL  PRIMARY KEY,
    user_id           BIGINT     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Candidate description fields. We don't link to a `users` row
    -- yet — the candidate might not exist in PG at submission time.
    -- The operator creates the row + flips `vip = TRUE` on approval.
    candidate_name    TEXT       NOT NULL,
    candidate_url     TEXT       NOT NULL,
    candidate_note    TEXT       NOT NULL DEFAULT '',
    -- Credit amount the user paid at submission time. Persisted
    -- here even though it's also visible in credit_events, so a
    -- pricing change later doesn't retroactively mutate the refund
    -- amount on pending rows.
    credits_paid      INTEGER    NOT NULL,
    -- Lifecycle: pending → approved | rejected.
    status            TEXT       NOT NULL DEFAULT 'pending'
                                 CHECK (status IN ('pending', 'approved', 'rejected')),
    -- When approved, points at the resulting users.id (so the VIP
    -- catalogue can be cross-referenced). NULL otherwise.
    resolved_user_id  BIGINT     REFERENCES users(id) ON DELETE SET NULL,
    -- Free-text comment the operator can leave on review.
    review_note       TEXT       NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_vip_sponsorships_status
    ON vip_sponsorships (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vip_sponsorships_user
    ON vip_sponsorships (user_id, created_at DESC);

COMMENT ON TABLE  vip_sponsorships IS
    'Submitted requests to onboard a new VIP. Each row debits credits at submission; refunds happen via the ledger when status = rejected.';
COMMENT ON COLUMN vip_sponsorships.candidate_url IS
    'URL the operator uses to verify the candidate (their X profile, personal site, …).';
