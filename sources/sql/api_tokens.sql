-- User-scoped API tokens.
--
-- Each row authorizes uploads (and any future bearer-auth endpoint)
-- for a specific user. The plaintext token is shown to the user
-- ONCE at creation time and never stored — only its sha256 hash
-- and the leading 8-char prefix (for display) live here.
--
-- Format of the plaintext (server-generated):
--   kn_<base64url(32 random bytes)>
--   → 32 byte secret = 256 bits of entropy
--
-- Auth flow:
--   1. Client sends `Authorization: Bearer kn_xxxxx`.
--   2. Server hashes the value with sha256 → hex.
--   3. SELECT user_id FROM api_tokens WHERE token_hash = $1
--        AND revoked_at IS NULL.
--   4. UPDATE last_used_at = now() (best-effort, fire-and-forget).
--
-- Revocation: setting revoked_at takes effect on the next request.
-- We never DELETE so the audit trail (who created what, when) lives
-- forever — the unique index is partial on revoked_at IS NULL so a
-- revoked-then-recreated token doesn't collide.

CREATE TABLE IF NOT EXISTS api_tokens (
    id            BIGSERIAL   PRIMARY KEY,
    user_id       BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Human-readable label the user picked at creation ("My laptop",
    -- "MCP — Cursor", etc.). Free-form, capped at 80 chars by the
    -- handler.
    name          TEXT        NOT NULL,
    -- SHA-256(plaintext_token), hex-encoded. Lookups hash the inbound
    -- bearer value with the same algorithm and equality-compare.
    token_hash    TEXT        NOT NULL,
    -- First N chars of the plaintext, kept for the management UI
    -- ("kn_AbCd…"). Not enough to reconstruct the secret — purely a
    -- visual handle so the user can identify which row to revoke.
    prefix        TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);

-- Auth lookup is the hot path; every bearer request hits this index.
-- Partial: revoked tokens never need to match.
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_tokens_active_hash
    ON api_tokens (token_hash)
    WHERE revoked_at IS NULL;

-- List/manage path — "show me my tokens, newest first".
CREATE INDEX IF NOT EXISTS idx_api_tokens_user_created
    ON api_tokens (user_id, created_at DESC);

COMMENT ON TABLE api_tokens IS
    'User-scoped bearer tokens. Plaintext is never stored — only sha256(token).';
