-- Schema for the `auth_sessions` table.
--
-- Server-side session store for GitHub-OAuth-authenticated users. One
-- row per active browser session. The cookie sent to the client carries
-- only the opaque `id` — user_id and expiry are looked up here on every
-- request, so revoking a session is a single DELETE.
--
-- This is separate from the analytics `sessions` table (which tracks
-- anonymous visitor state). They have orthogonal lifetimes and privacy
-- requirements; don't merge them.

CREATE TABLE IF NOT EXISTS auth_sessions (
    -- 256-bit random token, hex-encoded. Never reused; never logged.
    id          TEXT        PRIMARY KEY,

    user_id     BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Informational: helps the user spot unfamiliar sessions in a
    -- "Devices signed in" UI. Nullable since UA headers can be absent.
    user_agent  TEXT,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Sessions are sliding-window: each authenticated request extends
    -- expires_at by the configured TTL (default 30 days). A session
    -- that hasn't been touched in TTL is effectively logged out.
    expires_at  TIMESTAMPTZ NOT NULL
);

-- Fetching every session for a given user (revoke-all flow).
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
    ON auth_sessions (user_id);

-- Sweep expired sessions (partial index — only rows worth looking at).
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at
    ON auth_sessions (expires_at);

COMMENT ON TABLE auth_sessions IS
    'Server-side session store for GitHub-OAuth users. The browser cookie holds only the opaque id.';
COMMENT ON COLUMN auth_sessions.id         IS '256-bit random token, hex-encoded. Opaque to the client; never logged.';
COMMENT ON COLUMN auth_sessions.user_id    IS 'FK to users.id. ON DELETE CASCADE so deleting a user logs them out everywhere.';
COMMENT ON COLUMN auth_sessions.user_agent IS 'User-Agent header at session creation. Informational only.';
COMMENT ON COLUMN auth_sessions.created_at IS 'When the session was minted (first successful OAuth exchange).';
COMMENT ON COLUMN auth_sessions.expires_at IS 'Sliding expiry. Each authenticated request pushes this forward by TTL.';
