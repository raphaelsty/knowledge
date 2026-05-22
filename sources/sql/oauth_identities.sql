-- OAuth-verified third-party identities.
--
-- One row per (provider, provider_user_id) the user has proven they
-- own via the provider's OAuth flow. Distinct from `users.sources`,
-- which is user-typed and unverifiable — a row here means the
-- provider itself confirmed the linkage at sign-in time.
--
-- Today the only provider is "github". The schema is provider-keyed
-- so adding Google / GitLab / etc. doesn't require a new table.

CREATE TABLE IF NOT EXISTS oauth_identities (
    id                BIGSERIAL PRIMARY KEY,
    provider          TEXT      NOT NULL,
    -- The provider's stable numeric (or string) account id. GitHub
    -- exposes it as `id`; surviving a username rename is the whole
    -- point of storing it instead of the login.
    provider_user_id  TEXT      NOT NULL,
    -- The provider's current login (GitHub: `login`). Stored only for
    -- display + audit trail. Never used as a lookup key — that's what
    -- provider_user_id is for.
    provider_login    TEXT,
    -- Provider-supplied email at sign-in time. Null when the user
    -- has no public email or has scoped down the OAuth grant.
    provider_email    TEXT,
    user_id           BIGINT    NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS oauth_identities_provider_user_uq
    ON oauth_identities (provider, provider_user_id);

CREATE INDEX IF NOT EXISTS oauth_identities_user_id_idx
    ON oauth_identities (user_id);
