-- Schema for the `users` table.
--
-- One row per account, both human-signed-up users and seeded VIP
-- personalities. Each user owns exactly one library —
-- `documents.user_id` FKs into `id`.
--
-- Sparse / nested fields (`links`, `sources`) are stored as JSONB because
-- their keys vary heavily across personalities and new keys appear over
-- time (e.g. `bluesky`, `linkedin` in the long tail of `links`).

CREATE TABLE IF NOT EXISTS users (
    id           BIGSERIAL PRIMARY KEY,

    -- ── Credentials / auth ──────────────────────────────────────────────
    -- `username` is the URL-safe slug. New accounts authenticate with
    -- email + password. Seeded VIP personalities may have NULL
    -- email/password_hash until someone claims them.
    username       TEXT        NOT NULL UNIQUE,
    email          TEXT        UNIQUE,
    password_hash  TEXT,

    -- Email verification gate. Unverified accounts can read but cannot
    -- save, follow, or favorite. Flipped TRUE once the user follows the
    -- magic link sent to `email` (see auth handler).
    email_verified                 BOOLEAN     NOT NULL DEFAULT FALSE,
    email_verification_token       TEXT        UNIQUE,
    email_verification_expires_at  TIMESTAMPTZ,

    -- Password reset. Token is single-use and lives in this table
    -- (no separate reset table); cleared on consume or expiry.
    password_reset_token       TEXT        UNIQUE,
    password_reset_expires_at  TIMESTAMPTZ,

    -- Public libraries are visible to anonymous visitors. Private
    -- libraries require authentication to read.
    public       BOOLEAN     NOT NULL DEFAULT TRUE,

    -- ── Profile ─────────────────────────────────────────────────────────
    -- `name` is the display name ("Raphael Sourty"); `username` is the
    -- URL-safe slug ("raphael-sourty"). Keep both so URLs and display
    -- can diverge.
    name         TEXT        NOT NULL,
    description  TEXT        NOT NULL DEFAULT '',

    -- Avatar is optional — ~24% of personalities have no avatar set.
    avatar       TEXT,

    -- Search index name. Usually equals `username` but kept explicit so
    -- a user can point their library at a shared/aliased index without
    -- renaming the account.
    index_name   TEXT        NOT NULL,

    -- External profile links. Sparse: only `website`, `twitter`, `github`
    -- are common; the rest (linkedin, huggingface, bluesky, lab pages…)
    -- show up on < 1% of accounts. JSONB keeps the schema flexible and
    -- lets us query by provider with the GIN index below.
    --   e.g. {"github": "https://github.com/karpathy", "twitter": "..."}
    links        JSONB       NOT NULL DEFAULT '{}',

    -- Per-source fetcher configuration — deeply nested and varies per
    -- user (github user list, twitter handle + filters, blog feeds,
    -- sitemaps, youtube channels, etc.). Normalizing this into columns
    -- would fight the data; JSONB + GIN is the right shape.
    sources      JSONB       NOT NULL DEFAULT '{}',

    -- Raw social signals. Nullable — NULL means "not yet fetched"; the
    -- pipeline (`sources/utils/popularity`) populates each one on the
    -- first run that sees NULL. The frontend combines them on a log
    -- scale to rank personalities within a category.
    twitter_followers INTEGER,
    github_followers  INTEGER,
    citations         INTEGER,

    -- ── Twitter incremental cursor ──────────────────────────────────────
    -- Populated by the pipeline after each Twitter fetch; consumed on the
    -- next run to early-stop pagination once we reach tweets we already
    -- have. Both nullable until the first successful fetch.
    tweet_newest_date DATE,
    tweet_oldest_date DATE,

    -- ── Timestamps ──────────────────────────────────────────────────────
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migration for DBs that predate the cursor columns.
ALTER TABLE users ADD COLUMN IF NOT EXISTS tweet_newest_date DATE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS tweet_oldest_date DATE;

-- Idempotent migration for DBs that predate the social-count columns.
ALTER TABLE users DROP COLUMN IF EXISTS popularity;
ALTER TABLE users ADD COLUMN IF NOT EXISTS twitter_followers INTEGER;
ALTER TABLE users ADD COLUMN IF NOT EXISTS github_followers  INTEGER;
ALTER TABLE users ADD COLUMN IF NOT EXISTS citations         INTEGER;

-- Denormalised count of non-deleted documents owned by this user.
-- Refreshed hourly by the `knowledge-feed-snapshot` daemon (same one
-- that builds feed_snapshot + personal_snapshot). The /api/users
-- handler reads this directly so the response stays under 200 ms;
-- previously it ran a 450-way LATERAL `count(*)` against documents
-- (~4 s on the prod corpus) and was the right-rail bottleneck.
-- A few-minute staleness on the count is invisible to the UI — the
-- rail sorts on Twitter/GitHub followers + citations anyway, and
-- the count is just a label under the avatar.
ALTER TABLE users ADD COLUMN IF NOT EXISTS document_count BIGINT NOT NULL DEFAULT 0;

-- ── Migration: email/password authentication (replaces GitHub OAuth) ───
-- Add the new auth columns first…
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash                 TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified                BOOLEAN     NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_token      TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_expires_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_token          TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_expires_at     TIMESTAMPTZ;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'users_email_verification_token_key'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT users_email_verification_token_key UNIQUE (email_verification_token);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'users_password_reset_token_key'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT users_password_reset_token_key UNIQUE (password_reset_token);
    END IF;
END$$;

-- …then drop the legacy GitHub-OAuth columns and the old plain-text
-- `password` column. Idempotent: only drops if they exist.
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_github_id_key;
ALTER TABLE users DROP COLUMN     IF EXISTS github_id;
ALTER TABLE users DROP COLUMN     IF EXISTS github_login;
ALTER TABLE users DROP COLUMN     IF EXISTS password;
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

-- VIP flag. Defaults to false for new sign-ups. Existing users at
-- the time the column was first added were grandfathered to true
-- (one-shot UPDATE in the live DB).
ALTER TABLE users ADD COLUMN IF NOT EXISTS vip BOOLEAN NOT NULL DEFAULT FALSE;

-- Who paid to add this personality? NULL for users created via the
-- normal signup flow (or grandfathered in). Populated by the
-- POST /api/personalities handler so we can show "personalities
-- you've added" in the settings page and audit who's funding what.
-- ON DELETE SET NULL so a sponsor closing their account doesn't
-- cascade-delete the libraries they paid to create.
ALTER TABLE users ADD COLUMN IF NOT EXISTS sponsored_by BIGINT REFERENCES users(id) ON DELETE SET NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS sponsored_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_users_sponsored_by ON users(sponsored_by) WHERE sponsored_by IS NOT NULL;

-- ── Indices ─────────────────────────────────────────────────────────────
-- `public` is the only column-level filter predicate left on the
-- personality list. Topical grouping lives in `user_categories` (see
-- categories.sql); the legacy single-string `users.category` column
-- and its index were dropped in the categories ontology migration.
CREATE INDEX IF NOT EXISTS idx_users_public ON users (public);
DROP INDEX  IF EXISTS idx_users_category;
ALTER TABLE users DROP COLUMN IF EXISTS category;

-- Partial index: the welcome grid + library picker both gate on
-- `vip = TRUE`, but the table can hold 100k+ rows and only ~133
-- are flagged. A partial b-tree on the truthy subset keeps the
-- scan O(matching rows) instead of O(table) and lets PG plan a
-- pure index lookup. Cheap to maintain because non-vip rows never
-- touch the index.
CREATE INDEX IF NOT EXISTS idx_users_vip ON users (vip) WHERE vip;

-- Lookup by lower(email) on login. Sparse because seeded personalities
-- have NULL email until claimed.
CREATE INDEX IF NOT EXISTS idx_users_email_lower
    ON users (lower(email)) WHERE email IS NOT NULL;

-- GIN indices let us query JSONB payloads cheaply, e.g.
--   SELECT id FROM users WHERE links ? 'github';
--   SELECT id FROM users WHERE sources @> '{"twitter": {}}'::jsonb;
CREATE INDEX IF NOT EXISTS idx_users_links_gin   ON users USING GIN (links);
CREATE INDEX IF NOT EXISTS idx_users_sources_gin ON users USING GIN (sources);

-- ── DB-level documentation (visible via \d+ and pg_description) ─────────
COMMENT ON TABLE users IS
    'Accounts. One row per personality. Holds auth fields plus the profile + sources config that drives the pipeline.';

COMMENT ON COLUMN users.id          IS 'Surrogate primary key. Referenced by documents.user_id.';
COMMENT ON COLUMN users.username    IS 'URL-safe slug. Unique. Used as the personal-page route segment.';
COMMENT ON COLUMN users.email       IS 'Contact email. Required for self-signup, NULL for seeded VIP personalities until claimed.';
COMMENT ON COLUMN users.password_hash IS 'Argon2id password hash (PHC string). NULL for seeded personalities that have not been claimed.';
COMMENT ON COLUMN users.email_verified IS 'TRUE once the user has clicked the verification link emailed to `email`. Unverified accounts are read-only.';
COMMENT ON COLUMN users.email_verification_token IS 'Single-use opaque token sent in the verification email. NULL when no verification is pending.';
COMMENT ON COLUMN users.email_verification_expires_at IS 'Expiry for the active verification token (24h from issue). NULL when no verification is pending.';
COMMENT ON COLUMN users.password_reset_token IS 'Single-use opaque token sent in the password-reset email. NULL when no reset is pending.';
COMMENT ON COLUMN users.password_reset_expires_at IS 'Expiry for the active password-reset token (1h from issue). NULL when no reset is pending.';
COMMENT ON COLUMN users.public      IS 'True = library visible to anonymous visitors; false = auth required.';
COMMENT ON COLUMN users.name        IS 'Display name, e.g. "Raphael Sourty".';
COMMENT ON COLUMN users.description IS 'Short bio shown on the personality card.';
COMMENT ON COLUMN users.avatar      IS 'Avatar URL. Nullable — ~24% of accounts have none.';
COMMENT ON COLUMN users.index_name  IS 'Search index name. Usually equals username; kept explicit to allow aliasing.';
COMMENT ON COLUMN users.links       IS 'External profile links as JSONB, e.g. {"github":"...","twitter":"..."}. Sparse schema.';
COMMENT ON COLUMN users.sources     IS 'Per-source fetcher config as JSONB (github users, twitter filters, blog feeds, sitemaps, youtube channels, …).';
COMMENT ON COLUMN users.twitter_followers IS 'Raw Twitter/X follower count. NULL = not yet fetched. Populated lazily by the pipeline via api.twitterapi.io.';
COMMENT ON COLUMN users.github_followers  IS 'Raw GitHub follower count. NULL = not yet fetched. Populated lazily by the pipeline via api.github.com.';
COMMENT ON COLUMN users.citations         IS 'Total citation count (Semantic Scholar). Optional — NULL when the personality has no scholar profile or the lookup failed.';
COMMENT ON COLUMN users.tweet_newest_date IS 'Date of the most-recent tweet ingested. Used as the next run''s stop_date fence.';
COMMENT ON COLUMN users.tweet_oldest_date IS 'Date of the oldest tweet ingested. Bounds how far back a backfill can reach.';
COMMENT ON COLUMN users.created_at  IS 'Row creation timestamp.';
COMMENT ON COLUMN users.updated_at  IS 'Last mutation timestamp. Updated by the application, not by a trigger.';
