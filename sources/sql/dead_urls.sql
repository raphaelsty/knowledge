-- Schema for the `dead_urls` table.
--
-- Records URLs that failed the dead-link probe so the pipeline can
-- skip re-fetching them on subsequent runs. Without this, every run
-- re-discovers the same broken DBLP/Wikipedia/etc URLs as "new",
-- re-probes them, and re-discards them. Global (not user-scoped):
-- a URL that 404s for one user 404s for everyone.

CREATE TABLE IF NOT EXISTS dead_urls (
    url        TEXT        PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dead_urls_checked_at
    ON dead_urls (checked_at);

COMMENT ON TABLE dead_urls IS
    'URLs the dead-link probe rejected. Used to short-circuit refetching across runs.';
COMMENT ON COLUMN dead_urls.url        IS 'Canonical URL that failed the probe.';
COMMENT ON COLUMN dead_urls.checked_at IS 'When the URL was last confirmed dead. Useful for periodic re-checks.';
