-- ── View: user_source_counts ────────────────────────────────────────────
--
-- Per-user breakdown of documents by `source` type. Called from the
-- frontend on every personality page load, so we keep it fast:
--
--   * Backed by `idx_documents_user_source` (composite btree on
--     (user_id, source)) — the GROUP BY resolves via an index-only
--     scan, so the query returns in microseconds even with millions
--     of documents.
--   * Ordering by count is left to the caller; the view is intentionally
--     a thin aggregate so filters (e.g. `WHERE user_id = $1`) can push
--     down cleanly.
--
-- Typical usage:
--   SELECT source, count FROM user_source_counts
--    WHERE user_id = $1
--    ORDER BY count DESC;

CREATE OR REPLACE VIEW user_source_counts AS
SELECT user_id,
       source,
       count(*) AS count
  FROM documents
 WHERE deleted = FALSE
 GROUP BY user_id, source;

COMMENT ON VIEW user_source_counts IS
    'Per-user document count broken down by source type. Backed by idx_documents_user_source.';
