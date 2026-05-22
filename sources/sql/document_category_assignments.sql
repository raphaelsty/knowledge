-- Junction table: which documents were assigned which fine-grained
-- categories from the document_categories catalogue. Populated by
-- sources/utils/categorize.py.
--
-- One row per (document, category) assignment. A document gets 1-3
-- rows depending on how cleanly it fits a single bucket. The first
-- assignment (is_primary = TRUE) is the model's top pick; the rest
-- are secondary slugs that materially apply.
--
-- score is the LLM's self-reported confidence (0-1). We persist it
-- so the UI can de-prioritise low-confidence assignments or fall
-- back to the daemon's free-text title when nothing scored well.

CREATE TABLE IF NOT EXISTS document_category_assignments (
    user_id      BIGINT      NOT NULL,
    url          TEXT        NOT NULL,
    category_id  BIGINT      NOT NULL REFERENCES document_categories(id) ON DELETE CASCADE,
    score        REAL        NOT NULL DEFAULT 0,
    is_primary   BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, url, category_id),
    FOREIGN KEY (user_id, url)
        REFERENCES documents(user_id, url) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dca_category
    ON document_category_assignments(category_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dca_doc
    ON document_category_assignments(user_id, url);
-- Partial index so "primary category for these docs" lookups
-- (timeline filter) don't scan the secondary rows.
CREATE INDEX IF NOT EXISTS idx_dca_primary
    ON document_category_assignments(user_id, url) WHERE is_primary;

COMMENT ON TABLE  document_category_assignments IS 'M:N between documents and document_categories. Populated by sources/utils/categorize.py.';
COMMENT ON COLUMN document_category_assignments.score      IS 'LLM self-reported confidence (0-1) that the chosen slug fits.';
COMMENT ON COLUMN document_category_assignments.is_primary IS 'TRUE for the top pick from the LLM; secondary slugs are FALSE.';
