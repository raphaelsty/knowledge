-- Schema for `export_downloads`.
--
-- One row per `GET /api/personalities/{slug}/export.jsonl` call that
-- passes auth + access checks. Lets the platform answer "who exported
-- what, when, and at what scope" without having to scrape API logs.
--
-- Inserted by `api::handlers::exports::export_personality` *before*
-- the JSONL stream starts. Rows therefore represent "downloads that
-- the server agreed to serve"; a client that drops the connection
-- mid-stream still leaves a row, which is the right semantics for
-- abuse-tracking and quota work.
--
-- Date range columns (`date_from` / `date_to`) mirror the optional
-- query params on the export endpoint. NULL = "no lower / upper
-- bound" = "everything".

CREATE TABLE IF NOT EXISTS export_downloads (
    id              BIGSERIAL    PRIMARY KEY,
    user_id         BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_user_id  BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_count       BIGINT       NOT NULL DEFAULT 0,
    date_from       DATE,
    date_to         DATE,
    downloaded_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- "Recent exports by this user" is the hot read path (account history,
-- abuse review). DESC on the timestamp keeps the index aligned with
-- the most common scan direction.
CREATE INDEX IF NOT EXISTS idx_export_downloads_user_time
    ON export_downloads (user_id, downloaded_at DESC);

-- "Who's been exporting MY library lately?" — symmetric index for
-- the personality owner's audit view.
CREATE INDEX IF NOT EXISTS idx_export_downloads_target_time
    ON export_downloads (target_user_id, downloaded_at DESC);

COMMENT ON TABLE export_downloads IS
    'Audit log of /api/personalities/{slug}/export.jsonl requests served.';
COMMENT ON COLUMN export_downloads.user_id        IS 'Caller (signed-in account that initiated the export).';
COMMENT ON COLUMN export_downloads.target_user_id IS 'Personality whose library was exported.';
COMMENT ON COLUMN export_downloads.doc_count      IS 'Document rows the server agreed to stream (after date + limit filters).';
COMMENT ON COLUMN export_downloads.date_from      IS 'Inclusive lower bound on document date; NULL means no lower bound.';
COMMENT ON COLUMN export_downloads.date_to        IS 'Inclusive upper bound on document date; NULL means no upper bound.';
