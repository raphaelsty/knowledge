//! Shared PostgreSQL pool and schema migrations.

use sqlx::PgPool;

/// Run all schema migrations on startup.
/// Creates tables for documents, generated_data, favorites, and events.
pub async fn run_migrations(pool: &PgPool) {
    // Documents table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS documents (
            url         TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT '',
            summary     TEXT NOT NULL DEFAULT '',
            date        DATE,
            tags        TEXT[] NOT NULL DEFAULT '{}',
            extra_tags  TEXT[] NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await
    .expect("Failed to create documents table");

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_documents_date ON documents (date DESC NULLS LAST)",
    )
    .execute(pool)
    .await
    .expect("Failed to create documents date index");

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN (tags)")
        .execute(pool)
        .await
        .expect("Failed to create documents tags index");

    // Generated data table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS generated_data (
            key         TEXT PRIMARY KEY,
            data        JSONB NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await
    .expect("Failed to create generated_data table");

    // Favorites table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS favorites (
            url         TEXT PRIMARY KEY REFERENCES documents(url) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await
    .expect("Failed to create favorites table");

    // Events table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS events (
            id          BIGSERIAL PRIMARY KEY,
            session_id  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            payload     JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await
    .expect("Failed to create events table");

    for stmt in &[
        "CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type)",
        "CREATE INDEX IF NOT EXISTS idx_events_created ON events (created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id)",
    ] {
        sqlx::query(stmt)
            .execute(pool)
            .await
            .expect("Failed to create events index");
    }
}

/// Purge events older than `retention_days`.
pub async fn purge_old_events(pool: &PgPool, retention_days: i32) -> Result<u64, sqlx::Error> {
    let result = sqlx::query(&format!(
        "DELETE FROM events WHERE created_at < now() - interval '{} days'",
        retention_days
    ))
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}
