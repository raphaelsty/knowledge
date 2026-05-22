//! Shared PostgreSQL pool helpers.
//!
//! Schema (`users`, `documents`, `sessions`, `events`, views) lives in
//! `sources/sql/` and is applied via `run.py` at pipeline bootstrap. The
//! Rust API no longer owns any DDL.

use sqlx::PgPool;

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
