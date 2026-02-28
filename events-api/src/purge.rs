use sqlx::PgPool;

/// Deletes events older than `retention_days` from the database.
pub async fn purge_old_events(pool: &PgPool, retention_days: i32) -> Result<u64, sqlx::Error> {
    let result = sqlx::query(&format!(
        "DELETE FROM events WHERE created_at < now() - interval '{} days'",
        retention_days
    ))
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}
