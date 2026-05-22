//! Per-user storage stats.
//!
//! Lives in its own module so the storage-billing surface stays
//! decoupled from `credits.rs` (ledger) and `sponsorships.rs`
//! (credit-spending). The route registrations in `main.rs` are the
//! only file that knows all three.
//!
//! Routes:
//!   GET  /api/me/storage          — return cached snapshot + rates
//!   POST /api/me/storage/refresh  — recompute from PG + disk, upsert
//!
//! The snapshot is materialised in `user_storage` because the
//! per-document `pg_column_size` SUM and the filesystem walk of the
//! personal index directory are too expensive to do on every page
//! load. The pipeline (`run_pipeline`) bills storage based on the
//! credit-billing rates in `sources/storage.py`; this handler
//! mirrors those constants so the frontend can show consistent
//! projected costs without an extra round-trip.
//!
//! NOTE: rate constants here must stay in sync with
//! `sources/storage.py`. They're duplicated rather than served
//! because the values are part of the v1 contract and the Python
//! pipeline already hard-codes them.

use std::path::Path;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use serde::Serialize;

use crate::handlers::auth::current_user;
use crate::state::AppState;

// Must match sources/storage.py. Linear pricing — see that file
// for the derivation against Hetzner Volume cost.
const FREE_DOCS: i64 = 1_000;
const BYTES_PER_DOC: i64 = 20_000;
const USD_PER_DOC_PER_MONTH: f64 = 0.000_01;
const USD_PER_CREDIT: f64 = 0.01;
const BILLING_PERIOD_DAYS: i32 = 30;

#[derive(Serialize)]
pub struct StorageSnapshot {
    /// Number of `documents` rows owned by the user.
    #[serde(rename = "docCount")]
    pub doc_count: i64,
    /// Sum of `pg_column_size(d.*)` across the user's documents.
    #[serde(rename = "dbBytes")]
    pub db_bytes: i64,
    /// Size of `indexes/{users.index_name}/` on disk.
    #[serde(rename = "indexBytes")]
    pub index_bytes: i64,
    /// db_bytes + index_bytes.
    #[serde(rename = "totalBytes")]
    pub total_bytes: i64,
    /// Last successful refresh (ISO-8601 UTC). None when the user
    /// has never been refreshed.
    #[serde(rename = "updatedAt", skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Credits owed for the *next* billing period at the current
    /// doc count (0 if user is under free quota or is a VIP).
    #[serde(rename = "projectedCreditsPerMonth")]
    pub projected_credits_per_month: i32,
    /// Reflects the rate sheet that produced
    /// `projected_credits_per_month`. The frontend renders this
    /// so the user understands why the number is what it is.
    pub rates: RateInfo,
}

#[derive(Serialize)]
pub struct RateInfo {
    #[serde(rename = "freeDocs")]
    pub free_docs: i64,
    #[serde(rename = "bytesPerDoc")]
    pub bytes_per_doc: i64,
    /// Billed rate in USD per document per month above the free
    /// quota. Linear — no tiers.
    #[serde(rename = "usdPerDocPerMonth")]
    pub usd_per_doc_per_month: f64,
    #[serde(rename = "billingPeriodDays")]
    pub billing_period_days: i32,
}

fn rate_info() -> RateInfo {
    RateInfo {
        free_docs: FREE_DOCS,
        bytes_per_doc: BYTES_PER_DOC,
        usd_per_doc_per_month: USD_PER_DOC_PER_MONTH,
        billing_period_days: BILLING_PERIOD_DAYS,
    }
}

/// Linear storage cost in credits per BILLING_PERIOD_DAYS.
/// Mirror of `storage_credits()` in sources/storage.py.
fn projected_credits(doc_count: i64) -> i32 {
    if doc_count <= FREE_DOCS {
        return 0;
    }
    let surplus = doc_count - FREE_DOCS;
    let usd = (surplus as f64) * USD_PER_DOC_PER_MONTH;
    let credits = (usd / USD_PER_CREDIT).ceil() as i32;
    credits.max(1)
}

/// Recursively sum the size of every regular file under `root`.
/// Errors (missing dir, permission, race against a rebuild) collapse
/// to 0 — the user just sees their index as 0 bytes until the next
/// refresh lands.
fn dir_size_bytes(root: &Path) -> u64 {
    fn walk(p: &Path, acc: &mut u64) {
        let Ok(entries) = std::fs::read_dir(p) else {
            return;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                walk(&entry.path(), acc);
            } else if meta.is_file() {
                *acc += meta.len();
            }
        }
    }
    if !root.exists() {
        return 0;
    }
    let mut total = 0u64;
    walk(root, &mut total);
    total
}

/// GET /api/me/storage
///
/// Reads the cached row from `user_storage`. Returns zeros when no
/// snapshot exists yet — the caller is expected to POST /refresh on
/// first load (the frontend does this on mount).
pub async fn get_storage(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let Some(pool) = state.pg_pool.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "no database").into_response();
    };
    let Some(me) = current_user(pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let row: Option<(i32, i64, i64, String)> = sqlx::query_as(
        "SELECT doc_count, db_bytes, index_bytes,
                to_char(updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"')
           FROM user_storage WHERE user_id = $1",
    )
    .bind(me.id)
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    let (doc_count, db_bytes, index_bytes, updated_at) = match row {
        Some((d, b, i, t)) => (d as i64, b, i, Some(t)),
        None => (0, 0, 0, None),
    };
    let total = db_bytes + index_bytes;
    Json(StorageSnapshot {
        doc_count,
        db_bytes,
        index_bytes,
        total_bytes: total,
        updated_at,
        projected_credits_per_month: projected_credits(doc_count),
        rates: rate_info(),
    })
    .into_response()
}

/// POST /api/me/storage/refresh
///
/// Computes fresh values and upserts the snapshot. Cheaper than
/// running on every page load, but still O(rows) for `pg_column_size`
/// — the frontend should call this on explicit user action, not on
/// every navigation.
pub async fn refresh_storage(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let Some(pool) = state.pg_pool.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "no database").into_response();
    };
    let Some(me) = current_user(pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    // 1. Postgres footprint — row count + payload bytes.
    let agg: Option<(i64, i64)> = sqlx::query_as(
        "SELECT count(*)::bigint,
                COALESCE(SUM(pg_column_size(d.*))::bigint, 0::bigint)
           FROM documents d WHERE user_id = $1",
    )
    .bind(me.id)
    .fetch_one(pool)
    .await
    .ok();
    let (doc_count, db_bytes) = agg.unwrap_or((0, 0));

    // 2. Personal index footprint — walk `indexes/{index_name}/`.
    let index_name: Option<String> =
        sqlx::query_scalar("SELECT index_name FROM users WHERE id = $1")
            .bind(me.id)
            .fetch_one(pool)
            .await
            .ok();
    let index_bytes = match index_name.as_deref() {
        Some(name) if !name.is_empty() => {
            let path = state.config.index_dir.join(name);
            dir_size_bytes(&path) as i64
        }
        _ => 0,
    };

    // 3. Upsert.
    let _ = sqlx::query(
        "INSERT INTO user_storage (user_id, doc_count, db_bytes, index_bytes, updated_at)
         VALUES ($1, $2::int, $3, $4, now())
         ON CONFLICT (user_id) DO UPDATE
            SET doc_count = EXCLUDED.doc_count,
                db_bytes  = EXCLUDED.db_bytes,
                index_bytes = EXCLUDED.index_bytes,
                updated_at  = now()",
    )
    .bind(me.id)
    .bind(doc_count as i32)
    .bind(db_bytes)
    .bind(index_bytes)
    .execute(pool)
    .await;

    let total = db_bytes + index_bytes;
    Json(StorageSnapshot {
        doc_count,
        db_bytes,
        index_bytes,
        total_bytes: total,
        updated_at: Some(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        projected_credits_per_month: projected_credits(doc_count),
        rates: rate_info(),
    })
    .into_response()
}
