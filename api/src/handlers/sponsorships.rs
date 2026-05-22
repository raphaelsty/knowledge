//! Sponsor-a-VIP endpoints.
//!
//! Lives in its own module so the credit-spending side of the
//! billing feature is cleanly isolated from the ledger primitives in
//! `credits.rs`. The route registrations in `main.rs` are the only
//! ones that know both modules.
//!
//! Pricing for v1 is hard-coded here — same rationale as the pack
//! catalogue: rarely changes, no need for a config-table round-trip.
//! Move to env / a `credit_prices` table if it starts changing.
//!
//! Routes:
//!   POST /api/me/sponsorships    — submit a new VIP request (debits SPONSOR_COST)
//!   GET  /api/me/sponsorships    — list the caller's submissions

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::handlers::auth::current_user;

/// Credit cost to sponsor a new VIP. 200 credits = $2 at the
/// reference rate (1 credit = $0.01).
pub const SPONSOR_COST: i32 = 200;
/// Hard cap on how many pending requests a single user can carry,
/// stops spam from a logged-in account with credits to burn.
const MAX_PENDING_PER_USER: i64 = 10;

#[derive(Deserialize)]
pub struct CreateSponsorshipRequest {
    #[serde(rename = "candidateName")]
    pub candidate_name: String,
    #[serde(rename = "candidateUrl")]
    pub candidate_url: String,
    #[serde(rename = "candidateNote", default)]
    pub candidate_note: String,
}

#[derive(Serialize, sqlx::FromRow)]
pub struct SponsorshipRow {
    pub id: i64,
    #[serde(rename = "candidateName")]
    pub candidate_name: String,
    #[serde(rename = "candidateUrl")]
    pub candidate_url: String,
    #[serde(rename = "candidateNote")]
    pub candidate_note: String,
    #[serde(rename = "creditsPaid")]
    pub credits_paid: i32,
    pub status: String,
    #[serde(rename = "reviewNote")]
    pub review_note: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "resolvedAt", skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<String>,
}

#[derive(Serialize)]
pub struct CreateSponsorshipResponse {
    pub id: i64,
    /// Caller's new credit balance after the debit.
    pub balance: i32,
}

/// POST /api/me/sponsorships
///
/// Single transaction:
///   1. Debit the caller's credits via `credits_debit`.
///      Returns NULL when balance < SPONSOR_COST — we surface that
///      as 402 Payment Required.
///   2. Insert the sponsorship row.
///
/// Both operations happen inside one txn so a sponsorship is never
/// recorded without a successful debit and vice versa.
pub async fn create(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<CreateSponsorshipRequest>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let name = req.candidate_name.trim();
    let url = req.candidate_url.trim();
    let note = req.candidate_note.trim();
    if name.is_empty() || name.len() > 200 {
        return (StatusCode::BAD_REQUEST, "candidateName: 1–200 chars").into_response();
    }
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return (
            StatusCode::BAD_REQUEST,
            "candidateUrl must be an absolute http(s) URL",
        )
            .into_response();
    }
    if url.len() > 1000 {
        return (StatusCode::BAD_REQUEST, "candidateUrl too long").into_response();
    }
    if note.len() > 2000 {
        return (StatusCode::BAD_REQUEST, "candidateNote too long").into_response();
    }

    // Cap pending requests per user.
    let pending: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM vip_sponsorships WHERE user_id = $1 AND status = 'pending'",
    )
    .bind(me.id)
    .fetch_one(&pool)
    .await
    .unwrap_or(0);
    if pending >= MAX_PENDING_PER_USER {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "too many pending sponsorships — wait for one to be reviewed",
        )
            .into_response();
    }

    // Single transaction so the debit + insert are atomic.
    let mut tx = match pool.begin().await {
        Ok(t) => t,
        Err(e) => {
            tracing::error!(error = %e, "sponsorships.begin.failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };
    let new_balance: Option<i32> = sqlx::query_scalar(
        "SELECT credits_debit($1, $2::int, 'debit:vip-sponsor',
                jsonb_build_object('candidate_name', $3::text, 'candidate_url', $4::text))",
    )
    .bind(me.id)
    .bind(SPONSOR_COST)
    .bind(name)
    .bind(url)
    .fetch_one(&mut *tx)
    .await
    .unwrap_or(None);

    let Some(balance) = new_balance else {
        // Insufficient credits.
        let _ = tx.rollback().await;
        return (
            StatusCode::PAYMENT_REQUIRED,
            Json(serde_json::json!({
                "error":    "insufficient credits",
                "required": SPONSOR_COST,
            })),
        )
            .into_response();
    };

    let id: Result<(i64,), _> = sqlx::query_as(
        "INSERT INTO vip_sponsorships
            (user_id, candidate_name, candidate_url, candidate_note, credits_paid)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id",
    )
    .bind(me.id)
    .bind(name)
    .bind(url)
    .bind(note)
    .bind(SPONSOR_COST)
    .fetch_one(&mut *tx)
    .await;
    let Ok((id,)) = id else {
        let _ = tx.rollback().await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "could not record sponsorship",
        )
            .into_response();
    };

    if let Err(e) = tx.commit().await {
        tracing::error!(error = %e, "sponsorships.commit.failed");
        return (StatusCode::INTERNAL_SERVER_ERROR, "commit failed").into_response();
    }
    Json(CreateSponsorshipResponse { id, balance }).into_response()
}

/// GET /api/me/sponsorships — caller's submissions, most recent first.
pub async fn list(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let rows: Vec<SponsorshipRow> = sqlx::query_as(
        "SELECT id, candidate_name, candidate_url, candidate_note,
                credits_paid, status, review_note,
                to_char(created_at  AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS created_at,
                to_char(resolved_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS resolved_at
           FROM vip_sponsorships
          WHERE user_id = $1
          ORDER BY id DESC
          LIMIT 50",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    Json(rows).into_response()
}
