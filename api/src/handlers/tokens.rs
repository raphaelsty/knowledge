//! User-scoped API tokens.
//!
//! Lifecycle:
//!   - Creation: signed-in user POSTs `{name}` → server mints a 32-byte
//!     secret, stores `sha256(secret)` + an 8-char prefix, returns the
//!     plaintext ONCE. Caller must save it; subsequent reads only return
//!     the prefix.
//!   - Listing: cookie-authed GET returns the user's active tokens with
//!     `name`, `prefix`, `created_at`, `last_used_at`. No secret material.
//!   - Revocation: cookie-authed DELETE flips `revoked_at`. The unique
//!     index on `token_hash` is partial (`WHERE revoked_at IS NULL`) so a
//!     revoked-then-recreated token doesn't collide.
//!
//! Security notes:
//!   - The plaintext is never logged or persisted. We hash on the way in
//!     and the way out — the only place the secret exists is in the
//!     handler's local variable for the duration of one request.
//!   - The bearer auth path (`auth_middleware::RequireUserToken`) does
//!     a constant-time comparison via the unique-index lookup; we don't
//!     iterate user rows.
//!   - Active-token cap per user: 20 — keeps the management UI scannable
//!     and limits blast radius if a user's account is compromised.
//!
//! Routes (registered in main.rs):
//!   POST   /auth/me/tokens          create_token       (cookie session)
//!   GET    /auth/me/tokens          list_tokens        (cookie session)
//!   DELETE /auth/me/tokens/{id}     revoke_token       (cookie session)

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use axum_extra::extract::CookieJar;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::PgPool;

use crate::handlers::auth::current_user;

/// Plaintext-token prefix length: kept in PG as a display handle so the
/// user can pick the right row to revoke. 8 visible chars after the
/// `kn_` brand still leaves the secret safely opaque (~2^48 random
/// from those bytes — a brute-force across the prefix wouldn't hit a
/// real token in any reasonable time).
const PREFIX_LEN: usize = 8;
/// Random secret length in bytes. 32 = 256 bits = enough.
const SECRET_BYTES: usize = 32;
/// Active-token soft cap per user. Hard limit so a buggy or hostile
/// client can't spam create.
const MAX_ACTIVE_PER_USER: i64 = 20;
/// Max length of the user-supplied `name` label.
const MAX_NAME_LEN: usize = 80;

// ── Helpers ──────────────────────────────────────────────────────────

/// Hex-encoded sha256 of a bearer-token plaintext. Lookup key for the
/// `api_tokens.token_hash` column. Same algorithm on creation and on
/// auth — equality compare works.
pub fn hash_token(plaintext: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(plaintext.as_bytes());
    hex::encode(hasher.finalize())
}

/// Generate a fresh plaintext token. Format: `kn_<base64url(32 bytes)>`.
/// The `kn_` prefix is a brand cue: makes leaked tokens grep-able and
/// distinguishes from an admin api key in logs.
fn mint_token() -> String {
    let mut buf = [0u8; SECRET_BYTES];
    OsRng.fill_bytes(&mut buf);
    format!("kn_{}", URL_SAFE_NO_PAD.encode(buf))
}

// ── Request / response types ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateTokenRequest {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct CreateTokenResponse {
    pub id: i64,
    pub name: String,
    pub prefix: String,
    /// ⚠ Plaintext, returned ONCE at creation time. The caller must
    /// store this immediately — subsequent list calls only return
    /// the prefix.
    pub token: String,
    /// ISO-8601 timestamp. We render to text on the SQL side so we
    /// don't need the sqlx `chrono`/`time` feature flags, which
    /// aren't enabled in this crate.
    pub created_at: String,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct TokenSummary {
    pub id: i64,
    pub name: String,
    pub prefix: String,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
}

// ── Handlers ────────────────────────────────────────────────────────

/// POST /auth/me/tokens — mint a new token.
///
/// Body: `{ "name": "My laptop" }`. Returns the plaintext ONCE.
pub async fn create_token(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<CreateTokenRequest>,
) -> Result<Json<CreateTokenResponse>, (StatusCode, Json<ErrorResponse>)> {
    let me = current_user(&pool, &jar)
        .await
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "Sign in to create tokens"))?;

    let name = req.name.trim();
    if name.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "name is required"));
    }
    if name.chars().count() > MAX_NAME_LEN {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "name is too long (max 80 chars)",
        ));
    }

    // Per-user cap. Prevents accidental runaway from a buggy client and
    // limits blast radius on credential compromise. Counts active rows
    // only — revoked tokens don't count toward the limit.
    let active: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM api_tokens WHERE user_id = $1 AND revoked_at IS NULL")
            .bind(me.id)
            .fetch_one(&pool)
            .await
            .map_err(|e| {
                tracing::error!("count active tokens failed: {e}");
                err(StatusCode::INTERNAL_SERVER_ERROR, "Database error")
            })?;
    if active.0 >= MAX_ACTIVE_PER_USER {
        return Err(err(
            StatusCode::TOO_MANY_REQUESTS,
            "Too many active tokens — revoke an existing one first",
        ));
    }

    let plaintext = mint_token();
    let token_hash = hash_token(&plaintext);
    // Prefix = first PREFIX_LEN chars AFTER the `kn_` brand. Keeps
    // both the cue and an identifier that's distinct from the secret.
    let prefix: String = plaintext.chars().take(3 + PREFIX_LEN).collect();

    let row: (i64, String) = sqlx::query_as(
        "INSERT INTO api_tokens (user_id, name, token_hash, prefix)
         VALUES ($1, $2, $3, $4)
         RETURNING id, to_char(created_at AT TIME ZONE 'UTC',
                               'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"')",
    )
    .bind(me.id)
    .bind(name)
    .bind(&token_hash)
    .bind(&prefix)
    .fetch_one(&pool)
    .await
    .map_err(|e| {
        tracing::error!("insert api_token failed: {e}");
        err(StatusCode::INTERNAL_SERVER_ERROR, "Database error")
    })?;

    Ok(Json(CreateTokenResponse {
        id: row.0,
        name: name.to_string(),
        prefix,
        token: plaintext,
        created_at: row.1,
    }))
}

/// GET /auth/me/tokens — list the signed-in user's active tokens.
pub async fn list_tokens(State(pool): State<PgPool>, jar: CookieJar) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    let rows: Vec<TokenSummary> = sqlx::query_as(
        "SELECT id,
                name,
                prefix,
                to_char(created_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS created_at,
                to_char(last_used_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS last_used_at
           FROM api_tokens
          WHERE user_id = $1
            AND revoked_at IS NULL
          ORDER BY created_at DESC",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

    Json(rows).into_response()
}

/// Resolve a `Authorization: Bearer kn_…` header to a user_id.
/// Returns `None` when:
///   - The header is missing or malformed.
///   - The token isn't recognized or has been revoked.
///
/// Updates `last_used_at` as a fire-and-forget side effect on every
/// successful resolve — handy for the management UI's "last used" column.
pub async fn resolve_bearer(pool: &PgPool, headers: &HeaderMap) -> Option<i64> {
    let raw = headers.get("authorization")?.to_str().ok()?;
    // Tolerate any case for "bearer" but require the kn_ brand so a
    // misrouted GitHub PAT or admin key doesn't accidentally pass.
    let plaintext = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))?
        .trim();
    if !plaintext.starts_with("kn_") {
        return None;
    }
    let token_hash = hash_token(plaintext);

    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT user_id FROM api_tokens
          WHERE token_hash = $1 AND revoked_at IS NULL",
    )
    .bind(&token_hash)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let user_id = row?.0;

    // Best-effort last-used bump. We don't await — failure here mustn't
    // tank the request, and the user-visible side is decoration anyway.
    let pool_clone = pool.clone();
    let hash_clone = token_hash.clone();
    tokio::spawn(async move {
        let _ = sqlx::query("UPDATE api_tokens SET last_used_at = now() WHERE token_hash = $1")
            .bind(&hash_clone)
            .execute(&pool_clone)
            .await;
    });

    Some(user_id)
}

/// Single-doc upload payload accepted by the bearer-authed
/// `/api/me/documents` endpoint. Mirrors the cookie-authed
/// `SaveDocumentRequest` shape so any client built against either
/// path needs minimal divergence.
#[derive(Debug, Deserialize)]
pub struct UploadDocumentRequest {
    pub url: String,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub date: Option<String>,
    pub tags: Option<Vec<String>>,
    #[serde(rename = "extra-tags", alias = "extra_tags")]
    pub extra_tags: Option<Vec<String>>,
    pub source: Option<String>,
    pub source_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UploadDocumentResponse {
    pub status: String,
    pub url: String,
}

/// POST /api/me/documents
///
/// Insert (or upsert) one document into the bearer-token holder's
/// library. The token in `Authorization: Bearer kn_...` decides the
/// owning user — the URL is the natural key inside that scope.
///
/// On conflict (same URL already in this user's library) we update
/// metadata fields. The downstream pipeline picks the row up on the
/// next `make run` and embeds it into the search index — same path
/// as a doc inserted by the browser-sync flow.
pub async fn upload_document(
    State(pool): State<PgPool>,
    headers: HeaderMap,
    Json(req): Json<UploadDocumentRequest>,
) -> Result<Json<UploadDocumentResponse>, (StatusCode, Json<ErrorResponse>)> {
    let user_id = resolve_bearer(&pool, &headers).await.ok_or_else(|| {
        err(
            StatusCode::UNAUTHORIZED,
            "Missing or invalid bearer token. Send 'Authorization: Bearer kn_...'.",
        )
    })?;

    let url = req.url.trim();
    if url.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "url is required"));
    }

    let date = req.date.as_deref().unwrap_or("");
    let tags = req.tags.unwrap_or_default();
    let extra_tags = req.extra_tags.unwrap_or_default();

    let res = sqlx::query(
        "INSERT INTO documents (
             user_id, url, title, summary, date, tags, extra_tags,
             source, source_url
         )
         VALUES ($1, $2, $3, $4, NULLIF($5, '')::date, $6, $7, $8, $9)
         ON CONFLICT (user_id, url) DO UPDATE SET
             title       = EXCLUDED.title,
             summary     = EXCLUDED.summary,
             date        = EXCLUDED.date,
             tags        = EXCLUDED.tags,
             extra_tags  = EXCLUDED.extra_tags,
             source      = EXCLUDED.source,
             source_url  = EXCLUDED.source_url,
             updated_at  = now()",
    )
    .bind(user_id)
    .bind(url)
    .bind(req.title.unwrap_or_default())
    .bind(req.summary.unwrap_or_default())
    .bind(date)
    .bind(&tags)
    .bind(&extra_tags)
    .bind(req.source.unwrap_or_default())
    .bind(req.source_url.as_deref())
    .execute(&pool)
    .await;

    match res {
        Ok(_) => Ok(Json(UploadDocumentResponse {
            status: "ok".to_string(),
            url: url.to_string(),
        })),
        Err(e) => {
            tracing::error!("upload_document insert failed: {e}");
            Err(err(StatusCode::INTERNAL_SERVER_ERROR, "Database error"))
        }
    }
}

/// DELETE /auth/me/tokens/{id} — revoke. Idempotent; revoking twice is
/// a no-op. Cross-user revocation is impossible because the WHERE
/// scopes by `user_id = me.id`.
pub async fn revoke_token(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Path(id): Path<i64>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let me = current_user(&pool, &jar)
        .await
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "Sign in to revoke tokens"))?;

    sqlx::query(
        "UPDATE api_tokens
            SET revoked_at = now()
          WHERE id = $1
            AND user_id = $2
            AND revoked_at IS NULL",
    )
    .bind(id)
    .bind(me.id)
    .execute(&pool)
    .await
    .map_err(|e| {
        tracing::error!("revoke api_token failed: {e}");
        err(StatusCode::INTERNAL_SERVER_ERROR, "Database error")
    })?;

    Ok(StatusCode::NO_CONTENT)
}
