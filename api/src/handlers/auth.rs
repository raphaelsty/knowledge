//! Email + password authentication and session management.
//!
//! Flow:
//!   1. `POST /auth/signup { email, password, slug, name }` — creates a
//!      `users` row with `email_verified = FALSE` and a single-use
//!      `email_verification_token`. Mints a session cookie immediately
//!      so the user can browse, and emails them a verification link.
//!      Argon2id is used to hash the password.
//!   2. The user follows the link → `GET /auth/verify?token=...` flips
//!      `email_verified = TRUE` and clears the token.
//!   3. `POST /auth/login { email, password }` — looks up by lower(email),
//!      verifies the hash, mints a session cookie.
//!   4. Subsequent requests read the session cookie via `current_user`.
//!      Sessions slide (TTL is refreshed on every authenticated hit).
//!   5. `POST /auth/forgot { email }` issues a `password_reset_token`
//!      and emails it. `POST /auth/reset { token, password }` consumes
//!      the token and swaps the hash.
//!
//! Config (env):
//!   SESSION_COOKIE_SECURE — "1" in production (forces Secure flag).
//!   RESEND_API_KEY        — Resend.com API key for outbound mail.
//!   RESEND_FROM           — sender address for verification + reset emails.
//!   PUBLIC_BASE_URL       — base URL for links inside emails.

use argon2::password_hash::{
    rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString,
};
use argon2::Argon2;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Redirect, Response},
    Json,
};
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use time::Duration as TimeDuration;

// ── Constants ───────────────────────────────────────────────────────────

const SESSION_COOKIE: &str = "k_session";
const SESSION_TTL_DAYS: i32 = 30;
const VERIFY_TTL_HOURS: i64 = 24;
const RESET_TTL_HOURS: i64 = 1;
/// Used by the Stack Overflow OAuth flow (which only links an SO
/// account to the already-signed-in Knowledge user — it is NOT a
/// login mechanism). SO has its own state cookie name (see
/// `STACK_STATE_COOKIE` further down).
const OAUTH_STATE_TTL_MINUTES: i64 = 10;

fn post_login_url() -> String {
    std::env::var("OAUTH_POST_LOGIN_URL").unwrap_or_else(|_| "/me".to_string())
}

/// Slugs the signup endpoint refuses regardless of availability.
/// Includes route-shaped names that would otherwise collide with the
/// path-based `/<slug>` profile route.
const RESERVED_SLUGS: &[&str] = &[
    "admin",
    "administrator",
    "root",
    "support",
    "help",
    "about",
    "privacy",
    "terms",
    "tos",
    "api",
    "auth",
    "login",
    "logout",
    "signup",
    "signin",
    "register",
    "verify",
    "reset",
    "forgot",
    "search",
    "profile",
    "me",
    "settings",
    "feed",
    "data",
    "indexes",
    "favicon",
    "robots",
    "sitemap",
    "static",
    "assets",
    "public",
    "bundle",
    "source",
    "pkg",
    "img",
    "icons",
    "events",
    "stats",
    "ingest",
    "anon",
    "anonymous",
    "user",
    "users",
    "knowledge",
    "empty",
    "all",
];

fn cookie_secure() -> bool {
    matches!(
        std::env::var("SESSION_COOKIE_SECURE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// 256-bit hex token (64 chars). Used for both session ids and OAuth state.
fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

fn build_session_cookie(value: String, max_age_days: i32) -> Cookie<'static> {
    let mut c = Cookie::new(SESSION_COOKIE, value);
    c.set_path("/");
    c.set_http_only(true);
    c.set_same_site(SameSite::Lax);
    c.set_secure(cookie_secure());
    c.set_max_age(TimeDuration::days(max_age_days as i64));
    c
}

fn clear_cookie(name: &'static str) -> Cookie<'static> {
    let mut c = Cookie::new(name, "");
    c.set_path("/");
    c.set_http_only(true);
    c.set_same_site(SameSite::Lax);
    c.set_secure(cookie_secure());
    c.set_max_age(TimeDuration::seconds(0));
    c
}

/// Slugify an arbitrary input (display name, email local-part, etc.)
/// into a URL-safe lower-kebab string. Drops non-alphanumeric runs to
/// a single dash, lowercases, and trims leading/trailing dashes.
fn slugify(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("user");
    }
    out
}

/// Reject slugs that don't match `[a-z0-9-]{2,32}` after slugify (so a
/// user can't smuggle whitespace or punctuation past the frontend).
fn slug_is_valid(s: &str) -> bool {
    let n = s.len();
    if !(2..=32).contains(&n) {
        return false;
    }
    if s.starts_with('-') || s.ends_with('-') {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn slug_is_reserved(s: &str) -> bool {
    RESERVED_SLUGS.contains(&s)
}

/// Very light email shape check — not RFC-strict, just enough to reject
/// obviously malformed inputs before we hash a password against it.
fn email_is_valid(s: &str) -> bool {
    let s = s.trim();
    if s.len() > 254 || s.is_empty() {
        return false;
    }
    let mut parts = s.split('@');
    let (local, domain) = match (parts.next(), parts.next(), parts.next()) {
        (Some(l), Some(d), None) => (l, d),
        _ => return false,
    };
    !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
}

fn password_is_acceptable(s: &str) -> bool {
    // Length-only policy: NIST SP 800-63B explicitly recommends against
    // composition rules. 8+ chars; cap at 128 to bound argon2 cost.
    let n = s.chars().count();
    (8..=128).contains(&n)
}

fn hash_password(plain: &str) -> Result<String, String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(plain.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("argon2 hash failed: {e}"))
}

fn verify_password(plain: &str, encoded: &str) -> bool {
    match PasswordHash::new(encoded) {
        Ok(parsed) => Argon2::default()
            .verify_password(plain.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

async fn mint_session(pool: &PgPool, user_id: i64) -> Result<String, sqlx::Error> {
    let session_id = random_token();
    sqlx::query(
        "INSERT INTO auth_sessions (id, user_id, user_agent, expires_at)
         VALUES ($1, $2, NULL, now() + make_interval(days => $3::int))",
    )
    .bind(&session_id)
    .bind(user_id)
    .bind(SESSION_TTL_DAYS)
    .execute(pool)
    .await?;
    Ok(session_id)
}

// ── Models ──────────────────────────────────────────────────────────────

#[derive(Serialize, sqlx::FromRow)]
pub struct MeResponse {
    pub id: i64,
    pub slug: String,
    pub name: String,
    pub email: Option<String>,
    pub avatar: Option<String>,
    pub description: String,
    /// Topical-ontology slugs (read from the `user_categories` table).
    /// Replaces the legacy single-string `users.category` column.
    pub categories: Vec<String>,
    #[serde(rename = "indexName")]
    pub index_name: String,
    /// True = library visible to anonymous visitors. The profile form
    /// reads this to hydrate the "Public profile" toggle and writes it
    /// back via the same PUT.
    pub public: bool,
    pub links: serde_json::Value,
    pub sources: serde_json::Value,
    /// True once the user has clicked the verification link emailed at
    /// signup. Unverified accounts are read-only — the frontend nudges
    /// the user to verify before letting them save / follow / star.
    #[sqlx(rename = "email_verified")]
    #[serde(rename = "emailVerified")]
    pub email_verified: bool,
    #[sqlx(rename = "twitter_followers")]
    #[serde(rename = "twitterFollowers")]
    pub twitter_followers: Option<i32>,
    #[sqlx(rename = "github_followers")]
    #[serde(rename = "githubFollowers")]
    pub github_followers: Option<i32>,
    pub citations: Option<i32>,
    /// True when a HackerNews username is on file. Username alone is
    /// enough for the public Algolia-backed fetchers (comments +
    /// submissions); the encrypted password is only required for the
    /// private /upvoted page (see `has_hackernews_upvotes`).
    #[sqlx(rename = "has_hackernews")]
    #[serde(rename = "hasHackernews")]
    pub has_hackernews: bool,
    /// True when an encrypted password is *also* on file — unlocks the
    /// private /upvoted fetcher. The plaintext is never exposed.
    #[sqlx(rename = "has_hackernews_upvotes")]
    #[serde(rename = "hasHackernewsUpvotes")]
    pub has_hackernews_upvotes: bool,
    /// The HN username is safe to echo back (lets the form show
    /// "Connected as {username}" without leaking the password).
    #[sqlx(rename = "hackernews_username")]
    #[serde(rename = "hackernewsUsername")]
    pub hackernews_username: Option<String>,
    /// True when a Zotero API key is on file (discovery succeeded).
    #[sqlx(rename = "has_zotero")]
    #[serde(rename = "hasZotero")]
    pub has_zotero: bool,
    /// Numeric Zotero userID resolved from the key — displayed in the
    /// "Connected" badge. Safe to echo (it's visible on the user's
    /// public Zotero profile).
    #[sqlx(rename = "zotero_user_id")]
    #[serde(rename = "zoteroUserId")]
    pub zotero_user_id: Option<i64>,
    /// Group libraries the key can read, as JSONB:
    /// `[{ id, name, count }, ...]`. `count` is the total item count at
    /// discovery time — not refreshed on every request.
    #[sqlx(rename = "zotero_groups")]
    #[serde(rename = "zoteroGroups")]
    pub zotero_groups: serde_json::Value,
    /// Item count on the personal library at discovery time.
    #[sqlx(rename = "zotero_personal_count")]
    #[serde(rename = "zoteroPersonalCount")]
    pub zotero_personal_count: Option<i64>,
    /// True when Twitter/X auth cookies (auth_token + ct0) are on file.
    /// The plaintext cookies are never exposed.
    #[sqlx(rename = "has_twitter_cookies")]
    #[serde(rename = "hasTwitterCookies")]
    pub has_twitter_cookies: bool,
    /// True when a Stack Overflow access token is on file.
    #[sqlx(rename = "has_stackoverflow_auth")]
    #[serde(rename = "hasStackoverflowAuth")]
    pub has_stackoverflow_auth: bool,
    /// Stack Exchange sites the linked token unlocks, discovered via
    /// `/me/associated`: `[{ api_site_parameter, site_name, user_id,
    /// reputation }, ...]`. Empty when no token is on file.
    #[sqlx(rename = "stackoverflow_sites")]
    #[serde(rename = "stackoverflowSites")]
    pub stackoverflow_sites: serde_json::Value,
}

// ── Session extractor ───────────────────────────────────────────────────

/// Reads the session cookie, looks up (and slides) the session, returns
/// the user row. Returns `None` when no/invalid/expired cookie.
pub async fn current_user(pool: &PgPool, jar: &CookieJar) -> Option<MeResponse> {
    let session_id = jar.get(SESSION_COOKIE)?.value().to_string();
    // Session must be non-expired; update last-seen by pushing expires_at
    // forward. Keep the query a single round-trip.
    let row: Option<MeResponse> = sqlx::query_as(
        "UPDATE auth_sessions s
            SET expires_at = now() + make_interval(days => $2::int)
          FROM users u
         WHERE s.id = $1
           AND s.user_id = u.id
           AND s.expires_at > now()
         RETURNING u.id,
                   u.username      AS slug,
                   u.name,
                   u.email,
                   u.avatar,
                   u.description,
                   COALESCE(
                       (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
                          FROM user_categories uc
                          JOIN categories      cat ON cat.id = uc.category_id
                         WHERE uc.user_id = u.id),
                       '{}'::text[]
                   ) AS categories,
                   u.index_name,
                   u.public,
                   u.links,
                   u.sources,
                   u.email_verified,
                   u.twitter_followers,
                   u.github_followers,
                   u.citations,
                   (COALESCE(u.sources->'hackernews'->>'username', '') <> '')
                       AS has_hackernews,
                   (COALESCE(u.sources->'hackernews'->>'username', '') <> ''
                    AND COALESCE(u.sources->'hackernews'->>'password_enc', '') <> '')
                       AS has_hackernews_upvotes,
                   NULLIF(u.sources->'hackernews'->>'username', '')
                       AS hackernews_username,
                   (COALESCE(u.sources->'zotero'->>'api_key_enc', '') <> ''
                    AND COALESCE(u.sources->'zotero'->>'user_id', '') <> '')
                       AS has_zotero,
                   NULLIF(u.sources->'zotero'->>'user_id', '')::bigint
                       AS zotero_user_id,
                   COALESCE(u.sources->'zotero'->'groups', '[]'::jsonb)
                       AS zotero_groups,
                   NULLIF(u.sources->'zotero'->>'personal_count', '')::bigint
                       AS zotero_personal_count,
                   (COALESCE(u.sources->'twitter'->>'cookies_enc', '') <> '')
                       AS has_twitter_cookies,
                   (COALESCE(u.sources->'stackoverflow'->>'access_token_enc', '') <> '')
                       AS has_stackoverflow_auth,
                   COALESCE(u.sources->'stackoverflow'->'associated_sites', '[]'::jsonb)
                       AS stackoverflow_sites",
    )
    .bind(&session_id)
    .bind(SESSION_TTL_DAYS)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    row
}

// ── Handlers ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SignupRequest {
    pub email: String,
    pub password: String,
    pub slug: String,
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct VerifyParams {
    pub token: String,
}

#[derive(Deserialize)]
pub struct ForgotRequest {
    pub email: String,
}

#[derive(Deserialize)]
pub struct ResetRequest {
    pub token: String,
    pub password: String,
}

#[derive(Serialize)]
struct AuthErrorBody<'a> {
    error: &'a str,
}

fn err(status: StatusCode, msg: &str) -> Response {
    (status, Json(AuthErrorBody { error: msg })).into_response()
}

/// POST /auth/signup
///
/// Creates an account, mints a session cookie, and emails a
/// verification link. The new account is read-only until the user
/// follows the link (we surface `emailVerified: false` in /auth/me so
/// the frontend can gate write actions accordingly).
pub async fn signup(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<SignupRequest>,
) -> Response {
    let email = req.email.trim().to_lowercase();
    let slug = slugify(req.slug.trim());
    let name = req
        .name
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| slug.clone());

    if !email_is_valid(&email) {
        return err(StatusCode::BAD_REQUEST, "invalid email");
    }
    if !password_is_acceptable(&req.password) {
        return err(StatusCode::BAD_REQUEST, "password must be 8–128 characters");
    }
    if !slug_is_valid(&slug) {
        return err(
            StatusCode::BAD_REQUEST,
            "slug must be 2–32 chars (a–z, 0–9, dashes)",
        );
    }
    if slug_is_reserved(&slug) {
        return err(StatusCode::CONFLICT, "this slug is reserved");
    }

    // Slug taken? (case-insensitive)
    let slug_taken: Option<i64> =
        sqlx::query_scalar("SELECT id FROM users WHERE lower(username) = $1")
            .bind(&slug)
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);
    if slug_taken.is_some() {
        return err(StatusCode::CONFLICT, "slug already taken");
    }
    let email_taken: Option<i64> =
        sqlx::query_scalar("SELECT id FROM users WHERE lower(email) = $1")
            .bind(&email)
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);
    if email_taken.is_some() {
        return err(StatusCode::CONFLICT, "email already registered");
    }
    // Display-name dedup. On Knowledge every account is also a
    // library, so two accounts with the same `name` would collide in
    // every list view. Case-insensitive — "Yann LeCun" and "yann
    // lecun" are the same person.
    let name_taken: Option<i64> =
        sqlx::query_scalar("SELECT id FROM users WHERE lower(name) = lower($1)")
            .bind(&name)
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);
    if name_taken.is_some() {
        return err(
            StatusCode::CONFLICT,
            "that display name is already on Knowledge",
        );
    }

    let hash = match hash_password(&req.password) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!(error = %e, "signup.hash.failed");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "internal error");
        }
    };

    let verify_token = random_token();
    let user_id: i64 = match sqlx::query_scalar(
        "INSERT INTO users (
            username, email, password_hash, name, index_name,
            email_verified, email_verification_token,
            email_verification_expires_at
         ) VALUES (
            $1, $2, $3, $4, $1,
            FALSE, $5,
            now() + make_interval(hours => $6::int)
         ) RETURNING id",
    )
    .bind(&slug)
    .bind(&email)
    .bind(&hash)
    .bind(&name)
    .bind(&verify_token)
    .bind(VERIFY_TTL_HOURS as i32)
    .fetch_one(&pool)
    .await
    {
        Ok(id) => id,
        Err(e) => {
            tracing::error!(error = %e, "signup.insert.failed");
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not create account",
            );
        }
    };

    if let Err(e) = super::mailer::send_verification_email(&email, &name, &verify_token).await {
        // Don't fail the signup — the user is created, they can
        // ask for a new link via /auth/resend.
        tracing::error!(error = %e, "signup.mail.failed");
    }

    let session_id = match mint_session(&pool, user_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "signup.session.failed");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "session creation failed");
        }
    };

    let jar = jar.add(build_session_cookie(session_id, SESSION_TTL_DAYS));
    match current_user(&pool, &jar).await {
        Some(me) => (jar, Json(me)).into_response(),
        None => (jar, StatusCode::NO_CONTENT).into_response(),
    }
}

/// POST /auth/login
///
/// Looks up by lower(email), verifies the argon2 hash, mints a
/// session cookie. Returns a single generic error message for "no
/// such user" and "wrong password" so the endpoint doesn't double
/// as a user-enumeration oracle.
pub async fn login(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<LoginRequest>,
) -> Response {
    let email = req.email.trim().to_lowercase();
    if email.is_empty() || req.password.is_empty() {
        return err(StatusCode::BAD_REQUEST, "email and password are required");
    }
    let row: Option<(i64, Option<String>)> =
        sqlx::query_as("SELECT id, password_hash FROM users WHERE lower(email) = $1")
            .bind(&email)
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);

    let Some((user_id, Some(hash))) = row else {
        return err(StatusCode::UNAUTHORIZED, "invalid email or password");
    };
    if !verify_password(&req.password, &hash) {
        return err(StatusCode::UNAUTHORIZED, "invalid email or password");
    }

    let session_id = match mint_session(&pool, user_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "login.session.failed");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "session creation failed");
        }
    };
    let jar = jar.add(build_session_cookie(session_id, SESSION_TTL_DAYS));
    match current_user(&pool, &jar).await {
        Some(me) => (jar, Json(me)).into_response(),
        None => (jar, StatusCode::NO_CONTENT).into_response(),
    }
}

/// GET /auth/verify?token=...
///
/// Consumes an email-verification token. Idempotent: a second hit
/// after success silently no-ops because the token is cleared on
/// first use. Returns a tiny HTML page that links back to /.
pub async fn verify_email(
    State(pool): State<PgPool>,
    Query(params): Query<VerifyParams>,
) -> Response {
    let row: Option<(i64,)> = sqlx::query_as(
        "UPDATE users
            SET email_verified                = TRUE,
                email_verification_token      = NULL,
                email_verification_expires_at = NULL,
                updated_at                    = now()
          WHERE email_verification_token      = $1
            AND email_verification_expires_at > now()
        RETURNING id",
    )
    .bind(&params.token)
    .fetch_optional(&pool)
    .await
    .unwrap_or(None);

    let (ok, msg) = match row {
        Some(_) => (true, "Email verified — you're all set."),
        None => (
            false,
            "This verification link is invalid or has expired. Sign in and request a new one.",
        ),
    };
    let body = format!(
        "<!doctype html><meta charset=utf-8><title>Verify</title>\
         <style>body{{font:16px/1.5 system-ui;margin:40px auto;max-width:480px;padding:0 16px;color:#222}}\
         .ok{{color:#0a7d33}}.bad{{color:#b00020}}a{{color:#10b981}}</style>\
         <h1 class=\"{cls}\">{msg}</h1>\
         <p><a href=\"/\">Back to Knowledge →</a></p>",
        cls = if ok { "ok" } else { "bad" },
        msg = msg,
    );
    let mut resp = axum::response::Html(body).into_response();
    if !ok {
        *resp.status_mut() = StatusCode::BAD_REQUEST;
    }
    resp
}

/// POST /auth/resend
///
/// Re-issues an email-verification token for the signed-in account.
/// 401 when anonymous; 204 when already verified.
pub async fn resend_verification(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return err(StatusCode::UNAUTHORIZED, "not signed in");
    };
    if me.email_verified {
        return StatusCode::NO_CONTENT.into_response();
    }
    let Some(email) = me.email.clone() else {
        return err(StatusCode::BAD_REQUEST, "no email on file");
    };
    let token = random_token();
    if let Err(e) = sqlx::query(
        "UPDATE users
            SET email_verification_token      = $2,
                email_verification_expires_at = now() + make_interval(hours => $3::int)
          WHERE id = $1",
    )
    .bind(me.id)
    .bind(&token)
    .bind(VERIFY_TTL_HOURS as i32)
    .execute(&pool)
    .await
    {
        tracing::error!(error = %e, "resend.update.failed");
        return err(StatusCode::INTERNAL_SERVER_ERROR, "could not issue token");
    }
    if let Err(e) = super::mailer::send_verification_email(&email, &me.name, &token).await {
        tracing::error!(error = %e, "resend.mail.failed");
    }
    StatusCode::NO_CONTENT.into_response()
}

/// POST /auth/forgot
///
/// Issues a password-reset token and emails it. Always returns 204
/// regardless of whether the email is on file, so the endpoint
/// cannot be used to enumerate accounts.
pub async fn forgot_password(
    State(pool): State<PgPool>,
    Json(req): Json<ForgotRequest>,
) -> Response {
    let email = req.email.trim().to_lowercase();
    if !email_is_valid(&email) {
        return StatusCode::NO_CONTENT.into_response();
    }
    let row: Option<(i64, String)> = sqlx::query_as(
        "SELECT id, name FROM users WHERE lower(email) = $1 AND password_hash IS NOT NULL",
    )
    .bind(&email)
    .fetch_optional(&pool)
    .await
    .unwrap_or(None);
    if let Some((user_id, name)) = row {
        let token = random_token();
        if let Err(e) = sqlx::query(
            "UPDATE users
                SET password_reset_token       = $2,
                    password_reset_expires_at  = now() + make_interval(hours => $3::int),
                    updated_at                 = now()
              WHERE id = $1",
        )
        .bind(user_id)
        .bind(&token)
        .bind(RESET_TTL_HOURS as i32)
        .execute(&pool)
        .await
        {
            tracing::error!(error = %e, "forgot.update.failed");
        } else if let Err(e) = super::mailer::send_password_reset_email(&email, &name, &token).await
        {
            tracing::error!(error = %e, "forgot.mail.failed");
        }
    }
    StatusCode::NO_CONTENT.into_response()
}

/// POST /auth/reset
///
/// Consumes a password-reset token, swaps the hash, invalidates all
/// existing sessions for the account (defense in depth), and mints a
/// fresh session for the caller.
pub async fn reset_password(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<ResetRequest>,
) -> Response {
    if !password_is_acceptable(&req.password) {
        return err(StatusCode::BAD_REQUEST, "password must be 8–128 characters");
    }
    let new_hash = match hash_password(&req.password) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!(error = %e, "reset.hash.failed");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "internal error");
        }
    };
    let row: Option<(i64,)> = sqlx::query_as(
        "UPDATE users
            SET password_hash              = $2,
                password_reset_token       = NULL,
                password_reset_expires_at  = NULL,
                email_verified             = TRUE,
                updated_at                 = now()
          WHERE password_reset_token       = $1
            AND password_reset_expires_at  > now()
        RETURNING id",
    )
    .bind(&req.token)
    .bind(&new_hash)
    .fetch_optional(&pool)
    .await
    .unwrap_or(None);
    let Some((user_id,)) = row else {
        return err(StatusCode::BAD_REQUEST, "invalid or expired reset link");
    };
    let _ = sqlx::query("DELETE FROM auth_sessions WHERE user_id = $1")
        .bind(user_id)
        .execute(&pool)
        .await;
    let session_id = match mint_session(&pool, user_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "reset.session.failed");
            return err(StatusCode::INTERNAL_SERVER_ERROR, "session creation failed");
        }
    };
    let jar = jar.add(build_session_cookie(session_id, SESSION_TTL_DAYS));
    match current_user(&pool, &jar).await {
        Some(me) => (jar, Json(me)).into_response(),
        None => (jar, StatusCode::NO_CONTENT).into_response(),
    }
}

/// GET /auth/me — returns the current user or 401.
pub async fn me(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    match current_user(&pool, &jar).await {
        Some(u) => Json(u).into_response(),
        None => StatusCode::UNAUTHORIZED.into_response(),
    }
}

/// POST /auth/logout — drops the session row and clears the cookie.
pub async fn logout(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    if let Some(c) = jar.get(SESSION_COOKIE) {
        let _ = sqlx::query("DELETE FROM auth_sessions WHERE id = $1")
            .bind(c.value())
            .execute(&pool)
            .await;
    }
    let jar = jar.remove(clear_cookie(SESSION_COOKIE));
    (jar, StatusCode::NO_CONTENT).into_response()
}

// ── Profile update (PUT /api/users/me) ──────────────────────────────────

#[derive(Deserialize)]
pub struct UpdateMeRequest {
    /// Optional: rename the slug/username. Lower-kebab; must be unique.
    pub slug: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    /// Full replacement of the user's category set. `None` = leave
    /// `user_categories` alone; an empty `Some([])` clears all
    /// categories. Slugs unknown to the `categories` table are
    /// silently dropped by the JOIN.
    pub categories: Option<Vec<String>>,
    pub avatar: Option<String>,
    pub public: Option<bool>,
    /// Full replacement (JSONB).
    pub links: Option<serde_json::Value>,
    pub sources: Option<serde_json::Value>,
}

/// PUT /api/users/me — authenticated profile + sources editor.
///
/// Unchanged fields pass through COALESCE: `NULL` in the request body
/// means "leave as is". The `slug` is derived from the supplied value
/// via `slugify` so the frontend can send a human label.
pub async fn update_me(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<UpdateMeRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    let slug = req.slug.as_deref().map(slugify);

    // Preserve `sources.hackernews` and `sources.zotero` on generic
    // updates — those keys are owned by the dedicated
    // /auth/me/{hackernews,zotero} endpoints (they hold encrypted
    // secrets; a profile edit must never drop them).
    let result = sqlx::query(
        "UPDATE users SET
            username    = COALESCE($2, username),
            index_name  = COALESCE($2, index_name),
            name        = COALESCE($3, name),
            description = COALESCE($4, description),
            avatar      = COALESCE($5, avatar),
            public      = COALESCE($6, public),
            links       = COALESCE($7::jsonb, links),
            sources     = CASE
                WHEN $8::jsonb IS NULL THEN sources
                ELSE (
                    $8::jsonb
                    -- hackernews, zotero, stackoverflow: fully owned by
                    -- their dedicated endpoints (encrypted secrets +
                    -- OAuth discovery metadata); never overwrite from
                    -- the generic profile form.
                    || CASE WHEN sources ? 'hackernews'
                            THEN jsonb_build_object('hackernews', sources->'hackernews')
                            ELSE '{}'::jsonb
                       END
                    || CASE WHEN sources ? 'zotero'
                            THEN jsonb_build_object('zotero', sources->'zotero')
                            ELSE '{}'::jsonb
                       END
                    || CASE WHEN sources ? 'stackoverflow'
                            THEN jsonb_build_object('stackoverflow', sources->'stackoverflow')
                            ELSE '{}'::jsonb
                       END
                    -- twitter: preserve only the encrypted cookies so
                    -- the form can still update username / replies /
                    -- max_age_years without wiping the bookmark auth.
                    || CASE WHEN sources->'twitter' ? 'cookies_enc'
                            THEN jsonb_build_object(
                                'twitter',
                                COALESCE($8::jsonb->'twitter', '{}'::jsonb)
                                    || jsonb_build_object(
                                        'cookies_enc',
                                        sources->'twitter'->'cookies_enc'
                                    )
                            )
                            ELSE '{}'::jsonb
                       END
                )
            END,
            updated_at  = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .bind(slug.as_deref())
    .bind(req.name.as_deref())
    .bind(req.description.as_deref())
    .bind(req.avatar.as_deref())
    .bind(req.public)
    .bind(req.links.as_ref())
    .bind(req.sources.as_ref())
    .execute(&pool)
    .await;

    if let Err(e) = result {
        tracing::error!(error = %e, "users.update_me.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("update failed: {e}"),
        )
            .into_response();
    }

    // Replace the user's category set when the request supplies one.
    // None = leave as-is. Some(empty) = clear everything. Slugs not
    // present in `categories` are silently dropped by the inner JOIN.
    if let Some(cats) = req.categories.as_ref() {
        let mut tx = match pool.begin().await {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(error = %e, "users.update_me.cat.begin.failed");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "category update failed".to_string(),
                )
                    .into_response();
            }
        };
        if let Err(e) = sqlx::query("DELETE FROM user_categories WHERE user_id = $1")
            .bind(me.id)
            .execute(&mut *tx)
            .await
        {
            tracing::error!(error = %e, "users.update_me.cat.delete.failed");
            let _ = tx.rollback().await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "category update failed".to_string(),
            )
                .into_response();
        }
        if !cats.is_empty() {
            if let Err(e) = sqlx::query(
                "INSERT INTO user_categories (user_id, category_id)
                 SELECT $1, c.id
                   FROM categories c
                  WHERE c.slug = ANY($2::text[])
                ON CONFLICT DO NOTHING",
            )
            .bind(me.id)
            .bind(cats)
            .execute(&mut *tx)
            .await
            {
                tracing::error!(error = %e, "users.update_me.cat.insert.failed");
                let _ = tx.rollback().await;
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "category update failed".to_string(),
                )
                    .into_response();
            }
        }
        if let Err(e) = tx.commit().await {
            tracing::error!(error = %e, "users.update_me.cat.commit.failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "category update failed".to_string(),
            )
                .into_response();
        }
    }

    me_after_update(&pool, me.id).await
}

// ── HackerNews credentials ──────────────────────────────────────────────
//
// `PUT /auth/me/hackernews` — enable the upvote scraper.
//    body: { "username": "...", "password": "..." }
//    → encrypts the password and stores `sources.hackernews`
//
// `DELETE /auth/me/hackernews` — wipe creds so the scraper skips HN.

#[derive(Deserialize)]
pub struct HackernewsCredsRequest {
    pub username: String,
    /// Optional — when absent or empty we save the username only. The
    /// public Comments+Submissions fetchers run on the username alone
    /// (Algolia API, no auth). A password is only needed to unlock
    /// the private /upvoted page.
    #[serde(default)]
    pub password: Option<String>,
}

pub async fn set_hackernews(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<HackernewsCredsRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let username = req.username.trim();
    if username.is_empty() {
        return (StatusCode::BAD_REQUEST, "username required").into_response();
    }

    let typed_password = req.password.as_deref().unwrap_or("");
    // Two paths: username-only (preserve any existing password_enc) vs
    // username+password (encrypt and overwrite).
    let sql = if typed_password.is_empty() {
        // Merge: overwrite username, keep whatever password_enc exists.
        "UPDATE users
            SET sources = jsonb_set(
                    COALESCE(sources, '{}'::jsonb),
                    '{hackernews,username}',
                    to_jsonb($2::text),
                    true
                ),
                updated_at = now()
          WHERE id = $1"
    } else {
        "UPDATE users
            SET sources = jsonb_set(
                    COALESCE(sources, '{}'::jsonb),
                    '{hackernews}',
                    jsonb_build_object(
                        'username', $2::text,
                        'password_enc', $3::text
                    ),
                    true
                ),
                updated_at = now()
          WHERE id = $1"
    };

    let mut q = sqlx::query(sql).bind(me.id).bind(username);
    if !typed_password.is_empty() {
        let Some(enc) = super::secrets::encrypt(typed_password) else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "HN_ENCRYPTION_KEY is not configured — refusing to store plaintext",
            )
                .into_response();
        };
        q = q.bind(enc);
    }

    if let Err(e) = q.execute(&pool).await {
        tracing::error!(error = %e, "hackernews.set.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("store failed: {e}"),
        )
            .into_response();
    }
    me_after_update(&pool, me.id).await
}

/// GET /auth/me/hackernews/test — dry-run a login using the stored
/// credentials. Lets the form show "stored creds still work" vs
/// "password has been changed on HN" without requiring the user to
/// re-type anything. Returns the same ProbeResponse shape as the
/// POST variant below so the frontend can share render code.
pub async fn test_hackernews_stored(
    State(pool): State<PgPool>,
    jar: CookieJar,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(super::probe::ProbeResponse {
                ok: false,
                error: Some("not signed in".to_string()),
                ..Default::default()
            }),
        )
            .into_response();
    };
    let row: Option<(Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT sources->'hackernews'->>'username',
                sources->'hackernews'->>'password_enc'
           FROM users WHERE id = $1",
    )
    .bind(me.id)
    .fetch_optional(&pool)
    .await
    .ok()
    .flatten();
    let (username, enc) = match row {
        Some((Some(u), Some(e))) if !u.is_empty() && !e.is_empty() => (u, e),
        _ => {
            return Json(super::probe::ProbeResponse {
                ok: false,
                error: Some("no stored credentials".to_string()),
                ..Default::default()
            })
            .into_response()
        }
    };
    let Some(password) = super::secrets::decrypt(&enc) else {
        return Json(super::probe::ProbeResponse {
            ok: false,
            error: Some("stored credentials unreadable (key changed?)".to_string()),
            ..Default::default()
        })
        .into_response();
    };

    // cookie_store(true) is critical: HN's /login response sets the
    // `user` cookie AND redirects to /news in one shot. Without the
    // cookie jar, the redirect follows but drops the auth cookie, and
    // we land on an anonymous /news page — so the success markers
    // never appear and every login looks like a failure.
    let http = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")
        .cookie_store(true)
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_default();
    let body = match http
        .post("https://news.ycombinator.com/login?goto=news")
        .form(&[("acct", username.as_str()), ("pw", password.as_str())])
        .send()
        .await
    {
        Ok(r) => r.text().await.unwrap_or_default(),
        Err(e) => {
            return Json(super::probe::ProbeResponse {
                ok: false,
                error: Some(format!("network error: {e}")),
                ..Default::default()
            })
            .into_response()
        }
    };
    let mut outcome = hn_login_outcome(&body);
    // Personalise messages with the username on the stored-creds path.
    if outcome.ok {
        outcome.info = Some(format!("login OK for @{}", username));
    } else if outcome.error.as_deref() == Some("login failed — wrong username or password") {
        outcome.error = Some("stored password no longer works — re-enter it".to_string());
    }
    Json(outcome).into_response()
}

/// POST /auth/me/hackernews/test — dry-run an HN login with the
/// given credentials. Nothing persists; the form uses the result to
/// show a live ✓/✗ badge next to the password field.
pub async fn test_hackernews(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<HackernewsCredsRequest>,
) -> impl IntoResponse {
    if current_user(&pool, &jar).await.is_none() {
        return (
            StatusCode::UNAUTHORIZED,
            Json(super::probe::ProbeResponse {
                ok: false,
                error: Some("not signed in".to_string()),
                ..Default::default()
            }),
        )
            .into_response();
    }
    let username = req.username.trim();
    let password = req.password.as_deref().unwrap_or("");
    if username.is_empty() || password.is_empty() {
        return Json(super::probe::ProbeResponse {
            ok: false,
            error: Some("username and password required".to_string()),
            ..Default::default()
        })
        .into_response();
    }
    let http = reqwest::Client::builder()
        // Realistic browser UA — HN's anti-bot heuristics trip on
        // bespoke / empty UAs and respond with a captcha challenge.
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")
        .cookie_store(true)
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_default();
    let resp = http
        .post("https://news.ycombinator.com/login?goto=news")
        .form(&[("acct", username), ("pw", password)])
        .send()
        .await;
    let body = match resp {
        Ok(r) => r.text().await.unwrap_or_default(),
        Err(e) => {
            return Json(super::probe::ProbeResponse {
                ok: false,
                error: Some(format!("network error: {e}")),
                ..Default::default()
            })
            .into_response()
        }
    };
    Json(hn_login_outcome(&body)).into_response()
}

/// Map an HN login response body to a ProbeResponse. Centralised so
/// the stored-creds and typed-creds handlers agree on what "success"
/// looks like. Body-only detection is intentional — the status code
/// is noisy (HN sometimes 200s on everything) but the body is stable.
fn hn_login_outcome(body: &str) -> super::probe::ProbeResponse {
    let trimmed = body.trim();
    // 429 rate-limit — HN returns a 2xx or 429 with a tiny "Sorry."
    // body. Distinguish from real logins so the UI can say "wait"
    // instead of "wrong password".
    if trimmed == "Sorry." || trimmed.starts_with("Sorry.") {
        return super::probe::ProbeResponse {
            ok: false,
            error: Some(
                "HN rate-limited this server — wait ~30 min, or paste your HN session cookie to skip login entirely"
                    .to_string(),
            ),
            ..Default::default()
        };
    }
    if body.contains("Bad login") {
        return super::probe::ProbeResponse {
            ok: false,
            error: Some("login failed — wrong username or password".to_string()),
            ..Default::default()
        };
    }
    if body.contains("Validation required") || body.contains("recaptcha") {
        return super::probe::ProbeResponse {
            ok: false,
            error: Some(
                "HN is asking for a captcha — paste your HN session cookie to bypass login"
                    .to_string(),
            ),
            ..Default::default()
        };
    }
    // Logged-in /news has the user's top-nav link as
    //   <a id=me href="user?id=NAME">NAME</a>
    // plus a karma badge (<span id=karma>).  Either is enough to be
    // sure the session is authenticated, since neither appears on an
    // anonymous page or a "Bad login" page.  We check for any of the
    // three forms HN has shipped (`id=me`, `id="me"`, `id=logout`) so
    // a minor markup change doesn't break the detector.
    let authed_markers = [
        "id=me",
        "id=\"me\"",
        "id='me'",
        "id=logout",
        "id=\"logout\"",
        "id='logout'",
        "id=karma",
        "id=\"karma\"",
        "logout?auth=",
    ];
    if authed_markers.iter().any(|m| body.contains(m)) {
        return super::probe::ProbeResponse {
            ok: true,
            info: Some("login OK".to_string()),
            ..Default::default()
        };
    }
    // Unknown shape — dump the full body to /tmp for offline inspection
    // (too large to stream back through the form). The error carries a
    // short snippet plus the dump path so a developer can cat it.
    let dump_path = "/tmp/hn-login-response.html";
    let _ = std::fs::write(dump_path, body);
    let snippet: String = body
        .chars()
        .take(180)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    super::probe::ProbeResponse {
        ok: false,
        error: Some(format!(
            "couldn't confirm login — body dumped to {}; first 180 chars: {}",
            dump_path,
            if snippet.is_empty() {
                "<empty body>".to_string()
            } else {
                snippet
            }
        )),
        ..Default::default()
    }
}

// ── Websites list ─────────────────────────────────────────────────────
//
// `PUT /auth/me/websites` — replace the unified web-source list in
// one jsonb_set. The form uses this instead of the generic profile
// PUT so it can auto-save on every probe change without racing other
// in-flight edits. Legacy `sources.blog` / `sources.sitemap` are
// dropped in the same transaction — the pipeline reads `sources.websites`
// going forward, and leaving the old keys around would double-fetch.
#[derive(Deserialize)]
pub struct WebsitesPatch {
    /// Fully-resolved entries as produced by the website probe.
    /// Shape: `{ input, kind: "feed"|"sitemap", url, url_filter?, tags }`.
    pub websites: Vec<serde_json::Value>,
}

pub async fn set_websites(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<WebsitesPatch>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let payload = serde_json::Value::Array(req.websites);

    // Diff old vs new website-source hostnames so we can soft-delete
    // documents whose source disappeared from the list — and revive
    // ones whose source was re-added (avoiding a re-fetch + re-index
    // round trip when the user toggles a source off and back on).
    let old_value: Option<serde_json::Value> =
        sqlx::query_scalar("SELECT sources->'websites' FROM users WHERE id = $1")
            .bind(me.id)
            .fetch_optional(&pool)
            .await
            .ok()
            .flatten();
    let old_hosts = website_hosts(old_value.as_ref());
    let new_hosts = website_hosts(Some(&payload));
    let removed: Vec<String> = old_hosts.difference(&new_hosts).cloned().collect();
    let restored: Vec<String> = new_hosts.difference(&old_hosts).cloned().collect();

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            tracing::error!(error = %e, "websites.set.tx_begin");
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("tx error: {e}")).into_response();
        }
    };

    if let Err(e) = sqlx::query(
        "UPDATE users
            SET sources = jsonb_set(
                    COALESCE(sources, '{}'::jsonb) - 'blog' - 'sitemap',
                    '{websites}',
                    $2::jsonb,
                    true
                ),
                updated_at = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .bind(&payload)
    .execute(&mut *tx)
    .await
    {
        tracing::error!(error = %e, "websites.set.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("store failed: {e}"),
        )
            .into_response();
    }

    // Tombstone documents whose hostname source just left the list.
    // `documents.source` is the bare hostname for website-derived docs
    // (set by `hostname_source_key` in the Python pipeline); twikit /
    // arxiv / github docs use brand keys like "twitter" / "arxiv" and
    // can't collide with a hostname here.
    if !removed.is_empty() {
        if let Err(e) = sqlx::query(
            "UPDATE documents
                SET to_delete = TRUE, updated_at = now()
              WHERE user_id = $1
                AND source = ANY($2)
                AND to_delete = FALSE",
        )
        .bind(me.id)
        .bind(&removed)
        .execute(&mut *tx)
        .await
        {
            tracing::error!(error = %e, "websites.set.tombstone_failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("tombstone failed: {e}"),
            )
                .into_response();
        }
    }

    // Revive any tombstoned docs whose hostname source just came back.
    if !restored.is_empty() {
        if let Err(e) = sqlx::query(
            "UPDATE documents
                SET to_delete = FALSE, updated_at = now()
              WHERE user_id = $1
                AND source = ANY($2)
                AND to_delete = TRUE",
        )
        .bind(me.id)
        .bind(&restored)
        .execute(&mut *tx)
        .await
        {
            tracing::error!(error = %e, "websites.set.revive_failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("revive failed: {e}"),
            )
                .into_response();
        }
    }

    if let Err(e) = tx.commit().await {
        tracing::error!(error = %e, "websites.set.commit");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("commit error: {e}"),
        )
            .into_response();
    }
    me_after_update(&pool, me.id).await
}

/// Pull the set of source-hostname keys from a `sources.websites` JSONB
/// array. Each entry's `url` (the resolved feed/sitemap URL the probe
/// settled on) is parsed through `reqwest::Url` and `www.` is stripped
/// — same canonicalisation `hostname_source_key` does in the Python
/// pipeline so the strings line up with `documents.source`.
fn website_hosts(value: Option<&serde_json::Value>) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let arr = match value.and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return out,
    };
    for entry in arr {
        let url_str = entry
            .get("url")
            .and_then(|v| v.as_str())
            .or_else(|| entry.get("input").and_then(|v| v.as_str()))
            .unwrap_or("");
        if url_str.is_empty() {
            continue;
        }
        if let Ok(u) = reqwest::Url::parse(url_str) {
            if let Some(host) = u.host_str() {
                let h = host.strip_prefix("www.").unwrap_or(host).to_lowercase();
                if !h.is_empty() {
                    out.insert(h);
                }
            }
        }
    }
    out
}

pub async fn clear_hackernews(State(pool): State<PgPool>, jar: CookieJar) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "UPDATE users
            SET sources = sources - 'hackernews',
                updated_at = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .execute(&pool)
    .await;
    me_after_update(&pool, me.id).await
}

// ── Zotero credentials ──────────────────────────────────────────────────
//
// `PUT /auth/me/zotero` — enable the Zotero importer.
//    body: { "libraryId": "12345", "libraryType": "user"|"group",
//             "apiKey": "..." }
//    Library id is not secret (Zotero shows it openly), but the API key
//    IS — so it's encrypted with the same HN_ENCRYPTION_KEY.
//
// `DELETE /auth/me/zotero` — wipe creds.

#[derive(Deserialize)]
pub struct ZoteroCredsRequest {
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

pub async fn set_zotero(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<ZoteroCredsRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    if req.api_key.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "apiKey required").into_response();
    }

    // Discover everything the key unlocks so the pipeline doesn't have
    // to know a library id/type up front. The fetch uses the plaintext
    // key we just received; only the encrypted copy is persisted.
    let http = reqwest::Client::builder()
        .user_agent("knowledge-api/0.1 zotero-setup")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    // Trim ambient whitespace so a copy-paste with a trailing space
    // doesn't masquerade as a "rejected key" error.
    let api_key = req.api_key.trim().to_string();
    if api_key.is_empty() {
        return (StatusCode::BAD_REQUEST, "apiKey required").into_response();
    }

    let keys_resp = http
        .get("https://api.zotero.org/keys/current")
        .header("Zotero-API-Key", &api_key)
        .header("Zotero-API-Version", "3")
        .send()
        .await;
    let keys_json: serde_json::Value = match keys_resp {
        Ok(r) if r.status().is_success() => r.json().await.unwrap_or_default(),
        Ok(r) => {
            // 401/403 from /keys/current means Zotero doesn't recognise
            // the key. Most often: typo (trailing characters cut off),
            // the key was deleted from zotero.org/settings/keys, or it
            // was created without "Allow library access" enabled. The
            // hint below points the user at the right place to check.
            let status = r.status();
            let hint = match status.as_u16() {
                401 | 403 => {
                    " — invalid or revoked key. Recreate one at \
                     zotero.org/settings/keys/new with \"Allow library access\" checked."
                }
                _ => "",
            };
            return (
                StatusCode::BAD_REQUEST,
                format!("zotero rejected key: {status}{hint}"),
            )
                .into_response();
        }
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, format!("zotero probe failed: {e}")).into_response()
        }
    };
    let user_id = keys_json
        .get("userID")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if user_id <= 0 {
        return (StatusCode::BAD_REQUEST, "zotero key has no user id").into_response();
    }

    // Encrypt only after Zotero has accepted the key — saves a hop
    // when the key is bad and keeps a malformed cipher out of the DB.
    let Some(enc) = super::secrets::encrypt(&api_key) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "HN_ENCRYPTION_KEY is not configured — refusing to store plaintext",
        )
            .into_response();
    };

    // Count items by reading the `Total-Results` header from a
    // limit=1 GET on `/items/top`. One call per library; cheap.
    async fn count_items(http: &reqwest::Client, api_key: &str, path: &str) -> Option<i64> {
        let r = http
            .get(format!("https://api.zotero.org{}/items/top?limit=1", path))
            .header("Zotero-API-Key", api_key)
            .header("Zotero-API-Version", "3")
            .send()
            .await
            .ok()?;
        r.headers()
            .get("Total-Results")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<i64>().ok())
    }

    let personal_count = count_items(&http, &api_key, &format!("/users/{}", user_id))
        .await
        .unwrap_or(0);

    // Pull the group list (with item counts) so the pipeline has a fixed
    // set it can iterate without redoing discovery on every run. New
    // groups the user joins later will be picked up on their next save.
    let mut groups: Vec<serde_json::Value> = Vec::new();
    if let Ok(r) = http
        .get(format!("https://api.zotero.org/users/{}/groups", user_id))
        .header("Zotero-API-Key", &api_key)
        .header("Zotero-API-Version", "3")
        .send()
        .await
    {
        if let Ok(list) = r.json::<serde_json::Value>().await {
            if let Some(arr) = list.as_array() {
                for g in arr {
                    let id = g.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let name = g
                        .get("data")
                        .and_then(|d| d.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if id > 0 {
                        let count = count_items(&http, &api_key, &format!("/groups/{}", id))
                            .await
                            .unwrap_or(0);
                        groups.push(serde_json::json!({
                            "id": id,
                            "name": name,
                            "count": count,
                        }));
                    }
                }
            }
        }
    }

    let patch = serde_json::json!({
        "api_key_enc": enc,
        "user_id": user_id,
        "personal_count": personal_count,
        "groups": groups,
    });
    if let Err(e) = sqlx::query(
        "UPDATE users
            SET sources = jsonb_set(
                    COALESCE(sources, '{}'::jsonb),
                    '{zotero}',
                    $2::jsonb,
                    true
                ),
                updated_at = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .bind(&patch)
    .execute(&pool)
    .await
    {
        tracing::error!(error = %e, "zotero.set.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("store failed: {e}"),
        )
            .into_response();
    }
    me_after_update(&pool, me.id).await
}

pub async fn clear_zotero(State(pool): State<PgPool>, jar: CookieJar) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "UPDATE users
            SET sources = sources - 'zotero',
                updated_at = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .execute(&pool)
    .await;
    me_after_update(&pool, me.id).await
}

/// GET /auth/me/zotero/items
///
/// Server-side proxy for the browser sync. Decrypts the stored
/// Zotero API key, paginates through every library the key unlocks
/// (personal + each discovered group), normalises items to our
/// `{url, title, summary, date, tags}` shape, and returns the
/// flat array. Items without a URL are dropped.
///
/// Why this lives on the server: the API key is AES-encrypted at
/// rest and the `HN_ENCRYPTION_KEY` only exists on the server, so
/// the browser literally can't read the key. This endpoint is the
/// thinnest possible relay — auth-gated, decrypts in-memory, fans
/// out to Zotero, returns rows.
pub async fn fetch_zotero_items(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    // Pull the stored Zotero block (key + userID + group list).
    let row: Option<(Option<serde_json::Value>,)> =
        sqlx::query_as("SELECT sources->'zotero' FROM users WHERE id = $1")
            .bind(me.id)
            .fetch_optional(&pool)
            .await
            .unwrap_or_default();
    let zot = match row.and_then(|r| r.0) {
        Some(v) if !v.is_null() => v,
        _ => return (StatusCode::BAD_REQUEST, "Zotero not configured").into_response(),
    };
    let api_key_enc = zot
        .get("api_key_enc")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let user_id = zot.get("user_id").and_then(|v| v.as_i64()).unwrap_or(0);
    if api_key_enc.is_empty() || user_id <= 0 {
        return (StatusCode::BAD_REQUEST, "Zotero not configured").into_response();
    }
    let api_key = match super::secrets::decrypt(api_key_enc) {
        Some(k) => k,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "could not decrypt Zotero key",
            )
                .into_response()
        }
    };

    // Walk personal + every group library.
    let mut paths = vec![format!("/users/{user_id}")];
    if let Some(groups) = zot.get("groups").and_then(|v| v.as_array()) {
        for g in groups {
            if let Some(id) = g.get("id").and_then(|v| v.as_i64()) {
                paths.push(format!("/groups/{id}"));
            }
        }
    }

    let http = match reqwest::Client::builder()
        .user_agent("knowledge-api/0.1 zotero-fetch")
        .timeout(std::time::Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "client init").into_response();
        }
    };

    let mut docs: Vec<serde_json::Value> = Vec::new();
    'libs: for path in &paths {
        let mut start = 0i64;
        loop {
            let url = format!(
                "https://api.zotero.org{path}/items/top?limit=100&start={start}&include=data"
            );
            let resp = match http
                .get(&url)
                .header("Zotero-API-Key", &api_key)
                .header("Zotero-API-Version", "3")
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => r,
                Ok(_) | Err(_) => continue 'libs, // skip a library that errors
            };
            let items: Vec<serde_json::Value> = resp.json().await.unwrap_or_default();
            if items.is_empty() {
                break;
            }
            for item in &items {
                let Some(data) = item.get("data") else {
                    continue;
                };
                let item_url = data
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if item_url.is_empty() {
                    continue;
                }
                let title = data.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let summary = data
                    .get("abstractNote")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let raw_date = data.get("dateAdded").and_then(|v| v.as_str()).unwrap_or("");
                // Zotero `dateAdded` is `YYYY-MM-DDTHH:MM:SSZ`; we
                // only keep the date portion to match the doc shape.
                let date = if raw_date.len() >= 10 {
                    &raw_date[..10]
                } else {
                    ""
                };
                let tags: Vec<String> = data
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|t| t.get("tag").and_then(|v| v.as_str()))
                            .map(|s| s.to_lowercase())
                            .collect()
                    })
                    .unwrap_or_default();
                docs.push(serde_json::json!({
                    "url": item_url,
                    "title": title,
                    "summary": summary,
                    "date": date,
                    "tags": tags,
                }));
            }
            if items.len() < 100 {
                break;
            }
            start += 100;
            if start > 5000 {
                // Safety stop — a single library with > 5k items is
                // possible but rare; force a re-run rather than
                // hold the connection open indefinitely.
                break;
            }
        }
    }
    Json(docs).into_response()
}

// ── Twitter/X cookie credentials ────────────────────────────────────────
//
// Twitter's bookmarks endpoint is user-private — no public API exposes
// it. The only reliable auth path is the browser's session cookies
// (`auth_token` + `ct0`), which the user pastes from their DevTools.
// We encrypt them together as a single JSON blob into
// `sources.twitter.cookies_enc`; the Python pipeline decrypts + feeds
// them to twikit at crawl time.

#[derive(Deserialize)]
pub struct TwitterCookiesRequest {
    #[serde(rename = "authToken")]
    pub auth_token: String,
    pub ct0: String,
}

pub async fn set_twitter_cookies(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<TwitterCookiesRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let auth_token = req.auth_token.trim();
    let ct0 = req.ct0.trim();
    if auth_token.is_empty() || ct0.is_empty() {
        return (StatusCode::BAD_REQUEST, "authToken and ct0 required").into_response();
    }
    // Format sanity check — both cookies are lowercase hex. No live
    // call to Twitter: every publicly-reachable verify endpoint is
    // undocumented and rotates, so we'd just be trading today's paste
    // error for tomorrow's detector breakage. twikit at pipeline time
    // is the single source of truth for "do they still work".
    fn looks_like_hex_cookie(s: &str, min: usize) -> bool {
        s.len() >= min && s.chars().all(|c| c.is_ascii_hexdigit())
    }
    if !looks_like_hex_cookie(auth_token, 30) {
        return (
            StatusCode::BAD_REQUEST,
            "authToken doesn't look like an x.com cookie (expected lowercase hex, 40 chars)",
        )
            .into_response();
    }
    if !looks_like_hex_cookie(ct0, 30) {
        return (
            StatusCode::BAD_REQUEST,
            "ct0 doesn't look like an x.com cookie (expected lowercase hex, 32+ chars)",
        )
            .into_response();
    }
    let plaintext = serde_json::json!({
        "auth_token": auth_token,
        "ct0": ct0,
    })
    .to_string();
    let Some(enc) = super::secrets::encrypt(&plaintext) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "HN_ENCRYPTION_KEY is not configured — refusing to store plaintext",
        )
            .into_response();
    };
    let patch = serde_json::json!({ "cookies_enc": enc });
    if let Err(e) = sqlx::query(
        "UPDATE users
            SET sources = jsonb_set(
                    COALESCE(sources, '{}'::jsonb),
                    '{twitter}',
                    COALESCE(sources->'twitter', '{}'::jsonb) || $2::jsonb,
                    true
                ),
                updated_at = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .bind(&patch)
    .execute(&pool)
    .await
    {
        tracing::error!(error = %e, "twitter.cookies.set.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("store failed: {e}"),
        )
            .into_response();
    }
    me_after_update(&pool, me.id).await
}

pub async fn clear_twitter_cookies(
    State(pool): State<PgPool>,
    jar: CookieJar,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "UPDATE users
            SET sources = jsonb_set(
                    sources,
                    '{twitter}',
                    COALESCE(sources->'twitter', '{}'::jsonb) - 'cookies_enc',
                    true
                ),
                updated_at = now()
          WHERE id = $1
            AND sources ? 'twitter'",
    )
    .bind(me.id)
    .execute(&pool)
    .await;
    me_after_update(&pool, me.id).await
}

// ── Stack Overflow OAuth 2.0 ────────────────────────────────────────────
//
// Stack Exchange's explicit server-side flow:
//   /auth/stackoverflow/start       → mint state, redirect to authorize
//   /auth/stackoverflow/callback    → verify state, exchange code, fetch
//                                     /me, store encrypted access token +
//                                     user_id in sources.stackoverflow
//   DELETE /auth/me/stackoverflow/auth → wipe the access token only (the
//                                     user_id stays so the public
//                                     Answers fetcher keeps running).
//
// We request scope `no_expiry private_info`:
//   no_expiry   — access_token never expires (StackApps allows this;
//                 avoids a refresh dance the API doesn't expose anyway).
//   private_info — required for /me/favorites (their bookmarks).
//
// The app "Key" (distinct from client_id) lifts the daily quota from
// 300 → 10 000. Stored separately so the Python pipeline can attach it
// to every API call.

const STACK_STATE_COOKIE: &str = "k_stack_state";

fn stack_redirect_url() -> String {
    std::env::var("STACKOVERFLOW_OAUTH_REDIRECT_URL")
        .unwrap_or_else(|_| "http://localhost:8080/auth/stackoverflow/callback".to_string())
}

#[derive(Deserialize)]
pub struct StackCallbackParams {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Deserialize)]
struct StackTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

pub async fn stackoverflow_start(jar: CookieJar) -> impl IntoResponse {
    let Ok(client_id) = std::env::var("STACKOVERFLOW_CLIENT_ID") else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "STACKOVERFLOW_CLIENT_ID is not configured",
        )
            .into_response();
    };
    let state = random_token();
    let url = format!(
        "https://stackoverflow.com/oauth?client_id={}&redirect_uri={}&scope={}&state={}",
        urlencoding::encode(&client_id),
        urlencoding::encode(&stack_redirect_url()),
        urlencoding::encode("no_expiry private_info"),
        urlencoding::encode(&state),
    );

    let mut state_cookie = Cookie::new(STACK_STATE_COOKIE, state);
    state_cookie.set_path("/auth");
    state_cookie.set_http_only(true);
    state_cookie.set_same_site(SameSite::Lax);
    state_cookie.set_secure(cookie_secure());
    state_cookie.set_max_age(TimeDuration::minutes(OAUTH_STATE_TTL_MINUTES));

    (jar.add(state_cookie), Redirect::to(&url)).into_response()
}

pub async fn stackoverflow_callback(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Query(params): Query<StackCallbackParams>,
) -> Response {
    if let Some(err) = params.error {
        let desc = params.error_description.unwrap_or_default();
        return (
            StatusCode::BAD_REQUEST,
            format!("Stack Overflow OAuth error: {err} — {desc}"),
        )
            .into_response();
    }
    let Some(code) = params.code else {
        return (StatusCode::BAD_REQUEST, "missing `code`").into_response();
    };
    let Some(state) = params.state else {
        return (StatusCode::BAD_REQUEST, "missing `state`").into_response();
    };

    let Some(me) = current_user(&pool, &jar).await else {
        return (StatusCode::UNAUTHORIZED, "sign in first").into_response();
    };

    let cookie_state = jar.get(STACK_STATE_COOKIE).map(|c| c.value().to_string());
    if cookie_state.as_deref() != Some(state.as_str()) {
        return (StatusCode::BAD_REQUEST, "oauth state mismatch").into_response();
    }

    let (Ok(client_id), Ok(client_secret)) = (
        std::env::var("STACKOVERFLOW_CLIENT_ID"),
        std::env::var("STACKOVERFLOW_CLIENT_SECRET"),
    ) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "STACKOVERFLOW_CLIENT_ID / STACKOVERFLOW_CLIENT_SECRET not configured",
        )
            .into_response();
    };
    let key = std::env::var("STACKOVERFLOW_KEY").unwrap_or_default();

    let http = reqwest::Client::builder()
        .user_agent("knowledge-api/0.1 stackoverflow-oauth")
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_default();

    // Exchange code → access_token. Stack Exchange accepts the JSON
    // variant when we call the ".../json" suffix and returns
    // `{ access_token, expires? }`.
    let token = match http
        .post("https://stackoverflow.com/oauth/access_token/json")
        .form(&[
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("code", code.as_str()),
            ("redirect_uri", stack_redirect_url().as_str()),
        ])
        .send()
        .await
    {
        Ok(r) => match r.json::<StackTokenResponse>().await {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("Stack Overflow token parse failed: {e}"),
                )
                    .into_response()
            }
        },
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("Stack Overflow token exchange failed: {e}"),
            )
                .into_response()
        }
    };
    let Some(access) = token.access_token else {
        return (
            StatusCode::BAD_GATEWAY,
            format!(
                "Stack Overflow did not return an access token: {} — {}",
                token.error.unwrap_or_default(),
                token.error_description.unwrap_or_default()
            ),
        )
            .into_response();
    };

    // Discover the Stack Exchange user_id so we can pre-fill the form.
    let me_url = format!(
        "https://api.stackexchange.com/2.3/me?site=stackoverflow&access_token={}&key={}",
        urlencoding::encode(&access),
        urlencoding::encode(&key),
    );
    let so_user_id: i64 = match http.get(&me_url).send().await {
        Ok(r) => match r.json::<serde_json::Value>().await {
            Ok(body) => body
                .get("items")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|u| u.get("user_id"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            Err(_) => 0,
        },
        Err(_) => 0,
    };
    if so_user_id == 0 {
        return (
            StatusCode::BAD_GATEWAY,
            "failed to resolve Stack Overflow user id",
        )
            .into_response();
    }

    // Discover *every* Stack Exchange site this account is active on.
    // `/me/associated` doesn't need a `site` param (it crosses the
    // whole network) and returns one entry per site with the canonical
    // `api_site_parameter` we need to crawl per-site content.
    let assoc_url = format!(
        "https://api.stackexchange.com/2.3/me/associated?access_token={}&key={}&types=main_site&pagesize=100",
        urlencoding::encode(&access),
        urlencoding::encode(&key),
    );
    let associated_sites: Vec<serde_json::Value> = match http.get(&assoc_url).send().await {
        Ok(r) => match r.json::<serde_json::Value>().await {
            Ok(body) => body
                .get("items")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|it| {
                            let api = it.get("site_url").and_then(|v| v.as_str())?;
                            // Derive api_site_parameter from site_url
                            // (e.g. https://stackoverflow.com → stackoverflow,
                            //  https://serverfault.com → serverfault,
                            //  https://math.stackexchange.com → math).
                            let host = api
                                .trim_start_matches("https://")
                                .trim_start_matches("http://")
                                .split('/')
                                .next()
                                .unwrap_or("")
                                .to_string();
                            let param = if let Some(sub) = host.strip_suffix(".stackexchange.com") {
                                sub.to_string()
                            } else {
                                host.trim_end_matches(".com").to_string()
                            };
                            let user_id = it.get("user_id").and_then(|v| v.as_i64()).unwrap_or(0);
                            let name = it
                                .get("site_name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let reputation =
                                it.get("reputation").and_then(|v| v.as_i64()).unwrap_or(0);
                            Some(serde_json::json!({
                                "api_site_parameter": param,
                                "site_name": name,
                                "user_id": user_id,
                                "reputation": reputation,
                            }))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };

    let Some(enc) = super::secrets::encrypt(&access) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "HN_ENCRYPTION_KEY is not configured — refusing to store plaintext",
        )
            .into_response();
    };

    // Merge into sources.stackoverflow, preserving max_pages/min_score
    // if they were already set.
    let patch = serde_json::json!({
        "user_id": so_user_id,
        "access_token_enc": enc,
        "associated_sites": associated_sites,
    });
    if let Err(e) = sqlx::query(
        "UPDATE users
            SET sources = jsonb_set(
                    COALESCE(sources, '{}'::jsonb),
                    '{stackoverflow}',
                    COALESCE(sources->'stackoverflow', '{}'::jsonb) || $2::jsonb,
                    true
                ),
                updated_at = now()
          WHERE id = $1",
    )
    .bind(me.id)
    .bind(&patch)
    .execute(&pool)
    .await
    {
        tracing::error!(error = %e, "stackoverflow.oauth.store_failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("store failed: {e}"),
        )
            .into_response();
    }

    let jar = jar.remove(clear_cookie(STACK_STATE_COOKIE));
    (jar, Redirect::to(&post_login_url())).into_response()
}

// ── GitHub sign-in (OAuth) ───────────────────────────────────────────────
//
// Unlike the Stack Overflow flow above (which only attaches an SO
// account to an already-signed-in user for ingestion purposes),
// GitHub OAuth here is a *login* mechanism: completing the round-trip
// mints a session cookie.
//
// Account dispatch on callback follows three rules, in order:
//
//   1. We already know this GitHub user — `oauth_identities` has a
//      row with (provider='github', provider_user_id=<gh_id>). Log
//      in as the linked Knowledge user.
//
//   2. The GitHub login matches a VIP user's `sources.github`. This
//      is the "claim my personality" path. Crucially the lookup is
//      gated by `vip = TRUE`, so a non-VIP fraudster who typed a
//      VIP's handle into their own `sources.github` won't match.
//      We link the OAuth identity to that VIP row and log in.
//
//   3. Nothing matched — provision a brand-new (non-VIP) account
//      from the GitHub profile and log in.
//
// `sources.github` is user-typed and unverifiable; this table is the
// only source of truth for "the user owns that GitHub account."

const GITHUB_STATE_COOKIE: &str = "k_github_state";

fn github_redirect_url() -> String {
    std::env::var("OAUTH_REDIRECT_URL")
        .unwrap_or_else(|_| "http://localhost:8080/auth/github/callback".to_string())
}

#[derive(Deserialize)]
pub struct GithubCallbackParams {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

#[derive(Deserialize)]
struct GithubTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Deserialize)]
struct GithubUser {
    id: i64,
    login: String,
    name: Option<String>,
    email: Option<String>,
    avatar_url: Option<String>,
    bio: Option<String>,
}

pub async fn github_start(jar: CookieJar) -> impl IntoResponse {
    let Ok(client_id) = std::env::var("GITHUB_CLIENT_ID") else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "GITHUB_CLIENT_ID is not configured",
        )
            .into_response();
    };
    let state = random_token();
    // `read:user` reads the profile (id, login, name, bio, avatar_url).
    // `user:email` lets us see verified emails when the user has no
    // public email — purely so we can pre-fill the account row.
    let url = format!(
        "https://github.com/login/oauth/authorize?client_id={}&redirect_uri={}&scope={}&state={}&allow_signup=true",
        urlencoding::encode(&client_id),
        urlencoding::encode(&github_redirect_url()),
        urlencoding::encode("read:user user:email"),
        urlencoding::encode(&state),
    );

    let mut state_cookie = Cookie::new(GITHUB_STATE_COOKIE, state);
    state_cookie.set_path("/auth");
    state_cookie.set_http_only(true);
    state_cookie.set_same_site(SameSite::Lax);
    state_cookie.set_secure(cookie_secure());
    state_cookie.set_max_age(TimeDuration::minutes(OAUTH_STATE_TTL_MINUTES));

    (jar.add(state_cookie), Redirect::to(&url)).into_response()
}

pub async fn github_callback(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Query(params): Query<GithubCallbackParams>,
) -> Response {
    if let Some(err) = params.error {
        let desc = params.error_description.unwrap_or_default();
        return (
            StatusCode::BAD_REQUEST,
            format!("GitHub OAuth error: {err} — {desc}"),
        )
            .into_response();
    }
    let Some(code) = params.code else {
        return (StatusCode::BAD_REQUEST, "missing `code`").into_response();
    };
    let Some(state) = params.state else {
        return (StatusCode::BAD_REQUEST, "missing `state`").into_response();
    };

    let cookie_state = jar.get(GITHUB_STATE_COOKIE).map(|c| c.value().to_string());
    if cookie_state.as_deref() != Some(state.as_str()) {
        return (StatusCode::BAD_REQUEST, "oauth state mismatch").into_response();
    }
    // One-shot — consume the state cookie regardless of outcome below.
    let jar = jar.remove(clear_cookie(GITHUB_STATE_COOKIE));

    let (Ok(client_id), Ok(client_secret)) = (
        std::env::var("GITHUB_CLIENT_ID"),
        std::env::var("GITHUB_CLIENT_SECRET"),
    ) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET not configured",
        )
            .into_response();
    };

    let http = reqwest::Client::builder()
        .user_agent("knowledge-api/0.1 github-oauth")
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_default();

    // Exchange code → access_token.
    let token: GithubTokenResponse = match http
        .post("https://github.com/login/oauth/access_token")
        .header("Accept", "application/json")
        .form(&[
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("code", code.as_str()),
            ("redirect_uri", github_redirect_url().as_str()),
        ])
        .send()
        .await
    {
        Ok(r) => match r.json().await {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("GitHub token parse failed: {e}"),
                )
                    .into_response()
            }
        },
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("GitHub token exchange failed: {e}"),
            )
                .into_response()
        }
    };
    let Some(access) = token.access_token else {
        return (
            StatusCode::BAD_GATEWAY,
            format!(
                "GitHub did not return an access token: {} — {}",
                token.error.unwrap_or_default(),
                token.error_description.unwrap_or_default(),
            ),
        )
            .into_response();
    };

    // Fetch the authenticated user.
    let gh_user: GithubUser = match http
        .get("https://api.github.com/user")
        .header("Authorization", format!("Bearer {access}"))
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
    {
        Ok(r) => match r.json().await {
            Ok(u) => u,
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("GitHub /user parse failed: {e}"),
                )
                    .into_response()
            }
        },
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("GitHub /user fetch failed: {e}"),
            )
                .into_response()
        }
    };

    // If no public email was returned, try /user/emails for the primary
    // verified address. Fine to leave empty if the user denied the scope.
    let email = match gh_user.email.clone() {
        Some(e) if !e.is_empty() => Some(e.to_lowercase()),
        _ => fetch_github_primary_email(&http, &access).await,
    };

    let gh_id = gh_user.id.to_string();
    let gh_login = gh_user.login.clone();
    let gh_login_lower = gh_login.to_lowercase();

    // ── Dispatch ────────────────────────────────────────────────────────
    //
    // 1. Returning OAuth user: identity already linked.
    let existing_user_id: Option<i64> = sqlx::query_scalar(
        "SELECT user_id FROM oauth_identities
          WHERE provider = 'github' AND provider_user_id = $1",
    )
    .bind(&gh_id)
    .fetch_optional(&pool)
    .await
    .unwrap_or(None);

    let user_id = if let Some(uid) = existing_user_id {
        // Refresh login + email + updated_at — handles renames.
        let _ = sqlx::query(
            "UPDATE oauth_identities
                SET provider_login = $2,
                    provider_email = $3,
                    updated_at     = now()
              WHERE provider = 'github' AND provider_user_id = $1",
        )
        .bind(&gh_id)
        .bind(&gh_login)
        .bind(&email)
        .execute(&pool)
        .await;
        uid
    } else {
        // 2. VIP-claim path. Match the OAuth login against the
        //    `sources.github` JSON across VIP rows only.
        //
        //    `sources.github` can be:
        //      - a string ("karpathy")
        //      - an array of strings (["karpathy", "altcaped"])
        //      - an object ({"username": "karpathy"})
        //    so we normalize via jsonb_path_query_first / lower().
        let vip_id: Option<i64> = sqlx::query_scalar(
            "SELECT u.id FROM users u
              WHERE u.vip = TRUE
                AND (
                    lower(u.sources->>'github') = $1
                    OR lower(u.sources->'github'->>'username') = $1
                    OR EXISTS (
                        SELECT 1
                          FROM jsonb_array_elements_text(
                                   CASE WHEN jsonb_typeof(u.sources->'github') = 'array'
                                        THEN u.sources->'github'
                                        ELSE '[]'::jsonb END
                               ) AS elt
                         WHERE lower(elt) = $1
                    )
                )
              LIMIT 1",
        )
        .bind(&gh_login_lower)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);

        if let Some(vid) = vip_id {
            // Backfill the VIP row with fields the OAuth profile
            // now confirms — but only when they're currently null.
            // Never overwrite existing values (the operator-curated
            // VIP name/avatar takes precedence over GitHub's).
            let _ = sqlx::query(
                "UPDATE users SET
                    email   = COALESCE(email, $2),
                    name    = COALESCE(NULLIF(name, ''), $3),
                    avatar  = COALESCE(NULLIF(avatar, ''), $4)
                  WHERE id = $1",
            )
            .bind(vid)
            .bind(&email)
            .bind(gh_user.name.as_deref().unwrap_or(""))
            .bind(gh_user.avatar_url.as_deref().unwrap_or(""))
            .execute(&pool)
            .await;
            vid
        } else {
            // 3. Fresh signup. Slug starts from the GitHub login;
            //    append a numeric suffix if it's taken. Reserved
            //    slugs are skipped too — though "github" is reserved
            //    so users named after platform terms get nudged.
            let base_slug = slugify(&gh_login);
            let mut slug = base_slug.clone();
            let mut i = 2;
            while slug_is_reserved(&slug)
                || sqlx::query_scalar::<_, i64>("SELECT 1 FROM users WHERE username = $1 LIMIT 1")
                    .bind(&slug)
                    .fetch_optional(&pool)
                    .await
                    .unwrap_or(None)
                    .is_some()
            {
                slug = format!("{base_slug}-{i}");
                i += 1;
                if i > 50 {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "could not allocate a free slug",
                    )
                        .into_response();
                }
            }

            let name = gh_user.name.clone().unwrap_or_else(|| gh_login.clone());
            // Pre-fill `sources.github = [<login>]` so the new row
            // is internally consistent. The OAuth identity row is
            // still the security boundary — `sources` is just for
            // display.
            let sources = serde_json::json!({ "github": [gh_login.clone()] });

            let new_id: i64 = match sqlx::query_scalar(
                "INSERT INTO users (
                    username, email, name, index_name, avatar,
                    description, sources, email_verified
                 ) VALUES (
                    $1, $2, $3, $1, $4,
                    $5, $6, TRUE
                 ) RETURNING id",
            )
            .bind(&slug)
            .bind(&email)
            .bind(&name)
            .bind(gh_user.avatar_url.unwrap_or_default())
            .bind(gh_user.bio.unwrap_or_default())
            .bind(&sources)
            .fetch_one(&pool)
            .await
            {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!(error = %e, "github.oauth.insert.failed");
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "could not create account",
                    )
                        .into_response();
                }
            };
            new_id
        }
    };

    // Upsert the oauth_identities row (covers fresh + claim paths;
    // for returning users we already updated above, which is fine —
    // the ON CONFLICT below is a no-op then).
    let _ = sqlx::query(
        "INSERT INTO oauth_identities (provider, provider_user_id,
                                       provider_login, provider_email, user_id)
              VALUES ('github', $1, $2, $3, $4)
         ON CONFLICT (provider, provider_user_id) DO UPDATE SET
              provider_login = EXCLUDED.provider_login,
              provider_email = EXCLUDED.provider_email,
              user_id        = EXCLUDED.user_id,
              updated_at     = now()",
    )
    .bind(&gh_id)
    .bind(&gh_login)
    .bind(&email)
    .bind(user_id)
    .execute(&pool)
    .await;

    let session_id = match mint_session(&pool, user_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "github.oauth.session.failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "session creation failed").into_response();
        }
    };
    let jar = jar.add(build_session_cookie(session_id, SESSION_TTL_DAYS));
    // Land on the frontend with a query flag so the SPA can pop the
    // welcome panel / hide the login modal cleanly.
    (jar, Redirect::to("/?github=1")).into_response()
}

#[derive(Deserialize)]
struct GithubEmailEntry {
    email: String,
    primary: bool,
    verified: bool,
}

async fn fetch_github_primary_email(http: &reqwest::Client, token: &str) -> Option<String> {
    let r = http
        .get("https://api.github.com/user/emails")
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .ok()?;
    let list: Vec<GithubEmailEntry> = r.json().await.ok()?;
    list.into_iter()
        .find(|e| e.primary && e.verified)
        .map(|e| e.email.to_lowercase())
}

// ── Cross-user document bookmarking ─────────────────────────────────────
//
// Lets a signed-in user save another personality's document into their
// own library. The `documents` table is keyed by (user_id, url), so
// saving the same URL twice for the same user is a no-op via ON
// CONFLICT DO NOTHING. DELETE removes the row only for the current
// user — it can never touch other users' copies.

#[derive(Deserialize)]
pub struct SaveDocumentRequest {
    pub url: String,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub date: Option<String>,
    pub tags: Option<Vec<String>>,
    #[serde(rename = "extra-tags", alias = "extra_tags")]
    pub extra_tags: Option<Vec<String>>,
    pub source: Option<String>,
    pub source_url: Option<String>,
    /// Audience. Defaults to TRUE — visible to followers. When the
    /// caller flips the "Make private" toggle on compose this lands
    /// as FALSE so the doc stays in their library only.
    pub public: Option<bool>,
    /// Inline previews for every external URL the document points
    /// at. Each entry is shaped `{url, host, title, summary, image}`.
    /// The pipeline + bookmark dialog populate this so the card
    /// renderer can show preview tiles without a per-view OG fetch;
    /// passing an empty array (or omitting the field) leaves the
    /// row's existing `linked_urls` untouched on conflict.
    #[serde(rename = "linked_urls", alias = "linkedUrls")]
    pub linked_urls: Option<serde_json::Value>,
    /// Flat hostnames extracted from `linked_urls`. Must mirror the
    /// host values inside `linked_urls` — the source-filter SQL
    /// uses this column's GIN index so any divergence between the
    /// two surfaces would silently hide rows. We re-derive it from
    /// the payload server-side if `linked_urls` is provided without
    /// a matching `link_hosts`.
    #[serde(rename = "link_hosts", alias = "linkHosts")]
    pub link_hosts: Option<Vec<String>>,
}

pub async fn save_document(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<SaveDocumentRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let url = req.url.trim().to_string();
    if url.is_empty() {
        return (StatusCode::BAD_REQUEST, "url required").into_response();
    }

    // Parse the date, falling back to today when the caller didn't
    // supply one or sent something unparseable.
    let result = sqlx::query(
        "INSERT INTO documents (
            user_id, url, title, summary, date, tags, extra_tags,
            source, source_url
         ) VALUES (
            $1, $2, $3, $4,
            NULLIF($5, '')::date,
            COALESCE($6, '{}')::text[],
            COALESCE($7, '{}')::text[],
            COALESCE($8, ''),
            NULLIF($9, '')
         )
         ON CONFLICT (user_id, url) DO NOTHING",
    )
    .bind(me.id)
    .bind(&url)
    .bind(req.title.unwrap_or_default())
    .bind(req.summary.unwrap_or_default())
    .bind(req.date.unwrap_or_default())
    .bind(req.tags.as_deref())
    .bind(req.extra_tags.as_deref())
    .bind(req.source.as_deref())
    .bind(req.source_url.as_deref())
    .execute(&pool)
    .await;

    match result {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            tracing::error!(error = %e, "bookmark.save.failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("save failed: {e}"),
            )
                .into_response()
        }
    }
}

#[derive(Deserialize)]
pub struct BulkSaveRequest {
    pub documents: Vec<SaveDocumentRequest>,
    /// When true, every URL in `documents` is also inserted into
    /// `favorite_documents` so it shows up in the caller's starred
    /// list. The bookmark dialog sets this so saving and starring
    /// happen atomically — previously they were two separate
    /// requests and the star call was fire-and-forget, leaving
    /// some bookmarked URLs unstarred when the second call failed.
    #[serde(default)]
    pub favorite: bool,
}

#[derive(Serialize)]
pub struct BulkSaveResponse {
    pub received: usize,
    pub inserted: usize,
}

/// POST /auth/me/documents/bulk
///
/// Insert many documents for the signed-in user in a single round-trip.
/// Used by the client-side "Sync" action that ports each Python fetcher
/// to JS: the browser walks every configured source, builds doc rows
/// (url + title + summary + date + source + source_url, tags left empty
/// for the backend pipeline to fill), and POSTs them here. UNNEST gives
/// us one INSERT per request regardless of payload size; `ON CONFLICT
/// DO NOTHING` keeps re-sync idempotent so the user can mash the button
/// without growing duplicates.
pub async fn bulk_save_documents(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<BulkSaveRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let received = req.documents.len();
    if received == 0 {
        return Json(BulkSaveResponse {
            received: 0,
            inserted: 0,
        })
        .into_response();
    }

    // Hard cap per request so a runaway client can't OOM the server.
    // The JS orchestrator should chunk beyond this.
    const MAX_PER_CALL: usize = 5000;
    let docs = if received > MAX_PER_CALL {
        &req.documents[..MAX_PER_CALL]
    } else {
        &req.documents[..]
    };

    let mut urls = Vec::with_capacity(docs.len());
    let mut titles = Vec::with_capacity(docs.len());
    let mut summaries = Vec::with_capacity(docs.len());
    let mut dates = Vec::with_capacity(docs.len());
    let mut sources = Vec::with_capacity(docs.len());
    let mut source_urls: Vec<Option<String>> = Vec::with_capacity(docs.len());
    let mut publics: Vec<bool> = Vec::with_capacity(docs.len());
    // Comma-joined per-doc tags. PG decodes each cell back to text[]
    // via `string_to_array` in the SELECT below. We avoid passing a
    // 2-D array (text[][]) because every row would need the same
    // length, which doesn't fit a "tags vary per doc" payload.
    let mut tag_csvs: Vec<String> = Vec::with_capacity(docs.len());
    // Linked-URL previews. `linked_urls` is one JSON string per row
    // (PG casts back to JSONB in the INSERT); `link_hosts` is the
    // flat hosts list, encoded the same way as tags. We re-derive
    // `link_hosts` from the JSON when the caller didn't ship it so
    // the GIN-indexed column never diverges from the JSON payload.
    let mut linked_urls_json: Vec<String> = Vec::with_capacity(docs.len());
    let mut link_hosts_csv: Vec<String> = Vec::with_capacity(docs.len());

    for d in docs {
        let url = d.url.trim();
        if url.is_empty() {
            continue;
        }
        urls.push(url.to_string());
        titles.push(d.title.clone().unwrap_or_default());
        summaries.push(d.summary.clone().unwrap_or_default());
        dates.push(d.date.clone().unwrap_or_default());
        sources.push(d.source.clone().unwrap_or_default());
        source_urls.push(d.source_url.clone().filter(|s| !s.is_empty()));
        publics.push(d.public.unwrap_or(true));
        // Comma encoding — strip empties, normalise whitespace, dedupe
        // case-insensitively so the row's tags column stays clean.
        let mut seen = std::collections::HashSet::new();
        let mut cleaned: Vec<String> = Vec::new();
        for t in d.tags.clone().unwrap_or_default() {
            let t = t.trim();
            if t.is_empty() {
                continue;
            }
            let key = t.to_lowercase();
            if seen.insert(key) {
                cleaned.push(t.to_string());
            }
        }
        tag_csvs.push(cleaned.join(","));

        // Normalise the linked-URLs JSON to a canonical array string
        // and harvest its hosts. We accept either an array of objects
        // (the pipeline's shape) or an empty/missing value (sync from
        // older clients). Anything that isn't a JSON array gets
        // dropped on the floor rather than corrupting the column.
        let normalised_links: serde_json::Value = match d.linked_urls.clone() {
            Some(serde_json::Value::Array(arr)) => serde_json::Value::Array(arr),
            _ => serde_json::Value::Array(Vec::new()),
        };
        // Re-derive hosts from the JSON unless the caller passed an
        // explicit list — the JSON is the truth source.
        let hosts_from_json: Vec<String> = if let serde_json::Value::Array(arr) = &normalised_links
        {
            arr.iter()
                .filter_map(|v| v.get("host").and_then(|h| h.as_str()))
                .filter(|h| !h.is_empty())
                .map(|h| h.to_string())
                .collect()
        } else {
            Vec::new()
        };
        let hosts: Vec<String> = match d.link_hosts.clone() {
            Some(v) if !v.is_empty() => v,
            _ => hosts_from_json,
        };
        linked_urls_json.push(normalised_links.to_string());
        link_hosts_csv.push(hosts.join(","));
    }

    if urls.is_empty() {
        return Json(BulkSaveResponse {
            received,
            inserted: 0,
        })
        .into_response();
    }

    // Tags arrive comma-joined per row (see encode loop above); we
    // explode them back to text[] via string_to_array. Empty CSV
    // resolves to an empty array. extra_tags stays empty — the
    // tagger pipeline fills that column.
    // ON CONFLICT semantics:
    //   * date:    GREATEST(old, new) — re-posting today bumps an old
    //              entry to today so it surfaces at the top of the
    //              feed; pipeline imports of an older publication date
    //              never go backwards.
    //   * title /
    //     summary: COALESCE on non-empty EXCLUDED → keep existing
    //              content if the new payload didn't ship a value.
    //   * tags:    only overwrite when the new tag array is non-empty,
    //              so pipeline syncs don't blow away user-curated tags.
    //   * deleted: clear — re-posting a soft-deleted URL resurrects it,
    //              same way the merge branch of update_document does.
    let sql = "
        INSERT INTO documents (
            user_id, url, title, summary, date, tags, extra_tags,
            source, source_url, public, linked_urls, link_hosts
        )
        SELECT $1, u.url, u.title, u.summary,
               NULLIF(u.date, '')::date,
               CASE WHEN u.tags = '' THEN '{}'::text[]
                    ELSE string_to_array(u.tags, ',') END,
               '{}'::text[],
               u.source, u.source_url, u.public,
               COALESCE(u.linked_urls::jsonb, '[]'::jsonb),
               CASE WHEN u.link_hosts = '' THEN '{}'::text[]
                    ELSE string_to_array(u.link_hosts, ',') END
          FROM UNNEST($2::text[], $3::text[], $4::text[], $5::text[],
                      $6::text[], $7::text[], $8::bool[], $9::text[],
                      $10::text[], $11::text[])
               AS u(url, title, summary, date, source, source_url,
                    public, tags, linked_urls, link_hosts)
         ON CONFLICT (user_id, url) DO UPDATE
            SET date    = GREATEST(documents.date,
                                   EXCLUDED.date),
                title   = CASE WHEN EXCLUDED.title <> ''
                                THEN EXCLUDED.title
                                ELSE documents.title END,
                summary = CASE WHEN EXCLUDED.summary <> ''
                                THEN EXCLUDED.summary
                                ELSE documents.summary END,
                tags    = CASE WHEN cardinality(EXCLUDED.tags) > 0
                                THEN EXCLUDED.tags
                                ELSE documents.tags END,
                -- Replace `linked_urls` and `link_hosts` only when
                -- the caller sent a non-empty payload — sync-style
                -- inserts from older clients omit the column and
                -- shouldn't clobber a richer existing value.
                linked_urls = CASE
                    WHEN jsonb_array_length(EXCLUDED.linked_urls) > 0
                        THEN EXCLUDED.linked_urls
                    ELSE documents.linked_urls
                END,
                link_hosts = CASE
                    WHEN cardinality(EXCLUDED.link_hosts) > 0
                        THEN EXCLUDED.link_hosts
                    ELSE documents.link_hosts
                END,
                -- A real sync just confirmed this doc — promote it
                -- away from the favorite-only lifecycle so a later
                -- un-upvote no longer deletes the row.
                created_via_favorite = FALSE,
                deleted = FALSE,
                updated_at = now()
    ";

    // One transaction so the optional favorite-star can't land
    // without the document, or vice-versa.
    let mut tx = match pool.begin().await {
        Ok(t) => t,
        Err(e) => {
            tracing::error!(error = %e, "bulk_save.begin.failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "begin failed").into_response();
        }
    };

    let insert_result = sqlx::query(sql)
        .bind(me.id)
        .bind(&urls)
        .bind(&titles)
        .bind(&summaries)
        .bind(&dates)
        .bind(&sources)
        .bind(&source_urls)
        .bind(&publics)
        .bind(&tag_csvs)
        .bind(&linked_urls_json)
        .bind(&link_hosts_csv)
        .execute(&mut *tx)
        .await;

    let inserted = match insert_result {
        Ok(r) => r.rows_affected() as usize,
        Err(e) => {
            tracing::error!(error = %e, "bulk_save.failed");
            let _ = tx.rollback().await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("bulk save failed: {e}"),
            )
                .into_response();
        }
    };

    // Bookmarks flip `favorite=true` so the row that was just
    // inserted (or upserted) also lands in favorite_documents
    // without a second round-trip. ON CONFLICT keeps the call
    // idempotent — re-bookmarking an already-starred URL is a
    // no-op, not an error.
    if req.favorite && !urls.is_empty() {
        if let Err(e) = sqlx::query(
            "INSERT INTO favorite_documents (user_id, url)
             SELECT $1, u FROM UNNEST($2::text[]) AS u
             ON CONFLICT (user_id, url) DO NOTHING",
        )
        .bind(me.id)
        .bind(&urls)
        .execute(&mut *tx)
        .await
        {
            tracing::error!(error = %e, "bulk_save.favorite.failed");
            let _ = tx.rollback().await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("favorite failed: {e}"),
            )
                .into_response();
        }
    }

    if let Err(e) = tx.commit().await {
        tracing::error!(error = %e, "bulk_save.commit.failed");
        return (StatusCode::INTERNAL_SERVER_ERROR, "commit failed").into_response();
    }

    Json(BulkSaveResponse { received, inserted }).into_response()
}

#[derive(Deserialize)]
pub struct UpdateDocumentRequest {
    pub url: String,
    pub title: Option<String>,
    pub summary: Option<String>,
    /// Replace the row's full tags array. Omit to leave tags
    /// untouched; pass `[]` to clear.
    pub tags: Option<Vec<String>>,
    /// Optional new canonical URL — used when the editor detects a
    /// real URL in the body of a previously text-only post. Triggers
    /// an `UPDATE … SET url = $new_url` keyed on (user_id, old_url).
    pub new_url: Option<String>,
    /// Optional source key (e.g. "github", "mixedbread.com") — derived
    /// by the frontend via hostnameSourceKey. NULL = leave alone.
    pub source: Option<String>,
}

/// PATCH /auth/me/documents
///
/// Update editable fields (title, summary, tags) on a doc the caller
/// owns. Scoped strictly to (user_id, url) so a user can never
/// overwrite somebody else's row. Fields are optional — only the ones
/// in the payload are touched.
pub async fn update_document(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<UpdateDocumentRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let url = req.url.trim().to_string();
    if url.is_empty() {
        return (StatusCode::BAD_REQUEST, "url required").into_response();
    }
    if req.title.is_none()
        && req.summary.is_none()
        && req.tags.is_none()
        && req.new_url.is_none()
        && req.source.is_none()
    {
        return (StatusCode::BAD_REQUEST, "nothing to update").into_response();
    }
    // Normalise tags: trim, drop empties, case-dedupe.
    let normalised_tags: Option<Vec<String>> = req.tags.as_ref().map(|raw| {
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<String> = Vec::new();
        for t in raw {
            let t = t.trim();
            if t.is_empty() {
                continue;
            }
            if seen.insert(t.to_lowercase()) {
                out.push(t.to_string());
            }
        }
        out
    });
    let new_url_trimmed = req
        .new_url
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != &url);

    // Optional URL change: the row's primary key gets rewritten if
    // the editor spotted a real URL in the body. Two cases:
    //   * Target URL doesn't yet exist for this user → simple UPDATE.
    //   * Target URL already exists (e.g. user previously bookmarked
    //     the same link) → MERGE: apply the new title/summary/tags/
    //     source to the existing target row, soft-delete the original
    //     note in the same transaction. That collapses duplicates
    //     instead of failing on the composite PK.
    if let Some(target) = new_url_trimmed.as_deref() {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (
                SELECT 1 FROM documents WHERE user_id = $1 AND url = $2
             )",
        )
        .bind(me.id)
        .bind(target)
        .fetch_one(&pool)
        .await
        .unwrap_or(false);
        if exists {
            let mut tx = match pool.begin().await {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(error = %e, "documents.merge.begin_tx_failed");
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
            };
            // Apply edits to the existing row and clear any prior soft
            // delete so the merge surfaces immediately.
            if let Err(e) = sqlx::query(
                "UPDATE documents
                    SET title   = COALESCE($3, title),
                        summary = COALESCE($4, summary),
                        tags    = CASE WHEN $5::text[] IS NULL THEN tags
                                       ELSE $5::text[] END,
                        source  = COALESCE($6, source),
                        deleted = FALSE,
                        updated_at = now()
                  WHERE user_id = $1 AND url = $2",
            )
            .bind(me.id)
            .bind(target)
            .bind(req.title.as_deref())
            .bind(req.summary.as_deref())
            .bind(normalised_tags.as_deref())
            .bind(req.source.as_deref())
            .execute(&mut *tx)
            .await
            {
                tracing::error!(error = %e, "documents.merge.update_target_failed");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            // Drop the original note — its content has migrated.
            if let Err(e) = sqlx::query("DELETE FROM documents WHERE user_id = $1 AND url = $2")
                .bind(me.id)
                .bind(&url)
                .execute(&mut *tx)
                .await
            {
                tracing::error!(error = %e, "documents.merge.delete_original_failed");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            if let Err(e) = tx.commit().await {
                tracing::error!(error = %e, "documents.merge.commit_failed");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            return StatusCode::NO_CONTENT.into_response();
        }
    }

    // No URL change OR no collision — straight UPDATE.
    let result = sqlx::query(
        "UPDATE documents
            SET title   = COALESCE($3, title),
                summary = COALESCE($4, summary),
                tags    = CASE WHEN $5::text[] IS NULL THEN tags
                               ELSE $5::text[] END,
                url     = COALESCE($6, url),
                source  = COALESCE($7, source),
                updated_at = now()
          WHERE user_id = $1 AND url = $2",
    )
    .bind(me.id)
    .bind(&url)
    .bind(req.title.as_deref())
    .bind(req.summary.as_deref())
    .bind(normalised_tags.as_deref())
    .bind(new_url_trimmed.as_deref())
    .bind(req.source.as_deref())
    .execute(&pool)
    .await;
    match result {
        Ok(r) if r.rows_affected() == 0 => StatusCode::NOT_FOUND.into_response(),
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            tracing::error!(error = %e, "documents.update.failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[derive(Deserialize)]
pub struct DeleteDocumentQuery {
    pub url: String,
}

/// GET /auth/me/deleted-urls — return the set of URLs the caller has
/// soft-deleted. The frontend uses this to filter ColBERT search hits
/// (the index can still surface evicted docs until a re-index lands).
pub async fn list_deleted_urls(State(pool): State<PgPool>, jar: CookieJar) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return Json(Vec::<String>::new()).into_response();
    };
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT url FROM documents WHERE user_id = $1 AND deleted = TRUE")
            .bind(me.id)
            .fetch_all(&pool)
            .await
            .unwrap_or_default();
    let urls: Vec<String> = rows.into_iter().map(|(u,)| u).collect();
    Json(urls).into_response()
}

pub async fn delete_document(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Query(q): Query<DeleteDocumentQuery>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let url = q.url.trim().to_string();
    if url.is_empty() {
        return (StatusCode::BAD_REQUEST, "url required").into_response();
    }
    // Soft delete: keep the row but flip `deleted = TRUE`. Read paths
    // (timeline, /api/users/{slug}/{documents,sources}, favorites,
    // etc.) exclude `deleted = TRUE`, and the pipeline's
    // `ON CONFLICT DO NOTHING` bulk insert preserves the flag — so
    // re-running `make run` never resurrects a deleted doc. Scoped
    // to (user_id, url) so a user can only soft-delete their own rows.
    // Also remove from favorite_documents — the user's intent when
    // deleting a doc is for it to be gone everywhere, including the
    // Favorites chip.
    let _ = sqlx::query(
        "UPDATE documents
            SET deleted = TRUE, updated_at = now()
          WHERE user_id = $1 AND url = $2",
    )
    .bind(me.id)
    .bind(&url)
    .execute(&pool)
    .await;
    let _ = sqlx::query("DELETE FROM favorite_documents WHERE user_id = $1 AND url = $2")
        .bind(me.id)
        .bind(&url)
        .execute(&pool)
        .await;
    StatusCode::NO_CONTENT.into_response()
}

// ── /auth/me/sync/* — live-tracker endpoints for the JS orchestrator ──

#[derive(Serialize)]
pub struct SyncStartResponse {
    pub run_id: i64,
}

/// POST /auth/me/sync/start
///
/// Inserts a `running` row in `pipeline_runs` with trigger='js-sync'
/// and returns its id. The browser orchestrator calls this once, then
/// hands the id to `/auth/me/sync/end` when the sync finishes (or
/// fails). Kept deliberately lightweight — no payload.
pub async fn sync_start(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let row: Result<(i64,), _> = sqlx::query_as(
        "INSERT INTO pipeline_runs (user_id, trigger, status)
         VALUES ($1, 'js-sync', 'running')
         RETURNING id",
    )
    .bind(me.id)
    .fetch_one(&pool)
    .await;
    match row {
        Ok((id,)) => Json(SyncStartResponse { run_id: id }).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "sync_start.failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("sync_start failed: {e}"),
            )
                .into_response()
        }
    }
}

#[derive(Deserialize)]
pub struct SyncEndRequest {
    pub run_id: i64,
    pub success: bool,
    /// Total docs the JS extraction produced before dedup (matches the
    /// Python pipeline's `total_documents` roughly).
    pub total_documents: Option<i64>,
    /// Docs actually inserted into `documents` (idempotent ON CONFLICT
    /// DO NOTHING means this can be lower than `total_documents`).
    pub new_documents: Option<i64>,
    pub duration_secs: Option<f64>,
    /// Per-source {key, fetched, error?} list; stored verbatim.
    pub timings: Option<serde_json::Value>,
    /// Free-form error string when `success = false`.
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct SyncStatusResponse {
    pub run_id: Option<i64>,
    pub trigger: Option<String>,
    pub status: Option<String>,
    pub stage: Option<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub duration_secs: Option<f32>,
    pub new_documents: Option<i32>,
    pub total_documents: Option<i32>,
    pub error: Option<String>,
}

/// GET /auth/me/sync/status
///
/// Latest pipeline_runs row for the signed-in user — running or
/// completed. Frontend polls this every ~2s while the profile
/// modal is open to power a progress bar. Returns empty JSON
/// (`{run_id: null, …}`) when the user has never had a run.
pub async fn sync_status(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    // Prefer a currently-running row; otherwise fall back to the most
    // recent completed one. sqlx here is built without chrono, so we
    // coerce the timestamps to ISO-8601 strings on the PG side via
    // `to_char`, matching what `list_documents` does for `date`.
    #[allow(clippy::type_complexity)]
    let row: Option<(
        i64,
        String,
        String,
        Option<String>,
        String,
        Option<String>,
        Option<f32>,
        i32,
        i32,
        Option<String>,
    )> = sqlx::query_as(
        "SELECT id, trigger, status, stage,
                to_char(started_at  AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'),
                to_char(finished_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'),
                duration_secs, new_documents, total_documents, error
           FROM pipeline_runs
          WHERE user_id = $1
          ORDER BY (status = 'running') DESC, started_at DESC
          LIMIT 1",
    )
    .bind(me.id)
    .fetch_optional(&pool)
    .await
    .unwrap_or(None);

    let resp = match row {
        Some((id, trig, st, stage, started, finished, dur, new_d, total_d, err)) => {
            SyncStatusResponse {
                run_id: Some(id),
                trigger: Some(trig),
                status: Some(st),
                stage,
                started_at: Some(started),
                finished_at: finished,
                duration_secs: dur,
                new_documents: Some(new_d),
                total_documents: Some(total_d),
                error: err,
            }
        }
        None => SyncStatusResponse {
            run_id: None,
            trigger: None,
            status: None,
            stage: None,
            started_at: None,
            finished_at: None,
            duration_secs: None,
            new_documents: None,
            total_documents: None,
            error: None,
        },
    };
    Json(resp).into_response()
}

/// POST /auth/me/sync/end
///
/// Seals a `running` row (must belong to the caller). The id has to
/// match one of their own runs — otherwise a client could close or
/// tamper with another user's pending runs.
pub async fn sync_end(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<SyncEndRequest>,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let status = if req.success { "success" } else { "failed" };
    let timings = req
        .timings
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
    let result = sqlx::query(
        "UPDATE pipeline_runs SET
             status          = $1,
             stage           = NULL,
             finished_at     = now(),
             duration_secs   = $2,
             new_documents   = COALESCE($3, new_documents),
             total_documents = COALESCE($4, total_documents),
             timings         = $5::jsonb,
             error           = $6
         WHERE id = $7 AND user_id = $8",
    )
    .bind(status)
    .bind(req.duration_secs.map(|d| d as f32))
    .bind(req.new_documents)
    .bind(req.total_documents)
    .bind(timings)
    .bind(req.error.as_deref())
    .bind(req.run_id)
    .bind(me.id)
    .execute(&pool)
    .await;
    match result {
        Ok(r) if r.rows_affected() == 0 => {
            (StatusCode::NOT_FOUND, "run_id not found for this user").into_response()
        }
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => {
            tracing::error!(error = %e, "sync_end.failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("sync_end failed: {e}"),
            )
                .into_response()
        }
    }
}

/// GET /auth/me/documents/urls — just the URLs in the signed-in user's
/// library. Used by the search page to mark results as already saved
/// without pulling the full document metadata.
pub async fn list_document_urls(State(pool): State<PgPool>, jar: CookieJar) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let rows: Vec<(String,)> = sqlx::query_as("SELECT url FROM documents WHERE user_id = $1")
        .bind(me.id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();
    let urls: Vec<String> = rows.into_iter().map(|r| r.0).collect();
    Json(urls).into_response()
}

pub async fn clear_stackoverflow_auth(
    State(pool): State<PgPool>,
    jar: CookieJar,
) -> impl IntoResponse {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "UPDATE users
            SET sources = jsonb_set(
                    sources,
                    '{stackoverflow}',
                    COALESCE(sources->'stackoverflow', '{}'::jsonb) - 'access_token_enc',
                    true
                ),
                updated_at = now()
          WHERE id = $1
            AND sources ? 'stackoverflow'",
    )
    .bind(me.id)
    .execute(&pool)
    .await;
    me_after_update(&pool, me.id).await
}

async fn me_after_update(pool: &PgPool, id: i64) -> Response {
    let row: Option<MeResponse> = sqlx::query_as(
        "SELECT u.id,
                u.username      AS slug,
                u.name,
                u.email,
                u.avatar,
                u.description,
                COALESCE(
                    (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
                       FROM user_categories uc
                       JOIN categories      cat ON cat.id = uc.category_id
                      WHERE uc.user_id = u.id),
                    '{}'::text[]
                ) AS categories,
                u.index_name,
                u.public,
                u.links,
                u.sources,
                u.email_verified,
                u.twitter_followers,
                u.github_followers,
                u.citations,
                (COALESCE(u.sources->'hackernews'->>'username', '') <> '')
                    AS has_hackernews,
                (COALESCE(u.sources->'hackernews'->>'username', '') <> ''
                 AND COALESCE(u.sources->'hackernews'->>'password_enc', '') <> '')
                    AS has_hackernews_upvotes,
                NULLIF(u.sources->'hackernews'->>'username', '')
                    AS hackernews_username,
                (COALESCE(u.sources->'zotero'->>'api_key_enc', '') <> ''
                 AND COALESCE(u.sources->'zotero'->>'user_id', '') <> '')
                    AS has_zotero,
                NULLIF(u.sources->'zotero'->>'user_id', '')::bigint
                    AS zotero_user_id,
                COALESCE(u.sources->'zotero'->'groups', '[]'::jsonb)
                    AS zotero_groups,
                NULLIF(u.sources->'zotero'->>'personal_count', '')::bigint
                    AS zotero_personal_count,
                (COALESCE(u.sources->'twitter'->>'cookies_enc', '') <> '')
                    AS has_twitter_cookies,
                (COALESCE(u.sources->'stackoverflow'->>'access_token_enc', '') <> '')
                    AS has_stackoverflow_auth,
                COALESCE(u.sources->'stackoverflow'->'associated_sites', '[]'::jsonb)
                    AS stackoverflow_sites
           FROM users u
          WHERE u.id = $1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    match row {
        Some(u) => Json(u).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

// `urlencoding` isn't a dep yet but is trivially inlined.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        const HEX: &[u8] = b"0123456789ABCDEF";
        let mut out = String::with_capacity(s.len());
        for &b in s.as_bytes() {
            let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
            if unreserved {
                out.push(b as char);
            } else {
                out.push('%');
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0x0f) as usize] as char);
            }
        }
        out
    }
}

// ── Personality bookmarks (cross-user "follow") ─────────────────────────
//
// One row per (owner, target) pair in `personality_bookmarks`. Surfaced
// in the search-page library picker as a dedicated "Bookmarks" section
// above the by-category list, so a user's saved people are one click
// away from being added as an active library.

/// GET /auth/me/personality-bookmarks
///
/// Returns the slugs the signed-in user has bookmarked, ordered by
/// most-recent first. 401 when anonymous.
pub async fn list_personality_bookmarks(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT u.username
           FROM personality_bookmarks pb
           JOIN users u ON u.id = pb.bookmarked_user_id
          WHERE pb.user_id = $1
          ORDER BY pb.created_at DESC",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let slugs: Vec<String> = rows.into_iter().map(|(s,)| s).collect();
    Json(slugs).into_response()
}

/// PUT /auth/me/personality-bookmarks/:slug
///
/// Adds a bookmark. Idempotent (ON CONFLICT DO NOTHING).
pub async fn add_personality_bookmark(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Path(slug): Path<String>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let target: Option<(i64,)> = sqlx::query_as("SELECT id FROM users WHERE username = $1")
        .bind(&slug)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);
    let Some((target_id,)) = target else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if target_id == me.id {
        // Self-bookmarking is meaningless — also blocked by the
        // table CHECK constraint, but reject early with a clearer
        // status code.
        return StatusCode::BAD_REQUEST.into_response();
    }
    let _ = sqlx::query(
        "INSERT INTO personality_bookmarks (user_id, bookmarked_user_id)
         VALUES ($1, $2) ON CONFLICT DO NOTHING",
    )
    .bind(me.id)
    .bind(target_id)
    .execute(&pool)
    .await;
    StatusCode::NO_CONTENT.into_response()
}

/// DELETE /auth/me/personality-bookmarks/:slug
///
/// Removes a bookmark. Idempotent.
pub async fn remove_personality_bookmark(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Path(slug): Path<String>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "DELETE FROM personality_bookmarks
          WHERE user_id = $1
            AND bookmarked_user_id = (SELECT id FROM users WHERE username = $2)",
    )
    .bind(me.id)
    .bind(&slug)
    .execute(&pool)
    .await;
    StatusCode::NO_CONTENT.into_response()
}
