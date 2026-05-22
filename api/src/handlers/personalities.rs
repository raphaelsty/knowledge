//! Suggest a public personality for review.
//!
//! `POST /api/personalities` no longer creates `users` rows directly.
//! Anyone could provision a VIP personality (and the pipeline compute
//! that follows) by filling a form, which isn't the moderation model
//! the project wants. Submissions now land in
//! `personality_submissions` with `status='pending'`; the project
//! owner reviews the queue out-of-band and promotes approved rows to
//! the real `users` table by hand.
//!
//! The endpoint path stays the same so the existing JS client keeps
//! working; only the response semantics change (the response carries
//! `status: "submitted"` and no `balance` field because no debit
//! happens).

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};

use crate::handlers::auth::current_user;
use crate::state::AppState;

/// Entry fee to add a public personality, in cents. Free for now —
/// the sponsor still picks up ongoing Twitter + storage via the
/// Python pipeline, but creation itself doesn't debit anything.
const ADD_PERSONALITY_COST: i32 = 0;

#[derive(Deserialize)]
pub struct AddPersonalityRequest {
    pub name: String,
    /// Lowercase-hyphen handle the library lives at, e.g.
    /// "aravind-srinivas". Must be unique. We re-derive from name
    /// when omitted.
    #[serde(default)]
    pub slug: String,
    #[serde(default)]
    pub description: String,
    // Optional source handles. Stored verbatim in `users.sources`.
    // Field names mirror the keys the Python pipeline reads.
    #[serde(default, rename = "twitterHandle")]
    pub twitter_handle: String,
    #[serde(default, rename = "githubHandle")]
    pub github_handle: String,
    #[serde(default, rename = "huggingfaceHandle")]
    pub huggingface_handle: String,
    #[serde(default, rename = "redditHandle")]
    pub reddit_handle: String,
    /// HackerNews username (lowercase, no @).
    #[serde(default, rename = "hackernewsHandle")]
    pub hackernews_handle: String,
    /// Stack Overflow numeric user id.
    #[serde(default, rename = "stackoverflowUserId")]
    pub stackoverflow_user_id: String,
    /// Author name to search for on arXiv.
    #[serde(default, rename = "arxivAuthor")]
    pub arxiv_author: String,
    /// DBLP author name (CS papers).
    #[serde(default, rename = "dblpAuthor")]
    pub dblp_author: String,
    /// Google Scholar user id (the `user=` param in the profile URL).
    #[serde(default, rename = "scholarUserId")]
    pub scholar_user_id: String,
    /// One URL per line. Accept it as a single string so the
    /// frontend can reuse the same textarea component as the
    /// user-config "Websites" panel; we split + dedupe here.
    #[serde(default)]
    pub websites: String,
}

#[derive(Serialize)]
pub struct AddPersonalityResponse {
    /// Canonicalised slug we'd assign on review. The admin can rename
    /// during promotion; this is only an early read-back so the
    /// client can show "we've stored your suggestion for @<slug>".
    pub slug: String,
    /// Caller's current balance. Always returned (unchanged from the
    /// pre-submission value) so the frontend's credit-pill stays in
    /// sync without a separate round-trip.
    pub balance: i32,
    /// Always `"submitted"` — no pipeline spawn, no user row, no
    /// debit. Kept as the `status` field so existing JS clients can
    /// switch on it without a schema change.
    pub status: &'static str,
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_dash = true;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            for lo in c.to_lowercase() {
                out.push(lo);
            }
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// Strip a leading `@` and surrounding whitespace from a handle.
fn h(s: &str) -> String {
    s.trim().trim_start_matches('@').to_string()
}

/// Look for an existing personality that would conflict with the
/// caller's inputs. Returns `Some((slug, name, field))` on the first
/// hit so the handler can surface a single, specific message.
///
/// `exclude_id` lets `update()` ignore the row being edited (a
/// personality can keep its own slug / name / Twitter handle).
async fn find_personality_conflict(
    pool: &sqlx::PgPool,
    slug: &str,
    name: &str,
    twitter_handle_lower: &str,
    exclude_id: Option<i64>,
) -> Option<(String, String, String)> {
    let excl = exclude_id.unwrap_or(-1);

    // 1. Slug. Cheapest check — has a unique index on `username`.
    if let Ok(Some((s, n))) = sqlx::query_as::<_, (String, String)>(
        "SELECT username, COALESCE(name, username) FROM users
          WHERE lower(username) = lower($1) AND id <> $2 LIMIT 1",
    )
    .bind(slug)
    .bind(excl)
    .fetch_optional(pool)
    .await
    {
        return Some((s, n, "slug".into()));
    }

    // 2. Display name. Case-insensitive — "Yann LeCun" and "yann
    // lecun" are the same person on this platform.
    if !name.trim().is_empty() {
        if let Ok(Some((s, n))) = sqlx::query_as::<_, (String, String)>(
            "SELECT username, COALESCE(name, username) FROM users
              WHERE lower(name) = lower($1) AND id <> $2 LIMIT 1",
        )
        .bind(name)
        .bind(excl)
        .fetch_optional(pool)
        .await
        {
            return Some((s, n, "name".into()));
        }
    }

    // 3. Twitter handle. Stored at `sources.twitter.username`. Match
    // case-insensitively since Twitter handles aren't case-sensitive.
    if !twitter_handle_lower.is_empty() {
        if let Ok(Some((s, n))) = sqlx::query_as::<_, (String, String)>(
            "SELECT username, COALESCE(name, username) FROM users
              WHERE lower(sources->'twitter'->>'username') = $1
                AND id <> $2 LIMIT 1",
        )
        .bind(twitter_handle_lower)
        .bind(excl)
        .fetch_optional(pool)
        .await
        {
            return Some((s, n, "twitter".into()));
        }
    }

    None
}

/// Human-readable error string for the duplicate-personality 409.
/// The frontend also has the structured `existingSlug` / `existingName`
/// fields so it can render a clickable link to the existing profile.
fn conflict_message(field: &str, existing_name: &str, existing_slug: &str) -> String {
    match field {
        "slug" => format!("@{existing_slug} is already on Knowledge"),
        "name" => format!(
            "{existing_name} is already on Knowledge as @{existing_slug}"
        ),
        "twitter" => format!(
            "That Twitter handle belongs to {existing_name} (@{existing_slug}), already on Knowledge"
        ),
        _ => format!("@{existing_slug} is already on Knowledge"),
    }
}

/// Build a JSONB `sources` blob for the new user row. Keys mirror
/// the Python pipeline's source-config schema (`sources/utils/*`).
fn build_sources(req: &AddPersonalityRequest) -> serde_json::Value {
    let mut sources = serde_json::Map::new();
    if !req.twitter_handle.trim().is_empty() {
        sources.insert(
            "twitter".into(),
            serde_json::json!({ "username": h(&req.twitter_handle) }),
        );
    }
    if !req.github_handle.trim().is_empty() {
        sources.insert(
            "github".into(),
            serde_json::json!({ "username": h(&req.github_handle) }),
        );
    }
    if !req.huggingface_handle.trim().is_empty() {
        sources.insert(
            "huggingface".into(),
            serde_json::json!({ "username": h(&req.huggingface_handle) }),
        );
    }
    if !req.reddit_handle.trim().is_empty() {
        sources.insert(
            "reddit".into(),
            serde_json::json!({ "username": h(&req.reddit_handle) }),
        );
    }
    if !req.hackernews_handle.trim().is_empty() {
        sources.insert(
            "hackernews".into(),
            serde_json::json!({ "username": h(&req.hackernews_handle) }),
        );
    }
    if !req.stackoverflow_user_id.trim().is_empty() {
        sources.insert(
            "stackoverflow".into(),
            serde_json::json!({ "user_id": req.stackoverflow_user_id.trim() }),
        );
    }
    if !req.arxiv_author.trim().is_empty() {
        sources.insert(
            "arxiv".into(),
            serde_json::json!({ "author": req.arxiv_author.trim() }),
        );
    }
    if !req.dblp_author.trim().is_empty() {
        sources.insert(
            "dblp".into(),
            serde_json::json!({ "author": req.dblp_author.trim() }),
        );
    }
    if !req.scholar_user_id.trim().is_empty() {
        sources.insert(
            "scholar".into(),
            serde_json::json!({ "user_id": req.scholar_user_id.trim() }),
        );
    }
    let urls = split_websites(&req.websites);
    if !urls.is_empty() {
        sources.insert("websites".into(), serde_json::json!({ "urls": urls }));
    }
    serde_json::Value::Object(sources)
}

/// Parse the websites textarea into a deduped list of trimmed URLs.
/// Accepts newline- or comma-separated entries; trims whitespace;
/// drops obvious garbage (empty lines, comment lines). The Python
/// pipeline's `websites` resolver handles feed/sitemap detection at
/// fetch time, so we don't need to validate the URL shape here.
fn split_websites(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in s.split(['\n', ',']) {
        let u = line.trim();
        if u.is_empty() || u.starts_with('#') {
            continue;
        }
        if seen.insert(u.to_string()) {
            out.push(u.to_string());
        }
    }
    out
}

fn build_links(req: &AddPersonalityRequest) -> serde_json::Value {
    let mut links = serde_json::Map::new();
    if !req.twitter_handle.trim().is_empty() {
        links.insert(
            "twitter".into(),
            serde_json::json!(format!("https://x.com/{}", h(&req.twitter_handle))),
        );
    }
    if !req.github_handle.trim().is_empty() {
        links.insert(
            "github".into(),
            serde_json::json!(format!("https://github.com/{}", h(&req.github_handle))),
        );
    }
    if !req.huggingface_handle.trim().is_empty() {
        links.insert(
            "huggingface".into(),
            serde_json::json!(format!(
                "https://huggingface.co/{}",
                h(&req.huggingface_handle)
            )),
        );
    }
    if !req.reddit_handle.trim().is_empty() {
        links.insert(
            "reddit".into(),
            serde_json::json!(format!(
                "https://www.reddit.com/user/{}",
                h(&req.reddit_handle)
            )),
        );
    }
    if !req.hackernews_handle.trim().is_empty() {
        links.insert(
            "hackernews".into(),
            serde_json::json!(format!(
                "https://news.ycombinator.com/user?id={}",
                h(&req.hackernews_handle)
            )),
        );
    }
    if !req.stackoverflow_user_id.trim().is_empty() {
        links.insert(
            "stackoverflow".into(),
            serde_json::json!(format!(
                "https://stackoverflow.com/users/{}",
                req.stackoverflow_user_id.trim()
            )),
        );
    }
    if !req.scholar_user_id.trim().is_empty() {
        links.insert(
            "scholar".into(),
            serde_json::json!(format!(
                "https://scholar.google.com/citations?user={}",
                req.scholar_user_id.trim()
            )),
        );
    }
    // First URL becomes the canonical `links.website` so existing UI
    // surfaces that read a single string still work. The full list
    // is in `sources.websites.urls`.
    let urls = split_websites(&req.websites);
    if let Some(first) = urls.first() {
        links.insert("website".into(), serde_json::json!(first.clone()));
    }
    serde_json::Value::Object(links)
}

#[derive(Serialize, sqlx::FromRow)]
pub struct SponsoredRow {
    pub slug: String,
    pub name: String,
    pub description: String,
    /// ISO-8601 UTC. NULL only on historical rows we couldn't
    /// backfill — the SQL `to_char` returns the empty string in
    /// that case to keep the JSON shape stable.
    #[serde(rename = "sponsoredAt")]
    pub sponsored_at: String,
    /// Live document count for the personality. Computed at read
    /// time so we never serve a stale figure.
    #[serde(rename = "docCount")]
    pub doc_count: i64,
    /// Total in cents the sponsor has spent on this personality
    /// to date, broken down by kind so the UI can display a
    /// helpful tooltip / drill-down.
    #[serde(rename = "costCents")]
    pub cost_cents: i64,
    #[serde(rename = "costEntryCents")]
    pub cost_entry_cents: i64,
    #[serde(rename = "costTwitterCents")]
    pub cost_twitter_cents: i64,
    #[serde(rename = "costStorageCents")]
    pub cost_storage_cents: i64,
    /// Source-handle JSON we stamped on the row at creation. Pulled
    /// here so the frontend can pre-fill the edit form without an
    /// extra round-trip. Field shape matches the Python pipeline's
    /// `users.sources.*` schema.
    pub sources: serde_json::Value,
    pub links: serde_json::Value,
}

/// `GET /api/me/personalities` — public personalities the caller
/// has sponsored, newest first. Powers the "Personalities you've
/// added" table on the settings page.
pub async fn list_mine(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let Some(pool) = state.pg_pool.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "no database").into_response();
    };
    let Some(me) = current_user(pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    // Per-personality cost breakdown. We display the *current*
    // entry-fee price (`ADD_PERSONALITY_COST`) regardless of what
    // was historically charged — when the fee changes (e.g. $2 →
    // $0.30), the displayed total reflects today's pricing rather
    // than the legacy ledger row. The actual `credit_events`
    // history is left untouched.
    //
    //   • entry fee — fixed at ADD_PERSONALITY_COST (current price)
    //   • storage   — `kind='debit:storage'`         + meta.personality_user_id
    //   • twitter   — `kind='debit:twitter-api'`     + meta.personality_user_id
    //
    // The Twitter rows pre-dating the personality_user_id stamp were
    // backfilled by a one-shot UPDATE so this query covers them too.
    let rows: Vec<SponsoredRow> = sqlx::query_as(
        "SELECT u.username                       AS slug,
                u.name,
                u.description,
                COALESCE(
                    to_char(u.sponsored_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'),
                    ''
                )                                AS sponsored_at,
                (SELECT count(*) FROM documents d WHERE d.user_id = u.id)::bigint AS doc_count,
                $2::bigint                       AS cost_entry_cents,
                COALESCE(
                    (SELECT SUM(-delta)::bigint FROM credit_events e
                      WHERE e.user_id = $1
                        AND e.kind = 'debit:twitter-api'
                        AND (e.meta->>'personality_user_id')::bigint = u.id), 0
                )                                AS cost_twitter_cents,
                COALESCE(
                    (SELECT SUM(-delta)::bigint FROM credit_events e
                      WHERE e.user_id = $1
                        AND e.kind = 'debit:storage'
                        AND (e.meta->>'personality_user_id')::bigint = u.id), 0
                )                                AS cost_storage_cents,
                ($2::bigint + COALESCE(
                    (SELECT SUM(-delta)::bigint FROM credit_events e
                      WHERE e.user_id = $1
                        AND e.kind = 'debit:twitter-api'
                        AND (e.meta->>'personality_user_id')::bigint = u.id), 0
                ) + COALESCE(
                    (SELECT SUM(-delta)::bigint FROM credit_events e
                      WHERE e.user_id = $1
                        AND e.kind = 'debit:storage'
                        AND (e.meta->>'personality_user_id')::bigint = u.id), 0
                ))                               AS cost_cents,
                u.sources,
                u.links
           FROM users u
          WHERE u.sponsored_by = $1
          ORDER BY u.sponsored_at DESC NULLS LAST, u.id DESC
          LIMIT 200",
    )
    .bind(me.id)
    .bind(ADD_PERSONALITY_COST as i64)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    Json(rows).into_response()
}

/// `PUT /api/personalities/{slug}` — update a personality you
/// originally sponsored. Free (no debit). The slug itself is
/// immutable — changing it would break every external link to the
/// library. We only allow editing the editable fields: name,
/// description, sources, links.
pub async fn update(
    State(state): State<Arc<AppState>>,
    Path(slug): Path<String>,
    jar: CookieJar,
    Json(req): Json<AddPersonalityRequest>,
) -> Response {
    let Some(pool) = state.pg_pool.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "no database").into_response();
    };
    let Some(me) = current_user(pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    // Permission gate: the caller must be the sponsor of this slug.
    // 404 (not 403) when the slug isn't theirs — don't leak the
    // existence of personalities they can't touch.
    let target: Option<(i64, Option<i64>)> =
        sqlx::query_as("SELECT id, sponsored_by FROM users WHERE username = $1")
            .bind(&slug)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten();
    let Some((target_id, sponsor)) = target else {
        return (StatusCode::NOT_FOUND, "personality not found").into_response();
    };
    if sponsor != Some(me.id) {
        return (StatusCode::NOT_FOUND, "personality not found").into_response();
    }

    let name = req.name.trim();
    if name.is_empty() || name.len() > 200 {
        return (StatusCode::BAD_REQUEST, "name must be 1–200 chars").into_response();
    }
    let has_source = [
        &req.twitter_handle,
        &req.github_handle,
        &req.huggingface_handle,
        &req.reddit_handle,
        &req.hackernews_handle,
        &req.stackoverflow_user_id,
        &req.arxiv_author,
        &req.dblp_author,
        &req.scholar_user_id,
    ]
    .iter()
    .any(|s| !s.trim().is_empty())
        || !split_websites(&req.websites).is_empty();
    if !has_source {
        return (
            StatusCode::BAD_REQUEST,
            "keep at least one source filled in",
        )
            .into_response();
    }

    // Reject edits that would collide with another personality's
    // name or Twitter handle. Slug is immutable in this path so we
    // don't re-check it; `exclude_id=_target_id` lets the row keep
    // its own values.
    let tw_handle_norm = h(&req.twitter_handle).to_lowercase();
    if let Some((existing_slug, existing_name, field)) =
        find_personality_conflict(pool, &slug, name, &tw_handle_norm, Some(target_id)).await
    {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error":        conflict_message(&field, &existing_name, &existing_slug),
                "field":        field,
                "existingSlug": existing_slug,
                "existingName": existing_name,
            })),
        )
            .into_response();
    }

    let sources = build_sources(&req);
    let links = build_links(&req);

    if let Err(e) = sqlx::query(
        "UPDATE users
            SET name        = $2,
                description = $3,
                sources     = $4::jsonb,
                links       = $5::jsonb,
                updated_at  = now()
          WHERE username    = $1",
    )
    .bind(&slug)
    .bind(name)
    .bind(req.description.trim())
    .bind(&sources)
    .bind(&links)
    .execute(pool)
    .await
    {
        tracing::error!(error = %e, slug = %slug, "personalities.update.failed");
        return (StatusCode::INTERNAL_SERVER_ERROR, "could not update").into_response();
    }
    Json(serde_json::json!({
        "slug": slug,
        "status": "updated",
    }))
    .into_response()
}

pub async fn create(
    State(state): State<Arc<AppState>>,
    jar: CookieJar,
    Json(req): Json<AddPersonalityRequest>,
) -> Response {
    let Some(pool) = state.pg_pool.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "no database").into_response();
    };
    let Some(me) = current_user(pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };

    let name = req.name.trim();
    if name.is_empty() || name.len() > 200 {
        return (StatusCode::BAD_REQUEST, "name must be 1\u{2013}200 chars").into_response();
    }
    let slug = if req.slug.trim().is_empty() {
        slugify(name)
    } else {
        slugify(&req.slug)
    };
    if slug.is_empty() || slug.len() > 64 {
        return (
            StatusCode::BAD_REQUEST,
            "slug must be 1\u{2013}64 chars after normalising",
        )
            .into_response();
    }
    if !slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return (StatusCode::BAD_REQUEST, "slug must be a–z, 0–9, hyphens").into_response();
    }

    // Require at least one source handle. Empty submissions waste the
    // reviewer's time and we already have the structured fields.
    let has_source = [
        &req.twitter_handle,
        &req.github_handle,
        &req.huggingface_handle,
        &req.reddit_handle,
        &req.hackernews_handle,
        &req.stackoverflow_user_id,
        &req.arxiv_author,
        &req.dblp_author,
        &req.scholar_user_id,
    ]
    .iter()
    .any(|s| !s.trim().is_empty())
        || !split_websites(&req.websites).is_empty();
    if !has_source {
        return (StatusCode::BAD_REQUEST, "fill at least one source — Twitter, GitHub, Reddit, Hugging Face, Hacker News, Stack Overflow, arXiv, Scholar, DBLP, or a website").into_response();
    }

    // Soft uniqueness gate against the live `users` table. We surface
    // a structured 409 so the submitter sees "this person is already
    // on Knowledge as @existing" instead of duplicating the work for
    // the admin reviewer.
    let tw_handle_norm = h(&req.twitter_handle).to_lowercase();
    if let Some((existing_slug, existing_name, field)) =
        find_personality_conflict(pool, &slug, name, &tw_handle_norm, None).await
    {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error":        conflict_message(&field, &existing_name, &existing_slug),
                "field":        field,
                "existingSlug": existing_slug,
                "existingName": existing_name,
            })),
        )
            .into_response();
    }

    // Persist into the review queue. All handles ride through as
    // plain strings — `build_sources` / `build_links` aren't applied
    // here because the admin will re-derive them at integration time
    // from the canonical fields stored on the row.
    let tw = h(&req.twitter_handle);
    let gh = h(&req.github_handle);
    let hf = h(&req.huggingface_handle);
    let rd = h(&req.reddit_handle);
    let hn = h(&req.hackernews_handle);
    let so = req.stackoverflow_user_id.trim().to_string();
    let arx = req.arxiv_author.trim().to_string();
    let dblp = req.dblp_author.trim().to_string();
    let scholar = req.scholar_user_id.trim().to_string();
    let sites = split_websites(&req.websites).join("\n");

    if let Err(e) = sqlx::query(
        "INSERT INTO personality_submissions (
            submitter_id, name, slug, description,
            twitter_handle, github_handle, huggingface_handle,
            reddit_handle, hackernews_handle, stackoverflow_user_id,
            arxiv_author, dblp_author, scholar_user_id, websites
         ) VALUES (
            $1, $2, $3, $4,
            $5, $6, $7,
            $8, $9, $10,
            $11, $12, $13, $14
         )",
    )
    .bind(me.id)
    .bind(name)
    .bind(&slug)
    .bind(req.description.trim())
    .bind(&tw)
    .bind(&gh)
    .bind(&hf)
    .bind(&rd)
    .bind(&hn)
    .bind(&so)
    .bind(&arx)
    .bind(&dblp)
    .bind(&scholar)
    .bind(&sites)
    .execute(pool)
    .await
    {
        tracing::error!(error = %e, slug = %slug, "personalities.submission.insert.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "could not record suggestion",
        )
            .into_response();
    }

    // Balance stays the same — no debit happens, but we echo it so
    // the client's credit-pill doesn't have to round-trip again.
    let balance: i32 = sqlx::query_scalar("SELECT credits_balance($1)::int")
        .bind(me.id)
        .fetch_one(pool)
        .await
        .unwrap_or(0);

    Json(AddPersonalityResponse {
        slug,
        balance,
        status: "submitted",
    })
    .into_response()
}
