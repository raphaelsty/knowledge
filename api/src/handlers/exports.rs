//! Library export endpoint.
//!
//! `GET /api/personalities/{slug}/export.jsonl`
//!
//! Hybrid endpoint — same URL serves the file when clicked from the
//! UI (browser saves it via `Content-Disposition: attachment`) and
//! works as a CLI / scripting endpoint when called with a bearer
//! token. No second download URL, no temporary signed redirects.
//!
//! Auth: session cookie OR `Authorization: Bearer kn_…` token. A
//! signed-in caller is required — exports are free for everyone who
//! has an account, but anonymous callers cannot pull data out of the
//! platform. Anonymous → 401.
//!
//! Access:
//!   • Public library          → any signed-in caller may export.
//!   • Private library         → owner-only; non-owner gets 404 so
//!                                the library's existence isn't leaked.
//!
//! Query parameters:
//!   • `quote=1`               → return a JSON summary
//!                                (`{exportCount, docCount, slug}`)
//!                                instead of streaming. The UI uses
//!                                this to populate the confirmation
//!                                dialog with an accurate count after
//!                                the user picks a date range.
//!   • `limit=N`               → cap the export at N rows (after the
//!                                ORDER BY in the streaming SQL, so
//!                                a partial export keeps the newest
//!                                rows).
//!   • `date_from=YYYY-MM-DD`  → inclusive lower bound on document
//!                                date. Omit for no lower bound.
//!   • `date_to=YYYY-MM-DD`    → inclusive upper bound on document
//!                                date. Omit for no upper bound.
//!
//! Streaming: `sqlx::fetch()` opens a server-side cursor, so the
//! handler doesn't allocate the whole result set in RAM. JSONL is
//! emitted one object per line; the network drives backpressure.
//!
//! Hard cap: `EXPORT_HARD_CAP` (50k rows) ceilings every export to
//! keep a runaway caller from saturating the server's worker pool
//! or pegging an open connection for hours. The cap applies after
//! `date_from` / `date_to` and after any user-supplied `limit`, so a
//! request for "everything between Jan 1 and Dec 31" still gets at
//! most 50k rows. The quote response carries `maxLimit` so the UI
//! can show the ceiling and surface "limited by server cap" copy.
//!
//! Audit: every real download (not quote) inserts one row into
//! `export_downloads` *before* the stream starts, so a client that
//! drops mid-stream still leaves a record. The row captures caller,
//! target, doc_count, date range and timestamp — enough to answer
//! "who exported what, when" without mining API logs.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use sqlx::Row;

use crate::handlers::auth::current_user;
use crate::handlers::tokens::resolve_bearer;
use crate::state::AppState;

/// Maximum number of documents any single export request will
/// stream. Applied after `date_from` / `date_to` and after any
/// user-supplied `limit`, so the server can't be coerced into
/// shipping more than this. Picked to keep the worst-case payload
/// under ~50 MB of JSONL and the streaming SQL within a few seconds
/// even on a cold cache.
const EXPORT_HARD_CAP: i64 = 50_000;

#[derive(Deserialize, Default)]
pub struct ExportQuery {
    /// When set (any truthy value) return a JSON summary instead of
    /// streaming the bytes. The UI uses this to refresh the count
    /// after the user changes the date range or limit.
    #[serde(default)]
    pub quote: Option<String>,
    /// Cap the export at this many rows. Applied after `ORDER BY
    /// date DESC` so the user always gets the newest matching docs.
    /// Omitted = export everything in range.
    #[serde(default)]
    pub limit: Option<i64>,
    /// Inclusive lower bound on `documents.date` (ISO `YYYY-MM-DD`).
    /// NULL means no lower bound.
    #[serde(default)]
    pub date_from: Option<String>,
    /// Inclusive upper bound on `documents.date` (ISO `YYYY-MM-DD`).
    /// NULL means no upper bound.
    #[serde(default)]
    pub date_to: Option<String>,
}

#[derive(Serialize)]
struct ExportQuote {
    /// Documents that *would* be exported with the current limit +
    /// date range applied. Always ≤ `maxLimit`.
    #[serde(rename = "exportCount")]
    export_count: i64,
    /// Total documents in the library, ignoring the date range +
    /// hard cap. Used by the UI to show "N of M" so the user can
    /// see how restrictive their filter is.
    #[serde(rename = "docCount")]
    doc_count: i64,
    /// Server's hard ceiling per request (see `EXPORT_HARD_CAP`).
    /// The picker shows it as the "Max" preset so the UI never lets
    /// the user request more than the server will ship.
    #[serde(rename = "maxLimit")]
    max_limit: i64,
    /// True when `export_count` would have been larger without the
    /// hard cap — i.e. the picker should show a "limited by server
    /// cap" hint.
    #[serde(rename = "capped")]
    capped: bool,
    /// Personality slug for confirmation.
    slug: String,
}

/// Resolve the caller via either auth path. Returns None for
/// anonymous, which the handler rejects with 401.
async fn caller_user_id(pool: &PgPool, headers: &HeaderMap, jar: &CookieJar) -> Option<i64> {
    if let Some(uid) = resolve_bearer(pool, headers).await {
        return Some(uid);
    }
    current_user(pool, jar).await.map(|me| me.id)
}

/// Personality lookup result. `vip` isn't used here today but stays
/// in the struct/query because it's free to fetch and the caller log
/// may want it later.
#[allow(dead_code)]
struct Target {
    id: i64,
    public: bool,
    vip: bool,
    doc_count: i64,
}

async fn lookup_target(pool: &PgPool, slug: &str) -> Option<Target> {
    let row = sqlx::query(
        "SELECT u.id, u.public, u.vip,
                (SELECT count(*) FROM documents d
                  WHERE d.user_id = u.id
                    AND d.deleted = false)::bigint AS n
           FROM users u
          WHERE u.username = $1",
    )
    .bind(slug)
    .fetch_optional(pool)
    .await
    .ok()??;
    Some(Target {
        id: row.try_get::<i64, _>("id").ok()?,
        public: row.try_get::<bool, _>("public").ok()?,
        vip: row.try_get::<bool, _>("vip").ok()?,
        doc_count: row.try_get::<i64, _>("n").ok()?,
    })
}

/// Count the documents that match the date range, after applying
/// `limit` and the server's hard cap. Mirrors the WHERE / LIMIT
/// clauses used by the streaming query so the quote and the real
/// download agree. Returns `(effective_count, capped_by_server)`
/// where `capped_by_server` is true when the cap (not the user's
/// `limit`) is what's restricting the output — that's the signal
/// the UI uses to show the "limited by server cap" copy.
async fn count_filtered_docs(
    pool: &PgPool,
    target_id: i64,
    date_from: Option<&str>,
    date_to: Option<&str>,
    limit: Option<i64>,
) -> (i64, bool) {
    let matched: i64 = sqlx::query_scalar(
        "SELECT count(*)::bigint
           FROM documents
          WHERE user_id = $1
            AND deleted = false
            AND ($2::date IS NULL OR date >= $2::date)
            AND ($3::date IS NULL OR date <= $3::date)",
    )
    .bind(target_id)
    .bind(date_from)
    .bind(date_to)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    // Two ceilings: the user's optional cap and the server's hard
    // cap. We track which one bites so the UI can explain the count.
    let user_capped = match limit {
        Some(n) if n > 0 && n < matched => n,
        _ => matched,
    };
    let effective = user_capped.min(EXPORT_HARD_CAP);
    let capped_by_server =
        matched > EXPORT_HARD_CAP && limit.map(|n| n > EXPORT_HARD_CAP).unwrap_or(true);
    (effective, capped_by_server)
}

/// `GET /api/personalities/{slug}/export.jsonl`
pub async fn export_personality(
    State(state): State<Arc<AppState>>,
    Path(slug): Path<String>,
    Query(q): Query<ExportQuery>,
    headers: HeaderMap,
    jar: CookieJar,
) -> Response {
    let Some(pool) = state.pg_pool.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "no database").into_response();
    };

    let is_quote = q
        .quote
        .as_deref()
        .is_some_and(|s| !s.is_empty() && s != "0");

    // 1. Resolve target personality.
    let Some(target) = lookup_target(pool, &slug).await else {
        return (StatusCode::NOT_FOUND, "personality not found").into_response();
    };

    // 2. Auth required for every export (including quotes — there's
    //    no public quote API, the dialog is a signed-in-only surface).
    let caller = caller_user_id(pool, &headers, &jar).await;
    let Some(caller_id) = caller else {
        return (StatusCode::UNAUTHORIZED, "sign in to export").into_response();
    };

    // 3. Private libraries are owner-only — return 404 (not 403) so
    //    we don't leak the library's existence.
    if !target.public && caller_id != target.id {
        return (StatusCode::NOT_FOUND, "personality not found").into_response();
    }

    // 4. Validate date strings up front so a typo doesn't get all the
    //    way to the streaming SQL before failing.
    if !date_str_is_valid(q.date_from.as_deref()) || !date_str_is_valid(q.date_to.as_deref()) {
        return (
            StatusCode::BAD_REQUEST,
            "date_from / date_to must be ISO YYYY-MM-DD",
        )
            .into_response();
    }

    let (export_count, capped) = count_filtered_docs(
        pool,
        target.id,
        q.date_from.as_deref(),
        q.date_to.as_deref(),
        q.limit,
    )
    .await;

    // 5. Quote-only path: return the count, no audit row.
    if is_quote {
        return Json(ExportQuote {
            export_count,
            doc_count: target.doc_count,
            max_limit: EXPORT_HARD_CAP,
            capped,
            slug,
        })
        .into_response();
    }

    // 6. Audit row goes in BEFORE the stream so a client that drops
    //    mid-download still leaves a record. The query uses
    //    parameter casts because `q.date_*` may be NULL.
    if let Err(e) = sqlx::query(
        "INSERT INTO export_downloads
            (user_id, target_user_id, doc_count, date_from, date_to)
         VALUES ($1, $2, $3, $4::date, $5::date)",
    )
    .bind(caller_id)
    .bind(target.id)
    .bind(export_count)
    .bind(q.date_from.as_deref())
    .bind(q.date_to.as_deref())
    .execute(pool)
    .await
    {
        // Log + carry on. We'd rather serve the user than block them
        // on an audit-table write — the table is a nice-to-have, not
        // a correctness gate.
        tracing::warn!(
            error = %e,
            caller_id = %caller_id,
            target_id = %target.id,
            "exports.audit.insert.failed",
        );
    }

    // 7. Stream the documents as JSONL via a server-side cursor.
    //    The WHERE + LIMIT here MUST mirror count_filtered_docs above
    //    so `export_count` accurately reflects what we ship.
    let target_id = target.id;
    let date_from = q.date_from.clone();
    let date_to = q.date_to.clone();
    let stream_limit = export_count.max(0);
    let pool_for_stream = pool.clone();
    let body_stream = async_stream::stream! {
        let mut rows = sqlx::query(
            "SELECT url, title, summary, date::text AS date, tags, extra_tags,
                    source, source_url
               FROM documents
              WHERE user_id = $1
                AND deleted = false
                AND ($3::date IS NULL OR date >= $3::date)
                AND ($4::date IS NULL OR date <= $4::date)
              ORDER BY date DESC NULLS LAST, url
              LIMIT $2",
        )
        .bind(target_id)
        .bind(stream_limit)
        .bind(date_from.as_deref())
        .bind(date_to.as_deref())
        .fetch(&pool_for_stream);

        while let Some(row_res) = rows.try_next().await.transpose() {
            let row = match row_res {
                Ok(r) => r,
                Err(e) => {
                    yield Err::<bytes::Bytes, std::io::Error>(std::io::Error::other(
                        format!("db error: {e}"),
                    ));
                    return;
                }
            };
            let doc = serde_json::json!({
                "url":        row.try_get::<String, _>("url").unwrap_or_default(),
                "title":      row.try_get::<String, _>("title").unwrap_or_default(),
                "summary":    row.try_get::<String, _>("summary").unwrap_or_default(),
                "date":       row.try_get::<Option<String>, _>("date").ok().flatten(),
                "tags":       row.try_get::<Vec<String>, _>("tags").unwrap_or_default(),
                "extra_tags": row.try_get::<Vec<String>, _>("extra_tags").unwrap_or_default(),
                "source":     row.try_get::<String, _>("source").unwrap_or_default(),
                "source_url": row.try_get::<Option<String>, _>("source_url").ok().flatten(),
            });
            let mut line = serde_json::to_vec(&doc).unwrap_or_default();
            line.push(b'\n');
            yield Ok(bytes::Bytes::from(line));
        }
    };

    let filename = format!("{slug}-knowledge-export.jsonl");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        )
        // Per-caller stream — never cache.
        .header(header::CACHE_CONTROL, "private, no-store")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Cheap validator for the `date_from` / `date_to` query params.
/// `None` is valid (means "no bound"). For `Some`, we require an ISO
/// `YYYY-MM-DD` literal — Postgres' `date` cast would reject anything
/// else anyway, but failing here gives a 400 with a clear message
/// instead of a 500 mid-stream.
fn date_str_is_valid(s: Option<&str>) -> bool {
    let Some(s) = s else { return true };
    if s.is_empty() {
        return true;
    }
    if s.len() != 10 {
        return false;
    }
    let bytes = s.as_bytes();
    bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[..4].iter().all(|b| b.is_ascii_digit())
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
        && bytes[8..].iter().all(|b| b.is_ascii_digit())
}
