//! Private per-user document favorites (the "star" on every search card).
//!
//!   GET    /auth/me/favorite-docs            → [ "url1", "url2", ... ]
//!   POST   /auth/me/favorite-docs            → body `{"url":"..."}`
//!   DELETE /auth/me/favorite-docs?url=...    → remove one
//!
//! All endpoints are session-gated. The URL is the primary key so the
//! toggle is idempotent — repeating a POST is a no-op, repeating a
//! DELETE returns 204.
//!
//! The list never leaks a user's favorites to another user or to
//! unauthenticated callers; there's no public "favorited by" view.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use serde::Deserialize;
use sqlx::PgPool;

use crate::handlers::auth::current_user;

/// GET /auth/me/favorite-docs → list of URLs, most-recent first.
///
/// Drops orphaned favorites (URLs with no row in `documents`) so the
/// rail count matches what's actually viewable. Favorites are
/// personal: we don't filter by which library currently owns the row,
/// because the user expects their stars to follow them everywhere.
pub async fn list(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT f.url
           FROM favorite_documents f
          WHERE f.user_id = $1
            AND EXISTS (SELECT 1 FROM documents d WHERE d.url = f.url)
          ORDER BY f.created_at DESC",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    Json(rows.into_iter().map(|r| r.0).collect::<Vec<String>>()).into_response()
}

/// GET /auth/me/favorite-docs/owners → distinct slugs that own at
/// least one favorited document. Lets the search page expand its
/// per-library fanout to include those libs whenever the Favorites
/// filter is active, so query-time pre-filtering by `url IN (…)`
/// finds the docs even when the owning library isn't selected.
pub async fn list_owners(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT u.username
           FROM favorite_documents f
           JOIN documents d ON d.url = f.url
           JOIN users u ON u.id = d.user_id
          WHERE f.user_id = $1
          ORDER BY u.username",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    Json(rows.into_iter().map(|r| r.0).collect::<Vec<String>>()).into_response()
}

/// GET /auth/me/favorite-docs/full → list of hydrated doc objects.
///
/// Joins `favorite_documents` with the `documents` table so the UI can
/// render cards for every favorited URL regardless of which personality
/// page the user is on. A URL may appear in several personalities'
/// `documents` rows (multiple libraries sharing the same link); we use
/// DISTINCT ON to pick one representative row per URL (most recent date
/// wins), then re-sort by star time so the user sees the list in the
/// order they curated it.
#[allow(clippy::type_complexity)]
pub async fn list_full(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    // Resolve each favorite to the same shape the personal page
    // serves: when the URL is the user's mirror of an aggregated
    // anchor, swap in the feed_snapshot representative's metadata
    // (rich summary, linked URLs, sharers, …) so the favorites view
    // renders the EXACT card the feed shows. Long-tail favorites
    // that aren't in feed_snapshot fall back to the user's own
    // documents row.
    let sql = "
        WITH deduped AS (
            SELECT DISTINCT ON (d.url)
                   d.url,
                   d.title,
                   d.summary,
                   d.date,
                   d.tags,
                   d.extra_tags,
                   d.source,
                   d.source_url,
                   d.linked_urls,
                   d.link_hosts,
                   d.canonical_url,
                   d.canonical_referenced_urls
              FROM documents d
             WHERE d.deleted = FALSE
               AND d.url IN (
                   SELECT url FROM favorite_documents WHERE user_id = $1
               )
             ORDER BY d.url, d.date DESC NULLS LAST
        ),
        resolved AS (
            SELECT dd.*,
                   COALESCE(
                       (SELECT ref FROM unnest(dd.canonical_referenced_urls) ref
                         ORDER BY CASE
                           WHEN ref LIKE 'https://arxiv.org/abs/%'       THEN 1
                           WHEN ref LIKE 'https://huggingface.co/%'      THEN 2
                           WHEN ref LIKE 'https://github.com/%'          THEN 3
                           WHEN ref LIKE 'https://openreview.net/%'      THEN 4
                           WHEN ref LIKE 'https://doi.org/%'             THEN 5
                           WHEN ref LIKE 'https://paperswithcode.com/%'  THEN 6
                           WHEN ref LIKE 'https://aclanthology.org/%'    THEN 7
                           WHEN ref LIKE 'https://semanticscholar.org/%' THEN 8
                           WHEN ref LIKE 'https://distill.pub/%'         THEN 9
                           WHEN ref LIKE 'https://biorxiv.org/%'         THEN 10
                           WHEN ref LIKE 'https://medrxiv.org/%'         THEN 11
                           ELSE 99
                         END, ref LIMIT 1),
                       dd.canonical_url
                   ) AS anchor_url
              FROM deduped dd
        )
        SELECT COALESCE(fs.url,                   r.url)           AS url,
               COALESCE(NULLIF(fs.title, ''),     r.title)         AS title,
               COALESCE(NULLIF(fs.summary, ''),   r.summary)       AS summary,
               COALESCE(
                   to_char(COALESCE(fs.date, r.date), 'YYYY-MM-DD'),
                   ''
               )                                                    AS date,
               r.tags                                               AS tags,
               r.extra_tags                                         AS extra_tags,
               COALESCE(NULLIF(fs.source, ''),    r.source)         AS source,
               COALESCE(fs.source_url,            r.source_url)     AS source_url,
               COALESCE(fs.linked_urls,           r.linked_urls)    AS linked_urls,
               COALESCE(NULLIF(fs.link_hosts, '{}'::text[]),
                        r.link_hosts)                               AS link_hosts,
               r.anchor_url                                         AS anchor_url,
               COALESCE(fs.sharers,
                        '[]'::jsonb)                                AS sharers,
               COALESCE(fs.sharer_count, 0)                         AS sharer_count
          FROM resolved r
          LEFT JOIN feed_snapshot fs ON fs.anchor_url = r.anchor_url
          JOIN favorite_documents f ON f.url = r.url AND f.user_id = $1
         ORDER BY f.created_at DESC
    ";
    let rows: Vec<(
        String,            // url
        String,            // title
        String,            // summary
        String,            // date
        Vec<String>,       // tags
        Vec<String>,       // extra_tags
        String,            // source
        Option<String>,    // source_url
        serde_json::Value, // linked_urls
        Vec<String>,       // link_hosts
        String,            // anchor_url
        serde_json::Value, // sharers
        i32,               // sharer_count
    )> = sqlx::query_as(sql)
        .bind(me.id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();
    let out: Vec<_> = rows
        .into_iter()
        .map(
            |(
                url,
                title,
                summary,
                date,
                tags,
                extra_tags,
                source,
                source_url,
                linked_urls,
                link_hosts,
                anchor_url,
                sharers,
                sharer_count,
            )| {
                serde_json::json!({
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "date": date,
                    "tags": tags,
                    "extra-tags": extra_tags,
                    "source": source,
                    "source_url": source_url,
                    "linked_urls": linked_urls,
                    "link_hosts": link_hosts,
                    "anchor_url": anchor_url,
                    "sharers": sharers,
                    "sharer_count": sharer_count,
                })
            },
        )
        .collect();
    Json(out).into_response()
}

#[derive(Deserialize)]
pub struct AddRequest {
    pub url: String,
}

/// POST /auth/me/favorite-docs → { url: "..." }
pub async fn add(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<AddRequest>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let url = req.url.trim();
    if url.is_empty() {
        return (StatusCode::BAD_REQUEST, "url is required").into_response();
    }
    if let Err(e) = sqlx::query(
        "INSERT INTO favorite_documents (user_id, url)
         VALUES ($1, $2)
         ON CONFLICT (user_id, url) DO NOTHING",
    )
    .bind(me.id)
    .bind(url)
    .execute(&pool)
    .await
    {
        tracing::error!(error = %e, "favorite_docs.add.failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("insert failed: {e}"),
        )
            .into_response();
    }
    // Mirror the upvote into the caller's documents library so the row
    // surfaces on their /<slug> personal page immediately. We copy
    // metadata from any existing documents row with the same URL
    // (typically the user the feed card came from) and stamp
    // `created_at = NOW()` so the personal page's
    // `ORDER BY date DESC, created_at DESC` floats the freshly-saved
    // upvote to the top — matching the behaviour the compose dialog
    // already gives a manual save.
    //
    // ON CONFLICT DO NOTHING leaves an existing row alone: if the URL
    // was already in the user's library (e.g. ingested via the daily
    // sync), the upvote stays a pure favorite without resetting its
    // `created_at`.
    //
    // Skipped silently when no documents row exists anywhere — the
    // favorite is still recorded, but we have no metadata to populate
    // a useful library row with. The favorites rail already filters
    // orphan favorites via the JOIN in `list`, so this stays
    // consistent.
    // Stamp `date = CURRENT_DATE` (and not the source row's
    // publication date) so the mirrored doc lands at the top of the
    // upvoter's personal page: the page sorts by `date DESC` alone
    // (publication date), and inheriting an old paper's date would
    // sink the row back into that paper's year. From the upvoter's
    // perspective they DID add it today, so showing today's date on
    // the card matches the action.
    if let Err(e) = sqlx::query(
        "INSERT INTO documents (
             user_id, url, title, summary, clean_title, clean_summary,
             date, tags, extra_tags, source, source_url,
             linked_urls, link_hosts, created_via_favorite
         )
         SELECT $1, $2,
                d.title, d.summary, d.clean_title, d.clean_summary,
                CURRENT_DATE, d.tags, d.extra_tags, d.source, d.source_url,
                d.linked_urls, d.link_hosts, TRUE
           FROM documents d
          WHERE d.url = $2
            AND d.deleted = FALSE
          ORDER BY d.indexed DESC, d.cleaned DESC, d.created_at DESC
          LIMIT 1
         ON CONFLICT (user_id, url) DO NOTHING",
    )
    .bind(me.id)
    .bind(url)
    .execute(&pool)
    .await
    {
        // Non-fatal: the favorite itself succeeded; we just couldn't
        // mirror the doc into the user's library. Log + continue.
        tracing::error!(error = %e, "favorite_docs.add.mirror_failed");
    }
    StatusCode::NO_CONTENT.into_response()
}

#[derive(Deserialize)]
pub struct RemoveQuery {
    pub url: String,
}

/// DELETE /auth/me/favorite-docs?url=...
pub async fn remove(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Query(q): Query<RemoveQuery>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let url = q.url.trim();
    if url.is_empty() {
        return (StatusCode::BAD_REQUEST, "url is required").into_response();
    }
    let _ = sqlx::query("DELETE FROM favorite_documents WHERE user_id = $1 AND url = $2")
        .bind(me.id)
        .bind(url)
        .execute(&pool)
        .await;
    // Un-upvote: drop the documents row IFF it was synthesised by an
    // earlier upvote (`created_via_favorite = TRUE`). Rows justified by
    // a real sync (manual save, GitHub stars, HN submissions, …) stay
    // put — toggling the heart off shouldn't erase a doc the user
    // actually owns. Sync upserts clear the flag on conflict so a doc
    // that started life as an upvote but later got confirmed by a
    // real source is treated as "real" by this delete.
    let _ = sqlx::query(
        "DELETE FROM documents
          WHERE user_id = $1
            AND url = $2
            AND created_via_favorite = TRUE",
    )
    .bind(me.id)
    .bind(url)
    .execute(&pool)
    .await;
    StatusCode::NO_CONTENT.into_response()
}
