//! Read-only API for the document_categories catalogue.
//!
//! `GET /api/document-categories` returns every row in the table
//! ordered by `sort_order`, plus its `group_name` so the frontend can
//! render section headers without a second round-trip. Filtering /
//! grouping is left to the client — the catalogue is small (currently
//! 178 rows) and never changes per-request, so we don't paginate.
//!
//! `POST /api/document-categories/by-url` is the batch sibling: takes
//! a list of document URLs and returns the slugs the categorize
//! daemon assigned to each one. Used by the search-picker frontend
//! to aggregate "which categories are these search hits about?" so
//! the picker can promote categories that semantically match the
//! user's query (via ColBERT) on top of plain lexical name matching.

use axum::{extract::State, response::IntoResponse, response::Response, Json};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct CategoryRow {
    pub slug: String,
    pub name: String,
    /// Coarse UI grouping ("Pretraining & Architecture", "Releases", …).
    /// Lets the frontend render the dropdown as collapsible sections
    /// without a second per-request join.
    pub group: String,
    pub description: String,
    pub sort_order: i32,
}

#[derive(Deserialize)]
pub struct ListCategoriesParams {
    /// Optional user slug — when present, the result is restricted
    /// to categories that have at least one assignment against the
    /// user's documents. Used by the picker on personal-page views
    /// so the operator only sees topics that actually carve up that
    /// library, not the whole 178-row catalogue.
    pub user: Option<String>,
}

/// `GET /api/document-categories[?user=<slug>]` — full catalogue or,
/// when `user` is supplied, the subset of categories with at least
/// one assignment against the named user's documents.
pub async fn list_document_categories(
    State(pool): State<PgPool>,
    axum::extract::Query(params): axum::extract::Query<ListCategoriesParams>,
) -> Response {
    let user = params
        .user
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let rows: Vec<(String, String, String, String, i32)> = if let Some(slug) = user.as_ref() {
        // Restrict to categories the user's documents actually
        // cover. EXISTS over document_category_assignments joined
        // to users picks up every doc owned by the user (matched
        // by username, which is the public slug the frontend
        // sends). Order kept by sort_order so the desktop picker
        // can still group by `group_name` in catalogue order.
        match sqlx::query_as(
            "SELECT dc.slug, dc.name, dc.group_name, dc.description, dc.sort_order
               FROM document_categories dc
              WHERE EXISTS (
                  SELECT 1
                    FROM document_category_assignments a
                    JOIN users u ON u.id = a.user_id
                   WHERE a.category_id = dc.id
                     AND u.username = $1
              )
              ORDER BY dc.sort_order, dc.slug",
        )
        .bind(slug)
        .fetch_all(&pool)
        .await
        {
            Ok(rs) => rs,
            Err(e) => {
                tracing::error!(error = %e, "document_categories.list.user.failed");
                return Json::<Vec<CategoryRow>>(vec![]).into_response();
            }
        }
    } else {
        match sqlx::query_as(
            "SELECT slug, name, group_name, description, sort_order
               FROM document_categories
              ORDER BY sort_order, slug",
        )
        .fetch_all(&pool)
        .await
        {
            Ok(rs) => rs,
            Err(e) => {
                tracing::error!(error = %e, "document_categories.list.failed");
                return Json::<Vec<CategoryRow>>(vec![]).into_response();
            }
        }
    };
    let out: Vec<CategoryRow> = rows
        .into_iter()
        .map(|(slug, name, group, description, sort_order)| CategoryRow {
            slug,
            name,
            group,
            description,
            sort_order,
        })
        .collect();
    Json(out).into_response()
}

/// GET /api/document-categories/urls?slugs=a,b,c[&limit=20000]
///
/// Returns the deduplicated set of document URLs assigned to ANY of
/// the supplied slugs (OR semantics). Used as a pre-filter source by
/// the search index path: the frontend asks for the URL set first,
/// then embeds it as a `url IN (?, ?, …)` clause in the ColBERT
/// query so the index never has to scan or score docs that wouldn't
/// pass the category filter anyway.
///
/// Hard-capped at 50,000 URLs even if `limit` is higher, since the
/// downstream `url IN (...)` filter has to fit in PyLate's SQL
/// metadata parser. Practically every realistic category is well
/// under that cap (the most-assigned slug today carries ~22k URLs).
#[derive(Deserialize)]
pub struct UrlsBySlugsParams {
    pub slugs: Option<String>,
    pub limit: Option<i64>,
}

pub async fn urls_by_slugs(
    State(pool): State<PgPool>,
    axum::extract::Query(params): axum::extract::Query<UrlsBySlugsParams>,
) -> Response {
    let slugs: Vec<String> = params
        .slugs
        .as_deref()
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if slugs.is_empty() {
        return Json::<Vec<String>>(vec![]).into_response();
    }
    let limit = params.limit.unwrap_or(20_000).clamp(1, 50_000);
    let rows: Vec<(String,)> = match sqlx::query_as(
        "SELECT DISTINCT a.url
           FROM document_category_assignments a
           JOIN document_categories dc ON dc.id = a.category_id
          WHERE dc.slug = ANY($1)
          LIMIT $2",
    )
    .bind(&slugs)
    .bind(limit)
    .fetch_all(&pool)
    .await
    {
        Ok(rs) => rs,
        Err(e) => {
            tracing::error!(error = %e, "document_categories.urls_by_slugs.failed");
            return Json::<Vec<String>>(vec![]).into_response();
        }
    };
    let out: Vec<String> = rows.into_iter().map(|(u,)| u).collect();
    Json(out).into_response()
}

#[derive(Deserialize)]
pub struct CategoriesByUrlRequest {
    /// Document URLs to look up. Capped server-side at 500 to keep
    /// the IN-list bounded.
    pub urls: Vec<String>,
}

/// `POST /api/document-categories/by-url` — given a batch of doc
/// URLs, return the category slugs each one is assigned to.
///
/// Response shape: `{ "<url>": ["slug1", "slug2", ...], ... }`. URLs
/// the picker queried but the categorize daemon hasn't touched are
/// simply absent from the map (no empty array). Slug ordering inside
/// each list is `is_primary DESC, score DESC` so the most-confident
/// assignment is first.
pub async fn categories_by_url(
    State(pool): State<PgPool>,
    Json(req): Json<CategoriesByUrlRequest>,
) -> Response {
    // Defensive bound — the caller is the picker's ColBERT hit list,
    // currently capped at 200, but we don't want a future caller
    // accidentally fanning out a 100k-URL request.
    let mut urls = req.urls;
    if urls.len() > 500 {
        urls.truncate(500);
    }
    if urls.is_empty() {
        return Json::<HashMap<String, Vec<String>>>(HashMap::new()).into_response();
    }
    let rows: Vec<(String, String)> = match sqlx::query_as(
        "SELECT a.url, dc.slug
           FROM document_category_assignments a
           JOIN document_categories dc ON dc.id = a.category_id
          WHERE a.url = ANY($1)
          ORDER BY a.is_primary DESC, a.score DESC",
    )
    .bind(&urls)
    .fetch_all(&pool)
    .await
    {
        Ok(rs) => rs,
        Err(e) => {
            tracing::error!(error = %e, "document_categories.by_url.failed");
            return Json::<HashMap<String, Vec<String>>>(HashMap::new()).into_response();
        }
    };
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for (url, slug) in rows {
        out.entry(url).or_default().push(slug);
    }
    Json(out).into_response()
}
