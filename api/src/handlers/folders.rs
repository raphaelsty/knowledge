//! Custom folder CRUD handlers.
//!
//! Endpoints:
//!   GET    /api/folders        — list all folders (flat, ordered by created_at)
//!   POST   /api/folders        — create a folder
//!   PUT    /api/folders/:id    — update a folder
//!   DELETE /api/folders/:id    — delete a folder (children cascade)

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

/// A custom folder record returned from the API.
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct FolderRecord {
    pub id: String,
    pub label: String,
    pub filter_type: String,
    pub search_query: String,
    pub search_sources: Vec<String>,
    pub tag_filter: Vec<String>,
    pub tag_intersect: bool,
    pub live: bool,
    pub top_k: Option<i32>,
    pub urls: Vec<String>,
    pub pinned_urls: Vec<String>,
    pub excluded_docs: serde_json::Value,
    pub sort_by: String,
    pub parent_id: Option<String>,
}

/// Request body for create/update (same fields, id required for create).
#[derive(Debug, Deserialize)]
pub struct FolderBody {
    pub id: String,
    pub label: String,
    #[serde(default = "default_filter_type")]
    pub filter_type: String,
    #[serde(default)]
    pub search_query: String,
    #[serde(default)]
    pub search_sources: Vec<String>,
    #[serde(default)]
    pub tag_filter: Vec<String>,
    #[serde(default)]
    pub tag_intersect: bool,
    #[serde(default = "default_true")]
    pub live: bool,
    pub top_k: Option<i32>,
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub pinned_urls: Vec<String>,
    #[serde(default = "default_excluded_docs")]
    pub excluded_docs: serde_json::Value,
    #[serde(default = "default_sort_by")]
    pub sort_by: String,
    pub parent_id: Option<String>,
}

fn default_filter_type() -> String {
    "search".to_string()
}
fn default_true() -> bool {
    true
}
fn default_excluded_docs() -> serde_json::Value {
    serde_json::json!([])
}
fn default_sort_by() -> String {
    "default".to_string()
}

/// GET /api/folders — list all folders ordered by creation time.
pub async fn list(State(pool): State<PgPool>) -> impl IntoResponse {
    let rows: Result<Vec<FolderRecord>, _> = sqlx::query_as(
        "SELECT id, label, filter_type, search_query, search_sources, tag_filter,
                tag_intersect, live, top_k, urls, pinned_urls, excluded_docs,
                sort_by, parent_id
         FROM custom_folders
         ORDER BY created_at ASC",
    )
    .fetch_all(&pool)
    .await;

    match rows {
        Ok(folders) => Json(serde_json::json!(folders)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// POST /api/folders — create a folder.
pub async fn create(State(pool): State<PgPool>, Json(body): Json<FolderBody>) -> impl IntoResponse {
    let result = sqlx::query(
        "INSERT INTO custom_folders
            (id, label, filter_type, search_query, search_sources, tag_filter,
             tag_intersect, live, top_k, urls, pinned_urls, excluded_docs,
             sort_by, parent_id)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)",
    )
    .bind(&body.id)
    .bind(&body.label)
    .bind(&body.filter_type)
    .bind(&body.search_query)
    .bind(&body.search_sources)
    .bind(&body.tag_filter)
    .bind(body.tag_intersect)
    .bind(body.live)
    .bind(body.top_k)
    .bind(&body.urls)
    .bind(&body.pinned_urls)
    .bind(&body.excluded_docs)
    .bind(&body.sort_by)
    .bind(&body.parent_id)
    .execute(&pool)
    .await;

    match result {
        Ok(_) => StatusCode::CREATED.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// PUT /api/folders/:id — update a folder.
pub async fn update(
    State(pool): State<PgPool>,
    Path(id): Path<String>,
    Json(body): Json<FolderBody>,
) -> impl IntoResponse {
    let result = sqlx::query(
        "UPDATE custom_folders SET
            label=$1, filter_type=$2, search_query=$3, search_sources=$4,
            tag_filter=$5, tag_intersect=$6, live=$7, top_k=$8,
            urls=$9, pinned_urls=$10, excluded_docs=$11, sort_by=$12,
            parent_id=$13, updated_at=now()
         WHERE id=$14",
    )
    .bind(&body.label)
    .bind(&body.filter_type)
    .bind(&body.search_query)
    .bind(&body.search_sources)
    .bind(&body.tag_filter)
    .bind(body.tag_intersect)
    .bind(body.live)
    .bind(body.top_k)
    .bind(&body.urls)
    .bind(&body.pinned_urls)
    .bind(&body.excluded_docs)
    .bind(&body.sort_by)
    .bind(&body.parent_id)
    .bind(&id)
    .execute(&pool)
    .await;

    match result {
        Ok(r) if r.rows_affected() == 0 => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "folder not found"})),
        )
            .into_response(),
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// DELETE /api/folders/:id — delete a folder and its descendants (CASCADE).
pub async fn delete_folder(
    State(pool): State<PgPool>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query("DELETE FROM custom_folders WHERE id=$1")
        .bind(&id)
        .execute(&pool)
        .await;

    match result {
        Ok(r) if r.rows_affected() == 0 => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "folder not found"})),
        )
            .into_response(),
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
