//! Data API handlers — folder tree, sources, health.
//!
//! Ported from the Python FastAPI data server (sources/api.py).

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use sqlx::PgPool;

/// GET /api/folder_tree
pub async fn folder_tree(State(pool): State<PgPool>) -> impl IntoResponse {
    let row: Option<(serde_json::Value,)> =
        sqlx::query_as("SELECT data FROM generated_data WHERE key = $1")
            .bind("folder_tree")
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);

    match row {
        Some((data,)) => Json(data).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "folder_tree not found"})),
        )
            .into_response(),
    }
}

/// GET /api/sources
pub async fn sources(State(pool): State<PgPool>) -> impl IntoResponse {
    let row: Option<(serde_json::Value,)> =
        sqlx::query_as("SELECT data FROM generated_data WHERE key = $1")
            .bind("sources")
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);

    match row {
        Some((data,)) => Json(data).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "sources not found"})),
        )
            .into_response(),
    }
}

/// GET /api/pipeline_run — last pipeline run metadata (timings, counts).
pub async fn pipeline_run(State(pool): State<PgPool>) -> impl IntoResponse {
    let row: Option<(serde_json::Value,)> =
        sqlx::query_as("SELECT data FROM generated_data WHERE key = $1")
            .bind("pipeline_run")
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);

    match row {
        Some((data,)) => Json(data).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "no pipeline run recorded"})),
        )
            .into_response(),
    }
}

/// GET /api/health
pub async fn data_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}
