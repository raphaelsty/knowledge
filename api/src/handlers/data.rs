//! Data API handlers — folder tree, sources, favorites, health.
//!
//! Ported from the Python FastAPI data server (sources/api.py).

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::Deserialize;
use sqlx::PgPool;

/// Request body for toggling favorites.
#[derive(Debug, Deserialize)]
pub struct FavoriteRequest {
    pub url: String,
}

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

/// GET /api/favorites
pub async fn favorites(State(pool): State<PgPool>) -> Json<Vec<String>> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT url FROM favorites ORDER BY created_at DESC")
            .fetch_all(&pool)
            .await
            .unwrap_or_default();

    Json(rows.into_iter().map(|(url,)| url).collect())
}

/// POST /api/favorites
pub async fn toggle_favorite(
    State(pool): State<PgPool>,
    Json(body): Json<FavoriteRequest>,
) -> Json<serde_json::Value> {
    let existing: Option<(String,)> =
        sqlx::query_as("SELECT url FROM favorites WHERE url = $1")
            .bind(&body.url)
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);

    let favorited = if existing.is_some() {
        sqlx::query("DELETE FROM favorites WHERE url = $1")
            .bind(&body.url)
            .execute(&pool)
            .await
            .ok();
        false
    } else {
        sqlx::query("INSERT INTO favorites (url) VALUES ($1)")
            .bind(&body.url)
            .execute(&pool)
            .await
            .ok();
        true
    };

    Json(serde_json::json!({
        "favorited": favorited,
        "url": body.url,
    }))
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
