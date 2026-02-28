mod purge;
mod stats;

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

const MAX_BATCH_SIZE: usize = 100;
const MAX_PAYLOAD_BYTES: usize = 8192;
const RETENTION_DAYS: i32 = 90;

const ALLOWED_EVENT_TYPES: &[&str] = &["search", "click", "folder_browse", "filter_apply", "page_view"];

#[derive(Debug, Deserialize)]
struct EventInput {
    session_id: String,
    event_type: String,
    payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Debug, Serialize)]
struct IngestResponse {
    inserted: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

pub struct AppState {
    pub pool: PgPool,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn ingest_events(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<IngestResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Parse JSON from body directly — accepts both application/json and text/plain
    // (sendBeacon uses text/plain to avoid CORS preflight)
    let events: Vec<EventInput> = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Invalid JSON: {e}"),
            }),
        )
    })?;
    if events.is_empty() {
        return Ok(Json(IngestResponse { inserted: 0 }));
    }

    if events.len() > MAX_BATCH_SIZE {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Batch too large: {} events (max {})", events.len(), MAX_BATCH_SIZE),
            }),
        ));
    }

    // Validate all events before inserting
    for event in &events {
        if !ALLOWED_EVENT_TYPES.contains(&event.event_type.as_str()) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Unknown event_type: {}", event.event_type),
                }),
            ));
        }

        if event.session_id.is_empty() || event.session_id.len() > 64 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Invalid session_id".to_string(),
                }),
            ));
        }

        let payload_size = event.payload.to_string().len();
        if payload_size > MAX_PAYLOAD_BYTES {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Payload too large: {} bytes (max {})",
                        payload_size, MAX_PAYLOAD_BYTES
                    ),
                }),
            ));
        }
    }

    // Batch insert
    let mut inserted = 0;
    for event in &events {
        sqlx::query(
            "INSERT INTO events (session_id, event_type, payload) VALUES ($1, $2, $3)",
        )
        .bind(&event.session_id)
        .bind(&event.event_type)
        .bind(&event.payload)
        .execute(&state.pool)
        .await
        .map_err(|e| {
            eprintln!("DB insert error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Database error".to_string(),
                }),
            )
        })?;
        inserted += 1;
    }

    Ok(Json(IngestResponse { inserted }))
}

async fn run_migrations(pool: &PgPool) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS events (
            id          BIGSERIAL PRIMARY KEY,
            session_id  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            payload     JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )",
    )
    .execute(pool)
    .await
    .expect("Failed to create events table");

    // Create indices (IF NOT EXISTS for idempotency)
    for stmt in &[
        "CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type)",
        "CREATE INDEX IF NOT EXISTS idx_events_created ON events (created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id)",
    ] {
        sqlx::query(stmt)
            .execute(pool)
            .await
            .expect("Failed to create index");
    }
}

#[tokio::main]
async fn main() {
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "3002".to_string())
        .parse()
        .expect("PORT must be a valid u16");

    let pool = PgPool::connect(&database_url)
        .await
        .expect("Failed to connect to PostgreSQL");

    // Run migrations on startup
    run_migrations(&pool).await;

    // Purge old events on startup
    match purge::purge_old_events(&pool, RETENTION_DAYS).await {
        Ok(count) => {
            if count > 0 {
                println!("Purged {count} events older than {RETENTION_DAYS} days");
            }
        }
        Err(e) => eprintln!("Purge error: {e}"),
    }

    let state = Arc::new(AppState { pool });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/events", post(ingest_events))
        .route("/health", get(health))
        .route("/stats/overview", get(stats::overview))
        .route("/stats/activity", get(stats::activity))
        .route("/stats/top-queries", get(stats::top_queries))
        .route("/stats/top-clicks", get(stats::top_clicks))
        .route("/stats/sources", get(stats::sources))
        .route("/stats/folders", get(stats::folders))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("Failed to bind");
    println!("Events API listening on :{port}");
    axum::serve(listener, app).await.expect("Server error");
}
