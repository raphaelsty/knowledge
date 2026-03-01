use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use next_plaid::{filtering, IndexConfig, MmapIndex, UpdateConfig};
use next_plaid_onnx::Colbert;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::PgPool;
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

struct AppState {
    pool: PgPool,
    model: Mutex<Colbert>,
    index_path: String,
}

#[derive(Debug, Deserialize)]
struct BookmarkRequest {
    url: String,
    title: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    date: String,
}

#[derive(Debug, Serialize)]
struct BookmarkResponse {
    status: String,
    url: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
}

fn build_document_text(title: &str, tags: &[String], summary: &str) -> String {
    let tags_str = tags.join(" ");
    let summary_short: String = summary.chars().take(200).collect();
    format!("{} {} {}", title, tags_str, summary_short)
        .trim()
        .to_string()
}

fn build_metadata(url: &str, title: &str, summary: &str, date: &str, tags: &[String]) -> Value {
    json!({
        "url": url,
        "title": title,
        "summary": summary,
        "date": date,
        "tags": tags.join(","),
        "extra_tags": "",
    })
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn ingest_bookmark(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BookmarkRequest>,
) -> Result<Json<BookmarkResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.url.is_empty() || req.title.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "url and title are required"));
    }

    // Date as Option<&str> for SQL casting (NULL if empty)
    let date_str: Option<&str> = if req.date.is_empty() {
        None
    } else {
        Some(&req.date)
    };

    // 1. Upsert into PostgreSQL
    sqlx::query(
        "INSERT INTO documents (url, title, summary, date, tags, extra_tags)
         VALUES ($1, $2, $3, $4::date, $5, '{}')
         ON CONFLICT (url) DO UPDATE SET
             title = EXCLUDED.title,
             summary = EXCLUDED.summary,
             date = EXCLUDED.date,
             tags = EXCLUDED.tags,
             updated_at = now()",
    )
    .bind(&req.url)
    .bind(&req.title)
    .bind(&req.summary)
    .bind(date_str)
    .bind(&req.tags)
    .execute(&state.pool)
    .await
    .map_err(|e| {
        eprintln!("DB upsert error: {e}");
        err(StatusCode::INTERNAL_SERVER_ERROR, "Database error")
    })?;

    // 2. Build document text (same logic as embeddings crate)
    let text = build_document_text(&req.title, &req.tags, &req.summary);
    if text.is_empty() {
        return Ok(Json(BookmarkResponse {
            status: "ok".to_string(),
            url: req.url,
        }));
    }

    // 3. Embed + index in a blocking task (CPU-intensive)
    let url = req.url.clone();
    let title = req.title.clone();
    let summary = req.summary.clone();
    let date = req.date.clone();
    let tags = req.tags.clone();
    let state2 = Arc::clone(&state);

    tokio::task::spawn_blocking(move || {
        // Encode with ColBERT model
        let model = state2.model.lock().unwrap();
        let embeddings = model
            .encode_documents(&[text.as_str()], Some(2))
            .map_err(|e| {
                eprintln!("Embedding error: {e}");
                err(StatusCode::INTERNAL_SERVER_ERROR, "Embedding error")
            })?;
        drop(model);

        // Build metadata
        let metadata = vec![build_metadata(&url, &title, &summary, &date, &tags)];

        // Update index
        let index_path = &state2.index_path;
        let (_index, doc_ids) = MmapIndex::update_or_create(
            &embeddings,
            index_path,
            &IndexConfig {
                nbits: 2,
                ..Default::default()
            },
            &UpdateConfig::default(),
        )
        .map_err(|e| {
            eprintln!("Index error: {e}");
            err(StatusCode::INTERNAL_SERVER_ERROR, "Indexing error")
        })?;

        // Update metadata store
        filtering::create(index_path, &metadata, &doc_ids).map_err(|e| {
            eprintln!("Metadata error: {e}");
            err(StatusCode::INTERNAL_SERVER_ERROR, "Metadata store error")
        })?;

        println!("Indexed bookmark: {url} — {title}");
        Ok(url)
    })
    .await
    .map_err(|e| {
        eprintln!("Task join error: {e}");
        err(StatusCode::INTERNAL_SERVER_ERROR, "Internal error")
    })?
    .map(|url| {
        Json(BookmarkResponse {
            status: "ok".to_string(),
            url,
        })
    })
}

#[tokio::main]
async fn main() {
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/answerai-colbert-small-v1-onnx".to_string());
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| "multi-vector-database/knowledge".to_string());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "3003".to_string())
        .parse()
        .expect("PORT must be a valid u16");

    let pool = PgPool::connect(&database_url)
        .await
        .expect("Failed to connect to PostgreSQL");

    println!("Loading ColBERT model from {model_path}...");
    let model = Colbert::builder(&model_path)
        .with_quantized(true)
        .build()
        .expect("Failed to load ColBERT model");
    println!("Model loaded.");

    let state = Arc::new(AppState {
        pool,
        model: Mutex::new(model),
        index_path,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/bookmark", post(ingest_bookmark))
        .route("/health", get(health))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("Failed to bind");
    println!("Ingest API listening on :{port}");
    axum::serve(listener, app).await.expect("Server error");
}
