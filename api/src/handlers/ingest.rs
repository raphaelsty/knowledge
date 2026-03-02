//! Bookmark ingestion handler.
//!
//! Reuses the model pool and index from AppState.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};
#[cfg(feature = "model")]
use serde_json::{json, Value};

#[cfg(feature = "model")]
use next_plaid::{filtering, IndexConfig, MmapIndex, UpdateConfig};

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct BookmarkRequest {
    pub url: String,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub date: String,
}

#[derive(Debug, Serialize)]
pub struct BookmarkResponse {
    pub status: String,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct IngestErrorResponse {
    pub error: String,
}

pub fn build_document_text(
    title: &str,
    tags: &[String],
    extra_tags: &[String],
    summary: &str,
) -> String {
    let tags_str = tags.join(" ");
    let extra_tags_str = extra_tags.join(" ");
    let summary_short: String = summary.chars().take(200).collect();
    format!(
        "{} {} {} {}",
        title, tags_str, extra_tags_str, summary_short
    )
    .trim()
    .to_string()
}

#[cfg(feature = "model")]
pub fn build_metadata(
    url: &str,
    title: &str,
    summary: &str,
    date: &str,
    tags: &[String],
    extra_tags: &[String],
) -> Value {
    json!({
        "url": url,
        "title": title,
        "summary": summary,
        "date": date,
        "tags": tags.join(","),
        "extra_tags": extra_tags.join(","),
    })
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<IngestErrorResponse>) {
    (
        status,
        Json(IngestErrorResponse {
            error: msg.to_string(),
        }),
    )
}

/// POST /api/bookmark
pub async fn ingest_bookmark(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BookmarkRequest>,
) -> Result<Json<BookmarkResponse>, (StatusCode, Json<IngestErrorResponse>)> {
    if req.url.is_empty() || req.title.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "url and title are required"));
    }

    let pool = state
        .pg_pool
        .as_ref()
        .ok_or_else(|| err(StatusCode::SERVICE_UNAVAILABLE, "Database not configured"))?;

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
    .execute(pool)
    .await
    .map_err(|e| {
        tracing::error!("DB upsert error: {e}");
        err(StatusCode::INTERNAL_SERVER_ERROR, "Database error")
    })?;

    // 2. Build document text
    let text = build_document_text(&req.title, &req.tags, &[], &req.summary);
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
        // Use the model pool's encode_texts for encoding
        #[cfg(feature = "model")]
        {
            let model_pool = state2
                .model_pool
                .as_ref()
                .ok_or_else(|| err(StatusCode::SERVICE_UNAVAILABLE, "Model not loaded"))?;

            // Build a model from config for this blocking operation
            let model = build_model_from_config(&model_pool.model_config).map_err(|e| {
                tracing::error!("Model build error: {e}");
                err(StatusCode::INTERNAL_SERVER_ERROR, "Model error")
            })?;

            let embeddings = model
                .encode_documents(&[text.as_str()], Some(2))
                .map_err(|e| {
                    tracing::error!("Embedding error: {e}");
                    err(StatusCode::INTERNAL_SERVER_ERROR, "Embedding error")
                })?;

            // Build metadata
            let metadata = vec![build_metadata(&url, &title, &summary, &date, &tags, &[])];

            // Get the index path from state
            let index_name = "knowledge";
            let index_path = state2.index_path(index_name).to_string_lossy().to_string();

            // Update index
            let (_index, doc_ids) = MmapIndex::update_or_create(
                &embeddings,
                &index_path,
                &IndexConfig {
                    nbits: 2,
                    ..Default::default()
                },
                &UpdateConfig::default(),
            )
            .map_err(|e| {
                tracing::error!("Index error: {e}");
                err(StatusCode::INTERNAL_SERVER_ERROR, "Indexing error")
            })?;

            // Update metadata store (append if exists, create if new)
            if filtering::exists(&index_path) {
                filtering::update(&index_path, &metadata, &doc_ids).map_err(|e| {
                    tracing::error!("Metadata update error: {e}");
                    err(StatusCode::INTERNAL_SERVER_ERROR, "Metadata store error")
                })?;
            } else {
                filtering::create(&index_path, &metadata, &doc_ids).map_err(|e| {
                    tracing::error!("Metadata create error: {e}");
                    err(StatusCode::INTERNAL_SERVER_ERROR, "Metadata store error")
                })?;
            }

            // Reload index in AppState
            state2.reload_index(index_name).map_err(|e| {
                tracing::error!("Index reload error: {e}");
                err(StatusCode::INTERNAL_SERVER_ERROR, "Index reload error")
            })?;

            tracing::info!("Indexed bookmark: {url} — {title}");
            Ok(url)
        }

        #[cfg(not(feature = "model"))]
        {
            // Suppress unused variable warnings
            let _ = (text, url, title, summary, date, tags, state2);
            Err(err(
                StatusCode::SERVICE_UNAVAILABLE,
                "Model support not compiled",
            ))
        }
    })
    .await
    .map_err(|e| {
        tracing::error!("Task join error: {e}");
        err(StatusCode::INTERNAL_SERVER_ERROR, "Internal error")
    })?
    .map(|url| {
        Json(BookmarkResponse {
            status: "ok".to_string(),
            url,
        })
    })
}

/// Build a Colbert model from configuration (used for ingest encoding).
#[cfg(feature = "model")]
pub fn build_model_from_config(
    config: &crate::state::ModelConfig,
) -> Result<next_plaid_onnx::Colbert, String> {
    let execution_provider = if config.use_cuda {
        next_plaid_onnx::ExecutionProvider::Cuda
    } else {
        next_plaid_onnx::ExecutionProvider::Cpu
    };

    let mut builder = next_plaid_onnx::Colbert::builder(&config.path)
        .with_execution_provider(execution_provider)
        .with_quantized(config.use_int8);

    if let Some(parallel) = config.parallel_sessions {
        builder = builder.with_parallel(parallel);
    }
    if let Some(batch_size) = config.batch_size {
        builder = builder.with_batch_size(batch_size);
    }
    if let Some(threads) = config.threads {
        builder = builder.with_threads(threads);
    }
    if let Some(query_length) = config.query_length {
        builder = builder.with_query_length(query_length);
    }
    if let Some(document_length) = config.document_length {
        builder = builder.with_document_length(document_length);
    }

    builder.build().map_err(|e| e.to_string())
}
