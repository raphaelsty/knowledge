//! Buffer-based document ingestion scanner.
//!
//! Periodically scans a directory for JSON batch files written by the Python
//! pipeline, encodes them with ColBERT, updates the search index, and upserts
//! to PostgreSQL.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use crate::state::AppState;

#[cfg(feature = "model")]
use next_plaid::{filtering, IndexConfig, MmapIndex, UpdateConfig};

#[cfg(feature = "model")]
use serde_json::Value;

/// A document as written by the Python pipeline into a buffer JSON file.
#[derive(Debug, Deserialize)]
pub struct BufferDocument {
    pub url: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub extra_tags: Vec<String>,
}

#[cfg(feature = "model")]
const BATCH_SIZE: usize = 64;

/// Start the background buffer scanner task.
///
/// Spawns a tokio task that wakes every `interval_secs` seconds, scans
/// `buffer_dir` for `.json` files, and processes them.
pub fn start_buffer_scanner(
    state: Arc<AppState>,
    buffer_dir: String,
    interval_secs: u64,
) {
    let interval = Duration::from_secs(interval_secs);
    let dir = PathBuf::from(&buffer_dir);

    tracing::info!(
        buffer_dir = %buffer_dir,
        interval_secs,
        "buffer_scanner.starting"
    );

    tokio::spawn(async move {
        // Create directory if it doesn't exist
        if let Err(e) = tokio::fs::create_dir_all(&dir).await {
            tracing::error!(error = %e, "buffer_scanner.create_dir.failed");
            return;
        }

        loop {
            tokio::time::sleep(interval).await;

            if let Err(e) = scan_and_process(&state, &dir).await {
                tracing::error!(error = %e, "buffer_scanner.cycle.error");
            }
        }
    });
}

/// One scan cycle: recover stale files, then process all `.json` files.
async fn scan_and_process(
    state: &Arc<AppState>,
    dir: &Path,
) -> Result<(), String> {
    // 1. Recover any stale .processing files back to .json
    recover_stale_files(dir).await;

    // 2. Collect .json files sorted by name (oldest first)
    let mut files = collect_json_files(dir).await;
    if files.is_empty() {
        return Ok(());
    }
    files.sort();

    tracing::info!(count = files.len(), "buffer_scanner.files.found");

    // 3. Process each file
    for file in files {
        if let Err(e) = process_file(state, &file).await {
            tracing::error!(
                file = %file.display(),
                error = %e,
                "buffer_scanner.file.error"
            );
            // Rename back to .json on failure
            let json_path = file.with_extension("json");
            if file.extension().map(|e| e == "processing").unwrap_or(false) {
                let _ = tokio::fs::rename(&file, &json_path).await;
            }
        }
    }

    Ok(())
}

/// Rename any `.processing` files back to `.json` (crash recovery).
async fn recover_stale_files(dir: &Path) {
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(e) => e,
        Err(_) => return,
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().map(|e| e == "processing").unwrap_or(false) {
            let json_path = path.with_extension("json");
            tracing::warn!(file = %path.display(), "buffer_scanner.recovering_stale_file");
            let _ = tokio::fs::rename(&path, &json_path).await;
        }
    }
}

/// Collect all `.json` files in the directory.
async fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    let mut entries = match tokio::fs::read_dir(dir).await {
        Ok(e) => e,
        Err(_) => return result,
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            result.push(path);
        }
    }

    result
}

/// Process a single buffer file: rename → parse → encode → index → upsert → delete.
async fn process_file(
    state: &Arc<AppState>,
    json_path: &Path,
) -> Result<(), String> {
    // Rename to .processing for atomicity
    let processing_path = json_path.with_extension("processing");
    tokio::fs::rename(json_path, &processing_path)
        .await
        .map_err(|e| format!("rename to .processing: {e}"))?;

    // Read and parse
    let content = tokio::fs::read_to_string(&processing_path)
        .await
        .map_err(|e| format!("read file: {e}"))?;

    let docs: Vec<BufferDocument> =
        serde_json::from_str(&content).map_err(|e| format!("parse JSON: {e}"))?;

    if docs.is_empty() {
        let _ = tokio::fs::remove_file(&processing_path).await;
        return Ok(());
    }

    tracing::info!(
        file = %json_path.display(),
        count = docs.len(),
        "buffer_scanner.file.processing"
    );

    // Encode and index in a blocking task (docs moved into closure)
    let state2 = Arc::clone(state);
    let result = tokio::task::spawn_blocking(move || {
        process_documents_blocking(&state2, &docs)
    })
    .await
    .map_err(|e| format!("task join: {e}"))?;

    match result {
        Ok(count) => {
            // Upsert to PostgreSQL (re-parse from content since docs were moved)
            if let Some(pool) = state.pg_pool.as_ref() {
                if let Ok(pg_docs) = serde_json::from_str::<Vec<BufferDocument>>(&content) {
                    upsert_to_pg(pool, &pg_docs).await?;

                    // Rescue placement: insert new docs into folder tree
                    match super::rescue::rescue_into_tree(pool, &pg_docs).await {
                        Ok(n) if n > 0 => {
                            tracing::info!(count = n, "buffer_scanner.rescue.placed");
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!(error = %e, "buffer_scanner.rescue.failed");
                        }
                    }
                }
            }

            // Delete file on success
            let _ = tokio::fs::remove_file(&processing_path).await;
            tracing::info!(
                file = %json_path.display(),
                count,
                "buffer_scanner.file.processed"
            );
            Ok(())
        }
        Err(e) => {
            // Rename back on failure
            let _ = tokio::fs::rename(&processing_path, json_path).await;
            Err(e)
        }
    }
}

/// Blocking: encode documents with ColBERT, update index, create metadata.
#[cfg(feature = "model")]
fn process_documents_blocking(
    state: &Arc<AppState>,
    docs: &[BufferDocument],
) -> Result<usize, String> {
    use super::ingest::{build_document_text, build_metadata, build_model_from_config};

    let model_pool = state
        .model_pool
        .as_ref()
        .ok_or_else(|| "Model not loaded".to_string())?;

    // Build texts and metadata
    let mut texts: Vec<String> = Vec::with_capacity(docs.len());
    let mut metadata: Vec<Value> = Vec::with_capacity(docs.len());

    for doc in docs {
        let text = build_document_text(&doc.title, &doc.tags, &doc.extra_tags, &doc.summary);
        if text.is_empty() {
            continue;
        }
        texts.push(text);
        metadata.push(build_metadata(
            &doc.url,
            &doc.title,
            &doc.summary,
            &doc.date,
            &doc.tags,
            &doc.extra_tags,
        ));
    }

    if texts.is_empty() {
        return Ok(0);
    }

    // Build model
    let model = build_model_from_config(&model_pool.model_config)
        .map_err(|e| format!("model build: {e}"))?;

    // Encode in batches of BATCH_SIZE
    let mut all_embeddings = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH_SIZE) {
        let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let batch_embeddings = model
            .encode_documents(&refs, Some(2))
            .map_err(|e| format!("encoding: {e}"))?;
        all_embeddings.extend(batch_embeddings);
    }

    // Update index
    let index_name = "knowledge";
    let index_path = state.index_path(index_name).to_string_lossy().to_string();

    let (_index, doc_ids) = MmapIndex::update_or_create(
        &all_embeddings,
        &index_path,
        &IndexConfig {
            nbits: 2,
            ..Default::default()
        },
        &UpdateConfig::default(),
    )
    .map_err(|e| format!("index update: {e}"))?;

    // Update metadata store
    filtering::create(&index_path, &metadata, &doc_ids)
        .map_err(|e| format!("metadata: {e}"))?;

    // Reload index for live queries
    state
        .reload_index(index_name)
        .map_err(|e| format!("reload: {e}"))?;

    let count = doc_ids.len();
    tracing::info!(count, "buffer_scanner.indexed");
    Ok(count)
}

#[cfg(not(feature = "model"))]
fn process_documents_blocking(
    _state: &Arc<AppState>,
    _docs: &[BufferDocument],
) -> Result<usize, String> {
    Err("Model support not compiled".to_string())
}

/// Batch-upsert documents to PostgreSQL.
async fn upsert_to_pg(
    pool: &sqlx::PgPool,
    docs: &[BufferDocument],
) -> Result<(), String> {
    for doc in docs {
        let date_str: Option<&str> = if doc.date.is_empty() {
            None
        } else {
            Some(&doc.date)
        };

        sqlx::query(
            "INSERT INTO documents (url, title, summary, date, tags, extra_tags)
             VALUES ($1, $2, $3, $4::date, $5, $6)
             ON CONFLICT (url) DO UPDATE SET
                 title = EXCLUDED.title,
                 summary = EXCLUDED.summary,
                 date = EXCLUDED.date,
                 tags = EXCLUDED.tags,
                 extra_tags = EXCLUDED.extra_tags,
                 updated_at = now()",
        )
        .bind(&doc.url)
        .bind(&doc.title)
        .bind(&doc.summary)
        .bind(date_str)
        .bind(&doc.tags)
        .bind(&doc.extra_tags)
        .execute(pool)
        .await
        .map_err(|e| format!("DB upsert for {}: {e}", doc.url))?;
    }

    tracing::info!(count = docs.len(), "buffer_scanner.pg_upsert.complete");
    Ok(())
}
