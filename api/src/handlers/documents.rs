//! Document and index management handlers.
//!
//! Handles index creation, document upload, and deletion.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use ndarray::Array2;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::task;
use tokio::time::Instant;

use next_plaid::{filtering, IndexConfig, Metadata, MmapIndex, UpdateConfig};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_texts_internal;
use crate::models::{
    AddDocumentsRequest, CreateIndexRequest, CreateIndexResponse, DeleteDocumentsRequest,
    DeleteIndexResponse, DocumentEmbeddings, ErrorResponse, IndexConfigStored, IndexInfoResponse,
    InputType, UpdateIndexConfigRequest, UpdateIndexConfigResponse, UpdateIndexRequest,
    UpdateWithEncodingRequest,
};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;

// --- Concurrency Control ---

/// Get the maximum number of queued background tasks per index.
/// Configurable via MAX_QUEUED_TASKS_PER_INDEX env var (default: 10).
/// When exceeded, new requests get 503 Service Unavailable.
fn max_queued_tasks_per_index() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MAX_QUEUED_TASKS_PER_INDEX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10)
    })
}

// --- Index/DB Sync Repair ---

/// Global registry of per-index repair locks to prevent concurrent repair operations.
/// This ensures only one repair operation runs at a time for a given index,
/// preventing race conditions where multiple requests detect a mismatch and
/// attempt conflicting repairs simultaneously.
static REPAIR_LOCKS: OnceLock<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<()>>>>> =
    OnceLock::new();

/// Get or create a repair lock for the given index path.
/// Used to serialize repair operations on a specific index.
fn get_repair_lock(index_path: &str) -> Arc<std::sync::Mutex<()>> {
    let locks = REPAIR_LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = locks
        .lock()
        .expect("REPAIR_LOCKS mutex poisoned - a thread panicked while holding this lock");
    map.entry(index_path.to_string())
        .or_insert_with(|| Arc::new(std::sync::Mutex::new(())))
        .clone()
}

/// Automatically repair sync issues between the vector index and metadata DB.
///
/// This function handles two cases:
/// 1. DB has more records than index: Delete extra DB records (IDs >= index count)
/// 2. Index has more documents than DB: Delete extra documents from index (IDs >= DB count)
///
/// Returns Ok(true) if repair was performed, Ok(false) if no repair needed.
///
/// Thread-safety: Uses a per-index lock to prevent concurrent repair operations.
fn repair_index_db_sync(index_path: &str) -> Result<bool, String> {
    // Acquire per-index repair lock to prevent concurrent repairs
    let repair_lock = get_repair_lock(index_path);
    let _guard = repair_lock
        .lock()
        .map_err(|_| "Repair lock poisoned - a previous repair operation panicked".to_string())?;

    let path = std::path::Path::new(index_path);

    // Check if both exist
    if !path.join("metadata.json").exists() {
        return Ok(false); // No index yet
    }
    if !filtering::exists(index_path) {
        return Ok(false); // No DB yet
    }

    let index_metadata = Metadata::load_from_path(path)
        .map_err(|e| format!("Failed to load index metadata: {}", e))?;
    let db_count =
        filtering::count(index_path).map_err(|e| format!("Failed to get DB count: {}", e))?;

    let index_count = index_metadata.num_documents;

    if index_count == db_count {
        return Ok(false); // Already in sync
    }

    tracing::warn!(
        index = %index_path,
        index_count = index_count,
        db_count = db_count,
        "index.sync.mismatch"
    );

    let repair_start = std::time::Instant::now();
    let extra_count: usize;

    if db_count > index_count {
        // DB has extra records - delete them
        // Extra records have IDs >= index_count
        let extra_ids: Vec<i64> = (index_count as i64..db_count as i64).collect();
        extra_count = extra_ids.len();
        filtering::delete(index_path, &extra_ids)
            .map_err(|e| format!("Failed to delete extra DB records: {}", e))?;
    } else {
        // Index has extra documents - delete them from index
        // Extra documents have IDs >= db_count
        let extra_ids: Vec<i64> = (db_count as i64..index_count as i64).collect();
        extra_count = extra_ids.len();
        let mut index = MmapIndex::load(index_path)
            .map_err(|e| format!("Failed to load index for repair: {}", e))?;
        // Use delete_with_options with delete_metadata=false since DB doesn't have these
        index
            .delete_with_options(&extra_ids, false)
            .map_err(|e| format!("Failed to delete extra index documents: {}", e))?;
    }

    // Verify repair succeeded
    let new_index_metadata = Metadata::load_from_path(path)
        .map_err(|e| format!("Failed to reload index metadata after repair: {}", e))?;
    let new_db_count = filtering::count(index_path)
        .map_err(|e| format!("Failed to get DB count after repair: {}", e))?;

    if new_index_metadata.num_documents != new_db_count {
        return Err(format!(
            "Repair failed: index still has {} documents but DB has {} records",
            new_index_metadata.num_documents, new_db_count
        ));
    }

    let repair_ms = repair_start.elapsed().as_millis() as u64;

    tracing::info!(
        index = %index_path,
        deleted_count = extra_count,
        final_count = new_db_count,
        repair_ms = repair_ms,
        "index.repair.complete"
    );

    Ok(true)
}

// --- Batch Collection ---

/// Get the maximum number of documents to batch together before processing.
/// Configurable via MAX_BATCH_DOCUMENTS env var (default: 300).
fn max_batch_documents() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MAX_BATCH_DOCUMENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300)
    })
}

/// Maximum time to wait for more documents before processing a batch.
const BATCH_TIMEOUT: Duration = Duration::from_millis(100);

/// Get the channel buffer size for batch queue.
/// Configurable via BATCH_CHANNEL_SIZE env var (default: 100).
fn batch_channel_size() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("BATCH_CHANNEL_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100)
    })
}

/// A single item in the batch queue, representing one update request.
struct BatchItem {
    embeddings: Vec<Array2<f32>>,
    metadata: Vec<serde_json::Value>,
}

/// Handle to a batch queue for an index.
struct BatchQueue {
    sender: mpsc::Sender<BatchItem>,
}

/// Global registry of batch queues per index.
static BATCH_QUEUES: OnceLock<std::sync::Mutex<HashMap<String, BatchQueue>>> = OnceLock::new();

/// Global registry to manage locks per index name.
/// We use tokio::sync::Mutex to allow tasks to wait asynchronously without blocking threads.
static INDEX_LOCKS: OnceLock<std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();

/// Global registry to manage semaphores per index name.
/// Limits the number of queued background tasks to prevent resource exhaustion.
static INDEX_SEMAPHORES: OnceLock<std::sync::Mutex<HashMap<String, Arc<Semaphore>>>> =
    OnceLock::new();

/// Helper to get (or create) an async mutex for a specific index.
/// Used to serialize updates to a specific index.
/// Uses the index name as key (assumes unique names within a single server instance).
pub fn get_index_lock(name: &str) -> Arc<Mutex<()>> {
    let locks: &std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>> =
        INDEX_LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = locks
        .lock()
        .expect("INDEX_LOCKS mutex poisoned - a thread panicked while holding this lock");
    map.entry(name.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Helper to get (or create) an async mutex for a specific index path.
/// Used when full path isolation is needed (e.g., in tests with separate temp directories).
pub fn get_index_lock_by_path(path: &str) -> Arc<Mutex<()>> {
    let locks: &std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>> =
        INDEX_LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = locks
        .lock()
        .expect("INDEX_LOCKS mutex poisoned - a thread panicked while holding this lock");
    map.entry(path.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Helper to get (or create) a semaphore for a specific index name.
/// The semaphore limits queued background tasks to prevent unbounded growth.
fn get_index_semaphore(name: &str) -> Arc<Semaphore> {
    let sems: &std::sync::Mutex<HashMap<String, Arc<Semaphore>>> =
        INDEX_SEMAPHORES.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = sems
        .lock()
        .expect("INDEX_SEMAPHORES mutex poisoned - a thread panicked while holding this lock");
    map.entry(name.to_string())
        .or_insert_with(|| Arc::new(Semaphore::new(max_queued_tasks_per_index())))
        .clone()
}

/// Get or create a batch queue for the given index.
/// Spawns a batch worker if the queue doesn't exist yet.
/// Uses full index path as key to ensure isolation between different data directories.
fn get_or_create_batch_queue(name: &str, state: Arc<AppState>) -> mpsc::Sender<BatchItem> {
    // Use full path as key to ensure isolation between different data directories
    // (important for tests running in parallel with separate temp directories)
    let queue_key = state.index_path(name).to_string_lossy().to_string();

    let queues: &std::sync::Mutex<HashMap<String, BatchQueue>> =
        BATCH_QUEUES.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = queues
        .lock()
        .expect("BATCH_QUEUES mutex poisoned - a thread panicked while holding this lock");

    if let Some(queue) = map.get(&queue_key) {
        return queue.sender.clone();
    }

    // Create new channel and spawn worker
    let (sender, receiver) = mpsc::channel(batch_channel_size());
    let queue = BatchQueue {
        sender: sender.clone(),
    };
    map.insert(queue_key, queue);

    // Spawn the batch worker
    let index_name = name.to_string();
    tokio::spawn(batch_worker(receiver, index_name, state));

    sender
}

/// Background worker that collects batch items and processes them together.
///
/// The worker waits for items on the channel and batches them until either:
/// - The total document count reaches MAX_BATCH_DOCUMENTS, or
/// - BATCH_TIMEOUT has elapsed since the first item arrived
async fn batch_worker(
    mut receiver: mpsc::Receiver<BatchItem>,
    index_name: String,
    state: Arc<AppState>,
) {
    tracing::info!(index = %index_name, "update.worker.started");

    loop {
        // Wait for the first item (blocking)
        let first_item = match receiver.recv().await {
            Some(item) => item,
            None => {
                tracing::debug!(index = %index_name, "update.worker.stopped");
                break;
            }
        };

        // Start collecting batch
        let mut batch_embeddings: Vec<Array2<f32>> = first_item.embeddings;
        let mut batch_metadata: Vec<serde_json::Value> = first_item.metadata;
        let mut doc_count = batch_embeddings.len();
        let deadline = Instant::now() + BATCH_TIMEOUT;

        // Collect more items until timeout or max batch size
        while doc_count < max_batch_documents() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(item)) => {
                    let item_docs = item.embeddings.len();
                    batch_embeddings.extend(item.embeddings);
                    batch_metadata.extend(item.metadata);
                    doc_count += item_docs;
                }
                Ok(None) => {
                    // Channel closed - process remaining batch before exiting
                    if !batch_embeddings.is_empty() {
                        process_batch(&index_name, batch_embeddings, batch_metadata, &state).await;
                    }
                    tracing::debug!(index = %index_name, "update.worker.stopped");
                    return;
                }
                Err(_) => {
                    // Timeout reached
                    break;
                }
            }
        }

        // Process the collected batch
        if !batch_embeddings.is_empty() {
            process_batch(&index_name, batch_embeddings, batch_metadata, &state).await;
        }
    }
}

/// Metrics returned from the blocking batch processing.
struct BatchMetrics {
    index_update_ms: u64,
    metadata_update_ms: u64,
    first_doc_id: Option<i64>,
    last_doc_id: Option<i64>,
    evicted_count: usize,
}

/// Process a batch of documents for the given index.
async fn process_batch(
    index_name: &str,
    embeddings: Vec<Array2<f32>>,
    metadata: Vec<serde_json::Value>,
    state: &Arc<AppState>,
) {
    let doc_count = embeddings.len();
    let start = std::time::Instant::now();

    let name_inner = index_name.to_string();
    let state_clone = state.clone();
    let path_str = state.index_path(index_name).to_string_lossy().to_string();

    // Acquire per-index lock using full path for isolation
    let lock = get_index_lock_by_path(&path_str);
    let _guard = lock.lock().await;

    // Run heavy work in blocking thread
    let result = task::spawn_blocking(move || -> Result<BatchMetrics, String> {
        // Load stored config
        let config_path = state_clone.index_path(&name_inner).join("config.json");
        let config_file = std::fs::File::open(&config_path)
            .map_err(|e| format!("Failed to open config: {}", e))?;
        let stored_config: IndexConfigStored = serde_json::from_reader(config_file)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        // Check and automatically repair sync issues between index and DB
        if let Err(e) = repair_index_db_sync(&path_str) {
            return Err(format!("Index/DB sync repair failed: {}", e));
        }

        // Build IndexConfig
        let index_config = IndexConfig {
            nbits: stored_config.nbits,
            batch_size: stored_config.batch_size,
            seed: stored_config.seed,
            start_from_scratch: stored_config.start_from_scratch,
            ..Default::default()
        };
        let update_config = UpdateConfig::default();

        // STEP 1: Update vector index FIRST
        let index_update_start = std::time::Instant::now();
        let index_result =
            MmapIndex::update_or_create(&embeddings, &path_str, &index_config, &update_config);

        let (mut index, doc_ids) = match index_result {
            Ok((idx, ids)) => (idx, ids),
            Err(e) => {
                return Err(format!("Index update failed: {}", e));
            }
        };
        let index_update_ms = index_update_start.elapsed().as_millis() as u64;

        let first_doc_id = doc_ids.first().copied();
        let last_doc_id = doc_ids.last().copied();

        // STEP 2: Update metadata DB using the ACTUAL doc_ids from the index
        let metadata_update_start = std::time::Instant::now();
        let db_existed = filtering::exists(&path_str);
        let db_result = if db_existed {
            filtering::update(&path_str, &metadata, &doc_ids)
        } else {
            filtering::create(&path_str, &metadata, &doc_ids)
        };

        if let Err(e) = db_result {
            // ROLLBACK: Remove the documents we just added to the index
            if let Err(rollback_err) = index.delete_with_options(&doc_ids, false) {
                tracing::error!(
                    index = %name_inner,
                    error = %rollback_err,
                    operation = "rollback",
                    "update.rollback.failed"
                );
            }
            return Err(format!("Failed to update metadata: {}", e));
        }
        let metadata_update_ms = metadata_update_start.elapsed().as_millis() as u64;

        // Eviction: Check if over max_documents limit
        let evicted_count = if let Some(max_docs) = stored_config.max_documents {
            match evict_oldest_documents(&mut index, max_docs) {
                Ok(count) => count,
                Err(e) => {
                    tracing::warn!(
                        index = %name_inner,
                        error = %e,
                        max_documents = max_docs,
                        "update.eviction.failed"
                    );
                    0
                }
            }
        } else {
            0
        };

        // Reload State
        state_clone.unload_index(&name_inner);
        let idx = MmapIndex::load(&path_str).map_err(|e| format!("Failed to load index: {}", e))?;
        state_clone.register_index(&name_inner, idx);

        Ok(BatchMetrics {
            index_update_ms,
            metadata_update_ms,
            first_doc_id,
            last_doc_id,
            evicted_count,
        })
    })
    .await;

    let total_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(Ok(metrics)) => {
            tracing::info!(
                index = %index_name,
                num_documents = doc_count,
                first_doc_id = ?metrics.first_doc_id,
                last_doc_id = ?metrics.last_doc_id,
                index_update_ms = metrics.index_update_ms,
                metadata_update_ms = metrics.metadata_update_ms,
                evicted_count = metrics.evicted_count,
                total_ms = total_ms,
                "update.batch.complete"
            );

            // Warn on slow batch processing (>5s)
            if total_ms > 5000 {
                tracing::warn!(
                    index = %index_name,
                    num_documents = doc_count,
                    total_ms = total_ms,
                    "update.batch.slow"
                );
            }
        }
        Ok(Err(e)) => {
            tracing::error!(
                index = %index_name,
                num_documents = doc_count,
                error = %e,
                total_ms = total_ms,
                "update.batch.failed"
            );
        }
        Err(e) => {
            tracing::error!(
                index = %index_name,
                num_documents = doc_count,
                error = %e,
                total_ms = total_ms,
                "update.batch.panicked"
            );
        }
    }
}

// ---------------------------

/// Convert document embeddings from JSON format to ndarray.
fn to_ndarray(doc: &DocumentEmbeddings) -> ApiResult<Array2<f32>> {
    let rows = doc.embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty embeddings".to_string()));
    }

    let cols = doc.embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension embeddings".to_string(),
        ));
    }

    // Verify all rows have the same dimension
    for (i, row) in doc.embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent embedding dimension at row {}: expected {}, got {}",
                i,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = doc.embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| ApiError::BadRequest(format!("Failed to create array: {}", e)))
}

/// Evict oldest documents if index exceeds max_documents limit.
/// Returns the number of documents evicted.
fn evict_oldest_documents(index: &mut MmapIndex, max_documents: usize) -> ApiResult<usize> {
    let current_count = index.metadata.num_documents;

    if current_count <= max_documents {
        return Ok(0);
    }

    let num_to_evict = current_count - max_documents;
    // Oldest documents have the lowest IDs (0, 1, 2, ...)
    let ids_to_delete: Vec<i64> = (0..num_to_evict as i64).collect();

    let deleted = index.delete(&ids_to_delete)?;
    index.reload()?;

    tracing::debug!(
        current_count = current_count,
        max_documents = max_documents,
        evicted = deleted,
        remaining = index.metadata.num_documents,
        "index.eviction.complete"
    );

    Ok(deleted)
}

// =============================================================================
// Delete Batching Infrastructure
// =============================================================================

/// Minimum wait time after receiving first delete condition before processing.
/// Configurable via DELETE_BATCH_MIN_WAIT env var (default: 500ms).
/// This allows time for concurrent delete requests to accumulate before processing.
fn delete_batch_min_wait() -> Duration {
    static VALUE: OnceLock<Duration> = OnceLock::new();
    *VALUE.get_or_init(|| {
        Duration::from_millis(
            std::env::var("DELETE_BATCH_MIN_WAIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
        )
    })
}

/// Maximum time to wait for more delete conditions before processing.
/// Configurable via DELETE_BATCH_MAX_WAIT env var (default: 2000ms).
/// Higher values allow better batching for clients with slow HTTP connections.
fn delete_batch_max_wait() -> Duration {
    static VALUE: OnceLock<Duration> = OnceLock::new();
    *VALUE.get_or_init(|| {
        Duration::from_millis(
            std::env::var("DELETE_BATCH_MAX_WAIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2000),
        )
    })
}

/// Maximum number of delete conditions to batch together.
/// Configurable via MAX_DELETE_BATCH_CONDITIONS env var (default: 200).
/// Higher values allow processing more deletes in a single batch.
fn max_delete_batch_conditions() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MAX_DELETE_BATCH_CONDITIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200)
    })
}

/// A single delete condition to be batched.
struct DeleteBatchItem {
    condition: String,
    parameters: Vec<serde_json::Value>,
}

/// Handle to a delete batch queue for an index.
struct DeleteBatchQueue {
    sender: mpsc::Sender<DeleteBatchItem>,
}

/// Global registry of delete batch queues per index.
static DELETE_BATCH_QUEUES: OnceLock<std::sync::Mutex<HashMap<String, DeleteBatchQueue>>> =
    OnceLock::new();

/// Get or create a delete batch queue for the given index.
/// Uses full index path as key to ensure isolation between different data directories.
fn get_or_create_delete_batch_queue(
    name: &str,
    state: Arc<AppState>,
) -> mpsc::Sender<DeleteBatchItem> {
    // Use full path as key to ensure isolation between different data directories
    // (important for tests running in parallel with separate temp directories)
    let queue_key = state.index_path(name).to_string_lossy().to_string();

    let queues: &std::sync::Mutex<HashMap<String, DeleteBatchQueue>> =
        DELETE_BATCH_QUEUES.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = queues.lock().expect("DELETE_BATCH_QUEUES mutex poisoned");

    if let Some(queue) = map.get(&queue_key) {
        return queue.sender.clone();
    }

    // Create new channel and spawn worker
    let (sender, receiver) = mpsc::channel(batch_channel_size());
    let queue = DeleteBatchQueue {
        sender: sender.clone(),
    };
    map.insert(queue_key, queue);

    // Spawn the delete batch worker
    let index_name = name.to_string();
    tokio::spawn(delete_batch_worker(receiver, index_name, state));

    sender
}

/// Background worker that collects delete conditions and processes them together.
///
/// The worker waits for conditions and batches them:
/// - Waits at least DELETE_BATCH_MIN_WAIT after first condition
/// - Then processes when batch reaches MAX_DELETE_BATCH_CONDITIONS, or
/// - DELETE_BATCH_MAX_WAIT has elapsed
///
/// Conditions are resolved to IDs at delete time (inside the lock) to handle
/// the ID shifting that occurs when documents are deleted.
async fn delete_batch_worker(
    mut receiver: mpsc::Receiver<DeleteBatchItem>,
    index_name: String,
    state: Arc<AppState>,
) {
    tracing::info!(index = %index_name, "delete.worker.started");

    loop {
        // Wait for the first item (blocking)
        let first_item = match receiver.recv().await {
            Some(item) => item,
            None => {
                tracing::debug!(index = %index_name, "delete.worker.stopped");
                break;
            }
        };

        // Start collecting batch
        let mut batch: Vec<DeleteBatchItem> = vec![first_item];
        let min_deadline = Instant::now() + delete_batch_min_wait();
        let max_deadline = Instant::now() + delete_batch_max_wait();

        // First phase: wait at least min_wait to allow batching
        while Instant::now() < min_deadline && batch.len() < max_delete_batch_conditions() {
            let remaining = min_deadline.saturating_duration_since(Instant::now());
            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(item)) => batch.push(item),
                Ok(None) => {
                    // Channel closed - process remaining batch
                    if !batch.is_empty() {
                        process_delete_batch(&index_name, batch, &state).await;
                    }
                    tracing::debug!(index = %index_name, "delete.worker.stopped");
                    return;
                }
                Err(_) => break, // Timeout - proceed to second phase
            }
        }

        // Second phase: continue collecting until max_wait or max conditions
        while batch.len() < max_delete_batch_conditions() {
            let remaining = max_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, receiver.recv()).await {
                Ok(Some(item)) => batch.push(item),
                Ok(None) => {
                    // Channel closed - process remaining batch
                    if !batch.is_empty() {
                        process_delete_batch(&index_name, batch, &state).await;
                    }
                    tracing::debug!(index = %index_name, "delete.worker.stopped");
                    return;
                }
                Err(_) => break, // Timeout
            }
        }

        // Process the collected batch
        process_delete_batch(&index_name, batch, &state).await;
    }
}

/// Process a batch of delete conditions for the given index.
///
/// Conditions are resolved to IDs at delete time (inside the lock) to ensure
/// we get the correct IDs after any previous deletions have completed.
async fn process_delete_batch(
    index_name: &str,
    conditions: Vec<DeleteBatchItem>,
    state: &Arc<AppState>,
) {
    let num_conditions = conditions.len();
    let start = std::time::Instant::now();

    tracing::info!(
        index = %index_name,
        num_conditions = num_conditions,
        "delete.batch.processing"
    );

    let name_inner = index_name.to_string();
    let state_clone = state.clone();
    let path_str = state.index_path(index_name).to_string_lossy().to_string();

    // Acquire per-index lock using full path for isolation
    let lock = get_index_lock_by_path(&path_str);
    let _guard = lock.lock().await;

    // Run in blocking thread
    let result = task::spawn_blocking(move || -> Result<(usize, usize), String> {
        // Check and repair sync issues
        if let Err(e) = repair_index_db_sync(&path_str) {
            return Err(format!("Index/DB sync repair failed: {}", e));
        }

        let mut index =
            MmapIndex::load(&path_str).map_err(|e| format!("Failed to load index: {}", e))?;

        let mut total_deleted = 0;

        // Process each condition sequentially to handle ID shifting correctly.
        // When documents are deleted, all subsequent document IDs shift down.
        // So we must resolve each condition against the current state AFTER
        // any previous deletions have been applied.
        for item in &conditions {
            // Resolve condition to IDs from current metadata.db state
            let doc_ids =
                match filtering::where_condition(&path_str, &item.condition, &item.parameters) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tracing::warn!(
                            index = %name_inner,
                            condition = %item.condition,
                            error = %e,
                            "delete.condition.failed"
                        );
                        continue;
                    }
                };

            if doc_ids.is_empty() {
                continue;
            }

            // Delete these documents - this will shift remaining IDs
            // Note: delete() modifies disk but doesn't reload in-memory state
            match index.delete(&doc_ids) {
                Ok(deleted) => {
                    total_deleted += deleted;
                    tracing::debug!(
                        index = %name_inner,
                        condition = %item.condition,
                        deleted = deleted,
                        "delete.condition.complete"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        index = %name_inner,
                        condition = %item.condition,
                        num_ids = doc_ids.len(),
                        error = %e,
                        "delete.condition.failed"
                    );
                }
            }
        }

        // Reload index once after all deletions to get accurate metadata
        index
            .reload()
            .map_err(|e| format!("Failed to reload index after deletions: {}", e))?;
        let remaining = index.metadata.num_documents;

        // Reload state
        state_clone
            .reload_index(&name_inner)
            .map_err(|e| format!("Failed to reload index: {}", e))?;

        Ok((total_deleted, remaining))
    })
    .await;

    let total_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(Ok((deleted, remaining))) => {
            tracing::info!(
                index = %index_name,
                num_conditions = num_conditions,
                deleted = deleted,
                remaining = remaining,
                total_ms = total_ms,
                "delete.batch.complete"
            );
        }
        Ok(Err(e)) => {
            tracing::error!(
                index = %index_name,
                num_conditions = num_conditions,
                error = %e,
                total_ms = total_ms,
                "delete.batch.failed"
            );
        }
        Err(e) => {
            tracing::error!(
                index = %index_name,
                num_conditions = num_conditions,
                error = %e,
                total_ms = total_ms,
                "delete.batch.panicked"
            );
        }
    }
}

// =============================================================================

/// Declare a new index with its configuration.
#[utoipa::path(
    post,
    path = "/indices",
    tag = "indices",
    request_body = CreateIndexRequest,
    responses(
        (status = 200, description = "Index declared successfully", body = CreateIndexResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 409, description = "Index already exists", body = ErrorResponse)
    )
)]
pub async fn create_index(
    State(state): State<Arc<AppState>>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<CreateIndexRequest>,
) -> ApiResult<Json<CreateIndexResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();

    // Validate name
    if req.name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Lock mainly to prevent race condition on file existence check
    let lock = get_index_lock(&req.name);
    let _guard = lock.lock().await;

    // Check if index already exists (either declared or populated)
    let index_path = state.index_path(&req.name);
    if index_path.join("config.json").exists() || index_path.join("metadata.json").exists() {
        return Err(ApiError::IndexAlreadyExists(req.name.clone()));
    }

    // Build stored config
    let stored_config = IndexConfigStored {
        nbits: req.config.nbits.unwrap_or(4),
        batch_size: req.config.batch_size.unwrap_or(50_000),
        seed: req.config.seed,
        start_from_scratch: req.config.start_from_scratch.unwrap_or(999),
        max_documents: req.config.max_documents,
    };

    // Create index directory
    std::fs::create_dir_all(&index_path)
        .map_err(|e| ApiError::Internal(format!("Failed to create index directory: {}", e)))?;

    // Store config.json and cache it
    state.set_index_config(&req.name, stored_config.clone())?;

    tracing::info!(
        trace_id = %trace_id,
        index = %req.name,
        nbits = stored_config.nbits,
        max_documents = ?stored_config.max_documents,
        "index.create.complete"
    );

    Ok(Json(CreateIndexResponse {
        name: req.name,
        config: stored_config,
        message: "Index declared. Use POST /indices/{name}/update to add documents.".to_string(),
    }))
}

/// Get information about a specific index.
#[utoipa::path(
    get,
    path = "/indices/{name}",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "Index information", body = IndexInfoResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn get_index_info(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<IndexInfoResponse>> {
    // Use lock-free read access (never blocks during writes)
    let (path_str, num_documents, num_embeddings, num_partitions, avg_doclen, dimension) = {
        let idx = state.get_index_for_read(&name)?;
        (
            idx.path.clone(),
            idx.num_documents(),
            idx.num_embeddings(),
            idx.num_partitions(),
            idx.avg_doclen(),
            idx.embedding_dim(),
        )
    };

    // Get max_documents from cached config (fast, no disk I/O if cached)
    let max_documents = state.get_index_config(&name).and_then(|c| c.max_documents);

    // Run blocking SQLite operations in a separate thread
    let (has_metadata, metadata_count) = task::spawn_blocking(move || {
        let has_metadata = filtering::exists(&path_str);
        let metadata_count = if has_metadata {
            filtering::count(&path_str).ok()
        } else {
            None
        };

        (has_metadata, metadata_count)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    Ok(Json(IndexInfoResponse {
        name,
        num_documents,
        num_embeddings,
        num_partitions,
        avg_doclen,
        dimension,
        has_metadata,
        metadata_count,
        max_documents,
    }))
}

/// List all available indices.
#[utoipa::path(
    get,
    path = "/indices",
    tag = "indices",
    responses(
        (status = 200, description = "List of index names", body = Vec<String>)
    )
)]
pub async fn list_indices(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    // Run blocking filesystem iteration in a separate thread
    let indices = task::spawn_blocking(move || state.list_all())
        .await
        .unwrap_or_default();
    Json(indices)
}

/// Add documents to an existing index.
///
/// Returns 202 Accepted immediately and processes in background.
#[utoipa::path(
    post,
    path = "/indices/{name}/documents",
    tag = "documents",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = AddDocumentsRequest,
    responses(
        (status = 202, description = "Request accepted for background processing", body = String),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn add_documents(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddDocumentsRequest>,
) -> ApiResult<impl IntoResponse> {
    if req.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    // Validate metadata length (metadata is required) - cheap check first
    if req.metadata.len() != req.documents.len() {
        return Err(ApiError::BadRequest(format!(
            "Metadata length ({}) must match documents length ({})",
            req.metadata.len(),
            req.documents.len()
        )));
    }

    // Check index existence BEFORE expensive conversion - fail fast (lock-free read)
    let expected_dim = {
        let idx = state.get_index_for_read(&name)?;
        idx.embedding_dim()
    };

    // Check first document's dimension before converting all (fail fast on dimension mismatch)
    if let Some(first_doc) = req.documents.first() {
        if first_doc.embeddings.is_empty() {
            return Err(ApiError::BadRequest("Empty embeddings".to_string()));
        }
        let first_dim = first_doc.embeddings[0].len();
        if first_dim != expected_dim {
            return Err(ApiError::DimensionMismatch {
                expected: expected_dim,
                actual: first_dim,
            });
        }
    }

    // Now perform CPU-intensive validation/conversion (index exists, dimension likely correct)
    let embeddings: Vec<Array2<f32>> = req
        .documents
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    // Final dimension check for all embeddings (in case documents have mixed dimensions)
    for emb in embeddings.iter() {
        if emb.ncols() != expected_dim {
            return Err(ApiError::DimensionMismatch {
                expected: expected_dim,
                actual: emb.ncols(),
            });
        }
    }

    // Prepare data for background task
    let name_clone = name.clone();
    let state_clone = state.clone();
    let metadata = req.metadata;
    let lock = get_index_lock(&name);

    // Acquire semaphore permit to limit queued tasks
    let semaphore = get_index_semaphore(&name);
    let permit = semaphore.clone().try_acquire_owned().map_err(|_| {
        ApiError::ServiceUnavailable(format!(
            "Update queue full for index '{}'. Max {} pending updates. Retry later.",
            name,
            max_queued_tasks_per_index()
        ))
    })?;

    let doc_count = embeddings.len();

    // Spawn background task
    tokio::spawn(async move {
        // Permit is held until this task completes (dropped at end of async block)
        let _permit = permit;

        // 1. Acquire async lock
        let _guard = lock.lock().await;

        // Clone name AGAIN for the inner closure, so `name_clone` stays valid for error logging
        let name_inner = name_clone.clone();
        let start = std::time::Instant::now();

        // 2. Perform heavy IO work in a blocking task
        let result = task::spawn_blocking(move || -> ApiResult<u64> {
            // Load index for update
            let path_str = state_clone
                .index_path(&name_inner)
                .to_string_lossy()
                .to_string();

            // Check sync before updating: if filtering DB exists, counts must match
            let index_path = std::path::Path::new(&path_str);
            if filtering::exists(&path_str) {
                let index_metadata = Metadata::load_from_path(index_path)?;
                let filtering_count = filtering::count(&path_str)?;

                if index_metadata.num_documents != filtering_count {
                    return Err(ApiError::Internal(format!(
                        "Index out of sync: vector index has {} documents but metadata DB has {}. \
                         A full rebuild is required.",
                        index_metadata.num_documents, filtering_count
                    )));
                }
            }

            let mut index = MmapIndex::load(&path_str)?;

            // Update with metadata (metadata is required)
            let update_config = UpdateConfig::default();
            let index_update_start = std::time::Instant::now();
            index.update_with_metadata(&embeddings, &update_config, Some(&metadata))?;
            let index_update_ms = index_update_start.elapsed().as_millis() as u64;

            // Eviction: Load config to check max_documents
            let config_path = state_clone.index_path(&name_inner).join("config.json");
            if let Ok(config_file) = std::fs::File::open(&config_path) {
                if let Ok(stored_config) =
                    serde_json::from_reader::<_, IndexConfigStored>(config_file)
                {
                    if let Some(max_docs) = stored_config.max_documents {
                        if let Err(e) = evict_oldest_documents(&mut index, max_docs) {
                            tracing::warn!(
                                index = %name_inner,
                                error = %e,
                                max_documents = max_docs,
                                "documents.add.eviction.failed"
                            );
                        }
                    }
                }
            }

            // Reload state
            state_clone.reload_index(&name_inner)?;
            Ok(index_update_ms)
        })
        .await;

        let total_ms = start.elapsed().as_millis() as u64;

        // Log result
        match result {
            Ok(Ok(index_update_ms)) => {
                tracing::info!(
                    index = %name_clone,
                    num_documents = doc_count,
                    index_update_ms = index_update_ms,
                    total_ms = total_ms,
                    "documents.add.complete"
                );
            }
            Ok(Err(e)) => {
                tracing::error!(
                    index = %name_clone,
                    num_documents = doc_count,
                    error = %e,
                    total_ms = total_ms,
                    "documents.add.failed"
                );
            }
            Err(e) => {
                tracing::error!(
                    index = %name_clone,
                    num_documents = doc_count,
                    error = %e,
                    total_ms = total_ms,
                    "documents.add.panicked"
                );
            }
        }
    });

    // Return 202 Accepted immediately
    Ok((StatusCode::ACCEPTED, Json("Update queued in background")))
}

/// Delete documents from an index by metadata filter.
///
/// Returns 202 Accepted immediately. Delete conditions are batched together
/// for efficient processing - multiple delete requests within a short window
/// will be processed in a single batch operation.
///
/// Batching parameters can be configured via environment variables:
/// - DELETE_BATCH_MIN_WAIT: Minimum wait time before processing (default: 500ms)
/// - DELETE_BATCH_MAX_WAIT: Maximum wait time for batching (default: 2000ms)
/// - MAX_DELETE_BATCH_CONDITIONS: Max conditions per batch (default: 50)
#[utoipa::path(
    delete,
    path = "/indices/{name}/documents",
    tag = "documents",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = DeleteDocumentsRequest,
    responses(
        (status = 202, description = "Delete request accepted for background processing", body = String),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn delete_documents(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<DeleteDocumentsRequest>,
) -> ApiResult<impl IntoResponse> {
    if req.condition.is_empty() {
        return Err(ApiError::BadRequest(
            "Delete condition cannot be empty".to_string(),
        ));
    }

    // Verify index exists
    let path_str = state.index_path(&name).to_string_lossy().to_string();
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Verify metadata exists
    if !filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name));
    }

    // Get or create the delete batch queue for this index
    let sender = get_or_create_delete_batch_queue(&name, state.clone());

    // Try to queue the delete condition
    let batch_item = DeleteBatchItem {
        condition: req.condition.clone(),
        parameters: req.parameters.clone(),
    };

    sender.try_send(batch_item).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(format!(
            "Delete queue full for index '{}'. Max {} pending items. Retry later.",
            name,
            batch_channel_size()
        )),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal(format!("Delete worker for index '{}' is not running", name))
        }
    })?;

    tracing::debug!(
        index = %name,
        condition = %req.condition,
        "delete.queued"
    );

    // Return 202 Accepted immediately
    Ok((
        StatusCode::ACCEPTED,
        Json("Delete condition queued for batch processing".to_string()),
    ))
}

/// Delete an entire index and all its data.
#[utoipa::path(
    delete,
    path = "/indices/{name}",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "Index deleted successfully", body = DeleteIndexResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn delete_index(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
) -> ApiResult<Json<DeleteIndexResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();

    let lock = get_index_lock(&name);
    let _guard = lock.lock().await;

    // Stop background workers by removing their queue entries.
    // Dropping the sender closes the channel, causing workers to exit.
    let queue_key = state.index_path(&name).to_string_lossy().to_string();
    if let Some(queues) = BATCH_QUEUES.get() {
        if let Ok(mut map) = queues.lock() {
            map.remove(&queue_key);
        }
    }
    if let Some(queues) = DELETE_BATCH_QUEUES.get() {
        if let Ok(mut map) = queues.lock() {
            map.remove(&queue_key);
        }
    }

    // Unload from memory and invalidate caches
    state.unload_index(&name);
    state.invalidate_config_cache(&name);

    // Drop the lock guard before removing the directory, so the async mutex
    // isn't held while we delete (similar to colgrep's drop(lock) pattern).
    drop(_guard);

    // Delete from disk
    let path = state.index_path(&name);
    if path.exists() {
        std::fs::remove_dir_all(&path)
            .map_err(|e| ApiError::Internal(format!("Failed to delete index: {}", e)))?;
    }

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        "index.delete.complete"
    );

    Ok(Json(DeleteIndexResponse {
        deleted: true,
        name,
    }))
}

/// Update an index by adding documents.
///
/// Returns 202 Accepted immediately and processes in background.
/// Multiple concurrent requests to the same index are batched together
/// for more efficient processing (up to 300 documents per batch).
#[utoipa::path(
    post,
    path = "/indices/{name}/update",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateIndexRequest,
    responses(
        (status = 202, description = "Request accepted for background processing", body = String),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not declared", body = ErrorResponse)
    )
)]
pub async fn update_index(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateIndexRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate name
    if name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Basic Validation (Fail fast)
    let index_path = state.index_path(&name);
    let config_path = index_path.join("config.json");
    if !config_path.exists() {
        return Err(ApiError::IndexNotDeclared(name));
    }

    // Heavy CPU work: convert to ndarray
    let embeddings: Vec<Array2<f32>> = req
        .documents
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    if embeddings.is_empty() {
        return Err(ApiError::BadRequest(
            "At least one document is required".to_string(),
        ));
    }

    // Validate metadata length (metadata is required)
    if req.metadata.len() != embeddings.len() {
        return Err(ApiError::BadRequest(format!(
            "Metadata length ({}) must match documents length ({})",
            req.metadata.len(),
            embeddings.len()
        )));
    }

    let doc_count = embeddings.len();

    // Get or create the batch queue for this index
    let sender = get_or_create_batch_queue(&name, state.clone());

    // Create batch item
    let batch_item = BatchItem {
        embeddings,
        metadata: req.metadata,
    };

    // Send to batch queue (non-blocking if channel has capacity)
    // Check queue depth for warning
    let queue_capacity = batch_channel_size();
    let queue_available = sender.capacity();
    let queue_depth = queue_capacity - queue_available;

    // Warn when queue is >80% full
    if queue_depth > (queue_capacity * 8 / 10) {
        tracing::warn!(
            index = %name,
            queue_depth = queue_depth,
            max_capacity = queue_capacity,
            "update.queue.high"
        );
    }

    sender.try_send(batch_item).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(format!(
            "Update queue full for index '{}'. Max {} pending items. Retry later.",
            name,
            batch_channel_size()
        )),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal(format!("Batch worker for index '{}' is not running", name))
        }
    })?;

    tracing::debug!(
        index = %name,
        num_documents = doc_count,
        queue_depth = queue_depth + 1,
        "update.queued"
    );

    // Immediate Response
    Ok((StatusCode::ACCEPTED, Json("Update queued for batching")))
}

/// Update index configuration (max_documents).
///
/// Changes the max_documents limit. If the new limit is lower than the current
/// document count, eviction will NOT happen immediately - it will occur on the
/// next document addition.
#[utoipa::path(
    put,
    path = "/indices/{name}/config",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateIndexConfigRequest,
    responses(
        (status = 200, description = "Configuration updated", body = UpdateIndexConfigResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn update_index_config(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<UpdateIndexConfigRequest>,
) -> ApiResult<Json<UpdateIndexConfigResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();

    let lock = get_index_lock(&name);
    let _guard = lock.lock().await;

    // Load existing config from cache (or disk if not cached)
    let mut stored_config = state
        .get_index_config(&name)
        .ok_or_else(|| ApiError::IndexNotFound(name.clone()))?;

    // Update max_documents
    stored_config.max_documents = req.max_documents;

    // Save updated config (updates both disk and cache)
    state.set_index_config(&name, stored_config.clone())?;

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        max_documents = ?req.max_documents,
        "index.config.update.complete"
    );

    let message = match req.max_documents {
        Some(max) => format!(
            "max_documents set to {}. Eviction will occur on next document addition if over limit.",
            max
        ),
        None => "max_documents limit removed (unlimited).".to_string(),
    };

    Ok(Json(UpdateIndexConfigResponse {
        name,
        config: stored_config,
        message,
    }))
}

/// Update an index with document texts (requires model to be loaded).
///
/// This endpoint encodes the document texts using the loaded model and then
/// adds them to the index. Requires the server to be started with `--model <path>`.
///
/// Returns 202 Accepted immediately and processes in background.
#[utoipa::path(
    post,
    path = "/indices/{name}/update_with_encoding",
    tag = "indices",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateWithEncodingRequest,
    responses(
        (status = 202, description = "Request accepted for background processing", body = String),
        (status = 400, description = "Invalid request or model not loaded", body = ErrorResponse),
        (status = 404, description = "Index not declared", body = ErrorResponse)
    )
)]
pub async fn update_index_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateWithEncodingRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate name
    if name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Check index is declared
    let index_path = state.index_path(&name);
    let config_path = index_path.join("config.json");
    if !config_path.exists() {
        return Err(ApiError::IndexNotDeclared(name));
    }

    // Validate input
    if req.documents.is_empty() {
        return Err(ApiError::BadRequest(
            "At least one document is required".to_string(),
        ));
    }

    // Validate metadata length
    if req.metadata.len() != req.documents.len() {
        return Err(ApiError::BadRequest(format!(
            "Metadata length ({}) must match documents length ({})",
            req.metadata.len(),
            req.documents.len()
        )));
    }

    // Get or create the batch queue for this index FIRST
    let sender = get_or_create_batch_queue(&name, state.clone());

    // Reserve a slot in the batch queue BEFORE encoding
    // This ensures we don't waste encoding work if the queue is full
    let permit = sender.try_reserve().map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(format!(
            "Update queue full for index '{}'. Max {} pending items. Retry later.",
            name,
            batch_channel_size()
        )),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal(format!("Batch worker for index '{}' is not running", name))
        }
    })?;

    // Now encode - we have a guaranteed slot in the batch queue
    let embeddings = encode_texts_internal(
        state.clone(),
        &req.documents,
        InputType::Document,
        req.pool_factor,
    )
    .await?;

    let doc_count = embeddings.len();

    // Create batch item
    let batch_item = BatchItem {
        embeddings,
        metadata: req.metadata,
    };

    // Send using the reserved permit - this is guaranteed to succeed
    permit.send(batch_item);

    tracing::debug!(
        index = %name,
        num_documents = doc_count,
        "update.with_encoding.queued"
    );

    // Immediate Response
    Ok((StatusCode::ACCEPTED, Json("Update queued for batching")))
}
