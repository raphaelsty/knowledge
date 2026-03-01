//! Encode endpoint handler for the next-plaid API.
//!
//! Provides text encoding using a pool of ColBERT model workers for concurrent
//! encoding with automatic batching of requests for improved throughput.

#[cfg(feature = "model")]
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "model")]
use std::sync::OnceLock;
#[cfg(feature = "model")]
use std::time::Duration;

use axum::{extract::State, Json};
#[cfg(feature = "model")]
use tokio::sync::{mpsc, oneshot};
#[cfg(feature = "model")]
use tokio::time::Instant;

use crate::error::{ApiError, ApiResult};
#[cfg(feature = "model")]
use crate::models::InputType;
use crate::models::{EncodeRequest, EncodeResponse};
use crate::state::AppState;

// --- Batch Configuration ---

/// Get the maximum number of texts to batch together before processing.
/// Configurable via MAX_BATCH_TEXTS env var (default: 64).
/// Aligned with typical model batch size for optimal GPU utilization.
#[cfg(feature = "model")]
fn max_batch_texts() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MAX_BATCH_TEXTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

/// Maximum time to wait for more requests before processing a batch.
/// Lower = less latency for single requests, higher = better batching efficiency.
#[cfg(feature = "model")]
const BATCH_TIMEOUT: Duration = Duration::from_millis(10);

/// Get the channel buffer size for encode batch queue.
/// Configurable via ENCODE_BATCH_CHANNEL_SIZE env var (default: 256).
#[cfg(feature = "model")]
fn encode_batch_channel_size() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("ENCODE_BATCH_CHANNEL_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

// --- Batch Types ---

/// Type alias for encoding results: either embeddings or error message.
#[cfg(feature = "model")]
type EncodeResult = Result<Vec<Vec<Vec<f32>>>, String>;

/// A single item in the encode batch queue, representing one client request.
#[cfg(feature = "model")]
struct EncodeBatchItem {
    /// Texts to encode
    texts: Vec<String>,
    /// Input type (query or document)
    input_type: InputType,
    /// Optional pool factor for document encoding
    pool_factor: Option<usize>,
    /// Channel to send results back to the client
    response_tx: oneshot::Sender<EncodeResult>,
}

/// Handle to the encode worker pool.
#[cfg(feature = "model")]
struct EncodeWorkerPool {
    sender: mpsc::Sender<EncodeBatchItem>,
}

/// Global encode worker pool (singleton).
#[cfg(feature = "model")]
static ENCODE_WORKER_POOL: OnceLock<std::sync::Mutex<Option<EncodeWorkerPool>>> = OnceLock::new();

/// Get or create the global encode worker pool.
/// Spawns multiple workers, each owning its own Colbert model instance.
#[cfg(feature = "model")]
fn get_or_create_encode_pool(state: Arc<AppState>) -> ApiResult<mpsc::Sender<EncodeBatchItem>> {
    let pool_lock: &std::sync::Mutex<Option<EncodeWorkerPool>> =
        ENCODE_WORKER_POOL.get_or_init(|| std::sync::Mutex::new(None));

    let mut pool_opt = pool_lock.lock().unwrap();

    if let Some(pool) = pool_opt.as_ref() {
        return Ok(pool.sender.clone());
    }

    // Get model pool configuration
    let model_pool = state
        .model_pool
        .as_ref()
        .ok_or_else(|| ApiError::ModelNotLoaded)?;

    let pool_size = model_pool.pool_size;
    let model_config = model_pool.model_config.clone();

    // Create shared channel for all workers
    let (sender, receiver) = mpsc::channel(encode_batch_channel_size());

    // Wrap receiver in Arc<Mutex> for sharing among workers
    let shared_receiver = Arc::new(tokio::sync::Mutex::new(receiver));

    // Spawn N workers, each building and owning its own model
    tracing::info!(pool_size = pool_size, "encode.worker.pool.starting");

    for worker_id in 0..pool_size {
        let receiver_clone = Arc::clone(&shared_receiver);
        let config_clone = model_config.clone();

        // Spawn worker in a blocking task since model building is CPU-intensive
        tokio::spawn(async move {
            // Build model for this worker
            let model = match build_model_from_config(&config_clone) {
                Ok(m) => {
                    tracing::info!(worker_id = worker_id, "encode.worker.started");
                    m
                }
                Err(e) => {
                    tracing::error!(
                        worker_id = worker_id,
                        error = %e,
                        "encode.worker.start.failed"
                    );
                    return;
                }
            };

            // Run the worker loop with owned model
            encode_worker_loop(worker_id, model, receiver_clone).await;
        });
    }

    let pool = EncodeWorkerPool {
        sender: sender.clone(),
    };
    *pool_opt = Some(pool);

    Ok(sender)
}

/// Build a Colbert model from configuration.
#[cfg(feature = "model")]
fn build_model_from_config(
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

/// Worker loop that owns a model and processes batches from the shared queue.
#[cfg(feature = "model")]
async fn encode_worker_loop(
    worker_id: usize,
    model: next_plaid_onnx::Colbert,
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<EncodeBatchItem>>>,
) {
    loop {
        // Collect a batch of items
        let pending_items = {
            let mut rx = receiver.lock().await;

            // Wait for the first item (blocking)
            let first_item = match rx.recv().await {
                Some(item) => item,
                None => {
                    tracing::debug!(worker_id = worker_id, "encode.worker.stopped");
                    break;
                }
            };

            // Start collecting batch
            let mut pending_items: Vec<EncodeBatchItem> = vec![first_item];
            let mut total_texts = pending_items[0].texts.len();
            let deadline = Instant::now() + BATCH_TIMEOUT;

            // Collect more items until timeout or max batch size
            while total_texts < max_batch_texts() {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }

                match tokio::time::timeout(remaining, rx.recv()).await {
                    Ok(Some(item)) => {
                        total_texts += item.texts.len();
                        pending_items.push(item);
                    }
                    Ok(None) => {
                        // Channel closed - process remaining batch before exiting
                        if !pending_items.is_empty() {
                            process_encode_batch_with_model(worker_id, pending_items, &model).await;
                        }
                        tracing::debug!(worker_id = worker_id, "encode.worker.stopped");
                        return;
                    }
                    Err(_) => {
                        // Timeout reached
                        break;
                    }
                }
            }

            pending_items
        }; // Release the lock before processing

        // Process the collected batch (lock is released)
        if !pending_items.is_empty() {
            process_encode_batch_with_model(worker_id, pending_items, &model).await;
        }
    }
}

/// Process a batch of encode requests using an owned model.
///
/// Groups items by input_type and pool_factor, encodes them in batches,
/// then distributes results back to waiting clients.
#[cfg(feature = "model")]
async fn process_encode_batch_with_model(
    worker_id: usize,
    items: Vec<EncodeBatchItem>,
    model: &next_plaid_onnx::Colbert,
) {
    let start = std::time::Instant::now();
    let num_requests = items.len();
    let total_texts: usize = items.iter().map(|i| i.texts.len()).sum();

    // Group items by input type and pool_factor
    let mut query_items: Vec<(usize, &EncodeBatchItem)> = Vec::new();
    // Group documents by pool_factor: (pool_factor, items)
    let mut document_groups: HashMap<Option<usize>, Vec<(usize, &EncodeBatchItem)>> =
        HashMap::new();

    for (idx, item) in items.iter().enumerate() {
        match item.input_type {
            InputType::Query => query_items.push((idx, item)),
            InputType::Document => {
                document_groups
                    .entry(item.pool_factor)
                    .or_default()
                    .push((idx, item));
            }
        }
    }

    // Prepare results storage
    let mut results: HashMap<usize, EncodeResult> = HashMap::with_capacity(num_requests);

    // Process queries batch (no pool_factor for queries)
    if !query_items.is_empty() {
        let all_query_texts: Vec<String> = query_items
            .iter()
            .flat_map(|(_, item)| item.texts.clone())
            .collect();

        let query_results =
            encode_texts_with_model(worker_id, model, &all_query_texts, InputType::Query, None);

        match query_results {
            Ok(embeddings) => {
                // Distribute embeddings back to each request
                let mut offset = 0;
                for (idx, item) in &query_items {
                    let count = item.texts.len();
                    let item_embeddings: Vec<Vec<Vec<f32>>> =
                        embeddings[offset..offset + count].to_vec();
                    results.insert(*idx, Ok(item_embeddings));
                    offset += count;
                }
            }
            Err(e) => {
                // Send error to all query requests
                for (idx, _) in &query_items {
                    results.insert(*idx, Err(e.clone()));
                }
            }
        }
    }

    // Process documents grouped by pool_factor
    for (pool_factor, doc_items) in document_groups {
        let all_doc_texts: Vec<String> = doc_items
            .iter()
            .flat_map(|(_, item)| item.texts.clone())
            .collect();

        let doc_results = encode_texts_with_model(
            worker_id,
            model,
            &all_doc_texts,
            InputType::Document,
            pool_factor,
        );

        match doc_results {
            Ok(embeddings) => {
                // Distribute embeddings back to each request
                let mut offset = 0;
                for (idx, item) in &doc_items {
                    let count = item.texts.len();
                    let item_embeddings: Vec<Vec<Vec<f32>>> =
                        embeddings[offset..offset + count].to_vec();
                    results.insert(*idx, Ok(item_embeddings));
                    offset += count;
                }
            }
            Err(e) => {
                // Send error to all document requests in this group
                for (idx, _) in &doc_items {
                    results.insert(*idx, Err(e.clone()));
                }
            }
        }
    }

    // Send results back to clients
    for (idx, item) in items.into_iter().enumerate() {
        let result = results
            .remove(&idx)
            .unwrap_or_else(|| Err("Missing result".to_string()));
        // Ignore send errors (client may have disconnected)
        let _ = item.response_tx.send(result);
    }

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        worker_id = worker_id,
        num_requests = num_requests,
        num_texts = total_texts,
        total_ms = total_ms,
        "encode.batch.complete"
    );
}

/// Encode texts using an owned model (no mutex needed).
#[cfg(feature = "model")]
fn encode_texts_with_model(
    worker_id: usize,
    model: &next_plaid_onnx::Colbert,
    texts: &[String],
    input_type: InputType,
    pool_factor: Option<usize>,
) -> Result<Vec<Vec<Vec<f32>>>, String> {
    let num_texts = texts.len();
    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let encode_start = std::time::Instant::now();
    let embeddings_result = match input_type {
        InputType::Query => model.encode_queries(&texts_ref),
        InputType::Document => model.encode_documents(&texts_ref, pool_factor),
    };
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    let embeddings = embeddings_result.map_err(|e| e.to_string())?;

    // Convert Array2<f32> to Vec<Vec<Vec<f32>>>
    let result: Vec<Vec<Vec<f32>>> = embeddings
        .into_iter()
        .map(|arr| arr.rows().into_iter().map(|row| row.to_vec()).collect())
        .collect();

    tracing::debug!(
        worker_id = worker_id,
        input_type = ?input_type,
        num_texts = num_texts,
        pool_factor = ?pool_factor,
        encode_ms = encode_ms,
        "encode.texts.complete"
    );

    Ok(result)
}

/// Encode texts into ColBERT embeddings.
///
/// This endpoint requires the server to be started with `--model <path>`.
/// If no model is loaded, returns a 400 error.
///
/// Requests are automatically batched with other concurrent requests for
/// improved throughput on GPU.
#[cfg(feature = "model")]
#[utoipa::path(
    post,
    path = "/encode",
    tag = "encoding",
    request_body = EncodeRequest,
    responses(
        (status = 200, description = "Texts encoded successfully", body = EncodeResponse),
        (status = 400, description = "Model not loaded or invalid request"),
        (status = 500, description = "Encoding failed")
    )
)]
pub async fn encode(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EncodeRequest>,
) -> ApiResult<Json<EncodeResponse>> {
    // Validate request
    if request.texts.is_empty() {
        return Err(ApiError::BadRequest("No texts provided".to_string()));
    }

    // Use the single internal encoding function
    let embeddings_arr = encode_texts_internal(
        state,
        &request.texts,
        request.input_type,
        request.pool_factor,
    )
    .await?;

    // Convert Array2<f32> to Vec<Vec<Vec<f32>>> for JSON response
    let embeddings: Vec<Vec<Vec<f32>>> = embeddings_arr
        .into_iter()
        .map(|arr| arr.rows().into_iter().map(|row| row.to_vec()).collect())
        .collect();
    let num_texts = embeddings.len();

    Ok(Json(EncodeResponse {
        embeddings,
        num_texts,
    }))
}

/// Stub encode function when model feature is not enabled.
#[cfg(not(feature = "model"))]
#[utoipa::path(
    post,
    path = "/encode",
    tag = "encoding",
    request_body = EncodeRequest,
    responses(
        (status = 400, description = "Model support not compiled"),
    )
)]
pub async fn encode(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<EncodeRequest>,
) -> ApiResult<Json<EncodeResponse>> {
    Err(ApiError::ModelNotLoaded)
}

/// Internal function to encode texts (queries or documents).
/// This is async and uses the worker pool mechanism to avoid blocking.
/// All encoding in the API goes through this single function.
#[cfg(feature = "model")]
pub async fn encode_texts_internal(
    state: Arc<AppState>,
    texts: &[String],
    input_type: InputType,
    pool_factor: Option<usize>,
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    if !state.has_model() {
        return Err(ApiError::ModelNotLoaded);
    }

    // Create oneshot channel for receiving results
    let (response_tx, response_rx) = oneshot::channel();

    // Create batch item
    let batch_item = EncodeBatchItem {
        texts: texts.to_vec(),
        input_type,
        pool_factor,
        response_tx,
    };

    // Get or create the worker pool
    let sender = get_or_create_encode_pool(state)?;

    // Send to worker pool
    sender.try_send(batch_item).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => ApiError::ServiceUnavailable(
            "Encode queue full. Too many concurrent requests. Retry later.".to_string(),
        ),
        mpsc::error::TrySendError::Closed(_) => {
            ApiError::Internal("Encode worker pool is not running".to_string())
        }
    })?;

    // Wait for result from worker
    let result = response_rx
        .await
        .map_err(|_| ApiError::Internal("Worker dropped response channel".to_string()))?;

    let embeddings_3d = result.map_err(ApiError::ModelError)?;

    // Convert Vec<Vec<Vec<f32>>> back to Vec<Array2<f32>>
    let embeddings: Vec<ndarray::Array2<f32>> = embeddings_3d
        .into_iter()
        .map(|doc| {
            let rows = doc.len();
            let cols = if rows > 0 { doc[0].len() } else { 0 };
            let flat: Vec<f32> = doc.into_iter().flatten().collect();
            ndarray::Array2::from_shape_vec((rows, cols), flat).unwrap()
        })
        .collect();

    Ok(embeddings)
}

/// Stub for encode_texts_internal when model feature is not enabled.
#[cfg(not(feature = "model"))]
pub async fn encode_texts_internal(
    _state: Arc<AppState>,
    _texts: &[String],
    _input_type: crate::models::InputType,
    _pool_factor: Option<usize>,
) -> ApiResult<Vec<ndarray::Array2<f32>>> {
    Err(ApiError::ModelNotLoaded)
}
