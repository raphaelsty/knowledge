//! Application state management for the next_plaid API.
//!
//! Manages loaded indices and provides thread-safe access to them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use arc_swap::{ArcSwap, Guard};
use next_plaid::MmapIndex;
use parking_lot::RwLock;

/// Global registry of per-index loading locks to prevent concurrent loads.
/// This prevents race conditions where multiple requests try to load the same
/// index simultaneously, which can cause file conflicts.
static LOADING_LOCKS: OnceLock<std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<()>>>>> =
    OnceLock::new();

/// Slot for an index with lock-free read access using ArcSwap.
/// Allows atomic swapping of the index during writes while readers continue
/// with the old version uninterrupted.
pub struct IndexSlot {
    /// The active index, accessible via lock-free atomic load
    active: ArcSwap<MmapIndex>,
}

impl IndexSlot {
    /// Create a new index slot with the given index.
    pub fn new(index: MmapIndex) -> Self {
        Self {
            active: ArcSwap::from_pointee(index),
        }
    }

    /// Get a lock-free reference to the current index.
    /// This never blocks, even during writes.
    pub fn load(&self) -> Guard<Arc<MmapIndex>> {
        self.active.load()
    }

    /// Store a new index (like swap but discards the old one).
    pub fn store(&self, new_index: MmapIndex) {
        self.active.store(Arc::new(new_index));
    }
}

/// Get or create a loading lock for the given index name.
fn get_loading_lock(name: &str) -> Arc<std::sync::Mutex<()>> {
    let locks = LOADING_LOCKS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut locks_guard = locks
        .lock()
        .expect("LOADING_LOCKS mutex poisoned - a thread panicked while holding this lock");
    locks_guard
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(std::sync::Mutex::new(())))
        .clone()
}

use crate::error::{ApiError, ApiResult};
use crate::models::IndexConfigStored;

/// Configuration for the API server.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Base directory for storing indices
    pub index_dir: PathBuf,
    /// Default number of results to return
    pub default_top_k: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            index_dir: PathBuf::from("./indices"),
            default_top_k: 10,
        }
    }
}

/// Model configuration info for logging purposes.
#[cfg(feature = "model")]
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelInfo {
    /// Path to the model directory
    pub path: String,
    /// Whether the model is INT8 quantized
    pub quantized: bool,
}

/// Configuration for building a model instance.
/// Used by the worker pool to create model instances on each worker.
#[cfg(feature = "model")]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the model directory
    pub path: std::path::PathBuf,
    /// Whether to use CUDA execution provider
    pub use_cuda: bool,
    /// Whether to use INT8 quantization
    pub use_int8: bool,
    /// Number of parallel ONNX sessions per model
    pub parallel_sessions: Option<usize>,
    /// Batch size for encoding
    pub batch_size: Option<usize>,
    /// Threads per ONNX session
    pub threads: Option<usize>,
    /// Maximum query length in tokens
    pub query_length: Option<usize>,
    /// Maximum document length in tokens
    pub document_length: Option<usize>,
}

/// Model pool for concurrent encoding.
/// Stores configuration for workers to create their own model instances.
#[cfg(feature = "model")]
#[derive(Debug)]
pub struct ModelPool {
    /// Number of workers in the pool
    pub pool_size: usize,
    /// Configuration for building model instances
    pub model_config: ModelConfig,
    /// Cached model info for lock-free access (immutable after init)
    pub cached_info: CachedModelInfo,
}

/// Cached model information that doesn't require locking.
/// This information is immutable after model initialization.
#[cfg(feature = "model")]
#[derive(Debug, Clone)]
pub struct CachedModelInfo {
    /// Model name (from config)
    pub name: Option<String>,
    /// Path to the model directory
    pub path: String,
    /// Whether INT8 quantization is enabled
    pub quantized: bool,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Batch size used for encoding
    pub batch_size: usize,
    /// Number of parallel ONNX sessions
    pub num_sessions: usize,
    /// Query prefix token
    pub query_prefix: String,
    /// Document prefix token
    pub document_prefix: String,
    /// Maximum query length
    pub query_length: usize,
    /// Maximum document length
    pub document_length: usize,
    /// Whether query expansion is enabled
    pub do_query_expansion: bool,
    /// Whether the model uses token_type_ids
    pub uses_token_type_ids: bool,
    /// MASK token ID for query expansion
    pub mask_token_id: u32,
    /// PAD token ID
    pub pad_token_id: u32,
}

/// Application state containing loaded indices and shared database pool.
///
/// All indices are stored as MmapIndex for efficient memory usage.
/// Indices use ArcSwap for lock-free read access during write operations.
pub struct AppState {
    /// Configuration
    pub config: ApiConfig,
    /// Loaded indices by name (using IndexSlot for lock-free reads)
    indices: RwLock<HashMap<String, Arc<IndexSlot>>>,
    /// Cached index configurations to avoid repeated file reads
    index_configs: RwLock<HashMap<String, IndexConfigStored>>,
    /// Optional model pool for concurrent encoding
    #[cfg(feature = "model")]
    pub model_pool: Option<ModelPool>,
    /// Model configuration info (path, quantization status) - for logging
    #[cfg(feature = "model")]
    #[allow(dead_code)]
    pub model_info: Option<ModelInfo>,
    /// Shared PostgreSQL connection pool (for data, events, ingest endpoints)
    pub pg_pool: Option<sqlx::PgPool>,
}

impl AppState {
    /// Create a new application state (without model feature).
    #[cfg(not(feature = "model"))]
    pub fn new(config: ApiConfig) -> Self {
        // Ensure index directory exists
        if !config.index_dir.exists() {
            std::fs::create_dir_all(&config.index_dir).ok();
        }

        Self {
            config,
            indices: RwLock::new(HashMap::new()),
            index_configs: RwLock::new(HashMap::new()),
            pg_pool: None,
        }
    }

    /// Create a new application state with an optional model pool.
    #[cfg(feature = "model")]
    pub fn with_model_pool(
        config: ApiConfig,
        model_pool: Option<ModelPool>,
        model_info: Option<ModelInfo>,
    ) -> Self {
        // Ensure index directory exists
        if !config.index_dir.exists() {
            std::fs::create_dir_all(&config.index_dir).ok();
        }

        Self {
            config,
            indices: RwLock::new(HashMap::new()),
            index_configs: RwLock::new(HashMap::new()),
            model_pool,
            model_info,
            pg_pool: None,
        }
    }

    /// Set the PostgreSQL pool on this state.
    pub fn set_pg_pool(&mut self, pool: sqlx::PgPool) {
        self.pg_pool = Some(pool);
    }

    /// Check if model pool is available.
    #[cfg(feature = "model")]
    pub fn has_model(&self) -> bool {
        self.model_pool.is_some()
    }

    /// Get cached model info if model pool is available.
    #[cfg(feature = "model")]
    pub fn cached_model_info(&self) -> Option<&CachedModelInfo> {
        self.model_pool.as_ref().map(|p| &p.cached_info)
    }

    /// Get the path for an index by name.
    pub fn index_path(&self, name: &str) -> PathBuf {
        self.config.index_dir.join(name)
    }

    /// Check if an index exists on disk.
    pub fn index_exists_on_disk(&self, name: &str) -> bool {
        let path = self.index_path(name);
        path.join("metadata.json").exists()
    }

    /// Load an index from disk.
    pub fn load_index(&self, name: &str) -> ApiResult<()> {
        let path = self.index_path(name);
        let path_str = path.to_string_lossy().to_string();

        if !path.join("metadata.json").exists() {
            return Err(ApiError::IndexNotFound(name.to_string()));
        }

        let idx = MmapIndex::load(&path_str)?;

        let mut indices = self.indices.write();
        indices.insert(name.to_string(), Arc::new(IndexSlot::new(idx)));

        Ok(())
    }

    /// Get a loaded index slot by name (for write operations that need swapping).
    /// Returns the IndexSlot which supports atomic swapping.
    pub fn get_index_slot(&self, name: &str) -> ApiResult<Arc<IndexSlot>> {
        // First check if already loaded (fast path, no lock needed)
        {
            let indices = self.indices.read();
            if let Some(idx) = indices.get(name) {
                return Ok(Arc::clone(idx));
            }
        }

        // Acquire per-index loading lock to prevent concurrent loads.
        // This prevents race conditions where multiple requests try to load
        // the same index simultaneously, which can cause file conflicts.
        let loading_lock = get_loading_lock(name);
        let _guard = loading_lock.lock().unwrap();

        // Double-check: another request might have loaded it while we waited
        {
            let indices = self.indices.read();
            if let Some(idx) = indices.get(name) {
                return Ok(Arc::clone(idx));
            }
        }

        // Now safe to load from disk
        self.load_index(name)?;

        // Get the loaded index
        let indices = self.indices.read();
        indices
            .get(name)
            .cloned()
            .ok_or_else(|| ApiError::IndexNotFound(name.to_string()))
    }

    /// Get a lock-free reference to an index for read operations.
    /// This never blocks, even during write operations.
    pub fn get_index_for_read(&self, name: &str) -> ApiResult<Guard<Arc<MmapIndex>>> {
        let slot = self.get_index_slot(name)?;
        Ok(slot.load())
    }

    /// Register a new index (after creation).
    pub fn register_index(&self, name: &str, index: MmapIndex) {
        let mut indices = self.indices.write();
        // Check if slot already exists - if so, swap instead of replacing
        if let Some(slot) = indices.get(name) {
            slot.store(index);
        } else {
            indices.insert(name.to_string(), Arc::new(IndexSlot::new(index)));
        }
    }

    /// Unload an index from memory.
    pub fn unload_index(&self, name: &str) -> bool {
        let mut indices = self.indices.write();
        indices.remove(name).is_some()
    }

    /// Reload an index from disk using atomic swap.
    /// Readers continue with the old version during the load.
    pub fn reload_index(&self, name: &str) -> ApiResult<()> {
        let path = self.index_path(name);
        let path_str = path.to_string_lossy().to_string();

        if !path.join("metadata.json").exists() {
            return Err(ApiError::IndexNotFound(name.to_string()));
        }

        // Load the new index
        let new_idx = MmapIndex::load(&path_str)?;

        // Check if slot exists
        let indices = self.indices.read();
        if let Some(slot) = indices.get(name) {
            // Atomic swap - readers continue with old version
            slot.store(new_idx);
            Ok(())
        } else {
            // Need to create new slot
            drop(indices);
            let mut indices = self.indices.write();
            indices.insert(name.to_string(), Arc::new(IndexSlot::new(new_idx)));
            Ok(())
        }
    }

    /// Get cached index config, loading from disk if not cached.
    pub fn get_index_config(&self, name: &str) -> Option<IndexConfigStored> {
        // Check cache first
        {
            let configs = self.index_configs.read();
            if let Some(config) = configs.get(name) {
                return Some(config.clone());
            }
        }

        // Not cached - try to load from disk
        let config_path = self.index_path(name).join("config.json");
        let config = std::fs::File::open(&config_path)
            .ok()
            .and_then(|f| serde_json::from_reader::<_, IndexConfigStored>(f).ok())?;

        // Cache for future use
        {
            let mut configs = self.index_configs.write();
            configs.insert(name.to_string(), config.clone());
        }

        Some(config)
    }

    /// Set cached index config (and persist to disk).
    pub fn set_index_config(&self, name: &str, config: IndexConfigStored) -> ApiResult<()> {
        let config_path = self.index_path(name).join("config.json");

        // Persist to disk
        let config_file = std::fs::File::create(&config_path)
            .map_err(|e| ApiError::Internal(format!("Failed to create config file: {}", e)))?;
        serde_json::to_writer_pretty(config_file, &config)
            .map_err(|e| ApiError::Internal(format!("Failed to write config: {}", e)))?;

        // Update cache
        let mut configs = self.index_configs.write();
        configs.insert(name.to_string(), config);

        Ok(())
    }

    /// Invalidate cached config for an index.
    pub fn invalidate_config_cache(&self, name: &str) {
        let mut configs = self.index_configs.write();
        configs.remove(name);
    }

    /// List all indices (on disk).
    pub fn list_all(&self) -> Vec<String> {
        let mut names = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.config.index_dir) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let path = entry.path();
                    if path.join("metadata.json").exists() || path.join("config.json").exists() {
                        if let Some(name) = entry.file_name().to_str() {
                            names.push(name.to_string());
                        }
                    }
                }
            }
        }

        names.sort();
        names
    }

    /// Get the number of loaded indices.
    pub fn loaded_count(&self) -> usize {
        self.indices.read().len()
    }

    /// Get summary information for all indices on disk.
    pub fn get_all_index_summaries(&self) -> Vec<crate::models::IndexSummary> {
        let mut summaries = Vec::new();

        for name in self.list_all() {
            if let Ok(summary) = self.get_index_summary(&name) {
                summaries.push(summary);
            }
        }

        summaries
    }

    /// Get summary information for a specific index.
    pub fn get_index_summary(
        &self,
        name: &str,
    ) -> crate::error::ApiResult<crate::models::IndexSummary> {
        let path = self.index_path(name);
        let path_str = path.to_string_lossy().to_string();

        // Use cached config to get nbits and max_documents (avoids repeated file reads)
        let stored_config = self.get_index_config(name);
        let max_documents = stored_config.as_ref().and_then(|c| c.max_documents);
        let nbits = stored_config.as_ref().map(|c| c.nbits).unwrap_or(4);

        // Try to load metadata from disk (may not exist for empty/declared indices)
        let metadata_path = path.join("metadata.json");
        if !metadata_path.exists() {
            // Empty index (declared but no documents yet)
            return Ok(crate::models::IndexSummary {
                name: name.to_string(),
                num_documents: 0,
                num_embeddings: 0,
                num_partitions: 0,
                dimension: 0,
                nbits,
                avg_doclen: 0.0,
                has_metadata: false,
                max_documents,
            });
        }

        // Read metadata.json directly to avoid loading the full index
        let metadata_file = std::fs::File::open(&metadata_path).map_err(|e| {
            crate::error::ApiError::Internal(format!("Failed to open metadata: {}", e))
        })?;

        let metadata: serde_json::Value = serde_json::from_reader(metadata_file).map_err(|e| {
            crate::error::ApiError::Internal(format!("Failed to parse metadata: {}", e))
        })?;

        let has_metadata = next_plaid::filtering::exists(&path_str);

        Ok(crate::models::IndexSummary {
            name: name.to_string(),
            num_documents: metadata["num_documents"].as_u64().unwrap_or(0) as usize,
            num_embeddings: metadata["num_embeddings"].as_u64().unwrap_or(0) as usize,
            num_partitions: metadata["num_partitions"].as_u64().unwrap_or(0) as usize,
            dimension: metadata["embedding_dim"].as_u64().unwrap_or(0) as usize,
            nbits: metadata["nbits"].as_u64().unwrap_or(4) as usize,
            avg_doclen: metadata["avg_doclen"].as_f64().unwrap_or(0.0),
            has_metadata,
            max_documents,
        })
    }
}
