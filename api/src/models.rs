//! Request and response models for the next-plaid API.
//!
//! This module defines the JSON structures used for API communication.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// =============================================================================
// Index Management
// =============================================================================

/// Request to create/declare a new index.
///
/// This only declares the index with its configuration. Use the update endpoint
/// to add documents to the index.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateIndexRequest {
    /// Name/identifier for the index
    #[schema(example = "my_index")]
    pub name: String,
    /// Index configuration
    #[serde(default)]
    pub config: IndexConfigRequest,
}

/// Index configuration options.
#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct IndexConfigRequest {
    /// Number of bits for quantization (2 or 4, default: 4)
    #[serde(default)]
    #[schema(example = 4)]
    pub nbits: Option<usize>,
    /// Batch size for processing (default: 50000)
    #[serde(default)]
    #[schema(example = 50000)]
    pub batch_size: Option<usize>,
    /// Random seed for reproducibility
    #[serde(default)]
    #[schema(example = 42)]
    pub seed: Option<u64>,
    /// Threshold for rebuilding index from scratch (default: 999)
    /// When num_documents <= start_from_scratch, the index will be rebuilt
    /// entirely on updates instead of using incremental updates.
    #[serde(default)]
    #[schema(example = 999)]
    pub start_from_scratch: Option<usize>,
    /// Maximum number of documents to keep in the index (optional, None = unlimited)
    /// When the limit is exceeded after adding documents, oldest documents (lowest IDs) are evicted.
    #[serde(default)]
    #[schema(example = 10000)]
    pub max_documents: Option<usize>,
}

/// Response after declaring an index.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateIndexResponse {
    /// Index name
    #[schema(example = "my_index")]
    pub name: String,
    /// Index configuration that was stored
    pub config: IndexConfigStored,
    /// Message indicating next steps
    #[schema(example = "Index declared. Use POST /indices/{name}/update to add documents.")]
    pub message: String,
}

/// Stored index configuration.
#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct IndexConfigStored {
    /// Number of bits for quantization (2 or 4)
    #[schema(example = 4)]
    pub nbits: usize,
    /// Batch size for processing
    #[schema(example = 50000)]
    pub batch_size: usize,
    /// Random seed for reproducibility
    #[schema(example = 42)]
    pub seed: Option<u64>,
    /// Threshold for rebuilding index from scratch
    #[serde(default = "default_start_from_scratch")]
    #[schema(example = 999)]
    pub start_from_scratch: usize,
    /// Maximum number of documents to keep (None = unlimited)
    #[serde(default)]
    #[schema(example = 10000)]
    pub max_documents: Option<usize>,
}

fn default_start_from_scratch() -> usize {
    999
}

/// Index status/info response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IndexInfoResponse {
    /// Index name
    #[schema(example = "my_index")]
    pub name: String,
    /// Number of documents
    #[schema(example = 1000)]
    pub num_documents: usize,
    /// Number of embeddings (tokens)
    #[schema(example = 50000)]
    pub num_embeddings: usize,
    /// Number of centroids (partitions)
    #[schema(example = 512)]
    pub num_partitions: usize,
    /// Average document length
    #[schema(example = 50.0)]
    pub avg_doclen: f64,
    /// Embedding dimension
    #[schema(example = 128)]
    pub dimension: usize,
    /// Whether metadata database exists
    #[schema(example = true)]
    pub has_metadata: bool,
    /// Number of metadata entries (if metadata exists)
    #[schema(example = 1000)]
    pub metadata_count: Option<usize>,
    /// Maximum documents limit (None if unlimited)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = 10000)]
    pub max_documents: Option<usize>,
}

// =============================================================================
// Document Upload
// =============================================================================

/// Document embeddings with optional metadata.
#[derive(Debug, Deserialize, ToSchema)]
pub struct DocumentEmbeddings {
    /// Embedding matrix as nested array [num_tokens, dim]
    #[schema(example = json!([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))]
    pub embeddings: Vec<Vec<f32>>,
}

/// Request to add documents to an existing index.
#[derive(Debug, Deserialize, ToSchema)]
pub struct AddDocumentsRequest {
    /// List of document embeddings
    pub documents: Vec<DocumentEmbeddings>,
    /// Metadata for each document (must match documents length)
    pub metadata: Vec<serde_json::Value>,
}

/// Response after adding documents.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AddDocumentsResponse {
    /// Number of documents added
    #[schema(example = 10)]
    pub documents_added: usize,
    /// New total number of documents
    #[schema(example = 1010)]
    pub total_documents: usize,
    /// Starting document ID for new documents
    #[schema(example = 1000)]
    pub start_id: usize,
}

// =============================================================================
// Search
// =============================================================================

/// Query embeddings for search.
#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryEmbeddings {
    /// Embedding matrix as nested array [num_tokens, dim]
    #[schema(example = json!([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))]
    pub embeddings: Vec<Vec<f32>>,
}

/// Request to search the index.
#[derive(Debug, Deserialize, ToSchema)]
pub struct SearchRequest {
    /// Query embeddings (single query or batch)
    pub queries: Vec<QueryEmbeddings>,
    /// Search parameters
    #[serde(default)]
    pub params: SearchParamsRequest,
    /// Optional subset of document IDs to search within
    #[serde(default)]
    #[schema(example = json!([0, 5, 10, 15]))]
    pub subset: Option<Vec<i64>>,
}

/// Search parameters.
#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct SearchParamsRequest {
    /// Number of results to return per query (default: 10)
    #[serde(default)]
    #[schema(example = 10)]
    pub top_k: Option<usize>,
    /// Number of IVF cells to probe (default: 8)
    #[serde(default)]
    #[schema(example = 8)]
    pub n_ivf_probe: Option<usize>,
    /// Number of documents for exact re-ranking (default: 4096)
    #[serde(default)]
    #[schema(example = 4096)]
    pub n_full_scores: Option<usize>,
    /// Centroid score threshold for centroid pruning (default: None = disabled).
    /// Centroids with max score below this threshold are filtered out.
    /// Set to a float value (e.g., 0.4) to enable pruning for faster but potentially less accurate search.
    #[serde(default)]
    #[schema(example = 0.4)]
    pub centroid_score_threshold: Option<Option<f32>>,
}

/// Single query result.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QueryResultResponse {
    /// Query index in the batch
    #[schema(example = 0)]
    pub query_id: usize,
    /// Retrieved document IDs (ranked by relevance)
    #[schema(example = json!([42, 17, 89, 5]))]
    pub document_ids: Vec<i64>,
    /// Relevance scores for each document
    #[schema(example = json!([0.95, 0.87, 0.82, 0.75]))]
    pub scores: Vec<f32>,
    /// Metadata for each document (None if document has no metadata)
    #[schema(example = json!([{"title": "Doc 1", "category": "science"}, {"title": "Doc 2", "category": "history"}, null, {"title": "Doc 4"}]))]
    pub metadata: Vec<Option<serde_json::Value>>,
}

/// Response containing search results.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SearchResponse {
    /// Results for each query
    pub results: Vec<QueryResultResponse>,
    /// Total number of queries processed
    #[schema(example = 1)]
    pub num_queries: usize,
}

/// Request for filtered search combining metadata query and search.
#[derive(Debug, Deserialize, ToSchema)]
pub struct FilteredSearchRequest {
    /// Query embeddings
    pub queries: Vec<QueryEmbeddings>,
    /// Search parameters
    #[serde(default)]
    pub params: SearchParamsRequest,
    /// SQL WHERE condition for filtering
    #[schema(example = "category = ? AND score > ?")]
    pub filter_condition: String,
    /// Parameters for the filter condition
    #[serde(default)]
    #[schema(example = json!(["science", 90]))]
    pub filter_parameters: Vec<serde_json::Value>,
}

// =============================================================================
// Metadata
// =============================================================================

/// Request to check which documents have metadata.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CheckMetadataRequest {
    /// Document IDs to check
    #[schema(example = json!([0, 5, 10, 999]))]
    pub document_ids: Vec<i64>,
}

/// Response for metadata existence check.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CheckMetadataResponse {
    /// Document IDs that exist in the metadata database
    #[schema(example = json!([0, 5, 10]))]
    pub existing_ids: Vec<i64>,
    /// Document IDs that do not exist
    #[schema(example = json!([999]))]
    pub missing_ids: Vec<i64>,
    /// Total count of existing
    #[schema(example = 3)]
    pub existing_count: usize,
    /// Total count of missing
    #[schema(example = 1)]
    pub missing_count: usize,
}

/// Request to get metadata for documents.
#[derive(Debug, Deserialize, ToSchema)]
pub struct GetMetadataRequest {
    /// Optional document IDs to retrieve (if not provided, returns all)
    #[serde(default)]
    #[schema(example = json!([0, 5, 10]))]
    pub document_ids: Option<Vec<i64>>,
    /// Optional SQL WHERE condition
    #[serde(default)]
    #[schema(example = "category = ?")]
    pub condition: Option<String>,
    /// Parameters for the condition
    #[serde(default)]
    #[schema(example = json!(["science"]))]
    pub parameters: Vec<serde_json::Value>,
    /// Maximum number of results to return
    #[serde(default)]
    #[schema(example = 100)]
    pub limit: Option<usize>,
}

/// Response containing metadata.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GetMetadataResponse {
    /// Metadata entries
    #[schema(example = json!([{"_subset_": 0, "title": "Doc 1"}, {"_subset_": 1, "title": "Doc 2"}]))]
    pub metadata: Vec<serde_json::Value>,
    /// Number of entries returned
    #[schema(example = 2)]
    pub count: usize,
}

/// Request to query metadata with a SQL condition.
#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryMetadataRequest {
    /// SQL WHERE condition (e.g., "category = ? AND score > ?")
    #[schema(example = "category = ? AND score > ?")]
    pub condition: String,
    /// Parameters for the condition
    #[serde(default)]
    #[schema(example = json!(["science", 90]))]
    pub parameters: Vec<serde_json::Value>,
}

/// Response containing document IDs matching the condition.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct QueryMetadataResponse {
    /// Document IDs matching the condition
    #[schema(example = json!([0, 5, 42, 89]))]
    pub document_ids: Vec<i64>,
    /// Number of matching documents
    #[schema(example = 4)]
    pub count: usize,
}

/// Metadata count response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MetadataCountResponse {
    /// Number of metadata entries
    #[schema(example = 1000)]
    pub count: usize,
    /// Whether metadata database exists
    #[schema(example = true)]
    pub has_metadata: bool,
}

/// Request to update metadata using a SQL condition.
#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateMetadataRequest {
    /// SQL WHERE condition for selecting documents to update (e.g., "category = ? AND score > ?")
    #[schema(example = "category = ? AND score > ?")]
    pub condition: String,
    /// Parameters for the condition
    #[serde(default)]
    #[schema(example = json!(["science", 90]))]
    pub parameters: Vec<serde_json::Value>,
    /// JSON object with column names and new values to set
    #[schema(example = json!({"status": "reviewed", "updated_at": "2024-01-15"}))]
    pub updates: serde_json::Value,
}

/// Response after updating metadata.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct UpdateMetadataResponse {
    /// Number of rows updated
    #[schema(example = 5)]
    pub updated: usize,
}

// =============================================================================
// Delete
// =============================================================================

/// Request to delete documents by metadata filter.
///
/// Documents matching the SQL WHERE condition will be deleted from the index.
#[derive(Debug, Deserialize, ToSchema)]
pub struct DeleteDocumentsRequest {
    /// SQL WHERE condition for selecting documents to delete (e.g., "category = ? AND year < ?")
    #[schema(example = "category = ? AND year < ?")]
    pub condition: String,
    /// Parameters for the condition
    #[serde(default)]
    #[schema(example = json!(["outdated", 2020]))]
    pub parameters: Vec<serde_json::Value>,
}

/// Response after deleting documents.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DeleteDocumentsResponse {
    /// Number of documents deleted
    #[schema(example = 3)]
    pub deleted: usize,
    /// Remaining document count
    #[schema(example = 997)]
    pub remaining: usize,
}

/// Response after deleting an index.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DeleteIndexResponse {
    /// Whether deletion was successful
    #[schema(example = true)]
    pub deleted: bool,
    /// Name of deleted index
    #[schema(example = "my_index")]
    pub name: String,
}

// =============================================================================
// Health
// =============================================================================

/// Health check response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    /// Service status
    #[schema(example = "healthy")]
    pub status: String,
    /// API version
    #[schema(example = "0.1.0")]
    pub version: String,
    /// Number of loaded indices
    #[schema(example = 2)]
    pub loaded_indices: usize,
    /// Index directory path
    #[schema(example = "./indices")]
    pub index_dir: String,
    /// Memory usage of the API process in bytes
    #[schema(example = 104857600)]
    pub memory_usage_bytes: u64,
    /// List of available indices with their configuration
    pub indices: Vec<IndexSummary>,
    /// Model information (only present when --model is specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelHealthInfo>,
}

/// Model information for health endpoint.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelHealthInfo {
    /// Model name (from config)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = "GTE-ModernColBERT-v1")]
    pub name: Option<String>,
    /// Path to the model directory
    #[schema(example = "/models/GTE-ModernColBERT-v1")]
    pub path: String,
    /// Whether INT8 quantization is enabled
    #[schema(example = false)]
    pub quantized: bool,
    /// Embedding dimension
    #[schema(example = 128)]
    pub embedding_dim: usize,
    /// Batch size used for encoding
    #[schema(example = 32)]
    pub batch_size: usize,
    /// Number of parallel ONNX sessions
    #[schema(example = 1)]
    pub num_sessions: usize,
    /// Query prefix token
    #[schema(example = "[Q] ")]
    pub query_prefix: String,
    /// Document prefix token
    #[schema(example = "[D] ")]
    pub document_prefix: String,
    /// Maximum query length
    #[schema(example = 48)]
    pub query_length: usize,
    /// Maximum document length
    #[schema(example = 300)]
    pub document_length: usize,
    /// Whether query expansion is enabled
    #[schema(example = true)]
    pub do_query_expansion: bool,
    /// Whether the model uses token_type_ids
    #[schema(example = false)]
    pub uses_token_type_ids: bool,
    /// MASK token ID for query expansion
    #[schema(example = 103)]
    pub mask_token_id: u32,
    /// PAD token ID
    #[schema(example = 0)]
    pub pad_token_id: u32,
}

/// Summary information about an index.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct IndexSummary {
    /// Index name
    #[schema(example = "my_index")]
    pub name: String,
    /// Number of documents
    #[schema(example = 1000)]
    pub num_documents: usize,
    /// Number of embeddings (tokens)
    #[schema(example = 50000)]
    pub num_embeddings: usize,
    /// Number of centroids (partitions)
    #[schema(example = 512)]
    pub num_partitions: usize,
    /// Embedding dimension
    #[schema(example = 128)]
    pub dimension: usize,
    /// Number of bits for quantization (2 or 4)
    #[schema(example = 4)]
    pub nbits: usize,
    /// Average document length
    #[schema(example = 50.0)]
    pub avg_doclen: f64,
    /// Whether metadata database exists
    #[schema(example = true)]
    pub has_metadata: bool,
    /// Maximum documents limit (None if unlimited)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = 10000)]
    pub max_documents: Option<usize>,
}

/// Request to update an index by adding documents.
///
/// The index must have been declared first via `POST /indices`.
#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateIndexRequest {
    /// Document embeddings to add
    pub documents: Vec<DocumentEmbeddings>,
    /// Metadata for each document (must match documents length)
    pub metadata: Vec<serde_json::Value>,
}

/// Response after updating or creating an index.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct UpdateIndexResponse {
    /// Index name
    #[schema(example = "my_index")]
    pub name: String,
    /// Whether a new index was created (true) or existing index was updated (false)
    #[schema(example = false)]
    pub created: bool,
    /// Number of documents added
    #[schema(example = 10)]
    pub documents_added: usize,
    /// Total number of documents after update
    #[schema(example = 1010)]
    pub total_documents: usize,
    /// Number of embeddings (tokens)
    #[schema(example = 50500)]
    pub num_embeddings: usize,
    /// Number of centroids
    #[schema(example = 512)]
    pub num_partitions: usize,
    /// Embedding dimension
    #[schema(example = 128)]
    pub dimension: usize,
}

// =============================================================================
// Index Configuration
// =============================================================================

/// Request to update index configuration.
#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateIndexConfigRequest {
    /// New maximum documents limit (set to null to remove limit)
    #[schema(example = 5000)]
    pub max_documents: Option<usize>,
}

/// Response after updating index configuration.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct UpdateIndexConfigResponse {
    /// Index name
    #[schema(example = "my_index")]
    pub name: String,
    /// Updated configuration
    pub config: IndexConfigStored,
    /// Message about the update
    #[schema(
        example = "max_documents set to 5000. Eviction will occur on next document addition if over limit."
    )]
    pub message: String,
}

// =============================================================================
// Encoding (requires "model" feature)
// =============================================================================

/// Type of text input for encoding.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    /// Query text (uses query expansion with MASK tokens)
    Query,
    /// Document text (filters padding and skiplist tokens)
    Document,
}

/// Request to encode texts into embeddings.
#[derive(Debug, Deserialize, ToSchema)]
pub struct EncodeRequest {
    /// List of texts to encode (only used with "model" feature)
    #[allow(dead_code)]
    #[schema(example = json!(["Paris is the capital of France.", "What is machine learning?"]))]
    pub texts: Vec<String>,
    /// Type of input (query or document, only used with "model" feature)
    #[allow(dead_code)]
    #[schema(example = "document")]
    pub input_type: InputType,
    /// Optional pool factor for reducing document embeddings via hierarchical clustering.
    /// Only applies to document encoding (ignored for queries).
    /// When set, token embeddings are clustered and mean-pooled, reducing count by this factor.
    /// E.g., pool_factor=2 reduces ~100 tokens to ~50 embeddings.
    #[allow(dead_code)]
    #[serde(default)]
    #[schema(example = 2)]
    pub pool_factor: Option<usize>,
}

/// Response containing embeddings for encoded texts.
#[derive(Debug, Serialize, ToSchema)]
pub struct EncodeResponse {
    /// Embeddings for each text: \[num_texts\]\[num_tokens\]\[embedding_dim\]
    #[schema(example = json!([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]))]
    pub embeddings: Vec<Vec<Vec<f32>>>,
    /// Number of texts encoded
    #[schema(example = 2)]
    pub num_texts: usize,
}

/// Request to search using text queries (requires model to be loaded).
#[derive(Debug, Deserialize, ToSchema)]
pub struct SearchWithEncodingRequest {
    /// Text queries to search with
    #[schema(example = json!(["What is the capital of France?"]))]
    pub queries: Vec<String>,
    /// Search parameters
    #[serde(default)]
    pub params: SearchParamsRequest,
    /// Optional subset of document IDs to search within
    #[serde(default)]
    #[schema(example = json!([0, 5, 10, 15]))]
    pub subset: Option<Vec<i64>>,
}

/// Request for filtered search using text queries.
#[derive(Debug, Deserialize, ToSchema)]
pub struct FilteredSearchWithEncodingRequest {
    /// Text queries to search with
    #[schema(example = json!(["What is machine learning?"]))]
    pub queries: Vec<String>,
    /// Search parameters
    #[serde(default)]
    pub params: SearchParamsRequest,
    /// SQL WHERE condition for filtering
    #[schema(example = "category = ? AND score > ?")]
    pub filter_condition: String,
    /// Parameters for the filter condition
    #[serde(default)]
    #[schema(example = json!(["science", 90]))]
    pub filter_parameters: Vec<serde_json::Value>,
}

/// Request to update index with document texts (requires model to be loaded).
#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateWithEncodingRequest {
    /// Document texts to add
    #[schema(example = json!(["Paris is the capital of France.", "Machine learning is a type of AI."]))]
    pub documents: Vec<String>,
    /// Metadata for each document (must match documents length)
    #[schema(example = json!([{"title": "Geography"}, {"title": "Computer Science"}]))]
    pub metadata: Vec<serde_json::Value>,
    /// Optional pool factor for reducing document embeddings via hierarchical clustering.
    /// When set, token embeddings are clustered and mean-pooled, reducing count by this factor.
    /// E.g., pool_factor=2 reduces ~100 tokens to ~50 embeddings.
    #[serde(default)]
    #[schema(example = 2)]
    pub pool_factor: Option<usize>,
}

// =============================================================================
// Reranking
// =============================================================================

/// Request to rerank documents given a query using pre-computed embeddings.
///
/// Uses ColBERT's MaxSim scoring: for each query token, find the maximum similarity
/// with any document token, then sum these maximum similarities.
#[derive(Debug, Deserialize, ToSchema)]
pub struct RerankRequest {
    /// Query embeddings [num_tokens, dim]
    #[schema(example = json!([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))]
    pub query: Vec<Vec<f32>>,
    /// List of document embeddings, each [num_tokens, dim]
    pub documents: Vec<DocumentEmbeddings>,
}

/// Request to rerank documents using text inputs (requires model to be loaded).
///
/// The query and documents will be encoded using the loaded ColBERT model,
/// then scored using MaxSim.
#[derive(Debug, Deserialize, ToSchema)]
pub struct RerankWithEncodingRequest {
    /// Query text to encode (only used with "model" feature)
    #[allow(dead_code)]
    #[schema(example = "What is the capital of France?")]
    pub query: String,
    /// List of document texts to encode and rank (only used with "model" feature)
    #[allow(dead_code)]
    #[schema(example = json!(["Paris is the capital of France.", "Berlin is the capital of Germany."]))]
    pub documents: Vec<String>,
    /// Optional pool factor for reducing document embeddings via hierarchical clustering.
    /// When set, token embeddings are clustered and mean-pooled, reducing count by this factor.
    /// (only used with "model" feature)
    #[allow(dead_code)]
    #[serde(default)]
    #[schema(example = 2)]
    pub pool_factor: Option<usize>,
}

/// Single document result in reranking response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RerankResult {
    /// Original index of the document in the input list
    #[schema(example = 0)]
    pub index: usize,
    /// MaxSim score (sum of max similarities per query token)
    #[schema(example = 12.5)]
    pub score: f32,
}

/// Response containing reranked documents.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RerankResponse {
    /// Documents sorted by score in descending order
    pub results: Vec<RerankResult>,
    /// Number of documents reranked
    #[schema(example = 2)]
    pub num_documents: usize,
}

// =============================================================================
// Error
// =============================================================================

/// API error response.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// Error code for programmatic handling
    #[schema(example = "INDEX_NOT_FOUND")]
    pub code: String,
    /// Human-readable error message
    #[schema(example = "Index 'my_index' not found")]
    pub message: String,
    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}
