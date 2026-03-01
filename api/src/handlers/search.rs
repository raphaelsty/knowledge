//! Search handlers.
//!
//! Handles search operations on indices.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Extension, Json,
};
use ndarray::Array2;

use next_plaid::{filtering, SearchParameters};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_texts_internal;
use crate::models::{
    ErrorResponse, FilteredSearchRequest, FilteredSearchWithEncodingRequest, InputType,
    QueryEmbeddings, QueryResultResponse, SearchRequest, SearchResponse, SearchWithEncodingRequest,
};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;
use crate::PrettyJson;

/// Convert query embeddings from JSON format to ndarray.
fn to_ndarray(query: &QueryEmbeddings) -> ApiResult<Array2<f32>> {
    let rows = query.embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty query embeddings".to_string()));
    }

    let cols = query.embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension query embeddings".to_string(),
        ));
    }

    // Verify all rows have the same dimension
    for (i, row) in query.embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent query embedding dimension at row {}: expected {}, got {}",
                i,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = query.embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| ApiError::BadRequest(format!("Failed to create query array: {}", e)))
}

/// Fetch metadata for a list of document IDs.
/// Returns a Vec of Option<serde_json::Value> in the same order as document_ids.
/// If metadata doesn't exist for an index or a specific document, returns None for that entry.
///
/// # Errors
/// Returns an error if the metadata database exists but fails to query.
/// If no metadata database exists, returns Ok with None for all entries (not an error).
fn fetch_metadata_for_docs(
    path_str: &str,
    document_ids: &[i64],
) -> ApiResult<Vec<Option<serde_json::Value>>> {
    if !filtering::exists(path_str) {
        // No metadata database - return None for all (this is not an error)
        return Ok(vec![None; document_ids.len()]);
    }

    // Fetch metadata for the document IDs
    let metadata_list = filtering::get(path_str, None, &[], Some(document_ids)).map_err(|e| {
        tracing::error!("Failed to fetch metadata from database: {}", e);
        ApiError::Internal(format!("Failed to fetch metadata: {}", e))
    })?;

    // Build a map from _subset_ to metadata for quick lookup
    let meta_map: HashMap<i64, serde_json::Value> = metadata_list
        .into_iter()
        .filter_map(|m| m.get("_subset_").and_then(|v| v.as_i64()).map(|id| (id, m)))
        .collect();

    // Map document_ids to their metadata (or None if not found)
    Ok(document_ids
        .iter()
        .map(|doc_id| meta_map.get(doc_id).cloned())
        .collect())
}

/// Search an index with query embeddings.
#[utoipa::path(
    post,
    path = "/indices/{name}/search",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Convert queries to ndarrays
    let queries: Vec<Array2<f32>> = req
        .queries
        .iter()
        .map(to_ndarray)
        .collect::<ApiResult<Vec<_>>>()?;

    // Get index (lock-free read - never blocks during writes)
    let idx = state.get_index_for_read(&name)?;

    // Validate query dimensions
    let expected_dim = idx.embedding_dim();
    for query in queries.iter() {
        if query.ncols() != expected_dim {
            return Err(ApiError::DimensionMismatch {
                expected: expected_dim,
                actual: query.ncols(),
            });
        }
    }

    // Build search parameters
    let top_k = req.params.top_k.unwrap_or(state.config.default_top_k);
    let params = SearchParameters {
        top_k,
        n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
        n_full_scores: req.params.n_full_scores.unwrap_or(4096),
        batch_size: 2000,
        // Use provided threshold, or default (None = disabled) if not specified.
        // The Option<Option<f32>> allows distinguishing "not provided" from "explicitly null".
        // None (not provided) -> None (disabled), Some(None) (explicit null) -> None, Some(Some(x)) -> Some(x)
        centroid_score_threshold: req.params.centroid_score_threshold.unwrap_or_default(),
        ..Default::default()
    };

    // Get path for metadata lookup
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Perform search and collect raw results
    let index = &**idx;
    let index_query_start = std::time::Instant::now();
    let raw_results: Vec<(usize, Vec<i64>, Vec<f32>)> = if queries.len() == 1 {
        let result = index.search(&queries[0], &params, req.subset.as_deref())?;
        vec![(result.query_id, result.passage_ids, result.scores)]
    } else {
        let batch_results = index.search_batch(&queries, &params, true, req.subset.as_deref())?;
        batch_results
            .into_iter()
            .map(|r| (r.query_id, r.passage_ids, r.scores))
            .collect()
    };
    let index_query_ms = index_query_start.elapsed().as_millis() as u64;

    // Enrich results with metadata
    let metadata_fetch_start = std::time::Instant::now();
    let total_results: usize = raw_results.iter().map(|(_, ids, _)| ids.len()).sum();
    let results: Vec<QueryResultResponse> = raw_results
        .into_iter()
        .map(|(query_id, document_ids, scores)| {
            let metadata = fetch_metadata_for_docs(&path_str, &document_ids)?;
            Ok(QueryResultResponse {
                query_id,
                document_ids,
                scores,
                metadata,
            })
        })
        .collect::<ApiResult<Vec<_>>>()?;
    let metadata_fetch_ms = metadata_fetch_start.elapsed().as_millis() as u64;

    let total_ms = start.elapsed().as_millis() as u64;

    // Single comprehensive completion log
    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        num_queries = queries.len(),
        top_k = top_k,
        total_results = total_results,
        index_query_ms = index_query_ms,
        metadata_fetch_ms = metadata_fetch_ms,
        total_ms = total_ms,
        "search.complete"
    );

    // Warn on slow searches (>1s)
    if total_ms > 1000 {
        tracing::warn!(
            trace_id = %trace_id,
            index = %name,
            total_ms = total_ms,
            "search.slow"
        );
    }

    Ok(PrettyJson(SearchResponse {
        num_queries: queries.len(),
        results,
    }))
}

/// Search with a pre-filtered subset from metadata query.
///
/// This is a convenience endpoint that combines metadata filtering and search.
#[utoipa::path(
    post,
    path = "/indices/{name}/search/filtered",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = FilteredSearchRequest,
    responses(
        (status = 200, description = "Filtered search results", body = SearchResponse),
        (status = 400, description = "Invalid request or filter condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn search_filtered(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<FilteredSearchRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_ext = trace_id.clone();
    let trace_id_val = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Get the filtered subset first
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    if !next_plaid::filtering::exists(&path_str) {
        return Err(ApiError::MetadataNotFound(name.clone()));
    }

    let sql_filter_start = std::time::Instant::now();
    let subset = next_plaid::filtering::where_condition(
        &path_str,
        &req.filter_condition,
        &req.filter_parameters,
    )
    .map_err(|e| ApiError::BadRequest(format!("Invalid filter condition: {}", e)))?;
    let sql_filter_ms = sql_filter_start.elapsed().as_millis() as u64;
    let matching_docs = subset.len();

    // Convert to standard search request with subset
    let search_req = SearchRequest {
        queries: req.queries,
        params: req.params,
        subset: Some(subset),
    };

    // Delegate to normal search
    let result = search(
        State(state),
        Path(name.clone()),
        trace_id_ext,
        Json(search_req),
    )
    .await;

    let total_ms = start.elapsed().as_millis() as u64;

    // Log filtered search completion with filter-specific metrics
    tracing::info!(
        trace_id = %trace_id_val,
        index = %name,
        filter = %req.filter_condition,
        matching_docs = matching_docs,
        sql_filter_ms = sql_filter_ms,
        total_ms = total_ms,
        "search.filtered.complete"
    );

    result
}

/// Search an index using text queries (requires model to be loaded).
///
/// This endpoint encodes the text queries using the loaded model and then performs a search.
/// Requires the server to be started with `--model <path>`.
#[utoipa::path(
    post,
    path = "/indices/{name}/search_with_encoding",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = SearchWithEncodingRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request or model not loaded", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn search_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<SearchWithEncodingRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_val = trace_id.as_ref().map(|t| t.0.clone()).unwrap_or_default();
    let start = std::time::Instant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let num_queries = req.queries.len();

    // Encode the text queries (async, uses batch queue)
    let encode_start = std::time::Instant::now();
    let query_embeddings =
        encode_texts_internal(state.clone(), &req.queries, InputType::Query, None).await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    // Convert to QueryEmbeddings format
    let queries: Vec<QueryEmbeddings> = query_embeddings
        .into_iter()
        .map(|arr| QueryEmbeddings {
            embeddings: arr.rows().into_iter().map(|r| r.to_vec()).collect(),
        })
        .collect();

    // Create a standard SearchRequest
    let search_req = SearchRequest {
        queries,
        params: req.params,
        subset: req.subset,
    };

    // Delegate to the standard search
    let result = search(State(state), Path(name.clone()), trace_id, Json(search_req)).await;

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id_val,
        index = %name,
        num_queries = num_queries,
        encode_ms = encode_ms,
        total_ms = total_ms,
        "search.with_encoding.complete"
    );

    result
}

/// Search with text queries and a metadata filter (requires model to be loaded).
///
/// This endpoint encodes the text queries using the loaded model and performs a filtered search.
/// Requires the server to be started with `--model <path>`.
#[utoipa::path(
    post,
    path = "/indices/{name}/search/filtered_with_encoding",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = FilteredSearchWithEncodingRequest,
    responses(
        (status = 200, description = "Filtered search results", body = SearchResponse),
        (status = 400, description = "Invalid request, model not loaded, or filter condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn search_filtered_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<FilteredSearchWithEncodingRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_val = trace_id.as_ref().map(|t| t.0.clone()).unwrap_or_default();
    let start = std::time::Instant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let num_queries = req.queries.len();

    // Encode the text queries (async, uses batch queue)
    let encode_start = std::time::Instant::now();
    let query_embeddings =
        encode_texts_internal(state.clone(), &req.queries, InputType::Query, None).await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    // Convert to QueryEmbeddings format
    let queries: Vec<QueryEmbeddings> = query_embeddings
        .into_iter()
        .map(|arr| QueryEmbeddings {
            embeddings: arr.rows().into_iter().map(|r| r.to_vec()).collect(),
        })
        .collect();

    // Create a FilteredSearchRequest
    let filtered_req = FilteredSearchRequest {
        queries,
        params: req.params,
        filter_condition: req.filter_condition.clone(),
        filter_parameters: req.filter_parameters,
    };

    // Delegate to the filtered search
    let result = search_filtered(
        State(state),
        Path(name.clone()),
        trace_id,
        Json(filtered_req),
    )
    .await;

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id_val,
        index = %name,
        num_queries = num_queries,
        filter = %req.filter_condition,
        encode_ms = encode_ms,
        total_ms = total_ms,
        "search.filtered_with_encoding.complete"
    );

    result
}
