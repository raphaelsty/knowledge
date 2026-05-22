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

use next_plaid::{filtering, text_search, SearchParameters};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_texts_internal;
use crate::models::{
    ErrorResponse, FilteredSearchRequest, FilteredSearchWithEncodingRequest, InputType,
    QueryEmbeddings, QueryResultResponse, SearchRequest, SearchResponse, SearchWithEncodingRequest,
};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;
use crate::PrettyJson;

// Fusion algorithms are in next_plaid::text_search::{fuse_rrf, fuse_relative_score}

/// Convert query embeddings from JSON or base64 format to ndarray.
fn to_ndarray(query: &QueryEmbeddings) -> ApiResult<Array2<f32>> {
    // Prefer base64 if provided (more efficient)
    if let (Some(b64), Some(shape)) = (&query.embeddings_b64, &query.shape) {
        let floats =
            crate::models::decode_b64_embeddings(b64, *shape).map_err(ApiError::BadRequest)?;
        return Array2::from_shape_vec((shape[0], shape[1]), floats)
            .map_err(|e| ApiError::BadRequest(format!("Failed to create query array: {}", e)));
    }

    // Fall back to JSON array format
    let embeddings = query.embeddings.as_ref().ok_or_else(|| {
        ApiError::BadRequest(
            "Must provide either 'embeddings' or 'embeddings_b64' + 'shape'".to_string(),
        )
    })?;

    let rows = embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty query embeddings".to_string()));
    }

    let cols = embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension query embeddings".to_string(),
        ));
    }

    // Verify all rows have the same dimension
    for (i, row) in embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent query embedding dimension at row {}: expected {}, got {}",
                i,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
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
pub(crate) fn fetch_metadata_for_docs(
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

    let has_queries = req.queries.as_ref().map(|q| !q.is_empty()).unwrap_or(false);
    let has_text_query = req
        .text_query
        .as_ref()
        .map(|q| !q.is_empty())
        .unwrap_or(false);

    if !has_queries && !has_text_query {
        return Err(ApiError::BadRequest(
            "At least one of 'queries' (embeddings) or 'text_query' (keyword) must be provided"
                .to_string(),
        ));
    }

    let alpha = req.alpha.unwrap_or(0.75);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ApiError::BadRequest(
            "alpha must be between 0.0 and 1.0".to_string(),
        ));
    }

    let fusion_mode = req.fusion.as_deref().unwrap_or("rrf");
    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(ApiError::BadRequest(
            "fusion must be 'rrf' or 'relative_score'".to_string(),
        ));
    }

    // Hybrid mode: text_query is a single string, so queries must have exactly 1 element
    if has_queries && has_text_query {
        let queries_len = req.queries.as_ref().unwrap().len();
        if queries_len != 1 {
            return Err(ApiError::BadRequest(format!(
                "Hybrid search requires exactly 1 query embedding (got {}). \
                 text_query is a single string and can only fuse with one semantic query.",
                queries_len
            )));
        }
    }

    let top_k = req.params.top_k.unwrap_or(state.config.default_top_k);
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Resolve filter condition to subset
    let mut subset = req.subset.clone();
    if let Some(ref condition) = req.filter_condition {
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name.clone()));
        }
        let filter_params = req.filter_parameters.as_deref().unwrap_or(&[]);
        let filtered_ids = filtering::where_condition(&path_str, condition, filter_params)
            .map_err(|e| ApiError::BadRequest(format!("Invalid filter condition: {}", e)))?;
        subset = Some(filtered_ids);
    }

    // --- Pure semantic search (preserves batch query support) ---
    if has_queries && !has_text_query {
        let queries_vec = req.queries.as_ref().unwrap();
        let queries: Vec<Array2<f32>> = queries_vec
            .iter()
            .map(to_ndarray)
            .collect::<ApiResult<Vec<_>>>()?;

        let idx = state.get_index_for_read(&name)?;
        let expected_dim = idx.embedding_dim();
        for query in queries.iter() {
            if query.ncols() != expected_dim {
                return Err(ApiError::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.ncols(),
                });
            }
        }

        let params = SearchParameters {
            top_k,
            n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
            n_full_scores: req.params.n_full_scores.unwrap_or(4096),
            batch_size: 2000,
            centroid_score_threshold: req.params.centroid_score_threshold.unwrap_or_default(),
            ..Default::default()
        };

        let index = &**idx;
        let raw_results: Vec<(usize, Vec<i64>, Vec<f32>)> = if queries.len() == 1 {
            let r = index.search(&queries[0], &params, subset.as_deref())?;
            vec![(r.query_id, r.passage_ids, r.scores)]
        } else {
            let batch = index.search_batch(&queries, &params, true, subset.as_deref())?;
            batch
                .into_iter()
                .map(|r| (r.query_id, r.passage_ids, r.scores))
                .collect()
        };

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

        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            trace_id = %trace_id,
            index = %name,
            mode = "semantic",
            num_queries = queries.len(),
            top_k = top_k,
            total_results = total_results,
            total_ms = total_ms,
            "search.complete"
        );
        if total_ms > 1000 {
            tracing::warn!(trace_id = %trace_id, index = %name, total_ms = total_ms, "search.slow");
        }

        return Ok(PrettyJson(SearchResponse {
            num_queries: queries.len(),
            results,
        }));
    }

    // --- Keyword or hybrid search (supports batch) ---
    let empty_text: Vec<String> = vec![];
    let text_queries = req.text_query.as_ref().unwrap_or(&empty_text);
    let embedding_queries = req.queries.as_ref();

    // Validate: in hybrid mode, queries and text_query must have the same length
    if has_queries && has_text_query {
        let n_emb = embedding_queries.unwrap().len();
        let n_txt = text_queries.len();
        if n_emb != n_txt {
            return Err(ApiError::BadRequest(format!(
                "queries length ({}) must match text_query length ({}) in hybrid mode",
                n_emb, n_txt
            )));
        }
    }

    let num_queries = if has_text_query {
        text_queries.len()
    } else {
        embedding_queries.map(|q| q.len()).unwrap_or(0)
    };

    let fetch_k = if has_queries && has_text_query {
        top_k * 3
    } else {
        top_k
    };

    // Process each query
    let mut all_results: Vec<QueryResultResponse> = Vec::with_capacity(num_queries);

    #[allow(clippy::needless_range_loop)]
    for i in 0..num_queries {
        // Semantic component for this query
        let semantic: Option<(Vec<i64>, Vec<f32>)> = if has_queries {
            let query = to_ndarray(&embedding_queries.unwrap()[i])?;
            let idx = state.get_index_for_read(&name)?;
            let expected_dim = idx.embedding_dim();
            if query.ncols() != expected_dim {
                return Err(ApiError::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.ncols(),
                });
            }
            let params = SearchParameters {
                top_k: fetch_k,
                n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
                n_full_scores: req.params.n_full_scores.unwrap_or(4096),
                batch_size: 2000,
                centroid_score_threshold: req.params.centroid_score_threshold.unwrap_or_default(),
                ..Default::default()
            };
            let r = idx.search(&query, &params, subset.as_deref())?;
            Some((r.passage_ids, r.scores))
        } else {
            None
        };

        // Keyword component for this query
        let keyword: Option<(Vec<i64>, Vec<f32>)> = if has_text_query {
            let tq = &text_queries[i];
            let result = if let Some(ref sub) = subset {
                text_search::search_filtered(&path_str, tq, fetch_k, sub)
            } else {
                text_search::search(&path_str, tq, fetch_k)
            };
            match result {
                Ok(r) => Some((r.passage_ids, r.scores)),
                Err(e) => {
                    tracing::warn!(trace_id = %trace_id, index = %name, error = %e, "search.keyword.failed");
                    None
                }
            }
        } else {
            None
        };

        // Fuse
        let (document_ids, scores) = match (semantic, keyword) {
            (Some((sem_ids, sem_scores)), Some((kw_ids, kw_scores))) => match fusion_mode {
                "relative_score" => text_search::fuse_relative_score(
                    &sem_ids,
                    &sem_scores,
                    &kw_ids,
                    &kw_scores,
                    alpha,
                    top_k,
                ),
                _ => text_search::fuse_rrf(&sem_ids, &kw_ids, alpha, top_k),
            },
            (Some((ids, scores)), None) => {
                let mut r: Vec<(i64, f32)> = ids.into_iter().zip(scores).collect();
                r.truncate(top_k);
                (
                    r.iter().map(|x| x.0).collect(),
                    r.iter().map(|x| x.1).collect(),
                )
            }
            (None, Some((ids, scores))) => {
                let mut r: Vec<(i64, f32)> = ids.into_iter().zip(scores).collect();
                r.truncate(top_k);
                (
                    r.iter().map(|x| x.0).collect(),
                    r.iter().map(|x| x.1).collect(),
                )
            }
            (None, None) => (vec![], vec![]),
        };

        let metadata = fetch_metadata_for_docs(&path_str, &document_ids)?;
        all_results.push(QueryResultResponse {
            query_id: i,
            document_ids,
            scores,
            metadata,
        });
    }

    let total_results: usize = all_results.iter().map(|r| r.document_ids.len()).sum();
    let total_ms = start.elapsed().as_millis() as u64;

    let mode = if has_queries && has_text_query {
        "hybrid"
    } else {
        "keyword"
    };

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        mode = mode,
        num_queries = num_queries,
        top_k = top_k,
        total_results = total_results,
        total_ms = total_ms,
        "search.complete"
    );
    if total_ms > 1000 {
        tracing::warn!(trace_id = %trace_id, index = %name, total_ms = total_ms, "search.slow");
    }

    Ok(PrettyJson(SearchResponse {
        num_queries,
        results: all_results,
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
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Convert to unified SearchRequest with filter_condition
    let search_req = SearchRequest {
        queries: Some(req.queries),
        params: req.params,
        subset: None,
        text_query: None,
        alpha: None,
        fusion: None,
        filter_condition: Some(req.filter_condition),
        filter_parameters: Some(req.filter_parameters),
    };

    search(State(state), Path(name), trace_id, Json(search_req)).await
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
            embeddings: Some(arr.rows().into_iter().map(|r| r.to_vec()).collect()),
            embeddings_b64: None,
            shape: None,
        })
        .collect();

    // Create a standard SearchRequest (pass through hybrid fields)
    let search_req = SearchRequest {
        queries: Some(queries),
        params: req.params,
        subset: req.subset,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: None,
        filter_parameters: None,
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
            embeddings: Some(arr.rows().into_iter().map(|r| r.to_vec()).collect()),
            embeddings_b64: None,
            shape: None,
        })
        .collect();

    // Create a unified SearchRequest with filter (pass through hybrid fields)
    let search_req = SearchRequest {
        queries: Some(queries),
        params: req.params,
        subset: None,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: Some(req.filter_condition.clone()),
        filter_parameters: Some(req.filter_parameters),
    };

    // Delegate to the unified search handler
    let result = search(State(state), Path(name.clone()), trace_id, Json(search_req)).await;

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
