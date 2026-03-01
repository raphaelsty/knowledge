//! Rerank endpoint handlers for the next-plaid API.
//!
//! Provides document reranking using ColBERT's MaxSim scoring:
//! For each query token, find the maximum similarity with any document token,
//! then sum these maximum similarities.

use std::sync::Arc;

use axum::{extract::State, Extension, Json};
use ndarray::Array2;

use crate::error::{ApiError, ApiResult};
use crate::models::{RerankRequest, RerankResponse, RerankResult};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;
use crate::PrettyJson;

/// Convert a Vec<Vec<f32>> to an ndarray::Array2<f32>.
fn to_ndarray(embeddings: &[Vec<f32>]) -> ApiResult<Array2<f32>> {
    if embeddings.is_empty() {
        return Err(ApiError::BadRequest(
            "Empty embeddings provided".to_string(),
        ));
    }

    let rows = embeddings.len();
    let cols = embeddings[0].len();

    // Validate all rows have same dimension
    for (i, row) in embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent embedding dimensions: row 0 has {} elements, row {} has {}",
                cols,
                i,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| ApiError::Internal(format!("Failed to create ndarray: {}", e)))
}

/// Compute ColBERT MaxSim score between a query and a document.
///
/// For each query token, find the maximum cosine similarity with any document token,
/// then sum these maximum similarities.
///
/// Assumes embeddings are already L2-normalized (as ColBERT models produce).
fn compute_maxsim(query: &Array2<f32>, document: &Array2<f32>) -> f32 {
    let mut total_score = 0.0f32;

    // For each query token
    for query_row in query.rows() {
        let mut max_sim = f32::NEG_INFINITY;

        // Find max similarity with any document token
        for doc_row in document.rows() {
            // Dot product (cosine similarity for normalized vectors)
            let sim: f32 = query_row
                .iter()
                .zip(doc_row.iter())
                .map(|(q, d)| q * d)
                .sum();
            if sim > max_sim {
                max_sim = sim;
            }
        }

        // Sum the max similarities
        if max_sim > f32::NEG_INFINITY {
            total_score += max_sim;
        }
    }

    total_score
}

/// Rerank documents given pre-computed query and document embeddings.
///
/// Uses ColBERT's MaxSim scoring: for each query token, find the maximum
/// similarity with any document token, then sum these maximum similarities.
#[utoipa::path(
    post,
    path = "/rerank",
    tag = "reranking",
    request_body = RerankRequest,
    responses(
        (status = 200, description = "Documents reranked successfully", body = RerankResponse),
        (status = 400, description = "Invalid request (empty or mismatched dimensions)"),
    )
)]
pub async fn rerank(
    State(_state): State<Arc<AppState>>,
    trace_id: Option<Extension<TraceId>>,
    Json(request): Json<RerankRequest>,
) -> ApiResult<PrettyJson<RerankResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    // Validate request
    if request.query.is_empty() {
        return Err(ApiError::BadRequest("Empty query embeddings".to_string()));
    }
    if request.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    // Convert query to ndarray
    let query = to_ndarray(&request.query)?;
    let query_dim = query.ncols();
    let query_tokens = query.nrows();

    // Convert all documents and validate dimensions
    let documents: Vec<Array2<f32>> = request
        .documents
        .iter()
        .map(|doc| {
            let arr = to_ndarray(&doc.embeddings)?;
            if arr.ncols() != query_dim {
                return Err(ApiError::DimensionMismatch {
                    expected: query_dim,
                    actual: arr.ncols(),
                });
            }
            Ok(arr)
        })
        .collect::<ApiResult<Vec<_>>>()?;

    let num_documents = documents.len();

    // Compute MaxSim scores for all documents
    let scoring_start = std::time::Instant::now();
    let mut results: Vec<RerankResult> = documents
        .iter()
        .enumerate()
        .map(|(index, doc)| {
            let score = compute_maxsim(&query, doc);
            RerankResult { index, score }
        })
        .collect();
    let scoring_ms = scoring_start.elapsed().as_millis() as u64;

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        num_documents = num_documents,
        query_tokens = query_tokens,
        scoring_ms = scoring_ms,
        total_ms = total_ms,
        "rerank.complete"
    );

    Ok(PrettyJson(RerankResponse {
        results,
        num_documents,
    }))
}

/// Rerank documents given text inputs (requires model to be loaded).
///
/// The query and documents will be encoded using the loaded ColBERT model,
/// then scored using MaxSim.
#[cfg(feature = "model")]
#[utoipa::path(
    post,
    path = "/rerank_with_encoding",
    tag = "reranking",
    request_body = crate::models::RerankWithEncodingRequest,
    responses(
        (status = 200, description = "Documents reranked successfully", body = RerankResponse),
        (status = 400, description = "Model not loaded or invalid request"),
        (status = 500, description = "Encoding failed")
    )
)]
pub async fn rerank_with_encoding(
    State(state): State<Arc<AppState>>,
    trace_id: Option<Extension<TraceId>>,
    Json(request): Json<crate::models::RerankWithEncodingRequest>,
) -> ApiResult<PrettyJson<RerankResponse>> {
    use crate::handlers::encode::encode_texts_internal;
    use crate::models::InputType;

    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    // Validate request
    if request.query.is_empty() {
        return Err(ApiError::BadRequest("Empty query text".to_string()));
    }
    if request.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    let num_documents = request.documents.len();

    // Check if model is loaded
    if !state.has_model() {
        return Err(ApiError::ModelNotLoaded);
    }

    // Encode query
    let encode_start = std::time::Instant::now();
    let query_texts = vec![request.query];
    let query_embeddings = encode_texts_internal(
        state.clone(),
        &query_texts,
        InputType::Query,
        None, // No pool factor for queries
    )
    .await?;

    let query = query_embeddings
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Internal("Failed to encode query".to_string()))?;

    // Encode documents
    let doc_embeddings = encode_texts_internal(
        state,
        &request.documents,
        InputType::Document,
        request.pool_factor,
    )
    .await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    // Compute MaxSim scores for all documents
    let scoring_start = std::time::Instant::now();
    let mut results: Vec<RerankResult> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(index, doc)| {
            let score = compute_maxsim(&query, doc);
            RerankResult { index, score }
        })
        .collect();
    let scoring_ms = scoring_start.elapsed().as_millis() as u64;

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        num_documents = num_documents,
        encode_ms = encode_ms,
        scoring_ms = scoring_ms,
        total_ms = total_ms,
        "rerank.with_encoding.complete"
    );

    let result_count = results.len();

    Ok(PrettyJson(RerankResponse {
        results,
        num_documents: result_count,
    }))
}

/// Stub rerank_with_encoding function when model feature is not enabled.
#[cfg(not(feature = "model"))]
#[utoipa::path(
    post,
    path = "/rerank_with_encoding",
    tag = "reranking",
    request_body = crate::models::RerankWithEncodingRequest,
    responses(
        (status = 400, description = "Model support not compiled"),
    )
)]
pub async fn rerank_with_encoding(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<crate::models::RerankWithEncodingRequest>,
) -> ApiResult<PrettyJson<RerankResponse>> {
    Err(ApiError::ModelNotLoaded)
}
