//! Metadata handlers.
//!
//! Handles metadata operations: check, query, and get.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Extension, Json,
};
use tokio::task;

use next_plaid::filtering;

use crate::error::{ApiError, ApiResult};
use crate::models::{
    CheckMetadataRequest, CheckMetadataResponse, ErrorResponse, GetMetadataRequest,
    GetMetadataResponse, MetadataCountResponse, QueryMetadataRequest, QueryMetadataResponse,
    UpdateMetadataRequest, UpdateMetadataResponse,
};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;

/// Check if specific documents exist in the metadata database.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/check",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = CheckMetadataRequest,
    responses(
        (status = 200, description = "Metadata existence check result", body = CheckMetadataResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn check_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<CheckMetadataRequest>,
) -> ApiResult<Json<CheckMetadataResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Fast path: if no IDs requested, return empty result
    if req.document_ids.is_empty() {
        return Ok(Json(CheckMetadataResponse {
            existing_count: 0,
            missing_count: 0,
            existing_ids: Vec::new(),
            missing_ids: Vec::new(),
        }));
    }

    // Run blocking SQLite operations in a separate thread
    let document_ids = req.document_ids.clone();
    let num_ids = document_ids.len();
    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err("metadata_not_found".to_string());
        }

        // Query only the requested IDs (O(k) instead of O(n) where n is total metadata entries)
        let sql_query_start = std::time::Instant::now();
        let found_metadata = filtering::get(&path_str, None, &[], Some(&document_ids))
            .map_err(|e| format!("Failed to query metadata: {}", e))?;
        let sql_query_ms = sql_query_start.elapsed().as_millis() as u64;

        // Extract the IDs that were actually found
        let existing_set: std::collections::HashSet<i64> = found_metadata
            .iter()
            .filter_map(|m| m.get("_subset_").and_then(|v| v.as_i64()))
            .collect();

        // Partition requested IDs into existing and missing
        let mut existing_ids = Vec::with_capacity(existing_set.len());
        let mut missing_ids =
            Vec::with_capacity(document_ids.len().saturating_sub(existing_set.len()));

        for &doc_id in &document_ids {
            if existing_set.contains(&doc_id) {
                existing_ids.push(doc_id);
            } else {
                missing_ids.push(doc_id);
            }
        }

        Ok((existing_ids, missing_ids, sql_query_ms))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    match result {
        Ok((existing_ids, missing_ids, sql_query_ms)) => {
            let total_ms = start.elapsed().as_millis() as u64;
            tracing::info!(
                trace_id = %trace_id,
                index = %name,
                num_ids = num_ids,
                existing_count = existing_ids.len(),
                missing_count = missing_ids.len(),
                sql_query_ms = sql_query_ms,
                total_ms = total_ms,
                "metadata.check.complete"
            );
            Ok(Json(CheckMetadataResponse {
                existing_count: existing_ids.len(),
                missing_count: missing_ids.len(),
                existing_ids,
                missing_ids,
            }))
        }
        Err(e) if e == "metadata_not_found" => Err(ApiError::MetadataNotFound(name)),
        Err(e) => Err(ApiError::Internal(e)),
    }
}

/// Query metadata to get document IDs matching a SQL condition.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/query",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = QueryMetadataRequest,
    responses(
        (status = 200, description = "Document IDs matching the condition", body = QueryMetadataResponse),
        (status = 400, description = "Invalid SQL condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn query_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<QueryMetadataRequest>,
) -> ApiResult<Json<QueryMetadataResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Run blocking SQLite operations in a separate thread
    let condition = req.condition.clone();
    let parameters = req.parameters.clone();
    let name_clone = name.clone();
    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        // Query metadata
        let sql_query_start = std::time::Instant::now();
        let document_ids = filtering::where_condition(&path_str, &condition, &parameters)
            .map_err(|e| ApiError::BadRequest(format!("Invalid condition: {}", e)))?;
        let sql_query_ms = sql_query_start.elapsed().as_millis() as u64;
        Ok((document_ids, sql_query_ms))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let (document_ids, sql_query_ms) = result?;
    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        condition = %req.condition,
        num_results = document_ids.len(),
        sql_query_ms = sql_query_ms,
        total_ms = total_ms,
        "metadata.query.complete"
    );

    Ok(Json(QueryMetadataResponse {
        count: document_ids.len(),
        document_ids,
    }))
}

/// Get metadata for specific documents by ID or SQL condition.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/get",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = GetMetadataRequest,
    responses(
        (status = 200, description = "Metadata entries", body = GetMetadataResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn get_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<GetMetadataRequest>,
) -> ApiResult<Json<GetMetadataResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Cannot use both document_ids and condition
    if req.document_ids.is_some() && req.condition.is_some() {
        return Err(ApiError::BadRequest(
            "Cannot specify both document_ids and condition".to_string(),
        ));
    }

    // Run blocking SQLite operations in a separate thread
    let condition = req.condition.clone();
    let parameters = req.parameters.clone();
    let document_ids = req.document_ids.clone();
    let limit = req.limit;
    let name_clone = name.clone();

    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        let sql_query_start = std::time::Instant::now();
        let mut metadata = filtering::get(
            &path_str,
            condition.as_deref(),
            &parameters,
            document_ids.as_deref(),
        )
        .map_err(|e| ApiError::Internal(format!("Failed to get metadata: {}", e)))?;
        let sql_query_ms = sql_query_start.elapsed().as_millis() as u64;

        // Apply limit if specified
        if let Some(limit) = limit {
            metadata.truncate(limit);
        }

        Ok((metadata, sql_query_ms))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let (metadata, sql_query_ms) = result?;
    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        num_results = metadata.len(),
        sql_query_ms = sql_query_ms,
        total_ms = total_ms,
        "metadata.get.complete"
    );

    Ok(Json(GetMetadataResponse {
        count: metadata.len(),
        metadata,
    }))
}

/// Get all metadata entries for an index.
#[utoipa::path(
    get,
    path = "/indices/{name}/metadata",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "All metadata entries", body = GetMetadataResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn get_all_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
) -> ApiResult<Json<GetMetadataResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Run blocking SQLite operations in a separate thread
    let name_clone = name.clone();
    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        let sql_query_start = std::time::Instant::now();
        let metadata = filtering::get(&path_str, None, &[], None)
            .map_err(|e| ApiError::Internal(format!("Failed to get metadata: {}", e)))?;
        let sql_query_ms = sql_query_start.elapsed().as_millis() as u64;
        Ok((metadata, sql_query_ms))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let (metadata, sql_query_ms) = result?;
    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        num_results = metadata.len(),
        sql_query_ms = sql_query_ms,
        total_ms = total_ms,
        "metadata.get_all.complete"
    );

    Ok(Json(GetMetadataResponse {
        count: metadata.len(),
        metadata,
    }))
}

/// Get the count of metadata entries for an index.
#[utoipa::path(
    get,
    path = "/indices/{name}/metadata/count",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    responses(
        (status = 200, description = "Metadata count", body = MetadataCountResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn get_metadata_count(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
) -> ApiResult<Json<MetadataCountResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Run blocking SQLite operations in a separate thread
    let (has_metadata, count, sql_query_ms) = task::spawn_blocking(move || {
        let has_metadata = filtering::exists(&path_str);
        let (count, sql_query_ms) = if has_metadata {
            let sql_query_start = std::time::Instant::now();
            let count = filtering::count(&path_str).unwrap_or(0);
            let sql_query_ms = sql_query_start.elapsed().as_millis() as u64;
            (count, sql_query_ms)
        } else {
            (0, 0)
        };
        (has_metadata, count, sql_query_ms)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        count = count,
        has_metadata = has_metadata,
        sql_query_ms = sql_query_ms,
        total_ms = total_ms,
        "metadata.count.complete"
    );

    Ok(Json(MetadataCountResponse {
        count,
        has_metadata,
    }))
}

/// Update metadata rows matching a SQL condition.
///
/// This endpoint updates existing metadata rows that match the given condition.
/// The updates are provided as a JSON object where keys are column names and values
/// are the new values to set.
#[utoipa::path(
    post,
    path = "/indices/{name}/metadata/update",
    tag = "metadata",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = UpdateMetadataRequest,
    responses(
        (status = 200, description = "Metadata updated successfully", body = UpdateMetadataResponse),
        (status = 400, description = "Invalid request (bad condition or updates)", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse),
        (status = 429, description = "Rate limit exceeded")
    )
)]
pub async fn update_metadata(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<UpdateMetadataRequest>,
) -> ApiResult<Json<UpdateMetadataResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Check if index exists
    if !state.index_exists_on_disk(&name) {
        return Err(ApiError::IndexNotFound(name));
    }

    // Validate that updates is an object
    if !req.updates.is_object() {
        return Err(ApiError::BadRequest(
            "Updates must be a JSON object".to_string(),
        ));
    }

    // Run blocking SQLite operations in a separate thread
    let condition = req.condition.clone();
    let parameters = req.parameters.clone();
    let updates = req.updates.clone();
    let name_clone = name.clone();

    let result = task::spawn_blocking(move || {
        // Check if metadata database exists
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name_clone));
        }

        let sql_update_start = std::time::Instant::now();
        let updated = filtering::update_where(&path_str, &condition, &parameters, &updates)
            .map_err(|e| ApiError::BadRequest(format!("Update failed: {}", e)))?;
        let sql_update_ms = sql_update_start.elapsed().as_millis() as u64;
        Ok((updated, sql_update_ms))
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task failed: {}", e)))?;

    let (updated, sql_update_ms) = result?;
    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        condition = %req.condition,
        rows_updated = updated,
        sql_update_ms = sql_update_ms,
        total_ms = total_ms,
        "metadata.update.complete"
    );

    Ok(Json(UpdateMetadataResponse { updated }))
}
