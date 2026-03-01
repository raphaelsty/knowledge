//! Error handling for the next_plaid API.
//!
//! Provides a unified error type that can be converted to HTTP responses
//! with appropriate status codes and JSON error bodies.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

/// API error type that maps to HTTP responses.
#[derive(Error, Debug)]
pub enum ApiError {
    /// Index not found or not loaded
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    /// Index already exists
    #[error("Index already exists: {0}")]
    IndexAlreadyExists(String),

    /// Index not declared (must call create first)
    #[error("Index not declared: {0}. Call POST /indices first to declare the index.")]
    IndexNotDeclared(String),

    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    BadRequest(String),

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Metadata database not found
    #[error("Metadata database not found for index: {0}")]
    MetadataNotFound(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Service temporarily unavailable (e.g., queue full)
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Model not loaded (encoding endpoints require --model flag)
    #[error("Model not loaded. Start the server with --model <path> to enable encoding.")]
    ModelNotLoaded,

    /// Model encoding error (only used with "model" feature)
    #[error("Model error: {0}")]
    #[allow(dead_code)]
    ModelError(String),

    /// NextPlaid library error
    #[error("Next-Plaid error: {0}")]
    NextPlaid(#[from] next_plaid::Error),
}

/// JSON error response body.
#[derive(Serialize)]
pub struct ErrorResponse {
    /// Error code for programmatic handling
    pub code: &'static str,
    /// Human-readable error message
    pub message: String,
    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ApiError::IndexNotFound(msg) => (StatusCode::NOT_FOUND, "INDEX_NOT_FOUND", msg.clone()),
            ApiError::IndexAlreadyExists(msg) => {
                (StatusCode::CONFLICT, "INDEX_ALREADY_EXISTS", msg.clone())
            }
            ApiError::IndexNotDeclared(msg) => (
                StatusCode::NOT_FOUND,
                "INDEX_NOT_DECLARED",
                format!(
                    "Index '{}' not declared. Call POST /indices first to declare the index.",
                    msg
                ),
            ),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg.clone()),
            ApiError::DimensionMismatch { expected, actual } => (
                StatusCode::BAD_REQUEST,
                "DIMENSION_MISMATCH",
                format!("Expected dimension {}, got {}", expected, actual),
            ),
            ApiError::MetadataNotFound(msg) => {
                (StatusCode::NOT_FOUND, "METADATA_NOT_FOUND", msg.clone())
            }
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                msg.clone(),
            ),
            ApiError::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "SERVICE_UNAVAILABLE",
                msg.clone(),
            ),
            ApiError::ModelNotLoaded => (
                StatusCode::BAD_REQUEST,
                "MODEL_NOT_LOADED",
                "No model loaded. Start the server with --model <path> to enable encoding."
                    .to_string(),
            ),
            ApiError::ModelError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MODEL_ERROR",
                msg.clone(),
            ),
            ApiError::NextPlaid(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "NEXT_PLAID_ERROR",
                e.to_string(),
            ),
        };

        let body = ErrorResponse {
            code,
            message,
            details: None,
        };

        (status, Json(body)).into_response()
    }
}

/// Result type alias for API operations.
pub type ApiResult<T> = Result<T, ApiError>;
