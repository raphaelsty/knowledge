//! Knowledge API Library
//!
//! Unified REST API combining search (next-plaid), data, events, and ingest services.

pub mod db;
pub mod error;
pub mod handlers;
pub mod models;
pub mod state;
pub mod tracing_middleware;

pub use error::{ApiError, ApiResult};
pub use state::{ApiConfig, AppState};

/// JSON response with pretty-printing (indented output).
pub struct PrettyJson<T>(pub T);

impl<T: serde::Serialize> axum::response::IntoResponse for PrettyJson<T> {
    fn into_response(self) -> axum::response::Response {
        match serde_json::to_string_pretty(&self.0) {
            Ok(json) => (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json,
            )
                .into_response(),
            Err(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to serialize response: {}", e),
            )
                .into_response(),
        }
    }
}
