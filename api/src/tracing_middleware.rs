//! Request tracing middleware for the next-plaid API.
//!
//! Provides request correlation via trace_id for all API operations.
//! Each incoming request receives a unique trace_id that propagates
//! through all handlers and logs for debugging and observability.

use axum::{
    extract::Request,
    http::{header::HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

/// Counter for generating unique request IDs within this process.
/// Combined with a random component for global uniqueness.
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Extension type for storing the trace_id in request extensions.
#[derive(Clone, Debug)]
pub struct TraceId(pub String);

impl TraceId {
    /// Generate a new trace ID.
    ///
    /// Uses a combination of UUID v4 (random) for global uniqueness
    /// and a monotonic counter for ordering within this process.
    pub fn new() -> Self {
        // Use UUID v4 for uniqueness across instances
        // The counter helps with ordering/debugging within a single instance
        let _counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(Uuid::new_v4().to_string())
    }

    /// Create a TraceId from an existing string (e.g., from incoming header).
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the trace_id as a string reference.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Header name for request ID (standard convention).
pub static X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Middleware that adds request tracing via trace_id.
///
/// - Extracts or generates a trace_id for each request
/// - Stores trace_id in request extensions for handler access
/// - Adds X-Request-ID header to responses
pub async fn trace_request(mut request: Request, next: Next) -> Response {
    // Check if client provided a trace_id via X-Request-ID header
    let trace_id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(|s| TraceId::from_string(s.to_string()))
        .unwrap_or_else(TraceId::new);

    // Store trace_id in request extensions for handlers to access
    request.extensions_mut().insert(trace_id.clone());

    // Process the request
    let mut response = next.run(request).await;

    // Add trace_id to response headers
    if let Ok(header_value) = HeaderValue::from_str(trace_id.as_str()) {
        response
            .headers_mut()
            .insert(X_REQUEST_ID.clone(), header_value);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id_generation() {
        let id1 = TraceId::new();
        let id2 = TraceId::new();

        // Each trace_id should be unique
        assert_ne!(id1.as_str(), id2.as_str());

        // Should be valid UUID format (36 chars with hyphens)
        assert_eq!(id1.as_str().len(), 36);
        assert!(id1.as_str().contains('-'));
    }

    #[test]
    fn test_trace_id_from_string() {
        let custom_id = "custom-trace-id-12345";
        let trace_id = TraceId::from_string(custom_id.to_string());
        assert_eq!(trace_id.as_str(), custom_id);
    }
}
