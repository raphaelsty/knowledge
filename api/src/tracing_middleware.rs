//! Request tracing middleware for the API.

use axum::{
    extract::Request,
    http::{header::HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct TraceId(pub String);

impl TraceId {
    pub fn new() -> Self {
        let _counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }

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

pub static X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

pub async fn trace_request(mut request: Request, next: Next) -> Response {
    let trace_id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(|s| TraceId::from_string(s.to_string()))
        .unwrap_or_else(TraceId::new);

    request.extensions_mut().insert(trace_id.clone());

    let mut response = next.run(request).await;

    if let Ok(header_value) = HeaderValue::from_str(trace_id.as_str()) {
        response
            .headers_mut()
            .insert(X_REQUEST_ID.clone(), header_value);
    }

    response
}
