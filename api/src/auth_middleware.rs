//! API key authentication.
//!
//! Protects sensitive endpoints (index management, ingestion, deletion)
//! by requiring a valid `X-API-Key` header matching the `ADMIN_API_KEY` env var.
//!
//! Usage: add `_auth: RequireApiKey` as a parameter to any handler that needs protection.

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response},
};
use std::sync::OnceLock;

/// Cached admin API key from environment.
/// `None` means ADMIN_API_KEY is not set — the middleware then fails closed
/// (rejects every request) rather than allowing unauthenticated access.
fn admin_api_key() -> Option<&'static str> {
    static KEY: OnceLock<Option<String>> = OnceLock::new();
    KEY.get_or_init(|| {
        let key = std::env::var("ADMIN_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        if key.is_none() {
            tracing::error!(
                "ADMIN_API_KEY is not set — admin endpoints will reject all requests until it is configured"
            );
        }
        key
    })
    .as_deref()
}

/// Extractor that validates the API key. Add as a handler parameter to protect an endpoint.
///
/// ```rust
/// async fn my_admin_handler(_auth: RequireApiKey, ...) -> ... { }
/// ```
pub struct RequireApiKey;

/// Rejection type for missing/invalid API key.
pub struct ApiKeyRejection;

impl IntoResponse for ApiKeyRejection {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "code": "UNAUTHORIZED",
            "message": "Missing or invalid API key. Provide a valid X-API-Key header."
        });
        (
            StatusCode::UNAUTHORIZED,
            [("content-type", "application/json")],
            body.to_string(),
        )
            .into_response()
    }
}

impl<S: Send + Sync> FromRequestParts<S> for RequireApiKey {
    type Rejection = ApiKeyRejection;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Fail closed: if ADMIN_API_KEY is not configured, reject every request
        // rather than silently allowing unauthenticated access to admin endpoints.
        let Some(expected) = admin_api_key() else {
            return Err(ApiKeyRejection);
        };

        let provided = parts.headers.get("X-API-Key").and_then(|v| v.to_str().ok());

        // Constant-time comparison to avoid leaking the key byte-by-byte via timing.
        match provided {
            Some(key) if constant_time_eq(key.as_bytes(), expected.as_bytes()) => Ok(RequireApiKey),
            _ => Err(ApiKeyRejection),
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
