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
/// `None` means ADMIN_API_KEY is not set (dev mode — all requests allowed).
fn admin_api_key() -> Option<&'static str> {
    static KEY: OnceLock<Option<String>> = OnceLock::new();
    KEY.get_or_init(|| {
        let key = std::env::var("ADMIN_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        if key.is_none() {
            tracing::warn!("ADMIN_API_KEY not set — admin endpoints are unprotected");
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
        let Some(expected) = admin_api_key() else {
            // No key configured — allow all (dev mode)
            return Ok(RequireApiKey);
        };

        let provided = parts.headers.get("X-API-Key").and_then(|v| v.to_str().ok());

        match provided {
            Some(key) if key == expected => Ok(RequireApiKey),
            _ => Err(ApiKeyRejection),
        }
    }
}
