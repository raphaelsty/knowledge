//! Server-side fetch proxy for the browser sync.
//!
//!   GET /api/proxy/fetch?url=<percent-encoded URL>
//!
//! Many of the websites users add to their library (sitemaps, RSS
//! feeds, blog homepages) don't send `Access-Control-Allow-Origin`,
//! so a browser-side `fetch()` is rejected before the body lands.
//! The Python pipeline doesn't have this constraint — but the
//! "Sync now" button in the profile modal runs entirely in the
//! browser, so it inherits CORS.
//!
//! This proxy is the bridge: the browser asks our API to GET the
//! upstream URL on its behalf, the API streams the body back with
//! permissive CORS so the browser-side fetcher can read it.
//!
//! Safety:
//!   * Auth-required (session cookie). Anonymous users can't use
//!     the API as an open relay.
//!   * Only `http`/`https` schemes.
//!   * Loopback / private / link-local destinations rejected to
//!     avoid SSRF against the host network.
//!   * 20s timeout, 10MB body cap.

use axum::{
    extract::{Query, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use axum_extra::extract::cookie::CookieJar;
use serde::Deserialize;
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::handlers::auth::current_user;
use crate::handlers::url_safety::{is_blocked_host_literal, safe_get};

const MAX_BODY: usize = 10 * 1024 * 1024; // 10 MB
const TIMEOUT_SECS: u64 = 20;

// ── In-memory response cache ─────────────────────────────────────────
//
// The frontend hammers this endpoint to scrape `og:description` for
// short-summary cards (the "summary enhancer" loop in page.js). Each
// upstream call is 600–900ms when the remote is youtube.com or
// scholar.google.com, and several users will collectively request
// the same handful of URLs (any followed user's page re-asks for the
// same docs).
//
// A process-local cache keyed by URL turns the second hit into a
// memcpy. We store status + content-type + body. TTL is generous —
// these are static-ish OG metadata pages and the user wouldn't notice
// a 30-minute staleness window. Size cap prunes the LRU by oldest
// fetched timestamp.

const CACHE_TTL: Duration = Duration::from_secs(30 * 60); // 30 min
const CACHE_MAX_ENTRIES: usize = 4096;

struct CachedEntry {
    status: u16,
    content_type: String,
    body: bytes::Bytes,
    fetched_at: Instant,
}

fn cache() -> &'static Mutex<HashMap<String, CachedEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<String, CachedEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_lookup(url: &str) -> Option<(u16, String, bytes::Bytes)> {
    let mut map = cache().lock().ok()?;
    let entry = map.get(url)?;
    if entry.fetched_at.elapsed() > CACHE_TTL {
        map.remove(url);
        return None;
    }
    Some((entry.status, entry.content_type.clone(), entry.body.clone()))
}

fn cache_store(url: String, status: u16, content_type: String, body: bytes::Bytes) {
    let Ok(mut map) = cache().lock() else {
        return;
    };
    if map.len() >= CACHE_MAX_ENTRIES {
        // Evict the oldest entry. Linear scan is fine for 4k items.
        if let Some((oldest_k, _)) = map
            .iter()
            .min_by_key(|(_, e)| e.fetched_at)
            .map(|(k, e)| (k.clone(), e.fetched_at))
        {
            map.remove(&oldest_k);
        }
    }
    map.insert(
        url,
        CachedEntry {
            status,
            content_type,
            body,
            fetched_at: Instant::now(),
        },
    );
}

#[derive(Deserialize)]
pub struct ProxyParams {
    pub url: String,
}

pub async fn proxy_fetch(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Query(p): Query<ProxyParams>,
) -> Response {
    // Auth gate.
    if current_user(&pool, &jar).await.is_none() {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    // URL validation — scheme + literal-host check. The deep SSRF
    // guard (server-side DNS resolution + per-IP block list + redirect
    // re-validation) lives inside `safe_get`. The literal-host short
    // circuit here keeps the cache key clean for the `localhost` /
    // raw-IP cases that we never want to even reach DNS for.
    let parsed = match reqwest::Url::parse(&p.url) {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid url").into_response(),
    };
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return (StatusCode::BAD_REQUEST, "scheme not allowed").into_response();
    }
    match parsed.host_str() {
        Some(host) if is_blocked_host_literal(host) => {
            return (StatusCode::FORBIDDEN, "private network not allowed").into_response();
        }
        Some(_) => {}
        None => return (StatusCode::BAD_REQUEST, "missing host").into_response(),
    }

    // Cache hit shortcut. Returns the same shape as a live fetch so
    // callers can't tell the difference. We key on the *normalised*
    // URL string (reqwest::Url already normalises percent-encoding
    // and case-folds the host).
    let cache_key = parsed.as_str().to_string();
    if let Some((status_u16, ct, body)) = cache_lookup(&cache_key) {
        let status = StatusCode::from_u16(status_u16).unwrap_or(StatusCode::BAD_GATEWAY);
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE.as_str(),
            HeaderValue::from_str(&ct)
                .unwrap_or(HeaderValue::from_static("application/octet-stream")),
        );
        headers.insert("x-knowledge-cache", HeaderValue::from_static("HIT"));
        headers.insert("cache-control", HeaderValue::from_static("no-store"));
        return (status, headers, body).into_response();
    }

    // SSRF-safe fetch: resolves DNS server-side, blocks private IPs,
    // and re-validates every redirect hop. native-tls is used inside
    // `safe_get` so the fingerprint behaviour stays the same as the
    // previous reqwest builder (reddit.com etc.).
    let resp = match safe_get(
        parsed.as_str(),
        std::time::Duration::from_secs(TIMEOUT_SECS),
        "Knowledge/1.0 (research project; https://github.com/raphaelsty/knowledge)",
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            // Map structured fetch errors to the right HTTP status.
            use crate::handlers::url_safety::FetchError;
            let (status, msg) = match &e {
                FetchError::InvalidUrl
                | FetchError::SchemeNotAllowed
                | FetchError::HostMissing
                | FetchError::RedirectLocationMissing => (StatusCode::BAD_REQUEST, e.to_string()),
                FetchError::BlockedHost => (StatusCode::FORBIDDEN, e.to_string()),
                FetchError::DnsFailed
                | FetchError::Request(_)
                | FetchError::TooManyRedirects
                | FetchError::BuildClient => (StatusCode::BAD_GATEWAY, e.to_string()),
            };
            return (status, msg).into_response();
        }
    };
    let upstream_status = resp.status();
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();
    let body = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("body: {e}")).into_response(),
    };
    if body.len() > MAX_BODY {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("response > {MAX_BODY} bytes"),
        )
            .into_response();
    }

    // Mirror upstream status (so 404 stays 404 etc.) but always
    // forward the body.
    let status = StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    // Only cache successful responses; a 5xx blip shouldn't pin a bad
    // result for 30 min. 404 we keep — it's a stable "no" that we
    // don't want to keep re-asking.
    if upstream_status.is_success() || upstream_status == reqwest::StatusCode::NOT_FOUND {
        cache_store(cache_key, status.as_u16(), ct.clone(), body.clone());
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE.as_str(),
        HeaderValue::from_str(&ct).unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    headers.insert("x-knowledge-cache", HeaderValue::from_static("MISS"));
    headers.insert("cache-control", HeaderValue::from_static("no-store"));
    (status, headers, body).into_response()
}

// Loopback/private/link-local/ULA/CGNAT IP filtering plus DNS-resolve
// hardening live in `crate::handlers::url_safety` (see `safe_get`).
