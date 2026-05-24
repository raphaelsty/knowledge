//! Knowledge API — Unified Rust server
//!
//! Combines search (next-plaid), data, events, and ingest endpoints into one binary.
//!
//! # Endpoints
//!
//! ## Search
//! - `GET /health` - Health check with index info
//! - `GET /indices` - List indices
//! - `POST /indices/{name}/search` - Search with embeddings
//! - `POST /indices/{name}/search/filtered` - Filtered search
//! - etc.
//!
//! ## Events
//! - `POST /events` - Batch event ingestion
//! - `GET /stats/overview` - Analytics overview
//! - `GET /stats/activity` - Activity over time
//! - `GET /stats/top-queries` - Top search queries
//! - `GET /stats/top-clicks` - Top clicked URLs
//! - `GET /stats/sources` - Source filter usage
//! - `GET /stats/folders` - Folder browse counts
//!
//! ## Ingest
//! - `POST /api/bookmark` - Ingest a bookmark
//!
//! ## Pipeline
//! - `POST /api/pipeline` - Trigger the Python pipeline (run.py)
//! - `GET /api/pipeline` - Pipeline status and last run result

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use tower::limit::ConcurrencyLimitLayer;
use tower_governor::{
    governor::GovernorConfigBuilder, key_extractor::SmartIpKeyExtractor, GovernorLayer,
};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod auth_middleware;
mod db;
mod error;
mod handlers;
mod mcp;
mod models;
mod state;
mod tracing_middleware;

use knowledge_api::PrettyJson;
use models::HealthResponse;
use state::{ApiConfig, AppState};

const RETENTION_DAYS: i32 = 90;

// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Knowledge API",
        version = "0.1.0",
        description = "Unified REST API for knowledge base: search, data, events, and ingest.",
    ),
    servers(
        (url = "/", description = "Local server")
    ),
    tags(
        (name = "health", description = "Health check endpoints"),
        (name = "indices", description = "Index management operations"),
        (name = "documents", description = "Document upload and deletion"),
        (name = "search", description = "Search operations"),
        (name = "metadata", description = "Metadata management and filtering"),
        (name = "encoding", description = "Text encoding operations (requires --model)"),
        (name = "reranking", description = "Document reranking with ColBERT MaxSim scoring")
    ),
    paths(
        health,
        handlers::documents::list_indices,
        handlers::documents::create_index,
        handlers::documents::get_index_info,
        handlers::documents::delete_index,
        handlers::documents::promote_index,
        handlers::documents::add_documents,
        handlers::documents::delete_documents,
        handlers::documents::update_index,
        handlers::documents::update_index_config,
        handlers::documents::update_index_with_encoding,
        handlers::search::search,
        handlers::search::search_filtered,
        handlers::search::search_with_encoding,
        handlers::search::search_filtered_with_encoding,
        handlers::encode::encode,
        handlers::rerank::rerank,
        handlers::rerank::rerank_with_encoding,
        handlers::metadata::get_all_metadata,
        handlers::metadata::get_metadata_count,
        handlers::metadata::check_metadata,
        handlers::metadata::query_metadata,
        handlers::metadata::get_metadata,
        handlers::metadata::update_metadata,
    ),
    components(schemas(
        models::HealthResponse,
        models::ModelHealthInfo,
        models::IndexSummary,
        models::ErrorResponse,
        models::CreateIndexRequest,
        models::CreateIndexResponse,
        models::IndexConfigRequest,
        models::IndexConfigStored,
        models::IndexInfoResponse,
        models::DocumentEmbeddings,
        models::AddDocumentsRequest,
        models::AddDocumentsResponse,
        models::DeleteDocumentsRequest,
        models::DeleteDocumentsResponse,
        models::DeleteIndexResponse,
        models::PromoteIndexRequest,
        models::PromoteIndexResponse,
        models::UpdateIndexRequest,
        models::UpdateIndexResponse,
        models::QueryEmbeddings,
        models::SearchRequest,
        models::SearchParamsRequest,
        models::SearchResponse,
        models::QueryResultResponse,
        models::FilteredSearchRequest,
        models::CheckMetadataRequest,
        models::CheckMetadataResponse,
        models::GetMetadataRequest,
        models::GetMetadataResponse,
        models::QueryMetadataRequest,
        models::QueryMetadataResponse,
        models::MetadataCountResponse,
        models::UpdateMetadataRequest,
        models::UpdateMetadataResponse,
        models::UpdateIndexConfigRequest,
        models::UpdateIndexConfigResponse,
        models::InputType,
        models::EncodeRequest,
        models::EncodeResponse,
        models::SearchWithEncodingRequest,
        models::FilteredSearchWithEncodingRequest,
        models::UpdateWithEncodingRequest,
        models::RerankRequest,
        models::RerankWithEncodingRequest,
        models::RerankResult,
        models::RerankResponse,
    ))
)]
struct ApiDoc;

/// Cached sysinfo System for memory usage queries.
static SYSINFO_SYSTEM: std::sync::OnceLock<std::sync::Mutex<sysinfo::System>> =
    std::sync::OnceLock::new();

fn get_memory_usage_bytes() -> u64 {
    let pid = match sysinfo::get_current_pid() {
        Ok(pid) => pid,
        Err(_) => return 0,
    };

    let system_mutex = SYSINFO_SYSTEM.get_or_init(|| std::sync::Mutex::new(sysinfo::System::new()));

    let mut system = match system_mutex.lock() {
        Ok(guard) => guard,
        Err(_) => return 0,
    };

    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// Health check and root endpoint.
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
async fn health(state: axum::extract::State<Arc<AppState>>) -> PrettyJson<HealthResponse> {
    if !state.config.index_dir.exists() {
        let dir = state.config.index_dir.clone();
        tokio::task::spawn_blocking(move || std::fs::create_dir_all(&dir).ok());
    }

    let memory_usage_bytes = get_memory_usage_bytes();

    #[cfg(feature = "model")]
    let model_info = state
        .cached_model_info()
        .map(|info| models::ModelHealthInfo {
            name: info.name.clone(),
            path: info.path.clone(),
            quantized: info.quantized,
            embedding_dim: info.embedding_dim,
            batch_size: info.batch_size,
            num_sessions: info.num_sessions,
            query_prefix: info.query_prefix.clone(),
            document_prefix: info.document_prefix.clone(),
            query_length: info.query_length,
            document_length: info.document_length,
            do_query_expansion: info.do_query_expansion,
            uses_token_type_ids: info.uses_token_type_ids,
            mask_token_id: info.mask_token_id,
            pad_token_id: info.pad_token_id,
        });

    #[cfg(not(feature = "model"))]
    let model_info: Option<models::ModelHealthInfo> = None;

    PrettyJson(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_indices: state.loaded_count(),
        index_dir: state.config.index_dir.to_string_lossy().to_string(),
        memory_usage_bytes,
        indices: state.get_all_index_summaries(),
        model: model_info,
    })
}

fn rate_limit_error(_err: tower_governor::GovernorError) -> axum::http::Response<axum::body::Body> {
    let body = serde_json::json!({
        "code": "RATE_LIMITED",
        "message": "Too many requests. Please retry after the specified time.",
        "retry_after_seconds": 2
    });
    axum::http::Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("content-type", "application/json")
        .header("retry-after", "2")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap()
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!(signal = "SIGINT", "server.shutdown.initiated");
        },
        _ = terminate => {
            tracing::info!(signal = "SIGTERM", "server.shutdown.initiated");
        },
    }
}

/// Apply every schema file in `sources/sql/` against the live
/// database, in the same dependency order `run.py` uses.
///
/// Why this lives in the API process (not just in `run.py`): the
/// API binary deploys whenever code or schema changes, but the
/// pipeline daemon runs on its own cadence. Without this hook a
/// deploy that adds a new column ships the binary first and the
/// schema only minutes-to-hours later when the next pipeline pass
/// fires — every endpoint that references the new column would
/// 500 in the meantime (May 2026: `column e.dwell_ms does not
/// exist` outage). Running migrations on API boot collapses that
/// window to ~0.
///
/// All .sql files are baked in via `include_str!`, so the API
/// container doesn't need the sources/ tree at runtime — only the
/// binary. Each file is an idempotent batch (`CREATE TABLE IF NOT
/// EXISTS` / `ADD COLUMN IF NOT EXISTS` / `CREATE OR REPLACE
/// VIEW`), so re-applying on every boot is safe.
///
/// Order matters: tables with foreign keys must be applied after
/// their referent. The list below mirrors `run.py`'s
/// `create_*_table()` sequence one-for-one — keep them in sync.
async fn run_sql_migrations(pool: &sqlx::PgPool) -> Result<(), sqlx::Error> {
    // (filename-for-logs, baked SQL text). The ordering matches
    // run.py — DO NOT reorder without checking the FK graph.
    let migrations: &[(&str, &str)] = &[
        ("users.sql", include_str!("../../sources/sql/users.sql")),
        (
            "documents.sql",
            include_str!("../../sources/sql/documents.sql"),
        ),
        (
            "dead_urls.sql",
            include_str!("../../sources/sql/dead_urls.sql"),
        ),
        (
            "sessions.sql",
            include_str!("../../sources/sql/sessions.sql"),
        ),
        (
            "twitter_feed_status.sql",
            include_str!("../../sources/sql/twitter_feed_status.sql"),
        ),
        (
            "twitter_feed_attempts.sql",
            include_str!("../../sources/sql/twitter_feed_attempts.sql"),
        ),
        (
            "auth_sessions.sql",
            include_str!("../../sources/sql/auth_sessions.sql"),
        ),
        (
            "api_tokens.sql",
            include_str!("../../sources/sql/api_tokens.sql"),
        ),
        (
            "favorites.sql",
            include_str!("../../sources/sql/favorites.sql"),
        ),
        ("follows.sql", include_str!("../../sources/sql/follows.sql")),
        ("events.sql", include_str!("../../sources/sql/events.sql")),
        (
            "export_downloads.sql",
            include_str!("../../sources/sql/export_downloads.sql"),
        ),
        (
            "pipeline_runs.sql",
            include_str!("../../sources/sql/pipeline_runs.sql"),
        ),
        (
            "pipeline_source_runs.sql",
            include_str!("../../sources/sql/pipeline_source_runs.sql"),
        ),
        (
            "index_health_checks.sql",
            include_str!("../../sources/sql/index_health_checks.sql"),
        ),
        (
            "oauth_identities.sql",
            include_str!("../../sources/sql/oauth_identities.sql"),
        ),
        (
            "personality_submissions.sql",
            include_str!("../../sources/sql/personality_submissions.sql"),
        ),
        (
            "hn_frontpage.sql",
            include_str!("../../sources/sql/hn_frontpage.sql"),
        ),
        ("credits.sql", include_str!("../../sources/sql/credits.sql")),
        (
            "user_storage.sql",
            include_str!("../../sources/sql/user_storage.sql"),
        ),
        (
            "vip_sponsorships.sql",
            include_str!("../../sources/sql/vip_sponsorships.sql"),
        ),
        // Views depend on the documents + users tables — must run last.
        ("views.sql", include_str!("../../sources/sql/views.sql")),
    ];
    for (name, sql) in migrations {
        // Each file is executed in one round-trip; sqlx's `execute`
        // handles multi-statement batches against Postgres. Errors
        // surface with the filename so the operator knows which one
        // tripped without diffing the log against the migrations list.
        if let Err(e) = sqlx::raw_sql(sql).execute(pool).await {
            tracing::error!(file = %name, error = %e, "schema.migrate.statement_failed");
            return Err(e);
        }
        tracing::debug!(file = %name, "schema.migrate.applied");
    }
    Ok(())
}

/// Build the API router.
fn build_router(state: Arc<AppState>, pg_pool: Option<sqlx::PgPool>) -> Router {
    let rate_limit_enabled: bool = std::env::var("RATE_LIMIT_ENABLED")
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(false);
    let rate_limit_per_second: u64 = std::env::var("RATE_LIMIT_PER_SECOND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let rate_limit_burst_size: u32 = std::env::var("RATE_LIMIT_BURST_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let concurrency_limit: usize = std::env::var("CONCURRENCY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    if rate_limit_enabled {
        tracing::info!(
            rate_limit_per_second,
            rate_limit_burst_size,
            "rate_limiting.enabled"
        );
    } else {
        tracing::info!("rate_limiting.disabled");
    }

    // Permissive CORS for the public read API (search, metadata, MCP,
    // index_info). Allows any origin + standard methods, but restricts
    // `allow_headers` to an explicit list — notably *excluding*
    // `X-API-Key`. That stops cross-origin browser scripts from
    // pre-flighting admin endpoints even if they hold the key, while
    // still letting normal read calls work from any origin.
    //
    // Server-to-server callers (the Python pipeline, CLI scripts) do
    // not preflight, so they remain unaffected.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PUT,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::ACCEPT,
            axum::http::header::AUTHORIZATION,
        ]);

    // --- Search API routers ---

    let health_router = Router::new()
        .route("/health", get(health))
        .route("/", get(health))
        .layer(middleware::from_fn(tracing_middleware::trace_request))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(30),
        ))
        .with_state(state.clone());

    let index_info_router = Router::new()
        .without_v07_checks()
        .route("/indices", get(handlers::list_indices))
        .route("/indices/{name}", get(handlers::get_index_info))
        .layer(middleware::from_fn(tracing_middleware::trace_request))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(30),
        ))
        .layer(cors.clone())
        .with_state(state.clone());

    // --- Routers (auth is handled per-handler via RequireApiKey extractor) ---

    let update_router = Router::new()
        .without_v07_checks()
        .route("/indices/{name}/update", post(handlers::update_index))
        .route(
            "/indices/{name}/update_with_encoding",
            post(handlers::update_index_with_encoding),
        )
        .layer(middleware::from_fn(tracing_middleware::trace_request))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(cors.clone())
        .layer(ConcurrencyLimitLayer::new(concurrency_limit))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state.clone());

    let encode_router = Router::new()
        .route("/encode", post(handlers::encode))
        .route("/rerank", post(handlers::rerank))
        .route(
            "/rerank_with_encoding",
            post(handlers::rerank_with_encoding),
        )
        .layer(middleware::from_fn(tracing_middleware::trace_request))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(cors.clone())
        .layer(ConcurrencyLimitLayer::new(concurrency_limit))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state.clone());

    let delete_router = Router::new()
        .without_v07_checks()
        .route("/indices/{name}", delete(handlers::delete_index))
        .route(
            "/indices/{name}/documents",
            delete(handlers::delete_documents),
        )
        // Promote sits on the delete-router because it is logically
        // a destructive op on the target index (its old contents are
        // discarded). Sharing the router means it inherits the same
        // timeout / concurrency / CORS layers that govern delete.
        .route(
            "/indices/{name}/promote",
            post(handlers::documents::promote_index),
        )
        .layer(middleware::from_fn(tracing_middleware::trace_request))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(cors.clone())
        .layer(ConcurrencyLimitLayer::new(concurrency_limit))
        .with_state(state.clone());

    let search_api_router = Router::new()
        .without_v07_checks()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/indices", post(handlers::create_index))
        .route("/indices/{name}/documents", post(handlers::add_documents))
        .route("/indices/{name}/config", put(handlers::update_index_config))
        .route("/indices/{name}/search", post(handlers::search))
        .route(
            "/indices/{name}/search/filtered",
            post(handlers::search_filtered),
        )
        .route(
            "/indices/{name}/search_with_encoding",
            post(handlers::search_with_encoding),
        )
        .route(
            "/indices/{name}/search/filtered_with_encoding",
            post(handlers::search_filtered_with_encoding),
        )
        .route("/indices/{name}/metadata", get(handlers::get_all_metadata))
        .route(
            "/indices/{name}/metadata/count",
            get(handlers::get_metadata_count),
        )
        .route(
            "/indices/{name}/metadata/check",
            post(handlers::check_metadata),
        )
        .route(
            "/indices/{name}/metadata/query",
            post(handlers::query_metadata),
        )
        .route("/indices/{name}/metadata/get", post(handlers::get_metadata))
        .route(
            "/indices/{name}/metadata/update",
            post(handlers::update_metadata),
        )
        .layer(middleware::from_fn(tracing_middleware::trace_request))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(cors.clone());

    let search_api_router = if rate_limit_enabled {
        // SmartIpKeyExtractor reads X-Forwarded-For / X-Real-IP /
        // Forwarded before falling back to the peer IP. Required
        // because the API is fronted by Traefik + Caddy — without
        // it, every request looks like it came from caddy's
        // container IP and the limiter degenerates into a global
        // shared bucket.
        let governor_conf = GovernorConfigBuilder::default()
            .per_second(rate_limit_per_second)
            .burst_size(rate_limit_burst_size)
            .key_extractor(SmartIpKeyExtractor)
            .finish()
            .expect("Failed to build rate limiter config");
        let governor_layer = GovernorLayer::new(governor_conf).error_handler(rate_limit_error);
        search_api_router.layer(governor_layer)
    } else {
        search_api_router
    };

    let search_api_router = search_api_router
        .layer(ConcurrencyLimitLayer::new(concurrency_limit))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .with_state(state.clone());

    // --- Ingest router ---
    let ingest_router = Router::new()
        .route("/api/bookmark", post(handlers::ingest::ingest_bookmark))
        .layer(cors.clone())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(120),
        ))
        .with_state(state.clone());

    // --- Pipeline router ---
    let pipeline_state = handlers::pipeline::new_state();
    let pipeline_router = Router::new()
        .route("/api/pipeline", get(handlers::pipeline::status))
        .route("/api/pipeline", post(handlers::pipeline::trigger))
        .layer(cors.clone())
        .with_state(pipeline_state);

    // Start merging all routers
    let mut app = Router::new()
        .merge(health_router)
        .merge(index_info_router)
        .merge(update_router)
        .merge(encode_router)
        .merge(delete_router)
        .merge(search_api_router)
        .merge(ingest_router)
        .merge(pipeline_router);

    // --- MCP router (no PostgreSQL required — reads JSON files from data/) ---
    let mcp_router = Router::new()
        .route("/mcp", post(mcp::mcp_handler))
        .layer(cors.clone())
        .with_state(state.clone());
    app = app.merge(mcp_router);

    // --- PG-backed router (users + events + stats) ---
    if let Some(pool) = pg_pool {
        let users_router = Router::new()
            .route("/api/users", get(handlers::users::list_users))
            // Order matters: `/api/users/intersect` must be registered
            // before the `/api/users/{slug}` catch-all, otherwise axum
            // routes "intersect" as a slug.
            .route(
                "/api/users/intersect",
                get(handlers::users::intersect_documents),
            )
            .route("/api/co-owners", post(handlers::users::list_co_owners))
            .route("/api/users/{slug}", get(handlers::users::get_user))
            .route(
                "/api/users/{slug}/documents",
                get(handlers::users::list_documents),
            )
            .route(
                "/api/users/{slug}/sources",
                get(handlers::users::list_sources),
            )
            // SQL-backed search/feed for libraries whose ColBERT
            // index is missing on disk. Frontend wrappers in
            // search/api.js fall through to this endpoint only on
            // HTTP 404 from the plaid endpoint — so the fast path
            // stays plaid as long as the index exists.
            .route(
                "/api/personalities/{slug}/fallback",
                get(handlers::users::fallback_search),
            )
            .route("/api/sources", get(handlers::users::list_all_vip_sources))
            // Bearer-auth upload: the token (created from the profile
            // panel) decides the owning user. Inserts/upserts a single
            // document into that user's library; the pipeline picks it
            // up on the next `make run` and embeds it.
            .route("/api/me/documents", post(handlers::tokens::upload_document))
            .route("/api/feed", get(handlers::users::feed))
            .layer(cors.clone())
            .with_state(pool.clone());

        // GitHub OAuth + session management. Lives on its own router
        // because cookies require a credentials-aware CORS layer
        // (Allow-Origin: * + credentials is rejected by browsers).
        //
        // Allowed origins come from AUTH_ALLOWED_ORIGINS (comma-separated).
        // Locally we default to the two dev ports (web :3000, api :8080).
        let auth_origins: Vec<axum::http::HeaderValue> = std::env::var("AUTH_ALLOWED_ORIGINS")
            .unwrap_or_else(|_| {
                // Default includes every port the dev Makefile commonly
                // binds to so `make dev WEB_PORT=<n>` works out of the
                // box. 3001 is the default WEB_PORT; 3000 covers Vite /
                // CRA dev servers; 3002 is the fallback used when
                // Paradigm Mission Control's ui-console is already on
                // 3001 locally.
                "http://localhost:3002,http://127.0.0.1:3002,\
                 http://localhost:3001,http://127.0.0.1:3001,\
                 http://localhost:3000,http://127.0.0.1:3000,\
                 http://localhost:8080"
                    .to_string()
            })
            .split(',')
            .filter_map(|s| axum::http::HeaderValue::from_str(s.trim()).ok())
            .collect();

        let auth_cors = CorsLayer::new()
            .allow_origin(auth_origins)
            .allow_credentials(true)
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::PUT,
                axum::http::Method::PATCH,
                axum::http::Method::DELETE,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers([axum::http::header::CONTENT_TYPE, axum::http::header::COOKIE]);

        // Abuse-prone credential / email endpoints get a dedicated
        // tight rate limit (when RATE_LIMIT_ENABLED). Without this,
        // /auth/forgot can exhaust the Resend.com mailer quota and
        // /auth/login is open to password-stuffing. Keyed per IP
        // via SmartIpKeyExtractor (X-Forwarded-For aware).
        const AUTH_PER_SECOND: u64 = 1;
        const AUTH_BURST: u32 = 10;
        let auth_throttle_layer = if rate_limit_enabled {
            let conf = GovernorConfigBuilder::default()
                .per_second(AUTH_PER_SECOND)
                .burst_size(AUTH_BURST)
                .key_extractor(SmartIpKeyExtractor)
                .finish()
                .expect("Failed to build auth rate limiter");
            Some(GovernorLayer::new(conf).error_handler(rate_limit_error))
        } else {
            None
        };

        let mut auth_throttled = Router::new()
            .route("/auth/signup", post(handlers::auth::signup))
            .route("/auth/login", post(handlers::auth::login))
            .route("/auth/resend", post(handlers::auth::resend_verification))
            .route("/auth/forgot", post(handlers::auth::forgot_password))
            .route("/auth/reset", post(handlers::auth::reset_password));
        if let Some(layer) = auth_throttle_layer {
            auth_throttled = auth_throttled.layer(layer);
        }
        let auth_throttled = auth_throttled
            .layer(auth_cors.clone())
            .with_state(pool.clone());

        let auth_router = Router::new()
            // Email-verification, session, and logout aren't abuse-prone
            // in the same way as signup/login/forgot — kept on the
            // un-throttled router so a tab full of /auth/me calls
            // doesn't trip the limit.
            .route("/auth/verify", get(handlers::auth::verify_email))
            .route("/auth/me", get(handlers::auth::me))
            .route("/auth/logout", post(handlers::auth::logout))
            // Profile editor needs the same credentials-aware CORS as the
            // rest of /auth/* — it reads the session cookie.
            .route("/api/users/me", put(handlers::auth::update_me))
            .route("/auth/me/websites", put(handlers::auth::set_websites))
            .route("/api/profile/probe", post(handlers::probe::probe))
            .route("/api/proxy/fetch", get(handlers::proxy::proxy_fetch))
            .route(
                "/auth/me/hackernews",
                put(handlers::auth::set_hackernews).delete(handlers::auth::clear_hackernews),
            )
            .route(
                "/auth/me/hackernews/test",
                get(handlers::auth::test_hackernews_stored).post(handlers::auth::test_hackernews),
            )
            .route(
                "/auth/me/twitter/cookies",
                put(handlers::auth::set_twitter_cookies)
                    .delete(handlers::auth::clear_twitter_cookies),
            )
            .route(
                "/auth/me/zotero",
                put(handlers::auth::set_zotero).delete(handlers::auth::clear_zotero),
            )
            .route(
                "/auth/me/zotero/items",
                get(handlers::auth::fetch_zotero_items),
            )
            .route(
                "/auth/stackoverflow/start",
                get(handlers::auth::stackoverflow_start),
            )
            .route(
                "/auth/stackoverflow/callback",
                get(handlers::auth::stackoverflow_callback),
            )
            .route("/auth/github/start", get(handlers::auth::github_start))
            .route(
                "/auth/github/callback",
                get(handlers::auth::github_callback),
            )
            .route(
                "/auth/me/stackoverflow/auth",
                delete(handlers::auth::clear_stackoverflow_auth),
            )
            .route(
                "/auth/me/documents",
                post(handlers::auth::save_document)
                    .patch(handlers::auth::update_document)
                    .delete(handlers::auth::delete_document),
            )
            .route(
                "/auth/me/documents/bulk",
                post(handlers::auth::bulk_save_documents),
            )
            // User-scoped API tokens. Cookie session manages lifecycle;
            // the bearer-auth upload route below uses the token itself.
            .route(
                "/auth/me/tokens",
                post(handlers::tokens::create_token).get(handlers::tokens::list_tokens),
            )
            .route(
                "/auth/me/tokens/{id}",
                axum::routing::delete(handlers::tokens::revoke_token),
            )
            .route("/auth/me/sync/start", post(handlers::auth::sync_start))
            .route("/auth/me/sync/end", post(handlers::auth::sync_end))
            .route("/auth/me/sync/status", get(handlers::auth::sync_status))
            .route(
                "/auth/me/documents/urls",
                get(handlers::auth::list_document_urls),
            )
            .route(
                "/auth/me/personality-bookmarks",
                get(handlers::auth::list_personality_bookmarks),
            )
            .route(
                "/auth/me/personality-bookmarks/{slug}",
                put(handlers::auth::add_personality_bookmark)
                    .delete(handlers::auth::remove_personality_bookmark),
            )
            .route("/auth/favorites", get(handlers::favorites::list))
            .route(
                "/auth/favorites/{slug}",
                put(handlers::favorites::add).delete(handlers::favorites::remove),
            )
            // Private per-user document favorites (star on every search
            // card). Separate table from personality favorites above.
            .route(
                "/auth/me/favorite-docs",
                get(handlers::favorite_docs::list)
                    .post(handlers::favorite_docs::add)
                    .delete(handlers::favorite_docs::remove),
            )
            .route(
                "/auth/me/favorite-docs/full",
                get(handlers::favorite_docs::list_full),
            )
            .route(
                "/auth/me/favorite-docs/owners",
                get(handlers::favorite_docs::list_owners),
            )
            .route(
                "/auth/me/deleted-urls",
                get(handlers::auth::list_deleted_urls),
            )
            // Follow graph + timeline.
            .route(
                "/api/follow/{slug}",
                post(handlers::follows::follow).delete(handlers::follows::unfollow),
            )
            .route("/api/me/following", get(handlers::follows::list_following))
            .route("/api/me/follow/bulk", post(handlers::follows::follow_bulk))
            .route("/api/me/feed/sources", get(handlers::follows::feed_sources))
            .route("/api/timeline", get(handlers::follows::timeline))
            // Catalogue endpoint feeding the search-bar category picker.
            // Read-only, no auth — the 178-row taxonomy is public.
            .route(
                "/api/document-categories",
                get(handlers::document_categories::list_document_categories),
            )
            // Batch URL -> category-slugs map. Used by the picker to
            // aggregate the categories of a query's top ColBERT hits.
            .route(
                "/api/document-categories/by-url",
                post(handlers::document_categories::categories_by_url),
            )
            // Slug -> URL set lookup for the search-path pre-filter.
            // Given a comma-separated slug list, returns every URL
            // assigned to at least one of them (OR semantics, capped).
            .route(
                "/api/document-categories/urls",
                get(handlers::document_categories::urls_by_slugs),
            )
            // Co-retweet sharer enrichment for search results — see
            // `handlers::follows::coretweet_sharers` for the SQL.
            .route(
                "/api/documents/coretweet-sharers",
                post(handlers::follows::coretweet_sharers),
            )
            // Credit-billing endpoints (Polar.sh). The webhook lives
            // on a separate, no-auth router below — it's called by
            // Polar's servers, not the browser.
            .route("/api/me/credits", get(handlers::credits::get_credits))
            .route("/api/credits/packs", get(handlers::credits::list_packs))
            .route(
                "/api/credits/checkout",
                post(handlers::credits::start_checkout),
            )
            .route(
                "/api/me/sponsorships",
                get(handlers::sponsorships::list).post(handlers::sponsorships::create),
            )
            .layer(auth_cors.clone())
            .with_state(pool.clone());

        // Polar webhook receiver. Public (no auth, no CORS), the
        // request is authenticated by HMAC signature inside the
        // handler. Kept in its own router so future webhook senders
        // (Stripe, …) can join the same parent.
        let webhooks_router = Router::new()
            .route("/api/credits/webhook", post(handlers::credits::webhook))
            .with_state(pool.clone());

        // Events + stats endpoints. Use the same credentials-aware
        // CORS layer as /auth/* so:
        //   • `sendBeacon('/events', ...)` from a signed-in user sends
        //     the session cookie cross-origin without browser rejection
        //     (wildcard `Access-Control-Allow-Origin: *` is incompatible
        //     with cookies, which is what the old layer did),
        //   • the dashboard can pull /stats/* with credentials so the
        //     auth middleware on those endpoints sees a session.
        let events_router = Router::new()
            .route("/events", post(handlers::events::ingest_events))
            .route("/stats/overview", get(handlers::events::overview))
            .route("/stats/activity", get(handlers::events::activity))
            .route("/stats/top-queries", get(handlers::events::top_queries))
            .route("/stats/top-clicks", get(handlers::events::top_clicks))
            .route("/stats/sources", get(handlers::events::sources))
            .route("/stats/folders", get(handlers::events::folders))
            .layer(auth_cors.clone())
            .with_state(pool);

        // Storage stats — needs both PgPool and the on-disk index
        // directory, so it carries Arc<AppState> rather than PgPool.
        let storage_router = Router::new()
            .route("/api/me/storage", get(handlers::storage::get_storage))
            .route(
                "/api/me/storage/refresh",
                post(handlers::storage::refresh_storage),
            )
            .layer(auth_cors.clone())
            .with_state(state.clone());

        // Library export — JSONL stream, billed via credits unless
        // the caller is the owner or the target is a VIP. Same URL
        // works for browser downloads (cookie auth) and scripted use
        // (Bearer token).
        let exports_router = Router::new()
            .route(
                "/api/personalities/{slug}/export.jsonl",
                get(handlers::exports::export_personality),
            )
            .layer(auth_cors.clone())
            .with_state(state.clone());

        // Adding a public personality. Charges $2 and kicks off an
        // initial pipeline run for the new slug. AppState so we can
        // reach the pool + spawn a subprocess from the same handler.
        let personalities_router = Router::new()
            .route("/api/personalities", post(handlers::personalities::create))
            .route(
                "/api/personalities/{slug}",
                put(handlers::personalities::update),
            )
            .route(
                "/api/me/personalities",
                get(handlers::personalities::list_mine),
            )
            .layer(auth_cors.clone())
            .with_state(state.clone());

        // Admin panel. Read-only routes, every handler self-gates
        // against the session cookie (slug must equal `raphael-sourty`).
        // Sits on Arc<AppState> because the indices scan reaches into
        // the loaded-index map alongside its PG query.
        let admin_router = Router::new()
            .route("/api/admin/overview", get(handlers::admin::overview))
            .route("/api/admin/sources", get(handlers::admin::sources))
            .route(
                "/api/admin/sources/{name}/failures",
                get(handlers::admin::source_failures),
            )
            .route("/api/admin/users", get(handlers::admin::users_list))
            .route(
                "/api/admin/users/{slug}/runs",
                get(handlers::admin::user_runs),
            )
            .route("/api/admin/indices", get(handlers::admin::indices))
            .route("/api/admin/live", get(handlers::admin::live))
            .route("/api/admin/system", get(handlers::admin::system_stats))
            .route("/api/admin/indexer", get(handlers::admin::indexer_activity))
            .route("/api/admin/behaviour", get(handlers::admin::behaviour))
            // Twitter-feed health surface. POST is the heartbeat
            // receiver (shared-token auth, no session needed);
            // GET is the admin-panel read (session cookie).
            .route(
                "/api/admin/twitter-feed/heartbeat",
                post(handlers::admin::twitter_feed_heartbeat),
            )
            .route(
                "/api/admin/tweets/ingest",
                post(handlers::admin::admin_ingest_tweets),
            )
            .route(
                "/api/admin/twitter-queue",
                get(handlers::admin::admin_twitter_queue),
            )
            .route(
                "/api/admin/users/{slug}/twitter-urls",
                get(handlers::admin::admin_user_twitter_urls),
            )
            .route(
                "/api/admin/twitter-feed/status",
                get(handlers::admin::twitter_feed_status),
            )
            .route(
                "/api/admin/twitter-feed/attempt",
                post(handlers::admin::admin_twitter_feed_attempt),
            )
            .layer(auth_cors.clone())
            .with_state(state.clone());

        app = app
            .merge(users_router)
            // Order matters: the throttled router carries the same
            // path-prefix as auth_router for a subset of routes, but
            // axum's merge takes the first match. Throttled goes first
            // so /auth/login etc. hit the rate-limited handler.
            .merge(auth_throttled)
            .merge(auth_router)
            .merge(events_router)
            .merge(storage_router)
            .merge(exports_router)
            .merge(personalities_router)
            .merge(webhooks_router)
            .merge(admin_router);
    }

    app
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "knowledge_api=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let mut host = "0.0.0.0".to_string();
    let mut port: u16 = 8080;
    let mut index_dir = PathBuf::from("./indices");
    let mut model_path: Option<PathBuf> = None;
    let mut _use_cuda = false;
    let mut _use_int8 = false;
    let mut _parallel_sessions: Option<usize> = None;
    let mut _batch_size: Option<usize> = None;
    let mut _threads: Option<usize> = None;
    let mut _query_length: Option<usize> = None;
    let mut _document_length: Option<usize> = None;
    let mut _model_pool_size: Option<usize> = None;
    let mut buffer_dir: Option<String> = None;
    let mut buffer_interval: u64 = 30;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" | "-h" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --host requires a value");
                    std::process::exit(1);
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid port number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --port requires a value");
                    std::process::exit(1);
                }
            }
            "--index-dir" | "-d" => {
                if i + 1 < args.len() {
                    index_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --index-dir requires a value");
                    std::process::exit(1);
                }
            }
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --model requires a value");
                    std::process::exit(1);
                }
            }
            "--cuda" => {
                _use_cuda = true;
                i += 1;
            }
            "--int8" => {
                _use_int8 = true;
                i += 1;
            }
            "--parallel" => {
                if i + 1 < args.len() {
                    _parallel_sessions = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid number of parallel sessions");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --parallel requires a value");
                    std::process::exit(1);
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    _batch_size = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid batch size");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --batch-size requires a value");
                    std::process::exit(1);
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    _threads = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid number of threads");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --threads requires a value");
                    std::process::exit(1);
                }
            }
            "--query-length" => {
                if i + 1 < args.len() {
                    _query_length = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid query length");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --query-length requires a value");
                    std::process::exit(1);
                }
            }
            "--document-length" => {
                if i + 1 < args.len() {
                    _document_length = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid document length");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --document-length requires a value");
                    std::process::exit(1);
                }
            }
            "--model-pool-size" => {
                if i + 1 < args.len() {
                    _model_pool_size = Some(args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid model pool size");
                        std::process::exit(1);
                    }));
                    i += 2;
                } else {
                    eprintln!("Error: --model-pool-size requires a value");
                    std::process::exit(1);
                }
            }
            "--buffer-dir" => {
                if i + 1 < args.len() {
                    buffer_dir = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --buffer-dir requires a value");
                    std::process::exit(1);
                }
            }
            "--buffer-interval" => {
                if i + 1 < args.len() {
                    buffer_interval = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid buffer interval");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --buffer-interval requires a value");
                    std::process::exit(1);
                }
            }
            "--help" => {
                println!(
                    r#"Knowledge API Server

Usage: knowledge-api [OPTIONS]

Options:
  -h, --host <HOST>        Host to bind to (default: 0.0.0.0)
  -p, --port <PORT>        Port to bind to (default: 8080)
  -d, --index-dir <DIR>    Directory for storing indices (default: ./indices)
  -m, --model <PATH>       Path to ONNX model directory for encoding (optional)
  --cuda                   Use CUDA for model inference (requires --model)
  --int8                   Use INT8 quantized model (requires --model)
  --parallel <N>           Number of parallel ONNX sessions (default: 1)
  --batch-size <N>         Batch size per ONNX session
  --threads <N>            Threads per ONNX session
  --query-length <N>       Maximum query length in tokens
  --document-length <N>    Maximum document length in tokens
  --model-pool-size <N>    Number of model worker instances
  --buffer-dir <DIR>       Directory to scan for buffer JSON files (enables buffer scanner)
  --buffer-interval <SECS> Scan interval in seconds (default: 30)
  --help                   Show this help message

Environment Variables:
  DATABASE_URL             PostgreSQL connection string (enables data/events/ingest endpoints)
  RUST_LOG                 Set log level (e.g., RUST_LOG=debug)
"#
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Use --help for usage information");
                std::process::exit(1);
            }
        }
    }

    // Create config
    let config = ApiConfig {
        index_dir,
        default_top_k: 10,
    };

    tracing::info!(
        index_dir = %config.index_dir.display(),
        "server.starting"
    );

    // --- Connect to PostgreSQL (optional) ---
    let pg_pool = if let Ok(database_url) = std::env::var("DATABASE_URL") {
        // Disable PG's JIT compiler on every connection in the pool.
        // Our timeline / co-owners queries have a planner-estimated
        // cost of ~3.9M which trips the default jit_above_cost
        // threshold (100k), and the LLVM optimize+emit passes then
        // burn ~1s per cold request — bigger than the actual query
        // execution time. Empirically (May 2026), jit=off cuts
        // /api/timeline TTFB from ~2s to ~700ms with no measurable
        // regression on the rest of the query mix. Setting it via
        // `after_connect` here AND via `ALTER DATABASE … SET jit
        // = off` (already applied to prod) so a fresh container
        // build pre-deploy also benefits.
        // Self-healing pool. Without `test_before_acquire` + a
        // bounded `max_lifetime` / `idle_timeout`, a PG container
        // restart or NAT timeout silently leaves sqlx with dead TCP
        // sockets — every subsequent request hangs for the 30 s
        // acquire timeout, returns 200 with an empty body, and the
        // operator finds out via "prod is down" (May 2026).
        //   * test_before_acquire: ~0.3 ms ping per checkout, drops
        //     dead connections before they reach a handler.
        //   * max_lifetime 30 min: forces full reconnect once per
        //     window so any half-state (server restart, role
        //     reload, NAT) gets renegotiated.
        //   * idle_timeout 10 min: shrinks the pool when traffic
        //     dips, so we don't carry 20 stale sockets through a
        //     quiet night.
        //   * max_connections 20: doubles the default 10. The API
        //     hosts ~5 concurrent handlers under typical traffic;
        //     20 leaves room for the admin + ingest tail without
        //     starving the timeline.
        //   * acquire_timeout 5s: fail fast on a wedged pool —
        //     better to surface "we're overloaded" than hang for
        //     30 s with a 200-empty response.
        //
        // Note on statement_timeout: deliberately NOT set via
        // after_connect. The schema-migration step at boot replays
        // sources/sql/*.sql, and one of those (documents.sql adds a
        // GENERATED STORED column whose backfill scans every row)
        // can run for minutes on a large prod table. A per-connection
        // statement timeout would abort that mid-flight, leaving the
        // schema in a broken state. The same partial-write hazard
        // applies to any future big migration — we keep PG's own
        // unbounded default and rely on `acquire_timeout` instead to
        // protect against pool exhaustion.
        let opts = sqlx::postgres::PgPoolOptions::new()
            .max_connections(20)
            .min_connections(2)
            .test_before_acquire(true)
            .max_lifetime(Duration::from_secs(30 * 60))
            .idle_timeout(Duration::from_secs(10 * 60))
            .acquire_timeout(Duration::from_secs(5))
            .after_connect(|conn, _| {
                Box::pin(async move {
                    sqlx::query("SET jit = off").execute(conn).await?;
                    Ok(())
                })
            });
        match opts.connect(&database_url).await {
            Ok(pool) => {
                tracing::info!("database.connected");

                // Apply schema migrations before serving traffic.
                // The .sql files are baked into the binary at compile
                // time (include_str!) so the API container doesn't
                // need to ship the sources/ tree just to migrate.
                // Each statement is wrapped in IF NOT EXISTS / CREATE
                // OR REPLACE inside the file itself, so re-running on
                // every boot is a no-op once the schema is in place.
                //
                // On failure we LOG AND CONTINUE rather than crash
                // the API — the previous binary's schema is still
                // valid for read traffic, and a wedged migration
                // surfacing as 5xx on every endpoint would be worse
                // than serving stale-schema reads. The error stays
                // visible in the logs so it can be addressed.
                if let Err(e) = run_sql_migrations(&pool).await {
                    tracing::error!(error = %e, "schema.migrate.failed — continuing with existing schema");
                } else {
                    tracing::info!("schema.migrate.complete");
                }

                // Purge old events
                match db::purge_old_events(&pool, RETENTION_DAYS).await {
                    Ok(count) => {
                        if count > 0 {
                            tracing::info!(count, "events.purge.complete");
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "events.purge.failed"),
                }

                Some(pool)
            }
            Err(e) => {
                tracing::warn!(error = %e, "database.connection.failed — data/events/ingest endpoints disabled");
                None
            }
        }
    } else {
        tracing::info!("DATABASE_URL not set — data/events/ingest endpoints disabled");
        None
    };

    // --- Load model if specified ---
    #[cfg(feature = "model")]
    let model = if let Some(ref model_path) = model_path {
        let execution_provider = if _use_cuda {
            next_plaid_onnx::ExecutionProvider::Cuda
        } else {
            next_plaid_onnx::ExecutionProvider::Cpu
        };

        let mut builder = next_plaid_onnx::Colbert::builder(model_path)
            .with_execution_provider(execution_provider)
            .with_quantized(_use_int8);

        if let Some(parallel) = _parallel_sessions {
            builder = builder.with_parallel(parallel);
        }
        if let Some(batch_size) = _batch_size {
            builder = builder.with_batch_size(batch_size);
        }
        if let Some(threads) = _threads {
            builder = builder.with_threads(threads);
        }
        if let Some(query_length) = _query_length {
            builder = builder.with_query_length(query_length);
        }
        if let Some(document_length) = _document_length {
            builder = builder.with_document_length(document_length);
        }

        match builder.build() {
            Ok(model) => {
                let cfg = model.config();
                tracing::info!(
                    model_path = %model_path.display(),
                    model_name = ?cfg.model_name(),
                    execution_provider = if _use_cuda { "cuda" } else { "cpu" },
                    quantized = _use_int8,
                    embedding_dim = model.embedding_dim(),
                    batch_size = model.batch_size(),
                    num_sessions = model.num_sessions(),
                    "model.load.complete"
                );
                Some(model)
            }
            Err(e) => {
                tracing::error!(
                    model_path = %model_path.display(),
                    error = %e,
                    "model.load.failed"
                );
                eprintln!("Error: Failed to load model from {:?}: {}", model_path, e);
                std::process::exit(1);
            }
        }
    } else {
        tracing::debug!("model.disabled");
        None
    };

    // --- Create state ---
    #[cfg(feature = "model")]
    let state = {
        let model_info = model_path.as_ref().map(|path| state::ModelInfo {
            path: path.to_string_lossy().to_string(),
            quantized: _use_int8,
        });

        let model_pool = model.map(|m| {
            let model_cfg = m.config();
            let pool_size = _model_pool_size.unwrap_or(1);

            let cached_info = state::CachedModelInfo {
                name: model_cfg.model_name().map(|s| s.to_string()),
                path: model_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default(),
                quantized: _use_int8,
                embedding_dim: m.embedding_dim(),
                batch_size: m.batch_size(),
                num_sessions: m.num_sessions(),
                query_prefix: model_cfg.query_prefix.clone(),
                document_prefix: model_cfg.document_prefix.clone(),
                query_length: model_cfg.query_length,
                document_length: model_cfg.document_length,
                do_query_expansion: model_cfg.do_query_expansion,
                uses_token_type_ids: model_cfg.uses_token_type_ids,
                mask_token_id: model_cfg.mask_token_id,
                pad_token_id: model_cfg.pad_token_id,
            };

            let model_config = state::ModelConfig {
                path: model_path.clone().unwrap(),
                use_cuda: _use_cuda,
                use_int8: _use_int8,
                parallel_sessions: _parallel_sessions,
                batch_size: _batch_size,
                threads: _threads,
                query_length: _query_length,
                document_length: _document_length,
            };

            drop(m);

            state::ModelPool {
                pool_size,
                model_config,
                cached_info,
            }
        });

        let mut app_state = AppState::with_model_pool(config, model_pool, model_info);
        if let Some(ref pool) = pg_pool {
            app_state.set_pg_pool(pool.clone());
        }
        Arc::new(app_state)
    };

    #[cfg(not(feature = "model"))]
    let state = {
        if model_path.is_some() {
            tracing::warn!("Model path specified but 'model' feature is not enabled. Encoding will be disabled.");
        }
        let mut app_state = AppState::new(config);
        if let Some(ref pool) = pg_pool {
            app_state.set_pg_pool(pool.clone());
        }
        Arc::new(app_state)
    };

    // Start buffer scanner if configured
    if let Some(ref buf_dir) = buffer_dir {
        handlers::buffer::start_buffer_scanner(state.clone(), buf_dir.clone(), buffer_interval);
    }

    // Build router
    let app = build_router(state, pg_pool);

    // Start server
    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();

    tracing::info!(
        listen_addr = %addr,
        swagger_ui = %format!("http://{}/swagger-ui", addr),
        "server.started"
    );

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .unwrap();

    tracing::info!("server.shutdown.complete");
}
