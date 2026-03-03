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
//! ## Data
//! - `GET /api/folder_tree` - Folder tree structure
//! - `GET /api/sources` - Source filter list
//! - `GET /api/health` - Simple health check
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
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod db;
mod error;
mod handlers;
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

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

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
        let governor_conf = GovernorConfigBuilder::default()
            .per_second(rate_limit_per_second)
            .burst_size(rate_limit_burst_size)
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

    // --- Ingest router (uses Arc<AppState>) ---
    let ingest_router = Router::new()
        .route("/api/bookmark", post(handlers::ingest::ingest_bookmark))
        .layer(cors.clone())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(120),
        ))
        .with_state(state);

    // --- Pipeline router (trigger run.py) ---
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

    // --- PG-backed routers (data + events) ---
    if let Some(pool) = pg_pool {
        let data_router = Router::new()
            .route("/api/folder_tree", get(handlers::data::folder_tree))
            .route("/api/sources", get(handlers::data::sources))
            .route("/api/pipeline_run", get(handlers::data::pipeline_run))
            .route("/api/health", get(handlers::data::data_health))
            .layer(cors.clone())
            .with_state(pool.clone());

        let events_router = Router::new()
            .route("/events", post(handlers::events::ingest_events))
            .route("/stats/overview", get(handlers::events::overview))
            .route("/stats/activity", get(handlers::events::activity))
            .route("/stats/top-queries", get(handlers::events::top_queries))
            .route("/stats/top-clicks", get(handlers::events::top_clicks))
            .route("/stats/sources", get(handlers::events::sources))
            .route("/stats/folders", get(handlers::events::folders))
            .layer(cors)
            .with_state(pool);

        app = app.merge(data_router).merge(events_router);
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
        match sqlx::PgPool::connect(&database_url).await {
            Ok(pool) => {
                tracing::info!("database.connected");

                // Run migrations
                db::run_migrations(&pool).await;
                tracing::info!("database.migrations.complete");

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
