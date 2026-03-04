//! MCP (Model Context Protocol) server — Streamable HTTP transport.
//!
//! Implements the MCP protocol over a single `POST /mcp` endpoint so any
//! MCP-compatible client (Claude Desktop, etc.) can navigate and search the
//! knowledge base without needing a separate process.
//!
//! # Supported methods
//! - `initialize`      — handshake, returns server capabilities
//! - `tools/list`      — enumerate available tools
//! - `tools/call`      — invoke a tool
//! - `ping`            — liveness check
//!
//! # Tools
//! - `list_folders`    — list custom folders from PostgreSQL
//! - `read_folder`     — fetch docs belonging to a folder
//! - `list_sources`    — enumerate known sources with doc counts
//! - `read_source`     — fetch recent docs from a source
//! - `search`          — ColBERT semantic search (falls back to ILIKE without model)
//! - `get_document`    — fetch one document by URL

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::PgPool;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

fn ok_response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn err_response(id: Option<Value>, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

/// Successful tool result — `isError: false` is required by MCP spec.
fn tool_result(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": false })
}

/// Tool-level error result — returned as a *successful* JSON-RPC response with `isError: true`.
/// Per spec, tool execution errors must NOT be JSON-RPC errors; they are content with isError.
fn tool_error_result(msg: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": msg }], "isError": true })
}

// ---------------------------------------------------------------------------
// Source definitions (mirrors frontend SOURCE_ICONS / buildSourceCondition)
// ---------------------------------------------------------------------------

struct SourceDef {
    key: &'static str,
    label: &'static str,
    url_pattern: Option<&'static str>, // SQL LIKE pattern on url column
    tag_pattern: Option<&'static str>, // SQL LIKE pattern on tags/title
}

const SOURCES: &[SourceDef] = &[
    SourceDef {
        key: "github.com",
        label: "GitHub",
        url_pattern: Some("%github.com%"),
        tag_pattern: None,
    },
    SourceDef {
        key: "hackernews",
        label: "Hacker News",
        url_pattern: Some("%news.ycombinator.com%"),
        tag_pattern: Some("%hackernews%"),
    },
    SourceDef {
        key: "arxiv.org",
        label: "arXiv",
        url_pattern: Some("%arxiv.org%"),
        tag_pattern: None,
    },
    SourceDef {
        key: "huggingface.co",
        label: "Hugging Face",
        url_pattern: Some("%huggingface.co%"),
        tag_pattern: None,
    },
    SourceDef {
        key: "twitter.com",
        label: "Twitter / X",
        url_pattern: Some("%twitter.com%"),
        tag_pattern: None,
    },
    SourceDef {
        key: "superuser.com",
        label: "Super User",
        url_pattern: Some("%superuser.com%"),
        tag_pattern: None,
    },
    SourceDef {
        key: "ieeexplore.ieee.org",
        label: "IEEE Xplore",
        url_pattern: Some("%ieeexplore.ieee.org%"),
        tag_pattern: None,
    },
];

// ---------------------------------------------------------------------------
// MCP handler
// ---------------------------------------------------------------------------

pub async fn mcp_handler(
    State((app_state, pool)): State<(Arc<AppState>, PgPool)>,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    // MCP spec: notifications (no `id`) must receive HTTP 202, no body.
    if req.id.is_none() && req.method.starts_with("notifications/") {
        return StatusCode::ACCEPTED.into_response();
    }

    let id = req.id.clone();
    let result = dispatch(app_state, pool, req).await;
    match result {
        Ok(v) => Json(ok_response(id, v)).into_response(),
        Err((code, msg)) => Json(err_response(id, code, &msg)).into_response(),
    }
}

async fn dispatch(
    state: Arc<AppState>,
    pool: PgPool,
    req: JsonRpcRequest,
) -> Result<Value, (i32, String)> {
    match req.method.as_str() {
        "initialize" => Ok(handle_initialize()),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(handle_tools_list()),
        "tools/call" => {
            let params = req.params.unwrap_or(json!({}));
            let name = params
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing tool name".to_string()))?;
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            // Tool errors are content-level (isError: true), not JSON-RPC errors.
            Ok(handle_tool_call(state, pool, name, args).await)
        }
        _ => Err((-32601, format!("Method not found: {}", req.method))),
    }
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

fn handle_initialize() -> Value {
    json!({
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "knowledge-mcp",
            "version": "1.0.0"
        }
    })
}

// ---------------------------------------------------------------------------
// tools/list
// ---------------------------------------------------------------------------

fn handle_tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "list_folders",
                "description": "List custom folders in the knowledge base. Returns a flat list of all folders with their id, label, filter_type (search/tag/urls/none), and parent_id. Use parent_id to reconstruct the folder hierarchy. Call read_folder(id) to get the documents inside any folder.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "parent_id": {
                            "type": "string",
                            "description": "If provided, return only direct children of this folder id. Omit to get all folders."
                        }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "read_folder",
                "description": "Read documents belonging to a custom folder. Automatically resolves the folder's filter type: 'search' runs a text query, 'tag' filters by tags, 'urls' returns curated URLs. Pinned docs always appear first; excluded docs are removed. Returns url, title, summary, date, and tags for each document.",
                "inputSchema": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": { "type": "string", "description": "Folder id (from list_folders)" },
                        "limit": { "type": "integer", "description": "Max documents to return (default 50, max 200)" },
                        "offset": { "type": "integer", "description": "Pagination offset (default 0)" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "list_sources",
                "description": "List all indexed sources with document counts. Sources include: GitHub, Hacker News, arXiv, Hugging Face, Twitter/X, Super User, IEEE Xplore. Returns source key (used with read_source), label, and doc_count. Also returns total_docs across all sources.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            {
                "name": "read_source",
                "description": "Fetch recent documents from a specific source, ordered by date descending. Returns url, title, summary, date, and tags. Use list_sources first to get valid source keys (e.g. 'github.com', 'hackernews', 'arxiv.org').",
                "inputSchema": {
                    "type": "object",
                    "required": ["source_key"],
                    "properties": {
                        "source_key": {
                            "type": "string",
                            "description": "Source key from list_sources. Valid values: 'github.com', 'hackernews', 'arxiv.org', 'huggingface.co', 'twitter.com', 'superuser.com', 'ieeexplore.ieee.org'"
                        },
                        "limit": { "type": "integer", "description": "Max documents to return (default 50, max 200)" },
                        "offset": { "type": "integer", "description": "Pagination offset (default 0)" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "search",
                "description": "Semantic search over the entire knowledge base using ColBERT multi-vector retrieval (falls back to keyword search if model unavailable). Returns documents ranked by relevance with title, summary, date, tags, and relevance score. Best for conceptual or natural language queries. For browsing by topic, prefer list_folders or read_source.",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "Natural language search query, e.g. 'transformer attention mechanisms' or 'rust async runtime'" },
                        "limit": { "type": "integer", "description": "Max results to return (default 20, max 100)" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "get_document",
                "description": "Fetch full metadata for a single document by its URL. Returns url, title, summary, date, and tags. Useful after search or list operations to retrieve the full summary of a specific document.",
                "inputSchema": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": { "type": "string", "description": "Document URL (exact match)" }
                    },
                    "additionalProperties": false
                }
            }
        ],
        "nextCursor": null
    })
}

// ---------------------------------------------------------------------------
// tools/call dispatcher
// ---------------------------------------------------------------------------

async fn handle_tool_call(state: Arc<AppState>, pool: PgPool, name: &str, args: Value) -> Value {
    let result = match name {
        "list_folders" => tool_list_folders(pool, args).await,
        "read_folder" => tool_read_folder(pool, args).await,
        "list_sources" => tool_list_sources(pool).await,
        "read_source" => tool_read_source(pool, args).await,
        "search" => tool_search(state, pool, args).await,
        "get_document" => tool_get_document(pool, args).await,
        _ => return tool_error_result(&format!("Unknown tool: {name}")),
    };
    match result {
        Ok(v) => v,
        Err(e) => tool_error_result(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool: list_folders
// ---------------------------------------------------------------------------

async fn tool_list_folders(pool: PgPool, args: Value) -> Result<Value, String> {
    let parent_filter = args.get("parent_id").and_then(|v| v.as_str());

    let rows: Vec<(String, String, String, Option<String>)> = if let Some(pid) = parent_filter {
        sqlx::query_as(
            "SELECT id, label, filter_type, parent_id FROM custom_folders
             WHERE parent_id = $1 ORDER BY created_at ASC",
        )
        .bind(pid)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?
    } else {
        sqlx::query_as(
            "SELECT id, label, filter_type, parent_id FROM custom_folders
             ORDER BY created_at ASC",
        )
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?
    };

    let folders: Vec<Value> = rows
        .into_iter()
        .map(|(id, label, filter_type, parent_id)| {
            json!({ "id": id, "label": label, "filter_type": filter_type, "parent_id": parent_id })
        })
        .collect();

    Ok(tool_result(
        serde_json::to_string(&json!({ "folders": folders })).unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: read_folder
// ---------------------------------------------------------------------------

#[derive(sqlx::FromRow)]
struct FolderRow {
    filter_type: String,
    search_query: String,
    tag_filter: Vec<String>,
    tag_intersect: bool,
    live: bool,
    urls: Vec<String>,
    pinned_urls: Vec<String>,
    excluded_docs: serde_json::Value,
    sort_by: String,
}

#[derive(sqlx::FromRow, Serialize)]
struct DocRow {
    url: String,
    title: String,
    summary: String,
    date: Option<String>,
    tags: Vec<String>,
}

async fn tool_read_folder(pool: PgPool, args: Value) -> Result<Value, String> {
    let id = args
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or("Missing id")?;
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(50);
    let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);

    let folder: Option<FolderRow> = sqlx::query_as(
        "SELECT filter_type, search_query, tag_filter, tag_intersect, live, urls,
                pinned_urls, excluded_docs, sort_by
         FROM custom_folders WHERE id = $1",
    )
    .bind(id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| e.to_string())?;

    let folder = folder.ok_or_else(|| format!("Folder not found: {id}"))?;

    let excluded_urls: Vec<String> = folder
        .excluded_docs
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.get("url").and_then(|u| u.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Resolve docs based on filter type
    let use_stored = !folder.live || folder.filter_type == "urls";
    let mut docs: Vec<DocRow> = if use_stored && !folder.urls.is_empty() {
        sqlx::query_as(
            "SELECT url, title, summary, date::TEXT as date, tags
             FROM documents WHERE url = ANY($1)
             ORDER BY date DESC NULLS LAST LIMIT $2 OFFSET $3",
        )
        .bind(&folder.urls)
        .bind(limit)
        .bind(offset)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?
    } else if folder.filter_type == "search" && !folder.search_query.is_empty() {
        sqlx::query_as(
            "SELECT url, title, summary, date::TEXT as date, tags
             FROM documents
             WHERE (title ILIKE $1 OR summary ILIKE $1)
             ORDER BY date DESC NULLS LAST LIMIT $2 OFFSET $3",
        )
        .bind(format!("%{}%", folder.search_query))
        .bind(limit)
        .bind(offset)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?
    } else if folder.filter_type == "tag" && !folder.tag_filter.is_empty() {
        if folder.tag_intersect {
            sqlx::query_as(
                "SELECT url, title, summary, date::TEXT as date, tags
                 FROM documents WHERE tags @> $1
                 ORDER BY date DESC NULLS LAST LIMIT $2 OFFSET $3",
            )
            .bind(&folder.tag_filter)
            .bind(limit)
            .bind(offset)
            .fetch_all(&pool)
            .await
            .map_err(|e| e.to_string())?
        } else {
            sqlx::query_as(
                "SELECT url, title, summary, date::TEXT as date, tags
                 FROM documents WHERE tags && $1
                 ORDER BY date DESC NULLS LAST LIMIT $2 OFFSET $3",
            )
            .bind(&folder.tag_filter)
            .bind(limit)
            .bind(offset)
            .fetch_all(&pool)
            .await
            .map_err(|e| e.to_string())?
        }
    } else {
        Vec::new()
    };

    // Apply sort
    if folder.sort_by == "date" {
        docs.sort_by(|a, b| {
            b.date
                .as_deref()
                .unwrap_or("")
                .cmp(a.date.as_deref().unwrap_or(""))
        });
    }

    // Load pinned docs and prepend
    let mut pinned: Vec<DocRow> = if !folder.pinned_urls.is_empty() {
        sqlx::query_as(
            "SELECT url, title, summary, date::TEXT as date, tags
             FROM documents WHERE url = ANY($1)",
        )
        .bind(&folder.pinned_urls)
        .fetch_all(&pool)
        .await
        .map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };

    // Apply exclusions
    let excluded_set: std::collections::HashSet<&str> =
        excluded_urls.iter().map(|s| s.as_str()).collect();
    let pinned_set: std::collections::HashSet<&str> =
        folder.pinned_urls.iter().map(|s| s.as_str()).collect();

    pinned.retain(|d| !excluded_set.contains(d.url.as_str()));
    docs.retain(|d| !excluded_set.contains(d.url.as_str()) && !pinned_set.contains(d.url.as_str()));

    let mut all_docs: Vec<&DocRow> = pinned.iter().chain(docs.iter()).collect();
    all_docs.truncate(limit as usize);

    let result = json!({
        "folder_id": id,
        "docs": all_docs.iter().map(|d| json!({
            "url": d.url,
            "title": d.title,
            "summary": d.summary,
            "date": d.date,
            "tags": d.tags,
        })).collect::<Vec<_>>(),
    });

    Ok(tool_result(serde_json::to_string(&result).unwrap()))
}

// ---------------------------------------------------------------------------
// Tool: list_sources
// ---------------------------------------------------------------------------

async fn tool_list_sources(pool: PgPool) -> Result<Value, String> {
    let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM documents")
        .fetch_one(&pool)
        .await
        .map_err(|e| e.to_string())?;

    let mut sources = Vec::new();
    for src in SOURCES {
        let count: (i64,) = if let Some(up) = src.url_pattern {
            if let Some(tp) = src.tag_pattern {
                sqlx::query_as(
                    "SELECT COUNT(*) FROM documents WHERE url ILIKE $1 OR title ILIKE $2",
                )
                .bind(up)
                .bind(tp)
                .fetch_one(&pool)
                .await
                .map_err(|e| e.to_string())?
            } else {
                sqlx::query_as("SELECT COUNT(*) FROM documents WHERE url ILIKE $1")
                    .bind(up)
                    .fetch_one(&pool)
                    .await
                    .map_err(|e| e.to_string())?
            }
        } else {
            (0,)
        };
        sources.push(json!({
            "key": src.key,
            "label": src.label,
            "doc_count": count.0,
        }));
    }

    let result = json!({
        "total_docs": total.0,
        "sources": sources,
    });

    Ok(tool_result(serde_json::to_string(&result).unwrap()))
}

// ---------------------------------------------------------------------------
// Tool: read_source
// ---------------------------------------------------------------------------

async fn tool_read_source(pool: PgPool, args: Value) -> Result<Value, String> {
    let source_key = args
        .get("source_key")
        .and_then(|v| v.as_str())
        .ok_or("Missing source_key")?;
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(50);
    let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);

    let src = SOURCES
        .iter()
        .find(|s| s.key == source_key)
        .ok_or_else(|| format!("Unknown source: {source_key}"))?;

    let docs: Vec<DocRow> = if let Some(url_pat) = src.url_pattern {
        if let Some(tag_pat) = src.tag_pattern {
            sqlx::query_as(
                "SELECT url, title, summary, date::TEXT as date, tags
                 FROM documents WHERE url ILIKE $1 OR title ILIKE $2
                 ORDER BY date DESC NULLS LAST LIMIT $3 OFFSET $4",
            )
            .bind(url_pat)
            .bind(tag_pat)
            .bind(limit)
            .bind(offset)
            .fetch_all(&pool)
            .await
            .map_err(|e| e.to_string())?
        } else {
            sqlx::query_as(
                "SELECT url, title, summary, date::TEXT as date, tags
                 FROM documents WHERE url ILIKE $1
                 ORDER BY date DESC NULLS LAST LIMIT $2 OFFSET $3",
            )
            .bind(url_pat)
            .bind(limit)
            .bind(offset)
            .fetch_all(&pool)
            .await
            .map_err(|e| e.to_string())?
        }
    } else {
        Vec::new()
    };

    let result = json!({
        "source": source_key,
        "docs": docs.iter().map(|d| json!({
            "url": d.url,
            "title": d.title,
            "summary": d.summary,
            "date": d.date,
            "tags": d.tags,
        })).collect::<Vec<_>>(),
    });

    Ok(tool_result(serde_json::to_string(&result).unwrap()))
}

// ---------------------------------------------------------------------------
// Tool: search
// ---------------------------------------------------------------------------

async fn tool_search(
    #[allow(unused_variables)] state: Arc<AppState>,
    pool: PgPool,
    args: Value,
) -> Result<Value, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("Missing query")?
        .to_string();
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(20);

    // Try ColBERT in-process search first
    #[cfg(feature = "model")]
    if state.has_model() {
        if let Ok(docs) = colbert_search(&state, &pool, &query, limit).await {
            return Ok(tool_result(
                serde_json::to_string(&json!({ "query": query, "docs": docs })).unwrap(),
            ));
        }
    }

    // Fallback: PostgreSQL ILIKE search
    let pattern = format!("%{query}%");
    let docs: Vec<DocRow> = sqlx::query_as(
        "SELECT url, title, summary, date::TEXT as date, tags
         FROM documents
         WHERE title ILIKE $1 OR summary ILIKE $1
         ORDER BY date DESC NULLS LAST LIMIT $2",
    )
    .bind(&pattern)
    .bind(limit)
    .fetch_all(&pool)
    .await
    .map_err(|e| e.to_string())?;

    let result = json!({
        "query": query,
        "search_type": "text",
        "docs": docs.iter().map(|d| json!({
            "url": d.url,
            "title": d.title,
            "summary": d.summary,
            "date": d.date,
            "tags": d.tags,
        })).collect::<Vec<_>>(),
    });

    Ok(tool_result(serde_json::to_string(&result).unwrap()))
}

#[cfg(feature = "model")]
async fn colbert_search(
    state: &Arc<AppState>,
    pool: &PgPool,
    query: &str,
    limit: i64,
) -> Result<Vec<Value>, String> {
    use next_plaid::SearchParameters;

    use crate::handlers::encode::encode_texts_internal;
    use crate::handlers::search::fetch_metadata_for_docs;
    use crate::models::InputType;

    let embeddings =
        encode_texts_internal(state.clone(), &[query.to_string()], InputType::Query, None)
            .await
            .map_err(|e| e.to_string())?;

    // Try the "knowledge" index (default index name used by the pipeline)
    let index_name = "knowledge";
    let idx = state
        .get_index_for_read(index_name)
        .map_err(|e| e.to_string())?;

    let params = SearchParameters {
        top_k: limit as usize,
        n_ivf_probe: 8,
        n_full_scores: 4096,
        batch_size: 2000,
        ..Default::default()
    };

    let result = idx
        .search(&embeddings[0], &params, None)
        .map_err(|e| e.to_string())?;

    let path_str = state.index_path(index_name).to_string_lossy().to_string();
    let metadata =
        fetch_metadata_for_docs(&path_str, &result.passage_ids).map_err(|e| e.to_string())?;

    // Join with PostgreSQL for full doc data
    let urls: Vec<String> = metadata
        .iter()
        .filter_map(|m| {
            m.as_ref()
                .and_then(|v| v.get("url").and_then(|u| u.as_str()).map(String::from))
        })
        .collect();

    if urls.is_empty() {
        return Ok(Vec::new());
    }

    // Preserve score ordering by fetching all and reordering
    let rows: Vec<DocRow> = sqlx::query_as(
        "SELECT url, title, summary, date::TEXT as date, tags
         FROM documents WHERE url = ANY($1)",
    )
    .bind(&urls)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let url_to_doc: std::collections::HashMap<&str, &DocRow> =
        rows.iter().map(|r| (r.url.as_str(), r)).collect();

    let url_to_score: std::collections::HashMap<&str, f32> = urls
        .iter()
        .zip(result.scores.iter())
        .map(|(u, s)| (u.as_str(), *s))
        .collect();

    let mut docs: Vec<Value> = urls
        .iter()
        .filter_map(|url| {
            url_to_doc.get(url.as_str()).map(|doc| {
                json!({
                    "url": doc.url,
                    "title": doc.title,
                    "summary": doc.summary,
                    "date": doc.date,
                    "tags": doc.tags,
                    "score": url_to_score.get(url.as_str()).copied().unwrap_or(0.0),
                })
            })
        })
        .collect();

    docs.truncate(limit as usize);
    Ok(docs)
}

// ---------------------------------------------------------------------------
// Tool: get_document
// ---------------------------------------------------------------------------

async fn tool_get_document(pool: PgPool, args: Value) -> Result<Value, String> {
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or("Missing url")?;

    let doc: Option<DocRow> = sqlx::query_as(
        "SELECT url, title, summary, date::TEXT as date, tags
         FROM documents WHERE url = $1",
    )
    .bind(url)
    .fetch_optional(&pool)
    .await
    .map_err(|e| e.to_string())?;

    match doc {
        Some(d) => {
            let result = json!({
                "url": d.url,
                "title": d.title,
                "summary": d.summary,
                "date": d.date,
                "tags": d.tags,
            });
            Ok(tool_result(serde_json::to_string(&result).unwrap()))
        }
        None => Err(format!("Document not found: {url}")),
    }
}
