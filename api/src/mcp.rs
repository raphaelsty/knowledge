//! MCP (Model Context Protocol) server — Streamable HTTP transport.
//!
//! Multi-personality knowledge base server. All data is read from the
//! shared PostgreSQL database (the `data/{slug}/*.json` snapshots were
//! retired in commit 38cfe61 — Postgres is the only source of truth).
//! ColBERT semantic search still runs in-process via the AppState index
//! pool when the `model` feature is built in.
//!
//! # Supported methods
//! - `initialize`             — handshake, returns server capabilities
//! - `tools/list`             — enumerate available tools
//! - `tools/call`             — invoke a tool
//! - `ping`                   — liveness check
//!
//! # Tools — parity with the UI / REST API
//! - `list_personalities`   — vip personalities, ordered by doc count
//! - `search_personalities` — keyword filter over name/desc/category/slug
//! - `get_personality`      — full profile (counts, links, sources, …)
//! - `search`               — ColBERT search inside one personality, with
//!                            source/tag filters, optional date sort
//! - `search_across`        — multi-personality fan-out + RRF fusion
//! - `latest`               — most-recent docs (mirrors the UI default)
//! - `find_similar`         — ColBERT "Similar" button semantics
//! - `list_sources`         — `documents.source` keys + counts
//! - `list_tags`            — tag frequencies across the library
//! - `get_document`         — single doc by URL
//! - `feed`                 — cross-library activity feed
//!                            (mirrors `GET /api/feed`)
//! - `intersect_documents`  — URLs shared across N libraries
//!                            (mirrors `GET /api/users/intersect`)
//! - `save_document`        — upload a doc into the bearer-token holder's
//!                            library. Requires `Authorization: Bearer kn_…`
//!                            on the HTTP request. Mirrors `POST
//!                            /api/me/documents`.
//! - `my_library`           — search / list the bearer-token holder's own
//!                            library. Bearer-authed, no slug argument.
//! - `my_timeline`          — recent docs from the bearer-token holder's
//!                            follow graph (followees ∪ self). Mirrors
//!                            `GET /api/timeline`.

#![allow(clippy::doc_overindented_list_items, clippy::type_complexity)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
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
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn err_response(id: Option<Value>, code: i32, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Successful tool result — `isError: false` is required by MCP spec.
fn tool_result(text: String) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": false })
}

/// Tool-level error returned as a *successful* JSON-RPC response with
/// `isError: true` so MCP clients can surface it to the LLM as a normal
/// tool failure rather than a transport error.
fn tool_error_result(msg: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": msg }], "isError": true })
}

// ---------------------------------------------------------------------------
// Pagination — shared contract across every list-returning tool.
//
//   - `page`     : 1-indexed page number (default 1)
//   - `per_page` : page size (per-tool default, per-tool max)
//   - `limit`    : deprecated alias accepted for backward-compat
//   - `top`      : deprecated alias for `list_tags`
//
// Responses carry `pagination: {page, per_page, total, total_pages,
// has_more}`. Keeping payloads small matters for MCP clients where every
// result is fed back to an LLM — pagination lets the model pull more
// only if it actually needs it.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Pagination {
    page: usize,
    per_page: usize,
}

impl Pagination {
    fn start(&self) -> usize {
        (self.page - 1) * self.per_page
    }
    fn end(&self) -> usize {
        self.page * self.per_page
    }
}

fn parse_pagination(args: &Value, default_per_page: usize, max_per_page: usize) -> Pagination {
    let page = args
        .get("page")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        .max(1) as usize;
    let per_page_raw = args
        .get("per_page")
        .and_then(|v| v.as_i64())
        .or_else(|| args.get("limit").and_then(|v| v.as_i64()))
        .or_else(|| args.get("top").and_then(|v| v.as_i64()))
        .unwrap_or(default_per_page as i64);
    let per_page = per_page_raw.max(1).min(max_per_page as i64) as usize;
    Pagination { page, per_page }
}

/// Slice `items` for the requested page and return `(page_items, metadata)`.
fn paginate<T: Clone>(items: &[T], pg: Pagination) -> (Vec<T>, Value) {
    let total = items.len();
    let total_pages = if total == 0 {
        0
    } else {
        total.div_ceil(pg.per_page)
    };
    let start = pg.start().min(total);
    let end = pg.end().min(total);
    let slice = items[start..end].to_vec();
    let has_more = end < total;
    (
        slice,
        json!({
            "page": pg.page,
            "per_page": pg.per_page,
            "total": total,
            "total_pages": total_pages,
            "has_more": has_more,
        }),
    )
}

fn parse_string_set(v: Option<&Value>) -> Option<Vec<String>> {
    v.and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
}

fn truncate_summary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cut = s.floor_char_boundary(max_chars);
    let truncated = &s[..cut];
    match truncated.rfind(' ') {
        Some(pos) => format!("{}…", &truncated[..pos]),
        None => format!("{truncated}…"),
    }
}

// ---------------------------------------------------------------------------
// MCP handler
// ---------------------------------------------------------------------------

pub async fn mcp_handler(
    State(app_state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    // MCP spec: notifications (no `id`) must receive HTTP 202, no body.
    if req.id.is_none() && req.method.starts_with("notifications/") {
        return StatusCode::ACCEPTED.into_response();
    }
    let id = req.id.clone();
    match dispatch(app_state, headers, req).await {
        Ok(v) => Json(ok_response(id, v)).into_response(),
        Err((code, msg)) => Json(err_response(id, code, &msg)).into_response(),
    }
}

async fn dispatch(
    state: Arc<AppState>,
    headers: HeaderMap,
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
            Ok(handle_tool_call(state, &headers, name, args).await)
        }
        _ => Err((-32601, format!("Method not found: {}", req.method))),
    }
}

fn handle_initialize() -> Value {
    json!({
        "protocolVersion": "2025-03-26",
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "knowledge-mcp", "version": "3.0.0" }
    })
}

// ---------------------------------------------------------------------------
// tools/list
// ---------------------------------------------------------------------------

fn handle_tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "list_personalities",
                "description": "List the public knowledge libraries (vip personalities) ordered by document count desc. Returns slug (use as identifier in other tools), name, description, category, document_count, follower counts, and avatar. Paginated.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page": { "type": "integer", "description": "1-indexed page number (default 1)" },
                        "per_page": { "type": "integer", "description": "Items per page (default 50, max 200). `limit` accepted as deprecated alias." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "search_personalities",
                "description": "Rank personalities by how well they match a free-text query. Two tiers are merged: VIPs are ranked by ColBERT semantic search over their documents (top 200 passages from the cross-library `__all__` index, grouped by `owner`, size-normalized as `count / sqrt(total_docs)` so a focused library beats a noisy giant when its hits are denser — same algorithm as the welcome page). Non-VIPs aren't in `__all__`, so they're matched lexically against name/description/category/slug and appended after the ColBERT-ranked VIPs. Each row carries a `tier` field (`\"vip\"` vs `\"name-match\"`) so the caller can tell them apart. Falls back to a pure substring path across both tiers when ColBERT is unavailable or the query is shorter than 3 chars. Paginated.",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "Free-text query, e.g. 'transformer', 'Karpathy', 'NLP'" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 50, max 200." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "get_personality",
                "description": "Fetch the full profile of one personality by slug — counts, links, configured sources, follower counts, citation count, etc. Use this when the LLM needs the metadata of a specific library before searching it.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality"],
                    "properties": {
                        "personality": { "type": "string", "description": "Personality slug" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "search",
                "description": "Semantic search inside a single personality's library using ColBERT multi-vector retrieval, with popularity blended in: broadly-shared resources rank above equally-relevant long-tail docs. Falls back to a SQL keyword search (title/summary/tags ILIKE) if the model or index isn't available. Supports source + tag filters and optional date sort. Near-duplicates pointing at the same resource (a paper + the tweets linking it) collapse into one row. Each doc carries title, summary, date, tags, source, score, plus the cross-personality roll-up: `sharers` (every personality that saved the resource, with name/avatar/follower counts), `sharer_count`, `anchor_url` (canonical resource), `aggregated_urls` (companion docs that collapsed into it), `linked_urls` (resources the post links to) and `feed_score` (popularity).",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality", "query"],
                    "properties": {
                        "personality": { "type": "string", "description": "Personality slug" },
                        "query": { "type": "string", "description": "Natural-language query, e.g. 'transformer attention mechanisms'" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 20, max 100." },
                        "sources": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Optional source-key filter (OR). Valid keys: github, twitter, arxiv, hackernews, huggingface, youtube, reddit, scholar, dblp, zotero, wikipedia, semantic_scholar, or a website hostname. Discover via list_sources."
                        },
                        "tags": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Optional tag filter — AND semantics across the supplied tags (a doc must carry every tag in the list, looked up across both `tags` and `extra_tags`). Discover via list_tags."
                        },
                        "sort_by_date": {
                            "type": "boolean",
                            "description": "If true, return results sorted by date desc (most recent first). Default false (ColBERT relevance)."
                        }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "search_across",
                "description": "Run `search` across multiple personalities in parallel and fuse the per-library rankings with Reciprocal Rank Fusion (RRF), merging by canonical resource (anchor) so different posts of the same paper collapse into one row. Order-independent: a doc that several libraries rank highly bubbles to the top regardless of which slug appeared first. Each merged hit carries a `libraries` array listing every slug whose own ranking surfaced it, plus the same `sharers` / `sharer_count` / `anchor_url` / `aggregated_urls` / `linked_urls` roll-up as `search`.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personalities", "query"],
                    "properties": {
                        "personalities": {
                            "type": "array", "items": { "type": "string" },
                            "description": "List of personality slugs (≥ 2 to be useful, max 10)"
                        },
                        "query": { "type": "string" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 20, max 100." },
                        "sources": { "type": "array", "items": { "type": "string" } },
                        "tags": { "type": "array", "items": { "type": "string" } }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "latest",
                "description": "Most-recently-dated documents from a personality's library, with optional source / tag / url filters. Mirrors the UI's default browse view — use this when the user wants 'what has X been saving lately' rather than a topical search.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality"],
                    "properties": {
                        "personality": { "type": "string" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 30, max 200." },
                        "sources": { "type": "array", "items": { "type": "string" }, "description": "Source-key filter (OR)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tag filter (AND across supplied tags)" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "find_similar",
                "description": "Given a document URL in a personality's library, return semantically-similar documents from that same library using ColBERT — identical to the UI's 'Similar' button (query = title + tags + first 20 words of summary; the source URL is excluded). Falls back to a tag-overlap heuristic when ColBERT is unavailable.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality", "url"],
                    "properties": {
                        "personality": { "type": "string" },
                        "url": { "type": "string", "description": "URL of the source document (must exist in this library)" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 10, max 50." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "list_sources",
                "description": "List source-type buckets for a personality (github, arxiv, hackernews, blogs, …) with their document counts, ordered by count desc. Useful for source filter discovery before calling search/latest.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality"],
                    "properties": {
                        "personality": { "type": "string" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 50, max 200." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "list_tags",
                "description": "List tags used across a personality's library, ranked by document count. Combines `tags` and `extra_tags`. Useful for tag-filter discovery before calling search/latest.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality"],
                    "properties": {
                        "personality": { "type": "string" },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 100, max 1000. `top` accepted as alias." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "get_document",
                "description": "Fetch full metadata for a single document by URL from a personality's library: title, summary, date, tags, extra_tags, source, source_url, indexed flag.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personality", "url"],
                    "properties": {
                        "personality": { "type": "string" },
                        "url": { "type": "string", "description": "Document URL (exact match)" }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "feed",
                "description": "Activity feed. When the MCP HTTP request carries `Authorization: Bearer kn_...` this returns the bearer holder's personal follow-graph timeline — the exact docs the UI's feed renders for that user (followees ∪ self, deduped by URL, most-recent first). When no bearer is present, falls back to the public cross-library aggregate sorted by date desc then by sharer count. Each row carries the list of personalities that have the URL in their library (slug, name, avatar, follower counts). Mirrors `GET /api/timeline` (authed) or `GET /api/feed` (anonymous).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 50, max 500. `limit` accepted as alias." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "save_document",
                "description": "Upload a document into the authenticated user's library. The owning user is taken from the `Authorization: Bearer kn_...` header on the MCP HTTP request — there is no `personality`/`slug` argument and the token cannot be redirected. URL is the natural key inside that user's library: re-saving the same URL updates the existing row (upsert on (user_id, url)). Useful for clipping a page from the web, the LLM's context, or another tool's output.",
                "inputSchema": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": { "type": "string", "description": "Canonical URL of the document. Required." },
                        "title": { "type": "string", "description": "Display title. Optional but recommended." },
                        "summary": { "type": "string", "description": "Short summary or excerpt. Optional." },
                        "date": { "type": "string", "description": "ISO date (YYYY-MM-DD). Optional — leave blank if unknown." },
                        "tags": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Primary tags. Optional."
                        },
                        "extra_tags": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Secondary tags (free-form, not reconciled against the user's vocabulary). Optional."
                        },
                        "source": { "type": "string", "description": "Source key (e.g. 'manual', 'github', 'arxiv', a hostname). Defaults to '' if omitted." },
                        "source_url": { "type": "string", "description": "Optional URL for the source bucket (e.g. the repo or feed the doc came from)." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "intersect_documents",
                "description": "Return URLs that appear in *every* one of the listed libraries — the multi-library shared-resource pool. Each result carries an `owners` array of slugs that hold it. Ordered by owner_count desc, then date desc. Mirrors `GET /api/users/intersect`. Pass between 2 and 10 slugs.",
                "inputSchema": {
                    "type": "object",
                    "required": ["personalities"],
                    "properties": {
                        "personalities": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Slugs to intersect (2–10)"
                        },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 50, max 500. `limit` accepted as alias." }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "my_library",
                "description": "Search or list documents in the bearer-token holder's own library. The owning user is taken from the `Authorization: Bearer kn_...` header on the MCP HTTP request — there is no `personality`/`slug` argument and the token cannot be redirected. If `query` is supplied, runs a ColBERT semantic search (falls back to SQL keyword search when the model/index is unavailable). If `query` is omitted, returns the most recently dated documents (mirrors `latest`). Supports source + tag filters and optional date sort. Mint a token at `/profile → API tokens`.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Optional natural-language query. When omitted, returns most-recent docs instead of a ranked search." },
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 20 with query, 30 without. Max 200." },
                        "sources": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Optional source-key filter (OR). Discover via list_sources on your own slug."
                        },
                        "tags": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Optional tag filter — AND across the supplied tags (looked up across `tags` and `extra_tags`)."
                        },
                        "sort_by_date": {
                            "type": "boolean",
                            "description": "Only applies when `query` is set. If true, sort hits by date desc instead of ColBERT relevance. Default false."
                        }
                    },
                    "additionalProperties": false
                }
            },
            {
                "name": "my_timeline",
                "description": "Recent documents from the bearer-token holder's follow graph (followees ∪ self), most recent first. Same payload shape as `feed` — each row carries the list of personalities sharing the URL. Mirrors `GET /api/timeline`. Bearer-authed; the token decides whose follow graph is used. Useful for asking 'what's new in my world?'.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page": { "type": "integer" },
                        "per_page": { "type": "integer", "description": "Default 50, max 200. `limit` accepted as alias." },
                        "before": { "type": "string", "description": "ISO-8601 cursor — only return docs strictly older than this date." },
                        "sources": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Source-key include filter (OR)."
                        },
                        "exclude_sources": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Source-key exclude filter."
                        },
                        "tags": {
                            "type": "array", "items": { "type": "string" },
                            "description": "Tag filter — AND across the supplied tags."
                        }
                    },
                    "additionalProperties": false
                }
            }
        ]
    })
}

// ---------------------------------------------------------------------------
// tools/call dispatcher
// ---------------------------------------------------------------------------

async fn handle_tool_call(
    state: Arc<AppState>,
    headers: &HeaderMap,
    name: &str,
    args: Value,
) -> Value {
    let pool = match state.pg_pool.clone() {
        Some(p) => p,
        None => {
            return tool_error_result(
                "PostgreSQL is not configured on this server — MCP requires DATABASE_URL.",
            );
        }
    };

    let result: Result<Value, String> = match name {
        "list_personalities" => tool_list_personalities(&pool, args).await,
        "search_personalities" => tool_search_personalities(state.clone(), &pool, args).await,
        "get_personality" => tool_get_personality(&pool, args).await,
        "search" => tool_search(state.clone(), &pool, args).await,
        "search_across" => tool_search_across(state.clone(), &pool, args).await,
        "latest" => tool_latest(&pool, args).await,
        "find_similar" => tool_find_similar(state.clone(), &pool, args).await,
        "list_sources" => tool_list_sources(&pool, args).await,
        "list_tags" => tool_list_tags(&pool, args).await,
        "get_document" => tool_get_document(&pool, args).await,
        "feed" => tool_feed(&pool, headers, args).await,
        "intersect_documents" => tool_intersect_documents(&pool, args).await,
        "save_document" => tool_save_document(&pool, headers, args).await,
        "my_library" => tool_my_library(state.clone(), &pool, headers, args).await,
        "my_timeline" => tool_my_timeline(&pool, headers, args).await,
        _ => return tool_error_result(&format!("Unknown tool: {name}")),
    };
    match result {
        Ok(v) => v,
        Err(e) => tool_error_result(&e),
    }
}

// ---------------------------------------------------------------------------
// PG helpers
// ---------------------------------------------------------------------------

/// Resolve a slug → `(user_id, index_name)`. Returned as a tool error when
/// the slug doesn't exist.
async fn resolve_personality(pool: &PgPool, slug: &str) -> Result<(i64, String), String> {
    sqlx::query_as::<_, (i64, String)>("SELECT id, index_name FROM users WHERE username = $1")
        .bind(slug)
        .fetch_optional(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?
        .ok_or_else(|| format!("Personality not found: {slug}"))
}

/// Document metadata as we expose it through MCP — common shape for
/// search / latest / find_similar / get_document.
#[derive(Clone)]
struct DocMeta {
    url: String,
    title: String,
    summary: String,
    date: String,
    tags: Vec<String>,
    extra_tags: Vec<String>,
    source: String,
    source_url: Option<String>,
    indexed: bool,
}

impl DocMeta {
    fn to_json(&self) -> Value {
        json!({
            "url": self.url,
            "title": self.title,
            "summary": truncate_summary(&self.summary, 200),
            "date": self.date,
            "tags": self.tags,
            "extra-tags": self.extra_tags,
            "source": self.source,
            "source_url": self.source_url,
            "indexed": self.indexed,
        })
    }
}

fn row_to_doc(
    row: (
        String,
        String,
        String,
        String,
        Vec<String>,
        Vec<String>,
        String,
        Option<String>,
        bool,
    ),
) -> DocMeta {
    let (url, title, summary, date, tags, extra_tags, source, source_url, indexed) = row;
    DocMeta {
        url,
        title,
        summary,
        date,
        tags,
        extra_tags,
        source,
        source_url,
        indexed,
    }
}

#[allow(clippy::type_complexity)]
async fn fetch_docs_by_urls(
    pool: &PgPool,
    user_id: i64,
    urls: &[String],
) -> Result<HashMap<String, DocMeta>, String> {
    if urls.is_empty() {
        return Ok(HashMap::new());
    }
    let rows: Vec<(
        String,
        String,
        String,
        String,
        Vec<String>,
        Vec<String>,
        String,
        Option<String>,
        bool,
    )> = sqlx::query_as(
        "SELECT d.url,
                COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                COALESCE(to_char(d.date, 'YYYY-MM-DD'), '') AS date,
                d.tags, d.extra_tags, d.source, d.source_url, d.indexed
           FROM documents d
          WHERE d.user_id = $1 AND d.url = ANY($2)",
    )
    .bind(user_id)
    .bind(urls)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    Ok(rows
        .into_iter()
        .map(|r| {
            let d = row_to_doc(r);
            (d.url.clone(), d)
        })
        .collect())
}

/// Collapse docs by anchor URL and attach the cross-personality roll-up
/// — the same aggregation the web feed renders for each card:
///   • `sharers` / `sharer_count` — every personality that saved this
///     resource (slug, name, avatar, follower counts), from feed_snapshot
///   • `anchor_url` / `aggregated_urls` — the canonical resource and the
///     companion docs (tweets, abs/pdf pages) that collapsed into it
///   • `linked_urls` — inline resource cards the post links to
///   • `feed_score` — the precomputed popularity score
///
/// When `boost` is true and docs carry a ColBERT `score`, scores are
/// min-max normalized per result set (raw MaxSim magnitudes aren't
/// comparable across queries) and blended with popularity:
/// `score = norm + W × ln(1 + feed_score)`, then re-sorted. The raw
/// relevance score is preserved as `colbert_score`. With `boost` false
/// (keyword fallback, date-sorted paths) the input order is kept.
async fn aggregate_docs_with_feed(
    pool: &PgPool,
    docs: Vec<Value>,
    boost: bool,
) -> Result<Vec<Value>, String> {
    use crate::handlers::search::{fetch_feed_info, FEED_SCORE_WEIGHT_SEARCH};

    if docs.is_empty() {
        return Ok(docs);
    }
    let urls: Vec<String> = docs
        .iter()
        .filter_map(|d| d.get("url").and_then(|u| u.as_str()).map(String::from))
        .collect();
    let info = fetch_feed_info(pool, &urls)
        .await
        .map_err(|e| e.to_string())?;

    // Min-max bounds for relevance normalization (boost path only).
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for d in &docs {
        if let Some(s) = d.get("score").and_then(|v| v.as_f64()) {
            lo = lo.min(s);
            hi = hi.max(s);
        }
    }
    let span = (hi - lo).max(1e-9);

    struct Agg {
        doc: Value,
        best_blend: f64,
        aggregated_urls: Vec<String>,
        // Union of every candidate's `linked_urls` at this anchor,
        // deduped by each linked-URL object's `url` field — the same
        // merge the web path performs, so the surviving row carries
        // every distinct resource any collapsed duplicate linked.
        merged_linked: Vec<Value>,
        seen_linked: HashSet<String>,
    }
    let mut order: Vec<String> = Vec::new();
    let mut by_anchor: HashMap<String, Agg> = HashMap::new();

    for doc in docs {
        let Some(url) = doc
            .get("url")
            .and_then(|u| u.as_str())
            .map(|s| s.to_string())
        else {
            continue;
        };
        let fi = info.get(&url);
        let anchor = fi
            .map(|f| f.anchor_url.clone())
            .unwrap_or_else(|| url.clone());
        let fs = fi.and_then(|f| f.feed_score).unwrap_or(0.0).max(0.0);
        let raw_score = doc.get("score").and_then(|v| v.as_f64());
        let blend = match raw_score {
            Some(s) if boost => (s - lo) / span + FEED_SCORE_WEIGHT_SEARCH * (1.0 + fs).ln(),
            Some(s) => s,
            None => 0.0,
        };

        let entry = by_anchor.entry(anchor.clone()).or_insert_with(|| {
            order.push(anchor.clone());
            Agg {
                doc: Value::Null,
                best_blend: f64::NEG_INFINITY,
                aggregated_urls: Vec::new(),
                merged_linked: Vec::new(),
                seen_linked: HashSet::new(),
            }
        });
        if !entry.aggregated_urls.contains(&url) {
            entry.aggregated_urls.push(url.clone());
        }
        // Union this candidate's linked resources into the anchor's
        // bundle regardless of whether it wins the title slot.
        if let Some(linked) = fi
            .and_then(|f| f.linked_urls.as_ref())
            .and_then(|v| v.as_array())
        {
            for lu in linked {
                let key = lu
                    .get("url")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| lu.to_string());
                if entry.seen_linked.insert(key) {
                    entry.merged_linked.push(lu.clone());
                }
            }
        }
        if blend > entry.best_blend {
            entry.best_blend = blend;
            let mut doc = doc;
            if let Some(obj) = doc.as_object_mut() {
                obj.insert("anchor_url".to_string(), json!(anchor));
                obj.insert(
                    "sharers".to_string(),
                    fi.and_then(|f| f.sharers.clone()).unwrap_or(Value::Null),
                );
                obj.insert(
                    "sharer_count".to_string(),
                    json!(fi.map(|f| f.sharer_count).unwrap_or(0)),
                );
                obj.insert(
                    "feed_score".to_string(),
                    fi.and_then(|f| f.feed_score)
                        .map(|v| json!(v))
                        .unwrap_or(Value::Null),
                );
                if boost && raw_score.is_some() {
                    obj.insert("score".to_string(), json!(blend));
                    obj.insert("colbert_score".to_string(), json!(raw_score));
                }
            }
            entry.doc = doc;
        }
    }

    let mut out: Vec<(f64, Value)> = order
        .into_iter()
        .filter_map(|a| by_anchor.remove(&a))
        .map(|mut agg| {
            if let Some(obj) = agg.doc.as_object_mut() {
                obj.insert("aggregated_urls".to_string(), json!(agg.aggregated_urls));
                // The merged bundle replaces the winner's own list —
                // it's a superset by construction.
                obj.insert("linked_urls".to_string(), json!(agg.merged_linked));
            }
            (agg.best_blend, agg.doc)
        })
        .collect();
    if boost {
        out.sort_by(|a, b| b.0.total_cmp(&a.0));
    }
    Ok(out.into_iter().map(|(_, d)| d).collect())
}

// ---------------------------------------------------------------------------
// Tool: list_personalities
// ---------------------------------------------------------------------------

async fn tool_list_personalities(pool: &PgPool, args: Value) -> Result<Value, String> {
    let pg = parse_pagination(&args, 50, 200);

    #[allow(clippy::type_complexity)]
    let rows: Vec<(
        String,
        String,
        String,
        Vec<String>,
        Option<String>,
        i64,
        Option<i32>,
        Option<i32>,
        Option<i32>,
    )> = sqlx::query_as(
        "SELECT u.username, u.name, u.description,
                COALESCE(
                    (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
                       FROM user_categories uc
                       JOIN categories      cat ON cat.id = uc.category_id
                      WHERE uc.user_id = u.id),
                    '{}'::text[]
                ) AS categories,
                u.avatar,
                COALESCE(c.cnt, 0)::bigint AS document_count,
                u.twitter_followers, u.github_followers, u.citations
           FROM users u
           LEFT JOIN LATERAL (
               SELECT count(*) AS cnt FROM documents d WHERE d.user_id = u.id
           ) c ON true
          WHERE u.vip = TRUE AND COALESCE(c.cnt, 0) > 0
          ORDER BY c.cnt DESC, u.name",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    let items: Vec<Value> = rows
        .into_iter()
        .map(|(slug, name, desc, cats, avatar, docs, tw, gh, cit)| {
            json!({
                "slug": slug,
                "name": name,
                "description": desc,
                "categories": cats,
                "avatar": avatar,
                "document_count": docs,
                "twitter_followers": tw,
                "github_followers": gh,
                "citations": cit,
            })
        })
        .collect();

    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "count": page_items.len(),
            "personalities": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: search_personalities
// ---------------------------------------------------------------------------

/// Substring search across user metadata. `vip_filter`:
///   - `Some(true)`  → VIPs only
///   - `Some(false)` → non-VIPs only
///   - `None`        → both tiers
///
/// Returns rows shaped like the other personality-list outputs but with
/// each row tagged `tier = "vip" | "name-match"` so the caller can keep
/// the two flavors visually separate. Ordered by document_count desc
/// then name (cheap heuristic for "more-engaged users surface first").
async fn personality_substring_rows(
    pool: &PgPool,
    query: &str,
    vip_filter: Option<bool>,
) -> Result<Vec<Value>, String> {
    // Escape user-supplied % and _ so the input matches literally
    // rather than acting as a wildcard.
    let pat = format!(
        "%{}%",
        crate::handlers::sql_like::escape_like_pattern(query)
    );
    let mut sql = String::from(
        "SELECT u.username, u.name, u.description,
                COALESCE(
                    (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
                       FROM user_categories uc
                       JOIN categories      cat ON cat.id = uc.category_id
                      WHERE uc.user_id = u.id),
                    '{}'::text[]
                ) AS categories,
                u.avatar,
                u.vip,
                COALESCE(c.cnt, 0)::bigint AS document_count
           FROM users u
           LEFT JOIN LATERAL (
               SELECT count(*) AS cnt FROM documents d WHERE d.user_id = u.id
           ) c ON true
          WHERE (u.name ILIKE $1
                 OR u.description ILIKE $1
                 OR u.username ILIKE $1
                 OR EXISTS (
                     SELECT 1 FROM user_categories uc2
                       JOIN categories cat2 ON cat2.id = uc2.category_id
                      WHERE uc2.user_id = u.id
                        AND (cat2.name ILIKE $1 OR cat2.slug ILIKE $1)
                 ))",
    );
    match vip_filter {
        Some(true) => sql.push_str(" AND u.vip = TRUE AND COALESCE(c.cnt, 0) > 0"),
        Some(false) => sql.push_str(" AND u.vip = FALSE"),
        None => sql.push_str(" AND COALESCE(c.cnt, 0) >= 0"),
    }
    sql.push_str(" ORDER BY c.cnt DESC, u.name");

    #[allow(clippy::type_complexity)]
    let rows: Vec<(
        String,
        String,
        String,
        Vec<String>,
        Option<String>,
        bool,
        i64,
    )> = sqlx::query_as(&sql)
        .bind(&pat)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?;

    Ok(rows
        .into_iter()
        .map(|(slug, name, desc, cats, avatar, vip, docs)| {
            json!({
                "slug": slug,
                "name": name,
                "description": desc,
                "categories": cats,
                "avatar": avatar,
                "document_count": docs,
                "tier": if vip { "vip" } else { "name-match" },
            })
        })
        .collect())
}

async fn tool_search_personalities(
    state: Arc<AppState>,
    pool: &PgPool,
    args: Value,
) -> Result<Value, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("Missing query")?
        .to_string();
    let pg = parse_pagination(&args, 50, 200);

    // ColBERT path for VIPs — mirrors the welcome page's people-search:
    //   1. Hit the cross-library `__all__` index, top 200 passages.
    //   2. Group by `owner`, count hits.
    //   3. Size-normalize: score = count / sqrt(total_docs). Without
    //      this a 5000-doc lib dominates simply by having more chances
    //      to match. Classic IR length normalization — a focused
    //      library beats a noisy giant when its hits are denser, while
    //      scale still gets some credit.
    //   4. Hydrate top owners' personality rows from PG.
    //
    // Non-VIPs aren't in the `__all__` index (it only ingests VIP
    // libraries), so for them we fall back to lexical name/slug
    // substring matching and append those rows after the ColBERT-ranked
    // VIPs. Each row carries a `tier` field (`"vip"` vs `"name-match"`)
    // so the caller can tell the two flavors apart at a glance.
    //
    // Short queries (<3 chars) skip ColBERT (relevance is noisy on
    // 1–2 characters) and fall through to a pure substring path that
    // searches BOTH VIPs and non-VIPs in one shot.
    let _ = &state;
    #[cfg(feature = "model")]
    if state.has_model() && query.trim().chars().count() >= 3 {
        if let Ok(hits) = colbert_search_owners(&state, &query, 200).await {
            // Per-owner hit count.
            let mut counts: HashMap<String, u32> = HashMap::new();
            for (owner, _score) in &hits {
                *counts.entry(owner.clone()).or_insert(0) += 1;
            }

            let mut ranked: Vec<Value> = if counts.is_empty() {
                Vec::new()
            } else {
                let owner_slugs: Vec<String> = counts.keys().cloned().collect();

                // One round-trip: profile metadata + total document
                // count for every owner that scored a hit. We use the
                // PG count as the size-normalization denominator
                // (rather than the index size) because PG is the
                // canonical source of truth for library size.
                #[allow(clippy::type_complexity)]
                let rows: Vec<(
                    String,
                    String,
                    String,
                    Vec<String>,
                    Option<String>,
                    i64,
                )> = sqlx::query_as(
                    "SELECT u.username, u.name, u.description,
                                COALESCE(
                                    (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
                                       FROM user_categories uc
                                       JOIN categories      cat ON cat.id = uc.category_id
                                      WHERE uc.user_id = u.id),
                                    '{}'::text[]
                                ) AS categories,
                                u.avatar,
                                COALESCE(c.cnt, 0)::bigint AS document_count
                           FROM users u
                           LEFT JOIN LATERAL (
                               SELECT count(*) AS cnt FROM documents d
                                WHERE d.user_id = u.id
                           ) c ON true
                          WHERE u.username = ANY($1)",
                )
                .bind(&owner_slugs)
                .fetch_all(pool)
                .await
                .map_err(|e| format!("DB error: {e}"))?;

                let mut v: Vec<Value> = rows
                    .into_iter()
                    .filter_map(|(slug, name, desc, cats, avatar, docs)| {
                        let count = *counts.get(&slug)? as f64;
                        let total = docs.max(1) as f64;
                        let score = count / total.sqrt();
                        Some(json!({
                            "slug": slug,
                            "name": name,
                            "description": desc,
                            "categories": cats,
                            "avatar": avatar,
                            "document_count": docs,
                            "hit_count": count as u32,
                            "score": score,
                            "tier": "vip",
                        }))
                    })
                    .collect();

                v.sort_by(|a, b| {
                    let sa = a["score"].as_f64().unwrap_or(0.0);
                    let sb = b["score"].as_f64().unwrap_or(0.0);
                    sb.partial_cmp(&sa)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            let ca = a["hit_count"].as_u64().unwrap_or(0);
                            let cb = b["hit_count"].as_u64().unwrap_or(0);
                            cb.cmp(&ca)
                        })
                });
                v
            };

            // Append non-VIP name-match rows after the ColBERT-ranked
            // VIPs. Non-VIPs don't carry a relevance score, so we put
            // them strictly after — the LLM can still surface them via
            // pagination when the topical VIPs run out.
            let nonvip = personality_substring_rows(pool, &query, Some(false)).await?;
            ranked.extend(nonvip);

            if !ranked.is_empty() {
                let (page_items, meta) = paginate(&ranked, pg);
                return Ok(tool_result(
                    serde_json::to_string(&json!({
                        "count": page_items.len(),
                        "query": query,
                        "search_type": "colbert+name-match",
                        "personalities": page_items,
                        "pagination": meta,
                    }))
                    .unwrap(),
                ));
            }
        }
    }

    // Pure substring fallback — used when ColBERT is unavailable, the
    // index is missing, or the query is too short. Searches BOTH tiers
    // in one query so non-VIPs can be reached even without the model.
    let items = personality_substring_rows(pool, &query, None).await?;
    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "count": page_items.len(),
            "query": query,
            "search_type": "substring",
            "personalities": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: get_personality
// ---------------------------------------------------------------------------

async fn tool_get_personality(pool: &PgPool, args: Value) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?;

    #[allow(clippy::type_complexity)]
    let row: Option<(
        String,
        String,
        String,
        Vec<String>,
        Option<String>,
        String,
        Value,
        Value,
        i64,
        Option<i32>,
        Option<i32>,
        Option<i32>,
        bool,
    )> = sqlx::query_as(
        "SELECT u.username, u.name, u.description,
                COALESCE(
                    (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
                       FROM user_categories uc
                       JOIN categories      cat ON cat.id = uc.category_id
                      WHERE uc.user_id = u.id),
                    '{}'::text[]
                ) AS categories,
                u.avatar,
                u.index_name, u.links, u.sources,
                COALESCE(c.cnt, 0)::bigint AS document_count,
                u.twitter_followers, u.github_followers, u.citations, u.vip
           FROM users u
           LEFT JOIN LATERAL (
               SELECT count(*) AS cnt FROM documents d WHERE d.user_id = u.id
           ) c ON true
          WHERE u.username = $1",
    )
    .bind(slug)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    let (
        slug,
        name,
        description,
        categories,
        avatar,
        index_name,
        links,
        sources,
        document_count,
        twitter_followers,
        github_followers,
        citations,
        vip,
    ) = row.ok_or_else(|| format!("Personality not found: {slug}"))?;

    Ok(tool_result(
        serde_json::to_string(&json!({
            "slug": slug,
            "name": name,
            "description": description,
            "categories": categories,
            "avatar": avatar,
            "indexName": index_name,
            "links": links,
            "sources": sources,
            "document_count": document_count,
            "twitter_followers": twitter_followers,
            "github_followers": github_followers,
            "citations": citations,
            "vip": vip,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: search
// ---------------------------------------------------------------------------

async fn tool_search(state: Arc<AppState>, pool: &PgPool, args: Value) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?
        .to_string();
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("Missing query")?
        .to_string();
    let pg = parse_pagination(&args, 20, 100);
    let sort_by_date = args
        .get("sort_by_date")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let sources = parse_string_set(args.get("sources"));
    let tags = parse_string_set(args.get("tags"));

    let (user_id, _) = resolve_personality(pool, &slug).await?;

    // ColBERT path — fetch top-K from the index, hydrate metadata from PG,
    // then post-filter by source/tag.
    #[cfg(feature = "model")]
    if state.has_model() {
        let base_k = pg.end().max(100);
        // Over-fetch when filters are active so the post-filter still has
        // enough rows to fill the requested page.
        let fetch_k = if sources.is_some() || tags.is_some() {
            base_k * 3 + 20
        } else {
            base_k
        };

        if let Ok(scored_urls) = colbert_search_urls(&state, &slug, &query, fetch_k as i64).await {
            let urls: Vec<String> = scored_urls.iter().map(|(u, _)| u.clone()).collect();
            let meta = fetch_docs_by_urls(pool, user_id, &urls).await?;

            let mut docs: Vec<Value> = scored_urls
                .into_iter()
                .filter_map(|(url, score)| {
                    meta.get(&url).map(|d| {
                        let mut v = d.to_json();
                        if let Some(obj) = v.as_object_mut() {
                            obj.insert("score".to_string(), json!(score));
                        }
                        v
                    })
                })
                .collect();

            docs = filter_doc_values(docs, &sources, &tags);
            // Collapse near-duplicates by anchor, attach sharers /
            // linked resources, and blend popularity into the ranking —
            // the same aggregation the web feed search renders.
            docs = aggregate_docs_with_feed(pool, docs, !sort_by_date).await?;
            if sort_by_date {
                docs.sort_by(|a, b| {
                    let da = a["date"].as_str().unwrap_or("");
                    let db = b["date"].as_str().unwrap_or("");
                    db.cmp(da)
                });
            }

            let (page_docs, meta_pg) = paginate(&docs, pg);
            return Ok(tool_result(
                serde_json::to_string(&json!({
                    "personality": slug,
                    "query": query,
                    "search_type": "colbert",
                    "sort": if sort_by_date { "date" } else { "relevance" },
                    "count": page_docs.len(),
                    "docs": page_docs,
                    "pagination": meta_pg,
                }))
                .unwrap(),
            ));
        }
    }

    // Fallback: SQL keyword search over title/summary/tags. Returns
    // date-sorted results regardless of `sort_by_date` since there's no
    // relevance score to fall back on.
    #[cfg(not(feature = "model"))]
    let _ = (&state, sort_by_date);

    let pat = format!(
        "%{}%",
        crate::handlers::sql_like::escape_like_pattern(&query)
    );
    let mut sql = String::from(
        // Match against the raw title/summary (richer keyword
        // surface) but return the clean variants when available.
        "SELECT d.url,
                COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                COALESCE(to_char(d.date, 'YYYY-MM-DD'), '') AS date,
                d.tags, d.extra_tags, d.source, d.source_url, d.indexed
           FROM documents d
          WHERE d.user_id = $1
            AND (d.title ILIKE $2
                 OR d.summary ILIKE $2
                 OR EXISTS (
                     SELECT 1 FROM unnest(d.tags || d.extra_tags) t
                      WHERE t ILIKE $2
                 ))",
    );
    let mut idx = 3;
    if sources.is_some() {
        sql.push_str(&format!(" AND d.source = ANY(${idx})"));
        idx += 1;
    }
    if tags.is_some() {
        sql.push_str(&format!(" AND (d.tags || d.extra_tags) @> ${idx}"));
    }
    sql.push_str(" ORDER BY d.date DESC NULLS LAST");

    let mut q = sqlx::query_as::<
        _,
        (
            String,
            String,
            String,
            String,
            Vec<String>,
            Vec<String>,
            String,
            Option<String>,
            bool,
        ),
    >(&sql)
    .bind(user_id)
    .bind(&pat);
    if let Some(ref s) = sources {
        q = q.bind(s);
    }
    if let Some(ref t) = tags {
        let lower: Vec<String> = t.iter().map(|x| x.to_lowercase()).collect();
        q = q.bind(lower);
    }

    let rows = q
        .fetch_all(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?;

    let docs: Vec<Value> = rows.into_iter().map(|r| row_to_doc(r).to_json()).collect();
    // Attach sharers / resources (no re-rank: results stay date-sorted).
    let docs = aggregate_docs_with_feed(pool, docs, false).await?;
    let (page_docs, meta_pg) = paginate(&docs, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personality": slug,
            "query": query,
            "search_type": "keyword",
            "sort": "date",
            "count": page_docs.len(),
            "docs": page_docs,
            "pagination": meta_pg,
        }))
        .unwrap(),
    ))
}

#[cfg(feature = "model")]
async fn colbert_search_urls(
    state: &Arc<AppState>,
    owner: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<(String, f32)>, String> {
    use next_plaid::{filtering, SearchParameters};

    use crate::handlers::encode::encode_texts_internal;
    use crate::handlers::search::fetch_metadata_for_docs;
    use crate::models::InputType;

    // Per-user indices are gone — every library lives in the single
    // `__all__` index, scoped by the `owner` metadata column. This is
    // the same routing the web frontend uses; searching the (now
    // nonexistent) per-user index here used to fail and silently
    // demote every MCP search to the SQL keyword fallback.
    let path_str = state.index_path("__all__").to_string_lossy().to_string();
    let subset = filtering::where_condition(&path_str, "owner = ?", &[serde_json::json!(owner)])
        .map_err(|e| e.to_string())?;
    if subset.is_empty() {
        return Ok(Vec::new());
    }

    let embeddings =
        encode_texts_internal(state.clone(), &[query.to_string()], InputType::Query, None)
            .await
            .map_err(|e| e.to_string())?;

    let idx = state
        .get_index_for_read("__all__")
        .map_err(|e| e.to_string())?;

    let params = SearchParameters {
        top_k: limit as usize,
        n_ivf_probe: 8,
        n_full_scores: 4096,
        batch_size: 2000,
        ..Default::default()
    };

    let result = idx
        .search(&embeddings[0], &params, Some(&subset))
        .map_err(|e| e.to_string())?;

    let metadata =
        fetch_metadata_for_docs(&path_str, &result.passage_ids).map_err(|e| e.to_string())?;

    Ok(metadata
        .into_iter()
        .zip(result.scores.iter())
        .filter_map(|(m, s)| {
            m.and_then(|v| {
                v.get("url")
                    .and_then(|u| u.as_str())
                    .map(|u| (u.to_string(), *s))
            })
        })
        .collect())
}

/// ColBERT search variant that exposes the `owner` slug from each hit's
/// metadata. Used by the cross-library `__all__` index path so callers
/// can rank personalities by hit density. Identical to
/// `colbert_search_urls` otherwise — same encoder, same params, same
/// score semantics. Returns `Vec<(owner, score)>` with hit-level
/// multiplicity preserved (one entry per matching passage), so the
/// caller can either count hits or sum scores.
#[cfg(feature = "model")]
async fn colbert_search_owners(
    state: &Arc<AppState>,
    query: &str,
    limit: i64,
) -> Result<Vec<(String, f32)>, String> {
    use next_plaid::SearchParameters;

    use crate::handlers::encode::encode_texts_internal;
    use crate::handlers::search::fetch_metadata_for_docs;
    use crate::models::InputType;

    let embeddings =
        encode_texts_internal(state.clone(), &[query.to_string()], InputType::Query, None)
            .await
            .map_err(|e| e.to_string())?;

    let idx = state
        .get_index_for_read("__all__")
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

    let path_str = state.index_path("__all__").to_string_lossy().to_string();
    let metadata =
        fetch_metadata_for_docs(&path_str, &result.passage_ids).map_err(|e| e.to_string())?;

    Ok(metadata
        .into_iter()
        .zip(result.scores.iter())
        .filter_map(|(m, s)| {
            m.and_then(|v| {
                v.get("owner")
                    .and_then(|o| o.as_str())
                    .map(|o| (o.to_string(), *s))
            })
        })
        .collect())
}

/// Apply optional source / tag filters on already-hydrated doc JSON.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
fn filter_doc_values(
    docs: Vec<Value>,
    sources: &Option<Vec<String>>,
    tags: &Option<Vec<String>>,
) -> Vec<Value> {
    if sources.is_none() && tags.is_none() {
        return docs;
    }
    docs.into_iter()
        .filter(|d| doc_matches_filters(d, sources, tags))
        .collect()
}

#[cfg_attr(not(feature = "model"), allow(dead_code))]
fn doc_matches_filters(
    doc: &Value,
    sources: &Option<Vec<String>>,
    tags: &Option<Vec<String>>,
) -> bool {
    if let Some(srcs) = sources {
        let s = doc.get("source").and_then(|v| v.as_str()).unwrap_or("");
        if !srcs.iter().any(|x| x == s) {
            return false;
        }
    }
    if let Some(want) = tags {
        let want_lower: Vec<String> = want.iter().map(|x| x.to_lowercase()).collect();
        let mut have: Vec<String> = Vec::new();
        for key in ["tags", "extra-tags"] {
            if let Some(arr) = doc.get(key).and_then(|v| v.as_array()) {
                for t in arr {
                    if let Some(s) = t.as_str() {
                        have.push(s.to_lowercase());
                    }
                }
            }
        }
        // AND semantics — every requested tag must be present.
        for w in &want_lower {
            if !have.iter().any(|h| h == w) {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tool: search_across (multi-personality, RRF fusion)
// ---------------------------------------------------------------------------

async fn tool_search_across(
    state: Arc<AppState>,
    pool: &PgPool,
    args: Value,
) -> Result<Value, String> {
    let slugs: Vec<String> = args
        .get("personalities")
        .and_then(|v| v.as_array())
        .ok_or("Missing personalities")?
        .iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
    if slugs.is_empty() {
        return Err("personalities must be a non-empty array".into());
    }
    if slugs.len() > 10 {
        return Err(format!("too many libraries: {} > 10", slugs.len()));
    }
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("Missing query")?
        .to_string();
    let pg = parse_pagination(&args, 20, 100);
    let sources = args.get("sources").cloned();
    let tags = args.get("tags").cloned();

    const RRF_K: f64 = 60.0;
    let per_slug_fetch = (pg.end().max(50) * 3).clamp(20, 100);
    let mut fused: HashMap<String, (Value, f64, Vec<String>)> = HashMap::new();

    for slug in &slugs {
        let per_slug_args = json!({
            "personality": slug,
            "query": query,
            "per_page": per_slug_fetch,
            "sources": sources,
            "tags": tags,
        });
        let v = tool_search(state.clone(), pool, per_slug_args).await?;
        let text = v["content"][0]["text"].as_str().unwrap_or("{}");
        let parsed: Value = serde_json::from_str(text).unwrap_or(json!({}));
        let empty: Vec<Value> = Vec::new();
        let docs = parsed["docs"].as_array().unwrap_or(&empty);
        for (rank, doc) in docs.iter().enumerate() {
            // Fuse by anchor (canonical resource) when available so two
            // libraries surfacing different tweets of the same paper
            // merge into one row; fall back to the raw URL.
            let url = doc
                .get("anchor_url")
                .or_else(|| doc.get("url"))
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string();
            if url.is_empty() {
                continue;
            }
            let contribution = 1.0 / (RRF_K + rank as f64);
            fused
                .entry(url)
                .and_modify(|(_, score, libs)| {
                    *score += contribution;
                    if !libs.contains(slug) {
                        libs.push(slug.clone());
                    }
                })
                .or_insert_with(|| (doc.clone(), contribution, vec![slug.clone()]));
        }
    }

    let mut merged: Vec<(Value, f64, Vec<String>)> = fused.into_values().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let out: Vec<Value> = merged
        .into_iter()
        .map(|(mut doc, rrf, libraries)| {
            if let Some(obj) = doc.as_object_mut() {
                obj.insert("rrf_score".to_string(), json!(rrf));
                obj.insert("libraries".to_string(), json!(libraries));
            }
            doc
        })
        .collect();

    let (page_docs, meta) = paginate(&out, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "query": query,
            "personalities": slugs,
            "count": page_docs.len(),
            "docs": page_docs,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: latest
// ---------------------------------------------------------------------------

async fn tool_latest(pool: &PgPool, args: Value) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?
        .to_string();
    let pg = parse_pagination(&args, 30, 200);
    let sources = parse_string_set(args.get("sources"));
    let tags = parse_string_set(args.get("tags"));

    let (user_id, _) = resolve_personality(pool, &slug).await?;

    let mut sql = String::from(
        "SELECT d.url,
                COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                COALESCE(to_char(d.date, 'YYYY-MM-DD'), '') AS date,
                d.tags, d.extra_tags, d.source, d.source_url, d.indexed
           FROM documents d
          WHERE d.user_id = $1",
    );
    let mut idx = 2;
    if sources.is_some() {
        sql.push_str(&format!(" AND d.source = ANY(${idx})"));
        idx += 1;
    }
    if tags.is_some() {
        sql.push_str(&format!(" AND (d.tags || d.extra_tags) @> ${idx}"));
    }
    sql.push_str(" ORDER BY d.date DESC NULLS LAST");

    let mut q = sqlx::query_as::<
        _,
        (
            String,
            String,
            String,
            String,
            Vec<String>,
            Vec<String>,
            String,
            Option<String>,
            bool,
        ),
    >(&sql)
    .bind(user_id);
    if let Some(ref s) = sources {
        q = q.bind(s);
    }
    if let Some(ref t) = tags {
        let lower: Vec<String> = t.iter().map(|x| x.to_lowercase()).collect();
        q = q.bind(lower);
    }
    let rows = q
        .fetch_all(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?;

    let docs: Vec<Value> = rows.into_iter().map(|r| row_to_doc(r).to_json()).collect();
    let (page_docs, meta) = paginate(&docs, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personality": slug,
            "count": page_docs.len(),
            "docs": page_docs,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: find_similar
// ---------------------------------------------------------------------------

async fn tool_find_similar(
    state: Arc<AppState>,
    pool: &PgPool,
    args: Value,
) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?
        .to_string();
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or("Missing url")?
        .to_string();
    let pg = parse_pagination(&args, 10, 50);

    let (user_id, _) = resolve_personality(pool, &slug).await?;
    let urls = vec![url.clone()];
    let mut docs_meta = fetch_docs_by_urls(pool, user_id, &urls).await?;
    let entry = docs_meta
        .remove(&url)
        .ok_or_else(|| format!("Document not found: {url}"))?;

    // Same query construction as the UI's `fetchSimilar`.
    let mut parts: Vec<String> = vec![entry.title.clone()];
    if !entry.tags.is_empty() {
        parts.push(entry.tags.join(" "));
    }
    if !entry.summary.is_empty() {
        let head: String = entry
            .summary
            .split_whitespace()
            .take(20)
            .collect::<Vec<_>>()
            .join(" ");
        parts.push(head);
    }
    let query = parts.join(" ");

    #[cfg(feature = "model")]
    if state.has_model() {
        let fetch_k = (pg.end() as i64).max(50) + 5;
        let scored = colbert_search_urls(&state, &slug, &query, fetch_k)
            .await
            .map_err(|e| format!("colbert_search failed: {e}"))?;

        let urls: Vec<String> = scored
            .iter()
            .map(|(u, _)| u.clone())
            .filter(|u| u != &url)
            .collect();
        let meta = fetch_docs_by_urls(pool, user_id, &urls).await?;

        let docs: Vec<Value> = scored
            .into_iter()
            .filter(|(u, _)| u != &url)
            .filter_map(|(u, s)| {
                meta.get(&u).map(|d| {
                    let mut v = d.to_json();
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("score".to_string(), json!(s));
                    }
                    v
                })
            })
            .collect();

        let (page_docs, meta_pg) = paginate(&docs, pg);
        return Ok(tool_result(
            serde_json::to_string(&json!({
                "personality": slug,
                "source_url": url,
                "search_type": "colbert",
                "count": page_docs.len(),
                "docs": page_docs,
                "pagination": meta_pg,
            }))
            .unwrap(),
        ));
    }

    // Fallback: tag-overlap heuristic, sorted by date desc.
    #[cfg(not(feature = "model"))]
    let _ = (&state, &query);

    let tag_set: Vec<String> = entry.tags.iter().map(|t| t.to_lowercase()).collect();
    if tag_set.is_empty() {
        return Ok(tool_result(
            serde_json::to_string(&json!({
                "personality": slug,
                "source_url": url,
                "search_type": "keyword",
                "count": 0,
                "docs": [],
                "pagination": json!({"page": pg.page, "per_page": pg.per_page, "total": 0, "total_pages": 0, "has_more": false}),
            }))
            .unwrap(),
        ));
    }

    let rows: Vec<(
        String,
        String,
        String,
        String,
        Vec<String>,
        Vec<String>,
        String,
        Option<String>,
        bool,
    )> = sqlx::query_as(
        "SELECT d.url,
                COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                COALESCE(to_char(d.date, 'YYYY-MM-DD'), '') AS date,
                d.tags, d.extra_tags, d.source, d.source_url, d.indexed
           FROM documents d
          WHERE d.user_id = $1
            AND d.url <> $2
            AND (d.tags || d.extra_tags) && $3
          ORDER BY d.date DESC NULLS LAST",
    )
    .bind(user_id)
    .bind(&url)
    .bind(&tag_set)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    let docs: Vec<Value> = rows.into_iter().map(|r| row_to_doc(r).to_json()).collect();
    let (page_docs, meta_pg) = paginate(&docs, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personality": slug,
            "source_url": url,
            "search_type": "keyword",
            "count": page_docs.len(),
            "docs": page_docs,
            "pagination": meta_pg,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: list_sources
// ---------------------------------------------------------------------------

fn source_label(key: &str) -> String {
    match key {
        "github" => "GitHub".into(),
        "twitter" | "x" => "X".into(),
        "youtube" => "YouTube".into(),
        "hackernews" => "HackerNews".into(),
        "huggingface" => "HuggingFace".into(),
        "stackoverflow" => "StackOverflow".into(),
        "wikipedia" => "Wikipedia".into(),
        "reddit" => "Reddit".into(),
        "scholar" => "Scholar".into(),
        "semantic_scholar" => "Semantic Scholar".into(),
        "dblp" => "DBLP".into(),
        "arxiv" => "arXiv".into(),
        "extra" => "Extra".into(),
        "zotero" => "Zotero".into(),
        other if other.contains('.') && !other.contains(' ') => other.to_string(),
        other => {
            let mut chars = other.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        }
    }
}

async fn tool_list_sources(pool: &PgPool, args: Value) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?
        .to_string();
    let pg = parse_pagination(&args, 50, 200);

    let (user_id, _) = resolve_personality(pool, &slug).await?;

    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT v.source, v.count
           FROM user_source_counts v
          WHERE v.user_id = $1 AND v.source <> ''
          ORDER BY v.count DESC",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    let total: i64 = rows.iter().map(|(_, c)| *c).sum();
    let items: Vec<Value> = rows
        .into_iter()
        .map(|(key, count)| json!({ "key": key, "label": source_label(&key), "count": count }))
        .collect();

    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personality": slug,
            "total_docs": total,
            "sources": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: list_tags
// ---------------------------------------------------------------------------

async fn tool_list_tags(pool: &PgPool, args: Value) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?
        .to_string();
    let pg = parse_pagination(&args, 100, 1000);

    let (user_id, _) = resolve_personality(pool, &slug).await?;

    // Combine `tags` and `extra_tags`, count occurrences per tag.
    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT tag, count(*)::bigint AS cnt
           FROM (
                SELECT unnest(d.tags || d.extra_tags) AS tag
                  FROM documents d
                 WHERE d.user_id = $1
           ) t
          WHERE tag <> ''
          GROUP BY tag
          ORDER BY cnt DESC, tag",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    let total_unique = rows.len();
    let items: Vec<Value> = rows
        .into_iter()
        .map(|(tag, count)| json!({ "tag": tag, "count": count }))
        .collect();

    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personality": slug,
            "unique_tags": total_unique,
            "tags": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: get_document
// ---------------------------------------------------------------------------

async fn tool_get_document(pool: &PgPool, args: Value) -> Result<Value, String> {
    let slug = args
        .get("personality")
        .and_then(|v| v.as_str())
        .ok_or("Missing personality")?
        .to_string();
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or("Missing url")?
        .to_string();

    let (user_id, _) = resolve_personality(pool, &slug).await?;
    let urls = vec![url.clone()];
    let mut meta = fetch_docs_by_urls(pool, user_id, &urls).await?;
    let doc = meta
        .remove(&url)
        .ok_or_else(|| format!("Document not found: {url}"))?;

    let mut v = doc.to_json();
    if let Some(obj) = v.as_object_mut() {
        // Override the truncated summary with the full one — get_document
        // is the explicit "give me everything" tool.
        obj.insert("summary".to_string(), json!(doc.summary));
        obj.insert("personality".to_string(), json!(slug));
    }
    Ok(tool_result(serde_json::to_string(&v).unwrap()))
}

// ---------------------------------------------------------------------------
// Tool: feed
// ---------------------------------------------------------------------------

async fn tool_feed(pool: &PgPool, headers: &HeaderMap, args: Value) -> Result<Value, String> {
    // If the caller is bearer-authed, return their personal follow-graph
    // timeline — exactly what the UI feed shows them when libs.size === 0.
    // Anonymous callers fall back to the public cross-library aggregate
    // below so `feed` keeps doing something useful without credentials.
    if crate::handlers::tokens::resolve_bearer(pool, headers)
        .await
        .is_some()
    {
        return tool_my_timeline(pool, headers, args).await;
    }

    let pg = parse_pagination(&args, 50, 500);

    // Reuse the same builder that powers `GET /api/feed` (welcome page
    // RecentFeed). That handler does the heavy lifting we want for
    // "relevant + various":
    //   - Per-URL latest-by-date pick + cross-library sharer JSONB.
    //   - Score = sharer_count + 14-day recency bonus, so multi-shared
    //     URLs win even if not from today, and a fresh single-share
    //     can still surface.
    //   - Source-gap selector: never two consecutive same-source rows
    //     within a 4-slot window, so a same-day bulk import (e.g. a
    //     HuggingFace likes batch) can't dominate the feed.
    //   - 60-second TTL cache, so repeat MCP calls within a minute
    //     skip both the SQL and the selector.
    //
    // We DON'T apply `jitter_feed` here — pagination needs a stable
    // canonical ordering across pages, otherwise page 1 and page 2
    // could overlap or skip rows.
    //
    // Pull a slightly larger payload than the requested end so
    // pagination beyond page 1 still has rows to slice.
    let target = (pg.end() as i64).max(50);
    let payload = crate::handlers::users::build_feed_payload(pool, target).await;

    let arr: Vec<Value> = match payload {
        Value::Array(v) => v,
        _ => Vec::new(),
    };
    // Trim summary for the LLM channel — full text is available via
    // `get_document` if the caller wants it.
    let items: Vec<Value> = arr
        .into_iter()
        .map(|mut v| {
            if let Some(obj) = v.as_object_mut() {
                if let Some(s) = obj.get("summary").and_then(|x| x.as_str()) {
                    let trimmed = truncate_summary(s, 200);
                    obj.insert("summary".to_string(), Value::String(trimmed));
                }
            }
            v
        })
        .collect();

    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "count": page_items.len(),
            "docs": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: intersect_documents
// ---------------------------------------------------------------------------

async fn tool_intersect_documents(pool: &PgPool, args: Value) -> Result<Value, String> {
    let slugs: Vec<String> = args
        .get("personalities")
        .and_then(|v| v.as_array())
        .ok_or("Missing personalities")?
        .iter()
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
    if slugs.len() < 2 {
        return Err("intersect_documents needs at least 2 personality slugs".into());
    }
    if slugs.len() > 10 {
        return Err(format!("too many libraries: {} > 10", slugs.len()));
    }
    let pg = parse_pagination(&args, 50, 500);
    let fetch_limit = (pg.end() as i64).clamp(1, 1000);

    #[allow(clippy::type_complexity)]
    let rows: Vec<(
        String,
        String,
        String,
        String,
        Vec<String>,
        Vec<String>,
        String,
        Option<String>,
        bool,
        Vec<String>,
        i64,
    )> = sqlx::query_as(
        "WITH shared_urls AS (
             SELECT d.url,
                    array_agg(DISTINCT u.username ORDER BY u.username) AS owners,
                    count(DISTINCT u.username) AS owner_count
               FROM documents d
               JOIN users u ON u.id = d.user_id
              WHERE u.username = ANY($1)
              GROUP BY d.url
             HAVING count(DISTINCT u.username) >= 2
         ),
         canonical AS (
             SELECT DISTINCT ON (d.url)
                    d.url,
                    COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                    COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                    d.date,
                    d.tags, d.extra_tags, d.source, d.source_url, d.indexed,
                    s.owners, s.owner_count
               FROM documents d
               JOIN shared_urls s ON s.url = d.url
               JOIN users u ON u.id = d.user_id
              WHERE u.username = ANY($1)
              ORDER BY d.url, d.date DESC NULLS LAST
         )
         SELECT url, title, summary,
                COALESCE(to_char(date, 'YYYY-MM-DD'), '') AS date,
                tags, extra_tags, source, source_url, indexed,
                owners, owner_count
           FROM canonical
          ORDER BY owner_count DESC, date DESC NULLS LAST
          LIMIT $2",
    )
    .bind(&slugs)
    .bind(fetch_limit)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    let items: Vec<Value> = rows
        .into_iter()
        .map(
            |(
                url,
                title,
                summary,
                date,
                tags,
                extra_tags,
                source,
                source_url,
                indexed,
                owners,
                owner_count,
            )| {
                json!({
                    "url": url,
                    "title": title,
                    "summary": truncate_summary(&summary, 200),
                    "date": date,
                    "tags": tags,
                    "extra-tags": extra_tags,
                    "source": source,
                    "source_url": source_url,
                    "indexed": indexed,
                    "owners": owners,
                    "owner_count": owner_count,
                })
            },
        )
        .collect();

    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personalities": slugs,
            "count": page_items.len(),
            "docs": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// save_document — bearer-authed upload
// ---------------------------------------------------------------------------

/// MCP-side wrapper around the same upsert path as `POST /api/me/documents`.
/// Auth comes from the `Authorization: Bearer kn_...` header on the JSON-RPC
/// request — the `mcp-remote` shim that connects Claude Desktop to this
/// endpoint forwards the header verbatim. The owning user is derived from
/// the token; there is no way to redirect the write to another user.
async fn tool_save_document(
    pool: &PgPool,
    headers: &HeaderMap,
    args: Value,
) -> Result<Value, String> {
    let user_id = crate::handlers::tokens::resolve_bearer(pool, headers)
        .await
        .ok_or_else(|| {
            "Missing or invalid bearer token. The MCP HTTP request must \
             carry `Authorization: Bearer kn_...`. Mint a token at \
             /profile → API tokens."
                .to_string()
        })?;

    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "url is required".to_string())?;

    let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("");
    let summary = args.get("summary").and_then(|v| v.as_str()).unwrap_or("");
    let date = args.get("date").and_then(|v| v.as_str()).unwrap_or("");
    let tags = parse_string_set(args.get("tags")).unwrap_or_default();
    let extra_tags = parse_string_set(args.get("extra_tags")).unwrap_or_default();
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    let source_url = args.get("source_url").and_then(|v| v.as_str());

    sqlx::query(
        "INSERT INTO documents (
             user_id, url, title, summary, date, tags, extra_tags,
             source, source_url
         )
         VALUES ($1, $2, $3, $4, NULLIF($5, '')::date, $6, $7, $8, $9)
         ON CONFLICT (user_id, url) DO UPDATE SET
             title       = EXCLUDED.title,
             summary     = EXCLUDED.summary,
             date        = EXCLUDED.date,
             tags        = EXCLUDED.tags,
             extra_tags  = EXCLUDED.extra_tags,
             source      = EXCLUDED.source,
             source_url  = EXCLUDED.source_url,
             updated_at  = now()",
    )
    .bind(user_id)
    .bind(url)
    .bind(title)
    .bind(summary)
    .bind(date)
    .bind(&tags)
    .bind(&extra_tags)
    .bind(source)
    .bind(source_url)
    .execute(pool)
    .await
    .map_err(|e| format!("DB error: {e}"))?;

    Ok(tool_result(
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "url": url,
            "user_id": user_id,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Bearer-token → self resolution
// ---------------------------------------------------------------------------

/// Common error string for the my_* tools when the bearer is missing or
/// invalid. Kept identical to the `save_document` message so an LLM that
/// has already learned the remediation hint for one tool applies it to
/// the others.
const BEARER_REQUIRED_HINT: &str = "Missing or invalid bearer token. The MCP HTTP request must \
     carry `Authorization: Bearer kn_...`. Mint a token at \
     /profile → API tokens.";

/// Resolve `Authorization: Bearer kn_...` → `(user_id, username, index_name)`.
/// Returns `Err` with a user-facing hint when the header is missing,
/// malformed, revoked, or the user row has been deleted in the meantime.
async fn resolve_self_full(
    pool: &PgPool,
    headers: &HeaderMap,
) -> Result<(i64, String, String), String> {
    let user_id = crate::handlers::tokens::resolve_bearer(pool, headers)
        .await
        .ok_or_else(|| BEARER_REQUIRED_HINT.to_string())?;

    sqlx::query_as::<_, (String, String)>("SELECT username, index_name FROM users WHERE id = $1")
        .bind(user_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?
        .map(|(username, index_name)| (user_id, username, index_name))
        .ok_or_else(|| format!("Token resolved to user_id={user_id} but the user row is gone"))
}

// ---------------------------------------------------------------------------
// Tool: my_library — search / list the bearer holder's own library
// ---------------------------------------------------------------------------

async fn tool_my_library(
    state: Arc<AppState>,
    pool: &PgPool,
    headers: &HeaderMap,
    args: Value,
) -> Result<Value, String> {
    let (user_id, username, _index_name) = resolve_self_full(pool, headers).await?;

    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let sources = parse_string_set(args.get("sources"));
    let tags = parse_string_set(args.get("tags"));
    let sort_by_date = args
        .get("sort_by_date")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // No query → "latest" semantics. Different default page size than the
    // search path, same as the standalone `latest` tool.
    let Some(query) = query else {
        let pg = parse_pagination(&args, 30, 200);
        let mut sql = String::from(
            "SELECT d.url,
                    COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                    COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                    COALESCE(to_char(d.date, 'YYYY-MM-DD'), '') AS date,
                    d.tags, d.extra_tags, d.source, d.source_url, d.indexed
               FROM documents d
              WHERE d.user_id = $1",
        );
        let mut idx = 2;
        if sources.is_some() {
            sql.push_str(&format!(" AND d.source = ANY(${idx})"));
            idx += 1;
        }
        if tags.is_some() {
            sql.push_str(&format!(" AND (d.tags || d.extra_tags) @> ${idx}"));
        }
        sql.push_str(" ORDER BY d.date DESC NULLS LAST");

        let mut q = sqlx::query_as::<
            _,
            (
                String,
                String,
                String,
                String,
                Vec<String>,
                Vec<String>,
                String,
                Option<String>,
                bool,
            ),
        >(&sql)
        .bind(user_id);
        if let Some(ref s) = sources {
            q = q.bind(s);
        }
        if let Some(ref t) = tags {
            let lower: Vec<String> = t.iter().map(|x| x.to_lowercase()).collect();
            q = q.bind(lower);
        }
        let rows = q
            .fetch_all(pool)
            .await
            .map_err(|e| format!("DB error: {e}"))?;
        let docs: Vec<Value> = rows.into_iter().map(|r| row_to_doc(r).to_json()).collect();
        let (page_docs, meta) = paginate(&docs, pg);
        return Ok(tool_result(
            serde_json::to_string(&json!({
                "personality": username,
                "scope": "self",
                "count": page_docs.len(),
                "docs": page_docs,
                "pagination": meta,
            }))
            .unwrap(),
        ));
    };

    let pg = parse_pagination(&args, 20, 100);

    // ColBERT path — mirrors `tool_search`, restricted to the bearer
    // holder's own index. Falls back to a SQL keyword search if the
    // model or index isn't available.
    #[cfg(feature = "model")]
    if state.has_model() {
        let base_k = pg.end().max(100);
        let fetch_k = if sources.is_some() || tags.is_some() {
            base_k * 3 + 20
        } else {
            base_k
        };

        if let Ok(scored_urls) =
            colbert_search_urls(&state, &username, &query, fetch_k as i64).await
        {
            let urls: Vec<String> = scored_urls.iter().map(|(u, _)| u.clone()).collect();
            let meta = fetch_docs_by_urls(pool, user_id, &urls).await?;

            let mut docs: Vec<Value> = scored_urls
                .into_iter()
                .filter_map(|(url, score)| {
                    meta.get(&url).map(|d| {
                        let mut v = d.to_json();
                        if let Some(obj) = v.as_object_mut() {
                            obj.insert("score".to_string(), json!(score));
                        }
                        v
                    })
                })
                .collect();

            docs = filter_doc_values(docs, &sources, &tags);
            if sort_by_date {
                docs.sort_by(|a, b| {
                    let da = a["date"].as_str().unwrap_or("");
                    let db = b["date"].as_str().unwrap_or("");
                    db.cmp(da)
                });
            }

            let (page_docs, meta_pg) = paginate(&docs, pg);
            return Ok(tool_result(
                serde_json::to_string(&json!({
                    "personality": username,
                    "scope": "self",
                    "query": query,
                    "search_type": "colbert",
                    "sort": if sort_by_date { "date" } else { "relevance" },
                    "count": page_docs.len(),
                    "docs": page_docs,
                    "pagination": meta_pg,
                }))
                .unwrap(),
            ));
        }
    }
    #[cfg(not(feature = "model"))]
    let _ = (&state, sort_by_date);

    // SQL keyword fallback — date-sorted regardless of `sort_by_date`.
    let pat = format!(
        "%{}%",
        crate::handlers::sql_like::escape_like_pattern(&query)
    );
    let mut sql = String::from(
        // Match against the raw title/summary (richer keyword
        // surface) but return the clean variants when available.
        "SELECT d.url,
                COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                COALESCE(to_char(d.date, 'YYYY-MM-DD'), '') AS date,
                d.tags, d.extra_tags, d.source, d.source_url, d.indexed
           FROM documents d
          WHERE d.user_id = $1
            AND (d.title ILIKE $2
                 OR d.summary ILIKE $2
                 OR EXISTS (
                     SELECT 1 FROM unnest(d.tags || d.extra_tags) t
                      WHERE t ILIKE $2
                 ))",
    );
    let mut idx = 3;
    if sources.is_some() {
        sql.push_str(&format!(" AND d.source = ANY(${idx})"));
        idx += 1;
    }
    if tags.is_some() {
        sql.push_str(&format!(" AND (d.tags || d.extra_tags) @> ${idx}"));
    }
    sql.push_str(" ORDER BY d.date DESC NULLS LAST");

    let mut q = sqlx::query_as::<
        _,
        (
            String,
            String,
            String,
            String,
            Vec<String>,
            Vec<String>,
            String,
            Option<String>,
            bool,
        ),
    >(&sql)
    .bind(user_id)
    .bind(&pat);
    if let Some(ref s) = sources {
        q = q.bind(s);
    }
    if let Some(ref t) = tags {
        let lower: Vec<String> = t.iter().map(|x| x.to_lowercase()).collect();
        q = q.bind(lower);
    }
    let rows = q
        .fetch_all(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?;
    let docs: Vec<Value> = rows.into_iter().map(|r| row_to_doc(r).to_json()).collect();
    let (page_docs, meta) = paginate(&docs, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "personality": username,
            "scope": "self",
            "query": query,
            "search_type": "keyword",
            "sort": "date",
            "count": page_docs.len(),
            "docs": page_docs,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tool: my_timeline — bearer-authed follow-graph timeline
// ---------------------------------------------------------------------------

/// SQL twin of `handlers::follows::timeline` but parametrized by a
/// bearer-resolved `user_id` rather than the cookie session. The shape
/// of the per-row output matches `/api/feed` exactly so MCP clients can
/// reuse their feed-renderers without conditioning on the source.
#[allow(clippy::type_complexity)]
async fn tool_my_timeline(
    pool: &PgPool,
    headers: &HeaderMap,
    args: Value,
) -> Result<Value, String> {
    let (me_id, _, _) = resolve_self_full(pool, headers).await?;

    let pg = parse_pagination(&args, 50, 200);
    let before = args
        .get("before")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let sources_inc = parse_string_set(args.get("sources")).unwrap_or_default();
    let sources_exc = parse_string_set(args.get("exclude_sources")).unwrap_or_default();
    let tags_inc: Vec<String> = parse_string_set(args.get("tags"))
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.to_lowercase())
        .collect();

    // Over-fetch up to `end()` rows so a page-2 caller still sees the
    // correct slice (the timeline SQL itself caps at `$2 * 20`).
    let fetch_limit = (pg.end() as i64).clamp(50, 500);

    let sql = "
        WITH followed AS (
            SELECT followed_id AS user_id FROM follows WHERE follower_id = $1
            UNION
            SELECT $1::bigint AS user_id
        ),
        candidates AS (
            SELECT d.user_id, d.url,
                   COALESCE(NULLIF(d.clean_title, ''), d.title) AS title,
                   d.date,
                   COALESCE(NULLIF(d.clean_summary, ''), d.summary) AS summary,
                   d.tags,
                   d.extra_tags, d.source, d.source_url, d.created_at
              FROM documents d
              JOIN followed f ON f.user_id = d.user_id
             WHERE d.deleted = FALSE
               AND d.date IS NOT NULL
               AND ($3::timestamptz IS NULL OR d.date < $3::timestamptz)
               AND (cardinality($4::text[]) = 0 OR d.source = ANY($4::text[]))
               AND (cardinality($5::text[]) = 0 OR NOT d.source = ANY($5::text[]))
               AND (cardinality($6::text[]) = 0
                    OR (SELECT bool_and(
                            EXISTS (
                                SELECT 1 FROM unnest(d.tags) t WHERE lower(t) = q
                            ) OR EXISTS (
                                SELECT 1 FROM unnest(d.extra_tags) t WHERE lower(t) = q
                            )
                        )
                        FROM unnest($6::text[]) AS q))
             ORDER BY d.date DESC
             LIMIT LEAST(4000, $2 * 20)
        ),
        ranked AS (
            SELECT c.*,
                   ROW_NUMBER() OVER (PARTITION BY c.url ORDER BY c.date DESC) AS rn
              FROM candidates c
        ),
        dedup AS (
            SELECT url, title, date, summary, tags, source, source_url, created_at
              FROM ranked
             WHERE rn = 1
             ORDER BY date DESC, created_at DESC
             LIMIT $2
        )
        SELECT
            m.url, m.title,
            COALESCE(to_char(m.date, 'YYYY-MM-DD'), '') AS date_str,
            m.summary, m.tags, m.source, m.source_url,
            s.sharers, s.sharer_count
          FROM dedup m
          JOIN LATERAL (
              SELECT jsonb_agg(
                         jsonb_build_object(
                             'slug',             u.username,
                             'name',             u.name,
                             'avatar',           u.avatar,
                             'twitterFollowers', u.twitter_followers
                         )
                     )       AS sharers,
                     count(*) AS sharer_count
                FROM documents d
                JOIN users    u ON u.id = d.user_id
               WHERE d.url = m.url
                 AND d.deleted = FALSE
          ) s ON true
         ORDER BY m.date DESC, m.created_at DESC, s.sharer_count DESC, m.url
    ";

    let rows: Vec<(
        String,
        String,
        String,
        String,
        Vec<String>,
        String,
        Option<String>,
        Value,
        i64,
    )> = sqlx::query_as(sql)
        .bind(me_id)
        .bind(fetch_limit)
        .bind(before)
        .bind(&sources_inc)
        .bind(&sources_exc)
        .bind(&tags_inc)
        .fetch_all(pool)
        .await
        .map_err(|e| format!("DB error: {e}"))?;

    let items: Vec<Value> = rows
        .into_iter()
        .map(
            |(url, title, date, summary, tags, source, source_url, sharers, count)| {
                json!({
                    "url": url,
                    "title": title,
                    "date": date,
                    "summary": truncate_summary(&summary, 200),
                    "tags": tags,
                    "source": source,
                    "source_url": source_url,
                    "sharers": sharers,
                    "sharerCount": count,
                })
            },
        )
        .collect();

    let (page_items, meta) = paginate(&items, pg);
    Ok(tool_result(
        serde_json::to_string(&json!({
            "scope": "self",
            "count": page_items.len(),
            "docs": page_items,
            "pagination": meta,
        }))
        .unwrap(),
    ))
}
