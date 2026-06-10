//! Search handlers.
//!
//! Handles search operations on indices.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Extension, Json,
};
use ndarray::Array2;

use next_plaid::{filtering, text_search, SearchParameters};

use crate::error::{ApiError, ApiResult};
use crate::handlers::encode::encode_texts_internal;
use crate::models::{
    ErrorResponse, FilteredSearchRequest, FilteredSearchWithEncodingRequest, InputType,
    QueryEmbeddings, QueryResultResponse, SearchRequest, SearchResponse, SearchWithEncodingRequest,
};
use crate::state::AppState;
use crate::tracing_middleware::TraceId;
use crate::PrettyJson;

// Fusion algorithms are in next_plaid::text_search::{fuse_rrf, fuse_relative_score}

/// Convert query embeddings from JSON or base64 format to ndarray.
fn to_ndarray(query: &QueryEmbeddings) -> ApiResult<Array2<f32>> {
    // Prefer base64 if provided (more efficient)
    if let (Some(b64), Some(shape)) = (&query.embeddings_b64, &query.shape) {
        let floats =
            crate::models::decode_b64_embeddings(b64, *shape).map_err(ApiError::BadRequest)?;
        return Array2::from_shape_vec((shape[0], shape[1]), floats)
            .map_err(|e| ApiError::BadRequest(format!("Failed to create query array: {}", e)));
    }

    // Fall back to JSON array format
    let embeddings = query.embeddings.as_ref().ok_or_else(|| {
        ApiError::BadRequest(
            "Must provide either 'embeddings' or 'embeddings_b64' + 'shape'".to_string(),
        )
    })?;

    let rows = embeddings.len();
    if rows == 0 {
        return Err(ApiError::BadRequest("Empty query embeddings".to_string()));
    }

    let cols = embeddings[0].len();
    if cols == 0 {
        return Err(ApiError::BadRequest(
            "Zero dimension query embeddings".to_string(),
        ));
    }

    // Verify all rows have the same dimension
    for (i, row) in embeddings.iter().enumerate() {
        if row.len() != cols {
            return Err(ApiError::BadRequest(format!(
                "Inconsistent query embedding dimension at row {}: expected {}, got {}",
                i,
                cols,
                row.len()
            )));
        }
    }

    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| ApiError::BadRequest(format!("Failed to create query array: {}", e)))
}

/// Fetch metadata for a list of document IDs.
/// Returns a Vec of Option<serde_json::Value> in the same order as document_ids.
/// If metadata doesn't exist for an index or a specific document, returns None for that entry.
///
/// # Errors
/// Returns an error if the metadata database exists but fails to query.
/// If no metadata database exists, returns Ok with None for all entries (not an error).
pub(crate) fn fetch_metadata_for_docs(
    path_str: &str,
    document_ids: &[i64],
) -> ApiResult<Vec<Option<serde_json::Value>>> {
    if !filtering::exists(path_str) {
        // No metadata database - return None for all (this is not an error)
        return Ok(vec![None; document_ids.len()]);
    }

    // Fetch metadata for the document IDs
    let metadata_list = filtering::get(path_str, None, &[], Some(document_ids)).map_err(|e| {
        tracing::error!("Failed to fetch metadata from database: {}", e);
        ApiError::Internal(format!("Failed to fetch metadata: {}", e))
    })?;

    // Build a map from _subset_ to metadata for quick lookup
    let meta_map: HashMap<i64, serde_json::Value> = metadata_list
        .into_iter()
        .filter_map(|m| m.get("_subset_").and_then(|v| v.as_i64()).map(|id| (id, m)))
        .collect();

    // Map document_ids to their metadata (or None if not found)
    Ok(document_ids
        .iter()
        .map(|doc_id| meta_map.get(doc_id).cloned())
        .collect())
}

/// Filter + re-rank search results using `feed_snapshot`.
///
/// Two passes:
///   1. Drop any result whose URL is not in `feed_snapshot` — the feed-page
///      search bar would otherwise surface long-tail ColBERT matches that
///      never made it into the curated set.
///   2. Blend the ColBERT score with the feed_snapshot score so VIP-shared
///      tweets that anchor an arxiv / hf / github resource rise above text-
///      matchier but lower-signal candidates. Formula:
///      `final = colbert + weight × ln(1 + feed_score)`
///      then sort desc and trim to `top_k`. The blended score replaces the
///      ColBERT one in the response; the raw `feed_score` is attached to
///      each metadata entry so the client can apply the same blend after
///      its own re-rank pass.
///
/// Historically `FEED_SCORE_WEIGHT` was 0.5 — a relevance *proxy* picked
/// when the only ranking signal was raw ColBERT, whose magnitude (~0.015)
/// couldn't carry semantic intent on its own and let popular-but-unrelated
/// docs dominate. Now that the hybrid pipeline produces a real relevance
/// score (ColBERT + BM25 fused via per-query min-max normalization, range
/// ~[0, 1]), popularity should only break ties between equally-relevant
/// candidates — not override the user's expressed intent.
///
/// `_BROWSE`: feed view, no `text_query`. Some popularity nudge is still
///   useful as a weak relevance proxy for items the semantic side ranked
///   similarly, but kept modest so date/recency stays visible.
/// `_SEARCH`: `text_query` present. The user typed something specific;
///   popularity matters less than relevance, but broadly-shared
///   resources should still rise above equally-matched long-tail docs.
///   At 0.05 the boost tops out around +0.19 (ln(1+40) ≈ 3.7) on a
///   ~[0, 1] fused relevance score — enough to reorder near-ties and
///   lift VIP-consensus docs, not enough to override a clear match.
///   (Was 0.02, which made popularity invisible in practice.)
const FEED_SCORE_WEIGHT_BROWSE: f64 = 0.10;
pub(crate) const FEED_SCORE_WEIGHT_SEARCH: f64 = 0.05;

/// Cross-personality info for one result URL: the anchor it collapses
/// to, the feed_snapshot popularity roll-up, and the doc's linked
/// resources from Postgres. Shared between the web search path
/// (`apply_feed_scope_filter`) and the MCP tools so both render the
/// same "shared by N people" aggregation.
pub(crate) struct FeedInfo {
    pub anchor_url: String,
    /// `None` when the anchor has no feed_snapshot row (no breadth signal).
    pub feed_score: Option<f64>,
    pub sharers: Option<serde_json::Value>,
    pub sharer_count: i32,
    /// `documents.linked_urls` JSONB — inline resource cards (arxiv,
    /// github, hf, …) the post links to. Used by MCP; the web path
    /// reads linked_urls from the index metadata instead.
    // Only read from mcp.rs, which is compiled into the binary target
    // but not the lib — dead_code fires on the lib pass otherwise.
    #[allow(dead_code)]
    pub linked_urls: Option<serde_json::Value>,
}

/// Resolve each URL's anchor (priority-picked canonical referenced URL,
/// falling back to canonical_url) and join the feed_snapshot row for
/// that anchor. URLs absent from `documents` are absent from the map.
#[allow(clippy::type_complexity)]
pub(crate) async fn fetch_feed_info(
    pool: &sqlx::PgPool,
    urls: &[String],
) -> ApiResult<HashMap<String, FeedInfo>> {
    if urls.is_empty() {
        return Ok(HashMap::new());
    }
    let rows: Vec<(
        String,
        Option<String>,
        Option<f64>,
        Option<serde_json::Value>,
        Option<i32>,
        Option<serde_json::Value>,
    )> = sqlx::query_as(
        "WITH input AS (\n            SELECT d.url, d.canonical_url, d.canonical_referenced_urls, d.linked_urls\n              FROM documents d\n             WHERE d.url = ANY($1::text[])\n               -- `AND d.deleted = FALSE` forces the planner to use\n               -- `idx_documents_url_live` (partial, on url WHERE\n               -- deleted=false) instead of `documents_pkey`\n               -- (composite on `user_id, url`). The PK can't seek by\n               -- url alone — it scans the full key range and pays\n               -- ~1.5 M buffer hits per call.\n               AND d.deleted = FALSE\n         ),\n         resolved AS (\n            SELECT i.url, i.linked_urls,\n                   COALESCE(\n                       (SELECT ref FROM unnest(i.canonical_referenced_urls) ref\n                         ORDER BY CASE\n                           WHEN ref LIKE 'https://arxiv.org/abs/%'       THEN 1\n                           WHEN ref LIKE 'https://huggingface.co/%'      THEN 2\n                           WHEN ref LIKE 'https://github.com/%'          THEN 3\n                           WHEN ref LIKE 'https://openreview.net/%'      THEN 4\n                           WHEN ref LIKE 'https://doi.org/%'             THEN 5\n                           WHEN ref LIKE 'https://paperswithcode.com/%'  THEN 6\n                           WHEN ref LIKE 'https://aclanthology.org/%'    THEN 7\n                           WHEN ref LIKE 'https://semanticscholar.org/%' THEN 8\n                           WHEN ref LIKE 'https://distill.pub/%'         THEN 9\n                           WHEN ref LIKE 'https://biorxiv.org/%'         THEN 10\n                           WHEN ref LIKE 'https://medrxiv.org/%'         THEN 11\n                           ELSE 99\n                         END, ref LIMIT 1),\n                       i.canonical_url\n                   ) AS anchor_url\n              FROM input i\n         )\n         SELECT r.url, r.anchor_url, fs.score, fs.sharers, fs.sharer_count, r.linked_urls\n           FROM resolved r\n           LEFT JOIN feed_snapshot fs ON fs.anchor_url = r.anchor_url",
    )
    .bind(urls)
    .fetch_all(pool)
    .await
    .map_err(|e| ApiError::Internal(format!("feed_snapshot anchor lookup failed: {}", e)))?;

    let mut out: HashMap<String, FeedInfo> = HashMap::new();
    for (url, anchor, score, sharers, sharer_count, linked_urls) in rows {
        let anchor = anchor.unwrap_or_else(|| url.clone());
        out.insert(
            url,
            FeedInfo {
                anchor_url: anchor,
                feed_score: score,
                sharers,
                sharer_count: sharer_count.unwrap_or(0),
                linked_urls,
            },
        );
    }
    Ok(out)
}

/// Normalized content fingerprint for content-level dedup. Catches
/// the "same post under two URLs" cases the anchor collapse can't
/// see — an author re-posting the identical tweet (each copy is its
/// own anchor), or several accounts posting the same body verbatim.
///
/// Key choice:
///   * Substantial summary (≥ 80 normalized chars) → summary alone.
///     The body IS the content; including the title (which for
///     tweets is just the author name) would keep cross-author
///     copies of the same text apart.
///   * Shorter text → title + summary, requiring ≥ 40 chars total.
///     Title alone is never enough — every tweet by one author
///     shares its title ("Rohan Paul (@rohanpaul_ai)").
///   * Below that → None (too little text to be an identity).
pub(crate) fn content_signature(meta: Option<&serde_json::Value>) -> Option<String> {
    let m = meta?;
    let norm = |s: &str| {
        s.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let title = norm(m.get("title").and_then(|v| v.as_str()).unwrap_or(""));
    let summary = norm(m.get("summary").and_then(|v| v.as_str()).unwrap_or(""));
    // Cap the summary contribution so truncation differences between
    // copies (e.g. one channel trims at 200 chars) don't defeat the
    // match.
    let head: String = summary.chars().take(180).collect();
    if summary.chars().count() >= 80 {
        return Some(head);
    }
    if title.chars().count() + summary.chars().count() < 40 {
        return None;
    }
    Some(format!("{title}\u{1}{head}"))
}

/// Anchor-dedup + feed-score blend.
///
/// `strict_feed_filter=true`: drop any result whose anchor isn't in
/// feed_snapshot (the original feed-scope behaviour — used by the
/// global feed search bar to scope discovery to the curated set).
///
/// `strict_feed_filter=false`: keep every result (no URL filtering),
/// but still collapse near-duplicates by `anchor_url` and blend the
/// score with the cross-personality `feed_score`. This is what the
/// per-library search uses: a user typing on raphael-sourty's page
/// sees one row per anchor instead of three tweets quoting the same
/// paper, and broadly-shared resources rise above lonely matches.
#[allow(clippy::ptr_arg, clippy::type_complexity)]
async fn apply_feed_scope_filter(
    pool: &sqlx::PgPool,
    results: &mut Vec<QueryResultResponse>,
    top_k: usize,
    strict_feed_filter: bool,
    search_intent: bool,
) -> ApiResult<()> {
    let feed_weight = if search_intent {
        FEED_SCORE_WEIGHT_SEARCH
    } else {
        FEED_SCORE_WEIGHT_BROWSE
    };
    let filter_t0 = std::time::Instant::now();
    let mut url_set: HashSet<String> = HashSet::new();
    for r in results.iter() {
        for m in r.metadata.iter().flatten() {
            if let Some(u) = m.get("url").and_then(|v| v.as_str()) {
                url_set.insert(u.to_string());
            }
        }
    }
    if url_set.is_empty() {
        for r in results.iter_mut() {
            r.document_ids.clear();
            r.scores.clear();
            r.metadata.clear();
        }
        return Ok(());
    }
    // For each candidate URL we resolve its anchor (the priority-picked
    // canonical_referenced_url, falling back to canonical_url) and the
    // feed_snapshot row for that anchor. Two ColBERT results that point
    // at the same resource (e.g. the arxiv paper + a tweet that links
    // it) share an anchor_url, so the dedup below collapses them into
    // a single result row, keeping the highest-blended candidate.
    let urls: Vec<String> = url_set.into_iter().collect();
    // url → FeedInfo (anchor_url, feed_score, sharers, sharer_count).
    // The sharers / sharer_count come from feed_snapshot (the cross-user
    // roll-up) so a merged result can render the same avatar stack
    // the global feed shows.
    let url_info = fetch_feed_info(pool, &urls).await?;
    for r in results.iter_mut() {
        // Per-anchor aggregator: track the winning candidate's
        // metadata + every candidate's linked_urls so we can union
        // them when emitting the merged row. The "winning" candidate
        // is the highest-blended (= ColBERT × feed-score) — that's
        // the doc whose title/summary best matches the query.
        struct AnchorAgg {
            best_doc_id: i64,
            best_blended: f32,
            best_colbert: f32,
            feed_score: Option<f64>,
            best_meta: Option<serde_json::Value>,
            // Union of every candidate's `url` field at this anchor —
            // surfaces in the response as `aggregated_urls` so the
            // client can render the full resource bundle (e.g. the
            // paper, the abs page, and every tweet linking it).
            aggregated_urls: Vec<String>,
            seen_urls: HashSet<String>,
            // Union of every candidate's `linked_urls` array, deduped
            // by the `url` field within each linked-URL object.
            merged_linked: Vec<serde_json::Value>,
            seen_linked: HashSet<String>,
            // Cross-personality sharer aggregate from feed_snapshot.
            sharers: Option<serde_json::Value>,
            sharer_count: i32,
        }
        let mut by_anchor: HashMap<String, AnchorAgg> = HashMap::new();
        let mut anchor_order: Vec<String> = Vec::new();
        for i in 0..r.document_ids.len() {
            let url_opt = r
                .metadata
                .get(i)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get("url").and_then(|v| v.as_str()))
                .map(|s| s.to_string());
            let Some(url) = url_opt else { continue };
            let info = url_info.get(&url);
            let (anchor, fs, sharers_json, sharer_count) = match info {
                Some(fi) => (
                    fi.anchor_url.clone(),
                    fi.feed_score,
                    fi.sharers.clone(),
                    fi.sharer_count,
                ),
                None => {
                    if strict_feed_filter {
                        continue;
                    }
                    (url.clone(), None, None, 0)
                }
            };
            let fs_for_blend = fs.unwrap_or(0.0);
            if strict_feed_filter && fs.is_none() {
                continue;
            }
            let colbert = r.scores[i];
            let blended =
                (colbert as f64 + feed_weight * (1.0 + fs_for_blend.max(0.0)).ln()) as f32;

            // Linked URLs on this candidate (deduped on the way in).
            let linked_from_this: Vec<serde_json::Value> = r
                .metadata
                .get(i)
                .and_then(|m| m.as_ref())
                .and_then(|m| m.get("linked_urls"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let entry = by_anchor.entry(anchor.clone()).or_insert_with(|| {
                anchor_order.push(anchor.clone());
                AnchorAgg {
                    best_doc_id: r.document_ids[i],
                    best_blended: f32::NEG_INFINITY,
                    best_colbert: 0.0,
                    feed_score: fs,
                    best_meta: None,
                    aggregated_urls: Vec::new(),
                    seen_urls: HashSet::new(),
                    merged_linked: Vec::new(),
                    seen_linked: HashSet::new(),
                    sharers: sharers_json.clone(),
                    sharer_count,
                }
            });
            if !entry.seen_urls.contains(&url) {
                entry.seen_urls.insert(url.clone());
                entry.aggregated_urls.push(url.clone());
            }
            for lu in &linked_from_this {
                // Dedup by the `url` field of each linked-URL object;
                // fall back to the full JSON string when no url key.
                let key = lu
                    .get("url")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| lu.to_string());
                if entry.seen_linked.insert(key) {
                    entry.merged_linked.push(lu.clone());
                }
            }
            if blended > entry.best_blended {
                entry.best_doc_id = r.document_ids[i];
                entry.best_blended = blended;
                entry.best_colbert = colbert;
                entry.feed_score = fs;
                entry.best_meta = r.metadata[i].clone();
                // Refresh sharers/count from feed_snapshot when the
                // best candidate's anchor was in the snapshot. If
                // strict_feed_filter was false and the best one missed
                // feed_snapshot, we keep whatever sharers we already
                // had from an earlier candidate.
                if info.is_some() {
                    entry.sharers = sharers_json;
                    entry.sharer_count = sharer_count;
                }
            }
        }
        // Second-level dedup: same visible content under different
        // anchors. The anchor collapse can't merge an author's
        // re-post of the identical tweet (each copy self-anchors),
        // so fold anchor groups whose winning title+summary
        // normalize to the same signature into the first group.
        let mut sig_owner: HashMap<String, String> = HashMap::new();
        let mut deduped_order: Vec<String> = Vec::new();
        for anchor in anchor_order {
            let Some(agg) = by_anchor.get(&anchor) else {
                continue;
            };
            let Some(sig) = content_signature(agg.best_meta.as_ref()) else {
                deduped_order.push(anchor);
                continue;
            };
            let Some(owner_anchor) = sig_owner.get(&sig).cloned() else {
                sig_owner.insert(sig, anchor.clone());
                deduped_order.push(anchor);
                continue;
            };
            let Some(dup) = by_anchor.remove(&anchor) else {
                continue;
            };
            let Some(owner) = by_anchor.get_mut(&owner_anchor) else {
                continue;
            };
            for u in dup.aggregated_urls {
                if owner.seen_urls.insert(u.clone()) {
                    owner.aggregated_urls.push(u);
                }
            }
            for lu in dup.merged_linked {
                let key = lu
                    .get("url")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| lu.to_string());
                if owner.seen_linked.insert(key) {
                    owner.merged_linked.push(lu);
                }
            }
            if dup.best_blended > owner.best_blended {
                owner.best_doc_id = dup.best_doc_id;
                owner.best_blended = dup.best_blended;
                owner.best_colbert = dup.best_colbert;
                owner.feed_score = dup.feed_score;
                owner.best_meta = dup.best_meta;
                owner.sharers = dup.sharers;
                owner.sharer_count = dup.sharer_count;
            }
        }
        let mut anchor_order = deduped_order;
        anchor_order.sort_by(|a, b| {
            let sa = by_anchor.get(a).map(|t| t.best_blended).unwrap_or(0.0);
            let sb = by_anchor.get(b).map(|t| t.best_blended).unwrap_or(0.0);
            sb.total_cmp(&sa)
        });
        anchor_order.truncate(top_k);
        r.document_ids.clear();
        r.scores.clear();
        r.metadata.clear();
        for anchor in anchor_order {
            let Some(agg) = by_anchor.remove(&anchor) else {
                continue;
            };
            r.document_ids.push(agg.best_doc_id);
            r.scores.push(agg.best_blended);
            let mut m = agg.best_meta.unwrap_or_else(|| serde_json::json!({}));
            if let Some(obj) = m.as_object_mut() {
                obj.insert(
                    "feed_score".to_string(),
                    serde_json::Value::from(agg.feed_score),
                );
                obj.insert(
                    "colbert_score".to_string(),
                    serde_json::Value::from(agg.best_colbert as f64),
                );
                obj.insert("anchor_url".to_string(), serde_json::Value::from(anchor));
                // Override linked_urls with the merged set so the
                // surviving card carries every distinct linked URL
                // any candidate at this anchor reported.
                obj.insert(
                    "linked_urls".to_string(),
                    serde_json::Value::Array(agg.merged_linked),
                );
                // Companion URLs (the duplicates the dedup collapsed).
                // Empty when no merging happened.
                obj.insert(
                    "aggregated_urls".to_string(),
                    serde_json::Value::Array(
                        agg.aggregated_urls
                            .into_iter()
                            .map(serde_json::Value::String)
                            .collect(),
                    ),
                );
                // Cross-personality sharer roll-up (same shape as
                // feed_snapshot.sharers / sharer_count). When the
                // anchor isn't in feed_snapshot these stay
                // null / 0, which is the right "no breadth signal"
                // representation.
                obj.insert(
                    "sharers".to_string(),
                    agg.sharers.unwrap_or(serde_json::Value::Null),
                );
                obj.insert(
                    "sharer_count".to_string(),
                    serde_json::Value::from(agg.sharer_count),
                );
            }
            r.metadata.push(Some(m));
        }
    }
    let n_results: usize = results.iter().map(|r| r.document_ids.len()).sum();
    tracing::info!(
        filter_ms = filter_t0.elapsed().as_millis() as u64,
        n_results = n_results,
        strict = strict_feed_filter,
        "search.filter.complete"
    );
    Ok(())
}

/// Search an index with query embeddings.
#[utoipa::path(
    post,
    path = "/indices/{name}/search",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id = trace_id.map(|t| t.0).unwrap_or_default();
    let start = std::time::Instant::now();

    let has_queries = req.queries.as_ref().map(|q| !q.is_empty()).unwrap_or(false);
    let has_text_query = req
        .text_query
        .as_ref()
        .map(|q| !q.is_empty())
        .unwrap_or(false);

    if !has_queries && !has_text_query {
        return Err(ApiError::BadRequest(
            "At least one of 'queries' (embeddings) or 'text_query' (keyword) must be provided"
                .to_string(),
        ));
    }

    let alpha = req.alpha.unwrap_or(0.75);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ApiError::BadRequest(
            "alpha must be between 0.0 and 1.0".to_string(),
        ));
    }

    let fusion_mode = req.fusion.as_deref().unwrap_or("rrf");
    if fusion_mode != "rrf" && fusion_mode != "relative_score" {
        return Err(ApiError::BadRequest(
            "fusion must be 'rrf' or 'relative_score'".to_string(),
        ));
    }

    // Hybrid mode: text_query is a single string, so queries must have exactly 1 element
    if has_queries && has_text_query {
        let queries_len = req.queries.as_ref().unwrap().len();
        if queries_len != 1 {
            return Err(ApiError::BadRequest(format!(
                "Hybrid search requires exactly 1 query embedding (got {}). \
                 text_query is a single string and can only fuse with one semantic query.",
                queries_len
            )));
        }
    }

    let requested_top_k = req.params.top_k.unwrap_or(state.config.default_top_k);
    // Over-fetch internally so the post-search dedup (always on) and
    // feed-snapshot filter (strict mode only) still leave enough
    // candidates to fill the requested top_k. ColBERT IDs that miss
    // feed_snapshot drop out under strict; otherwise duplicates that
    // anchor-collapse fall away.
    let feed_scope = req.feed_scope.unwrap_or(false);
    let top_k = if feed_scope {
        // Strict feed-snapshot filter: 1.5× over-fetch. Past 3×
        // we paid a ~4s ColBERT scan tax for a top_k=200 request
        // (600 internal). The anchor-merge collapses ~10–30% of
        // raw hits, so 1.5× leaves the user-visible top_k full
        // without paying for headroom we rarely consume.
        requested_top_k
            .saturating_mul(3)
            .div_ceil(2)
            .clamp(120, 400)
            .max(requested_top_k)
    } else {
        // Anchor-dedup only: 1.3× over-fetch — duplicates are rarer
        // on per-library indices since they're scoped to one owner.
        requested_top_k
            .saturating_mul(13)
            .div_ceil(10)
            .clamp(80, 300)
            .max(requested_top_k)
    };
    let path_str = state.index_path(&name).to_string_lossy().to_string();

    // Resolve filter condition to subset
    let mut subset = req.subset.clone();
    if let Some(ref condition) = req.filter_condition {
        if !filtering::exists(&path_str) {
            return Err(ApiError::MetadataNotFound(name.clone()));
        }
        let filter_params = req.filter_parameters.as_deref().unwrap_or(&[]);
        let filtered_ids = filtering::where_condition(&path_str, condition, filter_params)
            .map_err(|e| ApiError::BadRequest(format!("Invalid filter condition: {}", e)))?;
        subset = Some(filtered_ids);
    }

    // --- Pure semantic search (preserves batch query support) ---
    if has_queries && !has_text_query {
        let queries_vec = req.queries.as_ref().unwrap();
        let queries: Vec<Array2<f32>> = queries_vec
            .iter()
            .map(to_ndarray)
            .collect::<ApiResult<Vec<_>>>()?;

        let idx = state.get_index_for_read(&name)?;
        let expected_dim = idx.embedding_dim();
        for query in queries.iter() {
            if query.ncols() != expected_dim {
                return Err(ApiError::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.ncols(),
                });
            }
        }

        let params = SearchParameters {
            top_k,
            n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
            n_full_scores: req.params.n_full_scores.unwrap_or(4096),
            batch_size: 2000,
            centroid_score_threshold: req.params.centroid_score_threshold.unwrap_or_default(),
            ..Default::default()
        };

        let index = &**idx;
        let raw_results: Vec<(usize, Vec<i64>, Vec<f32>)> = if queries.len() == 1 {
            let r = index.search(&queries[0], &params, subset.as_deref())?;
            vec![(r.query_id, r.passage_ids, r.scores)]
        } else {
            let batch = index.search_batch(&queries, &params, true, subset.as_deref())?;
            batch
                .into_iter()
                .map(|r| (r.query_id, r.passage_ids, r.scores))
                .collect()
        };

        let mut results: Vec<QueryResultResponse> = raw_results
            .into_iter()
            .map(|(query_id, document_ids, scores)| {
                let metadata = fetch_metadata_for_docs(&path_str, &document_ids)?;
                Ok(QueryResultResponse {
                    query_id,
                    document_ids,
                    scores,
                    metadata,
                })
            })
            .collect::<ApiResult<Vec<_>>>()?;

        // Always run anchor-dedup + feed-score blend. `feed_scope`
        // controls whether long-tail results (anchor not in
        // feed_snapshot) get dropped (strict) or kept as their own
        // anchors with no feed-score boost. The semantic-only branch
        // never has a text_query, so search_intent=false (browse mode
        // weight applies — popularity is a meaningful signal when the
        // user gave no relevance cue).
        if let Some(pool) = state.pg_pool.as_ref() {
            apply_feed_scope_filter(pool, &mut results, requested_top_k, feed_scope, false).await?;
        } else if feed_scope {
            return Err(ApiError::Internal(
                "feed_scope requested but PgPool unavailable".to_string(),
            ));
        }

        let total_results: usize = results.iter().map(|r| r.document_ids.len()).sum();
        let total_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            trace_id = %trace_id,
            index = %name,
            mode = "semantic",
            num_queries = queries.len(),
            top_k = requested_top_k,
            total_results = total_results,
            total_ms = total_ms,
            "search.complete"
        );
        if total_ms > 1000 {
            tracing::warn!(trace_id = %trace_id, index = %name, total_ms = total_ms, "search.slow");
        }

        return Ok(PrettyJson(SearchResponse {
            num_queries: queries.len(),
            results,
        }));
    }

    // --- Keyword or hybrid search (supports batch) ---
    let empty_text: Vec<String> = vec![];
    let text_queries = req.text_query.as_ref().unwrap_or(&empty_text);
    let embedding_queries = req.queries.as_ref();

    // Validate: in hybrid mode, queries and text_query must have the same length
    if has_queries && has_text_query {
        let n_emb = embedding_queries.unwrap().len();
        let n_txt = text_queries.len();
        if n_emb != n_txt {
            return Err(ApiError::BadRequest(format!(
                "queries length ({}) must match text_query length ({}) in hybrid mode",
                n_emb, n_txt
            )));
        }
    }

    let num_queries = if has_text_query {
        text_queries.len()
    } else {
        embedding_queries.map(|q| q.len()).unwrap_or(0)
    };

    let fetch_k = if has_queries && has_text_query {
        top_k * 3
    } else {
        top_k
    };

    // Process each query
    let mut all_results: Vec<QueryResultResponse> = Vec::with_capacity(num_queries);

    #[allow(clippy::needless_range_loop)]
    for i in 0..num_queries {
        // Semantic component for this query
        let semantic: Option<(Vec<i64>, Vec<f32>)> = if has_queries {
            let query = to_ndarray(&embedding_queries.unwrap()[i])?;
            let idx = state.get_index_for_read(&name)?;
            let expected_dim = idx.embedding_dim();
            if query.ncols() != expected_dim {
                return Err(ApiError::DimensionMismatch {
                    expected: expected_dim,
                    actual: query.ncols(),
                });
            }
            let params = SearchParameters {
                top_k: fetch_k,
                n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
                n_full_scores: req.params.n_full_scores.unwrap_or(4096),
                batch_size: 2000,
                centroid_score_threshold: req.params.centroid_score_threshold.unwrap_or_default(),
                ..Default::default()
            };
            let r = idx.search(&query, &params, subset.as_deref())?;
            Some((r.passage_ids, r.scores))
        } else {
            None
        };

        // Keyword component for this query.
        //
        // text_search::search passes the string straight into the FTS5
        // MATCH clause, which has its own mini-grammar (AND/OR/NOT,
        // quotes, parens, colons). Anything resembling an operator or
        // a stray quote raises a parse error and the keyword half
        // drops out — which silently degrades hybrid search to
        // semantic-only. sanitize_fts5_query strips operators and
        // wraps every word in literal quotes, joined by implicit AND.
        let keyword: Option<(Vec<i64>, Vec<f32>)> = if has_text_query {
            let tq_raw = &text_queries[i];
            let tq = text_search::sanitize_fts5_query(tq_raw);
            if tq.is_empty() {
                None
            } else {
                let result = if let Some(ref sub) = subset {
                    text_search::search_filtered(&path_str, &tq, fetch_k, sub)
                } else {
                    text_search::search(&path_str, &tq, fetch_k)
                };
                match result {
                    Ok(r) => Some((r.passage_ids, r.scores)),
                    Err(e) => {
                        tracing::warn!(trace_id = %trace_id, index = %name, error = %e, "search.keyword.failed");
                        None
                    }
                }
            }
        } else {
            None
        };

        // Fuse
        let (document_ids, scores) = match (semantic, keyword) {
            (Some((sem_ids, sem_scores)), Some((kw_ids, kw_scores))) => match fusion_mode {
                "relative_score" => text_search::fuse_relative_score(
                    &sem_ids,
                    &sem_scores,
                    &kw_ids,
                    &kw_scores,
                    alpha,
                    top_k,
                ),
                _ => text_search::fuse_rrf(&sem_ids, &kw_ids, alpha, top_k),
            },
            (Some((ids, scores)), None) => {
                let mut r: Vec<(i64, f32)> = ids.into_iter().zip(scores).collect();
                r.truncate(top_k);
                (
                    r.iter().map(|x| x.0).collect(),
                    r.iter().map(|x| x.1).collect(),
                )
            }
            (None, Some((ids, scores))) => {
                let mut r: Vec<(i64, f32)> = ids.into_iter().zip(scores).collect();
                r.truncate(top_k);
                (
                    r.iter().map(|x| x.0).collect(),
                    r.iter().map(|x| x.1).collect(),
                )
            }
            (None, None) => (vec![], vec![]),
        };

        let metadata = fetch_metadata_for_docs(&path_str, &document_ids)?;
        all_results.push(QueryResultResponse {
            query_id: i,
            document_ids,
            scores,
            metadata,
        });
    }

    // Always run anchor-dedup + feed-score blend (see semantic branch
    // above). `feed_scope` only controls whether non-snapshot
    // candidates are dropped. `has_text_query` tells the blend the
    // user has expressed a specific relevance intent, so popularity
    // gets a lighter weight (a tiebreaker, not a primary ranker).
    if let Some(pool) = state.pg_pool.as_ref() {
        apply_feed_scope_filter(
            pool,
            &mut all_results,
            requested_top_k,
            feed_scope,
            has_text_query,
        )
        .await?;
    } else if feed_scope {
        return Err(ApiError::Internal(
            "feed_scope requested but PgPool unavailable".to_string(),
        ));
    }

    let total_results: usize = all_results.iter().map(|r| r.document_ids.len()).sum();
    let total_ms = start.elapsed().as_millis() as u64;

    let mode = if has_queries && has_text_query {
        "hybrid"
    } else {
        "keyword"
    };

    tracing::info!(
        trace_id = %trace_id,
        index = %name,
        mode = mode,
        num_queries = num_queries,
        top_k = requested_top_k,
        total_results = total_results,
        total_ms = total_ms,
        "search.complete"
    );
    if total_ms > 1000 {
        tracing::warn!(trace_id = %trace_id, index = %name, total_ms = total_ms, "search.slow");
    }

    Ok(PrettyJson(SearchResponse {
        num_queries,
        results: all_results,
    }))
}

/// Search with a pre-filtered subset from metadata query.
///
/// This is a convenience endpoint that combines metadata filtering and search.
#[utoipa::path(
    post,
    path = "/indices/{name}/search/filtered",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = FilteredSearchRequest,
    responses(
        (status = 200, description = "Filtered search results", body = SearchResponse),
        (status = 400, description = "Invalid request or filter condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn search_filtered(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<FilteredSearchRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    // Convert to unified SearchRequest with filter_condition
    let search_req = SearchRequest {
        queries: Some(req.queries),
        params: req.params,
        subset: None,
        text_query: None,
        alpha: None,
        fusion: None,
        filter_condition: Some(req.filter_condition),
        filter_parameters: Some(req.filter_parameters),
        feed_scope: None,
    };

    search(State(state), Path(name), trace_id, Json(search_req)).await
}

/// Search an index using text queries (requires model to be loaded).
///
/// This endpoint encodes the text queries using the loaded model and then performs a search.
/// Requires the server to be started with `--model <path>`.
#[utoipa::path(
    post,
    path = "/indices/{name}/search_with_encoding",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = SearchWithEncodingRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 400, description = "Invalid request or model not loaded", body = ErrorResponse),
        (status = 404, description = "Index not found", body = ErrorResponse)
    )
)]
pub async fn search_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<SearchWithEncodingRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_val = trace_id.as_ref().map(|t| t.0.clone()).unwrap_or_default();
    let start = std::time::Instant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let num_queries = req.queries.len();

    // Encode the text queries (async, uses batch queue)
    let encode_start = std::time::Instant::now();
    let query_embeddings =
        encode_texts_internal(state.clone(), &req.queries, InputType::Query, None).await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    // Convert to QueryEmbeddings format
    let queries: Vec<QueryEmbeddings> = query_embeddings
        .into_iter()
        .map(|arr| QueryEmbeddings {
            embeddings: Some(arr.rows().into_iter().map(|r| r.to_vec()).collect()),
            embeddings_b64: None,
            shape: None,
        })
        .collect();

    // Create a standard SearchRequest (pass through hybrid fields)
    let search_req = SearchRequest {
        queries: Some(queries),
        params: req.params,
        subset: req.subset,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: None,
        filter_parameters: None,
        feed_scope: req.feed_scope,
    };

    // Delegate to the standard search
    let result = search(State(state), Path(name.clone()), trace_id, Json(search_req)).await;

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id_val,
        index = %name,
        num_queries = num_queries,
        encode_ms = encode_ms,
        total_ms = total_ms,
        "search.with_encoding.complete"
    );

    result
}

/// Search with text queries and a metadata filter (requires model to be loaded).
///
/// This endpoint encodes the text queries using the loaded model and performs a filtered search.
/// Requires the server to be started with `--model <path>`.
#[utoipa::path(
    post,
    path = "/indices/{name}/search/filtered_with_encoding",
    tag = "search",
    params(
        ("name" = String, Path, description = "Index name")
    ),
    request_body = FilteredSearchWithEncodingRequest,
    responses(
        (status = 200, description = "Filtered search results", body = SearchResponse),
        (status = 400, description = "Invalid request, model not loaded, or filter condition", body = ErrorResponse),
        (status = 404, description = "Index or metadata not found", body = ErrorResponse)
    )
)]
pub async fn search_filtered_with_encoding(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    trace_id: Option<Extension<TraceId>>,
    Json(req): Json<FilteredSearchWithEncodingRequest>,
) -> ApiResult<PrettyJson<SearchResponse>> {
    let trace_id_val = trace_id.as_ref().map(|t| t.0.clone()).unwrap_or_default();
    let start = std::time::Instant::now();

    if req.queries.is_empty() {
        return Err(ApiError::BadRequest("No queries provided".to_string()));
    }

    let num_queries = req.queries.len();

    // Encode the text queries (async, uses batch queue)
    let encode_start = std::time::Instant::now();
    let query_embeddings =
        encode_texts_internal(state.clone(), &req.queries, InputType::Query, None).await?;
    let encode_ms = encode_start.elapsed().as_millis() as u64;

    // Convert to QueryEmbeddings format
    let queries: Vec<QueryEmbeddings> = query_embeddings
        .into_iter()
        .map(|arr| QueryEmbeddings {
            embeddings: Some(arr.rows().into_iter().map(|r| r.to_vec()).collect()),
            embeddings_b64: None,
            shape: None,
        })
        .collect();

    // Create a unified SearchRequest with filter (pass through hybrid fields)
    let search_req = SearchRequest {
        queries: Some(queries),
        params: req.params,
        subset: None,
        text_query: req.text_query,
        alpha: req.alpha,
        fusion: req.fusion,
        filter_condition: Some(req.filter_condition.clone()),
        filter_parameters: Some(req.filter_parameters),
        feed_scope: req.feed_scope,
    };

    // Delegate to the unified search handler
    let result = search(State(state), Path(name.clone()), trace_id, Json(search_req)).await;

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        trace_id = %trace_id_val,
        index = %name,
        num_queries = num_queries,
        filter = %req.filter_condition,
        encode_ms = encode_ms,
        total_ms = total_ms,
        "search.filtered_with_encoding.complete"
    );

    result
}
