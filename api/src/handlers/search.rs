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

/// How deep into the unfiltered BM25 ranking `keyword_search_subset` reads on
/// its first pass. Deep enough that any filter admitting ~2 % of a query's
/// matches fills a 1 000-document `fetch_k` outright; shallow enough that a
/// stopword query (`the` matches 399 k rows in the `__all__` index) doesn't
/// drag the whole corpus through SQLite's sorter — bounded costs 0.5 s there
/// against 1.6 s unbounded.
const KEYWORD_SCAN_DEPTH: usize = 50_000;

/// Second-pass depth: past any real index, so the pass is effectively
/// unbounded. Not `usize::MAX`, which next-plaid casts to `LIMIT -1`.
const KEYWORD_SCAN_ALL: usize = i32::MAX as usize;

/// BM25 over a metadata-filtered subset of an index.
///
/// This replaces `text_search::search_filtered`, which narrows the FTS5 query
/// with `… MATCH ? AND rowid IN (<subset>)`. That reads like a pre-filter and
/// behaves like the opposite: SQLite hands the rowid list to FTS5's
/// `xBestIndex` as an *equality* constraint — the plan reads `SCAN
/// METADATA_FTS VIRTUAL TABLE INDEX 0:=M1` — then drives the loop from the
/// ids, re-running the whole full-text query once per id at ~100 µs a go.
///
/// So the keyword half of a filtered hybrid search cost O(subset), and got
/// *slower* the more documents the filter let through. Against the
/// 685 k-document `__all__` index: 16 ms unfiltered, 3.7 s behind the
/// `github` source chip (41 k ids), 6.5 s behind `github`+`arxiv`, 37 s
/// behind `twitter` (499 k) — which is what made picking a source in the left
/// rail feel like the search had hung.
///
/// Rank unfiltered and intersect here instead. FTS5 has to score every match
/// before it can honour `ORDER BY`, so a deeper `LIMIT` costs barely more
/// than a shallow one (94 k matches: 271 ms unbounded vs 322 ms at `LIMIT
/// 600`) and one wide pass beats anything that re-runs the MATCH.
///
/// The answer is exact — the unfiltered ranking restricted to the subset,
/// the same rows `search_filtered` returned. A first pass reading
/// `KEYWORD_SCAN_DEPTH` deep settles it whenever it either fills `fetch_k`
/// (everything below is lower-scoring by construction) or runs out of
/// matches. Only the combination that satisfies neither — a query matching
/// more than 50 k documents behind a filter narrow enough to contribute few
/// of them, e.g. `the` behind the `huggingface` chip — pays a second,
/// unbounded pass.
///
/// Fixed upstream in lightonai/next-plaid#183; this can go back to
/// `search_filtered` once the API builds against a release carrying it.
fn keyword_search_subset(
    path: &str,
    query: &str,
    fetch_k: usize,
    subset: &[i64],
) -> next_plaid::Result<next_plaid::search::QueryResult> {
    if subset.is_empty() || fetch_k == 0 {
        return Ok(next_plaid::search::QueryResult {
            query_id: 0,
            passage_ids: vec![],
            scores: vec![],
        });
    }

    let allowed: HashSet<i64> = subset.iter().copied().collect();
    let mut depth = fetch_k.max(KEYWORD_SCAN_DEPTH);

    loop {
        let ranked = text_search::search(path, query, depth)?;
        let saturated = ranked.passage_ids.len() >= depth;

        let mut passage_ids: Vec<i64> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for (id, score) in ranked.passage_ids.into_iter().zip(ranked.scores) {
            if !allowed.contains(&id) {
                continue;
            }
            passage_ids.push(id);
            scores.push(score);
            if passage_ids.len() == fetch_k {
                break;
            }
        }

        // Short of `fetch_k` while the pass was still saturated means more
        // subset rows may sit below the window — widen once and settle it.
        if passage_ids.len() < fetch_k && saturated && depth < KEYWORD_SCAN_ALL {
            depth = KEYWORD_SCAN_ALL;
            continue;
        }

        return Ok(next_plaid::search::QueryResult {
            query_id: 0,
            passage_ids,
            scores,
        });
    }
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
    /// Pedagogical rewrite from the clean daemon (empty when the doc
    /// hasn't been cleaned). The index metadata doesn't carry these,
    /// so search results pick them up here; dedup also prefers a
    /// cleaned candidate as a group's representative.
    pub clean_title: String,
    pub clean_summary: String,
}

impl FeedInfo {
    pub(crate) fn is_cleaned(&self) -> bool {
        !self.clean_summary.trim().is_empty()
    }
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
        String,
        String,
    )> = sqlx::query_as(
        "WITH input AS (\n            SELECT d.url, d.canonical_url, d.canonical_referenced_urls, d.linked_urls,\n                   d.clean_title, d.clean_summary\n              FROM documents d\n             WHERE d.url = ANY($1::text[])\n               -- `AND d.deleted = FALSE` forces the planner to use\n               -- `idx_documents_url_live` (partial, on url WHERE\n               -- deleted=false) instead of `documents_pkey`\n               -- (composite on `user_id, url`). The PK can't seek by\n               -- url alone — it scans the full key range and pays\n               -- ~1.5 M buffer hits per call.\n               AND d.deleted = FALSE\n         ),\n         resolved AS (\n            SELECT i.url, i.linked_urls, i.clean_title, i.clean_summary,\n                   COALESCE(\n                       (SELECT ref FROM unnest(i.canonical_referenced_urls) ref\n                         ORDER BY CASE\n                           WHEN ref LIKE 'https://arxiv.org/abs/%'       THEN 1\n                           WHEN ref LIKE 'https://huggingface.co/%'      THEN 2\n                           WHEN ref LIKE 'https://github.com/%'          THEN 3\n                           WHEN ref LIKE 'https://openreview.net/%'      THEN 4\n                           WHEN ref LIKE 'https://doi.org/%'             THEN 5\n                           WHEN ref LIKE 'https://paperswithcode.com/%'  THEN 6\n                           WHEN ref LIKE 'https://aclanthology.org/%'    THEN 7\n                           WHEN ref LIKE 'https://semanticscholar.org/%' THEN 8\n                           WHEN ref LIKE 'https://distill.pub/%'         THEN 9\n                           WHEN ref LIKE 'https://biorxiv.org/%'         THEN 10\n                           WHEN ref LIKE 'https://medrxiv.org/%'         THEN 11\n                           ELSE 99\n                         END, ref LIMIT 1),\n                       i.canonical_url\n                   ) AS anchor_url\n              FROM input i\n         )\n         SELECT r.url, r.anchor_url, fs.score, fs.sharers, fs.sharer_count, r.linked_urls,\n                r.clean_title, r.clean_summary\n           FROM resolved r\n           LEFT JOIN feed_snapshot fs ON fs.anchor_url = r.anchor_url",
    )
    .bind(urls)
    .fetch_all(pool)
    .await
    .map_err(|e| ApiError::Internal(format!("feed_snapshot anchor lookup failed: {}", e)))?;

    let mut out: HashMap<String, FeedInfo> = HashMap::new();
    for (url, anchor, score, sharers, sharer_count, linked_urls, clean_title, clean_summary) in rows
    {
        // Several users can hold the same url; keep a cleaned copy
        // over an uncleaned one so the rewrite isn't lost to row order.
        if clean_summary.trim().is_empty() && out.get(&url).is_some_and(|fi| fi.is_cleaned()) {
            continue;
        }
        let anchor = anchor.unwrap_or_else(|| url.clone());
        out.insert(
            url,
            FeedInfo {
                anchor_url: anchor,
                feed_score: score,
                sharers,
                sharer_count: sharer_count.unwrap_or(0),
                linked_urls,
                clean_title,
                clean_summary,
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

/// Twitter media ids embedded in a doc's summary (the `📷 …` / `🎬 …`
/// markers the tweet fetcher writes). An uploaded image or video keeps
/// its id when the post is retweeted or quoted, so two results sharing
/// one are reshares of the same post — commentary on top or not —
/// even though their text and anchors differ.
pub(crate) fn media_ids(meta: Option<&serde_json::Value>) -> Vec<String> {
    meta.and_then(|m| m.get("summary"))
        .and_then(|v| v.as_str())
        .map(twimg_ids)
        .unwrap_or_default()
}

/// Media ids in any text: `pbs.twimg.com/media/<id>`, video ids from
/// `video.twimg.com/amplify_video/<id>/…` and the matching poster
/// thumbnails, in order of appearance, deduped.
fn twimg_ids(text: &str) -> Vec<String> {
    const KINDS: [&str; 7] = [
        "media/",
        "amplify_video/",
        "amplify_video_thumb/",
        "ext_tw_video/",
        "ext_tw_video_thumb/",
        "tweet_video/",
        "tweet_video_thumb/",
    ];
    let mut out: Vec<String> = Vec::new();
    for (i, needle) in text.match_indices("twimg.com/") {
        let rest = &text[i + needle.len()..];
        let Some(tail) = KINDS.iter().find_map(|k| rest.strip_prefix(k)) else {
            continue;
        };
        let id: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        // Real ids are 15+ chars; anything shorter is a truncated
        // summary cutting the URL mid-id.
        if id.len() >= 10 && !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// Photo / video attachments of a tweet doc, parsed from the
/// `📷 <url>` and `🎬 <poster> | <mp4>` lines the fetcher writes into
/// the summary (same format the frontend's tweet renderer reads).
/// Each item is `(dedup key, tile JSON)`; the key is the twimg media
/// id so a reshare's copy of an image collapses onto the original.
/// `tweet` records which post the item came from, so a video tile on
/// a merged card links to the tweet that actually carries it.
pub(crate) fn media_items(
    meta: Option<&serde_json::Value>,
    tweet_url: &str,
) -> Vec<(String, serde_json::Value)> {
    let Some(summary) = meta.and_then(|m| m.get("summary")).and_then(|v| v.as_str()) else {
        return Vec::new();
    };
    let key_of = |urls: &[&str]| -> String {
        urls.iter()
            .find_map(|u| twimg_ids(u).into_iter().next())
            .unwrap_or_else(|| {
                urls.iter()
                    .find(|u| !u.is_empty())
                    .unwrap_or(&"")
                    .to_string()
            })
    };
    let mut out: Vec<(String, serde_json::Value)> = Vec::new();
    for line in summary.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('📷') {
            let Some(url) = rest.split_whitespace().next() else {
                continue;
            };
            out.push((
                key_of(&[url]),
                serde_json::json!({"kind": "photo", "url": url, "tweet": tweet_url}),
            ));
        } else if let Some(rest) = line.strip_prefix('🎬') {
            let rest = rest.trim();
            let (poster, mp4) = match rest.split_once(" | ") {
                Some((p, m)) => (p.trim(), m.trim()),
                None => ("", rest),
            };
            if poster.is_empty() && mp4.is_empty() {
                continue;
            }
            out.push((
                key_of(&[poster, mp4]),
                serde_json::json!({"kind": "video", "poster": poster, "mp4": mp4, "tweet": tweet_url}),
            ));
        }
    }
    out
}

/// Dedup keys for the second-level (cross-anchor) collapse: the
/// normalized content signature plus one key per embedded media id.
pub(crate) fn dedup_keys(meta: Option<&serde_json::Value>) -> Vec<String> {
    let mut keys: Vec<String> = media_ids(meta)
        .into_iter()
        .map(|id| format!("media:{id}"))
        .collect();
    if let Some(sig) = content_signature(meta) {
        keys.push(format!("text:{sig}"));
    }
    keys
}

/// Union two `sharers` arrays (feed_snapshot shape), deduped by slug.
pub(crate) fn merge_sharers(
    a: Option<serde_json::Value>,
    b: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    let mut merged: Vec<serde_json::Value> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for v in [a, b].into_iter().flatten() {
        let Some(arr) = v.as_array() else { continue };
        for s in arr {
            let slug = s.get("slug").and_then(|x| x.as_str()).unwrap_or("");
            if slug.is_empty() || seen.insert(slug.to_string()) {
                merged.push(s.clone());
            }
        }
    }
    (!merged.is_empty()).then_some(serde_json::Value::Array(merged))
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
        // Per-anchor aggregator: track the representative candidate's
        // metadata + every candidate's linked_urls so we can union
        // them when emitting the merged row. The representative is the
        // AI-rewritten (clean daemon) candidate when one exists, else
        // the highest-blended (= ColBERT × feed-score) one; the group
        // itself ranks by its best blended score either way.
        struct AnchorAgg {
            best_doc_id: i64,
            best_blended: f32,
            best_colbert: f32,
            best_clean: bool,
            best_url: String,
            best_meta: Option<serde_json::Value>,
            // Max blended score across every candidate in the group —
            // the group's rank, independent of which card represents it.
            score: f32,
            feed_score: Option<f64>,
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
            // Cross-personality sharer aggregate from feed_snapshot,
            // unioned across groups the second-level dedup merges.
            sharers: Option<serde_json::Value>,
            sharer_count: i32,
            // Library owners of every collapsed candidate (`__all__`
            // index rows carry `owner`), so the card's avatar stack
            // shows everyone who posted it.
            owners: Vec<String>,
            // Second-level dedup keys (media ids + text signature)
            // collected from every candidate, not just the winner.
            keys: Vec<String>,
            // Every distinct photo / video across the group's
            // candidates, keyed by media id — the kept card shows
            // them all, once each.
            media: Vec<(String, serde_json::Value)>,
            seen_media: HashSet<String>,
        }
        impl AnchorAgg {
            /// Should a candidate with (`clean`, `blended`) replace the
            /// current representative? Cleaned beats uncleaned; within
            /// the same tier, the better match wins.
            fn prefers(&self, clean: bool, blended: f32) -> bool {
                (clean, blended) > (self.best_clean, self.best_blended)
            }
            /// Fold `dup` (a different anchor showing the same post)
            /// into `self`.
            fn absorb(&mut self, dup: AnchorAgg) {
                for u in dup.aggregated_urls {
                    if self.seen_urls.insert(u.clone()) {
                        self.aggregated_urls.push(u);
                    }
                }
                for lu in dup.merged_linked {
                    let key = lu
                        .get("url")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| lu.to_string());
                    if self.seen_linked.insert(key) {
                        self.merged_linked.push(lu);
                    }
                }
                for o in dup.owners {
                    if !self.owners.contains(&o) {
                        self.owners.push(o);
                    }
                }
                for k in dup.keys {
                    if !self.keys.contains(&k) {
                        self.keys.push(k);
                    }
                }
                for (k, item) in dup.media {
                    if self.seen_media.insert(k.clone()) {
                        self.media.push((k, item));
                    }
                }
                self.sharer_count = self.sharer_count.max(dup.sharer_count);
                self.sharers = merge_sharers(self.sharers.take(), dup.sharers);
                self.score = self.score.max(dup.score);
                self.feed_score = match (self.feed_score, dup.feed_score) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
                if self.prefers(dup.best_clean, dup.best_blended) {
                    self.best_doc_id = dup.best_doc_id;
                    self.best_blended = dup.best_blended;
                    self.best_colbert = dup.best_colbert;
                    self.best_clean = dup.best_clean;
                    self.best_url = dup.best_url;
                    self.best_meta = dup.best_meta;
                }
            }
        }
        let mut by_anchor: HashMap<String, AnchorAgg> = HashMap::new();
        let mut anchor_order: Vec<String> = Vec::new();
        for i in 0..r.document_ids.len() {
            let meta_i = r.metadata.get(i).and_then(|m| m.as_ref());
            let url_opt = meta_i
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
            let clean = info.is_some_and(|fi| fi.is_cleaned());
            let owner = meta_i
                .and_then(|m| m.get("owner"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Linked URLs on this candidate (deduped on the way in).
            let linked_from_this: Vec<serde_json::Value> = meta_i
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
                    best_clean: false,
                    best_url: url.clone(),
                    best_meta: None,
                    score: f32::NEG_INFINITY,
                    feed_score: fs,
                    aggregated_urls: Vec::new(),
                    seen_urls: HashSet::new(),
                    merged_linked: Vec::new(),
                    seen_linked: HashSet::new(),
                    sharers: sharers_json.clone(),
                    sharer_count,
                    owners: Vec::new(),
                    keys: Vec::new(),
                    media: Vec::new(),
                    seen_media: HashSet::new(),
                }
            });
            if !entry.seen_urls.contains(&url) {
                entry.seen_urls.insert(url.clone());
                entry.aggregated_urls.push(url.clone());
            }
            if !owner.is_empty() && !entry.owners.contains(&owner) {
                entry.owners.push(owner);
            }
            for k in dedup_keys(meta_i) {
                if !entry.keys.contains(&k) {
                    entry.keys.push(k);
                }
            }
            for (k, item) in media_items(meta_i, &url) {
                if entry.seen_media.insert(k.clone()) {
                    entry.media.push((k, item));
                }
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
            entry.score = entry.score.max(blended);
            // Every candidate here shares the anchor, hence the same
            // feed_snapshot row — fill sharers from whichever has it.
            if entry.sharers.is_none() && info.is_some() {
                entry.sharers = sharers_json;
                entry.sharer_count = sharer_count;
                entry.feed_score = fs;
            }
            if entry.prefers(clean, blended) {
                entry.best_doc_id = r.document_ids[i];
                entry.best_blended = blended;
                entry.best_colbert = colbert;
                entry.best_clean = clean;
                entry.best_url = url.clone();
                entry.best_meta = r.metadata[i].clone();
            }
        }
        // Second-level dedup: same post under different anchors. The
        // anchor collapse can't merge an author's re-post of the
        // identical tweet (each copy self-anchors), nor the retweets
        // and quote-tweets of one post (each wrapper is its own URL
        // with its own commentary). Fold groups that share a media id
        // or a normalized title+summary signature into the first
        // (highest-retrieved) group carrying that key.
        let mut key_owner: HashMap<String, String> = HashMap::new();
        let mut deduped_order: Vec<String> = Vec::new();
        for anchor in anchor_order {
            let Some(agg) = by_anchor.get(&anchor) else {
                continue;
            };
            let keys = agg.keys.clone();
            let owner_anchor = keys.iter().find_map(|k| key_owner.get(k).cloned());
            let Some(owner_anchor) = owner_anchor else {
                for k in keys {
                    key_owner.insert(k, anchor.clone());
                }
                deduped_order.push(anchor);
                continue;
            };
            let Some(dup) = by_anchor.remove(&anchor) else {
                continue;
            };
            let Some(owner) = by_anchor.get_mut(&owner_anchor) else {
                continue;
            };
            owner.absorb(dup);
            // Keys only the duplicate carried now resolve to the
            // merged group, so a third post sharing them joins too.
            for k in keys {
                key_owner.entry(k).or_insert_with(|| owner_anchor.clone());
            }
        }
        let mut anchor_order = deduped_order;
        anchor_order.sort_by(|a, b| {
            let sa = by_anchor.get(a).map(|t| t.score).unwrap_or(0.0);
            let sb = by_anchor.get(b).map(|t| t.score).unwrap_or(0.0);
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
            r.scores.push(agg.score);
            // The kept card's own attachments lead, then every other
            // distinct one the folded duplicates carried.
            let mut group_media: Vec<serde_json::Value> = Vec::new();
            let mut media_seen: HashSet<String> = HashSet::new();
            for (k, item) in media_items(agg.best_meta.as_ref(), &agg.best_url)
                .into_iter()
                .chain(agg.media)
            {
                if media_seen.insert(k) {
                    group_media.push(item);
                }
            }
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
                // Pedagogical rewrite of the representative, when the
                // clean daemon has processed it. The frontend renders
                // `clean_title || title` and `clean_summary || summary`.
                if let Some(fi) = url_info.get(&agg.best_url).filter(|fi| fi.is_cleaned()) {
                    obj.insert(
                        "clean_title".to_string(),
                        serde_json::Value::from(fi.clean_title.clone()),
                    );
                    obj.insert(
                        "clean_summary".to_string(),
                        serde_json::Value::from(fi.clean_summary.clone()),
                    );
                }
                // Override linked_urls with the merged set so the
                // surviving card carries every distinct linked URL
                // any candidate at this anchor reported.
                obj.insert(
                    "linked_urls".to_string(),
                    serde_json::Value::Array(agg.merged_linked),
                );
                // Companion URLs (the duplicates the dedup collapsed).
                // Single-element when no merging happened.
                obj.insert(
                    "aggregated_urls".to_string(),
                    serde_json::Value::Array(
                        agg.aggregated_urls
                            .into_iter()
                            .map(serde_json::Value::String)
                            .collect(),
                    ),
                );
                obj.insert(
                    "group_media".to_string(),
                    serde_json::Value::Array(group_media),
                );
                // Library owners of every collapsed candidate.
                obj.insert(
                    "group_owners".to_string(),
                    serde_json::Value::Array(
                        agg.owners
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
                let sharer_count = agg
                    .sharers
                    .as_ref()
                    .and_then(|v| v.as_array())
                    .map(|a| (a.len() as i32).max(agg.sharer_count))
                    .unwrap_or(agg.sharer_count);
                obj.insert(
                    "sharers".to_string(),
                    agg.sharers.unwrap_or(serde_json::Value::Null),
                );
                obj.insert(
                    "sharer_count".to_string(),
                    serde_json::Value::from(sharer_count),
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

    // Resolve filter condition to subset. SQLite can block on the
    // database lock while an update batch writes to the same file, so
    // this must run on the blocking pool: a runtime worker thread that
    // blocks in sync code can take the whole server down with it (all
    // four workers parked = no I/O driver = frozen accept loop).
    let mut subset = req.subset.clone();
    if let Some(condition) = req.filter_condition.clone() {
        let filter_params = req.filter_parameters.clone().unwrap_or_default();
        let path_bg = path_str.clone();
        let name_bg = name.clone();
        let filtered_ids = tokio::task::spawn_blocking(move || {
            if !filtering::exists(&path_bg) {
                return Err(ApiError::MetadataNotFound(name_bg));
            }
            filtering::where_condition(&path_bg, &condition, &filter_params)
                .map_err(|e| ApiError::BadRequest(format!("Invalid filter condition: {}", e)))
        })
        .await
        .map_err(|e| ApiError::Internal(format!("Filter task panicked: {}", e)))??;
        subset = Some(filtered_ids);
    }

    // --- Pure semantic search (preserves batch query support) ---
    if has_queries && !has_text_query {
        let queries_vec = req.queries.as_ref().unwrap();
        let queries: Vec<Array2<f32>> = queries_vec
            .iter()
            .map(to_ndarray)
            .collect::<ApiResult<Vec<_>>>()?;
        let num_queries = queries.len();

        let params = SearchParameters {
            top_k,
            n_ivf_probe: req.params.n_ivf_probe.unwrap_or(8),
            n_full_scores: req.params.n_full_scores.unwrap_or(4096),
            batch_size: 2000,
            centroid_score_threshold: req.params.centroid_score_threshold.unwrap_or_default(),
            ..Default::default()
        };

        // PLAID scan + SQLite metadata fetch are synchronous CPU-bound
        // calls — keep them off the async runtime (see comment on the
        // filter block above).
        let state_bg = state.clone();
        let name_bg = name.clone();
        let path_bg = path_str.clone();
        let subset_bg = subset.clone();
        let mut results: Vec<QueryResultResponse> =
            tokio::task::spawn_blocking(move || -> ApiResult<Vec<QueryResultResponse>> {
                let idx = state_bg.get_index_for_read(&name_bg)?;
                let expected_dim = idx.embedding_dim();
                for query in queries.iter() {
                    if query.ncols() != expected_dim {
                        return Err(ApiError::DimensionMismatch {
                            expected: expected_dim,
                            actual: query.ncols(),
                        });
                    }
                }

                let index = &**idx;
                let raw_results: Vec<(usize, Vec<i64>, Vec<f32>)> = if queries.len() == 1 {
                    let r = index.search(&queries[0], &params, subset_bg.as_deref())?;
                    vec![(r.query_id, r.passage_ids, r.scores)]
                } else {
                    let batch =
                        index.search_batch(&queries, &params, true, subset_bg.as_deref())?;
                    batch
                        .into_iter()
                        .map(|r| (r.query_id, r.passage_ids, r.scores))
                        .collect()
                };

                raw_results
                    .into_iter()
                    .map(|(query_id, document_ids, scores)| {
                        let metadata = fetch_metadata_for_docs(&path_bg, &document_ids)?;
                        Ok(QueryResultResponse {
                            query_id,
                            document_ids,
                            scores,
                            metadata,
                        })
                    })
                    .collect::<ApiResult<Vec<_>>>()
            })
            .await
            .map_err(|e| ApiError::Internal(format!("Search task panicked: {}", e)))??;

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
            num_queries = num_queries,
            top_k = requested_top_k,
            total_results = total_results,
            total_ms = total_ms,
            "search.complete"
        );
        if total_ms > 1000 {
            tracing::warn!(trace_id = %trace_id, index = %name, total_ms = total_ms, "search.slow");
        }

        return Ok(PrettyJson(SearchResponse {
            num_queries,
            results,
        }));
    }

    // --- Keyword or hybrid search (supports batch) ---
    // Validate: in hybrid mode, queries and text_query must have the same length
    if has_queries && has_text_query {
        let n_emb = req.queries.as_ref().unwrap().len();
        let n_txt = req.text_query.as_ref().unwrap().len();
        if n_emb != n_txt {
            return Err(ApiError::BadRequest(format!(
                "queries length ({}) must match text_query length ({}) in hybrid mode",
                n_emb, n_txt
            )));
        }
    }

    let num_queries = if has_text_query {
        req.text_query.as_ref().unwrap().len()
    } else {
        req.queries.as_ref().map(|q| q.len()).unwrap_or(0)
    };

    let fetch_k = if has_queries && has_text_query {
        top_k * 3
    } else {
        top_k
    };

    // Process each query. The whole loop is synchronous PLAID/SQLite
    // work — keep it off the async runtime (see comment on the filter
    // block above).
    let state_bg = state.clone();
    let name_bg = name.clone();
    let path_bg = path_str.clone();
    let trace_id_bg = trace_id.to_string();
    let mut all_results: Vec<QueryResultResponse> = tokio::task::spawn_blocking(
        move || -> ApiResult<Vec<QueryResultResponse>> {
            let empty_text: Vec<String> = vec![];
            let text_queries = req.text_query.as_ref().unwrap_or(&empty_text);
            let embedding_queries = req.queries.as_ref();
            let fusion_mode = req.fusion.as_deref().unwrap_or("rrf");

            let mut all_results: Vec<QueryResultResponse> = Vec::with_capacity(num_queries);

            #[allow(clippy::needless_range_loop)]
            for i in 0..num_queries {
                // Semantic component for this query
                let semantic: Option<(Vec<i64>, Vec<f32>)> = if has_queries {
                    let query = to_ndarray(&embedding_queries.unwrap()[i])?;
                    let idx = state_bg.get_index_for_read(&name_bg)?;
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
                        centroid_score_threshold: req
                            .params
                            .centroid_score_threshold
                            .unwrap_or_default(),
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
                            keyword_search_subset(&path_bg, &tq, fetch_k, sub)
                        } else {
                            text_search::search(&path_bg, &tq, fetch_k)
                        };
                        match result {
                            Ok(r) => Some((r.passage_ids, r.scores)),
                            Err(e) => {
                                tracing::warn!(trace_id = %trace_id_bg, index = %name_bg, error = %e, "search.keyword.failed");
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

                let metadata = fetch_metadata_for_docs(&path_bg, &document_ids)?;
                all_results.push(QueryResultResponse {
                    query_id: i,
                    document_ids,
                    scores,
                    metadata,
                });
            }

            Ok(all_results)
        },
    )
    .await
    .map_err(|e| ApiError::Internal(format!("Search task panicked: {}", e)))??;

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn media_ids_reads_photo_and_video_markers() {
        let meta = json!({
            "summary": "launch!\n📷 https://pbs.twimg.com/media/HPmOJZlbUAAOgVt.jpg | \
                        🎬 https://video.twimg.com/amplify_video/2105708422884909057/vid/avc1/480x270/x.mp4?tag=29 \
                        Quoting @deepseek_ai\n📷 https://pbs.twimg.com/media/HPmOJZlbUAAOgVt.jpg"
        });
        assert_eq!(
            media_ids(Some(&meta)),
            vec![
                "HPmOJZlbUAAOgVt".to_string(),
                "2105708422884909057".to_string()
            ]
        );
    }

    #[test]
    fn media_ids_skips_truncated_ids_and_plain_text() {
        let meta = json!({"summary": "cut off 📷 https://pbs.twimg.com/media/HPm and no media"});
        assert!(media_ids(Some(&meta)).is_empty());
        assert!(media_ids(None).is_empty());
    }

    #[test]
    fn quote_tweets_of_one_post_share_a_dedup_key() {
        let a = json!({"title": "A (@a)", "summary": "my take 📷 https://pbs.twimg.com/media/HPmOJZlbUAAOgVt.jpg"});
        let b = json!({"title": "B (@b)", "summary": "different take, same image 📷 https://pbs.twimg.com/media/HPmOJZlbUAAOgVt.jpg"});
        let ka = dedup_keys(Some(&a));
        assert!(dedup_keys(Some(&b)).iter().any(|k| ka.contains(k)));
    }

    #[test]
    fn media_items_dedup_photos_and_key_videos_by_id() {
        let meta = json!({
            "summary": "take\n📷 https://pbs.twimg.com/media/HPmOJZlbUAAOgVt.jpg\n🎬 https://pbs.twimg.com/amplify_video_thumb/2105708422884909057/img/a.jpg | https://video.twimg.com/amplify_video/2105708422884909057/vid/x.mp4\nQuoting @x\n📷 https://pbs.twimg.com/media/HPmOJZlbUAAOgVt.jpg"
        });
        let items = media_items(Some(&meta), "https://x.com/a/status/1");
        let keys: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            vec!["HPmOJZlbUAAOgVt", "2105708422884909057", "HPmOJZlbUAAOgVt"]
        );
        assert_eq!(items[1].1["kind"], "video");
        assert_eq!(items[1].1["tweet"], "https://x.com/a/status/1");
    }

    #[test]
    fn merge_sharers_dedups_by_slug() {
        let a = Some(json!([{"slug": "x"}, {"slug": "y"}]));
        let b = Some(json!([{"slug": "y"}, {"slug": "z"}]));
        let merged = merge_sharers(a, b).unwrap();
        let slugs: Vec<&str> = merged
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["slug"].as_str().unwrap())
            .collect();
        assert_eq!(slugs, vec!["x", "y", "z"]);
        assert!(merge_sharers(None, None).is_none());
    }
}
