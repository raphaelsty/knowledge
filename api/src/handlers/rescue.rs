//! Rescue placement — insert newly ingested documents into the folder tree.
//!
//! After the buffer scanner indexes new documents, this places each one into
//! the best-matching leaf tag using tag overlap and word similarity, then
//! saves the updated tree back to PostgreSQL.
//!
//! This mirrors `sources/taxonomy.py:rescue_unplaced_docs` but uses word
//! matching instead of model2vec embeddings — fast and dependency-free.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use super::buffer::BufferDocument;

/// Place newly ingested documents into the folder tree.
///
/// Loads the tree from PG, finds unplaced docs, inserts them into the
/// best-matching leaf tag, recounts, and saves back. Returns the count rescued.
pub async fn rescue_into_tree(
    pool: &sqlx::PgPool,
    docs: &[BufferDocument],
) -> Result<usize, String> {
    // Load folder_tree from PG
    let row: Option<(Value,)> =
        sqlx::query_as("SELECT data FROM generated_data WHERE key = 'folder_tree'")
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("load folder_tree: {e}"))?;

    let mut tree = match row {
        Some((data,)) => data,
        None => return Ok(0),
    };

    // Collect all URLs already placed in the tree
    let placed = collect_placed_urls(&tree);

    // Filter to unplaced docs with valid title
    let unplaced: Vec<&BufferDocument> = docs
        .iter()
        .filter(|d| !d.url.is_empty() && !d.title.is_empty() && !placed.contains(&d.url))
        .collect();

    if unplaced.is_empty() {
        return Ok(0);
    }

    // Collect unique leaf tag names from the tree
    let leaf_tags = collect_unique_leaf_tags(&tree);
    if leaf_tags.is_empty() {
        return Ok(0);
    }

    // Precompute: leaf tag name → set of normalized words
    let tag_words: Vec<HashSet<String>> = leaf_tags.iter().map(|t| normalize_to_words(t)).collect();

    // Precompute: normalized full tag name → index (for exact matching)
    let tag_norm_map: HashMap<String, usize> = leaf_tags
        .iter()
        .enumerate()
        .map(|(i, t)| (normalize_tag(t), i))
        .collect();

    // Find best leaf for each unplaced doc and insert
    let mut rescued = 0;
    for doc in &unplaced {
        if let Some(tag_name) = find_best_tag(doc, &leaf_tags, &tag_words, &tag_norm_map) {
            let compact = json!({"u": &doc.url, "t": &doc.title, "d": &doc.date});
            if insert_into_leaf(&mut tree, &tag_name, compact) {
                rescued += 1;
            }
        }
    }

    if rescued == 0 {
        return Ok(0);
    }

    // Recount document totals up the tree
    recount(&mut tree);

    // Save back to PG
    sqlx::query(
        "INSERT INTO generated_data (key, data, updated_at) VALUES ('folder_tree', $1, now())
         ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
    )
    .bind(&tree)
    .execute(pool)
    .await
    .map_err(|e| format!("save folder_tree: {e}"))?;

    Ok(rescued)
}

/// Find the best matching leaf tag for a document.
///
/// Strategy:
/// 1. Exact match — doc's tags/extra_tags contain a leaf tag name (normalized)
/// 2. Word overlap — Jaccard-like score between doc words and tag words
fn find_best_tag(
    doc: &BufferDocument,
    leaf_tags: &[String],
    tag_words: &[HashSet<String>],
    tag_norm_map: &HashMap<String, usize>,
) -> Option<String> {
    // 1. Try exact tag match (doc's tags contain a leaf tag name)
    for tag in doc.tags.iter().chain(doc.extra_tags.iter()) {
        let norm = normalize_tag(tag);
        if let Some(&idx) = tag_norm_map.get(&norm) {
            return Some(leaf_tags[idx].clone());
        }
    }

    // 2. Word overlap between doc words (title + tags) and leaf tag words
    let mut doc_words: HashSet<String> = HashSet::new();
    for tag in &doc.tags {
        doc_words.extend(normalize_to_words(tag));
    }
    for tag in &doc.extra_tags {
        doc_words.extend(normalize_to_words(tag));
    }
    doc_words.extend(normalize_to_words(&doc.title));

    let mut best_score = 0.0_f64;
    let mut best_idx = None;

    for (i, tw) in tag_words.iter().enumerate() {
        if tw.is_empty() {
            continue;
        }
        let overlap = tw.intersection(&doc_words).count() as f64;
        // Score: fraction of tag words found in doc (prefer full matches)
        let score = overlap / tw.len() as f64;
        if score > best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }

    best_idx.map(|i| leaf_tags[i].clone())
}

// ─── Tree traversal helpers ────────────────────────────────────────

/// Collect all document URLs already placed in the folder tree.
fn collect_placed_urls(node: &Value) -> HashSet<String> {
    let mut urls = HashSet::new();
    if let Some(tags) = node.get("t").and_then(|t| t.as_array()) {
        for entry in tags {
            if let Some(arr) = entry.as_array() {
                if let Some(docs) = arr.get(2).and_then(|d| d.as_array()) {
                    for doc in docs {
                        if let Some(url) = doc.get("u").and_then(|u| u.as_str()) {
                            urls.insert(url.to_string());
                        }
                    }
                }
            }
        }
    }
    if let Some(children) = node.get("c").and_then(|c| c.as_array()) {
        for child in children {
            urls.extend(collect_placed_urls(child));
        }
    }
    urls
}

/// Collect all unique leaf tag names from the tree.
fn collect_unique_leaf_tags(node: &Value) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut tags = Vec::new();
    collect_leaf_tags_inner(node, &mut seen, &mut tags);
    tags
}

fn collect_leaf_tags_inner(node: &Value, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    if let Some(tag_entries) = node.get("t").and_then(|t| t.as_array()) {
        for entry in tag_entries {
            if let Some(arr) = entry.as_array() {
                if let Some(name) = arr.first().and_then(|n| n.as_str()) {
                    if !name.is_empty() && seen.insert(name.to_string()) {
                        out.push(name.to_string());
                    }
                }
            }
        }
    }
    if let Some(children) = node.get("c").and_then(|c| c.as_array()) {
        for child in children {
            collect_leaf_tags_inner(child, seen, out);
        }
    }
}

/// Insert a compact doc into the first leaf tag entry matching `tag_name`.
///
/// Tree tag entries are arrays: `[tag_name, doc_count, [docs...]]`
fn insert_into_leaf(node: &mut Value, tag_name: &str, doc: Value) -> bool {
    if let Some(tags) = node.get_mut("t").and_then(|t| t.as_array_mut()) {
        for entry in tags.iter_mut() {
            if let Some(arr) = entry.as_array_mut() {
                if arr.len() >= 3 && arr[0].as_str() == Some(tag_name) {
                    if let Some(docs) = arr[2].as_array_mut() {
                        docs.push(doc);
                        arr[1] = json!(docs.len());
                    }
                    return true;
                }
            }
        }
    }
    if let Some(children) = node.get_mut("c").and_then(|c| c.as_array_mut()) {
        for child in children.iter_mut() {
            if insert_into_leaf(child, tag_name, doc.clone()) {
                return true;
            }
        }
    }
    false
}

/// Recompute `n` (document count) at every node after insertions.
fn recount(node: &mut Value) {
    // Recurse into children first
    if let Some(children) = node.get_mut("c").and_then(|c| c.as_array_mut()) {
        for child in children.iter_mut() {
            recount(child);
        }
    }
    // Sum: tag doc counts + child subtree counts
    let tag_count: u64 = node
        .get("t")
        .and_then(|t| t.as_array())
        .map(|tags| {
            tags.iter()
                .filter_map(|e| e.as_array().and_then(|a| a.get(1)).and_then(|n| n.as_u64()))
                .sum()
        })
        .unwrap_or(0);

    let child_count: u64 = node
        .get("c")
        .and_then(|c| c.as_array())
        .map(|children| {
            children
                .iter()
                .filter_map(|c| c.get("n").and_then(|n| n.as_u64()))
                .sum()
        })
        .unwrap_or(0);

    node["n"] = json!(tag_count + child_count);
}

// ─── Text normalization ────────────────────────────────────────────

/// Normalize text to a set of lowercase words (splitting on hyphens/underscores/whitespace).
fn normalize_to_words(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .replace(['-', '_'], " ")
        .split_whitespace()
        .filter(|w| w.len() > 1)
        .map(|w| w.to_string())
        .collect()
}

/// Normalize a tag name for exact comparison.
fn normalize_tag(tag: &str) -> String {
    tag.to_lowercase()
        .replace(['-', '_'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
