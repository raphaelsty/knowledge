use indexmap::IndexMap;
use next_plaid::{filtering, IndexConfig, MmapIndex, UpdateConfig};
use next_plaid_onnx::Colbert;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;

const MODEL: &str = "lightonai/answerai-colbert-small-v1-onnx";
const INDEX_PATH: &str = "indices/knowledge";
const DATABASE_PATH: &str = "database/database.json";
const BATCH_SIZE: usize = 64;

#[derive(Deserialize)]
struct Doc {
    #[serde(default)]
    title: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "extra-tags")]
    extra_tags: Vec<String>,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    date: String,
}

fn build_document_text(doc: &Doc) -> String {
    let tags = doc.tags.join(" ");
    let extra_tags = doc.extra_tags.join(" ");
    let summary: String = doc.summary.chars().take(200).collect();
    format!("{} {} {} {}", doc.title, tags, extra_tags, summary)
        .trim()
        .to_string()
}

fn build_metadata(url: &str, doc: &Doc) -> Value {
    json!({
        "url": url,
        "title": doc.title,
        "summary": doc.summary,
        "date": doc.date,
        "tags": doc.tags.join(","),
        "extra_tags": doc.extra_tags.join(","),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read database
    let raw = fs::read_to_string(DATABASE_PATH)?;
    let data: IndexMap<String, Doc> = serde_json::from_str(&raw)?;
    println!("Database has {} documents.", data.len());

    // Build texts and metadata
    let mut texts: Vec<String> = Vec::with_capacity(data.len());
    let mut metadata: Vec<Value> = Vec::with_capacity(data.len());

    for (url, doc) in &data {
        let text = build_document_text(doc);
        if text.is_empty() {
            continue;
        }
        texts.push(text);
        metadata.push(build_metadata(url, doc));
    }

    println!("Encoding {} documents...", texts.len());

    // Load model
    let model = Colbert::builder(MODEL)
        .with_execution_provider(next_plaid_onnx::ExecutionProvider::CoreML)
        .with_quantized(true)
        .build()?;

    // Encode in batches
    let mut all_embeddings = Vec::with_capacity(texts.len());
    for (i, chunk) in texts.chunks(BATCH_SIZE).enumerate() {
        let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let batch_embeddings = model.encode_documents(&refs, Some(2))?;
        all_embeddings.extend(batch_embeddings);
        println!(
            "  Encoded batch {}/{} ({} docs)",
            i + 1,
            (texts.len() + BATCH_SIZE - 1) / BATCH_SIZE,
            all_embeddings.len()
        );
    }

    // Create index
    println!("Building index at {INDEX_PATH}...");
    fs::create_dir_all(INDEX_PATH)?;

    let index_config = IndexConfig {
        nbits: 2,
        ..Default::default()
    };
    let update_config = UpdateConfig::default();

    let (_index, doc_ids) =
        MmapIndex::update_or_create(&all_embeddings, INDEX_PATH, &index_config, &update_config)?;

    println!("Index created with {} documents.", doc_ids.len());

    // Store metadata
    println!("Storing metadata...");
    filtering::create(INDEX_PATH, &metadata, &doc_ids)?;

    println!("Done. Index written to {INDEX_PATH}/");
    Ok(())
}
