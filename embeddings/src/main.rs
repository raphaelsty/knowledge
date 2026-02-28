use indexmap::IndexMap;
use next_plaid::{filtering, IndexConfig, MmapIndex, UpdateConfig};
use next_plaid_onnx::Colbert;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;

const MODEL: &str = "models/answerai-colbert-small-v1-onnx";
const INDEX_PATH: &str = "multi-vector-database/knowledge";
const DATABASE_PATH: &str = "web/data/database.json";
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

/// Load documents from JSON file (original path).
fn load_from_json() -> Result<IndexMap<String, Doc>, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(DATABASE_PATH)?;
    let data: IndexMap<String, Doc> = serde_json::from_str(&raw)?;
    Ok(data)
}

/// Load documents from PostgreSQL when the `postgres` feature is enabled.
#[cfg(feature = "postgres")]
async fn load_from_pg(
    database_url: &str,
) -> Result<IndexMap<String, Doc>, Box<dyn std::error::Error>> {
    use sqlx::postgres::PgPoolOptions;
    use sqlx::Row;

    let pool = PgPoolOptions::new()
        .max_connections(2)
        .connect(database_url)
        .await?;

    let rows = sqlx::query(
        "SELECT url, title, summary, COALESCE(date::text, '') as date, tags, extra_tags FROM documents"
    )
    .fetch_all(&pool)
    .await?;

    let mut data = IndexMap::new();
    for row in rows {
        let url: String = row.get("url");
        let tags: Vec<String> = row.get("tags");
        let extra_tags: Vec<String> = row.get("extra_tags");
        let doc = Doc {
            title: row.get("title"),
            summary: row.get("summary"),
            date: row.get("date"),
            tags,
            extra_tags,
        };
        data.insert(url, doc);
    }

    pool.close().await;
    Ok(data)
}

fn run(data: IndexMap<String, Doc>) -> Result<(), Box<dyn std::error::Error>> {
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
            texts.len().div_ceil(BATCH_SIZE),
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

#[cfg(feature = "postgres")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database_url = std::env::var("DATABASE_URL").ok();

    let data = if let Some(ref url) = database_url {
        println!("Loading documents from PostgreSQL...");
        tokio::runtime::Runtime::new()?.block_on(load_from_pg(url))?
    } else {
        println!("Loading documents from {DATABASE_PATH}...");
        load_from_json()?
    };

    run(data)
}

#[cfg(not(feature = "postgres"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading documents from {DATABASE_PATH}...");
    let data = load_from_json()?;
    run(data)
}
