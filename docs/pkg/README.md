<div align="center">
    <h1>pylate-rs</h1>
</div>

<p align="center"><img width=500 src="https://github.com/lightonai/pylate-rs/blob/01ee9895d83d6bc0a52ba826f6b634d33be479ce/docs/logo.jpg"/></p>

<div align="center">
    <a href="https://lightonai.github.io/pylate-rs/"><img src="https://img.shields.io/badge/blog-%23000000.svg?style=for-the-badge&logoColor=white" alt="blog"></a>
    <a href="https://docs.rs/pylate-rs/latest/pylate_rs/all.html"><img src="https://img.shields.io/badge/crate-%23000000.svg?style=for-the-badge&logoColor=white" alt="crate"></a>

</div>

<div align="center">
<b>Efficient Inference for PyLate</b>
</div>

&nbsp;

## ⭐️ Overview

**pylate-rs** is a high-performance inference engine for [PyLate](https://github.com/lightonai/pylate) models, meticulously crafted in Rust for optimal speed and efficiency.

While model training is handled by PyLate, which supports a variety of late interaction models, `pylate-rs` is engineered to execute these models at speeds.

- **Accelerated Performance**: Experience significantly faster model loading and rapid cold starts, making it ideal for serverless environments and low-latency applications.

- **Lightweight Design**: Built on the [Candle](https://github.com/huggingface/candle) ML framework, `pylate-rs` maintains a minimal footprint suitable for resource-constrained systems like serverless functions and edge computing.

- **Broad Hardware Support**: Optimized for diverse hardware, with dedicated builds for standard CPUs, Intel (MKL), Apple Silicon (Accelerate & Metal), and NVIDIA GPUs (CUDA).

- **Cross-Platform Integration**: Seamlessly integrate `pylate-rs` into your projects with bindings for Python, Rust, and JavaScript/WebAssembly.

For a complete, high-performance multi-vector search pipeline, pair `pylate-rs` with its companion library, [**FastPlaid**](https://github.com/lightonai/fast-plaid), at inference time.

Explore our [**WebAssembly live demo**](https://lightonai.github.io/pylate-rs/).

&nbsp;

## 💻 Installation

Install the version of `pylate-rs` that matches your hardware for optimal performance.

### Python

| Target Hardware          | Installation Command               |
| :----------------------- | :--------------------------------- |
| **Standard CPU**         | `pip install pylate-rs`            |
| **Apple CPU** (macOS)    | `pip install pylate-rs-accelerate` |
| **Intel CPU** (MKL)      | `pip install pylate-rs-mkl`        |
| **Apple GPU** (M1/M2/M3) | `pip install pylate-rs-metal`      |

### Python GPU support

To install pylate-rs with GPU support, please built it from source using the following command:

```sh
pip install git+https://github.com/lightonai/pylate-rs.git
```

or by cloning the repository and installing it locally:

```sh
git clone https://github.com/lightonai/pylate-rs.git
cd pylate-rs
pip install .
```

Any help to pre-build and disribute the CUDA wheels would be greatly appreciated.

&nbsp;

### Rust

Add `pylate-rs` to your `Cargo.toml` by enabling the feature flag that corresponds to your backend.

| Feature      | Target Hardware          | Installation Command                        |
| :----------- | :----------------------- | :------------------------------------------ |
| _(default)_  | **Standard CPU**         | `cargo add pylate-rs`                       |
| `accelerate` | **Apple CPU** (macOS)    | `cargo add pylate-rs --features accelerate` |
| `mkl`        | **Intel CPU** (MKL)      | `cargo add pylate-rs --features mkl`        |
| `metal`      | **Apple GPU** (M1/M2/M3) | `cargo add pylate-rs --features metal`      |
| `cuda`       | **NVIDIA GPU** (CUDA)    | `cargo add pylate-rs --features cuda`       |

&nbsp;

## ⚡️ Quick Start

### Python

Get started in just a few lines of Python.

```python
from pylate_rs import models

# Initialize the model for your target device ("cpu", "cuda", or "mps")
model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    device="cuda"
)

# Encode queries and documents
queries_embeddings = model.encode(
    sentences=["What is the capital of France?", "How big is the sun?"],
    is_query=True
)

documents_embeddings = model.encode(
    sentences=["Paris is the capital of France.", "The sun is a star."],
    is_query=False
)

# Calculate similarity scores
similarities = model.similarity(queries_embeddings, documents_embeddings)

print(f"Similarity scores:\n{similarities}")

# Use hierarchical pooling to reduce document embedding size and speed up downstream tasks
pooled_documents_embeddings = model.encode(
    sentences=["Paris is the capital of France.", "The sun is a star."],
    is_query=False,
    pool_factor=2, # Halves the number of token embeddings
)

similarities_pooled = model.similarity(queries_embeddings, pooled_documents_embeddings)

print(f"Similarity scores with pooling:\n{similarities_pooled}")
```

&nbsp;

### Rust

```rust
use anyhow::Result;
use candle_core::Device;
use pylate_rs::{hierarchical_pooling, ColBERT};

fn main() -> Result<()> {
    // Set the device (e.g., Cpu, Cuda, Metal)
    let device = Device::Cpu;

    // Initialize the model
    let mut model: ColBERT = ColBERT::from("lightonai/GTE-ModernColBERT-v1")
        .with_device(device)
        .try_into()?;

    // Encode queries and documents
    let queries = vec!["What is the capital of France?".to_string()];
    let documents = vec!["Paris is the capital of France.".to_string()];

    let query_embeddings = model.encode(&queries, true)?;
    let document_embeddings = model.encode(&documents, false)?;

    // Calculate similarity
    let similarities = model.similarity(&query_embeddings, &document_embeddings)?;
    println!("Similarity score: {}", similarities.data[0][0]);

    // Use hierarchical pooling
    let pooled_document_embeddings = hierarchical_pooling(&document_embeddings, 2)?;
    let pooled_similarities = model.similarity(&query_embeddings, &pooled_document_embeddings)?;
    println!("Similarity score after hierarchical pooling: {}", pooled_similarities.data[0][0]);

    Ok(())
}
```

&nbsp;

## 📊 Benchmarks

```python
Device    backend        Queries per seconds        Documents per seconds        Model loading time
cpu       PyLate         350.10                     32.16                        2.06
cpu       pylate-rs      386.21 (+10%)              42.15 (+31%)                 0.07 (-97%)

cuda      PyLate         2236.48                    882.66                       3.62
cuda      pylate-rs      4046.88 (+81%)             976.23 (+11%)                1.95 (-46%)

mps       PyLate         580.81                     103.10                       1.95
mps       pylate-rs      291.71 (-50%)              23.26 (-77%)                 0.08 (-96%)
```

Benchmark were run with Python. `pylate-rs` provide significant performance improvement, especially in scenarios requiring fast startup times. While on a Mac it takes up to 5 seconds to load a model with the Transformers backend and encode a single query, `pylate-rs` achieves this in just 0.11 seconds, making it ideal for low-latency applications. Don't expect `pylate-rs` to be much faster than `PyLate` to encode a lot of content at the same time as PyTorch is heavily optimized.

&nbsp;

## 📦 Using Custom Models

`pylate-rs` is compatible with any model saved in the PyLate format, whether from the Hugging Face Hub or a local directory. PyLate itself is compatible with a wide range of models, including those from Sentence Transformers, Hugging Face Transformers, and custom models. So before using `pylate-rs`, ensure your model is saved in the PyLate format. You can easily convert and upload your own models using PyLate.

Pushing a model to the Hugging Face Hub in PyLate format is straightforward. Here’s how you can do it:

```bash
pip install pylate
```

Then, you can use the following Python code snippet to push your model:

```python
from pylate import models

# Load your model
model = models.ColBERT(model_name_or_path="your-base-model-on-hf")

# Push in PyLate format
model.push_to_hub(
    repo_id="YourUsername/YourModelName",
    private=False,
    token="YOUR_HUGGINGFACE_TOKEN",
)
```

If you want to save a model in PyLate format locally, you can do so with the following code snippet:

```python
from pylate import models

# Load your model
model = models.ColBERT(model_name_or_path="your-base-model-on-hf")

# Save in PyLate format
model.save_pretrained("path/to/save/GTE-ModernColBERT-v1-pylate")
```

An existing set of models compatible with `pylate-rs` is available on the Hugging Face Hub under the [**LightOn**](https://huggingface.co/collections/lightonai/pylate-6862b571946fe88330d65264) namespace.

&nbsp;

## Retrieval pipeline

```bash
pip install pylate-rs fast-plaid
```

Here is a sample code for running ColBERT with pylate-rs and fast-plaid.

```python
import torch
from fast_plaid import search
from pylate_rs import models

model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    device="cpu", # mps or cuda
)

documents = [
    "1st Arrondissement: Louvre, Tuileries Garden, Palais Royal, historic, tourist.",
    "2nd Arrondissement: Bourse, financial, covered passages, Sentier, business.",
    "3rd Arrondissement: Marais, Musée Picasso, galleries, trendy, historic.",
    "4th Arrondissement: Notre-Dame, Marais, Hôtel de Ville, LGBTQ+.",
    "5th Arrondissement: Latin Quarter, Sorbonne, Panthéon, student, intellectual.",
    "6th Arrondissement: Saint-Germain-des-Prés, Luxembourg Gardens, chic, artistic, cafés.",
    "7th Arrondissement: Eiffel Tower, Musée d'Orsay, Les Invalides, affluent, prestigious.",
    "8th Arrondissement: Champs-Élysées, Arc de Triomphe, luxury, shopping, Élysée.",
    "9th Arrondissement: Palais Garnier, department stores, shopping, theaters.",
    "10th Arrondissement: Gare du Nord, Gare de l'Est, Canal Saint-Martin.",
    "11th Arrondissement: Bastille, nightlife, Oberkampf, revolutionary, hip.",
    "12th Arrondissement: Bois de Vincennes, Opéra Bastille, Bercy, residential.",
    "13th Arrondissement: Chinatown, Bibliothèque Nationale, modern, diverse, street-art.",
    "14th Arrondissement: Montparnasse, Catacombs, residential, artistic, quiet.",
    "15th Arrondissement: Residential, family, populous, Parc André Citroën.",
    "16th Arrondissement: Trocadéro, Bois de Boulogne, affluent, elegant, embassies.",
    "17th Arrondissement: Diverse, Palais des Congrès, residential, Batignolles.",
    "18th Arrondissement: Montmartre, Sacré-Cœur, Moulin Rouge, artistic, historic.",
    "19th Arrondissement: Parc de la Villette, Cité des Sciences, canals, diverse.",
    "20th Arrondissement: Père Lachaise, Belleville, cosmopolitan, artistic, historic.",
]

# Encoding documents
documents_embeddings = model.encode(
    sentences=documents,
    is_query=False,
    pool_factor=2, # Let's divide the number of embeddings by 2.
)

# Creating the FastPlaid index
fast_plaid = search.FastPlaid(index="index")


fast_plaid.create(
    documents_embeddings=[torch.tensor(embedding) for embedding in documents_embeddings]
)
```

We can then load the existing index and search for the most relevant documents:

```python
import torch
from fast_plaid import search
from pylate_rs import models

fast_plaid = search.FastPlaid(index="index")

queries = [
    "arrondissement with the Eiffel Tower and Musée d'Orsay",
    "Latin Quarter and Sorbonne University",
    "arrondissement with Sacré-Cœur and Moulin Rouge",
    "arrondissement with the Louvre and Tuileries Garden",
    "arrondissement with Notre-Dame Cathedral and the Marais",
]

queries_embeddings = model.encode(
    sentences=queries,
    is_query=True,
)

scores = fast_plaid.search(
    queries_embeddings=torch.tensor(queries_embeddings),
    top_k=3,
)

print(scores)
```

## 📝 Citation

If you use `pylate-rs` in your research or project, please cite it as follows:

```bibtex
@misc{PyLate,
  title={PyLate: Flexible Training and Retrieval for Late Interaction Models},
  author={Chaffin, Antoine and Sourty, Raphaël},
  url={https://github.com/lightonai/pylate},
  year={2024}
}
```

&nbsp;

## WebAssembly

For JavaScript and TypeScript projects, install the WASM package from npm.

```bash
npm install pylate-rs
```

Load the model by fetching the required files from a local path or the Hugging Face Hub.

```javascript
import { ColBERT } from "pylate-rs";

const REQUIRED_FILES = [
  "tokenizer.json",
  "model.safetensors",
  "config.json",
  "config_sentence_transformers.json",
  "1_Dense/model.safetensors",
  "1_Dense/config.json",
  "special_tokens_map.json",
];

async function loadModel(modelRepo) {
  const fetchAllFiles = async (basePath) => {
    const responses = await Promise.all(
      REQUIRED_FILES.map((file) => fetch(`${basePath}/${file}`))
    );
    for (const response of responses) {
      if (!response.ok) throw new Error(`File not found: ${response.url}`);
    }
    return Promise.all(
      responses.map((res) => res.arrayBuffer().then((b) => new Uint8Array(b)))
    );
  };

  try {
    let modelFiles;
    try {
      // Attempt to load from a local `models` directory first
      modelFiles = await fetchAllFiles(`models/${modelRepo}`);
    } catch (e) {
      console.warn(
        `Local model not found, falling back to Hugging Face Hub.`,
        e
      );
      // Fallback to fetching directly from the Hugging Face Hub
      modelFiles = await fetchAllFiles(
        `https://huggingface.co/${modelRepo}/resolve/main`
      );
    }

    const [
      tokenizer,
      model,
      config,
      stConfig,
      dense,
      denseConfig,
      tokensConfig,
    ] = modelFiles;

    // Instantiate the model with the loaded files
    const colbertModel = new ColBERT(
      model,
      dense,
      tokenizer,
      config,
      stConfig,
      denseConfig,
      tokensConfig,
      32
    );

    // You can now use `colbertModel` for encoding
    console.log("Model loaded successfully!");
    return colbertModel;
  } catch (error) {
    console.error("Model Loading Error:", error);
  }
}
```
