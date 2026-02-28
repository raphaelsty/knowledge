.PHONY: install install-dev sync run index serve web lint lint-fix check pre-commit pre-commit-install docker-build docker-run launch docker-stop clean install-api

INDEX_DIR        = indices
MODEL            = lightonai/answerai-colbert-small-v1-onnx
PORT             = 8080
WEB_PORT         = 3000
NEXT_PLAID_API   = /Users/raphael/Documents/lighton/lategrep/target/release/next-plaid-api
ORT_DYLIB_PATH  ?= $(shell find ~/Library/Caches ~/.cache -name "libonnxruntime*.dylib" -print -quit 2>/dev/null)

# ── Dependencies ──────────────────────────────────────────────

# Install Python prod dependencies
install:
	uv sync --no-dev

# Install Python dev dependencies (ruff, mypy, pre-commit)
install-dev:
	uv sync --all-extras

sync: install-dev

# Install the Rust API server binary (with ONNX model support)
install-api:
	cargo install next-plaid-api --features model

# ── Pipeline ──────────────────────────────────────────────────

# Fetch sources, generate tags, build tree, and index
run:
	uv run python run.py

# Build only the Rust search index
index:
	cargo run --release --manifest-path indexer/Cargo.toml

# ── Serve ─────────────────────────────────────────────────────

# Start the Rust API (serves index + ONNX reranking model)
serve:
	ORT_DYLIB_PATH=$(ORT_DYLIB_PATH) $(NEXT_PLAID_API) --index-dir $(INDEX_DIR) --model $(MODEL) --int8 --port $(PORT)

# Serve the frontend locally
web:
	python3 -m http.server $(WEB_PORT) --directory docs

# ── Lint ──────────────────────────────────────────────────────

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy . --ignore-missing-imports

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

check: lint

# ── Pre-commit ────────────────────────────────────────────────

pre-commit:
	uv run pre-commit run --all-files

pre-commit-install:
	uv run pre-commit install

# ── Docker ────────────────────────────────────────────────────

docker-build:
	docker build -t knowledge .

docker-run:
	docker run -d --add-host host.docker.internal:host-gateway --name run_knowledge -p $(PORT):$(PORT) knowledge

launch: docker-build docker-run

docker-stop:
	docker stop run_knowledge || true
	docker rm run_knowledge || true

# ── Cleanup ───────────────────────────────────────────────────

clean:
	rm -rf .venv __pycache__ .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
