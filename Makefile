.PHONY: install install-dev sync run index serve web lint lint-fix check pre-commit pre-commit-install docker-build docker-run launch docker-stop clean install-api db api migrate up down events events-build

INDEX_DIR        = multi-vector-database
MODEL            = models/answerai-colbert-small-v1-onnx
PORT             = 8080
WEB_PORT         = 3000
API_PORT         = 3001
EVENTS_PORT      = 3002
DATABASE_URL    ?= postgresql://knowledge:knowledge@localhost:5433/knowledge
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

# ── Database ─────────────────────────────────────────────────

# Start PostgreSQL via Docker Compose
db:
	docker compose up -d postgres

# Run one-time JSON → PostgreSQL migration
migrate:
	DATABASE_URL=$(DATABASE_URL) uv run python scripts/migrate_json_to_pg.py

# ── Pipeline ──────────────────────────────────────────────────

# Fetch sources, generate tags, build tree, and index
run:
	DATABASE_URL=$(DATABASE_URL) uv run python run.py

# Build only the Rust search index (reads from PG when DATABASE_URL is set)
index:
	DATABASE_URL=$(DATABASE_URL) cargo run --release --manifest-path embeddings/Cargo.toml --features postgres

# ── Serve ─────────────────────────────────────────────────────

# Start the Rust search API (serves index + ONNX reranking model)
serve:
	ORT_DYLIB_PATH=$(ORT_DYLIB_PATH) $(NEXT_PLAID_API) --index-dir $(INDEX_DIR) --model $(MODEL) --int8 --port $(PORT)

# Start the FastAPI data server (serves tree/sources from PG)
api:
	DATABASE_URL=$(DATABASE_URL) uv run uvicorn sources.api:app --port $(API_PORT)

# Start the events analytics API (anonymous, RGPD-compliant)
events:
	DATABASE_URL=$(DATABASE_URL) PORT=$(EVENTS_PORT) cargo run --release --manifest-path events-api/Cargo.toml

# Build the events API binary
events-build:
	cargo build --release --manifest-path events-api/Cargo.toml

# Serve the frontend locally
web:
	python3 -m http.server $(WEB_PORT) --directory web

# ── Docker Compose ───────────────────────────────────────────

# Start all services via Docker Compose
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

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

# ── Docker (legacy single-container) ─────────────────────────

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
