.PHONY: install install-dev sync run index serve web lint lint-fix check pre-commit pre-commit-install docker-build docker-run launch docker-stop clean install-api db api migrate up down events events-build ingest ingest-build deploy deploy-build deploy-down deploy-logs ssh remote-status remote-logs remote-restart remote-update redeploy extension

# Load .env if present
-include .env
export

INDEX_DIR        = multi-vector-database
MODEL            = models/answerai-colbert-small-v1-onnx
PORT             = 8080
WEB_PORT         = 3000
API_PORT         = 3001
EVENTS_PORT      = 3002
INGEST_PORT      = 3003
DATABASE_URL    ?= postgresql://knowledge:knowledge@localhost:5433/knowledge
NEXT_PLAID_API   = /Users/raphael/Documents/lighton/lategrep/target/release/next-plaid-api
ORT_DYLIB_PATH  ?= $(shell find ~/Library/Caches ~/.cache -name "libonnxruntime*.dylib" -print -quit 2>/dev/null)

# Remote connection
HETZNER_IP      ?= 65.21.111.133
SSH_KEY         ?= ~/.ssh/hetzner_knowledge
SSH_USER        ?= root
DOMAIN          ?= knowledge-web.org
POSTGRES_PASSWORD ?= knowledge
SSH_CMD          = ssh -i $(SSH_KEY) $(SSH_USER)@$(HETZNER_IP)

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

# Start the ingest API (real-time bookmark embedding + indexing)
ingest:
	DATABASE_URL=$(DATABASE_URL) MODEL_PATH=$(MODEL) INDEX_PATH=$(INDEX_DIR)/knowledge PORT=$(INGEST_PORT) ORT_DYLIB_PATH=$(ORT_DYLIB_PATH) cargo run --release --manifest-path ingest-api/Cargo.toml --features coreml

# Build the ingest API binary
ingest-build:
	cargo build --release --manifest-path ingest-api/Cargo.toml --features coreml

# Serve the frontend locally
web:
	python3 -m http.server $(WEB_PORT) --directory web

# ── Docker Compose (local dev) ──────────────────────────────

# Start all services via Docker Compose (local dev)
up:
	docker compose up -d

# Stop all services (local dev)
down:
	docker compose down

# ── Production Deploy (Hetzner VPS) ────────────────────────

# Build and start production stack (Caddy + all services)
deploy:
	docker compose -f docker-compose.prod.yml up -d

# Rebuild and restart production stack
deploy-build:
	docker compose -f docker-compose.prod.yml up -d --build

# Stop production stack
deploy-down:
	docker compose -f docker-compose.prod.yml down

# View production logs
deploy-logs:
	docker compose -f docker-compose.prod.yml logs -f

# ── Remote Server ──────────────────────────────────────────

# SSH into the server
ssh:
	$(SSH_CMD)

# Show container status on server
remote-status:
	$(SSH_CMD) "cd knowledge && docker compose -f docker-compose.prod.yml ps"

# Stream server logs
remote-logs:
	$(SSH_CMD) "cd knowledge && docker compose -f docker-compose.prod.yml logs -f --tail 100"

# Restart all services on server
remote-restart:
	$(SSH_CMD) "cd knowledge && docker compose -f docker-compose.prod.yml restart"

# Pull latest code and rebuild on server
remote-update:
	$(SSH_CMD) "cd knowledge && git pull && DOMAIN=$(DOMAIN) POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) docker compose -f docker-compose.prod.yml up -d --build"

# One-shot redeploy: push local changes, pull + rebuild on server, stream logs
redeploy:
	git push
	$(SSH_CMD) "cd knowledge && git pull && DOMAIN=$(DOMAIN) POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) docker compose -f docker-compose.prod.yml up -d --build && docker compose -f docker-compose.prod.yml logs -f --tail 20"

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

# ── Extension ─────────────────────────────────────────────────

# Package the browser extension into a zip for download
extension:
	cd extension && zip -r ../web/extension.zip . -x ".*" "__MACOSX/*"

# ── Cleanup ───────────────────────────────────────────────────

clean:
	rm -rf .venv __pycache__ .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
