# Knowledge

Personal knowledge base: fetches bookmarks from GitHub, HackerNews, Zotero, HuggingFace, and X/Twitter, then serves them via a ColBERT search engine.

## Use the Makefile

Every operation goes through `make`. Do not run raw commands. Settings are loaded from `.env`.

```
# Local development
make install      # install prod dependencies
make install-dev  # install with dev tools (ruff, mypy, pre-commit)
make run          # fetch sources → generate tags → build tree → index
make index        # rebuild only the Rust search index
make serve        # start the unified API on :8080 (search + data + events + ingest)
make web          # serve the frontend on :3000
make up           # start all services via Docker Compose (local dev)
make down         # stop all local services
make lint         # ruff + mypy
make lint-fix     # auto-fix lint issues
make clean        # wipe caches and venv

# Production (Hetzner VPS)
make ssh             # SSH into the server
make remote-status   # show container status on server
make remote-logs     # stream server logs
make remote-restart  # restart all services on server
make remote-update   # pull latest code + rebuild on server
make deploy          # start production stack (run on server)
make deploy-build    # rebuild + start production stack (run on server)
make deploy-down     # stop production stack (run on server)
make deploy-logs     # view production logs (run on server)
```

## Project layout

- `sources/` — Python package: data fetchers, tag tree builder, and pipeline client
  - `sources/client.py` — main pipeline orchestrator (`from sources.client import main`)
  - `sources/taxonomy.py` — builds the folder tree from tag triples
  - `sources/database.py` — PostgreSQL abstraction layer
- `api/` — Unified Rust API: search + data + events + ingest in a single binary
- `embeddings/` — Rust binary that builds the ColBERT index
- `web/` — static frontend (index.html, app.jsx, dashboard.html, dashboard.jsx, CSS, WASM worker)
- `web/data/` — generated JSON files (database.json, sources.json, folder_tree.json, tree.json)
- `multi-vector-database/` — generated ColBERT index (committed for deploy)
- `run.py` — thin entry point (calls `sources.client.main()`)

## Deployment

- **Server:** Hetzner CX33 VPS (4 vCPU, 8GB RAM) at `65.21.111.133`
- **Domain:** https://knowledge-web.org
- **Stack:** Docker Compose with Caddy (reverse proxy + HTTPS), PostgreSQL, knowledge-api
- **Config:** `docker-compose.prod.yml` (production), `docker-compose.yml` (local dev), `Caddyfile` (routing)
- **Secrets:** `.env` file (gitignored) — contains SSH key path, server IP, domain, Postgres password

## Key details

- Python package is `sources`, not `knowledge_database` (renamed)
- The API is `knowledge-api` (Rust binary in `api/`, built in Docker or via `make serve`)
- Frontend API URLs auto-detect: `localhost` → hardcoded ports, production → relative paths (same origin via Caddy)
- The embeddings crate requires `libonnxruntime` — if it fails locally, the existing `multi-vector-database/` still works
- All routes go through the single knowledge-api on port 8080: `/indices/*` (search), `/api/*` (data + ingest), `/events` + `/stats/*` (analytics)
