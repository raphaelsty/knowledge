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

# Production (Hetzner VPS, managed by Dokploy)
# Deploys are GitHub-driven: `git push origin main` and Dokploy
# redeploys via its webhook. Dashboard at dokploy.knowledge-web.org.
make ssh             # SSH into the server (handy for ad-hoc shell work)
```

## Project layout

- `sources/` — Python package: data fetchers, tag tree builder, and pipeline client
  - `sources/utils/client.py` — main pipeline orchestrator (`from sources.utils import run_pipeline`)
  - `sources/database.py` — PostgreSQL abstraction layer
- `api/` — Unified Rust API: search + data + events + ingest in a single binary
- `web/` — static frontend
  - `index.html` + `search/page.js` (welcome page reuses the Search bundle)
  - `search.html` + `search/page.js` (plain JS)
  - `profile.html` + `profile/page.js` (plain JS)
  - shared: `api.js`, `config.js`, `colbert.worker.js` (WASM worker), CSS
- `indexes/` — generated ColBERT indices (gitignored, rebuilt per-deploy)
- `run.py` — iterates over personalities and runs the pipeline for each

## Deployment

- **Server:** Hetzner CX33 VPS (4 vCPU, 8GB RAM) at `65.21.111.133`
- **Domain:** https://knowledge-web.org · Dokploy UI at https://dokploy.knowledge-web.org
- **Stack:** Dokploy-managed Docker Compose (`docker-compose.dokploy.yml`) — Traefik terminates TLS, Caddy does path routing + serves the baked `web/` tree, knowledge-api + PostgreSQL behind it.
- **Deploy flow:** push to `origin/main` → Dokploy's GitHub webhook redeploys (~1-2 min). Manual redeploys and rollbacks happen in the Dokploy UI.
- **Local dev:** `docker-compose.yml` (no Caddy, just postgres + the API).
- **Secrets:** `.env` file (gitignored) — local dev. Production env vars live in Dokploy's project settings.

## Key details

- Python package is `sources`, not `knowledge_database` (renamed)
- The API is `knowledge-api` (Rust binary in `api/`, built in Docker or via `make serve`)
- Frontend API URLs auto-detect: `localhost` → hardcoded ports, production → relative paths (same origin via Caddy)
- All routes go through the single knowledge-api on port 8080: `/indices/*` (search), `/api/*` (data + ingest), `/events` + `/stats/*` (analytics)

## Prod daemons (systemd, host-side)

Four long-lived services run directly on the Hetzner host (NOT inside the Dokploy Docker stack). They're defined by unit files in `sources/*.service` and installed under `/etc/systemd/system/`. Code lives in the host's `/root/knowledge` git checkout — a `git pull` + `systemctl restart` is how you ship updates to them.

| Service | Source | Role |
|---|---|---|
| `knowledge-continuous` | `sources/knowledge-continuous.service` (wraps `sources/continuous_pipeline.sh`) | Long-running VIP-first loop: re-runs `run.py` for every personality, oldest-touched first. Pinned to CPU 0 with `CPUAffinity=0` so the runner can never starve the Rust API. This is the daemon that picks up new source fetchers (e.g. the recently added `huggingface.Activity`) — restart it after a code change to `sources/*` or `run.py`. |
| `knowledge-indexer` | `sources/knowledge-indexer.service` (wraps `sources/indexer_daemon.py`) | Detects broken ColBERT indices, backfills `indexed=FALSE` documents, owns the index lifecycle end-to-end (decoupled from the fetcher so `make run` is quota-bounded). Pinned to CPU 1 with `CPUQuota=50%` so it shares fairly with the Rust API on the second core. |
| `knowledge-categorize-daemon` | `sources/categorize_daemon.service` | Walks uncategorized `documents` newest-first, runs Potion static embeddings, writes 0–3 category slugs per doc into `document_category_assignments`. Niced to 19, CPU capped at 10%, memory capped at 384 MB. |
| `knowledge-clean-daemon` | `sources/clean_daemon.service` | Rewrites verbose `title` / `summary` into pedagogical `clean_title` / `clean_summary` via `gpt-4o-mini`. I/O-bound on the OpenAI API; niced to 19, CPU 20%, memory 256 MB. Only touches VIP documents. |

Operate via the standard Makefile shortcuts (`make ssh` then `systemctl status/start/stop/restart <name>`) or via the dedicated `clean-daemon-*` / `categorize-daemon-*` targets in the Makefile.
