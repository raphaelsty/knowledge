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

## Prod daemons (Dokploy compose services)

Four long-running Python daemons live in the same Dokploy stack as the API, defined in `docker-compose.dokploy.yml`. They share `Dockerfile.daemons` (Python 3.11 + uv + project deps) — one image, four entry points, only `command:` and `deploy.resources.limits` differ per service.

Updates ship the same way as the API: `git push origin main` → Dokploy webhook redeploys the affected services. **Code changes to `sources/*` or `run.py` are picked up automatically** because every daemon container's image is rebuilt on push (no manual `systemctl restart`).

| Service | Command | CPU / memory cap | Role |
|---|---|---|---|
| `knowledge-continuous` | `bash sources/continuous_pipeline.sh` | 1.0 vCPU / 2 G | VIP-first per-user pipeline runner: walks personalities oldest-touched first, invokes `run.py <slug>` for each. The daemon that picks up new source fetchers (e.g. `huggingface.Activity`). |
| `knowledge-indexer` | `python -m sources.indexer_daemon` | 0.5 vCPU / 2 G | Detects broken ColBERT indices, backfills `indexed=FALSE` documents, owns the index lifecycle. Talks to the API on the internal docker network (`http://knowledge-api:8080`). |
| `knowledge-categorize-daemon` | `python -m sources.utils.categorize_daemon` | 0.10 vCPU / 384 M | Assigns 0–3 category slugs per doc via Potion static embeddings, newest-first. |
| `knowledge-clean-daemon` | `python -m sources.utils.clean_daemon` | 0.20 vCPU / 256 M | Rewrites verbose `title` / `summary` into pedagogical `clean_title` / `clean_summary` via OpenAI. Default model is `gpt-4o-mini`; override with `OPENAI_CLEAN_MODEL` env (e.g. `gpt-4.1-nano` for cheaper). Requires `OPENAI_API_KEY`. VIP documents only. |

Operate via Docker on the host:
```
ssh -i ~/.ssh/hetzner_knowledge root@65.21.111.133
docker logs -f knowledge-prod-gjqqg2-knowledge-<name>-1
docker restart knowledge-prod-gjqqg2-knowledge-<name>-1
```

The continuous-pipeline state files (history, pid, rotation cursor) live in the named volume `knowledge_daemon_logs` so a `docker compose down` doesn't reset the 12 h per-user cooldown.

**Legacy:** the systemd unit files in `sources/*.service` are kept in the repo for rollback reference. They were stopped + disabled on the host during the cutover; do not re-enable them or both copies will compete for the same `pipeline_runs` rows.
