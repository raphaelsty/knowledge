.PHONY: install install-dev sync run index index-all serve web lint lint-fix check pre-commit pre-commit-install docker-build docker-run launch docker-stop clean install-api api-build db db-browse db-backup db-backup-if-stale up down ssh dev dev-stop delete purge hn-frontpage daily all-status all-rebuild load-test prod-db-dump prod-db-dump-if-stale prod-db-restore prod-db-sync

# Load .env if present
-include .env
export

INDEX_DIR        = indexes
MODEL            = models/answerai-colbert-small-v1-onnx
PORT             = 8080
WEB_PORT         = 3001
API_PORT         = 3001
DATABASE_URL    ?= postgresql://knowledge:knowledge@localhost:5433/knowledge
KNOWLEDGE_API    = api
ORT_DYLIB_PATH  ?= $(shell find ~/.cache ~/Library/Caches -name "libonnxruntime*.dylib" -print -quit 2>/dev/null)

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
install-dev: pre-commit-install
	uv sync --all-extras

sync: install-dev

# Build the unified Rust API (search + data + events + ingest)
install-api:
	cargo build --release --manifest-path $(KNOWLEDGE_API)/Cargo.toml --features model

# ── Database ─────────────────────────────────────────────────

# Start PostgreSQL via Docker Compose
db:
	docker compose up -d postgres

# Browse the database in a web UI (pgweb, opens on :8081)
db-browse:
	pgweb --url $(DATABASE_URL) --listen 8081

# Dump the entire knowledge database to ~/Desktop/knowledge/ as a
# gzipped SQL file timestamped to the current minute. Uses
# `pg_dump` inside the running postgres container so we don't need
# pg-tools installed on the host. Custom format (`-Fc`) gives a
# compressed, restorable archive — restore with:
#
#     pg_restore -d knowledge --clean --if-exists <dump.dump>
#
# The output directory is created on first run.
DB_BACKUP_DIR ?= $(HOME)/Desktop/knowledge
db-backup:
	@mkdir -p $(DB_BACKUP_DIR)
	@stamp=$$(date +%Y%m%d_%H%M%S); \
	  out="$(DB_BACKUP_DIR)/knowledge_$${stamp}.dump"; \
	  echo "Dumping to $$out ..."; \
	  docker compose exec -T postgres pg_dump -U knowledge -d knowledge -Fc > "$$out" \
	    && size=$$(du -sh "$$out" | cut -f1) \
	    && echo "✓ wrote $$out ($$size)"

# Guard target: produces a dump only if there isn't already one from
# today (local date). Used as a `make run` prerequisite so a long
# pipeline run is always preceded by a fresh on-disk snapshot — but
# we don't pay the dump cost more than once per calendar day.
# Filename pattern is `knowledge_YYYYMMDD_HHMMSS.dump`, so a glob on
# today's prefix tells us whether anything exists.
db-backup-if-stale:
	@today=$$(date +%Y%m%d); \
	  if ls -1 $(DB_BACKUP_DIR)/knowledge_$${today}_*.dump >/dev/null 2>&1; then \
	    echo "DB backup: today's dump exists — skipping."; \
	  else \
	    echo "DB backup: no dump from today — creating..."; \
	    $(MAKE) --no-print-directory db-backup; \
	  fi

# ── Pipeline ──────────────────────────────────────────────────

# Fetch sources, generate tags, build tree, and index via API.
#
#   make run                                # everyone, all sources
#   make run SLUG=max-halford               # one personality, all sources
#   make run SOURCE=twitter                 # everyone, twitter only
#   make run SLUG=max-halford SOURCE=reddit # one personality, reddit only
#   make run SOURCE=twitter,reddit          # everyone, two specific sources
#
# `SLUG=` is a positional arg; `SOURCE=` becomes `--source <value>` so
# argparse stays happy. The pipeline still runs cleanup → tagging →
# indexing on whatever the filtered fetchers produced, so any partial
# run leaves the user's library coherent.
#
# `TWIKIT=1` swaps the Twitter source from api.twitterapi.io to
# cookie-authenticated twikit (Safari cookies on this machine). The
# twikit path pulls each VIP's public tweets + likes through one
# shared session, so we can backfill the whole roster without
# spending twitterapi.io credit. Combine with `SOURCE=twitter` for
# a Twitter-only pass:
#
#   make run TWIKIT=1                            # everyone, all sources
#   make run TWIKIT=1 SOURCE=twitter             # everyone, twitter only via twikit
#   make run TWIKIT=1 SLUG=tony-wu SOURCE=twitter  # one VIP, twitter only via twikit
run: db-backup-if-stale
	DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) \
	uv run python run.py $(SLUG) $(if $(SOURCE),--source $(SOURCE),) $(if $(JOBS),--jobs $(JOBS),) $(if $(TWIKIT),--twikit,)

# Fill twitter_followers / github_followers / citations for every user
# whose column is still NULL. Idempotent — reruns skip already-populated
# rows. Picks up TWITTERAPIIO_API_KEY / GITHUB_TOKEN from .env (exported
# by this Makefile).
social-counts:
	DATABASE_URL=$(DATABASE_URL) uv run python -m sources.utils.popularity_cli

# Mechanical source-key backfill: derives `youtube_search` from display
# name, `websites` from links.website (probes /sitemap.xml, /feed.xml,
# /atom.xml, /rss.xml, /index.xml, /feed for an XML response),
# `github_repos` + `github_gists` from links.github, and `huggingface`
# from the github handle (only when the HF profile passes a non-squat
# check). Idempotent — never overwrites existing keys. Add `--dry` to
# preview without writing.
backfill-sources:
	DATABASE_URL=$(DATABASE_URL) uv run python -m sources.utils.backfill_all_sources

# ── Daily cron — run these ONCE per day to keep the app fresh ─
#
# This is the contract for the production scheduler. Anything not
# listed here is opt-in or driven by user action.
#
#   1. make run             — full pipeline for every personality
#                             (fetch all sources, clean, tag, index).
#                             Long-running. Independent of (2).
#   2. make hn-frontpage    — snapshot the HN front page and refresh
#                             every user's per-feed HN picks.
#                             Fast (~seconds × #users). Only needs the
#                             API up — does NOT need (1) to have run.
#
# Run order doesn't matter; they share no state. In cron, schedule
# them at separate times so the API isn't competing with the pipeline
# embedder. `make daily` runs both sequentially for convenience.

daily: run hn-frontpage

# Snapshot the current HackerNews front page and refresh every user's
# personalised picks. The picks are surfaced in the feed (NOT the
# personal page) until the next run replaces them. Bookmarking a pick
# copies it into the user's `documents` with indexed=FALSE so the
# next `make run` indexes it; from there it behaves like any other
# bookmark.
#
#   make hn-frontpage                              # everyone
#   make hn-frontpage SLUG=raphael-sourty          # one user (debug)
#   make hn-frontpage DRY=1                        # fetch + score, no writes
#   make hn-frontpage TOP_PER_USER=12 THRESHOLD=7  # tune relevance
#   make hn-frontpage NO_SNAPSHOT=1                # re-score against latest run
hn-frontpage:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) \
	  uv run python scripts/hn_frontpage.py \
	    $(if $(SLUG),--slug $(SLUG),) \
	    $(if $(TOP_PER_USER),--top-per-user $(TOP_PER_USER),) \
	    $(if $(THRESHOLD),--threshold $(THRESHOLD),) \
	    $(if $(TOP),--top $(TOP),) \
	    $(if $(LIMIT),--limit $(LIMIT),) \
	    $(if $(DRY),--dry,) \
	    $(if $(NO_SNAPSHOT),--no-snapshot,) \
	    $(if $(DEBUG),--debug,)

# ── Serve ─────────────────────────────────────────────────────

# Start the unified Rust API (search + data + events + ingest).
# Locally we reuse the _DEV OAuth credentials (separate GitHub OAuth App
# for localhost) so the prod GITHUB_CLIENT_ID/SECRET only ship with the
# production container. The leading `@` on every line suppresses command
# echo so the client secret never hits the terminal.
serve:
	@echo "serve: knowledge-api → :$(PORT)  (OAuth callback /auth/github/callback)"
	@DATABASE_URL='$(DATABASE_URL)' \
	ORT_DYLIB_PATH='$(ORT_DYLIB_PATH)' \
	GITHUB_CLIENT_ID='$(GITHUB_CLIENT_ID_DEV)' \
	GITHUB_CLIENT_SECRET='$(GITHUB_CLIENT_SECRET_DEV)' \
	HN_ENCRYPTION_KEY='$(HN_ENCRYPTION_KEY)' \
	STACKOVERFLOW_CLIENT_ID='$(STACKOVERFLOW_DEV_CLIENT_ID)' \
	STACKOVERFLOW_CLIENT_SECRET='$(STACKOVERFLOW_DEV_CLIENT_SECRET)' \
	STACKOVERFLOW_KEY='$(STACKOVERFLOW_DEV_KEY)' \
	STACKOVERFLOW_OAUTH_REDIRECT_URL='http://localhost:$(PORT)/auth/stackoverflow/callback' \
	OAUTH_REDIRECT_URL='http://localhost:$(PORT)/auth/github/callback' \
	OAUTH_POST_LOGIN_URL='http://localhost:$(WEB_PORT)/' \
	cargo run --release --manifest-path $(KNOWLEDGE_API)/Cargo.toml --features "accelerate,model" -- --index-dir $(INDEX_DIR) --model $(MODEL) --int8 --port $(PORT)

# Build the unified API binary
api-build:
	cargo build --release --manifest-path $(KNOWLEDGE_API)/Cargo.toml --features "accelerate,model"

# Serve the frontend locally (uv run ensures psycopg is available for /admin)
web:
	DATABASE_URL=$(DATABASE_URL) uv run python3 web/serve.py $(WEB_PORT)

# Build the cross-personality `__all__` index. Loads every doc owned by
# a VIP user from PG, formats it the same way per-user indexing does,
# and pushes it to the unified API. Idempotent — pre-purges existing
# chunks for each URL before re-pushing so reruns don't accumulate
# duplicates. Run AFTER per-user indices are healthy (PG `documents` is
# the source of truth).
index-all:
	@DATABASE_URL=$(DATABASE_URL) ADMIN_API_KEY=$(ADMIN_API_KEY) \
	  API_URL=http://localhost:$(PORT) \
	  uv run python -m sources.utils.build_all_index

# Read-only audit: print the `__all__` index status (doc count vs sum
# of VIP documents in PG). The underlying CLI exits non-zero when the
# index is stale (so it can be wired to a cron alert via
# `uv run python -m sources.utils.index_health all-status`), but the
# Makefile target swallows that so interactive invocations don't print
# the alarming "Error 1" line.
.PHONY: all-status
all-status:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) \
	  uv run python -m sources.utils.index_health all-status || true

# Drop and rebuild the `__all__` index from PG. Pass `IF_STALE=1` to
# skip the rebuild when the audit already says it's up-to-date — that
# variant is what `run.py` calls automatically after processing VIPs.
.PHONY: all-rebuild
all-rebuild:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) \
	  ADMIN_API_KEY=$(ADMIN_API_KEY) \
	  uv run python -m sources.utils.index_health all-rebuild \
	    $(if $(IF_STALE),--if-stale,)

# Saturation probe for the API — ramps a thread pool up the
# concurrency ladder and reports the highest level the host
# sustains within an SLO budget (p95 ≤ 2s, errors ≤ 5%). Pure
# stdlib + requests, no extra binary needed.
#
#   make load-test                            # localhost API, default ladder
#   make load-test URL=https://knowledge-web.org YES=1
#   make load-test LEVELS=1,5,10,20,50 DURATION=15 YES=1
#
# Defaults: 1→200 concurrency in 8 steps, 10s per level. Use
# LEVELS to override. URL outside localhost requires YES=1 so
# nobody accidentally hammers prod.
.PHONY: load-test
load-test:
	@LOAD_TEST_URL=$(if $(URL),$(URL),http://localhost:$(PORT)) \
	  LOAD_TEST_LEVELS=$(LEVELS) \
	  LOAD_TEST_DURATION=$(if $(DURATION),$(DURATION),10) \
	  LOAD_TEST_MAX=$(if $(MAX),$(MAX),200) \
	  YES=$(if $(YES),1,) \
	  uv run python -m sources.utils.load_test

# Delete documents (and their index entries) by personality, by source,
# or both. Refuses to run without a filter. The Rust API enforces an
# admin key for the index-delete endpoint, so this picks `ADMIN_API_KEY`
# out of the environment automatically.
#
#   make delete SLUG=simon-willison              # wipe one user entirely
#   make delete SOURCE=twitter                   # wipe all twitter docs across users
#   make delete SLUG=simon-willison SOURCE=reddit  # surgical
#   make delete SOURCE=twitter,reddit            # multiple sources
#   make delete SLUG=simon-willison DRY=1        # preview only
#   make delete SLUG=simon-willison YES=1        # skip confirmation prompt
delete:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) ADMIN_API_KEY=$(ADMIN_API_KEY) \
	uv run python -m sources.utils.delete_documents \
	  $(if $(SLUG),--slug $(SLUG),) \
	  $(if $(SOURCE),--source $(SOURCE),) \
	  $(if $(DRY),--dry,) \
	  $(if $(YES),--yes,)

# Hard-delete documents flagged `to_delete = TRUE` and remove their
# entries from the ColBERT index. The flag is set by the API when a
# user removes a source from their profile (e.g. drops a website
# entry); this job is the offline counterpart that actually frees the
# storage.
#
#   make purge                  # purge every tombstone
#   make purge SLUG=raphael-sourty   # only that user
#   make purge DRY=1                  # preview only
#   make purge YES=1                  # skip confirmation prompt
purge:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) ADMIN_API_KEY=$(ADMIN_API_KEY) \
	uv run python -m sources.delete.purge_tombstones \
	  $(if $(SLUG),--slug $(SLUG),) \
	  $(if $(DRY),--dry,) \
	  $(if $(YES),--yes,)

# Backfill image / video URLs into tweet documents whose summaries
# were ingested before the media-extraction pipeline shipped. Hits
# `/twitter/tweets?tweet_ids=...` in batches of 100 — one cheap call
# per ~100 tweets.
#
#   make backfill-tweet-media SLUG=raphael-sourty
#   make backfill-tweet-media SLUG=raphael-sourty DRY=1
backfill-tweet-media:
	@DATABASE_URL=$(DATABASE_URL) TWITTERAPIIO_API_KEY=$(TWITTERAPIIO_API_KEY) \
	uv run python -m sources.utils.backfill_tweet_media \
	  --slug $(SLUG) \
	  $(if $(DRY),--dry,)

# Backfill `documents.referenced_author` — the @handle a tweet
# refers to via retweet / quote / reply. Walks every twitter doc
# newest-first, batch-hydrates via cookie-authenticated twikit,
# writes the handle (or empty sentinel) and refreshes the engagement
# columns at the same time. Cookies come from Safari by default;
# override with `TWITTER_AUTH_TOKEN=...` `TWITTER_CT0=...` env vars.
#
#   make backfill-twitter-referenced-author                  # all users, newest-first
#   make backfill-twitter-referenced-author SLUG=tony-wu     # one personality
#   make backfill-twitter-referenced-author LIMIT=500 DRY=1  # plan-only sample
#   make backfill-twitter-referenced-author MARK_MISSING=1   # also stamp deleted/protected
.PHONY: backfill-twitter-referenced-author
backfill-twitter-referenced-author:
	@DATABASE_URL=$(DATABASE_URL) \
	uv run python scripts/backfill_twitter_referenced_author.py \
	  $(if $(SLUG),--slug $(SLUG),) \
	  $(if $(LIMIT),--limit $(LIMIT),) \
	  $(if $(MARK_MISSING),--mark-missing,) \
	  $(if $(DRY),--dry,)

# Backfill title + summary on YouTube docs whose metadata is missing
# (e.g. tweets that landed with a slug-style title like "K4i C5YYvr Qk").
# Uses YouTube's oEmbed endpoint — no API key needed — and writes
# results back via plain UPDATE. Never deletes a row.
#
#   make backfill-youtube                # apply to all users
#   make backfill-youtube DRY=1          # plan only, no writes
#   make backfill-youtube SLUG=alice     # restrict to one user
backfill-youtube:
	@DATABASE_URL=$(DATABASE_URL) \
	uv run python -m sources.utils.backfill_youtube \
	  $(if $(SLUG),--slug $(SLUG),) \
	  $(if $(DRY),--dry,)

# Evict user-soft-deleted documents from the ColBERT search index.
# The PG rows stay (so `make run` doesn't re-fetch), only the index
# entries are removed via `DELETE /indices/{name}/documents`.
#
# Run periodically (cron/nightly). Idempotent.
#
#   make prune-deleted              # prune every soft-deleted doc
#   make prune-deleted SLUG=alice   # restrict to one user
#   make prune-deleted DRY=1        # preview only, no API calls
prune-deleted:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) ADMIN_API_KEY=$(ADMIN_API_KEY) \
	uv run python -m sources.utils.prune_deleted \
	  $(if $(SLUG),--slug $(SLUG),) \
	  $(if $(DRY),--dry,)

# ── Local Development ────────────────────────────────────────

# Start everything locally: PostgreSQL (Docker) + API + web
dev:
	@echo "Starting PostgreSQL..."
	@docker compose up -d postgres
	@echo "Waiting for PostgreSQL to be healthy..."
	@docker compose exec -T postgres pg_isready -U knowledge -d knowledge > /dev/null 2>&1 || sleep 3
	@echo "Stopping Docker API/web containers (using local builds)..."
	@docker stop knowledge-knowledge-api-1 knowledge-web-1 2>/dev/null || true
	@echo "Starting API..."
	@DATABASE_URL=$(DATABASE_URL) ADMIN_API_KEY=$(ADMIN_API_KEY) ORT_DYLIB_PATH=$(ORT_DYLIB_PATH) cargo run --release --manifest-path $(KNOWLEDGE_API)/Cargo.toml --features "accelerate,model" -- --index-dir $(INDEX_DIR) --model $(MODEL) --int8 --port $(PORT) &
	@echo "Starting web server..."
	@DATABASE_URL=$(DATABASE_URL) uv run python3 web/serve.py $(WEB_PORT) &
	@echo ""
	@echo "  Web:    http://localhost:$(WEB_PORT)"
	@echo "  Admin:  http://localhost:$(WEB_PORT)/admin"
	@echo "  API:    http://localhost:$(PORT)"
	@echo "  DB:     postgresql://localhost:5433/knowledge"
	@echo ""

# Stop all local services
dev-stop:
	@echo "Stopping local services..."
	@pkill -f "knowledge-api" 2>/dev/null || true
	@pkill -f "serve.py" 2>/dev/null || true
	@docker compose stop postgres 2>/dev/null || true
	@echo "All stopped."

# ── Docker Compose (local dev — full stack) ──────────────────

# Start all services via Docker Compose (local dev)
up:
	docker compose up -d

# Stop all services (local dev)
down:
	docker compose down

# ── Remote Server ──────────────────────────────────────────
#
# Deploys are managed by Dokploy on the Hetzner box — push to
# `origin/main` and Dokploy's GitHub webhook redeploys via the
# `docker-compose.dokploy.yml` compose file. The Dokploy UI at
# https://dokploy.knowledge-web.org is where you watch builds, see
# logs, and roll back. The only target left here is `make ssh`, which
# is still handy for one-off shell work and the systemd-daemon
# wrappers below.

# SSH into the server
ssh:
	$(SSH_CMD)

# ── Clean daemon (gpt-4o-mini pedagogical rewriter) ──────────────
#
# Installs / manages the systemd service that runs
# `sources.utils.clean_daemon` on prod. The service file lives at
# sources/clean_daemon.service in the repo; this target copies it
# into /etc/systemd/system/, daemon-reloads, enables, and starts.
clean-daemon-install:
	$(SSH_CMD) "cd knowledge && sudo cp sources/clean_daemon.service /etc/systemd/system/knowledge-clean-daemon.service && sudo systemctl daemon-reload && sudo systemctl enable knowledge-clean-daemon"

clean-daemon-start:
	$(SSH_CMD) "sudo systemctl start knowledge-clean-daemon"

clean-daemon-stop:
	$(SSH_CMD) "sudo systemctl stop knowledge-clean-daemon"

clean-daemon-restart:
	$(SSH_CMD) "sudo systemctl restart knowledge-clean-daemon"

clean-daemon-status:
	$(SSH_CMD) "systemctl status knowledge-clean-daemon --no-pager || true"

clean-daemon-logs:
	$(SSH_CMD) "journalctl -u knowledge-clean-daemon -n 200 --no-pager"

# Reset cleaned=FALSE for every VIP twitter/x/huggingface/hf/arxiv/
# scholar/dblp/openreview/semanticscholar/paperswithcode doc in the
# last 90 days so the daemon picks them up on first run. Safe to
# re-run; idempotent.
clean-daemon-prime:
	$(SSH_CMD) "docker exec -i knowledge-prod-gjqqg2-postgres-1 psql -U knowledge -d knowledge -c \"UPDATE documents d SET cleaned = FALSE FROM users u WHERE u.id = d.user_id AND u.vip = TRUE AND lower(d.source) IN ('twitter','x','huggingface','hf','arxiv','scholar','dblp','openreview','semanticscholar','semantic_scholar','paperswithcode') AND d.date >= (now() - INTERVAL '90 days')::date AND d.deleted = FALSE;\""

# ── Categorize daemon (Potion static-embedding categorizer) ──
#
# Installs / manages the systemd service that runs
# `sources.utils.categorize_daemon` on prod. The service file lives
# at sources/categorize_daemon.service in the repo; this target
# copies it into /etc/systemd/system/, daemon-reloads, enables, and
# starts.
categorize-daemon-install:
	$(SSH_CMD) "cd knowledge && sudo cp sources/categorize_daemon.service /etc/systemd/system/knowledge-categorize-daemon.service && sudo systemctl daemon-reload && sudo systemctl enable knowledge-categorize-daemon"

categorize-daemon-start:
	$(SSH_CMD) "sudo systemctl start knowledge-categorize-daemon"

categorize-daemon-stop:
	$(SSH_CMD) "sudo systemctl stop knowledge-categorize-daemon"

categorize-daemon-restart:
	$(SSH_CMD) "sudo systemctl restart knowledge-categorize-daemon"

categorize-daemon-status:
	$(SSH_CMD) "systemctl status knowledge-categorize-daemon --no-pager || true"

categorize-daemon-logs:
	$(SSH_CMD) "journalctl -u knowledge-categorize-daemon -n 200 --no-pager"

# Force a prototype refresh on the next daemon iteration by removing
# the on-disk cache. Useful after editing `document_categories.sql`
# (description changes need a fresh anchor sweep to take effect).
categorize-daemon-refresh:
	$(SSH_CMD) "rm -f /root/knowledge/.cache/categorize/refined_protos_*.npz && sudo systemctl restart knowledge-categorize-daemon"

# ── Prod → local Postgres clone ────────────────────────────────
#
# Three targets that together replace the local dev DB with a copy
# of prod. Useful when you want to debug against real data, or as a
# pre-migration safety net before touching the prod stack.
#
# Files land in ./backups/ — gitignored, gzipped, timestamped:
#   backups/prod-YYYYMMDD-HHMMSS.sql.gz
#
#   make prod-db-dump      Stream a pg_dump from prod over SSH into
#                          backups/. Doesn't touch local PG. ~5 GB
#                          on a typical day → gzipped to ~800 MB.
#   make prod-db-restore   Drop + recreate the local `knowledge`
#                          database and replay the freshest dump in
#                          backups/. Local PG container must be up.
#   make prod-db-sync      Convenience: dump then restore in one go.
#
# Both commands run pg_dump / psql inside the postgres container on
# either side so we don't need the binaries on the host. The dump
# uses --no-owner --no-privileges so it replays cleanly into a
# differently-owned local DB.
# Discover the postgres container via compose labels rather than a
# hard-coded name — Dokploy prefixes the project name with a random
# suffix (e.g. `knowledge-prod-gjqqg2-postgres-1`) that changes
# whenever the project is re-created in the UI. The compose label is
# set automatically and is stable across deploys.
#
# `set -o pipefail` on the pg_dump pipeline means a failed dump exits
# non-zero, so we don't silently produce a valid-but-empty .sql.gz
# (which is what happened before this fix — a hard-coded container
# name miss produced an empty stdout that gzip wrapped as 20 bytes).
.PHONY: prod-db-dump prod-db-dump-if-stale prod-db-restore prod-db-sync
prod-db-dump:
	@mkdir -p backups
	@TS=$$(date '+%Y%m%d-%H%M%S'); \
	OUT="backups/prod-$$TS.sql.gz"; \
	echo "==> Dumping prod knowledge DB → $$OUT"; \
	CTN=$$($(SSH_CMD) "docker ps --filter label=com.docker.compose.service=postgres --format '{{.Names}}' | grep '^knowledge-' | head -1"); \
	if [ -z "$$CTN" ]; then \
	    echo "[!] could not find a postgres container with project=knowledge-* on prod." >&2; \
	    rm -f "$$OUT"; exit 1; \
	fi; \
	echo "    via container: $$CTN"; \
	set -o pipefail; \
	if $(SSH_CMD) "docker exec -i $$CTN pg_dump -U knowledge -d knowledge --no-owner --no-privileges --clean --if-exists" \
	    | gzip > "$$OUT.partial"; then \
	    mv "$$OUT.partial" "$$OUT"; \
	    SZ=$$(du -h "$$OUT" | cut -f1); \
	    echo "✓ Saved $$OUT ($$SZ)"; \
	else \
	    echo "[!] pg_dump failed — leaving truncated file at $$OUT.partial for inspection." >&2; \
	    exit 1; \
	fi

# Guard target: stream a prod dump only if there isn't already one
# from today (local date). Used as a `make twitter-feed` prerequisite
# so the first invocation each day always leaves a fresh on-disk
# snapshot of prod in ./backups/ before the long-running feeder
# starts. Subsequent runs the same day are no-ops.
#
# Failures (SSH down, dump errored mid-stream) are non-fatal: we
# remove the truncated file and proceed with the feeder anyway —
# losing the feeder over a snapshot hiccup would be worse than
# missing one day's backup.
prod-db-dump-if-stale:
	@mkdir -p backups
	@today=$$(date '+%Y%m%d'); \
	if ls -1 backups/prod-$${today}-*.sql.gz >/dev/null 2>&1; then \
	    echo "==> prod-db snapshot from today already exists in backups/ — skipping."; \
	else \
	    echo "==> No prod-db snapshot from today found — taking one before the feeder starts."; \
	    if $(MAKE) --no-print-directory prod-db-dump; then \
	        echo "==> Snapshot complete; starting feeder."; \
	    else \
	        echo "[!] prod-db-dump failed — continuing to the feeder anyway." >&2; \
	        rm -f backups/prod-$${today}-*.sql.gz.partial 2>/dev/null || true; \
	    fi; \
	fi

prod-db-restore:
	@LATEST=$$(ls -t backups/prod-*.sql.gz 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
	    echo "[!] no dump found in backups/. Run \`make prod-db-dump\` first." >&2; \
	    exit 1; \
	fi; \
	echo "==> Restoring $$LATEST into local PG (knowledge-postgres-1)"; \
	if ! docker ps --format '{{.Names}}' | grep -q '^knowledge-postgres-1$$'; then \
	    echo "[!] local PG container 'knowledge-postgres-1' is not running."; \
	    echo "    Start it with \`make up\` first."; \
	    exit 1; \
	fi; \
	echo "    Dropping + recreating local 'knowledge' database…"; \
	docker exec -i knowledge-postgres-1 psql -U knowledge -d postgres \
	    -c "DROP DATABASE IF EXISTS knowledge_old;" >/dev/null; \
	docker exec -i knowledge-postgres-1 psql -U knowledge -d postgres \
	    -c "ALTER DATABASE knowledge RENAME TO knowledge_old;" 2>/dev/null || true; \
	docker exec -i knowledge-postgres-1 psql -U knowledge -d postgres \
	    -c "CREATE DATABASE knowledge OWNER knowledge;" >/dev/null; \
	echo "    Replaying dump (gunzip | psql)…"; \
	gunzip -c "$$LATEST" | docker exec -i knowledge-postgres-1 \
	    psql -U knowledge -d knowledge >/dev/null; \
	echo "    Dropping the renamed-aside knowledge_old…"; \
	docker exec -i knowledge-postgres-1 psql -U knowledge -d postgres \
	    -c "DROP DATABASE IF EXISTS knowledge_old;" >/dev/null; \
	echo "✓ Local PG is now a copy of prod (from $$LATEST)"

prod-db-sync: prod-db-dump prod-db-restore

# ── Local Twitter feeder → prod PG ────────────────────────────
#
# Everything is prod: the queue is `GET /api/admin/twitter-queue` on
# https://$(DOMAIN) (prod PG), the already-known URLs come from prod,
# and every scraped tweet is POSTed to prod's
# `/api/admin/tweets/ingest`. Only the twikit fetches and the Safari
# cookies are local. Nothing touches the local dev DB.
#
# Coverage is every prod account with a non-empty
# `sources.twitter.username` — VIP or not, VIPs first. Today that's
# ~635 accounts.
#
# Logs go to `logs/twitter-feed-<ts>.log` and the terminal.
#
#   make twitter-feed                     # default: rest 1h between sweeps
#   make twitter-feed ARGS="--one-shot"   # single pass, exit
#   make twitter-feed ARGS="--min-age 0"  # ignore the 24h guard: walk ALL accounts
#   make twitter-feed ARGS="--rest 1800 --personality-delay 6"
#
# The first invocation each day depends on `prod-db-dump-if-stale`
# which streams a fresh prod pg_dump into ./backups/ before the
# feeder starts — that way at least one on-disk snapshot exists on
# this laptop per calendar day, regardless of whether the prod
# pg-backup sidecar volume survives a host event.
#
# `--min-age 24` is prepended so the server-side queue endpoint
# excludes any account whose `last_attempt_at` is within the last 24h.
# That gives "at most one parsing per personality per day" even
# across restarts (state lives in `twitter_feed_attempts`). Pass a
# different value via `ARGS="--min-age 0"` to override — argparse
# uses the last occurrence.
#
# So a SHORT queue is not a cap: restart at noon after the morning
# pass already covered 470 accounts and the queue is the ~165
# remainder. The banner now prints "queue: 165 of 635 … held back:
# 470 attempted < 24h ago" so that's unambiguous. `--min-age 0`
# forces the full roster (the today-attempted rows just sort last).
#
# Ctrl+C exits cleanly: in-flight personality finishes.
.PHONY: twitter-feed twitter-feed-logs
twitter-feed: prod-db-dump-if-stale
	KNOWLEDGE_ADMIN_TOKEN=$(KNOWLEDGE_ADMIN_TOKEN) \
	API_URL=https://$(DOMAIN) \
		scripts/twitter_feed.sh --min-age 24 $(ARGS)

# Tail the most recent twitter-feed log (handy if the client is
# running in another tmux pane / screen session).
twitter-feed-logs:
	@ls -t logs/twitter-feed-*.log 2>/dev/null | head -1 | xargs -I{} tail -f {}

# Install a global `knowledge-twitter-feed` shim into ~/.local/bin
# so the feeder is launchable from anywhere on this machine. The
# shim is self-contained — it bakes in the repo path, sources
# `.env`, then exec's `scripts/twitter_feed.sh`. Re-run any time
# you move the repo to regenerate the path.
.PHONY: install-twitter-feed uninstall-twitter-feed
install-twitter-feed:
	@mkdir -p $(HOME)/.local/bin
	@{ \
	  echo '#!/usr/bin/env bash'; \
	  echo '# Auto-generated by `make install-twitter-feed`. Do not edit;'; \
	  echo '# re-run `make install-twitter-feed` from the repo root to refresh.'; \
	  echo 'set -e'; \
	  echo 'REPO="$(CURDIR)"'; \
	  echo '# Extract just the keys the feeder needs from .env. Avoids'; \
	  echo '# `source .env`, which breaks on free-form lines (bare URLs,'; \
	  echo '# comments without `#`, etc.).'; \
	  echo 'if [ -f "$$REPO/.env" ]; then'; \
	  echo '  for var in HETZNER_IP SSH_KEY SSH_USER POSTGRES_PASSWORD LOCAL_PORT; do'; \
	  echo '    line=$$(grep -E "^$${var}=" "$$REPO/.env" | tail -1 || true)'; \
	  echo '    [ -n "$$line" ] && export "$${line%%=*}"="$${line#*=}"'; \
	  echo '  done'; \
	  echo 'fi'; \
	  echo 'exec "$$REPO/scripts/twitter_feed.sh" "$$@"'; \
	} > $(HOME)/.local/bin/knowledge-twitter-feed
	@chmod +x $(HOME)/.local/bin/knowledge-twitter-feed
	@echo "✓ installed $(HOME)/.local/bin/knowledge-twitter-feed → $(CURDIR)/scripts/twitter_feed.sh"
	@command -v knowledge-twitter-feed >/dev/null 2>&1 || \
	  echo '! ~/.local/bin is not on your PATH — add `export PATH="$$HOME/.local/bin:$$PATH"` to your shell rc'

uninstall-twitter-feed:
	@rm -f $(HOME)/.local/bin/knowledge-twitter-feed
	@echo "removed $(HOME)/.local/bin/knowledge-twitter-feed"

# ── launchd LaunchAgent for the twitter feeder ────────────────
#
# Installs a per-user LaunchAgent so the feeder starts automatically
# at login, restarts itself if it crashes (e.g. SSH tunnel dies), and
# keeps a rolling pair of logs under ~/Library/Logs. The plist is
# generated from scripts/com.knowledge.twitter-feed.plist.template by
# substituting absolute paths in for the @@REPO@@ / @@HOME@@ tokens —
# launchd doesn't expand env vars inside the file itself.
#
# Mac-only. Other platforms can keep using `make twitter-feed`.
LAUNCHD_LABEL  := com.knowledge.twitter-feed
LAUNCHD_PLIST  := $(HOME)/Library/LaunchAgents/$(LAUNCHD_LABEL).plist
LAUNCHD_TARGET := gui/$(shell id -u)/$(LAUNCHD_LABEL)

twitter-feed-launchd-install:
	@mkdir -p $(HOME)/Library/LaunchAgents $(HOME)/Library/Logs
	@sed -e 's|@@REPO@@|$(CURDIR)|g' \
	     -e 's|@@HOME@@|$(HOME)|g' \
	     scripts/com.knowledge.twitter-feed.plist.template > $(LAUNCHD_PLIST)
	@# Re-load if it was already installed; bootstrap fresh otherwise.
	@launchctl bootout gui/$(shell id -u)/$(LAUNCHD_LABEL) 2>/dev/null || true
	@launchctl bootstrap gui/$(shell id -u) $(LAUNCHD_PLIST)
	@launchctl enable $(LAUNCHD_TARGET)
	@launchctl kickstart -k $(LAUNCHD_TARGET) || true
	@echo "✓ launched $(LAUNCHD_LABEL)"
	@echo "  plist:      $(LAUNCHD_PLIST)"
	@echo "  stdout log: $(HOME)/Library/Logs/knowledge-twitter-feed.out.log"
	@echo "  stderr log: $(HOME)/Library/Logs/knowledge-twitter-feed.err.log"
	@echo "  feed logs:  $(CURDIR)/logs/twitter-feed-*.log"

twitter-feed-launchd-uninstall:
	@launchctl bootout gui/$(shell id -u)/$(LAUNCHD_LABEL) 2>/dev/null || true
	@rm -f $(LAUNCHD_PLIST)
	@echo "✓ removed $(LAUNCHD_LABEL)"

twitter-feed-launchd-status:
	@launchctl print $(LAUNCHD_TARGET) 2>/dev/null | head -30 || \
	  echo "(not loaded — run \`make twitter-feed-launchd-install\`)"

twitter-feed-launchd-logs:
	@echo "==> stderr (~/Library/Logs/knowledge-twitter-feed.err.log)"
	@tail -n 40 $(HOME)/Library/Logs/knowledge-twitter-feed.err.log 2>/dev/null || true
	@echo
	@echo "==> latest feed log under logs/"
	@ls -t logs/twitter-feed-*.log 2>/dev/null | head -1 | xargs -I{} tail -n 40 {} 2>/dev/null || true

# Convenience — bounce the agent so a freshly-pulled change to
# twitter_feeder.py or the wrapper script picks up.
twitter-feed-launchd-restart:
	@launchctl kickstart -k $(LAUNCHD_TARGET)
	@echo "✓ kicked $(LAUNCHD_LABEL)"

# Indexer daemon — long-running process that owns the ColBERT index.
# `run.py` no longer touches the index; this daemon picks the user
# with the most-urgent index work (broken > error > missing >
# pg_drift, biggest backlog first inside each tier) and reindexes.
#
#   make indexer-daemon                      # loop forever on local
#   make indexer-daemon ARGS="--dry"         # show queue, do nothing
#   make indexer-daemon ARGS="--once"        # process top user, exit
#   make indexer-daemon ARGS="--vip-only --exclude-drift"
#
# For prod, `make indexer-daemon-prod` opens an SSH tunnel to the
# prod PG and points the daemon at the public HTTPS API.
.PHONY: indexer-daemon indexer-daemon-prod
indexer-daemon:
	@DATABASE_URL=$(DATABASE_URL) API_URL=http://localhost:$(PORT) \
	  uv run python -m sources.indexer_daemon $(ARGS)

indexer-daemon-prod:
	HETZNER_IP=$(HETZNER_IP) SSH_KEY=$(SSH_KEY) SSH_USER=$(SSH_USER) \
	POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
		scripts/indexer_daemon_prod.sh $(ARGS)

# Push every twitter doc from the local dev PG into prod, mapping
# users by `username` (their IDs can differ between dev and prod).
# Safe: uses INSERT ... ON CONFLICT DO NOTHING — never touches a row
# prod already has. Per-user early stop when prod count ≥ local count.
#
#   make sync-tweets-to-prod                          # full run
#   make sync-tweets-to-prod ARGS="--dry-run"         # report only
#   make sync-tweets-to-prod ARGS="--limit 5"         # smoke test
.PHONY: sync-tweets-to-prod
sync-tweets-to-prod:
	HETZNER_IP=$(HETZNER_IP) SSH_KEY=$(SSH_KEY) SSH_USER=$(SSH_USER) \
	POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
		scripts/sync_tweets_to_prod.sh $(ARGS)

# ── Continuous-pipeline runner (prod, via systemd) ────────────────────
#
# `sources/continuous_pipeline.sh` is a long-lived loop that drains
# every VIP and non-VIP user, VIP-first, twitter excluded, 12 h
# cool-down per user (≤ 2 runs/day). One iteration = one user, and
# `run.py <slug> --source <every-non-twitter-source>` runs all of
# their sources in a single invocation — so the cool-down really is
# a per-user constraint, not per-source.
#
# Production reliability is owned by systemd:
#
#   * `Restart=on-failure` auto-restarts after any crash, including
#     server reboots and SIGKILL from OOM.
#   * `CPUAffinity=0` pins the runner and every spawned python
#     process to the first CPU, so it can never starve the Rust API
#     on the remaining cores.
#   * `MemoryMax=2G` caps the runner family so a pathological python
#     run can't push the host into swap.
#
# All four Makefile targets shell out to `systemctl` on the prod host.

.PHONY: continuous-up continuous-down continuous-status continuous-logs

# continuous-up: install/refresh the systemd unit and start (or
# restart) the continuous-pipeline service on prod. Idempotent —
# running on a fresh server and on one already serving both end at
# the same state.
continuous-up:
	# Step 1: fast-forward the prod checkout to the latest `terminal`
	# branch so the unit file + runner script the service launches
	# are in sync with what we just pushed from local.
	$(SSH_CMD) "cd knowledge && git fetch origin terminal && git reset --hard origin/terminal"
	# Step 2: copy the unit file into /etc/systemd/system, reload
	# systemd's view of unit files (`daemon-reload`), ensure it's
	# enabled-on-boot (`enable`), and (re)launch the service
	# (`restart` — same as `start` if it wasn't running). Each step
	# is &&-chained so a failure short-circuits and surfaces here.
	$(SSH_CMD) "install -m 0644 /root/knowledge/sources/knowledge-continuous.service /etc/systemd/system/knowledge-continuous.service && systemctl daemon-reload && systemctl enable knowledge-continuous && systemctl restart knowledge-continuous"
	# Give the service ~4 seconds to actually start; without this
	# the status query right below would race and report "starting".
	@sleep 4
	# Step 3: show the runtime status (active/inactive, PID, uptime,
	# the line systemd is currently waiting on). `--no-pager` keeps
	# the output piped straight back; `--lines 0` skips the journal
	# tail since we have our own log files.
	$(SSH_CMD) "systemctl --no-pager --lines 0 status knowledge-continuous"

# continuous-down: stop the service cleanly. The unit file stays in
# /etc/systemd/system, so a future `make continuous-up` re-uses the
# same definition without re-installing it.
continuous-down:
	# `systemctl stop` sends SIGTERM, waits for graceful exit, then
	# SIGKILLs if the process didn't honour TERM within the unit's
	# default TimeoutStopSec (90s). The bash loop exits on its next
	# iteration when its WHILE-loop checks the (now-deleted) stop
	# flag, well before the 90s timeout fires.
	$(SSH_CMD) "systemctl stop knowledge-continuous && echo stopped"

# continuous-status: one-shot status snapshot — service state, the
# main PID's CPU + uptime, the human-readable log tail, and the
# size of the history file (which doubles as a "how many user-runs
# have we done since last rotation" gauge).
#
# The ssh argument is single-quoted so the local shell doesn't try
# to expand `$(systemctl …)` before the SSH wrapper sends it; the
# remote shell does the expansion at the right time.
continuous-status:
	$(SSH_CMD) 'systemctl --no-pager --lines 0 status knowledge-continuous || true ; \
	  echo "--- runner pid + CPU + uptime" ; \
	  PID=$$(systemctl show -p MainPID --value knowledge-continuous); \
	  if [ -n "$$PID" ] && [ "$$PID" != 0 ]; then \
	    ps -p $$PID -o pid,psr,pcpu,etime,start,comm 2>/dev/null; \
	  fi ; \
	  echo "--- last 25 log lines" ; \
	  tail -25 knowledge/logs/continuous_pipeline.log 2>/dev/null ; \
	  echo "--- history size" ; \
	  wc -l knowledge/logs/continuous_pipeline.history 2>/dev/null'

# continuous-logs: follow the per-task pipeline stdout in real time.
# `tail -f` streams the Python output from each run.py invocation
# (one user at a time). Kill with Ctrl-C.
#
# Alternative: `journalctl -u knowledge-continuous -f` on the prod
# host gives the systemd-managed view (boot lines + restarts).
continuous-logs:
	$(SSH_CMD) "tail -f knowledge/logs/continuous_pipeline_runs.log"

# ── Ops UIs (Portainer + Dozzle) ──────────────────────────────
# Portainer = Docker management UI, Dozzle = live log viewer.
# Both run on the server inside the `knowledge_default` network
# and are reverse-proxied by Caddy at:
#   https://portainer.$(DOMAIN)
#   https://logs.$(DOMAIN)
# Requires DNS A records for those subdomains pointing at the VPS.

ops-install:
	$(SSH_CMD) "docker volume create portainer_data && \
	  docker run -d --name portainer --restart unless-stopped \
	    --network knowledge_default \
	    -v /var/run/docker.sock:/var/run/docker.sock \
	    -v portainer_data:/data \
	    portainer/portainer-ce:latest && \
	  docker run -d --name dozzle --restart unless-stopped \
	    --network knowledge_default \
	    -v /var/run/docker.sock:/var/run/docker.sock \
	    amir20/dozzle:latest && \
	  cd knowledge && git pull && \
	  docker compose -f docker-compose.prod.yml exec caddy caddy reload --config /etc/caddy/Caddyfile"

ops-uninstall:
	$(SSH_CMD) "docker rm -f portainer dozzle 2>/dev/null; docker volume rm portainer_data 2>/dev/null; true"

ops-restart:
	$(SSH_CMD) "docker restart portainer dozzle"

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

# Arm both git hook types — pre-commit (per-file on commit) and
# pre-push (changed-since-upstream on push) — so CI can't see anything
# the local hook didn't already. `make install-dev` calls this, so a
# fresh clone is set up after one command.
pre-commit-install:
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

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
