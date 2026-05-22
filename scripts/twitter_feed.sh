#!/usr/bin/env bash
# Launch `knowledge-twitter-feed` against the Knowledge prod API.
#
# No SSH tunnel any more — the feeder talks to the Rust admin API
# over plain HTTPS for queue, existing-URL and ingest calls. The
# only secrets it needs are KNOWLEDGE_ADMIN_TOKEN (so the admin
# endpoints accept it) and the local Safari twikit cookies for the
# Twitter fetches themselves.
#
# Logs are tee'd into `logs/twitter-feed-<timestamp>.log` AND echoed
# to the terminal, so you can leave a window open and watch progress
# live. Re-running creates a new log file; old ones stay on disk.
#
# Env (auto-loaded from .env via the Makefile, or set manually):
#   KNOWLEDGE_ADMIN_TOKEN   Shared secret for the admin endpoints.
#   API_URL                 Base URL (default: https://knowledge-web.org).
#
# CLI args are forwarded to `knowledge-twitter-feed` verbatim so you
# can override `--rest`, `--personality-delay`, `--one-shot` etc.
set -euo pipefail

: "${KNOWLEDGE_ADMIN_TOKEN:?set KNOWLEDGE_ADMIN_TOKEN (see .env)}"
: "${API_URL:=https://knowledge-web.org}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"
TS=$(date '+%Y%m%d-%H%M%S')
LOG="$REPO_ROOT/logs/twitter-feed-$TS.log"

echo "==> API:   $API_URL"
echo "==> Logs:  $LOG"
echo "==> Press Ctrl+C to stop cleanly (in-flight personality will finish)."
echo

# Prefer the venv binary so a bare `make twitter-feed` works
# without the user having to activate the venv first.
BIN="$REPO_ROOT/.venv/bin/knowledge-twitter-feed"
if [ ! -x "$BIN" ]; then
    if command -v knowledge-twitter-feed >/dev/null 2>&1; then
        BIN="knowledge-twitter-feed"
    else
        echo "[!] knowledge-twitter-feed not found — run \`make install-dev\` first" >&2
        exit 1
    fi
fi

# `tee -a` so the log persists even if the operator pipes through
# `less` or detaches the terminal.
API_URL="$API_URL" KNOWLEDGE_ADMIN_TOKEN="$KNOWLEDGE_ADMIN_TOKEN" \
    "$BIN" "$@" 2>&1 | tee -a "$LOG"
