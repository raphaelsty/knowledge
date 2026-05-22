#!/usr/bin/env bash
# Run `sources.indexer_daemon` against prod: opens an SSH tunnel
# to Hetzner's loopback-only Postgres (5433 → 15433 locally), points
# the daemon at the public HTTPS API, and tees stdout to a
# timestamped log file. Same pattern as `scripts/twitter_feed.sh`.
#
# Forwards all extra CLI args to the python daemon, so you can run:
#   make indexer-daemon-prod ARGS="--dry"
#   make indexer-daemon-prod ARGS="--once"
#   make indexer-daemon-prod ARGS="--vip-only --sleep 3"
set -euo pipefail

: "${HETZNER_IP:?set HETZNER_IP}"
: "${SSH_KEY:?set SSH_KEY}"
: "${SSH_USER:=root}"
: "${POSTGRES_PASSWORD:=knowledge}"

LOCAL_PORT="${LOCAL_PORT:-15433}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"
TS=$(date '+%Y%m%d-%H%M%S')
LOG="$REPO_ROOT/logs/indexer-daemon-$TS.log"

echo "==> Opening SSH tunnel  localhost:$LOCAL_PORT  →  $HETZNER_IP:5433"
ssh -i "$SSH_KEY" \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=4 \
    -fN -L "$LOCAL_PORT:localhost:5433" \
    "$SSH_USER@$HETZNER_IP"

TUNNEL_PID=$(pgrep -f "ssh .*-L $LOCAL_PORT:localhost:5433 $SSH_USER@$HETZNER_IP" | tail -1 || true)
if [ -z "$TUNNEL_PID" ]; then
    echo "[!] tunnel didn't start — aborting"
    exit 1
fi
echo "==> tunnel pid=$TUNNEL_PID"

cleanup() {
    if [ -n "${TUNNEL_PID:-}" ] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo
        echo "==> closing tunnel (pid=$TUNNEL_PID)"
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

DB_URL="postgresql://knowledge:$POSTGRES_PASSWORD@localhost:$LOCAL_PORT/knowledge"
API_URL="${INDEXER_API_URL:-https://knowledge-web.org}"

echo "==> Logs:    $LOG"
echo "==> API URL: $API_URL"
echo "==> DB URL:  postgresql://knowledge:***@localhost:$LOCAL_PORT/knowledge"
echo

PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

DATABASE_URL="$DB_URL" API_URL="$API_URL" \
    "$PY" -m sources.indexer_daemon "$@" 2>&1 | tee -a "$LOG"
