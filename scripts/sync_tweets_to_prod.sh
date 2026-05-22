#!/usr/bin/env bash
# Wrapper for `scripts/sync_tweets_to_prod.py` that opens the SSH
# tunnel to Hetzner's loopback-only Postgres before running the
# sync, then tears it down on exit. Same pattern as
# `scripts/twitter_feed.sh`.
#
# All extra CLI args (e.g. `--dry-run`, `--limit 5`) are forwarded
# verbatim to the python script.
set -euo pipefail

: "${HETZNER_IP:?set HETZNER_IP}"
: "${SSH_KEY:?set SSH_KEY}"
: "${SSH_USER:=root}"
: "${POSTGRES_PASSWORD:=knowledge}"

LOCAL_PORT="${LOCAL_PORT:-15433}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"
TS=$(date '+%Y%m%d-%H%M%S')
LOG="$REPO_ROOT/logs/sync-tweets-$TS.log"

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

PROD_URL="postgresql://knowledge:$POSTGRES_PASSWORD@localhost:$LOCAL_PORT/knowledge"
LOCAL_URL="${LOCAL_DATABASE_URL:-postgresql://knowledge:knowledge@localhost:5433/knowledge}"

echo "==> Logs:  $LOG"
echo "==> Local: $LOCAL_URL"
echo "==> Prod:  postgresql://knowledge:***@localhost:$LOCAL_PORT/knowledge"
echo

# Prefer the venv python so `psycopg` is available without manual activation.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

PROD_DATABASE_URL="$PROD_URL" \
    "$PY" "$REPO_ROOT/scripts/sync_tweets_to_prod.py" \
        --local-url "$LOCAL_URL" \
        --prod-url  "$PROD_URL" \
        "$@" 2>&1 | tee -a "$LOG"
