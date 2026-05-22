#!/usr/bin/env bash
# Bootstrap wrapper for the launchd-managed twitter feeder.
#
# launchd strips the inherited environment (PATH, locale, …) and
# never sources your shell rc, so the Makefile's
# `HETZNER_IP=$(HETZNER_IP) SSH_KEY=$(SSH_KEY) ...` prelude isn't
# available. This script does the equivalent at runtime: read the
# committed-but-gitignored .env, export every KEY=value line that
# the downstream `twitter_feed.sh` expects, then exec it.
#
# Quiet by design — every log line goes to twitter_feed.sh's
# tee'd file under logs/. The plist also captures stdout/stderr
# under ~/Library/Logs so you can `log show` for crash context.

set -euo pipefail

# Repo root = parent of this script's directory. `realpath` is
# macOS-fine; coreutils not needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Source .env — but defensively: only export lines that look like
# `KEY=VALUE`. The committed example file has stray URLs and
# bracketed text that `set -a; source .env` would choke on.
if [ -f .env ]; then
    while IFS= read -r line; do
        # skip comments / blanks
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        # accept only well-formed assignments
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        export "${line%%=*}=${line#*=}"
    done < .env
fi

# Defaults — same as `scripts/twitter_feed.sh` baked-ins. Set here
# too so the wrapper is self-contained even with an empty .env.
: "${SSH_USER:=root}"
: "${POSTGRES_PASSWORD:=knowledge}"

exec "$REPO_ROOT/scripts/twitter_feed.sh" "$@"
