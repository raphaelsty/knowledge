#!/usr/bin/env bash
# Launch wrapper for the continuous pipeline.
#
# Wraps `continuous_pipeline.sh` with the environment + CPU pinning we
# always want on the production host:
#
#   * `taskset -c 0` — pin to the first CPU only, so the runner
#     (whose Python work + downloaded payloads can spike) never
#     starves the Rust API on the other cores.
#   * `uv` Python 3.11 — the synced env on prod (the project's
#     pyproject pins pandas 1.5.3 which has no Python 3.12 wheels).
#   * 5 s inter-task sleep, 12 h cool-down per (user, source), 10 min
#     per-task hard timeout. Overridable via env.
#
# Always invoked from inside the repo root; `make continuous-up` does
# that for you. Stdout/stderr land in `logs/continuous_pipeline.boot`.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

export PATH="/root/.local/bin:${PATH}"
export DATABASE_URL="${DATABASE_URL:-postgresql://knowledge:knowledge@localhost:5433/knowledge}"
export API_URL="${API_URL:-https://knowledge-web.org}"
export PORT="${PORT:-8080}"
export UV_PYTHON="${UV_PYTHON:-3.11}"

# Runner knobs — see `continuous_pipeline.sh` for the meaning.
export SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"
export MIN_INTERVAL_HOURS="${MIN_INTERVAL_HOURS:-12}"
export EMPTY_QUEUE_WAIT="${EMPTY_QUEUE_WAIT:-600}"
export PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-600}"

# CPU pinning happens at the systemd-unit level (`CPUAffinity=0`),
# which is inherited by every child process. When running the script
# directly without systemd (local debugging) we fall back to
# `taskset` if it's available; on macOS where neither exists, we
# just exec.
if [ -n "${INVOCATION_ID:-}" ]; then
  # Started by systemd — CPUAffinity already in effect.
  exec bash sources/continuous_pipeline.sh
elif command -v taskset >/dev/null 2>&1; then
  exec taskset -c 0 bash sources/continuous_pipeline.sh
else
  exec bash sources/continuous_pipeline.sh
fi
