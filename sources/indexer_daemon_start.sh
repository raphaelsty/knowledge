#!/usr/bin/env bash
# Launch wrapper for the indexer daemon.
#
# Mirrors `continuous_pipeline_start.sh` so the operational model is
# identical: when systemd starts us we trust the unit's
# `CPUAffinity=` to do the pinning; when invoked by hand (e.g.
# debugging on the prod host outside systemd, or on a dev box) we
# fall back to `taskset -c 1` so the daemon still respects the
# single-CPU contract.
#
# Single-CPU pinning is non-negotiable on the prod box: the
# embedder (model2vec + ONNX) is CPU-bound and would otherwise
# trample the Rust API on the other cores. The unit pins to CPU 1
# (CPU 0 is reserved for `knowledge-continuous.service`).
#
# Logs land in `logs/indexer_daemon.boot` (systemd's StandardOutput
# target) plus whatever `indexer_daemon.py` writes to stdout, which
# systemd journals via the same pipe.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

export PATH="/root/.local/bin:${PATH}"
# Talk to the local prod stack — Postgres on the docker-published
# 127.0.0.1:5433, Rust API behind Caddy on the public domain. The
# unit can override these in /etc/default/knowledge-indexer if a
# different prod topology shows up.
export DATABASE_URL="${DATABASE_URL:-postgresql://knowledge:knowledge@localhost:5433/knowledge}"
export API_URL="${API_URL:-https://knowledge-web.org}"
export UV_PYTHON="${UV_PYTHON:-3.11}"

# Daemon knobs. See `sources/indexer_daemon.py --help`.
INDEXER_ARGS=( )
[ -n "${INDEXER_VIP_ONLY:-}" ] && INDEXER_ARGS+=("--vip-only")
[ -n "${INDEXER_EXCLUDE_DRIFT:-}" ] && INDEXER_ARGS+=("--exclude-drift")
[ -n "${INDEXER_SLEEP:-}" ] && INDEXER_ARGS+=("--sleep" "$INDEXER_SLEEP")
[ -n "${INDEXER_IDLE_SLEEP:-}" ] && INDEXER_ARGS+=("--idle-sleep" "$INDEXER_IDLE_SLEEP")

# Resolve the Python runtime — same fallback chain as
# `continuous_pipeline.sh`. Prod uses the local `.venv` (no `uv`
# installed); a dev box typically has `uv`; everything else falls
# back to whatever `python3` is on PATH.
if command -v uv >/dev/null 2>&1; then
  PY_RUN=(uv run python -m sources.indexer_daemon)
elif [ -x ".venv/bin/python3" ]; then
  PY_RUN=(.venv/bin/python3 -m sources.indexer_daemon)
else
  PY_RUN=(python3 -m sources.indexer_daemon)
fi

# CPU pinning happens at the systemd-unit level (`CPUAffinity=1`),
# inherited by every child process. When running outside systemd we
# fall back to `taskset -c 1` if available; on macOS where neither
# exists, we just exec without pinning (dev convenience).
if [ -n "${INVOCATION_ID:-}" ]; then
  # Started by systemd — CPUAffinity already in effect.
  exec "${PY_RUN[@]}" "${INDEXER_ARGS[@]}"
elif command -v taskset >/dev/null 2>&1; then
  exec taskset -c 1 "${PY_RUN[@]}" "${INDEXER_ARGS[@]}"
else
  exec "${PY_RUN[@]}" "${INDEXER_ARGS[@]}"
fi
