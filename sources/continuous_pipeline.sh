#!/usr/bin/env bash
# Continuous, per-USER pipeline runner.
#
# Unit of work is a *user*. Each iteration:
#   1. Picks the user whose last attempt is the oldest (or has never
#      been touched). VIPs always come before non-VIPs.
#   2. Runs `run.py <slug> --source <comma-joined non-twitter keys>`
#      so every source for that user refreshes in one invocation.
#   3. Records the attempt's epoch in the history file so the next
#      iteration skips that user for MIN_INTERVAL_HOURS (default 12h
#      → at most twice a day per user).
#
# Design choices for "run for months unattended":
#   * The launch wrapper (`continuous_pipeline_start.sh`) is started
#     by a systemd unit (`Restart=on-failure`) so unexpected crashes
#     and server reboots auto-recover.
#   * `CPUAffinity=0` in the systemd unit pins the runner (and every
#     child python process) to the first CPU so it can never starve
#     the Rust API serving the live site on the other cores.
#   * History + per-task stdout logs are trimmed in-place every N
#     iterations so disk usage stays bounded.
#   * Twitter ingestion is excluded twice over: it's filtered out of
#     the per-user source list before invocation, and a defensive
#     case clause refuses to invoke the pipeline if a twitter slug
#     ever slipped through.
#   * Strictly sequential. Each fetcher (arxiv, scholar, huggingface,
#     github) already throttles on rate-limit responses; we add a
#     SLEEP_BETWEEN seconds of dead-time between user-runs so the
#     external host windows refill.
#
# Per-source failure isolation:
#   `run.py <slug> --source a,b,c` wraps each source in
#   `sources/utils/client.py::_ts(...)` — a context manager that
#   catches per-source exceptions, records a `failed` row in
#   `pipeline_source_runs`, and lets the next source still run. The
#   final indexing pass at the end of `run_pipeline` rebuilds the
#   ColBERT index from every doc in PG with `indexed=FALSE`,
#   regardless of which source delivered it. Net effect: a single
#   fetcher hitting a 429 / 403 / crash never blocks the other
#   sources for that user, and never starves the user's search
#   index. If the whole `run.py` process is hard-killed (e.g.
#   PER_RUN_TIMEOUT), its unindexed rows simply get picked up on
#   the user's next iteration 12 h later — that's also handled by
#   the `indexed=FALSE` query.
#
# Stop with `make continuous-down` (removes the stop flag and signals
# the pid file).

set -uo pipefail

# ── Tunables (override via env) ────────────────────────────────────
MIN_INTERVAL_HOURS=${MIN_INTERVAL_HOURS:-12}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-15}
EMPTY_QUEUE_WAIT=${EMPTY_QUEUE_WAIT:-600}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1800}   # max 30 min per user-run
STOP_FLAG=${STOP_FLAG:-/tmp/continuous_pipeline.run}
# Log rotation thresholds. After every LOG_ROTATE_EVERY iterations,
# trim history to its last HISTORY_MAX_LINES rows and truncate the
# per-task stdout log if it exceeds RUNNER_LOG_MAX_BYTES.
LOG_ROTATE_EVERY=${LOG_ROTATE_EVERY:-50}
HISTORY_MAX_LINES=${HISTORY_MAX_LINES:-10000}
RUNNER_LOG_MAX_BYTES=${RUNNER_LOG_MAX_BYTES:-104857600}   # 100 MB
LOG_MAX_BYTES=${LOG_MAX_BYTES:-10485760}                  # 10 MB

# Sources we intentionally skip in this continuous mode. Twitter has
# its own dedicated paid process (TwitterAPI.io credits, separate
# rate-limit budget); we must never fire any twitter variant.
SKIPPED_SOURCES="'twitter','twitter_likes','twitter_bookmarks','twitter_tweets'"
SKIPPED_PREFIX="twitter"

cd "$(dirname "$0")/.." || exit 1

# ── Bootstrap ─────────────────────────────────────────────────────
mkdir -p logs
LOG=logs/continuous_pipeline.log
RUNNER_LOG=logs/continuous_pipeline_runs.log
HISTORY=${HISTORY:-logs/continuous_pipeline.history}
touch "$STOP_FLAG" "$HISTORY" "$RUNNER_LOG" "$LOG"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG" ; }

# Drop our pid so `make continuous-down` can target this specific
# process without `pkill -f` (which is unsafe inside an ssh command
# whose own command-line contains the script name and would
# self-terminate the remote shell). EXIT trap clears it on a clean
# stop; systemd handles the SIGKILL case by re-launching us.
PIDFILE=${PIDFILE:-/tmp/continuous_pipeline.pid}
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# Resolve the Python runtime. `uv` is the canonical path on the prod
# host (pyproject pins pandas 1.5.3 which only has Python ≤ 3.11
# wheels, so we explicitly pin UV_PYTHON in the launch wrapper).
if command -v uv >/dev/null 2>&1; then
  PY_RUN="uv run python3"
elif [ -x ".venv/bin/python3" ]; then
  PY_RUN=".venv/bin/python3"
else
  PY_RUN="python3"
fi
log "runtime: $PY_RUN"

# Postgres CLI helper. Uses the docker-compose container by default;
# override DB_CMD if running on a host with native psql installed.
DB_CMD=${DB_CMD:-"docker exec -i knowledge-postgres-1 psql -U knowledge -d knowledge -tAc"}

# Per-user history. Each line:
#
#     <epoch-secs>\t<username>\t<comma-joined-sources>
#
# We track per user (rather than per (user, source)) because the
# user-visible spec is "at most twice a day per user". Single source-
# label tracking gave us a per-source cool-down which a user with 5
# sources could compound into 5 runs in a single 12-hour window —
# now one user-run touches every non-twitter source at once and
# locks the whole user out for MIN_INTERVAL_HOURS.

# Candidate queue: every user that has at least one non-twitter
# source configured. We compute the (already-comma-joined, twitter-
# stripped) source list per user inside the SQL so the runner just
# needs to pass it through to `run.py --source`. Ordered by VIP
# first, then username for a stable tiebreak.
candidate_sql() {
  cat <<SQL
SELECT u.username || E'\t' ||
       string_agg(DISTINCT key, ',' ORDER BY key) AS sources
  FROM users u
  CROSS JOIN LATERAL jsonb_object_keys(u.sources) AS key
 WHERE u.sources IS NOT NULL
   AND u.sources <> '{}'::jsonb
   AND key NOT IN (${SKIPPED_SOURCES})
   AND key NOT LIKE '${SKIPPED_PREFIX}%'
 GROUP BY u.username, u.vip
 ORDER BY u.vip DESC, u.username;
SQL
}

# Read the history file; emit the set of users whose last attempt is
# inside the cool-down window.
recent_users() {
  local cutoff
  cutoff=$(( $(date +%s) - MIN_INTERVAL_HOURS * 3600 ))
  awk -F'\t' -v c="$cutoff" '$1 >= c { print $2 }' "$HISTORY" | sort -u
}

# Print the next eligible (slug, source-list) pair as "<slug>\t<csv>".
next_task() {
  local recent
  recent=$(recent_users)
  eval "$DB_CMD \"$(candidate_sql)\"" 2>>"$LOG" \
    | awk -F'\t' -v r="$recent" '
        BEGIN { n = split(r, a, "\n"); for (i = 1; i <= n; i++) skip[a[i]] = 1 }
        !skip[$1] { print; exit }
      '
}

mark_done() {
  printf '%s\t%s\t%s\n' "$(date +%s)" "$1" "$2" >> "$HISTORY"
}

# Bounded-size rotation. Called every LOG_ROTATE_EVERY iterations.
rotate_logs() {
  # History: keep the last N rows (the oldest cool-down windows are
  # informational only; the recent_users window is < 24h so anything
  # older than that doesn't affect scheduling).
  if [ "$(wc -l < "$HISTORY")" -gt "$HISTORY_MAX_LINES" ]; then
    tail -n "$HISTORY_MAX_LINES" "$HISTORY" > "$HISTORY.tmp" \
      && mv "$HISTORY.tmp" "$HISTORY"
    log "rotated history → last $HISTORY_MAX_LINES rows"
  fi
  # Per-task stdout log: hard-truncate when it exceeds the cap.
  local rsize
  rsize=$(stat -c%s "$RUNNER_LOG" 2>/dev/null || echo 0)
  if [ "$rsize" -gt "$RUNNER_LOG_MAX_BYTES" ]; then
    : > "$RUNNER_LOG"
    log "truncated runs log (was ${rsize} bytes)"
  fi
  # Human-readable log: same treatment but a tighter cap.
  local lsize
  lsize=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "$lsize" -gt "$LOG_MAX_BYTES" ]; then
    tail -c "$((LOG_MAX_BYTES / 2))" "$LOG" > "$LOG.tmp" \
      && mv "$LOG.tmp" "$LOG"
    log "trimmed log → last $((LOG_MAX_BYTES / 2)) bytes"
  fi
}

# ── Main loop ─────────────────────────────────────────────────────
log "starting continuous pipeline (min-interval=${MIN_INTERVAL_HOURS}h per user, sleep=${SLEEP_BETWEEN}s, empty-wait=${EMPTY_QUEUE_WAIT}s)"

iter=0
while [ -f "$STOP_FLAG" ]; do
  iter=$((iter + 1))
  if [ "$((iter % LOG_ROTATE_EVERY))" = 0 ]; then
    rotate_logs
  fi

  task=$(next_task || true)
  if [ -z "$task" ]; then
    # Queue empty = every user is still inside its 12 h fetch cool-down.
    # Index upkeep is no longer this runner's concern — the indexer
    # daemon keeps the single `__all__` index current incrementally — so
    # there's nothing to drain here; just wait for a cool-down to expire.
    log "queue empty — sleeping ${EMPTY_QUEUE_WAIT}s before re-polling"
    sleep "$EMPTY_QUEUE_WAIT"
    continue
  fi
  slug=$(printf '%s' "$task" | cut -f1)
  sources=$(printf '%s' "$task" | cut -f2)

  # Belt-and-suspenders: even if the SQL filter slipped a twitter
  # variant through (e.g. an old `users.sources` row encoded under a
  # Unicode lookalike), strip any twitter-prefixed entries before
  # invoking the pipeline. If nothing is left after stripping, skip
  # the user entirely — they only had twitter configured.
  sources=$(printf '%s' "$sources" | tr ',' '\n' \
    | awk -v p="$SKIPPED_PREFIX" 'index($0, p) != 1' \
    | paste -sd, -)
  if [ -z "$sources" ]; then
    log "  ✗ $slug had only twitter-prefixed sources — skipping"
    mark_done "$slug" "(empty after twitter strip)"
    sleep "$SLEEP_BETWEEN"
    continue
  fi

  log "running $slug × {$sources}"
  # Record the attempt immediately. Even on failure we don't want
  # to retry within the cool-down — a flaky third-party source can
  # otherwise eat the queue.
  mark_done "$slug" "$sources"
  if timeout "$PER_RUN_TIMEOUT" $PY_RUN run.py "$slug" --source "$sources" \
       >>"$RUNNER_LOG" 2>&1; then
    log "  ✓ done"
  else
    log "  ✗ failed (timeout=$PER_RUN_TIMEOUT, exit=$?)"
  fi
  sleep "$SLEEP_BETWEEN"
done

log "stop flag removed — exiting"
