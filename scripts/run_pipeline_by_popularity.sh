#!/usr/bin/env bash
# Run the Python pipeline for every personality, popular-first.
#
# Order: derived from `users` rows that have `sources` configured,
# scored as
#   ln(twitter_followers + 1)
# + ln(github_followers  + 1)
# + 0.4 * ln(citations    + 1)
# (matches the scoring used by web/welcome.jsx for sort).
#
# Behaviour:
#   1. Fetch the slug list once and write it to /tmp/run-popularity-slugs.txt.
#   2. Run `make run SLUG=<slug>` for slug #1, fail loudly if the pipeline
#      didn't insert any documents (smoke check that fetchers work).
#   3. After the smoke passes, iterate the rest sequentially. Each per-user
#      log is appended to /tmp/run-popularity.log.
#
# Usage:
#   bash scripts/run_pipeline_by_popularity.sh           # smoke + all
#   bash scripts/run_pipeline_by_popularity.sh --resume  # skip slug #1 smoke
#   bash scripts/run_pipeline_by_popularity.sh --dry     # only print order
set -euo pipefail

DB_CMD=(docker exec -i knowledge-postgres-1 psql -U knowledge -d knowledge -tAc)
LOG=/tmp/run-popularity.log
SLUG_LIST=/tmp/run-popularity-slugs.txt

"${DB_CMD[@]}" "
SELECT username
  FROM users
 WHERE sources IS NOT NULL
   AND sources <> '{}'::jsonb
 ORDER BY (
     LN(GREATEST(COALESCE(twitter_followers, 0), 0) + 1)
   + LN(GREATEST(COALESCE(github_followers,  0), 0) + 1)
   + 0.4 * LN(GREATEST(COALESCE(citations,   0), 0) + 1)
 ) DESC NULLS LAST,
   username
" > "$SLUG_LIST"
total=$(wc -l < "$SLUG_LIST" | tr -d ' ')
echo "[run-popularity] $total slugs queued, log → $LOG"

if [[ "${1:-}" == "--dry" ]]; then
  cat "$SLUG_LIST"
  exit 0
fi

resume=0
[[ "${1:-}" == "--resume" ]] && resume=1

verify_first_run() {
  local slug=$1
  echo "[run-popularity] verifying $slug parsed something..."
  local docs
  docs=$("${DB_CMD[@]}" "
    SELECT count(*) FROM documents d JOIN users u ON u.id = d.user_id
     WHERE u.username = '$slug'
  ")
  echo "[run-popularity] $slug now has $docs documents in PG"
  if [[ "$docs" -lt 1 ]]; then
    echo "[run-popularity] FAIL — no documents inserted for $slug" >&2
    echo "[run-popularity] check $LOG; aborting batch" >&2
    exit 2
  fi
  local pr_status
  pr_status=$("${DB_CMD[@]}" "
    SELECT status FROM pipeline_runs pr JOIN users u ON u.id = pr.user_id
     WHERE u.username = '$slug'
     ORDER BY pr.started_at DESC LIMIT 1
  ")
  echo "[run-popularity] $slug pipeline_runs.status = ${pr_status:-(no row)}"
}

# Read every slug into an array up-front so the loop doesn't depend on
# any inherited stdin — `make run` invokes `uv run python run.py` which
# can drain a piped stdin and bail the loop after a single iteration.
# (`mapfile` is bash 4+ and macOS ships bash 3.2, so we do it manually.)
SLUGS=()
while IFS= read -r line; do
  SLUGS+=("$line")
done < "$SLUG_LIST"
i=0
for slug in "${SLUGS[@]}"; do
  i=$((i + 1))
  [[ -z "$slug" ]] && continue
  echo
  echo "[run-popularity] [$i/$total] $slug" | tee -a "$LOG"
  date +"[run-popularity] start %Y-%m-%d %H:%M:%S" | tee -a "$LOG"
  if make run SLUG="$slug" </dev/null >> "$LOG" 2>&1; then
    date +"[run-popularity]   ok %Y-%m-%d %H:%M:%S" | tee -a "$LOG"
  else
    echo "[run-popularity]   FAILED — see $LOG" | tee -a "$LOG"
  fi
  if [[ $i -eq 1 && $resume -eq 0 ]]; then
    verify_first_run "$slug"
  fi
done

echo
echo "[run-popularity] all $total runs complete (log: $LOG)"
