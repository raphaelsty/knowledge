"""Functional helpers for the `pipeline_runs` table.

The table is a live tracker: callers INSERT a row at run start, UPDATE
its stage as work progresses, and SEAL it with `finish_pipeline_run` at
the end.

Stateless: take a `database_url` and perform a side-effect or return data.
Requires the `users` table to exist first (FK).
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "pipeline_runs.sql"


def create_pipeline_runs_table(database_url: str) -> None:
    """Create the `pipeline_runs` table and its indices if they don't exist."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def start_pipeline_run(database_url: str, user_id: int, trigger: str = "python") -> int:
    """Insert a `running` row and return its id.

    Call this as the first action of a parsing run. Subsequent progress
    goes through `update_pipeline_run_stage`; completion through
    `finish_pipeline_run`. On an exception the caller should still call
    `finish_pipeline_run(..., success=False, error=...)` so the row
    doesn't hang in `running` forever.
    """
    sql = "INSERT INTO pipeline_runs (user_id, trigger, status) " "VALUES (%s, %s, 'running') RETURNING id"
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, trigger))
            row = cur.fetchone()
            return int(row[0]) if row else 0


def update_pipeline_run_stage(database_url: str, run_id: int, stage: str) -> None:
    """Update the current stage of a running pipeline row.

    Cheap, called at each `step()` boundary in the pipeline so a
    dashboard can show "user X is currently in stage Y".
    """
    if run_id <= 0:
        return
    sql = "UPDATE pipeline_runs SET stage = %s WHERE id = %s"
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (stage, run_id))


def finish_pipeline_run(
    database_url: str,
    run_id: int,
    *,
    success: bool,
    new_documents: int = 0,
    total_documents: int = 0,
    duration_secs: float = 0.0,
    timings: list | None = None,
    error: str | None = None,
) -> None:
    """Seal a running pipeline row with its final stats.

    Sets `status`, `finished_at = now()`, `duration_secs`, stats, and
    clears `stage`. When `success = False`, also stores `error` (which
    should be a short description — the full traceback is not needed).
    """
    if run_id <= 0:
        return
    sql = (
        "UPDATE pipeline_runs SET "
        "   status           = %s, "
        "   stage            = NULL, "
        "   finished_at      = now(), "
        "   duration_secs    = %s, "
        "   new_documents    = %s, "
        "   total_documents  = %s, "
        "   timings          = %s::jsonb, "
        "   error            = %s "
        " WHERE id = %s"
    )
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    "success" if success else "failed",
                    float(duration_secs),
                    int(new_documents),
                    int(total_documents),
                    json.dumps(timings or []),
                    error,
                    run_id,
                ),
            )


def cleanup_stale_runs(database_url: str, max_age_hours: float = 2.0) -> int:
    """Mark `running` rows older than `max_age_hours` as `failed`.

    If the Python process that owned a row crashed hard (segfault,
    OOM, SIGKILL), the row would stay `running` forever. Calling this
    at the start of every pipeline invocation is a cheap safety net —
    any row we find is necessarily orphaned. Returns the count of rows
    swept.
    """
    sql = (
        "UPDATE pipeline_runs SET "
        "   status      = 'failed', "
        "   finished_at = COALESCE(finished_at, now()), "
        "   stage       = NULL, "
        "   error       = COALESCE(error, 'stale — previous run did not finish') "
        " WHERE status = 'running' "
        "   AND started_at < now() - make_interval(secs => %s) "
        "RETURNING id"
    )
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (float(max_age_hours) * 3600.0,))
            return len(cur.fetchall())
