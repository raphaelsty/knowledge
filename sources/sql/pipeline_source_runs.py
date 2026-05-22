"""Helpers for `pipeline_source_runs` — per-source breakdown of a run.

The `track_source` context manager is the public interface used by
`run_pipeline` to wrap each fetcher block. It:

  • times the block,
  • catches exceptions, records a `failed` row, and SUPPRESSES the
    exception so the next source still runs,
  • on clean exit records a `success` (or `skipped` if the caller
    asked for it via `.skip(reason)`).

It also appends a `(label, duration)` tuple to an optional `timings`
list so the existing per-step bar chart in the admin keeps rendering
without a separate timing pass.
"""

from __future__ import annotations

import time
from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "pipeline_source_runs.sql"


def create_pipeline_source_runs_table(database_url: str) -> None:
    """Create the `pipeline_source_runs` table and its indices."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def record_source_run(
    database_url: str,
    run_id: int,
    user_id: int,
    source: str,
    detail: str | None,
    status: str,
    duration_secs: float,
    new_documents: int,
    error: str | None,
) -> None:
    """Insert a sealed per-source row. Best-effort — never raises."""
    if run_id <= 0:
        return
    sql = (
        "INSERT INTO pipeline_source_runs "
        "(run_id, user_id, source, detail, status, started_at, finished_at, "
        " duration_secs, new_documents, error) "
        "VALUES (%s, %s, %s, %s, %s, "
        "        now() - make_interval(secs => %s), now(), %s, %s, %s)"
    )
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        int(run_id),
                        int(user_id),
                        source,
                        detail,
                        status,
                        float(duration_secs),
                        float(duration_secs),
                        int(new_documents),
                        error,
                    ),
                )
    except Exception as exc:
        print(f"  (pipeline_source_runs insert failed: {exc})")


class track_source:
    """Context manager that records one row in `pipeline_source_runs`.

    Usage:

        with track_source(db, run_id, user_id, "github", "@simonw", timings) as ts:
            added = fetcher(...)
            ts.add(len(added))

    On a raised exception the row is recorded as `failed` and the
    exception is SUPPRESSED so the next fetcher block still runs.
    Call `ts.skip("no api key")` from inside the block to record the
    block as `skipped`.
    """

    def __init__(
        self,
        database_url: str,
        run_id: int,
        user_id: int,
        source: str,
        detail: str = "",
        timings: list | None = None,
        timing_label: str | None = None,
    ) -> None:
        self.database_url = database_url
        self.run_id = run_id
        self.user_id = user_id
        self.source = source
        self.detail = detail
        self.new_documents = 0
        self._t0 = 0.0
        self._timings = timings
        self._timing_label = timing_label or (f"Fetch {source}" if not detail else f"Fetch {source} ({detail})")
        self._skipped = False
        self._skip_reason: str | None = None

    def __enter__(self) -> track_source:
        self._t0 = time.perf_counter()
        return self

    def add(self, n: int) -> None:
        """Bump the new-document counter for this fetcher."""
        self.new_documents += int(n or 0)

    def skip(self, reason: str = "") -> None:
        """Mark this block as deliberately skipped (no creds, cooldown)."""
        self._skipped = True
        self._skip_reason = reason

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._t0
        if self._timings is not None:
            self._timings.append((self._timing_label, duration))

        if self._skipped:
            status = "skipped"
            error: str | None = self._skip_reason
        elif exc_type is None:
            status = "success"
            error = None
        else:
            status = "failed"
            # Keep error short — full traceback is not what the dashboard
            # needs. Strip newlines so the panel renders predictably.
            error = f"{exc_type.__name__}: {exc_val}".strip().replace("\n", " ")
            print(f"    [{self.source}] failed: {error}")

        record_source_run(
            self.database_url,
            self.run_id,
            self.user_id,
            self.source,
            self.detail or None,
            status,
            duration,
            self.new_documents,
            error,
        )
        # Suppress exception so subsequent fetchers still run.
        return True
