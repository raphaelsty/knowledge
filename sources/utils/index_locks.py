"""PG advisory locks for per-user index writes.

Why this module exists
----------------------
Two code paths can mutate a user's ColBERT index:

  * ``sources.utils.client.run_pipeline`` (the regular fetch + embed
    pass triggered by ``make run`` / continuous_pipeline.sh).
  * ``scripts/repair_indexes_daemon.py`` (the parallel repair sweep
    that drops + rebuilds broken indexes).

If they touch the same user at the same instant — e.g. the daemon
DELETEs the index while the pipeline is mid-``update_with_encoding``
— the on-disk state goes inconsistent and the next search returns
``HTTP 500 "No data to merge"``. That failure mode is exactly what
the daemon is *trying* to fix, so a race here is doubly bad.

Postgres advisory locks are the cheapest cross-process mutex we
already have: every component of the stack opens a PG connection
anyway, and the lock is released automatically when the holding
session closes (so an OOM-kill or laptop sleep can never wedge the
lock forever).

Key layout
~~~~~~~~~~
``pg_advisory_lock`` takes two ``int4`` keys. We pin the high key to
a module-specific namespace (``_NAMESPACE``) so this lock space can
coexist with any future advisory-lock use without collisions, and
spend the low key on the user_id (BIGINT in Postgres, but only the
low 32 bits are needed — `users.id` is a BIGSERIAL but realistically
fits in 31 bits for the foreseeable future).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

# Arbitrary but stable namespace tag for this feature. Picked so it
# is unlikely to collide with any other advisory-lock convention in
# the codebase; if a second feature ever shows up here, give it its
# own constant and document it.
_NAMESPACE = 0x1DEC1  # "index" — leetspeak-ish, just needs to be unique


class IndexBusy(RuntimeError):
    """Another process holds the index lock for this user."""


@contextmanager
def acquire_index_lock(
    database_url: str,
    user_id: int,
    *,
    blocking: bool = True,
) -> Iterator[None]:
    """Hold a per-user advisory lock for the duration of the ``with`` block.

    Parameters
    ----------
    database_url
        Standard Postgres URL. A short-lived dedicated connection is
        opened for the lock so the caller's own conn(s) don't get
        accidentally tangled in long-running transactions.
    user_id
        ``users.id``. Used as the low 32 bits of the advisory-lock key.
    blocking
        * ``True`` (default) — wait until the lock is granted. Use this
          on the *pipeline* side: it must complete its indexing pass.
        * ``False`` — fail fast with :class:`IndexBusy` if held by
          someone else. Use this on the *repair* side: it should just
          move on to another user instead of stalling.

    Raises
    ------
    IndexBusy
        Only when ``blocking=False`` and the lock is currently held.
    """
    conn = psycopg.connect(database_url, autocommit=True)
    held = False
    try:
        with conn.cursor() as cur:
            if blocking:
                cur.execute(
                    "SELECT pg_advisory_lock(%s, %s)",
                    (_NAMESPACE, int(user_id)),
                )
                held = True
            else:
                cur.execute(
                    "SELECT pg_try_advisory_lock(%s, %s)",
                    (_NAMESPACE, int(user_id)),
                )
                row = cur.fetchone()
                held = bool(row and row[0])
                if not held:
                    raise IndexBusy(f"index lock held for user_id={user_id}")
        yield
    finally:
        # Release exactly when we acquired. Closing the connection
        # would also drop the lock (advisory locks are session-scoped),
        # but explicit unlock is friendlier to long-lived poolers and
        # surfaces "released a lock we didn't hold" mistakes during
        # development as a noisy warning instead of silent.
        try:
            if held:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_unlock(%s, %s)",
                        (_NAMESPACE, int(user_id)),
                    )
        finally:
            conn.close()
