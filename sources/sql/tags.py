"""Read-side helpers for the shared tag vocabulary.

Stateless: takes a `database_url` and returns plain data. No classes,
no module state, no hidden connections.

Tag-vocabulary policy
---------------------
The tagger seeds its keyword universe from a curated tag pool. To
prevent a non-VIP user's idiosyncratic tags from leaking into other
people's libraries, we restrict the cross-user pool to **VIP users
only**, then union it with the **target user's own tags** when we
build a per-user vocabulary:

    vocab(u) = tags(u) ∪ ⋃{tags(v) : v.vip}

For a VIP user the union collapses to the VIP pool (their own tags
are already in there). For a non-VIP user u1, the vocabulary picks
up u1's personal tags plus every VIP's tags, but never tags from
other non-VIP users.

`get_shared_tags` is preserved for backwards compatibility but is
no longer used by the pipeline — `get_vip_tags` + `get_user_tags`
replace it.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "tags.sql"
VIP_SQL_PATH = Path(__file__).parent / "vip_tags.sql"
USER_SQL_PATH = Path(__file__).parent / "user_tags.sql"


def get_shared_tags(database_url: str) -> list[str]:
    """Return the alphabetical union of `tags` across all documents.

    Legacy helper. Prefer `get_vip_tags` + `get_user_tags` so non-VIP
    users don't pollute everyone else's tag vocabulary.
    """
    sql = SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]


def get_vip_tags(database_url: str) -> list[str]:
    """Return the alphabetical union of `tags` across VIP users only.

    Used as the cross-user portion of every personality's tag
    vocabulary. The pipeline computes this once per run and unions
    it with each target user's own tags before passing it down.
    """
    sql = VIP_SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]


def get_user_tags(database_url: str, user_id: int) -> list[str]:
    """Return the alphabetical set of `tags` for a single user_id."""
    sql = USER_SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            return [row[0] for row in cur.fetchall()]
