"""Functional helpers for the `users` table.

All functions are stateless: they take a `database_url` and perform a
side-effect on Postgres. No classes, no module state, no hidden connections.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "users.sql"


def create_users_table(database_url: str) -> None:
    """Create the `users` table and its indices if they don't exist."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def get_twitter_cursor(database_url: str, user_id: int) -> tuple[str, str]:
    """Return (newest_date, oldest_date) as ISO strings for a user's twitter cursor.

    Either or both values are empty strings when the user has never had
    a successful twitter fetch.
    """
    sql = "SELECT tweet_newest_date, tweet_oldest_date FROM users WHERE id = %s"
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
    if not row:
        return ("", "")
    newest, oldest = row
    return (newest.isoformat() if newest else "", oldest.isoformat() if oldest else "")


def list_personalities(database_url: str) -> list[dict]:
    """Return every user row in a pipeline-friendly shape.

    Sort order is "who needs work most":
      1. VIPs first.
      2. Within each tier, oldest last successful run first (NULL =
         never-run goes to the very front).

    This way `make run` (no slug) starts on the people most worth
    refreshing — VIPs that haven't been updated in a while — and the
    long tail follows in staleness order. A new VIP joins the front
    of the queue automatically.

    Fields returned: ``id``, ``slug`` (= username), ``name``,
    ``indexName``, ``sources``, ``vip``, ``last_success_at``.
    """
    sql = (
        "SELECT u.id, u.username, u.name, u.index_name, u.sources, "
        "       u.vip, "
        "       (SELECT MAX(r.finished_at) "
        "          FROM pipeline_runs r "
        "         WHERE r.user_id = u.id "
        "           AND r.status = 'success') AS last_success_at "
        "  FROM users u "
        " ORDER BY u.vip DESC, "
        "          last_success_at ASC NULLS FIRST, "
        "          u.name"
    )
    out: list[dict] = []
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for id_, username, name, index_name, sources, vip, last_success_at in cur.fetchall():
                out.append(
                    {
                        "id": id_,
                        "slug": username,
                        "name": name or username,
                        "indexName": index_name or username,
                        "sources": sources if isinstance(sources, dict) else {},
                        "vip": bool(vip),
                        "last_success_at": last_success_at,
                    }
                )
    return out


def get_social_counts(database_url: str, user_id: int) -> dict:
    """Return raw social counts + links for a user.

    Shape:
        {
            "twitter_followers": int | None,
            "github_followers":  int | None,
            "citations":         int | None,
            "links":             dict,    # JSONB
            "sources":           dict,    # JSONB (handles live here too)
        }

    Returns all-None with empty dicts when the user row doesn't exist.
    """
    sql = "SELECT twitter_followers, github_followers, citations, avatar, links, sources   FROM users WHERE id = %s"
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
    if not row:
        return {
            "twitter_followers": None,
            "github_followers": None,
            "citations": None,
            "avatar": None,
            "links": {},
            "sources": {},
        }
    tw, gh, cit, av, links, sources = row
    return {
        "twitter_followers": int(tw) if tw is not None else None,
        "github_followers": int(gh) if gh is not None else None,
        "citations": int(cit) if cit is not None else None,
        "avatar": av if av else None,
        "links": links if isinstance(links, dict) else {},
        "sources": sources if isinstance(sources, dict) else {},
    }


def set_social_counts(
    database_url: str,
    user_id: int,
    *,
    twitter_followers: int | None = None,
    github_followers: int | None = None,
    citations: int | None = None,
    avatar: str | None = None,
) -> None:
    """Write any provided social count. ``None`` means "leave as-is".

    No-op if every argument is ``None`` — lets callers probe-and-skip
    without paying for a round-trip.
    """
    fields = []
    values: list[object] = []
    if twitter_followers is not None:
        fields.append("twitter_followers = %s")
        values.append(int(twitter_followers))
    if github_followers is not None:
        fields.append("github_followers = %s")
        values.append(int(github_followers))
    if citations is not None:
        fields.append("citations = %s")
        values.append(int(citations))
    if avatar is not None:
        fields.append("avatar = %s")
        values.append(str(avatar))
    if not fields:
        return
    fields.append("updated_at = now()")
    sql = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"
    values.append(user_id)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(values))


def update_twitter_cursor(
    database_url: str,
    user_id: int,
    newest: str | None,
    oldest: str | None,
) -> None:
    """Widen the user's tweet date cursor.

    Keeps the max(newest, existing) and min(oldest, existing) so the
    stored range always covers every tweet we've ever ingested. No-op
    when both inputs are falsy.
    """
    if not newest and not oldest:
        return
    sql = (
        "UPDATE users SET "
        "  tweet_newest_date = GREATEST(COALESCE(tweet_newest_date, %s::date), %s::date), "
        "  tweet_oldest_date = LEAST(COALESCE(tweet_oldest_date, %s::date), %s::date), "
        "  updated_at = now() "
        "WHERE id = %s"
    )
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (newest or None, newest or None, oldest or None, oldest or None, user_id),
            )
