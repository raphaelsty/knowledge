"""CLI: fill NULL social counts for every user.

Invoked by ``make social-counts`` — picks up env vars from the Makefile's
exported ``.env``. Iterates ``users`` rows and calls
``populate_social_counts`` for anyone still missing data. Output is one
line per user with the values written (or ``-`` when nothing changed).
"""

from __future__ import annotations

import os

import psycopg

from sources.utils.popularity import populate_social_counts

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"


def main() -> None:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, name "
                "  FROM users "
                " WHERE twitter_followers IS NULL "
                "    OR github_followers  IS NULL "
                "    OR citations         IS NULL "
                " ORDER BY name"
            )
            rows = cur.fetchall()

    if not rows:
        print("All users already have twitter / github / citation counts set.")
        return

    for user_id, username, name in rows:
        written = populate_social_counts(url, user_id, display_name=name or username)
        summary = ", ".join(f"{k}={v}" for k, v in written.items()) if written else "-"
        print(f"{username:<30} {summary}")


if __name__ == "__main__":
    main()
