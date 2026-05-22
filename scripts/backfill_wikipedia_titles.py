"""One-off: recover paper titles for Wikipedia-references docs whose
stored title is the author block.

Why this script exists
----------------------
The old Wikipedia references extractor fell back to "first 12 words of
the surrounding context" when the link's anchor was too short. On a
citation list those 12 words are the author block ("Pióro, Maciej; …"),
not the paper title. The fix (in `sources/wikipedia/references.py`)
pulls the title out of the quoted region in the citation BEFORE the
noise stripper drops the quotes. New runs use it automatically.

Existing rows can't be fixed in-place from PG alone because the summary
was already stripped of quotes when it was written. So we re-fetch the
small set of Wikipedia source pages, re-run the extractor, and UPDATE
the title field where the new title is materially better than the old.

Scope
-----
- Only touches the `title` column.
- Only acts on `documents.tags @> ARRAY['wikipedia']` rows.
- Only updates when the *current* title looks like an author block
  (`Lastname, Firstname` shape) AND the freshly-extracted title differs.

Usage::

    DATABASE_URL=... uv run python scripts/backfill_wikipedia_titles.py
    DATABASE_URL=... uv run python scripts/backfill_wikipedia_titles.py --dry
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict

import psycopg

from sources.wikipedia.references import References

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"

# Same shape we use to gate the UPDATE: "Lastname, Firstname …" or
# similar author-list openings.
_AUTHOR_TITLE_RE = re.compile(r"^[A-Z][a-zA-Z\-]+,\s+[A-Z]")


def _pick_subjects(conn) -> dict[str, set[tuple[int, str]]]:
    """Return {wikipedia_subject_key → {(user_id, url), …}} for the
    set of rows that look like author-titled wikipedia citations.

    `tags[2]` is where the extractor stamps the subject (lower-cased),
    e.g. ['wikipedia', 'tri dao', 'references']. We re-map the subject
    back to the Wikipedia page title form by capitalising and
    underscoring — fits the way `References()` accepts pages.
    """
    sql = """
        SELECT user_id, url, title, tags
          FROM documents
         WHERE tags && ARRAY['wikipedia']::text[]
           AND title ~ '^[A-Z][a-zA-Z\\-]+,\\s+[A-Z]'
           AND deleted = false
    """
    out: dict[str, set[tuple[int, str]]] = defaultdict(set)
    with conn.cursor() as cur:
        cur.execute(sql)
        for user_id, url, _title, tags in cur.fetchall():
            # Subject = tags[2] (1-indexed in SQL; in Python the
            # second element of the list).
            if len(tags) < 2:
                continue
            subject = tags[1]
            # Wikipedia page titles use Title_Case_With_Underscores.
            # The extractor accepts either — we use the underscored
            # form so the URL composes cleanly.
            page = "_".join(w.capitalize() for w in subject.split())
            out[page].add((user_id, url))
    return out


def _new_titles_for_page(page: str) -> dict[str, str]:
    """Re-run the extractor on one Wikipedia page and return
    {url → new_title}. Empty when the fetch fails."""
    refs = References(pages=[page])
    # `existing_urls=None` so every external link is returned.
    data = refs(existing_urls=None)
    return {url: rec["title"] for url, rec in data.items() if rec.get("title")}


def _is_improvement(old: str, new: str) -> bool:
    """Replace `old` with `new` when:
      • The new title is non-empty and different.
      • The new title does NOT itself open with `Lastname, Firstname`.
    Otherwise leave the row alone — we don't want to swap one
    author-block title for another.
    """
    if not new or new.strip() == old.strip():
        return False
    if _AUTHOR_TITLE_RE.match(new):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="print proposed changes, don't write")
    args = ap.parse_args()

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(database_url) as conn:
        candidates = _pick_subjects(conn)
        if not candidates:
            print("Nothing to do — no author-titled wikipedia rows in scope.")
            return 0

        print(
            f"Refetching {len(candidates)} Wikipedia page(s) "
            f"to recover titles for {sum(len(v) for v in candidates.values())} row(s).\n"
        )

        total_updates = 0
        total_skipped = 0
        for i, (page, rows) in enumerate(sorted(candidates.items()), 1):
            print(f"[{i}/{len(candidates)}] {page} ({len(rows)} row(s))")
            t0 = time.perf_counter()
            try:
                new_titles = _new_titles_for_page(page)
            except Exception as e:
                print(f"  ! fetch failed: {e}")
                continue
            print(f"  fetched: {len(new_titles)} link(s), {time.perf_counter() - t0:.1f}s")

            # Build the per-row update plan for this page.
            plan: list[tuple[int, str, str, str]] = []  # (uid, url, old, new)
            with conn.cursor() as cur:
                for user_id, url in rows:
                    new = new_titles.get(url)
                    if not new:
                        continue
                    cur.execute(
                        "SELECT title FROM documents WHERE user_id = %s AND url = %s",
                        (user_id, url),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    old = row[0]
                    if _is_improvement(old, new):
                        plan.append((user_id, url, old, new))
                    else:
                        total_skipped += 1

            for user_id, url, old, new in plan:
                print(f"  • uid={user_id:<4} {url}")
                print(f"      old: {old[:90]}")
                print(f"      new: {new[:90]}")
                if not args.dry:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE documents SET title = %s, updated_at = now() " "WHERE user_id = %s AND url = %s",
                            (new, user_id, url),
                        )
                total_updates += 1
            if not args.dry:
                conn.commit()

        action = "Would update" if args.dry else "Updated"
        print(f"\n{action} {total_updates} row(s). Skipped {total_skipped} (no better title found).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
