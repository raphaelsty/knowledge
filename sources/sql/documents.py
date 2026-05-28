"""Functional helpers for the `documents` table.

All functions are stateless: they take a `database_url` and perform a
side-effect on Postgres or return plain data. No classes, no module
state, no hidden connections.

Requires `users` to exist first — `documents.user_id` FKs into `users(id)`.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import psycopg

SQL_PATH = Path(__file__).parent / "documents.sql"


def create_documents_table(database_url: str) -> None:
    """Create the `documents` table and its indices if they don't exist."""
    with psycopg.connect(database_url) as conn:
        conn.execute(SQL_PATH.read_text(encoding="utf-8"))


def load_documents(database_url: str, user_id: int) -> dict[str, dict]:
    """Return all documents for a user, keyed by url.

    Same shape as the legacy `database.json` dict — keys are URLs, values
    have ``title``, ``summary``, ``date``, ``tags``, ``extra-tags``,
    ``source``, ``source_url``, ``linked_urls``, ``link_hosts``, plus the
    engagement signals (``citation_count`` / ``twitter_*``). NULL columns
    are returned as ``None`` so the caller can tell "never fetched" from
    "fetched and zero".
    """
    sql = (
        "SELECT url, title, summary, date, tags, extra_tags, source, "
        "source_url, linked_urls, link_hosts, "
        "citation_count, twitter_likes, twitter_retweets, twitter_replies, "
        "twitter_quotes, twitter_views, twitter_bookmarks, referenced_author "
        "FROM documents WHERE user_id = %s"
    )
    out: dict[str, dict] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            for (
                url,
                title,
                summary,
                dt,
                tags,
                extra_tags,
                source,
                source_url,
                linked_urls,
                link_hosts,
                citation_count,
                tw_likes,
                tw_retweets,
                tw_replies,
                tw_quotes,
                tw_views,
                tw_bookmarks,
                referenced_author,
            ) in cur.fetchall():
                out[url] = {
                    "title": title,
                    "summary": summary,
                    "date": dt.isoformat() if dt else "",
                    "tags": list(tags or []),
                    "extra-tags": list(extra_tags or []),
                    "source": source or "",
                    "source_url": source_url,
                    "linked_urls": linked_urls or [],
                    "link_hosts": list(link_hosts or []),
                    "citation_count": citation_count,
                    "twitter_likes": tw_likes,
                    "twitter_retweets": tw_retweets,
                    "twitter_replies": tw_replies,
                    "twitter_quotes": tw_quotes,
                    "twitter_views": tw_views,
                    "twitter_bookmarks": tw_bookmarks,
                    "referenced_author": referenced_author,
                }
    return out


def upsert_documents(database_url: str, user_id: int, docs: dict[str, dict]) -> None:
    """Bulk-upsert ``{url: doc}`` for a user. No-op when ``docs`` is empty.

    Doc dict keys read: ``title``, ``summary``, ``date`` (ISO-ish string),
    ``tags``, ``extra-tags`` (or ``extra_tags``), ``source``, ``source_url``.

    On INSERT the new ``indexed`` flag starts FALSE (default) so the
    pipeline picks the row up for embedding. On UPDATE we flip it back
    to FALSE only when the title/summary/tags changed materially —
    stale embeddings for identical content would just waste work.
    """
    if not docs:
        return

    def _linked_urls_for(doc: dict) -> str:
        """Serialise the doc's `linked_urls` into the JSON string PG
        expects. Accepts pre-built dicts (the twitter pipeline) or a
        legacy doc with no inline links (empty array). We never accept
        a non-list value — anything else would corrupt the column."""
        v = doc.get("linked_urls")
        if isinstance(v, list):
            return json.dumps(v)
        return "[]"

    def _engagement_int(doc: dict, key: str) -> int | None:
        """Coerce a doc-side engagement field to a non-negative int or None.

        Sources hand us ints, strings, or absent keys depending on what
        the upstream API ships. We treat ``None`` / missing / negative as
        "not measured" so the column stays NULL (and merges below preserve
        the prior value) instead of clobbering a real count with 0.
        """
        v = doc.get(key)
        if v is None:
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return n if n >= 0 else None

    def _any_engagement(doc: dict) -> bool:
        """True iff the doc carries at least one engagement signal — so we
        only stamp engagement_updated_at on rows where the sync actually
        fetched a count, not on every plain upsert."""
        return any(
            _engagement_int(doc, k) is not None
            for k in (
                "citation_count",
                "twitter_likes",
                "twitter_retweets",
                "twitter_replies",
                "twitter_quotes",
                "twitter_views",
                "twitter_bookmarks",
            )
        )

    rows = [
        (
            user_id,
            url,
            (doc.get("title") or "").strip(),
            (doc.get("summary") or "").strip(),
            _parse_date(doc.get("date") or ""),
            list(doc.get("tags") or []),
            list(doc.get("extra-tags") or doc.get("extra_tags") or []),
            (doc.get("source") or "").strip(),
            doc.get("source_url"),
            _linked_urls_for(doc),
            list(doc.get("link_hosts") or []),
            _engagement_int(doc, "citation_count"),
            _engagement_int(doc, "twitter_likes"),
            _engagement_int(doc, "twitter_retweets"),
            _engagement_int(doc, "twitter_replies"),
            _engagement_int(doc, "twitter_quotes"),
            _engagement_int(doc, "twitter_views"),
            _engagement_int(doc, "twitter_bookmarks"),
            _any_engagement(doc),
            # referenced_author: keep None for "field absent on this
            # doc" so the COALESCE merge below preserves the prior
            # value. The compose_thread_doc path always stamps a
            # value (handle, or '' for "checked, none"), so a real
            # twitter sync never sends None.
            doc.get("referenced_author"),
        )
        for url, doc in docs.items()
    ]
    sql = (
        "INSERT INTO documents "
        "  (user_id, url, title, summary, date, tags, extra_tags, "
        "   source, source_url, linked_urls, link_hosts, "
        "   citation_count, twitter_likes, twitter_retweets, twitter_replies, "
        "   twitter_quotes, twitter_views, twitter_bookmarks, engagement_updated_at, "
        "   referenced_author) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, "
        "        %s, %s, %s, %s, %s, %s, %s, "
        "        CASE WHEN %s THEN now() ELSE NULL END, "
        "        %s) "
        "ON CONFLICT (user_id, url) DO UPDATE SET "
        "   title       = EXCLUDED.title, "
        "   summary     = EXCLUDED.summary, "
        "   date        = EXCLUDED.date, "
        "   tags        = EXCLUDED.tags, "
        "   extra_tags  = EXCLUDED.extra_tags, "
        "   source      = EXCLUDED.source, "
        "   source_url  = EXCLUDED.source_url, "
        # Same merge rule the bulk endpoint uses: keep the previous
        # rich payload if the new doc shipped an empty `linked_urls`
        # array (e.g. a sync from an older client). The pipeline
        # always sends the current value so a real edit wins.
        "   linked_urls = CASE "
        "       WHEN jsonb_array_length(EXCLUDED.linked_urls) > 0 "
        "           THEN EXCLUDED.linked_urls "
        "       ELSE documents.linked_urls "
        "   END, "
        "   link_hosts  = CASE "
        "       WHEN cardinality(EXCLUDED.link_hosts) > 0 "
        "           THEN EXCLUDED.link_hosts "
        "       ELSE documents.link_hosts "
        "   END, "
        # Engagement merges: only overwrite when the incoming row
        # actually measured the signal. A non-twitter sync (e.g. a
        # zotero re-fetch of a tweeted URL) ships NULLs and must
        # not clobber the like count we cached last twikit run.
        "   citation_count   = COALESCE(EXCLUDED.citation_count,   documents.citation_count), "
        "   twitter_likes    = COALESCE(EXCLUDED.twitter_likes,    documents.twitter_likes), "
        "   twitter_retweets = COALESCE(EXCLUDED.twitter_retweets, documents.twitter_retweets), "
        "   twitter_replies  = COALESCE(EXCLUDED.twitter_replies,  documents.twitter_replies), "
        "   twitter_quotes   = COALESCE(EXCLUDED.twitter_quotes,   documents.twitter_quotes), "
        "   twitter_views    = COALESCE(EXCLUDED.twitter_views,    documents.twitter_views), "
        "   twitter_bookmarks = COALESCE(EXCLUDED.twitter_bookmarks, documents.twitter_bookmarks), "
        "   engagement_updated_at = COALESCE(EXCLUDED.engagement_updated_at, documents.engagement_updated_at), "
        "   referenced_author = COALESCE(EXCLUDED.referenced_author, documents.referenced_author), "
        "   indexed     = CASE "
        "       WHEN documents.title   IS DISTINCT FROM EXCLUDED.title "
        "         OR documents.summary IS DISTINCT FROM EXCLUDED.summary "
        "         OR documents.tags    IS DISTINCT FROM EXCLUDED.tags "
        "         OR documents.extra_tags IS DISTINCT FROM EXCLUDED.extra_tags "
        "         OR documents.linked_urls IS DISTINCT FROM EXCLUDED.linked_urls "
        "       THEN FALSE "
        "       ELSE documents.indexed "
        "   END, "
        # A real sync just confirmed this doc — promote it away from
        # the favorite-only lifecycle so a later un-upvote no longer
        # deletes the row. Mirrors the same rule in the Rust bulk_save
        # path so both ingestion routes agree.
        "   created_via_favorite = FALSE, "
        "   updated_at  = now()"
    )
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)


def load_unindexed_documents(database_url: str, user_id: int) -> dict[str, dict]:
    """Return only the docs still waiting to be embedded, keyed by url.

    Same shape as ``load_documents`` — caller passes the text + metadata
    straight to the ColBERT API.
    """
    sql = (
        "SELECT url, title, summary, date, tags, extra_tags, source, source_url "
        "FROM documents WHERE user_id = %s AND indexed = FALSE"
    )
    out: dict[str, dict] = {}
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            for url, title, summary, dt, tags, extra_tags, source, source_url in cur.fetchall():
                out[url] = {
                    "title": title,
                    "summary": summary,
                    "date": dt.isoformat() if dt else "",
                    "tags": list(tags or []),
                    "extra-tags": list(extra_tags or []),
                    "source": source or "",
                    "source_url": source_url,
                }
    return out


def mark_documents_indexed(database_url: str, user_id: int, urls: list[str]) -> None:
    """Flip `indexed = TRUE` for the given URLs after a successful embed push."""
    if not urls:
        return
    sql = "UPDATE documents SET indexed = TRUE, updated_at = now() WHERE user_id = %s AND url = ANY(%s::text[])"
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, urls))


def _parse_date(s: str) -> date | None:
    """Best-effort YYYY-MM-DD parse, capped at today.

    Some upstream sources (RSS feeds with bad clocks, sitemaps with
    placeholder lastmods, scraped pages with future-dated articles)
    occasionally hand us dates in the future. Storing those breaks
    the date-sorted views — they pin to the top of every list. Cap
    here so the rule applies to every fetcher in one place, no
    matter how the date came in.
    """
    if not s:
        return None
    parsed: date | None = None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%B %d, %Y"):
        try:
            parsed = datetime.strptime(s, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    today = date.today()
    return today if parsed > today else parsed
