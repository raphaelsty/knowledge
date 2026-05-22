"""CLI: backfill image / video URLs into tweet documents whose summaries
were ingested before the media-extraction pipeline shipped.

For each ``documents`` row whose URL matches a Twitter status, re-asks
TwitterAPI.io for the tweet via the batched ``/twitter/tweets`` endpoint
(100 IDs per call — cheap) and prepends ``📷 <url>`` / ``🎬 <url>`` lines
to the existing summary when they aren't already there. Thread documents
get the root tweet's media; per-part media on threads we don't yet have
IDs for is out of scope.

Usage::

    make backfill-tweet-media SLUG=raphael-sourty       # one user
    make backfill-tweet-media SLUG=raphael-sourty DRY=1 # plan only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import psycopg

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
API_BASE = "https://api.twitterapi.io"
BATCH = 100

_STATUS_RE = re.compile(r"https?://(?:www\.)?(?:twitter|x)\.com/[^/]+/status/(\d+)", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="backfill_tweet_media.py")
    p.add_argument("--slug", required=True, help="Personality slug (required).")
    p.add_argument("--dry", action="store_true")
    return p.parse_args()


def _fetch_tweets(ids: list[str], api_key: str) -> list[dict]:
    qs = urllib.parse.urlencode({"tweet_ids": ",".join(ids)})
    req = urllib.request.Request(f"{API_BASE}/twitter/tweets?{qs}", headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  ! fetch error: {e}", file=sys.stderr)
        return []
    return data.get("tweets") or (data.get("data") or {}).get("tweets") or []


def _media_urls(tweet: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    media = (tweet.get("extendedEntities") or {}).get("media") or []
    for m in media:
        kind = m.get("type") or ""
        if kind == "photo":
            u = m.get("media_url_https") or m.get("media_url") or ""
            if u:
                out.append(("photo", u))
        elif kind in ("video", "animated_gif"):
            variants = (m.get("video_info") or {}).get("variants") or []
            mp4s = [v for v in variants if v.get("content_type") == "video/mp4"]
            mp4s.sort(key=lambda v: int(v.get("bitrate") or 0))
            mp4_url = (mp4s[0].get("url") if mp4s else "") or ""
            poster = m.get("media_url_https") or ""
            if poster and mp4_url:
                out.append((kind, f"{poster} | {mp4_url}"))
            elif mp4_url or poster:
                out.append((kind, mp4_url or poster))
    return out


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("TWITTERAPIIO_API_KEY")
    if not api_key:
        print("error: TWITTERAPIIO_API_KEY not set in env", file=sys.stderr)
        sys.exit(2)
    db_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT u.id, d.url, d.summary FROM documents d "
                " JOIN users u ON u.id = d.user_id "
                " WHERE u.username = %s "
                "   AND d.deleted = FALSE "
                "   AND 'twitter' = ANY(d.tags)",
                (args.slug,),
            )
            rows = cur.fetchall()

        # Map url → (user_id, summary), keyed by tweet id when parseable.
        by_id: dict[str, tuple[int, str, str]] = {}
        for uid, url, summary in rows:
            m = _STATUS_RE.match(url)
            if not m:
                continue
            tid = m.group(1)
            s = summary or ""
            # Skip docs that already carry a photo (📷 pbs.twimg.com) OR
            # a video stored in the NEW "poster | mp4" format. Docs that
            # only have an old-style "🎬 <mp4>" (no poster) still need
            # work: the renderer needs the poster, so re-process those.
            has_photo = "📷 https://pbs.twimg.com/" in s
            has_new_video = "🎬 https://pbs.twimg.com/" in s and " | " in s
            if has_photo or has_new_video:
                continue
            by_id[tid] = (uid, url, s)
        ids = list(by_id.keys())
        print(f"Inspected {len(rows)} twitter docs, {len(ids)} need media backfill.")
        if not ids:
            return
        if args.dry:
            for tid in ids[:15]:
                _, url, _ = by_id[tid]
                print(f"  - {url}")
            if len(ids) > 15:
                print(f"  … (+{len(ids) - 15} more)")
            print("\n--dry: no API / DB calls.")
            return

        fixed = no_media = 0
        for i in range(0, len(ids), BATCH):
            chunk = ids[i : i + BATCH]
            tweets = _fetch_tweets(chunk, api_key)
            tweets_by_id = {str(t.get("id") or ""): t for t in tweets}
            for tid in chunk:
                tweet = tweets_by_id.get(tid)
                if not tweet:
                    no_media += 1
                    continue
                media = _media_urls(tweet)
                if not media:
                    no_media += 1
                    continue
                uid, url, summary = by_id[tid]
                # Strip any pre-existing 📷/🎬 marker lines from the
                # summary so we don't double them up when re-running
                # against rows that received the old (mp4-only)
                # encoding.
                cleaned = re.sub(
                    r"^\s*[📷🎬]\s+https?://\S+(?: \| https?://\S+)?\s*\n?",
                    "",
                    summary or "",
                    flags=re.MULTILINE,
                ).strip()
                marker_lines = "\n".join(f"{'📷' if kind == 'photo' else '🎬'} {u}" for kind, u in media)
                new_summary = f"{marker_lines}\n\n{cleaned}" if cleaned else marker_lines
                with conn.cursor() as cur:
                    # Flip `indexed = FALSE` so the next pipeline pass
                    # re-indexes the doc — otherwise the ColBERT
                    # sidecar keeps the old summary and the new media
                    # URLs never reach the search results / browse
                    # cards.
                    cur.execute(
                        "UPDATE documents SET summary = %s, indexed = FALSE, "
                        "       updated_at = now()"
                        "  WHERE user_id = %s AND url = %s",
                        (new_summary, uid, url),
                    )
                conn.commit()
                fixed += 1
            time.sleep(0.3)
            print(f"  batch {i // BATCH + 1}: +{len([t for t in chunk if t in tweets_by_id])} returned")

        print(f"\nDone. fixed={fixed} no_media={no_media}")


if __name__ == "__main__":
    main()
