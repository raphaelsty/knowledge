"""Third pass: fill missing `avatar` fields from twitterapi.io.

For each row in data/people/enriched_candidates.tsv whose `avatar`
column is empty, hit twitterapi.io's `/twitter/user/info` endpoint
with the candidate's Twitter handle and pull the full-resolution
profile picture URL. We reuse `fetch_twitter_avatar` from
sources.utils.popularity so the URL-cleanup rules (strip the
`_normal` suffix) stay consistent with the live pipeline.

Usage::

    TWITTERAPIIO_API_KEY=... uv run python scripts/enrich_candidates_pass3_avatar.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_REQ_DELAY_S = 0.4
_USER_AGENT = "knowledge-avatar-backfill/1.0"


def fetch_twitter_avatar(handle: str, api_key: str) -> str | None:
    """Hit twitterapi.io's `/twitter/user/info` and return the
    profile picture URL. The current API exposes the avatar at
    `data.profilePicture` (already full-resolution). Returns None
    on any failure or when the field is empty.

    Inlined here — the schema differs from the older shape used by
    `sources/utils/popularity.fetch_twitter_avatar`, which still
    looks for keys like `profile_image_url_https`. We keep that
    helper for the pipeline (where it's gated on multiple key
    fallbacks) and ship the current shape here.
    """
    if not handle:
        return None
    try:
        req = urllib.request.Request(
            f"https://api.twitterapi.io/twitter/user/info?userName={urllib.parse.quote(handle)}",
            headers={"X-API-Key": api_key, "User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status != 200:
                return None
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    # Modern field. Strip `_normal.` if present (legacy thumbnail
    # suffix) so we keep the original-size image.
    for key in ("profilePicture", "profile_image_url_https", "profileImageUrlHttps"):
        v = data.get(key)
        if isinstance(v, str) and v:
            return v.replace("_normal.", ".")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/people/enriched_candidates.tsv",
        help="input/output TSV (overwritten in-place)",
    )
    args = ap.parse_args()

    api_key = os.environ.get("TWITTERAPIIO_API_KEY") or ""
    if not api_key:
        print("TWITTERAPIIO_API_KEY missing in env — bailing.")
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    path = (repo_root / args.inp).resolve()
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        print("Empty input.")
        return 1
    cols = list(rows[0].keys())

    todo = [r for r in rows if not (r.get("avatar") or "").strip() and r.get("handle")]
    print(f"{len(rows)} rows total; {len(todo)} missing avatar — probing twitterapi.io one by one.\n")

    filled = 0
    failed = 0
    for i, r in enumerate(todo, 1):
        handle = r["handle"].strip()
        print(f"[{i}/{len(todo)}] @{handle}", end="")
        try:
            url = fetch_twitter_avatar(handle, api_key)
        except Exception as e:
            print(f"  ! error: {e}")
            failed += 1
            time.sleep(_REQ_DELAY_S)
            continue
        if url:
            r["avatar"] = url
            filled += 1
            print(f"  ✓ {url[:80]}")
        else:
            failed += 1
            print("  — no avatar returned")
        time.sleep(_REQ_DELAY_S)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"\n✓ wrote {path}\n  filled: {filled}\n  failed: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
