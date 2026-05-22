"""
Stack Overflow answer extractor.

Fetches a user's top answers from Stack Overflow and extracts the
parent question URL. If someone answered a question, that topic is
part of their expertise.

Uses the public Stack Exchange API (no auth needed for basic usage,
30 req/min without key).
"""

import gzip
import json
import time
import urllib.request

__all__ = ["Answers"]

_API_BASE = "https://api.stackexchange.com/2.3"
_DELAY = 2.0  # seconds between requests (conservative)


def _fetch_json(url: str, timeout: int = 15) -> dict:
    """Fetch from Stack Exchange API (returns gzip-compressed JSON)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Knowledge/1.0", "Accept-Encoding": "gzip"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        try:
            return json.loads(gzip.decompress(raw))
        except (gzip.BadGzipFile, OSError):
            return json.loads(raw)


class Answers:
    """
    Extract questions answered by a Stack Overflow user.

    Parameters
    ----------
    user_id : int | None
        Stack Overflow numeric user ID. If provided, fetches directly.
    username : str | None
        Display name to search for. Uses the top result by reputation.
    site : str, default="stackoverflow"
        Stack Exchange site (e.g. "stackoverflow", "serverfault").
    max_pages : int, default=5
        Max pages to fetch (100 answers per page).
    min_score : int, default=1
        Only include answers with at least this score.
    """

    def __init__(
        self,
        user_id: int | None = None,
        username: str | None = None,
        site: str = "stackoverflow",
        max_pages: int = 5,
        min_score: int = 1,
    ):
        self.user_id = user_id
        self.username = username
        self.site = site
        self.max_pages = max_pages
        self.min_score = min_score

    def _resolve_user_id(self) -> int | None:
        """Search for a user by name and return their numeric ID."""
        if self.user_id:
            return self.user_id
        if not self.username:
            return None

        url = (
            f"{_API_BASE}/users?order=desc&sort=reputation"
            f"&inname={urllib.parse.quote(self.username)}"
            f"&site={self.site}&pagesize=3"
        )
        try:
            data = _fetch_json(url)
            items = data.get("items", [])
            if items:
                best = items[0]
                print(f"    Found SO user: {best['display_name']} (id={best['user_id']}, rep={best['reputation']})")
                return best["user_id"]
        except Exception as e:
            print(f"    SO user search failed: {e}")
        return None

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        """Fetch answers and extract question URLs."""

        uid = self._resolve_user_id()
        if not uid:
            print("    Could not find Stack Overflow user")
            return {}

        import re

        print(f"    Fetching Stack Overflow answers for user {uid}...")
        # Phase 1: collect all answer data (question IDs, dates, bodies)
        answers_by_qid: dict[int, dict] = {}

        for page in range(1, self.max_pages + 1):
            url = (
                f"{_API_BASE}/users/{uid}/answers"
                f"?order=desc&sort=votes&site={self.site}"
                f"&pagesize=100&page={page}&filter=withbody"
            )

            try:
                result = _fetch_json(url)
            except Exception as e:
                print(f"    SO API error (page {page}): {e}")
                break

            items = result.get("items", [])
            if not items:
                break

            for answer in items:
                score = answer.get("score", 0)
                if score < self.min_score:
                    continue

                qid = answer.get("question_id")
                if not qid or qid in answers_by_qid:
                    continue

                question_url = f"https://stackoverflow.com/q/{qid}"
                if existing_urls and question_url in existing_urls:
                    continue

                created = answer.get("creation_date", 0)
                date = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else ""

                body = answer.get("body", "")
                summary = re.sub(r"<[^>]+>", " ", body)
                summary = re.sub(r"\s+", " ", summary).strip()
                if len(summary) > 200:
                    summary = summary[:197] + "..."

                answers_by_qid[qid] = {"url": question_url, "date": date, "summary": summary}

            if not result.get("has_more", False):
                break

            time.sleep(_DELAY)

        if not answers_by_qid:
            print("    0 questions from answers")
            return {}

        # Phase 2: batch-fetch question titles (up to 100 per request)
        qids = list(answers_by_qid.keys())
        for i in range(0, len(qids), 100):
            batch = qids[i : i + 100]
            ids_str = ";".join(str(q) for q in batch)
            url = f"{_API_BASE}/questions/{ids_str}?site={self.site}&filter=!nNPvSNOTRz&pagesize=100"
            try:
                result = _fetch_json(url)
                for q in result.get("items", []):
                    qid = q.get("question_id")
                    if qid in answers_by_qid:
                        answers_by_qid[qid]["title"] = q.get("title", "")
            except Exception as e:
                print(f"    SO question fetch error: {e}")

            if i + 100 < len(qids):
                time.sleep(_DELAY)

        # Phase 3: build final data
        data: dict[str, dict] = {}
        for qid, info in answers_by_qid.items():
            title = info.get("title", "")
            data[info["url"]] = {
                "title": title or f"Stack Overflow #{qid}",
                "summary": info["summary"],
                "date": info["date"],
                "tags": ["stackoverflow"],
            }

        print(f"    {len(data)} questions from answers")
        return data


# ─────────────────────────────────────────────────────────────────────────
# Favorites — requires a Stack Overflow access_token (scope private_info).
# Paired with the app quota "key" it counts against the 10 000/day bucket.
# ─────────────────────────────────────────────────────────────────────────


class Favorites:
    """Extract URLs from the user's favorited (bookmarked) questions.

    Stack Exchange's /me/favorites is scoped to the OAuth token's user,
    so we don't need a user_id. Uses the same gzip-aware fetcher as
    Answers for consistency.

    Parameters
    ----------
    access_token : str
        OAuth access token (scope private_info).
    key : str
        App quota key from StackApps (lifts the daily quota cap).
    max_pages : int, default=5
        Max pages (100 favorites per page).
    """

    def __init__(
        self,
        access_token: str,
        key: str,
        max_pages: int = 5,
        site: str = "stackoverflow",
    ):
        self.access_token = access_token
        self.key = key
        self.max_pages = max_pages
        self.site = site

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        data: dict[str, dict] = {}
        import urllib.parse

        for page in range(1, self.max_pages + 1):
            params = urllib.parse.urlencode(
                {
                    "site": self.site,
                    "access_token": self.access_token,
                    "key": self.key,
                    "page": page,
                    "pagesize": 100,
                    "order": "desc",
                    "sort": "added",
                    "filter": "default",
                }
            )
            url = f"{_API_BASE}/me/favorites?{params}"
            try:
                payload = _fetch_json(url)
            except Exception as e:
                print(f"    Stack Overflow /me/favorites error: {e}")
                break
            items = payload.get("items") or []
            if not items:
                break
            for it in items:
                link = it.get("link")
                if not link:
                    continue
                if existing_urls and link in existing_urls:
                    continue
                if link in data:
                    continue
                title = (it.get("title") or "").strip()
                created = it.get("creation_date", 0)
                date = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else ""
                tags = it.get("tags") or []
                data[link] = {
                    "title": title,
                    "summary": f"Stack Overflow favorite: {title}",
                    "date": date,
                    "tags": ["stackoverflow", *tags[:3]],
                }
            if not payload.get("has_more"):
                break
            time.sleep(_DELAY)

        print(f"    {len(data)} URLs from Stack Overflow favorites")
        return data


# ─────────────────────────────────────────────────────────────────────────
# Questions — the user's own questions. Public; no auth needed.
# Complements Answers (topics they've answered) with Questions (topics
# they've wondered about).
# ─────────────────────────────────────────────────────────────────────────


class Questions:
    """Fetch questions asked by a Stack Overflow user.

    Parameters
    ----------
    user_id, username : see Answers
    site, max_pages, min_score : see Answers
    """

    def __init__(
        self,
        user_id: int | None = None,
        username: str | None = None,
        site: str = "stackoverflow",
        max_pages: int = 5,
        min_score: int = 0,
    ):
        self.user_id = user_id
        self.username = username
        self.site = site
        self.max_pages = max_pages
        self.min_score = min_score

    def _resolve_user_id(self) -> int | None:
        if self.user_id:
            return self.user_id
        if not self.username:
            return None
        import urllib.parse as _up

        # Fallback: search by display name, take the top-reputation hit.
        url = (
            f"{_API_BASE}/users?order=desc&sort=reputation"
            f"&inname={_up.quote(self.username)}&site={self.site}&pagesize=1"
        )
        try:
            items = _fetch_json(url).get("items") or []
        except Exception:
            return None
        return items[0].get("user_id") if items else None

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        uid = self._resolve_user_id()
        if not uid:
            return {}

        import re as _re

        data: dict[str, dict] = {}
        print(f"    Fetching Stack Overflow questions for user {uid}...")
        for page in range(1, self.max_pages + 1):
            url = (
                f"{_API_BASE}/users/{uid}/questions"
                f"?order=desc&sort=votes&site={self.site}"
                f"&pagesize=100&page={page}&filter=withbody"
            )
            try:
                result = _fetch_json(url)
            except Exception as e:
                print(f"    SO API error (questions p{page}): {e}")
                break
            for q in result.get("items") or []:
                score = q.get("score", 0)
                if score < self.min_score:
                    continue
                link = q.get("link")
                if not link:
                    continue
                if existing_urls and link in existing_urls:
                    continue
                if link in data:
                    continue
                title = (q.get("title") or "").strip()
                body = q.get("body", "")
                summary = _re.sub(r"<[^>]+>", " ", body)
                summary = _re.sub(r"\s+", " ", summary).strip()
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                created = q.get("creation_date", 0)
                date = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else ""
                tags = q.get("tags") or []
                data[link] = {
                    "title": title,
                    "summary": summary or title,
                    "date": date,
                    "tags": ["stackoverflow", *tags[:3]],
                }
            if not result.get("has_more", False):
                break
            time.sleep(_DELAY)

        print(f"    {len(data)} URLs from Stack Overflow questions")
        return data
