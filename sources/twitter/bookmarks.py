"""Twitter/X cookie-authenticated extractor.

Uses twikit (session-cookie auth) instead of api.twitterapi.io so we
can hit user-private endpoints (`/2/users/:id/bookmarks`,
`/get_user_tweets`). Runs the async twikit client inside a small
asyncio.run wrapper so the rest of the pipeline stays synchronous.

Two streams, both opt-in via flags (default both on):
  - **Bookmarks** — tweets the cookie-owner bookmarked (auth-user
    only; never available in target-user mode).
  - **Tweets**    — posts by the resolved user (the cookie owner in
    auth mode, or `target_username` when set).

Likes are intentionally not fetched: X only exposes the
authenticated user's own liked-tweets timeline, so in target-user
mode (the operator browsing someone else's library) there's nothing
useful to pull.

Output shape — kept in lockstep with the twitterapi.io ``Tweets``
extractor — is **one document per tweet**, never one-doc-per-URL.
Every external URL the tweet links to lives on the parent tweet doc
under ``linked_urls`` (a list of ``{url, host, title, summary,
image}`` dicts) plus ``link_hosts`` (a string array used by the
source-chip filter). Retweets recurse into the inner tweet so the
card surfaces the original's full text, photos, and videos — same
self-sufficient summary the twitterapi.io path produces.

Tags carry the stream of origin (``twitter-bookmark``) so the
filter chips can distinguish bookmarks from regular tweets.
"""

from __future__ import annotations

import asyncio
import datetime
import re
import time

from twikit import Client
from twikit.errors import Forbidden, TooManyRequests, Unauthorized


def _refresh_safari_cookies() -> tuple[str, str] | None:
    """Pull fresh `auth_token` + `ct0` from the local Safari cookie jar.

    Used when the cached cookies a `Bookmarks` instance is holding
    have gone stale mid-run (Safari rotates them periodically;
    Twitter also revokes sessions). Returns `None` quietly if the
    Safari jar isn't readable on this machine — the caller logs
    the failure and stops trying.
    """
    try:
        from .cookies import get_safari_cookies

        creds = get_safari_cookies()
        tok = creds.get("auth_token") or ""
        ct0 = creds.get("ct0") or ""
        if tok and ct0:
            return tok, ct0
    except Exception:
        pass
    return None


# ─── rate-limit guard ───────────────────────────────────────────────────
#
# Twitter's GraphQL endpoints enforce a per-session quota on each
# distinct resource (tweets, likes, user lookups…). Hitting it
# returns HTTP 429 with an `x-rate-limit-reset` epoch — twikit
# parses that into `TooManyRequests.rate_limit_reset`. Burning the
# quota in a 140-personality run is the obvious failure mode, so we
# wrap every twikit call in this guard:
#
#   1. Try the call.
#   2. On `TooManyRequests`, sleep until the server-supplied reset
#      (capped to keep a corrupted timestamp from parking the
#      process for hours).
#   3. Retry once. If the second attempt also 429s we give up on
#      this call and let the caller move on — better to skip one
#      personality than block the whole queue.
#
# `_RATE_LIMIT_MAX_WAIT_SEC` is the cap. Twitter's longest window
# is 15 minutes (900 s); 1 200 s leaves slack for clock skew.

_RATE_LIMIT_MAX_WAIT_SEC = 1200


async def _rate_limit_aware(coro_factory, *, label: str):
    """Run an async twikit call with one rate-limit retry.

    `coro_factory` is a zero-arg callable that returns a fresh
    coroutine — we call it twice (initial + post-sleep retry)
    rather than caching the awaitable, since awaiting a coroutine
    twice raises RuntimeError. On non-429 errors we re-raise so the
    caller can decide how to react.
    """
    for attempt in range(2):
        try:
            return await coro_factory()
        except TooManyRequests as e:
            reset = getattr(e, "rate_limit_reset", None)
            now = time.time()
            wait = (reset - now) if reset else 60.0
            wait = max(15.0, min(_RATE_LIMIT_MAX_WAIT_SEC, wait + 5.0))
            if attempt == 0:
                print(f"    twikit {label}: rate-limited; sleeping {int(wait)}s until reset…")
                await asyncio.sleep(wait)
                continue
            # Second 429 — give up so the run keeps moving.
            print(f"    twikit {label}: rate-limited again after wait — giving up on this call")
            return None


__all__ = ["Bookmarks"]


# ─── twikit monkey-patch ─────────────────────────────────────────────────
# twikit 2.3.3 can't parse x.com's March-2026 main-bundle format, so it
# raises "Couldn't get KEY_BYTE indices" before any request goes out.
# Mirrors twikit PR #410 (https://github.com/d60/twikit/pull/410) in
# process so a fresh `uv sync` never loses the fix. Removable once the
# PR merges and our pin bumps past it.
def _patch_twikit_key_byte_indices() -> None:
    from twikit.x_client_transaction import transaction as _tx

    on_demand_file_regex = re.compile(r""",(\d+):["']ondemand\.s["']""", flags=(re.VERBOSE | re.MULTILINE))
    indices_regex = re.compile(r"\[(\d+)\],\s*16")
    on_demand_hash_pattern = r',{}:"([0-9a-f]+)"'

    async def get_indices(self, home_page_response, session, headers):
        key_byte_indices: list[str] = []
        response = self.validate_response(home_page_response) or self.home_page_response
        response_str = str(response)
        on_demand_file = on_demand_file_regex.search(response_str)
        if on_demand_file:
            on_demand_file_index = on_demand_file.group(1)
            hash_regex = re.compile(on_demand_hash_pattern.format(on_demand_file_index))
            hash_match = hash_regex.search(response_str)
            if hash_match:
                filename = hash_match.group(1)
                on_demand_file_url = f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{filename}a.js"
                on_demand_file_response = await session.request(method="GET", url=on_demand_file_url, headers=headers)
                for item in indices_regex.finditer(str(on_demand_file_response.text)):
                    key_byte_indices.append(item.group(1))
        if not key_byte_indices:
            raise Exception("Couldn't get KEY_BYTE indices")
        key_byte_indices = list(map(int, key_byte_indices))
        return key_byte_indices[0], key_byte_indices[1:]

    _tx.ON_DEMAND_FILE_REGEX = on_demand_file_regex
    _tx.INDICES_REGEX = indices_regex
    _tx.ON_DEMAND_HASH_PATTERN = on_demand_hash_pattern
    _tx.ClientTransaction.get_indices = get_indices


# Second twikit workaround: User.__init__ hard-indexes a bunch of
# legacy fields that Twitter routinely omits for accounts with sparse
# profiles (no bio, no location, private, etc.). One missing key
# tanks the whole bookmarks fetch. We pre-populate safe defaults into
# the raw data dict before twikit's constructor touches it.
_USER_LEGACY_DEFAULTS = {
    "location": "",
    "description": "",
    "pinned_tweet_ids_str": [],
    "verified": False,
    "possibly_sensitive": False,
    "can_dm": False,
    "can_media_tag": False,
    "want_retweets": False,
    "default_profile": False,
    "default_profile_image": False,
    "has_custom_timelines": False,
    "followers_count": 0,
    "fast_followers_count": 0,
    "normal_followers_count": 0,
    "friends_count": 0,
    "favourites_count": 0,
    "listed_count": 0,
    "media_count": 0,
    "statuses_count": 0,
    "created_at": "",
    "withheld_in_countries": [],
    "profile_banner_url": None,
    "url": None,
    "name": "",
    "screen_name": "",
    "profile_image_url_https": "",
}


def _patch_twikit_user_defaults() -> None:
    from twikit import user as _user

    original_init = _user.User.__init__

    def safe_init(self, client, data):
        # `data` is the raw user node from Twitter's GraphQL response.
        # Make sure every legacy key twikit reads with [] has *some*
        # value, and that the nested entities dict always has the
        # description.urls list that twikit reads unconditionally.
        legacy = data.setdefault("legacy", {}) if isinstance(data, dict) else {}
        for k, v in _USER_LEGACY_DEFAULTS.items():
            legacy.setdefault(k, v)
        entities = legacy.setdefault("entities", {})
        description = entities.setdefault("description", {})
        description.setdefault("urls", [])
        data.setdefault("is_blue_verified", False)
        original_init(self, client, data)

    _user.User.__init__ = safe_init


_patch_twikit_key_byte_indices()
_patch_twikit_user_defaults()

# Polite cap — 40 per page, so 50 pages ≈ 2000 bookmarks. The 1.5 s
# delay between pages keeps the total run under Twitter's rate-limit
# threshold (~75 s of cooldown spread across the walk). Bookmarks
# come newest-first, so re-runs hit known URLs quickly even at the
# higher cap.
_DEFAULT_MAX_PAGES = 50
_PAGE_SIZE = 40
_POLITE_DELAY = 1.5  # seconds between pages


# ─── twikit → twitterapi.io dict adapter ────────────────────────────────
#
# Every helper in `sources/twitter/tweets.py` operates on a plain dict
# shaped like twitterapi.io's payload: `{id, text, user.screen_name,
# entities.urls, extendedEntities.media, retweeted_tweet, quoted_tweet,
# createdAt, ...}`. Twikit hands us a `Tweet` object with similarly-
# named but distinct attributes. Rather than maintain a parallel set of
# extractors for twikit, we adapt once and route every downstream call
# through the existing functions — the upshot is that any improvement
# to the twitterapi.io path automatically applies to the twikit one.
#
# The adapter is *tolerant* by design (lots of `getattr(..., default)`):
# twikit ships breaking attribute renames between releases and we'd
# rather degrade to "shorter summary" than crash the whole bookmarks
# fetch over a missing field.


def _twikit_url(raw) -> str:
    """Normalise a twikit URL entry to a bare expanded URL string.

    twikit hands us either a dict (`{expanded_url, url, ...}`), a raw
    string, or a `Url`-shaped object with attributes; this collapses
    all three to whichever string looks most like the destination
    URL.
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return (raw.get("expanded_url") or raw.get("url") or "").strip()
    for attr in ("expanded_url", "url"):
        v = getattr(raw, attr, None)
        if v:
            return str(v).strip()
    return ""


def _twikit_media(tweet) -> list[dict]:
    """Convert twikit's `.media` list to twitterapi.io's `media` shape.

    Each item becomes `{type, media_url_https, video_info}` so
    `_tweet_media_urls` can pick the right photo/video URL without
    branching on the attribute style.
    """
    out: list[dict] = []
    for m in getattr(tweet, "media", None) or []:
        if isinstance(m, dict):
            kind = m.get("type") or ""
            mu = m.get("media_url_https") or m.get("media_url") or ""
            vi = m.get("video_info") or m.get("videoInfo") or None
        else:
            kind = getattr(m, "type", "") or ""
            mu = getattr(m, "media_url_https", "") or getattr(m, "media_url", "") or ""
            vi = getattr(m, "video_info", None) or getattr(m, "videoInfo", None)
        item: dict = {"type": kind}
        if mu:
            item["media_url_https"] = mu
        if vi is not None:
            # `_tweet_media_urls` reads `video_info.variants[*]` —
            # accept twikit's object-shape too via duck-typing.
            if isinstance(vi, dict):
                item["video_info"] = vi
            else:
                variants = getattr(vi, "variants", []) or []
                # Normalise variants to dicts of {content_type, bitrate, url}.
                norm = []
                for v in variants:
                    if isinstance(v, dict):
                        norm.append(v)
                    else:
                        norm.append(
                            {
                                "content_type": getattr(v, "content_type", "") or "",
                                "bitrate": getattr(v, "bitrate", 0) or 0,
                                "url": getattr(v, "url", "") or "",
                            }
                        )
                item["video_info"] = {"variants": norm}
        out.append(item)
    return out


def _twikit_article(tweet) -> dict | None:
    """Pull the X long-form Article (`x.com/i/article/<id>`) metadata
    out of a twikit `Tweet`'s raw payload.

    Twikit doesn't expose articles on its `Tweet` model (the
    `thumbnail_*` slots come back null), but the raw GraphQL
    response — stashed on `tweet._data` — carries a node that
    looks like ``{title, preview_text, rest_id, ...}``. We
    recurse the tree and surface that as ``{title, summary,
    rest_id}`` so `_build_linked_urls` can render a real
    preview card without going through `_fetch_page_preview`
    (which would hit X's consent wall and return nothing).
    """
    raw = getattr(tweet, "_data", None)
    if not isinstance(raw, dict):
        return None

    def _walk(node, depth=0):
        if depth > 12 or not isinstance(node, dict | list):
            return None
        if isinstance(node, dict):
            # The article node consistently has these three keys
            # together. Match on the trio so we don't pick up the
            # surrounding tweet's `rest_id`.
            if (
                "title" in node
                and "preview_text" in node
                and "rest_id" in node
                and str(node.get("rest_id", "")).isdigit()
            ):
                return node
            for v in node.values():
                hit = _walk(v, depth + 1)
                if hit is not None:
                    return hit
        else:
            for v in node:
                hit = _walk(v, depth + 1)
                if hit is not None:
                    return hit
        return None

    art = _walk(raw)
    if not art:
        return None
    return {
        "title": (art.get("title") or "").strip(),
        "summary": (art.get("preview_text") or "").strip(),
        "rest_id": str(art.get("rest_id") or ""),
    }


def _twikit_to_dict(tweet, depth: int = 0) -> dict | None:
    """Adapt a twikit `Tweet` to a twitterapi.io-shaped dict.

    `depth` caps the recursion into `retweeted_tweet` / `quoted_tweet`
    — twikit can in theory wire a tweet back into its own quote chain
    via reposts of reposts, and two levels is enough to surface "RT of
    a quote tweet" without risking pathological loops.
    """
    if tweet is None:
        return None

    user = getattr(tweet, "user", None)
    user_d: dict = {}
    if user is not None:
        user_d["screen_name"] = getattr(user, "screen_name", "") or ""
        user_d["name"] = getattr(user, "name", "") or ""

    # twikit's `urls` is a property that dereferences
    # `note_tweet_results['result']['entity_set']` unconditionally and
    # KeyErrors on tweets where that structure isn't populated (notably
    # some community-note-tagged tweets). `getattr` would propagate the
    # exception because it only swallows AttributeError on missing
    # attributes, not arbitrary errors raised inside a property body.
    # Treat any failure as "no URLs attached" — losing the link preview
    # on the rare malformed tweet is far better than aborting the whole
    # bookmarks pass.
    try:
        raw_urls = getattr(tweet, "urls", None) or []
    except Exception:
        raw_urls = []
    urls = [{"expanded_url": e} for e in (_twikit_url(u) for u in raw_urls) if e]

    created = getattr(tweet, "created_at", None)
    if created is not None and hasattr(created, "strftime"):
        # datetime — `_parse_date` accepts both, but we prefer to
        # hand it a string in the twitterapi.io format so behaviour
        # matches that path verbatim.
        try:
            created_str = created.strftime("%a %b %d %H:%M:%S %z %Y")
        except Exception:
            created_str = str(created)
    else:
        created_str = str(created or "")

    # Thread plumbing: twikit exposes `conversation_id` and either
    # `in_reply_to` (parent tweet id) or `in_reply_to_status_id` depending
    # on version. We carry both so the thread-merger downstream can
    # detect a continuation regardless of which one is populated.
    conv_id = getattr(tweet, "conversation_id", None) or getattr(tweet, "conversationId", None)
    reply_to = (
        getattr(tweet, "in_reply_to", None)
        or getattr(tweet, "in_reply_to_status_id", None)
        or getattr(tweet, "in_reply_to_status_id_str", None)
    )
    # The *author* of the parent tweet — needed to tell self-replies
    # (legitimate thread continuations) apart from the user's
    # @-replies to other people, which we never want to keep as the
    # user's own documents. twikit doesn't expose this as a public
    # attribute, but it stows the raw v1.1 payload under `._legacy`,
    # which has `in_reply_to_user_id_str`. Fall back to direct
    # attribute names for compatibility with future versions.
    legacy = getattr(tweet, "_legacy", None) or {}
    reply_to_user = (
        legacy.get("in_reply_to_user_id_str")
        or legacy.get("in_reply_to_user_id")
        or getattr(tweet, "in_reply_to_user_id", None)
        or getattr(tweet, "in_reply_to_user_id_str", None)
    )
    # The reply target's *@handle* — only available on the v1.1 legacy
    # payload, not as a public twikit attribute. Needed by the
    # referenced-author backfill which records who the user is replying
    # to (and therefore worth looking at as a potential new VIP).
    reply_to_screen = legacy.get("in_reply_to_screen_name") or ""

    d: dict = {
        "id": str(getattr(tweet, "id", "") or ""),
        "text": (getattr(tweet, "full_text", None) or getattr(tweet, "text", "") or ""),
        "user": user_d,
        "createdAt": created_str,
        "entities": {"urls": urls},
        # `_tweet_media_urls` reads `extendedEntities.media` first.
        "extendedEntities": {"media": _twikit_media(tweet)},
        "conversationId": str(conv_id) if conv_id else "",
        "in_reply_to_status_id": str(reply_to) if reply_to else "",
        "in_reply_to_user_id": str(reply_to_user) if reply_to_user else "",
        "in_reply_to_screen_name": reply_to_screen,
    }

    # Engagement metrics. twikit exposes them as Tweet attributes
    # (`favorite_count`, `retweet_count`, …); some live only on the raw
    # `_legacy` v1.1 payload (notably `bookmark_count` and `view_count`
    # on older Tweet objects). We accept either spelling and let
    # `tweets._tweet_engagement` walk the resulting dict — copying these
    # onto the dict keeps the twikit and twitterapi.io paths converging
    # on the same downstream code.
    def _eng_attr(name: str, *legacy_keys: str):
        v = getattr(tweet, name, None)
        if v is None:
            for k in legacy_keys or (name,):
                v = legacy.get(k)
                if v is not None:
                    break
        return v

    eng_pairs = (
        ("favorite_count", "favorite_count"),
        ("retweet_count", "retweet_count"),
        ("reply_count", "reply_count"),
        ("quote_count", "quote_count"),
        ("view_count", "view_count"),
        ("bookmark_count", "bookmark_count"),
    )
    for attr, legacy_key in eng_pairs:
        val = _eng_attr(attr, legacy_key)
        if val is not None:
            d[attr] = val

    # X long-form Article attached to this tweet (only present when
    # the tweet's t.co link expands to `x.com/i/article/<id>`).
    # Surfaced so `_build_linked_urls` can fill the linked-url
    # preview card with the article's real title + preview text
    # instead of an empty placeholder.
    art = _twikit_article(tweet)
    if art:
        d["article_card"] = art

    if depth < 2:
        rt = getattr(tweet, "retweeted_tweet", None)
        if rt is not None:
            d["retweeted_tweet"] = _twikit_to_dict(rt, depth + 1)
        # twikit exposes the quote on `.quote`; older versions had
        # `.quoted_tweet`. Accept both for forward compatibility.
        q = getattr(tweet, "quote", None) or getattr(tweet, "quoted_tweet", None)
        if q is not None:
            d["quoted_tweet"] = _twikit_to_dict(q, depth + 1)

    return d


class Bookmarks:
    """Fetch a Twitter/X account's tweets and bookmarks as {url: doc}.

    Two modes, switched by ``target_username``:

      * **Auth-user mode** (``target_username=None``, default) — pulls
        the account that owns the cookies. Bookmarks (private) + own
        tweets.
      * **Target-user mode** (``target_username="@<handle>"``) — pulls
        tweets for an arbitrary public account, using the cookies
        only to authenticate the request. Bookmarks are skipped (the
        endpoint is auth-user-only). This is what the
        ``make run TWIKIT=1`` path uses: one Safari session covers
        every VIP, no per-personality cookies required.

    Parameters
    ----------
    auth_token : str
        The ``auth_token`` cookie value from x.com.
    ct0 : str
        The ``ct0`` cookie value from x.com.
    target_username : str | None
        Public Twitter handle to fetch. None means the
        cookie-authenticated user (legacy behaviour).
    max_pages : int
        Safety cap on pagination *per stream*. Default 50.
    include_bookmarks : bool
        Pull /bookmarks. Default True. Forced False when
        ``target_username`` is set (endpoint is auth-user-only).
    include_tweets : bool
        Pull /get_user_tweets. Default True.
    on_page_flush : callable | None
        Optional callback invoked after every drained page with a
        ``{url: doc}`` dict of the docs emitted by *that page* (i.e.
        new since the previous flush). Lets the caller durably
        persist tweets as they stream in so a crash mid-pagination
        doesn't throw away everything fetched so far. Exceptions
        from the callback are caught and logged; they don't abort
        the stream.
    """

    def __init__(
        self,
        auth_token: str,
        ct0: str,
        target_username: str | None = None,
        max_pages: int = _DEFAULT_MAX_PAGES,
        include_bookmarks: bool = True,
        include_tweets: bool = True,
        on_page_flush=None,
    ):
        self.auth_token = auth_token
        self.ct0 = ct0
        self.target_username = (target_username or "").lstrip("@") or None
        self.max_pages = max_pages
        # /bookmarks is only readable for the auth'd account; when
        # fetching someone else's library we can't get their saves.
        self.include_bookmarks = include_bookmarks and self.target_username is None
        self.include_tweets = include_tweets
        self.on_page_flush = on_page_flush

    def __call__(self, existing_urls: set[str] | None = None) -> dict[str, dict]:
        existing = existing_urls or set()
        try:
            return asyncio.run(self._fetch(existing))
        except Exception as e:
            import traceback

            print(f"    Twitter cookie sync failed: {e!r}")
            traceback.print_exc()
            return {}

    def lookup(self, tweet_ids: list[str]) -> dict[str, dict]:
        """Fetch tweets by ID and return them in twitterapi.io payload shape.

        Used by backfill scripts that already know the tweet ids they
        want to re-render (e.g. "every doc whose summary still starts
        with 'RT @'"). Each returned dict is the same shape
        ``_twikit_to_dict`` produces, which means it can be fed
        straight into ``_tweet_self_sufficient_summary`` /
        ``_build_linked_urls`` without any further adaptation.

        Returns a ``{tweet_id: tweet_dict}`` map; ids that twikit
        couldn't fetch (protected, deleted, rate-limited) are simply
        absent from the map. No twitterapi.io credit consumed.
        """
        if not tweet_ids:
            return {}
        try:
            return asyncio.run(self._lookup(list(tweet_ids)))
        except Exception as e:
            import traceback

            print(f"    Twitter lookup failed: {e!r}")
            traceback.print_exc()
            return {}

    async def _lookup(self, tweet_ids: list[str]) -> dict[str, dict]:
        client = Client("en-US")
        client.set_cookies({"auth_token": self.auth_token, "ct0": self.ct0})
        out: dict[str, dict] = {}
        # `get_tweets_by_ids` hits the `TweetResultsByRestIds` GraphQL
        # endpoint — one batched call per chunk, and unlike
        # `get_tweet_by_id` it doesn't trip over twikit 2.3.3's
        # broken `TweetDetail` parser. Twitter caps each call at
        # around 100 ids; 50 leaves headroom for retries on flaky
        # ones.
        chunk = 50
        for i in range(0, len(tweet_ids), chunk):
            batch = [str(t) for t in tweet_ids[i : i + chunk]]
            try:
                tweets = await _rate_limit_aware(
                    lambda b=batch: client.get_tweets_by_ids(b),
                    label=f"get_tweets_by_ids({len(batch)})",
                )
            except Exception as e:
                print(f"    lookup chunk {i}-{i + len(batch)}: {e!r}")
                tweets = []
            for tw in tweets or []:
                if tw is None:
                    continue
                tid = str(getattr(tw, "id", "") or "")
                if not tid:
                    continue
                adapted = _twikit_to_dict(tw)
                if adapted:
                    out[tid] = adapted
            # Pace between batches — last batch has nothing to wait for.
            if i + chunk < len(tweet_ids):
                time.sleep(_POLITE_DELAY)
        return out

    async def _fetch(self, existing: set[str]) -> dict[str, dict]:
        client = Client("en-US")
        client.set_cookies({"auth_token": self.auth_token, "ct0": self.ct0})

        # Cookie health check — resolve the authenticated user id
        # before any per-stream fetch fires. If the cached cookies
        # have gone stale (Safari rotated them, Twitter revoked the
        # session) we silently re-read from the live Safari jar and
        # try once more before giving up. Process-wide cache of the
        # fresh tokens so all 452 personalities in the same run
        # share the same refreshed credentials.
        async def _probe() -> str | None:
            try:
                return await _rate_limit_aware(
                    lambda: _maybe_await(_resolve_user_id(client)),
                    label="cookie health probe",
                )
            except (Unauthorized, Forbidden):
                return None
            except Exception:
                # Network blips or twikit shape errors aren't
                # auth-related; let the caller see them as a failed
                # probe but don't trigger a cookie refresh (which
                # wouldn't help).
                raise

        token_tail = (self.auth_token or "")[-6:] or "?"
        try:
            me_id = await _probe()
        except Exception as e:
            print(
                f"    twikit cookies (auth_token …{token_tail}): probe failed: {e!r} "
                "— fetches will likely return nothing"
            )
            me_id = None

        if not me_id:
            print(f"    twikit cookies (auth_token …{token_tail}) stale — trying Safari refresh…")
            fresh = _refresh_safari_cookies()
            if fresh:
                self.auth_token, self.ct0 = fresh
                token_tail = self.auth_token[-6:] or "?"
                client.set_cookies({"auth_token": self.auth_token, "ct0": self.ct0})
                try:
                    me_id = await _probe()
                except Exception as e:
                    print(f"    twikit cookies (auth_token …{token_tail}): refresh probe failed: {e!r}")
                    me_id = None
            else:
                print(
                    "    twikit cookie refresh: Safari jar unavailable (headless host? Safari closed?) — cannot recover"
                )

        if me_id:
            print(f"    twikit cookies (auth_token …{token_tail}) OK — authenticated user id={me_id}")
        else:
            print(
                f"    twikit cookies (auth_token …{token_tail}): could not "
                "resolve authenticated user id; fetches will return nothing"
            )

        data: dict[str, dict] = {}

        if self.include_bookmarks:
            await self._drain(
                lambda: client.get_bookmarks(count=_PAGE_SIZE),
                kind="bookmark",
                data=data,
                existing=existing,
            )

        # Resolve which X account's tweets we're pulling. In
        # target-user mode we look up the requested handle's numeric
        # id; in auth-user mode we ask twikit for the id of the
        # account that owns the cookies.
        if self.include_tweets:
            if self.target_username:
                user_id = await _resolve_target_user_id(client, self.target_username)
                who = f"@{self.target_username}"
            else:
                user_id = await _resolve_user_id(client)
                who = "authenticated user"
            if not user_id:
                print(f"    Could not resolve user id for {who}; skipping tweets")
                return data

            await self._drain(
                lambda: client.get_user_tweets(user_id, "Tweets", count=_PAGE_SIZE),
                kind="tweet",
                data=data,
                existing=existing,
            )

        # Merge same-author multi-tweet threads (e.g. the operator
        # liked or retweeted every part of a numbered thread). Each
        # surviving doc gets the box-drawing separator the frontend's
        # `renderTweetSummary` already splits on, so threads land as
        # one card with per-part media instead of N near-duplicates.
        self._merge_threads(data)
        for d in data.values():
            for k in ("_conv_id", "_author", "_tweet_id", "_raw", "_kind"):
                d.pop(k, None)

        print(f"    {len(data)} URLs from Twitter cookie sync ({who if self.include_tweets else 'bookmarks only'})")
        return data

    @staticmethod
    def _merge_threads(data: dict[str, dict]) -> None:
        """In-place merge of thread parts inside `data`.

        Group key: ``(author, conversation_id)`` where
        ``conversation_id`` is set AND differs from the part's tweet
        id — that combination identifies a continuation tweet by the
        original author. Singletons (size 1 groups) are left alone.

        Output shape is produced by ``tweets.compose_thread_doc``, the
        same function the twitterapi.io path uses. The only addition
        on this path is a union of per-part kind tags
        (``twitter-like`` / ``twitter-retweet`` / ``twitter-bookmark``)
        so source filters still surface a merged thread under each of
        the user actions that contributed to it.
        """
        from collections import defaultdict

        from .tweets import compose_thread_doc

        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for url, d in data.items():
            conv = d.get("_conv_id") or ""
            author = d.get("_author") or ""
            if not conv or not author:
                continue
            # Group every doc with a conversation id — roots
            # (conv == tweet_id) AND continuations (conv != tweet_id).
            # A standalone tweet still gets its own singleton bucket
            # and is filtered out below by the `len < 2` guard.
            groups[(author, conv)].append(url)

        for (author, conv), urls in groups.items():
            if len(urls) < 2:
                continue
            # Sort by date so part numbering is chronological.
            sorted_parts = sorted(
                (data[u] for u in urls),
                key=lambda d: (d.get("date") or "", d.get("_tweet_id") or ""),
            )
            raw_parts = [p["_raw"] for p in sorted_parts]
            # `compose_thread_doc` picks the canonical anchor URL (true
            # root if present, oldest otherwise) and produces the same
            # six-key doc shape the twitterapi.io path emits.
            new_url, doc = compose_thread_doc(raw_parts, username=author)
            # Carry per-part kind tags into the merged doc — these are
            # the one legitimate per-flow signal the twitterapi.io path
            # doesn't have (it knows everything is the user's own).
            tag_set = set(doc.get("tags") or [])
            for p in sorted_parts:
                tag_set.update(p.get("tags") or [])
            doc["tags"] = sorted(tag_set)
            # Replace the per-part rows with the merged doc, keyed on
            # the compose-function's chosen URL.
            for u in urls:
                if u != new_url:
                    data.pop(u, None)
            # Carry the private fields through so the strip-pass at
            # the end of `_fetch` cleans them out uniformly.
            doc["_conv_id"] = conv
            doc["_author"] = author
            doc["_tweet_id"] = sorted_parts[0].get("_tweet_id") or ""
            doc["_raw"] = sorted_parts[0]["_raw"]
            doc["_kind"] = sorted_parts[0].get("_kind") or ""
            data[new_url] = doc

    async def _drain(
        self,
        page_loader,
        *,
        kind: str,
        data: dict[str, dict],
        existing: set[str],
    ) -> None:
        """Walk a paginated twikit Result and emit one doc per tweet/url.

        ``page_loader`` is a no-arg callable that returns the *first*
        page (an async coroutine). Subsequent pages come from
        ``page.next()``.

        Both the initial fetch and every `page.next()` go through
        `_rate_limit_aware`, so a 429 mid-stream is absorbed by a
        single sleep-to-reset instead of dropping the whole stream
        on the floor.
        """
        try:
            page = await _rate_limit_aware(page_loader, label=f"{kind} first page")
        except Exception as e:
            print(f"    Twitter {kind}s: initial fetch failed ({e!r})")
            return
        if page is None:
            return

        before = len(data)
        page_index = 0
        keys_before_page: set[str] = set(data.keys())
        for _ in range(self.max_pages):
            if not page:
                break
            page_index += 1
            before_page = len(data)
            n_tweets_in_page = 0
            for tw in page:
                n_tweets_in_page += 1
                self._emit(tw, kind, data, existing)
            # Per-page line so the operator can see the stream is
            # actually pulling tweets (not silently stuck behind a
            # rate-limit wait). `+N` is the number of new docs we
            # accepted off this page (after dedup against `existing`
            # and the in-flight `data`); `seen=` is the raw tweet
            # count the page handed us.
            page_new = len(data) - before_page
            print(
                f"    Twitter {kind}s: page {page_index} +{page_new} new (seen={n_tweets_in_page})",
                flush=True,
            )
            # Early stop when a whole page produced zero new docs and
            # we actually had something to dedup against. Twitter's
            # timeline is reverse-chronological, so a page of all-
            # already-known tweets means everything below it is also
            # known — no point spending another 3-second RPC chasing
            # rows we'll just upsert as no-ops. The caller (the
            # twitter feeder loop) passes the user's existing-URL set
            # so `_emit` can drop already-stored tweets at source.
            if page_new == 0 and n_tweets_in_page > 0 and existing:
                print(
                    f"    Twitter {kind}s: page {page_index} added nothing new "
                    f"({n_tweets_in_page} already known) — stopping early",
                    flush=True,
                )
                # page_new == 0 ⇒ no new keys ⇒ nothing to flush; just
                # exit the pagination loop.
                break
            # Durably flush this page so a crash mid-stream doesn't
            # discard everything fetched so far. The callback gets
            # ONLY the keys added during this page (set diff against
            # the snapshot we took at the top of the loop). Errors
            # in the callback are caught and logged — they must
            # never abort the stream.
            if self.on_page_flush is not None:
                new_keys = set(data.keys()) - keys_before_page
                if new_keys:
                    page_docs = {k: data[k] for k in new_keys}
                    try:
                        self.on_page_flush(page_docs)
                    except Exception as e:
                        print(
                            f"    Twitter {kind}s: page {page_index} flush callback failed ({e!r}); continuing",
                            flush=True,
                        )
            keys_before_page = set(data.keys())
            try:
                page = await _rate_limit_aware(page.next, label=f"{kind} next page")
            except Exception:
                break
            if page is None:
                break
            time.sleep(_POLITE_DELAY)
        print(
            f"    Twitter {kind}s: total +{len(data) - before} new URLs over {page_index} page(s)",
            flush=True,
        )

    # ── Per-tweet emission ───────────────────────────────────────────────
    def _emit(self, tweet, kind: str, data: dict[str, dict], existing: set[str]) -> None:
        """Emit one document per tweet, in the twitterapi.io shape.

        ``kind`` is one of ``"bookmark"`` / ``"tweet"`` — used to tag
        the document so downstream filters can distinguish bookmarks
        from regular posts.

        The output doc carries:

          * ``summary``      — the self-sufficient summary produced by
                               :func:`tweets._tweet_self_sufficient_summary`,
                               which recurses into ``retweeted_tweet``
                               so retweets surface the original's full
                               text + media markers.
          * ``linked_urls``  — list of ``{url, host, title, summary,
                               image}`` dicts for every external URL
                               the tweet (or its retweeted inner)
                               points at, with server-side OG previews.
          * ``link_hosts``   — distinct hostnames from ``linked_urls``
                               (powers the multi-source filter chip).

        Tweets that are pure retweets pull their ``linked_urls`` from
        the wrapped inner tweet (the wrapper carries no URLs of its
        own) — mirroring what ``Tweets._collect_own_tweets`` does on
        the twitterapi.io path.
        """
        from .tweets import (
            _build_linked_urls,
            _link_source_for,
            _parse_date,
            _retweet_extra_tags,
            _tweet_display_title,
            _tweet_engagement,
            _tweet_self_sufficient_summary,
            _tweet_url,
        )

        tw_dict = _twikit_to_dict(tweet)
        if not tw_dict or not tw_dict.get("id"):
            return

        # Self-reply gate. The "Replies" stream fetch sets
        # `self._self_reply_user_id` so we keep only the target user's
        # replies to themselves (= thread continuations). Replies to
        # other people are not "the user's own documents" and would
        # clutter their library.
        gate = getattr(self, "_self_reply_user_id", None)
        if gate:
            reply_to_user = (tw_dict.get("in_reply_to_user_id") or "").strip()
            if reply_to_user != str(gate):
                return

        fallback_handle = (tw_dict.get("user") or {}).get("screen_name") or "unknown"
        url = _tweet_url(tw_dict, fallback_handle)
        if not url:
            return
        if url in existing or url in data:
            return

        # `_link_source_for` handles the retweet-vs-quote logic so the
        # twikit path stays in lockstep with the twitterapi.io one:
        # retweets pull from the inner tweet, quotes pull from both
        # the wrapper and the quoted side, plain tweets just from
        # themselves.
        rt_inner = tw_dict.get("retweeted_tweet") or {}
        linked_urls, link_hosts = _build_linked_urls(_link_source_for(tw_dict))

        # `twitter-bookmark` stays as a filterable provenance chip.
        # `twitter-tweet` (own posts) used to ride along too but
        # was dropped — every own tweet is already implied by
        # `source = twitter` and the absence of a bookmark/retweet
        # marker, so the extra chip was pure noise. Mirrored in
        # `_collect_own_tweets` for the twitterapi.io path.
        tags = ["twitter"]
        if kind != "tweet":
            tags.append(f"twitter-{kind}")
        if rt_inner:
            tags.append("retweet")

        # Thread-grouping identity: when the outer tweet wraps a
        # retweet, the conversation belongs to the *inner* (original)
        # author, not the retweeter. Unwrap so a mixed like/retweet
        # set of the same thread lands in one group.
        inner = rt_inner if isinstance(rt_inner, dict) else {}
        inner_user = inner.get("user") or {}
        thread_author = (inner_user.get("screen_name") or fallback_handle).lower()
        thread_conv = (inner.get("conversationId") or tw_dict.get("conversationId") or "").strip()
        # When we unwrap to the inner tweet, the "tweet id" used to
        # tell standalone tweets apart from continuations is also the
        # inner one. Otherwise we'd false-positive ("inner conv == outer
        # tweet id" rejects the merge for retweets-of-a-thread-root).
        thread_tweet_id = str(inner.get("id") or tw_dict.get("id") or "")

        # Engagement: `_tweet_engagement` recurses into the retweet
        # wrapper so a retweet's counts come from the SOURCE tweet,
        # which is the right thing for ranking — the wrapper itself
        # never accumulates likes.
        engagement = _tweet_engagement(tw_dict)

        data[url] = {
            "title": _tweet_display_title(tw_dict, fallback_handle),
            "summary": _tweet_self_sufficient_summary(tw_dict),
            "date": _parse_date(tw_dict),
            "tags": tags,
            **engagement,
            # `extra_tags` carries `retweet @<inner-handle>` so the
            # card chip strip can show the attribution that used to
            # live as a "Retweet @x" prefix in the summary text.
            "extra-tags": _retweet_extra_tags(tw_dict),
            "linked_urls": linked_urls,
            "link_hosts": link_hosts,
            "source": "twitter",
            # Private fields used by `_merge_threads`. Stripped before
            # the dict is returned from `_fetch`. We keep the raw twikit
            # dict so the merger can reuse `_tweet_self_sufficient_summary`
            # on each part (preserving every photo/video marker).
            "_conv_id": thread_conv,
            "_author": thread_author,
            "_tweet_id": thread_tweet_id,
            "_raw": tw_dict,
            "_kind": kind,
        }

    # ── twikit field extraction ──────────────────────────────────────────
    @staticmethod
    def _tweet_permalink(tweet) -> str:
        tid = getattr(tweet, "id", None)
        if not tid:
            return ""
        user = getattr(getattr(tweet, "user", None), "screen_name", None) or "i"
        return f"https://x.com/{user}/status/{tid}"

    @staticmethod
    def _format_date(tweet) -> str:
        created = getattr(tweet, "created_at", None)
        if isinstance(created, datetime.datetime):
            return created.strftime("%Y-%m-%d")
        if isinstance(created, str):
            try:
                return datetime.datetime.strptime(created, "%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
        return ""

    @staticmethod
    def _clean_url(raw) -> str:
        """twikit's tweet.urls entries are either dicts with
        `expanded_url` / `url` keys, or bare strings. Normalize to a
        plain absolute URL string."""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            return (raw.get("expanded_url") or raw.get("url") or "").strip()
        return ""


# ─── twikit version-tolerant user-id resolution ─────────────────────────
async def _resolve_user_id(client) -> str | None:
    """Return the authenticated user's id, tolerating twikit API drift.

    twikit has shipped at least three different shapes for "who am I":
    a coroutine `client.user_id()`, a sync property `client.user_id`,
    and an async `client.user()` returning a User with `.id`. We try
    each in turn and fall back to None if all fail — the caller skips
    the dependent streams quietly when this happens.
    """
    # Coroutine method
    try:
        candidate = client.user_id
        if callable(candidate):
            value = candidate()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return str(value)
        elif candidate:
            return str(candidate)
    except Exception:
        pass
    # client.user() → User
    try:
        get_user = getattr(client, "user", None)
        if callable(get_user):
            me = get_user()
            if asyncio.iscoroutine(me):
                me = await me
            uid = getattr(me, "id", None)
            if uid:
                return str(uid)
    except Exception:
        pass
    return None


async def _resolve_target_user_id(client, screen_name: str) -> str | None:
    """Resolve a public Twitter handle to its numeric id.

    Used by target-user mode (``make run TWIKIT=1``): one set of
    Safari cookies authenticates the request, the handle picks who
    we read. Falls back through twikit's naming variants so a
    future renaming of `get_user_by_screen_name` doesn't break the
    pipeline. The lookup itself is rate-limit-aware so a burst of
    backfills doesn't tank on the user-resolution endpoint before
    even getting to the tweet stream.
    """
    handle = (screen_name or "").lstrip("@")
    if not handle:
        return None
    for method_name in (
        "get_user_by_screen_name",
        "get_user_by_username",
    ):
        getter = getattr(client, method_name, None)
        if getter is None:
            continue
        try:
            user = await _rate_limit_aware(
                lambda g=getter, h=handle: _maybe_await(g(h)),
                label=f"{method_name}({handle})",
            )
        except Exception:
            continue
        if user is None:
            continue
        uid = getattr(user, "id", None) or getattr(user, "rest_id", None)
        if uid:
            return str(uid)
    return None


async def _maybe_await(value):
    """Await `value` if it's a coroutine; otherwise return it as-is.

    Twikit's user-by-handle getters return either a User or a
    coroutine depending on the version, so the call site wraps the
    return in this helper to keep the rate-limit guard uniform.
    """
    if asyncio.iscoroutine(value):
        return await value
    return value
