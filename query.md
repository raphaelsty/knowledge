# Feed query — how it works

The feed has **two SQL paths** that produce the same JSON. Which one
runs depends on whether the hourly `feed_snapshot` table is fresh.

```
GET /api/timeline
        │
        ▼
   anon cache hit?  ─ yes ─►  return cached JSON (≤60 s old)
        │ no
        ▼
   snapshot fresh (<3 h)
   AND no category filter?  ─ yes ─►  Snapshot path
        │ no                              SELECT … FROM feed_snapshot
        ▼                                 ORDER BY score DESC LIMIT N
   Live path
   400 k-doc CTE with windowed
   candidates + dedup + scoring
```

Both branches share the same output schema and feed the same
Rust-side diversity pass + JSON envelope. Code lives in
[`api/src/handlers/follows.rs::timeline`](api/src/handlers/follows.rs).

---

## 1. Snapshot path — the fast one

Used for ≥99 % of timeline requests. The shape is:

```sql
WITH followed AS (
    SELECT followed_id AS user_id FROM follows WHERE follower_id = $me
    UNION
    SELECT $me WHERE $me IS NOT NULL
    UNION
    SELECT id FROM users WHERE vip = TRUE AND $me IS NULL
),
followed_ids AS (
    SELECT COALESCE(array_agg(user_id), '{}'::bigint[]) AS ids FROM followed
)
SELECT s.*, /* score + per-viewer bonuses */
  FROM feed_snapshot s
 WHERE  -- audience filter
       ($me IS NULL
        OR s.sharer_user_ids && (SELECT ids FROM followed_ids LIMIT 1))
   AND ($me IS NOT NULL OR s.any_vip_sharer = TRUE)
   AND  -- source/tag/date/hide-seen filters …
 ORDER BY effective_score DESC, s.date DESC, s.url
 LIMIT $limit * 2;
```

### What's in `feed_snapshot`

One row per **anchor URL** (a resource — paper, repo, model, blog
post). Multiple docs that map to the same anchor (a paper + tweets
linking it) collapse into one row, with the visually-richest doc as
the representative. Built hourly by the `knowledge-feed-snapshot`
daemon — see
[`sources/sql/feed_snapshot.py::refresh_feed_snapshot`](sources/sql/feed_snapshot.py).

Key columns:

| Column | Meaning |
|---|---|
| `url`, `canonical_url`, `anchor_url` | Three forms of the URL. `anchor_url` is the deduplication key; `url` is the representative doc's URL. |
| `score` | Viewer-agnostic score (formula below). |
| `sharer_user_ids[]` | Every user_id who owns a doc that maps to this anchor. **GIN-indexed** for the `&&` overlap test. |
| `any_vip_sharer` | TRUE iff at least one sharer is a VIP. Partial-indexed for the anon path. |
| `sharers` (jsonb) | Pre-rendered avatar stack. |
| `refreshed_at` | Freshness probe — the handler bypasses the snapshot if `MAX(refreshed_at)` is more than 3 h old. |

### Audience filter — the two-line trick

The `WHERE` clause does the heavy lifting in just two predicates:

```sql
($me IS NULL OR s.sharer_user_ids && (SELECT ids FROM followed_ids LIMIT 1))
AND ($me IS NOT NULL OR s.any_vip_sharer = TRUE)
```

- **Logged-in**: `$me IS NOT NULL`, so the first predicate becomes
  `sharer_user_ids && followee_array`. The GIN index handles the
  array overlap in microseconds. The second predicate is satisfied
  by `$me IS NOT NULL`.
- **Anonymous**: `$me IS NULL`, so the first predicate is satisfied
  by the short-circuit. The second predicate becomes `any_vip_sharer
  = TRUE` — served by the partial index
  `idx_feed_snapshot_vip_score (score DESC) WHERE any_vip_sharer`.

So the same query handles both audiences with no `CASE` branch and
each branch hits a dedicated index.

### Per-viewer score bonuses

The `score` column is **viewer-agnostic** (it can't know who's
asking). The handler adds three viewer-specific terms on read:

```sql
s.score
+ LEAST(3, GREATEST(0, (
      SELECT count(*) FROM unnest(s.sharer_user_ids) sid
       WHERE sid IN (SELECT user_id FROM followed)
  ) - 1)) * 1.5
+ CASE WHEN s.primary_user_id = $me
            AND s.refreshed_at > now() - interval '1 hour'
       THEN 50 ELSE 0 END
AS score
```

- **Followee-share bonus** — count of overlap between the doc's
  sharers and your followees. Cap at 3 extras × 1.5 = 4.5 so one
  super-popular doc doesn't dominate.
- **Fresh-self boost** — +50 if *you* authored the doc less than an
  hour ago. Makes your own brand-new post jump to the top of your
  own feed instantly.

### Hide-seen filter

Logged-in callers also get a `NOT EXISTS` that drops cards they've
already engaged with. The dwell threshold is per-call:

```sql
AND ($me IS NULL
     OR $include_seen::bool = TRUE
     OR COALESCE((
         SELECT SUM(COALESCE(e.dwell_ms, 0))::bigint
           FROM events e
          WHERE e.viewer_user_id = $me
            AND e.event_type    = 7      -- card_seen
            AND e.doc_url       = s.url
            AND e.created_at    < now() - interval '10 minutes'
            AND e.created_at    > now() - ($horizon || ' days')::interval
     ), 0) < $min_seen_dwell_ms)
```

- 10-minute grace prevents cards from vanishing mid-scroll.
- Threshold defaults to 1500 ms aggregated — matches the client's
  `MIN_DWELL_MS` floor, so any `card_seen` event the tracker fired
  counts as a real impression.
- `$include_seen = TRUE` bypasses the filter (the "Show seen" chip).

The same `NOT EXISTS` shape also drives the per-row `alreadySeen`
flag returned to the client, which the frontend uses to dim the
card and add a "Seen" hint.

---

## 2. Live path — the fallback

Runs when the snapshot is empty / stale / a category filter is
active. The full CTE lives in
[`api/src/handlers/follows.rs::timeline`](api/src/handlers/follows.rs)
(around line ~470). It's built in four layers:

```
candidates    →  url_share  →  candidate_anchors  →  dedup  →  scored
(window scan)    (per-URL      (anchor-priority      (one row    (composite
                  sharer-counts)  + viz richness)      per         score)
                                                       anchor)
```

| Layer | What it does |
|---|---|
| `candidates` | Most-recent `$limit × 16` rows (cap 2000) across the `followed` user set. Stamps each with `sci_score` (3 for arxiv/scholar/HF or sci-linking tweets, 1 for github, 0 else). |
| `url_share` | For each canonical URL among the candidates: `followee_share` (how many followees own it) and `total_share` (how many users own it globally). Both feed the score. |
| `candidate_anchors` | Adds `anchor_url` (priority host lookup: arxiv → HF → github → openreview → DOI → …) plus `image_count` and `url_count` for the representative pick. |
| `dedup` | `DISTINCT ON (anchor_url)` keeps the visually-richest doc per anchor (most preview images, then most referenced URLs, then most recent, then VIP / follower count / citations as tiebreakers). |
| `scored` | Composite score (see formula below). |

The candidate scan's "most-recent" bound is why the live path
struggles to surface older deep cuts. The snapshot path avoids this
by scoring over the full 180-day window once an hour.

---

## 3. The score formula

Both paths use the same coefficients. Tunables in **bold**.

```
score = sci_score      × 6                          -- scientific bonus
      + recency bucket  ∈ {5, 4, 3, 2, 1, 0}        -- weekly steps, 5 weeks
      + LEAST(3, max(0, followee_share - 1)) × 1.5  -- viewer-side, ≤ 3 extras
      + LEAST(2, LN(total_share))             × 0.7 -- popularity, sublinear
      + 0.8         if primary sharer is VIP
      + LEAST(1.5, LN(twitter_followers / 10_000))  -- notability, sublinear
      + 1.5         if it's a tweet with a preview image
      + 50          if viewer authored it < 1 h ago -- fresh-self boost
```

Why each term:

- **`sci_score × 6`** — heaviest single coefficient. Scientific
  content (arxiv abs, HF model, OpenReview, distill.pub, etc.) wins
  by default. Tweets that *link* a scientific host get the same
  bonus — caught by `link_hosts && ARRAY[...]` rather than parsing
  the tweet body.
- **Weekly recency buckets** — `5, 4, 3, 2, 1, 0` over 5 weeks (not
  daily). Two docs in the same 7-day window get the same recency
  bonus, so the feed reads as "this week's activity" not "last 24
  hours". Beyond 35 days, docs compete purely on sharer count.
- **Followee-share** — capped at 3 extras × 1.5 = 4.5 so a single
  hyper-popular doc can't bury everything else. **Live path only**
  in SQL; in the snapshot path it's added on read.
- **Total-share, sublinear** — `LN(total_share)` capped at 2, then
  ×0.7. Lets globally popular resources surface even when the
  viewer follows zero people who saved them; the log keeps it from
  drowning the followee signal.
- **VIP / followers** — small notability nudges. `LN(followers /
  10_000)` capped at 1.5 means ≈10k followers = +1.0, 100k = +1.5,
  zero = 0.
- **Rich tweet** — +1.5 if the linked_urls payload has at least one
  entry with a preview image. Detected by structure, never by
  reading the tweet text.
- **Fresh-self** — +50 for an hour. Big enough to guarantee a brand-
  new post lands at the top of the author's own feed, even when its
  sharer count is still 1.

---

## 4. Diversity pass (in Rust, not SQL)

After SQL returns `$limit × 2` rows, the handler does a O(N²) pass
over a queue capped at ~400 items to avoid bunching the same
author:

```
effective = base_score
          - DECAY     × prior_appearances ^ EXP
          - ADJACENT  if same primary_user_id as the previous emit
```

Tuned values: `DECAY = 2.0`, `EXP = 1.3`, `ADJACENT = 18.0`. The
effect: a prolific author's 2nd post lands ~10 slots after their
1st, their 3rd ~30 slots later, etc. **Nothing is hidden** — every
fetched row eventually appears; they just shift down so first-time
sharers get a fair chance at the top of the page.

Same algorithm whether the rows came from the snapshot or the live
CTE.

---

## 5. Anonymous in-process cache

One more layer on top: an in-memory `HashMap<String,
AnonTimelineEntry>` keyed by canonical query-param signature, TTL
60 s. Lives in [`api/src/handlers/follows.rs`](api/src/handlers/follows.rs)
near the top of the file. Logged-in callers skip the cache (their
view is per-user); anon callers all share entries.

This is why anonymous front-page loads typically return in <1 ms —
the SQL only runs once per minute per unique filter combo.

---

## Summary

The feed is fast because three layers of caching share the load:

1. **Anonymous in-process cache** (60 s) — `<1 ms` cache hits.
2. **`feed_snapshot` table** (hourly refresh) — `50–80 ms` single
   indexed scan when the cache misses.
3. **Live CTE** — `700 ms–2 s` fallback for category filters or a
   stale snapshot.

Everything funnels into the same Rust-side diversity pass and JSON
envelope, so the response shape is identical regardless of which
path served the request.

The hourly refresh and the 180-day window are what let the snapshot
beat the live CTE on the long tail — the live path can only score
the most-recent 2000 candidate rows, the snapshot scores the
entire 180-day window once and then any read is just `ORDER BY
score DESC LIMIT N`.
