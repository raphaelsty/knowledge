# Twitter API Cost Report

## TL;DR

twitterapi.io bills **~15 credits per tweet returned**, regardless of which endpoint. The 5× cost ratio between `/twitter/tweets` (1500/call) and `/twitter/user/last_tweets` (300/call) is just the batch-size ratio (100 vs 20 tweets per call) — there is no cheap endpoint to substitute for the expensive one.

A full `make run` over 133 VIPs costs roughly **6-7M credits per fresh backfill**, dominated by Twitter. The single biggest lever for saving money is **lowering `max_tweets` for VIPs** (from 3000 to 1500-2000). Most other knobs save ≤10%.

---

## Cost model

Confirmed from production logs (4/29/2026 10:44–10:49 AM):

| Endpoint | Tweets returned | Credits | Per-tweet |
|---|---|---|---|
| `/twitter/user/last_tweets` (timeline pagination) | ~20 / page | 300 | 15 |
| `/twitter/tweets` (batch hydrate up to 100 IDs) | up to 100 | 1500 (or 675 for partial) | 15 |
| `/twitter/user/info` | 1 user | 18 | — |

→ **Cost is purely a function of total tweets fetched.** Only two ways to save money: fetch fewer tweets, or hit cache more.

---

## What we currently fetch — and why

For every personality with `sources.twitter`, the pipeline does **two phases**:

### Phase 1 — Timeline pagination (`/twitter/user/last_tweets`)
- Pulls the user's own tweets AND their replies, paginated ~20/page.
- Capped at `max_tweets` (currently **3000 for VIP**, **300 for non-VIP**, no date fence).
- This phase produces:
  - **Own tweets** → indexed as standalone documents (titles, URLs they posted, etc.)
  - **Replies** → used to discover OTHER users' content via Phase 2.

### Phase 2 — Thread root hydration (`/twitter/tweets`)
- For every reply, look up the parent tweet (the one being replied to).
- Collect distinct `conversation_id`s, batch-fetch in chunks of 100.
- **Why it exists:** the URL we care about is usually in the *parent* (the tweet being shared), not in the reply text ("great article!"). Without this phase, we lose the curatorial signal of replies.
- Capped at `max_parents=5000` per user.
- Cache is **per-user** in `data/{slug}/twitter_cache.json` so a second run skips already-hydrated parents.

### Per-VIP fresh-backfill cost
- Phase 1: ~150 pages × 300 = **~45k credits**
- Phase 2: ~700 thread roots / 100 per call × 1500 = **~12-15k credits**
- **Total: ~55-60k credits per VIP fresh.**
- Across 133 VIPs: **~7-8M credits per first run.**

### Per-non-VIP fresh-backfill cost
- Phase 1: ~15 pages × 300 = ~4.5k credits
- Phase 2: ~80 parents / 100 × 1500 = ~1.5k credits
- **Total: ~6k credits per non-VIP fresh.**
- These are mostly noise unless the person actually tweets resources — for VIP-only users this is 0.

---

## What we **definitely need**

| Item | Why | Removable? |
|---|---|---|
| Phase 1 timeline (own tweets) | The user's actual posts — original signal. | **No.** Core value. |
| Phase 1 timeline (replies) | Triggers Phase 2 → URL discovery via curation. | **No.** Needed to feed Phase 2. |
| Phase 2 thread root hydration | URL in parent tweet is the *whole point* of capturing replies. Dropping this would silently kill 30-50% of Twitter-sourced docs. | **No** (with refinements — see below). |
| `last_seen` cursor | Subsequent runs short-circuit when a full page of known IDs is hit. Without it, every run is a fresh backfill. | **No.** Already saves ~80-95% on incremental runs. |

---

## What we **don't need** (or can trim) — ordered by ROI

### 1. **Lower the VIP `max_tweets` cap from 3000 → 1500-2000** ⭐ biggest lever

**Cost per 1000 fewer tweets per VIP:** ~15k credits saved × 133 VIPs = **~2M credits per run**.

The 3000 cap was chosen for "comprehensive backfill". In practice:
- The most recent ~1000-1500 tweets capture nearly all current resources.
- Tweets older than 1-2 years rarely link to things still alive on the web (dead-link probe culls many).
- Once `last_seen` cursors warm up (after first run), the cap matters less anyway — incremental runs stop early.

**Recommendation:** drop default `max_tweets` to **1500 for VIP**. Keep per-personality override for power-VIPs (e.g. Karpathy) who need the full backfill. Saves ~30% on Twitter cost in the steady state.

### 2. **Keep non-VIPs at 300 OR consider 100** — small savings, big honesty

Currently non-VIPs are capped at 300. If the goal is "we just want their *recent* signal," 100 would still capture the last month of activity for active accounts and saves ~3-4k credits/non-VIP × ~80 non-VIPs = **~250k credits**.

Open question: do we even *want* Twitter for every non-VIP? Many non-VIPs have empty/junk Twitter — disabling Twitter for non-VIPs entirely could save ~500k credits/run with arguably zero loss. **Decision needed.**

### 3. **Cross-user thread-root cache** — modest, easy

Today the cache is per-user. If user A and user B both reply to Sam Altman's tweet, we hydrate it twice. A shared cache (one JSON / table column at the slug level → at the global level) would dedupe.

**Estimated savings:** 5-15% on Phase 2. **Not huge, but free** once implemented (no quality loss).

### 4. **Filter self-thread continuations from Phase 2 hydration** — small, safe

When the parent tweet is by the same user as the reply (the user threading their own tweets), the parent's content is already in our Phase 1 results. We're paying to re-fetch our own data.

**Estimated savings:** 10-20% of Phase 2 calls (depends on how much each VIP threads). Easy to detect: skip hydration when `conversation_id` belongs to a tweet we already pulled in Phase 1.

### 5. **Drop `includeReplies` for users whose replies are mostly noise** — opt-in

Currently `includeReplies=true` is hardcoded on Phase 1 calls. For users like @sama who tweet a lot but reply rarely with substance, this doubles the volume for little gain.

Would need per-personality flag in `sources.twitter.include_replies` (already exists, default True). Could flip default to **False** and let curators opt in.

**Estimated savings:** 30-50% on Phase 1 for users who flip it. Hard to estimate global savings without per-user analysis.

### 6. **Bookmarks fetcher — already free**

`twitter.Bookmarks` uses cookie-auth (not twitterapi.io), so doesn't count against credits. Keep as-is.

### 7. **`/twitter/user/info` calls — already negligible**

18 credits per call, called ~once per user per run. Total: ~133 credits / run. Don't bother.

---

## What is NOT a good lever (looked at and dismissed)

| Idea | Why not |
|---|---|
| "Cheaper endpoint substitute" | Doesn't exist — both endpoints are 15 cred/tweet. |
| "Fetch fewer URLs from each tweet" | Cost is per-tweet, not per-URL. |
| "Skip parent hydration entirely (option 1 from earlier discussion)" | Would lose 30-50% of Twitter-sourced docs. The parent IS where the URL lives. |
| "Use Twitter's own API instead" | X API v2 is even more expensive (~$100-200/mo for Basic tier with worse limits). |

---

## Recommended action plan (ranked)

| # | Change | Effort | Savings/run | Risk |
|---|---|---|---|---|
| 1 | `max_tweets` VIP: 3000 → 1500 | 1-line code change in `client.py` | **~2M credits (~30%)** | None for steady-state. First-time backfill captures less old history. |
| 2 | Cross-user thread-root cache | New table or shared JSON file | ~500k–1M credits (~10%) | None — pure dedupe. |
| 3 | Skip self-thread parents in Phase 2 | One conditional in `_hydrate_thread_roots` | ~1M credits (~15%) | None — we already have those tweets. |
| 4 | Disable Twitter for non-VIPs by default | Config flag | ~500k credits (~8%) | Loses Twitter signal for non-VIPs. **User decision.** |
| 5 | Drop `includeReplies` to opt-in default | Config flag | Variable, 0-30% | Loses URL discovery from replies for users who don't opt in. **User decision.** |

**Combined potential savings if 1+2+3 ship:** roughly **~4M credits per run**, or ~50-60% of current Twitter spend. Subsequent runs (with warm `last_seen` cursors) cost a fraction of this anyway.

---

## Open questions for you

1. **Lower VIP cap to 1500 or hold at 3000?** This is the biggest call.
2. **Disable Twitter entirely for non-VIPs?** Many of them are dormant on Twitter.
3. **Cross-user cache: per-slug shared file or global PG table?** Latter is cleaner, former is faster to ship.
4. **Should `filter_replies=True` happen *before* `_hydrate_tweets` to skip parent hydration for filtered-out replies?** The earlier concern was valid (we'd lose URLs from junk replies); the safer version is to always hydrate, then filter just the docs. Confirms current behavior is correct — no savings here.
