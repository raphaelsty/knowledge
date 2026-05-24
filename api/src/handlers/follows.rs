//! Follow graph endpoints.
//!
//! Routes:
//!   POST   /api/follow/{slug}   — caller follows the user at {slug}
//!   DELETE /api/follow/{slug}   — caller unfollows
//!   GET    /api/me/following    — caller's followees (slug, name, avatar)
//!   GET    /api/timeline        — recent docs from followees + self
//!
//! All endpoints require a session cookie. The unauthenticated cases
//! return 401 / empty arrays as appropriate so callers can render the
//! signed-out state without special-casing.

use axum::{
    extract::{Path, Query, State},
    http::{
        header::{CACHE_CONTROL, VARY},
        StatusCode,
    },
    response::{IntoResponse, Json, Response},
};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::handlers::auth::current_user;

// ── Anonymous-timeline cache ─────────────────────────────────────────
//
// Logged-out callers all see the SAME deterministic VIP-wide
// timeline (the SQL falls back to `users WHERE vip = TRUE` when
// $1 IS NULL). That makes the response a perfect candidate for an
// in-process cache — one entry per unique query-param signature,
// shared across every anonymous visitor.
//
// Cache hit ratio in production is dominated by the "open the
// front page" call, which is always `/api/timeline` with the
// default 50 limit and no filters. Adding the cache cut that
// path from ~700 ms (SQL) to <1 ms in dev.
//
// Logged-in viewers never read or write this cache — their
// timeline is filtered by their personal follow graph + their own
// `card_seen` events, which is unique per user. A shared cache
// there would either leak data across users or thrash on every
// request.
const ANON_TIMELINE_TTL: Duration = Duration::from_secs(60);

struct AnonTimelineEntry {
    payload: serde_json::Value,
    cached_at: Instant,
}

/// Lazy global cache. The cache key is the canonical query-param
/// signature (CSV-joined source/tags/etc) so two requests with
/// equivalent filters share an entry, but a sources=github query
/// doesn't poison the no-filter cache.
fn anon_timeline_cache() -> &'static RwLock<HashMap<String, AnonTimelineEntry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, AnonTimelineEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

// ── Follow / unfollow ───────────────────────────────────────────────────

/// POST /api/follow/{slug}
pub async fn follow(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Path(slug): Path<String>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let target: Option<i64> = sqlx::query_scalar("SELECT id FROM users WHERE username = $1")
        .bind(&slug)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);
    let Some(target) = target else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if target == me.id {
        return (StatusCode::BAD_REQUEST, "cannot follow yourself").into_response();
    }
    if let Err(e) = sqlx::query(
        "INSERT INTO follows (follower_id, followed_id)
         VALUES ($1, $2)
         ON CONFLICT DO NOTHING",
    )
    .bind(me.id)
    .bind(target)
    .execute(&pool)
    .await
    {
        tracing::error!(error = %e, "follows.add.failed");
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("follow: {e}")).into_response();
    }
    Json(serde_json::json!({ "ok": true, "following": true })).into_response()
}

#[derive(Deserialize)]
pub struct BulkFollowRequest {
    pub slugs: Vec<String>,
}

/// POST /api/me/follow/bulk { slugs: [...] }
///
/// Follow many users in a single round-trip. Used by the onboarding
/// flow so picking a category and committing 8–10 follows doesn't
/// fan out to that many individual POSTs. Idempotent — pre-existing
/// follow rows are left alone.
///
/// Returns `{added: N}` with the count of newly-inserted rows.
pub async fn follow_bulk(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<BulkFollowRequest>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let slugs: Vec<String> = req
        .slugs
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if slugs.is_empty() {
        return Json(serde_json::json!({ "added": 0 })).into_response();
    }
    // ON CONFLICT DO NOTHING on the (follower_id, followed_id) PK gives
    // us idempotency for free. Self-follow is filtered out in SQL via
    // `u.id <> $1` so a slug that happens to be the caller's own
    // doesn't error the whole batch.
    let res = sqlx::query(
        "INSERT INTO follows (follower_id, followed_id)
         SELECT $1, u.id
           FROM users u
          WHERE u.username = ANY($2::text[])
            AND u.id      <> $1
         ON CONFLICT DO NOTHING",
    )
    .bind(me.id)
    .bind(&slugs)
    .execute(&pool)
    .await;
    match res {
        Ok(r) => Json(serde_json::json!({ "added": r.rows_affected() })).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "follows.bulk.failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("bulk follow: {e}"),
            )
                .into_response()
        }
    }
}

/// DELETE /api/follow/{slug}
pub async fn unfollow(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Path(slug): Path<String>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "DELETE FROM follows
           WHERE follower_id = $1
             AND followed_id = (SELECT id FROM users WHERE username = $2)",
    )
    .bind(me.id)
    .bind(&slug)
    .execute(&pool)
    .await;
    Json(serde_json::json!({ "ok": true, "following": false })).into_response()
}

// ── List followees ──────────────────────────────────────────────────────

#[derive(Serialize, sqlx::FromRow)]
pub struct FollowedUser {
    pub id: i64,
    #[sqlx(rename = "username")]
    pub slug: String,
    pub name: String,
    pub avatar: Option<String>,
    pub description: String,
    #[sqlx(rename = "document_count")]
    #[serde(rename = "documentCount")]
    pub document_count: i64,
}

/// GET /api/me/following — followees with display metadata. Used by
/// the frontend to render Follow/Following button state alongside any
/// personality card without a second roundtrip.
pub async fn list_following(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return Json(Vec::<FollowedUser>::new()).into_response();
    };
    let rows = sqlx::query_as::<_, FollowedUser>(
        "SELECT u.id, u.username, u.name, u.avatar, u.description,
                COALESCE(c.cnt, 0)::bigint AS document_count
           FROM follows f
           JOIN users   u ON u.id = f.followed_id
           LEFT JOIN LATERAL (
                SELECT count(*) AS cnt FROM documents d
                  WHERE d.user_id = u.id AND d.deleted = FALSE
           ) c ON true
          WHERE f.follower_id = $1
          ORDER BY f.created_at DESC",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    Json(rows).into_response()
}

// ── Aggregated sources across (followees ∪ self) ────────────────────────

/// GET /api/me/feed/sources — returns one row per distinct source key
/// across the caller's follow graph (plus their own library), with the
/// summed document count. Replaces the per-followee fan-out the feed
/// rail used to do client-side (N round-trips → 1).
///
/// Shape: `[{ key, label, count, user_count }]`, ordered by
/// `user_count DESC, count DESC` — surfaces sources lots of people in
/// the follow graph read, with raw doc volume as the tiebreaker.
pub async fn feed_sources(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    // Anonymous callers see the global VIP feed (same scope as the
    // anonymous timeline), so the source rail must mirror that scope.
    // The `followed` CTE below switches on whether $1 is NULL.
    let me_id: Option<i64> = current_user(&pool, &jar).await.map(|u| u.id);
    // Rank by total document volume across (VIP followees ∪ me).
    // `SUM(v.count)` aggregates the per-user doc counts from the
    // `user_source_counts` view, so a source with 1000 GitHub stars
    // outranks one with 5 niche-blog hits even if more followees
    // happen to use the niche one. The VIP gate skips noise from
    // newly-signed-up followees who haven't been promoted yet —
    // matching the user's mental model of "people I follow on the
    // platform". The caller's own row is included unconditionally
    // (signed-in user = always counted).
    // Top-50 keeps every meaningful chip and trims the long tail.
    let sql = "
        WITH followed AS (
            -- Logged-in: VIP followees ∪ self.
            SELECT u.id AS user_id
              FROM follows f JOIN users u ON u.id = f.followed_id
             WHERE f.follower_id = $1 AND u.vip = TRUE
            UNION
            SELECT $1::bigint AS user_id WHERE $1 IS NOT NULL
            -- Logged-out: every VIP, so the source rail mirrors the
            -- anonymous timeline scope.
            UNION
            SELECT id AS user_id FROM users WHERE vip = TRUE AND $1 IS NULL
        )
        SELECT
            v.source,
            COUNT(DISTINCT v.user_id)::bigint AS user_count,
            SUM(v.count)::bigint            AS doc_count
          FROM user_source_counts v
          JOIN followed f ON f.user_id = v.user_id
         WHERE v.source <> ''
         GROUP BY v.source
         -- Primary key = number of *people* that have this source.
         -- A source that 12 followees use beats one with twice the
         -- raw doc count owned by just 1 person. `doc_count` is the
         -- tiebreaker so equally-popular-among-people sources rank
         -- by reading volume.
         ORDER BY user_count DESC, doc_count DESC
         LIMIT 50
    ";
    let rows: Vec<(String, i64, i64)> = sqlx::query_as(sql)
        .bind(me_id)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();
    let out: Vec<_> = rows
        .into_iter()
        .map(|(key, user_count, doc_count)| {
            let label = crate::handlers::users::source_label(&key);
            // `count` stays on the doc-volume value so existing
            // clients keep showing the doc count next to the chip;
            // `user_count` ships alongside for clients that want to
            // surface "N people" on hover or in a tooltip.
            serde_json::json!({
                "key": key,
                "label": label,
                "count": doc_count,
                "user_count": user_count
            })
        })
        .collect();
    Json(out).into_response()
}

// ── Timeline (recent docs from followees + self) ────────────────────────

#[derive(Deserialize)]
pub struct TimelineParams {
    pub limit: Option<i64>,
    /// ISO-8601 cursor — return docs strictly older than this.
    pub before: Option<String>,
    /// ISO-8601 floor — return docs with `date >= since`. Honoured
    /// by the "Past week / Past month / Past year" filter so the
    /// SQL pre-filter sees the same date range as the index search
    /// path (no JS post-filter).
    pub since: Option<String>,
    /// Comma-separated source keys to include (e.g. `github,blog`).
    /// Empty / absent → no source filter.
    pub sources: Option<String>,
    /// Comma-separated source keys to exclude.
    pub exclude_sources: Option<String>,
    /// Comma-separated tags (AND semantics — all must be present on
    /// the row, matched across `tags` and `extra_tags`).
    pub tags: Option<String>,
    /// Optional fine-grained category slug(s) from `document_categories`.
    /// Comma-separated for multi-select; when one or more slugs are
    /// supplied the candidate scan is restricted to documents the
    /// categorize daemon has assigned to ANY of them (OR semantics —
    /// same convention as the source chips on the left rail). The
    /// singular `category` name is preserved for backward compatibility
    /// with saved deep-links from earlier versions of the picker.
    pub category: Option<String>,
    /// When `true`, the hide-already-seen filter is bypassed. Default
    /// `false`: signed-in viewers don't get re-shown cards whose URL
    /// already has a `card_seen` event from them within the horizon.
    /// Anonymous callers (`me_id IS NULL`) are never filtered — there
    /// is no per-viewer identity to scope against.
    pub include_seen: Option<bool>,
    /// Lookback horizon for the seen filter, in days. Default 30. Older
    /// `card_seen` rows stop hiding the doc so a paper that's been
    /// dormant in your library for a month can resurface naturally.
    pub seen_horizon_days: Option<i32>,
    /// Minimum aggregated dwell (ms) before a card counts as "really
    /// seen". Below this the impression is treated as a scroll-past
    /// and the doc stays in the feed. Default 3000 (3 s). The sum is
    /// taken over every `card_seen` event for the same URL within
    /// the horizon, so multiple short glances can add up to a
    /// genuine read.
    pub min_seen_dwell_ms: Option<i32>,
    /// When `true`, the user's own library + favorites stay in the
    /// feed. Default `false` — for logged-in viewers we hide every
    /// URL they already have in `documents` (own personality) or
    /// `favorite_documents` (starred). Anon callers are never
    /// filtered. Useful for debugging or for an explicit
    /// "show me everything" mode.
    pub include_owned: Option<bool>,
}

/// GET /api/timeline — same payload shape as `/api/feed` (per-URL rows
/// with `sharers` + `sharerCount`) so the existing card renderer works
/// unchanged. Scoped to the caller's follow graph (including their
/// own library, so a user with zero follows still sees their saves).
#[allow(clippy::type_complexity)]
pub async fn timeline(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Query(params): Query<TimelineParams>,
) -> Response {
    // Logged-out callers get a VIP-wide timeline (as if they followed
    // every VIP). Logged-in callers get the classic followees+self
    // timeline. The `followed` CTE below switches on whether $1 is NULL.
    let me_id: Option<i64> = current_user(&pool, &jar).await.map(|u| u.id);
    let limit = params.limit.unwrap_or(50).clamp(1, 200);
    let before: Option<String> = params.before.clone();
    let split_csv = |s: &Option<String>| -> Vec<String> {
        s.as_deref()
            .map(|raw| {
                raw.split(',')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    };
    let sources_inc: Vec<String> = split_csv(&params.sources);
    let sources_exc: Vec<String> = split_csv(&params.exclude_sources);
    let tags_inc: Vec<String> = split_csv(&params.tags)
        .into_iter()
        .map(|t| t.to_lowercase())
        .collect();
    // Multi-select category filter: parse the CSV into a Vec<String>.
    // Empty vec → SQL sees an empty array → the cardinality guard
    // short-circuits to true and the timeline is unfiltered. Same
    // shape as the sources / tags lists already bound below.
    let categories: Vec<String> = split_csv(&params.category)
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect();

    // Anonymous cache lookup. The cache key is the full canonical
    // signature of the request — same key in == same response out.
    // Logged-in viewers skip this entirely; their timeline is
    // per-user and would either leak data across users or thrash.
    //
    // Cache headers on a hit:
    //   * `Cache-Control: public, max-age=60` — every layer (browser,
    //     Caddy if it ever caches, any CDN) can hold the response
    //     for the same TTL window as the in-process cache.
    //   * `Vary: Cookie` — once the visitor signs in, the browser
    //     won't serve a cached anon body to a logged-in request.
    let anon_cache_key: Option<String> = if me_id.is_none() {
        Some(build_anon_cache_key(
            limit,
            before.as_deref(),
            params.since.as_deref(),
            &sources_inc,
            &sources_exc,
            &tags_inc,
            &categories,
        ))
    } else {
        None
    };
    if let Some(key) = &anon_cache_key {
        let cache = anon_timeline_cache().read().await;
        if let Some(entry) = cache.get(key) {
            if entry.cached_at.elapsed() < ANON_TIMELINE_TTL {
                return (
                    [
                        (
                            CACHE_CONTROL,
                            "public, max-age=60, stale-while-revalidate=120",
                        ),
                        (VARY, "Cookie"),
                    ],
                    Json(entry.payload.clone()),
                )
                    .into_response();
            }
        }
    }

    let since: Option<String> = params.since.clone();
    let include_seen: bool = params.include_seen.unwrap_or(false);
    let seen_horizon_days: i32 = params.seen_horizon_days.unwrap_or(30).clamp(1, 365);
    // Aggregated-dwell threshold to count as "really seen". Default
    // 1.5 s — matches the client's MIN_DWELL_MS floor, so any
    // card_seen event the tracker bothered to fire counts as a real
    // impression. Earlier 3 s was leaving too many "I saw this for
    // ~2 s" URLs in the feed (May 2026: 19 / 80 hides instead of
    // 48 / 80). Clamped to [0, MAX_DWELL_MS=120 000] since values
    // outside that bound would never match the data the client
    // actually emits.
    let min_seen_dwell_ms: i32 = params.min_seen_dwell_ms.unwrap_or(1500).clamp(0, 120_000);
    // `include_owned` defaults to false — logged-in viewers don't
    // want to discover docs they've already saved. Toggling it on
    // bypasses the documents + favorite_documents NOT EXISTS
    // filters in both query paths.
    let include_owned: bool = params.include_owned.unwrap_or(false);

    // Row struct used by both the snapshot fast-path and the live
    // CTE fallback. Field order matches the final SELECT projection
    // of both queries 1:1.
    #[derive(Clone, sqlx::FromRow)]
    struct TimelineRow {
        url: String,
        title: String,
        date_str: String,
        summary: String,
        clean_title: String,
        clean_summary: String,
        urls: Vec<String>,
        tags: Vec<String>,
        source: String,
        source_url: Option<String>,
        linked_urls: serde_json::Value,
        link_hosts: Vec<String>,
        primary_user_id: i64,
        score: f64,
        sharers: serde_json::Value,
        sharer_count: i64,
        already_seen: bool,
    }

    // ── Snapshot fast-path ────────────────────────────────────────
    //
    // `feed_snapshot` is rebuilt hourly by knowledge-feed-snapshot.
    // When fresh (<3 h old), every timeline request becomes a
    // single indexed scan against ~20 k pre-scored rows instead of
    // replaying the 400 k-doc CTE — empirically ~50 ms instead of
    // ~700 ms for a logged-in feed.
    //
    // Bypass conditions:
    //   * Category filter active — the snapshot doesn't carry per-
    //     doc category assignments. Live query handles the
    //     document_category_assignments join.
    //   * Snapshot empty or stale (>3 h). The freshness probe below
    //     is a single-row index lookup.
    //
    // Either bypass cleanly drops to the live query lower down.
    let snapshot_fresh: bool = if !categories.is_empty() {
        false
    } else {
        sqlx::query_scalar::<_, Option<bool>>(
            "SELECT (MAX(refreshed_at) > now() - interval '3 hours') FROM feed_snapshot",
        )
        .fetch_one(&pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(false)
    };

    // Snapshot query. Same final projection as the live CTE so both
    // share the TimelineRow struct + the diversity-pass code below.
    // Per-viewer score additions (followee_share, fresh-self) ride
    // on top of the precomputed score column.
    let snapshot_sql = "
        WITH followed AS (
            SELECT followed_id AS user_id FROM follows WHERE follower_id = $1
            UNION
            SELECT $1::bigint AS user_id WHERE $1 IS NOT NULL
            UNION
            SELECT id AS user_id FROM users WHERE vip = TRUE AND $1 IS NULL
        ),
        followed_ids AS (
            SELECT COALESCE(array_agg(user_id), '{}'::bigint[]) AS ids FROM followed
        )
        SELECT
            s.url,
            s.title,
            COALESCE(to_char(s.date, 'YYYY-MM-DD'), '') AS date_str,
            s.summary, s.clean_title, s.clean_summary,
            s.urls, s.tags, s.source, s.source_url,
            s.linked_urls, s.link_hosts,
            s.primary_user_id,
            -- effective score = precomputed score
            --                 + followee_share bonus (capped at 3 extras × 1.5)
            --                 + 50 if the viewer authored it < 1 h ago
            -- IN (SELECT user_id FROM followed) rather than
            -- ANY((SELECT ids FROM followed_ids)) because PG
            -- otherwise interprets the inner subquery as
            -- compare-scalar-sid-against-an-array-literal and
            -- fails type resolution at parse time.
            (s.score
             + LEAST(3, GREATEST(0, (
                   SELECT count(*)::int
                     FROM unnest(s.sharer_user_ids) sid
                    WHERE sid IN (SELECT user_id FROM followed)
               ) - 1)) * 1.5
             + CASE
                   WHEN s.primary_user_id = $1
                        AND s.refreshed_at > now() - interval '1 hour'
                     THEN 50
                   ELSE 0
               END
            ) AS score,
            s.sharers,
            -- Cast INT4 → INT8 so TimelineRow.sharer_count (i64)
            -- matches both code paths. The snapshot stores it as
            -- INT to save 4 bytes per row; the live CTE's
            -- count(*) is bigint by default.
            s.sharer_count::bigint AS sharer_count,
            -- Dwell-aware already_seen flag — same semantics as the
            -- live query so the dim-and-pill UI looks identical.
            ($9::bool AND $1 IS NOT NULL AND COALESCE((
                 SELECT SUM(COALESCE(e.dwell_ms, 0))::bigint
                   FROM events e
                  WHERE e.viewer_user_id = $1
                    AND e.event_type    = 7
                    AND e.doc_url       = s.url
                    AND e.created_at    < now() - interval '10 minutes'
                    AND e.created_at    > now() - ($10::int || ' days')::interval
             ), 0) >= $11::int) AS already_seen
          FROM feed_snapshot s
         WHERE
               -- Logged-in: sharer_user_ids must intersect followees.
               -- Anon: rely on any_vip_sharer (partial-indexed scan).
               -- The `&&` operator wants `bigint[]` on both sides, so
               -- we pull the array out of `followed_ids` directly.
               ($1 IS NULL
                OR s.sharer_user_ids && (SELECT ids FROM followed_ids LIMIT 1))
           AND ($1 IS NOT NULL OR s.any_vip_sharer = TRUE)
           -- Date filters: `before` (cursor) and `since` (window).
           AND ($3::timestamptz IS NULL OR s.date <  $3::timestamptz)
           AND ($7::timestamptz IS NULL OR s.date >= $7::timestamptz)
           -- Source filters (same two-pronged source / link_hosts).
           AND (cardinality($4::text[]) = 0
                OR s.source = ANY($4::text[])
                OR s.link_hosts && $4::text[])
           AND (cardinality($5::text[]) = 0 OR NOT s.source = ANY($5::text[]))
           -- Tag AND-semantics (matched on s.tags only — the snapshot
           -- folds extra_tags into the source `tags` column at
           -- refresh time).
           AND (cardinality($6::text[]) = 0
                OR (SELECT bool_and(
                        EXISTS (
                            SELECT 1 FROM unnest(s.tags) t WHERE lower(t) = q
                        )
                    )
                    FROM unnest($6::text[]) AS q))
           -- Hide-seen filter (logged-in, include_seen=false).
           AND ($1 IS NULL
                OR $9::bool = TRUE
                OR COALESCE((
                    SELECT SUM(COALESCE(e.dwell_ms, 0))::bigint
                      FROM events e
                     WHERE e.viewer_user_id = $1
                       AND e.event_type    = 7
                       AND e.doc_url       = s.url
                       AND e.created_at    < now() - interval '10 minutes'
                       AND e.created_at    > now() - ($10::int || ' days')::interval
               ), 0) < $11::int)
           -- Exclude-owned filter — logged-in callers don't want to
           -- discover docs they've already saved or starred. Drops
           -- any URL present in their own `documents` rows OR in
           -- `favorite_documents`. `$12::bool = TRUE` bypasses the
           -- filter entirely (`include_owned` query param).
           AND ($1 IS NULL OR $12::bool = TRUE OR (
                NOT EXISTS (
                    SELECT 1 FROM documents d
                     WHERE d.user_id = $1
                       AND d.url     = s.url
                       AND d.deleted = FALSE
                )
                AND NOT EXISTS (
                    SELECT 1 FROM favorite_documents fd
                     WHERE fd.user_id = $1
                       AND fd.url     = s.url
                )
                -- Also catch canonical-url overlap so a paper saved
                -- as arxiv.org/abs/X hides arxiv.org/pdf/X in the
                -- feed. `documents.canonical_url` is a GENERATED
                -- STORED column so the lookup is cheap.
                AND NOT EXISTS (
                    SELECT 1 FROM documents d
                     WHERE d.user_id       = $1
                       AND d.canonical_url = s.canonical_url
                       AND d.deleted       = FALSE
                )
           ))
         ORDER BY score DESC, s.date DESC NULLS LAST, s.url
         LIMIT $2 * 2
    ";

    // The feed is built in four layers (followed → candidates →
    // dedup → diversified) and scored along the way:
    //
    //   1. `candidates`  — pull the most-recent `$2 * 40` rows
    //                      (capped at 8000) across (followees ∪ self).
    //                      Each row carries a precomputed `sci_score`
    //                      based purely on the resource type (no
    //                      keyword matching): direct scientific
    //                      sources (arxiv/scholar/huggingface) score
    //                      3, tweets linking to a scientific host
    //                      (arxiv.org / huggingface.co / openreview…)
    //                      also score 3, github code releases score
    //                      1, everything else 0.
    //   2. `url_share`   — for every URL among the candidates, how
    //                      many followees share it (`followee_share`)
    //                      and how many users overall share it
    //                      (`total_share`). Both feed the score.
    //   3. `dedup`       — collapse to one row per URL. When several
    //                      followees share the same URL the row is
    //                      attributed to the most notable sharer
    //                      (vip → twitter_followers → citations).
    //   4. `scored`      — composite score: scientific bonus + a
    //                      recency bucket (last 1 / 3 / 7 / 14 / 30
    //                      days) + a multi-sharer term (sublinear in
    //                      the followee count, sublinear in the total
    //                      sharer count) + a notability bonus driven
    //                      by the primary sharer's follower count.
    //   5. `diversified` — ROW_NUMBER() OVER (PARTITION BY
    //                      primary_user_id ORDER BY score DESC). The
    //                      final ORDER BY is `user_rank, score DESC`,
    //                      which gives a round-robin: everyone's best
    //                      post first, then everyone's second-best,
    //                      etc. Guarantees no user appears twice
    //                      consecutively as long as there are enough
    //                      sharers to fill `$2` slots.
    //
    // The 40× over-fetch is sized for the rerank + diversity pass —
    // dropping the cap below ~20× starves the round-robin for users
    // who don't post often.
    let sql = "
        WITH followed AS (
            -- Logged-in: followees ∪ self.
            SELECT followed_id AS user_id FROM follows WHERE follower_id = $1
            UNION
            SELECT $1::bigint AS user_id WHERE $1 IS NOT NULL
            -- Logged-out: act as if following every VIP.
            UNION
            SELECT id AS user_id FROM users WHERE vip = TRUE AND $1 IS NULL
        ),
        candidates AS MATERIALIZED (
            SELECT d.user_id, d.url, d.title, d.date, d.summary,
                   d.clean_title, d.clean_summary, d.urls,
                   d.tags,
                   d.extra_tags, d.source, d.source_url, d.created_at,
                   d.linked_urls, d.link_hosts,
                   -- Canonical URL drives the share aggregation below
                   -- so arxiv.org/abs/X, /pdf/X, /abs/Xv2 collapse to
                   -- one card. `canonical_referenced_urls` is the
                   -- doc's own canonical URL PLUS every URL it cites
                   -- (linked_urls + the cleaned `urls` array) — used
                   -- by the LATERAL sharer expansion to surface
                   -- anyone who also references at least one of the
                   -- same URLs (e.g. a tweet linking the same paper).
                   d.canonical_url,
                   d.canonical_referenced_urls,
                   -- Resource-type signal — no keyword / tag matching.
                   -- 3 = direct scientific source OR a tweet that
                   --     links to a scientific host;
                   -- 1 = code release (github);
                   -- 0 = everything else.
                   CASE
                     WHEN d.source IN ('arxiv', 'scholar', 'huggingface')
                       THEN 3
                     WHEN d.source = 'twitter'
                          AND d.link_hosts && ARRAY[
                            'arxiv', 'arxiv.org',
                            'huggingface', 'huggingface.co', 'hf.co',
                            'paperswithcode.com', 'openreview.net',
                            'aclanthology.org', 'distill.pub',
                            'jmlr.org', 'biorxiv.org', 'medrxiv.org',
                            'semanticscholar.org', 'scholar.google.com',
                            'neurips.cc', 'icml.cc'
                          ]::text[]
                       THEN 3
                     WHEN d.source IN ('github', 'github_repos')
                       THEN 1
                     ELSE 0
                   END AS sci_score
              FROM documents d
              JOIN followed f ON f.user_id = d.user_id
             WHERE d.deleted = FALSE
               AND d.date IS NOT NULL
               AND ($3::timestamptz IS NULL OR d.date < $3::timestamptz)
               AND ($7::timestamptz IS NULL OR d.date >= $7::timestamptz)
               -- Source filter is two-pronged: the canonical `source`
               -- column OR any host inside `link_hosts`. A tweet
               -- linking hornet.dev therefore surfaces under both
               -- the `twitter` chip and the `hornet.dev` chip
               -- without us minting a second row.
               AND (cardinality($4::text[]) = 0
                    OR d.source = ANY($4::text[])
                    OR d.link_hosts && $4::text[])
               AND (cardinality($5::text[]) = 0 OR NOT d.source = ANY($5::text[]))
               AND (cardinality($6::text[]) = 0
                    OR (SELECT bool_and(
                            EXISTS (
                                SELECT 1 FROM unnest(d.tags) t WHERE lower(t) = q
                            ) OR EXISTS (
                                SELECT 1 FROM unnest(d.extra_tags) t WHERE lower(t) = q
                            )
                        )
                        FROM unnest($6::text[]) AS q))
               -- Fine-grained category filter (multi-select, OR
               -- semantics). The categorize daemon writes 0-3 rows
               -- per doc into document_category_assignments keyed on
               -- (user_id, url, category_id) with a slug FK through
               -- document_categories. With multiple selected slugs
               -- we accept a doc if it has ANY of them assigned
               -- (primary OR secondary) — matches the source-chip
               -- convention on the left rail.
               AND (cardinality($8::text[]) = 0
                    OR EXISTS (
                        SELECT 1
                          FROM document_category_assignments a
                          JOIN document_categories dc ON dc.id = a.category_id
                         WHERE a.user_id = d.user_id
                           AND a.url     = d.url
                           AND dc.slug   = ANY($8::text[])
                    ))
               -- Hide-already-seen filter. Logged-in only ($1 IS NOT
               -- NULL); bypassed when the caller passes
               -- include_seen=true ($9 = TRUE).
               --
               -- A row only counts as really seen if the viewer
               -- aggregated dwell on the URL meets $11 ms inside
               -- the horizon. Scroll-pasts (sub-second card_seen
               -- events) do not add up to a hide. Multiple short
               -- glances do — the SUM lets a card the user came
               -- back to three times for ~1.2 s each cross a 3 s
               -- threshold the way a single deliberate read would.
               --
               -- Also: a 10-minute grace window — only events older
               -- than that contribute to the sum, so a card the
               -- viewer just engaged with stays visible for the
               -- current scroll and only disappears on the next
               -- visit. Indexed by idx_events_viewer_doc_seen.
               AND ($1 IS NULL
                    OR $9::bool = TRUE
                    OR COALESCE((
                        SELECT SUM(COALESCE(e.dwell_ms, 0))::bigint
                          FROM events e
                         WHERE e.viewer_user_id = $1
                           AND e.event_type    = 7
                           AND e.doc_url       = d.url
                           AND e.created_at    < now() - interval '10 minutes'
                           AND e.created_at    > now() - ($10::int || ' days')::interval
                    ), 0) < $11::int)
               -- Exclude-owned filter — drop docs the viewer already
               -- has in their own library or has starred. Bypassed
               -- by include_owned=true ($12). Mirrors the snapshot
               -- path's equivalent clause; uses the same indexes
               -- (documents PK on (user_id, url) and
               -- favorite_documents PK on (user_id, url)).
               AND ($1 IS NULL OR $12::bool = TRUE OR (
                    NOT EXISTS (
                        SELECT 1 FROM documents od
                         WHERE od.user_id = $1
                           AND od.url     = d.url
                           AND od.deleted = FALSE
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM favorite_documents fd
                         WHERE fd.user_id = $1
                           AND fd.url     = d.url
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM documents od2
                         WHERE od2.user_id       = $1
                           AND od2.canonical_url = d.canonical_url
                           AND od2.deleted       = FALSE
                    )
               ))
             ORDER BY d.date DESC
             -- Over-fetch by 16x the requested limit — gives the
             -- diversity pass headroom to reorder past consecutive
             -- duplicates without re-paginating, while keeping the
             -- candidate scan small enough to return fast (~600ms
             -- when scanning every VIP, ~50ms for a 30-follow user).
             LIMIT LEAST(2000, $2 * 16)
        ),
        url_share AS MATERIALIZED (
            -- Followee share count is bounded by candidates (so the
            -- scan stays small); total share count needs a separate
            -- lookup but only for canonical URLs that survived the
            -- candidate pass, which is the same order of magnitude
            -- as $2 * 16. Grouping by `canonical_url` collapses
            -- arxiv.org/abs/X, /pdf/X, /abs/Xv2 etc. so a paper saved
            -- under three URL variants counts as one shared resource.
            SELECT c.canonical_url AS canonical_url,
                   COUNT(DISTINCT c.user_id) AS followee_share,
                   (SELECT COUNT(DISTINCT d2.user_id)
                      FROM documents d2
                     WHERE d2.canonical_url = c.canonical_url
                       AND d2.deleted = FALSE) AS total_share
              FROM candidates c
             GROUP BY c.canonical_url
        ),
        candidate_anchors AS MATERIALIZED (
            -- Per-candidate anchor URL used by the dedup step below.
            -- The anchor identifies WHICH RESOURCE this doc is about,
            -- so different docs that discuss the same paper / repo /
            -- model collapse into one feed card.
            --
            -- Priority: known scientific / code hosts first (arxiv,
            -- huggingface, github, openreview, doi, …). Among URLs of
            -- the same priority, lexicographic min is the stable
            -- tiebreaker. Falls back to the doc's own canonical_url
            -- when no priority URL is present — so a plain bookmark
            -- with no external references just anchors on itself
            -- (and stays distinct from other plain bookmarks).
            --
            -- Pre-computes `image_count` and `url_count` here so the
            -- dedup step can pick the visually-richest representative
            -- (most preview images, then most referenced URLs).
            SELECT c.*,
                   COALESCE(
                       (SELECT ref
                          FROM unnest(c.canonical_referenced_urls) ref
                         ORDER BY CASE
                             WHEN ref LIKE 'https://arxiv.org/abs/%'       THEN 1
                             WHEN ref LIKE 'https://huggingface.co/%'      THEN 2
                             WHEN ref LIKE 'https://github.com/%'          THEN 3
                             WHEN ref LIKE 'https://openreview.net/%'      THEN 4
                             WHEN ref LIKE 'https://doi.org/%'             THEN 5
                             WHEN ref LIKE 'https://paperswithcode.com/%'  THEN 6
                             WHEN ref LIKE 'https://aclanthology.org/%'    THEN 7
                             WHEN ref LIKE 'https://semanticscholar.org/%' THEN 8
                             WHEN ref LIKE 'https://distill.pub/%'         THEN 9
                             WHEN ref LIKE 'https://biorxiv.org/%'         THEN 10
                             WHEN ref LIKE 'https://medrxiv.org/%'         THEN 11
                             ELSE 99
                         END, ref
                         LIMIT 1),
                       c.canonical_url
                   ) AS anchor_url,
                   COALESCE((
                       SELECT count(*)::int
                         FROM jsonb_array_elements(c.linked_urls) e
                        WHERE COALESCE(e->>'image', '') <> ''
                   ), 0) AS image_count,
                   cardinality(c.canonical_referenced_urls) AS url_count
              FROM candidates c
        ),
        dedup AS (
            -- One row per ANCHOR — avoid showing the same resource
            -- twice in the feed. Two distinct tweets that link the
            -- same paper now collapse to one card; the arxiv doc
            -- itself and tweets about it also collapse.
            --
            -- Representative selection: pick the visually richest
            -- copy first (most preview images, then most referenced
            -- URLs), then the existing tiebreakers (recency,
            -- VIP-ness, follower count, citations). The avatar stack
            -- still shows every sharer in the cluster because the
            -- LATERAL `s.sharers` query expands via
            -- `canonical_referenced_urls && m.canonical_referenced_urls`.
            SELECT DISTINCT ON (ca.anchor_url)
                   ca.user_id AS primary_user_id,
                   ca.url, ca.title, ca.date, ca.summary,
                   ca.clean_title, ca.clean_summary, ca.urls,
                   ca.tags,
                   ca.extra_tags, ca.source, ca.source_url, ca.created_at,
                   ca.linked_urls, ca.link_hosts, ca.sci_score,
                   ca.canonical_url,
                   ca.canonical_referenced_urls,
                   u.twitter_followers AS pri_twitter_followers,
                   u.citations         AS pri_citations,
                   u.vip               AS pri_vip
              FROM candidate_anchors ca
              JOIN users u ON u.id = ca.user_id
             ORDER BY ca.anchor_url,
                      ca.image_count DESC,
                      ca.url_count DESC,
                      ca.date DESC,
                      u.vip DESC,
                      COALESCE(u.twitter_followers, 0) DESC,
                      COALESCE(u.citations, 0) DESC
        ),
        scored AS (
            SELECT d.*,
                   COALESCE(us.followee_share, 1) AS followee_share,
                   COALESCE(us.total_share, 1)    AS total_share,
                   -- Composite score. Tunables:
                   --   sci  ×6       — pushes scientific resources
                   --                   to the top of the feed.
                   --   recency      — weekly buckets so the whole
                   --                   current-week cohort competes
                   --                   on signal, not on hour-of-day.
                   --                   Docs inside the same rolling
                   --                   7-day window all get the same
                   --                   recency bonus; the bonus steps
                   --                   down at each week boundary.
                   --   followee_share × 1.5 (capped at 3 extras) —
                   --                   reward URLs surfaced by
                   --                   multiple followees.
                   --   total_share   — sublinear (log) so popular
                   --                   resources outside the follow
                   --                   graph also bubble up, without
                   --                   drowning the followee signal.
                   --   primary VIP   — +0.8 nudge.
                   --   followers     — sublinear bump on the most
                   --                   notable sharer (10k twitter
                   --                   followers ≈ +1.0).
                   d.sci_score::float * 6
                   + CASE
                       WHEN d.date >= now() - INTERVAL '7 days'  THEN 5
                       WHEN d.date >= now() - INTERVAL '14 days' THEN 4
                       WHEN d.date >= now() - INTERVAL '21 days' THEN 3
                       WHEN d.date >= now() - INTERVAL '28 days' THEN 2
                       WHEN d.date >= now() - INTERVAL '35 days' THEN 1
                       ELSE 0
                     END
                   + LEAST(3, GREATEST(0, COALESCE(us.followee_share, 1) - 1)) * 1.5
                   + LEAST(2, LN(GREATEST(1, COALESCE(us.total_share, 1)::float))) * 0.7
                   + CASE WHEN d.pri_vip THEN 0.8 ELSE 0 END
                   + LEAST(
                       1.5,
                       LN(GREATEST(1, COALESCE(d.pri_twitter_followers, 0) / 10000.0))
                     )
                   -- Rich-tweet bonus: tweets whose `linked_urls`
                   -- payload contains at least one entry with a
                   -- non-empty `image` field have a preview card
                   -- (paper card / image / video thumbnail) and tend
                   -- to be more interesting visually. Detected by
                   -- structure, never by inspecting tweet text.
                   + CASE
                       WHEN d.source = 'twitter'
                            AND EXISTS (
                              SELECT 1
                                FROM jsonb_array_elements(d.linked_urls) e
                               WHERE COALESCE(e->>'image', '') <> ''
                            )
                         THEN 1.5
                       ELSE 0
                     END
                   -- Fresh-self-post boost: docs the signed-in user
                   -- authored in the last hour get a huge score
                   -- bump so they surface at the very top of their
                   -- own feed immediately after a compose submit
                   -- (or any other client-side bulk-insert). The
                   -- boost decays - at 1h the doc settles back
                   -- into its natural score order, so it does not
                   -- permanently camp at the top of the feed.
                   -- Without this a source=bookmark post
                   -- (sci_score=0) was being scored below ~200
                   -- followee posts.
                   + CASE
                       WHEN d.primary_user_id = $1
                            AND d.created_at > NOW() - INTERVAL '1 hour'
                         THEN 50
                       ELSE 0
                     END
                   AS score
              FROM dedup d
              LEFT JOIN url_share us ON us.canonical_url = d.canonical_url
        ),
        ordered AS (
            -- Pure score-based ranking — we no longer round-robin
            -- the users in SQL. The Rust handler does a light
            -- post-pass that *only* prevents two consecutive posts
            -- from the same primary user, so most of someone's
            -- output remains visible in the feed; a follower's
            -- recent posts can sit two slots apart instead of being
            -- buried at the end.
            SELECT s.*
              FROM scored s
        )
        SELECT
            m.url, m.title,
            COALESCE(to_char(m.date, 'YYYY-MM-DD'), '') AS date_str,
            m.summary,
            m.clean_title, m.clean_summary, m.urls,
            m.tags, m.source, m.source_url,
            m.linked_urls, m.link_hosts,
            m.primary_user_id, m.score,
            s.sharers, s.sharer_count,
            -- Per-row already-seen flag. Mirrors the hide-filter
            -- semantics: a row is flagged seen only when the
            -- viewer aggregated dwell on this URL meets the
            -- threshold within the horizon (and outside the 10-min
            -- grace). Short scroll-pasts do not trip the dim.
            -- Short-circuited by `$9::bool AND ...` so the common
            -- hide-seen path skips the SUM entirely.
            ($9::bool AND $1 IS NOT NULL AND COALESCE((
                SELECT SUM(COALESCE(e.dwell_ms, 0))::bigint
                  FROM events e
                 WHERE e.viewer_user_id = $1
                   AND e.event_type    = 7
                   AND e.doc_url       = m.url
                   AND e.created_at    < now() - interval '10 minutes'
                   AND e.created_at    > now() - ($10::int || ' days')::interval
            ), 0) >= $11::int) AS already_seen
          FROM ordered m
          JOIN LATERAL (
              -- Sharers = EVERY non-deleted owner of this URL OR of a
              -- doc that references this URL. Two-pronged lookup:
              --
              --   1. `d.canonical_url = m.canonical_url`
              --        Direct owners — anyone whose own `url`
              --        canonicalizes to the same thing (arxiv abs/pdf
              --        variants collapse here).
              --
              --   2. `d.canonical_referenced_urls && m.canonical_referenced_urls`
              --        Indirect: anyone whose doc REFERENCES at least
              --        one URL the current card also references. A
              --        tweet linking a paper, a blog post linking a
              --        paper, and the paper's own arxiv page all
              --        cross-pollinate — so the avatar stack on the
              --        paper card surfaces every personality who
              --        tweeted about it, and vice versa.
              --
              -- Both prongs are GIN-indexed (canonical_url btree;
              -- canonical_referenced_urls gin), so this is cheap.
              -- `canonical_referenced_urls` always includes the
              -- doc's own canonical_url so prong 2 also matches
              -- direct owners — but we keep prong 1 explicit so
              -- the query optimiser can pick the cheaper plan.
              SELECT jsonb_agg(DISTINCT
                         jsonb_build_object(
                             'slug',             u.username,
                             'name',             u.name,
                             'avatar',           u.avatar,
                             'twitterFollowers', u.twitter_followers
                         )
                     )       AS sharers,
                     count(DISTINCT u.id) AS sharer_count
                FROM documents d
                JOIN users    u ON u.id = d.user_id
               WHERE d.deleted = FALSE
                 AND (
                       d.canonical_url = m.canonical_url
                    OR (cardinality(m.canonical_referenced_urls) > 0
                        AND d.canonical_referenced_urls && m.canonical_referenced_urls)
                 )
          ) s ON true
         -- Pure score-based ordering. Over-fetch by 2× so the
         -- Rust post-pass can defer immediate same-user duplicates
         -- without running out of rows to backfill.
         ORDER BY m.score DESC,
                  m.date DESC,
                  m.created_at DESC,
                  m.url
         LIMIT $2 * 2
    ";

    // Helper closure that runs the live CTE query. Used both as the
    // primary path (snapshot bypassed / stale / empty) and as the
    // fallback when the snapshot path returns zero rows.
    let bind_live = || {
        sqlx::query_as::<_, TimelineRow>(sql)
            .bind(me_id)
            .bind(limit)
            .bind(before.clone())
            .bind(&sources_inc)
            .bind(&sources_exc)
            .bind(&tags_inc)
            .bind(since.clone())
            .bind(&categories)
            .bind(include_seen)
            .bind(seen_horizon_days)
            .bind(min_seen_dwell_ms)
            .bind(include_owned)
    };

    let rows: Vec<TimelineRow> = if snapshot_fresh {
        // Try the snapshot fast path. Fall back to the live query
        // when it returns no rows (cold filter combo, daemon outage
        // mid-rebuild) or errors out — better to serve a slow page
        // than an empty one.
        let snapshot_result = sqlx::query_as::<_, TimelineRow>(snapshot_sql)
            .bind(me_id)
            .bind(limit)
            .bind(before.clone())
            .bind(&sources_inc)
            .bind(&sources_exc)
            .bind(&tags_inc)
            .bind(since.clone())
            .bind(&categories)
            .bind(include_seen)
            .bind(seen_horizon_days)
            .bind(min_seen_dwell_ms)
            .bind(include_owned)
            .fetch_all(&pool)
            .await;
        match snapshot_result {
            Ok(rs) if !rs.is_empty() => rs,
            Ok(_) => {
                tracing::debug!("timeline.snapshot.empty — falling back to live");
                bind_live().fetch_all(&pool).await.unwrap_or_default()
            }
            Err(e) => {
                tracing::warn!(error = %e, "timeline.snapshot.failed — falling back to live");
                bind_live().fetch_all(&pool).await.unwrap_or_default()
            }
        }
    } else {
        match bind_live().fetch_all(&pool).await {
            Ok(rs) => rs,
            Err(e) => {
                tracing::error!(error = %e, "timeline.query.failed");
                Vec::new()
            }
        }
    };

    // Diversity pass — soft, super-linear decay so prolific sharers
    // get spread out without us hiding any of their posts.
    //
    //   effective_score = base_score
    //                     - DECAY * prior_appearances ^ EXP
    //                     - (ADJACENT if same user as the last emit)
    //
    // Tuned for "see most of someone's posts, spread out". With
    // DECAY=2.0 and EXP=1.3 a prolific sharer's 2nd post lands ~10
    // slots after their 1st, their 3rd ~30 slots later, etc. No
    // posts get hidden — they just shift down so first-time sharers
    // get a fair chance at the top of the feed.
    // O(N²) over a queue capped at limit × 2 (≤ 400) — negligible.
    const DECAY: f64 = 2.0;
    const EXP: f64 = 1.3;
    const ADJACENT: f64 = 18.0;
    let mut emit_order: Vec<usize> = Vec::with_capacity(rows.len());
    let mut remaining: Vec<usize> = (0..rows.len()).collect();
    let mut emit_count: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    let mut last_user: Option<i64> = None;
    while !remaining.is_empty() && emit_order.len() < limit as usize {
        let (best_pos, _) = remaining
            .iter()
            .enumerate()
            .map(|(pos, &row_idx)| {
                let user = rows[row_idx].primary_user_id;
                let prior = *emit_count.get(&user).unwrap_or(&0) as f64;
                let adjacent_pen = if last_user == Some(user) {
                    ADJACENT
                } else {
                    0.0
                };
                let eff = rows[row_idx].score - DECAY * prior.powf(EXP) - adjacent_pen;
                (pos, eff)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty remaining");
        let chosen = remaining.remove(best_pos);
        let user = rows[chosen].primary_user_id;
        *emit_count.entry(user).or_insert(0) += 1;
        last_user = Some(user);
        emit_order.push(chosen);
    }

    let mut out: Vec<serde_json::Value> = emit_order
        .into_iter()
        .map(|i| {
            let r = rows[i].clone();
            serde_json::json!({
                "url": r.url,
                "title": r.title,
                "date": r.date_str,
                "summary": r.summary,
                // Pedagogical-rewriter outputs. Empty string when the
                // clean daemon hasn't processed this row yet, or when
                // it processed it but found no useful summary to add
                // (skeletal HF cards, mood tweets). The frontend
                // falls back to `summary` when these are empty.
                "cleanTitle": r.clean_title,
                "cleanSummary": r.clean_summary,
                // Flat URL list extracted from the raw `summary` +
                // OG-cluster — the frontend uses this to surface
                // every URL the original post referenced even when
                // the cleaned summary dropped the label that wrapped
                // it (e.g. a trailing 'Paper:' line).
                "urls": r.urls,
                "tags": r.tags,
                "source": r.source,
                "source_url": r.source_url,
                "linked_urls": r.linked_urls,
                "link_hosts": r.link_hosts,
                "sharers": r.sharers,
                "sharerCount": r.sharer_count,
                "picked": false,
                // True when the viewer has a `card_seen` event for
                // this URL within the horizon. Only flips on when the
                // client passed include_seen=1 (otherwise the row was
                // filtered out upstream).
                "alreadySeen": r.already_seen,
            })
        })
        .collect();

    // ── HackerNews front-page picks (latest run only) ────────────
    //
    // Surfaced in the feed alongside followees' bookmarks, never in
    // anyone's personal page. We:
    //   • emit source = "hackernews" so the existing HN logo / source
    //     pill renders unchanged on the card;
    //   • set "picked" = true so the frontend can offer a "Save to
    //     library" action instead of the favorite heart (the doc
    //     isn't yet a row in `documents`);
    //   • honour source include/exclude + the `before` cursor so
    //     filtering and pagination stay coherent;
    //   • skip when the caller filtered to a source set that doesn't
    //     include hackernews (cheap short-circuit).
    let hn_allowed = me_id.is_some()
        && (sources_inc.is_empty() || sources_inc.iter().any(|s| s == "hackernews"))
        && !sources_exc.iter().any(|s| s == "hackernews")
        && tags_inc.is_empty();
    // Picks belong to the FIRST page only — they're a personalised
    // surface, not paginated content. Once the caller scrolls past
    // today's picks they shouldn't reappear in older windows.
    if hn_allowed && params.before.is_none() {
        // Picks come pre-ordered by upvote count: the script writes
        // them with `rank` reflecting "top N by ColBERT mean, then
        // re-sorted by HN points desc" so the highest-traffic
        // relevant item gets `rank = 1`.
        let pick_sql = "
            SELECT
                i.url,
                i.title,
                to_char(r.fetched_at, 'YYYY-MM-DD') AS date_str,
                i.summary,
                ('https://news.ycombinator.com/item?id=' || i.hn_id::text) AS source_url
              FROM hn_user_picks p
              JOIN hn_frontpage_items i
                ON i.run_id = p.run_id AND i.hn_id = p.hn_id
              JOIN hn_frontpage_runs r
                ON r.id = p.run_id
             WHERE p.user_id = $1
               AND p.run_id = (SELECT MAX(id) FROM hn_frontpage_runs)
             ORDER BY p.rank
        ";
        let pick_rows: Vec<(String, String, String, String, String)> =
            match sqlx::query_as(pick_sql).bind(me_id).fetch_all(&pool).await {
                Ok(rs) => rs,
                Err(e) => {
                    tracing::error!(error = %e, "timeline.hn_picks.query.failed");
                    Vec::new()
                }
            };
        let existing: std::collections::HashSet<String> = out
            .iter()
            .filter_map(|v| v.get("url").and_then(|u| u.as_str()).map(String::from))
            .collect();
        let mut picks: std::collections::VecDeque<serde_json::Value> =
            std::collections::VecDeque::new();
        for (url, title, date, summary, source_url) in pick_rows {
            // A user that already bookmarked the same URL sees the
            // real document in their feed — don't dupe with a pick.
            if existing.contains(&url) {
                continue;
            }
            picks.push_back(serde_json::json!({
                "url": url,
                "title": title,
                "date": date,
                "summary": summary,
                // HN picks bypass the documents table — they're never
                // cleaned by the daemon, so cleanTitle / cleanSummary
                // are always empty. The frontend's `cleanTitle ||
                // title` fallback handles this transparently.
                "cleanTitle": "",
                "cleanSummary": "",
                "urls": Vec::<String>::new(),
                "tags": Vec::<String>::new(),
                "source": "hackernews",
                "source_url": source_url,
                "sharers": Vec::<serde_json::Value>::new(),
                "sharerCount": 0,
                "picked": true,
            }));
        }

        // Two placement modes:
        //
        //  • Mixed feed (user hasn't filtered to hackernews) — drop
        //    a pick into the followees timeline after every ~10
        //    documents. Two guardrails: never break a same-source
        //    run; force-insert after 20 docs without a boundary so
        //    long single-source stretches don't swallow the picks.
        //
        //  • Hackernews filter active — every followee doc shares
        //    the source with the picks, so the boundary rule never
        //    fires and the picks would never appear in the place
        //    the user explicitly asked for them. Prepend all picks
        //    at the top instead (rank order, which is upvote desc).
        let hn_only_filter = sources_inc.iter().any(|s| s == "hackernews");
        if !picks.is_empty() {
            if hn_only_filter {
                let pick_count = picks.len();
                let doc_room = (limit as usize).saturating_sub(pick_count);
                out.truncate(doc_room);
                let mut combined: Vec<serde_json::Value> = picks.into_iter().collect();
                combined.append(&mut out);
                out = combined;
            } else {
                const PICK_EVERY: usize = 10;
                const MAX_GAP: usize = 20;
                let mut woven: Vec<serde_json::Value> = Vec::with_capacity(out.len() + picks.len());
                let mut prev_source: Option<String> = None;
                let mut docs_since_last_pick: usize = 0;
                for doc in out.into_iter() {
                    let cur_source = doc
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let at_boundary = prev_source.as_deref() != Some(cur_source.as_str());
                    let force = docs_since_last_pick >= MAX_GAP;
                    let due = docs_since_last_pick >= PICK_EVERY && at_boundary;
                    if (due || force) && !picks.is_empty() {
                        if let Some(pick) = picks.pop_front() {
                            woven.push(pick);
                            docs_since_last_pick = 0;
                        }
                    }
                    woven.push(doc);
                    docs_since_last_pick += 1;
                    prev_source = Some(cur_source);
                }
                // Drop leftover picks (didn't fit before the page ended).
                out = woven;
            }
        }
    }

    let payload = serde_json::Value::Array(out);

    // Anonymous: stash in the shared cache + emit public cache
    // headers so the response can also live in browser caches /
    // any future CDN layer for the same TTL window. Logged-in
    // path stays private to prevent cross-user bleed.
    if let Some(key) = anon_cache_key {
        {
            let mut cache = anon_timeline_cache().write().await;
            // Evict expired entries opportunistically so a flood
            // of distinct filter combinations can't grow the map
            // unbounded.
            cache.retain(|_, e| e.cached_at.elapsed() < ANON_TIMELINE_TTL);
            cache.insert(
                key,
                AnonTimelineEntry {
                    payload: payload.clone(),
                    cached_at: Instant::now(),
                },
            );
        }
        return (
            [
                (
                    CACHE_CONTROL,
                    "public, max-age=60, stale-while-revalidate=120",
                ),
                (VARY, "Cookie"),
            ],
            Json(payload),
        )
            .into_response();
    }

    // Logged-in path: timeline is per-user (the `me_id` derived
    // from the session cookie drives the SQL). Without these
    // headers a shared cache — browser bf-cache after logout, a
    // CDN that ignores cookies, even a future Caddy `cache`
    // directive — could serve one user's feed to another.
    // `no-store` forbids any cache from holding the response;
    // `Vary: Cookie` is the belt-and-braces signal for any cache
    // that DOES decide to cache (per RFC 9111 a Vary on the auth
    // cookie makes the cache key per-user, so even a noncompliant
    // `no-store` cache still segregates responses).
    (
        [
            (CACHE_CONTROL, "private, no-store, must-revalidate"),
            (VARY, "Cookie"),
        ],
        Json(payload),
    )
        .into_response()
}

/// Build the canonical signature used as the anon-timeline cache key.
///
/// All comma-separated lists are sorted in-key so callers that send
/// `sources=a,b` and `sources=b,a` share an entry. Same for tags and
/// categories. `before` and `since` are normalised by trim + lowercase.
///
/// We do NOT include `include_seen` / `min_seen_dwell_ms` / horizon —
/// those parameters only affect the per-viewer seen filter, which is
/// skipped entirely for anon callers (`$1 IS NULL` short-circuit).
/// Including them in the key would create useless duplicate entries.
fn build_anon_cache_key(
    limit: i64,
    before: Option<&str>,
    since: Option<&str>,
    sources_inc: &[String],
    sources_exc: &[String],
    tags: &[String],
    categories: &[String],
) -> String {
    fn norm(values: &[String]) -> String {
        let mut v: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v.join(",")
    }
    let before = before.unwrap_or("").trim();
    let since = since.unwrap_or("").trim();
    format!(
        "v1|l={}|b={}|s={}|src={}|exc={}|tag={}|cat={}",
        limit,
        before,
        since,
        norm(sources_inc),
        norm(sources_exc),
        norm(tags),
        norm(categories),
    )
}

// ── Co-retweet sharers (search-result enrichment) ───────────────────────

#[derive(Deserialize)]
pub struct CoRetweetRequest {
    pub urls: Vec<String>,
}

/// `POST /api/documents/coretweet-sharers` — given a batch of doc URLs,
/// return the set of personalities who retweeted the *same source
/// tweet* as each URL but kept it under their own wrapper URL.
///
/// The pipeline (and the recent backfill) rebuilds every retweet's
/// summary verbatim from the inner tweet, so two personalities who
/// retweeted the same source tweet end up with identical
/// `summary` text on different wrapper URLs. We use that property
/// as the grouping key via the partial index
/// `idx_documents_retweet_summary_md5`. Non-retweet URLs are
/// returned as empty arrays, which is convenient for the caller —
/// it can blindly look up every visible URL in the result map and
/// only retweets contribute new avatars.
///
/// Used by the search-page card renderer to enrich the avatar stack
/// of retweet cards with their co-retweeters when the timeline's
/// own JOIN (see `timeline()` above) didn't run because the doc
/// came from the ColBERT index instead of the feed query.
pub async fn coretweet_sharers(
    State(pool): State<PgPool>,
    Json(req): Json<CoRetweetRequest>,
) -> Response {
    if req.urls.is_empty() {
        return Json(serde_json::json!({})).into_response();
    }
    // Cap the batch so a runaway client can't blow up the LATERAL
    // scan. 500 covers a full screen of search results with plenty
    // of headroom.
    let urls: Vec<String> = req.urls.into_iter().take(500).collect();

    // Two-step query:
    //   1. seed:    pull the (url, summary) pairs for the URLs the
    //               caller actually has on screen — gates the
    //               LATERAL JOIN on the (small) caller set.
    //   2. lateral: for each retweet-summary seed, find every owner
    //               of a doc with the same summary. Hash equality
    //               keys into `idx_documents_retweet_summary_md5`.
    let sql = "
        WITH seed AS (
            SELECT url, summary
              FROM documents
             WHERE url = ANY($1::text[])
               AND deleted = FALSE
               AND source = 'twitter'
               AND summary LIKE 'Retweet @%'
        )
        SELECT seed.url,
               s.sharers
          FROM seed
          JOIN LATERAL (
              SELECT jsonb_agg(DISTINCT
                         jsonb_build_object(
                             'slug',             u.username,
                             'name',             u.name,
                             'avatar',           u.avatar,
                             'twitterFollowers', u.twitter_followers
                         )
                     ) AS sharers
                FROM documents d
                JOIN users    u ON u.id = d.user_id
               WHERE d.deleted = FALSE
                 AND d.source = 'twitter'
                 AND d.summary LIKE 'Retweet @%'
                 AND md5(d.summary) = md5(seed.summary)
          ) s ON true
    ";
    let rows: Vec<(String, serde_json::Value)> =
        match sqlx::query_as(sql).bind(&urls).fetch_all(&pool).await {
            Ok(rs) => rs,
            Err(e) => {
                tracing::error!(error = %e, "coretweet_sharers.query.failed");
                Vec::new()
            }
        };

    let mut out = serde_json::Map::new();
    for (url, sharers) in rows {
        out.insert(url, sharers);
    }
    Json(serde_json::Value::Object(out)).into_response()
}
