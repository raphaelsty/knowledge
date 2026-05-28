//! User (a.k.a. personality) endpoints.
//!
//!   GET /api/users                         — list everyone
//!   GET /api/users/{slug}                  — single profile
//!   GET /api/users/{slug}/documents        — all docs for a library
//!   GET /api/users/{slug}/sources          — source-type filter list
//!
//! All data is sourced from the Postgres `users`, `documents`, and
//! `user_categories` tables — there are no on-disk static files.

use axum::{
    extract::{Path, Query, State},
    http::{header::CACHE_CONTROL, StatusCode},
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// In-memory feed cache.
///
/// The `/api/feed` query is heavy: a CTE over every document for the
/// per-URL latest-by-date pick, a GROUP BY for the sharers JSONB
/// aggregate, a score sort, plus the post-query source-gap selector.
/// On a 100k-doc corpus this runs in the hundreds of ms even on a
/// warm cache. The welcome page hits it on every visit (and again
/// when scrolling triggers the top-up).
///
/// Caching here makes sense because:
///   * the feed mutates only when the pipeline upserts new docs,
///     which is a batch event, not a per-second event;
///   * the result is identical for every anonymous visitor for the
///     same `limit`, so a single cached payload serves them all;
///   * a 60-second staleness window is invisible at human reading
///     pace and lets the welcome page render instantly on repeat
///     visits within a session.
///
/// Keyed on `limit` (the public query param). The full
/// gap-selected JSON payload is cached, so cache hits skip both the
/// SQL and the selector. A `OnceLock` lazily initialises a single
/// shared `RwLock<HashMap<…>>` — read locks for the fast path, a
/// write lock only on miss.
// Short app-layer TTL so a fresh bookmark surfaces in the welcome
// feed quickly. Longer windows save a SQL query but the staleness
// reads as "the site forgot my new bookmark" — not worth it.
const FEED_TTL: Duration = Duration::from_secs(15);

struct FeedCacheEntry {
    payload: serde_json::Value,
    cached_at: Instant,
}

fn feed_cache() -> &'static RwLock<HashMap<i64, FeedCacheEntry>> {
    static CACHE: OnceLock<RwLock<HashMap<i64, FeedCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// In-memory cache for `/api/users` (the VIP list shown in the right-
/// rail people panel + welcome grid).
///
/// The query is heavy: a per-VIP LATERAL `count(*)` against `documents`,
/// which takes 4 s on the local dataset and dominates first-paint on
/// the search page. The list itself rarely changes — new VIPs land at
/// most ~1×/day; doc counts shift on each pipeline pass but a few
/// minutes of staleness is invisible in this UI (the rail just shows
/// names + avatars + counts, no time-sensitive data).
///
/// 60-second TTL: shaves the 4 s SQL down to a single in-memory
/// `clone()` for the next minute of requests. Cache is shared across
/// all callers (the response has no per-user data, only public VIP
/// profiles). A short TTL keeps a freshly-added VIP visible within ~1
/// min of the next pipeline pass.
const LIST_USERS_TTL: Duration = Duration::from_secs(60);

struct ListUsersCacheEntry {
    payload: serde_json::Value,
    cached_at: Instant,
}

fn list_users_cache() -> &'static RwLock<Option<ListUsersCacheEntry>> {
    static CACHE: OnceLock<RwLock<Option<ListUsersCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(None))
}

// ── /api/users{,/{slug}} ───────────────────────────────────────────────

#[derive(Serialize, sqlx::FromRow)]
pub struct UserResponse {
    pub id: i64,
    /// Aliased so the frontend keeps using `personality.slug`.
    #[sqlx(rename = "username")]
    pub slug: String,
    /// Nullable in the DB — community-added personalities (via
    /// POST /api/personalities) carry no email. Kept on the wire
    /// so admin tooling still sees it for users that have one.
    pub email: Option<String>,
    pub public: bool,
    pub name: String,
    pub description: String,
    /// Topical-ontology slugs the user belongs to, ordered by the
    /// category's `sort_order`. Empty when the user is unclassified.
    /// Read from the `user_categories` junction table — the legacy
    /// single-string `users.category` column was dropped in favour of
    /// this many-to-many shape so onboarding can group people by
    /// topic.
    pub categories: Vec<String>,
    pub avatar: Option<String>,
    #[sqlx(rename = "index_name")]
    #[serde(rename = "indexName")]
    pub index_name: String,
    pub links: serde_json::Value,
    pub sources: serde_json::Value,
    /// Total documents in PG for this user. Read from the canonical
    /// `documents` table, not the ColBERT search index.
    #[serde(rename = "documentCount")]
    pub document_count: i64,
    /// Raw social-follower counts. `None` when the pipeline hasn't
    /// populated them yet; the frontend combines the non-null ones on a
    /// log scale to rank personalities within a category.
    #[serde(rename = "twitterFollowers")]
    pub twitter_followers: Option<i32>,
    #[serde(rename = "githubFollowers")]
    pub github_followers: Option<i32>,
    pub citations: Option<i32>,
    /// Grandfathered-in personalities (the original 133) are flagged
    /// vip = true; new sign-ups default to false. The picker /
    /// welcome grid surface only vip rows, plus the caller's own
    /// page when signed in (handled client-side).
    pub vip: bool,
}

// LEFT-JOINs a per-user document count via LATERAL so the count
// only runs for the rows actually fetched (the picker / welcome
// list both gate on `vip`, so we touch ~133 rows out of the table
// instead of aggregating every document up front). Each per-user
// count is a fast index lookup on the `(user_id, url)` PK of
// `documents`. This is the PG source of truth; the ColBERT index
// at indexes/{name}/ can drift and is not used here.
//
// `document_count` reads from the denormalised `users.document_count`
// column (refreshed hourly by the feed-snapshot daemon). The previous
// shape ran a LATERAL `count(*)` per row — fine for one user but the
// /api/users list version dragged 4 s on the 100 k-doc corpus. Single
// user lookups also benefit from the index scan even though the
// LATERAL was cheap; staleness ≤ 1 h is invisible for a "N bookmarks"
// label.
const USER_SELECT: &str = "SELECT u.id,
        u.username,
        u.email,
        u.public,
        u.name,
        u.description,
        COALESCE(
            (SELECT array_agg(cat.slug ORDER BY cat.sort_order)
               FROM user_categories uc
               JOIN categories      cat ON cat.id = uc.category_id
              WHERE uc.user_id = u.id),
            '{}'::text[]
        ) AS categories,
        u.avatar,
        u.index_name,
        u.links,
        u.sources,
        u.document_count,
        u.twitter_followers,
        u.github_followers,
        u.citations,
        u.vip
   FROM users u";

/// GET /api/users
///
/// Public personality list — the welcome grid + the search-page
/// picker both consume this. With the table sized for 100k+ rows,
/// every shaping decision lives in SQL:
///
///   * `u.vip` — only the grandfathered-in cohort. Backed by the
///     partial index `idx_users_vip` (created in users.sql), so the
///     scan touches only the ~133 vip rows even when the table is
///     huge.
///   * `u.document_count > 0` — empty libraries are noise. Uses the
///     denormalised column refreshed hourly by the feed-snapshot
///     daemon; no JOIN to `documents` at read time.
///   * ORDER BY name — the welcome grid re-bins by category and the
///     rail's primary sort is twitter/github/citations followers, so
///     a simple alphabetical default is the cleanest stable order.
///
/// Callers that need a non-vip user (the signed-in user looking at
/// their own page; an already-active lib whose owner hasn't been
/// promoted) hit `GET /api/users/{slug}` directly.
pub async fn list_users(State(pool): State<PgPool>) -> impl IntoResponse {
    let headers = [(
        axum::http::header::CACHE_CONTROL,
        "public, max-age=300, stale-while-revalidate=3600",
    )];

    // Fast path: serve from the in-memory cache when the entry is
    // fresh. The list is identical for every caller (no per-viewer
    // data) so a single shared cache suffices.
    if let Some(entry) = list_users_cache().read().await.as_ref() {
        if entry.cached_at.elapsed() < LIST_USERS_TTL {
            return (headers, Json(entry.payload.clone())).into_response();
        }
    }

    // Cold-path query reads `users.document_count` directly. That
    // column is denormalised and refreshed hourly by the
    // `knowledge-feed-snapshot` daemon's `_refresh_user_document_counts`
    // step — see sources/utils/feed_snapshot_daemon.py. Previously
    // we ran a per-VIP LATERAL `count(*)` against documents (~4 s on
    // a 100 k-doc corpus); now the read is a plain index scan.
    //
    // Categories are pulled in one batch via array_agg on a single
    // JOIN, not a per-row subquery, so the planner can use a hash
    // aggregate instead of 450 nested-loops.
    //
    // VIPs with zero docs are filtered out via `document_count > 0`,
    // mirroring the previous `EXISTS` gate but without the extra
    // index lookup.
    let sql = "
        WITH user_cats AS (
            SELECT uc.user_id,
                   array_agg(cat.slug ORDER BY cat.sort_order) AS categories
              FROM user_categories uc
              JOIN categories      cat ON cat.id = uc.category_id
              JOIN users           u  ON u.id  = uc.user_id AND u.vip = TRUE
             GROUP BY uc.user_id
        )
        SELECT u.id,
               u.username,
               u.email,
               u.public,
               u.name,
               u.description,
               COALESCE(uc.categories, '{}'::text[]) AS categories,
               u.avatar,
               u.index_name,
               u.links,
               u.sources,
               u.document_count,
               u.twitter_followers,
               u.github_followers,
               u.citations,
               u.vip
          FROM users u
          LEFT JOIN user_cats uc ON uc.user_id = u.id
         WHERE u.vip = TRUE
           AND u.document_count > 0
         ORDER BY u.name
    ";
    let rows = sqlx::query_as::<_, UserResponse>(sql)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();
    let payload = serde_json::to_value(&rows).unwrap_or(serde_json::Value::Null);

    // Populate the cache. A single shared entry — no key — since
    // `/api/users` has no query params that vary the result.
    {
        let mut guard = list_users_cache().write().await;
        *guard = Some(ListUsersCacheEntry {
            payload: payload.clone(),
            cached_at: Instant::now(),
        });
    }
    (headers, Json(payload)).into_response()
}

/// GET /api/users/{slug}
pub async fn get_user(State(pool): State<PgPool>, Path(slug): Path<String>) -> impl IntoResponse {
    let sql = format!("{USER_SELECT} WHERE u.username = $1");
    let row = sqlx::query_as::<_, UserResponse>(&sql)
        .bind(&slug)
        .fetch_optional(&pool)
        .await
        .unwrap_or(None);
    match row {
        Some(u) => {
            // Per-user row — doc counts move on every pipeline pass,
            // but the rest (name, avatar, sources config) is stable
            // for minutes. 60s cache keeps the search page and the
            // personality picker snappy on a return visit without
            // showing wildly stale data.
            let headers = [(
                axum::http::header::CACHE_CONTROL,
                "public, max-age=60, stale-while-revalidate=600",
            )];
            (headers, Json(u)).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

// ── /api/users/{slug}/documents ────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ListDocumentsParams {
    /// When `true`, restrict to indexed rows; when `false`, restrict to
    /// the not-yet-embedded backlog. Omitted → return everything.
    /// The frontend pulls `?indexed=false` to surface freshly-synced
    /// docs in the search page before the next pipeline run reaches
    /// them.
    indexed: Option<bool>,
    /// Comma-separated source keys (e.g. `github,lighton.ai`). Match
    /// is exact against `documents.source`.
    sources: Option<String>,
    /// Comma-separated source keys to EXCLUDE (e.g.
    /// `twitter,reddit`). AND-combined with `sources` (so callers
    /// can both narrow to a set and remove a few sources from it).
    exclude_sources: Option<String>,
    /// Comma-separated tag values (e.g. `retrieval,benchmark`). All
    /// tags must be present on a row for it to match (AND semantics).
    /// Searches `tags` and `extra_tags` together — a hit in either
    /// counts.
    tags: Option<String>,
    /// Comma-separated URL list (used by the favorites pre-filter).
    /// Restricts results to rows whose `url` is in the supplied set.
    /// Comes through as `?urls=...` so the JS layer can pass the
    /// user's session-side favorites verbatim.
    urls: Option<String>,
    /// Comma-separated category slugs (e.g. `ai-safety,ml-theory`).
    /// Restricts results to rows whose URL has at least one matching
    /// row in `document_category_assignments`. Used by the Topics
    /// filter on the personal page so the same selection that
    /// narrows the feed also narrows `/<slug>`. The frontend would
    /// otherwise have to fan out a separate
    /// `document-categories/urls` call and pass the result via
    /// `?urls=...`, which collapses for big categories (URL length
    /// limit on the query string).
    category: Option<String>,
    /// Cap the response to the top-N rows in the canonical
    /// (date DESC, created_at DESC) order. The personal-page
    /// browser passes `limit=300` to keep the initial payload small
    /// — without this Raphael's full 3,800-row library shipped 3.4MB
    /// of JSON for a view that only paints 60 cards. Clamped to
    /// [1, 1000] server-side; omit for the historical "return
    /// everything" behaviour MCP / scripts depend on.
    limit: Option<i64>,
}

/// GET /api/users/{slug}/documents[?indexed=false&sources=...&tags=...]
///
/// Returns `{ url: { title, summary, date, tags, "extra-tags", source,
/// source_url, indexed } }` — same shape as the old `database.json`,
/// plus the boolean `indexed` flag so callers can mark a doc as still
/// awaiting embedding.
///
/// Pre-filtering is done in SQL so callers don't have to fetch
/// thousands of rows just to drop most of them client-side: passing
/// `sources=github,lighton.ai&tags=retrieval` runs a single indexed
/// query rather than streaming the whole library.
#[allow(clippy::type_complexity)]
pub async fn list_documents(
    State(pool): State<PgPool>,
    Path(slug): Path<String>,
    Query(params): Query<ListDocumentsParams>,
) -> impl IntoResponse {
    // VIP fast path: read from `personal_snapshot`, which carries the
    // same anchor-collapsed shape but ordered by the feed-style score
    // (sci × 6 + recency tier + tweet-with-resource bonus). Falls back
    // to the legacy date-DESC path when:
    //   * the user is not VIP (no snapshot for them),
    //   * the snapshot is empty (e.g. brand-new VIP, daemon not yet swept),
    //   * the caller passes `indexed=false` (snapshot only carries the
    //     180-day rolling window of indexed docs — unindexed-only
    //     queries belong on the legacy path),
    //   * the caller passes `urls=` (the snapshot is per-anchor, so a
    //     specific-URL lookup may dedup the very row the caller asked
    //     for; the legacy path's `skip_dedup` honors the 1:1 contract).
    let want_unindexed = matches!(params.indexed, Some(false));
    let has_url_filter = params
        .urls
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if !want_unindexed && !has_url_filter {
        if let Some(resp) = try_list_documents_from_snapshot(&pool, &slug, &params).await {
            return resp;
        }
    }

    // Build the SQL incrementally so each $N placeholder lines up
    // with the bind below. $1 = slug; the `indexed`/sources/tags
    // bindings each reserve their own next slot.
    //
    // The personal page applies the same anchor-based dedup as the
    // feed: a candidate's `anchor_url` is selected from its
    // `canonical_referenced_urls` (priority: arxiv > huggingface >
    // github > openreview > doi > etc., lexicographic min within a
    // tier), falling back to its own canonical_url. DISTINCT ON
    // (anchor_url) then keeps the visually-richest copy first
    // (most preview images, then most referenced URLs). So a user's
    // two tweets linking the same paper collapse into the better
    // one; an arxiv abs/pdf pair collapses; a tweet linking a paper
    // collapses with the standalone arxiv row.
    //
    // Pagination cursor escape hatch: when the caller passes an
    // explicit `urls=` filter (favorites pre-filter, deep links),
    // we skip the dedup and return one row per requested URL so the
    // 1-to-1 caller contract holds.
    let skip_dedup = params
        .urls
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    // NOTE: every line continuation here uses `\n\` rather than `\`.
    // A bare `\<newline>` collapses the newline AND all following
    // whitespace, which previously produced `FROM documents dJOIN
    // users` (no space) and made `fetch_all` return a syntax error
    // that the old `.unwrap_or_default()` silently turned into `{}`.
    let mut sql = String::from(
        "WITH candidates AS (\n\
            SELECT d.url,\n\
                   d.title,\n\
                   d.summary,\n\
                   d.clean_title,\n\
                   d.clean_summary,\n\
                   d.date,\n\
                   d.tags,\n\
                   d.extra_tags,\n\
                   d.source,\n\
                   d.source_url,\n\
                   d.indexed,\n\
                   d.linked_urls,\n\
                   d.link_hosts,\n\
                   d.created_at,\n\
                   d.created_via_post,\n\
                   d.canonical_url,\n\
                   d.canonical_referenced_urls\n\
              FROM documents d\n\
              JOIN users     u ON u.id = d.user_id\n\
             WHERE u.username = $1\n\
               AND d.deleted = FALSE",
    );
    let mut next_idx: usize = 2;
    if params.indexed.is_some() {
        sql.push_str(&format!(" AND d.indexed = ${}", next_idx));
        next_idx += 1;
    }
    let sources_vec: Vec<String> = params
        .sources
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !sources_vec.is_empty() {
        sql.push_str(&format!(" AND d.source = ANY(${})", next_idx));
        next_idx += 1;
    }
    let exclude_sources_vec: Vec<String> = params
        .exclude_sources
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !exclude_sources_vec.is_empty() {
        sql.push_str(&format!(" AND NOT (d.source = ANY(${}))", next_idx));
        next_idx += 1;
    }
    let tags_vec: Vec<String> = params
        .tags
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_lowercase())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !tags_vec.is_empty() {
        // `(d.tags || d.extra_tags) @> $N` — a tag matches when it's
        // present in either array. AND semantics across tags are
        // implied by passing the full required set as one array.
        sql.push_str(&format!(" AND (d.tags || d.extra_tags) @> ${}", next_idx));
        next_idx += 1;
    }
    let urls_vec: Vec<String> = params
        .urls
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !urls_vec.is_empty() {
        sql.push_str(&format!(" AND d.url = ANY(${})", next_idx));
        next_idx += 1;
    }
    let category_vec: Vec<String> = params
        .category
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_lowercase())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !category_vec.is_empty() {
        sql.push_str(&format!(
            " AND EXISTS (\
                SELECT 1 FROM document_category_assignments a \
                  JOIN document_categories dc ON dc.id = a.category_id \
                 WHERE a.url = d.url AND dc.slug = ANY(${}))",
            next_idx
        ));
        next_idx += 1;
    }
    // Close the `candidates` CTE and tack on the anchor + dedup +
    // final-select. The two branches diverge only in whether we
    // collapse near-duplicates; the column projection is identical
    // so the tuple decoder below stays the same.
    //
    // Branch A — `?urls=` was supplied: callers want 1 row per
    // requested URL (favorites pre-filter, deep-link expansion).
    // Just project `candidates` straight through. Dedup would
    // silently drop URLs from the response and break the contract.
    //
    // Branch B — normal browse: insert `candidate_anchors` + `dedup`
    // CTEs (same algorithm as the feed). DISTINCT ON (anchor_url)
    // picks the visually-richest representative; the final ORDER
    // restores chronological order so the page reads
    // newest-first.
    //
    // Sort key: `date DESC NULLS LAST, created_at DESC`. Using
    // `d.date` alone keeps the personal page chronological by the
    // underlying content, not by when our crawler happened to see
    // it — a YouTube video published in 2023 but discovered today
    // would otherwise read as "today" and float above a tweet
    // posted yesterday. The upvote-mirror path
    // (`handlers::favorite_docs::add`) stamps `d.date =
    // CURRENT_DATE` so an upvoted old paper still floats to the
    // top, matching the user's mental model. `created_at DESC` is
    // the tiebreaker for two same-day rows.
    if skip_dedup {
        sql.push_str(
            ")\n\
             SELECT c.url, c.title, c.summary,\n\
                    c.clean_title, c.clean_summary,\n\
                    COALESCE(to_char(c.date, 'YYYY-MM-DD'), '') AS date,\n\
                    c.tags, c.extra_tags, c.source, c.source_url, c.indexed,\n\
                    c.linked_urls, c.link_hosts,\n\
                    to_char(c.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS created_at\n\
               FROM candidates c\n\
               LEFT JOIN users uu ON uu.username = $1\n\
               LEFT JOIN favorite_documents fav\n\
                 ON fav.user_id = uu.id AND fav.url = c.url\n\
              ORDER BY CASE\n\
                           WHEN GREATEST(\n\
                                    fav.created_at,\n\
                                    CASE WHEN c.created_via_post THEN c.created_at END\n\
                                ) > now() - interval '14 days'\n\
                           THEN GREATEST(\n\
                                    fav.created_at,\n\
                                    CASE WHEN c.created_via_post THEN c.created_at END\n\
                                )\n\
                       END DESC NULLS LAST,\n\
                       -- After the 14-d hard pin expires, user-action\n\
                       -- rows keep a +12 bump on their date-based sort\n\
                       -- so they stay elevated; a strong recent parsed\n\
                       -- doc still surpasses them when its score wins.\n\
                       -- The legacy path doesn't carry a per-doc score\n\
                       -- column, so we approximate \"score\" via\n\
                       -- `c.date` and add a virtual day-bump on the\n\
                       -- date axis only for user-action rows.\n\
                       (c.date + CASE\n\
                           WHEN GREATEST(\n\
                                    fav.created_at,\n\
                                    CASE WHEN c.created_via_post THEN c.created_at END\n\
                                ) IS NOT NULL\n\
                           THEN INTERVAL '30 days'\n\
                           ELSE INTERVAL '0'\n\
                       END) DESC NULLS LAST,\n\
                       c.created_at DESC",
        );
    } else {
        sql.push_str(
            "),\n\
             candidate_anchors AS (\n\
                SELECT c.*,\n\
                       COALESCE(\n\
                           (SELECT ref\n\
                              FROM unnest(c.canonical_referenced_urls) ref\n\
                             ORDER BY CASE\n\
                                 WHEN ref LIKE 'https://arxiv.org/abs/%'       THEN 1\n\
                                 WHEN ref LIKE 'https://huggingface.co/%'      THEN 2\n\
                                 WHEN ref LIKE 'https://github.com/%'          THEN 3\n\
                                 WHEN ref LIKE 'https://openreview.net/%'      THEN 4\n\
                                 WHEN ref LIKE 'https://doi.org/%'             THEN 5\n\
                                 WHEN ref LIKE 'https://paperswithcode.com/%'  THEN 6\n\
                                 WHEN ref LIKE 'https://aclanthology.org/%'    THEN 7\n\
                                 WHEN ref LIKE 'https://semanticscholar.org/%' THEN 8\n\
                                 WHEN ref LIKE 'https://distill.pub/%'         THEN 9\n\
                                 WHEN ref LIKE 'https://biorxiv.org/%'         THEN 10\n\
                                 WHEN ref LIKE 'https://medrxiv.org/%'         THEN 11\n\
                                 ELSE 99\n\
                             END, ref\n\
                             LIMIT 1),\n\
                           c.canonical_url\n\
                       ) AS anchor_url,\n\
                       COALESCE((\n\
                           SELECT count(*)::int\n\
                             FROM jsonb_array_elements(c.linked_urls) e\n\
                            WHERE COALESCE(e->>'image', '') <> ''\n\
                       ), 0) AS image_count,\n\
                       cardinality(c.canonical_referenced_urls) AS url_count\n\
                  FROM candidates c\n\
             ),\n\
             dedup AS (\n\
                SELECT DISTINCT ON (anchor_url)\n\
                       url, title, summary, clean_title, clean_summary,\n\
                       date, tags, extra_tags,\n\
                       source, source_url, indexed, linked_urls, link_hosts,\n\
                       created_at, created_via_post\n\
                  FROM candidate_anchors\n\
                 ORDER BY anchor_url, image_count DESC, url_count DESC,\n\
                          date DESC NULLS LAST, created_at DESC\n\
             )\n\
             SELECT dedup.url, dedup.title, dedup.summary,\n\
                    dedup.clean_title, dedup.clean_summary,\n\
                    COALESCE(to_char(dedup.date, 'YYYY-MM-DD'), '') AS date,\n\
                    dedup.tags, dedup.extra_tags, dedup.source, dedup.source_url,\n\
                    dedup.indexed, dedup.linked_urls, dedup.link_hosts,\n\
                    to_char(dedup.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS created_at\n\
               FROM dedup\n\
               LEFT JOIN users uu ON uu.username = $1\n\
               LEFT JOIN favorite_documents fav\n\
                 ON fav.user_id = uu.id AND fav.url = dedup.url\n\
              ORDER BY CASE\n\
                           WHEN GREATEST(\n\
                                    fav.created_at,\n\
                                    CASE WHEN dedup.created_via_post THEN dedup.created_at END\n\
                                ) > now() - interval '14 days'\n\
                           THEN GREATEST(\n\
                                    fav.created_at,\n\
                                    CASE WHEN dedup.created_via_post THEN dedup.created_at END\n\
                                )\n\
                       END DESC NULLS LAST,\n\
                       -- Same date-shift as the skip_dedup branch:\n\
                       -- after the 14-d pin expires, user-action rows\n\
                       -- ride 30 days ahead of their actual date so\n\
                       -- they stay elevated. Pipeline rows shift 0d.\n\
                       (dedup.date + CASE\n\
                           WHEN GREATEST(\n\
                                    fav.created_at,\n\
                                    CASE WHEN dedup.created_via_post THEN dedup.created_at END\n\
                                ) IS NOT NULL\n\
                           THEN INTERVAL '30 days'\n\
                           ELSE INTERVAL '0'\n\
                       END) DESC NULLS LAST,\n\
                       dedup.created_at DESC",
        );
    }
    // Server-side cap so the personal page doesn't have to ship the
    // user's entire library on every refresh. Clamp to a sane range
    // — too small breaks infinite scroll past the visible window;
    // too large defeats the whole point.
    let mut limit_val: Option<i64> = None;
    if let Some(n) = params.limit {
        let clamped = n.clamp(1, 1000);
        sql.push_str(&format!(" LIMIT ${next_idx}"));
        limit_val = Some(clamped);
        // No further bindings after limit, so we don't bump next_idx.
    }

    let mut q = sqlx::query_as::<
        _,
        (
            String,            // url
            String,            // title
            String,            // summary
            String,            // clean_title
            String,            // clean_summary
            String,            // date
            Vec<String>,       // tags
            Vec<String>,       // extra_tags
            String,            // source
            Option<String>,    // source_url
            bool,              // indexed
            serde_json::Value, // linked_urls
            Vec<String>,       // link_hosts
            String,            // created_at (ISO-8601 UTC)
        ),
    >(&sql)
    .bind(&slug);
    if let Some(flag) = params.indexed {
        q = q.bind(flag);
    }
    if !sources_vec.is_empty() {
        q = q.bind(&sources_vec);
    }
    if !exclude_sources_vec.is_empty() {
        q = q.bind(&exclude_sources_vec);
    }
    if !tags_vec.is_empty() {
        q = q.bind(&tags_vec);
    }
    if !urls_vec.is_empty() {
        q = q.bind(&urls_vec);
    }
    if !category_vec.is_empty() {
        q = q.bind(&category_vec);
    }
    if let Some(n) = limit_val {
        q = q.bind(n);
    }
    let rows = match q.fetch_all(&pool).await {
        Ok(r) => r,
        Err(e) => {
            // Log the full error so it isn't silently swallowed by
            // the previous `.unwrap_or_default()` which returned `{}`
            // and made every personal page look empty on prod.
            tracing::error!(
                slug = %slug,
                error = %e,
                "list_documents.sql.failed"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "documents query failed",
                    "detail": e.to_string(),
                })),
            )
                .into_response();
        }
    };

    let mut out = serde_json::Map::with_capacity(rows.len());
    for (
        url,
        title,
        summary,
        clean_title,
        clean_summary,
        date,
        tags,
        extra_tags,
        source,
        source_url,
        indexed,
        linked_urls,
        link_hosts,
        created_at,
    ) in rows
    {
        out.insert(
            url,
            serde_json::json!({
                "title": title,
                "summary": summary,
                // Pedagogical title/summary produced by the clean
                // daemon. Empty until the row has been cleaned;
                // the frontend's `cleanTitle || title` fallback keeps
                // the raw text on uncleaned rows.
                "clean_title": clean_title,
                "clean_summary": clean_summary,
                "date": date,
                "tags": tags,
                "extra-tags": extra_tags,
                "source": source,
                "source_url": source_url,
                "indexed": indexed,
                "linked_urls": linked_urls,
                "link_hosts": link_hosts,
                "created_at": created_at,
            }),
        );
    }
    Json(out).into_response()
}

/// VIP personal-page fast path. Returns `Some(response)` when the
/// snapshot path applies and produced rows; `None` to fall through to
/// the legacy date-DESC query.
///
/// The response shape MUST match `list_documents` exactly so the
/// frontend doesn't have to special-case which path served it.
#[allow(clippy::type_complexity)]
async fn try_list_documents_from_snapshot(
    pool: &PgPool,
    slug: &str,
    params: &ListDocumentsParams,
) -> Option<axum::response::Response> {
    // Cheap VIP gate first — non-VIPs never have a snapshot row.
    let vip_row: Option<(bool,)> = sqlx::query_as("SELECT vip FROM users WHERE username = $1")
        .bind(slug)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    let is_vip = matches!(vip_row, Some((true,)));
    if !is_vip {
        return None;
    }

    let sources_vec: Vec<String> = params
        .sources
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let exclude_sources_vec: Vec<String> = params
        .exclude_sources
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let tags_vec: Vec<String> = params
        .tags
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_lowercase())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let category_vec: Vec<String> = params
        .category
        .as_deref()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_lowercase())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();

    // Build the SQL incrementally to keep $N placeholders aligned.
    // Two LEFT JOINs feed the ORDER BY's "user activity" sort key:
    //   * favorite_documents → upvotes (fav.created_at)
    //   * documents.created_via_post → manual Post-button rows
    //                                  (d_post.created_at)
    // Whichever timestamp is more recent wins, so an upvote made
    // after a post still rises above it. Pipeline-imported rows
    // leave both slots NULL and fall back to feed-score order.
    let mut sql = String::from(
        "SELECT ps.url,\n         ps.title,\n         ps.summary,\n         ps.clean_title,\n         ps.clean_summary,\n         COALESCE(to_char(ps.date, 'YYYY-MM-DD'), '') AS date,\n         ps.tags,\n         ps.extra_tags,\n         ps.source,\n         ps.source_url,\n         ps.indexed,\n         ps.linked_urls,\n         ps.link_hosts,\n         COALESCE(to_char(ps.date AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'), '') AS created_at,\n         ps.sharers,\n         ps.sharer_count\n    FROM personal_snapshot ps\n    JOIN users u ON u.id = ps.user_id\n    LEFT JOIN favorite_documents fav\n      ON fav.user_id = ps.user_id AND fav.url = ps.url\n    LEFT JOIN documents d_post\n      ON d_post.user_id = ps.user_id AND d_post.url = ps.url\n     AND d_post.deleted = FALSE\n   WHERE u.username = $1",
    );
    let mut idx: usize = 2;
    if !sources_vec.is_empty() {
        sql.push_str(&format!(" AND ps.source = ANY(${idx})"));
        idx += 1;
    }
    if !exclude_sources_vec.is_empty() {
        sql.push_str(&format!(" AND NOT (ps.source = ANY(${idx}))"));
        idx += 1;
    }
    if !tags_vec.is_empty() {
        // Same AND-of-required-tags semantics as the legacy path:
        // every requested tag must be present in either `tags` or
        // `extra_tags` for the row to match.
        sql.push_str(&format!(" AND (ps.tags || ps.extra_tags) @> ${idx}"));
        idx += 1;
    }
    if !category_vec.is_empty() {
        // `personal_snapshot.categories` rolls up the per-(user, url)
        // assignments to the anchor level at refresh time — no JOIN
        // back to `document_category_assignments` at read time.
        sql.push_str(&format!(" AND ps.categories && ${idx}"));
        idx += 1;
    }
    // User-activity boost (manual Post + upvote) on the personal
    // page — two-tier so fresh actions hard-pin and older ones still
    // stay elevated but yield to a strong recent parsed doc.
    //
    //   * Tier 1 — hard pin (≤ 14 d): the sort uses
    //     `GREATEST(fav.created_at, post.created_at)` as the first
    //     key. Today's upvote beats yesterday's post; a Post lands
    //     instantly at the top.
    //   * Tier 2 — score boost (any age): every user-action row gets
    //     a +12 bump on `ps.score` for the secondary sort key. After
    //     the 14-d pin expires the row still ranks like a top-tier
    //     doc, so it stays "quite high" — but a parsed doc that
    //     genuinely scores higher (recent, broadly shared, sci-
    //     anchored) can now surpass it.
    //   * Pipeline-imported docs leave both inputs NULL (the
    //     `created_via_post` guard masks them on the post side, no
    //     favorite_documents row on the upvote side) so they sort
    //     by raw `ps.score` without the bump.
    sql.push_str(
        "\n   ORDER BY CASE\n                WHEN GREATEST(\n                         fav.created_at,\n                         CASE WHEN d_post.created_via_post THEN d_post.created_at END\n                     ) > now() - interval '14 days'\n                THEN GREATEST(\n                         fav.created_at,\n                         CASE WHEN d_post.created_via_post THEN d_post.created_at END\n                     )\n            END DESC NULLS LAST,\n            (ps.score + CASE\n                WHEN GREATEST(\n                         fav.created_at,\n                         CASE WHEN d_post.created_via_post THEN d_post.created_at END\n                     ) IS NOT NULL\n                THEN 12.0\n                ELSE 0\n            END) DESC,\n            ps.date DESC NULLS LAST,\n            ps.url",
    );
    let mut limit_val: Option<i64> = None;
    if let Some(n) = params.limit {
        let clamped = n.clamp(1, 1000);
        sql.push_str(&format!(" LIMIT ${idx}"));
        limit_val = Some(clamped);
    }

    let mut q = sqlx::query_as::<
        _,
        (
            String,            // url
            String,            // title
            String,            // summary
            String,            // clean_title
            String,            // clean_summary
            String,            // date
            Vec<String>,       // tags
            Vec<String>,       // extra_tags
            String,            // source
            Option<String>,    // source_url
            bool,              // indexed
            serde_json::Value, // linked_urls
            Vec<String>,       // link_hosts
            String,            // created_at
            serde_json::Value, // sharers (jsonb array)
            i32,               // sharer_count
        ),
    >(&sql)
    .bind(slug);
    if !sources_vec.is_empty() {
        q = q.bind(&sources_vec);
    }
    if !exclude_sources_vec.is_empty() {
        q = q.bind(&exclude_sources_vec);
    }
    if !tags_vec.is_empty() {
        q = q.bind(&tags_vec);
    }
    if !category_vec.is_empty() {
        q = q.bind(&category_vec);
    }
    if let Some(n) = limit_val {
        q = q.bind(n);
    }
    let rows = match q.fetch_all(pool).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(slug = %slug, error = %e, "list_documents.snapshot.failed");
            return None;
        }
    };
    // Empty snapshot for this user — likely the daemon hasn't swept
    // them yet. Fall through so the legacy path serves SOMETHING.
    if rows.is_empty() {
        return None;
    }
    let mut out = serde_json::Map::with_capacity(rows.len());
    for (
        url,
        title,
        summary,
        clean_title,
        clean_summary,
        date,
        tags,
        extra_tags,
        source,
        source_url,
        indexed,
        linked_urls,
        link_hosts,
        created_at,
        sharers,
        sharer_count,
    ) in rows
    {
        out.insert(
            url,
            serde_json::json!({
                "title": title,
                "summary": summary,
                // Pedagogical title/summary produced by the clean
                // daemon. Empty until the row has been cleaned; the
                // frontend's `cleanTitle || title` fallback keeps the
                // raw text on uncleaned rows. Wire format is
                // snake_case; the api.js mapper translates to the
                // camelCase keys the renderer reads.
                "clean_title": clean_title,
                "clean_summary": clean_summary,
                "date": date,
                "tags": tags,
                "extra-tags": extra_tags,
                "source": source,
                "source_url": source_url,
                "indexed": indexed,
                "linked_urls": linked_urls,
                "link_hosts": link_hosts,
                "created_at": created_at,
                // Cross-user sharer roll-up — surfaces the same
                // avatar-stack data the global feed renders, so the
                // personal page can show "X others also have this".
                "sharers": sharers,
                "sharer_count": sharer_count,
            }),
        );
    }
    Some(Json(out).into_response())
}

// ── /api/personalities/{slug}/fallback ────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FallbackParams {
    /// Free-text query. Empty/missing → browse mode (latest by date).
    pub q: Option<String>,
    /// Cap, 1..=600.
    pub limit: Option<i64>,
}

/// GET /api/personalities/{slug}/fallback[?q=...&limit=N]
///
/// SQL-backed search/feed for libraries whose ColBERT index is
/// `missing` on disk. Returns BOTH shapes — `metadata` (for the
/// browse-mode caller that reads `data.metadata`) and `results`
/// (for the search caller that reads `data.results[0].metadata` /
/// `.scores`) — so the frontend can fall through with a single
/// endpoint regardless of which call site triggered the fallback.
///
/// When `q` is empty: ORDER BY date DESC LIMIT N.
/// When `q` is set:   ILIKE filter on title/summary/source/owner,
///                    then ORDER BY date DESC LIMIT N. ILIKE is
///                    intentionally simple — proper FTS would need
///                    a `tsvector` column + index, which we don't
///                    have today. ILIKE on title+summary is fast
///                    enough for the small per-user corpus this
///                    endpoint serves (one user's docs, capped at
///                    600 rows). A user with 5,000 docs still scans
///                    in <50ms thanks to the `(user_id)` index.
///
/// Frontend opt-in path: `search/api.js` falls through to this
/// endpoint ONLY when the plaid endpoint returns HTTP 404 (index
/// not declared). For `broken`/`error` indices the heal hook in
/// `run.py` does the rebuild on the user's next pipeline pass,
/// so we never see those failure modes in steady state.
pub async fn fallback_search(
    State(pool): State<PgPool>,
    Path(slug): Path<String>,
    Query(params): Query<FallbackParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(60).clamp(1, 600);
    let q = params.q.unwrap_or_default().trim().to_string();

    // 404 if the user doesn't exist — same status the plaid index
    // returns for an unknown name, so the frontend's "treat 404 as
    // empty" path works without special-casing.
    let user_exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM users WHERE username = $1)")
            .bind(&slug)
            .fetch_one(&pool)
            .await
            .unwrap_or(false);
    if !user_exists {
        return (StatusCode::NOT_FOUND, "user not found").into_response();
    }

    let mut sql = String::from(
        "SELECT d.url, d.title, d.summary,
                COALESCE(to_char(d.date,'YYYY-MM-DD'),'') AS date,
                d.tags, d.extra_tags, d.source, d.source_url,
                d.linked_urls, d.link_hosts, u.username AS owner
           FROM documents d
           JOIN users u ON u.id = d.user_id
          WHERE u.username = $1
            AND d.deleted = FALSE",
    );
    let mut next: usize = 2;
    if !q.is_empty() {
        // One placeholder, four ILIKEs — cheaper than four binds and
        // PG plans it identically. The `source` match lets queries
        // like "github" still return github docs even when title/
        // summary don't mention the word.
        sql.push_str(&format!(
            " AND (d.title    ILIKE ${i}
                OR d.summary  ILIKE ${i}
                OR d.source   ILIKE ${i}
                OR u.username ILIKE ${i})",
            i = next
        ));
        next += 1;
    }
    sql.push_str(&format!(" ORDER BY d.date DESC NULLS LAST LIMIT ${}", next));

    // Escape user-supplied % and _ so they match literally instead of
    // acting as wildcards (a query of `%` would otherwise match all rows).
    let pattern = format!("%{}%", crate::handlers::sql_like::escape_like_pattern(&q));
    let mut query = sqlx::query_as::<
        _,
        (
            String,            // url
            String,            // title
            String,            // summary
            String,            // date
            Vec<String>,       // tags
            Vec<String>,       // extra_tags
            String,            // source
            Option<String>,    // source_url
            serde_json::Value, // linked_urls
            Vec<String>,       // link_hosts
            String,            // owner
        ),
    >(&sql)
    .bind(&slug);
    if !q.is_empty() {
        query = query.bind(&pattern);
    }
    query = query.bind(limit);

    let rows = match query.fetch_all(&pool).await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("fallback query failed: {}", e),
            )
                .into_response();
        }
    };

    // Build the metadata array once, shaped to mirror what the
    // search-index endpoints emit. `transformMeta` on the frontend
    // accepts arrays directly for tags/linked_urls/link_hosts, so we
    // ship them as real PG arrays — no comma-encoding round-trip.
    let metadata: Vec<serde_json::Value> = rows
        .into_iter()
        .map(
            |(
                url,
                title,
                summary,
                date,
                tags,
                extra_tags,
                source,
                source_url,
                linked_urls,
                link_hosts,
                owner,
            )| {
                serde_json::json!({
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "date": date,
                    "tags": tags,
                    "extra_tags": extra_tags,
                    "source": source,
                    "source_url": source_url,
                    "linked_urls": linked_urls,
                    "link_hosts": link_hosts,
                    "owner": owner,
                })
            },
        )
        .collect();

    // Score every row the same — there's no ranking signal here,
    // so the frontend falls back to date order anyway.
    let scores: Vec<f32> = vec![0.0; metadata.len()];

    Json(serde_json::json!({
        "metadata": metadata,
        "results": [{
            "metadata": metadata,
            "scores": scores,
        }],
        "fallback": true,
    }))
    .into_response()
}

// ── /api/feed ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct FeedParams {
    /// Max rows to return. Defaults to 500; clamped to [1, 2000].
    limit: Option<i64>,
}

/// GET /api/feed
///
/// Cross-library activity feed — one row per unique URL.
///
/// Ranking is a composite score that **weights intersection size over
/// raw recency** but still rewards freshness. Recency is bucketed by
/// week so the feed reads as "this week's events" rather than "today's
/// events" — every doc inside the same rolling 7-day window gets the
/// same recency bonus:
///
/// ```text
/// weeks_old = FLOOR(days_old / 7)
/// score = sharer_count + GREATEST(0, 5 - weeks_old) * 0.56
/// ```
///
/// So a 5-sharer doc from 2 weeks ago (≈ 5 + 3·0.56 = 6.68) still
/// outranks a single-sharer doc from this week (≈ 1 + 5·0.56 = 3.8),
/// matching the "shared deep cuts beat fresh long-tail" preference.
/// The 5-week cap means anything older than ~35 days competes purely
/// on intersection size.
///
/// Then we apply a **4-slot source gap** post-query: the same
/// `documents.source` value cannot appear within 4 consecutive feed
/// slots. The greedy selector walks the score-sorted candidates and
/// defers any whose source is still inside the rolling 4-slot window,
/// reconsidering them on later iterations once the window has moved
/// on. If no source-legal candidate is available we relax the gap so
/// the feed never stalls.
#[allow(clippy::type_complexity)]
/// Shared feed builder used by both `GET /api/feed` and the MCP `feed`
/// tool. Returns the cached, score-sorted + source-gap-selected
/// payload (a JSON array of documents). The caller decides whether
/// to apply `jitter_feed` on top — the welcome page wants per-visit
/// jitter so consecutive loads aren't identical; MCP wants stable
/// pagination so it doesn't apply it.
///
/// `limit` is the requested feed length; the cache keys by it.
pub async fn build_feed_payload(pool: &PgPool, limit: i64) -> serde_json::Value {
    let limit = limit.clamp(1, 2000);

    // Cache fast path. The shared `feed_cache()` keys by `limit` and
    // expires after `FEED_TTL`. A read lock lets concurrent visitors
    // share the same cached payload without contending on the
    // database.
    {
        let cache = feed_cache().read().await;
        if let Some(entry) = cache.get(&limit) {
            if entry.cached_at.elapsed() < FEED_TTL {
                return entry.payload.clone();
            }
        }
    }

    // Pull a generous over-fetch so the source-gap selector has slack.
    // Three times the requested limit covers the worst case where one
    // bursty source eats the top of the score order: we still have
    // enough other-source rows queued up to fill the 4-slot gaps.
    let fetch_limit = (limit * 3).clamp(50, 6000);

    let sql = "
        WITH latest_meta AS (
            SELECT DISTINCT ON (d.url)
                d.url,
                d.title,
                d.date,
                d.summary,
                d.tags,
                d.source,
                d.source_url
              FROM documents d
             WHERE d.date IS NOT NULL
               AND d.deleted = FALSE
             ORDER BY d.url, d.date DESC
        ),
        sharers_per_url AS (
            SELECT
                d.url,
                jsonb_agg(
                    jsonb_build_object(
                        'slug',             u.username,
                        'name',             u.name,
                        'avatar',           u.avatar,
                        'twitterFollowers', u.twitter_followers,
                        'githubFollowers',  u.github_followers,
                        'citations',        u.citations
                    )
                )        AS sharers,
                count(*) AS sharer_count
              FROM documents d
              JOIN users u ON u.id = d.user_id
             WHERE d.date IS NOT NULL
               AND d.deleted = FALSE
             GROUP BY d.url
        ),
        scored AS (
            SELECT
                m.url, m.title, m.date, m.summary, m.tags,
                m.source, m.source_url,
                s.sharers, s.sharer_count,
                -- Weekly bucketed recency: every doc inside the
                -- same rolling 7-day window gets the same bonus, so
                -- the feed reads as current-week activity instead
                -- of last-24h activity. 5 weekly steps span the
                -- same ~5-week horizon as the old 14-day linear
                -- decay; top bonus matches the old peak (2.8).
                (
                    s.sharer_count::double precision
                    + GREATEST(
                          0.0,
                          5.0 - FLOOR(EXTRACT(EPOCH FROM (now() - m.date)) / (7.0 * 86400.0))
                      ) * 0.56
                ) AS score
              FROM latest_meta m
              JOIN sharers_per_url s ON s.url = m.url
        )
        SELECT
            url,
            title,
            COALESCE(to_char(date, 'YYYY-MM-DD'), '') AS date,
            summary,
            tags,
            source,
            source_url,
            sharers,
            sharer_count
          FROM scored
         ORDER BY score DESC, date DESC, sharer_count DESC, url
         LIMIT $1
    ";

    let rows: Vec<(
        String,            // url
        String,            // title
        String,            // date
        String,            // summary
        Vec<String>,       // tags
        String,            // source
        Option<String>,    // source_url
        serde_json::Value, // sharers (JSONB)
        i64,               // sharer_count
    )> = match sqlx::query_as(sql).bind(fetch_limit).fetch_all(pool).await {
        Ok(rs) => rs,
        Err(e) => {
            tracing::error!(error = %e, "feed query failed");
            Vec::new()
        }
    };

    // Source-gap selector. At every output slot we pick the
    // highest-scoring candidate whose `source` isn't in the rolling
    // 4-slot window. Both `rows` (still iterating) and `deferred` are
    // score-desc, so the head-of-deferred always wins over the next
    // unseen row when both are gap-legal.
    //
    // When no candidate is gap-legal (the whole tail is 1–2 sources —
    // common when only HF & github have high sharer counts), we shrink
    // the gap by 1 and retry. This degrades gracefully to the maximum
    // gap the data can support (e.g. 3 if only 4 unique sources are
    // left) instead of abandoning the rule and dumping a long run.
    type FeedRow = (
        String,
        String,
        String,
        String,
        Vec<String>,
        String,
        Option<String>,
        serde_json::Value,
        i64,
    );
    const SOURCE_GAP: usize = 4;
    let target = limit as usize;
    let mut out: Vec<serde_json::Value> = Vec::with_capacity(target);
    let mut recent: std::collections::VecDeque<String> = Default::default();
    let mut deferred: std::collections::VecDeque<FeedRow> = std::collections::VecDeque::new();
    let mut iter = rows.into_iter();
    let row_to_json = |row: &FeedRow| {
        let (url, title, date, summary, tags, source, source_url, sharers, count) = row;
        serde_json::json!({
            "url": url,
            "title": title,
            "date": date,
            "summary": summary,
            "tags": tags,
            "source": source,
            "source_url": source_url,
            "sharers": sharers,
            "sharerCount": count,
        })
    };

    while out.len() < target {
        // 1. Highest-scored deferred row whose source has cleared the gap.
        let legal_deferred = deferred.iter().position(|r| !recent.contains(&r.5));
        if let Some(idx) = legal_deferred {
            let row = deferred.remove(idx).expect("idx in range");
            recent.push_back(row.5.clone());
            while recent.len() > SOURCE_GAP {
                recent.pop_front();
            }
            out.push(row_to_json(&row));
            continue;
        }
        // 2. Pull the next score-sorted row from the SQL stream.
        if let Some(row) = iter.next() {
            if recent.contains(&row.5) {
                deferred.push_back(row);
            } else {
                recent.push_back(row.5.clone());
                while recent.len() > SOURCE_GAP {
                    recent.pop_front();
                }
                out.push(row_to_json(&row));
            }
            continue;
        }
        // 3. No fresh rows left and nothing in deferred is gap-legal.
        //    Shrink the rolling window so the next-best deferred row
        //    becomes legal — keeps the maximum gap the data supports.
        if !deferred.is_empty() && !recent.is_empty() {
            recent.pop_front();
            continue;
        }
        // 4. Out of work.
        break;
    }

    // Stash the gap-selected payload in the cache before returning so
    // subsequent requests for the same `limit` skip both the SQL and
    // the selector for the next FEED_TTL window. Also cleans up any
    // expired entries to keep the map bounded.
    let payload = serde_json::Value::Array(out);
    {
        let mut cache = feed_cache().write().await;
        cache.retain(|_, entry| entry.cached_at.elapsed() < FEED_TTL);
        cache.insert(
            limit,
            FeedCacheEntry {
                payload: payload.clone(),
                cached_at: Instant::now(),
            },
        );
    }
    payload
}

pub async fn feed(
    State(pool): State<PgPool>,
    Query(params): Query<FeedParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(500).clamp(1, 2000);
    let payload = build_feed_payload(&pool, limit).await;
    // Per-request jitter so two visits don't look identical. MCP
    // callers go through `build_feed_payload` directly to keep
    // pagination stable.
    //
    // `Cache-Control: no-store` keeps every caller's browser /
    // intermediate cache out of the response — the jitter would
    // otherwise be frozen at the first visit, and even though the
    // payload has no PII, reloading expecting an updated feed
    // would silently return the cached one.
    ([(CACHE_CONTROL, "no-store")], Json(jitter_feed(payload))).into_response()
}

/// Light per-request shuffle on top of the cached canonical feed.
///
/// The cached payload is the deterministic, score-sorted +
/// source-gap-selected list. To keep two consecutive visits from
/// looking identical, we apply a *cheap* in-place reorder before
/// returning:
///
///   * The first 3 slots stay frozen — those are the heavy hitters
///     (highest sharer counts × freshness) and they're the page's
///     hero. Shuffling them would hide the strongest signal.
///   * Every other slot is offered for swap with a small forward
///     window (next 3 neighbours). With per-pair probability 35%
///     the items swap; otherwise they stay put. Net effect: most
///     pairs hold their position, a handful of items drift one or
///     two slots up/down per visit.
///
/// Total cost is O(n) with a tiny constant — negligible compared
/// to the SQL query we just skipped via the cache.
fn jitter_feed(payload: serde_json::Value) -> serde_json::Value {
    use rand::Rng;
    const FROZEN_HEAD: usize = 3;
    const SWAP_WINDOW: usize = 3;
    const SWAP_PROB: f64 = 0.35;

    let serde_json::Value::Array(mut arr) = payload else {
        return payload;
    };
    if arr.len() <= FROZEN_HEAD + 1 {
        return serde_json::Value::Array(arr);
    }
    let mut rng = rand::thread_rng();
    let n = arr.len();
    let mut i = FROZEN_HEAD;
    while i < n {
        // Random partner within [i+1, i+SWAP_WINDOW] (clamped).
        let max_j = (i + SWAP_WINDOW).min(n - 1);
        if max_j > i && rng.gen_bool(SWAP_PROB) {
            let j = rng.gen_range(i + 1..=max_j);
            arr.swap(i, j);
        }
        i += 1;
    }
    serde_json::Value::Array(arr)
}

// ── /api/users/intersect ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct IntersectParams {
    /// Comma-separated list of usernames. The intersection is computed
    /// over exactly these libraries — a URL must exist for every one of
    /// them to be included.
    slugs: String,
    /// Max rows to return. Defaults to 200; clamped to [1, 1000].
    limit: Option<i64>,
}

/// GET /api/users/intersect?slugs=a,b,c[&limit=200]
///
/// Returns the documents that exist in **all** of the specified
/// libraries — the multi-library "shared resource" pool. Same shape as
/// `/api/users/{slug}/documents`, but augmented with an `_owners` array
/// listing every slug that actually has the URL (always equal to the
/// requested slug list, by construction). Ordered by date desc.
///
/// Why a dedicated endpoint: the per-library `latest` call only sees
/// the top-N most-recent rows from one library, and shared URLs tend
/// to be older deep cuts that don't surface in either library's
/// recent slice. A direct intersection query is the only way to find
/// them without paginating thousands of rows client-side.
#[allow(clippy::type_complexity)]
pub async fn intersect_documents(
    State(pool): State<PgPool>,
    Query(params): Query<IntersectParams>,
) -> impl IntoResponse {
    let slugs: Vec<String> = params
        .slugs
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if slugs.len() < 2 {
        // No intersection to compute with < 2 libraries.
        return Json(serde_json::json!({ "owners": slugs, "documents": {} })).into_response();
    }
    /* Hard cap on multi-library queries: 10. Each extra slug fans
     * the SQL out (ANY($1) over `documents`) and the per-library
     * search/latest fanout on the client multiplies request count,
     * so a runaway slug list would trash both ends. The frontend
     * picker enforces the same limit; this is the server-side
     * backstop. */
    const MAX_LIBS: usize = 10;
    if slugs.len() > MAX_LIBS {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("too many libraries: {} > {}", slugs.len(), MAX_LIBS),
            })),
        )
            .into_response();
    }
    let limit = params.limit.unwrap_or(200).clamp(1, 1000);

    // Three-stage query:
    //   1. `shared_urls` — URLs owned by ≥ 2 of the input libraries,
    //      with the per-URL list of owning slugs and the count.
    //      `array_agg(DISTINCT)` avoids inflating the list when a
    //      single library has duplicate rows for the same URL.
    //   2. `canonical` — for each shared URL, pick the most-recent
    //      copy across the chosen libraries (DISTINCT ON) so the
    //      title/summary/tags reflect the freshest tagging pass.
    //   3. final SELECT — sort by intersection size desc, then by
    //      date desc, so a 3-way overlap from two years ago beats
    //      yesterday's 2-way overlap; ties within a tier fall back
    //      to recency.
    let sql = "
        WITH shared_urls AS (
            SELECT d.url,
                   array_agg(DISTINCT u.username
                             ORDER BY u.username) AS owners,
                   count(DISTINCT u.username)     AS owner_count
              FROM documents d
              JOIN users u ON u.id = d.user_id
             WHERE u.username = ANY($1)
               AND d.deleted = FALSE
             GROUP BY d.url
            HAVING count(DISTINCT u.username) >= 2
        ),
        canonical AS (
            SELECT DISTINCT ON (d.url)
                   d.url, d.title, d.summary, d.date,
                   d.tags, d.extra_tags, d.source, d.source_url, d.indexed,
                   s.owners, s.owner_count
              FROM documents d
              JOIN shared_urls s ON s.url = d.url
              JOIN users u ON u.id = d.user_id
             WHERE u.username = ANY($1)
               AND d.deleted = FALSE
             ORDER BY d.url, d.date DESC NULLS LAST
        )
        SELECT url, title, summary,
               COALESCE(to_char(date, 'YYYY-MM-DD'), '') AS date,
               tags, extra_tags, source, source_url, indexed,
               owners, owner_count
          FROM canonical
         ORDER BY owner_count DESC, date DESC NULLS LAST
         LIMIT $2
    ";

    let rows: Vec<(
        String,         // url
        String,         // title
        String,         // summary
        String,         // date
        Vec<String>,    // tags
        Vec<String>,    // extra_tags
        String,         // source
        Option<String>, // source_url
        bool,           // indexed
        Vec<String>,    // owners
        i64,            // owner_count
    )> = match sqlx::query_as(sql)
        .bind(&slugs)
        .bind(limit)
        .fetch_all(&pool)
        .await
    {
        Ok(rs) => rs,
        Err(e) => {
            tracing::error!(error = %e, "intersect query failed");
            Vec::new()
        }
    };

    let mut documents = serde_json::Map::with_capacity(rows.len());
    for (
        url,
        title,
        summary,
        date,
        tags,
        extra_tags,
        source,
        source_url,
        indexed,
        owners,
        _owner_count,
    ) in rows
    {
        documents.insert(
            url,
            serde_json::json!({
                "title": title,
                "summary": summary,
                "date": date,
                "tags": tags,
                "extra-tags": extra_tags,
                "source": source,
                "source_url": source_url,
                "indexed": indexed,
                "owners": owners,
            }),
        );
    }
    Json(serde_json::json!({ "owners": slugs, "documents": documents })).into_response()
}

// ── /api/users/{slug}/sources ──────────────────────────────────────────

/// POST /api/co-owners
///
/// Body: `{ "urls": ["...", ...], "exclude_slug": "raphael-sourty" }`.
///
/// Returns `{ "<url>": [ { slug, name, avatar, twitterFollowers }, ... ] }`
/// — the set of VIP personalities (other than `exclude_slug`) who share
/// the URL OR reference one of the URLs the page card surfaces. Used by
/// the personal-page renderer to surface "people who also liked this".
///
/// Matching prongs (mirrors the timeline LATERAL in `follows.rs`):
///   1. Direct: `d2.canonical_url = seed.canonical_url` — covers URL
///      variants (arxiv abs/pdf/vN, www., utm_*) that previously
///      fragmented into separate cards.
///   2. Overlap: `d2.canonical_referenced_urls && seed.canonical_referenced_urls`
///      — picks up anyone whose tweet / blog post links at least one
///      of the URLs the seed doc references, so a paper card surfaces
///      every personality who tweeted about it without us needing a
///      separate co-retweet sweep.
#[derive(serde::Deserialize)]
pub struct CoOwnersRequest {
    pub urls: Vec<String>,
    pub exclude_slug: Option<String>,
}

#[allow(clippy::type_complexity)]
pub async fn list_co_owners(
    State(pool): State<PgPool>,
    Json(req): Json<CoOwnersRequest>,
) -> impl IntoResponse {
    let urls: Vec<String> = req
        .urls
        .into_iter()
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty())
        .collect();
    if urls.is_empty() {
        return Json(serde_json::json!({})).into_response();
    }
    // Cap at a reasonable batch size so a runaway client can't pull
    // the whole co-owner graph in one request.
    let urls: Vec<String> = urls.into_iter().take(500).collect();
    let exclude = req.exclude_slug.unwrap_or_default();
    // Two-step:
    //   1. `seed` resolves each input URL to its canonical_url and
    //      canonical_referenced_urls. The same URL can appear in
    //      multiple libraries; we just need ONE row to pick up the
    //      canonical metadata since those columns are deterministic
    //      from (url, urls, linked_urls) via the generated-column
    //      formula.
    //   2. The LATERAL aggregates everyone who matches either prong,
    //      same shape (`{slug, name, avatar, twitterFollowers}`) the
    //      old endpoint returned so the frontend doesn't need to
    //      change.
    let sql = "
        WITH input AS (SELECT DISTINCT u AS raw_url FROM unnest($1::text[]) u),
        seed AS (
            SELECT DISTINCT ON (i.raw_url)
                   i.raw_url,
                   COALESCE(d.canonical_url, canonicalize_url(i.raw_url)) AS canonical_url,
                   COALESCE(d.canonical_referenced_urls, '{}'::text[])    AS canonical_referenced_urls
              FROM input i
              LEFT JOIN documents d
                ON d.url = i.raw_url AND d.deleted = FALSE
             ORDER BY i.raw_url, d.date DESC NULLS LAST
        )
        SELECT seed.raw_url AS url,
               jsonb_agg(
                   jsonb_build_object(
                       'slug',             s.username,
                       'name',             s.name,
                       'avatar',           s.avatar,
                       'twitterFollowers', s.twitter_followers
                   ) ORDER BY COALESCE(s.twitter_followers, 0) DESC
               ) AS sharers
          FROM seed
          JOIN LATERAL (
              SELECT DISTINCT u.id, u.username, u.name, u.avatar, u.twitter_followers
                FROM documents d2
                JOIN users u ON u.id = d2.user_id
               WHERE d2.deleted = FALSE
                 AND u.vip = TRUE
                 AND ($2 = '' OR u.username <> $2)
                 AND (
                       d2.canonical_url = seed.canonical_url
                    OR (cardinality(seed.canonical_referenced_urls) > 0
                        AND d2.canonical_referenced_urls && seed.canonical_referenced_urls)
                 )
          ) s ON true
         GROUP BY seed.raw_url
    ";
    let rows: Vec<(String, serde_json::Value)> = sqlx::query_as(sql)
        .bind(&urls)
        .bind(&exclude)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();
    let mut out = serde_json::Map::new();
    for (url, payload) in rows {
        out.insert(url, payload);
    }
    Json(serde_json::Value::Object(out)).into_response()
}

/// GET /api/users/{slug}/sources
///
/// Returns `[{ key, label, count }]` ordered by count desc — same shape
/// as the old `sources.json`.
///
/// Bypasses the `user_source_counts` view: that view does
/// `GROUP BY user_id, source` over the entire `documents` table and
/// PostgreSQL can't push the `username = $1` filter past the
/// aggregate, so the per-user request used to scan all 460k rows
/// (~830 ms cold). Resolving the user_id in an inline subquery and
/// aggregating only that user's rows lets the planner use
/// `idx_documents_user_source` for an index-only scan over the few
/// thousand rows that user actually has — output is byte-identical
/// to the view-based form.
pub async fn list_sources(
    State(pool): State<PgPool>,
    Path(slug): Path<String>,
) -> impl IntoResponse {
    let sql = "SELECT source, count(*)::bigint
         FROM documents
        WHERE user_id = (SELECT id FROM users WHERE username = $1)
          AND deleted = FALSE
          AND source <> ''
        GROUP BY source
        ORDER BY count(*) DESC";

    let rows: Vec<(String, i64)> = sqlx::query_as(sql)
        .bind(&slug)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

    let out: Vec<_> = rows
        .into_iter()
        .map(|(key, count)| {
            let label = source_label(&key);
            serde_json::json!({ "key": key, "label": label, "count": count })
        })
        .collect();

    Json(out).into_response()
}

/// GET /api/sources
///
/// Aggregate source-filter list across every VIP user's documents.
/// One round-trip alternative to fanning out
/// `/api/users/{slug}/sources` for each selected library — used by
/// the search page when the user picks 5+ libs and we route through
/// the unified `__all__` index. Returns the COMPLETE set of sources
/// (not just those visible in a particular result set), so the rail
/// can offer every available filter even though we never loaded
/// per-slug data.
///
/// Shape: `[{ key, label, count }]`, ordered by count desc.
pub async fn list_all_vip_sources(State(pool): State<PgPool>) -> impl IntoResponse {
    // Aggregate `documents` directly against the small (~452 row)
    // VIP user set instead of going through `user_source_counts`
    // (which pre-aggregates the full table by `(user_id, source)`
    // and forces a 460k-row scan even when only the VIP slice is
    // needed). The straight JOIN lets the planner hash the VIP id
    // set in memory and probe per doc — cold drops from ~830 ms
    // to ~650 ms, and the floor here is fundamental (we have to
    // touch every VIP doc to count by source).
    let sql = "SELECT d.source, count(*)::bigint
         FROM documents d
         JOIN users u ON u.id = d.user_id
        WHERE u.vip = TRUE
          AND d.deleted = FALSE
          AND d.source <> ''
        GROUP BY d.source
        ORDER BY count(*) DESC";

    let rows: Vec<(String, i64)> = sqlx::query_as(sql)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

    let out: Vec<_> = rows
        .into_iter()
        .map(|(key, count)| {
            let label = source_label(&key);
            serde_json::json!({ "key": key, "label": label, "count": count })
        })
        .collect();

    Json(out).into_response()
}

/// Brand-friendly label for a `documents.source` value.
pub fn source_label(key: &str) -> String {
    match key {
        "github" => "GitHub",
        "twitter" | "x" => "X",
        "youtube" => "YouTube",
        "hackernews" => "HackerNews",
        "huggingface" => "HuggingFace",
        "stackoverflow" => "StackOverflow",
        "wikipedia" => "Wikipedia",
        "reddit" => "Reddit",
        "scholar" => "Scholar",
        "semantic_scholar" => "Semantic Scholar",
        "dblp" => "DBLP",
        "arxiv" => "arXiv",
        "extra" => "Extra",
        "zotero" => "Zotero",
        // Per-website source keys are the raw hostname (e.g.
        // `mixedbread.com`) — keep them lowercase so they read as
        // domains instead of branded proper nouns.
        other if other.contains('.') && !other.contains(' ') => return other.to_string(),
        other => return capitalize(other),
    }
    .to_string()
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}
