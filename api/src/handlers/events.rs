//! Events + stats handlers. Backed by the typed `events` and `sessions`
//! tables defined in `sources/sql/`.
//!
//! Wire format (kept compatible with the existing frontend `analytics.js`):
//!     { "session_id": "<uuid>", "event_type": "<name>", "payload": { ... } }
//!
//! The `payload` object carries typed fields that the server extracts into
//! columns. `user_id` inside the payload is required (it identifies the
//! library being browsed, not the viewer). `device_type` + `referrer_domain`
//! are session-level and only set on the first event of a session.

use axum::{
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

const MAX_BATCH_SIZE: usize = 100;

// ── Event-type enum ─────────────────────────────────────────────────────
// Matches the CHECK constraint on `events.event_type` (1..=6). Keep in
// sync with the COMMENT ON COLUMN in `sources/sql/events.sql`.

const EVT_VIEW: i16 = 1;
const EVT_SEARCH: i16 = 2;
const EVT_CLICK: i16 = 3;
const EVT_FIND_SIMILAR: i16 = 4;
const EVT_FILTER_APPLY: i16 = 5;
const EVT_FOLDER_BROWSE: i16 = 6;

fn event_type_code(name: &str) -> Option<i16> {
    match name {
        // `page_view` kept as an alias for backwards compat with the
        // current analytics.js; the canonical name is `view`.
        "view" | "page_view" => Some(EVT_VIEW),
        "search" => Some(EVT_SEARCH),
        "click" => Some(EVT_CLICK),
        "find_similar" | "click_similar" => Some(EVT_FIND_SIMILAR),
        "filter_apply" => Some(EVT_FILTER_APPLY),
        "folder_browse" => Some(EVT_FOLDER_BROWSE),
        _ => None,
    }
}

fn device_code(name: Option<&str>) -> i16 {
    match name {
        Some("mobile") => 1,
        _ => 0, // desktop / unknown
    }
}

fn sort_mode_code(name: Option<&str>) -> Option<i16> {
    match name? {
        "date" => Some(1),
        "relevance" => Some(0),
        _ => None,
    }
}

// ── Ingest ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct EventInput {
    session_id: Uuid,
    event_type: String,
    #[serde(default)]
    payload: EventPayload,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct EventPayload {
    // Required for every event — identifies the library being browsed.
    user_id: Option<i64>,

    // Session-scoped (only read on the first event per session).
    device_type: Option<String>,
    referrer_domain: Option<String>,

    // Search context.
    query: Option<String>,
    result_count: Option<i16>,
    latency_ms: Option<i32>,
    source_filter: Option<String>,
    sort_mode: Option<String>,

    // Click / find_similar context.
    doc_url: Option<String>,
    position: Option<i16>,
    score: Option<f32>,

    // Recommendation-training signals.
    personality_slug: Option<String>,
    viewer_user_id: Option<i64>,
    // Client wall-clock at event-fire time, ISO-8601 (e.g.
    // "2026-05-22T18:12:03.412Z"). Bound as text; PG parses on insert
    // into the TIMESTAMPTZ column. Keeping the chrono dep out of sqlx
    // avoids a libsqlite3-sys conflict with next-plaid.
    client_ts: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    inserted: usize,
}

#[derive(Debug, Serialize)]
pub struct EventErrorResponse {
    error: String,
}

fn bad_request(msg: impl Into<String>) -> (StatusCode, Json<EventErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(EventErrorResponse { error: msg.into() }),
    )
}

/// POST /events
pub async fn ingest_events(
    State(pool): State<PgPool>,
    body: Bytes,
) -> Result<Json<IngestResponse>, (StatusCode, Json<EventErrorResponse>)> {
    let events: Vec<EventInput> =
        serde_json::from_slice(&body).map_err(|e| bad_request(format!("Invalid JSON: {e}")))?;

    if events.is_empty() {
        return Ok(Json(IngestResponse { inserted: 0 }));
    }

    if events.len() > MAX_BATCH_SIZE {
        return Err(bad_request(format!(
            "Batch too large: {} events (max {MAX_BATCH_SIZE})",
            events.len()
        )));
    }

    // Validate up-front so we never do a partial insert.
    for ev in &events {
        if event_type_code(&ev.event_type).is_none() {
            return Err(bad_request(format!(
                "Unknown event_type: {}",
                ev.event_type
            )));
        }
        if ev.payload.user_id.is_none() {
            return Err(bad_request("payload.user_id is required"));
        }
    }

    let mut tx = pool.begin().await.map_err(|e| {
        tracing::error!("tx begin error: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(EventErrorResponse {
                error: "Database error".to_string(),
            }),
        )
    })?;

    let mut inserted = 0usize;
    for ev in &events {
        let evt_code = event_type_code(&ev.event_type).expect("validated above");
        let user_id = ev.payload.user_id.expect("validated above");

        // Upsert session (ignore device/referrer on subsequent events).
        sqlx::query(
            "INSERT INTO sessions (id, user_id, device, referrer_domain)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (id) DO UPDATE SET last_seen_at = now()",
        )
        .bind(ev.session_id.to_string())
        .bind(user_id)
        .bind(device_code(ev.payload.device_type.as_deref()))
        .bind(ev.payload.referrer_domain.as_deref())
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            tracing::error!("session upsert error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(EventErrorResponse {
                    error: "Database error".to_string(),
                }),
            )
        })?;

        sqlx::query(
            "INSERT INTO events
                (session_id, user_id, event_type,
                 query, result_count, latency_ms, source_filter, sort_mode,
                 doc_url, position, score,
                 personality_slug, viewer_user_id, client_ts)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                     $12, $13, $14::timestamptz)",
        )
        .bind(ev.session_id.to_string())
        .bind(user_id)
        .bind(evt_code)
        .bind(ev.payload.query.as_deref())
        .bind(ev.payload.result_count)
        .bind(ev.payload.latency_ms)
        .bind(ev.payload.source_filter.as_deref())
        .bind(sort_mode_code(ev.payload.sort_mode.as_deref()))
        .bind(ev.payload.doc_url.as_deref())
        .bind(ev.payload.position)
        .bind(ev.payload.score)
        .bind(ev.payload.personality_slug.as_deref())
        .bind(ev.payload.viewer_user_id)
        .bind(ev.payload.client_ts.as_deref())
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            tracing::error!("event insert error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(EventErrorResponse {
                    error: "Database error".to_string(),
                }),
            )
        })?;

        inserted += 1;
    }

    tx.commit().await.map_err(|e| {
        tracing::error!("tx commit error: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(EventErrorResponse {
                error: "Database error".to_string(),
            }),
        )
    })?;

    Ok(Json(IngestResponse { inserted }))
}

// ── Stats ───────────────────────────────────────────────────────────────
// All endpoints accept a shared set of query params. `user_id` is
// optional: when set, stats are scoped to one library; when absent,
// aggregated across all libraries (admin view).

#[derive(Debug, Deserialize)]
pub struct StatsParams {
    days: Option<i32>,
    limit: Option<i64>,
    user_id: Option<i64>,
}

impl StatsParams {
    fn days(&self) -> i32 {
        self.days.unwrap_or(7).clamp(1, 90)
    }
    fn limit(&self) -> i64 {
        self.limit.unwrap_or(15).clamp(1, 100)
    }
}

/// Appends an optional `AND user_id = $N` clause.
/// Returns the new SQL plus whether the user_id parameter should be bound.
fn scope_user_id(base_sql: &str, user_id: Option<i64>, next_param: usize) -> (String, bool) {
    match user_id {
        Some(_) => (format!("{base_sql} AND user_id = ${next_param}"), true),
        None => (base_sql.to_string(), false),
    }
}

#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    page_views: i64,
    searches: i64,
    clicks: i64,
    ctr: f64,
    avg_latency_ms: f64,
    sessions: i64,
}

/// GET /stats/overview
pub async fn overview(
    State(pool): State<PgPool>,
    Query(params): Query<StatsParams>,
) -> Json<OverviewResponse> {
    let days = params.days();

    let base = "SELECT
            COUNT(*) FILTER (WHERE event_type = $2),
            COUNT(*) FILTER (WHERE event_type = $3),
            COUNT(*) FILTER (WHERE event_type = $4),
            COUNT(DISTINCT session_id),
            AVG(latency_ms::double precision) FILTER (WHERE event_type = $3 AND latency_ms IS NOT NULL)
         FROM events
         WHERE created_at >= now() - make_interval(days => $1)";

    let (sql, bind_user) = scope_user_id(base, params.user_id, 5);

    let mut q = sqlx::query_as::<_, (i64, i64, i64, i64, Option<f64>)>(&sql)
        .bind(days)
        .bind(EVT_VIEW)
        .bind(EVT_SEARCH)
        .bind(EVT_CLICK);
    // NB: COUNT(DISTINCT session_id) is bound positionally but doesn't need
    // its own placeholder because it doesn't reference a param.
    if bind_user {
        q = q.bind(params.user_id.unwrap());
    }

    let row = q.fetch_one(&pool).await.unwrap_or((0, 0, 0, 0, None));

    let (page_views, searches, clicks, sessions, avg_latency) = row;
    let avg_latency_ms = avg_latency.unwrap_or(0.0);
    let ctr = if searches > 0 {
        (clicks as f64 / searches as f64) * 100.0
    } else {
        0.0
    };

    Json(OverviewResponse {
        page_views,
        searches,
        clicks,
        ctr: (ctr * 100.0).round() / 100.0,
        avg_latency_ms: (avg_latency_ms * 100.0).round() / 100.0,
        sessions,
    })
}

#[derive(Debug, Serialize)]
pub struct ActivityBucket {
    period: String,
    page_views: i64,
    searches: i64,
    clicks: i64,
    browses: i64,
    filters: i64,
}

/// GET /stats/activity
pub async fn activity(
    State(pool): State<PgPool>,
    Query(params): Query<StatsParams>,
) -> Json<Vec<ActivityBucket>> {
    let days = params.days();
    let trunc = if days <= 2 { "hour" } else { "day" };

    let base = format!(
        "SELECT
            date_trunc('{trunc}', created_at)::text AS period,
            COUNT(*) FILTER (WHERE event_type = $2),
            COUNT(*) FILTER (WHERE event_type = $3),
            COUNT(*) FILTER (WHERE event_type = $4),
            COUNT(*) FILTER (WHERE event_type = $5),
            COUNT(*) FILTER (WHERE event_type = $6)
         FROM events
         WHERE created_at >= now() - make_interval(days => $1)"
    );

    let (mut sql, bind_user) = scope_user_id(&base, params.user_id, 7);
    sql.push_str(" GROUP BY 1 ORDER BY 1");

    let mut q = sqlx::query_as::<_, (String, i64, i64, i64, i64, i64)>(&sql)
        .bind(days)
        .bind(EVT_VIEW)
        .bind(EVT_SEARCH)
        .bind(EVT_CLICK)
        .bind(EVT_FOLDER_BROWSE)
        .bind(EVT_FILTER_APPLY);
    if bind_user {
        q = q.bind(params.user_id.unwrap());
    }

    let rows = q.fetch_all(&pool).await.unwrap_or_default();

    Json(
        rows.into_iter()
            .map(|r| ActivityBucket {
                period: r.0,
                page_views: r.1,
                searches: r.2,
                clicks: r.3,
                browses: r.4,
                filters: r.5,
            })
            .collect(),
    )
}

#[derive(Debug, Serialize)]
pub struct TopQuery {
    query: String,
    count: i64,
}

/// GET /stats/top-queries
pub async fn top_queries(
    State(pool): State<PgPool>,
    Query(params): Query<StatsParams>,
) -> Json<Vec<TopQuery>> {
    let days = params.days();
    let limit = params.limit();

    let base = "SELECT query, COUNT(*) AS c
         FROM events
         WHERE event_type = $2
           AND query IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)";

    let (mut sql, bind_user) = scope_user_id(base, params.user_id, 3);
    if bind_user {
        sql.push_str(" GROUP BY query ORDER BY c DESC LIMIT $4");
    } else {
        sql.push_str(" GROUP BY query ORDER BY c DESC LIMIT $3");
    }

    let mut q = sqlx::query_as::<_, (String, i64)>(&sql)
        .bind(days)
        .bind(EVT_SEARCH);
    if bind_user {
        q = q.bind(params.user_id.unwrap());
    }
    q = q.bind(limit);

    let rows = q.fetch_all(&pool).await.unwrap_or_default();

    Json(
        rows.into_iter()
            .map(|r| TopQuery {
                query: r.0,
                count: r.1,
            })
            .collect(),
    )
}

#[derive(Debug, Serialize)]
pub struct TopClick {
    doc_url: String,
    count: i64,
}

/// GET /stats/top-clicks
pub async fn top_clicks(
    State(pool): State<PgPool>,
    Query(params): Query<StatsParams>,
) -> Json<Vec<TopClick>> {
    let days = params.days();
    let limit = params.limit();

    let base = "SELECT doc_url, COUNT(*) AS c
         FROM events
         WHERE event_type = $2
           AND doc_url IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)";

    let (mut sql, bind_user) = scope_user_id(base, params.user_id, 3);
    if bind_user {
        sql.push_str(" GROUP BY doc_url ORDER BY c DESC LIMIT $4");
    } else {
        sql.push_str(" GROUP BY doc_url ORDER BY c DESC LIMIT $3");
    }

    let mut q = sqlx::query_as::<_, (String, i64)>(&sql)
        .bind(days)
        .bind(EVT_CLICK);
    if bind_user {
        q = q.bind(params.user_id.unwrap());
    }
    q = q.bind(limit);

    let rows = q.fetch_all(&pool).await.unwrap_or_default();

    Json(
        rows.into_iter()
            .map(|r| TopClick {
                doc_url: r.0,
                count: r.1,
            })
            .collect(),
    )
}

#[derive(Debug, Serialize)]
pub struct SourceUsage {
    source_key: String,
    count: i64,
}

/// GET /stats/sources
pub async fn sources(
    State(pool): State<PgPool>,
    Query(params): Query<StatsParams>,
) -> Json<Vec<SourceUsage>> {
    let days = params.days();

    let base = "SELECT source_filter, COUNT(*) AS c
         FROM events
         WHERE event_type = $2
           AND source_filter IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)";

    let (mut sql, bind_user) = scope_user_id(base, params.user_id, 3);
    sql.push_str(" GROUP BY source_filter ORDER BY c DESC");

    let mut q = sqlx::query_as::<_, (String, i64)>(&sql)
        .bind(days)
        .bind(EVT_FILTER_APPLY);
    if bind_user {
        q = q.bind(params.user_id.unwrap());
    }

    let rows = q.fetch_all(&pool).await.unwrap_or_default();

    Json(
        rows.into_iter()
            .map(|r| SourceUsage {
                source_key: r.0,
                count: r.1,
            })
            .collect(),
    )
}

#[derive(Debug, Serialize)]
pub struct FolderUsage {
    folder_name: String,
    count: i64,
}

/// GET /stats/folders — placeholder endpoint returning an empty list.
///
/// The typed schema no longer stores a `folder_name` column; when we wire
/// folder-browse telemetry from the frontend we'll add a dedicated column
/// and populate it here.
pub async fn folders(
    State(_pool): State<PgPool>,
    Query(_params): Query<StatsParams>,
) -> Json<Vec<FolderUsage>> {
    Json(Vec::new())
}
