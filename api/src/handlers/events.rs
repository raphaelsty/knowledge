//! Events and stats handlers.

use axum::{
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

const MAX_BATCH_SIZE: usize = 100;
const MAX_PAYLOAD_BYTES: usize = 8192;
const ALLOWED_EVENT_TYPES: &[&str] = &[
    "search",
    "click",
    "folder_browse",
    "filter_apply",
    "page_view",
];

// --- Event ingestion ---

#[derive(Debug, Deserialize)]
struct EventInput {
    session_id: String,
    event_type: String,
    payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    inserted: usize,
}

#[derive(Debug, Serialize)]
pub struct EventErrorResponse {
    error: String,
}

/// POST /events
pub async fn ingest_events(
    State(pool): State<PgPool>,
    body: Bytes,
) -> Result<Json<IngestResponse>, (StatusCode, Json<EventErrorResponse>)> {
    let events: Vec<EventInput> = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(EventErrorResponse {
                error: format!("Invalid JSON: {e}"),
            }),
        )
    })?;

    if events.is_empty() {
        return Ok(Json(IngestResponse { inserted: 0 }));
    }

    if events.len() > MAX_BATCH_SIZE {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(EventErrorResponse {
                error: format!(
                    "Batch too large: {} events (max {})",
                    events.len(),
                    MAX_BATCH_SIZE
                ),
            }),
        ));
    }

    for event in &events {
        if !ALLOWED_EVENT_TYPES.contains(&event.event_type.as_str()) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(EventErrorResponse {
                    error: format!("Unknown event_type: {}", event.event_type),
                }),
            ));
        }

        if event.session_id.is_empty() || event.session_id.len() > 64 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(EventErrorResponse {
                    error: "Invalid session_id".to_string(),
                }),
            ));
        }

        let payload_size = event.payload.to_string().len();
        if payload_size > MAX_PAYLOAD_BYTES {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(EventErrorResponse {
                    error: format!(
                        "Payload too large: {} bytes (max {})",
                        payload_size, MAX_PAYLOAD_BYTES
                    ),
                }),
            ));
        }
    }

    let mut inserted = 0;
    for event in &events {
        sqlx::query("INSERT INTO events (session_id, event_type, payload) VALUES ($1, $2, $3)")
            .bind(&event.session_id)
            .bind(&event.event_type)
            .bind(&event.payload)
            .execute(&pool)
            .await
            .map_err(|e| {
                tracing::error!("DB insert error: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(EventErrorResponse {
                        error: "Database error".to_string(),
                    }),
                )
            })?;
        inserted += 1;
    }

    Ok(Json(IngestResponse { inserted }))
}

// --- Stats ---

#[derive(Debug, Deserialize)]
pub struct StatsParams {
    days: Option<i32>,
    limit: Option<i64>,
}

impl StatsParams {
    fn days(&self) -> i32 {
        self.days.unwrap_or(7).clamp(1, 90)
    }

    fn limit(&self) -> i64 {
        self.limit.unwrap_or(15).clamp(1, 100)
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

    let row = sqlx::query_as::<_, (i64, i64, i64, i64, Option<f64>)>(
        "SELECT
            COUNT(*) FILTER (WHERE event_type = 'page_view'),
            COUNT(*) FILTER (WHERE event_type = 'search'),
            COUNT(*) FILTER (WHERE event_type = 'click'),
            COUNT(DISTINCT session_id),
            AVG((payload->>'latency_ms')::double precision) FILTER (WHERE event_type = 'search' AND payload->>'latency_ms' IS NOT NULL)
         FROM events
         WHERE created_at >= now() - make_interval(days => $1)",
    )
    .bind(days)
    .fetch_one(&pool)
    .await
    .unwrap_or((0, 0, 0, 0, None));

    let page_views = row.0;
    let searches = row.1;
    let clicks = row.2;
    let sessions = row.3;
    let avg_latency_ms = row.4.unwrap_or(0.0);
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

    let query = format!(
        "SELECT
            date_trunc('{trunc}', created_at)::text AS period,
            COUNT(*) FILTER (WHERE event_type = 'page_view'),
            COUNT(*) FILTER (WHERE event_type = 'search'),
            COUNT(*) FILTER (WHERE event_type = 'click'),
            COUNT(*) FILTER (WHERE event_type = 'folder_browse'),
            COUNT(*) FILTER (WHERE event_type = 'filter_apply')
         FROM events
         WHERE created_at >= now() - make_interval(days => $1)
         GROUP BY 1
         ORDER BY 1"
    );

    let rows = sqlx::query_as::<_, (String, i64, i64, i64, i64, i64)>(&query)
        .bind(days)
        .fetch_all(&pool)
        .await
        .unwrap_or_default();

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

    let rows = sqlx::query_as::<_, (String, i64)>(
        "SELECT payload->>'query' AS q, COUNT(*) AS c
         FROM events
         WHERE event_type = 'search'
           AND payload->>'query' IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)
         GROUP BY q
         ORDER BY c DESC
         LIMIT $2",
    )
    .bind(days)
    .bind(limit)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

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

    let rows = sqlx::query_as::<_, (String, i64)>(
        "SELECT payload->>'doc_url' AS u, COUNT(*) AS c
         FROM events
         WHERE event_type = 'click'
           AND payload->>'doc_url' IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)
         GROUP BY u
         ORDER BY c DESC
         LIMIT $2",
    )
    .bind(days)
    .bind(limit)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

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

    let rows = sqlx::query_as::<_, (String, i64)>(
        "SELECT payload->>'source_key' AS s, COUNT(*) AS c
         FROM events
         WHERE event_type = 'filter_apply'
           AND payload->>'source_key' IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)
         GROUP BY s
         ORDER BY c DESC",
    )
    .bind(days)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

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

/// GET /stats/folders
pub async fn folders(
    State(pool): State<PgPool>,
    Query(params): Query<StatsParams>,
) -> Json<Vec<FolderUsage>> {
    let days = params.days();
    let limit = params.limit();

    let rows = sqlx::query_as::<_, (String, i64)>(
        "SELECT payload->>'folder_name' AS f, COUNT(*) AS c
         FROM events
         WHERE event_type = 'folder_browse'
           AND payload->>'folder_name' IS NOT NULL
           AND created_at >= now() - make_interval(days => $1)
         GROUP BY f
         ORDER BY c DESC
         LIMIT $2",
    )
    .bind(days)
    .bind(limit)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();

    Json(
        rows.into_iter()
            .map(|r| FolderUsage {
                folder_name: r.0,
                count: r.1,
            })
            .collect(),
    )
}
