use axum::{
    extract::{Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

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

// --- Overview ---

#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    page_views: i64,
    searches: i64,
    clicks: i64,
    ctr: f64,
    avg_latency_ms: f64,
    sessions: i64,
}

pub async fn overview(
    State(state): State<Arc<AppState>>,
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
    .fetch_one(&state.pool)
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

// --- Activity over time ---

#[derive(Debug, Serialize)]
pub struct ActivityBucket {
    period: String,
    page_views: i64,
    searches: i64,
    clicks: i64,
    browses: i64,
    filters: i64,
}

pub async fn activity(
    State(state): State<Arc<AppState>>,
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
        .fetch_all(&state.pool)
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

// --- Top queries ---

#[derive(Debug, Serialize)]
pub struct TopQuery {
    query: String,
    count: i64,
}

pub async fn top_queries(
    State(state): State<Arc<AppState>>,
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
    .fetch_all(&state.pool)
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

// --- Top clicks ---

#[derive(Debug, Serialize)]
pub struct TopClick {
    doc_url: String,
    count: i64,
}

pub async fn top_clicks(
    State(state): State<Arc<AppState>>,
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
    .fetch_all(&state.pool)
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

// --- Source filter usage ---

#[derive(Debug, Serialize)]
pub struct SourceUsage {
    source_key: String,
    count: i64,
}

pub async fn sources(
    State(state): State<Arc<AppState>>,
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
    .fetch_all(&state.pool)
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

// --- Folder browse counts ---

#[derive(Debug, Serialize)]
pub struct FolderUsage {
    folder_name: String,
    count: i64,
}

pub async fn folders(
    State(state): State<Arc<AppState>>,
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
    .fetch_all(&state.pool)
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
