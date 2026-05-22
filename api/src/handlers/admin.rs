//! Admin panel endpoints.
//!
//! Read-only dashboard for the operator. Every route is gated by
//! [`require_raphael`] which checks the session cookie and refuses
//! to serve anyone other than the `raphael-sourty` account.
//!
//! Routes (mounted under the auth router so the session cookie /
//! CORS layer applies):
//!   GET /api/admin/overview                  — KPI tiles
//!   GET /api/admin/sources?days=N            — per-source 7d aggregates
//!   GET /api/admin/sources/{name}/failures   — failure groups by error message
//!   GET /api/admin/users?q=…                 — user list with last run
//!   GET /api/admin/users/{slug}/runs         — last 50 runs for one user
//!   GET /api/admin/indices                   — per-user index verdicts
//!   GET /api/admin/live                      — tail of recent pipeline activity

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::PgPool;

use crate::handlers::auth::current_user;
use crate::state::AppState;
use std::sync::Arc;

/// Slug allowed to access the admin panel. Hard-coded: this is a
/// single-operator app and we don't want a runtime config knob that
/// could be flipped accidentally. Anyone else gets 403.
const ADMIN_SLUG: &str = "raphael-sourty";

/// Pull the Postgres pool out of the shared `AppState`. Every admin
/// endpoint needs it; this trims the boilerplate.
#[allow(clippy::result_large_err)]
fn pg(state: &Arc<AppState>) -> Result<PgPool, Response> {
    state.pg_pool.clone().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "DATABASE_URL not configured",
        )
            .into_response()
    })
}

/// Session guard. Returns `Ok(pool)` for the admin, `Err(response)`
/// to short-circuit the handler with 401 (no session) or 403 (wrong
/// user). Bundles the pool fetch so handlers only call this once.
#[allow(clippy::result_large_err)]
async fn require_raphael(state: &Arc<AppState>, jar: &CookieJar) -> Result<PgPool, Response> {
    let pool = pg(state)?;
    let Some(me) = current_user(&pool, jar).await else {
        return Err(StatusCode::UNAUTHORIZED.into_response());
    };
    if me.slug != ADMIN_SLUG {
        return Err(StatusCode::FORBIDDEN.into_response());
    }
    Ok(pool)
}

// ── /api/admin/overview ──────────────────────────────────────────────

/// One-shot KPI tile payload. Single round-trip to PG.
pub async fn overview(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };

    let row: Result<KpiRow, _> = sqlx::query_as::<_, KpiRow>(
        r#"
        SELECT
            (SELECT COUNT(*) FROM users)::bigint                                          AS total_users,
            (SELECT COUNT(*) FROM users WHERE vip)::bigint                                AS vip_users,
            (SELECT COUNT(*) FROM documents WHERE deleted = FALSE)::bigint                AS total_docs,
            (SELECT COUNT(*) FROM documents d JOIN users u ON u.id=d.user_id
               WHERE u.vip AND d.deleted = FALSE)::bigint                                 AS vip_docs,
            (SELECT COUNT(*) FROM pipeline_runs WHERE status = 'running')::bigint         AS running_now,
            (SELECT COUNT(*) FROM pipeline_runs WHERE started_at > NOW()-INTERVAL '7 days')::bigint
                                                                                          AS runs_7d,
            (SELECT COUNT(*) FROM pipeline_runs
              WHERE started_at > NOW()-INTERVAL '7 days' AND status = 'success')::bigint  AS runs_7d_ok,
            (SELECT COUNT(*) FROM pipeline_runs
              WHERE started_at > NOW()-INTERVAL '7 days' AND status = 'failed')::bigint   AS runs_7d_failed,
            (SELECT COUNT(*) FROM pipeline_source_runs
              WHERE started_at > NOW()-INTERVAL '7 days' AND status = 'failed')::bigint   AS source_runs_7d_failed,
            (SELECT COALESCE(SUM(new_documents),0) FROM pipeline_runs
              WHERE started_at > NOW()-INTERVAL '7 days')::bigint                         AS new_docs_7d
        "#,
    )
    .fetch_one(&pool)
    .await;

    match row {
        Ok(r) => Json(r).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("overview query failed: {}", e),
        )
            .into_response(),
    }
}

#[derive(Serialize, sqlx::FromRow)]
pub struct KpiRow {
    pub total_users: i64,
    pub vip_users: i64,
    pub total_docs: i64,
    pub vip_docs: i64,
    pub running_now: i64,
    pub runs_7d: i64,
    pub runs_7d_ok: i64,
    pub runs_7d_failed: i64,
    pub source_runs_7d_failed: i64,
    pub new_docs_7d: i64,
}

// ── /api/admin/sources?days=N ────────────────────────────────────────

#[derive(Deserialize)]
pub struct DaysQuery {
    #[serde(default = "default_days")]
    pub days: i32,
}
fn default_days() -> i32 {
    7
}

#[derive(Serialize, sqlx::FromRow)]
pub struct SourceHealthRow {
    pub source: String,
    pub total_runs: i64,
    pub success_runs: i64,
    pub failed_runs: i64,
    pub skipped_runs: i64,
    pub users_touched: i64,
    pub users_failing: i64,
    pub total_new_docs: i64,
    pub avg_duration_ok: f64,
    pub last_failure_at: Option<String>,
    pub last_success_at: Option<String>,
}

pub async fn sources(
    State(state): State<Arc<AppState>>,
    jar: CookieJar,
    Query(q): Query<DaysQuery>,
) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    let days = q.days.clamp(1, 90);
    let rows = sqlx::query_as::<_, SourceHealthRow>(
        r#"
        SELECT
            source,
            COUNT(*)                                            AS total_runs,
            COUNT(*) FILTER (WHERE status = 'success')          AS success_runs,
            COUNT(*) FILTER (WHERE status = 'failed')           AS failed_runs,
            COUNT(*) FILTER (WHERE status = 'skipped')          AS skipped_runs,
            COUNT(DISTINCT user_id)                             AS users_touched,
            COUNT(DISTINCT user_id) FILTER (WHERE status='failed') AS users_failing,
            COALESCE(SUM(new_documents),0)::bigint              AS total_new_docs,
            COALESCE(AVG(duration_secs) FILTER (WHERE status='success'), 0)::float8
                                                                AS avg_duration_ok,
            to_char(MAX(started_at) FILTER (WHERE status='failed'),
                    'YYYY-MM-DD"T"HH24:MI:SS"Z"')               AS last_failure_at,
            to_char(MAX(started_at) FILTER (WHERE status='success'),
                    'YYYY-MM-DD"T"HH24:MI:SS"Z"')               AS last_success_at
        FROM pipeline_source_runs
        WHERE started_at > NOW() - make_interval(days => $1)
        GROUP BY source
        ORDER BY (COUNT(*) FILTER (WHERE status='failed')) DESC, total_runs DESC
        "#,
    )
    .bind(days)
    .fetch_all(&pool)
    .await;

    match rows {
        Ok(r) => Json(r).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("sources query failed: {}", e),
        )
            .into_response(),
    }
}

// ── /api/admin/sources/{name}/failures ───────────────────────────────

pub async fn source_failures(
    State(state): State<Arc<AppState>>,
    jar: CookieJar,
    Path(source): Path<String>,
    Query(q): Query<DaysQuery>,
) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    let days = q.days.clamp(1, 90);
    #[derive(sqlx::FromRow)]
    struct Row {
        error: Option<String>,
        detail: Option<String>,
        username: String,
        name: Option<String>,
        avatar: Option<String>,
        vip: Option<bool>,
        started_at: String,
        duration_secs: Option<f64>,
    }
    let rows = sqlx::query_as::<_, Row>(
        r#"
        SELECT psr.error, psr.detail,
               to_char(psr.started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
               psr.duration_secs::float8 AS duration_secs,
               u.username, u.name, u.avatar, u.vip
          FROM pipeline_source_runs psr
          JOIN users u ON u.id = psr.user_id
         WHERE psr.source = $1 AND psr.status = 'failed'
           AND psr.started_at > NOW() - make_interval(days => $2)
         ORDER BY psr.started_at DESC
         LIMIT 500
        "#,
    )
    .bind(&source)
    .bind(days)
    .fetch_all(&pool)
    .await;

    let rows = match rows {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failures query failed: {}", e),
            )
                .into_response();
        }
    };

    // Group by error message — same shape the old admin used so the
    // frontend stays simple. {message, count, users[], samples[]}.
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for r in rows {
        groups
            .entry(r.error.clone().unwrap_or_default())
            .or_default()
            .push(r);
    }
    let mut out: Vec<Value> = groups
        .into_iter()
        .map(|(msg, rows)| {
            let mut seen = std::collections::HashSet::new();
            let users: Vec<Value> = rows
                .iter()
                .filter_map(|r| {
                    if seen.insert(r.username.clone()) {
                        Some(json!({
                            "username": r.username,
                            "name": r.name,
                            "avatar": r.avatar,
                            "vip": r.vip,
                        }))
                    } else {
                        None
                    }
                })
                .collect();
            let samples: Vec<Value> = rows
                .iter()
                .take(8)
                .map(|r| {
                    json!({
                        "username": r.username,
                        "name": r.name,
                        "detail": r.detail,
                        "started_at": r.started_at,
                        "duration_secs": r.duration_secs,
                    })
                })
                .collect();
            json!({
                "message": msg,
                "count": rows.len(),
                "users": users,
                "samples": samples,
            })
        })
        .collect();
    out.sort_by_key(|g| -(g["count"].as_i64().unwrap_or(0)));
    Json(json!({"source": source, "groups": out})).into_response()
}

// ── /api/admin/users?q=… ─────────────────────────────────────────────

#[derive(Deserialize)]
pub struct UsersQuery {
    #[serde(default)]
    pub q: String,
}

pub async fn users_list(
    State(state): State<Arc<AppState>>,
    jar: CookieJar,
    Query(uq): Query<UsersQuery>,
) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    #[derive(sqlx::FromRow, Serialize)]
    struct Row {
        id: i64,
        username: String,
        name: Option<String>,
        vip: Option<bool>,
        avatar: Option<String>,
        index_name: Option<String>,
        doc_count: i64,
        last_run_status: Option<String>,
        last_run_trigger: Option<String>,
        last_run_started_at: Option<String>,
        last_run_duration_secs: Option<f64>,
        last_run_new_docs: Option<i64>,
        last_run_error: Option<String>,
        last_run_stage: Option<String>,
    }
    let q = uq.q.trim().to_string();
    let rows = sqlx::query_as::<_, Row>(
        r#"
        WITH latest AS (
            SELECT DISTINCT ON (user_id)
                   user_id, status, trigger, started_at,
                   duration_secs, new_documents, error, stage
              FROM pipeline_runs
             ORDER BY user_id, started_at DESC
        ),
        doc_counts AS (
            SELECT user_id, COUNT(*)::bigint AS n
              FROM documents WHERE deleted = FALSE GROUP BY user_id
        )
        SELECT u.id, u.username, u.name, u.vip, u.avatar, u.index_name,
               COALESCE(d.n, 0)::bigint                 AS doc_count,
               l.status                                 AS last_run_status,
               l.trigger                                AS last_run_trigger,
               to_char(l.started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
                                                        AS last_run_started_at,
               l.duration_secs::float8                  AS last_run_duration_secs,
               l.new_documents::bigint                  AS last_run_new_docs,
               l.error                                  AS last_run_error,
               l.stage                                  AS last_run_stage
          FROM users u
          LEFT JOIN latest     l ON l.user_id = u.id
          LEFT JOIN doc_counts d ON d.user_id = u.id
         WHERE ($1 = '' OR u.username ILIKE '%' || $1 || '%'
                       OR u.name     ILIKE '%' || $1 || '%')
         ORDER BY u.vip DESC NULLS LAST, l.started_at DESC NULLS LAST, u.name
         LIMIT 300
        "#,
    )
    // Escape % / _ in the user-supplied filter so they match literally
    // (a search of `%` would otherwise return every row).
    .bind(crate::handlers::sql_like::escape_like_pattern(&q))
    .fetch_all(&pool)
    .await;

    match rows {
        Ok(r) => Json(r).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("users query failed: {}", e),
        )
            .into_response(),
    }
}

// ── /api/admin/users/{slug}/runs ─────────────────────────────────────

pub async fn user_runs(
    State(state): State<Arc<AppState>>,
    jar: CookieJar,
    Path(slug): Path<String>,
) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    let rows = sqlx::query(
        r#"
        SELECT r.id, r.trigger, r.status, r.stage,
               to_char(r.started_at,  'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
               to_char(r.finished_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS finished_at,
               r.duration_secs::float8 AS duration_secs,
               r.new_documents::bigint  AS new_documents,
               r.total_documents::bigint AS total_documents,
               r.error, r.timings
          FROM pipeline_runs r
          JOIN users u ON u.id = r.user_id
         WHERE u.username = $1
         ORDER BY r.started_at DESC
         LIMIT 50
        "#,
    )
    .bind(&slug)
    .fetch_all(&pool)
    .await;

    let rows = match rows {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("user runs failed: {}", e),
            )
                .into_response();
        }
    };
    use sqlx::Row;
    let out: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "id": r.try_get::<i64, _>("id").ok(),
                "trigger": r.try_get::<Option<String>, _>("trigger").ok().flatten(),
                "status": r.try_get::<Option<String>, _>("status").ok().flatten(),
                "stage": r.try_get::<Option<String>, _>("stage").ok().flatten(),
                "started_at": r.try_get::<Option<String>, _>("started_at").ok().flatten(),
                "finished_at": r.try_get::<Option<String>, _>("finished_at").ok().flatten(),
                // chrono types aren't enabled in our sqlx build (would
                // pull a conflicting sqlite native lib); we cast to
                // ISO-8601 via to_char in the SQL above.
                "duration_secs": r.try_get::<Option<f64>, _>("duration_secs").ok().flatten(),
                "new_documents": r.try_get::<Option<i64>, _>("new_documents").ok().flatten(),
                "total_documents": r.try_get::<Option<i64>, _>("total_documents").ok().flatten(),
                "error": r.try_get::<Option<String>, _>("error").ok().flatten(),
                "timings": r.try_get::<Option<serde_json::Value>, _>("timings").ok().flatten(),
            })
        })
        .collect();
    Json(out).into_response()
}

// ── /api/admin/indices — live scan via in-memory state ───────────────

pub async fn indices(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    #[derive(sqlx::FromRow)]
    struct Row {
        username: String,
        name: Option<String>,
        vip: Option<bool>,
        index_name: Option<String>,
        pg_total: i64,
        pg_indexed: i64,
    }
    let rows = sqlx::query_as::<_, Row>(
        r#"
        SELECT u.username, u.name, u.vip, u.index_name,
               COUNT(d.url)::bigint                                                AS pg_total,
               COUNT(d.url) FILTER (WHERE d.indexed = TRUE)::bigint                AS pg_indexed
          FROM users u
          LEFT JOIN documents d ON d.user_id = u.id AND d.deleted = FALSE
         GROUP BY u.id
         ORDER BY u.vip DESC NULLS LAST, u.username
        "#,
    )
    .fetch_all(&pool)
    .await;

    let rows = match rows {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("indices query failed: {}", e),
            )
                .into_response();
        }
    };

    // Classify each row using the loaded index info directly from
    // AppState — no HTTP round-trip needed. Mirrors the verdicts the
    // Python `classify_index` returns so the frontend rendering is
    // identical to `make repair-indexes`.
    const DRIFT_ABS: i64 = 5;
    const DRIFT_FRAC: f64 = 0.05;
    let mut summary: std::collections::BTreeMap<&'static str, i64> = Default::default();
    let mut details: Vec<Value> = Vec::with_capacity(rows.len());
    for r in rows {
        let name = r.index_name.clone().unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let (verdict, reason) = classify_one(
            &state,
            &name,
            r.pg_total,
            r.pg_indexed,
            DRIFT_ABS,
            DRIFT_FRAC,
        );
        *summary.entry(verdict).or_insert(0) += 1;
        details.push(json!({
            "username": r.username,
            "name": r.name,
            "vip": r.vip,
            "index_name": name,
            "pg_total": r.pg_total,
            "pg_indexed": r.pg_indexed,
            "verdict": verdict,
            "reason": reason,
        }));
    }
    Json(json!({"summary": summary, "details": details})).into_response()
}

fn classify_one(
    state: &AppState,
    name: &str,
    pg_total: i64,
    pg_indexed: i64,
    drift_abs: i64,
    drift_frac: f64,
) -> (&'static str, String) {
    if !state.index_exists_on_disk(name) {
        return if pg_total > 0 {
            ("missing", "API 404".into())
        } else {
            ("empty", "no docs".into())
        };
    }
    // `get_index_summary` reads metadata.json directly — no mmap
    // load — so we can scan all 450 indices in milliseconds.
    let summary = match state.get_index_summary(name) {
        Ok(s) => s,
        Err(_) => return ("error", "metadata unreadable".into()),
    };
    let n_docs = summary.num_documents as i64;
    let n_emb = summary.num_embeddings as i64;
    // Two flavours of broken — search returns nothing in both.
    //   a) num_documents > 0 but num_embeddings == 0 — embedder
    //      crashed mid-write (the original "broken" case).
    //   b) num_documents == 0 while PG has docs — index file
    //      exists (declared/loads OK) but is empty. Until now
    //      this was bucketed under `pg_drift`, which is opt-in
    //      for repair; promote it to `broken` so the in-pipeline
    //      heal hook auto-rebuilds it on the user's next pass.
    if n_docs > 0 && n_emb == 0 {
        return (
            "broken",
            format!("num_documents={n_docs}, num_embeddings=0"),
        );
    }
    if n_docs == 0 && pg_total > 0 {
        return ("broken", format!("api=0, pg has {pg_total} doc(s)"));
    }
    let baseline = pg_indexed.max(pg_total);
    if baseline > 0 {
        let drift = (pg_indexed - n_docs).abs();
        let threshold = drift_abs.max((baseline as f64 * drift_frac) as i64);
        if drift > threshold {
            return (
                "pg_drift",
                format!("pg_indexed={pg_indexed} api={n_docs} drift={drift}"),
            );
        }
    }
    // `backlog` mirrors the indexer-daemon's PRI_BACKLOG tier: the
    // index agrees with pg_indexed but PG still carries indexed=false
    // rows (e.g. tweets that arrived via sync-tweets-to-prod after
    // the last embed pass). Without this we'd report `healthy` and
    // the admin panel would hide those users from view while the
    // daemon happily queues them in the background.
    let backlog = pg_total - pg_indexed;
    if backlog > 0 {
        return ("backlog", format!("{backlog} indexed=false in PG"));
    }
    ("healthy", String::new())
}

// ── /api/admin/live ──────────────────────────────────────────────────

pub async fn live(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    let runs = sqlx::query(
        r#"
        SELECT r.id, r.status, r.trigger, r.stage,
               to_char(r.started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
               r.duration_secs::float8 AS duration_secs,
               r.new_documents::bigint AS new_documents,
               r.error, u.username, u.name, u.vip
          FROM pipeline_runs r
          JOIN users u ON u.id = r.user_id
         ORDER BY r.started_at DESC
         LIMIT 50
        "#,
    )
    .fetch_all(&pool)
    .await;
    let source_runs = sqlx::query(
        r#"
        SELECT psr.id, psr.source, psr.status,
               to_char(psr.started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
               psr.duration_secs::float8 AS duration_secs,
               psr.new_documents::bigint AS new_documents,
               psr.error, u.username, u.name, u.vip
          FROM pipeline_source_runs psr
          JOIN users u ON u.id = psr.user_id
         ORDER BY psr.started_at DESC
         LIMIT 80
        "#,
    )
    .fetch_all(&pool)
    .await;

    use sqlx::Row;
    let runs_v: Vec<Value> = runs
        .unwrap_or_default()
        .into_iter()
        .map(|r| {
            json!({
                "id": r.try_get::<i64, _>("id").ok(),
                "status": r.try_get::<Option<String>, _>("status").ok().flatten(),
                "trigger": r.try_get::<Option<String>, _>("trigger").ok().flatten(),
                "stage": r.try_get::<Option<String>, _>("stage").ok().flatten(),
                "started_at": r.try_get::<Option<String>, _>("started_at").ok().flatten(),
                "duration_secs": r.try_get::<Option<f64>, _>("duration_secs").ok().flatten(),
                "new_documents": r.try_get::<Option<i64>, _>("new_documents").ok().flatten(),
                "error": r.try_get::<Option<String>, _>("error").ok().flatten(),
                "username": r.try_get::<String, _>("username").ok(),
                "name": r.try_get::<Option<String>, _>("name").ok().flatten(),
                "vip": r.try_get::<Option<bool>, _>("vip").ok().flatten(),
            })
        })
        .collect();
    let source_runs_v: Vec<Value> = source_runs
        .unwrap_or_default()
        .into_iter()
        .map(|r| {
            json!({
                "id": r.try_get::<i64, _>("id").ok(),
                "source": r.try_get::<Option<String>, _>("source").ok().flatten(),
                "status": r.try_get::<Option<String>, _>("status").ok().flatten(),
                "started_at": r.try_get::<Option<String>, _>("started_at").ok().flatten(),
                "duration_secs": r.try_get::<Option<f64>, _>("duration_secs").ok().flatten(),
                "new_documents": r.try_get::<Option<i64>, _>("new_documents").ok().flatten(),
                "error": r.try_get::<Option<String>, _>("error").ok().flatten(),
                "username": r.try_get::<String, _>("username").ok(),
                "name": r.try_get::<Option<String>, _>("name").ok().flatten(),
                "vip": r.try_get::<Option<bool>, _>("vip").ok().flatten(),
            })
        })
        .collect();
    Json(json!({"runs": runs_v, "source_runs": source_runs_v})).into_response()
}

// ── /api/admin/system — host metrics for the box this API runs on ────

/// CPU load + memory + disk snapshot of the API host. Linux-only
/// reads — on macOS dev boxes the fields degrade to `null` and the
/// frontend renders a "—" placeholder. Cheap: each request is three
/// small file reads + one `df` exec, total <2 ms warm.
pub async fn system_stats(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    // Auth-only — admin can see node metrics, nobody else.
    if let Err(e) = require_raphael(&state, &jar).await {
        return e;
    }

    let load = read_loadavg();
    let cpu_count = std::thread::available_parallelism()
        .ok()
        .map(|n| n.get() as i64);
    let mem = read_meminfo();
    // Disk fill for the volume holding the index dir — that's where
    // the bulk of the bytes live (300k VIP docs × ~few KB metadata +
    // mmap'd embedding arrays). Falls back to "." if the index dir
    // hasn't been created yet (fresh checkout / test runner).
    let disk_path = if state.config.index_dir.exists() {
        state.config.index_dir.clone()
    } else {
        std::path::PathBuf::from(".")
    };
    let disk = read_disk(&disk_path);

    Json(json!({
        "cpu": {
            "load_1m":  load.map(|l| l.0),
            "load_5m":  load.map(|l| l.1),
            "load_15m": load.map(|l| l.2),
            "count":    cpu_count,
        },
        "memory": mem.map(|(total_kb, avail_kb)| json!({
            "total_bytes":     total_kb * 1024,
            "available_bytes": avail_kb * 1024,
            "used_bytes":      total_kb.saturating_sub(avail_kb) * 1024,
            "used_fraction":   if total_kb > 0 {
                (total_kb - avail_kb) as f64 / total_kb as f64
            } else { 0.0 },
        })).unwrap_or(Value::Null),
        "disk": disk.map(|(total_kb, avail_kb, path)| json!({
            "path":            path,
            "total_bytes":     total_kb * 1024,
            "available_bytes": avail_kb * 1024,
            "used_bytes":      total_kb.saturating_sub(avail_kb) * 1024,
            "used_fraction":   if total_kb > 0 {
                (total_kb - avail_kb) as f64 / total_kb as f64
            } else { 0.0 },
        })).unwrap_or(Value::Null),
    }))
    .into_response()
}

// Linux /proc/loadavg → "0.42 0.38 0.35 1/123 4567"
fn read_loadavg() -> Option<(f64, f64, f64)> {
    let s = std::fs::read_to_string("/proc/loadavg").ok()?;
    let mut parts = s.split_whitespace();
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    ))
}

// Linux /proc/meminfo → returns (MemTotal kB, MemAvailable kB).
// MemAvailable is the "what userspace can use without swapping"
// estimate the kernel publishes (since 3.14) — better than
// MemFree, which excludes page cache reclaim.
fn read_meminfo() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total: Option<u64> = None;
    let mut avail: Option<u64> = None;
    for line in s.lines() {
        let kv = |prefix: &str| -> Option<u64> {
            line.strip_prefix(prefix)?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        };
        if total.is_none() {
            if let Some(n) = kv("MemTotal:") {
                total = Some(n);
                continue;
            }
        }
        if avail.is_none() {
            if let Some(n) = kv("MemAvailable:") {
                avail = Some(n);
            }
        }
        if total.is_some() && avail.is_some() {
            break;
        }
    }
    Some((total?, avail?))
}

// Shell out to `df -Pk <path>` and parse the data row.
//   header        → Filesystem 1024-blocks Used Available Capacity Mounted on
//   data row[0]   → /dev/sda1
//   data row[1]   → total 1K-blocks
//   data row[3]   → available 1K-blocks
//   data row[5]   → mount point
fn read_disk(path: &std::path::Path) -> Option<(u64, u64, String)> {
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    // Some BSD `df` wraps the filesystem name onto its own line for
    // long device paths; the "1024-blocks" column may then be on the
    // line below. Collect all whitespace-separated tokens past the
    // header and pick by index.
    let mut tokens: Vec<&str> = Vec::new();
    for (i, line) in s.lines().enumerate() {
        if i == 0 {
            continue;
        }
        tokens.extend(line.split_whitespace());
    }
    // After the header we expect at least 6 tokens.
    if tokens.len() < 6 {
        return None;
    }
    let total_kb: u64 = tokens.get(1)?.parse().ok()?;
    let avail_kb: u64 = tokens.get(3)?.parse().ok()?;
    let mount = tokens.get(5).map(|s| s.to_string()).unwrap_or_default();
    Some((total_kb, avail_kb, mount))
}

// ── /api/admin/indexer — what the indexer daemon is doing ────────────

/// The PG advisory-lock namespace constant the indexer daemon uses
/// when it takes a per-user lock (see `sources/utils/index_locks.py`:
/// `_NAMESPACE = 0x1DEC1`). We grep `pg_locks` for rows with this
/// classid to discover which users are currently being processed —
/// the daemon doesn't write a separate state row, so this is the
/// cheapest source-of-truth for "what's in flight right now".
const INDEX_LOCK_NAMESPACE: i32 = 0x1DEC1;

/// GET /api/admin/indexer
///
/// Returns two lists used by the admin Overview's "Indexer activity"
/// section:
///
///   * `active` — users whose index is currently being written to.
///     Computed by joining `pg_locks` (advisory locks held in our
///     namespace) to `users` + the user's currently-running
///     `pipeline_runs` row, so we get username + stage in flight.
///   * `recent` — last 20 finished indexer runs (status = success
///     or failed). One row per user-run with how many fresh docs
///     got embedded; `new_documents > 0` reads as "updated",
///     `= 0` reads as "cleaned" (heal-then-re-embed case where
///     nothing new came in).
pub async fn indexer_activity(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };

    // Active set. pg_locks gives us every advisory lock by (classid,
    // objid). We filter to our namespace then join `users` for the
    // slug + the user's most recent `pipeline_runs` row so the
    // payload includes the running stage. NOTE: a lock without a
    // matching running run is normal during the brief window between
    // lock acquisition and the first `_mark_stage("fetch")` write.
    let active = sqlx::query(
        r#"
        WITH active_locks AS (
            SELECT objid::bigint AS user_id
              FROM pg_locks
             WHERE locktype = 'advisory' AND classid = $1
        ),
        latest_run AS (
            SELECT DISTINCT ON (user_id)
                   user_id, id, status, stage, started_at, duration_secs,
                   new_documents
              FROM pipeline_runs
             ORDER BY user_id, started_at DESC
        )
        SELECT u.username,
               u.name,
               u.vip,
               l.stage,
               to_char(l.started_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
               EXTRACT(EPOCH FROM (now() - l.started_at))::float8  AS running_for_secs,
               l.new_documents::bigint AS new_documents
          FROM active_locks a
          JOIN users u       ON u.id = a.user_id
          LEFT JOIN latest_run l ON l.user_id = a.user_id
                                AND l.status = 'running'
         ORDER BY u.vip DESC NULLS LAST, u.username
        "#,
    )
    .bind(INDEX_LOCK_NAMESPACE)
    .fetch_all(&pool)
    .await;

    use sqlx::Row;
    let active_v: Vec<Value> = active
        .unwrap_or_default()
        .into_iter()
        .map(|r| {
            json!({
                "username": r.try_get::<String, _>("username").ok(),
                "name":     r.try_get::<Option<String>, _>("name").ok().flatten(),
                "vip":      r.try_get::<Option<bool>, _>("vip").ok().flatten(),
                "stage":    r.try_get::<Option<String>, _>("stage").ok().flatten(),
                "started_at":      r.try_get::<Option<String>, _>("started_at").ok().flatten(),
                "running_for_secs": r.try_get::<Option<f64>, _>("running_for_secs").ok().flatten(),
                "new_documents":    r.try_get::<Option<i64>, _>("new_documents").ok().flatten(),
            })
        })
        .collect();

    // Recent finished runs. The indexer's process is the only thing
    // that produces rows with stage='index' completed and trigger
    // matching the python pipeline; cap at 20.
    let recent = sqlx::query(
        r#"
        SELECT r.id,
               r.status,
               r.stage,
               to_char(r.started_at,  'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS started_at,
               to_char(r.finished_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS finished_at,
               r.duration_secs::float8 AS duration_secs,
               r.new_documents::bigint AS new_documents,
               r.error,
               u.username,
               u.name,
               u.vip
          FROM pipeline_runs r
          JOIN users u ON u.id = r.user_id
         WHERE r.status IN ('success', 'failed')
         ORDER BY r.started_at DESC
         LIMIT 20
        "#,
    )
    .fetch_all(&pool)
    .await;

    let recent_v: Vec<Value> = recent
        .unwrap_or_default()
        .into_iter()
        .map(|r| {
            let new_docs = r.try_get::<Option<i64>, _>("new_documents").ok().flatten();
            let status = r.try_get::<Option<String>, _>("status").ok().flatten();
            // "kind" gives the UI a one-word label without forcing
            // the frontend to re-derive the success/zero-new split:
            //   updated  — succeeded AND added rows to the index
            //   cleaned  — succeeded with 0 new docs (heal/re-embed
            //              of existing content)
            //   failed   — pipeline aborted
            let kind = match (status.as_deref(), new_docs.unwrap_or(0)) {
                (Some("success"), n) if n > 0 => "updated",
                (Some("success"), _) => "cleaned",
                (Some("failed"), _) => "failed",
                _ => "other",
            };
            json!({
                "id":             r.try_get::<i64, _>("id").ok(),
                "status":         status,
                "kind":           kind,
                "stage":          r.try_get::<Option<String>, _>("stage").ok().flatten(),
                "started_at":     r.try_get::<Option<String>, _>("started_at").ok().flatten(),
                "finished_at":    r.try_get::<Option<String>, _>("finished_at").ok().flatten(),
                "duration_secs":  r.try_get::<Option<f64>, _>("duration_secs").ok().flatten(),
                "new_documents":  new_docs,
                "error":          r.try_get::<Option<String>, _>("error").ok().flatten(),
                "username":       r.try_get::<String, _>("username").ok(),
                "name":           r.try_get::<Option<String>, _>("name").ok().flatten(),
                "vip":            r.try_get::<Option<bool>, _>("vip").ok().flatten(),
            })
        })
        .collect();

    Json(json!({ "active": active_v, "recent": recent_v })).into_response()
}

// ── /api/admin/twitter-feed ──────────────────────────────────────────
//
// Health surface for the launchd-managed twitter feeder running on
// the operator's Mac. The feeder POSTs heartbeats to
//   POST /api/admin/twitter-feed/heartbeat
// authenticated with the `KNOWLEDGE_ADMIN_TOKEN` shared secret
// (X-Admin-Token header) — it doesn't carry a session cookie. The
// admin panel reads back the current row via
//   GET  /api/admin/twitter-feed/status
// behind the usual session-cookie admin guard.
//
// Storage is a single row (see sources/sql/twitter_feed_status.sql)
// — old state is overwritten on every heartbeat. The goal is "is
// the daemon running right now", not a history.

#[derive(Deserialize)]
pub struct TwitterFeedHeartbeat {
    /// Client-side state machine: starting | running | idle |
    /// sleeping | error | unknown.
    pub state: String,
    /// Personality currently being processed (running state).
    pub current_slug: Option<String>,
    pub current_handle: Option<String>,
    /// Pass progress so far.
    pub pass_processed: Option<i32>,
    pub pass_total: Option<i32>,
    /// Optional wall-clock for the active pass.
    pub pass_started_at: Option<String>,
    pub pass_finished_at: Option<String>,
    /// Set when the feeder just finished a pass — accumulates so we
    /// can spot "it ran once then died" vs "running normally".
    pub pass_completed: Option<bool>,
    /// Free-text error, truncated server-side.
    pub last_error: Option<String>,
}

/// `POST /api/admin/twitter-feed/heartbeat`
///
/// Auth: shared-secret token in `X-Admin-Token` header, matched
/// against `KNOWLEDGE_ADMIN_TOKEN` from the server env. If the env
/// var isn't set the endpoint refuses every request (fail-closed) —
/// no silent "auth disabled" mode.
pub async fn twitter_feed_heartbeat(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<TwitterFeedHeartbeat>,
) -> Response {
    let expected = match std::env::var("KNOWLEDGE_ADMIN_TOKEN") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "KNOWLEDGE_ADMIN_TOKEN not configured",
            )
                .into_response();
        }
    };
    let provided = headers
        .get("x-admin-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    // Constant-time compare so a probe can't recover the token by
    // measuring response latency.
    let ok = provided.len() == expected.len()
        && provided
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0;
    if !ok {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let pool = match pg(&state) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let last_error = body
        .last_error
        .as_deref()
        .map(|s| s.chars().take(500).collect::<String>());

    let pass_completed = body.pass_completed.unwrap_or(false);

    // Upsert the single sentinel row. RFC3339 strings are cast to
    // timestamptz inside the SQL so we don't need sqlx's chrono
    // feature (the rest of the codebase reads timestamps back as
    // strings via to_char). The COALESCE pattern lets a partial
    // heartbeat (e.g. mid-pass slug ping) update only the fields it
    // carries, while a full pass-end heartbeat overwrites
    // everything. last_error_at is server-stamped when the
    // heartbeat carries a non-null error.
    let sql = r#"
        INSERT INTO twitter_feed_status (
            id, heartbeat_at, state,
            current_slug, current_handle,
            pass_processed, pass_total,
            pass_started_at, pass_finished_at,
            last_error, last_error_at,
            pass_count, updated_at
        ) VALUES (
            1, NOW(), $1,
            $2, $3,
            COALESCE($4, 0), COALESCE($5, 0),
            NULLIF($6, '')::timestamptz, NULLIF($7, '')::timestamptz,
            $8,
            CASE WHEN $8 IS NULL THEN NULL ELSE NOW() END,
            CASE WHEN $9 THEN 1 ELSE 0 END, NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            heartbeat_at = NOW(),
            state        = EXCLUDED.state,
            current_slug   = COALESCE(EXCLUDED.current_slug,   twitter_feed_status.current_slug),
            current_handle = COALESCE(EXCLUDED.current_handle, twitter_feed_status.current_handle),
            pass_processed = COALESCE(EXCLUDED.pass_processed, twitter_feed_status.pass_processed),
            pass_total     = COALESCE(EXCLUDED.pass_total,     twitter_feed_status.pass_total),
            pass_started_at  = COALESCE(EXCLUDED.pass_started_at,  twitter_feed_status.pass_started_at),
            pass_finished_at = COALESCE(EXCLUDED.pass_finished_at, twitter_feed_status.pass_finished_at),
            last_error    = COALESCE(EXCLUDED.last_error,    twitter_feed_status.last_error),
            last_error_at = COALESCE(EXCLUDED.last_error_at, twitter_feed_status.last_error_at),
            pass_count = twitter_feed_status.pass_count + CASE WHEN $9 THEN 1 ELSE 0 END,
            updated_at = NOW()
    "#;

    let res = sqlx::query(sql)
        .bind(&body.state)
        .bind(body.current_slug.as_deref())
        .bind(body.current_handle.as_deref())
        .bind(body.pass_processed)
        .bind(body.pass_total)
        .bind(body.pass_started_at.as_deref().unwrap_or(""))
        .bind(body.pass_finished_at.as_deref().unwrap_or(""))
        .bind(last_error.as_deref())
        .bind(pass_completed)
        .execute(&pool)
        .await;
    match res {
        Ok(_) => (StatusCode::NO_CONTENT, "").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "twitter_feed.heartbeat.failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "write failed").into_response()
        }
    }
}

// ── /api/admin/tweets/ingest ────────────────────────────────────────
//
// Replaces the direct PG-via-SSH-tunnel writes the local twitter
// feeder used to do. Two reasons to route through the API instead:
//
//   1. The feeder needs to know WHICH urls were actually new vs
//      already-stored so it can bail early when a page produces
//      zero inserts (lex hasn't tweeted in a week → page 1 is
//      already entirely in PG → no point paginating 30 more pages
//      of known history). RETURNING (xmax = 0) gives us that signal
//      per row; the old `upsert_documents()` callable didn't expose
//      it.
//   2. One less direct-DB consumer simplifies operational surface —
//      auth, schema migrations, write-shape evolution all stay
//      behind the Rust layer.

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct IngestDoc {
    pub url: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub source: String,
    pub source_url: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub extra_tags: Vec<String>,
    #[serde(default)]
    pub linked_urls: Option<serde_json::Value>,
    #[serde(default)]
    pub link_hosts: Vec<String>,
    // Behavioural / engagement metrics. `None` = the feeder didn't
    // measure this signal for this doc; the upsert below COALESCEs
    // onto the prior value so a re-sync that happens to ship without
    // engagement (e.g. a malformed twikit payload) never resets a
    // tweet's like count back to zero.
    #[serde(default)]
    pub twitter_likes: Option<i64>,
    #[serde(default)]
    pub twitter_retweets: Option<i64>,
    #[serde(default)]
    pub twitter_replies: Option<i64>,
    #[serde(default)]
    pub twitter_quotes: Option<i64>,
    #[serde(default)]
    pub twitter_views: Option<i64>,
    #[serde(default)]
    pub twitter_bookmarks: Option<i64>,
}

#[derive(Debug, serde::Deserialize)]
pub struct IngestRequest {
    pub slug: String,
    pub documents: Vec<IngestDoc>,
}

#[derive(Debug, serde::Serialize)]
pub struct IngestResponse {
    pub n_inserted: usize,
    pub n_existed: usize,
    pub inserted: Vec<String>,
    pub existed: Vec<String>,
}

/// `POST /api/admin/tweets/ingest`
/// Auth: shared-secret `X-Admin-Token` header.
pub async fn admin_ingest_tweets(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<IngestRequest>,
) -> Response {
    // Reuse the same fail-closed admin-token check as the heartbeat.
    let expected = match std::env::var("KNOWLEDGE_ADMIN_TOKEN") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "KNOWLEDGE_ADMIN_TOKEN not configured",
            )
                .into_response();
        }
    };
    let provided = headers
        .get("x-admin-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ok = provided.len() == expected.len()
        && provided
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0;
    if !ok {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let pool = match pg(&state) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if body.documents.is_empty() {
        return Json(IngestResponse {
            n_inserted: 0,
            n_existed: 0,
            inserted: Vec::new(),
            existed: Vec::new(),
        })
        .into_response();
    }
    // Resolve slug → user_id once. The feeder always targets a single
    // personality per call so we don't need to mix users.
    let user_id: i64 =
        match sqlx::query_scalar::<_, i64>("SELECT id FROM users WHERE username = $1")
            .bind(&body.slug)
            .fetch_optional(&pool)
            .await
        {
            Ok(Some(id)) => id,
            Ok(None) => return (StatusCode::NOT_FOUND, "user not found").into_response(),
            Err(e) => {
                tracing::error!(error = %e, "admin_ingest.user_lookup.failed");
                return (StatusCode::INTERNAL_SERVER_ERROR, "db error").into_response();
            }
        };

    // Build the docs as a jsonb payload for `jsonb_to_recordset`. One
    // round-trip + one `RETURNING` set is far cheaper than per-row
    // INSERTs in a loop, and keeps the ordering deterministic.
    let payload = serde_json::to_value(&body.documents).unwrap_or(serde_json::Value::Null);

    // ON CONFLICT semantics mirror the bulk_save_documents handler
    // (which the browser-side sync hits): keep existing rich values
    // when the new payload ships empties, GREATEST the date so a
    // newly-discovered tweet bumps stale rows to its own date, clear
    // created_via_favorite (a real sync just confirmed the row), and
    // resurrect soft-deleted rows.
    let sql = r#"
        WITH input AS (
            SELECT *
              FROM jsonb_to_recordset($2::jsonb) AS x(
                  url               text,
                  title             text,
                  summary           text,
                  date              text,
                  source            text,
                  source_url        text,
                  tags              text[],
                  extra_tags        text[],
                  linked_urls       jsonb,
                  link_hosts        text[],
                  twitter_likes     bigint,
                  twitter_retweets  bigint,
                  twitter_replies   bigint,
                  twitter_quotes    bigint,
                  twitter_views     bigint,
                  twitter_bookmarks bigint
              )
        )
        INSERT INTO documents (
            user_id, url, title, summary, date, tags, extra_tags,
            source, source_url, linked_urls, link_hosts,
            twitter_likes, twitter_retweets, twitter_replies,
            twitter_quotes, twitter_views, twitter_bookmarks,
            engagement_updated_at
        )
        SELECT $1, i.url, COALESCE(i.title, ''), COALESCE(i.summary, ''),
               NULLIF(i.date, '')::date,
               COALESCE(i.tags, '{}'::text[]),
               COALESCE(i.extra_tags, '{}'::text[]),
               COALESCE(i.source, ''),
               i.source_url,
               COALESCE(i.linked_urls, '[]'::jsonb),
               COALESCE(i.link_hosts, '{}'::text[]),
               -- BIGINT on the wire → INT4 on disk for the non-view
               -- columns. PG will narrow the cast; values above 2.1B
               -- here would mean a tweet hit 2 billion likes, which
               -- isn't a real concern.
               i.twitter_likes::int,
               i.twitter_retweets::int,
               i.twitter_replies::int,
               i.twitter_quotes::int,
               i.twitter_views,
               i.twitter_bookmarks::int,
               CASE WHEN i.twitter_likes IS NOT NULL
                      OR i.twitter_retweets IS NOT NULL
                      OR i.twitter_replies IS NOT NULL
                      OR i.twitter_quotes IS NOT NULL
                      OR i.twitter_views IS NOT NULL
                      OR i.twitter_bookmarks IS NOT NULL
                    THEN now() ELSE NULL END
          FROM input i
        ON CONFLICT (user_id, url) DO UPDATE
            SET date = GREATEST(documents.date, EXCLUDED.date),
                title = CASE WHEN EXCLUDED.title <> ''
                              THEN EXCLUDED.title
                              ELSE documents.title END,
                summary = CASE WHEN EXCLUDED.summary <> ''
                                THEN EXCLUDED.summary
                                ELSE documents.summary END,
                tags = CASE WHEN cardinality(EXCLUDED.tags) > 0
                              THEN EXCLUDED.tags
                              ELSE documents.tags END,
                extra_tags = CASE WHEN cardinality(EXCLUDED.extra_tags) > 0
                                    THEN EXCLUDED.extra_tags
                                    ELSE documents.extra_tags END,
                source = CASE WHEN EXCLUDED.source <> ''
                                THEN EXCLUDED.source
                                ELSE documents.source END,
                source_url = COALESCE(EXCLUDED.source_url, documents.source_url),
                linked_urls = CASE
                    WHEN jsonb_array_length(EXCLUDED.linked_urls) > 0
                        THEN EXCLUDED.linked_urls
                    ELSE documents.linked_urls
                END,
                link_hosts = CASE
                    WHEN cardinality(EXCLUDED.link_hosts) > 0
                        THEN EXCLUDED.link_hosts
                    ELSE documents.link_hosts
                END,
                -- Engagement: COALESCE so a payload that ships NULLs
                -- (e.g. a non-engagement re-sync) keeps the prior
                -- measurement intact. A real measurement always wins
                -- because EXCLUDED is non-NULL only when it was set.
                twitter_likes     = COALESCE(EXCLUDED.twitter_likes,     documents.twitter_likes),
                twitter_retweets  = COALESCE(EXCLUDED.twitter_retweets,  documents.twitter_retweets),
                twitter_replies   = COALESCE(EXCLUDED.twitter_replies,   documents.twitter_replies),
                twitter_quotes    = COALESCE(EXCLUDED.twitter_quotes,    documents.twitter_quotes),
                twitter_views     = COALESCE(EXCLUDED.twitter_views,     documents.twitter_views),
                twitter_bookmarks = COALESCE(EXCLUDED.twitter_bookmarks, documents.twitter_bookmarks),
                engagement_updated_at = COALESCE(EXCLUDED.engagement_updated_at, documents.engagement_updated_at),
                created_via_favorite = FALSE,
                deleted = FALSE,
                updated_at = now()
        RETURNING url, (xmax = 0) AS inserted
    "#;

    #[allow(clippy::type_complexity)]
    let rows: Vec<(String, bool)> = match sqlx::query_as(sql)
        .bind(user_id)
        .bind(&payload)
        .fetch_all(&pool)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "admin_ingest.upsert.failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}")).into_response();
        }
    };
    let mut inserted = Vec::new();
    let mut existed = Vec::new();
    for (url, is_insert) in rows {
        if is_insert {
            inserted.push(url);
        } else {
            existed.push(url);
        }
    }
    Json(IngestResponse {
        n_inserted: inserted.len(),
        n_existed: existed.len(),
        inserted,
        existed,
    })
    .into_response()
}

// ── /api/admin/twitter-queue ────────────────────────────────────────
//
// Returns the same VIP-with-twitter ordering the local feeder used
// to compute via direct PG (`_vips_by_staleness`). Moving this read
// behind the API kills the last reason the feeder needed the SSH
// tunnel to prod — every page write was already going through the
// ingest endpoint, but the queue + existing-URL reads were still
// touching PG via the tunnel.

#[derive(Debug, serde::Deserialize)]
pub struct TwitterQueueParams {
    #[serde(default)]
    pub min_age_hours: f64,
}

#[derive(Debug, serde::Serialize)]
pub struct TwitterQueueEntry {
    pub user_id: i64,
    pub slug: String,
    pub handle: String,
    pub last_touch: Option<String>,
    pub twitter_followers: i64,
}

#[allow(clippy::result_large_err)]
fn check_admin_token(headers: &axum::http::HeaderMap) -> Result<(), Response> {
    let expected = match std::env::var("KNOWLEDGE_ADMIN_TOKEN") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "KNOWLEDGE_ADMIN_TOKEN not configured",
            )
                .into_response());
        }
    };
    let provided = headers
        .get("x-admin-token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let ok = provided.len() == expected.len()
        && provided
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0;
    if !ok {
        return Err(StatusCode::UNAUTHORIZED.into_response());
    }
    Ok(())
}

/// `GET /api/admin/twitter-queue?min_age_hours=N`
/// Auth: shared-secret `X-Admin-Token` header.
pub async fn admin_twitter_queue(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(params): axum::extract::Query<TwitterQueueParams>,
) -> Response {
    if let Err(e) = check_admin_token(&headers) {
        return e;
    }
    let pool = match pg(&state) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Order rules (per-row, applied in this priority):
    //
    //   1. Cooldown: rows with `consecutive_failures >= 1` are hidden
    //      until `last_attempt_at + LEAST(30d, 24h × 2^failures)`. So
    //      a deleted/locked account hit once is held off for 24h,
    //      twice for 48h, … capped at 30 days. Resets the moment the
    //      next ok/up_to_date attempt lands.
    //
    //   2. Today-attempted demote: a slug whose `last_attempt_at` is
    //      today (UTC date) goes to the END of the queue. So on a
    //      mid-day restart the feeder skips past everyone we just
    //      touched and gets to the people we haven't tried yet.
    //
    //   3. Within the remaining tier: popularity DESC (high
    //      twitter_followers first), then never-attempted, then
    //      oldest-attempt-first, alphabetical tiebreaker.
    //
    // The `min_age_hours` param is kept as a coarse filter — set it
    // to 0 to get every eligible slug, or to a positive value to
    // require last_attempt_at to be at least that old (skipping
    // every slug touched within the window).
    let sql = r#"
        SELECT u.id, u.username,
               COALESCE(u.sources->'twitter'->>'username', '') AS handle,
               to_char(ta.last_attempt_at AT TIME ZONE 'UTC',
                       'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS last_attempt,
               COALESCE(u.twitter_followers, 0)::bigint AS twitter_followers
          FROM users u
          LEFT JOIN twitter_feed_attempts ta ON ta.user_id = u.id
         WHERE u.vip = TRUE
           AND u.sources ? 'twitter'
           -- Cooldown for known-broken accounts. `LEAST(...)` caps
           -- the backoff at 30 days; the inner POWER doubles on each
           -- consecutive failure (24h, 48h, 96h, 192h, …).
           AND (
                 ta.user_id IS NULL
              OR ta.consecutive_failures = 0
              OR ta.last_attempt_at < NOW() - LEAST(
                     INTERVAL '30 days',
                     POWER(2, LEAST(ta.consecutive_failures, 10))::float
                       * INTERVAL '24 hours'
                 )
           )
           -- Optional coarse age filter (kept for backward compat
           -- with the --min-age CLI flag).
           AND ($1::float8 <= 0
                OR ta.last_attempt_at IS NULL
                OR ta.last_attempt_at < NOW() - ($1::float8 * INTERVAL '1 hour'))
         ORDER BY
                  -- Today-attempted go LAST.
                  COALESCE(
                      DATE(ta.last_attempt_at AT TIME ZONE 'UTC')
                          = (NOW() AT TIME ZONE 'UTC')::date,
                      FALSE
                  ) ASC,
                  -- Among remaining: popular first.
                  COALESCE(u.twitter_followers, 0) DESC,
                  -- Then never-attempted (NULL) before stalest.
                  ta.last_attempt_at ASC NULLS FIRST,
                  u.username
    "#;
    #[allow(clippy::type_complexity)]
    let rows: Vec<(i64, String, String, Option<String>, i64)> = match sqlx::query_as(sql)
        .bind(params.min_age_hours)
        .fetch_all(&pool)
        .await
    {
        Ok(rs) => rs,
        Err(e) => {
            tracing::error!(error = %e, "admin_twitter_queue.failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "db error").into_response();
        }
    };
    let out: Vec<TwitterQueueEntry> = rows
        .into_iter()
        .filter(|(_, _, handle, _, _)| !handle.is_empty())
        .map(
            |(user_id, slug, handle, last_touch, twitter_followers)| TwitterQueueEntry {
                user_id,
                slug,
                handle,
                last_touch,
                twitter_followers,
            },
        )
        .collect();
    Json(out).into_response()
}

// ── /api/admin/twitter-feed/attempt ────────────────────────────────
//
// Called by the feeder at the end of every slug. UPSERTs the row in
// `twitter_feed_attempts`. The handler resolves the success vs
// failure path from the supplied `status` string:
//
//   • 'ok' | 'up_to_date' — reset consecutive_failures to 0
//   • everything else      — increment consecutive_failures
//
// `last_attempt_at` is always stamped to now() — that's how the queue
// endpoint above demotes today-attempted slugs and decides which
// broken accounts are still in cooldown.

#[derive(Debug, serde::Deserialize)]
pub struct TwitterAttemptBody {
    pub user_id: i64,
    pub status: String,
}

pub async fn admin_twitter_feed_attempt(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<TwitterAttemptBody>,
) -> Response {
    if let Err(e) = check_admin_token(&headers) {
        return e;
    }
    let pool = match pg(&state) {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Whitelist the statuses so a typo from the client doesn't bypass
    // the failure-counting branch. Anything outside this set is
    // accepted but routed through the increment path (treated as an
    // error). Keep the list aligned with the feeder's call sites in
    // `clients/twitter_feeder.py:_one_pass`.
    let is_success = matches!(body.status.as_str(), "ok" | "up_to_date");
    let sql = r#"
        INSERT INTO twitter_feed_attempts
                    (user_id, last_attempt_at, last_status, consecutive_failures)
             VALUES ($1, now(), $2, CASE WHEN $3::boolean THEN 0 ELSE 1 END)
        ON CONFLICT (user_id) DO UPDATE
              SET last_attempt_at      = now(),
                  last_status          = EXCLUDED.last_status,
                  consecutive_failures = CASE
                      WHEN $3::boolean THEN 0
                      ELSE twitter_feed_attempts.consecutive_failures + 1
                  END
    "#;
    match sqlx::query(sql)
        .bind(body.user_id)
        .bind(&body.status)
        .bind(is_success)
        .execute(&pool)
        .await
    {
        Ok(_) => (StatusCode::NO_CONTENT, "").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "admin_twitter_feed_attempt.failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "db error").into_response()
        }
    }
}

// ── /api/admin/users/{slug}/twitter-urls ────────────────────────────
//
// Returns every twitter URL we already have stored for `{slug}`. The
// feeder feeds this set into the Bookmarks fetcher's `existing_urls`
// param so the paginator can early-stop the moment a whole page
// returns nothing new.

pub async fn admin_user_twitter_urls(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Path(slug): axum::extract::Path<String>,
) -> Response {
    if let Err(e) = check_admin_token(&headers) {
        return e;
    }
    let pool = match pg(&state) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let rows: Vec<(String,)> = match sqlx::query_as(
        r#"
        SELECT d.url
          FROM documents d
          JOIN users u ON u.id = d.user_id
         WHERE u.username = $1
           AND d.source = 'twitter'
           AND d.deleted = FALSE
        "#,
    )
    .bind(&slug)
    .fetch_all(&pool)
    .await
    {
        Ok(rs) => rs,
        Err(e) => {
            tracing::error!(error = %e, "admin_user_twitter_urls.failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "db error").into_response();
        }
    };
    let urls: Vec<String> = rows.into_iter().map(|(u,)| u).collect();
    Json(urls).into_response()
}

/// `GET /api/admin/twitter-feed/status` — current row, admin-only.
#[allow(clippy::type_complexity)]
pub async fn twitter_feed_status(State(state): State<Arc<AppState>>, jar: CookieJar) -> Response {
    let pool = match require_raphael(&state, &jar).await {
        Ok(p) => p,
        Err(e) => return e,
    };
    // Pull everything as strings — the project already round-trips
    // timestamps as RFC3339 text through every other admin
    // endpoint (sqlx isn't built with the chrono feature).
    let row: Result<
        (
            String,         // heartbeat_at
            i64,            // age in seconds (server-computed)
            String,         // state
            Option<String>, // pass_started_at
            Option<String>, // pass_finished_at
            i32,            // pass_processed
            i32,            // pass_total
            Option<String>, // current_slug
            Option<String>, // current_handle
            Option<String>, // last_error
            Option<String>, // last_error_at
            i32,            // pass_count
        ),
        _,
    > = sqlx::query_as(
        r#"
        SELECT
            to_char(heartbeat_at AT TIME ZONE 'UTC',
                    'YYYY-MM-DD"T"HH24:MI:SS"Z"')               AS heartbeat_at,
            EXTRACT(EPOCH FROM (NOW() - heartbeat_at))::bigint  AS age_secs,
            state,
            to_char(pass_started_at AT TIME ZONE 'UTC',
                    'YYYY-MM-DD"T"HH24:MI:SS"Z"')               AS pass_started_at,
            to_char(pass_finished_at AT TIME ZONE 'UTC',
                    'YYYY-MM-DD"T"HH24:MI:SS"Z"')               AS pass_finished_at,
            pass_processed,
            pass_total,
            current_slug,
            current_handle,
            last_error,
            to_char(last_error_at AT TIME ZONE 'UTC',
                    'YYYY-MM-DD"T"HH24:MI:SS"Z"')               AS last_error_at,
            pass_count
          FROM twitter_feed_status
         WHERE id = 1
        "#,
    )
    .fetch_one(&pool)
    .await;
    match row {
        Ok((
            heartbeat_at,
            age_secs,
            st,
            ps,
            pf,
            processed,
            total,
            slug,
            handle,
            err,
            err_at,
            pass_count,
        )) => {
            // > 15 minutes since last heartbeat → mark as stale so
            // the panel can flag it red even if the recorded state
            // still says "running".
            let stale = age_secs > 15 * 60;
            Json(json!({
                "heartbeat_at": heartbeat_at,
                "heartbeat_age_secs": age_secs,
                "stale": stale,
                "state": st,
                "pass_started_at": ps,
                "pass_finished_at": pf,
                "pass_processed": processed,
                "pass_total": total,
                "current_slug": slug,
                "current_handle": handle,
                "last_error": err,
                "last_error_at": err_at,
                "pass_count": pass_count,
            }))
            .into_response()
        }
        Err(_) => Json(json!({
            "heartbeat_at": Value::Null,
            "heartbeat_age_secs": Value::Null,
            "stale": true,
            "state": "unknown",
            "pass_processed": 0,
            "pass_total": 0,
            "pass_count": 0,
        }))
        .into_response(),
    }
}
