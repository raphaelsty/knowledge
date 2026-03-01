//! Pipeline execution — trigger `run.py` from the API.
//!
//! - `POST /api/pipeline` starts the Python pipeline as a background process.
//! - `GET  /api/pipeline` returns the current status, with live output while running.
//!
//! Output is captured line-by-line so the frontend can poll for incremental logs.
//! Only one pipeline run is allowed at a time.

use std::sync::Arc;

use axum::{extract::State, response::Json};
use serde::Serialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

// ─── Shared state ──────────────────────────────────────────────────

pub struct PipelineState {
    running: bool,
    started: Option<(std::time::Instant, String)>,
    lines: Vec<String>,
    last_run: Option<RunResult>,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            running: false,
            started: None,
            lines: Vec::new(),
            last_run: None,
        }
    }
}

#[derive(Clone, Serialize)]
pub struct RunResult {
    pub success: bool,
    pub started_at: String,
    pub finished_at: String,
    pub duration_secs: f64,
    pub output: String,
}

pub type SharedPipeline = Arc<Mutex<PipelineState>>;

pub fn new_state() -> SharedPipeline {
    Arc::new(Mutex::new(PipelineState::default()))
}

// ─── Handlers ──────────────────────────────────────────────────────

/// GET /api/pipeline — returns status + live output lines (even while running).
pub async fn status(State(state): State<SharedPipeline>) -> Json<serde_json::Value> {
    let inner = state.lock().await;

    if inner.running {
        let (instant, timestamp) = inner.started.as_ref().unwrap();
        Json(json!({
            "status": "running",
            "started_at": timestamp,
            "elapsed_secs": instant.elapsed().as_secs_f64(),
            "output": inner.lines.join("\n"),
        }))
    } else {
        Json(json!({
            "status": "idle",
            "last_run": inner.last_run,
        }))
    }
}

/// POST /api/pipeline — start the pipeline. Returns immediately.
pub async fn trigger(State(state): State<SharedPipeline>) -> Json<serde_json::Value> {
    let mut inner = state.lock().await;

    if inner.running {
        let started_at = inner.started.as_ref().map(|(_, s)| s.as_str()).unwrap_or("");
        return Json(json!({
            "status": "already_running",
            "started_at": started_at,
        }));
    }

    let now = chrono::Utc::now().to_rfc3339();
    inner.running = true;
    inner.started = Some((std::time::Instant::now(), now.clone()));
    inner.lines.clear();
    drop(inner);

    let state2 = Arc::clone(&state);
    tokio::spawn(async move {
        let result = run_pipeline(&state2).await;
        if result.success {
            tracing::info!(duration_secs = result.duration_secs, "pipeline.completed");
        } else {
            tracing::error!(duration_secs = result.duration_secs, "pipeline.failed");
        }
        let mut inner = state2.lock().await;
        inner.running = false;
        inner.last_run = Some(result);
        inner.started = None;
    });

    Json(json!({
        "status": "started",
        "started_at": now,
    }))
}

// ─── Subprocess with line-by-line capture ──────────────────────────

async fn run_pipeline(state: &SharedPipeline) -> RunResult {
    let started = std::time::Instant::now();
    let started_at = chrono::Utc::now().to_rfc3339();

    tracing::info!("pipeline.starting");

    let buffer_dir = std::env::var("BUFFER_DIR").unwrap_or_else(|_| "buffer".into());

    let child = tokio::process::Command::new("uv")
        .args(["run", "python", "-u", "run.py"])
        .env("BUFFER_DIR", &buffer_dir)
        .env("PYTHONUNBUFFERED", "1")
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("Failed to start pipeline: {e}");
            state.lock().await.lines.push(msg.clone());
            return RunResult {
                success: false,
                started_at,
                finished_at: chrono::Utc::now().to_rfc3339(),
                duration_secs: started.elapsed().as_secs_f64(),
                output: msg,
            };
        }
    };

    // Read stdout and stderr line-by-line, appending to shared state
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let state_out = Arc::clone(state);
    let stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            state_out.lock().await.lines.push(line);
        }
    });

    let state_err = Arc::clone(state);
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            state_err.lock().await.lines.push(line);
        }
    });

    let exit_status = child.wait().await;
    // Give reader tasks a moment to drain remaining output, then abort if stuck
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), stdout_task).await;
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), stderr_task).await;

    let finished_at = chrono::Utc::now().to_rfc3339();
    let duration_secs = started.elapsed().as_secs_f64();
    let output = state.lock().await.lines.join("\n");
    let success = exit_status.map(|s| s.success()).unwrap_or(false);

    RunResult {
        success,
        started_at,
        finished_at,
        duration_secs,
        output,
    }
}
