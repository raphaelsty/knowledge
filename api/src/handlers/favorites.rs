//! Favorites — per-user starred personalities.
//!
//!   GET    /auth/favorites          → list of slugs the user has starred
//!   PUT    /auth/favorites/{slug}   → add
//!   DELETE /auth/favorites/{slug}   → remove
//!
//! All endpoints require an authenticated session (reads the cookie via
//! `current_user`) and enforce the user-can't-favorite-themselves rule
//! at the DB level via the CHECK on `favorites`.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use sqlx::PgPool;

use crate::handlers::auth::current_user;

/// GET /auth/favorites
///
/// Returns `[ "slug1", "slug2", ... ]` ordered by when each was starred,
/// most recent first. Empty when signed in but no favorites.
pub async fn list(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT u.username
           FROM favorites f
           JOIN users u ON u.id = f.favorite_id
          WHERE f.user_id = $1
          ORDER BY f.created_at DESC",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let slugs: Vec<String> = rows.into_iter().map(|r| r.0).collect();
    Json(slugs).into_response()
}

/// PUT /auth/favorites/{slug}
pub async fn add(State(pool): State<PgPool>, jar: CookieJar, Path(slug): Path<String>) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    // Resolve the slug to an id, then insert. One round-trip via CTE so
    // a missing slug surfaces as 404 instead of a FK violation.
    let result = sqlx::query(
        "INSERT INTO favorites (user_id, favorite_id)
         SELECT $1, u.id
           FROM users u
          WHERE u.username = $2
         ON CONFLICT DO NOTHING",
    )
    .bind(me.id)
    .bind(&slug)
    .execute(&pool)
    .await;

    match result {
        Ok(r) if r.rows_affected() == 1 => StatusCode::NO_CONTENT.into_response(),
        Ok(_) => {
            // Either the slug doesn't exist, the user tried to favorite
            // themselves (CHECK violation → 0 rows because `ON CONFLICT`
            // doesn't see CHECK, but INSERT…SELECT emits 0 when the
            // SELECT returns no rows), or the row already existed.
            // Disambiguate quickly: is there a user with that slug?
            let exists: Option<i64> =
                sqlx::query_scalar("SELECT id FROM users WHERE username = $1")
                    .bind(&slug)
                    .fetch_optional(&pool)
                    .await
                    .unwrap_or(None);
            match exists {
                Some(id) if id == me.id => {
                    (StatusCode::BAD_REQUEST, "cannot favorite yourself").into_response()
                }
                Some(_) => StatusCode::NO_CONTENT.into_response(), // already starred
                None => StatusCode::NOT_FOUND.into_response(),
            }
        }
        Err(e) => {
            tracing::error!(error = %e, "favorites.add.failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("favorite failed: {e}"),
            )
                .into_response()
        }
    }
}

/// DELETE /auth/favorites/{slug}
pub async fn remove(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Path(slug): Path<String>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let _ = sqlx::query(
        "DELETE FROM favorites
           WHERE user_id = $1
             AND favorite_id = (SELECT id FROM users WHERE username = $2)",
    )
    .bind(me.id)
    .bind(&slug)
    .execute(&pool)
    .await;
    StatusCode::NO_CONTENT.into_response()
}
