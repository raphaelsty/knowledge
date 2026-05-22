//! Credit-billing endpoints.
//!
//! Routes (registered in `main.rs`):
//!   GET  /api/me/credits           — current balance + recent ledger
//!   POST /api/credits/checkout     — start a Polar-hosted checkout,
//!                                    returns the URL to redirect to
//!   POST /api/credits/webhook      — Polar webhook receiver (no auth)
//!
//! All credit movements go through the SQL helper functions
//! `credits_top_up()` / `credits_debit()` defined in
//! `sources/sql/credits.sql`. Those functions own the locking +
//! invariants; this module never touches `credit_events` directly.
//!
//! v1 ships the schema + the buy-credits flow end-to-end. No
//! operations debit credits yet — the debit pathway lives here too,
//! ready to be called from `pipeline_runs.rs` / the Twitter fetcher
//! / etc. as soon as you're ready to flip on billing.

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::handlers::auth::current_user;
use crate::handlers::polar;

// ── Pack catalogue ──────────────────────────────────────────────────
//
// Polar product ids + their credit grants live in env so the
// operator can change pricing without redeploying the API. Each
// entry maps an internal `id` (used by the frontend to pick a pack)
// to:
//   - the Polar product id (used at checkout)
//   - the displayed price (in USD cents)
//   - the credit grant (either fixed or per-cent for custom amount)
//
// Env keys:
//   POLAR_PACK_<ID>_PRODUCT     — Polar product id
//   POLAR_PACK_<ID>_PRICE_CENTS — display price for fixed packs
//   POLAR_PACK_<ID>_CREDITS     — fixed credit grant
// Custom-amount packs use:
//   POLAR_PACK_<ID>_MIN_CENTS   — min user-entered amount
//   POLAR_PACK_<ID>_MAX_CENTS   — max user-entered amount
//   POLAR_PACK_<ID>_CREDITS_PER_CENT — multiplier
//
// We keep the catalogue order stable (starter, small, medium, large,
// custom) by hard-coding the slugs here rather than enumerating env.

#[derive(Serialize)]
pub struct Pack {
    pub id: String,
    pub label: String,
    /// "fixed" | "custom"
    pub kind: String,
    #[serde(rename = "productId")]
    pub product_id: String,
    /// Currencies the product can be paid in. The actual currency the
    /// user pays in is decided at checkout by Polar's Localized
    /// Checkout based on their region — this is the display hint so
    /// the frontend can render "$1 / €1" instead of one-of.
    pub currencies: Vec<String>,
    /// Display price in minor units for fixed packs. Same numeric
    /// value in both USD cents and EUR cents — we don't track a
    /// per-currency price because the catalogue uses identical
    /// integer amounts in both. None when the pack is custom-amount.
    #[serde(rename = "priceCents", skip_serializing_if = "Option::is_none")]
    pub price_cents: Option<i64>,
    /// Credit grant for fixed packs. None when custom-amount.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credits: Option<i64>,
    /// Custom-amount range (minor units).
    #[serde(rename = "minCents", skip_serializing_if = "Option::is_none")]
    pub min_cents: Option<i64>,
    #[serde(rename = "maxCents", skip_serializing_if = "Option::is_none")]
    pub max_cents: Option<i64>,
    /// Credits granted per cent paid (custom-amount packs only).
    #[serde(rename = "creditsPerCent", skip_serializing_if = "Option::is_none")]
    pub credits_per_cent: Option<i64>,
}

/// Currencies the Polar products are configured for. Kept in sync
/// with the per-product `prices` array we maintain on Polar. If a
/// new currency is added there, list it here so the frontend renders
/// the matching badge.
fn supported_currencies() -> Vec<String> {
    vec!["USD".to_string(), "EUR".to_string()]
}

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}
fn env_i64(key: &str) -> Option<i64> {
    env(key).and_then(|s| s.parse().ok())
}

fn fixed_pack(id: &str, label: &str) -> Option<Pack> {
    let upper = id.to_uppercase().replace('-', "_");
    let product = env(&format!("POLAR_PACK_{upper}_PRODUCT"))?;
    let price = env_i64(&format!("POLAR_PACK_{upper}_PRICE_CENTS"))?;
    let credits = env_i64(&format!("POLAR_PACK_{upper}_CREDITS"))?;
    Some(Pack {
        id: id.to_string(),
        label: label.to_string(),
        kind: "fixed".to_string(),
        product_id: product,
        currencies: supported_currencies(),
        price_cents: Some(price),
        credits: Some(credits),
        min_cents: None,
        max_cents: None,
        credits_per_cent: None,
    })
}

fn custom_pack(id: &str, label: &str) -> Option<Pack> {
    let upper = id.to_uppercase().replace('-', "_");
    let product = env(&format!("POLAR_PACK_{upper}_PRODUCT"))?;
    let min = env_i64(&format!("POLAR_PACK_{upper}_MIN_CENTS")).unwrap_or(100);
    let max = env_i64(&format!("POLAR_PACK_{upper}_MAX_CENTS")).unwrap_or(1000);
    let per = env_i64(&format!("POLAR_PACK_{upper}_CREDITS_PER_CENT")).unwrap_or(1);
    Some(Pack {
        id: id.to_string(),
        label: label.to_string(),
        kind: "custom".to_string(),
        product_id: product,
        currencies: supported_currencies(),
        price_cents: None,
        credits: None,
        min_cents: Some(min),
        max_cents: Some(max),
        credits_per_cent: Some(per),
    })
}

fn build_pack_catalogue() -> Vec<Pack> {
    let mut packs = Vec::new();
    if let Some(p) = fixed_pack("starter", "$1 · 100 credits") {
        packs.push(p);
    }
    if let Some(p) = fixed_pack("boost", "$3 · 320 credits (+7%)") {
        packs.push(p);
    }
    if let Some(p) = fixed_pack("stack", "$5 · 550 credits (+10%)") {
        packs.push(p);
    }
    if let Some(p) = fixed_pack("pro", "$10 · 1200 credits (+20%)") {
        packs.push(p);
    }
    if let Some(p) = custom_pack("custom", "Choose your amount") {
        packs.push(p);
    }
    packs
}

/// GET /api/credits/packs — list of pack configurations the
/// frontend renders. Operator controls the contents via env vars
/// (POLAR_PACK_*_*). Returns the empty list when nothing is
/// configured — the UI then shows a "billing not configured" state.
pub async fn list_packs() -> Response {
    Json(build_pack_catalogue()).into_response()
}

// ── Balance + ledger ────────────────────────────────────────────────

#[derive(Serialize, sqlx::FromRow)]
pub struct CreditEventRow {
    pub id: i64,
    pub delta: i32,
    #[serde(rename = "balanceAfter")]
    pub balance_after: i32,
    pub kind: String,
    pub meta: serde_json::Value,
    /// ISO-8601 string. We render `created_at` as `text` server-side
    /// to keep this module free of the `chrono` sqlx feature flag —
    /// it would otherwise pull in a transitive sqlite3 dep that
    /// conflicts with `next-plaid`'s rusqlite.
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

#[derive(Serialize)]
pub struct CreditsResponse {
    pub balance: i32,
    pub history: Vec<CreditEventRow>,
}

/// GET /api/me/credits — current balance + the 50 most recent events.
pub async fn get_credits(State(pool): State<PgPool>, jar: CookieJar) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let balance: i32 = sqlx::query_scalar("SELECT credits_balance($1)::int")
        .bind(me.id)
        .fetch_one(&pool)
        .await
        .unwrap_or(0);
    let history: Vec<CreditEventRow> = sqlx::query_as(
        "SELECT id, delta, balance_after, kind, meta,
                to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS created_at
           FROM credit_events
          WHERE user_id = $1
          ORDER BY id DESC
          LIMIT 50",
    )
    .bind(me.id)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    Json(CreditsResponse { balance, history }).into_response()
}

// ── Checkout ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CheckoutRequest {
    /// Polar product id of the pack the user wants to buy.
    /// We don't validate against an allow-list here — the
    /// `metadata.user_id` stamp + the webhook signature check are
    /// what stop someone topping up the wrong user. (Polar itself
    /// only honours product ids registered in your org.)
    #[serde(rename = "productId")]
    pub product_id: String,
    /// Optional absolute URL to send the user to after a successful
    /// payment. Defaults to the public app URL.
    #[serde(rename = "successUrl")]
    pub success_url: Option<String>,
    /// Amount in minor units (cents), required for pay-what-you-want
    /// products. Clamped server-side to the pack's configured range
    /// so the user can't bypass the cap by hitting the API directly.
    #[serde(rename = "amountCents")]
    pub amount_cents: Option<i64>,
}

#[derive(Serialize)]
pub struct CheckoutResponse {
    pub url: String,
    pub id: String,
}

/// POST /api/credits/checkout — mint a Polar checkout URL for the
/// signed-in caller. We don't create the Polar customer eagerly;
/// Polar will create one on the first checkout and the webhook
/// gives us back the id so we can store it for next time.
pub async fn start_checkout(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<CheckoutRequest>,
) -> Response {
    let Some(me) = current_user(&pool, &jar).await else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let polar_customer_id: Option<String> =
        sqlx::query_scalar("SELECT polar_customer_id FROM polar_customers WHERE user_id = $1")
            .bind(me.id)
            .fetch_optional(&pool)
            .await
            .unwrap_or(None);

    let success_url = req.success_url.as_deref().or(Some(
        // Fall back to the public app URL when the caller didn't
        // specify a return target.
        Box::leak(public_base_url().into_boxed_str()),
    ));

    // Match the request against the configured catalogue so we can:
    //   1. clamp the custom amount to the configured min/max,
    //   2. refuse arbitrary product ids the operator hasn't blessed.
    let packs = build_pack_catalogue();
    let pack = packs.iter().find(|p| p.product_id == req.product_id);
    let amount = match pack {
        Some(p) if p.kind == "custom" => {
            let raw = req.amount_cents.unwrap_or(0);
            let lo = p.min_cents.unwrap_or(100);
            let hi = p.max_cents.unwrap_or(1000);
            Some(raw.clamp(lo, hi))
        }
        Some(_) => None, // fixed pack — amount is implied by the product
        None => {
            // Unknown product id. Reject rather than trust the caller.
            return (StatusCode::BAD_REQUEST, "unknown pack").into_response();
        }
    };

    match polar::create_checkout(
        &req.product_id,
        me.id,
        success_url,
        polar_customer_id.as_deref(),
        amount,
    )
    .await
    {
        Ok(c) => Json(CheckoutResponse {
            url: c.url,
            id: c.id,
        })
        .into_response(),
        Err(e) => {
            tracing::error!(error = %e, "credits.checkout.failed");
            (StatusCode::BAD_GATEWAY, e).into_response()
        }
    }
}

fn public_base_url() -> String {
    std::env::var("PUBLIC_BASE_URL").unwrap_or_else(|_| "http://localhost:3001".to_string())
}

// ── Webhook receiver ────────────────────────────────────────────────

/// POST /api/credits/webhook
///
/// Polar fires this on order events. We verify the Standard
/// Webhooks signature, dedupe on `webhook-id`, look up the
/// internal user from `metadata.user_id`, and `credits_top_up()`
/// for the credit amount specified on the purchased product's
/// metadata (key `credits`, integer).
///
/// Returns 2xx even on no-op (idempotency replays, unknown event
/// types) so Polar doesn't retry forever. Only signature/parse
/// failures surface as 4xx.
pub async fn webhook(State(pool): State<PgPool>, headers: HeaderMap, body: Bytes) -> Response {
    // ── 1. Signature verification ─────────────────────────────────
    let Some(secret) = polar::webhook_secret() else {
        tracing::error!("polar.webhook: POLAR_WEBHOOK_SECRET is not set");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "webhook receiver not configured",
        )
            .into_response();
    };
    let header = |name: &str| -> Option<&str> { headers.get(name).and_then(|v| v.to_str().ok()) };
    let (Some(webhook_id), Some(timestamp), Some(signature)) = (
        header("webhook-id"),
        header("webhook-timestamp"),
        header("webhook-signature"),
    ) else {
        return (StatusCode::BAD_REQUEST, "missing standard-webhooks headers").into_response();
    };
    if !polar::verify_webhook(&secret, webhook_id, timestamp, &body, signature) {
        tracing::warn!(webhook_id = %webhook_id, "polar.webhook.bad_signature");
        return (StatusCode::UNAUTHORIZED, "invalid signature").into_response();
    }

    // ── 2. Parse envelope ─────────────────────────────────────────
    let envelope: polar::WebhookEnvelope = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(error = %e, "polar.webhook.parse.failed");
            return (StatusCode::BAD_REQUEST, "could not parse webhook body").into_response();
        }
    };

    // ── 3. Route by event type ────────────────────────────────────
    // We treat "order.created", "order.paid", and "order.updated"
    // as the credit-grant trigger — Polar's docs have varied on
    // which one fires when, so we accept any of them and rely on
    // the unique-event-id dedupe to prevent double-crediting.
    let kind = envelope.event_type.as_str();
    if !matches!(
        kind,
        "order.created" | "order.paid" | "order.updated" | "checkout.completed"
    ) {
        // Other events (subscription.created, customer.updated, …)
        // are no-ops in v1. ACK so Polar moves on.
        return (StatusCode::OK, "ignored").into_response();
    }

    let order: polar::OrderData = match serde_json::from_value(envelope.data.clone()) {
        Ok(o) => o,
        Err(e) => {
            tracing::error!(error = %e, body = %String::from_utf8_lossy(&body), "polar.webhook.order.parse.failed");
            return (StatusCode::OK, "ignored").into_response();
        }
    };

    // ── 4. Map back to internal user ──────────────────────────────
    let user_id: Option<i64> = order.metadata.get("user_id").and_then(|v| {
        v.as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| v.as_i64())
    });
    let Some(user_id) = user_id else {
        tracing::warn!(order_id = %order.id, "polar.webhook.no_user_metadata");
        return (StatusCode::OK, "ignored: missing user_id").into_response();
    };

    // ── 5. Pull the credit grant from product metadata ────────────
    //
    // Two paths:
    //   - product.metadata.credits        — fixed grant (starter / small / …)
    //   - product.metadata.credits_per_cent — variable grant for
    //     the custom-amount pack; credits = order.amount * factor,
    //     where order.amount is in minor units (cents).
    //
    // The first one wins when both are set, so a misconfigured
    // product never accidentally over-credits.
    //
    // Polar's metadata API stores values as strings (the SDK type is
    // `Mapping[str, str | int | bool]` but JSON-serialises to strings
    // by default). We accept both shapes — i64 directly OR a string
    // that parses to one — so we don't silently miss a grant because
    // of a JSON type mismatch.
    fn meta_int(meta: &serde_json::Value, key: &str) -> Option<i64> {
        let v = meta.get(key)?;
        v.as_i64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    }

    // The order webhook may not expand `product.metadata` — Polar's
    // schema marks it optional. When that happens we look up the
    // product via the REST API so the credit grant still lands. The
    // resolved metadata is identical in both cases.
    let mut product_metadata: serde_json::Value = order
        .product
        .as_ref()
        .map(|p| p.metadata.clone())
        .unwrap_or_else(|| serde_json::json!({}));
    if product_metadata
        .as_object()
        .map(|o| o.is_empty())
        .unwrap_or(true)
    {
        if let Some(pid) = order.product_id().map(|s| s.to_string()) {
            if let Some(fetched) = polar::fetch_product_metadata(&pid).await {
                product_metadata = fetched;
            }
        }
    }
    let fixed_credits = meta_int(&product_metadata, "credits");
    let per_cent = meta_int(&product_metadata, "credits_per_cent");
    let credits = if let Some(c) = fixed_credits {
        c
    } else if let Some(rate) = per_cent {
        // Use `total_amount` (what the customer actually paid,
        // including any VAT) as the credit-grant basis. Not
        // `subtotal_amount` — for tax-inclusive jurisdictions
        // (e.g. EU VAT) Polar back-computes subtotal as
        // total - tax, so a €3 PWYW order would only grant 250
        // credits at 1 credit/cent when the user clearly paid
        // 300 cents. Not `amount` either — that's the net we
        // receive after Polar fees. Fall back to subtotal, then
        // amount, only if Polar didn't populate total.
        let basis = if order.total_amount > 0 {
            order.total_amount
        } else if order.subtotal_amount > 0 {
            order.subtotal_amount
        } else {
            order.amount
        };
        basis.saturating_mul(rate)
    } else {
        0
    };
    if credits <= 0 {
        tracing::warn!(
            order_id = %order.id,
            "polar.webhook.no_credits_metadata — set `credits` or `credits_per_cent` on the Polar product"
        );
        return (StatusCode::OK, "ignored: no credits on product").into_response();
    }

    // ── 6. Atomic top-up via the SQL helper ───────────────────────
    let polar_event_id = envelope.id.clone().unwrap_or_else(|| order.id.clone());
    let meta = serde_json::json!({
        "polar_order_id":     order.id,
        // What the customer paid (gross). For PWYW grants this is
        // the basis we charge credits against.
        "subtotal_minor":     order.subtotal_amount,
        // Final charged amount including tax.
        "total_minor":        order.total_amount,
        // What we receive after Polar fees + tax (revenue).
        "net_minor":          order.amount,
        "currency":           order.currency.clone().unwrap_or_default(),
        "polar_event":        envelope.event_type,
    });
    let result =
        sqlx::query_scalar::<_, i32>("SELECT credits_top_up($1, $2::int, 'top_up', $3, $4::jsonb)")
            .bind(user_id)
            .bind(credits as i32)
            .bind(&polar_event_id)
            .bind(&meta)
            .fetch_one(&pool)
            .await;

    match result {
        Ok(new_balance) => {
            tracing::info!(
                user_id = user_id,
                credits = credits,
                new_balance = new_balance,
                order_id = %order.id,
                "credits.top_up.ok"
            );
            // ── 7. Best-effort: persist the Polar customer id for
            // future checkouts. Run after the credit landing so a
            // mapping failure can't block the top-up.
            if let Some(cust) = order.customer_id.as_deref() {
                let _ = sqlx::query(
                    "INSERT INTO polar_customers (user_id, polar_customer_id)
                     VALUES ($1, $2)
                     ON CONFLICT (user_id) DO UPDATE SET polar_customer_id = EXCLUDED.polar_customer_id",
                )
                .bind(user_id)
                .bind(cust)
                .execute(&pool)
                .await;
            }
            (StatusCode::OK, "ok").into_response()
        }
        Err(e) => {
            // 23505 = unique_violation. The unique index on
            // polar_event_id is the idempotency guard — a replay of
            // the same event ends up here and we ACK as a no-op.
            if let Some(db_err) = e.as_database_error() {
                if db_err.code().as_deref() == Some("23505") {
                    tracing::info!(order_id = %order.id, "credits.top_up.duplicate");
                    return (StatusCode::OK, "duplicate").into_response();
                }
            }
            tracing::error!(error = %e, order_id = %order.id, "credits.top_up.failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "top up failed").into_response()
        }
    }
}
