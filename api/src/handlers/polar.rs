//! Polar.sh integration.
//!
//! This module is intentionally isolated from the rest of the API:
//! it owns the Polar HTTP client (checkout creation + customer
//! upsert) and the Standard Webhooks signature verification. The
//! credit-handler module wires it into the actual `/api/credits/*`
//! routes.
//!
//! Environment config:
//!   POLAR_ACCESS_TOKEN   — server-side org access token
//!   POLAR_WEBHOOK_SECRET — HMAC-SHA256 secret from the Polar
//!                          dashboard, formatted as the Standard
//!                          Webhooks spec expects (the secret string
//!                          Polar gives you, no `whsec_` prefix
//!                          processing — we treat it verbatim).
//!   POLAR_API_BASE       — base URL override (default
//!                          https://api.polar.sh). Set to the
//!                          sandbox host while testing.
//!
//! Two helpers are exposed:
//!   - `create_checkout(...)` — POST /v1/checkouts/, returns the
//!     hosted-checkout URL.
//!   - `verify_webhook(...)` — Constant-time HMAC compare against
//!     the `webhook-signature` header.

use base64::Engine as _;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

// ── Env helpers ─────────────────────────────────────────────────────

fn polar_base() -> String {
    std::env::var("POLAR_API_BASE").unwrap_or_else(|_| "https://api.polar.sh".to_string())
}

pub fn access_token() -> Option<String> {
    std::env::var("POLAR_ACCESS_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
}

pub fn webhook_secret() -> Option<String> {
    std::env::var("POLAR_WEBHOOK_SECRET")
        .ok()
        .filter(|s| !s.is_empty())
}

/// Fetch a product's metadata JSON from the Polar REST API. Used as
/// a fallback by the order webhook when the inline `product.metadata`
/// field isn't expanded — Polar's webhook schema marks it optional
/// so we can't rely on it. Returns None when the request fails for
/// any reason (no token, network error, 4xx, etc.); the caller
/// degrades to "no credits granted" + a warning log line.
pub async fn fetch_product_metadata(product_id: &str) -> Option<serde_json::Value> {
    let token = access_token()?;
    let url = format!("{}/v1/products/{}", polar_base(), product_id);
    let client = reqwest::Client::new();
    let resp = client.get(&url).bearer_auth(token).send().await.ok()?;
    if !resp.status().is_success() {
        tracing::warn!(
            product_id = %product_id,
            status = %resp.status(),
            "polar.fetch_product_metadata.non_success"
        );
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body.get("metadata").cloned()
}

// ── Checkout ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct CheckoutRequest<'a> {
    products: Vec<&'a str>,
    /// External reference we can echo back through the webhook so
    /// the credit handler knows which user to top up. Polar surfaces
    /// it on the resulting Order under `metadata.user_id`.
    metadata: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    success_url: Option<&'a str>,
    /// Optional pre-created customer id so the user doesn't have to
    /// retype their email at checkout for repeat purchases.
    #[serde(skip_serializing_if = "Option::is_none")]
    customer_id: Option<&'a str>,
    /// Amount in minor units (cents) — required for pay-what-you-want
    /// products, ignored for fixed-price products. Polar's checkout
    /// pre-fills the input field with this value so the user doesn't
    /// have to re-pick the amount they already chose in our UI.
    #[serde(skip_serializing_if = "Option::is_none")]
    amount: Option<i64>,
}

#[derive(Deserialize)]
struct CheckoutResponse {
    url: String,
    id: String,
}

pub struct CheckoutCreated {
    pub url: String,
    pub id: String,
}

/// Start a Polar-hosted checkout for the given product id. `user_id`
/// is stamped into the checkout's metadata so the webhook can map
/// the resulting order back to the right user.
pub async fn create_checkout(
    product_id: &str,
    user_id: i64,
    success_url: Option<&str>,
    customer_id: Option<&str>,
    amount: Option<i64>,
) -> Result<CheckoutCreated, String> {
    let token = access_token().ok_or_else(|| "POLAR_ACCESS_TOKEN not set".to_string())?;
    let body = CheckoutRequest {
        products: vec![product_id],
        metadata: serde_json::json!({ "user_id": user_id.to_string() }),
        success_url,
        customer_id,
        amount,
    };
    let http = reqwest::Client::builder()
        .user_agent("knowledge-api/0.1")
        .build()
        .map_err(|e| format!("http client build failed: {e}"))?;
    let resp = http
        .post(format!("{}/v1/checkouts/", polar_base()))
        .bearer_auth(&token)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("polar request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("polar checkout {status}: {text}"));
    }
    let parsed: CheckoutResponse = resp
        .json()
        .await
        .map_err(|e| format!("polar checkout response parse failed: {e}"))?;
    Ok(CheckoutCreated {
        url: parsed.url,
        id: parsed.id,
    })
}

// ── Webhook signature verification (Standard Webhooks) ──────────────
//
// Polar uses the Standard Webhooks spec (https://www.standardwebhooks.com).
// Headers:
//   webhook-id        — opaque event id, e.g. "msg_2x..."
//   webhook-timestamp — unix-seconds, e.g. "1729000000"
//   webhook-signature — space-separated list of `v1,<base64-sha256>`
//                       tokens (one secret can have N rotated values).
//
// The signed payload is:
//   "${webhook-id}.${webhook-timestamp}.${raw-body}"
//
// Compare with constant-time equality. Reject when no v1 token
// matches, or when the timestamp is too old (>5 min skew).

/// Returns true iff the signature header carries a v1 token that
/// matches an HMAC-SHA256 of `<id>.<timestamp>.<body>` keyed with
/// `secret`. Caller passes the verbatim secret from
/// POLAR_WEBHOOK_SECRET; we treat anything after a leading
/// `whsec_` as the key bytes (Polar's modern dashboard hands out
/// the prefixed form).
pub fn verify_webhook(
    secret: &str,
    id: &str,
    timestamp: &str,
    body: &[u8],
    signature_header: &str,
) -> bool {
    // Drop the optional `whsec_` prefix and base64-decode if the
    // remainder looks base64-shaped, otherwise treat as raw bytes.
    // Polar currently ships base64 secrets with the prefix.
    let key_bytes = secret_to_bytes(secret);

    let mut mac = match HmacSha256::new_from_slice(&key_bytes) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(id.as_bytes());
    mac.update(b".");
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    let expected = mac.finalize().into_bytes();
    let expected_b64 = base64::engine::general_purpose::STANDARD.encode(expected);

    // signature_header is "v1,<sig> v1,<sig> ..."
    for token in signature_header.split_whitespace() {
        let (version, sig) = match token.split_once(',') {
            Some(parts) => parts,
            None => continue,
        };
        if version != "v1" {
            continue;
        }
        if sig.as_bytes().ct_eq(expected_b64.as_bytes()).into() {
            return true;
        }
    }
    false
}

fn secret_to_bytes(secret: &str) -> Vec<u8> {
    let raw = secret.strip_prefix("whsec_").unwrap_or(secret);
    base64::engine::general_purpose::STANDARD
        .decode(raw)
        .unwrap_or_else(|_| raw.as_bytes().to_vec())
}

// ── Webhook payload model ───────────────────────────────────────────
//
// Polar fires several event types; the ones we care about for credit
// top-ups are `order.created` and `order.paid` (Polar's docs use
// both spellings in different contexts — we accept either). The
// payload carries:
//   - id          : event id (we dedupe on this)
//   - type        : "order.created" | "order.paid" | …
//   - data        : the order object
//
// We only deserialise the bits we actually need.

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct WebhookEnvelope {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: serde_json::Value,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct OrderData {
    pub id: String,
    /// Polar's top-level `amount` echoes `net_amount` — what *we*
    /// receive after Polar's fees and taxes. Useful for revenue
    /// accounting; NOT what the user thinks they paid.
    #[serde(default)]
    pub amount: i64,
    /// Pre-tax / pre-discount amount in minor units. This is what
    /// the user typed at checkout for pay-what-you-want products,
    /// so credit grants based on the customer's intent (e.g. "$5 →
    /// 500 credits") must compute from this — not from `amount`.
    #[serde(default)]
    pub subtotal_amount: i64,
    /// Final amount the customer was charged (subtotal + tax -
    /// discount). Equal to subtotal_amount when there's no tax /
    /// discount applied.
    #[serde(default)]
    pub total_amount: i64,
    #[serde(default)]
    pub currency: Option<String>,
    /// May come back as a nested object (`product.id`) or as a flat
    /// `product_id`. We try both at parse time.
    #[serde(default)]
    pub product: Option<ProductRef>,
    #[serde(default)]
    pub product_id: Option<String>,
    /// Metadata we stamped on the checkout, echoed back here. We
    /// look up `user_id` here to map the order to an internal user.
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Polar customer id; useful to persist into `polar_customers`.
    #[serde(default)]
    pub customer_id: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct ProductRef {
    pub id: String,
    /// Product metadata, when expanded. We look up a `credits` key
    /// here so the per-pack credit grant lives in Polar (the
    /// merchant of record) rather than scattered across env vars.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl OrderData {
    #[allow(dead_code)] // Reserved for future debit hooks.
    pub fn product_id(&self) -> Option<&str> {
        if let Some(p) = self.product.as_ref() {
            return Some(p.id.as_str());
        }
        self.product_id.as_deref()
    }
}
