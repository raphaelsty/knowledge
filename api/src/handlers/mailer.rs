//! Outbound email via Resend (resend.com).
//!
//! Configured by env:
//!   RESEND_API_KEY     — required to actually send. When unset, every
//!                        send is logged to tracing at INFO and treated
//!                        as a success (dev mode).
//!   RESEND_FROM        — sender address, e.g. "Knowledge <no-reply@knowledge-web.org>".
//!                        Required when RESEND_API_KEY is set.
//!   PUBLIC_BASE_URL    — base URL embedded in the verification +
//!                        reset links sent to users (default
//!                        "http://localhost:3001").

use serde::Serialize;

const RESEND_URL: &str = "https://api.resend.com/emails";

fn public_base_url() -> String {
    std::env::var("PUBLIC_BASE_URL").unwrap_or_else(|_| "http://localhost:3001".to_string())
}

fn from_addr() -> Option<String> {
    std::env::var("RESEND_FROM").ok()
}

fn api_key() -> Option<String> {
    std::env::var("RESEND_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
}

#[derive(Serialize)]
struct ResendPayload<'a> {
    from: &'a str,
    to: [&'a str; 1],
    subject: &'a str,
    html: &'a str,
    text: &'a str,
}

async fn deliver(to: &str, subject: &str, html: &str, text: &str) -> Result<(), String> {
    let Some(key) = api_key() else {
        // Dev mode: no API key configured. Print the body so the
        // developer can copy/paste the link locally.
        tracing::info!(target: "mailer.dev", to = %to, subject = %subject, body = %text, "(no RESEND_API_KEY — printing email)");
        return Ok(());
    };
    let from = from_addr()
        .ok_or_else(|| "RESEND_FROM is required when RESEND_API_KEY is set".to_string())?;
    let payload = ResendPayload {
        from: &from,
        to: [to],
        subject,
        html,
        text,
    };
    let resp = reqwest::Client::builder()
        .user_agent("knowledge-api/0.1")
        .build()
        .map_err(|e| format!("http client build failed: {e}"))?
        .post(RESEND_URL)
        .bearer_auth(&key)
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("resend send failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("resend {status}: {body}"));
    }
    Ok(())
}

pub async fn send_verification_email(to: &str, name: &str, token: &str) -> Result<(), String> {
    let link = format!("{}/auth/verify?token={}", public_base_url(), token);
    let subject = "Verify your Knowledge email";
    let safe_name = html_escape(name);
    let safe_link = html_escape(&link);
    let html = format!(
        "<p>Hi {safe_name},</p>\
         <p>Click the link below to verify your Knowledge account. The link expires in 24 hours.</p>\
         <p><a href=\"{safe_link}\">{safe_link}</a></p>\
         <p>If you didn't sign up, you can ignore this email.</p>"
    );
    let text = format!(
        "Hi {name},\n\nVerify your Knowledge account by opening this link (expires in 24h):\n{link}\n\nIf you didn't sign up, ignore this email.\n"
    );
    deliver(to, subject, &html, &text).await
}

pub async fn send_password_reset_email(to: &str, name: &str, token: &str) -> Result<(), String> {
    let link = format!("{}/auth/reset?token={}", public_base_url(), token);
    let subject = "Reset your Knowledge password";
    let safe_name = html_escape(name);
    let safe_link = html_escape(&link);
    let html = format!(
        "<p>Hi {safe_name},</p>\
         <p>You requested a password reset. Click the link below to choose a new password — the link expires in 1 hour.</p>\
         <p><a href=\"{safe_link}\">{safe_link}</a></p>\
         <p>If you didn't request a reset, you can ignore this email — your password won't change.</p>"
    );
    let text = format!(
        "Hi {name},\n\nReset your Knowledge password by opening this link (expires in 1h):\n{link}\n\nIf you didn't request a reset, ignore this email.\n"
    );
    deliver(to, subject, &html, &text).await
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}
