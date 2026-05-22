//! Shared SSRF guards for outbound HTTP from request handlers.
//!
//! Every request that fetches a user-supplied URL (the `/api/proxy/fetch`
//! relay and the profile-source probes) must route through `safe_get`.
//!
//! The guard is two layered:
//!
//!   1. The hostname is resolved server-side via the OS resolver, and
//!      every returned IP is checked against the private/loopback/
//!      link-local/ULA/CGNAT block list.
//!   2. The resulting `reqwest::Client` is built with
//!      `resolve_to_addrs` pinned to the validated IPs only, so even
//!      if a DNS-rebinding attacker flips the record between our
//!      lookup and reqwest's connect, the connection still targets a
//!      vetted address.
//!
//! Redirects are followed manually with re-validation on every hop
//! (`reqwest::redirect::Policy::none()`), preventing a public 302 →
//! private IP escape.

use reqwest::{redirect::Policy, Client, Response, Url};
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;
use tokio::net::lookup_host;

pub const MAX_REDIRECTS: u8 = 8;

#[derive(Debug)]
pub enum FetchError {
    InvalidUrl,
    SchemeNotAllowed,
    HostMissing,
    BlockedHost,
    DnsFailed,
    TooManyRedirects,
    BuildClient,
    Request(reqwest::Error),
    RedirectLocationMissing,
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::InvalidUrl => f.write_str("invalid url"),
            FetchError::SchemeNotAllowed => f.write_str("scheme not allowed"),
            FetchError::HostMissing => f.write_str("missing host"),
            FetchError::BlockedHost => f.write_str("private network not allowed"),
            FetchError::DnsFailed => f.write_str("dns lookup failed"),
            FetchError::TooManyRedirects => f.write_str("too many redirects"),
            FetchError::BuildClient => f.write_str("client init"),
            FetchError::Request(e) => write!(f, "upstream: {}", e),
            FetchError::RedirectLocationMissing => f.write_str("redirect without location"),
        }
    }
}

impl std::error::Error for FetchError {}

/// IPs we will never connect to from a user-supplied URL.
pub fn is_blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_unspecified()
                || v4.is_documentation()
                // Carrier-grade NAT 100.64.0.0/10 — used by some cloud
                // metadata layers.
                || (o[0] == 100 && (o[1] & 0xc0) == 64)
                // 0.0.0.0/8 — "this network".
                || o[0] == 0
            // 169.254.169.254 in particular is the cloud metadata
            // endpoint; covered by is_link_local() already but
            // worth noting.
        }
        IpAddr::V6(v6) => {
            let segs = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                // ULA fc00::/7
                || (segs[0] & 0xfe00) == 0xfc00
                // Link-local fe80::/10
                || (segs[0] & 0xffc0) == 0xfe80
                // IPv4-mapped (::ffff:0:0/96) → re-check the embedded v4
                || matches!(v6.to_ipv4_mapped(), Some(v4) if is_blocked_ip(&IpAddr::V4(v4)))
        }
    }
}

/// Hostnames we reject before even attempting resolution.
pub fn is_blocked_host_literal(host: &str) -> bool {
    let h = host.to_ascii_lowercase();
    if h == "localhost" || h.ends_with(".localhost") {
        return true;
    }
    // Reject literal IP hostnames that fall in a blocked range. This
    // is also caught by `is_blocked_ip` after DNS resolution, but
    // short-circuiting here avoids leaking "DNS failed" vs "blocked".
    if let Ok(ip) = h.parse::<IpAddr>() {
        return is_blocked_ip(&ip);
    }
    false
}

/// Resolve a URL's host and return only the IPs that pass `is_blocked_ip`.
async fn resolve_safe(url: &Url) -> Result<(String, u16, Vec<SocketAddr>), FetchError> {
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(FetchError::SchemeNotAllowed);
    }
    let host = url.host_str().ok_or(FetchError::HostMissing)?.to_string();
    if is_blocked_host_literal(&host) {
        return Err(FetchError::BlockedHost);
    }
    let port = url.port_or_known_default().ok_or(FetchError::HostMissing)?;

    let lookup_target = format!("{}:{}", host, port);
    let addrs: Vec<SocketAddr> = lookup_host(lookup_target)
        .await
        .map_err(|_| FetchError::DnsFailed)?
        .collect();
    if addrs.is_empty() {
        return Err(FetchError::DnsFailed);
    }
    let safe: Vec<SocketAddr> = addrs
        .into_iter()
        .filter(|sa| !is_blocked_ip(&sa.ip()))
        .collect();
    if safe.is_empty() {
        return Err(FetchError::BlockedHost);
    }
    Ok((host, port, safe))
}

/// SSRF-safe HTTP GET.
///
/// Resolves DNS, validates every returned IP, pins reqwest's resolver
/// to the validated set, and follows up to `MAX_REDIRECTS` redirects
/// manually — re-validating each hop's URL from scratch.
pub async fn safe_get(
    url: &str,
    timeout: Duration,
    user_agent: &str,
) -> Result<Response, FetchError> {
    let mut current = Url::parse(url).map_err(|_| FetchError::InvalidUrl)?;
    for _hop in 0..=MAX_REDIRECTS {
        let (host, _port, safe_addrs) = resolve_safe(&current).await?;

        let mut builder = Client::builder()
            .use_native_tls()
            .user_agent(user_agent)
            .timeout(timeout)
            .redirect(Policy::none());
        // Pin DNS for this host to validated IPs only. reqwest's
        // resolver replaces the OS lookup with these entries, so even
        // a flipped DNS record cannot redirect the connect() call.
        for addr in &safe_addrs {
            builder = builder.resolve_to_addrs(&host, &[*addr]);
        }
        let client = builder.build().map_err(|_| FetchError::BuildClient)?;

        let resp = client
            .get(current.as_str())
            .send()
            .await
            .map_err(FetchError::Request)?;

        if resp.status().is_redirection() {
            let loc = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or(FetchError::RedirectLocationMissing)?;
            current = current.join(loc).map_err(|_| FetchError::InvalidUrl)?;
            continue;
        }
        return Ok(resp);
    }
    Err(FetchError::TooManyRedirects)
}
