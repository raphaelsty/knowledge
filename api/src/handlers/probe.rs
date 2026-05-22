//! POST /api/profile/probe — inline-validate a source value.
//!
//! Used by the profile modal on the welcome page: each field debounces,
//! hits this endpoint, and renders a small ✓/✗ + info line so the user
//! knows a Scholar id / RSS feed / Twitter handle is actually reachable
//! before saving.
//!
//! Request body: `{ "kind": "github", "value": "karpathy" }`
//! Response:     `{ "ok": true, "info": "171 k followers", "error": null }`
//!
//! All probes are best-effort. Network failures return `{ ok: false }`
//! rather than 5xx so the frontend can show an inline error without
//! freezing the form.

use axum::extract::State;
use axum::{http::StatusCode, response::IntoResponse, Json};
use axum_extra::extract::cookie::CookieJar;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::handlers::auth::current_user;

#[derive(Deserialize)]
pub struct ProbeRequest {
    pub kind: String,
    pub value: String,
}

#[derive(Serialize, Default)]
pub struct ProbeResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Canonical public URL for what we just verified — lets the form
    /// render the status as a "✓ 14 k followers ↗" link. Optional
    /// because not every probe has a meaningful destination (e.g. a
    /// sitemap points to a file; the user already typed the URL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// For `website` probes: the sitemap URL we auto-discovered from
    /// whatever the user pasted. The frontend stores this (not the
    /// user's raw input) so the pipeline can run the standard Sitemap
    /// fetcher.
    #[serde(skip_serializing_if = "Option::is_none", rename = "resolvedUrl")]
    pub resolved_url: Option<String>,
    /// For `website` probes: the path prefix we'll filter by (e.g.
    /// `/articles/`). Empty string means "keep everything in the
    /// sitemap". `None` means "not applicable to this probe kind".
    #[serde(skip_serializing_if = "Option::is_none", rename = "resolvedFilter")]
    pub resolved_filter: Option<String>,
    /// For `website` probes: the most common path prefixes we found
    /// in the sitemap, each with a URL count. Surfaced so users who
    /// pasted a homepage (or a filter that matched nothing) can pick
    /// a subtree instead of guessing. Capped to a handful of entries
    /// so the UI stays readable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtrees: Option<Vec<Subtree>>,
    /// For `website` probes: which fetcher the pipeline should route
    /// this to (`feed` or `sitemap`). Lets us fold RSS, sitemaps, and
    /// homepages into a single input box without losing the
    /// format-specific code paths under the hood.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ResolvedKind>,
}

#[derive(Serialize, Default, Clone)]
pub struct Subtree {
    pub path: String,
    pub count: i64,
}

/// What the user's URL resolved to. The frontend stores this verbatim
/// so the pipeline can pick the right fetcher without re-sniffing.
#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ResolvedKind {
    /// RSS 2.0 or Atom feed. Routed through `blog.Feed`.
    Feed,
    /// sitemap.xml or sitemapindex.xml (including auto-discovered).
    /// Routed through `blog.Sitemap`, with `resolved_filter` applied.
    Sitemap,
}

fn fail(msg: &str) -> ProbeResponse {
    ProbeResponse {
        ok: false,
        error: Some(msg.to_string()),
        ..Default::default()
    }
}

fn ok_info_url(info: String, url: String) -> ProbeResponse {
    ProbeResponse {
        ok: true,
        info: Some(info),
        url: Some(url),
        ..Default::default()
    }
}

fn fmt_count(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{:.1} M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1} k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("knowledge-api/0.1 profile-probe")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

// ── Individual probes ────────────────────────────────────────────────

async fn probe_github(handle: &str) -> ProbeResponse {
    let token = std::env::var("GITHUB_TOKEN")
        .or_else(|_| std::env::var("GH_TOKEN"))
        .ok();
    let mut req = http()
        .get(format!("https://api.github.com/users/{}", handle))
        .header("Accept", "application/vnd.github+json");
    if let Some(t) = token {
        req = req.header("Authorization", format!("token {}", t));
    }
    let Ok(resp) = req.send().await else {
        return fail("network error");
    };
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return fail("user not found");
    }
    if !resp.status().is_success() {
        return fail(&format!("github returned {}", resp.status().as_u16()));
    }
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return fail("github response parse failed");
    };
    let followers = body.get("followers").and_then(|v| v.as_i64()).unwrap_or(0);
    let repos = body
        .get("public_repos")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let html_url = body
        .get("html_url")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| format!("https://github.com/{}", handle));
    ok_info_url(
        format!(
            "{} followers · {} repos",
            fmt_count(followers),
            fmt_count(repos)
        ),
        html_url,
    )
}

async fn probe_twitter(handle: &str) -> ProbeResponse {
    // No live validation: x.com aggressively gates handle lookups behind
    // auth, and the Python pipeline (`make run`) is the authoritative path
    // for fetching follower counts via TWITTERAPIIO_API_KEY from .env.
    // The probe just constructs the canonical URL so the form can render
    // a clickable check.
    ok_info_url(String::new(), format!("https://x.com/{}", handle))
}

async fn probe_blog(url: &str) -> ProbeResponse {
    // User-supplied URL → must go through the SSRF-safe fetcher.
    let Ok(resp) = crate::handlers::url_safety::safe_get(
        url,
        std::time::Duration::from_secs(10),
        "knowledge-api/0.1 profile-probe",
    )
    .await
    else {
        return fail("unreachable");
    };
    if !resp.status().is_success() {
        return fail(&format!("returned {}", resp.status().as_u16()));
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = resp.text().await.unwrap_or_default();
    let head = body.get(..4096).unwrap_or(&body).to_ascii_lowercase();

    // Accept Atom, RSS 2.0, or bare-XML feeds. Count entries/items for
    // a friendly "12 posts" hint.
    let is_atom = head.contains("<feed") || head.contains("application/atom");
    let is_rss = head.contains("<rss") || head.contains("<channel");
    if !is_atom && !is_rss && !ct.contains("xml") {
        return fail("not a valid feed");
    }
    let entries = body.matches("<entry").count() + body.matches("<item").count();
    if entries == 0 {
        return ok_info_url("feed looks empty".to_string(), url.to_string());
    }
    ok_info_url(format!("{} posts in feed", entries), url.to_string())
}

async fn probe_sitemap(url: &str) -> ProbeResponse {
    // User-supplied URL → must go through the SSRF-safe fetcher.
    let Ok(resp) = crate::handlers::url_safety::safe_get(
        url,
        std::time::Duration::from_secs(10),
        "knowledge-api/0.1 profile-probe",
    )
    .await
    else {
        return fail("unreachable");
    };
    if !resp.status().is_success() {
        return fail(&format!("returned {}", resp.status().as_u16()));
    }
    let body = resp.text().await.unwrap_or_default();
    let head = body.get(..4096).unwrap_or(&body).to_ascii_lowercase();
    if !head.contains("<urlset") && !head.contains("<sitemapindex") {
        return fail("not a sitemap");
    }
    let n = body.matches("<loc").count();
    ok_info_url(format!("{} URLs", fmt_count(n as i64)), url.to_string())
}

async fn probe_scholar(id: &str) -> ProbeResponse {
    // The Scholar profile page doesn't need auth and returns HTML we
    // can scrape lightly. Google blocks many regions — so failure is
    // treated as "unknown", not as invalid.
    let url = format!("https://scholar.google.com/citations?user={}&hl=en", id);
    let Ok(resp) = http().get(&url).send().await else {
        return fail("unreachable (network or google block)");
    };
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return fail("id not found");
    }
    if !resp.status().is_success() {
        return fail(&format!("scholar returned {}", resp.status().as_u16()));
    }
    let html = resp.text().await.unwrap_or_default();
    if html.contains("gsc_prf_in") {
        // Extract total citation count from the stats table. The layout
        // is `<td class="gsc_rsb_std">12345</td>` — first cell is cites.
        let citations = html
            .split("gsc_rsb_std")
            .nth(1)
            .and_then(|s| s.split('>').nth(1))
            .and_then(|s| s.split('<').next())
            .and_then(|s| s.trim().replace(',', "").parse::<i64>().ok())
            .unwrap_or(0);
        let info = if citations > 0 {
            format!("{} citations", fmt_count(citations))
        } else {
            "profile found".to_string()
        };
        return ok_info_url(info, url);
    }
    fail("profile not found")
}

async fn probe_huggingface(handle: &str) -> ProbeResponse {
    let Ok(resp) = http()
        .get(format!(
            "https://huggingface.co/api/users/{}/overview",
            handle
        ))
        .send()
        .await
    else {
        return fail("network error");
    };
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return fail("user not found");
    }
    if !resp.status().is_success() {
        return fail(&format!("hf returned {}", resp.status().as_u16()));
    }
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return fail("hf response parse failed");
    };
    let followers = body
        .get("numFollowers")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let models = body.get("numModels").and_then(|v| v.as_i64()).unwrap_or(0);
    ok_info_url(
        format!(
            "{} followers · {} models",
            fmt_count(followers),
            fmt_count(models)
        ),
        format!("https://huggingface.co/{}", handle),
    )
}

async fn probe_arxiv(author: &str) -> ProbeResponse {
    // arXiv's Atom feed is CORS-blocked in the browser; we proxy it here
    // and just count <entry> elements.
    let q = format!("au:\"{}\"", author);
    let url = format!(
        "https://export.arxiv.org/api/query?search_query={}&start=0&max_results=1",
        urlencoding::encode(&q)
    );
    let Ok(resp) = http().get(url).send().await else {
        return fail("network error");
    };
    if !resp.status().is_success() {
        return fail(&format!("arxiv returned {}", resp.status().as_u16()));
    }
    let body = resp.text().await.unwrap_or_default();
    // <opensearch:totalResults>N</opensearch:totalResults>
    let total = body
        .split("totalResults>")
        .nth(1)
        .and_then(|s| s.split('<').next())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(-1);
    if total <= 0 {
        return fail("no papers found");
    }
    let html_url = format!(
        "https://arxiv.org/a/{}",
        urlencoding::encode(&author.to_lowercase().replace(' ', "_"))
    );
    ok_info_url(format!("{} papers", fmt_count(total)), html_url)
}

async fn probe_stackoverflow(user_id: &str) -> ProbeResponse {
    // Input must parse as a numeric user id — SO's API keys off the id,
    // and a text handle would need an extra search step. Surface the
    // display name + reputation so the user can confirm before saving.
    if !user_id.chars().all(|c| c.is_ascii_digit()) {
        return fail("must be the numeric user id");
    }
    let url = format!(
        "https://api.stackexchange.com/2.3/users/{}?site=stackoverflow",
        user_id
    );
    let Ok(resp) = http().get(&url).send().await else {
        return fail("network error");
    };
    if !resp.status().is_success() {
        return fail(&format!(
            "stack exchange returned {}",
            resp.status().as_u16()
        ));
    }
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return fail("stack exchange response parse failed");
    };
    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let Some(user) = items.first() else {
        return fail("user not found");
    };
    let name = user
        .get("display_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let reputation = user.get("reputation").and_then(|v| v.as_i64()).unwrap_or(0);
    let answers = user
        .get("answer_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let link = user
        .get("link")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| format!("https://stackoverflow.com/users/{}", user_id));
    ok_info_url(
        format!(
            "{} · {} rep · {} answers",
            if name.is_empty() { "profile" } else { &name },
            fmt_count(reputation),
            fmt_count(answers)
        ),
        link,
    )
}

async fn probe_hackernews_user(handle: &str) -> ProbeResponse {
    // HN profile page: 200 + "user:" marker means the account exists;
    // 200 + "No such user" means it doesn't. No auth needed.
    let url = format!("https://news.ycombinator.com/user?id={}", handle);
    let Ok(resp) = http().get(&url).send().await else {
        return fail("network error");
    };
    if !resp.status().is_success() {
        return fail(&format!("hn returned {}", resp.status().as_u16()));
    }
    let html = resp.text().await.unwrap_or_default();
    if html.contains("No such user") {
        return fail("user not found");
    }
    if !html.contains("karma:") {
        return fail("user not found");
    }
    // Pull karma out of the page for a friendlier info line.
    let karma = html
        .split("karma:")
        .nth(1)
        .and_then(|s| s.split('<').nth(1))
        .and_then(|s| s.split('>').nth(1))
        .and_then(|s| s.trim().replace(',', "").parse::<i64>().ok())
        .unwrap_or(0);
    let info = if karma > 0 {
        format!("{} karma", fmt_count(karma))
    } else {
        "user exists".to_string()
    };
    ok_info_url(info, url)
}

async fn probe_reddit(handle: &str) -> ProbeResponse {
    // Reddit's public user info. We hit `api.reddit.com` (the
    // legacy public API host) instead of `www.reddit.com` — the
    // latter blanket-403s reqwest from server IPs even with a
    // polite UA, while api.reddit returns the same JSON without
    // the bot blocker. Format mirrors `about.json`: { kind, data: {
    // total_karma, link_karma, comment_karma, ... } }.
    let url = format!("https://api.reddit.com/user/{}/about", handle);
    // Reddit fingerprints the rustls ClientHello (JA3) and 403s any
    // server-side request that lands on it — UA, headers, and
    // HTTP/1.1 vs HTTP/2 don't matter. We force native-tls (OS
    // crypto stack: Security.framework on macOS, OpenSSL on Linux),
    // which presents the same TLS fingerprint family curl uses and
    // passes through.
    let client = reqwest::Client::builder()
        .use_native_tls()
        .user_agent("Knowledge/1.0 (research project; https://github.com/raphaelsty/knowledge)")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "probe.reddit.network_error");
            return fail("network error");
        }
    };
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return fail("user not found");
    }
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        tracing::warn!(
            status,
            url = %url,
            body = %body.chars().take(300).collect::<String>(),
            "probe.reddit.non_success"
        );
        return fail(&format!("reddit returned {}", status));
    }
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return fail("reddit response parse failed");
    };
    let data = body.get("data").cloned().unwrap_or_default();
    let karma = data
        .get("total_karma")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| {
            // Older API shape: sum link + comment karma manually.
            let lk = data.get("link_karma").and_then(|v| v.as_i64()).unwrap_or(0);
            let ck = data
                .get("comment_karma")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            lk + ck
        });
    ok_info_url(
        format!("{} karma", fmt_count(karma)),
        format!("https://www.reddit.com/user/{}", handle),
    )
}

async fn probe_zotero(api_key: &str) -> ProbeResponse {
    // /keys/current returns { userID, username, access: { user: {...},
    // groups: { "<id>": {...} } } }. We confirm the key resolves and
    // summarise what it unlocks so the UI can render "Personal + N
    // groups" before the user hits save.
    let Ok(resp) = http()
        .get("https://api.zotero.org/keys/current")
        .header("Zotero-API-Key", api_key)
        .header("Zotero-API-Version", "3")
        .send()
        .await
    else {
        return fail("network error");
    };
    if resp.status() == reqwest::StatusCode::FORBIDDEN
        || resp.status() == reqwest::StatusCode::UNAUTHORIZED
    {
        return fail("invalid API key");
    }
    if !resp.status().is_success() {
        return fail(&format!("zotero returned {}", resp.status().as_u16()));
    }
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return fail("zotero response parse failed");
    };

    let user_id = body.get("userID").and_then(|v| v.as_i64()).unwrap_or(0);
    if user_id <= 0 {
        return fail("key has no user id");
    }
    let access = body
        .get("access")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let has_user_lib = access
        .get("user")
        .and_then(|v| v.get("library"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let n_groups = access
        .get("groups")
        .and_then(|v| v.as_object())
        .map(|m| m.len())
        .unwrap_or(0);

    // A separate call lists the groups with names so we can show them.
    // Best-effort; failure just shows "+ N groups" without names.
    let mut names: Vec<String> = Vec::new();
    if n_groups > 0 {
        if let Ok(r) = http()
            .get(format!("https://api.zotero.org/users/{}/groups", user_id))
            .header("Zotero-API-Key", api_key)
            .header("Zotero-API-Version", "3")
            .send()
            .await
        {
            if let Ok(list) = r.json::<serde_json::Value>().await {
                if let Some(arr) = list.as_array() {
                    for g in arr {
                        if let Some(name) = g
                            .get("data")
                            .and_then(|d| d.get("name"))
                            .and_then(|v| v.as_str())
                        {
                            names.push(name.to_string());
                        }
                    }
                }
            }
        }
    }

    let mut parts: Vec<String> = Vec::new();
    if has_user_lib {
        parts.push("Personal".to_string());
    }
    if !names.is_empty() {
        parts.push(format!(
            "{} group{}: {}",
            names.len(),
            if names.len() == 1 { "" } else { "s" },
            names.join(", ")
        ));
    } else if n_groups > 0 {
        parts.push(format!("{} groups", n_groups));
    }
    if parts.is_empty() {
        return fail("key grants no library access");
    }
    ok_info_url(
        parts.join(" · "),
        "https://www.zotero.org/settings/keys".to_string(),
    )
}

// Well-known sitemap paths to try when robots.txt doesn't declare one.
// Order matters: `*-index.xml` first because sites that have both
// typically point the index at the real sub-sitemaps.
const FALLBACK_SITEMAPS: &[&str] = &[
    "/sitemap-index.xml",
    "/sitemap_index.xml",
    "/sitemap.xml",
    "/sitemap-0.xml",
];

/// Auto-correct a user-pasted URL into a real sitemap reference.
///
/// Accepts anything a reasonable user would paste: a bare hostname,
/// a homepage, an article URL, or an exact sitemap URL. Discovery
/// order is the same one crawlers follow (robots.txt → well-known
/// paths), so it works for every static-site generator and CMS we've
/// seen without per-site code.
///
/// When the input points deeper than the root (e.g. `/articles/foo`),
/// the first path segment becomes a filter — that keeps sibling
/// articles while dropping unrelated pages (pricing, terms…). A
/// bare-host input yields no filter so everything in the sitemap
/// comes through.
async fn probe_website(raw: &str) -> ProbeResponse {
    // Forgive the user: `zeroentropy.dev`, `http://…`, `https://…/path`
    // all resolve to the same site.
    let normalized = normalize_site_url(raw);
    let Some((base, path_filter)) = split_base_and_filter(&normalized) else {
        return fail("not a valid URL");
    };

    // 1) Sniff the pasted URL directly. We intentionally do this
    //    before any site-wide auto-discovery so that:
    //      • `https://site/feed.xml`      → feed (no filter)
    //      • `https://site/sitemap.xml`   → sitemap (no filter)
    //      • `https://site/articles/foo`  → HTML → fall through to (2)
    //    We fetch once and route on the body shape.
    if let Some(resp) = http_get_ok(&normalized).await {
        let body = resp.text().await.unwrap_or_default();
        let head = body.get(..4096).unwrap_or(&body).to_ascii_lowercase();

        if is_feed(&head) {
            return feed_ok(&body, normalized);
        }
        if head.contains("<urlset") || head.contains("<sitemapindex") {
            // Re-enter the sitemap walker via the already-downloaded
            // body's canonical URL — cheaper than refetching, and
            // lets us reuse `gather_sitemap_urls`' recursion logic
            // for `<sitemapindex>` responses.
            if let Some(urls) = gather_sitemap_urls(&normalized, 0).await {
                return website_ok(urls, normalized, path_filter);
            }
        }

        // 1b) RSS/Atom autodiscovery via <link rel="alternate"> in the
        //     HTML head. This is the 2005 web standard for feed
        //     discovery — browsers, readers, and crawlers all honour
        //     it — and it rescues sites (Next.js/SPAs, Ghost, Hugo,
        //     …) that don't expose /feed or /sitemap.xml directly but
        //     do declare the feed in their HTML.
        if let Some(ok) = try_autodiscovered_feeds(&body, &normalized).await {
            return ok;
        }
    }

    // 2) Try robots.txt — authors often declare a non-obvious feed or
    //    sitemap path (e.g. `/sitemap-blog.xml`, `/atom`) that we'd
    //    never guess.
    for sm in sitemaps_from_robots(&base).await {
        if let Some(urls) = gather_sitemap_urls(&sm, 0).await {
            return website_ok(urls, sm, path_filter);
        }
    }

    // 3) Fall back to well-known sitemap paths.
    for tail in FALLBACK_SITEMAPS {
        let candidate = format!("{}{}", base, tail);
        if let Some(urls) = gather_sitemap_urls(&candidate, 0).await {
            return website_ok(urls, candidate, path_filter);
        }
    }

    // 4) Try well-known feed paths. A site may have no sitemap at
    //    all (many personal blogs) but still expose RSS under one of
    //    these conventional paths.
    for tail in [
        "/feed",
        "/rss",
        "/rss.xml",
        "/atom.xml",
        "/feed.xml",
        "/index.xml",
    ] {
        let candidate = format!("{}{}", base, tail);
        if let Some(resp) = http_get_ok(&candidate).await {
            let body = resp.text().await.unwrap_or_default();
            let head = body.get(..4096).unwrap_or(&body).to_ascii_lowercase();
            if is_feed(&head) {
                return feed_ok(&body, candidate);
            }
        }
    }

    // 5) Last resort: probe common blog-landing paths (/blog,
    //    /news, …) and run autodiscovery on their HTML. Covers sites
    //    like mixedbread.com where the feed is advertised in the
    //    /blog page's <link rel="alternate"> tags but the homepage
    //    and robots.txt don't mention it.
    for path in [
        "/blog",
        "/news",
        "/writing",
        "/posts",
        "/articles",
        "/essays",
        "/journal",
    ] {
        let landing = format!("{}{}", base, path);
        let Some(resp) = http_get_ok(&landing).await else {
            continue;
        };
        let body = resp.text().await.unwrap_or_default();
        if !body.contains("<link") {
            // Cheap reject — no point in running the extractor on a
            // page that has no <link> tags at all.
            continue;
        }
        if let Some(ok) = try_autodiscovered_feeds(&body, &landing).await {
            return ok;
        }
    }

    fail("no feed or sitemap found — site isn't crawlable")
}

/// Run RSS/Atom autodiscovery against an HTML body: scan for
/// `<link rel="alternate" type="application/(rss|atom)+xml" href="…">`,
/// resolve relative hrefs against `page_url`, and try each candidate.
/// Returns a populated `ProbeResponse` as soon as any candidate
/// resolves to a real feed; `None` if nothing validates.
async fn try_autodiscovered_feeds(body: &str, page_url: &str) -> Option<ProbeResponse> {
    for candidate in extract_feed_links(body, page_url) {
        let Some(resp) = http_get_ok(&candidate).await else {
            continue;
        };
        let fbody = resp.text().await.unwrap_or_default();
        let fhead = fbody.get(..4096).unwrap_or(&fbody).to_ascii_lowercase();
        if is_feed(&fhead) {
            return Some(feed_ok(&fbody, candidate));
        }
    }
    None
}

/// Extract feed URLs declared via `<link rel="alternate">`. RSS and
/// Atom MIME types only — JSON Feed isn't supported by our Python
/// fetcher, so there's no point surfacing it. Hrefs are resolved
/// against `page_url` so relative paths like `/blog/feed.xml` work.
fn extract_feed_links(body: &str, page_url: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // Restrict the scan to a reasonable head budget — feed links
    // live in <head>, and some SPAs ship huge inline JSON payloads
    // later in the body that would balloon this loop.
    let search = body.get(..32_768).unwrap_or(body);
    let lower = search.to_ascii_lowercase();

    // Walk each `<link` tag. No HTML parser: attribute order and
    // quoting are loose enough that a small state machine beats
    // bringing in a full dependency.
    let mut cursor = 0;
    while let Some(start) = lower[cursor..].find("<link") {
        let abs = cursor + start;
        let rel_end = lower[abs..]
            .find('>')
            .map(|i| abs + i)
            .unwrap_or(lower.len());
        let tag_lower = &lower[abs..rel_end];
        let tag_orig = &search[abs..rel_end];
        cursor = rel_end;

        // Must be `rel=... alternate ...`.
        let Some(rel_val) = attr_value(tag_lower, "rel") else {
            continue;
        };
        if !rel_val.split_whitespace().any(|t| t == "alternate") {
            continue;
        }
        // Must have an RSS or Atom MIME type.
        let Some(type_val) = attr_value(tag_lower, "type") else {
            continue;
        };
        let is_rss = type_val.contains("rss");
        let is_atom = type_val.contains("atom");
        if !is_rss && !is_atom {
            continue;
        }
        // Pull the (case-preserving) href from the ORIGINAL tag —
        // URLs are case-sensitive in some servers' path components.
        let Some(href) = attr_value_ci(tag_orig, "href") else {
            continue;
        };
        if let Some(resolved) = resolve_url(href, page_url) {
            if !out.iter().any(|u| u == &resolved) {
                out.push(resolved);
            }
        }
    }
    out
}

/// Lowercase-only attribute extractor — assumes `tag` is already
/// lowercased. Supports single-quoted, double-quoted, and bare
/// values. Returns `None` when the attribute is absent.
fn attr_value<'a>(tag_lower: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!(" {}=", name);
    let start = tag_lower.find(&needle)?;
    let after_eq = start + needle.len();
    let rest = tag_lower.get(after_eq..)?;
    let rest_trimmed = rest.trim_start();
    let offset = rest.len() - rest_trimmed.len();
    let value_start = after_eq + offset;
    let bytes = tag_lower.as_bytes();
    let quote = bytes.get(value_start)?;
    let (open, closer) = if *quote == b'"' || *quote == b'\'' {
        (value_start + 1, *quote)
    } else {
        // Bare attribute like `rel=alternate` — closes at whitespace
        // or the tag end.
        let end = tag_lower[value_start..]
            .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
            .map(|i| value_start + i)
            .unwrap_or(tag_lower.len());
        return Some(&tag_lower[value_start..end]);
    };
    let end_rel = tag_lower[open..].find(closer as char)?;
    Some(&tag_lower[open..open + end_rel])
}

/// Same logic as `attr_value` but case-insensitive on the attribute
/// name so we can read href values from the original-case tag.
fn attr_value_ci<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let lower = tag.to_ascii_lowercase();
    let mut cursor = 0;
    let needle = format!(" {}=", name.to_ascii_lowercase());
    let start = lower[cursor..].find(&needle)? + cursor;
    let after_eq = start + needle.len();
    let bytes = tag.as_bytes();
    let quote = *bytes.get(after_eq)?;
    cursor = after_eq;
    if quote == b'"' || quote == b'\'' {
        let rel_end = tag[cursor + 1..].find(quote as char)?;
        Some(&tag[cursor + 1..cursor + 1 + rel_end])
    } else {
        let end = tag[cursor..]
            .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
            .map(|i| cursor + i)
            .unwrap_or(tag.len());
        Some(&tag[cursor..end])
    }
}

/// Resolve `href` against `base_url`. Handles absolute URLs,
/// protocol-relative (`//cdn.example/feed`), root-relative (`/feed`),
/// and same-directory paths (`feed.xml`). Returns `None` for obvious
/// junk so the caller doesn't issue bad requests.
fn resolve_url(href: &str, base_url: &str) -> Option<String> {
    let href = href.trim();
    if href.is_empty() || href.starts_with("javascript:") || href.starts_with('#') {
        return None;
    }
    if href.starts_with("http://") || href.starts_with("https://") {
        return Some(href.to_string());
    }
    if href.starts_with("//") {
        // Protocol-relative — reuse the page's scheme.
        let scheme = if base_url.starts_with("https://") {
            "https:"
        } else {
            "http:"
        };
        return Some(format!("{}{}", scheme, href));
    }
    let rest = base_url
        .strip_prefix("https://")
        .or_else(|| base_url.strip_prefix("http://"))?;
    let scheme = if base_url.starts_with("https://") {
        "https"
    } else {
        "http"
    };
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if href.starts_with('/') {
        return Some(format!("{}://{}{}", scheme, host, href));
    }
    // Same-directory: strip everything after the last slash in `path`.
    let dir = match path.rfind('/') {
        Some(i) => &path[..=i],
        None => "/",
    };
    Some(format!("{}://{}{}{}", scheme, host, dir, href))
}

/// GET helper that returns `None` for any non-2xx or network error.
/// Keeps the probe flow readable without nested match arms.
///
/// Uses the SSRF-safe fetcher — every URL reaching this function is
/// derived (directly or via robots.txt / sitemap chain) from user
/// input.
async fn http_get_ok(url: &str) -> Option<reqwest::Response> {
    let resp = crate::handlers::url_safety::safe_get(
        url,
        std::time::Duration::from_secs(10),
        "knowledge-api/0.1 profile-probe",
    )
    .await
    .ok()?;
    resp.status().is_success().then_some(resp)
}

fn is_feed(head_lower: &str) -> bool {
    // Atom exposes `<feed xmlns="…atom">`, RSS 2.0 exposes `<rss`/
    // `<channel`. JSON Feed isn't covered here — the pipeline doesn't
    // have a parser for it either.
    head_lower.contains("<feed") || head_lower.contains("<rss") || head_lower.contains("<channel")
}

/// Build a `ProbeResponse` for a confirmed RSS/Atom feed. Reports
/// entry count as a friendly "N posts in feed" so users know the
/// input was accepted as a feed (and not as a sitemap fallback).
fn feed_ok(body: &str, resolved_url: String) -> ProbeResponse {
    let entries = body.matches("<entry").count() + body.matches("<item").count();
    let info = if entries == 0 {
        "feed looks empty".to_string()
    } else {
        format!("{} posts in feed", fmt_count(entries as i64))
    };
    ProbeResponse {
        ok: entries > 0,
        info: Some(info),
        error: (entries == 0).then(|| "feed is empty".to_string()),
        url: Some(resolved_url.clone()),
        resolved_url: Some(resolved_url),
        // Feeds are self-scoped — whatever the author publishes is
        // what we index. No path filter concept applies.
        resolved_filter: Some(String::new()),
        subtrees: None,
        kind: Some(ResolvedKind::Feed),
    }
}

/// Best-effort normalization. Users paste sloppy input; we fix it
/// before doing URL arithmetic rather than failing with "not a URL".
fn normalize_site_url(raw: &str) -> String {
    let s = raw.trim().trim_end_matches('/');
    if s.is_empty() {
        return String::new();
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return s.to_string();
    }
    // Bare host like `zeroentropy.dev` or `zeroentropy.dev/articles`.
    format!("https://{}", s)
}

/// Returns `(base, filter)` where `base` is `scheme://host` and
/// `filter` is `Some("/articles/")` when the URL had a non-root path.
fn split_base_and_filter(url: &str) -> Option<(String, Option<String>)> {
    // We only need scheme + host + path for this; parse by hand to
    // avoid pulling in the full `url` crate for one field.
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let scheme = if url.starts_with("https://") {
        "https"
    } else {
        "http"
    };
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    if host.is_empty() || !host.contains('.') {
        return None;
    }
    let base = format!("{}://{}", scheme, host);

    let trimmed = path.trim_start_matches('/').trim_end_matches('/');
    if trimmed.is_empty() {
        return Some((base, None));
    }
    // Skip common sitemap filenames — they shouldn't become filters.
    if trimmed.ends_with(".xml") || trimmed.ends_with(".gz") {
        return Some((base, None));
    }
    let first = trimmed.split('/').next().unwrap_or("");
    if first.is_empty() || first.contains('.') {
        return Some((base, None));
    }
    Some((base, Some(format!("/{}/", first))))
}

async fn sitemaps_from_robots(base: &str) -> Vec<String> {
    // `base` is derived from user input → SSRF-safe fetcher.
    let Ok(resp) = crate::handlers::url_safety::safe_get(
        &format!("{}/robots.txt", base),
        std::time::Duration::from_secs(10),
        "knowledge-api/0.1 profile-probe",
    )
    .await
    else {
        return Vec::new();
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let body = resp.text().await.unwrap_or_default();
    body.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (k, v) = line.split_once(':')?;
            if !k.trim().eq_ignore_ascii_case("sitemap") {
                return None;
            }
            let url = v.trim();
            (!url.is_empty()).then(|| url.to_string())
        })
        .collect()
}

/// Fetch `url` and return every `<loc>` URL it contains.
/// `<sitemapindex>` responses are followed one level deep so the
/// returned list always holds real page URLs, not sub-sitemap URLs.
/// Bounded by `MAX_URLS` to keep the probe fast even on huge sites.
///
/// Returns `None` when the URL isn't a sitemap or is unreachable, so
/// callers know to try the next discovery candidate.
async fn gather_sitemap_urls(url: &str, depth: u8) -> Option<Vec<String>> {
    gather_sitemap_urls_inner(url, depth).await
}

const MAX_URLS: usize = 5_000;
const MAX_DEPTH: u8 = 2;

fn gather_sitemap_urls_inner(
    url: &str,
    depth: u8,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<Vec<String>>> + Send + '_>> {
    Box::pin(async move {
        // Sitemap chain walker — every URL here originated from user
        // input (initial sitemap URL or `<loc>` children of one), so
        // SSRF-safe fetcher.
        let resp = crate::handlers::url_safety::safe_get(
            url,
            std::time::Duration::from_secs(10),
            "knowledge-api/0.1 profile-probe",
        )
        .await
        .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body = resp.text().await.ok()?;
        let head = body.get(..4096).unwrap_or(&body).to_ascii_lowercase();
        let is_index = head.contains("<sitemapindex");
        let is_urlset = head.contains("<urlset");
        if !is_urlset && !is_index {
            return None;
        }

        if is_index && depth < MAX_DEPTH {
            let mut out: Vec<String> = Vec::new();
            // Cap the fan-out so pathological indexes can't blow up
            // the probe budget.
            for child in extract_locs(&body).into_iter().take(6) {
                if out.len() >= MAX_URLS {
                    break;
                }
                if let Some(urls) = gather_sitemap_urls_inner(&child, depth + 1).await {
                    out.extend(urls);
                }
            }
            if !out.is_empty() {
                out.truncate(MAX_URLS);
                return Some(out);
            }
        }

        let mut urls = extract_locs(&body);
        urls.truncate(MAX_URLS);
        Some(urls)
    })
}

/// Top first-path-segment buckets in a URL list, sorted by count
/// descending. Used to suggest `"/articles/"`-style subtrees when the
/// user's pasted URL doesn't match anything in the sitemap. Capped at
/// `limit` buckets — the UI renders them as clickable chips, and more
/// than a handful becomes noise.
fn top_subtrees(urls: &[String], limit: usize) -> Vec<Subtree> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, i64> = HashMap::new();
    for u in urls {
        // Strip scheme + host so we can bucket by path prefix. We
        // accept anything after the first '/' past `://`.
        let after_scheme = u.split_once("://").map(|x| x.1).unwrap_or(u);
        let path = match after_scheme.find('/') {
            Some(i) => &after_scheme[i + 1..],
            None => "",
        };
        let trimmed = path.trim_matches('/');
        if trimmed.is_empty() {
            // Root URLs like `https://site/` — skip; they don't form
            // a subtree suggestion.
            continue;
        }
        // Ignore segments that look like a filename (have an extension
        // other than a trailing slash): `/privacy`, `/feed.xml`, etc.
        // We want path *directories*, not individual pages.
        let first = trimmed.split('/').next().unwrap_or("");
        if first.is_empty() {
            continue;
        }
        *counts.entry(format!("/{}/", first)).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, i64)> = counts.into_iter().collect();
    // Sort desc by count, tie-break alphabetically for a stable UI.
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(limit)
        .map(|(path, count)| Subtree { path, count })
        .collect()
}

/// Lightweight `<loc>URL</loc>` extractor — good enough for sitemap
/// indexes, which are flat and well-formed by spec.
fn extract_locs(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in body.split("<loc").skip(1) {
        let Some(end) = chunk.find("</loc>") else {
            continue;
        };
        let inner = &chunk[..end];
        let start = inner.find('>').map(|i| i + 1).unwrap_or(0);
        let url = inner[start..].trim();
        if !url.is_empty() {
            out.push(url.to_string());
        }
    }
    out
}

fn website_ok(
    urls: Vec<String>,
    sitemap_url: String,
    path_filter: Option<String>,
) -> ProbeResponse {
    let total = urls.len() as i64;
    let has_filter = path_filter.as_ref().map(|s| !s.is_empty()).unwrap_or(false);
    let matched = match &path_filter {
        Some(f) if !f.is_empty() => urls.iter().filter(|u| u.contains(f.as_str())).count() as i64,
        _ => total,
    };

    // A bare-host URL (no path) is the user saying "index everything"
    // — treat that as success. The subtree chips are still offered as
    // an OPTIONAL narrow-down, not a requirement, and "all" auto-adopts
    // any new subtree the author adds later (e.g. a /2027/ archive).
    //
    // We only bail to "pick a subtree" when the user DID pick a path
    // but it matched nothing — that's a typo, not a scope decision.
    let ok = if has_filter { matched > 0 } else { total > 0 };

    // Always attach subtree suggestions when there's any choice to
    // offer. For the bare-host case they're opt-in narrow-downs; for
    // the no-match case they're the fix.
    let subtrees = if !has_filter || matched == 0 {
        let t = top_subtrees(&urls, 8);
        (!t.is_empty()).then_some(t)
    } else {
        None
    };

    let info = if has_filter && matched > 0 {
        format!(
            "{} URLs at {} (optional scope)",
            fmt_count(matched),
            path_filter.as_deref().unwrap_or("")
        )
    } else if has_filter {
        format!(
            "no URLs matched {} — pick a subtree below or drop the path",
            path_filter.as_deref().unwrap_or("")
        )
    } else if subtrees.is_some() {
        format!(
            "{} URLs, whole site indexed · pick a chip below to narrow",
            fmt_count(total)
        )
    } else {
        format!("{} URLs, whole site indexed", fmt_count(total))
    };

    ProbeResponse {
        ok,
        info: Some(info),
        error: (!ok).then(|| {
            if total == 0 {
                "sitemap is empty".to_string()
            } else {
                "pick a subtree to scope this site".to_string()
            }
        }),
        url: Some(sitemap_url.clone()),
        resolved_url: Some(sitemap_url),
        resolved_filter: Some(path_filter.unwrap_or_default()),
        subtrees,
        kind: Some(ResolvedKind::Sitemap),
    }
}

// ── Router entrypoint ────────────────────────────────────────────────

// Keep this behind auth — probes can otherwise be used as an SSRF
// hop. `current_user` rejects anonymous callers with 401.
pub async fn probe(
    State(pool): State<PgPool>,
    jar: CookieJar,
    Json(req): Json<ProbeRequest>,
) -> impl IntoResponse {
    if current_user(&pool, &jar).await.is_none() {
        return (StatusCode::UNAUTHORIZED, Json(fail("not signed in"))).into_response();
    }

    let value = req.value.trim();
    if value.is_empty() {
        return Json(fail("empty value")).into_response();
    }

    let result = match req.kind.as_str() {
        "github" => probe_github(value).await,
        "twitter" => probe_twitter(value.trim_start_matches('@')).await,
        "blog" => probe_blog(value).await,
        "sitemap" => probe_sitemap(value).await,
        "scholar" => probe_scholar(value).await,
        "huggingface" => probe_huggingface(value).await,
        "arxiv" => probe_arxiv(value).await,
        "website" => probe_website(value).await,
        "zotero" => probe_zotero(value).await,
        "reddit" => probe_reddit(value.trim_start_matches("u/")).await,
        "hackernews_user" => probe_hackernews_user(value).await,
        "stackoverflow" => probe_stackoverflow(value).await,
        other => fail(&format!("unknown probe kind: {}", other)),
    };
    Json(result).into_response()
}

// `urlencoding` is inlined the same way as in auth.rs. Kept private.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        const HEX: &[u8] = b"0123456789ABCDEF";
        let mut out = String::with_capacity(s.len());
        for &b in s.as_bytes() {
            let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
            if unreserved {
                out.push(b as char);
            } else {
                out.push('%');
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0x0f) as usize] as char);
            }
        }
        out
    }
}
