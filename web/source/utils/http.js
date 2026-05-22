// Shared HTTP helpers used by every JS source fetcher.
//
// Mirrors the tiny wrapper each Python module grew its own copy of
// (urllib.request.urlopen with a User-Agent + JSON parse). Kept here
// so each source module can stay focused on its specific endpoint
// semantics.
//
// Cross-origin fallback: many of the URLs the sync fetchers hit
// (sitemaps, RSS feeds, blog homepages) are on third-party hosts
// that don't send `Access-Control-Allow-Origin`. The browser refuses
// to expose those responses to JS, so a direct `fetch()` fails with
// a TypeError ("Failed to fetch"). When we detect that, we retry the
// request through `${KNOWLEDGE_API}/api/proxy/fetch?url=…`, which
// fetches server-side (no CORS) and streams the body back with the
// session cookie.

const DEFAULT_UA = "Knowledge/1.0 (web sync)";

function apiBase() {
  if (typeof window === "undefined") return "";
  if (window.KNOWLEDGE_API) return window.KNOWLEDGE_API;
  // Dev convention: when served from localhost, the API runs on :8080.
  if (window.location && window.location.hostname === "localhost") {
    return "http://localhost:8080";
  }
  return "";
}

function proxyUrl(url) {
  return `${apiBase()}/api/proxy/fetch?url=${encodeURIComponent(url)}`;
}

/** Most reliable signal that a `fetch()` failure was a CORS / network
 *  block rather than e.g. a 500. Browsers throw TypeError with no
 *  status; we also catch the explicit AbortError so timeouts don't
 *  trigger the proxy fallback. */
function isCrossOriginError(err) {
  if (!err) return false;
  if (err.name === "AbortError") return false;
  if (err.name === "TypeError") return true;
  return /failed to fetch|network ?error|cors/i.test(
    String(err.message || err),
  );
}

async function rawFetch(url, { timeoutMs, headers, parser }) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      headers: { "User-Agent": DEFAULT_UA, ...headers },
      signal: ctrl.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
    return await parser(resp);
  } finally {
    clearTimeout(timer);
  }
}

/** Fetch via direct cross-origin first; on CORS/network failure,
 *  retry through the auth-required server-side proxy. */
async function fetchWithFallback(url, opts, parser) {
  try {
    return await rawFetch(url, { ...opts, parser });
  } catch (err) {
    if (!apiBase() || !isCrossOriginError(err)) throw err;
    // Same-origin proxy request — sends the session cookie so the
    // server can authenticate this user.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs);
    try {
      const resp = await fetch(proxyUrl(url), {
        credentials: "include",
        signal: ctrl.signal,
      });
      if (!resp.ok) {
        const detail = await resp.text().catch(() => resp.statusText);
        throw new Error(`proxy ${resp.status}: ${detail || "fetch failed"}`);
      }
      return await parser(resp);
    } finally {
      clearTimeout(timer);
    }
  }
}

/** Fetch a URL and parse the body as JSON. Throws on non-2xx. */
export async function fetchJson(url, { timeoutMs = 15000, headers = {} } = {}) {
  return fetchWithFallback(url, { timeoutMs, headers }, (r) => r.json());
}

/** Fetch a URL and return body text. Throws on non-2xx. */
export async function fetchText(url, { timeoutMs = 15000, headers = {} } = {}) {
  return fetchWithFallback(url, { timeoutMs, headers }, (r) => r.text());
}

/** Small delay helper; matches the per-source pacing in the Python code. */
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
