// Browser-side Zotero fetcher.
//
// The user's Zotero API key is AES-encrypted at rest on the server
// (the encryption key never leaves the server), so we can't talk to
// `api.zotero.org` directly from the browser. Instead we hit our own
// `/auth/me/zotero/items` proxy: it auths via the session cookie,
// decrypts the key in-memory, paginates through every library
// (personal + each discovered group), normalises rows to our doc
// shape, and returns the flat array. We then dedupe against the URLs
// the user already owns so re-syncs don't ship duplicates back to
// the bulk-upload endpoint.

export async function items({ apiBase = "", existingUrls = null } = {}) {
  // Direct fetch with credentials — the shared `fetchJson` helper
  // omits cookies, which works for the public third-party endpoints
  // it usually wraps but breaks here: `/auth/me/zotero/items` is
  // session-gated and would 401 without the cookie. In dev the API
  // runs on a different port (cross-origin), so we must opt in to
  // credentials explicitly.
  let resp;
  try {
    resp = await fetch(`${apiBase}/auth/me/zotero/items`, {
      credentials: "include",
    });
  } catch (err) {
    throw new Error(err && err.message ? err.message : String(err));
  }
  if (!resp.ok) {
    const detail = await resp.text().catch(() => resp.statusText);
    throw new Error(`zotero proxy ${resp.status}: ${detail || "fetch failed"}`);
  }
  const raw = await resp.json().catch(() => null);
  if (!Array.isArray(raw)) return {};

  const out = {};
  for (const d of raw) {
    if (!d || !d.url) continue;
    if (existingUrls && existingUrls.has(d.url)) continue;
    // No `source` key here — the orchestrator's URL-routing layer
    // (`applyUrlSourceOverrides` in sync.js) buckets each Zotero item
    // by its content type: arxiv → "arxiv", youtube → "youtube",
    // a hostname like `aclanthology.org` → "aclanthology.org", etc.
    // Stamping "zotero" here would mask all that.
    out[d.url] = {
      title: d.title || "",
      summary: d.summary || "",
      date: d.date || "",
      tags: Array.isArray(d.tags) ? d.tags : [],
    };
  }
  return out;
}
