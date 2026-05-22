/* Tiny utilities shared by the page scripts.
 *
 * Loaded as a plain <script> before any consumer, so everything
 * here attaches to `window` and is read by global lookup from
 * inside the consumers' IIFEs.
 *
 * Keep this file small. Anything page-specific stays in that
 * page's own JS bundle.
 */

(function () {
  // Escape the five HTML-special characters so a string can be
  // dropped into innerHTML without opening an XSS hole. Treats
  // null / undefined as empty so call sites don't have to
  // pre-check. Both HTML-text and HTML-attribute contexts need the
  // same five characters, so escapeAttr is just an alias.
  const ESC = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  window.escapeHtml = function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ESC[c]);
  };
  window.escapeAttr = window.escapeHtml;

  // Absolute URL prefix for the Rust API. In production the API and
  // the static site sit behind the same Caddy, so paths like
  // `/auth/me` are same-origin and the prefix is empty. In dev the
  // static server lives on :3001 and the API on :8080, so requests
  // need the full http://localhost:8080 prefix.
  //
  // Honor a pre-set window.KNOWLEDGE_API_BASE so the page can
  // override it from a <script> earlier in the HTML if needed.
  if (typeof window.KNOWLEDGE_API_BASE !== "string") {
    const host = window.location.hostname;
    window.KNOWLEDGE_API_BASE =
      host === "localhost" || host === "127.0.0.1"
        ? "http://localhost:8080"
        : "";
  }

  // Tiny credentialed-JSON GET. Throws on non-2xx so callers can
  // `.catch()` and render a fallback. POST has too many shape
  // variants (body / no body, error-payload extraction) to be
  // worth a shared helper — leave that to per-page code.
  window.K_getJson = async function getJson(path) {
    const r = await fetch(`${window.KNOWLEDGE_API_BASE}${path}`, {
      credentials: "include",
    });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  };
})();
