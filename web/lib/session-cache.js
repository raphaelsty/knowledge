/* Tiny TTL-tagged sessionStorage wrapper used by the feed + personal-
 * page caches.
 *
 * Knowledge's links are full page reloads, so the in-memory Map
 * caches in page.js (`_timelineCache`) and api.js
 * (`_personalDocsCache`) evaporate every time the user navigates.
 * That makes "bounce between feed and personal page" feel like two
 * cold loads. Mirror the same payloads into sessionStorage with a
 * short TTL so the next navigation paints instantly from cache; an
 * explicit refresh (pull-to-refresh, the "More" button) still calls
 * the API with `fresh: true` and bypasses both layers.
 *
 * sessionStorage (not localStorage) so the data dies with the tab —
 * a stale feed from days ago is worse than no cache. The TTL is
 * the second guard: anything older than the caller-supplied window
 * is treated as a miss and lazily evicted.
 */
(function () {
  "use strict";

  const PREFIX = "kn.cache:";

  function _read(key) {
    try {
      return sessionStorage.getItem(PREFIX + key);
    } catch {
      // Private mode / disabled storage / quota — treat as no cache.
      return null;
    }
  }
  function _write(key, value) {
    try {
      sessionStorage.setItem(PREFIX + key, value);
      return true;
    } catch {
      // Quota or storage off — silently drop. We never want a cache
      // write to break the live rendering path.
      return false;
    }
  }
  function _remove(key) {
    try {
      sessionStorage.removeItem(PREFIX + key);
    } catch {
      /* ignore */
    }
  }

  /** Return the cached value or null when missing / expired. */
  function get(key, ttlMs) {
    const raw = _read(key);
    if (!raw) return null;
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      _remove(key);
      return null;
    }
    if (!parsed || typeof parsed.at !== "number") {
      _remove(key);
      return null;
    }
    if (Date.now() - parsed.at > ttlMs) {
      _remove(key);
      return null;
    }
    return parsed.v;
  }

  /** Stash a value under `key`. JSON-serialised with a wall-clock stamp. */
  function set(key, value) {
    try {
      const payload = JSON.stringify({ at: Date.now(), v: value });
      // Skip absurdly large payloads — they'd blow the per-origin
      // sessionStorage budget (5–10 MB on most browsers) and the
      // hot-path callers aren't designed to gracefully recover. 4 MB
      // is a comfortable headroom against the feed's biggest payloads
      // we've observed (~2 MB).
      if (payload.length > 4_000_000) return;
      _write(key, payload);
    } catch {
      /* serialisation failure — just skip the cache write */
    }
  }

  /** Drop every entry whose key starts with `prefix`. */
  function invalidatePrefix(prefix) {
    const full = PREFIX + prefix;
    const toDrop = [];
    try {
      for (let i = 0; i < sessionStorage.length; i++) {
        const k = sessionStorage.key(i);
        if (k && k.startsWith(full)) toDrop.push(k);
      }
      for (const k of toDrop) sessionStorage.removeItem(k);
    } catch {
      /* ignore — storage may be unavailable */
    }
  }

  /** Drop every cache entry across every namespace. Useful for
   * sign-out / aggressive resets. */
  function clearAll() {
    const toDrop = [];
    try {
      for (let i = 0; i < sessionStorage.length; i++) {
        const k = sessionStorage.key(i);
        if (k && k.startsWith(PREFIX)) toDrop.push(k);
      }
      for (const k of toDrop) sessionStorage.removeItem(k);
    } catch {
      /* ignore */
    }
  }

  window.KnowledgeSessionCache = { get, set, invalidatePrefix, clearAll };
})();
