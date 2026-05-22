/* Scroll-position memory across in-app navigation.
 *
 * Knowledge uses plain <a href="..."> links between the feed (/),
 * personal pages (/<slug>), the search shell (/search), and the
 * settings page (/profile). Every link triggers a full reload, so
 * window.scrollY resets even when the user is just bouncing between
 * two views they already loaded once. On mobile that costs a lot of
 * re-scrolling for no good reason.
 *
 * This module:
 *   1. Snapshots window.scrollY + the internal scrollTops of the
 *      three known scrollable rails (sources, people, library) on
 *      `pagehide` and stores them in sessionStorage, keyed by route.
 *   2. On the next load of the *same* route, restores those values
 *      once the result list has populated (polled with a budget so
 *      we don't sit on a forever-empty page).
 *
 * sessionStorage (not localStorage) so positions vanish when the
 * tab closes — the user expects "where I left off" to be a within-
 * session concept, not something that follows them across days.
 *
 * Hash-only navigation is ignored; the browser handles those
 * natively and we'd otherwise stomp the anchor jump.
 */
(function () {
  "use strict";

  const STORAGE_PREFIX = "kn.scroll:";
  // Per-route entry shape:
  //   { y: number, rails: { [selector]: number }, savedAt: number }
  // The selector keys match the global selectors below so the restore
  // step doesn't need to know which page rendered which rail — it
  // tries every selector and silently no-ops on absent elements.
  const RAIL_SELECTORS = [
    "#peopleRailList",
    "#grpSrc .group-body",
    "#grpLibs .group-body",
  ];
  // Drop entries older than this on read — a stale snapshot from
  // hours ago is almost never what the user wants to come back to.
  const TTL_MS = 60 * 60 * 1000; // 1h
  // Hard cap on the wait-for-results loop. Knowledge's feed paints
  // within ~1.5s on a warm cache; allow some headroom for cold loads.
  const RESTORE_TIMEOUT_MS = 4000;
  const RESTORE_POLL_MS = 80;

  function routeKey() {
    // Path + search, no hash. Same route across hash-only changes so
    // the memory stays useful when the user clicks a same-page anchor.
    return STORAGE_PREFIX + location.pathname + location.search;
  }

  function ssRead(key) {
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
  function ssWrite(key, value) {
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* quota / private mode — ignore */
    }
  }
  function ssDelete(key) {
    try {
      sessionStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  }

  function snapshot() {
    const rails = {};
    for (const sel of RAIL_SELECTORS) {
      const el = document.querySelector(sel);
      if (el && typeof el.scrollTop === "number" && el.scrollTop > 0) {
        rails[sel] = el.scrollTop;
      }
    }
    return {
      y: window.scrollY || 0,
      rails,
      savedAt: Date.now(),
    };
  }

  function save() {
    const snap = snapshot();
    // No point persisting a top-of-page state — that's the default,
    // and storing it would just overwrite a useful older snapshot
    // when the user reloads the page without actually scrolling.
    if (snap.y === 0 && Object.keys(snap.rails).length === 0) {
      ssDelete(routeKey());
      return;
    }
    ssWrite(routeKey(), snap);
  }

  function restore() {
    const entry = ssRead(routeKey());
    if (!entry) return;
    if (Date.now() - (entry.savedAt || 0) > TTL_MS) {
      ssDelete(routeKey());
      return;
    }
    // Tell the browser we'll handle restoration manually — otherwise
    // a back/forward could fight our explicit scrollTo on top of its
    // own restore.
    try {
      history.scrollRestoration = "manual";
    } catch {
      /* unsupported browser — best-effort fallthrough */
    }

    // Wait until the results list has SOMETHING in it before we try
    // to scroll. Without this the feed scrolls to a Y that exceeds
    // the current page height and the browser clamps it to top.
    const start = performance.now();
    const tick = () => {
      const list = document.getElementById("results");
      const ready =
        // Results painted, OR a no-results empty state showed up,
        // OR we ran out of patience.
        (list && list.childElementCount > 0) ||
        document.getElementById("empty")?.style?.display === "" ||
        performance.now() - start > RESTORE_TIMEOUT_MS;
      if (!ready) {
        setTimeout(tick, RESTORE_POLL_MS);
        return;
      }
      // Page-level scroll. Use 'auto' (instant) instead of 'smooth'
      // so the user lands exactly where they were — a smooth scroll
      // would visibly animate from top, which reads as a glitch.
      try {
        window.scrollTo({ top: entry.y, left: 0, behavior: "auto" });
      } catch {
        window.scrollTo(0, entry.y);
      }
      // Per-rail scroll. Skipped silently when the rail doesn't
      // exist on the current page (e.g. /profile has no rails).
      for (const sel of Object.keys(entry.rails || {})) {
        const el = document.querySelector(sel);
        if (el) el.scrollTop = entry.rails[sel] || 0;
      }
    };
    tick();
  }

  // Snapshot on every plausible "leaving the page" moment. pagehide
  // is the canonical one (works for bfcache + plain unload); the
  // two beforeunload + visibilitychange handlers exist as backups
  // because Safari sometimes withholds pagehide.
  window.addEventListener("pagehide", save, { capture: true });
  window.addEventListener("beforeunload", save, { capture: true });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") save();
  });

  // Restore once the DOM is ready — the wait-for-results loop above
  // handles the async result render.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", restore, { once: true });
  } else {
    restore();
  }
})();
