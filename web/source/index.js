// Entry point for the browser sync feature. Exposes runSync and the
// registry on `window.KnowledgeSync` so the React bundle (plain
// script, not ES module) can call them from the Sync button.
//
// This module ALSO drives background autosync for signed-in non-VIPs:
// non-VIPs don't have the Python ingestion pipeline running for them
// server-side, so without a periodic browser-side refresh their
// libraries only grow when they manually click "Sync now". The
// autosync runs silently every ~30 min while any of their tabs are
// open — at most one in-flight sync per browser thanks to the
// localStorage throttle.

import { runSync } from "./sync.js";
import { REGISTRY, enabledFetchers } from "./registry.js";
import { hostnameSourceKey } from "./utils/hostname.js";

window.KnowledgeSync = {
  runSync,
  REGISTRY,
  enabledFetchers,
  hostnameSourceKey,
};

/* ── Background autosync (non-VIPs only) ──────────────────────────────
 *
 * Constants are intentionally generous: GitHub stars / HN submissions /
 * blog feeds don't change in seconds, and we'd rather under-sync than
 * hammer the API + every upstream's rate limiter. The localStorage
 * timestamp is shared across tabs, so opening five Knowledge tabs
 * doesn't multiply the request rate.
 */
const AUTOSYNC_LS_KEY = "kn.lastAutoSync";
const AUTOSYNC_INTERVAL_MS = 30 * 60 * 1000; // 30 min between passes
const AUTOSYNC_FIRST_DELAY_MS = 10 * 1000; // 10s after page load before first attempt
const AUTOSYNC_INFLIGHT_KEY = "kn.autoSyncInflight"; // cross-tab lock

function lsRead(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function lsWrite(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private mode / quota — ignore */
  }
}
function lsDelete(key) {
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}

async function fetchJsonSafe(url, opts) {
  try {
    const r = await fetch(url, opts);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

async function resolveSignedInNonVip(apiBase) {
  const me = await fetchJsonSafe(`${apiBase}/auth/me`, {
    credentials: "include",
  });
  if (!me || !me.slug) return null;
  const u = await fetchJsonSafe(
    `${apiBase}/api/users/${encodeURIComponent(me.slug)}`,
  );
  if (!u) return null;
  if (u.vip) return null;
  return { me, user: u };
}

async function runAutosyncOnce(apiBase) {
  const ctx = await resolveSignedInNonVip(apiBase);
  if (!ctx) return; // anonymous or VIP — nothing to do
  const sources = (ctx.user && ctx.user.sources) || {};
  if (!Object.keys(sources).length) return; // no sources wired yet

  // Cross-tab lock: another tab is mid-sync, skip this round entirely
  // and let the active tab finish. The lock has a TTL so a crashed tab
  // can't permanently wedge autosync — anything older than 10 min is
  // treated as stale and ignored.
  const inflightAt = parseInt(lsRead(AUTOSYNC_INFLIGHT_KEY) || "0", 10);
  if (inflightAt && Date.now() - inflightAt < 10 * 60_000) return;
  lsWrite(AUTOSYNC_INFLIGHT_KEY, String(Date.now()));

  try {
    const urlsArr = await fetchJsonSafe(`${apiBase}/auth/me/documents/urls`, {
      credentials: "include",
    });
    const existingUrls = new Set(Array.isArray(urlsArr) ? urlsArr : []);
    let inserted = 0;
    await runSync({
      sources,
      existingUrls,
      apiBase,
      onProgress(evt) {
        if (evt.type === "upload") inserted += evt.inserted || 0;
      },
    });
    lsWrite(AUTOSYNC_LS_KEY, String(Date.now()));
    if (inserted > 0) {
      // Drop the persistent sessionStorage caches directly — the
      // `knowledge:bookmark-added` event only invalidates them on
      // pages that listen for it (feed + personal). When autosync
      // fires on /profile or some other shell, the listener isn't
      // wired and the feed would otherwise hand back a stale payload
      // on next nav.
      window.KnowledgeSessionCache?.invalidatePrefix?.("timeline:");
      const slug = ctx.user?.slug;
      if (slug) {
        window.KnowledgeAPI?.invalidatePersonalDocs?.(slug);
        window.KnowledgeAPI?.invalidateUnindexed?.(slug);
      }
      // Nudge the feed / personal-page cache layer's in-memory Maps
      // when those pages are in front of the user — same event the
      // compose dialog and the manual Sync button fire so the rest
      // of the app doesn't need to know autosync exists.
      window.dispatchEvent(new CustomEvent("knowledge:bookmark-added"));
    }
  } catch {
    /* Silent failure by design — next interval will retry. Surfacing
     * background-sync errors as toasts would be noisy and rarely
     * actionable. */
  } finally {
    lsDelete(AUTOSYNC_INFLIGHT_KEY);
  }
}

function scheduleAutosync(apiBase) {
  // Throttle: if a sync happened in another tab within the interval,
  // sleep for the remainder before our first attempt.
  const lastAt = parseInt(lsRead(AUTOSYNC_LS_KEY) || "0", 10);
  const ageMs = Date.now() - lastAt;
  const initialDelay = Math.max(
    AUTOSYNC_FIRST_DELAY_MS,
    AUTOSYNC_INTERVAL_MS - ageMs,
  );

  const tick = async () => {
    if (document.visibilityState === "hidden") {
      // Defer until the tab is focused again — running sync while
      // hidden is fine in theory, but lots of browsers throttle
      // setTimeout in background tabs anyway and we'd rather avoid
      // burning fetcher rate limit on a tab the user abandoned.
      const onVisible = () => {
        document.removeEventListener("visibilitychange", onVisible);
        if (document.visibilityState === "visible") {
          runAutosyncOnce(apiBase);
        }
      };
      document.addEventListener("visibilitychange", onVisible);
      return;
    }
    await runAutosyncOnce(apiBase);
  };

  setTimeout(tick, initialDelay);
  setInterval(tick, AUTOSYNC_INTERVAL_MS);
}

// Boot. `KNOWLEDGE_API_BASE` is wired by lib/utils.js on every page
// that ships the search/profile shell — empty on prod (same-origin via
// Caddy), absolute on local dev.
const _apiBase =
  window.KNOWLEDGE_API_BASE != null ? window.KNOWLEDGE_API_BASE : "";
scheduleAutosync(_apiBase);
