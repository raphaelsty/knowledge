// Behavioural-event tracker. Posts batched events to POST /events; the
// server fans them into the `events` table where they become training
// data for recommendation + ranking models.
//
// Public API exposed on `window.kn` (legacy global naming used elsewhere):
//
//   kn.track(type, payload)        — buffer an event
//   kn.setViewer({ id })           — remember the signed-in user
//   kn.setPersonality({ id, slug }) — remember which library is being browsed
//   kn.setLastQuery(query)         — remember the most recent search query
//   kn.flush()                     — push the buffer immediately (idempotent)
//
// Flush triggers:
//   - 30 events buffered
//   - 8 s since the oldest buffered event
//   - pagehide / visibilitychange:hidden (sendBeacon — survives navigation)
//
// Field shape matches `api/src/handlers/events.rs::EventPayload`.

(function () {
  if (window.kn) return; // idempotent — `<script>` may load twice during dev
  const kn = (window.kn = {});

  // ── Session id (UUID v4, persisted per tab) ───────────────────────
  // sessionStorage scopes it to the tab; closing/reopening starts a
  // fresh session, which matches how the analytics dashboards group
  // events into sessions.
  const SK = "kn.session_id";
  let sessionId = sessionStorage.getItem(SK);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SK, sessionId);
  }

  // ── Mutable context (set as the user navigates) ───────────────────
  let viewerId = null; // logged-in user id, or null for anon
  let personalityId = null; // user_id of the library being browsed
  let personalitySlug = null;
  let lastQuery = null;
  let lastSourceFilter = null;
  let lastSortMode = null;

  // ── Buffer + flush plumbing ───────────────────────────────────────
  const MAX_BUFFER = 30;
  const MAX_AGE_MS = 8000;
  const ENDPOINT = `${window.KNOWLEDGE_API_BASE || ""}/events`;

  let buffer = [];
  let flushTimer = null;
  let firstEventAt = 0;

  function deviceType() {
    return window.matchMedia && window.matchMedia("(max-width: 768px)").matches
      ? "mobile"
      : "desktop";
  }
  function referrerDomain() {
    try {
      return document.referrer ? new URL(document.referrer).hostname : null;
    } catch {
      return null;
    }
  }
  // Session-scoped flags so device/referrer only ride along with the
  // first event of the tab — the server only reads them on session
  // upsert and ignores them after.
  let sessionStamped = false;

  function scheduleFlush() {
    if (flushTimer) return;
    flushTimer = setTimeout(() => {
      flushTimer = null;
      kn.flush();
    }, MAX_AGE_MS);
  }

  kn.track = function track(type, payload = {}) {
    // Library id is the only hard requirement on the server. Fall back
    // to the remembered personality if the call site didn't pass one.
    const userId = payload.user_id ?? personalityId;
    if (!userId) return; // nothing to attribute to — drop silently

    const ev = {
      session_id: sessionId,
      event_type: type,
      payload: {
        user_id: userId,
        personality_slug: payload.personality_slug ?? personalitySlug,
        viewer_user_id: payload.viewer_user_id ?? viewerId,
        client_ts: new Date().toISOString(),
        // Click events without an explicit query inherit the last search
        // query — that's the signal recommenders care about.
        query: payload.query ?? (type === "click" ? lastQuery : undefined),
        source_filter: payload.source_filter ?? lastSourceFilter,
        sort_mode: payload.sort_mode ?? lastSortMode,
        result_count: payload.result_count,
        latency_ms: payload.latency_ms,
        doc_url: payload.doc_url,
        position: payload.position,
        score: payload.score,
      },
    };
    // Only stamp device + referrer on the first event of a session —
    // the server ignores them after the session upsert anyway, but
    // shipping less per event keeps the payload tight.
    if (!sessionStamped) {
      ev.payload.device_type = deviceType();
      ev.payload.referrer_domain = referrerDomain();
      sessionStamped = true;
    }

    if (buffer.length === 0) firstEventAt = Date.now();
    buffer.push(ev);

    if (
      buffer.length >= MAX_BUFFER ||
      Date.now() - firstEventAt >= MAX_AGE_MS
    ) {
      kn.flush();
    } else {
      scheduleFlush();
    }
  };

  kn.flush = function flush() {
    if (buffer.length === 0) return;
    const payload = JSON.stringify(buffer);
    buffer = [];
    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    // Prefer sendBeacon: fire-and-forget, survives page navigation, no
    // CORS preflight. Fallback to fetch+keepalive for browsers where
    // sendBeacon isn't available (edge cases) or returns false (queue
    // full).
    let ok = false;
    if (navigator.sendBeacon) {
      try {
        ok = navigator.sendBeacon(
          ENDPOINT,
          new Blob([payload], { type: "application/json" }),
        );
      } catch {
        ok = false;
      }
    }
    if (!ok) {
      fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive: true,
        credentials: "include",
      }).catch(() => {
        // Drop on the floor. Events are best-effort — never block the
        // UI or surface failures to the user.
      });
    }
  };

  kn.setViewer = function setViewer({ id } = {}) {
    viewerId = typeof id === "number" ? id : null;
  };
  kn.setPersonality = function setPersonality({ id, slug } = {}) {
    if (typeof id === "number") personalityId = id;
    if (typeof slug === "string") personalitySlug = slug;
  };
  kn.setLastQuery = function setLastQuery(query) {
    lastQuery = typeof query === "string" && query.trim() ? query : null;
  };
  kn.setLastFilter = function setLastFilter({ source, sort } = {}) {
    if (typeof source === "string") lastSourceFilter = source;
    if (typeof sort === "string") lastSortMode = sort;
  };

  // Always flush on page hide so single-event sessions still get sent.
  // visibilitychange fires reliably across browsers/mobile; pagehide as
  // a secondary safety net on iOS Safari.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") kn.flush();
  });
  window.addEventListener("pagehide", () => kn.flush());
})();
