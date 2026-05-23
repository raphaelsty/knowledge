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
        // Cumulative viewport dwell in ms — only set on card_seen.
        dwell_ms: payload.dwell_ms,
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

  // ── card_seen + dwell observer ───────────────────────────────────
  // Per-card viewport tracker. Measures *cumulative* time the card
  // was ≥60% on screen and reports it as `dwell_ms` on the
  // `card_seen` event. Logged-in only.
  //
  // Lifecycle for one card:
  //   1. enters viewport (≥60% visible) → start a stopwatch
  //   2. leaves viewport               → freeze the stopwatch,
  //                                       accumulate the elapsed slice
  //   3. re-enters                      → restart stopwatch (cumulative)
  //   4. card has accumulated ≥ MIN     → eligible to fire
  //   5. fires on viewport-leave OR     → emits the final dwell_ms
  //      pagehide OR cap reached         (clamped to MAX_DWELL_MS)
  //
  // Two thresholds:
  //   • MIN_DWELL_MS — below this we ignore the impression entirely.
  //     Scrolling past quickly shouldn't count.
  //   • MAX_DWELL_MS — anything above this is "tab left open", not
  //     engagement. Clamp so a forgotten tab doesn't dominate the
  //     ML training set or push a card permanently into the
  //     hide-seen bucket.
  //
  // De-dup is local to the tab session: once a card fires we won't
  // fire it again for the same URL in the same tab. The server
  // accepts multiple rows per (viewer, url) — the dedup is here only
  // to keep the event stream lean.
  const SEEN_THRESHOLD = 0.6;
  const MIN_DWELL_MS = 1500;
  const MAX_DWELL_MS = 120_000; // 2 minutes hard cap
  // Per-element dwell state. WeakMap so DOM removal frees the entry.
  //   { accumulated_ms, enteredAt_ms | null, capTimer | null, fired }
  const dwellState = new WeakMap();
  // We also keep a parallel Set of currently-observed elements so
  // the pagehide flush can walk them — WeakMap doesn't enumerate.
  const observedEls = new Set();
  const firedUrls = new Set(); // doc_urls already reported this tab
  let cardObserver = null;

  function commitDwell(el, reason) {
    const s = dwellState.get(el);
    if (!s || s.fired) return;
    // If still in viewport at commit time, fold the open slice in.
    if (s.enteredAt != null) {
      s.accumulated += performance.now() - s.enteredAt;
      s.enteredAt = null;
    }
    if (s.capTimer != null) {
      clearTimeout(s.capTimer);
      s.capTimer = null;
    }
    const dwell = Math.min(MAX_DWELL_MS, Math.round(s.accumulated));
    if (dwell < MIN_DWELL_MS) return; // didn't qualify
    const docUrl = el.dataset.url;
    if (!docUrl || firedUrls.has(docUrl) || viewerId == null) return;
    s.fired = true;
    firedUrls.add(docUrl);
    kn.track("card_seen", {
      doc_url: docUrl,
      // user_id is the NOT NULL FK; we attribute the impression to
      // the viewer themselves since the card isn't scoped to a
      // single library on the timeline.
      user_id: viewerId,
      viewer_user_id: viewerId,
      dwell_ms: dwell,
    });
    if (cardObserver) cardObserver.unobserve(el);
    observedEls.delete(el);
    if (reason === "page-hide") kn.flush();
  }

  function ensureCardObserver() {
    if (cardObserver || typeof IntersectionObserver === "undefined") {
      return cardObserver;
    }
    cardObserver = new IntersectionObserver(
      (entries) => {
        for (const ent of entries) {
          const el = ent.target;
          let s = dwellState.get(el);
          if (!s) {
            s = {
              accumulated: 0,
              enteredAt: null,
              capTimer: null,
              fired: false,
            };
            dwellState.set(el, s);
          }
          if (s.fired) continue;
          const visible =
            ent.isIntersecting && ent.intersectionRatio >= SEEN_THRESHOLD;
          if (visible && s.enteredAt == null) {
            // Entering viewport — start the stopwatch and schedule
            // a cap-timer at MAX - already-accumulated so a card
            // pinned on screen still fires at the hard cap.
            s.enteredAt = performance.now();
            const remaining = MAX_DWELL_MS - s.accumulated;
            s.capTimer = setTimeout(() => commitDwell(el, "cap"), remaining);
          } else if (!visible && s.enteredAt != null) {
            // Leaving viewport — close the slice. If we crossed the
            // min threshold this fires the event; otherwise it just
            // accumulates and waits for a future re-entry.
            s.accumulated += performance.now() - s.enteredAt;
            s.enteredAt = null;
            if (s.capTimer != null) {
              clearTimeout(s.capTimer);
              s.capTimer = null;
            }
            if (s.accumulated >= MIN_DWELL_MS) {
              commitDwell(el, "viewport-leave");
            }
          }
        }
      },
      { threshold: [SEEN_THRESHOLD] },
    );
    return cardObserver;
  }
  kn.observeCard = function observeCard(el, docUrl) {
    if (!el || !docUrl) return;
    if (viewerId == null) return; // logged-out: never observe
    if (firedUrls.has(docUrl)) return; // already counted this tab
    el.dataset.url = docUrl;
    const obs = ensureCardObserver();
    if (obs) {
      observedEls.add(el);
      obs.observe(el);
    }
  };
  // Clear the in-tab firedUrls dedup + reset per-element state so
  // flipping "Show seen" can re-observe and re-fire as the user
  // engages with the same cards again.
  kn.resetSeenSuppression = function resetSeenSuppression() {
    firedUrls.clear();
    for (const el of observedEls) {
      const s = dwellState.get(el);
      if (s && s.capTimer != null) clearTimeout(s.capTimer);
      dwellState.delete(el);
    }
  };

  // Page-hide flush — walk every observed card and commit any
  // accumulated dwell. The buffered `card_seen` event then rides
  // out on the same pagehide → sendBeacon path as the rest of the
  // batch (see the visibilitychange + pagehide listeners above).
  function flushPendingDwell() {
    for (const el of [...observedEls]) commitDwell(el, "page-hide");
  }
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushPendingDwell();
  });
  window.addEventListener("pagehide", flushPendingDwell);
})();
