/* Export dialog — pick a date range + row cap, confirm, download.
 *
 * Self-contained module. Loaded on the welcome / search / personality
 * pages; exposes `window.KnowledgeExport.open(slug)`. The profile
 * header rendered by search/page.js wires its Export button to this
 * function via a delegated click handler at the bottom of the file.
 *
 * Flow:
 *   1. POST quote — `GET /api/personalities/{slug}/export.jsonl?quote=1`
 *      → { exportCount, docCount, slug }
 *      Anonymous → 401, dialog flips to "Sign in to export".
 *   2. Render the picker. The user can optionally narrow the export
 *      by `date_from` / `date_to` and the row cap. Changing any
 *      input refetches the quote so `exportCount` is always accurate.
 *   3. Confirm → navigate to the same URL without `quote=1`. The
 *      server logs one row in `export_downloads`, then streams the
 *      JSONL with `Content-Disposition: attachment` so the browser
 *      saves it directly.
 *
 * Exports are free for every signed-in user. The previous credits /
 * 402 paid path was removed when the platform stopped billing
 * exports; the surface kept the dialog because the date-range
 * picker is still useful, but the slider no longer shows a price.
 */
(function () {
  "use strict";

  const ABS = window.KNOWLEDGE_API_BASE;

  let dialog = null;
  let bodyEl = null;
  // Guards `onConfirm` so a double click can't fire two navigations.
  let busy = false;

  // escapeHtml, escapeAttr come from /lib/utils.js
  const fmtNumber = (n) => Number(n || 0).toLocaleString();

  // ── Dialog plumbing ────────────────────────────────────────────────
  function ensureDialog() {
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "exportDialog";
    dialog.className = "export-dialog";
    dialog.setAttribute("aria-label", "Export library");
    bodyEl = document.createElement("div");
    bodyEl.className = "export-body";
    bodyEl.id = "exportBody";
    dialog.appendChild(bodyEl);
    document.body.appendChild(dialog);
    // Backdrop click closes.
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) close();
    });
    return dialog;
  }
  function showDialog() {
    ensureDialog();
    if (typeof dialog.showModal === "function" && !dialog.open)
      dialog.showModal();
    else dialog.setAttribute("open", "");
  }
  function close() {
    if (!dialog) return;
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else dialog.removeAttribute("open");
    bodyEl.innerHTML = "";
  }
  function wireClose() {
    bodyEl
      .querySelectorAll("[data-export-close]")
      .forEach((b) => b.addEventListener("click", close));
  }

  // ── Renderers ──────────────────────────────────────────────────────
  function renderLoading(slug) {
    bodyEl.innerHTML = `
      <h2 class="export-title">Export @${escapeHtml(slug)}</h2>
      <p class="export-status">Counting documents…</p>
      <div class="export-actions">
        <button type="button" data-export-close>Cancel</button>
      </div>
    `;
    wireClose();
  }

  function renderError(slug, message) {
    bodyEl.innerHTML = `
      <h2 class="export-title">Export @${escapeHtml(slug)}</h2>
      <p class="export-error">${escapeHtml(message)}</p>
      <div class="export-actions">
        <button type="button" class="action-btn" data-export-close>Close</button>
      </div>
    `;
    wireClose();
  }

  function renderSignInRequired(slug) {
    bodyEl.innerHTML = `
      <h2 class="export-title">Sign in to export</h2>
      <p>Exporting <strong>@${escapeHtml(slug)}</strong> is free, but
         we need an account so we can record the download in your
         history.</p>
      <div class="export-actions">
        <button type="button" data-export-close>Close</button>
        <button type="button" class="action-btn primary" id="exportSignIn">Sign in</button>
      </div>
    `;
    wireClose();
    document.getElementById("exportSignIn")?.addEventListener("click", () => {
      close();
      // Welcome / search pages have an `#authBtn` that opens the
      // auth modal; the profile page just lives at /profile so a
      // sign-in flow there is handled by the page itself.
      document.getElementById("authBtn")?.click();
    });
  }

  // Build the download URL from the current picker state. Empty
  // strings on the date inputs map to "no bound"; the server treats
  // missing query params as NULL.
  function buildDownloadUrl(slug, { limit, dateFrom, dateTo }, asQuote) {
    const params = new URLSearchParams();
    if (asQuote) params.set("quote", "1");
    if (limit && limit > 0) params.set("limit", String(limit));
    if (dateFrom) params.set("date_from", dateFrom);
    if (dateTo) params.set("date_to", dateTo);
    const qs = params.toString();
    return `${ABS}/api/personalities/${encodeURIComponent(slug)}/export.jsonl${
      qs ? `?${qs}` : ""
    }`;
  }

  // Fire-and-debounce a quote refresh. The server-side count is
  // cheap (one indexed COUNT(*) on the documents table), so we just
  // re-request on every input change. A 200ms debounce keeps date
  // typing from drowning the server.
  let refreshTimer = null;
  function scheduleQuoteRefresh(slug, state) {
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(async () => {
      try {
        const r = await fetch(buildDownloadUrl(slug, state, true), {
          credentials: "include",
        });
        if (!r.ok) return;
        const fresh = await r.json();
        const out = document.getElementById("exportLiveCount");
        if (out) out.textContent = fmtNumber(fresh.exportCount || 0);
        const btn = document.getElementById("exportConfirm");
        if (btn) {
          btn.textContent = `Download ${fmtNumber(fresh.exportCount || 0)} docs`;
          btn.disabled = !fresh.exportCount;
        }
        // Show/hide the "trimmed to server cap" hint as the filter
        // moves above/below the ceiling.
        const hint = document.getElementById("exportCapHint");
        if (hint) hint.hidden = !fresh.capped;
      } catch {
        /* swallowed — the picker stays on the stale count, the user
         * can still hit Download */
      }
    }, 200);
  }

  // Today / today-minus-N as ISO YYYY-MM-DD, in the user's local
  // timezone — matches what the native <input type="date"> emits, so
  // preset clicks and manual edits round-trip without drift.
  function isoToday() {
    return isoDaysAgo(0);
  }
  function isoDaysAgo(days) {
    const d = new Date();
    d.setDate(d.getDate() - days);
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}`;
  }

  // Date-range presets. Each entry produces a (from, to) pair the
  // picker can drop into the inputs in one click. "All time" clears
  // both bounds — the server treats missing params as NULL.
  const DATE_PRESETS = [
    { id: "7d", label: "Last 7 days", days: 7 },
    { id: "30d", label: "Last 30 days", days: 30 },
    { id: "90d", label: "Last 90 days", days: 90 },
    { id: "1y", label: "Last year", days: 365 },
    { id: "all", label: "All time", days: null },
  ];

  // Detect which (if any) preset the current (from, to) pair
  // matches, so we can highlight the active chip. We only mark a
  // preset active when BOTH bounds line up, otherwise the user has
  // diverged with a manual edit and no chip should look selected.
  function activeDatePreset(from, to) {
    if (!from && !to) return "all";
    const today = isoToday();
    if (to !== today) return null;
    for (const p of DATE_PRESETS) {
      if (p.days != null && from === isoDaysAgo(p.days)) return p.id;
    }
    return null;
  }

  // Cap presets. "Max" maps to the server-supplied ceiling so the UI
  // never offers a value the server would refuse.
  function capPresets(maxLimit) {
    return [
      { id: "100", label: "100", value: 100 },
      { id: "1k", label: "1,000", value: 1000 },
      { id: "10k", label: "10,000", value: 10000 },
      { id: "max", label: `Max (${fmtNumber(maxLimit)})`, value: maxLimit },
    ];
  }

  function activeCapPreset(limit, maxLimit) {
    if (!limit || limit <= 0) return "max";
    for (const p of capPresets(maxLimit)) {
      if (p.value === limit) return p.id;
    }
    return null;
  }

  function renderPicker(quote) {
    const { slug, exportCount, docCount, maxLimit, capped } = quote;
    // Picker state mirrors the URL params. Limit 0 = use the server
    // cap (no explicit `?limit=` sent); dates empty = no bound.
    const state = {
      limit: 0,
      dateFrom: "",
      dateTo: "",
    };

    bodyEl.innerHTML = `
      <h2 class="export-title">Export @${escapeHtml(slug)}</h2>
      <p class="export-docs">
        <strong id="exportLiveCount">${fmtNumber(exportCount)}</strong>
        of <strong>${fmtNumber(docCount)}</strong> documents will be
        included. Newest first.
        <span id="exportCapHint" class="export-cap-hint"
              ${capped ? "" : "hidden"}>
          Trimmed to the ${fmtNumber(maxLimit)}-row server cap.
        </span>
      </p>

      <section class="export-section">
        <header class="export-section-head">
          <span class="export-section-title">Date range</span>
          <div class="export-pick-presets" data-preset-group="date">
            ${DATE_PRESETS.map(
              (p) => `
              <button type="button"
                      data-date-preset="${p.id}"
                      aria-pressed="false">${escapeHtml(p.label)}</button>
            `,
            ).join("")}
          </div>
        </header>
        <div class="export-date-row">
          <label class="export-input">
            <span>From</span>
            <input id="exportDateFrom" type="date" />
          </label>
          <label class="export-input">
            <span>To</span>
            <input id="exportDateTo" type="date" />
          </label>
        </div>
      </section>

      <section class="export-section">
        <header class="export-section-head">
          <span class="export-section-title">Row cap</span>
          <div class="export-pick-presets" data-preset-group="cap">
            ${capPresets(maxLimit)
              .map(
                (p) => `
              <button type="button"
                      data-cap-preset="${p.id}"
                      data-cap-value="${p.value}"
                      aria-pressed="false">${escapeHtml(p.label)}</button>
            `,
              )
              .join("")}
          </div>
        </header>
        <label class="export-input">
          <span>Custom cap</span>
          <input id="exportLimit" type="number" min="0"
                 max="${maxLimit}" step="1" inputmode="numeric"
                 placeholder="Server cap (${fmtNumber(maxLimit)})" />
        </label>
      </section>

      <p class="export-rate">Free for signed-in users. Each download
         is logged to your account history.</p>

      <div class="export-actions">
        <button type="button" data-export-close>Cancel</button>
        <button type="button"
                class="action-btn primary"
                id="exportConfirm"
                ${!exportCount ? "disabled" : ""}>
          Download ${fmtNumber(exportCount)} docs
        </button>
      </div>
    `;
    wireClose();

    const dateFromEl = document.getElementById("exportDateFrom");
    const dateToEl = document.getElementById("exportDateTo");
    const limitEl = document.getElementById("exportLimit");

    // Reflect the current state into the chip highlights. Called
    // after every state change (preset click, input edit, etc.) so
    // chips and inputs stay coherent.
    function refreshChipHighlights() {
      const date = activeDatePreset(state.dateFrom, state.dateTo);
      bodyEl
        .querySelectorAll("[data-date-preset]")
        .forEach((b) =>
          b.setAttribute(
            "aria-pressed",
            b.getAttribute("data-date-preset") === date ? "true" : "false",
          ),
        );
      const cap = activeCapPreset(state.limit, maxLimit);
      bodyEl
        .querySelectorAll("[data-cap-preset]")
        .forEach((b) =>
          b.setAttribute(
            "aria-pressed",
            b.getAttribute("data-cap-preset") === cap ? "true" : "false",
          ),
        );
    }

    function commitChange() {
      refreshChipHighlights();
      scheduleQuoteRefresh(slug, state);
    }

    function syncFromInputs() {
      state.dateFrom = dateFromEl.value || "";
      state.dateTo = dateToEl.value || "";
      state.limit = Math.max(0, Number(limitEl.value) || 0);
      commitChange();
    }
    dateFromEl.addEventListener("change", syncFromInputs);
    dateToEl.addEventListener("change", syncFromInputs);
    limitEl.addEventListener("input", syncFromInputs);

    bodyEl.querySelectorAll("[data-date-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-date-preset");
        const preset = DATE_PRESETS.find((p) => p.id === id);
        if (!preset) return;
        if (preset.days == null) {
          state.dateFrom = "";
          state.dateTo = "";
        } else {
          state.dateFrom = isoDaysAgo(preset.days);
          state.dateTo = isoToday();
        }
        dateFromEl.value = state.dateFrom;
        dateToEl.value = state.dateTo;
        commitChange();
      });
    });

    bodyEl.querySelectorAll("[data-cap-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const v = Number(btn.getAttribute("data-cap-value")) || 0;
        // "Max" chip leaves the limit at 0 so we send no `?limit=`
        // and the server applies its own cap — saves a click vs.
        // typing the ceiling in the box.
        state.limit = v >= maxLimit ? 0 : v;
        limitEl.value = state.limit > 0 ? String(state.limit) : "";
        commitChange();
      });
    });

    document
      .getElementById("exportConfirm")
      ?.addEventListener("click", () => onConfirm(slug, state));

    // Initial highlight: nothing entered yet means "All time" +
    // "Max" are the active presets.
    refreshChipHighlights();
  }

  // Confirm — navigate to the streaming URL. The server's
  // `Content-Disposition: attachment` header turns the navigation
  // into a save dialog instead of a render. We don't pre-flight with
  // fetch() because there's no longer a 402 path to recover from;
  // any 4xx the server returns will just appear in the browser
  // download UI as a failed save, which is the right UX for the
  // (rare) edge cases left (e.g. session expired mid-dialog).
  function onConfirm(slug, state) {
    if (busy) return;
    busy = true;
    const url = buildDownloadUrl(slug, state, false);
    const btn = document.getElementById("exportConfirm");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Downloading…";
    }
    window.location.assign(url);
    // Give the browser a beat to start the download before we close
    // the dialog — closing too quickly looks like nothing happened.
    setTimeout(() => {
      busy = false;
      close();
    }, 1200);
  }

  // ── Entry point ────────────────────────────────────────────────────
  async function open(slug) {
    if (!slug) return;
    showDialog();
    renderLoading(slug);
    try {
      const url = buildDownloadUrl(
        slug,
        { limit: 0, dateFrom: "", dateTo: "" },
        true,
      );
      const r = await fetch(url, { credentials: "include" });
      if (r.status === 401) {
        renderSignInRequired(slug);
        return;
      }
      if (r.status === 404) {
        renderError(slug, "This library is private or doesn't exist.");
        return;
      }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      renderPicker(await r.json());
    } catch (err) {
      renderError(
        slug,
        `Could not load the export picker: ${err.message || err}`,
      );
    }
  }

  // Document-level delegation so the Export button rendered by
  // search/page.js (`data-action="ph-export"`) works regardless of
  // when this script finished loading relative to that render.
  document.addEventListener("click", (e) => {
    const btn = e.target.closest('[data-action="ph-export"]');
    if (!btn) return;
    e.preventDefault();
    const slug = btn.getAttribute("data-slug");
    if (slug) open(slug);
  });

  window.KnowledgeExport = { open };
})();
