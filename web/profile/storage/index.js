/* Billing & storage hero — renders inside #storagePanel on the
 * settings page. Combines three things into one card:
 *
 *   1. Current balance (in dollars).
 *   2. Quick add-money chips that kick straight to Polar checkout
 *      via the helper exposed by web/credits.js. No detour through
 *      the dialog.
 *   3. Live disk usage for the signed-in user's library, plus the
 *      projected monthly storage fee.
 *
 * Dropped from the old version: the credits-vs-storage split, the
 * "Top up credits" CTA (now inline as chips), the "Updated …" line
 * (kept only as a quiet timestamp on the storage half).
 */
(function () {
  "use strict";

  const ABS = window.KNOWLEDGE_API_BASE;

  const $ = (id) => document.getElementById(id);
  const M = () => window.KnowledgeMoney;

  const getJson = window.K_getJson;
  async function postJson(path) {
    const r = await fetch(`${ABS}${path}`, {
      method: "POST",
      credentials: "include",
    });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  }

  function fmtBytes(n) {
    if (n == null) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let v = Number(n);
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    const fixed = v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2;
    return `${v.toFixed(fixed)} ${units[i]}`;
  }
  function fmtNumber(n) {
    return Number(n || 0).toLocaleString();
  }
  function fmtDateTime(iso) {
    if (!iso) return "never";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
    } catch {
      return iso;
    }
  }

  // Map a credit_events.kind value to the verb shown next to the
  // last-activity line. Kept here, not in credits.js, because this
  // hero needs to read concisely (one phrase) — credits.js uses
  // longer labels in its history list.
  function lastActivityLine(history) {
    if (!history || !history.length) return "No activity yet.";
    const e = history[0];
    const when = e.createdAt
      ? new Date(e.createdAt).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
        })
      : "";
    const amount = M().fmt(Math.abs(e.delta));
    const verb = e.delta >= 0 ? `Added ${amount}` : `Spent ${amount}`;
    const why =
      {
        top_up: "",
        refund: " (refund)",
        manual_adjustment: " (adjustment)",
        "debit:twitter-api": " on Twitter ingest",
        "debit:storage": " on storage",
        "debit:pipeline-run": " on a pipeline run",
        "debit:vip-sponsor": " sponsoring a personality",
        "debit:add-personality": " adding a personality",
        "debit:export": " on a library export",
      }[e.kind] || "";
    return `${verb}${why} on ${when}.`;
  }

  function render(panel, balance, storage, packs, history) {
    const r = storage.rates || {};
    const docs = storage.docCount || 0;
    const free = r.freeDocs || 0;
    const projected = storage.projectedCreditsPerMonth || 0;
    // Meter shows progress against the free quota. Above the quota
    // the bar saturates and we surface the cost-per-month line below.
    const overQuota = docs > free;
    const tier = overQuota ? "Paid" : "Free";
    const tierClass = overQuota ? "storage-tier paid" : "storage-tier free";
    const overBy = Math.max(0, docs - free);
    // Short, plain-English line that explains the relationship to
    // the free quota without a chart. Reads naturally regardless of
    // whether the user is below or above it.
    const positionLine = overQuota
      ? `${fmtNumber(overBy)} above the ${fmtNumber(free)}-document free quota.`
      : docs === 0
        ? `Your first ${fmtNumber(free)} documents are free.`
        : `${fmtNumber(free - docs)} documents to go before storage starts costing money.`;

    // Build the add-money chips from the live pack catalogue. Only
    // fixed packs surface as chips (one tap = one checkout). If the
    // catalogue is empty (no Polar creds configured), the chips
    // block is hidden entirely.
    const fixed = (packs || []).filter((p) => p.kind === "fixed");
    const chipsHtml = fixed
      .map((p) => {
        const bonus = (p.credits || 0) - (p.priceCents || 0);
        const bonusBadge =
          bonus > 0
            ? `<span class="billing-chip-bonus">${M().fmtBonus(bonus)}</span>`
            : "";
        return `<button type="button"
                        class="billing-chip"
                        data-product-id="${p.productId}">
                  <span class="billing-chip-amount">${M().fmt(p.priceCents)}</span>
                  ${bonusBadge}
                </button>`;
      })
      .join("");

    panel.innerHTML = `
      <div class="billing-hero">
        <div class="billing-balance">
          <span class="billing-balance-label">Balance</span>
          <span class="billing-balance-amount">${M().fmt(balance)}</span>
          <span class="billing-balance-sub">${lastActivityLine(history)}</span>
        </div>
        ${
          fixed.length
            ? `
          <div class="billing-add">
            <span class="billing-add-label">Add money</span>
            <div class="billing-chips">
              ${chipsHtml}
            </div>
          </div>
        `
            : ""
        }
      </div>

      <div class="billing-storage">
        <div class="billing-storage-head">
          <div>
            <span class="billing-storage-label">Your library</span>
            <span class="billing-storage-docs">${fmtNumber(docs)} documents</span>
            <span class="billing-storage-position">${positionLine}</span>
          </div>
          <div class="billing-storage-right">
            <span class="${tierClass}">${tier}</span>
            ${
              overQuota
                ? `<span class="billing-storage-cost"><strong>${M().fmt(projected)}</strong> / month</span>`
                : `<span class="billing-storage-cost-free">$0 / month</span>`
            }
          </div>
        </div>
        <div class="billing-storage-foot">
          <span>
            ${fmtBytes(storage.totalBytes)} on disk
            <span class="billing-storage-breakdown">
              (${fmtBytes(storage.dbBytes)} database + ${fmtBytes(storage.indexBytes)} index)
            </span>
          </span>
          <button type="button" class="billing-refresh-link" id="storageRefreshBtn"
                  title="Last updated ${fmtDateTime(storage.updatedAt)}">refresh</button>
        </div>
      </div>
    `;

    // Quick add-money chips: hit Polar checkout directly via the
    // shared helper in credits.js. If credits.js hasn't loaded yet
    // (it does on this page, but defensively), fall back to opening
    // the dialog.
    panel.querySelectorAll(".billing-chip[data-product-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const productId = btn.getAttribute("data-product-id");
        if (window.KnowledgeCredits?.checkout) {
          window.KnowledgeCredits.checkout(productId, null);
        } else if (window.KnowledgeCredits?.open) {
          window.KnowledgeCredits.open();
        }
      });
    });
    panel
      .querySelector("#storageRefreshBtn")
      ?.addEventListener("click", (e) => {
        e.preventDefault();
        init({ silent: true });
      });
  }

  function renderError(panel, msg) {
    panel.innerHTML = `<div class="billing-error">${msg}</div>`;
  }

  async function init(opts = {}) {
    const panel = $("storagePanel");
    if (!panel) return;
    // Twitter parsing is the only paid action. While it's gated to
    // VIPs (alpha), the billing widget shouldn't render or fetch
    // anything — page.js leaves `data-vip` off <html> for non-VIPs.
    if (!document.documentElement.hasAttribute("data-vip")) {
      panel.hidden = true;
      panel.innerHTML = "";
      return;
    }
    panel.hidden = false;
    if (!opts.silent) {
      panel.innerHTML = `<div class="billing-loading">Loading…</div>`;
    }
    try {
      const [storage, credits, packs] = await Promise.all([
        postJson("/api/me/storage/refresh").catch(() =>
          getJson("/api/me/storage"),
        ),
        getJson("/api/me/credits").catch(() => ({ balance: 0, history: [] })),
        getJson("/api/credits/packs").catch(() => []),
      ]);
      render(panel, credits.balance || 0, storage, packs, credits.history);
    } catch {
      renderError(
        panel,
        "Couldn't load your balance and storage right now. Try refreshing the page.",
      );
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init());
  } else {
    init();
  }
  window.KnowledgeStorage = { reload: () => init() };
})();
