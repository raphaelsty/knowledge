/* Credits panel — buy credits + sponsor a new VIP.
 *
 * Self-contained module. Loaded once on every page; exposes
 * `window.KnowledgeCredits.open()` to pop the panel from anywhere.
 * The rail "Credits" pill calls that open() on click.
 *
 * Structure:
 *   <dialog id="creditsDialog">
 *     <div class="credits-body">
 *       [tabs: Buy credits · Sponsor a VIP · History]
 *       [active panel]
 *     </div>
 *   </dialog>
 *
 * Keeping everything inside this one file + credits.css keeps the
 * billing feature cleanly separated from page.js / search.css.
 */
(function () {
  "use strict";

  const ABS = window.KNOWLEDGE_API_BASE;

  const $ = (id) => document.getElementById(id);

  // ── State ──────────────────────────────────────────────────────────
  let dialog = null;
  let bodyEl = null;
  let packs = []; // populated on first open
  let balance = 0;
  let history = [];
  let sponsorships = [];
  let activeTab = "buy"; // "buy" | "sponsor" | "history"
  let busy = false;

  // ── Tiny utilities ────────────────────────────────────────────────
  // escapeHtml comes from /lib/utils.js
  const escapeAttr = escapeHtml;
  function fmtUSD(cents) {
    if (cents == null) return "";
    const d = cents / 100;
    return d % 1 === 0 ? `$${d}` : `$${d.toFixed(2)}`;
  }
  // Format the price with both supported currencies (e.g. "$5 / €5").
  // We use identical integer amounts in USD and EUR on Polar, so the
  // numeric value is shared; only the symbol differs. Polar's
  // Localized Checkout picks the actual currency by the buyer's region
  // at checkout time — this is just the display hint.
  function fmtPrice(cents, currencies) {
    if (cents == null) return "";
    const d = cents / 100;
    const trim = (n) => (n % 1 === 0 ? `${n}` : n.toFixed(2));
    const SYM = { USD: "$", EUR: "€" };
    const list = currencies && currencies.length ? currencies : ["USD"];
    return list.map((c) => `${SYM[c] || c + " "}${trim(d)}`).join(" / ");
  }
  const getJson = window.K_getJson;
  async function postJson(path, body) {
    const r = await fetch(`${ABS}${path}`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let payload = null;
    try {
      payload = await r.json();
    } catch {}
    if (!r.ok) {
      const err = new Error(
        (payload && payload.error) || `${path}: HTTP ${r.status}`,
      );
      err.status = r.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  // ── Data loaders ──────────────────────────────────────────────────
  async function loadAll() {
    const [creditsResp, packsResp] = await Promise.allSettled([
      getJson("/api/me/credits"),
      getJson("/api/credits/packs"),
    ]);
    if (creditsResp.status === "fulfilled") {
      balance = creditsResp.value.balance;
      history = creditsResp.value.history || [];
    }
    if (packsResp.status === "fulfilled") {
      packs = packsResp.value || [];
    }
  }

  // Balance / cost / pack values are stored as cents internally
  // (the legacy `credits` column — 1 credit = 1¢). The UI never
  // shows "credits"; everything renders in USD via KnowledgeMoney.
  const M = () => window.KnowledgeMoney;

  // ── Render ────────────────────────────────────────────────────────
  function render() {
    if (!bodyEl) return;
    bodyEl.innerHTML = `
      <header class="credits-head">
        <div class="credits-balance">
          <span class="credits-balance-label">Balance</span>
          <span class="credits-balance-amount">${M().fmt(balance)}</span>
        </div>
        <button type="button" class="credits-close" data-credits-close aria-label="Close">×</button>
      </header>
      <nav class="credits-tabs" role="tablist">
        <button type="button"
                class="credits-tab ${activeTab === "buy" ? "is-active" : ""}"
                data-credits-tab="buy">Add funds</button>
        <button type="button"
                class="credits-tab ${activeTab === "history" ? "is-active" : ""}"
                data-credits-tab="history">History</button>
      </nav>
      <div class="credits-panel">${panelHtml()}</div>`;
    wire();
  }

  function panelHtml() {
    if (activeTab === "buy") return panelBuyHtml();
    // Any non-"buy" value (including a stale `sponsor` left over in
    // local state from before paid personalities moved to their own
    // /api/personalities flow) falls through to the ledger view.
    return panelHistoryHtml();
  }

  function panelBuyHtml() {
    if (!packs.length) {
      return `
        <div class="credits-empty">
          <p>Billing isn't configured yet. The operator needs to set
             <code>POLAR_ACCESS_TOKEN</code>, <code>POLAR_WEBHOOK_SECRET</code>
             and one or more <code>POLAR_PACK_*</code> env vars before
             top-ups can run.</p>
        </div>`;
    }
    const fixed = packs.filter((p) => p.kind === "fixed");
    const custom = packs.find((p) => p.kind === "custom");
    // Pack tiles read as "pay $X → balance goes up by $Y". The Y is
    // the credits field on the API (still named that internally; 1
    // credit = 1¢). Bonus packs grant more than they cost, e.g. pay
    // $3, get $3.20 added to your balance.
    const fixedHtml = fixed
      .map(
        (p) => `
        <button type="button"
                class="credits-pack"
                data-pack-id="${escapeAttr(p.id)}"
                data-product-id="${escapeAttr(p.productId)}">
          <span class="credits-pack-price">${M().fmt(p.priceCents)}</span>
          <span class="credits-pack-credits">→ ${M().fmt(p.credits)} added</span>
          ${bonusBadge(p)}
        </button>`,
      )
      .join("");
    const customHtml = custom
      ? `
        <section class="credits-custom">
          <h3>Pick your amount</h3>
          <p>Between ${M().fmt(custom.minCents)} and ${M().fmt(custom.maxCents)} —
             every dollar adds a dollar to your balance.
             EU residents can pay in euros at checkout (Polar handles the conversion).</p>
          <div class="credits-custom-row">
            <span class="credits-custom-currency">$</span>
            <input type="number"
                   class="credits-custom-input"
                   id="creditsCustomAmount"
                   min="${(custom.minCents / 100).toFixed(2)}"
                   max="${(custom.maxCents / 100).toFixed(2)}"
                   step="1"
                   value="5"
                   inputmode="decimal" />
            <span class="credits-custom-equiv" id="creditsCustomEquiv">= $5.00 added</span>
          </div>
          <button type="button"
                  class="credits-buy-custom"
                  data-pack-id="${escapeAttr(custom.id)}"
                  data-product-id="${escapeAttr(custom.productId)}">
            Buy
          </button>
        </section>`
      : "";
    return `
      <div class="credits-buy">
        <div class="credits-packs-grid">${fixedHtml}</div>
        ${customHtml}
      </div>`;
  }

  // Bonus = how much more than they paid the user receives. Shown as
  // "+$0.20" so the value-add is concrete (vs an opaque "+7%").
  function bonusBadge(p) {
    if (!p.credits || !p.priceCents) return "";
    if (p.credits <= p.priceCents) return "";
    const bonus = p.credits - p.priceCents;
    return `<span class="credits-pack-bonus">${M().fmtBonus(bonus)} bonus</span>`;
  }

  function panelSponsorHtml() {
    return `
      <div class="credits-sponsor">
        <p class="credits-sponsor-pitch">
          Spot someone whose library would belong here? Sponsor them.
          We review every submission and onboard approved candidates
          into the VIP feed.
        </p>
        <p class="credits-sponsor-cost">
          <strong>Cost:</strong> 200 credits ($2). Refunded if we
          can't onboard the candidate.
        </p>
        <form class="credits-sponsor-form" id="creditsSponsorForm">
          <label class="credits-field">
            <span>Candidate name</span>
            <input type="text"
                   id="sponsorName"
                   required
                   minlength="1"
                   maxlength="200"
                   placeholder="Andrej Karpathy" />
          </label>
          <label class="credits-field">
            <span>Candidate URL <em>(profile, blog, GitHub…)</em></span>
            <input type="url"
                   id="sponsorUrl"
                   required
                   placeholder="https://x.com/karpathy" />
          </label>
          <label class="credits-field">
            <span>Why this candidate? <em>(optional)</em></span>
            <textarea id="sponsorNote"
                      rows="3"
                      maxlength="2000"
                      placeholder="Their tweets on training tricks are gold."></textarea>
          </label>
          <div class="credits-sponsor-actions">
            <span class="credits-sponsor-balance">
              Balance after: <strong>${M().fmt(Math.max(0, balance - 200))}</strong>
            </span>
            <button type="submit"
                    class="credits-sponsor-submit"
                    ${balance < 200 ? "disabled" : ""}>
              ${balance < 200 ? "Need $2.00 to sponsor" : "Sponsor for $2.00"}
            </button>
          </div>
          <div class="credits-sponsor-msg" id="creditsSponsorMsg" hidden></div>
        </form>
        ${sponsorshipsListHtml()}
      </div>`;
  }

  function sponsorshipsListHtml() {
    if (!sponsorships.length) return "";
    const rows = sponsorships
      .map(
        (s) => `
          <li>
            <div class="credits-sponsor-row">
              <div>
                <a class="credits-sponsor-name" href="${escapeAttr(s.candidateUrl)}" target="_blank" rel="noopener">${escapeHtml(s.candidateName)}</a>
                <p class="credits-sponsor-meta">${escapeHtml(s.createdAt.slice(0, 10))} · ${M().fmt(s.creditsPaid)}</p>
              </div>
              <span class="credits-sponsor-status is-${escapeAttr(s.status)}">${escapeHtml(s.status)}</span>
            </div>
          </li>`,
      )
      .join("");
    return `
      <section class="credits-sponsor-history">
        <h4>Your submissions</h4>
        <ul>${rows}</ul>
      </section>`;
  }

  // Map the `credits_ledger.kind` enum from Postgres to a label
  // suitable for the history list. SQL keeps the canonical key
  // (used in WHERE filters and analytics); the UI just needs
  // something readable. Unknown kinds fall back to the raw value
  // so a freshly added kind isn't silently mis-labeled.
  function kindLabel(kind) {
    return (
      {
        top_up: "Top-up",
        refund: "Refund",
        manual_adjustment: "Adjustment",
        "debit:twitter-api": "Twitter ingest",
        "debit:storage": "Storage",
        "debit:pipeline-run": "Pipeline run",
        "debit:vip-sponsor": "Sponsored a personality",
        "debit:add-personality": "Added a personality",
        "debit:export": "Library export",
      }[kind] || kind
    );
  }

  function panelHistoryHtml() {
    if (!history.length) {
      return `<div class="credits-empty"><p>No activity yet. Add funds to get started.</p></div>`;
    }
    const rows = history
      .map((e) => {
        const sign = e.delta >= 0 ? "+" : "−";
        const formatted = `${sign}${M().fmt(Math.abs(e.delta))}`;
        return `
          <li class="credits-history-row ${e.delta >= 0 ? "is-credit" : "is-debit"}">
            <div>
              <span class="credits-history-kind">${escapeHtml(kindLabel(e.kind))}</span>
              <span class="credits-history-date">${escapeHtml(e.createdAt.slice(0, 10))}</span>
            </div>
            <span class="credits-history-delta">${formatted}</span>
          </li>`;
      })
      .join("");
    return `<ul class="credits-history-list">${rows}</ul>`;
  }

  // ── Wiring ────────────────────────────────────────────────────────
  function wire() {
    bodyEl
      .querySelector("[data-credits-close]")
      ?.addEventListener("click", close);
    bodyEl.querySelectorAll("[data-credits-tab]").forEach((b) => {
      b.addEventListener("click", () => {
        activeTab = b.dataset.creditsTab;
        render();
      });
    });
    // Fixed-pack buttons.
    bodyEl.querySelectorAll(".credits-pack").forEach((b) => {
      b.addEventListener("click", () => checkout(b.dataset.productId, null));
    });
    // Custom-pack live "= $N added" hint. credits_per_cent on the
    // custom pack is 1 (= no bonus) so the amount added equals what
    // the user pays. We still go through the formula in case the
    // rate ever changes.
    const amt = bodyEl.querySelector("#creditsCustomAmount");
    const equiv = bodyEl.querySelector("#creditsCustomEquiv");
    if (amt && equiv) {
      const custom = packs.find((p) => p.kind === "custom");
      const update = () => {
        const dollars = parseFloat(amt.value || "0");
        const cents = Math.round(dollars * 100);
        const credited = Math.max(0, cents * (custom?.creditsPerCent || 1));
        equiv.textContent = `= ${M().fmt(credited)} added`;
      };
      amt.addEventListener("input", update);
      update();
    }
    bodyEl
      .querySelector(".credits-buy-custom")
      ?.addEventListener("click", (e) => {
        const productId = e.currentTarget.dataset.productId;
        const dollars = parseFloat(amt?.value || "0");
        const cents = Math.round(dollars * 100);
        checkout(productId, cents);
      });
    // Sponsorship form.
    bodyEl
      .querySelector("#creditsSponsorForm")
      ?.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (busy) return;
        busy = true;
        const name = bodyEl.querySelector("#sponsorName").value.trim();
        const url = bodyEl.querySelector("#sponsorUrl").value.trim();
        const note = bodyEl.querySelector("#sponsorNote").value.trim();
        const msg = bodyEl.querySelector("#creditsSponsorMsg");
        try {
          const resp = await postJson("/api/me/sponsorships", {
            candidateName: name,
            candidateUrl: url,
            candidateNote: note,
          });
          balance = resp.balance;
          // Reload to pick up the new row.
          sponsorships = await getJson("/api/me/sponsorships");
          history = (await getJson("/api/me/credits")).history;
          render();
        } catch (err) {
          if (msg) {
            msg.hidden = false;
            msg.textContent =
              err.status === 402
                ? `Not enough credits — buy ${err.payload?.required || 200}+ first.`
                : err.message || "Could not submit sponsorship.";
          }
        } finally {
          busy = false;
        }
      });
  }

  async function checkout(productId, amountCents) {
    if (busy) return;
    busy = true;
    try {
      const resp = await postJson("/api/credits/checkout", {
        productId,
        amountCents,
        successUrl: `${window.location.origin}/?credits=topped-up`,
      });
      if (resp && resp.url) {
        window.location.href = resp.url;
        return;
      }
    } catch (e) {
      console.warn("[credits] checkout failed", e);
      alert(`Checkout failed: ${e.message}`);
    } finally {
      busy = false;
    }
  }

  // ── Open / close ──────────────────────────────────────────────────
  async function open() {
    dialog = $("creditsDialog");
    bodyEl = $("creditsBody");
    if (!dialog || !bodyEl) {
      console.warn("[credits] dialog markup not present on this page");
      return;
    }
    // Show a loading shell immediately, then refresh.
    bodyEl.innerHTML = `<div class="credits-loading">Loading…</div>`;
    if (typeof dialog.showModal === "function" && !dialog.open) {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
    try {
      await loadAll();
      render();
    } catch (e) {
      bodyEl.innerHTML = `<div class="credits-empty"><p>Could not load credits. ${escapeHtml(e.message)}</p></div>`;
    }
  }

  function close() {
    if (!dialog) return;
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else dialog.removeAttribute("open");
    bodyEl.innerHTML = "";
  }

  // Backdrop click closes.
  document.addEventListener("click", (e) => {
    if (e.target?.id === "creditsDialog") close();
  });

  // The credits dialog is opened from the settings page only — the
  // "Top up credits" button inside the storage panel calls
  // `window.KnowledgeCredits.open()`. No rail-pill wiring needed.

  // Gate the public surface behind the VIP flag while Twitter
  // parsing — the only paid action — sits in alpha. Non-VIP
  // callers get a no-op so a stray click on a hidden chip can
  // never hit Polar.
  function vipGuard(fn) {
    return function (...args) {
      if (!document.documentElement.hasAttribute("data-vip")) {
        console.info(
          "Credits flow paused while tweets extraction is in alpha.",
        );
        return;
      }
      return fn.apply(this, args);
    };
  }
  window.KnowledgeCredits = {
    open: vipGuard(open),
    checkout: vipGuard(checkout),
  };
})();
