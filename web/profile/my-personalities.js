/* "Personalities you've added" table — settings page section that
 * lists every public library the signed-in user has paid to create.
 *
 * Hangs off the same /api/me/personalities endpoint that the Rust
 * handler exposes. Renders into #myPersonalities if present, quiet
 * no-op otherwise. We keep the module separate from add-personality
 * so adding the form on a page doesn't drag the listing logic.
 */
(function () {
  "use strict";

  const ABS = (() => {
    const host = window.location.hostname;
    return host === "localhost" || host === "127.0.0.1"
      ? "http://localhost:8080"
      : "";
  })();

  function $(id) {
    return document.getElementById(id);
  }
  // escapeHtml comes from /lib/utils.js
  function fmtNumber(n) {
    return Number(n || 0).toLocaleString();
  }
  function fmtDate(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleDateString(undefined, { dateStyle: "medium" });
    } catch {
      return iso;
    }
  }

  // Translate the JSONB sources/links blob back into the flat shape
  // the edit form expects. Mirrors `build_sources` /  `build_links`
  // in api/src/handlers/personalities.rs so a round-trip
  // (load → edit → save) doesn't lose data.
  function prefillFromRow(r) {
    const s = r.sources || {};
    const l = r.links || {};
    const handle = (obj, k) => (obj && obj[k] ? String(obj[k]) : "");
    return {
      slug: r.slug,
      name: r.name || "",
      description: r.description || "",
      twitterHandle: handle(s.twitter, "username"),
      githubHandle: handle(s.github, "username"),
      huggingfaceHandle: handle(s.huggingface, "username"),
      redditHandle: handle(s.reddit, "username"),
      hackernewsHandle: handle(s.hackernews, "username"),
      stackoverflowUserId: handle(s.stackoverflow, "user_id"),
      arxivAuthor: handle(s.arxiv, "author"),
      dblpAuthor: handle(s.dblp, "author"),
      scholarUserId: handle(s.scholar, "user_id"),
      websites:
        s.websites && Array.isArray(s.websites.urls)
          ? s.websites.urls.join("\n")
          : typeof l.website === "string"
            ? l.website
            : "",
    };
  }

  // Stash row data on the button so the edit handler doesn't need
  // a re-fetch — single source of truth is the table render.
  const rowStore = new Map();

  // Format a cents value as a dollar string. Mirror of money.js's
  // helper — we avoid the import to keep this module independent.
  function fmtCost(cents) {
    if (cents == null || cents === 0) return "$0";
    const n = Number(cents);
    const d = Math.abs(n) / 100;
    const sign = n < 0 ? "-" : "";
    return `${sign}${d % 1 === 0 ? `$${d}` : `$${d.toFixed(2)}`}`;
  }
  // Human-readable tooltip showing the per-kind breakdown so the
  // table doesn't have to surface four columns. Empty lines drop
  // out so a brand-new sponsorship reads "$0.30 entry fee" only.
  function costBreakdown(r) {
    const lines = [];
    if (r.costEntryCents) lines.push(`${fmtCost(r.costEntryCents)} entry fee`);
    if (r.costTwitterCents)
      lines.push(`${fmtCost(r.costTwitterCents)} tweets extraction`);
    if (r.costStorageCents)
      lines.push(`${fmtCost(r.costStorageCents)} storage`);
    return lines.join("\n");
  }

  function render(panel, rows) {
    rowStore.clear();
    if (!rows.length) {
      panel.innerHTML = `
        <p class="myp-empty">You haven't added any personalities yet. Use the button above to add the first one.</p>
      `;
      return;
    }
    const body = rows
      .map((r, i) => {
        rowStore.set(r.slug, r);
        const breakdown = costBreakdown(r);
        return `
        <tr>
          <td>
            <a class="myp-name" href="/?libs=${encodeURIComponent(r.slug)}">${escapeHtml(r.name || r.slug)}</a>
            <span class="myp-slug">@${escapeHtml(r.slug)}</span>
            ${r.description ? `<span class="myp-bio">${escapeHtml(r.description)}</span>` : ""}
          </td>
          <td class="myp-docs">${fmtNumber(r.docCount)}</td>
          <td class="myp-cost" title="${escapeHtml(breakdown)}">${fmtCost(r.costCents)}</td>
          <td class="myp-date">${fmtDate(r.sponsoredAt)}</td>
          <td class="myp-actions">
            <button type="button" class="myp-edit" data-myp-edit="${escapeHtml(r.slug)}">Edit</button>
          </td>
        </tr>`;
      })
      .join("");
    panel.innerHTML = `
      <table class="myp-table">
        <thead>
          <tr>
            <th scope="col">Personality</th>
            <th scope="col">Documents</th>
            <th scope="col">Cost so far</th>
            <th scope="col">Added</th>
            <th scope="col"></th>
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    `;
    panel.querySelectorAll("[data-myp-edit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const slug = btn.getAttribute("data-myp-edit");
        const row = rowStore.get(slug);
        if (row && window.KnowledgeAddPersonality) {
          window.KnowledgeAddPersonality.open(prefillFromRow(row), {
            editing: true,
          });
        }
      });
    });
  }

  function renderEmpty(panel) {
    panel.innerHTML = `
      <p class="myp-empty">You haven't added any personalities yet. Use the button above to add the first one.</p>
    `;
  }
  function renderError(panel, msg) {
    panel.innerHTML = `<p class="myp-error">${escapeHtml(msg)}</p>`;
  }

  async function init() {
    const panel = $("myPersonalities");
    if (!panel) return;
    panel.innerHTML = `<p class="myp-loading">Loading…</p>`;
    try {
      const r = await fetch(`${ABS}/api/me/personalities`, {
        credentials: "include",
      });
      if (r.status === 401) {
        renderEmpty(panel);
        return;
      }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const rows = await r.json();
      render(panel, Array.isArray(rows) ? rows : []);
    } catch (err) {
      renderError(
        panel,
        `Couldn't load your personalities: ${err.message || err}`,
      );
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  window.KnowledgeMyPersonalities = { reload: init };
})();
