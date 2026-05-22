/* One-shot bookmark dialog — shared by welcome.html and search.html.
 *
 * Exposes `window.KnowledgeBookmark.open()`. The dialog asks for a URL,
 * shows a live editable preview card derived from the URL (no server
 * proxy), persists the doc via `POST /auth/me/documents/bulk` with
 * `source: "bookmark"`, then stars it via `POST /auth/me/favorite-docs`
 * so it surfaces in the Favorites chip immediately.
 *
 * Auth is session-cookie based; if the caller isn't signed in, the
 * dialog renders a prompt instead of the form.
 */
(function () {
  const API_BASE = window.KNOWLEDGE_API_BASE;

  let dialogEl = null;
  let cachedMe = undefined;

  async function getMe() {
    if (cachedMe !== undefined) return cachedMe;
    try {
      const r = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
      cachedMe = r.ok ? await r.json() : null;
    } catch {
      cachedMe = null;
    }
    return cachedMe;
  }

  function hostOf(url) {
    try {
      return new URL(url).hostname.replace(/^www\./, "");
    } catch {
      return url;
    }
  }

  function faviconUrl(host) {
    return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=32`;
  }

  function titleFromUrl(url) {
    try {
      const u = new URL(url);
      const parts = u.pathname.split("/").filter(Boolean);
      if (!parts.length) return u.hostname.replace(/^www\./, "");
      const last = decodeURIComponent(parts[parts.length - 1])
        .replace(/\.[^.]+$/, "") // strip extension
        .replace(/[-_]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      if (!last) return u.hostname.replace(/^www\./, "");
      return last.charAt(0).toUpperCase() + last.slice(1);
    } catch {
      return url;
    }
  }

  function parseMeta(html) {
    const headEnd = html.search(/<\/head>/i);
    const slice =
      headEnd > 0 ? html.slice(0, headEnd + 7) : html.slice(0, 50000);
    let doc;
    try {
      doc = new DOMParser().parseFromString(slice, "text/html");
    } catch {
      return { title: null, description: null };
    }
    const get = (sel, attr) => {
      const el = doc.querySelector(sel);
      if (!el) return null;
      const v = attr ? el.getAttribute(attr) : el.textContent;
      return v ? v.trim() || null : null;
    };
    return {
      title:
        get('meta[property="og:title"]', "content") || get("title") || null,
      description:
        get('meta[property="og:description"]', "content") ||
        get('meta[name="twitter:description"]', "content") ||
        get('meta[name="description"]', "content") ||
        null,
    };
  }

  async function tryFetchMeta(url) {
    // Try direct browser fetch first (works for CORS-friendly sites).
    try {
      const r = await fetch(url, { mode: "cors", credentials: "omit" });
      if (r.ok) {
        const ct = r.headers.get("content-type") || "";
        if (ct.includes("text/html")) {
          const meta = parseMeta(await r.text());
          if (meta.title || meta.description) return meta;
        }
      }
    } catch {
      /* CORS blocked — fall through to proxy */
    }
    // Fall back to the API proxy for everything else.
    try {
      const r = await fetch(
        `${API_BASE}/api/proxy/fetch?url=${encodeURIComponent(url)}`,
        { credentials: "include" },
      );
      if (!r.ok) return null;
      return parseMeta(await r.text());
    } catch {
      return null;
    }
  }

  function normalizeUrl(raw) {
    const trimmed = (raw || "").trim();
    if (!trimmed) return "";
    if (/^https?:\/\//i.test(trimmed)) return trimmed;
    return `https://${trimmed}`;
  }

  function todayIso() {
    const d = new Date();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}`;
  }

  // Save the bookmark to the user's library. Starring is a separate
  // user action — adding something you posted shouldn't auto-add it
  // to your starred list.
  async function saveBookmark(url, title, summary, tags = []) {
    const body = {
      documents: [
        {
          url,
          title: title || url,
          summary: summary || "",
          tags,
          date: todayIso(),
          source: "bookmark",
        },
      ],
    };
    const r = await fetch(`${API_BASE}/auth/me/documents/bulk`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`bulk save: HTTP ${r.status}`);
    return r.json();
  }

  function ensureDialog() {
    if (dialogEl) return dialogEl;
    const root = document.createElement("div");
    root.className = "kb-back";
    root.setAttribute("hidden", "");
    root.innerHTML = `
      <div class="kb-card" role="dialog" aria-modal="true" aria-labelledby="kbTitle">
        <div class="kb-head">
          <h2 id="kbTitle">Add bookmark</h2>
          <button class="kb-close" type="button" aria-label="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.2" stroke-linecap="round" aria-hidden="true">
              <line x1="6" y1="6" x2="18" y2="18"/>
              <line x1="18" y1="6" x2="6" y2="18"/>
            </svg>
          </button>
        </div>
        <p class="kb-hint">
          Paste a URL — edit the title and description, then save.
        </p>
        <p class="kb-indexing-note">
          Indexing runs once a day. New bookmarks become searchable
          starting tomorrow.
        </p>
        <form class="kb-form">
          <input
            class="kb-input"
            type="url"
            name="url"
            placeholder="https://…"
            autocomplete="off"
            required
            spellcheck="false"
          />
          <div class="kb-preview" hidden>
            <div class="kb-preview-kicker">
              <img class="kb-preview-fav" src="" alt="" width="14" height="14"/>
              <span class="kb-preview-host"></span>
            </div>
            <div
              class="kb-preview-title"
              contenteditable="true"
              aria-label="Title"
              spellcheck="true"
            ></div>
            <div
              class="kb-preview-desc"
              contenteditable="true"
              aria-label="Description"
              data-placeholder="Add a description…"
              spellcheck="true"
            ></div>
            <div class="kb-tags-section">
              <label class="kb-tags-label">Tags</label>
              <div class="kb-tags-field">
                <div class="kb-tags-chips"></div>
                <input
                  class="kb-tags-input"
                  type="text"
                  placeholder="machine-learning, rust, …"
                  autocomplete="off"
                  spellcheck="false"
                />
              </div>
              <p class="kb-tags-hint">Separate tags with commas or press Enter</p>
            </div>
          </div>
          <p class="kb-zotero-tip">
            Tip: install
            <a href="https://www.zotero.org/download/" target="_blank" rel="noopener">Zotero</a>
            with its
            <a href="https://www.zotero.org/download/connectors" target="_blank" rel="noopener">browser connector</a>
            or mobile app —
            <a href="https://apps.apple.com/app/zotero/id1513554812" target="_blank" rel="noopener">iOS</a>
            /
            <a href="https://play.google.com/store/apps/details?id=org.zotero.android" target="_blank" rel="noopener">Android</a>
            — to save articles in one click, then add your
            <a href="/profile.html" target="_blank" rel="noopener">Zotero API key in Settings</a>
            to sync them here automatically.
          </p>
          <div class="kb-actions">
            <button type="button" class="kb-cancel">Cancel</button>
            <button type="submit" class="kb-submit" disabled>
              <span class="kb-submit-label">Save</span>
            </button>
          </div>
        </form>
        <div class="kb-status" role="status" aria-live="polite"></div>
      </div>
    `;
    document.body.appendChild(root);
    dialogEl = root;

    const card = root.querySelector(".kb-card");
    const closeBtn = root.querySelector(".kb-close");
    const cancelBtn = root.querySelector(".kb-cancel");
    const form = root.querySelector(".kb-form");
    const input = root.querySelector(".kb-input");
    const submit = root.querySelector(".kb-submit");
    const submitLabel = root.querySelector(".kb-submit-label");
    const status = root.querySelector(".kb-status");
    const preview = root.querySelector(".kb-preview");
    const prevFav = root.querySelector(".kb-preview-fav");
    const prevHost = root.querySelector(".kb-preview-host");
    const prevTitle = root.querySelector(".kb-preview-title");
    const prevDesc = root.querySelector(".kb-preview-desc");
    const tagsChips = root.querySelector(".kb-tags-chips");
    const tagsInput = root.querySelector(".kb-tags-input");

    const tags = new Set();

    function addTag(raw) {
      const t = raw
        .trim()
        .toLowerCase()
        .replace(/\s+/g, "-")
        .replace(/[^a-z0-9-]/g, "");
      if (!t || tags.has(t)) return;
      tags.add(t);
      const chip = document.createElement("span");
      chip.className = "kb-tag-chip";
      const label = document.createTextNode(t);
      const del = document.createElement("button");
      del.type = "button";
      del.className = "kb-tag-del";
      del.setAttribute("aria-label", `Remove ${t}`);
      del.innerHTML = `<svg viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><line x1="2" y1="2" x2="8" y2="8"/><line x1="8" y1="2" x2="2" y2="8"/></svg>`;
      del.addEventListener("click", () => {
        tags.delete(t);
        chip.remove();
      });
      chip.appendChild(label);
      chip.appendChild(del);
      tagsChips.appendChild(chip);
    }

    function flushInput(value) {
      value.split(",").forEach((p) => {
        if (p.trim()) addTag(p);
      });
      tagsInput.value = "";
    }

    tagsInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        flushInput(tagsInput.value);
      } else if (e.key === "Backspace" && !tagsInput.value) {
        const last = tagsChips.lastElementChild;
        if (last) {
          const t = last.querySelector("svg")
            ? last.childNodes[0].textContent.trim()
            : last.textContent.replace("×", "").trim();
          tags.delete(t);
          last.remove();
        }
      }
    });
    tagsInput.addEventListener("input", () => {
      if (tagsInput.value.includes(",")) flushInput(tagsInput.value);
    });
    tagsInput.addEventListener("blur", () => {
      if (tagsInput.value.trim()) flushInput(tagsInput.value);
    });

    let fetchAbort = null;

    function updatePreview() {
      const url = normalizeUrl(input.value);
      if (!url) {
        preview.hidden = true;
        submit.disabled = true;
        return;
      }
      const host = hostOf(url);
      prevFav.src = faviconUrl(host);
      prevHost.textContent = host;
      if (!prevTitle.textContent.trim()) {
        prevTitle.textContent = titleFromUrl(url);
      }
      preview.hidden = false;
      submit.disabled = false;

      if (fetchAbort) fetchAbort();
      let cancelled = false;
      fetchAbort = () => {
        cancelled = true;
      };
      tryFetchMeta(url).then((meta) => {
        if (cancelled || !meta) return;
        if (meta.title && !prevTitle.dataset.userEdited) {
          prevTitle.textContent = meta.title;
        }
        if (meta.description && !prevDesc.dataset.userEdited) {
          prevDesc.textContent = meta.description;
        }
      });
    }

    prevTitle.addEventListener("input", () => {
      prevTitle.dataset.userEdited = "1";
    });
    prevDesc.addEventListener("input", () => {
      prevDesc.dataset.userEdited = "1";
    });

    input.addEventListener("input", updatePreview);
    input.addEventListener("paste", () => setTimeout(updatePreview, 0));

    const close = () => {
      root.setAttribute("hidden", "");
      status.textContent = "";
      status.className = "kb-status";
      submit.disabled = true;
      submitLabel.textContent = "Save";
      if (fetchAbort) {
        fetchAbort();
        fetchAbort = null;
      }
      preview.hidden = true;
      prevTitle.textContent = "";
      prevDesc.textContent = "";
      tagsChips.innerHTML = "";
      tagsInput.value = "";
      tags.clear();
      delete prevTitle.dataset.userEdited;
      delete prevDesc.dataset.userEdited;
      form.reset();
    };
    closeBtn.addEventListener("click", close);
    cancelBtn.addEventListener("click", close);
    root.addEventListener("click", (e) => {
      if (e.target === root) close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !root.hasAttribute("hidden")) close();
    });
    card.addEventListener("click", (e) => e.stopPropagation());

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const url = normalizeUrl(input.value);
      if (!url) return;
      if (tagsInput.value.trim()) {
        addTag(tagsInput.value);
        tagsInput.value = "";
      }
      const title = prevTitle.textContent.trim() || titleFromUrl(url);
      const summary = prevDesc.textContent.trim();
      const tagList = [...tags];
      submit.disabled = true;
      submitLabel.textContent = "Saving…";
      status.textContent = "";
      status.className = "kb-status";
      try {
        await saveBookmark(url, title, summary, tagList);
        status.textContent = "Saved to your library.";
        status.className = "kb-status ok";
        setTimeout(close, 900);
        window.dispatchEvent(
          new CustomEvent("knowledge:bookmark-added", { detail: { url } }),
        );
      } catch (err) {
        status.textContent = `Couldn't save — ${err.message || "try again"}`;
        status.className = "kb-status err";
        submit.disabled = false;
        submitLabel.textContent = "Save";
      }
    });

    return root;
  }

  async function open() {
    const me = await getMe();
    if (!me) {
      window.KnowledgeAuth?.open("login");
      return;
    }
    const root = ensureDialog();
    root.removeAttribute("hidden");
    const input = root.querySelector(".kb-input");
    setTimeout(() => input && input.focus(), 0);
  }

  function invalidateMe() {
    cachedMe = undefined;
  }

  window.KnowledgeBookmark = { open, invalidateMe };
})();
