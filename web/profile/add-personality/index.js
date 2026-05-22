/* Suggest a public personality — form that lands a row in
 * `personality_submissions` for the project owner to review by hand.
 *
 * Visual language mirrors the export dialog (paper card on a
 * blurred backdrop, ink-on-paper primary button, theme tokens only).
 * Self-contained; exposes `window.KnowledgeAddPersonality.open()`.
 *
 * Backend contract: POST /api/personalities (see
 * api/src/handlers/personalities.rs). The endpoint no longer
 * creates `users` rows; submissions queue up in
 * `personality_submissions` and are reviewed off-line. Nothing is
 * provisioned until the suggestion is manually promoted.
 */
(function () {
  "use strict";

  const ABS = window.KNOWLEDGE_API_BASE;

  let dialog = null;
  let bodyEl = null;
  let busy = false;
  // Edit-mode state. When true, the form posts to PUT /api/personalities/{slug}
  // (free, no entry-fee charge) instead of POST. The slug becomes
  // read-only because changing it would break every external link
  // to the library.
  let editing = false;
  let editSlug = null;

  function $(id) {
    return document.getElementById(id);
  }
  // escapeHtml comes from /lib/utils.js
  // escapeAttr comes from /lib/utils.js
  // Same slug normaliser the backend uses — a–z, 0–9, hyphens. We
  // mirror it client-side so the user sees the cleaned slug before
  // they submit. Server still validates.
  function slugify(s) {
    return String(s || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64);
  }
  // Probe shape badge — sits next to each handle input and turns
  // into ✓ / × / spinner after a debounced /api/profile/probe call.
  // The classes mirror the user-config (.probe-status.ok / .bad /
  // .probing) so profile.css does the styling.
  function probeBadgeHtml(kind, id) {
    return `<span class="probe-status" data-probe-for="${kind}" id="${id}" aria-live="polite"></span>`;
  }

  async function sendJson(method, path, body) {
    const r = await fetch(`${ABS}${path}`, {
      method,
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let payload = null;
    try {
      payload = await r.json();
    } catch {}
    if (!r.ok) {
      const text =
        payload && payload.error
          ? payload.error
          : typeof payload === "string"
            ? payload
            : `HTTP ${r.status}`;
      const err = new Error(text);
      err.status = r.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  function ensureDialog() {
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "addPersonalityDialog";
    dialog.className = "addp-dialog";
    dialog.setAttribute("aria-label", "Suggest a personality for review");
    bodyEl = document.createElement("div");
    bodyEl.className = "addp-body";
    dialog.appendChild(bodyEl);
    document.body.appendChild(dialog);
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) close();
    });
    return dialog;
  }
  /**
   * Open the modal.
   *   open()                                 → fresh creation form, free.
   *   open(prefill)                          → fresh form pre-filled (rare).
   *   open(prefill, { editing: true })       → edit mode against `prefill.slug`.
   */
  function open(prefill, opts) {
    editing = !!(opts && opts.editing);
    editSlug = editing && prefill ? prefill.slug || null : null;
    ensureDialog();
    render(prefill || {});
    if (typeof dialog.showModal === "function" && !dialog.open)
      dialog.showModal();
    else dialog.setAttribute("open", "");
  }
  function close() {
    if (!dialog) return;
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else dialog.removeAttribute("open");
    bodyEl.innerHTML = "";
    busy = false;
    editing = false;
    editSlug = null;
  }

  function render(prefill = {}) {
    const v = (k) => escapeAttr(prefill[k] || "");
    const title = editing
      ? `Edit @${escapeHtml(editSlug || "")}`
      : "Suggest a personality";
    const blurb = editing
      ? `Update the handles or bio for <strong>@${escapeHtml(editSlug || "")}</strong>.
         The slug stays the same so existing links don't break. Changes apply
         on the next pipeline run.`
      : `Spot someone whose tweets, papers, or repos belong on
         Knowledge? Send their public handles below. The project
         owner reviews each suggestion by hand — nothing is created
         the moment you submit. We'll get back to you once your
         suggestion is integrated.`;
    bodyEl.innerHTML = `
      <h2 class="addp-title">${title}</h2>
      <p class="addp-blurb">${blurb}</p>
      <form class="addp-form" id="addpForm">
        <label class="addp-field">
          <span class="addp-label">Name</span>
          <input id="addpName" type="text" required maxlength="200"
                 placeholder="Display name" autocomplete="off"
                 value="${v("name")}" />
        </label>
        <label class="addp-field">
          <span class="addp-label">Slug
            ${
              editing
                ? `<span class="addp-label-hint">locked — slug can't change once a library exists</span>`
                : `<span class="addp-label-hint">lowercase, hyphens</span>`
            }
          </span>
          <input id="addpSlug" type="text" maxlength="64"
                 placeholder="auto from name"
                 pattern="[a-z0-9-]+" autocomplete="off"
                 value="${v("slug")}"
                 ${editing ? "readonly" : ""} />
        </label>
        <label class="addp-field addp-field-wide">
          <span class="addp-label">Short bio <span class="addp-label-hint">optional</span></span>
          <textarea id="addpDescription" rows="2" maxlength="500"
                    placeholder="One line about who they are and why their feed is worth indexing.">${escapeHtml(prefill.description || "")}</textarea>
        </label>
        <fieldset class="addp-sources">
          <legend class="addp-sources-legend">Public handles <span class="addp-label-hint">fill any you know — at least one is required</span></legend>
          <div class="addp-row">
            <label class="addp-field">
              <span class="addp-label">X / Twitter ${probeBadgeHtml("twitter", "addpTwitterStatus")}</span>
              <div class="addp-input-prefix"><span>@</span>
                <input id="addpTwitter" type="text" maxlength="64"
                       placeholder="handle" autocomplete="off"
                       data-probe="twitter" value="${v("twitterHandle")}" />
              </div>
            </label>
            <label class="addp-field">
              <span class="addp-label">GitHub ${probeBadgeHtml("github", "addpGithubStatus")}</span>
              <div class="addp-input-prefix"><span>@</span>
                <input id="addpGithub" type="text" maxlength="64"
                       placeholder="username" autocomplete="off"
                       data-probe="github" value="${v("githubHandle")}" />
              </div>
            </label>
          </div>
          <div class="addp-row">
            <label class="addp-field">
              <span class="addp-label">Hugging Face ${probeBadgeHtml("huggingface", "addpHuggingfaceStatus")}</span>
              <div class="addp-input-prefix"><span>@</span>
                <input id="addpHuggingface" type="text" maxlength="64"
                       placeholder="username" autocomplete="off"
                       data-probe="huggingface" value="${v("huggingfaceHandle")}" />
              </div>
            </label>
            <label class="addp-field">
              <span class="addp-label">Reddit ${probeBadgeHtml("reddit", "addpRedditStatus")}</span>
              <div class="addp-input-prefix"><span>u/</span>
                <input id="addpReddit" type="text" maxlength="64"
                       placeholder="username" autocomplete="off"
                       data-probe="reddit" value="${v("redditHandle")}" />
              </div>
            </label>
          </div>
          <div class="addp-row">
            <label class="addp-field">
              <span class="addp-label">Hacker News <span class="addp-label-hint">username</span> ${probeBadgeHtml("hackernews_user", "addpHackernewsStatus")}</span>
              <input id="addpHackernews" type="text" maxlength="64"
                     placeholder="username" autocomplete="off"
                     data-probe="hackernews_user" value="${v("hackernewsHandle")}" />
            </label>
            <label class="addp-field">
              <span class="addp-label">Stack Overflow <span class="addp-label-hint">user id</span> ${probeBadgeHtml("stackoverflow", "addpStackoverflowStatus")}</span>
              <input id="addpStackoverflow" type="text" maxlength="32"
                     placeholder="numeric id" autocomplete="off"
                     data-probe="stackoverflow" value="${v("stackoverflowUserId")}" />
            </label>
          </div>
          <div class="addp-row">
            <label class="addp-field">
              <span class="addp-label">arXiv author ${probeBadgeHtml("arxiv", "addpArxivStatus")}</span>
              <input id="addpArxiv" type="text" maxlength="120"
                     placeholder="Full name as it appears on papers" autocomplete="off"
                     data-probe="arxiv" value="${v("arxivAuthor")}" />
            </label>
            <label class="addp-field">
              <span class="addp-label">DBLP author</span>
              <input id="addpDblp" type="text" maxlength="120"
                     placeholder="Full name as it appears on papers" autocomplete="off"
                     value="${v("dblpAuthor")}" />
            </label>
          </div>
          <label class="addp-field addp-field-wide">
            <span class="addp-label">Google Scholar <span class="addp-label-hint">user id from the profile URL</span> ${probeBadgeHtml("scholar", "addpScholarStatus")}</span>
            <input id="addpScholar" type="text" maxlength="32"
                   placeholder="the ?user= value from the Scholar URL" autocomplete="off"
                   data-probe="scholar" value="${v("scholarUserId")}" />
          </label>
          <label class="addp-field addp-field-wide">
            <span class="addp-label">Websites <span class="addp-label-hint">one URL per line — RSS feeds, sitemaps, or any page</span></span>
            <textarea id="addpWebsites" rows="3" maxlength="2000"
                      placeholder="https://example.com/blog/&#10;https://example.com/feed.xml&#10;https://example.com/sitemap.xml">${escapeHtml(prefill.websites || "")}</textarea>
            <ul class="websites-status" id="addpWebsitesStatus"></ul>
          </label>
        </fieldset>
        <p class="addp-fineprint">
          ${
            editing
              ? "Anyone will be able to browse the library once it's created. Sources stay editable from the personality's own settings later."
              : "Your suggestion lands in a review queue — nothing is provisioned automatically. The project owner integrates approved personalities by hand."
          }
        </p>
        <div class="addp-msg" id="addpMsg" hidden></div>
        <div class="addp-actions">
          <button type="button" data-addp-close>Cancel</button>
          <button type="submit" class="primary" id="addpSubmit">
            ${editing ? "Save changes" : "Submit for review"}
          </button>
        </div>
      </form>
    `;
    wire(prefill);
  }

  // ── Live probes ─────────────────────────────────────────────────────
  // Single-handle probes (Twitter, GitHub, …) hit /api/profile/probe
  // with {kind, value} and paint a ✓ / × badge next to the field.
  // Debounced per input so we don't fire on every keystroke.
  const probeTimers = new WeakMap();
  function setProbeUI(kind, status, info, error) {
    const el = bodyEl?.querySelector(`[data-probe-for="${kind}"]`);
    if (!el) return;
    el.className = `probe-status ${status || ""}`.trim();
    if (status === "idle" || !status) {
      el.innerHTML = "";
      return;
    }
    const glyph = status === "ok" ? "✓" : status === "bad" ? "×" : "·"; // probing
    const meta =
      status === "ok"
        ? info || "ok"
        : status === "bad"
          ? error || "not found"
          : "checking…";
    el.innerHTML = `<span class="glyph">${glyph}</span><span class="meta">${escapeHtml(meta)}</span>`;
  }
  async function runHandleProbe(input) {
    const kind = input.dataset.probe;
    if (!kind) return;
    const raw = (input.value || "").split(/[\n,]/)[0].trim();
    clearTimeout(probeTimers.get(input));
    if (!raw) {
      setProbeUI(kind, "idle");
      return;
    }
    setProbeUI(kind, "probing");
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`${ABS}/api/profile/probe`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind, value: raw }),
        });
        const d = await r.json().catch(() => ({}));
        if (d.ok) setProbeUI(kind, "ok", d.info);
        else setProbeUI(kind, "bad", null, d.error || "not found");
      } catch {
        setProbeUI(kind, "bad", null, "network");
      }
    }, 600);
    probeTimers.set(input, t);
  }

  // Multi-URL website probe — mirrors profile/page.js. Each line in
  // the textarea gets its own row in #addpWebsitesStatus showing
  // "checking…" / "RSS valid" / "sitemap · filter: /blog/" / etc.
  const websiteCache = new Map();
  const websiteInflight = new Map();
  let websitesDebTimer = null;
  function splitWebsitesText(text) {
    return Array.from(
      new Set(
        (text || "")
          .split(/\n+/)
          .map((x) => x.trim())
          .filter(Boolean),
      ),
    );
  }
  function renderWebsitesStatus() {
    const list = bodyEl?.querySelector("#addpWebsitesStatus");
    const ta = bodyEl?.querySelector("#addpWebsites");
    if (!list || !ta) return;
    const urls = splitWebsitesText(ta.value);
    if (!urls.length) {
      list.innerHTML = "";
      return;
    }
    list.innerHTML = urls
      .map((u) => {
        const p = websiteCache.get(u) || { status: "probing" };
        const cls = p.status;
        const kindBadge =
          p.status === "ok" && p.kind
            ? `<span class="kind ${p.kind}">${p.kind === "feed" ? "RSS" : "sitemap"}</span>`
            : "";
        const msg =
          p.status === "probing"
            ? "checking…"
            : p.status === "ok"
              ? p.info || "valid"
              : p.error || "invalid";
        return `<li class="${cls}">
        <div class="websites-row">
          <span class="websites-url">${escapeHtml(u)}</span>
          <span class="websites-msg">${kindBadge}<span>${escapeHtml(msg)}</span></span>
        </div>
      </li>`;
      })
      .join("");
  }
  function probeWebsites() {
    const ta = bodyEl?.querySelector("#addpWebsites");
    if (!ta) return;
    const urls = splitWebsitesText(ta.value);
    // Cancel in-flight probes for URLs that the user removed.
    for (const [u, ctrl] of websiteInflight) {
      if (!urls.includes(u)) {
        ctrl.abort();
        websiteInflight.delete(u);
      }
    }
    for (const u of urls) {
      if (websiteCache.has(u) && websiteCache.get(u).status !== "probing")
        continue;
      if (websiteInflight.has(u)) continue;
      websiteCache.set(u, { status: "probing" });
      const ctrl = new AbortController();
      websiteInflight.set(u, ctrl);
      fetch(`${ABS}/api/profile/probe`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "website", value: u }),
        signal: ctrl.signal,
      })
        .then((r) => r.json())
        .then((d) => {
          const next = d.ok
            ? { status: "ok", info: d.info, kind: d.kind || null }
            : {
                status: "bad",
                error: d.error || "invalid",
                kind: d.kind || null,
              };
          websiteCache.set(u, next);
          websiteInflight.delete(u);
          renderWebsitesStatus();
        })
        .catch((err) => {
          if (err.name !== "AbortError") {
            websiteCache.set(u, { status: "bad", error: "network error" });
            websiteInflight.delete(u);
            renderWebsitesStatus();
          }
        });
    }
    renderWebsitesStatus();
  }

  function wire(prefill) {
    bodyEl
      .querySelectorAll("[data-addp-close]")
      .forEach((b) => b.addEventListener("click", close));
    // Bind handle-input probes. Fires once on render so pre-filled
    // edit-mode fields turn green immediately when they're valid.
    bodyEl.querySelectorAll("[data-probe]").forEach((input) => {
      input.addEventListener("input", () => runHandleProbe(input));
      runHandleProbe(input);
    });
    // Wire the websites textarea: re-render the status list on
    // every keystroke (cheap), debounce the network probe.
    const websitesInput = bodyEl.querySelector("#addpWebsites");
    if (websitesInput) {
      websitesInput.addEventListener("input", () => {
        clearTimeout(websitesDebTimer);
        renderWebsitesStatus();
        websitesDebTimer = setTimeout(probeWebsites, 550);
      });
      renderWebsitesStatus();
      probeWebsites();
    }
    const nameInput = $("addpName");
    const slugInput = $("addpSlug");
    const slugPreview = $("addpSlugPreview");
    // Slug preview lives in the blurb in create mode. In edit mode
    // the slug is locked so the preview element doesn't exist.
    const updatePreview = () => {
      if (!slugPreview) return;
      const s = slugInput.value.trim()
        ? slugify(slugInput.value)
        : slugify(nameInput.value);
      slugPreview.textContent = s || "name";
    };
    if (slugPreview) {
      nameInput.addEventListener("input", updatePreview);
      slugInput.addEventListener("input", updatePreview);
      updatePreview();
    }

    $("addpForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      if (busy) return;
      busy = true;
      const submitBtn = $("addpSubmit");
      const msg = $("addpMsg");
      submitBtn.disabled = true;
      submitBtn.textContent = editing ? "Saving…" : "Submitting…";
      msg.hidden = true;
      msg.className = "addp-msg";

      const payload = {
        name: nameInput.value.trim(),
        slug: slugInput.value.trim() || slugify(nameInput.value),
        description: $("addpDescription").value.trim(),
        twitterHandle: $("addpTwitter").value.trim(),
        githubHandle: $("addpGithub").value.trim(),
        huggingfaceHandle: $("addpHuggingface").value.trim(),
        redditHandle: $("addpReddit").value.trim(),
        hackernewsHandle: $("addpHackernews").value.trim(),
        stackoverflowUserId: $("addpStackoverflow").value.trim(),
        arxivAuthor: $("addpArxiv").value.trim(),
        dblpAuthor: $("addpDblp").value.trim(),
        scholarUserId: $("addpScholar").value.trim(),
        // Multi-line textarea: server splits on newlines / commas
        // and dedupes.
        websites: $("addpWebsites").value,
      };

      try {
        if (editing) {
          await sendJson(
            "PUT",
            `/api/personalities/${encodeURIComponent(editSlug)}`,
            payload,
          );
          submitBtn.textContent = "Saved";
          msg.hidden = false;
          msg.classList.add("ok");
          msg.innerHTML = `Updated <strong>@${escapeHtml(editSlug)}</strong>. Changes apply on the next pipeline run.`;
          // Refresh the "Personalities you've added" table so the
          // edited row picks up the new fields without a full page
          // reload.
          if (window.KnowledgeMyPersonalities?.reload) {
            window.KnowledgeMyPersonalities.reload();
          }
          setTimeout(close, 1000);
        } else {
          const resp = await sendJson("POST", "/api/personalities", payload);
          submitBtn.textContent = "Submitted";
          msg.hidden = false;
          msg.classList.add("ok");
          msg.innerHTML =
            `Thanks — your suggestion for <strong>@${escapeHtml(resp.slug)}</strong> ` +
            `is in the review queue. The project owner integrates personalities ` +
            `by hand, so it might take a few days before the library shows up on ` +
            `Knowledge.`;
          // Close the dialog after a beat — no redirect, no new
          // library to open since nothing was provisioned.
          setTimeout(close, 1800);
        }
      } catch (err) {
        busy = false;
        submitBtn.disabled = false;
        submitBtn.textContent = editing ? "Save changes" : "Submit for review";
        msg.hidden = false;
        msg.classList.add("error");
        if (err.status === 402) {
          // Should not happen anymore (creation is free) but keep
          // the handler so an old build of the API still surfaces a
          // useful message instead of a generic error.
          msg.innerHTML = `Insufficient balance for this action. Add funds in the Balance section, then try again.`;
        } else if (err.status === 409) {
          // Backend hands us {error, field, existingSlug, existingName}
          // when the conflict is a slug/name/twitter dup. Show a
          // clickable link to the existing personality so the user
          // can jump straight to their profile.
          const p = err.payload || {};
          if (p.existingSlug) {
            const safeSlug = escapeHtml(p.existingSlug);
            const safeName = escapeHtml(p.existingName || p.existingSlug);
            msg.innerHTML =
              `${escapeHtml(err.message || "Already on Knowledge.")} ` +
              `<a href="/?libs=${encodeURIComponent(p.existingSlug)}" target="_blank" rel="noopener">View ${safeName}'s profile (@${safeSlug})</a>`;
          } else {
            msg.innerHTML = escapeHtml(err.message || "Already on Knowledge.");
          }
        } else if (err.status === 401) {
          msg.innerHTML = `Sign in first.`;
        } else if (err.status === 404 && editing) {
          msg.innerHTML = `You don't have permission to edit @${escapeHtml(editSlug)}.`;
        } else {
          msg.innerHTML = escapeHtml(err.message || "Could not save changes.");
        }
      }
    });
  }

  // Document-level delegation so any element marked as the trigger
  // opens the modal — works regardless of when this script loaded.
  document.addEventListener("click", (e) => {
    const btn = e.target.closest('[data-action="add-personality"]');
    if (!btn) return;
    e.preventDefault();
    open();
  });

  window.KnowledgeAddPersonality = { open };
})();
