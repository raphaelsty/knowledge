/* profile/page.js — standalone profile editor at /profile.
 *
 * Self-contained: no search engine, no rail, no result list. The page
 * loads /auth/me, hydrates every input, wires live URL probes,
 * auto-save and the browser-side sync orchestrator, and posts any
 * secret blobs back to their dedicated endpoints.
 *
 * Page-level globals consumed (loaded by profile.html in order):
 *   - window.KNOWLEDGE_API_BASE          (lib/utils.js)
 *   - window.escapeHtml / escapeAttr     (lib/utils.js)
 *   - window.K_getJson                   (lib/utils.js)
 *   - window.KnowledgeMoney              (lib/money.js)
 *   - window.KnowledgeAPI                (search/api.js, reused for
 *                                         the same fetch + cache layer)
 *
 * Sibling subviews (web/profile/credits/, /storage/, /add-personality/,
 * my-personalities.js) attach their own click handlers and read
 * `me` from `window` after this script publishes it.
 */
(async function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const API_BASE = window.KNOWLEDGE_API_BASE;
  const K = window.KnowledgeAPI || {
    // Inert fallbacks so probe caching just turns into a no-op if api.js
    // didn't load. The page still works, it just probes more often.
    probeCacheGet: () => null,
    probeCacheSet: () => {},
    invalidateCaches: () => {},
    invalidateUnindexed: () => {},
  };

  // escapeHtml + escapeAttr come from /lib/utils.js

  // ── Theme toggle (segmented Light / Dark) ───────────────────────────
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "dark"
      ? "dark"
      : "light";
  }
  function applyTheme(next) {
    // Always set the attribute (including for light) so CSS vars under
    // [data-theme="light"|"dark"] resolve.
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch {}
    syncThemePicker();
  }
  function syncThemePicker() {
    const cur = currentTheme();
    document.querySelectorAll("#themeToggle .theme-opt").forEach((b) => {
      const on = b.dataset.themeValue === cur;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-checked", String(on));
    });
  }
  syncThemePicker();
  document.querySelectorAll("#themeToggle .theme-opt").forEach((b) => {
    b.addEventListener("click", () => applyTheme(b.dataset.themeValue));
  });

  // ── Auth ─────────────────────────────────────────────────────────────
  let me = null;
  async function loadMe() {
    try {
      const r = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
      if (!r.ok) return null;
      return await r.json();
    } catch {
      return null;
    }
  }
  me = await loadMe();
  if (!me) {
    // Anonymous on /profile → bounce them back to the feed and pop
    // the login modal so they can sign in / sign up.
    location.href = "/?signin=1";
    return;
  }

  // Twitter ingestion is the only paid source and is currently
  // gated to VIPs while the commerce side is being legally vetted.
  // The `vip` flag isn't on /auth/me; fetch the public user record
  // (already cached server-side) to learn it. Errors stay opaque —
  // we fall through as non-VIP, which is the safer default.
  try {
    const u = await fetch(
      `${API_BASE}/api/users/${encodeURIComponent(me.slug)}`,
    ).then((r) => (r.ok ? r.json() : null));
    me.vip = !!(u && u.vip);
  } catch {
    me.vip = false;
  }
  window.K_meVip = me.vip;
  document.documentElement.toggleAttribute("data-vip", me.vip);

  // Behavioural tracker — on /profile the viewer is also the personality
  // being browsed (it's the settings page for your own library), so
  // both ids resolve to `me`.
  if (window.kn && me.id) {
    window.kn.setViewer({ id: me.id });
    window.kn.setPersonality({ id: me.id, slug: me.slug });
    window.kn.track("view", {
      user_id: me.id,
      personality_slug: me.slug,
    });
  }

  // Admin-only entry point. Reveal the Settings → Admin link iff the
  // signed-in slug matches the single operator account. Mirrors the
  // server-side `require_raphael` guard in api/src/handlers/admin.rs
  // — both sides hard-code the same slug. The link is just a
  // shortcut; manual visits to /admin still go through the same
  // session-cookie + slug check, so flipping this row doesn't widen
  // the trust boundary.
  if (me.slug === "raphael-sourty") {
    const row = document.getElementById("pfAdminRow");
    if (row) row.hidden = false;
    const dokploy = document.getElementById("pfDokployRow");
    if (dokploy) dokploy.hidden = false;
  }

  // Toggle the Twitter card's alpha gate. VIPs see the regular paid
  // copy + the storage widget; everyone else sees the alpha notice
  // and has the handle field locked (still readable, just not
  // editable until we open it up).
  (function applyTwitterAlphaGate() {
    const alphaNote = document.querySelector("[data-alpha-notice]");
    const handle = document.getElementById("pfTwitter");
    document
      .querySelectorAll("[data-vip-only]")
      .forEach((el) => (el.hidden = !me.vip));
    if (alphaNote) alphaNote.hidden = !!me.vip;
    if (handle && !me.vip) {
      handle.disabled = true;
      handle.setAttribute("aria-disabled", "true");
      handle.title = "Tweets extraction is paused while the app is in alpha.";
    }
    // Storage panel boots on DOMContentLoaded, which can fire
    // before the VIP flag lands. Trigger a reload now that we know,
    // so VIPs see the billing hero and non-VIPs see nothing.
    window.KnowledgeStorage?.reload?.();
  })();

  /* Mobile bottom-nav wiring. The Settings tab is already marked
   * `is-current` in the HTML; we only need to point Personal at
   * the signed-in user's library. People + Feed already have the
   * right hrefs. */
  (function wireMobileNav() {
    const personal = document.getElementById("mbnPersonal");
    if (personal && me?.slug) {
      personal.href = `/search?libs=${encodeURIComponent(me.slug)}`;
    }
  })();

  /* Resolve the avatar shown in the settings hero. We removed the
   * GitHub OAuth flow, but most users still link their handle in
   * `sources.github`; GitHub serves a stable avatar PNG at
   * `https://github.com/<login>.png` for any public account, so we
   * use it as the implicit default when the user hasn't uploaded
   * their own. The <img> falls back to empty on 404 via `onerror`. */
  function resolveDefaultAvatar(meRow) {
    if (meRow.avatar) return meRow.avatar;
    const ghRaw = meRow.sources && meRow.sources.github;
    let ghLogin = "";
    if (Array.isArray(ghRaw) && ghRaw.length) {
      ghLogin = String(ghRaw[0] || "").trim();
    } else if (ghRaw && typeof ghRaw === "object" && ghRaw.username) {
      ghLogin = String(ghRaw.username).trim();
    } else if (typeof ghRaw === "string") {
      ghLogin = ghRaw.trim();
    }
    if (!ghLogin) return "";
    // GitHub also accepts /<login>.png?size=N to scale — 240 covers
    // hero + rail uses without burning bandwidth on the originals.
    return `https://github.com/${encodeURIComponent(ghLogin)}.png?size=240`;
  }

  /* Pull initials from a display name / slug. Mirrors the personal-
   * page fallback in web/search/page.js so the same user gets the
   * same letter avatar on both pages when no image is available. */
  function initialsFor(meRow) {
    return (meRow.name || meRow.slug || "?")
      .split(/\s+/)
      .slice(0, 2)
      .map((w) => (w[0] || "").toUpperCase())
      .join("");
  }

  /* Swap the hero <img> for an initials circle (or vice-versa).
   * Used on first hydrate AND when the GitHub avatar 404s, so the
   * fallback stays in sync with whatever me.avatar actually is. */
  function applyHeroAvatar(meRow) {
    const img = $("pfAvatar");
    const url = resolveDefaultAvatar(meRow);
    const initials = initialsFor(meRow);
    if (url) {
      img.style.display = "";
      img.src = url;
      img.alt = meRow.name || meRow.slug || "";
      img.onerror = () => {
        // GitHub /<login>.png 404s on deleted/private accounts or
        // when the user typed a slug that isn't a real GH handle.
        // Drop in the initials fallback instead of a broken image.
        img.onerror = null;
        renderInitialsFallback(initials);
      };
    } else {
      renderInitialsFallback(initials);
    }
  }

  function renderInitialsFallback(initials) {
    const img = $("pfAvatar");
    img.style.display = "none";
    let fallback = document.getElementById("pfAvatarFallback");
    if (!fallback) {
      fallback = document.createElement("span");
      fallback.id = "pfAvatarFallback";
      fallback.className = "hero-avatar hero-avatar-fallback";
      fallback.setAttribute("aria-hidden", "true");
      img.insertAdjacentElement("afterend", fallback);
    }
    fallback.textContent = initials;
  }

  // ── Hydrate the form ─────────────────────────────────────────────────
  function fillForm() {
    applyHeroAvatar(me);
    $("pfHandle").textContent = `@${me.githubLogin || me.slug || ""}`;
    $("pfName").value = me.name || "";
    $("pfDesc").value = me.description || "";
    $("pfPublic").checked = !!me.public;
    const sx = me.sources || {};
    $("pfGithub").value = Array.isArray(sx.github)
      ? sx.github.join(", ")
      : (sx.github && sx.github.username) || "";
    $("pfTwitter").value = (sx.twitter && sx.twitter.username) || "";
    $("pfReddit").value = (sx.reddit && sx.reddit.username) || "";
    $("pfSO").value = (sx.stackoverflow && sx.stackoverflow.user_id) || "";
    $("pfHF").value = (sx.huggingface && sx.huggingface.username) || "";
    $("pfArxiv").value = (sx.arxiv && sx.arxiv.author) || "";
    $("pfScholar").value = (sx.scholar && sx.scholar.user_id) || "";
    $("pfDblp").value = (sx.dblp && sx.dblp.author) || "";
    $("pfHnUser").value =
      (sx.hackernews && sx.hackernews.username) || me.hackernewsUsername || "";
    $("pfWebsites").value = Array.isArray(sx.websites)
      ? sx.websites
          .map((w) => (typeof w === "string" ? w : w.input || w.url))
          .filter(Boolean)
          .join("\n")
      : "";
    // Secrets are never round-tripped — clear them so plaintext never lingers.
    // Defensive against removed inputs (e.g. the X / Twitter cookies fields
    // were retired — we now push users to the paid Twitter API).
    ["pfHnPassword", "pfTwAuth", "pfTwCt0", "pfZotero"].forEach((id) => {
      const el = $(id);
      if (el) el.value = "";
    });
    document.querySelectorAll("[data-probe]").forEach(runProbe);
    // Seed websites cache from saved entries (skip "checking…" for known-good lines).
    const sxw = (me.sources && me.sources.websites) || [];
    for (const w of sxw) {
      const input = typeof w === "string" ? w : w.input || w.url;
      if (!input) continue;
      if (typeof w === "object" && w.url) {
        websiteCache.set(input, {
          status: "ok",
          kind: w.kind || (w.url_filter !== undefined ? "sitemap" : "feed"),
          resolvedUrl: w.url,
          resolvedFilter: w.url_filter !== undefined ? w.url_filter : null,
          info: w.url_filter ? `filter: ${w.url_filter}` : "",
          subtrees: null,
        });
      }
    }
    renderWebsitesStatus();
    probeWebsites();
    refreshHnConnBadge();
    setHnVerify("", "");
    renderZoteroConnection();
    renderTwitterConnection();
    maybePrefillFromGithub();
  }

  // ── Live source probes (kind="github" / "twitter" / etc.) ────────────
  const probeTimers = new WeakMap();
  function setProbeUI(kind, status, info, error) {
    const slot = document.querySelector(
      `.probe-status[data-probe-for="${kind}"]`,
    );
    if (!slot) return;
    slot.classList.remove("probing", "ok", "bad");
    if (status === "idle") {
      slot.innerHTML = "";
      return;
    }
    slot.classList.add(status);
    if (status === "probing") {
      slot.innerHTML = `<span class="glyph">·</span>`;
      return;
    }
    const glyph = status === "ok" ? "✓" : "✕";
    const meta =
      status === "ok" && info
        ? info.count != null
          ? `${info.count}`
          : info.label || info.name || ""
        : status === "bad"
          ? error || "not found"
          : "";
    slot.innerHTML = `<span class="glyph">${glyph}</span>${meta ? `<span class="meta">${escapeHtml(String(meta))}</span>` : ""}`;
  }
  function runProbe(input) {
    const kind = input.dataset.probe;
    const raw = (input.value || "").split(/[\n,]/)[0].trim();
    clearTimeout(probeTimers.get(input));
    if (!raw) {
      setProbeUI(kind, "idle");
      return;
    }
    const cached = K.probeCacheGet(kind, raw);
    if (cached) {
      setProbeUI(kind, cached.status, cached.info, cached.error);
      return;
    }
    setProbeUI(kind, "probing");
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/profile/probe`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind, value: raw }),
        });
        const d = await r.json().catch(() => ({}));
        const next = d.ok
          ? { status: "ok", info: d.info }
          : { status: "bad", error: d.error || "invalid" };
        K.probeCacheSet(kind, raw, next);
        setProbeUI(kind, next.status, next.info, next.error);
      } catch {
        setProbeUI(kind, "bad", null, "network");
      }
    }, 700);
    probeTimers.set(input, t);
  }
  document.addEventListener("input", (e) => {
    if (e.target.matches && e.target.matches("[data-probe]"))
      runProbe(e.target);
  });

  // ── Secret show/hide eye toggle ──────────────────────────────────────
  document.addEventListener("click", (e) => {
    const btn = e.target.closest && e.target.closest(".secret-eye");
    if (!btn) return;
    e.preventDefault();
    const id = btn.dataset.secretToggle;
    const inp = id && $(id);
    if (!inp) return;
    const reveal = inp.type === "password";
    inp.type = reveal ? "text" : "password";
    btn.setAttribute("aria-pressed", String(reveal));
  });

  // ── Websites — per-line probes ───────────────────────────────────────
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
    const urls = splitWebsitesText($("pfWebsites").value);
    if (!urls.length) {
      $("pfWebsitesStatus").innerHTML = "";
      return;
    }
    $("pfWebsitesStatus").innerHTML = urls
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
        const subtrees =
          Array.isArray(p.subtrees) && p.subtrees.length
            ? `<div class="websites-subtrees"><span class="websites-subtrees-label">narrow to:</span>${p.subtrees
                .map(
                  (s) =>
                    `<button type="button" class="websites-chip" data-url="${escapeAttr(u)}" data-path="${escapeAttr(s.path)}"><code>${escapeHtml(s.path)}</code><span class="websites-chip-count">${escapeHtml(String(s.count))}</span></button>`,
                )
                .join("")}</div>`
            : "";
        return `<li class="${cls}"><div class="websites-row"><span class="websites-url">${escapeHtml(u)}</span><span class="websites-msg">${kindBadge}<span>${escapeHtml(msg)}</span></span></div>${subtrees}</li>`;
      })
      .join("");
    $("pfWebsitesStatus")
      .querySelectorAll(".websites-chip")
      .forEach((b) => {
        b.addEventListener("click", () => {
          const u = b.dataset.url;
          const path = b.dataset.path;
          const baseMatch = u.match(/^https?:\/\/[^/]+/i);
          if (!baseMatch) return;
          const replacement = baseMatch[0] + path;
          const lines = $("pfWebsites").value.split("\n");
          const idx = lines.findIndex((l) => l.trim() === u);
          if (idx === -1) return;
          lines[idx] = replacement;
          $("pfWebsites").value = lines.join("\n");
          probeWebsites();
        });
      });
  }
  function probeWebsites() {
    const urls = splitWebsitesText($("pfWebsites").value);
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
      const cached = K.probeCacheGet("website", u);
      if (cached && cached.status === "ok") {
        websiteCache.set(u, cached);
        continue;
      }
      websiteCache.set(u, { status: "probing" });
      const ctrl = new AbortController();
      websiteInflight.set(u, ctrl);
      fetch(`${API_BASE}/api/profile/probe`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "website", value: u }),
        signal: ctrl.signal,
      })
        .then((r) => r.json())
        .then((d) => {
          const next = d.ok
            ? {
                status: "ok",
                info: d.info,
                kind: d.kind || null,
                resolvedUrl: d.resolvedUrl || null,
                resolvedFilter:
                  d.resolvedFilter !== undefined ? d.resolvedFilter : null,
                subtrees: Array.isArray(d.subtrees) ? d.subtrees : null,
              }
            : {
                status: "bad",
                error: d.error || "invalid",
                subtrees: Array.isArray(d.subtrees) ? d.subtrees : null,
                kind: d.kind || null,
              };
          websiteCache.set(u, next);
          websiteInflight.delete(u);
          if (next.status === "ok") K.probeCacheSet("website", u, next);
          renderWebsitesStatus();
          document.dispatchEvent(new CustomEvent("websites:probe-done"));
        })
        .catch((err) => {
          if (err.name !== "AbortError") {
            websiteCache.set(u, { status: "bad", error: "network error" });
            websiteInflight.delete(u);
            renderWebsitesStatus();
            document.dispatchEvent(new CustomEvent("websites:probe-done"));
          }
        });
    }
    renderWebsitesStatus();
  }
  $("pfWebsites").addEventListener("input", () => {
    clearTimeout(websitesDebTimer);
    renderWebsitesStatus();
    websitesDebTimer = setTimeout(probeWebsites, 550);
  });

  // ── GitHub bio prefill (only blanks) ─────────────────────────────────
  let prefillDoneFor = null;
  async function maybePrefillFromGithub() {
    const login = me && (me.githubLogin || me.slug);
    if (!login || prefillDoneFor === login) return;
    prefillDoneFor = login;
    try {
      const r = await fetch(
        `https://api.github.com/users/${encodeURIComponent(login)}`,
      );
      if (!r.ok) return;
      const gh = await r.json();
      if (!$("pfName").value) $("pfName").value = gh.name || gh.login || "";
      if (!$("pfDesc").value) $("pfDesc").value = gh.bio || "";
    } catch {}
  }

  // ── composeSources — same shape the API expects ──────────────────────
  function composeSources() {
    const out = { ...(me.sources || {}) };
    const ghLines = $("pfGithub")
      .value.split(/[\n,]/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (ghLines.length) out.github = ghLines;
    else delete out.github;

    const setOrDelete = (key, primary, body) => {
      if (primary) out[key] = { ...(out[key] || {}), ...body };
      else delete out[key];
    };
    const tw = $("pfTwitter").value.trim();
    const rd = $("pfReddit").value.trim();
    const so = $("pfSO").value.trim();
    const hf = $("pfHF").value.trim();
    const ar = $("pfArxiv").value.trim();
    const sc = $("pfScholar").value.trim();
    const db = $("pfDblp").value.trim();
    const hn = $("pfHnUser").value.trim();
    setOrDelete("twitter", tw, { username: tw });
    setOrDelete("reddit", rd, { username: rd });
    setOrDelete("stackoverflow", so, { user_id: so });
    setOrDelete("huggingface", hf, { username: hf });
    setOrDelete("arxiv", ar, {
      author: ar,
      max_results: (out.arxiv && out.arxiv.max_results) || 50,
    });
    setOrDelete("scholar", sc, { user_id: sc });
    setOrDelete("dblp", db, {
      author: db,
      max_results: (out.dblp && out.dblp.max_results) || 200,
    });
    setOrDelete("hackernews", hn, { username: hn });

    const urls = splitWebsitesText($("pfWebsites").value);
    const resolved = urls
      .map((u) => {
        const p = websiteCache.get(u);
        if (!p || p.status !== "ok" || !p.resolvedUrl) return null;
        const kind = p.kind === "feed" ? "feed" : "sitemap";
        const entry = { input: u, kind, url: p.resolvedUrl, tags: ["blog"] };
        if (kind === "sitemap") entry.url_filter = p.resolvedFilter || "";
        return entry;
      })
      .filter(Boolean);
    if (resolved.length) out.websites = resolved;
    else delete out.websites;
    delete out.blog;
    delete out.sitemap;
    return out;
  }

  // ── Auto-save (debounced 800 ms; coalesces concurrent in-flights) ────
  let autoSaveTimer = null;
  let autoSaveLastBody = null;
  let autoSaveStatusTimer = null;
  let autoSaveInflight = false;
  let autoSaveDeferred = false;
  let autoSavePriority = false;

  function setStatus(text, ok = false) {
    const el = $("pfStatus");
    el.textContent = text;
    el.classList.toggle("ok", !!ok);
    clearTimeout(autoSaveStatusTimer);
    if (ok) {
      autoSaveStatusTimer = setTimeout(() => {
        el.textContent = "Auto-saves as you type";
        el.classList.remove("ok");
      }, 1800);
    }
  }

  async function autoSaveNow() {
    if (!me) return;
    if (autoSaveInflight) {
      autoSaveDeferred = true;
      return;
    }
    // Preserve any external `links` we don't manage from this page
    // (Twitter URL etc. live elsewhere in the product).
    const links = { ...(me.links || {}) };
    const body = {
      name: $("pfName").value.trim(),
      description: $("pfDesc").value.trim(),
      public: $("pfPublic").checked,
      links,
      sources: composeSources(),
    };
    const serialized = JSON.stringify(body);
    if (serialized === autoSaveLastBody) return;
    autoSaveLastBody = serialized;
    autoSaveInflight = true;
    setStatus("saving…");
    try {
      const r = await fetch(`${API_BASE}/api/users/me`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: serialized,
      });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      const updated = await r.json().catch(() => ({ ...me, ...body }));
      me = updated;
      K.invalidateCaches(me.slug);
      // /api/users/{slug} ships with a 60s browser cache, so opening
      // /<slug> right after saving still shows the old bio. Replay the
      // GET with `cache: 'reload'` to evict the stale entry from the
      // browser's HTTP cache. Fire-and-forget — we don't use the body,
      // just the side effect on the cache.
      fetch(`${API_BASE}/api/users/${encodeURIComponent(me.slug)}`, {
        cache: "reload",
        credentials: "include",
      }).catch(() => {});
      setStatus("saved", true);
    } catch (err) {
      autoSaveLastBody = null;
      setStatus(`couldn't save · ${err.message || err}`);
    } finally {
      autoSaveInflight = false;
      if (autoSaveDeferred) {
        autoSaveDeferred = false;
        if (autoSavePriority) {
          autoSavePriority = false;
          autoSaveNow();
        } else scheduleAutoSave();
      }
    }
  }
  function scheduleAutoSave() {
    clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(autoSaveNow, 800);
  }

  [
    "pfName",
    "pfDesc",
    "pfGithub",
    "pfTwitter",
    "pfReddit",
    "pfSO",
    "pfHF",
    "pfArxiv",
    "pfScholar",
    "pfDblp",
    "pfHnUser",
    "pfWebsites",
  ].forEach((id) => $(id).addEventListener("input", scheduleAutoSave));
  $("pfPublic").addEventListener("change", () => {
    clearTimeout(autoSaveTimer);
    autoSavePriority = true;
    autoSaveNow();
  });
  document.addEventListener("websites:probe-done", scheduleAutoSave);

  // ── HN verify + connection badge + disconnect ────────────────────────
  function setHnVerify(status, msg) {
    const el = $("pfHnVerifyStatus");
    el.className = `verify-status ${status || ""}`;
    const glyph =
      status === "ok"
        ? "✓"
        : status === "bad"
          ? "✕"
          : status === "probing"
            ? "·"
            : "";
    el.innerHTML = status
      ? `<span class="glyph">${glyph}</span><span>${escapeHtml(msg || "")}</span>`
      : "";
  }
  function refreshHnConnBadge() {
    const badge = $("hnConnBadge");
    const dis = $("pfHnDisconnect");
    const hasStoredPwd = !!(me && me.hasHackernewsUpvotes);
    if (hasStoredPwd) {
      badge.className = "conn-badge";
      badge.textContent = "Connected";
      dis.style.display = "inline-block";
    } else {
      badge.textContent = "";
      dis.style.display = "none";
    }
  }
  $("pfHnVerify").addEventListener("click", async () => {
    const username = $("pfHnUser").value.trim();
    const password = $("pfHnPassword").value;
    if (!username) {
      setHnVerify("bad", "Enter a username first");
      return;
    }
    const hasStoredPwd = !!(me && me.hasHackernewsUpvotes);
    if (!password && !hasStoredPwd) {
      setHnVerify("bad", "Enter a password or save credentials first");
      return;
    }
    setHnVerify("probing", "Checking…");
    try {
      const opts = password
        ? {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
          }
        : { method: "GET", credentials: "include" };
      const r = await fetch(`${API_BASE}/auth/me/hackernews/test`, opts);
      const d = await r.json().catch(() => ({}));
      if (d.ok) setHnVerify("ok", d.info || "Login works");
      else setHnVerify("bad", d.error || "Login failed");
    } catch {
      setHnVerify("bad", "Network error");
    }
  });
  $("pfHnDisconnect").addEventListener("click", async () => {
    if (
      !confirm(
        "Disconnect Hacker News? Wipes the saved password (the username connection stays).",
      )
    )
      return;
    try {
      const r = await fetch(`${API_BASE}/auth/me/hackernews`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      me = await r.json().catch(() => me);
      $("pfHnPassword").value = "";
      setHnVerify("", "");
      refreshHnConnBadge();
      setStatus("HN disconnected", true);
    } catch (e) {
      setHnVerify("bad", `Disconnect failed · ${e.message || e}`);
    }
  });

  // ── Secret saves on blur ─────────────────────────────────────────────
  $("pfHnPassword").addEventListener("blur", async () => {
    const pwd = $("pfHnPassword").value;
    const user = $("pfHnUser").value.trim();
    if (!pwd || !user) return;
    setStatus("saving HN credentials…");
    try {
      const r = await fetch(`${API_BASE}/auth/me/hackernews`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: user, password: pwd }),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      me = await r.json().catch(() => me);
      $("pfHnPassword").value = "";
      refreshHnConnBadge();
      setStatus("saved", true);
    } catch (err) {
      setStatus(`HN save failed · ${err.message || err}`);
    }
  });

  // Twitter cookies path retired — users now top up credits and go through
  // the paid TwitterAPI.io route. The save listeners would crash on a null
  // element so they're gone entirely; if you re-introduce the inputs, wire
  // a new handler here.

  $("pfZotero").addEventListener("blur", async () => {
    const key = $("pfZotero").value.trim();
    if (!key) return;
    setStatus("saving Zotero key…");
    try {
      const r = await fetch(`${API_BASE}/auth/me/zotero`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey: key }),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      me = await r.json().catch(() => me);
      $("pfZotero").value = "";
      renderZoteroConnection();
      setStatus("saved", true);
    } catch (err) {
      setStatus(`Zotero key not accepted · ${err.message || err}`);
    }
  });

  // ── Twitter / Zotero connection rendering ─────────────────────────────
  function renderTwitterConnection() {
    const badge = $("twConnBadge");
    if (!badge) return;
    badge.className = "conn-badge";
    badge.textContent = "";
    if (me && me.hasTwitterCookies) {
      badge.textContent = "Connected";
    }
  }
  function renderZoteroConnection() {
    const badge = $("zotConnBadge");
    const summary = $("zotSummary");
    const actions = $("zotActions");
    const connectedAs = $("zotConnectedAs");
    if (!me || !me.hasZotero) {
      badge.textContent = "";
      summary.innerHTML = "";
      actions.style.display = "none";
      return;
    }
    badge.className = "conn-badge";
    badge.textContent = "Connected";
    actions.style.display = "flex";
    connectedAs.textContent = me.zoteroUserId
      ? `Connected · user ${me.zoteroUserId}`
      : "Connected";
    const personal = me.zoteroPersonalCount || 0;
    const groups = Array.isArray(me.zoteroGroups) ? me.zoteroGroups : [];
    const groupTotal = groups.reduce((acc, g) => acc + (g.count || 0), 0);
    const total = personal + groupTotal;
    let html = `<div><strong>Personal library</strong> · ${personal.toLocaleString()} items</div>`;
    for (const g of groups) {
      html += `<div>${escapeHtml(g.name || `Group ${g.id || ""}`)} · ${(g.count || 0).toLocaleString()}</div>`;
    }
    if (groups.length > 0) {
      html += `<div style="margin-top:6px"><strong>Total</strong> · ${total.toLocaleString()}</div>`;
    }
    summary.innerHTML = html;
  }
  $("pfZotDisconnect").addEventListener("click", async () => {
    if (
      !confirm(
        "Disconnect Zotero? Wipes the saved API key — already-indexed library data stays.",
      )
    )
      return;
    try {
      const r = await fetch(`${API_BASE}/auth/me/zotero`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      me = await r.json().catch(() => me);
      renderZoteroConnection();
      setStatus("Zotero disconnected", true);
    } catch (err) {
      setStatus(`Zotero disconnect failed · ${err.message || err}`);
    }
  });

  // ── Sync now ─────────────────────────────────────────────────────────
  // Browser-side ingestion. Reads the signed-in user's `sources` JSON
  // (cached on `/api/users/{slug}`), the list of URLs already in the
  // library (so fetchers can early-exit), and runs every enabled
  // fetcher from `web/source/registry.js`. Each batch is POSTed to
  // `/auth/me/documents/bulk` and lands in PG immediately — the
  // personal-page browse path reads from PG directly (no ColBERT
  // dependency for non-VIPs) so new docs are visible on the next
  // refresh of `/<slug>`.
  (function wireSync() {
    const btn = $("syncRun");
    const log = $("syncProgress");
    const hint = $("syncHint");
    if (!btn || !log) return;

    const setHint = (txt) => {
      if (hint) hint.textContent = txt || "";
    };

    function appendStep(item) {
      const li = document.createElement("li");
      li.className = `sync-item sync-item--${item.state}`;
      li.dataset.key = item.key;
      li.innerHTML = `
        <span class="sync-item-label">${escapeHtml(item.label)}</span>
        <span class="sync-item-state">${escapeHtml(item.detail || "")}</span>
      `;
      log.appendChild(li);
      return li;
    }

    function updateStep(key, state, detail) {
      const li = log.querySelector(`[data-key="${CSS.escape(key)}"]`);
      if (!li) return;
      li.className = `sync-item sync-item--${state}`;
      li.querySelector(".sync-item-state").textContent = detail || "";
    }

    btn.addEventListener("click", async () => {
      btn.disabled = true;
      log.hidden = false;
      log.innerHTML = "";
      setHint("Loading library state…");
      try {
        // 1. Sources config from the user row.
        const meRow = await fetch(
          `${API_BASE}/api/users/${encodeURIComponent(me.slug)}`,
        ).then((r) => (r.ok ? r.json() : null));
        const sources = (meRow && meRow.sources) || {};
        // 2. URLs already in library — passed as Set so fetchers can
        // skip URLs that the bulk endpoint would just upsert-as-no-op.
        const urlsResp = await fetch(`${API_BASE}/auth/me/documents/urls`, {
          credentials: "include",
        });
        const urlsArr = urlsResp.ok ? await urlsResp.json() : [];
        const existingUrls = new Set(Array.isArray(urlsArr) ? urlsArr : []);
        setHint(
          `Library: ${existingUrls.size.toLocaleString()} URLs already saved`,
        );

        if (!window.KnowledgeSync || !window.KnowledgeSync.runSync) {
          throw new Error(
            "Sync module not loaded — refresh the page and try again.",
          );
        }

        let totalFetched = 0;
        let totalInserted = 0;
        await window.KnowledgeSync.runSync({
          sources,
          existingUrls,
          apiBase: API_BASE,
          onProgress(evt) {
            switch (evt.type) {
              case "start":
                for (const s of evt.steps) {
                  appendStep({
                    key: s.key,
                    label: s.label,
                    state: "pending",
                    detail: "queued",
                  });
                }
                break;
              case "step.start":
                updateStep(evt.key, "running", "fetching…");
                break;
              case "step.done": {
                const fetched = evt.fetched || 0;
                if (evt.error) {
                  updateStep(
                    evt.key,
                    "error",
                    `failed: ${String(evt.error).slice(0, 80)}`,
                  );
                } else {
                  updateStep(
                    evt.key,
                    "ok",
                    fetched > 0
                      ? `${fetched.toLocaleString()} fetched`
                      : "no new docs",
                  );
                }
                totalFetched += fetched;
                break;
              }
              case "upload":
                totalInserted += evt.inserted || 0;
                setHint(
                  `Uploaded ${totalInserted.toLocaleString()} new docs — ${totalFetched.toLocaleString()} fetched`,
                );
                break;
              case "done":
                setHint(
                  `Done — ${(evt.totalInserted || 0).toLocaleString()} new docs saved. Open your personal page to see them.`,
                );
                // Drop the cached personal-page payload so the next
                // visit to /<my-slug> reflects the freshly-synced
                // rows. Same shape the upvote / compose / autosync
                // paths use — keep the invalidation surface minimal
                // and consistent across writers.
                if ((evt.totalInserted || 0) > 0 && me?.slug) {
                  window.KnowledgeAPI?.invalidatePersonalDocs?.(me.slug);
                  window.KnowledgeAPI?.invalidateUnindexed?.(me.slug);
                  window.KnowledgeSessionCache?.invalidatePrefix?.("timeline:");
                }
                break;
            }
          },
        });
      } catch (err) {
        setHint(`Error: ${err && err.message ? err.message : String(err)}`);
      } finally {
        btn.disabled = false;
      }
    });
  })();

  // ── API tokens ───────────────────────────────────────────────────────
  // Server-minted bearer tokens for the user. Plaintext is shown ONCE
  // by `create_token`; we surface it in #tokenFresh and rely on the
  // user to copy it before it's gone. Subsequent reads only return the
  // 11-char prefix (`kn_xxxxxxxx`) for visual identification.
  async function fetchTokens() {
    try {
      const r = await fetch(`${API_BASE}/auth/me/tokens`, {
        credentials: "include",
      });
      if (!r.ok) return [];
      return await r.json();
    } catch {
      return [];
    }
  }

  function fmtTokenDate(s) {
    if (!s) return "never";
    try {
      const d = new Date(s);
      return d.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return s;
    }
  }

  function renderTokens(rows) {
    const list = $("tokenList");
    if (!list) return;
    if (!rows.length) {
      list.innerHTML = '<li class="token-empty">No active tokens yet.</li>';
      return;
    }
    list.innerHTML = rows
      .map(
        (t) => `
        <li class="token-row" data-id="${t.id}">
          <div class="token-row-main">
            <span class="token-row-name">${escapeHtml(t.name)}</span>
            <code class="token-row-prefix">${escapeHtml(t.prefix)}…</code>
          </div>
          <div class="token-row-meta">
            <span>Created ${fmtTokenDate(t.created_at)}</span>
            <span>· Last used ${fmtTokenDate(t.last_used_at)}</span>
          </div>
          <button
            class="action-link danger token-revoke"
            type="button"
            data-id="${t.id}"
          >Revoke</button>
        </li>`,
      )
      .join("");
    list.querySelectorAll(".token-revoke").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        if (!confirm("Revoke this token? This can't be undone.")) return;
        btn.disabled = true;
        try {
          await fetch(`${API_BASE}/auth/me/tokens/${id}`, {
            method: "DELETE",
            credentials: "include",
          });
          await refreshTokens();
        } catch {
          btn.disabled = false;
        }
      });
    });
  }

  async function refreshTokens() {
    renderTokens(await fetchTokens());
  }

  $("tokenForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("tokenName").value.trim();
    if (!name) return;
    const btn = $("tokenCreate");
    btn.disabled = true;
    try {
      const r = await fetch(`${API_BASE}/auth/me/tokens`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        alert(body.error || "Failed to create token");
        return;
      }
      const data = await r.json();
      $("tokenFreshValue").textContent = data.token;
      $("tokenFresh").style.display = "";
      $("tokenName").value = "";
      await refreshTokens();
    } finally {
      btn.disabled = false;
    }
  });

  $("tokenCopy").addEventListener("click", async () => {
    const txt = $("tokenFreshValue").textContent;
    try {
      await navigator.clipboard.writeText(txt);
      const btn = $("tokenCopy");
      const orig = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => {
        btn.textContent = orig;
      }, 1500);
    } catch {}
  });

  refreshTokens();

  // ── MCP server ───────────────────────────────────────────────────────
  // Static reference for the /mcp endpoint: install recipes per client +
  // a per-tool accordion built from the same metadata the Rust server
  // advertises in `tools/list`. Everything renders client-side from the
  // arrays below so adding a tool is a single-file change.
  (function renderMcp() {
    const mcpUrl = location.origin + "/mcp";
    const ep = $("mcpEndpointUrl");
    if (!ep) return;
    ep.textContent = mcpUrl;

    // The bearer header is what makes `feed`, `my_library`, `my_timeline`,
    // and `save_document` operate on the caller's own library — without it,
    // `feed` falls back to the public cross-library aggregate. Mint a token
    // in the API tokens section above (`kn_...`) and paste it in place of
    // the placeholder.
    const configStr = JSON.stringify(
      {
        mcpServers: {
          knowledge: {
            command: "npx",
            args: [
              "-y",
              "mcp-remote",
              mcpUrl,
              "--header",
              "Authorization:Bearer kn_xxxxxxxx",
            ],
          },
        },
      },
      null,
      2,
    );

    const installs = [
      {
        key: "claude-desktop",
        label: "Claude Desktop",
        sub: "macOS / Windows / Linux — edit claude_desktop_config.json",
        hint: `Open <strong>Settings → Developer → Edit Config</strong> in Claude
          Desktop, then merge the snippet below — replace
          <code>kn_xxxxxxxx</code> with a token you minted above. Without the
          token the bearer-authed tools (<code>save_document</code>,
          <code>my_library</code>, <code>my_timeline</code>) won't work, and
          <code>feed</code> falls back to the public cross-library aggregate
          rather than your personal timeline.
          <strong>macOS</strong>:
          <code>~/Library/Application Support/Claude/claude_desktop_config.json</code>.
          <strong>Windows</strong>:
          <code>%APPDATA%\\Claude\\claude_desktop_config.json</code>.
          <strong>Linux</strong>:
          <code>~/.config/Claude/claude_desktop_config.json</code>.
          Restart the app afterwards.`,
        code: configStr,
      },
      {
        key: "claude-code",
        label: "Claude Code (CLI)",
        sub: "One command — adds an HTTP MCP server to the CLI",
        hint: `Run the command in any project where you want the tools
          available. Add <code>--scope user</code> to make it available across
          every project. Replace <code>kn_xxxxxxxx</code> with a token you
          minted above — without it <code>feed</code> returns the public
          aggregate instead of your personal timeline, and the bearer-only
          tools won't work at all.`,
        code: `claude mcp add knowledge --transport http ${mcpUrl} \\
  --header "Authorization: Bearer kn_xxxxxxxx"`,
      },
      {
        key: "cursor",
        label: "Cursor",
        sub: "~/.cursor/mcp.json (or per-workspace .cursor/mcp.json)",
        hint: `Cursor uses the same shape as Claude Desktop. Open
          <code>~/.cursor/mcp.json</code> (create it if missing), paste the
          snippet below, replace <code>kn_xxxxxxxx</code> with a token from
          the API tokens section above, then restart Cursor.`,
        code: configStr,
      },
      {
        key: "mcp-remote",
        label: "Raw mcp-remote",
        sub: "Any MCP-compatible client that speaks stdio",
        hint: `<code>mcp-remote</code> bridges stdio clients to the plain HTTP
          endpoint. For bearer-authed tools (<code>save_document</code>,
          <code>my_library</code>, <code>my_timeline</code>) append a header
          flag with your <code>kn_…</code> token.`,
        code: `npx -y mcp-remote ${mcpUrl} \\
  --header "Authorization:Bearer kn_xxxxxxxx"`,
      },
    ];

    const tools = [
      {
        fn: "list_personalities",
        title: "List Personalities",
        desc: "Public libraries ordered by document count.",
        auth: "public",
        args: [
          {
            name: "page",
            type: "int",
            req: false,
            doc: "1-indexed (default 1)",
          },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 50, max 200",
          },
        ],
        returns:
          "Array of `{slug, name, description, categories, avatar, document_count, twitter_followers, github_followers, citations}` ordered by doc count desc. The `slug` is the identifier used by every other tool.",
        example: { name: "list_personalities", arguments: { per_page: 20 } },
      },
      {
        fn: "search_personalities",
        title: "Search Personalities",
        desc: "Filter personalities by name, category, or topic.",
        auth: "public",
        args: [
          {
            name: "query",
            type: "string",
            req: true,
            doc: "Free-text — name, topic, or category",
          },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 50, max 200",
          },
        ],
        returns:
          'VIPs ranked by ColBERT hit density inside their library (size-normalized: `count / sqrt(total_docs)`), then non-VIPs appended by lexical match. Each row carries `tier: "vip" | "name-match"`. Substring fallback when the model is unavailable or query <3 chars.',
        example: {
          name: "search_personalities",
          arguments: { query: "transformer", per_page: 10 },
        },
      },
      {
        fn: "get_personality",
        title: "Get Personality",
        desc: "Full profile, counts, and configured sources.",
        auth: "public",
        args: [
          {
            name: "personality",
            type: "string",
            req: true,
            doc: "Slug from list_personalities",
          },
        ],
        returns:
          "`{slug, name, description, categories, avatar, indexName, links, sources, document_count, twitter_followers, github_followers, citations, vip}`.",
        example: {
          name: "get_personality",
          arguments: { personality: "karpathy" },
        },
      },
      {
        fn: "search",
        title: "Search",
        desc: "ColBERT semantic search inside one library.",
        auth: "public",
        args: [
          { name: "personality", type: "string", req: true },
          {
            name: "query",
            type: "string",
            req: true,
            doc: "Natural-language query",
          },
          {
            name: "sources",
            type: "string[]",
            req: false,
            doc: "OR filter — github, arxiv, …",
          },
          {
            name: "tags",
            type: "string[]",
            req: false,
            doc: "AND across tags + extra_tags",
          },
          {
            name: "sort_by_date",
            type: "bool",
            req: false,
            doc: "Sort by date desc instead of relevance",
          },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 20, max 100",
          },
        ],
        returns:
          "`{personality, query, search_type, sort, docs: [{url, title, summary, date, tags, source, source_url, indexed, score}], pagination}`. Falls back to SQL ILIKE keyword search when the model or index is unavailable.",
        example: {
          name: "search",
          arguments: {
            personality: "karpathy",
            query: "attention mechanism intuition",
            sources: ["arxiv", "youtube"],
            per_page: 10,
          },
        },
      },
      {
        fn: "search_across",
        title: "Search Across",
        desc: "Multi-library fan-out fused with RRF.",
        auth: "public",
        args: [
          {
            name: "personalities",
            type: "string[]",
            req: true,
            doc: "2–10 slugs",
          },
          { name: "query", type: "string", req: true },
          { name: "sources", type: "string[]", req: false },
          { name: "tags", type: "string[]", req: false },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 20, max 100",
          },
        ],
        returns:
          "Each merged hit carries `rrf_score` and a `libraries` array listing every slug whose ranking surfaced it. RRF k=60 — order-independent.",
        example: {
          name: "search_across",
          arguments: {
            personalities: ["karpathy", "ylecun", "geoffreyhinton"],
            query: "scaling laws",
            per_page: 15,
          },
        },
      },
      {
        fn: "latest",
        title: "Latest",
        desc: "Most-recently saved docs from a library.",
        auth: "public",
        args: [
          { name: "personality", type: "string", req: true },
          { name: "sources", type: "string[]", req: false },
          { name: "tags", type: "string[]", req: false },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 30, max 200",
          },
        ],
        returns:
          "Docs sorted by date desc, same shape as `search` minus the relevance score. Use for 'what has X saved lately' rather than a topical query.",
        example: {
          name: "latest",
          arguments: { personality: "karpathy", per_page: 20 },
        },
      },
      {
        fn: "find_similar",
        title: "Find Similar",
        desc: "ColBERT neighbours of a given document URL.",
        auth: "public",
        args: [
          { name: "personality", type: "string", req: true },
          {
            name: "url",
            type: "string",
            req: true,
            doc: "Must exist in this library",
          },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 10, max 50",
          },
        ],
        returns:
          "Builds the query from `title + tags + first 20 words of summary`, with the source URL excluded from results. Falls back to a tag-overlap heuristic when the model is unavailable.",
        example: {
          name: "find_similar",
          arguments: {
            personality: "karpathy",
            url: "https://arxiv.org/abs/1706.03762",
          },
        },
      },
      {
        fn: "list_sources",
        title: "List Sources",
        desc: "Source-type buckets with document counts.",
        auth: "public",
        args: [
          { name: "personality", type: "string", req: true },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 50, max 200",
          },
        ],
        returns:
          "`{personality, total_docs, sources: [{key, label, count}]}` ordered by count desc. Use to discover valid keys before filtering search / latest.",
        example: {
          name: "list_sources",
          arguments: { personality: "karpathy" },
        },
      },
      {
        fn: "list_tags",
        title: "List Tags",
        desc: "Tag frequencies across a library.",
        auth: "public",
        args: [
          { name: "personality", type: "string", req: true },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 100, max 1000",
          },
        ],
        returns:
          "Combined `tags + extra_tags` ranked by document count. `{personality, unique_tags, tags: [{tag, count}]}`.",
        example: {
          name: "list_tags",
          arguments: { personality: "karpathy", per_page: 50 },
        },
      },
      {
        fn: "get_document",
        title: "Get Document",
        desc: "Full metadata for a single URL.",
        auth: "public",
        args: [
          { name: "personality", type: "string", req: true },
          { name: "url", type: "string", req: true, doc: "Exact URL match" },
        ],
        returns:
          "Full row — title, untruncated summary, date, tags, extra-tags, source, source_url, indexed.",
        example: {
          name: "get_document",
          arguments: {
            personality: "karpathy",
            url: "https://arxiv.org/abs/1706.03762",
          },
        },
      },
      {
        fn: "feed",
        title: "Feed",
        desc: "Cross-library activity feed, newest first.",
        auth: "public",
        args: [
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 50, max 500",
          },
        ],
        returns:
          "Per-URL rows ranked by sharer count + 14-day recency. Each row carries `sharers: [{slug, name, avatar, twitterFollowers}]` and `sharerCount`.",
        example: { name: "feed", arguments: { per_page: 30 } },
      },
      {
        fn: "intersect_documents",
        title: "Intersect Documents",
        desc: "URLs shared across multiple libraries.",
        auth: "public",
        args: [
          {
            name: "personalities",
            type: "string[]",
            req: true,
            doc: "2–10 slugs",
          },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 50, max 500",
          },
        ],
        returns:
          "URLs in *every* listed library. Each row carries `owners: [slug]` and `owner_count`. Ordered by owner_count desc then date desc.",
        example: {
          name: "intersect_documents",
          arguments: { personalities: ["karpathy", "ylecun"] },
        },
      },
      {
        fn: "save_document",
        title: "Save Document",
        desc: "Upload a doc into your library — bearer-token authenticated.",
        auth: "bearer",
        args: [
          {
            name: "url",
            type: "string",
            req: true,
            doc: "Canonical URL — natural key inside your library",
          },
          { name: "title", type: "string", req: false },
          { name: "summary", type: "string", req: false },
          { name: "date", type: "string", req: false, doc: "ISO YYYY-MM-DD" },
          { name: "tags", type: "string[]", req: false },
          {
            name: "extra_tags",
            type: "string[]",
            req: false,
            doc: "Free-form secondary tags",
          },
          {
            name: "source",
            type: "string",
            req: false,
            doc: "e.g. `manual`, `github`, hostname",
          },
          { name: "source_url", type: "string", req: false },
        ],
        returns:
          '`{status: "ok", url, user_id}`. Re-saving the same URL upserts on the (user_id, url) unique key. The owning user is taken from the bearer — no way to redirect the write.',
        example: {
          name: "save_document",
          arguments: {
            url: "https://example.com/post",
            title: "Worth keeping",
            summary: "Why this matters in one or two lines.",
            tags: ["topic", "subtopic"],
            source: "manual",
          },
        },
      },
      {
        fn: "my_library",
        title: "My Library",
        desc: "Search or list your own docs — bearer-token authenticated.",
        auth: "bearer",
        args: [
          {
            name: "query",
            type: "string",
            req: false,
            doc: "Omit → most-recent. Supply → ColBERT search.",
          },
          { name: "sources", type: "string[]", req: false },
          { name: "tags", type: "string[]", req: false },
          {
            name: "sort_by_date",
            type: "bool",
            req: false,
            doc: "Only honoured when `query` is set",
          },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 20 with query, 30 without. Max 200.",
          },
        ],
        returns:
          '`{personality, scope: "self", query?, search_type, docs, pagination}`. Same doc shape as `search`. Owning user comes from the bearer — no slug argument.',
        example: {
          name: "my_library",
          arguments: {
            query: "rust borrow checker",
            sources: ["github", "blog"],
            per_page: 15,
          },
        },
      },
      {
        fn: "my_timeline",
        title: "My Timeline",
        desc: "Recent docs from your follow graph — bearer-token authenticated.",
        auth: "bearer",
        args: [
          {
            name: "before",
            type: "string",
            req: false,
            doc: "ISO-8601 cursor — older-than",
          },
          {
            name: "sources",
            type: "string[]",
            req: false,
            doc: "Include filter",
          },
          { name: "exclude_sources", type: "string[]", req: false },
          { name: "tags", type: "string[]", req: false },
          { name: "page", type: "int", req: false },
          {
            name: "per_page",
            type: "int",
            req: false,
            doc: "Default 50, max 200",
          },
        ],
        returns:
          "Per-URL rows from followees ∪ self, newest first. Same `{sharers, sharerCount}` shape as `feed`. Mirrors `GET /api/timeline`.",
        example: {
          name: "my_timeline",
          arguments: { sources: ["github", "arxiv"], per_page: 30 },
        },
      },
    ];

    const countEl = $("mcpToolCount");
    if (countEl) countEl.textContent = tools.length;

    // ── Endpoint copy ──────────────────────────────────────────────────
    function copyToClipboard(text, btn, doneLabel) {
      try {
        navigator.clipboard.writeText(text);
      } catch {}
      if (btn) {
        const orig = btn.textContent;
        btn.textContent = doneLabel || "Copied";
        setTimeout(() => {
          btn.textContent = orig;
        }, 1500);
      }
    }
    const epBtn = $("mcpCopyEndpoint");
    if (epBtn) {
      epBtn.addEventListener("click", () => copyToClipboard(mcpUrl, epBtn));
    }

    // ── Install tabs ───────────────────────────────────────────────────
    const tabsEl = $("mcpInstallTabs");
    const panelEl = $("mcpInstallPanel");
    let activeInstall = installs[0].key;

    function renderInstall() {
      tabsEl.innerHTML = installs
        .map(
          (i) => `
            <button
              type="button"
              class="mcp-install-tab${i.key === activeInstall ? " mcp-install-tab--active" : ""}"
              data-key="${i.key}"
            >${escapeHtml(i.label)}</button>`,
        )
        .join("");
      tabsEl.querySelectorAll(".mcp-install-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
          activeInstall = btn.dataset.key;
          renderInstall();
        });
      });
      const cur = installs.find((i) => i.key === activeInstall);
      panelEl.innerHTML = `
        <div class="mcp-install-sub">${escapeHtml(cur.sub)}</div>
        <div class="mcp-install-hint">${cur.hint}</div>
        <div class="mcp-code-wrap">
          <pre class="mcp-code">${escapeHtml(cur.code)}</pre>
          <button class="mcp-copy-btn" type="button" data-copy="install">Copy</button>
        </div>`;
      const copyBtn = panelEl.querySelector('[data-copy="install"]');
      if (copyBtn) {
        copyBtn.addEventListener("click", () =>
          copyToClipboard(cur.code, copyBtn),
        );
      }
    }
    renderInstall();

    // ── Tools quick grid ───────────────────────────────────────────────
    const gridEl = $("mcpToolsGrid");
    gridEl.innerHTML = tools
      .map(
        (t) => `
          <div class="mcp-tool-card">
            <code class="mcp-tool-fn">${escapeHtml(t.fn)}</code>
            <div class="mcp-tool-title">${escapeHtml(t.title)}</div>
            <div class="mcp-tool-desc">${escapeHtml(t.desc)}</div>
          </div>`,
      )
      .join("");

    // ── Function reference accordion ──────────────────────────────────
    const refEl = $("mcpRefList");
    const openRefs = new Set();

    function renderRef() {
      refEl.innerHTML = tools
        .map((t) => {
          const open = openRefs.has(t.fn);
          const argsHtml = t.args.length
            ? `<table class="mcp-ref-args">
                 <thead>
                   <tr>
                     <th>name</th><th>type</th><th>required</th><th>notes</th>
                   </tr>
                 </thead>
                 <tbody>
                   ${t.args
                     .map(
                       (a) => `
                       <tr>
                         <td><code>${escapeHtml(a.name)}</code></td>
                         <td><code>${escapeHtml(a.type)}</code></td>
                         <td>${a.req ? "yes" : "no"}</td>
                         <td>${escapeHtml(a.doc || "")}</td>
                       </tr>`,
                     )
                     .join("")}
                 </tbody>
               </table>`
            : `<div class="mcp-ref-empty">No arguments.</div>`;

          const envelope = JSON.stringify(
            {
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: t.example,
            },
            null,
            2,
          );
          const cmd = `curl -X POST ${mcpUrl} \\
  -H "Content-Type: application/json" \\${
    t.auth === "bearer"
      ? `
  -H "Authorization: Bearer kn_xxxxxxxx" \\`
      : ""
  }
  -d '${envelope.replace(/'/g, "'\\''")}'`;

          return `
            <div class="mcp-ref-card${open ? " mcp-ref-card--open" : ""}" data-fn="${escapeHtml(t.fn)}">
              <button type="button" class="mcp-ref-head" data-toggle="${escapeHtml(t.fn)}">
                <div class="mcp-ref-head-left">
                  <code class="mcp-ref-fn">${escapeHtml(t.fn)}</code>
                  <span class="mcp-ref-title">${escapeHtml(t.title)}</span>
                </div>
                <div class="mcp-ref-head-right">
                  <span class="mcp-auth-pill mcp-auth-pill--${t.auth}">${t.auth}</span>
                  <svg class="mcp-ref-chevron${open ? " mcp-ref-chevron--open" : ""}"
                    width="12" height="12" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2.5"
                    stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </div>
              </button>
              ${
                open
                  ? `<div class="mcp-ref-body">
                       <div class="mcp-ref-desc">${escapeHtml(t.desc)}</div>
                       <div class="mcp-ref-section-label">Arguments</div>
                       ${argsHtml}
                       <div class="mcp-ref-section-label">Returns</div>
                       <div class="mcp-ref-returns">${t.returns}</div>
                       <div class="mcp-ref-section-label">Example call (curl)</div>
                       <div class="mcp-code-wrap">
                         <pre class="mcp-code">${escapeHtml(cmd)}</pre>
                         <button class="mcp-copy-btn" type="button"
                           data-copy-curl="${escapeHtml(t.fn)}">Copy</button>
                       </div>
                     </div>`
                  : ""
              }
            </div>`;
        })
        .join("");

      refEl.querySelectorAll("[data-toggle]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const fn = btn.dataset.toggle;
          if (openRefs.has(fn)) openRefs.delete(fn);
          else openRefs.add(fn);
          renderRef();
        });
      });
      refEl.querySelectorAll("[data-copy-curl]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const fn = btn.dataset.copyCurl;
          const t = tools.find((x) => x.fn === fn);
          if (!t) return;
          const envelope = JSON.stringify(
            {
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: t.example,
            },
            null,
            2,
          );
          const cmd = `curl -X POST ${mcpUrl} \\
  -H "Content-Type: application/json" \\${
    t.auth === "bearer"
      ? `
  -H "Authorization: Bearer kn_xxxxxxxx" \\`
      : ""
  }
  -d '${envelope.replace(/'/g, "'\\''")}'`;
          copyToClipboard(cmd, btn);
        });
      });
    }
    renderRef();
  })();

  // ── Sign out ─────────────────────────────────────────────────────────
  $("pfSignout").addEventListener("click", async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch {}
    location.href = "/";
  });

  // ── Back link — return to whichever page sent the user here ─────────
  // history.back() honours the real browser history, so a user who came
  // from a search page goes back to that exact search page (with their
  // query and filters intact). When the page was opened directly (typed
  // URL, bookmark, OAuth landing), document.referrer is empty or external
  // and we fall back to the welcome page.
  $("backLink").addEventListener("click", (e) => {
    e.preventDefault();
    const ref = document.referrer;
    const cameFromSameOrigin =
      ref && ref.startsWith(location.origin) && ref !== location.href;
    if (cameFromSameOrigin) history.back();
    else location.href = "/";
  });

  // ── Boot ─────────────────────────────────────────────────────────────
  fillForm();
})();
