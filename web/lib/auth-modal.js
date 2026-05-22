/* Knowledge auth modal — exposes `window.KnowledgeAuth` with:
 *
 *   open(view = "login")  — show the dialog on the given view
 *   close()               — close the dialog
 *   onLogin(cb)           — register a callback fired after a successful
 *                           signup / login (passes the `me` payload)
 *
 * The dialog markup lives in the page (web/search.html →
 * `<dialog id="authModal">`). This script is loaded before page.js so
 * call sites can do `window.KnowledgeAuth.open()` without timing
 * dances. Login + signup POST to the unified API, set the session
 * cookie, then reload the page so the rest of the app re-hydrates
 * with the fresh `me`. Forgot-password keeps the modal open and
 * shows an info banner.
 */
(function () {
  const ABS = window.KNOWLEDGE_API_BASE;

  const TITLES = {
    login: ["Welcome back", "Sign in to your Knowledge account."],
    signup: [
      "Create an account",
      "Build your own library of bookmarks, papers, and threads.",
    ],
    forgot: [
      "Reset your password",
      "We'll email you a link to choose a new password.",
    ],
  };

  const loginCallbacks = [];

  function $(id) {
    return document.getElementById(id);
  }
  function $$(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  function setView(view) {
    const root = $("authModal");
    if (!root) return;
    root.setAttribute("data-view", view);
    const [title, sub] = TITLES[view] || TITLES.login;
    $("authTitle").textContent = title;
    $("authSub").textContent = sub;
    $("authError").hidden = true;
    $("authError").textContent = "";
    const ok = $("authSuccess");
    if (ok) {
      ok.hidden = true;
      ok.textContent = "";
    }
    // Move focus to the first visible field for keyboard users.
    const active = root.querySelector(`.auth-view[data-view="${view}"] input`);
    if (active) setTimeout(() => active.focus(), 0);
  }

  function showError(msg) {
    const e = $("authError");
    if (!e) return;
    e.textContent = msg || "Something went wrong.";
    e.hidden = false;
  }

  function showSuccess(msg) {
    const s = $("authSuccess");
    if (!s) return;
    s.textContent = msg;
    s.hidden = false;
  }

  async function postJson(path, body) {
    const r = await fetch(`${ABS}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(body),
    });
    let payload = null;
    try {
      payload = await r.json();
    } catch {
      /* may be empty */
    }
    if (!r.ok) {
      const msg = (payload && payload.error) || `HTTP ${r.status}`;
      const err = new Error(msg);
      err.status = r.status;
      throw err;
    }
    return payload;
  }

  function setBusy(btn, busy) {
    if (!btn) return;
    btn.disabled = !!busy;
    btn.dataset._label = btn.dataset._label || btn.textContent;
    btn.textContent = busy ? "Working…" : btn.dataset._label;
  }

  async function handleLogin() {
    const email = $("loginEmail").value.trim();
    const password = $("loginPassword").value;
    if (!email || !password) {
      showError("Email and password are required.");
      return;
    }
    const btn = $("loginSubmit");
    setBusy(btn, true);
    try {
      const me = await postJson("/auth/login", { email, password });
      loginCallbacks.forEach((cb) => {
        try {
          cb(me);
        } catch {}
      });
      // Full reload is simplest — the rest of the app boots from /auth/me.
      window.location.reload();
    } catch (e) {
      showError(e.message);
    } finally {
      setBusy(btn, false);
    }
  }

  async function handleSignup() {
    const email = $("signupEmail").value.trim();
    const slug = $("signupSlug").value.trim().toLowerCase();
    const name = $("signupName").value.trim();
    const password = $("signupPassword").value;
    if (!email || !slug || !password) {
      showError("Email, username, and password are required.");
      return;
    }
    if (password.length < 8) {
      showError("Password must be at least 8 characters.");
      return;
    }
    const btn = $("signupSubmit");
    setBusy(btn, true);
    try {
      const me = await postJson("/auth/signup", {
        email,
        slug,
        name,
        password,
      });
      loginCallbacks.forEach((cb) => {
        try {
          cb(me);
        } catch {}
      });
      window.location.reload();
    } catch (e) {
      showError(e.message);
    } finally {
      setBusy(btn, false);
    }
  }

  async function handleForgot() {
    const email = $("forgotEmail").value.trim();
    if (!email) {
      showError("Email is required.");
      return;
    }
    const btn = $("forgotSubmit");
    setBusy(btn, true);
    try {
      await postJson("/auth/forgot", { email });
      // Endpoint is deliberately non-enumerating — always succeeds.
      showSuccess(
        "If that email is on file, we just sent a reset link. Check your inbox.",
      );
    } catch (e) {
      // Even on transport errors we don't want to leak existence.
      showSuccess(
        "If that email is on file, we just sent a reset link. Check your inbox.",
      );
    } finally {
      setBusy(btn, false);
    }
  }

  function open(view = "login") {
    const root = $("authModal");
    if (!root) return;
    setView(view);
    if (typeof root.showModal === "function") {
      if (!root.open) root.showModal();
    } else {
      root.setAttribute("open", "");
    }
  }

  function close() {
    const root = $("authModal");
    if (!root) return;
    if (typeof root.close === "function" && root.open) root.close();
    else root.removeAttribute("open");
  }

  function onLogin(cb) {
    if (typeof cb === "function") loginCallbacks.push(cb);
  }

  /* Render a full-page status card on /auth/verify (and /auth/reset
   * once that flow lands). The email links point at the public app
   * URL, so the frontend has to handle these paths — the actual
   * API lives on a different origin in dev and behind Caddy in
   * production, never directly on the link. */
  function renderAuthStatusPage({ title, sub, kind }) {
    document.title = `${title} — Knowledge`;
    document.body.innerHTML = `
      <main class="auth-status">
        <div class="auth-status-card auth-status-${kind}">
          <h1>${title}</h1>
          <p>${sub}</p>
          <a class="auth-status-link" href="/">Back to Knowledge →</a>
        </div>
      </main>`;
  }

  /* Render the password-reset form on /auth/reset?token=…
   * This is its own takeover page (not the modal) because the user
   * arrives here from an emailed link, possibly signed out, possibly
   * on a different device than the one that requested the reset. */
  function renderResetForm(token) {
    document.title = "Reset your password — Knowledge";
    document.body.innerHTML = `
      <main class="auth-status">
        <form class="auth-status-card auth-reset-form" id="resetForm" autocomplete="off">
          <h1>Choose a new password</h1>
          <p>Pick a password at least 8 characters long. After saving, you'll be signed in automatically.</p>
          <div class="auth-error" id="resetError" hidden></div>
          <label class="auth-field">
            <span>New password</span>
            <input
              type="password"
              id="resetPassword"
              autocomplete="new-password"
              required
              minlength="8"
              maxlength="128"
            />
          </label>
          <label class="auth-field">
            <span>Confirm password</span>
            <input
              type="password"
              id="resetConfirm"
              autocomplete="new-password"
              required
              minlength="8"
              maxlength="128"
            />
          </label>
          <button type="submit" class="auth-submit" id="resetSubmit">Save and sign in</button>
        </form>
      </main>`;
    const form = document.getElementById("resetForm");
    const errBox = document.getElementById("resetError");
    const btn = document.getElementById("resetSubmit");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      errBox.hidden = true;
      const p1 = document.getElementById("resetPassword").value;
      const p2 = document.getElementById("resetConfirm").value;
      if (p1.length < 8) {
        errBox.textContent = "Password must be at least 8 characters.";
        errBox.hidden = false;
        return;
      }
      if (p1 !== p2) {
        errBox.textContent = "Passwords don't match.";
        errBox.hidden = false;
        return;
      }
      setBusy(btn, true);
      try {
        await postJson("/auth/reset", { token, password: p1 });
        renderAuthStatusPage({
          kind: "ok",
          title: "Password updated",
          sub: "You're signed in. Welcome back.",
        });
      } catch (e) {
        errBox.textContent = e.message || "Couldn't reset the password.";
        errBox.hidden = false;
      } finally {
        setBusy(btn, false);
      }
    });
  }

  function handleResetRoute() {
    if (window.location.pathname !== "/auth/reset") return false;
    const token = new URLSearchParams(window.location.search).get("token");
    if (!token) {
      renderAuthStatusPage({
        kind: "bad",
        title: "Missing token",
        sub: "This reset link is incomplete. Request a new one from the sign-in panel.",
      });
      return true;
    }
    renderResetForm(token);
    return true;
  }

  async function handleVerifyRoute() {
    if (window.location.pathname !== "/auth/verify") return false;
    const token = new URLSearchParams(window.location.search).get("token");
    if (!token) {
      renderAuthStatusPage({
        kind: "bad",
        title: "Missing token",
        sub: "This link is incomplete. Check the email and try again.",
      });
      return true;
    }
    renderAuthStatusPage({
      kind: "pending",
      title: "Verifying…",
      sub: "Hang on while we confirm your email.",
    });
    try {
      const r = await fetch(
        `${ABS}/auth/verify?token=${encodeURIComponent(token)}`,
        {
          method: "GET",
          credentials: "include",
        },
      );
      if (r.ok) {
        renderAuthStatusPage({
          kind: "ok",
          title: "Email verified",
          sub: "You're all set — your account is ready to follow, save, and post.",
        });
      } else {
        renderAuthStatusPage({
          kind: "bad",
          title: "Link expired",
          sub: "This verification link is invalid or has expired. Sign in and request a new one.",
        });
      }
    } catch {
      renderAuthStatusPage({
        kind: "bad",
        title: "Network error",
        sub: "Couldn't reach the server. Check your connection and reload this page.",
      });
    }
    return true;
  }

  function wireOnce() {
    // Intercept /auth/verify (and future /auth/reset) BEFORE wiring
    // up the dialog — those routes don't show the modal, they take
    // over the page chrome. NB: `handleVerifyRoute` is async, so it
    // returns a Promise; comparing the URL synchronously avoids the
    // "Promise is truthy → always bail" trap.
    if (window.location.pathname === "/auth/verify") {
      handleVerifyRoute();
      return;
    }
    if (handleResetRoute()) return;

    const root = $("authModal");
    if (!root || root.dataset.wired) return;
    root.dataset.wired = "1";

    // Close behavior: × button + click-on-backdrop + Esc (native).
    const closeBtn = $("authClose");
    if (closeBtn) closeBtn.addEventListener("click", close);
    root.addEventListener("click", (e) => {
      // Native <dialog> exposes the backdrop as the dialog element itself
      // when the click lands outside the .auth-modal-card child.
      if (e.target === root) close();
    });

    // View switchers (data-go = "login" / "signup" / "forgot").
    $$("button[data-go]", root).forEach((b) => {
      b.addEventListener("click", () => setView(b.dataset.go));
    });

    // Submit handlers (data-action triggers + Enter on the active form).
    $$("button[data-action]", root).forEach((b) => {
      b.addEventListener("click", (e) => {
        e.preventDefault();
        const action = b.dataset.action;
        if (action === "login") handleLogin();
        else if (action === "signup") handleSignup();
        else if (action === "forgot") handleForgot();
        else if (action === "github-oauth") {
          // Top-level redirect — the OAuth callback sets the
          // session cookie and bounces back to "/?github=1".
          window.location.href = "/auth/github/start";
        }
      });
    });
    // Enter in any input → submit the active view's primary button.
    root.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" || e.target.tagName !== "INPUT") return;
      e.preventDefault();
      const view = root.getAttribute("data-view");
      const btn = root.querySelector(
        `.auth-view[data-view="${view}"] [data-action]`,
      );
      if (btn) btn.click();
    });

    // Auto-open on /auth/reset?token=... or /auth/verify?... (verify is
    // a server-rendered page, so we only act on reset here). The
    // reset view isn't shown in the modal — instead we redirect to a
    // dedicated reset page if you build one later. For now, drop a
    // hint in the URL: ?signin=1 opens the login view automatically.
    const u = new URLSearchParams(window.location.search);
    if (u.get("signin") === "1") open("login");
    else if (u.get("signup") === "1") open("signup");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wireOnce);
  } else {
    wireOnce();
  }

  window.KnowledgeAuth = { open, close, onLogin };
})();
