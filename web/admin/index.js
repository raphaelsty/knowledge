// Knowledge admin — read-only dashboard, raphael-sourty only.
// Single file, vanilla ES modules, no build step.

// `KNOWLEDGE_API_BASE` is set by /lib/utils.js — empty on prod (Caddy
// routes /api/* to knowledge-api), `http://localhost:8080` on dev so
// the admin panel works against the running local API without needing
// serve.py to proxy anything.
const API_HOST = window.KNOWLEDGE_API_BASE || "";
const API = `${API_HOST}/api/admin`;

// ── Tiny DOM helpers ─────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== undefined && v !== null && v !== false) {
      node.setAttribute(k, v === true ? "" : String(v));
    }
  }
  for (const c of children) {
    if (c == null || c === false) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

// ── Formatting ───────────────────────────────────────────────────────

const nf = new Intl.NumberFormat("en-US");
const fmtInt = (n) => (n == null ? "—" : nf.format(n));
const fmtPct = (n, d) => (d > 0 ? `${((100 * n) / d).toFixed(1)}%` : "—");
const fmtSec = (s) => {
  if (s == null) return "—";
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s - 60 * m).toFixed(0)}s`;
};
const fmtAgo = (iso) => {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso;
  const dt = (Date.now() - t) / 1000;
  if (dt < 60) return `${dt.toFixed(0)}s ago`;
  if (dt < 3600) return `${(dt / 60).toFixed(0)}m ago`;
  if (dt < 86400) return `${(dt / 3600).toFixed(1)}h ago`;
  return `${(dt / 86400).toFixed(1)}d ago`;
};
const pill = (status) =>
  el(
    "span",
    { class: `pill pill-${(status || "").toLowerCase()}` },
    status || "—",
  );

// ── API layer ────────────────────────────────────────────────────────

async function api(path) {
  const res = await fetch(API + path, { credentials: "include" });
  if (res.status === 401) {
    throw new GateError(
      "not-signed-in",
      "Sign in first at /search.html, then come back.",
    );
  }
  if (res.status === 403) {
    throw new GateError("not-admin", "Your account doesn't have admin access.");
  }
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `HTTP ${res.status}${body ? ` — ${body.slice(0, 200)}` : ""}`,
    );
  }
  return res.json();
}
class GateError extends Error {
  constructor(kind, msg) {
    super(msg);
    this.kind = kind;
  }
}

// ── Tabs ─────────────────────────────────────────────────────────────

const TABS = ["overview", "sources", "users", "indices", "live"];
let currentTab = "overview";

function setTab(name) {
  if (!TABS.includes(name)) name = "overview";
  currentTab = name;
  $$(".adm-tab").forEach((b) =>
    b.setAttribute("aria-selected", b.dataset.tab === name),
  );
  TABS.forEach((t) => ($(`#tab-${t}`).hidden = t !== name));
  history.replaceState(null, "", `#${name}`);
  renderTab(name);
}

$$(".adm-tab").forEach((b) =>
  b.addEventListener("click", () => setTab(b.dataset.tab)),
);
$("#adm-refresh").addEventListener("click", () =>
  renderTab(currentTab, { force: true }),
);

// ── Gate ─────────────────────────────────────────────────────────────

function showGate(msg) {
  const g = $("#adm-gate");
  g.hidden = false;
  $("#adm-gate-msg").textContent = msg;
  TABS.forEach((t) => ($(`#tab-${t}`).hidden = true));
}

// ── Section: Overview ────────────────────────────────────────────────

async function renderOverview(root) {
  root.innerHTML = "";
  root.appendChild(el("div", { class: "adm-loading" }, "Loading…"));
  let ov, src, sys, idx, ix, tf;
  try {
    [ov, src, sys, idx, ix, tf] = await Promise.all([
      api("/overview"),
      api("/sources?days=7"),
      // System metrics are best-effort: when the host is a Mac dev
      // box the /proc reads degrade to null and the tile shows "—".
      api("/system").catch(() => null),
      // Indexer activity is best-effort too — the daemon may be
      // paused (systemd stop) and that's not an error per se.
      api("/indexer").catch(() => null),
      // Indices summary — gives us the daemon's queue depth without
      // a second fetch from the Indexer Activity panel. Best-effort:
      // a transient scan failure shouldn't break Overview.
      api("/indices").catch(() => null),
      // Twitter-feed heartbeat — best-effort; if the schema isn't
      // applied yet the panel just hides the tile.
      api("/twitter-feed/status").catch(() => null),
    ]);
  } catch (e) {
    return rerror(root, e);
  }
  root.innerHTML = "";

  // KPI tiles
  const kpis = el("div", { class: "adm-kpis" });
  const okPct =
    ov.runs_7d > 0
      ? `${((100 * ov.runs_7d_ok) / ov.runs_7d).toFixed(1)}%`
      : "—";
  const kpiData = [
    {
      label: "Users",
      value: fmtInt(ov.total_users),
      sub: `${fmtInt(ov.vip_users)} VIP`,
      mod: "",
    },
    {
      label: "Documents",
      value: fmtInt(ov.total_docs),
      sub: `${fmtInt(ov.vip_docs)} VIP-owned`,
      mod: "",
    },
    {
      label: "Runs (7d)",
      value: fmtInt(ov.runs_7d),
      sub: `${okPct} success`,
      mod: ov.runs_7d_failed > 0 ? "kpi-warn" : "kpi-ok",
    },
    {
      label: "Failed runs (7d)",
      value: fmtInt(ov.runs_7d_failed),
      sub: `${fmtInt(ov.source_runs_7d_failed)} source-level`,
      mod: ov.runs_7d_failed > 0 ? "kpi-bad" : "kpi-ok",
    },
    {
      label: "Running now",
      value: fmtInt(ov.running_now),
      sub: "in flight",
      mod: ov.running_now > 0 ? "kpi-warn" : "",
    },
    {
      label: "New docs (7d)",
      value: fmtInt(ov.new_docs_7d),
      sub: "added by pipeline",
      mod: "",
    },
  ];
  for (const k of kpiData) {
    kpis.appendChild(
      el(
        "div",
        { class: `kpi ${k.mod}` },
        el("div", { class: "kpi-label" }, k.label),
        el("div", { class: "kpi-value" }, k.value),
        el("div", { class: "kpi-sub" }, k.sub),
      ),
    );
  }
  root.appendChild(kpis);

  // Host metrics tile row — CPU load (1-min normalised by core count),
  // memory available, disk available on the index volume. Tinted
  // yellow at 70%+ used, red at 90%+ so a glance is enough.
  if (sys) root.appendChild(renderHostKpis(sys));

  // Twitter-feed agent (the launchd-managed client running on the
  // operator's Mac). Single tile: state + heartbeat age + pass
  // progress + the slug currently being processed. Stale heartbeat
  // colours the tile red so a sleeping Mac is visible without the
  // operator having to dig through logs.
  if (tf) root.appendChild(renderTwitterFeedTile(tf));

  // Indexer activity — what's being written right now + recent
  // outcomes + the daemon's queue depth by verdict. Read-only
  // window; the heavy lifting (queue prioritisation, throttling) is
  // owned by the systemd service, this just makes the state
  // observable.
  if (idx) {
    root.appendChild(el("div", { class: "adm-h" }, "Indexer activity"));
    root.appendChild(renderIndexerActivity(idx, ix && ix.summary));
  }

  // Top-5 source health snapshot
  root.appendChild(
    el("div", { class: "adm-h" }, "Source health · last 7 days · top 8"),
  );
  root.appendChild(
    renderSourceTable((src || []).slice(0, 8), {
      onClick: openFailures,
      compact: true,
    }),
  );
}

// ── Host metrics ────────────────────────────────────────────────────

function fmtBytes(n) {
  if (n == null || !isFinite(n)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function pctMod(used_fraction, warn = 0.7, bad = 0.9) {
  if (used_fraction == null) return "";
  if (used_fraction >= bad) return "kpi-bad";
  if (used_fraction >= warn) return "kpi-warn";
  return "kpi-ok";
}

function renderHostKpis(sys) {
  const kpis = el("div", { class: "adm-kpis" });

  // CPU. Show 1-min load against the core count: 1.0 per core = full
  // utilization. A 4-core box at load 4 is at 100%; load 8 = saturated.
  const cpu = sys.cpu || {};
  const cores = cpu.count;
  const load1 = cpu.load_1m;
  let cpuValue = "—";
  let cpuSub = "load 1m · 5m · 15m";
  let cpuMod = "";
  if (typeof load1 === "number" && cores) {
    const pct = (100 * load1) / cores;
    cpuValue = `${pct.toFixed(0)}%`;
    cpuMod = pct >= 90 ? "kpi-bad" : pct >= 70 ? "kpi-warn" : "kpi-ok";
    cpuSub = `${load1.toFixed(2)} · ${(cpu.load_5m ?? 0).toFixed(2)} · ${(cpu.load_15m ?? 0).toFixed(2)} · ${cores}c`;
  } else if (typeof load1 === "number") {
    cpuValue = load1.toFixed(2);
  }
  kpis.appendChild(
    el(
      "div",
      { class: `kpi ${cpuMod}` },
      el("div", { class: "kpi-label" }, "CPU"),
      el("div", { class: "kpi-value" }, cpuValue),
      el("div", { class: "kpi-sub" }, cpuSub),
    ),
  );

  // Memory.
  const mem = sys.memory;
  let memValue = "—";
  let memSub = "—";
  let memMod = "";
  if (mem) {
    memValue = fmtBytes(mem.available_bytes);
    memSub = `${fmtBytes(mem.used_bytes)} used / ${fmtBytes(mem.total_bytes)} total`;
    memMod = pctMod(mem.used_fraction);
  }
  kpis.appendChild(
    el(
      "div",
      { class: `kpi ${memMod}` },
      el("div", { class: "kpi-label" }, "Memory available"),
      el("div", { class: "kpi-value" }, memValue),
      el("div", { class: "kpi-sub" }, memSub),
    ),
  );

  // Disk on the index volume.
  const disk = sys.disk;
  let diskValue = "—";
  let diskSub = "—";
  let diskMod = "";
  if (disk) {
    diskValue = fmtBytes(disk.available_bytes);
    diskSub = `${fmtBytes(disk.used_bytes)} used / ${fmtBytes(disk.total_bytes)} total · ${disk.path || "/"}`;
    diskMod = pctMod(disk.used_fraction);
  }
  kpis.appendChild(
    el(
      "div",
      { class: `kpi ${diskMod}` },
      el("div", { class: "kpi-label" }, "Disk available"),
      el("div", { class: "kpi-value" }, diskValue),
      el("div", { class: "kpi-sub" }, diskSub),
    ),
  );

  return kpis;
}

// ── Twitter-feed agent (Overview section) ───────────────────────────

function _fmtAge(secs) {
  if (secs == null) return "—";
  const n = Math.max(0, Math.floor(secs));
  if (n < 60) return `${n}s ago`;
  if (n < 3600) return `${Math.floor(n / 60)}m ago`;
  if (n < 86400) return `${Math.floor(n / 3600)}h ago`;
  return `${Math.floor(n / 86400)}d ago`;
}

function renderTwitterFeedTile(tf) {
  // Pick the visual tier:
  //   bad   — heartbeat stale (>15 min), or last_error since the
  //           last heartbeat → the agent is probably dead.
  //   warn  — sleeping (rate-limit backoff) or starting.
  //   ok    — running or recently idle.
  const stale = !!tf.stale;
  const hasError = !!tf.last_error;
  let mod = "kpi-ok";
  if (stale) mod = "kpi-bad";
  else if (hasError && tf.state !== "running") mod = "kpi-warn";
  else if (tf.state === "sleeping" || tf.state === "starting") mod = "kpi-warn";

  // Headline value — what the agent is currently doing. Falls back
  // to the recorded state machine when we don't have a slug to
  // surface (e.g. between passes).
  let value = (tf.state || "unknown").toUpperCase();
  if (stale) value = "STALE";
  else if (tf.state === "running" && tf.current_slug) {
    value = `@${tf.current_handle || tf.current_slug}`;
  }

  // Subline: "<processed>/<total> · 5m ago · 12 passes".
  const parts = [];
  if (tf.pass_total) {
    parts.push(`${tf.pass_processed ?? 0}/${tf.pass_total}`);
  }
  parts.push(_fmtAge(tf.heartbeat_age_secs));
  if (tf.pass_count) {
    parts.push(`${tf.pass_count} ${tf.pass_count === 1 ? "pass" : "passes"}`);
  }

  const tile = el(
    "div",
    { class: `kpi ${mod}` },
    el("div", { class: "kpi-label" }, "Twitter feed"),
    el(
      "div",
      {
        class: "kpi-value",
        // Long handles can break the tile — clip on overflow rather
        // than re-flowing the whole row.
        style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
        title: tf.current_slug || tf.state || "",
      },
      value,
    ),
    el("div", { class: "kpi-sub" }, parts.join(" · ")),
  );

  // If the last heartbeat carried an error, surface it as a one-line
  // hint under the tile (truncated). Better than hiding it behind a
  // hover — the operator just wants to see "something is wrong".
  if (hasError) {
    const err = String(tf.last_error).slice(0, 180);
    tile.appendChild(
      el(
        "div",
        {
          class: "kpi-sub",
          style:
            "margin-top:6px;color:var(--text-error,#c0392b);font-size:11px;line-height:1.35;",
        },
        err,
      ),
    );
  }

  // Wrap in the same .adm-kpis container the host-metrics row uses
  // so spacing / typography match.
  return el("div", { class: "adm-kpis" }, tile);
}

// ── Indexer activity (Overview section) ─────────────────────────────

function renderIndexerActivity(idx, summary) {
  const wrap = el("div", { class: "adm-indexer" });
  const active = Array.isArray(idx.active) ? idx.active : [];
  const recent = Array.isArray(idx.recent) ? idx.recent : [];

  // Queue depth strip — one chip per non-zero queueable verdict. The
  // empty/healthy tiers are skipped: they're not work the daemon
  // does. Mirrors the priority order so the worst states sit on the
  // left, same as the Indices tab.
  if (summary) {
    const chips = el("div", { class: "adm-queue-chips" });
    let any = false;
    for (const v of VERDICTS) {
      if (v === "healthy" || v === "empty") continue;
      const n = summary[v] || 0;
      if (n === 0) continue;
      any = true;
      chips.appendChild(
        el("span", { class: `pill pill-${v}` }, `${v} · ${fmtInt(n)}`),
      );
    }
    wrap.appendChild(
      el(
        "div",
        { class: "adm-queue-row" },
        el("span", { class: "adm-queue-label" }, "Queue"),
        any
          ? chips
          : el(
              "span",
              { class: "adm-empty--quiet" },
              "Empty — daemon is idle.",
            ),
      ),
    );
  }

  // "Now processing" — one row per user currently holding an
  // advisory lock. Stage column reads whatever pipeline_runs set
  // last (fetch/clean/link_check/tag/index); "running for" is the
  // wall-clock since pipeline_runs.started_at.
  const active_h = el(
    "div",
    { class: "adm-indexer-head" },
    `Now processing · ${active.length}`,
  );
  wrap.appendChild(active_h);
  if (active.length === 0) {
    wrap.appendChild(
      el(
        "div",
        { class: "adm-empty adm-empty--quiet" },
        "Idle — nothing in flight.",
      ),
    );
  } else {
    const t = el(
      "table",
      { class: "adm-table adm-table--compact" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Slug"),
          el("th", {}, "Stage"),
          el("th", { class: "num" }, "Running for"),
          el("th", { class: "num" }, "New docs (so far)"),
        ),
      ),
      el(
        "tbody",
        {},
        ...active.map((a) =>
          el(
            "tr",
            {},
            el(
              "td",
              {},
              a.vip ? el("span", { class: "pill pill-vip" }, "★") : "",
              " ",
              a.username || "?",
            ),
            el("td", {}, a.stage ? pill(a.stage) : "—"),
            el("td", { class: "mono num" }, fmtSec(a.running_for_secs)),
            el("td", { class: "num" }, fmtInt(a.new_documents)),
          ),
        ),
      ),
    );
    wrap.appendChild(t);
  }

  // Recent — last 20 succeeded/failed runs. `kind` is the
  // server-computed label so the UI doesn't re-derive it:
  //   updated  — success AND new docs landed
  //   cleaned  — success with 0 new docs (heal + re-embed only)
  //   failed   — pipeline aborted; error column carries the reason
  wrap.appendChild(
    el(
      "div",
      { class: "adm-indexer-head adm-indexer-head--secondary" },
      "Recent · last 20",
    ),
  );
  if (recent.length === 0) {
    wrap.appendChild(el("div", { class: "adm-empty adm-empty--quiet" }, "—"));
  } else {
    const t = el(
      "table",
      { class: "adm-table adm-table--compact" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Slug"),
          el("th", {}, "Kind"),
          el("th", { class: "num" }, "When"),
          el("th", { class: "num" }, "Took"),
          el("th", { class: "num" }, "New docs"),
          el("th", {}, "Error"),
        ),
      ),
      el(
        "tbody",
        {},
        ...recent.map((r) =>
          el(
            "tr",
            {},
            el(
              "td",
              {},
              r.vip ? el("span", { class: "pill pill-vip" }, "★") : "",
              " ",
              r.username || "?",
            ),
            el("td", {}, pill(r.kind || r.status || "—")),
            el("td", { class: "mono num" }, fmtAgo(r.started_at)),
            el("td", { class: "num" }, fmtSec(r.duration_secs)),
            el("td", { class: "num" }, fmtInt(r.new_documents)),
            el(
              "td",
              { class: "mono", title: r.error || "" },
              truncate(r.error, 40) || "",
            ),
          ),
        ),
      ),
    );
    wrap.appendChild(t);
  }

  return wrap;
}

// ── Section: Sources ─────────────────────────────────────────────────

let sourcesState = { days: 7, openSource: null };

async function renderSources(root) {
  root.innerHTML = "";
  const toolbar = el(
    "div",
    { class: "adm-toolbar" },
    el("label", {}, "Window:"),
    el(
      "select",
      {
        onChange: (e) => {
          sourcesState.days = +e.target.value;
          renderSources(root);
        },
      },
      ...[1, 3, 7, 14, 30].map((d) =>
        el(
          "option",
          { value: d, selected: d === sourcesState.days ? true : undefined },
          `${d} day${d > 1 ? "s" : ""}`,
        ),
      ),
    ),
  );
  root.appendChild(toolbar);
  root.appendChild(
    el("div", { class: "adm-loading", id: "src-loading" }, "Loading…"),
  );
  let rows;
  try {
    rows = await api(`/sources?days=${sourcesState.days}`);
  } catch (e) {
    return rerror(root, e);
  }
  $("#src-loading").remove();
  root.appendChild(
    renderSourceTable(rows, { onClick: openFailures, compact: false }),
  );

  if (sourcesState.openSource) {
    root.appendChild(
      await renderFailuresPanel(sourcesState.openSource, sourcesState.days),
    );
  }
}

function renderSourceTable(rows, { onClick, compact }) {
  if (!rows || rows.length === 0) {
    return el(
      "div",
      { class: "adm-empty" },
      "No source activity in this window.",
    );
  }
  const tbl = el("table", { class: "adm-table" });
  const head = el(
    "thead",
    {},
    el(
      "tr",
      {},
      el("th", {}, "Source"),
      el("th", { class: "num" }, "Total"),
      el("th", { class: "num" }, "OK"),
      el("th", { class: "num" }, "Failed"),
      el("th", { class: "num" }, "Skipped"),
      el("th", { class: "num" }, "Users failing"),
      el("th", { class: "num" }, "New docs"),
      el("th", { class: "num" }, "Avg ok"),
      compact ? null : el("th", {}, "Last failure"),
      compact ? null : el("th", {}, "Last success"),
    ),
  );
  tbl.appendChild(head);
  const body = el("tbody");
  for (const r of rows) {
    const tr = el(
      "tr",
      { onClick: () => onClick && onClick(r.source), style: "cursor:pointer;" },
      el("td", {}, r.source),
      el("td", { class: "num" }, fmtInt(r.total_runs)),
      el("td", { class: "num" }, fmtInt(r.success_runs)),
      el(
        "td",
        { class: "num" },
        r.failed_runs > 0
          ? el("span", { class: "pill pill-failed" }, String(r.failed_runs))
          : "0",
      ),
      el("td", { class: "num" }, fmtInt(r.skipped_runs)),
      el("td", { class: "num" }, fmtInt(r.users_failing)),
      el("td", { class: "num" }, fmtInt(r.total_new_docs)),
      el("td", { class: "num" }, fmtSec(r.avg_duration_ok)),
      compact
        ? null
        : el(
            "td",
            { class: "mono" },
            r.last_failure_at ? fmtAgo(r.last_failure_at) : "—",
          ),
      compact
        ? null
        : el(
            "td",
            { class: "mono" },
            r.last_success_at ? fmtAgo(r.last_success_at) : "—",
          ),
    );
    body.appendChild(tr);
  }
  tbl.appendChild(body);
  return tbl;
}

function openFailures(source) {
  sourcesState.openSource = source;
  if (currentTab !== "sources") setTab("sources");
  else renderSources($("#tab-sources"));
}

async function renderFailuresPanel(source, days) {
  const panel = el("div", { class: "adm-failures-panel" });
  panel.appendChild(
    el("div", { class: "adm-h" }, `Failures · ${source} · ${days}d`),
  );
  panel.appendChild(el("div", { class: "adm-loading" }, "Loading…"));
  try {
    const data = await api(
      `/sources/${encodeURIComponent(source)}/failures?days=${days}`,
    );
    panel.lastChild.remove();
    if (!data.groups || data.groups.length === 0) {
      panel.appendChild(el("div", { class: "adm-empty" }, "No failures."));
      return panel;
    }
    for (const g of data.groups) {
      const grp = el(
        "div",
        { class: "adm-fail-group" },
        el("div", { class: "adm-fail-msg" }, g.message || "(no error message)"),
        el(
          "div",
          { class: "adm-fail-meta" },
          `${g.count} run(s), ${g.users.length} user(s)`,
        ),
        el(
          "div",
          { class: "adm-fail-users" },
          ...g.users
            .slice(0, 30)
            .map((u) =>
              el(
                "span",
                { class: "adm-fail-user", title: u.name || u.username },
                u.vip ? "★ " : "",
                u.username,
              ),
            ),
          g.users.length > 30
            ? el("span", { class: "adm-fail-user" }, `+${g.users.length - 30}`)
            : null,
        ),
      );
      panel.appendChild(grp);
    }
  } catch (e) {
    panel.lastChild.remove();
    rerror(panel, e);
  }
  return panel;
}

// ── Section: Users ───────────────────────────────────────────────────

let usersState = { q: "" };

async function renderUsers(root) {
  root.innerHTML = "";
  const toolbar = el(
    "div",
    { class: "adm-toolbar" },
    el("input", {
      type: "search",
      placeholder: "filter by slug or name…",
      value: usersState.q,
      onInput: debounce((e) => {
        usersState.q = e.target.value;
        renderUsers(root);
      }, 250),
    }),
  );
  root.appendChild(toolbar);
  root.appendChild(
    el("div", { class: "adm-loading", id: "users-loading" }, "Loading…"),
  );
  let rows;
  try {
    rows = await api(`/users?q=${encodeURIComponent(usersState.q || "")}`);
  } catch (e) {
    return rerror(root, e);
  }
  $("#users-loading").remove();
  if (rows.length === 0) {
    root.appendChild(el("div", { class: "adm-empty" }, "No matching users."));
    return;
  }
  const tbl = el(
    "table",
    { class: "adm-table" },
    el(
      "thead",
      {},
      el(
        "tr",
        {},
        el("th", {}, "Slug"),
        el("th", {}, "Name"),
        el("th", { class: "num" }, "Docs"),
        el("th", {}, "Last run"),
        el("th", {}, "When"),
        el("th", { class: "num" }, "New"),
        el("th", { class: "num" }, "Duration"),
        el("th", {}, "Error"),
      ),
    ),
  );
  const body = el("tbody");
  for (const u of rows) {
    body.appendChild(
      el(
        "tr",
        {},
        el(
          "td",
          {},
          u.vip ? el("span", { class: "pill pill-vip" }, "★") : "",
          " ",
          u.username,
        ),
        el("td", {}, u.name || "—"),
        el("td", { class: "num" }, fmtInt(u.doc_count)),
        el("td", {}, u.last_run_status ? pill(u.last_run_status) : "—"),
        el("td", { class: "mono" }, fmtAgo(u.last_run_started_at)),
        el("td", { class: "num" }, fmtInt(u.last_run_new_docs)),
        el("td", { class: "num" }, fmtSec(u.last_run_duration_secs)),
        el(
          "td",
          { class: "mono", title: u.last_run_error || "" },
          truncate(u.last_run_error, 80) || "",
        ),
      ),
    );
  }
  tbl.appendChild(body);
  root.appendChild(tbl);
}

// ── Section: Indices ─────────────────────────────────────────────────

// Plain-language definition for each verdict the classifier emits.
// Rendered as a collapsible legend at the top of the Indices tab so
// the operator doesn't have to remember what each label means.
// Order here drives the legend + KPI tile order in the Indices tab —
// keep it priority-descending so the worst states sit on the left.
const VERDICT_DEFS = {
  broken:
    'Index exists on disk but is unusable. Two sub-cases: (a) num_documents > 0 with num_embeddings = 0 — embedder crashed mid-write, search returns HTTP 500 "No data to merge"; (b) num_documents = 0 while PG has docs — index file loads but holds zero docs, search returns nothing. Both auto-repaired by the indexer daemon.',
  error:
    "GET /indices/{name} returns 5xx (corrupt on-disk files) or the metadata.json is unreadable. Search will fail. Auto-repaired.",
  missing:
    "No index directory on disk but PG has documents for the user. The pipeline either never ran an indexing pass or its output was wiped. Auto-repaired.",
  pg_drift:
    "Index loads but its document count disagrees with PG beyond the drift threshold (>5 docs or >5%). The daemon's drift-purge step compares the index against the live (deleted=false) PG set and removes ghost rows — typically self-heals on the next pass.",
  backlog:
    "Index agrees with pg_indexed but PG still carries indexed=false rows (e.g. tweets just synced in from the local feeder). The indexer daemon queues these last; small backlogs drain naturally on the next sweep.",
  healthy:
    "Index loads and its document count agrees with PG within the drift threshold; no indexed=false rows pending. Search works as expected.",
  empty:
    "No index AND no PG documents. Clean state for a fresh personality — nothing to repair.",
};

// Single source of truth: verdict → KPI/pill class. Mirrors the
// daemon's priority tiers (broken < error < missing < pg_drift <
// backlog < healthy/empty) so the colour temperature matches the
// urgency the operator should attach to each verdict.
const VERDICT_MOD = {
  broken: "kpi-bad",
  error: "kpi-bad",
  missing: "kpi-warn",
  pg_drift: "kpi-warn",
  backlog: "kpi-warn",
  healthy: "kpi-ok",
  empty: "",
};
const VERDICT_ORDER = {
  broken: 0,
  error: 1,
  missing: 2,
  pg_drift: 3,
  backlog: 4,
  empty: 5,
  healthy: 6,
};
const VERDICTS = Object.keys(VERDICT_DEFS);

function renderVerdictLegend() {
  const details = el("details", { class: "adm-legend" });
  const summary = el("summary", {}, "What does each verdict mean?");
  details.appendChild(summary);
  const grid = el("div", { class: "adm-legend-grid" });
  for (const [k, def] of Object.entries(VERDICT_DEFS)) {
    grid.appendChild(
      el(
        "div",
        { class: "adm-legend-row" },
        el("span", { class: `pill pill-${k}` }, k),
        el("div", { class: "adm-legend-text" }, def),
      ),
    );
  }
  details.appendChild(grid);
  return details;
}

async function renderIndices(root) {
  root.innerHTML = "";
  root.appendChild(el("div", { class: "adm-loading" }, "Scanning indices…"));
  let data;
  try {
    data = await api("/indices");
  } catch (e) {
    return rerror(root, e);
  }
  root.innerHTML = "";

  const kpis = el("div", { class: "adm-kpis" });
  for (const v of VERDICTS) {
    const n = data.summary[v] || 0;
    kpis.appendChild(
      el(
        "div",
        { class: `kpi ${VERDICT_MOD[v] || ""}` },
        el("div", { class: "kpi-label" }, v),
        el("div", { class: "kpi-value" }, fmtInt(n)),
      ),
    );
  }
  root.appendChild(kpis);
  root.appendChild(renderVerdictLegend());

  // Priority-first ordering — same tiers the indexer daemon uses to
  // pick the next user to process.
  const rows = (data.details || [])
    .slice()
    .sort(
      (a, b) =>
        (VERDICT_ORDER[a.verdict] ?? 9) - (VERDICT_ORDER[b.verdict] ?? 9) ||
        a.username.localeCompare(b.username),
    );

  // Default: hide healthy
  let showHealthy = false;
  const toolbar = el(
    "div",
    { class: "adm-toolbar" },
    el(
      "label",
      {},
      el("input", {
        type: "checkbox",
        onChange: (e) => {
          showHealthy = e.target.checked;
          redraw();
        },
      }),
      " show healthy",
    ),
  );
  root.appendChild(toolbar);
  const tblHolder = el("div");
  root.appendChild(tblHolder);

  function redraw() {
    tblHolder.innerHTML = "";
    const filtered = rows.filter((r) => showHealthy || r.verdict !== "healthy");
    if (filtered.length === 0) {
      tblHolder.appendChild(
        el("div", { class: "adm-empty" }, "Nothing to show."),
      );
      return;
    }
    const tbl = el(
      "table",
      { class: "adm-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "Slug"),
          el("th", {}, "Verdict"),
          el("th", { class: "num" }, "PG idx/total"),
          el("th", {}, "Reason"),
        ),
      ),
    );
    const body = el("tbody");
    for (const r of filtered) {
      body.appendChild(
        el(
          "tr",
          {},
          el(
            "td",
            {},
            r.vip ? el("span", { class: "pill pill-vip" }, "★") : "",
            " ",
            r.username,
          ),
          el("td", {}, pill(r.verdict)),
          el(
            "td",
            { class: "num" },
            `${fmtInt(r.pg_indexed)} / ${fmtInt(r.pg_total)}`,
          ),
          el("td", { class: "mono" }, r.reason),
        ),
      );
    }
    tbl.appendChild(body);
    tblHolder.appendChild(tbl);
  }
  redraw();
}

// ── Section: Live ────────────────────────────────────────────────────

let liveTimer = null;

async function renderLive(root, { autoRefresh = true } = {}) {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  root.innerHTML = "";
  root.appendChild(el("div", { class: "adm-loading" }, "Loading…"));
  await refreshLive(root);
  if (autoRefresh) {
    liveTimer = setInterval(() => {
      if (currentTab === "live") refreshLive(root);
    }, 5000);
  }
}

async function refreshLive(root) {
  let data;
  try {
    data = await api("/live");
  } catch (e) {
    return rerror(root, e);
  }
  root.innerHTML = "";

  const cols = el(
    "div",
    { class: "adm-live-cols" },
    el(
      "div",
      {},
      el("div", { class: "adm-h" }, "Pipeline runs · last 50"),
      renderLiveRuns(data.runs),
    ),
    el(
      "div",
      {},
      el("div", { class: "adm-h" }, "Source runs · last 80"),
      renderLiveSourceRuns(data.source_runs),
    ),
  );
  root.appendChild(cols);
}

function renderLiveRuns(rows) {
  if (!rows || rows.length === 0)
    return el("div", { class: "adm-empty" }, "No recent runs.");
  const tbl = el(
    "table",
    { class: "adm-table" },
    el(
      "thead",
      {},
      el(
        "tr",
        {},
        el("th", {}, "Slug"),
        el("th", {}, "Status"),
        el("th", { class: "num" }, "Started"),
        el("th", { class: "num" }, "Took"),
        el("th", { class: "num" }, "New"),
        el("th", {}, "Stage"),
      ),
    ),
    el(
      "tbody",
      {},
      ...rows.map((r) =>
        el(
          "tr",
          {},
          el("td", {}, r.vip ? "★ " : "", r.username || "?"),
          el("td", {}, pill(r.status)),
          el("td", { class: "mono num" }, fmtAgo(r.started_at)),
          el("td", { class: "num" }, fmtSec(r.duration_secs)),
          el("td", { class: "num" }, fmtInt(r.new_documents)),
          el(
            "td",
            { class: "mono", title: r.error || "" },
            r.stage || (r.error ? truncate(r.error, 40) : ""),
          ),
        ),
      ),
    ),
  );
  return tbl;
}

function renderLiveSourceRuns(rows) {
  if (!rows || rows.length === 0)
    return el("div", { class: "adm-empty" }, "No recent source runs.");
  const tbl = el(
    "table",
    { class: "adm-table" },
    el(
      "thead",
      {},
      el(
        "tr",
        {},
        el("th", {}, "Slug"),
        el("th", {}, "Source"),
        el("th", {}, "Status"),
        el("th", { class: "num" }, "When"),
        el("th", { class: "num" }, "Took"),
        el("th", { class: "num" }, "New"),
      ),
    ),
    el(
      "tbody",
      {},
      ...rows.map((r) =>
        el(
          "tr",
          {},
          el("td", {}, r.vip ? "★ " : "", r.username || "?"),
          el("td", {}, r.source || "—"),
          el("td", {}, pill(r.status)),
          el("td", { class: "mono num" }, fmtAgo(r.started_at)),
          el("td", { class: "num" }, fmtSec(r.duration_secs)),
          el("td", { class: "num" }, fmtInt(r.new_documents)),
        ),
      ),
    ),
  );
  return tbl;
}

// ── Dispatch ─────────────────────────────────────────────────────────

async function renderTab(name, { force = false } = {}) {
  const root = $(`#tab-${name}`);
  if (!root) return;
  if (name === "live") return renderLive(root);
  // Other tabs: stop any live-refresh timer.
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
  }
  try {
    if (name === "overview") return await renderOverview(root);
    if (name === "sources") return await renderSources(root);
    if (name === "users") return await renderUsers(root);
    if (name === "indices") return await renderIndices(root);
  } catch (e) {
    if (e instanceof GateError) showGate(e.message);
    else rerror(root, e);
  }
}

// ── Utils ────────────────────────────────────────────────────────────

function rerror(root, e) {
  if (e instanceof GateError) return showGate(e.message);
  root.innerHTML = "";
  root.appendChild(el("div", { class: "adm-error" }, e.message || String(e)));
}
function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
function truncate(s, n) {
  if (!s) return s;
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// ── Boot ─────────────────────────────────────────────────────────────

function tickClock() {
  const d = new Date();
  $("#adm-clock").textContent = d.toISOString().slice(11, 19) + "Z";
}
setInterval(tickClock, 1000);
tickClock();

// Pick tab from URL hash
const hashTab = (location.hash || "").replace(/^#/, "");
setTab(TABS.includes(hashTab) ? hashTab : "overview");
