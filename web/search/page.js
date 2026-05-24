/* Search-page rail engine.
 *
 * Boots the per-personality search view: state, library/source rail,
 * spotlight search, sort toggle, results render, and the profile
 * modal (form auto-save, secret saves, sync orchestrator).
 *
 * Depends on two globals already attached to `window`:
 *   - KnowledgeAPI  (web/search/api.js)        — fetch helpers, caches, doc shape
 *   - KnowledgeSync (web/source/index.js)      — runSync + REGISTRY
 */
(async function () {
  const K = window.KnowledgeAPI;
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  // Slug resolution order:
  //   1. ?slug=... query parameter
  //   2. on /search → first slug in ?libs= (host = first lib)
  //   3. first path segment (so /max-halford still works as a real route)
  //   4. fall back to the demo default
  const firstPathSeg = location.pathname
    .replace(/^\/+|\/+$/g, "")
    .split("/")[0];
  // `/` and `/search` are the same feed/search shell — no host slug
  // by default, so an empty `?libs=` lands on the timeline instead of
  // accidentally booting `K.DEFAULT_SLUG`'s library.
  const onSearchRoute = firstPathSeg === "search" || firstPathSeg === "";
  const libsParamRaw = (params.get("libs") || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const slug =
    params.get("slug") ||
    (onSearchRoute ? libsParamRaw[0] || "empty" : "") ||
    (!onSearchRoute ? firstPathSeg : "") ||
    K.DEFAULT_SLUG;
  /* Sentinel slug for the "no library selected" state. Surfaces when
   * the user lands on bare /search (no `?libs=`), so a refresh keeps
   * the empty picker view instead of re-loading the user's profile
   * (state.libs would otherwise default to [hostSlug] at boot). When
   * slug === EMPTY_SLUG we skip the host-meta fetches and start with
   * an empty state.libs. */
  const EMPTY_SLUG = "empty";
  const isEmptyHost = slug === EMPTY_SLUG;
  // Hoisted up because `rebuildAllSources()` (called early during
  // boot, line ~541) needs `useAllOnly` → which reads
  // `ALL_INDEX_THRESHOLD`. `const` is in TDZ until its declaration
  // line is executed, so leaving these alongside the rest of the
  // routing helpers further down crashes Safari with "Cannot
  // access uninitialized variable".
  const MAX_NONVIPS = 10;
  const ALL_INDEX_THRESHOLD = 5;

  /* Ontology slug → display label. Mirrors the seed in
   * sources/sql/categories.sql so the library picker and onboarding
   * can render human-friendly section titles without hitting the
   * API for the categories table. Keep in sync if the SQL seed is
   * edited. */
  const CATEGORY_LABELS = {
    "llm-research": "LLM Research",
    "nlp-retrieval": "NLP & Retrieval",
    "computer-vision": "Computer Vision",
    "generative-media": "Generative Media",
    multimodal: "Multimodal AI",
    "rl-robotics": "Reinforcement Learning & Robotics",
    "ai-safety": "AI Safety & Alignment",
    "ml-theory": "ML Theory & Foundations",
    "ml-infra": "ML Infra & Systems",
    "efficient-inference": "Efficient Inference",
    "oss-tools": "Open-Source ML Tools",
    founders: "AI Founders & Builders",
    "lab-leaders": "AI Lab Leadership",
    educators: "Educators & Bloggers",
    pioneers: "Pioneers & Laureates",
  };
  /* Stable display order: matches the `sort_order` ranks in the SQL
   * seed. Used by the library picker to group personalities. */
  const CATEGORY_ORDER = [
    "llm-research",
    "nlp-retrieval",
    "computer-vision",
    "generative-media",
    "multimodal",
    "rl-robotics",
    "ai-safety",
    "ml-theory",
    "ml-infra",
    "efficient-inference",
    "oss-tools",
    "founders",
    "lab-leaders",
    "educators",
    "pioneers",
  ];
  /* Pick a personality's primary category — first slug from their
   * `categories[]` (already sorted by sort_order from the API).
   * Falls back to "other" so unclassified rows still render. */
  function primaryCategoryLabel(p) {
    const slug = (p && p.categories && p.categories[0]) || null;
    return slug ? CATEGORY_LABELS[slug] || slug : "Other";
  }
  const ALL_INDEX_NAME = "__all__";
  if (!isEmptyHost) $("q").placeholder = `Search ${slug} knowledge`;
  else $("q").placeholder = "Search your feed";
  /* Synthetic source key — when the user has favorites, an extra
   * "Favorites" entry is injected into the source rail at this key.
   * Selecting it adds a `url IN (...)` clause to the SQL filter
   * pushed to every backend pool, so favorite-only browsing is a
   * real pre-filter, not a JS post-filter, and it composes cleanly
   * with the other source / tag chips (e.g. "Favorites" + "github"
   * = favorited GitHub stars only). The leading underscore guards
   * against collision with any real source key. */
  const FAV_SOURCE_KEY = "_favorites";
  /* Hoisted out of the auth section because rebuildAllSources()
   * (called from the initial URL-state load path) reads `me` to
   * decide whether to surface the Favorites chip. With `let` and
   * the original placement deeper in the file, that early read
   * tripped the TDZ and the rail crashed empty. */
  let me = null; // current user (null = anonymous)
  // Hoisted alongside `me` for the same reason — `loadMe()` runs in
  // the boot `Promise.all` (above the const-declaration site for
  // `API_BASE` further down the file), and a TDZ ReferenceError on
  // the template literal inside `loadMe` would silently fall into
  // the function's `catch` and resolve as `null`, leaving the auth
  // button stuck on the anonymous pill.
  const API_BASE = window.KNOWLEDGE_API_BASE;
  const state = {
    query: "",
    sources: new Set(),
    /* Sources the user has explicitly excluded from results — pushed
     * down to the index as `source NOT IN (...)`, AND-combined with
     * the include set. A source can never be in both at once (the
     * tri-state chip enforces neutral → include → exclude → neutral). */
    excludedSources: new Set(),
    tags: new Set(),
    sortByDate: false,
    /* Date-range filter — empty = any time. Values: "7d", "30d",
     * "90d", "365d" (parsed in `_sinceDateString`). Applied as a
     * `date >= ?` clause in `buildIndexFilter` and as `&since=` in
     * the timeline query string. */
    dateSince: "",
    /* Fine-grained category slugs from `document_categories` — the
     * topical filter wired to the right-rail Topics tab on desktop
     * and the bottom-nav Topics tab on phone. Empty set = no filter.
     * When non-empty, the timeline passes the CSV as
     * `&category=<a>,<b>,…` and the search path pre-fetches the URL
     * set for these slugs and intersects it with the ColBERT hits.
     * OR semantics across slugs (matches the source-chip behaviour
     * on the left rail). The catalogue is fetched lazily on first
     * open of either picker surface. */
    categories: new Set(),
    /* Desktop right-rail mode — either 'people' (the default
     * "Peoples to follow" panel) or 'categories' (the topic
     * picker UI). The two share the right column; only one is
     * rendered at a time. Persisted in localStorage under
     * `kn.right_rail` so a user who flips to Topics keeps Topics
     * as they navigate between routes (feed ↔ personal page).
     * Mobile ignores this state — the bottom-nav already exposes
     * separate entries for each. */
    rightRail: (function () {
      try {
        const v = localStorage.getItem("kn.right_rail");
        return v === "categories" ? "categories" : "people";
      } catch {
        return "people";
      }
    })(),
    /* When true (and signed in), feed search narrows to followees +
     * self. Off by default so first-time users still discover content
     * across libraries — the toggle next to the date filter lets them
     * opt back into the focused view. */
    followingOnly: false,
    /* When true, the timeline includes cards the viewer has already
     * seen (≥1.5 s viewport dwell tracked via the card_seen event).
     * Default false → seen cards are filtered out server-side.
     * Logged-in only — the button is hidden for anonymous visitors.
     * Persisted in localStorage as `feed.showSeen`. */
    showSeen: (() => {
      try {
        return localStorage.getItem("feed.showSeen") === "1";
      } catch {
        return false;
      }
    })(),
    lastDocs: [],
    /* URLs the user has been shown this session — drives the "More"
     * button's "fetch posts I haven't seen this week" filter.
     * Populated by markShownUrls() at every feed-render site. Wiped
     * by refresh() when the user changes filters, navigates, or
     * starts a search (a new search surface starts with a clean
     * slate). */
    shownUrls: new Set(),
    favorites: new Set(),
    /* Slugs the signed-in user has bookmarked (cross-user
     * "follow"). Populated after /auth/me resolves and surfaces in
     * the library picker's "Bookmarks" section. */
    personalityBookmarks: new Set(),
    allSources: [],
    allPersonalities: [],
    hostSlug: slug,
    libs: isEmptyHost ? new Set() : new Set([slug]),
    perSlugMeta: {},
    perSlugSources: {},
    perSlugTwitterFreshness: {},
    // Lowercase Twitter handle → slug map. Built once from
    // /api/users (which already returns each personality's
    // `sources.twitter.username`). Used by `renderResult` to find
    // the original author of a "Retweet @handle" / "Quoting
    // @handle" doc so we can surface their avatar inline on the
    // card instead of just the retweeter's.
    slugByTwitterHandle: {},
  };
  function parseQ(raw) {
    const tokens = (raw || "").split(/\s+/).filter(Boolean);
    const sites = new Set(),
      tags = new Set();
    const plain = [];
    for (const t of tokens) {
      if (t.startsWith("site:")) sites.add(t.slice(5).toLowerCase());
      else if (t.startsWith("tag:")) tags.add(t.slice(4).toLowerCase());
      else plain.push(t);
    }
    return { sites, tags, plain: plain.join(" ") };
  }
  function writeQ() {
    /* Tags and the Favorites toggle no longer round-trip through the
     * search input — both live as chips (tag pills below results,
     * Favorites in the source rail) and pre-filter server-side. The
     * input is reserved for free-text only. */
    $("q").value = state.query || "";
  }

  /* URL state codec — keep the address bar in sync with what's
   * filtered/searched so a user can copy-paste the URL and a
   * collaborator lands on the same view. We use replaceState (not
   * pushState) on every change so back/forward isn't polluted by
   * keystroke-by-keystroke edits; opening / changing libraries via
   * the rail is the granularity for forward/back, but right now
   * those also use replaceState — simpler, predictable.
   *
   * Param glossary:
   *   q       — query text (omitted when blank)
   *   tags    — comma-joined tag list (omitted when empty)
   *   sources — comma-joined source-key list (omitted when empty)
   *   libs    — comma-joined extra library slugs (host is implicit
   *             from the URL path, never written here)
   *   sort    — "date" when the user manually flipped to date sort
   *             on a query (omitted otherwise; relevance is default)
   *   fav     — "1" when the synthetic Favorites source is selected
   *
   * Reads from `state` directly; writes via history.replaceState so
   * the page never reloads. Slug stays in the path. */
  function writeUrl() {
    const q = new URLSearchParams();
    if (state.query) q.set("q", state.query);
    if (state.tags.size) q.set("tags", [...state.tags].join(","));
    // Real-source list only — the synthetic Favorites key rides in
    // the dedicated `fav` param so the URL stays semantic.
    const realSources = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
    if (realSources.length) q.set("sources", realSources.join(","));
    if (state.excludedSources.size)
      q.set("exclude", [...state.excludedSources].join(","));
    // On /search the path carries no slug, so all libs (including the
    // synthetic host = first lib) need to round-trip through ?libs=.
    // On /<slug> we keep the historical behavior: host is implicit in
    // the path and only extras go to ?libs=.
    const libList = onSearchRoute
      ? [...state.libs]
      : [...state.libs].filter((s) => s !== state.hostSlug);
    if (libList.length) q.set("libs", libList.join(","));
    if (state.sortByDate) q.set("sort", "date");
    if (state.dateSince) q.set("since", state.dateSince);
    if (state.followingOnly) q.set("scope", "following");
    if (state.categories && state.categories.size) {
      q.set("category", [...state.categories].join(","));
    }
    if (state.sources.has(FAV_SOURCE_KEY)) q.set("fav", "1");
    const qs = q.toString();
    const next = location.pathname + (qs ? `?${qs}` : "") + location.hash;
    if (next !== location.pathname + location.search + location.hash) {
      history.replaceState(null, "", next);
    }
    // SEO: keep <link rel="canonical"> in sync with the URL the user
    // actually sees. Static HTML ships a placeholder canonical (the
    // bare /search or /); when the SPA narrows by `libs=<slug>` the
    // canonical needs to point at the narrowed view so Google
    // indexes each personality page as a distinct URL instead of
    // collapsing them under the bare /search canonical.
    syncCanonical();
  }
  function syncCanonical() {
    const link = document.querySelector('link[rel="canonical"]');
    if (!link) return;
    // Only the same params that appear in the sitemap matter for
    // canonical purposes — `q`, `tags`, `since`, `sort`, `scope`,
    // `fav` are user-state and shouldn't fragment indexing.
    const cur = new URLSearchParams(location.search);
    const keep = new URLSearchParams();
    if (cur.get("libs")) keep.set("libs", cur.get("libs"));
    const qs = keep.toString();
    const path = location.pathname;
    const next = `https://knowledge-web.org${path}${qs ? `?${qs}` : ""}`;
    if (link.getAttribute("href") !== next) link.setAttribute("href", next);
  }
  /* Hydrate state from the URL on boot. Returns a list of the extra
   * libraries to async-load (so the caller can `Promise.all` their
   * source fetches before the first refresh fires). */
  function readUrl() {
    const u = new URLSearchParams(location.search);
    const q = u.get("q") || "";
    const tagsCsv = u.get("tags") || "";
    const sourcesCsv = u.get("sources") || "";
    const excludeCsv = u.get("exclude") || "";
    const libsCsv = u.get("libs") || "";
    const sort = u.get("sort") || "";
    const fav = u.get("fav") || "";
    const since = u.get("since") || "";
    if (q) state.query = q;
    if (tagsCsv)
      for (const t of tagsCsv.split(",").filter(Boolean))
        state.tags.add(t.toLowerCase());
    if (sourcesCsv)
      for (const s of sourcesCsv.split(",").filter(Boolean))
        state.sources.add(s);
    if (excludeCsv)
      for (const s of excludeCsv.split(",").filter(Boolean))
        if (!state.sources.has(s)) state.excludedSources.add(s);
    if (sort === "date") state.sortByDate = true;
    if (since) state.dateSince = since;
    if (u.get("scope") === "following") state.followingOnly = true;
    // Multi-select: the URL carries a CSV of slugs in the same
    // `category` param (kept singular for backward compatibility
    // with deep-links saved against the v1 single-select picker).
    const catCsv = (u.get("category") || u.get("categories") || "").trim();
    if (catCsv) {
      for (const s of catCsv
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean)) {
        state.categories.add(s.toLowerCase());
      }
      // Deep link → reset the persisted set to match this URL. The
      // user who shares /?category=ai-safety with a friend expects
      // the friend's subsequent navigations to keep that filter on.
      saveCatSelected(state.categories);
    } else {
      // No URL hint — hydrate from the localStorage memory so the
      // selection survives reloads and route changes (feed / personal
      // / search all share the same key).
      for (const s of loadCatSelected()) state.categories.add(s);
    }
    if (fav === "1") state.sources.add(FAV_SOURCE_KEY);
    // VIPs are unlimited (they get routed through the __all__ index
    // when ≥5 selected), non-VIPs are capped at MAX_NONVIPS. The
    // truncation here mirrors the picker's runtime cap so a saved URL
    // with too many slugs loads cleanly instead of erroring out.
    const slugSplit = libsCsv
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s && s !== state.hostSlug);
    const extras = [];
    let nonvipCount = 0;
    for (const s of slugSplit) {
      if (isVipSlug(s)) {
        extras.push(s);
      } else if (nonvipCount < MAX_NONVIPS) {
        extras.push(s);
        nonvipCount++;
      }
    }
    for (const s of extras) state.libs.add(s);
    return extras;
  }

  /* In-browser ColBERT re-ranker (pylate-rs WASM, lives in
   * /pkg/pylate_rs.js, loaded by /colbert.worker.js). Refines the
   * order of the top-29 search results against the user's query so
   * the first-stage API ranking gets a finer-grained late-interaction
   * pass before display.
   *
   * Wiring:
   *   - Worker spins up on page load and starts fetching the model
   *     (cached in the browser's CacheStorage, so subsequent loads
   *     are instant).
   *   - After every search-mode refresh, we post the candidate list
   *     to the worker tagged with a fresh `queryId`. The worker
   *     drops messages whose `queryId` is no longer the latest, so
   *     fast typing doesn't backlog stale rerank work.
   *   - When `rank-complete` lands with a matching queryId AND the
   *     user still has a query in the bar, we replace the rendered
   *     list with the re-ranked output. We ignore `rank-update`
   *     (partial progress) to avoid mid-stream layout flicker.
   *   - All failure modes degrade gracefully: no worker, no WASM,
   *     model download error → the user just sees the un-reranked
   *     API order. Reranking is an improvement, never a hard dep. */
  let rerankWorker = null;
  let rerankReady = false;
  let rerankQueryId = 0;
  /* requestAnimationFrame coalescer — the worker fires one
   * `rank-update` per scored doc (~30 events). Without batching,
   * each event would re-render the full result list and the page
   * would jitter. Schedule at most one render per frame against
   * the latest payload; in practice this collapses 30 updates
   * into 4–6 renders. */
  let rerankPending = null;
  /* Update only the score badge of an existing card. The rerank
   * stream only changes per-doc scores + their order — every other
   * field (title, summary, tags, source, avatars) is identical
   * between batches, so there's no need to re-render the whole
   * card. */
  function updateResultScore(node, d) {
    const foot = node.querySelector(".result-foot-right");
    if (!foot) return;
    const hasRerank = typeof d.colbertScore === "number";
    const score = hasRerank ? d.colbertScore : d.similarity;
    let span = foot.querySelector(".score");
    if (!state.query || typeof score !== "number") {
      if (span) span.remove();
      return;
    }
    if (!span) {
      span = document.createElement("span");
      span.className = "score";
      foot.insertBefore(span, foot.firstChild);
    }
    span.textContent = score.toFixed(3);
    span.classList.toggle("reranked", hasRerank);
    span.title = (hasRerank ? "ColBERT re-ranker" : "Retriever") + " score";
  }

  function applyRerank(payload) {
    state.lastDocs = payload;
    if (rerankPending !== null) return;
    rerankPending = requestAnimationFrame(() => {
      rerankPending = null;
      if (!state.query) return; // user cleared the box mid-stream
      const container = $("results");

      /* Keyed reconciliation: nodes already on screen are reused
       * (their listeners, :hover state, and any open .similar-panel
       * survive untouched); only nodes whose position in the new
       * order differs from their current position are moved. New
       * docs are rendered fresh and wired. Orphans are removed. */
      const existing = new Map();
      for (const article of container.querySelectorAll(".result")) {
        existing.set(article.dataset.url, article);
      }
      let cursor = container.firstElementChild;
      for (const d of state.lastDocs) {
        let node = existing.get(d.url);
        if (node) {
          updateResultScore(node, d);
          existing.delete(d.url);
          if (node === cursor) {
            // Already in the right place — leave it alone, advance.
            cursor = cursor.nextElementSibling;
          } else {
            // Out of order — slot in before the current cursor.
            container.insertBefore(node, cursor);
            // cursor unchanged: node was inserted *before* it.
          }
        } else {
          // First-time render for this URL.
          const tmp = document.createElement("template");
          tmp.innerHTML = renderResult(d).trim();
          const fresh = tmp.content.firstElementChild;
          container.insertBefore(fresh, cursor);
          wireResults(fresh);
        }
      }
      // Remove anything that's no longer in the new order.
      for (const node of existing.values()) node.remove();

      restoreOpenSimilarPanels();
    });
  }
  /* Status pastille — the small coloured dot inside the spotlight,
   * anchored at the right end of the bar. Reflects the worker's lifecycle so
   * the user can see at a glance whether re-ranking is active.
   *   off     — no worker (page just loaded, or worker unsupported)
   *   loading — model files downloading from huggingface.co (orange + pulse)
   *   ready   — model instantiated, re-ranks happening (green)
   *   error   — model load failed; un-reranked results still shown (red) */
  function setRerankStatus(state, label) {
    const dot = $("rerankerDot");
    if (!dot) return;
    dot.dataset.state = state;
    if (label) dot.title = label;
  }
  function postRerank(docs) {
    if (!rerankWorker || !rerankReady || !state.query || !docs.length) return;
    const queryId = ++rerankQueryId;
    rerankWorker.postMessage({
      type: "rank",
      payload: {
        query: state.query,
        // The worker reads `extra-tags` (the legacy hyphen key from
        // database.json); our doc shape exposes `extraTags`. Map at
        // post time so the worker doesn't need to know about both.
        documents: docs.map((d) => ({ ...d, "extra-tags": d.extraTags || [] })),
        queryId,
      },
    });
  }
  try {
    rerankWorker = new Worker("/colbert.worker.js", { type: "module" });
    setRerankStatus("loading", "Loading ColBERT re-ranker…");
    rerankWorker.postMessage({ type: "load" });
    rerankWorker.onmessage = (e) => {
      const { type, payload, queryId } = e.data || {};
      if (type === "status") {
        // The worker emits human-readable progress strings during
        // download/decode/instantiate. Mirror them in the tooltip
        // so a user hovering the dot can see exactly what's
        // happening, without us reserving inline page real estate.
        setRerankStatus("loading", `ColBERT re-ranker · ${payload}`);
      } else if (type === "model-ready") {
        rerankReady = true;
        setRerankStatus("ready", "ColBERT re-ranker · ready");
        // If a query was already on screen when the model finished
        // loading, run a rerank pass now so the first-load query
        // benefits too.
        if (state.query && state.lastDocs.length) postRerank(state.lastDocs);
      } else if (type === "rank-update" || type === "rank-complete") {
        // Stream every per-doc score as it lands. The worker
        // re-sorts internally on each scored doc and emits the
        // partial list, so consuming `rank-update` (not just
        // `rank-complete`) is what produces the visible
        // "documents shuffle into place" effect — items climb the
        // list one by one as their colbertScore comes in.
        if (queryId === rerankQueryId && state.query) {
          applyRerank(payload);
        }
      } else if (type === "error") {
        setRerankStatus(
          "error",
          `ColBERT re-ranker error · ${payload || "unknown"}`,
        );
        console.warn("[colbert] worker error:", payload);
      }
    };
    rerankWorker.onerror = (e) => {
      setRerankStatus(
        "error",
        `ColBERT re-ranker · ${e.message || "worker error"}`,
      );
      console.warn("[colbert] worker:", e.message || e);
    };
  } catch (e) {
    console.warn("[colbert] worker unavailable:", e);
    rerankWorker = null;
    setRerankStatus("off", "ColBERT re-ranker unavailable in this browser");
  }

  /* Composite popularity score from raw social counts. Log-combined
   * so 10k followers ≈ 2× the "weight" of 100, not 100×. Citations
   * are weighted softer since they accumulate across a career and
   * aren't directly comparable to follower counts. Mirrors the
   * `popularityScore` used by the welcome page's Recent Feed. */
  const popularityScore = (p) => {
    const tw = Number((p && p.twitterFollowers) || 0);
    const gh = Number((p && p.githubFollowers) || 0);
    const ci = Number((p && p.citations) || 0);
    return Math.log10(1 + tw) + Math.log10(1 + gh) + 0.4 * Math.log10(1 + ci);
  };
  /* Weighted shuffle (Efraimidis–Spirakis): key = random^(1/weight).
   * Higher popularity → higher expected rank, but a low-popularity
   * sharer can still land first on a lucky draw. The avatars on
   * each card are reshuffled on every refresh, so the order isn't
   * frozen the way a plain `sort by popularity` would be. */
  const weightedShuffleByPopularity = (arr) =>
    arr
      .map((s) => ({ s, k: Math.random() ** (1 / (1 + popularityScore(s))) }))
      .sort((a, b) => b.k - a.k)
      .map((x) => x.s);

  /* People-rail click memory. Tracked in localStorage so the order
   * survives across page loads — the user expects "the people I open
   * most" to stay at the top of their rail. Stored as a flat
   * `{ slug: { at: timestampMs, n: clickCount } }` map. Capped at
   * MAX_TRACKED so a tour of the whole catalogue doesn't dilute the
   * pinning signal; the eviction policy is least-recently-touched.
   *
   * Click recording fires on the rail's avatar / body links (see
   * `armPeopleClickTracker`). Reads happen during `renderPeopleRail`
   * to compute the pinned bucket. */
  const PEOPLE_CLICKS_LS_KEY = "kn.people.clicked";
  const PEOPLE_CLICKS_MAX_TRACKED = 60;
  function _readPeopleClicks() {
    try {
      const raw = localStorage.getItem(PEOPLE_CLICKS_LS_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }
  function _writePeopleClicks(map) {
    try {
      localStorage.setItem(PEOPLE_CLICKS_LS_KEY, JSON.stringify(map));
    } catch {
      /* private mode / quota — drop silently */
    }
  }
  function recordPeopleClick(slug) {
    if (!slug || typeof slug !== "string") return;
    const map = _readPeopleClicks();
    const prev = map[slug] || { at: 0, n: 0 };
    map[slug] = { at: Date.now(), n: (prev.n || 0) + 1 };
    // Cap the tracked set so a user who browses widely doesn't end up
    // with hundreds of "recently clicked" pins. Evict by oldest `at`.
    const slugs = Object.keys(map);
    if (slugs.length > PEOPLE_CLICKS_MAX_TRACKED) {
      slugs
        .map((s) => [s, map[s].at || 0])
        .sort((a, b) => a[1] - b[1])
        .slice(0, slugs.length - PEOPLE_CLICKS_MAX_TRACKED)
        .forEach(([s]) => delete map[s]);
    }
    _writePeopleClicks(map);
  }

  /* Reorder docs for the empty-query browse view.
   *
   * The hierarchy is tiered, owners-first:
   *
   *   1. Bucket every doc by owner count (how many of the active
   *      libraries returned it). Highest tier ships first — every
   *      multi-owner doc lands at the top of the page before any
   *      solo (1-owner) doc, so adding a library always surfaces
   *      the intersection before the long tail.
   *   2. Within each tier, group by the owner-set signature first
   *      (e.g. `a|b` vs `b|c` — distinct overlaps are distinct
   *      buckets) and sort each bucket by date desc. Tier-1 (solo)
   *      docs fall back to bucketing by source key, since every
   *      solo doc has the same owner-set size and signature would
   *      degenerate.
   *   3. Within each tier, run a diversity round-robin: at every
   *      emit, pick the most-recent head from a bucket that hasn't
   *      appeared in the last DIVERSITY_WINDOW (6) emissions. The
   *      window persists across tier boundaries.
   *   4. The window auto-relaxes when fewer than 7 distinct buckets
   *      remain in play: each step drops the oldest forbidden entry
   *      until something is eligible. With one bucket left we end
   *      up emitting it consecutively, which is fine — no choice.
   *
   * Single-source pool returns unchanged (e.g. when the user
   * filters to `source=github`, every doc has the same key and
   * diversity isn't applicable).
   *
   * Caller must skip this in query mode — the relevance ranker
   * owns its own ordering and the user explicitly asked for
   * matches, not variety. */
  const DIVERSITY_WINDOW = 6;

  /* Feed sort — date desc primary, source clustering secondary so
   * same-source rows are guaranteed adjacent within a date, sharer
   * count desc to break ties inside a source bucket.
   *
   * Why this order: the collapse logic downstream merges any run of
   * adjacent same-source rows into one "See more" pill. If a higher-
   * sharer row from another source slipped between two same-source
   * rows, the downstream merge would emit two separate pills. Keeping
   * sources clustered inside the date — and letting same-source rows
   * carry across the date boundary when alphabetical ordering puts
   * the source first on both sides — produces a single pill for an
   * uninterrupted HF run that spans yesterday→today. */
  /* Frontend dedup pass for the feed.
   *
   * Three signals collapse rows into one card:
   *   1. Same canonical URL (lowercase host + path, trailing slash
   *      and query/fragment stripped). Catches near-dupes like
   *      `scholar.google.com/citations?user=X` vs `?user=Y`.
   *   2. Same retweet body — source=twitter and the summary starts
   *      with "Retweet @": hash the canonicalized summary. Each
   *      retweeter's wrapper URL is a distinct row in PG but the
   *      content is identical, so they collapse to one card and the
   *      avatar stack gathers every retweeter.
   *   3. Same coreference URL — if any of `linkedUrls[0].url` matches
   *      the canonical URL of another doc (a tweet linking to an
   *      arxiv paper, plus a direct arxiv bookmark): keep the
   *      richest of the pair.
   *
   * Within each group the "richest" representative wins:
   *   linkedUrls.length, then sharerCount, then summary length,
   *   then most-recent createdAt. The dropped rows contribute their
   *   sharers (deduped by slug) to the representative so the avatar
   *   stack reflects everyone who has the URL in their library. */
  function _canonicalUrl(href) {
    if (!href || typeof href !== "string") return "";
    try {
      const u = new URL(href);
      const host = u.hostname.toLowerCase().replace(/^www\./, "");
      const path = u.pathname.replace(/\/+$/, "");
      return `${host}${path}`;
    } catch {
      return href.toLowerCase();
    }
  }
  /* Cheap, deterministic hash for retweet-body keys — we only need
   * collision resistance across the in-memory doc list (a few hundred
   * rows), not cryptographic strength. djb2 over the canonicalized
   * string is plenty. */
  function _hashStr(s) {
    let h = 5381;
    for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
    return h.toString(36);
  }
  function _retweetKey(doc) {
    const src = (doc.source || "").toLowerCase();
    if (src !== "twitter") return null;
    const body = String(doc.summary || "").trim();
    // Strip any "Retweet @<handle>: " / "RT @<handle>: " prefix, then
    // collapse whitespace. The pipeline sometimes stores the same
    // original tweet body under multiple wrapper URLs (one per
    // followee who quoted / retweeted / replied), often without the
    // explicit RT prefix on every row. Hashing the body itself catches
    // those cases too — at length >= 60 chars the false-positive risk
    // of two distinct tweets sharing the same body is negligible
    // (twitter posts are inherently varied at that length).
    const stripped = body.replace(/^(?:Retweet|RT)\s+@[\w_]+:\s*/i, "").trim();
    const norm = stripped.toLowerCase().replace(/\s+/g, " ");
    if (norm.length < 60) return null;
    return `tw:${_hashStr(norm)}`;
  }
  function _docRichness(d) {
    const linked = Array.isArray(d.linkedUrls) ? d.linkedUrls.length : 0;
    const sharers = d.sharerCount || (d.sharers ? d.sharers.length : 0) || 0;
    const sumLen = (d.summary || "").length;
    const created = d.createdAt || "";
    return [linked, sharers, sumLen, created];
  }
  function _pickRicher(a, b) {
    const ra = _docRichness(a);
    const rb = _docRichness(b);
    for (let i = 0; i < ra.length; i++) {
      if (ra[i] > rb[i]) return a;
      if (ra[i] < rb[i]) return b;
    }
    return a;
  }
  function _mergeSharers(into, from) {
    const a = Array.isArray(into.sharers) ? into.sharers : [];
    const b = Array.isArray(from.sharers) ? from.sharers : [];
    if (b.length === 0) return into;
    const seen = new Set(a.map((s) => s && s.slug).filter(Boolean));
    const merged = a.slice();
    for (const s of b) {
      if (!s || !s.slug || seen.has(s.slug)) continue;
      merged.push(s);
      seen.add(s.slug);
    }
    into.sharers = merged;
    into.sharerCount = merged.length;
    // Keep `_owners` (slug-only list) in sync — the avatar-stack
    // renderer reads it on cards that came from the search path.
    if (Array.isArray(into._owners)) {
      const ownSet = new Set(into._owners);
      for (const s of merged)
        if (s && s.slug && !ownSet.has(s.slug)) into._owners.push(s.slug);
    }
    return into;
  }
  function dedupFeedDocs(docs) {
    if (!Array.isArray(docs) || docs.length < 2) return docs || [];
    // Two-pass grouping: first by retweet signature (most reliable
    // multi-row collapse), then by canonical URL. Each doc maps to at
    // most one key from each pass — when both keys collide with an
    // existing group we still only merge once (the second pass sees
    // the representative already in `byKey`).
    const byKey = new Map(); // key -> representative doc index in out[]
    const out = [];
    function assign(doc, keys) {
      for (const k of keys) {
        if (!k) continue;
        if (byKey.has(k)) {
          const idx = byKey.get(k);
          const existing = out[idx];
          const winner = _pickRicher(existing, doc);
          if (winner === existing) {
            _mergeSharers(existing, doc);
          } else {
            _mergeSharers(doc, existing);
            // Replace in-place so all keys pointing at this group
            // continue to resolve to the new representative.
            out[idx] = doc;
          }
          // Re-point every existing key for this group at idx so the
          // next overlapping doc hits the same slot.
          for (const k2 of keys) if (k2) byKey.set(k2, idx);
          return;
        }
      }
      const idx = out.length;
      out.push(doc);
      for (const k of keys) if (k) byKey.set(k, idx);
    }
    for (const d of docs) {
      const keys = [_retweetKey(d), _canonicalUrl(d.url)];
      assign(d, keys);
    }
    return out;
  }

  function reorderFeed(docs) {
    if (docs.length === 0) return [];
    // Collapse retweet + URL near-duplicates BEFORE anything else.
    // Otherwise the picks-merge below treats each duplicate as a
    // fresh same-source row and produces awkward visual runs.
    docs = dedupFeedDocs(docs);
    // User-shuffled order short-circuits — the click handler did
    // its own Fisher-Yates and any re-sort here would undo it on
    // the next infinite-scroll append.
    if (state.feedShuffled) return docs;
    // Honour the server's score-based order. The API now ranks via
    // the precomputed `feed_snapshot.score` (sci × 6 + flat
    // recency + LN-of-VIP-share + rich-tweet …), plus per-viewer
    // bonuses applied at read time, plus a Rust-side diversity
    // pass that anti-bunches the same primary user. Re-sorting by
    // date DESC here used to override all of that — a 1-week-old
    // single-VIP card outranked a 1-month-old 15-VIP consensus
    // doc on the page even though the server scored them the
    // other way. We now pass the array through untouched.
    //
    // HN front-page picks still ride along at the position the
    // server interleaved them at; nothing to do for them either.
    return docs;
  }

  /* Build a render plan for feed docs. Walk the (already-sorted) list
   * and, whenever 7+ consecutive items share the same `source`, emit
   * the first 6 as normal cards and bundle the rest into a single
   * "see more" pill the user can expand inline. Threshold is one
   * higher than the visible cap so a run of exactly 6 still renders
   * flat — there's nothing to hide. */
  const FEED_COLLAPSE_VISIBLE = 3;
  const FEED_COLLAPSE_THRESHOLD = FEED_COLLAPSE_VISIBLE + 1;
  // Manual collapse threshold: any same-source run of ≥6 flat cards
  // (i.e. >5) gets a small "collapse" chevron on every card in the
  // run except the last. Kicks in mainly after the user expanded
  // an auto-collapse pill.
  const MANUAL_COLLAPSE_RUN_MIN = 6;
  function buildFeedRenderPlan(docs) {
    const plan = [];
    let i = 0;
    while (i < docs.length) {
      const src = docs[i].source || "";
      let j = i;
      while (j < docs.length && (docs[j].source || "") === src) j++;
      const runLen = j - i;
      if (runLen >= FEED_COLLAPSE_THRESHOLD && src) {
        for (let k = 0; k < FEED_COLLAPSE_VISIBLE; k++)
          plan.push({ kind: "doc", doc: docs[i + k] });
        plan.push({
          kind: "collapse",
          source: src,
          docs: docs.slice(i + FEED_COLLAPSE_VISIBLE, j),
        });
      } else {
        for (let k = i; k < j; k++) plan.push({ kind: "doc", doc: docs[k] });
      }
      i = j;
    }
    return plan;
  }

  /* Cache of unexpanded collapse groups, keyed by an id we stamp on
   * the pill. When the user clicks "See more N", we look up the docs
   * and splice them into the DOM, then drop the cache entry. */
  const _feedCollapseCache = new Map();
  let _feedCollapseSeq = 0;

  function renderFeedPlanItems(plan) {
    const parts = [];
    for (const item of plan) {
      if (item.kind === "doc") {
        parts.push(renderResult(item.doc));
      } else {
        const id = `fc_${++_feedCollapseSeq}`;
        _feedCollapseCache.set(id, item.docs);
        const icon = K.sourceIconUrl(item.source);
        const iconHtml = icon
          ? `<img class="fc-ico" src="${escapeAttr(icon)}" alt="" onerror="this.style.display='none'"/>`
          : `<span class="fc-ico fc-ico-fallback" aria-hidden="true">●</span>`;
        const label = displaySource(item.source) || item.source;
        parts.push(`<button class="feed-collapse" data-fc-id="${id}" type="button">
          ${iconHtml}
          <span class="fc-label">See ${item.docs.length} more from ${escapeHtml(label)}</span>
          <svg class="fc-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>`);
      }
    }
    return parts.join("");
  }

  /* Collapse rules are disabled by request — the feed never folds
   * same-source runs into a "See N more" pill. Every doc renders
   * as its own card. `buildFeedRenderPlan` + `renderFeedPlanItems`
   * stay defined for the on-demand expand path (`wireFeedCollapse`)
   * but are unreachable from the initial render. */
  function renderFeedDocsHtml(docs) {
    // Idempotent: reorderFeed already runs dedup before the sort, but
    // a few render paths (favorites view, feed-search post-merge,
    // manual collapse expand) reach here without going through the
    // sort. Running the pass twice is a no-op when there's nothing to
    // collapse, so it's cheaper to always re-apply than to track which
    // caller already dedup'd.
    return dedupFeedDocs(docs).map(renderResult).join("");
  }

  /* Manual collapse button is disabled by request. The function
   * stays in place (called from a few hot paths) but only strips
   * any stale buttons that may have leaked into the DOM from
   * cached HTML; it never *adds* a fresh button. */
  function armManualCollapseButtons() {
    const host = $("results");
    if (!host) return;
    for (const el of host.querySelectorAll(".result-manual-collapse")) {
      el.remove();
    }
  }

  function collapseAfter(article, src) {
    const host = $("results");
    if (!host) return;
    // Walk BACKWARDS from `article` to find the first card of the run
    // — we want to fold the whole same-source sequence into ONE pill,
    // not just the cards after the clicked one.
    let first = article;
    while (true) {
      const prev = first.previousElementSibling;
      if (!prev) break;
      if (prev.tagName !== "ARTICLE") break;
      const url = prev.dataset?.url;
      const doc = (state.lastDocs || []).find((d) => d.url === url);
      if ((doc?.source || "") !== src) break;
      first = prev;
    }
    // Walk FORWARDS from `first` to gather every card in the run.
    const toCollapse = [];
    let cur = first;
    while (cur) {
      if (cur.classList?.contains("feed-collapse")) break;
      if (cur.tagName !== "ARTICLE") break;
      const url = cur.dataset?.url;
      const doc = (state.lastDocs || []).find((d) => d.url === url);
      if ((doc?.source || "") !== src) break;
      toCollapse.push({ el: cur, doc });
      cur = cur.nextElementSibling;
    }
    if (!toCollapse.length) return;
    // Build a collapse pill that re-uses the same cache + handlers as
    // the auto-generated ones. The pill HTML mirrors renderFeedPlanItems.
    const id = `fc_${++_feedCollapseSeq}`;
    _feedCollapseCache.set(id, toCollapse.map((x) => x.doc).filter(Boolean));
    const icon = K.sourceIconUrl(src);
    const iconHtml = icon
      ? `<img class="fc-ico" src="${escapeAttr(icon)}" alt="" onerror="this.style.display='none'"/>`
      : `<span class="fc-ico fc-ico-fallback" aria-hidden="true">●</span>`;
    const label = displaySource(src) || src;
    const pillHtml = `<button class="feed-collapse" data-fc-id="${id}" type="button">
        ${iconHtml}
        <span class="fc-label">See ${toCollapse.length} more from ${escapeHtml(label)}</span>
        <svg class="fc-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>`;
    // Insert the pill where the run started, then remove every card
    // in the run (including the one the user clicked).
    first.insertAdjacentHTML("beforebegin", pillHtml);
    const pill = first.previousElementSibling;
    for (const x of toCollapse) x.el.remove();
    mergeAdjacentCollapsePills();
    wireFeedCollapse(host);
    // Bring the pill into view so the user isn't left scrolled to
    // empty space where their click target used to be. `nearest`
    // means "only scroll if it's out of view", and we offset by a
    // small margin (~80px) so it doesn't hug the very top edge.
    if (pill) {
      requestAnimationFrame(() => {
        const rect = pill.getBoundingClientRect();
        // Only adjust if the pill is above the visible area or close
        // to the top — otherwise leave the user's scroll alone.
        if (rect.top < 80 || rect.bottom > window.innerHeight) {
          window.scrollTo({
            top: window.scrollY + rect.top - 96,
            behavior: "smooth",
          });
        }
      });
    }
  }

  /* Merge any adjacent `feed-collapse` pills in #results that share
   * the same source into a single pill. Called after every collapse
   * insertion and after every expand restoration. Idempotent. */
  function mergeAdjacentCollapsePills() {
    const host = $("results");
    if (!host) return;
    const children = [...host.children];
    for (let i = 0; i < children.length - 1; i++) {
      const a = children[i];
      const b = children[i + 1];
      if (!a.classList?.contains("feed-collapse")) continue;
      if (!b.classList?.contains("feed-collapse")) continue;
      const idA = a.dataset.fcId;
      const idB = b.dataset.fcId;
      const docsA = _feedCollapseCache.get(idA);
      const docsB = _feedCollapseCache.get(idB);
      if (!docsA || !docsB) continue;
      // Match by source — peek at the first doc of each cache entry.
      const srcA = docsA[0]?.source || "";
      const srcB = docsB[0]?.source || "";
      if (!srcA || srcA !== srcB) continue;
      // Merge the two: replace `a`'s cache with the concatenated docs
      // and re-label, remove `b`. Re-wire to pick up the new total.
      const merged = [...docsA, ...docsB];
      _feedCollapseCache.set(idA, merged);
      _feedCollapseCache.delete(idB);
      const label = a.querySelector(".fc-label");
      if (label) {
        const srcLabel = displaySource(srcA) || srcA;
        label.textContent = `See ${merged.length} more from ${srcLabel}`;
      }
      b.remove();
      // Loop is over a stale `children` snapshot — restart so we keep
      // merging chains (A → A+B; if there was a C, A+B and C now sit
      // adjacent and we want them to fuse too).
      mergeAdjacentCollapsePills();
      return;
    }
  }

  function wireFeedCollapse(scope) {
    (scope || $("results"))
      .querySelectorAll(".feed-collapse")
      .forEach((btn) => {
        if (btn.dataset.fcWired === "1") return;
        btn.dataset.fcWired = "1";
        btn.addEventListener("click", () => {
          const id = btn.dataset.fcId;
          const docs = _feedCollapseCache.get(id);
          if (!docs || !docs.length) return;
          _feedCollapseCache.delete(id);
          // Replace the pill with the hidden docs, in order. They might
          // themselves contain runs of 4+ same-source rows (the original
          // batch had ≥1 run already), but since the user explicitly
          // asked for these, render them flat — no recursive collapse.
          const html = docs.map(renderResult).join("");
          btn.insertAdjacentHTML("afterend", html);
          const inserted = [];
          let cur = btn.nextElementSibling;
          for (let i = 0; i < docs.length && cur; i++) {
            inserted.push(cur);
            cur = cur.nextElementSibling;
          }
          btn.remove();
          // Re-wire the newly inserted cards AND re-arm the manual
          // collapse buttons so the user can fold the freshly-expanded
          // same-source run back up.
          wireResults($("results"));
          armManualCollapseButtons();
        });
      });
  }

  function reorderForBrowse(docs) {
    if (docs.length === 0) return [];

    // Tier by owner count desc. Each tier is a flat list that we'll
    // bucket-by-source inside the loop below.
    const tiers = new Map();
    for (const d of docs) {
      const ow = (d._owners || []).length;
      if (!tiers.has(ow)) tiers.set(ow, []);
      tiers.get(ow).push(d);
    }
    const tierKeys = [...tiers.keys()].sort((a, b) => b - a);

    const out = [];
    const recent = []; // window of last-emitted source keys, persists across tiers

    for (const ow of tierKeys) {
      const buckets = new Map();
      // For multi-owner tiers, the bucket key is the canonical
      // owner-set signature (e.g. `a|b` vs `b|c`) so a heavy a∩b
      // overlap doesn't drown out a smaller b∩c overlap. For tier-1
      // (solo docs) the signature degenerates to one slug per doc,
      // which would make round-robin meaningless — fall back to the
      // source key there to preserve the original cross-source mix.
      for (const d of tiers.get(ow)) {
        const k =
          ow >= 2
            ? (d._owners || []).slice().sort().join("|")
            : d.source || "_misc";
        if (!buckets.has(k)) buckets.set(k, []);
        buckets.get(k).push(d);
      }
      for (const arr of buckets.values()) {
        // Date desc, then createdAt desc as tiebreaker so two docs
        // posted on the same day surface most-recent-first. Most
        // docs in the indexed pool have an empty createdAt — those
        // tie with each other and keep their server order, which
        // is fine because they were inserted into PG earlier than
        // any fresh compose-dialog post which DOES carry the
        // stamp. Empty strings sort last in localeCompare.
        arr.sort((a, b) => {
          const byDate = (b.date || "").localeCompare(a.date || "");
          if (byDate !== 0) return byDate;
          return (b.createdAt || "").localeCompare(a.createdAt || "");
        });
      }
      const anyLeft = () => {
        for (const arr of buckets.values()) if (arr.length) return true;
        return false;
      };
      while (anyLeft()) {
        let pickKey = null,
          pickDoc = null;
        // Try with the full window; if no eligible bucket, drop the
        // oldest forbidden entry and retry. Terminates because
        // `recent` is bounded and shrinking, and at recent=[] every
        // non-empty bucket is eligible.
        while (!pickDoc) {
          const forbidden = new Set(recent);
          for (const [k, arr] of buckets) {
            if (!arr.length || forbidden.has(k)) continue;
            const head = arr[0];
            // Same date + createdAt comparator as the intra-bucket
            // sort above — so cross-bucket interleave also surfaces
            // a fresh same-day post ahead of an older one.
            let cmp = 0;
            if (pickDoc) {
              cmp = (head.date || "").localeCompare(pickDoc.date || "");
              if (cmp === 0) {
                cmp = (head.createdAt || "").localeCompare(
                  pickDoc.createdAt || "",
                );
              }
            }
            if (!pickDoc || cmp > 0) {
              pickKey = k;
              pickDoc = head;
            }
          }
          if (pickDoc) break;
          if (recent.length === 0) break; // safety; anyLeft guards above
          recent.shift();
        }
        if (!pickDoc) break;
        buckets.get(pickKey).shift();
        out.push(pickDoc);
        recent.push(pickKey);
        if (recent.length > DIVERSITY_WINDOW) recent.shift();
      }
    }
    return out;
  }
  /* On /empty there is no host personality to fetch — the slug is
   * a sentinel, not a real user. Skip the per-host calls and let
   * the picker fan out the public list / bookmarks normally. */
  // Show the rail spinner from the very first frame. Hidden again
  // by the first renderSrc() that has rows.
  showSrcSpinner();
  // `loadMe` rides along with the other boot fetches so the first
  // paint already knows whether the viewer is signed in. We used to
  // call it from a `.then(...)` after `refresh()` had already
  // painted, then re-`innerHTML` the whole result list to flip on
  // the fav buttons — which read as a visible blink (every card
  // unmounted and re-mounted in one frame). With `me` resolved
  // pre-paint the rebuild is no longer necessary.
  const [hostMeta, allPersonalities, hostSources, favs, favOwners, meEarly] =
    await Promise.all([
      isEmptyHost
        ? Promise.resolve(null)
        : K.getPersonality(slug).catch(() => null),
      K.listPersonalities().catch(() => []),
      isEmptyHost ? Promise.resolve([]) : K.getSources(slug),
      K.getFavoriteUrls(),
      K.getFavoriteOwners(),
      loadMe(),
    ]);
  me = meEarly || null;
  state.favoriteOwners = new Set(Array.isArray(favOwners) ? favOwners : []);
  if (!isEmptyHost) {
    state.perSlugMeta[slug] = hostMeta || { slug, name: slug, avatar: "" };
    state.perSlugSources[slug] = hostSources || [];
    if (hostMeta && hostMeta.name)
      $("q").placeholder = `Search ${hostMeta.name} knowledge`;
  }
  // Behavioural tracker — fire-and-forget. Identifies the viewer (for
  // personalised recs) and the library being browsed (for per-library
  // stats), then logs the page view.
  if (window.kn) {
    window.kn.setViewer({ id: me?.id });
    window.kn.setPersonality({ id: hostMeta?.id, slug });
    window.kn.track("view", { user_id: hostMeta?.id, personality_slug: slug });
  }
  state.favorites = favs;
  // Soft-deleted URLs — the index can still hold these until a
  // re-index lands. We filter them out of every search / similar /
  // latest result client-side so the user never sees a doc they
  // removed.
  state.deletedUrls = new Set();
  (async () => {
    try {
      const r = await fetch(`${API_BASE}/auth/me/deleted-urls`, {
        credentials: "include",
      });
      if (r.ok) state.deletedUrls = new Set(await r.json());
    } catch {}
  })();
  const dropDeleted = (rows) =>
    Array.isArray(rows) && state.deletedUrls?.size
      ? rows.filter((d) => !state.deletedUrls.has(d.url))
      : rows;
  // Wrap the index-side fetchers once. Every call site (feed search,
  // personal-page browse, findSimilar, latest) now post-filters out
  // soft-deleted URLs transparently.
  for (const fn of ["search", "findSimilar", "latest"]) {
    if (typeof K[fn] !== "function") continue;
    const orig = K[fn].bind(K);
    K[fn] = async (...args) => dropDeleted(await orig(...args));
  }
  // The /api/users endpoint already filters server-side to
  // (vip = TRUE AND documentCount > 0), so we don't redo that
  // here. The host slug is always represented via hostMeta further
  // down, which guarantees a brand-new (non-vip) account viewing
  // its own page still has a row to render.
  state.allPersonalities = Array.isArray(allPersonalities)
    ? allPersonalities
    : [];
  // Inject the host (which may be non-vip — e.g. a brand-new account
  // viewing its own page) so the picker / rail always has a row for
  // them. `hostMeta` already carries the full UserResponse shape.
  if (
    hostMeta &&
    !state.allPersonalities.find((p) => p.slug === state.hostSlug)
  ) {
    state.allPersonalities.unshift(hostMeta);
  }
  for (const p of state.allPersonalities) {
    if (!state.perSlugMeta[p.slug]) state.perSlugMeta[p.slug] = p;
    // Index by twitter handle for the retweet-author avatar lookup.
    // `sources.twitter` may be a string (legacy) or an object with
    // a `username` key (current shape).
    const tw = p.sources && p.sources.twitter;
    const handle =
      typeof tw === "string"
        ? tw
        : tw && typeof tw === "object"
          ? tw.username
          : "";
    if (handle) {
      state.slugByTwitterHandle[String(handle).toLowerCase()] = p.slug;
    }
  }
  rebuildAllSources();
  /** Build the left-rail source list from the current result set.
   *
   * Used when ≥5 libs are selected and we skipped per-slug
   * `getSources()` calls. The rail shows ONLY the sources that
   * actually appear in the visible docs — keeps the list short and
   * relevant. No counts: in big-selection mode counts derived from
   * a top-N result slice are misleading, so we drop the column
   * entirely. The user can still search the rail by typing.
   */
  /* Feed-mode source rail.
   *
   * Aggregates per-source totals across (signed-in user ∪ followees)
   * by reusing the same /api/sources endpoint we hit for personal
   * pages. Loads each followee's source list once and memoises into
   * state.perSlugSources, so subsequent renders are free.
   *
   * Returns the merged list (also assigned to state.allSources).
   */
  async function rebuildAllSourcesForFeed() {
    // Surface the rail spinner only when the list is currently empty
    // — re-renders triggered by toggling a chip should NOT flash a
    // loader since the existing rows are still valid.
    if (!state.allSources?.length) showSrcSpinner();
    // One round-trip to /api/me/feed/sources — the server aggregates
    // across (followees ∪ self) in a single GROUP BY against the
    // `user_source_counts` view. Replaces the previous per-followee
    // fan-out (N calls → 1) and the client-side merge loop.
    try {
      const r = await fetch(`${API_BASE}/api/me/feed/sources`, {
        credentials: "include",
      });
      if (!r.ok) {
        state.allSources = [];
        return;
      }
      const rows = await r.json();
      const real = (Array.isArray(rows) ? rows : []).map((s) =>
        s.key === "twitter" ? { ...s, label: "X" } : s,
      );
      // Pin Favorites as the first chip whenever the signed-in user
      // has at least one star. Same behaviour as the personal-page
      // rail — Favorites are personal-only, so we only show them on
      // the feed (which always includes the caller's library).
      if (me && state.favorites && state.favorites.size > 0) {
        state.allSources = [
          {
            key: FAV_SOURCE_KEY,
            label: "Upvoted",
            count: state.favorites.size,
            _synthetic: true,
          },
          ...real,
        ];
      } else {
        state.allSources = real;
      }
    } catch {
      state.allSources = [];
    }
  }

  function rebuildAllSourcesFromDocs(docs) {
    const seen = new Set();
    const real = [];
    for (const d of docs || []) {
      const key = d.source || "";
      if (!key || seen.has(key)) continue;
      seen.add(key);
      real.push({ key, label: key === "twitter" ? "X" : key });
    }
    // Stable alphabetical order — without counts, sorting by
    // descending count is meaningless.
    real.sort((a, b) => a.label.localeCompare(b.label));
    const showFavorites =
      me &&
      me.slug &&
      state.libs.has(me.slug) &&
      state.favorites &&
      state.favorites.size > 0;
    state.allSources = showFavorites
      ? [
          {
            key: FAV_SOURCE_KEY,
            label: "Upvoted",
            _synthetic: true,
          },
          ...real,
        ]
      : real;
  }

  function rebuildAllSources() {
    const map = new Map();
    for (const s of state.libs) {
      for (const src of state.perSlugSources[s] || []) {
        const ex = map.get(src.key);
        if (!ex) map.set(src.key, { ...src });
        else ex.count = (ex.count || 0) + (src.count || 0);
      }
    }
    // Display alias: the `twitter` source chip reads as "X" in the
    // rail. Logo / data key stay the same; only the visible label
    // changes.
    const twitter = map.get("twitter");
    if (twitter) twitter.label = "X";
    const real = Array.from(map.values()).sort(
      (a, b) => (b.count || 0) - (a.count || 0),
    );
    // Pin the Favorites pseudo-source at the top whenever the user
    // has at least one star — and their own library is part of the
    // current view (host OR any extra lib). Favorites are personal:
    // they only make sense when "self" is in the lib set.
    const showFavorites =
      me &&
      me.slug &&
      state.libs.has(me.slug) &&
      state.favorites &&
      state.favorites.size > 0;
    if (showFavorites) {
      state.allSources = [
        {
          key: FAV_SOURCE_KEY,
          label: "Upvoted",
          count: state.favorites.size,
          _synthetic: true,
        },
        ...real,
      ];
    } else {
      state.allSources = real;
    }
    // Kick off Twitter-freshness probes for any active lib that
    // has a twitter source but hasn't been probed yet. Fire-and-
    // forget: each probe re-renders the source list when it lands
    // so the dot updates without blocking the rest of the panel.
    for (const slug of state.libs) {
      const has = (state.perSlugSources[slug] || []).some(
        (src) => src.key === "twitter",
      );
      if (has && !(slug in state.perSlugTwitterFreshness)) {
        probeTwitterFreshness(slug);
      }
    }
  }
  async function ensureLibLoaded(s) {
    if (!state.perSlugSources[s]) {
      try {
        state.perSlugSources[s] = await K.getSources(s);
      } catch {
        state.perSlugSources[s] = [];
      }
    }
  }

  /* Twitter-bookmark freshness probe.
   *
   * Pulls the single most-recent indexed twitter doc per active
   * library and computes "days since last bookmark synced". The
   * result paints a tiny coloured dot on the Twitter source chip:
   *
   *   green  — last bookmark < 7 days  (cookies are working)
   *   amber  — 7-30 days                (probably still fine)
   *   red    — > 30 days or none       (cookies likely expired)
   *
   * 100% client-side: reuses the existing /metadata/get index call
   * (the same endpoint K.latest hits in browse mode), no new
   * server endpoint, no leaking of the encrypted cookies.
   *
   * Cached in `state.perSlugTwitterFreshness[slug]` (initialised in
   * the state object literal) so toggling the library checkbox
   * doesn't refetch every time. */
  async function probeTwitterFreshness(slug) {
    if (slug in state.perSlugTwitterFreshness) return; // already computed
    state.perSlugTwitterFreshness[slug] = null; // mark in-flight to dedup races
    try {
      const rs = await K.latest({
        indexName: slug,
        count: 1,
        condition: "source = ?",
        parameters: ["twitter"],
      });
      const top = Array.isArray(rs) && rs[0] && rs[0].date ? rs[0] : null;
      if (!top) {
        state.perSlugTwitterFreshness[slug] = {
          date: null,
          daysSince: Infinity,
        };
      } else {
        const ms = Date.now() - new Date(top.date + "T00:00:00").getTime();
        const daysSince = Math.max(0, Math.floor(ms / 86400000));
        state.perSlugTwitterFreshness[slug] = { date: top.date, daysSince };
      }
    } catch {
      state.perSlugTwitterFreshness[slug] = { date: null, daysSince: Infinity };
    }
    // Source rail listens to freshness for sort weighting;
    // re-render so the chip order picks up the new daysSince.
    renderSrc();
  }
  /* Search-bar placeholder reflects the current library selection.
   * Single-lib reads as "Search {name} knowledge"; multi-lib swaps
   * to "Search across N libraries" because dropping one user's
   * name there is misleading when several are active; empty
   * selection prompts the user to pick one. Re-runs whenever
   * `state.libs` changes. */
  function updatePlaceholder() {
    const input = $("q");
    if (!input) return;
    const libs = [...state.libs];
    if (libs.length === 0) {
      input.placeholder = "Search your feed";
      return;
    }
    if (libs.length === 1) {
      const slug = libs[0];
      const meta = state.perSlugMeta[slug];
      const name =
        (meta && meta.name) ||
        (state.allPersonalities.find((p) => p.slug === slug) || {}).name ||
        slug;
      input.placeholder = `Search ${name} knowledge`;
      return;
    }
    input.placeholder = `Search across ${libs.length} libraries`;
  }
  /* Multi-lib selection has no rail UI of its own — the library
   * picker modal owns add/remove. The only side effect of a libs
   * change on the rail surface is the search-bar placeholder, so
   * the refresh entry point reduces to a single call. Kept as a
   * named alias because half a dozen call sites read better as
   * `renderLibs()` than as `updatePlaceholder()`. */
  const renderLibs = updatePlaceholder;
  /* When the user unchecks their own (host) library we pivot to the
   * unified /search route. Any remaining libs ride along through the
   * `?libs=` param (in their existing order); a fully-empty selection
   * lands on bare /search so a refresh restores the empty picker view
   * rather than re-loading the user's profile. The rest of the query
   * string (q / tags / sources) is preserved verbatim. */
  function pivotAwayFromHost() {
    const remainingLibs = [...state.libs].filter((s) => s !== state.hostSlug);
    const u = new URLSearchParams(location.search);
    if (remainingLibs.length) u.set("libs", remainingLibs.join(","));
    else u.delete("libs");
    const tail = u.toString();
    location.href = `/search${tail ? `?${tail}` : ""}`;
  }

  /* Discover overlay — top-of-rail button. Reuses the onboarding
   * module's category picker rendered into the `#discoverBody`
   * inside a native <dialog>. Distinct copy ("discover" mode) for
   * users who already have follows. */
  function openDiscoverOverlay() {
    const dialog = $("discoverDialog");
    const body = $("discoverBody");
    if (!dialog || !body || !window.KnowledgeOnboarding) return;
    if (typeof dialog.showModal === "function" && !dialog.open) {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
    window.KnowledgeOnboarding.open({
      personalities: state.allPersonalities,
      apiBase: API_BASE,
      mode: "discover",
      host: body,
      onSkip: () => {
        // "Skip for now" inside the discover overlay just closes it.
        if (typeof dialog.close === "function" && dialog.open) dialog.close();
        else dialog.removeAttribute("open");
        body.innerHTML = "";
      },
    });
  }
  $("discoverBtn")?.addEventListener("click", () => {
    // Heading click → open the overlay when signed in; anonymous
    // users get the auth modal on the sign-in view (they can switch
    // to signup from there if they don't have an account yet).
    if (me) openDiscoverOverlay();
    else window.KnowledgeAuth?.open("login");
  });
  /* Close paths besides the in-panel Skip button:
   *   - Esc key (native <dialog> behaviour, free)
   *   - Click on the backdrop (i.e. the dialog element itself,
   *     not the card child) */
  function closeDiscoverOverlay() {
    const dialog = $("discoverDialog");
    const body = $("discoverBody");
    if (!dialog) return;
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else dialog.removeAttribute("open");
    if (body) body.innerHTML = "";
  }
  $("discoverDialog")?.addEventListener("click", (e) => {
    if (e.target === $("discoverDialog")) closeDiscoverOverlay();
  });

  /* ── Add-library picker (centred multi-select modal) ──────────
   * Buffers a tentative selection until the user clicks Done, so
   * mid-pick state changes don't trigger refresh storms. Cancel /
   * close discards the buffer. Currently-active libs are pinned to
   * the top so they read as "already in your library" candidates
   * for removal. */
  /* New cap model:
   *   - VIP libraries: no per-library cap. ≥5 selected routes through
   *     the unified `__all__` index (one ColBERT query, owner-filtered).
   *   - Non-VIP libraries: hard cap of 10. Each non-VIP fans out to
   *     its own per-user index (no shared index for non-VIPs), so the
   *     cap exists to keep the parallel-fanout cost bounded.
   *   - The host slug counts as a non-VIP toward MAX_NONVIPS only when
   *     the host is itself non-vip (rare — usually VIP).
   * Threshold for switching to the `__all__` index. Below this we
   * keep the per-library fanout for accuracy (each user's index is
   * tighter than the cross-personality merged one).
   * `MAX_NONVIPS`, `ALL_INDEX_THRESHOLD`, `ALL_INDEX_NAME` are
   * declared at the top of the IIFE so `rebuildAllSources()` can
   * call `useAllOnly()` during boot without hitting a `const` TDZ.
   */

  /** True when the selection is large enough that we should skip every
   *  per-slug round-trip and serve the whole UI from `__all__` only.
   *  Same threshold drives picker hydration and the search/browse
   *  paths so the behaviour is consistent across boot + interaction.
   */
  function useAllOnly(libs) {
    return libs.length >= ALL_INDEX_THRESHOLD;
  }

  /** Lookup: is this slug a VIP? Pulls from already-loaded personalities. */
  function isVipSlug(slug) {
    const p = state.allPersonalities.find((u) => u.slug === slug);
    return !!(p && p.vip);
  }

  /** Split a list of slugs into {vips, nonvips}. */
  function splitByVip(slugs) {
    const vips = [];
    const nonvips = [];
    for (const s of slugs) {
      if (isVipSlug(s)) vips.push(s);
      else nonvips.push(s);
    }
    return { vips, nonvips };
  }
  /* Map of `source key → result count` populated by the source
   * filter's relevance lookup. `null` means "input is empty, fall
   * back to the static count-based ordering". The counts come from
   * a real ColBERT search across the active libraries — so typing
   * "embeddings" in the source box re-ranks chips by how many
   * results actually match that query, not just by chip-label
   * substring. */
  state.sourceRelevance = null;
  let srcRelevanceDebT = null;
  let srcRelevanceQueryId = 0;
  async function runSourceRelevance(query) {
    const myId = ++srcRelevanceQueryId;
    const libs = [...state.libs];
    try {
      let results;
      if (libs.length === 0) {
        // Feed mode: no per-lib indices to fan out to. Hit the cross-
        // library __all__ index, then scope results to (me ∪ followees)
        // so the count matches the user's actual feed. topK scales with
        // pool size — ~30 hits/source signal still holds when split
        // across many owners.
        const scope = me
          ? new Set([...(_peopleRail?.following || []), me.slug])
          : null;
        let docs = await K.search({
          indexName: ALL_INDEX_NAME,
          query,
          topK: 800,
        }).catch(() => []);
        if (scope) docs = docs.filter((d) => scope.has(d.owner));
        results = [docs];
      } else {
        // Personal-page / multi-lib path. Per-lib fan-out — bumped to
        // top-150 so small / focused sources whose top hits sit
        // deeper in the relevance list still surface as candidates.
        results = await Promise.all(
          libs.map((s) =>
            K.search({ indexName: s, query, topK: 150 }).catch(() => []),
          ),
        );
      }
      if (myId !== srcRelevanceQueryId) return; // a newer query supersedes
      // Record the rank of each source's FIRST appearance across the
      // merged result set. Lower = earlier = more relevant. We still
      // keep a `counts` shape under state.sourceRelevance so the
      // displayCount in renderSrc keeps working, but the sort key is
      // now the rank map below.
      const counts = {};
      const firstRank = {};
      let rank = 0;
      const seenSources = new Set();
      for (const arr of results) {
        for (const d of arr) {
          rank++;
          const k = d.source || "";
          if (!k) continue;
          counts[k] = (counts[k] || 0) + 1;
          if (firstRank[k] == null) firstRank[k] = rank;
          seenSources.add(k);
        }
      }
      // The feed source rail is capped at the top-50 popular sources,
      // so the long tail (e.g. mixedbread.com) is missing from
      // `state.allSources`. ColBERT just told us which sources
      // actually contain rows matching the typed query — splice the
      // novel keys in so renderSrc can surface them. Each novel chip
      // gets a synthetic label = key (no count column on the feed).
      const knownKeys = new Set((state.allSources || []).map((s) => s.key));
      const novel = [...seenSources]
        .filter((k) => !knownKeys.has(k))
        .map((k) => ({
          key: k,
          label: k === "twitter" ? "X" : k,
          count: counts[k] || 0,
        }));
      if (novel.length)
        state.allSources = [...(state.allSources || []), ...novel];
      state.sourceRelevance = counts;
      state.sourceFirstRank = firstRank;
      renderSrc();
    } catch {
      if (myId !== srcRelevanceQueryId) return;
      state.sourceRelevance = null;
      state.sourceFirstRank = null;
      renderSrc();
    }
  }

  function renderSrc() {
    hideSrcSpinner();
    const q = $("srcFilter").value.trim().toLowerCase();
    const rel = state.sourceRelevance;
    // Normalise both sides by stripping non-alphanumeric characters,
    // so user-typed spaces / dots / hyphens don't break the match
    // against source keys like `maxhalford.github.io` or
    // `ourworldindata.org`. The same `norm`/`nq` are reused by the
    // sort comparator below.
    const norm = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    const nq = norm(q);
    const labelHit = (s) => {
      if (!nq) return false;
      const nLabel = norm(s.label);
      const nKey = norm(s.key);
      return nLabel.includes(nq) || nKey.includes(nq);
    };
    // Three modes:
    //   1. Empty input → original behaviour (all chips, count desc).
    //   2. Relevance ready → keep chips with hits, label substring,
    //      or active selection. Sort by hit count (desc), then label
    //      substring (alpha among match), then alpha for the rest.
    //   3. Input typed but relevance not yet returned → fall back to
    //      label substring so the panel doesn't go blank during the
    //      ~200 ms debounce.
    let list;
    if (q && rel) {
      list = state.allSources.filter(
        (s) => (rel[s.key] || 0) > 0 || labelHit(s) || state.sources.has(s.key),
      );
    } else {
      list = state.allSources.filter((s) => !q || labelHit(s));
    }
    /* Score a chip's lexical affinity to the query. The higher
     * the score, the closer the chip's name is to what the user
     * typed — so it should sort above ColBERT-relevance hits.
     *
     * Both sides are *normalised* by stripping every non-alphanumeric
     * character before comparing. That way:
     *   "max halford"   ↔ "maxhalford.github.io"   ✓ (prefix)
     *   "lighton.ai"    ↔ "lighton.ai"              ✓ (exact)
     *   "git hub"       ↔ "github"                  ✓ (exact)
     *   "ourworld data" ↔ "ourworldindata.org"     ✓ (substring)
     * The dots / spaces / hyphens that vary between human input
     * and source-key formatting stop mattering.
     *
     * Score levels (relative to the normalised forms):
     *   3 — exact equality
     *   2 — chip starts with the query
     *   1 — chip contains the query anywhere
     *   0 — no hit (relevance pass takes over)
     */
    const lexScore = (s) => {
      if (!nq) return 0;
      const nLabel = norm(s.label);
      const nKey = norm(s.key);
      if (nLabel === nq || nKey === nq) return 3;
      if (nLabel.startsWith(nq) || nKey.startsWith(nq)) return 2;
      if (nLabel.includes(nq) || nKey.includes(nq)) return 1;
      return 0;
    };
    list.sort((a, b) => {
      // 0. Favorites is always the first chip — synthetic personal
      //    source, the user's primary "saved" bucket.
      if (a.key === FAV_SOURCE_KEY && b.key !== FAV_SOURCE_KEY) return -1;
      if (b.key === FAV_SOURCE_KEY && a.key !== FAV_SOURCE_KEY) return 1;
      // 1. Selected chips always pinned to the top.
      const aOn = state.sources.has(a.key) ? 0 : 1;
      const bOn = state.sources.has(b.key) ? 0 : 1;
      if (aOn !== bOn) return aOn - bOn;
      // 2. Lexical affinity to the typed query — a chip whose label
      //    *is* (or starts with) what the user typed always beats a
      //    higher-scored ColBERT match. Saves the user from chasing
      //    semantically similar chips when they meant a specific one.
      const lxA = lexScore(a);
      const lxB = lexScore(b);
      if (lxA !== lxB) return lxB - lxA;
      // 3. Order by FIRST appearance in the ColBERT result set.
      //    A source whose top hit ranks first wins over one whose
      //    top hit ranks later — no aggregation, no normalisation.
      //    Sources with no hit at all fall through to step 4.
      const firstRank = state.sourceFirstRank;
      if (q && firstRank) {
        const ra = firstRank[a.key];
        const rb = firstRank[b.key];
        if (ra != null && rb != null) {
          if (ra !== rb) return ra - rb;
        } else if (ra != null) {
          return -1;
        } else if (rb != null) {
          return 1;
        }
      }
      // 4. Preserve original order (count desc from rebuildAllSources).
      return 0;
    });
    const fullCount = list.length;
    // Paginate: 50 chips initially, grow by 50 on scroll-to-bottom.
    // Selected chips are pinned at the top of the sorted list already,
    // so the slice will never drop them.
    list = list.slice(0, _srcRail.page);
    let lastSelectedIdx = -1;
    for (let i = 0; i < list.length; i++)
      if (state.sources.has(list[i].key)) lastSelectedIdx = i;
    $("srcList").innerHTML = list
      .map((s, i) => {
        const on = state.sources.has(s.key);
        const cls = ["opt"];
        if (on) cls.push("on");
        if (on && i === lastSelectedIdx && lastSelectedIdx < list.length - 1)
          cls.push("last-selected");
        const icon = K.sourceIconUrl(s.key);
        // Fallback glyph: a small dot in a square so unknown source
        // categories don't render as a broken-image gap. The synthetic
        // Favorites entry gets a filled star — distinct from any
        // hostname favicon so the chip reads as "your starred set".
        let iconHtml;
        if (s.key === FAV_SOURCE_KEY) {
          iconHtml = `<span class="fav fav-star" aria-hidden="true"><svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true"><path d="M12 4 L20 13 L15.5 13 L15.5 20 L8.5 20 L8.5 13 L4 13 Z"/></svg></span>`;
        } else if (icon) {
          iconHtml = `<img class="fav" src="${icon}" alt="" onerror="this.replaceWith(document.querySelector('#srcFav-fallback').content.cloneNode(true))"/>`;
        } else {
          iconHtml = `<span class="fav fav-fallback" aria-hidden="true">●</span>`;
        }
        // In relevance mode, show the per-source hit count from
        // ColBERT (helps the user see which chips are most relevant).
        // Otherwise show the static total — unchanged behaviour.
        // EXCEPTION: in big-selection mode the rail is built from
        // the result set without counts, so we omit the column
        // entirely. The chip still shows the icon + label and stays
        // searchable.
        const _allOnly = useAllOnly([...state.libs]);
        // Feed (libs.size === 0): hide the count column entirely. The
        // aggregated count is still used to rank the chips (see the
        // sort comparator above), it's just visual clutter at that
        // scale. Personal page (libs.size === 1) keeps the count so
        // the user can see how many docs each source contributes.
        const _onFeed = state.libs.size === 0;
        const displayCount =
          q && rel && rel[s.key] != null ? rel[s.key] : s.count || 0;
        const countHtml =
          _allOnly || _onFeed
            ? ""
            : `<span class="count">${displayCount}</span>`;
        return `<label class="${cls.join(" ")}">
        <input type="checkbox" data-src="${escapeAttr(s.key)}" ${on ? "checked" : ""}/>
        ${iconHtml}
        <span class="label">${escapeHtml(s.label || s.key)}</span>
        ${countHtml}
        <span class="check">✓</span>
      </label>`;
      })
      .join("");
    // Sentinel for infinite scroll inside the rail. Disconnected when
    // we've already rendered every available chip.
    const hasMore = fullCount > list.length;
    if (hasMore) {
      $("srcList").insertAdjacentHTML(
        "beforeend",
        `<div class="src-rail-sentinel" aria-hidden="true"></div>`,
      );
      armSrcRailScroll();
    } else if (_srcRail.observer) {
      _srcRail.observer.disconnect();
      _srcRail.observer = null;
    }
    $("clearSrc").style.display =
      state.sources.size || state.excludedSources.size
        ? "inline-block"
        : "none";
  }
  function armSrcRailScroll() {
    const sentinel = document.querySelector(".src-rail-sentinel");
    if (!sentinel) return;
    if (_srcRail.observer) _srcRail.observer.disconnect();
    // Scroll root = the rail's scrollable body. `.group-body` wraps
    // `#srcList` and is the element with overflow-y.
    const root =
      document.querySelector("#grpSrc .group-body") ||
      document.querySelector(".rail");
    _srcRail.observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            _srcRail.page += SRC_RAIL_PAGE;
            renderSrc();
            return;
          }
        }
      },
      { root, rootMargin: "120px 0px" },
    );
    _srcRail.observer.observe(sentinel);
  }
  $("srcFilter").addEventListener("input", () => {
    const q = $("srcFilter").value.trim();
    clearTimeout(srcRelevanceDebT);
    // Filter typing → start from the top page so the user always
    // sees the most relevant matches first.
    _srcRail.page = SRC_RAIL_PAGE;
    if (!q) {
      // Empty input — fall back to the static ordering, no search.
      state.sourceRelevance = null;
      state.sourceFirstRank = null;
      srcRelevanceQueryId++; // invalidate any in-flight relevance
      renderSrc();
      return;
    }
    // Render the substring-only fallback now (so the panel reacts
    // to the keystroke immediately), then debounce the ColBERT
    // search by 250 ms — typing fast collapses to one network call.
    renderSrc();
    srcRelevanceDebT = setTimeout(() => runSourceRelevance(q), 250);
  });
  $("srcList").addEventListener("change", (e) => {
    const t = e.target;
    if (!t.matches("[data-src]")) return;
    const key = t.dataset.src;
    if (t.checked) {
      // Including a source clears any prior exclusion on it — the
      // tri-state chip can never be on AND off at the same time.
      state.excludedSources.delete(key);
      state.sources.add(key);
    } else {
      state.sources.delete(key);
    }
    renderSrc();
    writeUrl();
    refresh();
    // Telemetry: record the new active source-filter string so the
    // recommender can learn per-user preference distributions.
    if (window.kn) {
      const sf = [...state.sources].sort().join(",") || null;
      window.kn.setLastFilter({ source: sf || "" });
      window.kn.track("filter_apply", { source_filter: sf });
    }
  });
  $("clearSrc").addEventListener("click", () => {
    state.sources.clear();
    state.excludedSources.clear();
    renderSrc();
    writeUrl();
    refresh();
  });
  let debT = null;
  $("q").addEventListener("input", () => {
    clearTimeout(debT);
    debT = setTimeout(() => {
      const p = parseQ($("q").value);
      state.query = p.plain;
      // state.tags + the Favorites chip are independent of the search
      // input now — driven by chip toggles + URL params. Don't let
      // typing wipe them.
      if (p.sites.size) {
        for (const s of p.sites) state.sources.add(s);
        writeQ();
      }
      rebuildAllSources();
      renderSrc();
      writeUrl();
      // The "Following only" toggle's visibility depends on whether
      // there's an active query — sync after each keystroke.
      syncFollowingOnlyButton();
      refresh();
    }, 220);
  });
  $("qClear").addEventListener("click", () => {
    $("q").value = "";
    $("q").focus();
    $("q").dispatchEvent(new Event("input"));
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      $("q").focus();
      $("q").select();
    }
  });
  // Sort toggle was removed — Relevance is implicit while a query is
  // active, browse-mode is always date-desc. The relevant filter lives
  // on the date-range <select> below.
  /* "Following only" toggle — restricts feed search to followees+self.
   * Persists in the URL as `&scope=following` so a shared link
   * reproduces the same view. Only relevant when a search is running
   * on the feed (libs.size === 0) — otherwise the button is hidden.
   * Anonymous users see it but a click pops the login modal. */
  function syncFollowingOnlyButton() {
    const btn = $("qFollowingOnly");
    if (!btn) return;
    const onFeed = state.libs.size === 0;
    const searching = !!state.query;
    btn.hidden = !(onFeed && searching);
    btn.classList.toggle("is-on", !!state.followingOnly);
    btn.setAttribute("aria-pressed", state.followingOnly ? "true" : "false");
  }
  /* "Show seen" chip — only meaningful on the logged-in timeline
   * (libs empty, no query). Visibility / enabled state split into
   * two attributes so desktop and mobile can style them differently:
   *
   *   • `hidden` — never set for logged-in users. Anonymous viewers
   *     get the chip hidden entirely so it doesn't tease a feature
   *     that requires an account.
   *   • `disabled` — set whenever the active state isn't the empty
   *     feed (a search is running, or a personal/library page is
   *     active). Desktop CSS collapses the disabled state to
   *     `display:none` so the chip vanishes; mobile CSS keeps it in
   *     place but grey + non-interactive, so the bottom row layout
   *     stays consistent as the user moves between surfaces. */
  function syncShowSeenButton() {
    // Two surfaces, one source of truth:
    //   • Desktop chip (#qShowSeen) — sits in the spotlight row.
    //   • Mobile bottom-nav action (#mbnShowSeen) — sits next to
    //     Filter/Post on the floating capsule.
    // Both follow the same visibility rules:
    //   • logged-out                         → hidden
    //   • on a personal/library page        → hidden (no eye at all)
    //   • on the feed with a search running → desktop disabled,
    //     mobile bottom-nav stays hidden (the eye lives next to the
    //     feed surface, not the search results)
    //   • on the empty feed                  → enabled
    const onPersonalPage = state.libs.size > 0;
    const searching = !!state.query;
    const onEmptyFeed = !onPersonalPage && !searching;

    const chip = $("qShowSeen");
    if (chip) {
      chip.hidden = !me || onPersonalPage;
      if (searching) chip.setAttribute("disabled", "");
      else chip.removeAttribute("disabled");
      chip.classList.toggle("is-on", !!state.showSeen);
      chip.setAttribute("aria-pressed", state.showSeen ? "true" : "false");
      const lbl = chip.querySelector("span");
      if (lbl) lbl.textContent = state.showSeen ? "Showing seen" : "Show seen";
    }

    const mobile = $("mbnShowSeen");
    if (mobile) {
      // The mobile button only ever shows on the empty feed for
      // logged-in users; off-feed there is no slot to dim into.
      mobile.hidden = !me || !onEmptyFeed;
      mobile.classList.toggle("is-on", !!state.showSeen);
      mobile.setAttribute("aria-pressed", state.showSeen ? "true" : "false");
    }
  }
  /* No-op kept as a stub so legacy call sites (refresh, append,
   * timeline render) don't blow up. The shuffle button was
   * removed — pull-to-refresh handles the same job now. */
  function syncShuffleButton() {}
  /* Date-filter chip mirror — the mobile build paints the disc
   * with an ink fill when a non-empty range is active so the
   * "on" state is unmistakable. */
  function syncSinceFilterActive() {
    const wrap = document.querySelector(".since-filter");
    if (!wrap) return;
    wrap.classList.toggle("has-filter", !!state.dateSince);
  }
  syncFollowingOnlyButton();
  syncShowSeenButton();
  syncSinceFilterActive();
  $("qFollowingOnly")?.addEventListener("click", () => {
    // Anonymous → invite them to sign in. Otherwise toggle the
    // scope filter and re-run the search.
    if (!me) {
      window.KnowledgeAuth?.open("login");
      return;
    }
    state.followingOnly = !state.followingOnly;
    syncFollowingOnlyButton();
    writeUrl();
    refresh();
  });
  function toggleShowSeen() {
    if (!me) {
      window.KnowledgeAuth?.open("login");
      return;
    }
    state.showSeen = !state.showSeen;
    try {
      localStorage.setItem("feed.showSeen", state.showSeen ? "1" : "0");
    } catch {}
    // Clear the in-tab dedup so flipping ON immediately re-fires
    // observers for cards that were previously suppressed.
    if (window.kn && typeof window.kn.resetSeenSuppression === "function") {
      window.kn.resetSeenSuppression();
    }
    syncShowSeenButton();
    // The timeline URL embeds `include_seen`, so flipping the chip
    // changes the cache key — next refresh() naturally re-fetches.
    writeUrl();
    refresh();
  }
  $("qShowSeen")?.addEventListener("click", toggleShowSeen);
  $("mbnShowSeen")?.addEventListener("click", toggleShowSeen);
  // Run a single sync after qSince changes — declared below.
  $("qSince")?.addEventListener("change", (e) => {
    state.dateSince = e.target.value || "";
    syncSinceFilterActive();
    writeUrl();
    refresh();
  });

  /* ── Category picker ───────────────────────────────────────────
   * Wires the spotlight-row button (see #catPickerBtn / #catPickerPanel
   * in search.html) to state.category. Behaviour:
   *
   *  - Lazy fetch the 178-row catalogue from /api/document-categories
   *    on the first popover open. Cache it on the closure for the
   *    rest of the session (it never changes per-request).
   *  - Render the list grouped by `group` with collapsible-looking
   *    section headers — actually one flat scroll, headers are
   *    sticky so they read as topic dividers.
   *  - Type-to-filter via the search box at the top of the panel:
   *    matches against slug + name + description, case-insensitive.
   *  - Selecting a row sets state.category, closes the popover,
   *    syncs the URL, and refreshes the feed. Clicking the row
   *    that's already current clears the filter.
   *  - Outside-click and Escape both close the popover without
   *    changing the filter. */
  const catPickerWrap = $("catPickerWrap");
  const catPickerBtn = $("catPickerBtn");
  const catPickerPanel = $("catPickerPanel");
  const catPickerLabel = $("catPickerLabel");
  const catPickerList = $("catPickerList");
  const catPickerSearch = $("catPickerSearch");
  // Mobile-only sister of the desktop popover. Same data, same
  // selection semantics — but lives inside a Cupertino Pane sheet
  // so it feels like the People drawer on phone. Hidden on desktop.
  const categoryRail = $("categoryRail");
  const categoryRailList = $("categoryRailList");
  const categoryRailFilter = $("categoryRailFilter");
  // The catalogue is keyed by the "active user" — `""` for the
  // global feed / multi-lib view, or the single lib slug when the
  // user is on a personal page (state.libs.size === 1). Server
  // returns a different subset in each case so the cache key has
  // to differ. _catalogueCache always points at the array for the
  // currently-active user so the synchronous render path stays
  // unchanged; `activeCatalogueUser()` decides which one that is.
  const _catalogueByUser = new Map(); // user slug or "" → array
  let _catalogueCache = null;
  let _catalogueLoading = null; // per-user promise; cleared on resolve
  let _catalogueLoadingUser = ""; // which slug the in-flight fetch is for

  /* Recents — localStorage-backed list of category slugs the user
   * has selected, most-recent first. When the picker renders the
   * unfiltered catalogue (no search query), these slugs are pulled
   * to the top of the list ahead of the grouped catalogue so the
   * user's habitual filters live at thumb height. Cap kept small
   * (12) because the recents area sits inline with the catalogue;
   * a longer list crowds the rest of the picker. */
  const CAT_RECENTS_KEY = "kn.cat_recents";
  const CAT_RECENTS_MAX = 12;
  function loadCatRecents() {
    try {
      const raw = localStorage.getItem(CAT_RECENTS_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr)
        ? arr.filter((s) => typeof s === "string").slice(0, CAT_RECENTS_MAX)
        : [];
    } catch {
      return [];
    }
  }
  let _catRecents = loadCatRecents();
  function pushCatRecent(slug) {
    if (!slug) return;
    _catRecents = [slug, ..._catRecents.filter((s) => s !== slug)].slice(
      0,
      CAT_RECENTS_MAX,
    );
    try {
      localStorage.setItem(CAT_RECENTS_KEY, JSON.stringify(_catRecents));
    } catch {
      /* private-mode storage block — drop silently */
    }
  }
  // Persisted multi-select. Mirrors `state.categories` to localStorage
  // so the active filter survives page reloads, tab restarts, and
  // navigation across feed / personal / search routes. URL params
  // still take precedence at boot (deep links), but any URL with a
  // category set rewrites this store so subsequent navigations carry
  // the selection forward.
  const CAT_SELECTED_LS_KEY = "kn.cat_selected";
  function loadCatSelected() {
    try {
      const raw = localStorage.getItem(CAT_SELECTED_LS_KEY);
      if (!raw) return new Set();
      const arr = JSON.parse(raw);
      return new Set(
        Array.isArray(arr)
          ? arr
              .filter((s) => typeof s === "string" && s.length > 0)
              .map((s) => s.toLowerCase())
          : [],
      );
    } catch {
      return new Set();
    }
  }
  function saveCatSelected(set) {
    try {
      const arr = set && set.size ? [...set] : [];
      if (arr.length === 0) {
        localStorage.removeItem(CAT_SELECTED_LS_KEY);
      } else {
        localStorage.setItem(CAT_SELECTED_LS_KEY, JSON.stringify(arr));
      }
    } catch {
      /* ignore */
    }
  }
  // `.spotlight-row` (the parent bar) ships a backdrop-filter so the
  // search-bar acrylic blur reads through behind the input. A side
  // effect of that property: it makes the bar the containing block
  // for any `position: fixed` descendant, so our mobile rule
  // (`top: auto; bottom: 12px;`) anchors the panel to the bar
  // instead of the viewport — and the panel ends up floating
  // ~440 px above where it should. Portal the panel out to
  // `document.body` once so it lives outside the acrylic-blur
  // subtree and `position: fixed` behaves as expected on every
  // screen size.
  if (catPickerPanel && catPickerPanel.parentElement !== document.body) {
    document.body.appendChild(catPickerPanel);
  }

  /* Which user's catalogue we should be showing right now. The feed
   * and multi-library views get the full 178-row catalogue (empty
   * key). A single-library view (a personal page) gets only the
   * categories that have at least one assignment for that lib, so
   * the picker doesn't show 178 rows when 30 actually carve up the
   * library. */
  function activeCatalogueUser() {
    const libs = state.libs;
    if (!libs || libs.size !== 1) return "";
    return [...libs][0] || "";
  }

  /* Fetch the catalogue scoped to `userSlug` (empty string = global)
   * and cache it. Re-uses the in-memory promise when a fetch is in
   * flight for the same key, so a rapid keystroke + boot race
   * doesn't double-hit the API. Returns the cached array
   * synchronously on every subsequent call. */
  async function fetchCatalogue(userSlug) {
    const key = typeof userSlug === "string" ? userSlug : activeCatalogueUser();
    if (_catalogueByUser.has(key)) {
      _catalogueCache = _catalogueByUser.get(key);
      return _catalogueCache;
    }
    if (_catalogueLoading && _catalogueLoadingUser === key) {
      return _catalogueLoading;
    }
    _catalogueLoadingUser = key;
    _catalogueLoading = (async () => {
      try {
        const url = key
          ? `${API_BASE}/api/document-categories?user=${encodeURIComponent(key)}`
          : `${API_BASE}/api/document-categories`;
        const r = await fetch(url, { credentials: "include" });
        if (!r.ok) return [];
        const out = await r.json();
        return Array.isArray(out) ? out : [];
      } catch {
        return [];
      }
    })();
    const arr = await _catalogueLoading;
    _catalogueByUser.set(key, arr);
    _catalogueCache = arr;
    _catalogueLoading = null;
    return arr;
  }

  /* Called whenever state.libs changes (URL hydration, navigation,
   * rail tweaks). Re-fetches the catalogue for the new active user
   * and re-renders the picker if the user-scoped subset differs
   * from what's currently on screen. */
  async function refreshCatalogueForActiveUser() {
    const key = activeCatalogueUser();
    if (_catalogueByUser.has(key)) {
      // Already cached — flip the pointer and re-render synchronously.
      const cached = _catalogueByUser.get(key);
      if (cached !== _catalogueCache) {
        _catalogueCache = cached;
        renderCatPickerList(
          (catPickerSearch?.value || categoryRailFilter?.value || "").trim(),
        );
        syncCatPickerLabel();
      }
      return;
    }
    await fetchCatalogue(key);
    renderCatPickerList(
      (catPickerSearch?.value || categoryRailFilter?.value || "").trim(),
    );
    syncCatPickerLabel();
  }

  function syncCatPickerLabel() {
    const slugs = [...(state.categories || [])];
    // Show / hide the rail-level "Clear selection" link based on
    // whether anything is picked. Cheap and idempotent — runs on
    // every selection toggle.
    const clearBtn = document.getElementById("catRailClear");
    if (clearBtn) clearBtn.hidden = slugs.length === 0;
    // Selection-count badges — desktop spotlight button + the
    // right-rail Topics tab. Hidden when nothing is selected so the
    // chrome stays quiet for users who never touch the filter.
    const countBadge = document.getElementById("catPickerCount");
    if (countBadge) {
      if (slugs.length > 0) {
        countBadge.textContent = String(slugs.length);
        countBadge.hidden = false;
      } else {
        countBadge.hidden = true;
      }
    }
    const railTabCount = document.getElementById("rrtTabCount");
    if (railTabCount) {
      if (slugs.length > 0) {
        railTabCount.textContent = String(slugs.length);
        railTabCount.hidden = false;
      } else {
        railTabCount.hidden = true;
      }
    }
    if (!catPickerLabel || !catPickerWrap) return;
    if (slugs.length === 0) {
      catPickerLabel.textContent = "All categories";
      catPickerWrap.classList.remove("is-on");
      return;
    }
    catPickerWrap.classList.add("is-on");
    // Single pick → show the human name. Multi-pick → show the first
    // name + a count of the rest. Falls back to the raw slug until the
    // catalogue cache lands.
    const cats = _catalogueCache || [];
    const findName = (s) => {
      const c = cats.find((x) => x.slug === s);
      return c ? c.name : s;
    };
    if (slugs.length === 1) {
      catPickerLabel.textContent = findName(slugs[0]);
    } else {
      catPickerLabel.textContent = `${findName(slugs[0])} +${slugs.length - 1}`;
    }
  }

  /* ── ColBERT-augmented category search ───────────────────────
   * Mirrors the people-rail behaviour. When the picker's search box
   * holds a query of ≥ 3 chars we send it to
   * `/indices/__all__/search_with_encoding` (the cross-library
   * ColBERT index). The top documents come back as URL+metadata; we
   * then ask `/api/document-categories/by-url` for each URL's slug
   * list and score categories by Σ 1/(rank+1) — categories
   * appearing in the most highly-ranked hits win. The top N
   * slugs become a "Best matches" section pinned to the top of the
   * picker list. Lexical filtering runs in parallel below it.
   *
   * Empty query / fewer than 3 chars → no ColBERT call, no
   * augmentation, the full grouped catalogue is rendered. */
  let _catColbertCtrl = null;
  let _catColbertTimer = null;
  let _catColbertSlugs = []; // [{slug, score}] — promoted to top
  let _catColbertForQuery = ""; // query the current augmentation is for

  async function _catColbertFetch(q) {
    if (_catColbertTimer) clearTimeout(_catColbertTimer);
    if (_catColbertCtrl) _catColbertCtrl.abort();
    _catColbertSlugs = [];
    if (!q || q.length < 3) {
      _catColbertForQuery = q;
      return;
    }
    _catColbertCtrl = new AbortController();
    const signal = _catColbertCtrl.signal;
    _catColbertForQuery = q;
    _catColbertTimer = setTimeout(async () => {
      try {
        // Stage 1 — ColBERT over the cross-library index.
        const r = await fetch(
          `${API_BASE}/indices/${ALL_INDEX_NAME}/search_with_encoding`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              queries: [q],
              params: { top_k: 200 },
            }),
            signal,
          },
        );
        if (!r.ok) return;
        const data = await r.json();
        const meta = (data.results && data.results[0]?.metadata) || [];
        const urls = [];
        const rankByUrl = new Map();
        for (let i = 0; i < meta.length && urls.length < 200; i++) {
          const u = meta[i]?.url;
          if (u && !rankByUrl.has(u)) {
            rankByUrl.set(u, i);
            urls.push(u);
          }
        }
        if (urls.length === 0) return;
        // Stage 2 — fetch the categories assigned to each hit URL.
        const r2 = await fetch(`${API_BASE}/api/document-categories/by-url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls }),
          signal,
        });
        if (!r2.ok) return;
        const map = await r2.json();
        // Stage 3 — Σ 1/(rank+1) per slug. The reciprocal-rank
        // weighting matches the people-rail's "first-appearance"
        // ranking in spirit but is smoother for tail categories
        // that surface 4-5 times across mid-rank hits.
        const score = new Map();
        for (const [u, slugs] of Object.entries(map)) {
          const rank = rankByUrl.get(u);
          if (rank === undefined) continue;
          const w = 1.0 / (rank + 1);
          for (const s of slugs || []) {
            score.set(s, (score.get(s) || 0) + w);
          }
        }
        const ranked = [...score.entries()]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([slug, s]) => ({ slug, score: s }));
        _catColbertSlugs = ranked;
        if (_catColbertForQuery === q) renderCatPickerList(q);
      } catch {
        // aborted / network blip — keep the lexical list visible
      }
    }, 250);
  }

  /* Render the picker list — works for both surfaces (desktop
   * popover + mobile category rail) by writing to whichever
   * targets exist. The two share state.category, the catalogue
   * cache, and the ColBERT augmentation set so a selection on
   * either surface stays in sync. */
  function renderCatPickerList(filter = "") {
    const targets = [catPickerList, categoryRailList].filter(Boolean);
    if (targets.length === 0) return;
    const cats = _catalogueCache || [];
    const bySlug = new Map(cats.map((c) => [c.slug, c]));
    const needle = filter.trim().toLowerCase();
    const hasSearch = needle.length > 0;
    // Lexical filter (instant, no network).
    const lexical = hasSearch
      ? cats.filter(
          (c) =>
            c.slug.toLowerCase().includes(needle) ||
            (c.name || "").toLowerCase().includes(needle) ||
            (c.description || "").toLowerCase().includes(needle) ||
            (c.group || "").toLowerCase().includes(needle),
        )
      : cats;
    // ColBERT augmentation — promoted slugs that aren't already
    // covered by the lexical match get surfaced as a top section.
    // Stays empty until the debounced ColBERT round-trip lands.
    const lexSlugSet = new Set(lexical.map((c) => c.slug));
    const colbertHits = hasSearch
      ? _catColbertSlugs.map(({ slug }) => bySlug.get(slug)).filter(Boolean)
      : [];
    const selected = state.categories || new Set();
    const parts = [];
    parts.push(
      `<button type="button" class="cat-picker-clear${
        selected.size === 0 ? " is-current" : ""
      }" data-slug="" data-action="clear">All categories</button>`,
    );
    if (hasSearch && colbertHits.length) {
      parts.push(`<div class="cat-picker-group cat-picker-group-colbert">`);
      parts.push(`<div class="cat-picker-group-head">Best matches</div>`);
      for (const c of colbertHits) {
        parts.push(renderCatItem(c, selected));
      }
      parts.push(`</div>`);
    }
    if (hasSearch && lexical.length === 0 && colbertHits.length === 0) {
      parts.push(
        `<div class="cat-picker-empty">No category matches “${escapeHtml(
          filter,
        )}”.</div>`,
      );
    } else if (hasSearch) {
      // Flat lexical list when searching — grouping headers would
      // crowd a short results list.
      parts.push(`<div class="cat-picker-group cat-picker-group-lexical">`);
      if (colbertHits.length) {
        parts.push(`<div class="cat-picker-group-head">Other matches</div>`);
      }
      for (const c of lexical) {
        // Skip duplicates that already appear in the ColBERT
        // section so the user doesn't see the same row twice.
        if (colbertHits.some((h) => h.slug === c.slug)) continue;
        parts.push(renderCatItem(c, selected));
      }
      parts.push(`</div>`);
    } else {
      // No query — three layers, from top to bottom:
      //   1. SELECTED categories — pinned at the top with their own
      //      header. Critical on phone where the user just toggled
      //      a slug deep in the list and would otherwise lose track
      //      of what they had picked. Selected items still appear
      //      again in their natural group below (user explicitly
      //      asked to keep them in the list).
      //   2. RECENTS — most-recently-selected first, no group
      //      wrapper. Drops anything already shown in the SELECTED
      //      section so the user doesn't see the same row twice in
      //      a row (one above the other).
      //   3. Grouped catalogue, minus recents.
      const selectedCats = [];
      const selectedSet = new Set();
      for (const slug of selected) {
        const c = bySlug.get(slug);
        if (!c || selectedSet.has(slug)) continue;
        selectedCats.push(c);
        selectedSet.add(slug);
      }
      if (selectedCats.length) {
        parts.push(`<div class="cat-picker-group cat-picker-group-selected">`);
        parts.push(
          `<div class="cat-picker-group-head">Selected (${selectedCats.length})</div>`,
        );
        for (const c of selectedCats) {
          parts.push(renderCatItem(c, selected));
        }
        parts.push(`</div>`);
      }
      const recentsCats = [];
      const recentSet = new Set();
      for (const slug of _catRecents) {
        // Skip slugs already shown in the SELECTED section above.
        if (selectedSet.has(slug)) continue;
        const c = bySlug.get(slug);
        if (!c || recentSet.has(slug)) continue;
        recentsCats.push(c);
        recentSet.add(slug);
      }
      if (recentsCats.length) {
        // No group wrapper — recents render as bare items at the
        // top of the list so they read as a continuation of the
        // catalogue, not a separate section.
        for (const c of recentsCats) {
          parts.push(renderCatItem(c, selected));
        }
      }
      // Grouped catalogue. Recents are filtered out (already shown
      // above without a header); SELECTED rows ARE allowed through
      // — the user asked for that explicitly so the catalogue stays
      // navigable.
      const byGroup = new Map();
      for (const c of lexical) {
        if (recentSet.has(c.slug)) continue;
        if (!byGroup.has(c.group)) byGroup.set(c.group, []);
        byGroup.get(c.group).push(c);
      }
      for (const [group, list] of byGroup) {
        parts.push(`<div class="cat-picker-group">`);
        parts.push(
          `<div class="cat-picker-group-head">${escapeHtml(group)}</div>`,
        );
        for (const c of list) {
          parts.push(renderCatItem(c, selected));
        }
        parts.push(`</div>`);
      }
    }
    const html = parts.join("");
    for (const target of targets) target.innerHTML = html;
  }

  function renderCatItem(c, selected) {
    const on = selected && selected.has(c.slug);
    return (
      `<button type="button" class="cat-picker-item${
        on ? " is-current" : ""
      }" data-slug="${escapeAttr(c.slug)}" aria-pressed="${
        on ? "true" : "false"
      }" title="${escapeAttr(c.description || "")}">` +
      `<span class="cat-picker-item-name">${escapeHtml(c.name)}</span>` +
      `<span class="cat-picker-item-desc">${escapeHtml(
        c.description || "",
      )}</span>` +
      `</button>`
    );
  }

  /* Position the panel relative to the button. On phones (viewport
   * ≤ 768 px) sit it full-width near the bottom of the viewport so
   * it's thumb-friendly. On desktop anchor it 8 px below the button
   * and right-align to the button's right edge, clamping inside the
   * viewport so the panel never spills off the left or right when
   * the button is near a screen edge. */
  function positionCatPickerPanel() {
    if (!catPickerPanel || !catPickerBtn) return;
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    const btnRect = catPickerBtn.getBoundingClientRect();
    if (isMobile) {
      const margin = 12;
      catPickerPanel.style.left = margin + "px";
      catPickerPanel.style.right = margin + "px";
      catPickerPanel.style.top = "auto";
      catPickerPanel.style.bottom = margin + "px";
    } else {
      // Measure once it's rendered (display:flex, visibility:hidden
      // trick isn't needed because the panel was just unhidden).
      const panelW = catPickerPanel.getBoundingClientRect().width;
      const margin = 8;
      // Align the panel's right edge to the button's right edge,
      // clamped inside the viewport with a 12 px gutter.
      let left = btnRect.right - panelW;
      const gutter = 12;
      if (left < gutter) left = gutter;
      if (left + panelW > window.innerWidth - gutter)
        left = window.innerWidth - gutter - panelW;
      catPickerPanel.style.left = left + "px";
      catPickerPanel.style.right = "auto";
      catPickerPanel.style.top = btnRect.bottom + margin + "px";
      catPickerPanel.style.bottom = "auto";
    }
  }

  function openCatPicker() {
    if (!catPickerPanel || !catPickerBtn) return;
    catPickerPanel.hidden = false;
    catPickerBtn.setAttribute("aria-expanded", "true");
    // Reset the search box on every open so the operator starts from
    // the full list — re-typing is cheaper than remembering what was
    // in the box last time.
    if (catPickerSearch) catPickerSearch.value = "";
    positionCatPickerPanel();
    fetchCatalogue().then(() => {
      renderCatPickerList("");
      syncCatPickerLabel();
      // Re-position once the list has grown — desktop panels right-
      // align to the button, and the panel width can shift slightly
      // once the items are inserted (scrollbar gutter, etc.).
      positionCatPickerPanel();
      // Defer focus to next tick so the popover animation doesn't
      // steal the keyboard mid-paint.
      setTimeout(() => catPickerSearch?.focus(), 0);
    });
  }
  function closeCatPicker() {
    if (!catPickerPanel || !catPickerBtn) return;
    catPickerPanel.hidden = true;
    catPickerBtn.setAttribute("aria-expanded", "false");
  }
  function toggleCatPicker() {
    if (catPickerPanel?.hidden === false) closeCatPicker();
    else openCatPicker();
  }

  catPickerBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleCatPicker();
  });
  // Shared input handler: fires lexical render immediately + kicks
  // off the debounced ColBERT augmentation. Used by both the
  // desktop popover input and the mobile rail input.
  function onCatSearchInput(q) {
    renderCatPickerList(q);
    _catColbertFetch(q);
  }
  catPickerSearch?.addEventListener("input", (e) => {
    onCatSearchInput(e.target.value || "");
  });
  categoryRailFilter?.addEventListener("input", (e) => {
    onCatSearchInput(e.target.value || "");
  });
  // Shared list click handler — multi-select toggle. Clicking a row
  // adds/removes its slug from state.categories. The picker stays
  // open so the user can stack several filters without re-opening
  // it for every one. Tapping "All categories" wipes the set in
  // one shot.
  function onCatListClick(e) {
    const tgt = e.target.closest("button[data-slug]");
    if (!tgt) return;
    const slug = tgt.getAttribute("data-slug") || "";
    const action = tgt.getAttribute("data-action") || "";
    if (action === "clear" || !slug) {
      // The "All categories" reset row — drop every selection.
      if (state.categories.size === 0) return;
      state.categories.clear();
    } else if (state.categories.has(slug)) {
      state.categories.delete(slug);
    } else {
      state.categories.add(slug);
      // Bump the slug to the top of the localStorage recents list
      // so the next picker open surfaces it ahead of the grouped
      // catalogue. Removal doesn't touch recents — the user might
      // be toggling a slug off temporarily and we want it to stay
      // close at hand for the next reselect.
      pushCatRecent(slug);
    }
    // Persist the new selection so it survives page reloads and
    // navigation across feed / personal / search routes.
    saveCatSelected(state.categories);
    syncCatPickerLabel();
    // Telemetry: each topic toggle is a "folder_browse" — feeds the
    // recommender a per-user topic-preference signal.
    if (window.kn) {
      window.kn.track("folder_browse", {
        source_filter: [...state.categories].sort().join(",") || null,
      });
    }
    // Re-render the list in-place so check marks update without
    // the user closing the picker. Reuses the current search box
    // contents so the visible filter stays the same.
    const q = (
      catPickerSearch?.value ||
      categoryRailFilter?.value ||
      ""
    ).trim();
    renderCatPickerList(q);
    writeUrl();
    refresh();
    // Bottom-nav: the Topics tab's active state tracks
    // state.categories.size, so re-sync immediately on every
    // toggle (don't wait for the sheet's dismiss animation).
    if (typeof window._syncMobileChrome === "function") {
      window._syncMobileChrome();
    }
  }
  catPickerList?.addEventListener("click", onCatListClick);
  categoryRailList?.addEventListener("click", onCatListClick);
  // The rail-level "Clear selection" button shares the same data
  // attributes as the in-list reset row, so the same handler can
  // service it.
  document
    .getElementById("catRailClear")
    ?.addEventListener("click", onCatListClick);
  // Outside-click + Escape both close the popover without changing
  // anything. `mousedown` (not `click`) so an in-panel click that
  // doesn't bubble can still close-on-second-tap reliably. The panel
  // is portaled to <body>, so a click "inside the picker" can land
  // on either the button wrap OR the panel itself — both must
  // exclude an outside-click verdict.
  document.addEventListener("mousedown", (e) => {
    if (!catPickerPanel || catPickerPanel.hidden) return;
    if (catPickerWrap?.contains(e.target)) return;
    if (catPickerPanel.contains(e.target)) return;
    closeCatPicker();
  });
  // Re-anchor the panel if the viewport changes shape while it's
  // open (orientation, soft-keyboard, resize, scroll). Cheap — only
  // touches inline styles, no layout-thrash.
  window.addEventListener("resize", () => {
    if (catPickerPanel && !catPickerPanel.hidden) positionCatPickerPanel();
  });
  window.addEventListener(
    "scroll",
    () => {
      if (catPickerPanel && !catPickerPanel.hidden) positionCatPickerPanel();
    },
    { passive: true },
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && catPickerPanel && !catPickerPanel.hidden) {
      closeCatPicker();
      catPickerBtn?.focus();
    }
  });
  // Boot — sync the button label with the URL-deserialised state.
  // The catalogue fetch is lazy on first open of either picker
  // surface.
  syncCatPickerLabel();
  if (state.categories && state.categories.size) {
    // We have slugs but no display names yet — populate the cache
    // so the button label upgrades from raw slugs to human names.
    fetchCatalogue().then(syncCatPickerLabel);
  }

  /* ── Desktop right-rail toggle ─────────────────────────────────
   * On desktop, two `.rrt-tab` buttons sit at the top of each rail
   * head. Clicking one swaps the rail content via
   * `body[data-rail-mode]`. The category rail starts hidden;
   * clicking "Topics" reveals it AND warms the catalogue cache so
   * the list paints immediately. The mode is purely a UI choice —
   * not persisted in the URL (it's per-session ambience). */
  function syncRightRailMode() {
    const mode = state.rightRail === "categories" ? "categories" : "people";
    document.body.setAttribute("data-rail-mode", mode);
    // Reflect on every tab so both rails' toggles stay in sync.
    for (const t of document.querySelectorAll(".rrt-tab")) {
      const isOn = t.getAttribute("data-rail") === mode;
      t.classList.toggle("is-current", isOn);
      t.setAttribute("aria-selected", isOn ? "true" : "false");
    }
    // When switching to categories, paint the list eagerly — the
    // user expects "click Topics, see categories now" without
    // having to also tap the search input.
    if (mode === "categories") {
      if (_catalogueCache) {
        renderCatPickerList(catPickerSearch?.value || "");
      } else {
        fetchCatalogue().then(() => renderCatPickerList(""));
      }
    }
  }
  for (const t of document.querySelectorAll(".rrt-tab")) {
    t.addEventListener("click", () => {
      const mode = t.getAttribute("data-rail") || "people";
      if (state.rightRail === mode) return;
      state.rightRail = mode;
      // Persist so the next navigation (or full reload) lands on
      // the same rail. Wrapped because private-mode browsers may
      // refuse the write.
      try {
        localStorage.setItem("kn.right_rail", mode);
      } catch {
        /* storage blocked — drop silently */
      }
      syncRightRailMode();
    });
  }
  // Initial paint — defaults to People mode.
  syncRightRailMode();

  /* Fetch the set of URLs assigned to ANY of the currently-selected
   * category slugs. Used as a pre-filter source by the search path
   * (`buildIndexFilter` consumes it as `url IN (…)`), and as the
   * post-filter source for the feed-search merge step below. Empty
   * Set when no categories are selected — callers should treat the
   * "no filter" case explicitly. Caches the result in-memory per
   * sorted slug-set string so a rapid sequence of selections
   * doesn't re-hit the API. */
  const _catUrlCache = new Map(); // key: "a,b,c" → array of URLs
  async function fetchUrlsForSelectedCategories() {
    const slugs = [...(state.categories || [])].sort();
    if (slugs.length === 0) return [];
    const key = slugs.join(",");
    if (_catUrlCache.has(key)) return _catUrlCache.get(key);
    try {
      const r = await fetch(
        `${API_BASE}/api/document-categories/urls?slugs=${encodeURIComponent(
          key,
        )}`,
        { credentials: "include" },
      );
      if (!r.ok) {
        _catUrlCache.set(key, []);
        return [];
      }
      const urls = await r.json();
      const out = Array.isArray(urls) ? urls : [];
      _catUrlCache.set(key, out);
      return out;
    } catch {
      _catUrlCache.set(key, []);
      return [];
    }
  }
  // Cheap synchronous post-filter for the search-merge step: takes
  // an array of docs and a Set of allowed URLs, keeps only docs in
  // that set. The caller pre-fetched the URL set via
  // fetchUrlsForSelectedCategories(); separating the steps keeps
  // the network call out of the hot loop and lets us share the
  // same set across both the post-filter and the
  // buildIndexFilter (which embeds it as `url IN (…)`).
  function filterDocsByUrlSet(docs, urlSet) {
    if (!urlSet || urlSet.size === 0) return [];
    return docs.filter((d) => urlSet.has(d.url));
  }
  // Expose to other closures (the feed-search merge step calls them).
  window._fetchUrlsForSelectedCategories = fetchUrlsForSelectedCategories;
  window._filterDocsByUrlSet = filterDocsByUrlSet;
  /* Pull-to-refresh handler — feed AND personal pages.
   *
   * Behaviour: invalidate every cache the current view reads from,
   * then re-run the normal load path with the current state. NO
   * special "trailing 7-day window" or unseen-URL filtering — the
   * user's pull simply means "give me whatever's actually fresh on
   * the server right now, with my current filters". The server's
   * snapshot path (refreshed hourly) plus the anon cache (60 s)
   * make this near-free; on a personal page the per-slug cache
   * (`K.invalidateUnindexed`) is the one that matters.
   *
   * Caches busted:
   *   * `_timelineCache` (in-memory Map, keyed by URL+params).
   *   * `KnowledgeSessionCache "timeline:"` (sessionStorage mirror —
   *     survives page navigations).
   *   * `K.invalidateUnindexed(slug)` for every slug in scope —
   *     covers freshly-saved docs that haven't hit the ColBERT
   *     index yet on a personal page.
   *
   * Idempotent against concurrent calls via `_pullRefreshBusy`.
   * `refresh()` itself wipes `state.shownUrls` so the next render
   * doesn't filter out cards we've shown this session. */
  let _pullRefreshBusy = false;
  async function pullRefreshFeed() {
    if (_pullRefreshBusy) return;
    _pullRefreshBusy = true;
    try {
      _timelineCache.clear();
      window.KnowledgeSessionCache?.invalidatePrefix?.(_TIMELINE_SS_PREFIX);
      // Personal-page caches. Two layers: the per-slug unindexed-doc
      // memo (recent saves not yet in ColBERT) and the per-slug
      // browse cache (full PG library) — both keyed on slug, both
      // shadow the freshest data for ~30 s if we don't clear them.
      if (typeof K.invalidateUnindexed === "function") {
        for (const slug of state.libs) K.invalidateUnindexed(slug);
        if (me?.slug) K.invalidateUnindexed(me.slug);
      }
      if (typeof K.invalidatePersonalDocs === "function") {
        for (const slug of state.libs) K.invalidatePersonalDocs(slug);
        if (me?.slug) K.invalidatePersonalDocs(me.slug);
      }
      await refresh();
      window.scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      _pullRefreshBusy = false;
    }
  }

  /* Pull-to-refresh — Twitter-style overscroll gesture from the
   * top of the feed. Only active on the feed view (libs.size = 0,
   * no query), and only when document.scrollY = 0 at the start of
   * a touch. Drags the indicator down with the finger using a
   * rubber-band easing; releasing past `TRIGGER_PX` fires
   * pullRefreshFeed(). Below the threshold the indicator snaps
   * back without firing.
   *
   * The indicator itself is injected once on first activation —
   * keeps the static HTML markup clean and means non-mobile
   * builds never pay for the gesture wiring. */
  function wirePullRefresh() {
    if (document.body.dataset.pullRefreshWired === "1") return;
    document.body.dataset.pullRefreshWired = "1";

    // Lower than the initial 70 — the chip becomes draggable at a
    // smaller threshold so the refresh feels responsive on phones
    // where a 70px drag is most of the screen width.
    const TRIGGER_PX = 55;
    const MAX_PULL_PX = 130;
    const SPRING = "cubic-bezier(0.2, 0.8, 0.2, 1)";
    // Resting offset for the position:fixed chip — sits this many
    // pixels above the top edge while hidden, then translates DOWN
    // by `visual` so the chip slides into view from above. Adds a
    // small landing gap (16) once pulled past the trigger so the
    // chip floats below the very top edge rather than hugging it.
    const REST_OFFSET = -56;
    const LANDING_GAP = 16;

    // Lazy-build the indicator. Inserted before #results so it
    // sits in the same flow as the feed but tucks above the first
    // card. CSS keeps it absolutely hidden until JS sets the
    // transform / opacity inline.
    let indicator = document.getElementById("feedPullRefresh");
    if (!indicator) {
      indicator = document.createElement("div");
      indicator.id = "feedPullRefresh";
      indicator.className = "feed-pull-refresh";
      indicator.setAttribute("aria-hidden", "true");
      indicator.innerHTML = `
        <div class="feed-pull-refresh-spinner" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.2" stroke-linecap="round"
               stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"/>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
          </svg>
        </div>
      `;
      const results = $("results");
      results?.parentNode?.insertBefore(indicator, results);
    }

    let startY = 0;
    let armed = false;
    let pulling = false;
    let lastPull = 0;

    function shouldArm() {
      // Eligible surfaces: the feed (libs empty) and a single
      // personal page (libs.size === 1). Multi-library
      // intersections + active searches are out — those views
      // benefit less from a refresh (the user can re-run the
      // search by hitting return) and the gesture would just
      // fight the normal scroll. Also requires the page to be
      // already scrolled to the very top so a mid-list pull
      // doesn't accidentally fire.
      if (state.libs.size > 1) return false;
      if (state.query) return false;
      // Skip when a Cupertino sheet is open (People / Sources) —
      // the user's downward drag there is meant to dismiss the
      // pane, not refresh the feed underneath. The body class is
      // toggled by openSheet / closeSheets / onDidDismiss in
      // wireMobileChrome.
      if (document.body.classList.contains("cupertino-pane-presented")) {
        return false;
      }
      const top =
        window.pageYOffset ??
        document.documentElement.scrollTop ??
        document.body.scrollTop;
      return top <= 0;
    }

    // The indicator is position:fixed and the resting state is
    // translateY(REST_OFFSET) (above the viewport). All transforms
    // here keep `translateX(-50%)` so the chip stays horizontally
    // centred — losing it on any frame would jump the chip to the
    // left edge.
    function setIndicatorY(y) {
      indicator.style.transform = `translate3d(-50%, ${y}px, 0)`;
    }

    function setPull(px) {
      // Rubber-band: pull tracks 1:1 until TRIGGER_PX, then
      // resistance kicks in so the indicator never stretches
      // unboundedly. Match the iOS rubber-band feel.
      let visual = px;
      if (px > TRIGGER_PX) {
        visual = TRIGGER_PX + (px - TRIGGER_PX) * 0.45;
      }
      visual = Math.min(visual, MAX_PULL_PX);
      lastPull = visual;
      const progress = Math.min(1, visual / TRIGGER_PX);
      const armed = progress >= 1;
      indicator.style.transition = "none";
      // Chip Y in viewport = visual + REST_OFFSET. At visual=0 the
      // chip sits at REST_OFFSET (off-screen above); at visual ≈
      // -REST_OFFSET it crosses the top edge; at visual ≥ TRIGGER_PX
      // (armed) we add LANDING_GAP so the chip floats below the top
      // edge with a small native-feeling breathing room.
      setIndicatorY(visual + REST_OFFSET + (armed ? LANDING_GAP : 0));
      indicator.style.opacity = String(progress);
      // Rotate the icon proportionally so it reads as "winding up"
      // before release.
      const rot = progress * 270;
      const svg = indicator.querySelector("svg");
      if (svg) {
        svg.style.transition = "none";
        svg.style.transform = `rotate(${rot}deg)`;
      }
      indicator.classList.toggle("is-armed", armed);
    }

    function snapBack() {
      indicator.style.transition = `transform 280ms ${SPRING}, opacity 220ms ease`;
      setIndicatorY(REST_OFFSET);
      indicator.style.opacity = "0";
      const svg = indicator.querySelector("svg");
      if (svg) {
        svg.style.transition = `transform 280ms ${SPRING}`;
        svg.style.transform = "rotate(0deg)";
      }
      indicator.classList.remove("is-armed");
    }

    function spinAndFire() {
      // Park the chip at the landing position (just below the top
      // edge) while the refresh is in flight, then snap back. CSS
      // keyframe handles the continuous rotation.
      indicator.style.transition = `transform 220ms ${SPRING}`;
      setIndicatorY(LANDING_GAP);
      indicator.style.opacity = "1";
      indicator.classList.add("is-spinning");
      const svg = indicator.querySelector("svg");
      if (svg) svg.style.transform = "";
      pullRefreshFeed().finally(() => {
        indicator.classList.remove("is-spinning");
        snapBack();
      });
    }

    document.addEventListener(
      "touchstart",
      (e) => {
        if (!shouldArm()) {
          armed = false;
          return;
        }
        armed = true;
        pulling = false;
        startY = e.touches[0].clientY;
      },
      { passive: true },
    );
    document.addEventListener(
      "touchmove",
      (e) => {
        if (!armed) return;
        const dy = e.touches[0].clientY - startY;
        if (dy <= 0) {
          // Upward / no movement — don't engage; let normal scroll
          // take over.
          if (pulling) {
            pulling = false;
            snapBack();
          }
          return;
        }
        // Downward pull from scroll-top: take over.
        pulling = true;
        // preventDefault lets us own the gesture instead of
        // triggering the browser's own pull-to-reload.
        if (e.cancelable) e.preventDefault();
        setPull(dy);
      },
      { passive: false },
    );
    function onEnd() {
      if (!pulling) {
        armed = false;
        return;
      }
      pulling = false;
      armed = false;
      if (lastPull >= TRIGGER_PX) {
        spinAndFire();
      } else {
        snapBack();
      }
      lastPull = 0;
    }
    document.addEventListener("touchend", onEnd, { passive: true });
    document.addEventListener("touchcancel", onEnd, { passive: true });
  }

  /* Track which URLs the user has been shown this session — drives
   * the "More" button's filter so repeated clicks reveal fresh
   * content. Reset on refresh() when the user changes filters,
   * navigates, or starts a search. */
  function markShownUrls(docs) {
    if (!Array.isArray(docs)) return;
    for (const d of docs) {
      if (d && d.url) state.shownUrls.add(d.url);
    }
  }
  // Sync the select with state on boot — `state.dateSince` may have
  // been set from the URL.
  if ($("qSince")) $("qSince").value = state.dateSince || "";
  syncSinceFilterActive();
  /* Build the SQL filter pushed down to every backend pool.
   *
   *   source  → `source IN (?, ?, …)`
   *   tags    → AND of comma-boundary LIKE clauses
   *             `(',' || tags || ',' || extra_tags || ',') LIKE '%,t,%'`
   *             (avoids needing array operators the metadata
   *              index doesn't expose).
   *   _favorites → `url IN (?, ?, …)`. The synthetic source key is
   *             stripped from the source list before building the
   *             `source IN` clause; instead its bound URLs come
   *             from the user's session-side favorites set.
   *
   * Conditions are AND-combined, so e.g. selecting "Favorites" +
   * "github" returns favorited GitHub stars only.
   *
   * Returns `null` when no filterable state is active so callers
   * can route to the unfiltered endpoint and skip the cost of a
   * trivial `WHERE 1=1` round trip. */
  /* Convert a "7d" / "30d" / "365d" / "" range into a YYYY-MM-DD
   * threshold (UTC midnight) the filter clause can compare against. */
  function _sinceDateString(range) {
    if (!range) return null;
    const m = /^(\d+)d$/.exec(range);
    if (!m) return null;
    const days = parseInt(m[1], 10);
    if (!Number.isFinite(days) || days <= 0) return null;
    const d = new Date();
    d.setUTCDate(d.getUTCDate() - days);
    const y = d.getUTCFullYear();
    const mo = String(d.getUTCMonth() + 1).padStart(2, "0");
    const da = String(d.getUTCDate()).padStart(2, "0");
    return `${y}-${mo}-${da}`;
  }

  function buildIndexFilter() {
    const conditions = [];
    const parameters = [];
    const sinceDate = _sinceDateString(state.dateSince);
    if (sinceDate) {
      // The metadata sidecar stores `date` as ISO YYYY-MM-DD text — a
      // lexical >= comparison works.
      conditions.push("date >= ?");
      parameters.push(sinceDate);
    }
    const favOn = state.sources.has(FAV_SOURCE_KEY);
    const realSources = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
    if (realSources.length) {
      // Source match is `source IN (?, ?, ...)` — the canonical
      // source assigned at ingest time.
      //
      // We *used to* OR-in a `link_hosts LIKE …` set so a tweet
      // linking out to youtube.com would surface under the
      // `youtube` chip too. PyLate's index-filter parser doesn't
      // know about the `link_hosts` column, so every filtered
      // search came back as HTTP 400 the moment a source chip
      // was active. Indirect link-host matching still works on
      // the Postgres timeline path (`/api/timeline`) where the
      // column exists; on the index path we drop it to keep the
      // pre-filter shape compatible with the Plaid metadata.
      const sourcePlaceholders = realSources.map(() => "?").join(", ");
      conditions.push(`source IN (${sourcePlaceholders})`);
      parameters.push(...realSources);
    }
    if (state.excludedSources.size) {
      const ex = [...state.excludedSources];
      const placeholders = ex.map(() => "?").join(", ");
      conditions.push(`source NOT IN (${placeholders})`);
      parameters.push(...ex);
    }
    if (state.tags.size) {
      /* Comma-aware tag match WITHOUT the `||` concat operator —
       * PyLate's index filter parser rejects `|` outright. Each
       * tag spans 8 OR'd clauses per column (start / middle / end
       * / exact) to catch the four positions a tag can take in a
       * comma-separated cell. Whole-word: "blog" won't match
       * "blogger". */
      for (const t of state.tags) {
        const lower = t.toLowerCase();
        conditions.push(
          "(tags = ? OR tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR extra_tags = ? OR extra_tags LIKE ? OR extra_tags LIKE ? OR extra_tags LIKE ?)",
        );
        parameters.push(
          lower,
          `${lower},%`,
          `%,${lower},%`,
          `%,${lower}`,
          lower,
          `${lower},%`,
          `%,${lower},%`,
          `%,${lower}`,
        );
      }
    }
    if (favOn) {
      const favs = [...state.favorites];
      if (favs.length === 0) {
        // Filter is on but no favs (rare race: list cleared mid-render).
        // Force a no-match condition so the API returns nothing rather
        // than ignoring the filter.
        conditions.push("1 = 0");
      } else {
        const placeholders = favs.map(() => "?").join(", ");
        conditions.push(`url IN (${placeholders})`);
        parameters.push(...favs);
      }
    }
    // Category pre-filter — when one or more categories are
    // selected, narrow the search to URLs already known to be in
    // those categories. The set is pre-fetched by refresh() into
    // `_categoryUrlSet` before this function is called; we just
    // unpack the same array here as a `url IN (?, ?, …)` clause.
    if (state.categories && state.categories.size) {
      const allowed = _categoryUrlSet || [];
      if (allowed.length === 0) {
        // Either still warming or empty for this slug set —
        // force-empty rather than silently ignoring the filter so
        // the result list matches the filter the user picked.
        conditions.push("1 = 0");
      } else {
        const placeholders = allowed.map(() => "?").join(", ");
        conditions.push(`url IN (${placeholders})`);
        parameters.push(...allowed);
      }
    }
    return conditions.length
      ? { condition: conditions.join(" AND "), parameters }
      : null;
  }
  // Cache the URL set across a single refresh() call so all the
  // call sites of buildIndexFilter (initial query, pagination, the
  // hint route) see the same allowed list without each re-fetching.
  let _categoryUrlSet = null;

  /* ── Real-time summary enhancer ──────────────────────────────────
   *
   * For each rendered card whose summary is missing or short, fetch
   * the underlying webpage through the auth-gated proxy and pull a
   * description out of the head:
   *
   *   1. <meta property="og:description">
   *   2. <meta name="twitter:description">
   *   3. <meta name="description">
   *
   * Walked top-to-bottom, one at a time — that way the topmost cards
   * (the ones the user is reading) get enhanced first and the cost
   * is amortised across the time the user spends scanning the
   * results. A token bumped on every refresh() cancels the in-flight
   * loop on filter / query / library changes so stale enhancements
   * never paint over freshly-rendered cards.
   *
   * Cache is in-memory (session-scoped) so re-rendering the same
   * cards after a sort flip or rerank pass doesn't re-fetch. We
   * cache misses (null) too so a 404'd or CORS-blocked URL doesn't
   * keep retrying. */
  const enhanceCache = new Map(); // url → { description, image, title } or null
  let enhanceToken = 0;

  // Abort the enhance loop the moment the page is unloaded or hidden
  // (back-navigation, tab switch, bfcache). Without this, in-flight
  // `/api/proxy/fetch` requests keep firing through the page-transition
  // and surface as a wall of CORS / "access control" errors in the
  // console of whichever page the user landed on. Bumping the token
  // makes the next iteration of `enhanceResults` exit cleanly.
  window.addEventListener("pagehide", () => {
    enhanceToken++;
  });
  window.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") enhanceToken++;
  });

  function parsePageMeta(html) {
    // Snip to the <head> when possible — the body is huge and
    // DOMParser is happy enough with a half-truncated document.
    const headEnd = html.search(/<\/head>/i);
    const slice =
      headEnd > 0 ? html.slice(0, headEnd + 7) : html.slice(0, 50000);
    let parsed;
    try {
      parsed = new DOMParser().parseFromString(slice, "text/html");
    } catch {
      return null;
    }
    const get = (sel) => {
      const el = parsed.querySelector(sel);
      if (!el) return null;
      const v = el.getAttribute("content") || el.textContent || "";
      return v.trim() || null;
    };
    return {
      description:
        get('meta[property="og:description"]') ||
        get('meta[name="twitter:description"]') ||
        get('meta[name="description"]') ||
        null,
      image:
        get('meta[property="og:image"]') ||
        get('meta[name="twitter:image"]') ||
        null,
      title: get('meta[property="og:title"]') || get("title") || null,
    };
  }

  async function fetchPageMeta(url) {
    if (enhanceCache.has(url)) return enhanceCache.get(url);
    try {
      const proxy = `${API_BASE}/api/proxy/fetch?url=${encodeURIComponent(url)}`;
      const r = await fetch(proxy, { credentials: "include" });
      if (!r.ok) {
        enhanceCache.set(url, null);
        return null;
      }
      const html = await r.text();
      const meta = parsePageMeta(html);
      enhanceCache.set(url, meta);
      return meta;
    } catch {
      enhanceCache.set(url, null);
      return null;
    }
  }

  /* Walk visible result cards top-down, enhance summaries one at a
   * time. Skip cards whose existing summary is already substantial. */
  async function enhanceResults(token, docs) {
    const SHORT_SUMMARY = 120; // chars; below this we try to enrich
    for (const d of docs) {
      if (token !== enhanceToken) return;
      if (!d.url) continue;
      const existing = (d.summary || "").trim();
      if (existing.length >= SHORT_SUMMARY) continue;
      const meta = await fetchPageMeta(d.url);
      if (token !== enhanceToken) return;
      if (!meta || !meta.description) continue;
      // Only swap if the fetched description is meaningfully better
      // than what we have (don't replace a 100-char summary with a
      // 50-char og description).
      if (meta.description.length <= existing.length) continue;
      const card = $("results").querySelector(
        `.result[data-url="${window.CSS && CSS.escape ? CSS.escape(d.url) : d.url.replace(/"/g, '\\"')}"]`,
      );
      if (!card) continue;
      const sumEl = card.querySelector(".result-summary");
      const text =
        meta.description.length > 320
          ? meta.description.slice(0, 320).replace(/\s+\S*$/, "") + "…"
          : meta.description;
      if (sumEl) {
        sumEl.textContent = text;
        sumEl.classList.add("enhanced");
      } else {
        const titleAnchor = card.querySelector(".result-body > a");
        if (!titleAnchor) continue;
        const p = document.createElement("p");
        p.className = "result-summary enhanced";
        p.textContent = text;
        titleAnchor.insertAdjacentElement("afterend", p);
      }
    }
  }

  /* Active-tag chip strip — renders the current `state.tags` as a
   * row of removable chips just under the spotlight. Each chip is
   * a click target that removes the tag, syncs the URL, and
   * refreshes the result set. The strip itself stays `hidden` when
   * no tags are active so the search bar isn't crowded. */
  function renderActiveTags() {
    const host = $("activeTags");
    if (!host) return;
    if (!state.tags.size) {
      host.hidden = true;
      host.innerHTML = "";
      return;
    }
    host.hidden = false;
    const chips = [...state.tags]
      .map(
        (
          t,
        ) => `<button class="active-tag" data-active-tag="${escapeAttr(t)}" type="button" title="Remove filter — ${escapeAttr(t)}">
          <span>${escapeHtml(t)}</span>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
            <line x1="6" y1="6" x2="18" y2="18"/>
            <line x1="18" y1="6" x2="6" y2="18"/>
          </svg>
        </button>`,
      )
      .join("");
    const clearAll =
      state.tags.size > 1
        ? '<button class="active-tag-clear" id="activeTagsClear" type="button">Clear all</button>'
        : "";
    host.innerHTML = chips + clearAll;
    host.querySelectorAll("[data-active-tag]").forEach((b) =>
      b.addEventListener("click", () => {
        state.tags.delete(b.dataset.activeTag);
        writeUrl();
        refresh();
      }),
    );
    const clearBtn = host.querySelector("#activeTagsClear");
    if (clearBtn)
      clearBtn.addEventListener("click", () => {
        state.tags.clear();
        writeUrl();
        refresh();
      });
  }

  /* ── Follow-graph timeline (default state for /search) ──────
   *
   * When the user lands on /search with no libraries selected, we
   * surface the activity of the people they follow (plus their own
   * library) as a Twitter-style timeline. The payload comes from
   * the dedicated `/api/timeline` endpoint and is mapped onto the
   * existing search-card shape so the renderResult template needs
   * zero changes.
   *
   * Cached for the page session so navigating in and out of the
   * empty-libs state doesn't refetch on every refresh. */
  // Cache keyed by filter signature so source/tag toggles don't
  // re-fetch the timeline within a single state. Mirrors into
  // sessionStorage via KnowledgeSessionCache so a fresh page load
  // (e.g. bouncing from /<slug> back to the feed) repaints instantly
  // instead of running another /api/timeline round-trip.
  const _timelineCache = new Map();
  const _TIMELINE_SS_TTL_MS = 10 * 60 * 1000; // 10 min
  const _TIMELINE_SS_PREFIX = "timeline:";
  async function loadFollowingTimeline(overrides = {}) {
    const srcs = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
    const excl = [...state.excludedSources];
    const tags = [...state.tags];
    // Bigger default page size on the initial timeline pull (was 50).
    // The timeline SQL hits the same Postgres plan whether we ask for
    // 50 or 75 rows — the dominant cost is the candidate scan + JSONB
    // aggregation, which sizes by `$2 * 16` and is the same either
    // way. Net effect: ~50% more cards land on first paint, so the
    // user has more content to read before the next pagination
    // trigger fires.
    const qs = new URLSearchParams({ limit: String(overrides.limit || 75) });
    if (srcs.length) qs.set("sources", srcs.join(","));
    if (excl.length) qs.set("exclude_sources", excl.join(","));
    if (tags.length) qs.set("tags", tags.join(","));
    // Date pre-filter — same SQL `date >= ?` clause the index path
    // applies, so "Past week / month / year" narrows the feed at
    // the database layer instead of a JS post-filter. The optional
    // `overrides.since` arg lets callers (like the "More" button)
    // force a specific trailing window regardless of the active
    // date-filter chip.
    const sinceDate =
      overrides.since !== undefined
        ? overrides.since
        : _sinceDateString(state.dateSince);
    if (sinceDate) qs.set("since", sinceDate);
    // Topical filter — when one or more categories are selected,
    // restrict the timeline to docs the categorize daemon assigned
    // to ANY of them (OR semantics handled server-side). Same param
    // name on the wire (the Rust handler accepts CSV).
    if (state.categories && state.categories.size) {
      qs.set("category", [...state.categories].join(","));
    }
    // Hide-seen is the default for signed-in viewers; only opt-in
    // when the user has clicked "Show seen". Anonymous viewers never
    // get filtered server-side (the SQL gates on $1 IS NOT NULL).
    if (state.showSeen) qs.set("include_seen", "true");
    const url = `${API_BASE}/api/timeline?${qs.toString()}`;
    // Bypass the cache when the caller wants a fresh fetch — the
    // pull-to-refresh handler and the "More" button set fresh:true
    // so the user always sees the freshest data when they ask for
    // it explicitly. The same flag also gates sessionStorage below.
    if (!overrides.fresh) {
      if (_timelineCache.has(url)) {
        return _timelineCache.get(url);
      }
      // Cross-navigation hit: the in-memory Map was wiped by the
      // page reload but sessionStorage carried the last payload
      // through. Hydrate the Map so subsequent calls in this page
      // session don't re-parse the JSON on every refresh.
      const cached =
        window.KnowledgeSessionCache &&
        window.KnowledgeSessionCache.get(
          _TIMELINE_SS_PREFIX + url,
          _TIMELINE_SS_TTL_MS,
        );
      if (Array.isArray(cached)) {
        _timelineCache.set(url, cached);
        // Re-populate perSlugMeta from the cached sharers so the
        // avatar stack renderer doesn't need to re-fetch per row.
        for (const d of cached) {
          const sharers = Array.isArray(d.sharers) ? d.sharers : [];
          for (const s of sharers) {
            if (s && s.slug && !state.perSlugMeta[s.slug]) {
              state.perSlugMeta[s.slug] = {
                slug: s.slug,
                name: s.name || s.slug,
                avatar: s.avatar || null,
                twitterFollowers: s.twitterFollowers || 0,
              };
            }
          }
        }
        return cached;
      }
    }
    try {
      const r = await fetch(url, { credentials: "include" });
      if (!r.ok) {
        _timelineCache.set(url, []);
        return [];
      }
      const docs = await r.json();
      const mapped = (Array.isArray(docs) ? docs : []).map((d) => {
        // Stamp the avatar+name for every sharer into state.perSlugMeta
        // so the result-card's "shared-by" stack can render them
        // without hitting /api/users per row.
        const sharers = Array.isArray(d.sharers) ? d.sharers : [];
        for (const s of sharers) {
          if (s && s.slug && !state.perSlugMeta[s.slug]) {
            state.perSlugMeta[s.slug] = {
              slug: s.slug,
              name: s.name || s.slug,
              avatar: s.avatar || null,
              // Carry through twitter-follower count so the avatar
              // shuffler can rank non-followees by popularity.
              twitterFollowers: s.twitterFollowers || 0,
            };
          }
        }
        return {
          url: d.url,
          title: d.title,
          summary: d.summary || "",
          // Pedagogical-rewriter fields populated by the clean
          // daemon (sources/utils/clean_daemon.py). Empty string
          // when the daemon hasn't processed the row yet or chose
          // to leave the summary blank. renderResult prefers these
          // over the raw fields when present.
          cleanTitle: d.cleanTitle || "",
          cleanSummary: d.cleanSummary || "",
          // Flat URL list extracted by the pipeline from the raw
          // summary + linked_urls. The render path uses this so
          // no URL the original post referenced ever gets lost
          // even when the daemon's clean_summary drops the label
          // around it.
          urls: Array.isArray(d.urls) ? d.urls : [],
          date: d.date || "",
          tags: Array.isArray(d.tags) ? d.tags : [],
          extraTags: [],
          source: d.source || "",
          source_url: d.source_url || null,
          // Carry the link cluster through so a tweet linking to an
          // arxiv / huggingface / github resource renders the same
          // inline preview here that it does on the personal page.
          // Backend sends `linked_urls` (snake_case) — normalise to
          // the camelCase the card renderer expects.
          linkedUrls: Array.isArray(d.linked_urls) ? d.linked_urls : [],
          linkHosts: Array.isArray(d.link_hosts) ? d.link_hosts : [],
          sharers,
          sharerCount: d.sharerCount || sharers.length,
          // HackerNews front-page picks: not yet in the user's library
          // (no row in `documents`), so the card renders a "Save" button
          // instead of the favorite heart. See renderResult below.
          picked: !!d.picked,
          // Carry through the server-side already-seen flag so the
          // card renderer can dim cards the viewer has already
          // absorbed (only meaningful when include_seen=1).
          alreadySeen: !!d.alreadySeen,
          // The renderResult template renders avatars off `_owners`
          // (a list of slugs); map the timeline's `sharers` array to
          // that field so the existing stack-renderer works unchanged.
          _owners: sharers.map((s) => s.slug).filter(Boolean),
        };
      });
      _timelineCache.set(url, mapped);
      if (window.KnowledgeSessionCache) {
        window.KnowledgeSessionCache.set(_TIMELINE_SS_PREFIX + url, mapped);
      }
      return mapped;
    } catch {
      _timelineCache.set(url, []);
      return [];
    }
  }

  /* Small kicker shown above the result list when we're rendering
   * the follow-graph timeline instead of search results. Inserted
   * lazily so the search.html markup stays untouched. */
  /* ── Right-side people-to-follow rail ─────────────────────────
   *
   * Renders a filterable list of personalities with a Follow /
   * Following toggle button per row. The interaction mirrors the
   * existing libraries picker: a debounced filter input narrows
   * the list as the user types; buttons hit /api/follow/{slug}.
   *
   * Hydrated lazily on first call so the page boot stays cheap. */
  /* Same algorithm as the libraries picker: substring filter over
   * name|description|slug|category, augmented by ColBERT-ranked
   * owners (queries ≥ 3 chars hit /indices/__all__/search_with_encoding,
   * group hits by owner, score with `count / sqrt(documentCount)`,
   * take top 10 not already in the substring set). Debounced 250 ms
   * with abort-on-new-query so fast typing doesn't backlog requests.
   */
  const PEOPLE_RAIL_PAGE = 30;
  const SRC_RAIL_PAGE = 50;
  const _srcRail = { page: SRC_RAIL_PAGE, observer: null };

  /* Lift the follow-set fetch out of the people-rail so the profile
   * header can read it before the rail is hydrated. We cache the
   * in-flight promise so concurrent callers (rail boot + personal
   * page header) only hit /api/me/following once. */
  let _followingPromise = null;
  function loadFollowingSet() {
    if (_followingPromise) return _followingPromise;
    _followingPromise = (async () => {
      if (!me) return new Set();
      try {
        const r = await fetch(`${API_BASE}/api/me/following`, {
          credentials: "include",
        });
        if (!r.ok) return new Set();
        const list = await r.json();
        return new Set(list.map((u) => u.slug));
      } catch {
        return new Set();
      }
    })();
    return _followingPromise;
  }
  const _peopleRail = {
    populated: false,
    rows: [], // [{slug, name, avatar, description, documentCount, category}, …]
    following: new Set(),
    filter: "",
    filterTimer: null,
    // How many rows we're currently rendering. Grows by PEOPLE_RAIL_PAGE
    // each time the user scrolls the rail near its bottom.
    page: PEOPLE_RAIL_PAGE,
    observer: null,
    colbertOwners: [], // top-N slugs by length-normalized ColBERT hit count
    colbertCtrl: null,
    colbertTimer: null,
  };

  function _peopleRailColbertFetch(q) {
    if (_peopleRail.colbertTimer) clearTimeout(_peopleRail.colbertTimer);
    if (_peopleRail.colbertCtrl) _peopleRail.colbertCtrl.abort();
    if (!q || q.length < 3) {
      _peopleRail.colbertOwners = [];
      return;
    }
    _peopleRail.colbertCtrl = new AbortController();
    _peopleRail.colbertTimer = setTimeout(async () => {
      try {
        const r = await fetch(
          `${API_BASE}/indices/${ALL_INDEX_NAME}/search_with_encoding`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ queries: [q], params: { top_k: 200 } }),
            signal: _peopleRail.colbertCtrl.signal,
          },
        );
        if (!r.ok) return;
        const data = await r.json();
        const meta = (data.results && data.results[0]?.metadata) || [];
        // Rank by FIRST appearance — owner of the top-ranked hit wins,
        // owner of the second wins next, etc. No aggregation, no size
        // normalisation. Matches the source-rail behaviour.
        const seen = new Set();
        const ordered = [];
        for (const m of meta) {
          const o = m.owner || "";
          if (!o || seen.has(o)) continue;
          seen.add(o);
          ordered.push(o);
          if (ordered.length >= 10) break;
        }
        _peopleRail.colbertOwners = ordered;
        // Only re-render if the user hasn't moved on — same race
        // guard as the lib picker.
        if (_peopleRail.filter === q) renderPeopleRail();
      } catch {
        // aborted or network blip — keep previous list
      }
    }, 250);
  }

  async function setupPeopleRail() {
    const root = $("peopleRail");
    if (!root || _peopleRail.populated) return;
    _peopleRail.populated = true;
    // Surface the rail spinner immediately. Hidden by the first
    // renderPeopleRail() that has rows.
    showPeopleSpinner();

    const input = $("peopleRailFilter");

    // VIP personalities — already cached by K.listPersonalities.
    let users = [];
    try {
      users = (await K.listPersonalities()) || [];
    } catch {}
    // Right rail shows only VIP personalities (the grandfathered cohort).
    // Default order: shuffled per page-load. Showing the same top-of-
    // list faces every time made the rail feel stale; randomising
    // surfaces the long tail and gives the page some life. Filter
    // matches still rank by relevance below.
    const pool = users.filter((u) => u && u.slug && u.vip);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    _peopleRail.rows = pool;

    // Current follow set — shared loader, so the profile header can
    // also consume it without re-fetching.
    _peopleRail.following = await loadFollowingSet();
    _peopleRail.followingHydrated = true;

    renderPeopleRail();

    input.addEventListener("input", () => {
      const next = input.value.trim().toLowerCase();
      _peopleRail.filter = next;
      // Filter change → start from the top page.
      _peopleRail.page = PEOPLE_RAIL_PAGE;
      // Substring pass renders synchronously; ColBERT augmentation
      // lands later and re-renders when its results match the
      // still-current query.
      _peopleRailColbertFetch(next);
      renderPeopleRail();
    });
  }

  function armPeopleRailScroll() {
    const sentinel = document.querySelector(".people-rail-sentinel");
    if (!sentinel) return;
    if (_peopleRail.observer) _peopleRail.observer.disconnect();
    // Use the rail itself as the scroll root so we pick up scroll
    // inside `.people-rail-list` even if the rail doesn't scroll the
    // whole document.
    const root = document.querySelector(".people-rail-list");
    _peopleRail.observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            _peopleRail.page += PEOPLE_RAIL_PAGE;
            renderPeopleRail();
            return;
          }
        }
      },
      { root, rootMargin: "120px 0px" },
    );
    _peopleRail.observer.observe(sentinel);
  }

  function renderPeopleRail() {
    hidePeopleSpinner();
    const listHost = $("peopleRailList");
    if (!listHost) return;

    const mySlug = me?.slug;
    const q = _peopleRail.filter;

    /* Substring match first — same predicate the libraries picker
     * uses: name | description | slug | category. */
    const matchesQuery = (p) =>
      !q ||
      (p.name || "").toLowerCase().includes(q) ||
      (p.description || "").toLowerCase().includes(q) ||
      (p.slug || "").toLowerCase().includes(q) ||
      // categories[] is an array of ontology slugs (post-migration
      // shape). Match against the slug AND its human label so a
      // query like "safety" still surfaces `ai-safety`.
      (p.categories || []).some(
        (c) =>
          c.toLowerCase().includes(q) ||
          (CATEGORY_LABELS[c] || "").toLowerCase().includes(q),
      );
    const substr = _peopleRail.rows.filter(matchesQuery);

    /* ColBERT augmentation — append up to 10 owners whose libraries
     * are densely about the typed topic but whose profile text
     * doesn't mention it (so substring missed them). */
    let merged = substr;
    if (q && _peopleRail.colbertOwners.length) {
      const seen = new Set(substr.map((p) => p.slug));
      const bySlug = new Map(_peopleRail.rows.map((p) => [p.slug, p]));
      const extras = [];
      for (const slug of _peopleRail.colbertOwners) {
        if (seen.has(slug)) continue;
        const p = bySlug.get(slug);
        if (!p) continue;
        extras.push(p);
        seen.add(slug);
        if (extras.length >= 10) break;
      }
      if (extras.length) merged = [...substr, ...extras];
    }

    // Hide the caller's own row — you can't follow yourself.
    let visible = merged.filter((u) => u.slug !== mySlug);
    // No-query default. Three buckets, in render order:
    //   1. Recently-clicked people — read from localStorage so the
    //      browser remembers across visits. Sorted by recency desc
    //      then click count desc, so the person the user last visited
    //      lands at the top of their rail.
    //   2. Everyone else — deterministic popularity. Primary key is
    //      Twitter followers desc; tiebreakers fall through to
    //      GitHub followers, then citations, then document count.
    //      No randomness: Karpathy at 2.3M Twitter followers lands
    //      first every time and the order doesn't shift between
    //      refreshes. Followed-vs-unfollowed is no longer a bucket
    //      boundary — popularity wins outright so the rail reads as
    //      one consistent ranking.
    if (!q) {
      const clicks = _readPeopleClicks();
      const pinned = [];
      const others = [];
      for (const u of visible) {
        if (clicks[u.slug]) pinned.push(u);
        else others.push(u);
      }
      pinned.sort((a, b) => {
        const ca = clicks[a.slug];
        const cb = clicks[b.slug];
        const byAt = (cb.at || 0) - (ca.at || 0);
        if (byAt !== 0) return byAt;
        return (cb.n || 0) - (ca.n || 0);
      });
      const cmpDesc = (av, bv) => (Number(bv) || 0) - (Number(av) || 0);
      others.sort((a, b) => {
        const byTw = cmpDesc(a.twitterFollowers, b.twitterFollowers);
        if (byTw !== 0) return byTw;
        const byGh = cmpDesc(a.githubFollowers, b.githubFollowers);
        if (byGh !== 0) return byGh;
        const byCi = cmpDesc(a.citations, b.citations);
        if (byCi !== 0) return byCi;
        return cmpDesc(a.documentCount, b.documentCount);
      });
      visible = [...pinned, ...others];
    }
    // Cap the underlying pool so the render math stays bounded even
    // for huge follow lists; pagination below decides how many of
    // the capped list we actually paint.
    visible = visible.slice(0, 500);
    const fullCount = visible.length;
    const pageSlice = visible.slice(0, _peopleRail.page);

    if (!pageSlice.length) {
      listHost.innerHTML = `<div class="people-rail-empty">${
        q ? `No people match "${escapeHtml(q)}"` : "No people available"
      }</div>`;
      return;
    }

    const hasMore = fullCount > pageSlice.length;
    listHost.innerHTML = pageSlice
      .map((u) => {
        const initials = (u.name || u.slug || "?")
          .split(/\s+/)
          .slice(0, 2)
          .map((w) => (w[0] || "").toUpperCase())
          .join("");
        // encodeURI keeps the URL functional while making it
        // impossible to break out of the CSS `url('…')` literal —
        // `(`, `)`, `;`, `'`, `"`, newline are all percent-encoded
        // by encodeURI's stricter cousin encodeURIComponent. Anything
        // that fails URL parsing falls back to no avatar.
        let avatarStyle = "";
        if (u.avatar) {
          try {
            const safe = new URL(u.avatar, window.location.origin);
            if (safe.protocol === "http:" || safe.protocol === "https:") {
              // CSS `url()` argument is wrapped in SINGLE quotes
              // because the surrounding HTML attribute uses double
              // quotes (`style="${avatarStyle}"`). With double
              // quotes inside double quotes the HTML parser closes
              // the style attribute at the first inner `"` and the
              // avatar URL disappears — empirically the people
              // panel showed `style="background-image: url("` and
              // every avatar was blank. encodeURI doesn't encode
              // `'`, so we also `.replace(/'/g, '%27')` to keep the
              // single-quoted literal safe even on the (very rare)
              // avatar URL that contains one.
              const encoded = encodeURI(safe.toString()).replace(/'/g, "%27");
              avatarStyle = `background-image: url('${encoded}');`;
            }
          } catch (_) {
            /* leave avatarStyle empty */
          }
        }
        const following = _peopleRail.following.has(u.slug);
        return `
          <div class="people-row" role="listitem" data-slug="${escapeAttr(u.slug)}">
            <a class="pr-avatar" href="/search?libs=${encodeURIComponent(u.slug)}" style="${avatarStyle}" aria-label="${escapeAttr(u.name || u.slug)}">${u.avatar ? "" : escapeHtml(initials)}</a>
            <a class="pr-body" href="/search?libs=${encodeURIComponent(u.slug)}">
              <div class="pr-name">${escapeHtml(u.name || u.slug)}</div>
              ${u.description ? `<div class="pr-desc">${escapeHtml(u.description)}</div>` : ""}
            </a>
            <button class="pr-follow ${following ? "is-following" : ""}"
                    type="button"
                    data-slug="${escapeAttr(u.slug)}"
                    aria-pressed="${following}"></button>
          </div>
        `;
      })
      .join("");

    // Sentinel for infinite-scroll within the rail. The
    // IntersectionObserver lives on _peopleRail.observer so we can
    // reset it cleanly on filter changes.
    if (hasMore) {
      listHost.insertAdjacentHTML(
        "beforeend",
        `<div class="people-rail-sentinel" aria-hidden="true"></div>`,
      );
      armPeopleRailScroll();
    } else if (_peopleRail.observer) {
      _peopleRail.observer.disconnect();
      _peopleRail.observer = null;
    }

    // Click memory: record every tap on the avatar or body link so
    // the next page load surfaces the visited people at the top of
    // the rail. Delegated to the list host so newly-appended rows
    // (infinite scroll) are covered without per-row wiring. Guarded
    // by a sentinel attribute so renderPeopleRail() can repaint the
    // list as often as it wants without stacking listeners.
    // Capture-phase so the navigation that follows the click doesn't
    // outrun us — most users tap and immediately leave the page,
    // and a non-capture listener might miss the write.
    if (listHost.dataset.peopleClickTrackerWired !== "1") {
      listHost.dataset.peopleClickTrackerWired = "1";
      listHost.addEventListener(
        "click",
        (e) => {
          const link = e.target?.closest?.("a.pr-avatar, a.pr-body");
          if (!link) return;
          const row = link.closest(".people-row");
          const slug = row?.dataset?.slug;
          if (slug) recordPeopleClick(slug);
        },
        { capture: true },
      );
    }

    listHost.querySelectorAll(".pr-follow").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Unauthenticated → pop the login modal. After a successful
        // login the page reloads and the rail rebuilds itself.
        if (!me) {
          window.KnowledgeAuth?.open("login");
          return;
        }
        const slug = btn.dataset.slug;
        const isFollowing = _peopleRail.following.has(slug);
        // Optimistic toggle.
        if (isFollowing) _peopleRail.following.delete(slug);
        else _peopleRail.following.add(slug);
        btn.classList.toggle("is-following", !isFollowing);
        btn.setAttribute("aria-pressed", String(!isFollowing));
        try {
          const r = await fetch(
            `${API_BASE}/api/follow/${encodeURIComponent(slug)}`,
            {
              method: isFollowing ? "DELETE" : "POST",
              credentials: "include",
            },
          );
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          // Bust the timeline cache so the new follow shows up the
          // next time the empty-libs landing state renders.
          _timelineCache.clear();
          window.KnowledgeSessionCache?.invalidatePrefix?.(_TIMELINE_SS_PREFIX);
        } catch (err) {
          // Roll back optimistic state on failure.
          if (isFollowing) _peopleRail.following.add(slug);
          else _peopleRail.following.delete(slug);
          btn.classList.toggle("is-following", isFollowing);
          btn.setAttribute("aria-pressed", String(isFollowing));
          console.warn("[follow]", err);
        }
      });
    });
  }

  /* Twitter-style profile header — only renders when the current
   * selection is exactly the signed-in user's own library, so the
   * page reads as a personal page (own bookmarks only) rather than
   * the feed (own bookmarks + followees).
   *
   * Pass `show = false` to force hide regardless of the rule. */
  function showProfileHeader(show) {
    const h = $("profileHeader");
    if (!h) return;
    if (show === false) {
      h.hidden = true;
      h.innerHTML = "";
      return;
    }
    // Render the header whenever EXACTLY ONE library is selected —
    // regardless of whether it's the signed-in user's own or
    // someone else's. The header reads as "you're looking at @slug's
    // page" so it makes sense for any personality.
    if (state.libs.size !== 1) {
      h.hidden = true;
      h.innerHTML = "";
      return;
    }
    const slug = [...state.libs][0];
    const meta =
      state.allPersonalities.find((p) => p.slug === slug) ||
      state.perSlugMeta?.[slug] ||
      (me && me.slug === slug ? me : null);
    if (!meta) {
      // Personality list might not be hydrated yet — try again after
      // the user-catalog cache promise resolves.
      h.hidden = true;
      h.innerHTML = "";
      return;
    }
    const initials = (meta.name || slug || "?")
      .split(/\s+/)
      .slice(0, 2)
      .map((w) => (w[0] || "").toUpperCase())
      .join("");
    const avatarHtml = meta.avatar
      ? `<img class="ph-avatar" src="${escapeAttr(meta.avatar)}" alt="@${escapeAttr(slug)}" onerror="this.style.display='none'"/>`
      : `<span class="ph-avatar ph-avatar-fallback" aria-hidden="true">${escapeHtml(initials)}</span>`;
    const docs =
      typeof meta.documentCount === "number" ? meta.documentCount : null;
    // The follow set might not be hydrated yet on a cold personal-page
    // load (the people rail is set up lazily). Kick off the shared
    // loader and re-render once it lands so the Follow / Following
    // label reflects the actual state.
    if (me && !_peopleRail.followingHydrated) {
      loadFollowingSet().then((set) => {
        _peopleRail.following = set;
        _peopleRail.followingHydrated = true;
        // Only re-render if we're still on the same personality.
        if (state.libs.size === 1 && [...state.libs][0] === slug) {
          showProfileHeader();
        }
      });
    }
    const isFollowing = _peopleRail?.following?.has?.(slug);
    const isMe = me && me.slug === slug;
    // Export button — visible to everyone (anonymous, signed-in,
    // owner, VIP). The click handler in web/export.js handles
    // pricing + auth-gating: it fetches a quote first, shows a
    // dialog with the cost, then triggers the JSONL download via
    // the same URL. Wiring lives in export.js so this file doesn't
    // know about credits.
    const exportBtnHtml = `<button type="button"
               class="ph-export"
               data-action="ph-export"
               data-slug="${escapeAttr(slug)}"
               title="Export this library as JSONL">Export</button>`;
    const actionHtml = isMe
      ? exportBtnHtml
      : `<button type="button"
                 class="ph-follow ${isFollowing ? "is-following" : ""}"
                 data-action="ph-follow"
                 data-slug="${escapeAttr(slug)}"></button>${exportBtnHtml}`;
    h.innerHTML = `
      <div class="ph-body">
        ${avatarHtml}
        <div class="ph-meta">
          <div class="ph-row">
            <div class="ph-name">${escapeHtml(meta.name || slug)}</div>
            ${actionHtml}
          </div>
          <div class="ph-handle">@${escapeHtml(slug)}</div>
          ${meta.description ? `<div class="ph-bio">${escapeHtml(meta.description)}</div>` : ""}
          <div class="ph-stats">
            ${docs !== null ? `<span><strong>${docs}</strong> bookmarks</span>` : ""}
            ${
              // "N following" — only on the signed-in user's own
              // personal page. We pull from the shared follow-set
              // loader so the number stays in sync with the people
              // rail without an extra round-trip.
              isMe && _peopleRail?.followingHydrated
                ? `<span><strong>${_peopleRail.following.size}</strong> following</span>`
                : ""
            }
          </div>
        </div>
      </div>
    `;
    // Wire the Follow / Following button.
    const fb = h.querySelector("[data-action='ph-follow']");
    if (fb) {
      fb.addEventListener("click", async (e) => {
        e.preventDefault();
        if (!me) {
          $("authBtn")?.click();
          return;
        }
        const wasOn = fb.classList.contains("is-following");
        fb.classList.toggle("is-following", !wasOn);
        try {
          const r = await fetch(
            `${API_BASE}/api/follow/${encodeURIComponent(slug)}`,
            { method: wasOn ? "DELETE" : "POST", credentials: "include" },
          );
          if (!r.ok) throw new Error("HTTP " + r.status);
          if (_peopleRail?.following) {
            if (wasOn) _peopleRail.following.delete(slug);
            else _peopleRail.following.add(slug);
            // Keep the right-rail row in sync without a full re-render.
            const railBtn = document.querySelector(
              `.people-row .pr-follow[data-slug="${slug.replace(/"/g, '\\"')}"]`,
            );
            if (railBtn) railBtn.classList.toggle("is-following", !wasOn);
          }
        } catch {
          fb.classList.toggle("is-following", wasOn);
        }
      });
    }
    // Export button is wired via document-level event delegation
    // in /export.js — survives re-renders and avoids ordering
    // assumptions between this file and the export module.
    h.hidden = false;
  }

  function showFollowingHeader(_show) {
    // Header removed — keep the function as a no-op so existing
    // call sites don't need to be touched.
    const h = document.getElementById("followingHeader");
    if (h) h.remove();
  }

  /* Banned-sources strip — sits next to the search bar and shows
   * one tiny chip per source the user has hidden via the per-card
   * ✕ button. Click a chip to restore that source. Hidden when no
   * sources are banned. The list mirrors state.excludedSources
   * verbatim — we don't filter by "still present in active libs"
   * here because a user who explicitly hid a source should still
   * see the running ban so they can undo it. */
  function renderResultSources() {
    const host = $("resultSources");
    if (!host) return;
    const banned = [...state.excludedSources];
    if (!banned.length) {
      host.hidden = true;
      host.innerHTML = "";
      return;
    }
    host.hidden = false;
    host.innerHTML = banned
      .map((key) => {
        const icon = K.sourceIconUrl(key);
        const iconHtml = icon
          ? `<img src="${escapeAttr(icon)}" alt="" onerror="this.style.display='none'"/>`
          : "";
        return `<button class="result-source banned" type="button"
                data-restore-source="${escapeAttr(key)}"
                title="Restore ${escapeAttr(key)} to results">
          ${iconHtml}
          <span>${escapeHtml(key)}</span>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 3-6.7"/>
            <path d="M3 4v5h5"/>
          </svg>
        </button>`;
      })
      .join("");
    host.querySelectorAll("[data-restore-source]").forEach((b) =>
      b.addEventListener("click", () => {
        const key = b.dataset.restoreSource;
        state.excludedSources.delete(key);
        renderSrc();
        writeUrl();
        refresh();
      }),
    );
  }

  let reqId = 0;

  /* Update the metrics pill above the results list. `total` is
   * the candidate-pool size (sum of documentCount across selected
   * libs), `nResults` is what we ended up rendering, `tMs` is the
   * wall-clock from refresh() start to render. The pill is hidden
   * when no library is selected so a "0 candidates" line doesn't
   * flash on the empty-state screen. */
  function setQueryMetrics({ nResults, tMs, total }) {
    const host = $("queryMetrics");
    if (!host) return;
    if (nResults == null && total == null) {
      host.hidden = true;
      return;
    }
    const fmt = (n) => {
      if (n == null) return "—";
      if (n >= 1_000_000)
        return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
      if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "k";
      return String(n);
    };
    $("qmResults").textContent =
      nResults == null ? "" : `${nResults} result${nResults === 1 ? "" : "s"}`;
    $("qmTime").textContent = tMs == null ? "" : `${Math.round(tMs)} ms`;
    $("qmCands").textContent =
      total == null ? "" : `${fmt(total)} candidate${total === 1 ? "" : "s"}`;
    host.hidden = false;

    // Behavioural tracking — log every completed search. Latency clamps
    // to int32 since the column is `latency_ms INTEGER`. Result count
    // clamps to smallint range so freak overflows don't blow up the
    // server validator.
    if (window.kn && state.query) {
      window.kn.setLastQuery(state.query);
      window.kn.track("search", {
        query: state.query,
        result_count: Math.min(32767, nResults ?? 0),
        latency_ms: tMs == null ? null : Math.min(2147483647, Math.round(tMs)),
      });
    }
  }

  /* Sum of `documentCount` across the personalities currently in
   * `state.libs`. Falls back to 0 when the field isn't populated
   * (e.g. mid-load). Used to fill the "{N} candidates" segment in
   * the metrics pill above the results. */
  function candidatePoolSize() {
    let total = 0;
    for (const slug of state.libs) {
      const p = state.allPersonalities.find((x) => x.slug === slug);
      if (p && typeof p.documentCount === "number") total += p.documentCount;
    }
    return total;
  }

  /* ── Infinite scroll ─────────────────────────────────────────────
   *
   * One IntersectionObserver watches a sentinel below #results. When
   * the user scrolls it into view, we extend the current view by one
   * page. Two modes:
   *
   *   Feed (libs.size===0, no query) → GET /api/timeline?before=<oldest>
   *                                    with the same source/tag filters
   *                                    as the initial load.
   *   Personal page (libs.size===1, no query) → K.latest with the
   *                                    composed filter PLUS
   *                                    `date < ?` cursor.
   *
   * Search results aren't paginated yet — the top-K from ColBERT is
   * the full set we surface.
   */
  let _scrollPagingBusy = false;
  let _scrollObserver = null;
  let _scrollListenerWired = false;
  /* Drives infinite scroll on the feed + personal pages.
   *
   * Two redundant triggers, because a bare IntersectionObserver
   * proved flaky in some hosting contexts (the callback would
   * silently never fire even with the sentinel inside the viewport):
   *   1. IO on `#loadMoreSentinel` with a wide rootMargin
   *      (pre-fetches before the user actually hits bottom).
   *   2. A plain scroll-event listener that measures the document
   *      and fires when the user is within a screen-and-a-half of
   *      the bottom.
   *
   * Both funnel through the same `tryLoadMore` guard so we never
   * double-fire while a fetch is in flight. */
  function armInfiniteScroll() {
    const sentinel = $("loadMoreSentinel");
    if (!sentinel) return;
    const eligible =
      // Feed, personal page, or feed-search: each has a working
      // pagination path in loadMoreDocs.
      (state.libs.size === 0 || (state.libs.size === 1 && !state.query)) &&
      state.lastDocs &&
      state.lastDocs.length > 0;
    if (!eligible) {
      sentinel.hidden = true;
      return;
    }
    sentinel.hidden = false;
    async function tryLoadMore() {
      if (_scrollPagingBusy) return;
      if (sentinel.hidden) return;
      _scrollPagingBusy = true;
      // Flip the spinner on. The CSS keys on the attribute presence
      // (`#loadMoreSentinel[aria-busy="true"]`) and fades in the
      // spinner + "Loading more…" label.
      sentinel.setAttribute("aria-busy", "true");
      try {
        await loadMoreDocs();
        // The list grew — if the user is still near the bottom
        // (e.g. they scrolled past one screen of new content)
        // keep paging. Bounded so a single gesture can't cascade
        // forever.
        for (let i = 0; i < 4; i++) {
          if (sentinel.hidden) break;
          const r = sentinel.getBoundingClientRect();
          const within =
            r.top <
            (window.innerHeight || document.documentElement.clientHeight) +
              1500;
          if (!within) break;
          await loadMoreDocs();
        }
      } finally {
        _scrollPagingBusy = false;
        // Fade the spinner out. If the sentinel got hidden because
        // we ran out of rows the attribute removal is moot (the
        // element is `display:none`) but we still clear it for
        // cleanliness so the next load comes back to a clean state.
        sentinel.removeAttribute("aria-busy");
      }
    }
    if (!_scrollObserver && "IntersectionObserver" in window) {
      _scrollObserver = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) tryLoadMore();
        },
        // Pre-fetch ~2 full mobile viewports before the user reaches
        // the bottom (was 1200px). On a slow cold timeline (~1 s
        // TTFB) this gives the new batch enough head start that the
        // user's continued scroll lands on already-rendered cards
        // instead of an empty space.
        { rootMargin: "2400px 0px" },
      );
      _scrollObserver.observe(sentinel);
    }
    if (!_scrollListenerWired) {
      _scrollListenerWired = true;
      const scroller = document.scrollingElement || document.documentElement;
      const check = () => {
        if (sentinel.hidden) return;
        const viewport = window.innerHeight || scroller.clientHeight;
        const scrollTop = window.scrollY || scroller.scrollTop || 0;
        const slack = scroller.scrollHeight - (scrollTop + viewport);
        // Fire when we're within ~2.5 screens of the bottom (was 1.5).
        // Earlier trigger = the next batch is in flight by the time
        // the user actually approaches the bottom, so they see a
        // continuous scroll instead of a pause + render.
        if (slack < viewport * 2.5) {
          tryLoadMore();
        }
      };
      const handler = () => check();
      // Plain scroll/resize listeners — no rAF coalescing so a single
      // programmatic scroll that updates `scrollTop` without dispatching
      // a sequence of events still gets a check on the next event.
      window.addEventListener("scroll", handler, { passive: true });
      window.addEventListener("resize", handler, { passive: true });
      document.addEventListener("scroll", handler, { passive: true });
      // Belt-and-suspenders polling. IntersectionObserver + scroll
      // events should be enough on a real device, but some hosting
      // contexts (controlled Chromium, iframe sandboxes) silently
      // drop one or both. A 600 ms tick is cheap and guarantees the
      // page never stalls at the bottom waiting for an event that
      // never fires.
      setInterval(check, 600);
      check();
    }
  }

  /* Feed minimum-visible guard. After the initial render and after
   * each infinite-scroll batch, count the cards actually visible in
   * the DOM (i.e. not folded behind a "See N more" pill). If we're
   * below the floor, keep firing loadMoreDocs() until we cross the
   * threshold or the server stops yielding new rows. Prevents the
   * "single HF run collapsed → only 6 cards visible" trap. */
  const FEED_MIN_VISIBLE = 50;
  let _ensureVisibleBusy = false;
  async function ensureMinVisibleOnFeed() {
    if (state.libs.size !== 0) return;
    if (state.query) return;
    if (_ensureVisibleBusy) return;
    _ensureVisibleBusy = true;
    try {
      // Bounded retry — at 50 rows per batch the worst case is
      // ~20 calls (1000 hidden + 50 visible). The break-out below
      // catches the "no more rows" case earlier.
      //
      // Render suppression: each `loadMoreDocs()` in feed mode used
      // to do a full `results.innerHTML = …` rebuild on every
      // iteration, which read as a visible blink on first load
      // (5–10 successive rebuilds in tens of milliseconds). We
      // suppress the per-iteration repaint via a flag and trigger
      // ONE final repaint once the loop has settled — the visible
      // result is identical, only the DOM churn is gone.
      _suppressFeedRepaint = true;
      let didLoad = false;
      try {
        for (let i = 0; i < 20; i++) {
          const visible =
            document.querySelectorAll("#results > article").length;
          if (visible >= FEED_MIN_VISIBLE) break;
          const lenBefore = state.lastDocs ? state.lastDocs.length : 0;
          await loadMoreDocs();
          const lenAfter = state.lastDocs ? state.lastDocs.length : 0;
          if (lenAfter === lenBefore) break;
          didLoad = true;
        }
      } finally {
        _suppressFeedRepaint = false;
      }
      // Final, single repaint with everything we've gathered.
      if (didLoad && state.lastDocs && state.lastDocs.length) {
        const sorted = reorderFeed(state.lastDocs);
        state.lastDocs = sorted;
        $("results").innerHTML = renderFeedDocsHtml(sorted);
        wireResults();
        wireFeedCollapse();
        armManualCollapseButtons();
        mergeAdjacentCollapsePills();
        armInfiniteScroll();
        markShownUrls(sorted);
      }
    } finally {
      _ensureVisibleBusy = false;
    }
  }
  // Flag respected by the feed branch of `loadMoreDocs`: when on,
  // it appends to `state.lastDocs` but skips the DOM repaint, so
  // batched `ensureMinVisibleOnFeed` loops only touch the DOM once.
  let _suppressFeedRepaint = false;

  async function loadMoreDocs() {
    if (!state.lastDocs?.length) return;
    let extra = [];
    // ── Search-mode pagination (feed search) ──────────────────
    // We page by re-running ColBERT with a larger `top_k` and
    // taking everything past the current `lastDocs` size. ColBERT
    // returns results in relevance order, so the new slab is the
    // next-best matches. Filter conditions (source / tag / date)
    // already ride inside `buildIndexFilter`, so the same SQL
    // pre-filter applied to the initial query is reapplied here —
    // no JS post-filter.
    if (state.query && state.libs.size === 0) {
      const currentCount = state.lastDocs.length;
      const requestK = Math.min(
        2000,
        Math.max(200, Math.ceil(currentCount * 2 + 80)),
      );
      let raw = [];
      try {
        raw = await K.search({
          indexName: ALL_INDEX_NAME,
          query: state.query,
          topK: requestK,
          filter: buildIndexFilter(),
        });
      } catch {
        return;
      }
      // Same scope filter the initial search applies.
      const scope =
        me && state.followingOnly
          ? new Set([...(_peopleRail?.following || []), me.slug])
          : null;
      if (scope) raw = raw.filter((d) => scope.has(d.owner));
      // Group by URL just like the initial search (avoids double
      // rows when the same doc lives in multiple followee libraries).
      const byUrl = new Map();
      for (const d of raw) {
        const ex = byUrl.get(d.url);
        if (ex) {
          if (d.owner && !ex._owners.includes(d.owner))
            ex._owners.push(d.owner);
          if ((d.similarity || 0) > (ex.similarity || 0))
            ex.similarity = d.similarity;
        } else {
          byUrl.set(d.url, { ...d, _owners: d.owner ? [d.owner] : [] });
        }
      }
      const merged = [...byUrl.values()];
      // Dedup against BOTH the currently-visible docs and the
      // session-wide `shownUrls` set. The latter matters after a
      // "More" click replaces state.lastDocs — the previously-shown
      // docs are no longer in state.lastDocs but the user did see
      // them; infinite-scroll bringing them back here would surface
      // duplicates.
      const seen = new Set(state.lastDocs.map((d) => d.url));
      extra = merged.filter(
        (d) => d.url && !seen.has(d.url) && !state.shownUrls.has(d.url),
      );
      if (!extra.length) {
        $("loadMoreSentinel").hidden = true;
        if (_scrollObserver) {
          _scrollObserver.disconnect();
          _scrollObserver = null;
        }
        return;
      }
      // Cap at 60 new rows per pull so the page doesn't grow by a
      // huge slab on a single trigger.
      extra = extra.slice(0, 60);
      state.lastDocs = state.lastDocs.concat(extra);
      markShownUrls(extra);
      const results = $("results");
      results.insertAdjacentHTML("beforeend", extra.map(renderResult).join(""));
      wireResults();
      armInfiniteScroll();
      return;
    }

    // ── Non-search cursor pagination (feed / personal page) ───
    // The list may end with one or two date-less docs (e.g. a
    // bookmark whose source didn't populate the field). Walk
    // backwards to find the most recent dated row and use *its*
    // date as the cursor so pagination never dead-ends prematurely.
    let before = null;
    for (let i = state.lastDocs.length - 1; i >= 0; i--) {
      const d = state.lastDocs[i];
      if (d && d.date) {
        before = d.date;
        break;
      }
    }
    if (!before) return;
    if (state.libs.size === 0) {
      // Feed timeline cursor pagination.
      const srcs = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
      const excl = [...state.excludedSources];
      const tags = [...state.tags];
      // Match the initial timeline limit (75) so a paginated batch
      // feels as substantial as the first paint.
      const qs = new URLSearchParams({ limit: "75", before });
      if (srcs.length) qs.set("sources", srcs.join(","));
      if (excl.length) qs.set("exclude_sources", excl.join(","));
      if (tags.length) qs.set("tags", tags.join(","));
      // Date-range pre-filter — re-applied at the SQL layer so the
      // pagination cursor honours "past week / past month / past
      // year" without a JS post-filter.
      const sinceDate = _sinceDateString(state.dateSince);
      if (sinceDate) qs.set("since", sinceDate);
      // Same multi-slug category filter the initial loader honours
      // — keeps cursor-paginated rows narrowed to the picked set.
      if (state.categories && state.categories.size) {
        qs.set("category", [...state.categories].join(","));
      }
      // Same hide-seen contract as the initial loader — pagination
      // must respect the chip state, otherwise scrolling resurrects
      // every card the user has already seen.
      if (state.showSeen) qs.set("include_seen", "true");
      try {
        const r = await fetch(`${API_BASE}/api/timeline?${qs.toString()}`, {
          credentials: "include",
        });
        if (!r.ok) return;
        const docs = await r.json();
        extra = (Array.isArray(docs) ? docs : []).map((d) => {
          const sharers = Array.isArray(d.sharers) ? d.sharers : [];
          for (const s of sharers) {
            if (s && s.slug && !state.perSlugMeta[s.slug]) {
              state.perSlugMeta[s.slug] = {
                slug: s.slug,
                name: s.name || s.slug,
                avatar: s.avatar || null,
                twitterFollowers: s.twitterFollowers || 0,
              };
            }
          }
          return {
            url: d.url,
            title: d.title,
            summary: d.summary || "",
            date: d.date || "",
            tags: Array.isArray(d.tags) ? d.tags : [],
            extraTags: [],
            source: d.source || "",
            source_url: d.source_url || null,
            // Same link-cluster passthrough as the initial timeline
            // mapper — otherwise pagination would render plain text
            // tweets where the first page renders rich previews.
            linkedUrls: Array.isArray(d.linked_urls) ? d.linked_urls : [],
            linkHosts: Array.isArray(d.link_hosts) ? d.link_hosts : [],
            sharers,
            sharerCount: d.sharerCount || sharers.length,
            _owners: sharers.map((s) => s.slug).filter(Boolean),
          };
        });
      } catch {
        return;
      }
    } else if (state.libs.size === 1) {
      // Personal-page browse pagination — refetch the full PG library
      // (free from the 30s memoised cache after the initial load) and
      // slice out the slab that follows the docs already on screen.
      // Same PG ordering as the initial paint (date DESC, created_at
      // DESC), so the cursor is just "everything we haven't shown yet".
      const slug = [...state.libs][0];
      const sourcesArr = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
      const excludeArr = [...state.excludedSources];
      const tagsArr = [...state.tags];
      const favOn = state.sources.has(FAV_SOURCE_KEY);
      const urlsArr = favOn ? [...state.favorites] : [];
      // Same Topics narrowing as the initial paint — see the matching
      // block under `libs.length === 1` in refresh(). Without this the
      // first 60 cards would respect the category filter but
      // infinite-scroll fetches would silently widen back to the
      // full library.
      const catsArr = state.categories ? [...state.categories] : [];
      try {
        const all = await K.getPersonalPageDocuments(slug, {
          sources: sourcesArr,
          excludeSources: excludeArr,
          tags: tagsArr,
          urls: urlsArr,
          categories: catsArr,
        });
        const shown = new Set(state.lastDocs.map((d) => d.url));
        // 75 to match the feed-pagination batch size (was 50). The
        // backing list is already in memory (single GET earlier in
        // the page lifecycle), so this is just a slice — no extra
        // network cost.
        extra = all
          .filter((d) => !shown.has(d.url))
          .slice(0, 75)
          .map((d) => ({ ...d, _from: slug, _owners: [slug] }));
      } catch {
        return;
      }
    } else {
      return;
    }
    if (!extra.length) {
      // Out of rows — stop further triggers.
      $("loadMoreSentinel").hidden = true;
      if (_scrollObserver) {
        _scrollObserver.disconnect();
        _scrollObserver = null;
      }
      return;
    }
    // Dedup against BOTH the currently-visible docs and the
    // session-wide `shownUrls` set. The latter matters after a
    // "More" click replaces state.lastDocs — previously-shown docs
    // are no longer in state.lastDocs, but the user did see them;
    // re-appending them here would surface duplicates.
    const seen = new Set(state.lastDocs.map((d) => d.url));
    let fresh = extra.filter(
      (d) => d.url && !seen.has(d.url) && !state.shownUrls.has(d.url),
    );
    if (!fresh.length) {
      // The server returned rows but they're all already on the
      // page. Could happen at the very edge of the dataset (e.g.
      // we asked `before=2026-05-12` and got back exactly the
      // 2026-05-12 row we already had). Stop paging — there's
      // nothing new to surface and re-firing with the same cursor
      // would loop forever.
      $("loadMoreSentinel").hidden = true;
      if (_scrollObserver) {
        _scrollObserver.disconnect();
        _scrollObserver = null;
      }
      return;
    }
    // Same diversity pass as the initial load on the feed — date is
    // the primary key, source-bucket round-robin reorders ties.
    state.lastDocs = state.lastDocs.concat(fresh);
    markShownUrls(fresh);
    const results = $("results");
    if (state.libs.size === 0) {
      // Feed mode: re-sort and re-render the full visible list so a
      // same-source run that spans the batch boundary collapses into
      // one pill instead of one-pill-per-batch. innerHTML replacement
      // preserves scroll position (height only grows) and the open
      // similar-panels are restored by restoreOpenSimilarPanels().
      //
      // When `ensureMinVisibleOnFeed` is running its top-up loop it
      // sets `_suppressFeedRepaint` so we accumulate fresh docs
      // without paying for a DOM rebuild per iteration; the loop
      // itself does the single final repaint at the end.
      if (_suppressFeedRepaint) return;
      const sorted = reorderFeed(state.lastDocs);
      state.lastDocs = sorted;
      results.innerHTML = renderFeedDocsHtml(sorted);
      wireResults();
      wireFeedCollapse();
      armManualCollapseButtons();
      mergeAdjacentCollapsePills();
      armInfiniteScroll();
      syncShuffleButton();
      markShownUrls(sorted);
    } else {
      results.insertAdjacentHTML("beforeend", fresh.map(renderResult).join(""));
      wireResults();
      // Re-arm the observer in case the previous batch hit a
      // duplicate-only response and disconnected.
      armInfiniteScroll();
    }
  }

  function showResultsSpinner() {
    const el = $("resultsLoading");
    if (el) el.hidden = false;
  }
  function hideResultsSpinner() {
    const el = $("resultsLoading");
    if (el) el.hidden = true;
  }
  function showSrcSpinner() {
    const el = $("srcLoading");
    if (el) el.hidden = false;
  }
  function hideSrcSpinner() {
    const el = $("srcLoading");
    if (el) el.hidden = true;
  }
  function showPeopleSpinner() {
    const el = $("peopleLoading");
    if (el) el.hidden = false;
  }
  function hidePeopleSpinner() {
    const el = $("peopleLoading");
    if (el) el.hidden = true;
  }

  /* Restore the default "No results." text/styling on the shared
   * empty-state element. We swap its content for the onboarding card
   * below, so every other empty path has to call this before showing
   * the generic empty pill again. */
  function resetEmptyMessage() {
    const empty = $("empty");
    if (!empty) return;
    empty.classList.remove("empty-onboarding");
    empty.innerHTML = "No results.";
  }

  /* Onboarding empty-state shown on a logged-in user's OWN personal
   * page when they have zero documents. The /profile page is where
   * they connect GitHub / X / Zotero / HuggingFace etc., so we point
   * them there with a clear primary CTA. Replaces the generic
   * "No results." pill for this specific case. */
  function renderPersonalEmptyOnboarding() {
    const empty = $("empty");
    if (!empty) return;
    empty.style.display = "";
    empty.classList.add("empty-onboarding");
    empty.innerHTML = `
      <div class="onboarding-card">
        <h2>Your library is empty</h2>
        <p>
          Connect your sources to start building your knowledge base.
          Add GitHub, X, Zotero, HuggingFace, or any blog feed — we'll
          pull your bookmarks, stars, and papers into one searchable
          library.
        </p>
        <a class="onboarding-cta" href="/profile">
          Open settings
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round" aria-hidden="true">
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="13 6 19 12 13 18" />
          </svg>
        </a>
      </div>`;
  }

  /* Personal-page live watcher.
   *
   * Out-of-band ingestion (the launchd twitter feeder writing
   * directly to prod PG) never touches our web caches, so newly
   * parsed tweets don't surface until the user reloads. Poll the
   * personal-docs endpoint every 10s while the tab is visible and
   * a personal page (`libs.size === 1`, no query) is on screen.
   * On the first poll where new URLs appear we run `refresh()` to
   * repaint — which in turn restarts the watcher with the new
   * state.lastDocs as the "already seen" baseline.
   *
   * Stopped on:
   *   • navigation (libs change)
   *   • search activation (state.query set)
   *   • tab hidden (we'd burn requests for nothing)
   *   • duplicate start (the helper clears any prior timer first)
   */
  const PERSONAL_WATCH_INTERVAL_MS = 10_000;
  let _personalWatchTimer = null;
  let _personalWatchSlug = null;
  function stopPersonalPageWatch() {
    if (_personalWatchTimer) {
      clearInterval(_personalWatchTimer);
      _personalWatchTimer = null;
    }
    _personalWatchSlug = null;
  }
  async function _pollPersonalPage(slug) {
    // Re-check the world before each fetch — the user may have
    // navigated, started a search, or hidden the tab since the
    // interval fired.
    if (document.visibilityState !== "visible") return;
    if (
      !state.libs ||
      state.libs.size !== 1 ||
      ![...state.libs].includes(slug)
    ) {
      stopPersonalPageWatch();
      return;
    }
    if (state.query) return;
    // Bypass the in-memory + sessionStorage caches so the poll
    // actually reaches PG. The endpoint already returns
    // (date DESC, created_at DESC) so the top of the list is the
    // freshest content.
    try {
      K.invalidatePersonalDocs?.(slug);
      const sourcesArr = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
      const excludeArr = [...state.excludedSources];
      const tagsArr = [...state.tags];
      const favOn = state.sources.has(FAV_SOURCE_KEY);
      const urlsArr = favOn ? [...state.favorites] : [];
      const catsArr = state.categories ? [...state.categories] : [];
      const next = await K.getPersonalPageDocuments(slug, {
        sources: sourcesArr,
        excludeSources: excludeArr,
        tags: tagsArr,
        urls: urlsArr,
        categories: catsArr,
      });
      const haveUrls = new Set((state.lastDocs || []).map((d) => d.url));
      let anyNew = false;
      for (const d of next) {
        if (d && d.url && !haveUrls.has(d.url)) {
          anyNew = true;
          break;
        }
      }
      if (anyNew) {
        // Re-render in place. `refresh()` will hit the freshly
        // populated cache (we just wrote it via getPersonal...),
        // so no second round-trip; just a paint.
        refresh();
      }
    } catch {
      /* silent — next tick retries */
    }
  }
  function startPersonalPageWatch(slug) {
    stopPersonalPageWatch();
    if (!slug) return;
    _personalWatchSlug = slug;
    _personalWatchTimer = setInterval(
      () => _pollPersonalPage(slug),
      PERSONAL_WATCH_INTERVAL_MS,
    );
  }
  // Visibility-aware: pause polling cost when the tab is backgrounded,
  // and as soon as it returns trigger an immediate refresh so the
  // user doesn't have to wait up to 10 s for the next tick.
  document.addEventListener("visibilitychange", () => {
    if (
      document.visibilityState === "visible" &&
      _personalWatchSlug &&
      state.libs?.size === 1
    ) {
      _pollPersonalPage(_personalWatchSlug);
    }
  });

  async function refresh() {
    const my = ++reqId;
    const _refreshT0 = performance.now();
    // Keep the Topics picker in sync with the active library
    // selection: on a personal page we only want to surface the
    // categories that lib's documents actually fall into, never the
    // full 178-row catalogue. Cheap when the cache is warm
    // (synchronous flip + re-render), one extra fetch when the user
    // navigates between routes. Fire-and-forget so the rest of
    // refresh() keeps running in parallel — the picker is on a
    // different surface and doesn't gate the main result render.
    refreshCatalogueForActiveUser();
    // Pre-fetch the URL set for the active category selection so
    // buildIndexFilter (sync) can embed it as a `url IN (…)`
    // clause. Awaited up-front so every search call in this refresh
    // — initial query, pagination, hint route — sees the same set
    // without each having to re-fetch. The fetchUrls helper caches
    // by sorted slug-key so repeated refreshes with the same set
    // are free after the first.
    if (state.categories && state.categories.size) {
      _categoryUrlSet = await fetchUrlsForSelectedCategories();
      if (my !== reqId) return;
    } else {
      _categoryUrlSet = null;
    }
    // A refresh means new content from the API — drop the
    // user-shuffled flag so the new docs flow back through the
    // normal date-desc + source-cluster reorder, and reset the
    // shown-URLs set so a fresh "More" cycle has a full pool to
    // draw from.
    state.feedShuffled = false;
    state.shownUrls = new Set();
    // Keep the "Following only" toggle's visibility in lockstep with
    // (query active × on the feed). Cheap and idempotent.
    syncFollowingOnlyButton();
    syncShowSeenButton();
    syncShuffleButton();
    // Same idea for the mobile chrome — keep the active tab + the
    // source-count badge tracking state.libs / state.sources.
    window._syncMobileChrome?.();
    // Invalidate any in-flight rerank stream from the previous
    // refresh: the worker keeps emitting `rank-update` payloads
    // tagged with the old queryId until it receives the next
    // `rank` message (which only happens at the end of *this*
    // refresh, after the await). Bumping rerankQueryId here means
    // those late events fail the `queryId === rerankQueryId` guard
    // in onmessage and won't paint stale docs back over the
    // freshly-filtered list. We also cancel any pending rAF render
    // so a queued stale frame can't slip past either.
    rerankQueryId++;
    if (rerankPending !== null) {
      cancelAnimationFrame(rerankPending);
      rerankPending = null;
    }
    // Same dance for the page-meta enhancer: bumping the token
    // tells the in-flight enhancement loop to bail out, so a
    // stale fetch doesn't insert an enriched summary into a card
    // that's no longer on screen.
    enhanceToken++;
    // New refresh = new result set, so any open similar panels
    // belong to the previous list. Clear the open-state tracker
    // so the rerank restorer doesn't keep stale URLs alive.
    openSimilarUrls.clear();
    $("results").innerHTML = "";
    $("empty").style.display = "none";
    // Show the spinner now that the result list is empty and we're
    // about to fetch. Hidden again by every render path below
    // (success or empty). Skip on tiny refreshes (e.g. a tag toggle
    // when state.lastDocs already had docs we're about to replace)
    // — those re-renders are synchronous and don't need a spinner.
    showResultsSpinner();
    $("sortToggle")?.classList?.toggle("show", !!state.query);
    renderActiveTags();
    /* Render the result-sources strip with the *previous* doc set so
     * the strip doesn't flicker out during the in-flight fetch.
     * Re-rendered with fresh data once the new docs land. */
    renderResultSources();
    const libs = [...state.libs];
    /* Zero libraries → this is "feed" mode. Two behaviours:
     *
     *   ── No query → render the follow-graph timeline (recent docs
     *      from followees + the caller's own library). Anonymous
     *      callers with no follows fall back to the noLibs template.
     *
     *   ── With a query → run a ColBERT search on the cross-library
     *      __all__ index, then keep only docs whose owner is in
     *      (followees ∪ self) so the feed search stays scoped to
     *      the people you actually follow. The user explicitly
     *      asked for this: "search using __all__ + my index".
     */
    if (libs.length === 0) {
      showProfileHeader(false);

      // Favourites chip on the feed → bypass /api/timeline and pull
      // the user's full favourited set straight from the hydrated
      // endpoint. Favourites are personal and may span libraries the
      // user doesn't follow, so the timeline scope (followees + self)
      // would miss them.
      if (state.sources.has(FAV_SOURCE_KEY) && !state.query) {
        try {
          let favDocs = await K.getFavoriteDocs();
          if (my !== reqId) return;
          // Apply any extra source/tag filters the user layered on top.
          const srcs = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
          const excl = state.excludedSources;
          const tags = [...state.tags];
          favDocs = favDocs.filter((d) => {
            if (srcs.length && !srcs.includes(d.source || "")) return false;
            if (excl.size && excl.has(d.source || "")) return false;
            if (tags.length) {
              const docTags = (d.tags || []).map((t) => t.toLowerCase());
              for (const t of tags)
                if (!docTags.includes(t.toLowerCase())) return false;
            }
            return true;
          });
          // Sort: same date-desc + diversity recipe as the regular feed.
          favDocs = reorderFeed(favDocs);
          state.lastDocs = favDocs;
          renderResultSources();
          $("resultCount").textContent =
            `${favDocs.length} upvote${favDocs.length === 1 ? "" : "s"}`;
          $("resultCount").hidden = false;
          setQueryMetrics({
            nResults: favDocs.length,
            tMs: performance.now() - _refreshT0,
            total: favDocs.length,
          });
          if (!favDocs.length) {
            $("results").innerHTML = "";
            $("empty").style.display = "";
            hideResultsSpinner();
          } else {
            $("empty").style.display = "none";
            $("results").innerHTML = renderFeedDocsHtml(favDocs);
            wireResults();
            wireFeedCollapse();
            armManualCollapseButtons();
            armInfiniteScroll();
          }
          rebuildAllSourcesForFeed().then(renderSrc);
          return;
        } catch (e) {
          console.warn("[feed-favs]", e);
          // Fall through to the regular timeline path on failure.
        }
      }

      if (state.query) {
        // Feed search defaults to the global library set so non-
        // followed sharers can surface (with the faded avatar +
        // Follow popover) and discovery stays open. The "Following
        // only" toggle next to the date filter narrows the scope to
        // followees + self when the user wants a focused view.
        const scope =
          me && state.followingOnly
            ? new Set([...(_peopleRail?.following || []), me.slug])
            : null;
        let docs = [];
        try {
          // Pre-filter at the index level — sources, excluded sources,
          // tags. Pushed down so the ColBERT scorer only sees rows that
          // already match the user's chip selection.
          docs = await K.search({
            indexName: ALL_INDEX_NAME,
            query: state.query,
            topK: 200,
            filter: buildIndexFilter(),
          });
        } catch (e) {
          console.warn("[feed-search]", e);
        }
        if (my !== reqId) return;
        if (scope) docs = docs.filter((d) => scope.has(d.owner));

        // Group by URL so multiple owners of the same doc surface as
        // one row with a stacked avatar list. The __all__ index can
        // return one entry per (user, url) pair — collapse those.
        const byUrl = new Map();
        for (const d of docs) {
          const ex = byUrl.get(d.url);
          if (ex) {
            if (!ex._owners.includes(d.owner)) ex._owners.push(d.owner);
            // Keep the highest similarity score across rows.
            if ((d.similarity || 0) > (ex.similarity || 0)) {
              ex.similarity = d.similarity;
            }
          } else {
            byUrl.set(d.url, { ...d, _owners: d.owner ? [d.owner] : [] });
          }
        }
        const merged = [...byUrl.values()];
        // Category post-filter — when the user has one or more
        // categories selected, keep only docs that fell inside the
        // pre-fetched URL set for those slugs. The URL set is
        // already in memory because `buildIndexFilter` consumed it
        // moments ago to constrain the ColBERT query; we just look
        // it up again here through the same cached helper.
        let filteredByCat = merged;
        if (state.categories && state.categories.size) {
          const urls = await fetchUrlsForSelectedCategories();
          if (my !== reqId) return;
          filteredByCat = filterDocsByUrlSet(merged, new Set(urls));
        }
        const capped = filteredByCat.slice(0, 60);
        showFollowingHeader(false);
        state.lastDocs = capped;
        renderResultSources();
        $("resultCount").textContent =
          `${capped.length} ${capped.length === 1 ? "result" : "results"} in your feed`;
        $("resultCount").hidden = false;
        setQueryMetrics({
          nResults: capped.length,
          tMs: performance.now() - _refreshT0,
          total: docs.length,
        });
        if (!capped.length) {
          $("results").innerHTML = "";
          $("empty").style.display = "";
          hideResultsSpinner();
          return;
        }
        $("empty").style.display = "none";
        $("results").innerHTML = renderFeedDocsHtml(capped);
        wireResults();
        wireFeedCollapse();
        armManualCollapseButtons();
        mergeAdjacentCollapsePills();
        // Search-mode infinite scroll — the next batch comes from
        // re-running ColBERT at a larger top_k and slicing past
        // the rows already on the page (see `loadMoreDocs`).
        armInfiniteScroll();
        postRerank(capped);
        // Feed mode: aggregate per-source totals across (me + followees)
        // so the rail mirrors a personal page's filter behaviour.
        rebuildAllSourcesForFeed().then(renderSrc);
        return;
      }

      // No query → the timeline.
      const tlDocs = await loadFollowingTimeline();
      if (my !== reqId) return;
      if (tlDocs && tlDocs.length) {
        showFollowingHeader(true);
        // Populate the Sources panel from the union of (me + followees)
        // source lists, with counts. Async — re-renders the rail when
        // the followee fetches land.
        rebuildAllSourcesForFeed().then(renderSrc);
        // Source/tag filters are pushed into /api/timeline directly,
        // so tlDocs is already filtered — no client-side narrowing.
        // Apply the feed diversity reorder: date desc primary, source
        // round-robin within each date so a 10-tweet day still shows
        // the lone HuggingFace card on it instead of burying it.
        const filtered = reorderFeed(tlDocs);
        state.lastDocs = filtered;
        renderResultSources();
        $("resultCount").textContent = me
          ? `${filtered.length} from people you follow`
          : `${filtered.length} from featured libraries`;
        $("resultCount").hidden = false;
        setQueryMetrics({
          nResults: filtered.length,
          tMs: performance.now() - _refreshT0,
          total: tlDocs.length,
        });
        if (!filtered.length) {
          $("results").innerHTML = "";
          $("empty").style.display = "";
          hideResultsSpinner();
        } else {
          $("empty").style.display = "none";
          $("results").innerHTML = renderFeedDocsHtml(filtered);
          wireResults();
          wireFeedCollapse();
          armManualCollapseButtons();
          mergeAdjacentCollapsePills();
          armInfiniteScroll();
          syncShuffleButton();
          markShownUrls(filtered);
          // Top up until at least FEED_MIN_VISIBLE cards are unfolded.
          ensureMinVisibleOnFeed();
        }
        return;
      }
      showFollowingHeader(false);
      // No timeline docs. When the cause is "signed in but following
      // nobody yet", show the same discover-people panel the
      // personal page renders — it's the most useful next action.
      // Filter-narrowed empties still get the plain "no results" pill.
      $("results").innerHTML = "";
      hideResultsSpinner();
      setQueryMetrics({ nResults: 0, tMs: null, total: 0 });
      const noFilters =
        !state.query &&
        !state.tags.size &&
        !state.sources.size &&
        !state.excludedSources.size;
      if (me && noFilters && window.KnowledgeOnboarding) {
        const followsSet = await loadFollowingSet();
        if (followsSet.size === 0) {
          window.KnowledgeOnboarding.open({
            personalities: state.allPersonalities,
            apiBase: API_BASE,
            // First-run flow → show the welcome intro before the
            // picker. The Discover overlay (existing user clicking
            // "Discover Peoples") is the only place that should skip
            // straight to the categories.
            mode: "onboard",
            onSkip: () => {
              resetEmptyMessage();
              $("empty").style.display = "";
            },
          });
          return;
        }
      }
      resetEmptyMessage();
      $("empty").style.display = "";
      return;
    }
    showFollowingHeader(false);
    showProfileHeader();
    let docs = [];
    // One filter, used by every pool: index-side condition for
    // search + latest, query params for the unindexed PG endpoint.
    const filter = buildIndexFilter();
    // Strip the synthetic Favorites key from the source list we
    // pass to the unindexed endpoint — that key isn't a real
    // `documents.source` value. The favorites pre-filter is
    // applied via the `urls` param instead.
    const sourcesArr = [...state.sources].filter((s) => s !== FAV_SOURCE_KEY);
    const excludeArr = [...state.excludedSources];
    const tagsArr = [...state.tags];
    const favOn = state.sources.has(FAV_SOURCE_KEY);
    const urlsArr = favOn ? [...state.favorites] : [];
    try {
      // Browse-mode shortcut: when the Favorites filter is on (and no
      // query/tag/source narrows it further), bypass the per-lib
      // fanout and pull every favorited doc straight from the hydrated
      // endpoint. Without this, stars whose owning library isn't in
      // the current selection silently disappear from the result list.
      if (
        favOn &&
        !state.query &&
        sourcesArr.length === 0 &&
        tagsArr.length === 0
      ) {
        const favDocs = await K.getFavoriteDocs();
        docs = favDocs;
        if (my !== reqId) return;
        state.lastDocs = docs;
        renderResultSources();
        $("resultCount").textContent =
          `${docs.length} result${docs.length === 1 ? "" : "s"}`;
        setQueryMetrics({
          nResults: docs.length,
          tMs: performance.now() - _refreshT0,
          total: state.favorites ? state.favorites.size : null,
        });
        if (docs.length === 0) {
          $("empty").style.display = "";
          hideResultsSpinner();
          return;
        }
        $("results").innerHTML = docs.map(renderResult).join("");
        wireResults();
        return;
      }
      if (state.query) {
        // When the Favorites filter is on, broaden the per-lib fanout
        // to include every library that owns one of the user's stars.
        // Without this, a query + fav filter would silently drop docs
        // whose owning library isn't in the current selection.
        const queryLibs =
          favOn && state.favoriteOwners
            ? Array.from(new Set([...libs, ...state.favoriteOwners]))
            : libs;

        // Routing:
        //   - ≥ ALL_INDEX_THRESHOLD selected → ONE `__all__` query,
        //     no per-slug fanout. Non-VIPs in the selection are
        //     skipped (they aren't in `__all__`, but at this scale
        //     hitting their indices individually would defeat the
        //     point of the threshold). The user explicitly opted
        //     into "fast" by picking a large library set.
        //   - Below threshold → per-lib fanout for everyone, the
        //     per-user indices are tighter than the merged one.
        const { vips, nonvips } = splitByVip(queryLibs);
        const allOnly = useAllOnly(queryLibs);
        const useAllIndex = vips.length >= ALL_INDEX_THRESHOLD;

        const tasks = [];
        if (useAllIndex && vips.length) {
          // Drop the `owner IN (…)` clause when the selection covers
          // every VIP `__all__` knows about — passing 133 placeholders
          // costs ~70ms server-side vs ~10ms unfiltered.
          const _fullVipCount = state.allPersonalities.filter(
            (p) => p.vip,
          ).length;
          const _coversAllVips =
            _fullVipCount > 0 && vips.length >= _fullVipCount;

          let composedFilter;
          if (_coversAllVips) {
            composedFilter = filter && filter.condition ? filter : null;
          } else {
            const ownerPlaceholders = vips.map(() => "?").join(",");
            const ownerClause = `owner IN (${ownerPlaceholders})`;
            composedFilter =
              filter && filter.condition
                ? {
                    condition: `(${filter.condition}) AND ${ownerClause}`,
                    parameters: [...(filter.parameters || []), ...vips],
                  }
                : { condition: ownerClause, parameters: [...vips] };
          }
          // topK widened so each owner still gets a fair share of
          // the result pool — at 60/N we'd starve when N is large.
          const allTopK = Math.min(
            300,
            60 * Math.max(1, Math.ceil(vips.length / 5)),
          );
          tasks.push(
            K.search({
              indexName: ALL_INDEX_NAME,
              query: state.query,
              topK: allTopK,
              filter: composedFilter,
            })
              .then((rs) =>
                rs.map((d) => ({
                  ...d,
                  // The `__all__` index carries the owner slug as
                  // metadata; fall back to slugless if missing so
                  // the dedupe + result render still work.
                  _from: d.owner || ALL_INDEX_NAME,
                })),
              )
              .catch(() => []),
          );
        }
        // When the selection is large (`allOnly`), drop non-VIP
        // fanout entirely — the threshold guarantees the user wants
        // a fast aggregate response, not a careful per-personality
        // merge. Below threshold, fall back to the existing logic.
        const perLibTargets = allOnly ? [] : useAllIndex ? nonvips : queryLibs;
        for (const s of perLibTargets) {
          tasks.push(
            K.search({ indexName: s, query: state.query, topK: 60, filter })
              .then((rs) => rs.map((d) => ({ ...d, _from: s })))
              .catch(() => []),
          );
        }
        const all = await Promise.all(tasks);
        // Dedup by URL but track every library that returned this doc
        // — gives the bottom-right "shared by" avatar stack its data.
        const map = new Map();
        for (const arr of all)
          for (const d of arr) {
            const ex = map.get(d.url);
            if (!ex) {
              map.set(d.url, { ...d, _owners: [d._from] });
            } else {
              if (!ex._owners.includes(d._from)) ex._owners.push(d._from);
              if ((d.similarity || 0) > (ex.similarity || 0)) {
                const owners = ex._owners;
                Object.assign(ex, d);
                ex._owners = owners;
              }
            }
          }
        docs = Array.from(map.values()).sort(
          (a, b) => (b.similarity || 0) - (a.similarity || 0),
        );
      } else if (libs.length === 1) {
        // Personal-page browse — skip ColBERT entirely and pull the
        // user's full library straight from Postgres. The PG endpoint
        // returns rows ordered by (date DESC, created_at DESC) and
        // carries `created_at` on every row, so same-day posts (very
        // common: the pipeline stamps `date` to today on a lot of
        // sources) sort by when the user actually saved them rather
        // than by SQLite-insertion order in the ColBERT sidecar.
        //
        // Topics filter is threaded through `categories=` so the same
        // Selection the user made on the feed narrows /<slug>. The
        // server JOINs against document_category_assignments — cleaner
        // than a client-side URL-list intersection that would blow
        // the query string for big categories.
        const slug = libs[0];
        const catsArr = state.categories ? [...state.categories] : [];
        const pgDocs = await K.getPersonalPageDocuments(slug, {
          sources: sourcesArr,
          excludeSources: excludeArr,
          tags: tagsArr,
          urls: urlsArr,
          categories: catsArr,
        });
        // Stamp `_from` so the renderer's owner-stack machinery (which
        // expects every doc to carry the library it came from) doesn't
        // miss this single-lib case.
        docs = pgDocs.map((d) => ({ ...d, _from: slug, _owners: [slug] }));
      } else {
        // Browse mode pulls THREE pools in parallel:
        //   - intersection (Postgres) — only when 2+ libs selected,
        //     finds URLs shared by EVERY active library.
        //   - indexed (ColBERT)        — latest by date. With ≥3 VIPs
        //     selected we route this through the unified `__all__`
        //     index (one filtered query) instead of fanning N
        //     parallel `latest` calls — the latter chokes the API
        //     pool when N is large. Non-VIPs always go per-user
        //     (they aren't in `__all__`).
        //   - unindexed (Postgres only) — freshly-synced docs awaiting
        //     `make run`. Per-user PG hits are cheap; we still fan
        //     these out, but only over a non-VIP cap so a "select
        //     all" doesn't spam 100+ unindexed lookups.
        const { vips: browseVips, nonvips: browseNonvips } = splitByVip(libs);
        const useAllForBrowse = browseVips.length >= ALL_INDEX_THRESHOLD;

        // The universe of VIPs `__all__` indexes — everyone with vip=true.
        // If the user's selection covers this universe, the
        // `owner IN (…)` clause is a no-op that just slows the
        // server (a 133-slug IN-list takes ~500ms vs ~50ms unfiltered).
        // Drop the filter entirely in that case.
        const fullVipCount = state.allPersonalities.filter((p) => p.vip).length;
        const coversAllVips =
          useAllForBrowse &&
          fullVipCount > 0 &&
          browseVips.length >= fullVipCount;

        // Compose the ColBERT-side filter for the latest pool. Two
        // pieces, ANDed together:
        //   1. existing source/tag filter (if any)
        //   2. `owner IN (…)` when we're targeting a subset of VIPs
        // The server-side `limit` on `/metadata/get` (wired up in
        // K.latest) keeps the response payload bounded regardless
        // of how many docs match.
        let allLatestCondition = filter ? filter.condition : null;
        let allLatestParams = filter ? [...(filter.parameters || [])] : [];
        if (useAllForBrowse && !coversAllVips) {
          const ownerPlaceholders = browseVips.map(() => "?").join(",");
          const ownerClause = `owner IN (${ownerPlaceholders})`;
          allLatestCondition = allLatestCondition
            ? `(${allLatestCondition}) AND ${ownerClause}`
            : ownerClause;
          allLatestParams = [...allLatestParams, ...browseVips];
        }

        // Build the indexed-pool tasks. Two routing modes:
        //
        //   • Above ALL_INDEX_THRESHOLD: ONE `__all__` query covers
        //     every selected VIP plus a per-user call for each
        //     non-VIP (they're not in `__all__`).
        //   • Below threshold: per-lib fanout for everyone —
        //     individual indices are tighter than the merged one,
        //     and the smaller selection makes the parallelism cheap.
        const indexedTasks = [];
        if (useAllForBrowse) {
          // Browse mode renders the most-recent pool across all
          // selected libraries. The view shows ~50-100 cards before
          // the user scrolls; pulling more is wasted bytes. Widen
          // modestly with selection size so a 30-lib selection still
          // gets diverse owners in the visible top, but cap at 200 —
          // anything more pushes JSON payload size into the
          // hundreds-of-KB and tanks UX.
          const widenedCount = Math.min(200, 60 + 5 * browseVips.length);
          indexedTasks.push(
            K.latest({
              indexName: ALL_INDEX_NAME,
              count: widenedCount,
              condition: allLatestCondition,
              parameters: allLatestParams,
            })
              .then((rs) =>
                rs.map((d) => ({ ...d, _from: d.owner || ALL_INDEX_NAME })),
              )
              .catch(() => []),
          );
          // Above the threshold we skip non-VIP fanout — the user
          // is asking for a fast aggregate, not careful merging.
          if (!useAllOnly(libs)) {
            for (const s of browseNonvips) {
              indexedTasks.push(
                K.latest({
                  indexName: s,
                  count: 50,
                  condition: filter ? filter.condition : null,
                  parameters: filter ? filter.parameters : null,
                })
                  .then((rs) => rs.map((d) => ({ ...d, _from: s })))
                  .catch(() => []),
              );
            }
          }
        } else {
          for (const s of libs) {
            indexedTasks.push(
              K.latest({
                indexName: s,
                count: 50,
                condition: filter ? filter.condition : null,
                parameters: filter ? filter.parameters : null,
              })
                .then((rs) => rs.map((d) => ({ ...d, _from: s })))
                .catch(() => []),
            );
          }
        }

        // Above the threshold we skip both the intersection tier
        // (server caps at 10 slugs anyway) and the per-slug
        // unindexed PG fanout. The `__all__` query alone covers
        // the fast-aggregate case; users who want shared-resource
        // ranking or unindexed-yet docs should narrow their
        // selection below the threshold.
        const skipFanout = useAllOnly(libs);
        const UNINDEXED_FANOUT_CAP = 25;
        const unindexedTargets = skipFanout
          ? []
          : libs.slice(0, UNINDEXED_FANOUT_CAP);

        const [intersection, indexedAll, unindexedAll] = await Promise.all([
          !skipFanout && libs.length >= 2
            ? K.intersect(libs, 200).catch(() => [])
            : Promise.resolve([]),
          Promise.all(indexedTasks),
          Promise.all(
            unindexedTargets.map((s) =>
              K.getUnindexedDocuments(s, {
                sources: sourcesArr,
                excludeSources: excludeArr,
                tags: tagsArr,
                urls: urlsArr,
              })
                .then((rs) => rs.map((d) => ({ ...d, _from: s })))
                .catch(() => []),
            ),
          ),
        ]);
        const map = new Map();
        // Intersection first — its `_owners` is the authoritative
        // full list (server-verified), so docs landing here have
        // the highest owners count and float to the top tier. The
        // intersection endpoint doesn't take SQL filters; apply the
        // matching predicate in JS for parity with the other pools.
        for (const d of intersection) {
          if (sourcesArr.length && !state.sources.has(d.source)) continue;
          if (state.excludedSources.has(d.source)) continue;
          if (tagsArr.length) {
            const all = new Set([...d.tags, ...d.extraTags]);
            let ok = true;
            for (const t of state.tags)
              if (!all.has(t)) {
                ok = false;
                break;
              }
            if (!ok) continue;
          }
          if (favOn && !state.favorites.has(d.url)) continue;
          map.set(d.url, { ...d });
        }
        // Indexed pool. If a URL was already added by the
        // intersection query, keep that record's owners (they're
        // server-verified across all libs) and just don't overwrite.
        for (const arr of indexedAll)
          for (const d of arr) {
            const ex = map.get(d.url);
            if (!ex) {
              map.set(d.url, { ...d, _owners: [d._from] });
            } else if (!ex._owners.includes(d._from)) {
              ex._owners.push(d._from);
            }
          }
        for (const arr of unindexedAll)
          for (const d of arr) {
            const ex = map.get(d.url);
            if (!ex) map.set(d.url, { ...d, _owners: [d._from] });
            else if (!ex._owners.includes(d._from)) ex._owners.push(d._from);
          }
        docs = Array.from(map.values());
      }
      // No JS post-filter: source, tags, AND favorites are all
      // resolved at the API now. The synthetic Favorites pseudo-
      // source is rewritten in buildIndexFilter into a `url IN (...)`
      // clause bound to the user's session-side favorites set.
      if (state.query) {
        // Query mode: relevance is default, manual Date toggle still
        // wins when picked. Diversity is intentionally NOT applied —
        // the user typed a query and expects matches, not variety.
        if (state.sortByDate)
          docs.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
      } else if (libs.length === 1) {
        // Personal-page browse: pure publication-date order. Mixing
        // `created_at` into the sort key let pipeline-sync time leak
        // into the visible order — a YouTube video published in 2023
        // but discovered today would float above a tweet posted
        // yesterday because its `created_at` was newer. Upvotes still
        // land at the top because the favorite-mirror SQL stamps
        // `date = CURRENT_DATE` on the inserted row (see
        // api/src/handlers/favorite_docs.rs::add).
        docs = docs
          .slice()
          .sort((a, b) => {
            const byDate = (b.date || "").localeCompare(a.date || "");
            if (byDate !== 0) return byDate;
            // Same-day tiebreak: insertion time descending so two
            // compose-dialog saves come back newest-first.
            return (b.createdAt || "").localeCompare(a.createdAt || "");
          })
          .slice(0, 60);
      } else {
        // Browse mode: owners desc → date desc → source-diversity
        // interleave. Adding a library naturally floats shared URLs
        // up (intersected results first); within the same owners
        // tier, newest wins; the interleave then breaks up runs of
        // the same source so a single GitHub-heavy library doesn't
        // monopolise the top of the pane.
        docs = reorderForBrowse(docs).slice(0, 60);
      }

      // Note: unindexed docs are fetched in parallel with the
      // indexed pool above and merged into the same `docs` array,
      // so the diversity reorder treats them as first-class
      // citizens. The "not indexed yet" badge still distinguishes
      // them in renderResult — they just aren't pinned to the top.
    } catch (e) {
      console.warn(e);
    }
    if (my !== reqId) return;
    state.lastDocs = docs;
    // Source rail: when the selection is large enough that we
    // skipped per-slug `getSources()` calls, derive the rail from
    // the result set itself — only sources that appear in the
    // visible docs, no counts. Search-by-text on the rail still
    // works against this list. Below the threshold the rail stays
    // driven by per-slug data (canonical counts).
    if (useAllOnly([...state.libs])) {
      rebuildAllSourcesFromDocs(docs);
      renderSrc();
    }
    renderResultSources();
    $("resultCount").textContent =
      `${docs.length} result${docs.length === 1 ? "" : "s"}`;
    setQueryMetrics({
      nResults: docs.length,
      tMs: performance.now() - _refreshT0,
      total: candidatePoolSize(),
    });
    if (docs.length === 0) {
      hideResultsSpinner();
      // Special case: signed-in user is on their OWN personal page
      // with zero docs and no query/source/tag filters narrowing the
      // pool — that means they haven't connected any sources yet.
      // Drop a friendly CTA pointing them at /profile (Settings) so
      // the page doesn't feel broken on day one.
      const onOwnPersonal =
        me &&
        state.libs.size === 1 &&
        state.libs.has(me.slug) &&
        !state.query &&
        !state.tags.size &&
        !state.sources.size &&
        !state.excludedSources.size;
      if (onOwnPersonal) {
        // Follow-graph empty? Take that as a stronger "new user"
        // signal than "no docs" — propose people to follow first, the
        // library-is-empty CTA falls through after.
        const followsSet = await loadFollowingSet();
        if (followsSet.size === 0 && window.KnowledgeOnboarding) {
          window.KnowledgeOnboarding.open({
            personalities: state.allPersonalities,
            apiBase: API_BASE,
            onSkip: () => renderPersonalEmptyOnboarding(),
          });
        } else {
          renderPersonalEmptyOnboarding();
        }
      } else {
        resetEmptyMessage();
        $("empty").style.display = "";
      }
      return;
    }
    $("empty").style.display = "none";
    $("results").innerHTML = docs.map(renderResult).join("");
    wireResults();
    armInfiniteScroll();
    // Personal page (single library) → ask the backend which OTHER
    // VIPs also have each URL in their library. Stamps the result onto
    // state.lastDocs and re-paints just the avatar stack so the user
    // sees a transparent "people who also liked this" row per card.
    if (state.libs.size === 1) {
      decoratePersonalPageWithCoOwners(docs);
    }
    // For every visible retweet, ask the backend "who else
    // retweeted this exact source tweet?" and fold the extras into
    // the avatar stack. Independent from the co-owner strip above:
    // co-owners is "this URL is in N libraries", co-retweeters is
    // "this source tweet has N wrapper URLs across libraries".
    enrichRetweetSharers(docs);
    // Re-rank only when there's an active query — browse mode has
    // no notion of relevance and the diversity reorder owns its
    // ordering. The worker's queryId guard ignores stale messages.
    if (state.query) postRerank(docs);
    // Kick off the page-meta enhancer (top-down, sequential) so the
    // first cards the user reads get richer summaries quickly. Fire
    // and forget — token-cancellable, never throws out of band.
    enhanceResults(enhanceToken, docs);
    // Start or stop the personal-page live watcher. Single-library
    // browse, no active query → poll every 10s for freshly-ingested
    // docs (e.g. the local twitter feeder writing tweets directly to
    // prod PG). Any other state cancels the watcher so we never
    // burn requests on the wrong route.
    if (state.libs.size === 1 && !state.query) {
      startPersonalPageWatch([...state.libs][0]);
    } else {
      stopPersonalPageWatch();
    }
  }

  /* Twitter / X helpers — extract the @handle from a status URL
   * and decide whether a doc is a tweet (so the kicker can swap
   * its source pill for a clickable @handle). */
  function extractTwitterHandle(url) {
    if (!url) return null;
    const m =
      /^https?:\/\/(?:x\.com|twitter\.com|mobile\.twitter\.com)\/([^/?#]+)/i.exec(
        url,
      );
    if (!m) return null;
    const handle = m[1];
    const reserved = [
      "home",
      "search",
      "i",
      "explore",
      "notifications",
      "compose",
      "settings",
      "login",
      "signup",
      "messages",
      "tos",
      "privacy",
    ];
    if (reserved.includes(handle.toLowerCase())) return null;
    return handle;
  }
  function isTweetDoc(d) {
    if ((d.source || "").toLowerCase() === "twitter") return true;
    const h = K.hostOf(d.url) || "";
    return h === "x.com" || h === "twitter.com" || h === "mobile.twitter.com";
  }

  /* Date label for the result cards. Feed mode (no library
   * selected) shows weekly buckets — "This week", "A week ago",
   * "3 weeks ago" — matching the server-side weekly recency
   * scoring. Per-library and search views stay day-granular
   * ("3d ago") because those are sorted by raw date. */
  function dateLabel(iso) {
    if (!iso) return "";
    return state.libs.size === 0
      ? K.feedRelativeDate(iso)
      : `${K.relativeDate(iso)} ago`;
  }
  /* User-facing source label. The `twitter` key is renamed to "X"
   * for display (the platform's current name); the bird logo we
   * render alongside the label is bundled separately so the icon
   * stays put. Everything else passes through unchanged. */
  function displaySource(key) {
    const k = String(key || "").toLowerCase();
    if (
      k === "twitter" ||
      k === "x.com" ||
      k === "twitter.com" ||
      k === "mobile.twitter.com"
    ) {
      return "X";
    }
    return key || "";
  }
  /* For tweet docs the stored `title` is just the author's display
   * name, which reads as filler when fifty results in a row repeat
   * the same handle. Pull a real title out of the tweet body
   * instead: drop any leading "RT @handle:" wrapper, prefer the
   * first sentence boundary that lands in a reasonable range, fall
   * back to a word-boundary truncation otherwise. Empty or
   * RT-only text falls through to the original title. */
  function tweetTitle(d) {
    let raw = String(d.summary || "").trim();
    // Strip pipeline-emitted media markers so they don't end up as the
    // rendered title. The video format is "🎬 <poster> | <mp4>" — the
    // optional second URL after " | " must be consumed too, otherwise
    // the title becomes "| https://video.twimg.com/...".
    raw = raw
      .replace(
        /^\s*[📷🎬]\s+https?:\/\/\S+(?:\s+\|\s+https?:\/\/\S+)?\s*/gmu,
        "",
      )
      .replace(/\[\d+\/\d+\]\s*/gu, "")
      .replace(/[📷🎬]+/gu, "")
      .trim();
    const cleaned = raw.replace(/^RT\s+@[A-Za-z0-9_]+:\s*/i, "").trim();
    if (!cleaned) return d.title || "";
    const MIN = 25; // shorter than this and we'd undersell the tweet
    const TARGET = 55; // ideal title length
    const MAX = 80; // hard cap before we truncate
    // 1. First sentence-boundary that lands in [MIN, MAX].
    const sentenceRe = /[.!?](\s|$)/g;
    let m;
    while ((m = sentenceRe.exec(cleaned)) !== null) {
      const cut = m.index + 1;
      if (cut >= MIN && cut <= MAX) {
        return cleaned.slice(0, cut).trim();
      }
    }
    // 2. Short tweet — render the whole thing as the title.
    if (cleaned.length <= MAX) return cleaned;
    // 3. Truncate near TARGET on a word boundary, ellipsize.
    const window = cleaned.slice(0, TARGET);
    const lastSpace = window.lastIndexOf(" ");
    const cut = lastSpace > MIN ? lastSpace : TARGET;
    return cleaned.slice(0, cut).trim() + "…";
  }
  /* X / Twitter brand mark — uses the bundled icon at /icons/twitter.png
   * to match the rest of the app rather than rolling our own SVG path. */
  const TWEET_ICON_SVG = `<img class="tweet-icon" src="/icons/twitter.png" alt="" aria-hidden="true"/>`;

  /* ── Editorial: airy magazine layout, no boxes ────────────── */
  /* Ownership predicate for the Edit-card affordance.
   *
   * Edit is shown only when the signed-in user owns the row. Three
   * cases produce ownership:
   *   1. Personal page where the host slug is the signed-in user
   *      (libs.size === 1 && libs has me.slug) — every doc is mine.
   *   2. Feed cards (libs.size !== 1) where `d.owner === me.slug`
   *      (the search/__all__ path).
   *   3. Feed timeline rows where `d.sharers` includes me, or
   *      `_owners` includes my slug (timeline returns per-URL
   *      records with multiple sharers; if I'm one of them, I
   *      can edit *my* copy of the row).
   */
  function isDocOwnedByMe(d) {
    if (!me?.slug) return false;
    if (state.libs.size === 1 && state.libs.has(me.slug)) return true;
    if (d.owner && d.owner === me.slug) return true;
    if (Array.isArray(d._owners) && d._owners.includes(me.slug)) return true;
    if (Array.isArray(d.sharers) && d.sharers.some((s) => s?.slug === me.slug))
      return true;
    return false;
  }

  /* Image lightbox — clicking a `.tweet-media-tile` photo opens the
   * full-resolution image in an overlay instead of navigating to a
   * new tab. Click anywhere outside the image or press Escape to
   * dismiss.
   *
   * Multi-image navigation: when a tile is clicked we collect every
   * `[data-zoom]` image inside the SAME `article.result` (i.e. the
   * same document — never crossing into another card). The user
   * can flip through them with ←/→ arrow keys or a horizontal
   * swipe on touch. Prev/next chevron buttons sit on the left and
   * right edges and disappear when only one image is available. */
  const _lightboxState = { list: [], idx: 0 };

  function _collectImagesForCard(triggerEl) {
    // The result card is the natural "document" scope. Fall back to
    // the tile's nearest media container so older markup (find-
    // similar drawer, etc.) still gets correct grouping.
    const scope =
      triggerEl?.closest("article.result, .result, .doc, .card") ||
      triggerEl?.closest(".tweet-media") ||
      document;
    const tiles = Array.from(scope.querySelectorAll("[data-zoom]"));
    return tiles.map((t) => t.dataset.zoom).filter(Boolean);
  }

  function _showLightboxAt(idx) {
    const img = $("imgLightboxImg");
    const list = _lightboxState.list;
    if (!img || !list.length) return;
    // Clamp instead of wrap so reaching the first/last image hits
    // a visible wall — the user knows they've seen everything in
    // this doc instead of looping back unexpectedly.
    const n = list.length;
    const next = Math.max(0, Math.min(n - 1, idx));
    _lightboxState.idx = next;
    img.src = list[next];
    // Toggle nav-button visibility & active state — hide entirely
    // when there's only one image; otherwise dim the side that's
    // already at the boundary so the user reads "this is the end".
    const prev = document.getElementById("imgLightboxPrev");
    const nxt = document.getElementById("imgLightboxNext");
    const multi = n > 1;
    if (prev) {
      prev.hidden = !multi;
      prev.classList.toggle("img-lightbox-nav-disabled", next === 0);
    }
    if (nxt) {
      nxt.hidden = !multi;
      nxt.classList.toggle("img-lightbox-nav-disabled", next === n - 1);
    }
  }

  function _openImgLightbox(url, triggerEl) {
    const root = $("imgLightbox");
    const img = $("imgLightboxImg");
    if (!root || !img) return;
    _ensureLightboxControls();
    const list = _collectImagesForCard(triggerEl);
    // Defensive: if collection somehow missed the clicked URL,
    // fall back to a single-image list so the lightbox still
    // opens — just without prev/next.
    if (!list.includes(url)) list.push(url);
    _lightboxState.list = list;
    _showLightboxAt(list.indexOf(url));
    root.hidden = false;
    document.body.classList.add("img-lightbox-open");
  }
  function _closeImgLightbox() {
    const root = $("imgLightbox");
    const img = $("imgLightboxImg");
    if (!root || !img) return;
    root.hidden = true;
    img.removeAttribute("src");
    _lightboxState.list = [];
    _lightboxState.idx = 0;
    document.body.classList.remove("img-lightbox-open");
  }
  function _lightboxNext() {
    _showLightboxAt(_lightboxState.idx + 1);
  }
  function _lightboxPrev() {
    _showLightboxAt(_lightboxState.idx - 1);
  }

  // Inject the prev/next chevron buttons once. The HTML template
  // only ships the close button; nav buttons are a runtime add so
  // we don't have to touch the static markup in three pages.
  function _ensureLightboxControls() {
    const root = $("imgLightbox");
    if (!root || root.dataset.navWired === "1") return;
    root.dataset.navWired = "1";
    const mkBtn = (id, label, dir) => {
      const b = document.createElement("button");
      b.type = "button";
      b.id = id;
      b.className = `img-lightbox-nav img-lightbox-nav-${dir}`;
      b.setAttribute("aria-label", label);
      b.innerHTML =
        dir === "prev"
          ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg>'
          : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>';
      b.hidden = true;
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        if (dir === "prev") _lightboxPrev();
        else _lightboxNext();
      });
      return b;
    };
    root.appendChild(mkBtn("imgLightboxPrev", "Previous image", "prev"));
    root.appendChild(mkBtn("imgLightboxNext", "Next image", "next"));

    // Touch-swipe navigation. Threshold tuned so a small wobble
    // while pinching/zooming doesn't trip a navigation — only a
    // deliberate horizontal flick paginates.
    let touch = null;
    root.addEventListener(
      "touchstart",
      (e) => {
        if (root.hidden || _lightboxState.list.length < 2) return;
        const t = e.touches[0];
        touch = { x0: t.clientX, y0: t.clientY, t0: performance.now() };
      },
      { passive: true },
    );
    root.addEventListener(
      "touchend",
      (e) => {
        if (!touch) return;
        const t = e.changedTouches[0];
        const dx = t.clientX - touch.x0;
        const dy = t.clientY - touch.y0;
        const dt = Math.max(1, performance.now() - touch.t0);
        const vx = Math.abs(dx) / dt; // px/ms
        touch = null;
        // Horizontal-dominant flick OR ≥60px translation commits.
        const horizontal = Math.abs(dx) > Math.abs(dy) * 1.4;
        if (!horizontal) return;
        if (Math.abs(dx) < 60 && vx < 0.35) return;
        if (dx < 0) _lightboxNext();
        else _lightboxPrev();
      },
      { passive: true },
    );
  }

  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-zoom]");
    if (btn) {
      e.preventDefault();
      _openImgLightbox(btn.dataset.zoom, btn);
      return;
    }
    if (e.target.closest("#imgLightboxClose")) {
      _closeImgLightbox();
      return;
    }
    // Don't dismiss when the click was on the nav buttons.
    if (e.target.closest(".img-lightbox-nav")) return;
    // Click on the image itself paginates when the doc has more
    // than one photo. Left half → prev, right half → next. This
    // matches the Instagram / Stories convention so a tap reads
    // the same as a swipe. With a single image we fall back to
    // closing on click (the user has nowhere else to navigate).
    if (e.target.id === "imgLightboxImg") {
      if (_lightboxState.list.length > 1) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        if (x < rect.width / 2) _lightboxPrev();
        else _lightboxNext();
      } else {
        _closeImgLightbox();
      }
      return;
    }
    // Click on the backdrop (anywhere inside .img-lightbox but not on
    // the image itself) closes.
    const lb = e.target.closest(".img-lightbox");
    if (lb) {
      _closeImgLightbox();
    }
  });
  document.addEventListener("keydown", (e) => {
    if ($("imgLightbox")?.hidden) return;
    if (e.key === "Escape") _closeImgLightbox();
    else if (e.key === "ArrowRight") _lightboxNext();
    else if (e.key === "ArrowLeft") _lightboxPrev();
  });

  /* Tweet summary renderer — parses the self-sufficient summary the
   * pipeline produces (see `_tweet_self_sufficient_summary` in
   * sources/twitter/tweets.py) and yields a richer DOM:
   *
   *   - Threads (parts separated by `────────`) become a vertical
   *     stack of `.tweet-part` boxes.
   *   - Lines starting with `📷 <url>` / `🎬 <url>` are pulled out
   *     of the text and rendered as a horizontally-scrolling
   *     `.tweet-media` strip with real `<img>` / `<video>` tags.
   *
   * Falls back to a plain truncated paragraph if the summary doesn't
   * match the expected shape (e.g. older docs from before the new
   * pipeline). */
  const _TWEET_PHOTO_RE = /^📷\s+(https?:\/\/\S+)/u;
  const _TWEET_VIDEO_RE = /^🎬\s+(https?:\/\/\S+)/u;
  const _TWEET_SEPARATOR = "────────";

  function _parseTweetPart(raw) {
    let text = (raw || "").trim();
    const photos = [];
    // Videos carry both a poster image and an mp4 URL, encoded as
    //   "🎬 <poster> | <mp4>"
    // The poster is what we render inline (loads fine — pbs.twimg.com
    // doesn't gate on Referer), and the mp4 is the click-out target.
    const videos = []; // {poster, mp4}
    const lines = text.split("\n");
    const kept = [];
    for (const line of lines) {
      let m = _TWEET_PHOTO_RE.exec(line.trim());
      if (m) {
        photos.push(m[1]);
        continue;
      }
      m = _TWEET_VIDEO_RE.exec(line.trim());
      if (m) {
        // Match the rest of the line so we capture both halves.
        const rest = line.trim().slice(2).trim();
        const sepIdx = rest.indexOf(" | ");
        if (sepIdx >= 0) {
          videos.push({
            poster: rest.slice(0, sepIdx).trim(),
            mp4: rest.slice(sepIdx + 3).trim(),
          });
        } else {
          // Older format: one URL — best guess it's the mp4.
          videos.push({ poster: "", mp4: rest });
        }
        continue;
      }
      kept.push(line);
    }
    text = kept.join("\n").trim();
    // Strip the [i/N] thread-part prefix from wherever it ended up
    // (after media extraction it may have moved off the very first
    // line). Loop until no more match — handles consecutive markers.
    while (true) {
      const next = text.replace(/^\[\d+\/\d+\]\s*/u, "");
      if (next === text) break;
      text = next.trim();
    }
    // Drop any leftover bare 📷 / 🎬 emojis the user (or the pipeline)
    // typed without an accompanying URL — they read as stray symbols
    // once their URL has been hoisted into a tile.
    text = text
      .replace(/[ \t]*[📷🎬]+[ \t]*/gu, " ")
      .replace(/[ \t]{2,}/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    return { text, photos, videos };
  }

  function renderTweetSummary(d) {
    const summary = d.summary || "";
    // Don't early-out on empty summary — a retweet of a URL-only
    // tweet (e.g. an X long-form Article) leaves the summary
    // blank but still has a populated `linkedUrls` cluster the
    // reader needs to see. Falling through lets the link-card
    // renderer below paint that preview.
    const rawParts = summary.includes(_TWEET_SEPARATOR)
      ? summary.split(_TWEET_SEPARATOR)
      : [summary];
    const parts = rawParts
      .map(_parseTweetPart)
      .filter((p) => p.text || p.photos.length || p.videos.length);
    const hasLinks = Array.isArray(d.linkedUrls) && d.linkedUrls.length > 0;
    if (parts.length === 0 && !hasLinks) return "";
    const isThread = parts.length > 1;
    const renderMedia = (p) => {
      if (!p.photos.length && !p.videos.length) return "";
      // Photos open the in-app lightbox via `data-zoom`.
      // Videos: render the poster image with a ▶ overlay and link to
      // the tweet's own status URL on x.com — playback happens on
      // Twitter (where the Referer check is satisfied). The doc URL
      // is the same status URL by construction.
      const statusUrl = d.url || "";
      const tiles = [
        ...p.photos.map(
          (u) =>
            `<button type="button" class="tweet-media-tile" data-zoom="${escapeAttr(u)}">
               <img loading="lazy" src="${escapeAttr(u)}" alt="" onerror="this.parentElement.style.display='none'"/>
             </button>`,
        ),
        ...p.videos.map((v) => {
          const poster = v.poster || v.mp4;
          if (!poster) return "";
          return `<a class="tweet-media-tile tweet-media-video" href="${safeHref(statusUrl)}" target="_blank" rel="noopener" title="Watch on twitter.com">
               <img loading="lazy" src="${escapeAttr(poster)}" alt="" onerror="this.parentElement.style.display='none'"/>
               <span class="tweet-media-play" aria-hidden="true">▶</span>
             </a>`;
        }),
      ].join("");
      return `<div class="tweet-media">${tiles}</div>`;
    };
    const renderPart = (p) => {
      let textHtml = "";
      if (p.text) {
        // Split into real paragraphs on blank lines (the source uses
        // `\n\n` between logical paragraphs — e.g. `Retweet @x\n\nbody
        // \n\nQuoting @y\n\nbody`). Each becomes its own <p> so the
        // browser handles paragraph spacing typographically instead
        // of relying on `white-space: pre-wrap` to render a blank
        // line. Single `\n` inside a paragraph still renders as a
        // line break via `pre-wrap` (kept in the CSS) so tweets like
        // `Title\nDescription` preserve their visual rhythm.
        const cleaned = cleanDescription(p.text, 1200);
        const paragraphs = cleaned
          .split(/\n{2,}/)
          .map((s) => s.trim())
          .filter(Boolean);
        textHtml = paragraphs
          .map(
            (para) =>
              `<p class="tweet-part-text">${highlightMatches(para, state.query)}</p>`,
          )
          .join("");
      }
      return `<div class="tweet-part${isThread ? " tweet-part--threaded" : ""}">
          ${textHtml}
          ${renderMedia(p)}
        </div>`;
    };
    const inner = parts.map(renderPart).join("");
    // Link cards live at the doc level (one cluster per tweet, not
    // per thread-part) because the rich metadata is stored on the
    // document's `linkedUrls` column rather than parsed out of the
    // summary text. Render after the parts so the cards sit below
    // every paragraph, matching the visual layout the user expects
    // ("the tweet, then the things it links to").
    const linksHtml = renderDocLinkCards(d);
    return `<div class="tweet-summary${isThread ? " tweet-summary--thread" : ""}">${inner}${linksHtml}</div>`;
  }

  /* Render the inline link-preview cluster for any doc. Reads
   * `d.linkedUrls` (an array of `{url, host, title, summary, image}`
  /* Photo + video media tiles for a tweet, without the body text.
   * Used by the clean-summary render path so the original tweet's
   * media survives even though the daemon's rewritten text doesn't
   * mention the media markers. Mirrors the media-emit half of
   * renderTweetSummary so the lightbox-zoom wiring stays identical. */
  function renderTweetMediaOnly(d) {
    const summary = d.summary || "";
    const rawParts = summary.includes(_TWEET_SEPARATOR)
      ? summary.split(_TWEET_SEPARATOR)
      : [summary];
    const parts = rawParts.map(_parseTweetPart);
    const statusUrl = d.url || "";
    const tiles = parts
      .flatMap((p) => [
        ...p.photos.map(
          (u) =>
            `<button type="button" class="tweet-media-tile" data-zoom="${escapeAttr(u)}">
               <img loading="lazy" src="${escapeAttr(u)}" alt="" onerror="this.parentElement.style.display='none'"/>
             </button>`,
        ),
        ...p.videos.map((v) => {
          const poster = v.poster || v.mp4;
          if (!poster) return "";
          return `<a class="tweet-media-tile tweet-media-video" href="${safeHref(statusUrl)}" target="_blank" rel="noopener" title="Watch on twitter.com">
               <img loading="lazy" src="${escapeAttr(poster)}" alt="" onerror="this.parentElement.style.display='none'"/>
               <span class="tweet-media-play" aria-hidden="true">▶</span>
             </a>`;
        }),
      ])
      .filter(Boolean)
      .join("");
    return tiles ? `<div class="tweet-media">${tiles}</div>` : "";
  }

  /* populated by the pipeline at ingest time) and emits one card
   * per entry. The data is already enriched server-side so there's
   * no client-side OG fetch involved — each card renders fully on
   * first paint.
   *
   * Empty array → empty string, so callers can append unconditionally. */
  function renderDocLinkCards(d) {
    const list = Array.isArray(d.linkedUrls) ? d.linkedUrls : [];
    if (!list.length) return "";
    return `<div class="tweet-links">${list
      .map((link) => {
        const url = link && link.url ? String(link.url) : "";
        if (!url) return "";
        // `link.host` is stored truncated at ingest time (e.g.
        // "arxiv" instead of "arxiv.org"), so it's fine for display
        // but useless as a Google-favicon lookup key — querying
        // `?domain=arxiv` returns Google's default globe icon, which
        // is what the user was seeing as a mis-aligned fallback.
        // Derive the favicon's lookup domain from the URL itself
        // (preserving www-stripping for nicer parity with `link.host`)
        // so the favicon service hits the real registered domain.
        let favDomain = "";
        try {
          favDomain = new URL(url).hostname.replace(/^www\./, "");
        } catch {
          favDomain = "";
        }
        const host =
          link.host ||
          favDomain ||
          (() => {
            try {
              return new URL(url).hostname.replace(/^www\./, "");
            } catch {
              return url;
            }
          })();
        const fav = favDomain
          ? `https://www.google.com/s2/favicons?domain=${encodeURIComponent(favDomain)}&sz=64`
          : "";
        const title = link.title || url;
        const summary = link.summary || "";
        const image = link.image || "";
        // Card shape:
        //   - With OG image → left thumbnail + meta column.
        //   - Without image → no thumbnail tile; favicon sits inline
        //     next to the host name so the row stays compact.
        //
        // When the OG image fails (404) the onerror handler drops the
        // `--has-image` modifier so the card collapses to the no-image
        // layout without leaving a broken-image placeholder.
        const imgBlock = image
          ? `<div class="link-card-img">
              <img class="link-card-og" loading="lazy" src="${escapeAttr(image)}" alt=""
                onerror="const c=this.closest('.link-card');if(c){c.classList.remove('link-card--has-image');c.classList.add('link-card--no-image');}this.remove();"/>
            </div>`
          : "";
        return `<a class="link-card ${image ? "link-card--has-image" : "link-card--no-image"}"
                   href="${safeHref(url)}"
                   target="_blank" rel="noopener"
                   onclick="event.stopPropagation()">
            ${imgBlock}
            <div class="link-card-meta">
              <div class="link-card-host">
                <img class="link-card-fav" loading="lazy"
                     src="${escapeAttr(fav)}" alt=""
                     onerror="this.style.display='none'"/>
                <span>${escapeHtml(host)}</span>
              </div>
              <div class="link-card-title">${escapeHtml(title)}</div>
              ${summary ? `<div class="link-card-summary">${escapeHtml(summary)}</div>` : ""}
            </div>
          </a>`;
      })
      .join("")}</div>`;
  }

  /* Co-retweet enrichment. Search results come from the ColBERT
   * index, so `_owners` only reflects shards that returned the URL —
   * which, for a retweet, is just the single personality whose
   * wrapper URL it is. The timeline endpoint already aggregates
   * co-retweeters via `md5(summary)` in SQL (see
   * `coretweet_sharers` in api/src/handlers/follows.rs), so for
   * search-mode we hit the same logic via the batch
   * `/api/documents/coretweet-sharers` endpoint and merge the
   * extra slugs into `_owners`, then re-render just the avatar
   * stack of the affected cards. No-op on non-retweet docs (the
   * endpoint returns them as empty).
   */
  async function enrichRetweetSharers(docs) {
    if (!Array.isArray(docs) || !docs.length) return;
    const urls = docs
      .filter(
        (d) =>
          d &&
          d.url &&
          d.source === "twitter" &&
          /^Retweet @/.test(d.summary || ""),
      )
      .map((d) => d.url);
    if (!urls.length) return;
    let payload;
    try {
      const r = await fetch(`${API_BASE}/api/documents/coretweet-sharers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls }),
      });
      if (!r.ok) return;
      payload = await r.json();
    } catch {
      return;
    }
    if (!payload || typeof payload !== "object") return;
    const byUrl = new Map();
    for (const d of state.lastDocs || []) byUrl.set(d.url, d);
    const changedUrls = [];
    for (const [url, sharers] of Object.entries(payload)) {
      const d = byUrl.get(url);
      if (!d) continue;
      const list = Array.isArray(sharers) ? sharers : [];
      if (!list.length) continue;
      const existing = new Set(d._owners || []);
      let added = false;
      for (const s of list) {
        if (!s || !s.slug) continue;
        if (!state.perSlugMeta[s.slug]) {
          state.perSlugMeta[s.slug] = {
            slug: s.slug,
            name: s.name || s.slug,
            avatar: s.avatar || null,
            twitterFollowers: s.twitterFollowers || 0,
          };
        }
        if (!existing.has(s.slug)) {
          existing.add(s.slug);
          added = true;
        }
      }
      if (added) {
        d._owners = [...existing];
        changedUrls.push(url);
      }
    }
    if (!changedUrls.length) return;
    // Surgical avatar-stack swap — DON'T replace the whole card.
    // Replacing the article element makes the card disappear and
    // reappear (visible blink + scroll position twitch + loss of
    // :hover and any open similar panel). The only thing the
    // co-retweet enrichment ever changes is the `.shared-by`
    // avatar strip, so we render the full result template offline,
    // pluck out just the new `.shared-by` subtree, and slot it
    // into the existing card's foot-right cluster. The rest of the
    // DOM (title, summary, media, tags, score) is untouched.
    const host = $("results");
    if (!host) return;
    for (const url of changedUrls) {
      const node = host.querySelector(
        `article.result[data-url="${CSS.escape(url)}"]`,
      );
      const d = byUrl.get(url);
      if (!node || !d) continue;
      const tmp = document.createElement("template");
      tmp.innerHTML = renderResult(d).trim();
      const fresh = tmp.content.firstElementChild;
      if (!fresh) continue;
      const freshStack = fresh.querySelector(".shared-by");
      const existingStack = node.querySelector(".shared-by");
      if (existingStack && freshStack) {
        existingStack.replaceWith(freshStack);
        wireResults(node);
      } else if (freshStack && !existingStack) {
        // Card had no avatar stack yet (e.g. single-library mode
        // without an original-author bubble). The renderResult
        // template puts the stack inside `.result-foot-right`; create
        // that cluster if it doesn't exist, then append the strip.
        let foot = node.querySelector(".result-foot-right");
        if (!foot) {
          foot = document.createElement("div");
          foot.className = "result-foot-right";
          node.appendChild(foot);
        }
        foot.appendChild(freshStack);
        wireResults(node);
      }
    }
  }

  /* Personal-page co-owner decorator. Posts the visible URLs to
   * /api/co-owners (cheap LATERAL on documents) and stamps the
   * resulting VIP-owner list onto each doc as `_co_owners`. Then
   * walks the existing DOM and slots the avatars into a sibling
   * row underneath each card's foot-right cluster — light & half
   * transparent so the user reads them as "social context", not
   * primary content. */
  async function decoratePersonalPageWithCoOwners(docs) {
    if (!Array.isArray(docs) || docs.length === 0) return;
    const urls = docs.map((d) => d.url).filter(Boolean);
    if (!urls.length) return;
    const excludeSlug = state.hostSlug || me?.slug || "";
    let payload;
    try {
      // No `credentials: include` — endpoint is public and the broader
      // /api/* CORS layer doesn't allow credentialed requests
      // (Access-Control-Allow-Origin: *). Cookies aren't needed.
      const r = await fetch(`${API_BASE}/api/co-owners`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls, exclude_slug: excludeSlug }),
      });
      if (!r.ok) return;
      payload = await r.json();
    } catch {
      return;
    }
    if (!payload || typeof payload !== "object") return;
    // Stamp results onto state.lastDocs so re-renders preserve them.
    const byUrl = new Map();
    for (const d of state.lastDocs || []) byUrl.set(d.url, d);
    for (const [url, owners] of Object.entries(payload)) {
      const d = byUrl.get(url);
      if (!d) continue;
      d._co_owners = Array.isArray(owners) ? owners : [];
      // Cache meta so future renders can resolve slug → avatar/name.
      for (const o of d._co_owners) {
        if (o && o.slug && !state.perSlugMeta[o.slug]) {
          state.perSlugMeta[o.slug] = {
            slug: o.slug,
            name: o.name || o.slug,
            avatar: o.avatar || null,
            twitterFollowers: o.twitterFollowers || 0,
          };
        }
      }
    }
    // Re-paint the avatar strips in place. Cheap — we only touch the
    // cards that gained data.
    paintCoOwners();
  }

  function paintCoOwners() {
    const host = $("results");
    if (!host) return;
    for (const art of host.querySelectorAll(":scope > article")) {
      const url = art.dataset?.url;
      const doc = (state.lastDocs || []).find((d) => d.url === url);
      if (!doc || !Array.isArray(doc._co_owners) || !doc._co_owners.length) {
        continue;
      }
      // Avoid double-paint.
      if (art.querySelector(".co-owners")) continue;
      const owners = doc._co_owners.slice(0, 12);
      const followingSet = _peopleRail?.following || new Set();
      const sorted = [
        ...owners.filter((o) => followingSet.has(o.slug)),
        ...owners.filter((o) => !followingSet.has(o.slug)),
      ];
      // The "also liked by" label used to live here as a small
      // muted span before the avatar stack. Dropped on the user's
      // request — the avatars themselves carry the meaning, and the
      // aria-label below keeps the social context discoverable for
      // assistive tech.
      const html = `<div class="co-owners" aria-label="${owners.length} other ${owners.length === 1 ? "person has" : "people have"} this">
        ${sorted
          .map(
            (o) =>
              `<span class="ava-host"><a class="ava co-ava" href="/search?libs=${encodeURIComponent(o.slug)}" title="${escapeAttr(o.name || o.slug)}"><img src="${escapeAttr(o.avatar || "")}" alt="" onerror="this.style.opacity=0"/></a></span>`,
          )
          .join("")}
      </div>`;
      // Slot the strip into the foot-right cluster as a new row
      // below the existing actions+avatars. If that cluster doesn't
      // exist on this card (no score, no sharers), create one.
      let foot = art.querySelector(".result-foot-right");
      if (!foot) {
        foot = document.createElement("div");
        foot.className = "result-foot-right";
        art.appendChild(foot);
      }
      foot.insertAdjacentHTML("beforeend", html);
    }
  }

  function renderResult(d) {
    const host = K.hostOf(d.url);
    const isFav = state.favorites.has(d.url);
    const tags = [...d.tags, ...d.extraTags].slice(0, 5);

    // Tweet pill: when this doc is a tweet, swap the {favicon + source}
    // pair in the kicker for a single clickable @handle pill.
    const tweetUrl = d.source_url || d.url;
    const tweetHandle = isTweetDoc(d) ? extractTwitterHandle(tweetUrl) : null;
    const sourceLabel = displaySource(d.source || host);
    /* Per-card "exclude this source" affordance — a tiny ✕ tucked
     * next to the source pill. Click adds the source to
     * state.excludedSources and re-runs the search; the running ban
     * list surfaces as small chips next to the search bar so the
     * user can restore. We don't render it for the tweet pill
     * variant (the @handle pill is itself a link) or when the
     * source key is empty. */
    const banBtn =
      d.source || ""
        ? `<button class="src-ban" type="button"
          data-ban-source="${escapeAttr(d.source)}"
          title="Hide ${escapeAttr(sourceLabel)} from results"
          aria-label="Hide ${escapeAttr(sourceLabel)} from results"
          onclick="event.stopPropagation()">✕</button>`
        : "";
    // HN front-page picks: replace the article-host favicon with the
    // HN logo so the card visually announces "this came from HN" even
    // when the linked article lives on a third-party site (WSJ,
    // niemanlab, …).
    const srcIconUrl = d.picked
      ? K.sourceIconUrl("hackernews")
      : K.faviconUrl(host);
    const sourcePill = tweetHandle
      ? `<a class="tweet-link" href="${safeHref(tweetUrl)}" target="_blank" rel="noopener" title="View original tweet" onclick="event.stopPropagation()">${TWEET_ICON_SVG}@${highlightMatches(tweetHandle, state.query)}</a>${banBtn}`
      : `<img class="src-fav" src="${srcIconUrl}" alt="" onerror="this.style.display='none'"/><span class="src">${highlightMatches(sourceLabel, state.query)}</span>${banBtn}`;

    // "via @handle" attribution: when the doc itself is NOT a tweet but
    // we know it surfaced through one (source_url points at a tweet
    // permalink — e.g. a paper / blog post / repo extracted from a
    // user's twitter thread or bookmark), show a small pill that links
    // to the originating tweet. Skipped when the doc IS the tweet
    // (the source pill above already exposes the @handle).
    const viaTweetUrl = d.source_url && !isTweetDoc(d) ? d.source_url : null;
    const viaHandle = viaTweetUrl ? extractTwitterHandle(viaTweetUrl) : null;
    const viaPill = viaHandle
      ? `<span class="dot">·</span><a class="via-tweet" href="${safeHref(viaTweetUrl)}" target="_blank" rel="noopener" title="View originating tweet" onclick="event.stopPropagation()">${TWEET_ICON_SVG}via @${highlightMatches(viaHandle, state.query)}</a>`
      : "";

    // "Shared by" avatar stack: every owner of this doc, including
    // the page's host — so a doc shared by raphael + max shows both
    // faces, the user gets the full membership at a glance. Order
    // is a weighted shuffle by popularity (popular accounts tend
    // toward the front, but it's randomised on each refresh so the
    // sequence isn't frozen). Single-library mode is suppressed —
    // every card would just show the host on its own.
    // Avatar stack ordering:
    //   1. People the signed-in user already follows come first
    //      (stable order by Twitter-follower count) — they're who
    //      the caller cares about most.
    //   2. Everyone else is weighted-shuffled by popularity so a
    //      well-known sharer surfaces near the front *most* of the
    //      time, but the order isn't frozen each refresh.
    //   3. The caller themselves is pinned at the very front when
    //      they're a sharer too.
    const _ownersAllMeta = (d._owners || [])
      .map((s) => state.perSlugMeta[s])
      .filter(Boolean);
    const _followingSet = _peopleRail?.following || new Set();
    const _meSlug = me?.slug || "";
    const _meBucket = [];
    const _followedBucket = [];
    const _restBucket = [];
    for (const p of _ownersAllMeta) {
      if (p.slug === _meSlug) _meBucket.push(p);
      else if (_followingSet.has(p.slug)) _followedBucket.push(p);
      else _restBucket.push(p);
    }
    _followedBucket.sort(
      (a, b) => (b.twitterFollowers || 0) - (a.twitterFollowers || 0),
    );
    // Original-author avatar: when this doc is a retweet/quote
    // whose source author is itself one of our indexed
    // personalities (e.g. raphael retweets antoine_chaffin), we
    // surface the source author's avatar at the head of the stack
    // so the card reads "antoine's tweet, shared by raphael".
    // The `is-original` class lets the CSS mark the bubble (small
    // badge, accent border) so the user can tell it apart from the
    // sharers.
    const _origMeta = (() => {
      if (!isTweetDoc(d)) return null;
      const m = /^(?:Retweet|Quoting)\s+@(\w+)/.exec(d.summary || "");
      if (!m) return null;
      const handle = m[1].toLowerCase();
      const slug = state.slugByTwitterHandle?.[handle];
      if (!slug) return null;
      // Don't double-render the original-author bubble when the
      // sharer IS the original author (shouldn't happen for
      // retweets, but a self-quote would).
      if (_ownersAllMeta.some((p) => p.slug === slug)) return null;
      const meta = state.perSlugMeta[slug];
      if (!meta) return null;
      return { ...meta, _original: true };
    })();
    const ownersMeta = [
      ...(_origMeta ? [_origMeta] : []),
      ..._meBucket,
      ..._followedBucket,
      ...weightedShuffleByPopularity(_restBucket),
    ];
    // Show the avatar stack whenever we're NOT on a single
    // personality's page — i.e. multi-library mode AND feed mode
    // (libs.size === 0). On a single library, the owner is implicit
    // from the page header so the stack would be redundant — *except*
    // when this card surfaces an indexed personality as the original
    // author of a retweet/quote, which the page header doesn't say.
    const showStack =
      (state.libs.size !== 1 || !!_origMeta) && ownersMeta.length;
    const ownersHtml = showStack
      ? `<div class="shared-by" aria-label="Shared by ${ownersMeta.length} ${ownersMeta.length === 1 ? "person" : "people"}">${ownersMeta
          .map((p) => {
            // Each avatar is its own hover-popover host. The avatar
            // itself links to the user's library; the popover
            // surfaces "View profile" plus (depending on follow
            // state) a Follow or Unfollow action.
            const followed = _peopleRail?.following?.has?.(p.slug);
            const isMe = me && me.slug === p.slug;
            // Anonymous viewers see neutral avatars (no follow
            // affordance). Signed-in viewers see followed people
            // at full opacity and non-followed strangers slightly
            // faded — the popover lets them flip the state.
            const showUnfollow = !!me && !isMe && followed;
            const showFollow = !!me && !isMe && !followed;
            const hostClass =
              "ava-host" +
              (showFollow ? " is-not-followed" : "") +
              (p._original ? " is-original" : "");
            const popName = p._original
              ? `${escapeHtml(p.name || p.slug)} · original author`
              : escapeHtml(p.name || p.slug);
            const titleAttr = p._original
              ? `Original author: ${p.name || p.slug}`
              : p.name || p.slug;
            return `<span class="${hostClass}">
                <a class="ava" href="/search?libs=${encodeURIComponent(p.slug)}" data-name="${escapeAttr(p.name || p.slug)}" onclick="event.stopPropagation()" title="${escapeAttr(titleAttr)}"><img src="${escapeAttr(p.avatar || "")}" alt="" onerror="this.style.opacity=0"/></a>
                <span class="ava-pop" role="menu">
                  <span class="ava-pop-name">${popName}</span>
                  <a class="ava-pop-link" href="/search?libs=${encodeURIComponent(p.slug)}">View profile</a>
                  ${
                    showUnfollow
                      ? `<button class="ava-pop-action" type="button" data-unfollow="${escapeAttr(p.slug)}">Unfollow</button>`
                      : showFollow
                        ? `<button class="ava-pop-action is-follow" type="button" data-follow="${escapeAttr(p.slug)}">Follow</button>`
                        : ""
                  }
                </span>
              </span>`;
          })
          .join("")}</div>`
      : "";
    // Score badge: prefer the in-browser ColBERT re-ranker score
    // (per-doc, late-interaction) once the worker has scored this
    // particular row. Until then, fall back to the API's first-stage
    // retriever score so every row carries a number from the moment
    // it appears. The `reranked` class is a hook for finer styling
    // (slightly bolder ink) that highlights rows the worker has
    // already touched, separating them visually from the still-to-
    // be-scored tail.
    const hasRerank = typeof d.colbertScore === "number";
    const score = hasRerank ? d.colbertScore : d.similarity;
    const scoreHtml =
      state.query && typeof score === "number"
        ? `<span class="score${hasRerank ? " reranked" : ""}" title="${hasRerank ? "ColBERT re-ranker" : "Retriever"} score">${score.toFixed(3)}</span>`
        : "";
    // Action cluster (edit / delete / favourite) — composed first so
    // we can inline it into the foot-right row below. With the cluster
    // living in the same flex container as the avatar stack, it never
    // gets hidden behind a wide row of sharer pictures: it just sits
    // to the left of them, scoot included.
    // HN front-page pick: not in the user's library yet, so the heart
    // (which only toggles `favorite_documents`) would target a row
    // that doesn't exist. We swap in a "Save to library" button that
    // POSTs to /auth/me/documents/bulk; the next pipeline run picks
    // up the freshly-inserted row (indexed = FALSE) and embeds it.
    const isPick = !!d.picked;
    const actionsHtml = `<div class="result-actions">
        ${
          isDocOwnedByMe(d) && !isPick
            ? `<button class="act act-edit" title="Edit title & summary" data-edit="${encodeURIComponent(d.url)}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M12 20h9"/>
                  <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/>
                </svg>
              </button>
              <button class="act act-delete" title="Delete from your library" data-delete="${encodeURIComponent(d.url)}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                  <path d="M10 11v6M14 11v6"/>
                  <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                </svg>
              </button>`
            : ""
        }
        ${
          me
            ? (() => {
                // Unified upvote control. For an HN front-page pick
                // (not yet in `documents`), the click handler will
                // first POST the doc to the user's library AND mark
                // it favorited; for a regular doc it just toggles
                // the favorite. The same icon + state class is used
                // either way so the filter chip (state.favorites)
                // works uniformly.
                const pickAttrs = isPick
                  ? `data-pick="1" data-pick-title="${escapeAttr(d.title || "")}" data-pick-summary="${escapeAttr(d.summary || "")}" data-pick-date="${escapeAttr(d.date || "")}" data-pick-source-url="${escapeAttr(d.source_url || "")}"`
                  : "";
                return `<button class="act act-fav${isFav ? " on" : ""}" title="${isFav ? "Remove upvote" : "Upvote"}" aria-pressed="${isFav}" data-fav="${encodeURIComponent(d.url)}" ${pickAttrs}>
                <svg viewBox="0 0 24 24" fill="${isFav ? "currentColor" : "none"}" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">
                  <path d="M12 4 L20 13 L15.5 13 L15.5 20 L8.5 20 L8.5 13 L4 13 Z"/>
                </svg>
              </button>`;
              })()
            : ""
        }
      </div>`;

    const footRightInner = `${actionsHtml}${scoreHtml}${ownersHtml}`;
    const footRight = footRightInner.trim()
      ? `<div class="result-foot-right">${footRightInner}</div>`
      : "";

    // `is-seen` dims the card so the user can tell at a glance which
    // ones are recycled, no textual label. Only set when the server
    // flagged this row as already seen (include_seen=1 path) — the
    // default hide-seen state never emits flagged rows.
    const isSeen = !!d.alreadySeen;
    return `<article class="result${isSeen ? " is-seen" : ""}" data-url="${escapeAttr(d.url)}"${isSeen ? ' data-seen="1"' : ""}>
      <div class="result-body">
        <div class="result-kicker">
          ${sourcePill}
          ${d.date ? `<span class="dot">·</span><span>${dateLabel(d.date)}</span>` : ""}
          ${viaPill}
        </div>
        <a href="${safeHref(d.url)}" target="_blank" rel="noopener"><h3>${highlightMatches((d.cleanTitle && d.cleanTitle.trim()) || (isTweetDoc(d) ? tweetTitle(d) : d.title), state.query)}</h3></a>
        ${(() => {
          // Display preference: the pedagogical clean_summary takes
          // precedence over the raw `summary` whenever the daemon
          // has populated it. Search continues to run against the
          // raw `summary` server-side, so this is purely a display-
          // layer override. Newlines in clean_summary are preserved
          // verbatim — the `.result-summary--clean` CSS uses
          // `white-space: pre-line` so paragraph breaks render as
          // the daemon intended.
          const cs = (d.cleanSummary || "").trim();
          if (cs) {
            // Three pieces, in order:
            //   1. The cleaned <p> with bare-URL autolinking.
            //   2. Any URLs the original post referenced that the
            //      cleaned summary didn't already surface — rendered
            //      as a quiet anchor row so no URL is ever lost.
            //   3. The OG-preview tiles for any `linked_urls` the
            //      pipeline cached.
            //   4. Tweet media (photos / video posters) for the
            //      twitter / x path so the visual content survives.
            const summaryHtml = `<p class="result-summary result-summary--clean">${renderCleanSummaryHtml(cs, state.query)}</p>`;
            const cleanLower = cs.toLowerCase();
            const allUrls = Array.isArray(d.urls) ? d.urls : [];
            const linkedUrlsSet = new Set(
              (Array.isArray(d.linkedUrls) ? d.linkedUrls : [])
                .map((l) => (l && l.url) || "")
                .filter(Boolean),
            );
            const missingUrls = allUrls.filter(
              (u) =>
                u &&
                !cleanLower.includes(u.toLowerCase()) &&
                !linkedUrlsSet.has(u),
            );
            const missingHtml = missingUrls.length
              ? `<div class="result-summary-extra-links">${missingUrls
                  .map(
                    (u) =>
                      `<a href="${safeHref(u)}" target="_blank" rel="noopener">${escapeHtml(u)}</a>`,
                  )
                  .join("")}</div>`
              : "";
            const linksHtml = renderDocLinkCards(d);
            const mediaHtml = isTweetDoc(d) ? renderTweetMediaOnly(d) : "";
            return summaryHtml + missingHtml + mediaHtml + linksHtml;
          }
          // Tweet-specific renderer: thread parts get their own
          // boxes, inline media URLs become real thumbnails, etc.
          // Only used when clean_summary is empty.
          if (isTweetDoc(d)) {
            const hasLinks =
              Array.isArray(d.linkedUrls) && d.linkedUrls.length > 0;
            if (!d.summary && !hasLinks) return "";
            return renderTweetSummary(d);
          }
          if (!d.summary) return "";
          // For non-arxiv: keep the existing 320 char clip. For
          // arxiv / scholar / paperswithcode: show the full
          // abstract (the user explicitly asked for this).
          const isPaper =
            /^(arxiv|scholar|dblp|openreview|semantic|paperswithcode)/i.test(
              d.source || "",
            );
          const s = cleanDescription(d.summary, isPaper ? 100000 : 320);
          return s
            ? `<p class="result-summary">${highlightMatches(s, state.query)}</p>`
            : "";
        })()}
      </div>
      <div class="result-tags-row">
        <div class="result-tags">
          ${tags
            .map((t) => {
              // `state.tags` is canonicalised to lowercase on URL
              // hydration, so we lowercase the chip's `data-tag` to
              // keep the click handler's `state.tags.has(tag)` check
              // case-insensitive. Without this a doc tag "Twitter"
              // would never match the lowercase URL state and the
              // toggle ping-ponged across multiple clicks.
              //
              // The `.active` highlight that used to paint clicked
              // tags an accent colour has been dropped — the
              // applied filter is visible in the top strip already.
              const lower = (t || "").toLowerCase();
              return `<button class="result-tag" data-tag="${escapeAttr(lower)}">${highlightMatches(t, state.query)}</button>`;
            })
            .join("")}
          <button class="result-similar result-similar-as-tag" title="Show similar documents" aria-expanded="false" data-similar="${encodeURIComponent(d.url)}">
            <span class="result-similar-label">Related</span>
            <svg class="result-similar-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>
        </div>
      </div>
      ${footRight}
      <div class="similar-panel" data-similar-url="${encodeURIComponent(d.url)}"></div>
    </article>`;
  }
  /* Visual update for every `[data-fav]` button on the page that
   * targets `url` — keeps the main result row's heart and the heart
   * inside any open similar-panel in sync, since the same URL can
   * appear in multiple places at once. */
  function _setFavVisualState(url, on) {
    const encoded = encodeURIComponent(url);
    document.querySelectorAll(`[data-fav="${encoded}"]`).forEach((btn) => {
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-pressed", String(on));
      btn.title = on ? "Remove upvote" : "Upvote";
      const svg = btn.querySelector("svg");
      if (svg) svg.setAttribute("fill", on ? "currentColor" : "none");
    });
  }

  /* Wire every favourite-toggle button under `scope`. Idempotent —
   * each button keeps its own listener via a sentinel attribute so
   * repaints (the rerank repaint, similar-panel repaint, etc.) can
   * call this as often as they want without doubling up handlers. */
  function wireFavButtons(scope) {
    scope.querySelectorAll("[data-fav]").forEach((b) => {
      if (b.dataset.favWired === "1") return;
      b.dataset.favWired = "1";
      b.addEventListener("click", async (e) => {
        // The similar-panel rows are wrapped in <a>, so a heart-click
        // on a similar-row would otherwise navigate the link too.
        e.preventDefault();
        e.stopPropagation();
        // Defensive auth gate — buttons are normally hidden when `me`
        // is null, but a stale render could leak through.
        if (!me) {
          $("authBtn").click();
          return;
        }
        const url = decodeURIComponent(b.dataset.fav);
        const wasOn = state.favorites.has(url);
        // Optimistic local update — every button bound to this URL
        // flips at once so the upvote arrow in the main row and the
        // one in the similar-panel stay coherent.
        _setFavVisualState(url, !wasOn);
        if (wasOn) state.favorites.delete(url);
        else state.favorites.add(url);
        // HN front-page picks aren't yet rows in `documents`, so
        // toggling the favorite alone would orphan the upvote. When
        // the user upvotes a pick we first POST the doc into their
        // library (indexed=FALSE, next pipeline run will embed it);
        // the favorite write rides along on the same request via
        // `favorite: true`.
        const isPick = b.dataset.pick === "1";
        if (isPick && !wasOn) {
          try {
            await fetch(`${API_BASE}/auth/me/documents/bulk`, {
              method: "POST",
              credentials: "include",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                documents: [
                  {
                    url,
                    title: b.dataset.pickTitle || "",
                    summary: b.dataset.pickSummary || "",
                    date: b.dataset.pickDate || "",
                    source: "hackernews",
                    source_url: b.dataset.pickSourceUrl || null,
                    public: true,
                  },
                ],
                favorite: true,
              }),
            });
            window.dispatchEvent(new CustomEvent("knowledge:bookmark-added"));
          } catch (err) {
            console.warn("pick.upvote.save_failed", err);
          }
        } else {
          // Fire-and-forget the favorite write, then drop the personal-
          // page + timeline caches so the next visit to /<my-slug>
          // sees the freshly-mirrored row (server-side POST
          // /auth/me/favorite-docs also INSERTs the doc into the
          // caller's documents table — see
          // api/src/handlers/favorite_docs.rs). We intentionally do
          // NOT dispatch `knowledge:bookmark-added` here: that event
          // re-renders the feed, and the user expects the feed to
          // stay put after an upvote (the visual heart state is
          // already updated optimistically by `_setFavVisualState`).
          K.toggleFavorite(url, wasOn).then((ok) => {
            if (!ok) return;
            // Both branches need cache invalidation now that
            // un-upvote also deletes the mirrored documents row
            // server-side (when created_via_favorite = TRUE — see
            // api/src/handlers/favorite_docs.rs::remove). Without
            // this the personal page would keep showing a doc the
            // user just removed until the cache expired.
            _timelineCache.clear();
            window.KnowledgeSessionCache?.invalidatePrefix?.(
              _TIMELINE_SS_PREFIX,
            );
            if (me?.slug) {
              K.invalidatePersonalDocs?.(me.slug);
              K.invalidateUnindexed?.(me.slug);
            }
          });
        }
        if (state.favorites.size === 0) state.sources.delete(FAV_SOURCE_KEY);
        // Feed mode has its own source aggregator (across followees +
        // self); calling rebuildAllSources() there would wipe the rail
        // because state.libs is empty. Pick the right rebuilder.
        if (state.libs.size === 0) {
          rebuildAllSourcesForFeed().then(renderSrc);
        } else {
          rebuildAllSources();
          renderSrc();
        }
        if (state.sources.has(FAV_SOURCE_KEY) || wasOn) {
          writeUrl();
          refresh();
        }
      });
    });
  }

  function wireResults(root) {
    const scope = root || $("results");
    // First render of this refresh — the spinner has served its
    // purpose, tear it down.
    hideResultsSpinner();
    wireFavButtons(scope);

    // Behavioural click tracking via delegation on the results
    // container. One handler per refresh — flag the container so the
    // listener doesn't stack on re-renders. Captures clicks on the
    // title link, the inline media tiles, and the link card embeds.
    if (window.kn && !scope.dataset.trackWired) {
      scope.dataset.trackWired = "1";
      scope.addEventListener("click", (e) => {
        const link = e.target.closest("a[href]");
        if (!link) return;
        const article = link.closest("article.result");
        if (!article) return;
        const docUrl = article.dataset.url;
        if (!docUrl) return;
        const docs = state.lastDocs || [];
        const idx = docs.findIndex((x) => x.url === docUrl);
        const doc = idx >= 0 ? docs[idx] : null;
        window.kn.track("click", {
          doc_url: docUrl,
          // 0-based rank on the result list; recommenders use this to
          // distinguish "clicked the top hit" from "scrolled past 20".
          position: idx >= 0 ? Math.min(32767, idx) : null,
          score:
            doc && typeof doc.similarity === "number" ? doc.similarity : null,
        });
      });
    }
    // card_seen: per-card viewport observer. Active on every list
    // surface — feed, personal pages, search — so the dwell signal
    // covers the full engagement footprint (used both to tune the
    // hide-seen filter and as ML training input). The observer
    // itself no-ops for anonymous viewers.
    if (window.kn && typeof window.kn.observeCard === "function") {
      scope.querySelectorAll("article.result[data-url]").forEach((el) => {
        window.kn.observeCard(el, el.dataset.url);
      });
    }
    scope.querySelectorAll("[data-tag]").forEach((b) =>
      b.addEventListener("click", () => {
        // Lowercase so the toggle matches `state.tags` even when
        // the doc's original tag has different casing.
        const tag = (b.dataset.tag || "").toLowerCase();
        if (!tag) return;
        // Toggle: clicking an active chip clears that filter.
        if (state.tags.has(tag)) state.tags.delete(tag);
        else state.tags.add(tag);
        writeUrl();
        refresh();
      }),
    );
    scope
      .querySelectorAll("[data-similar]")
      .forEach((b) => b.addEventListener("click", () => toggleSimilar(b)));
    scope.querySelectorAll("[data-ban-source]").forEach((b) =>
      b.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const key = b.dataset.banSource;
        if (!key) return;
        // Mutual exclusion with the include set — banning a source
        // that was previously a positive filter clears it there.
        state.sources.delete(key);
        state.excludedSources.add(key);
        renderSrc();
        writeUrl();
        refresh();
      }),
    );
    scope.querySelectorAll("[data-edit]").forEach((b) => {
      if (b.dataset.editWired === "1") return;
      b.dataset.editWired = "1";
      b.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const url = decodeURIComponent(b.dataset.edit);
        openInlineEditor(url, b);
      });
    });
    scope.querySelectorAll("[data-unfollow]").forEach((b) => {
      if (b.dataset.unfollowWired === "1") return;
      b.dataset.unfollowWired = "1";
      b.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!me) return;
        const slug = b.dataset.unfollow;
        // Optimistic flip: drop the slug from the cached follow set
        // so every other rendering surface (people rail, profile
        // header, future cards) sees the change immediately.
        _peopleRail.following.delete(slug);
        // Repaint every avatar referencing this slug in the current
        // view — the row stays, but it now reads as "not followed"
        // (faded + Follow option). Find the closest .ava-host and
        // flip the class + swap the popover button.
        scope
          .querySelectorAll(`[data-unfollow="${CSS.escape(slug)}"]`)
          .forEach((btn) => {
            const host = btn.closest(".ava-host");
            if (host) host.classList.add("is-not-followed");
            btn.outerHTML = `<button class="ava-pop-action is-follow" type="button" data-follow="${escapeAttr(slug)}">Follow</button>`;
          });
        try {
          const r = await fetch(
            `${API_BASE}/api/follow/${encodeURIComponent(slug)}`,
            { method: "DELETE", credentials: "include" },
          );
          if (!r.ok) throw new Error("HTTP " + r.status);
        } catch (err) {
          // Roll back the optimistic state on failure.
          _peopleRail.following.add(slug);
          console.warn("[avatar-unfollow]", err);
        }
        // Re-wire the freshly-inserted Follow buttons.
        wireFollowButtonsIn(scope);
      });
    });

    /* Idempotently wire every [data-follow] inside `root`. The
     * unfollow handler also calls this after it transmutes a button
     * into Follow so newly-inserted nodes pick up listeners. */
    function wireFollowButtonsIn(root) {
      root.querySelectorAll("[data-follow]").forEach((b) => {
        if (b.dataset.followWired === "1") return;
        b.dataset.followWired = "1";
        b.addEventListener("click", async (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!me) {
            window.KnowledgeAuth?.open("login");
            return;
          }
          const slug = b.dataset.follow;
          // Optimistic add: mark followed in the shared set so the
          // people rail, profile header, and other cards re-paint
          // consistently.
          _peopleRail.following.add(slug);
          // Repaint every avatar referencing this slug in the
          // current view — drop the faded state and swap the popover
          // button to Unfollow.
          root
            .querySelectorAll(`[data-follow="${CSS.escape(slug)}"]`)
            .forEach((btn) => {
              const host = btn.closest(".ava-host");
              if (host) host.classList.remove("is-not-followed");
              btn.outerHTML = `<button class="ava-pop-action" type="button" data-unfollow="${escapeAttr(slug)}">Unfollow</button>`;
            });
          try {
            const r = await fetch(
              `${API_BASE}/api/follow/${encodeURIComponent(slug)}`,
              { method: "POST", credentials: "include" },
            );
            if (!r.ok) throw new Error("HTTP " + r.status);
          } catch (err) {
            _peopleRail.following.delete(slug);
            console.warn("[avatar-follow]", err);
          }
        });
      });
    }
    wireFollowButtonsIn(scope);
    scope.querySelectorAll("[data-delete]").forEach((b) => {
      if (b.dataset.deleteWired === "1") return;
      b.dataset.deleteWired = "1";
      b.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!me) return;
        const url = decodeURIComponent(b.dataset.delete);
        if (!confirm("Delete this document from your library?")) return;
        const article = b.closest(".result");
        // Optimistic remove — restore the card if the server says no.
        const placeholder = article?.previousElementSibling;
        article?.remove();
        try {
          const qs = new URLSearchParams({ url });
          const r = await fetch(
            `${API_BASE}/auth/me/documents?${qs.toString()}`,
            { method: "DELETE", credentials: "include" },
          );
          if (!r.ok) throw new Error("HTTP " + r.status);
          // Drop the doc from local state so subsequent renders don't
          // bring it back. We never resurrect via /api/timeline because
          // the server now filters `deleted = TRUE`. We also stamp the
          // URL into state.deletedUrls so the search wrappers filter
          // it out of ColBERT hits this session.
          if (Array.isArray(state.lastDocs)) {
            state.lastDocs = state.lastDocs.filter((d) => d.url !== url);
          }
          state.favorites?.delete?.(url);
          state.deletedUrls?.add?.(url);
        } catch (err) {
          console.warn("[doc-delete]", err);
          // Restore the card by re-rendering from state.lastDocs.
          if (article && placeholder?.parentElement) {
            placeholder.insertAdjacentElement("afterend", article);
          }
        }
      });
    });
  }

  /* Inline title/summary editor for the signed-in user's own rows.
   *
   * Replaces the card's title <h3> and summary <p> with form inputs
   * in-place. Save → PATCH /auth/me/documents; on success patches
   * state.lastDocs and re-renders just this card. Cancel restores
   * the original DOM by re-rendering from the unchanged doc. */
  function openInlineEditor(urlArg, triggerBtn) {
    const article = triggerBtn?.closest(".result, article");
    if (!article) return;
    // url is mutable inside the editor so a successful URL rewrite
    // points the restore() lookup at the new key.
    let url = urlArg;
    const doc = (state.lastDocs || []).find((x) => x.url === url);
    if (!doc) return;
    if (article.dataset.editing === "1") return;
    article.dataset.editing = "1";

    const titleEl = article.querySelector(".result-title, h3");
    const summaryEl = article.querySelector(".result-summary, .summary");
    const origTitle = doc.title || "";
    const origSummary = doc.summary || "";
    const origTags = Array.isArray(doc.tags) ? doc.tags.slice() : [];

    if (titleEl) {
      titleEl.innerHTML = `<input type="text" class="edit-title" maxlength="200" value="${escapeAttr(
        origTitle,
      )}" />`;
    }
    if (summaryEl) {
      summaryEl.innerHTML = `<textarea class="edit-summary" rows="4">${escapeHtml(origSummary)}</textarea>
        <div class="edit-tags" data-edit-tags>
          <div class="edit-tags-chips"></div>
          <input type="text" class="edit-tags-input" placeholder="Add tags (press comma or Enter)" autocomplete="off" spellcheck="false" />
        </div>
        <div class="edit-actions">
          <button type="button" class="edit-cancel">Cancel</button>
          <button type="button" class="edit-save">Save</button>
        </div>`;
    }
    const titleInput = article.querySelector(".edit-title");
    const summaryInput = article.querySelector(".edit-summary");
    const tagsHost = article.querySelector(".edit-tags-chips");
    const tagsInput = article.querySelector(".edit-tags-input");
    const editTags = origTags.slice();
    const renderEditTags = () => {
      if (!tagsHost) return;
      tagsHost.innerHTML = editTags
        .map(
          (t, i) =>
            `<span class="compose-tag-chip" data-i="${i}">${escapeHtml(t)}<button type="button" class="x" data-remove="${i}" aria-label="Remove tag">×</button></span>`,
        )
        .join("");
    };
    const commitEditTag = (raw) => {
      const t = (raw || "")
        .trim()
        .replace(/^,+|,+$/g, "")
        .trim();
      if (!t) return;
      const k = t.toLowerCase();
      if (editTags.some((x) => x.toLowerCase() === k)) return;
      editTags.push(t);
      renderEditTags();
    };
    renderEditTags();
    tagsInput?.addEventListener("input", (e) => {
      const v = e.target.value;
      if (v.includes(",")) {
        const parts = v.split(",");
        const tail = parts.pop();
        for (const p of parts) commitEditTag(p);
        e.target.value = tail.trimStart();
      }
    });
    tagsInput?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (e.target.value.trim()) {
          commitEditTag(e.target.value);
          e.target.value = "";
        }
      } else if (
        e.key === "Backspace" &&
        e.target.value === "" &&
        editTags.length
      ) {
        editTags.pop();
        renderEditTags();
      }
    });
    tagsHost?.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove]");
      if (!btn) return;
      const i = +btn.dataset.remove;
      if (Number.isInteger(i)) {
        editTags.splice(i, 1);
        renderEditTags();
      }
    });
    titleInput?.focus();

    const restore = () => {
      article.dataset.editing = "";
      const idx = (state.lastDocs || []).findIndex((x) => x.url === url);
      if (idx >= 0) {
        const html = renderResult(state.lastDocs[idx]);
        const tmp = document.createElement("div");
        tmp.innerHTML = html;
        const fresh = tmp.firstElementChild;
        if (fresh) {
          article.replaceWith(fresh);
          wireResults(fresh.parentElement || $("results"));
        }
      }
    };

    article.querySelector(".edit-cancel")?.addEventListener("click", restore);
    article.querySelector(".edit-save")?.addEventListener("click", async () => {
      const newTitle = (titleInput?.value || "").trim();
      const newSummary = (summaryInput?.value || "").trim();
      if (!newTitle) {
        titleInput?.focus();
        return;
      }
      // Flush any in-progress tag the user hasn't comma-committed yet.
      const pending = (tagsInput?.value || "").trim();
      if (pending) {
        commitEditTag(pending);
        if (tagsInput) tagsInput.value = "";
      }
      // Mirror compose: spot a URL in the body, derive a source key
      // from its hostname via the shared sync module. If the row was
      // previously a text-only note (`knowledge://note/...`), the new
      // URL replaces it as the primary key.
      const spottedUrl = extractUrl(newSummary) || extractUrl(newTitle);
      const isOriginalNote = url.startsWith("knowledge://note/");
      const newUrl = spottedUrl && spottedUrl !== url ? spottedUrl : null;
      let newSource = null;
      if (spottedUrl) {
        const hostKey =
          (window.KnowledgeSync &&
            typeof window.KnowledgeSync.hostnameSourceKey === "function" &&
            window.KnowledgeSync.hostnameSourceKey(spottedUrl)) ||
          "";
        if (hostKey) newSource = hostKey;
        else if (isOriginalNote && newUrl) newSource = "bookmark";
      } else if (isOriginalNote) {
        // No URL anywhere → keep it a note.
        newSource = "note";
      }
      const saveBtn = article.querySelector(".edit-save");
      if (saveBtn) saveBtn.disabled = true;
      try {
        const body = {
          url,
          title: newTitle,
          summary: newSummary,
          tags: editTags.slice(),
        };
        if (newUrl) body.new_url = newUrl;
        if (newSource) body.source = newSource;
        const r = await fetch(`${API_BASE}/auth/me/documents`, {
          method: "PATCH",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const idx = (state.lastDocs || []).findIndex((x) => x.url === url);
        const finalUrl = newUrl || url;
        if (idx >= 0) {
          state.lastDocs[idx] = {
            ...state.lastDocs[idx],
            url: finalUrl,
            title: newTitle,
            summary: newSummary,
            tags: editTags.slice(),
            source: newSource || state.lastDocs[idx].source,
          };
        }
        // Re-point the closure's `url` so restore() finds the row.
        url = finalUrl;
        // Re-render the edited card in place first — that way the
        // user sees the change immediately even if the subsequent
        // refresh() comes back a beat later or doesn't yet see the
        // new row (cache race, source filter mismatch, etc.).
        restore();
        // Then dispatch the same event the compose dialog uses: it
        // busts the timeline + unindexed caches and runs a full
        // refresh that pulls the canonical post-merge state.
        window.dispatchEvent(new CustomEvent("knowledge:bookmark-added"));
      } catch (err) {
        console.warn("[doc-edit]", err);
        if (saveBtn) saveBtn.disabled = false;
      }
    });
  }

  /* Per-doc similar-docs cache so reopening doesn't re-fetch. */
  const similarCache = new Map();
  /* URLs whose similar-panel is currently open. Lifted out of the
   * DOM so a rerank pass (which blows away #results.innerHTML) can
   * restore the open state by repainting each panel after the
   * re-render — otherwise opening "Similar" while the worker is
   * still streaming scores would silently close itself. */
  const openSimilarUrls = new Set();

  function paintSimilarPanel(panel, doc) {
    const docs = similarCache.get(doc.url);
    const truncated =
      doc.title.length > 60 ? doc.title.slice(0, 60) + "…" : doc.title;
    const loadingHtml = docs
      ? ""
      : '<span class="similar-loading"><span></span><span></span><span></span></span>';
    let listHtml = "";
    if (docs) {
      // Use the exact same renderer the feed / personal page uses
      // so the related cards carry every affordance (favourite,
      // related-of-related, source pills, link preview cards,
      // tweet media, picked-from chips, etc). The wrapper class
      // `.similar-results` adds the visual indent that signals
      // "these are children of the Related button above".
      listHtml = docs.length
        ? docs.map((d) => renderResult(d)).join("")
        : '<div class="similar-empty">— no echoes found —</div>';
    }
    panel.innerHTML = `<div class="similar-head">
      <svg class="spark" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 2.5l1.55 4.95L18.5 9l-4.95 1.55L12 15.5l-1.55-4.95L5.5 9l4.95-1.55z"/>
        <path d="M19 14l.85 2.65L22.5 17.5l-2.65.85L19 21l-.85-2.65L15.5 17.5l2.65-.85z"/>
      </svg>
      <span>Related to</span>
      <span class="target"><b>${escapeHtml(truncated)}</b></span>
      ${loadingHtml}
    </div><div class="similar-list similar-results">${listHtml}</div>`;
    panel.classList.add("open");
    // Once the docs are in, drop the max-height cap so tall panels
    // (full feed-style cards × 6 can easily exceed 1500px) aren't
    // clipped at the bottom. The toggle path does this on
    // transitionend; here (re-paint after a rerank, or paint after
    // the fetch resolves) we don't have a transition to listen to.
    if (docs) panel.classList.add("is-fully-open");
    // Wire every interaction the inner cards expose — same suite
    // the top-level feed uses (favourites, tags, similar buttons,
    // source pills, lightbox triggers, etc).
    wireResults(panel);
  }

  /* Repaint every still-open similar-panel after #results was
   * re-rendered (e.g. by an in-flight rerank batch). Called from
   * applyRerank() right after wireResults(). */
  function restoreOpenSimilarPanels() {
    if (!openSimilarUrls.size) return;
    for (const url of openSimilarUrls) {
      const encoded = encodeURIComponent(url);
      const panel = $("results").querySelector(
        `.similar-panel[data-similar-url="${encoded}"]`,
      );
      if (!panel) continue;
      const article = panel.closest(".result");
      const btn = article?.querySelector(`[data-similar="${encoded}"]`);
      if (btn) btn.classList.add("on");
      const doc = state.lastDocs.find((d) => d.url === url);
      if (!doc) continue;
      paintSimilarPanel(panel, doc);
    }
  }

  /* Walk every place a doc could be stashed and return the first
   * match by URL. Used by toggleSimilar so opening Related on a
   * card nested inside another related panel works — those nested
   * docs aren't in state.lastDocs, they live in the similarCache
   * entry that produced the parent panel. */
  function findDocByUrl(url) {
    for (const d of state.lastDocs || []) {
      if (d && d.url === url) return d;
    }
    for (const cached of similarCache.values()) {
      if (!Array.isArray(cached)) continue;
      for (const d of cached) {
        if (d && d.url === url) return d;
      }
    }
    return null;
  }

  async function toggleSimilar(btn) {
    const url = decodeURIComponent(btn.dataset.similar);
    // Scope the panel lookup to the immediate `.result` ancestor —
    // querySelector would otherwise traverse into any nested
    // related panel's `.result` descendants and match the wrong
    // `.similar-panel` for the same URL.
    const article = btn.closest(".result");
    const panel = article.querySelector(
      `:scope > .similar-panel[data-similar-url="${btn.dataset.similar}"]`,
    );
    if (!panel) return;
    const isOpen = panel.classList.contains("open");

    if (isOpen) {
      // Closing — drop the no-cap class so max-height kicks back
      // in and the collapse transition has something to animate
      // against. Then remove .open on the next frame so the
      // browser registers the max-height change before the
      // collapse starts.
      panel.classList.remove("is-fully-open");
      requestAnimationFrame(() => {
        panel.classList.remove("open");
      });
      btn.classList.remove("on");
      btn.setAttribute("aria-expanded", "false");
      openSimilarUrls.delete(url);
      return;
    }
    btn.classList.add("on");
    btn.setAttribute("aria-expanded", "true");
    // Behavioural signal: "related" expansion = the user wanted more
    // like this doc. Stronger relevance signal than a click for the
    // recommender to learn from.
    if (window.kn) window.kn.track("find_similar", { doc_url: url });
    openSimilarUrls.add(url);
    // After the max-height open transition settles, unbind the cap
    // so panels with many related cards (full feed-style rows can
    // each be 250–400px tall) aren't clipped. transitionend fires
    // once per animated property — guard for the max-height one.
    const onOpenEnd = (e) => {
      if (e.propertyName !== "max-height") return;
      panel.removeEventListener("transitionend", onOpenEnd);
      if (panel.classList.contains("open")) {
        panel.classList.add("is-fully-open");
      }
    };
    panel.addEventListener("transitionend", onOpenEnd);
    // Safety net — if the transitionend never fires (e.g. tab was
    // backgrounded mid-animation), drop the cap after the expected
    // duration anyway.
    setTimeout(() => {
      if (panel.classList.contains("open")) {
        panel.classList.add("is-fully-open");
      }
    }, 450);

    // Resolve the doc from anywhere we've seen it: top-level feed
    // first, then every related-cache entry. This is what makes
    // Related-within-Related work — a card painted inside another
    // related panel isn't in state.lastDocs, but it IS in the
    // similarCache slot under its parent's URL.
    const doc = findDocByUrl(url);
    if (!doc) return;

    // Paint immediately — loading state if no cached results yet.
    paintSimilarPanel(panel, doc);

    if (!similarCache.has(url)) {
      let docs;
      try {
        const libs = [...state.libs];
        if (libs.length === 0) {
          // Feed mode: query the cross-library __all__ index, then
          // scope to (followees ∪ self) and group by URL so multiple
          // owners of the same doc collapse into one row with a
          // stacked avatar list. Mirrors the feed-search path.
          const scope = me
            ? new Set([...(_peopleRail?.following || []), me.slug])
            : null;
          let rows = await K.findSimilar({
            indexName: ALL_INDEX_NAME,
            doc,
            topK: 50,
          }).catch(() => []);
          if (scope) rows = rows.filter((d) => scope.has(d.owner));
          const map = new Map();
          for (const d of rows) {
            if (d.url === url) continue;
            const owner = d.owner || "";
            const ex = map.get(d.url);
            if (!ex) {
              map.set(d.url, {
                ...d,
                _from: owner,
                _owners: owner ? [owner] : [],
              });
            } else {
              if (owner && !ex._owners.includes(owner)) ex._owners.push(owner);
              if ((d.similarity || 0) > (ex.similarity || 0)) {
                const owners = ex._owners;
                Object.assign(ex, d);
                ex._owners = owners;
                ex._from = owner;
              }
            }
          }
          docs = Array.from(map.values())
            .sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
            .slice(0, 6);
        } else {
          // Library-scoped path. Fan out across every active library —
          // same logic as main refresh.
          const all = await Promise.all(
            libs.map((s) =>
              K.findSimilar({ indexName: s, doc, topK: 7 })
                .then((rs) => rs.map((d) => ({ ...d, _from: s })))
                .catch(() => []),
            ),
          );
          const map = new Map();
          for (const arr of all)
            for (const d of arr) {
              if (d.url === url) continue;
              const ex = map.get(d.url);
              if (!ex) map.set(d.url, { ...d, _owners: [d._from] });
              else {
                if (!ex._owners.includes(d._from)) ex._owners.push(d._from);
                if ((d.similarity || 0) > (ex.similarity || 0)) {
                  const owners = ex._owners;
                  Object.assign(ex, d);
                  ex._owners = owners;
                }
              }
            }
          docs = Array.from(map.values())
            .sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
            .slice(0, 6);
        }
      } catch {
        docs = [];
      }
      similarCache.set(url, docs);
    }

    // Bail if the panel was closed while we were fetching, or if a
    // rerank pass replaced #results since we started — the rerank
    // restorer will repaint from cache.
    if (!openSimilarUrls.has(url)) return;
    const livePanel = $("results").querySelector(
      `.similar-panel[data-similar-url="${btn.dataset.similar}"]`,
    );
    if (!livePanel) return;
    paintSimilarPanel(livePanel, doc);
  }

  function renderSimilarRow(d) {
    const host = K.hostOf(d.url);
    const fromMeta =
      d._from && d._from !== state.hostSlug ? state.perSlugMeta[d._from] : null;
    const isFav = state.favorites.has(d.url);
    // Heart only renders when signed-in. The button lives inside the
    // <a> row, so its click handler must `preventDefault` +
    // `stopPropagation` — `wireFavButtons` does both.
    const favBtn = me
      ? `<button class="act act-fav similar-fav${isFav ? " on" : ""}" title="${isFav ? "Remove upvote" : "Upvote"}" aria-pressed="${isFav}" data-fav="${encodeURIComponent(d.url)}">
          <svg viewBox="0 0 24 24" fill="${isFav ? "currentColor" : "none"}" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">
            <path d="M12 4 L20 13 L15.5 13 L15.5 20 L8.5 20 L8.5 13 L4 13 Z"/>
          </svg>
        </button>`
      : "";
    /* Tweet rows swap the generic favicon for the bundled X / Twitter
     * bird, matching how the main result cards' source pill renders.
     * Falls back to the favicon for everything else. */
    const tweetRow = isTweetDoc(d);
    const iconSrc = tweetRow ? "/icons/twitter.png" : K.faviconUrl(host);
    return `<a class="similar-row" href="${safeHref(d.url)}" target="_blank" rel="noopener">
      <span class="ico"><img src="${escapeAttr(iconSrc)}" alt="" onerror="this.style.display='none'"/></span>
      <div class="body">
        <h4>${escapeHtml(tweetRow ? tweetTitle(d) : d.title)}</h4>
        <div class="meta">
          <span class="src">${escapeHtml(displaySource(d.source || host))}</span>
          ${d.date ? `<span class="dot">·</span><span>${dateLabel(d.date)}</span>` : ""}
          ${fromMeta ? `<span class="dot">·</span><span><img class="from-ava" src="${escapeAttr(fromMeta.avatar || "")}" alt="" onerror="this.style.display='none'"/>${escapeHtml((fromMeta.name || d._from).split(" ")[0])}</span>` : ""}
        </div>
      </div>
      ${typeof d.similarity === "number" ? `<span class="similar-score">${d.similarity.toFixed(2)}</span>` : ""}
      ${favBtn}
      <span class="arrow">→</span>
    </a>`;
  }
  // escapeHtml + escapeAttr come from /lib/utils.js

  /* Highlight every occurrence of a query token in a piece of text.
   * Mirrors the highlight() in web/search.jsx: lowercase tokenise,
   * drop short tokens, build one regex with all keywords, split,
   * wrap matches in <mark class="hl">, escape each part along the
   * way so the output is HTML-safe. */

  /* Render a clean_summary that may contain bare URLs from the
   * original tweet. The daemon is instructed to leave plain URLs
   * verbatim (no Markdown link syntax — the frontend escapes the
   * brackets), so we autolink them here at render time. Splits
   * the text on the URL regex, escapes-and-highlights the
   * non-URL chunks, and wraps the URL chunks in <a target=_blank>.
   * Result is HTML-safe end-to-end. */
  function renderCleanSummaryHtml(text, query) {
    if (!text) return "";
    const URL_RE = /(https?:\/\/[^\s<>"'`]+)/g;
    const parts = String(text).split(URL_RE);
    // With a capture group in split's regex, odd indices are the
    // captured URLs and even indices are the text between them.
    return parts
      .map((part, i) => {
        if (i % 2 === 1) {
          // Trim trailing punctuation that almost certainly isn't
          // part of the URL ('paper at https://x.y/abs.' → drop
          // the trailing '.').
          let url = part;
          let trailing = "";
          while (url && /[.,;:!?)]$/.test(url)) {
            trailing = url.slice(-1) + trailing;
            url = url.slice(0, -1);
          }
          return `<a href="${safeHref(url)}" target="_blank" rel="noopener">${escapeHtml(url)}</a>${escapeHtml(trailing)}`;
        }
        return highlightMatches(part, query);
      })
      .join("");
  }

  function highlightMatches(text, query) {
    const escaped = escapeHtml(text || "");
    if (!text) return escaped;
    /* Token universe = free-text query (filtered to ≥3 chars so
     * stop-words don't paint everything green) PLUS every active
     * tag chip (used verbatim — the user picked it explicitly, so
     * even a 2-letter tag like "ai" should highlight). The two
     * paths are complementary: a tag-only filter still gets its
     * matches lit up across the title / summary / source pill. */
    const queryTokens = (query || "")
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 2);
    const tagTokens = state.tags
      ? [...state.tags].map((t) => String(t).toLowerCase()).filter(Boolean)
      : [];
    const tokens = [...queryTokens, ...tagTokens];
    if (!tokens.length) return escaped;
    const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(${tokens.map(escRe).join("|")})`, "gi");
    const lookup = new Set(tokens);
    return text
      .split(re)
      .map((part) => {
        if (!part) return "";
        return lookup.has(part.toLowerCase())
          ? `<mark class="hl">${escapeHtml(part)}</mark>`
          : escapeHtml(part);
      })
      .join("");
  }

  /* ── Description cleaner ────────────────────────────────────
   * Runs at render time on every card's d.summary. Pipeline:
   *   1. decode HTML entities  (&amp;  &nbsp;  &#39; …)
   *   2. convert :emoji: shortcodes to native unicode
   *   3. strip markdown leftovers (**, _, `, [text](url) …)
   *   4. drop residual HTML tags
   *   5. collapse whitespace & trim leading punctuation
   * The output is plain text + unicode emoji; the OS renders
   * emojis with its colour font (Apple Color Emoji on Mac).
   */

  /* GitHub-flavoured shortcodes → unicode. ~120 most-common
   * entries cover the vast majority of repo descriptions. Add
   * more as you spot misses. Reference: api.github.com/emojis */
  const EMOJI = {
    rocket: "🚀",
    sparkles: "✨",
    fire: "🔥",
    art: "🎨",
    bug: "🐛",
    zap: "⚡",
    hammer: "🔨",
    wrench: "🔧",
    gear: "⚙️",
    tools: "🛠️",
    package: "📦",
    memo: "📝",
    books: "📚",
    book: "📖",
    computer: "💻",
    keyboard: "⌨️",
    mag: "🔍",
    mag_right: "🔎",
    lock: "🔒",
    unlock: "🔓",
    key: "🔑",
    white_check_mark: "✅",
    heavy_check_mark: "✔️",
    warning: "⚠️",
    no_entry: "⛔",
    x: "❌",
    rotating_light: "🚨",
    construction: "🚧",
    tada: "🎉",
    trophy: "🏆",
    100: "💯",
    gem: "💎",
    star: "⭐",
    glowing_star: "🌟",
    rainbow: "🌈",
    heart: "❤️",
    blue_heart: "💙",
    green_heart: "💚",
    purple_heart: "💜",
    yellow_heart: "💛",
    orange_heart: "🧡",
    black_heart: "🖤",
    white_heart: "🤍",
    thumbsup: "👍",
    "+1": "👍",
    thumbsdown: "👎",
    "-1": "👎",
    clap: "👏",
    raised_hands: "🙌",
    muscle: "💪",
    eyes: "👀",
    wave: "👋",
    pray: "🙏",
    ok_hand: "👌",
    v: "✌️",
    point_right: "👉",
    point_down: "👇",
    point_up: "👆",
    point_left: "👈",
    bulb: "💡",
    bookmark: "🔖",
    bookmark_tabs: "📑",
    clipboard: "📋",
    calendar: "📅",
    date: "📆",
    hourglass: "⌛",
    stopwatch: "⏱️",
    alarm_clock: "⏰",
    tv: "📺",
    movie_camera: "🎥",
    camera: "📷",
    microphone: "🎤",
    musical_note: "🎵",
    musical_notes: "🎶",
    speaker: "🔊",
    loudspeaker: "📢",
    cloud: "☁️",
    sunny: "☀️",
    crystal_ball: "🔮",
    compass: "🧭",
    brain: "🧠",
    electric_plug: "🔌",
    battery: "🔋",
    satellite: "🛰️",
    airplane: "✈️",
    car: "🚗",
    train: "🚆",
    ship: "🚢",
    recycle: "♻️",
    infinity: "♾️",
    repeat: "🔁",
    arrow_right: "➡️",
    arrow_left: "⬅️",
    arrow_up: "⬆️",
    arrow_down: "⬇️",
    arrow_forward: "▶️",
    arrow_backward: "◀️",
    leftwards_arrow_with_hook: "↩️",
    rightwards_arrow_with_hook: "↪️",
    smile: "😄",
    smiley: "😃",
    grinning: "😀",
    joy: "😂",
    laughing: "😆",
    wink: "😉",
    sunglasses: "😎",
    thinking: "🤔",
    100: "💯",
    mortar_board: "🎓",
    school: "🏫",
    office: "🏢",
    house: "🏠",
    earth_americas: "🌎",
    earth_asia: "🌏",
    earth_africa: "🌍",
    coffee: "☕",
    tea: "🍵",
    beer: "🍺",
    pizza: "🍕",
    cake: "🍰",
    cat: "🐱",
    dog: "🐶",
    whale: "🐳",
    penguin: "🐧",
    bee: "🐝",
    fish: "🐟",
    tropical_fish: "🐠",
    dolphin: "🐬",
    octopus: "🐙",
    turtle: "🐢",
    snake: "🐍",
    lion: "🦁",
    elephant: "🐘",
    seedling: "🌱",
    herb: "🌿",
    evergreen_tree: "🌲",
    deciduous_tree: "🌳",
    sun_with_face: "🌞",
    first_quarter_moon_with_face: "🌛",
    page_facing_up: "📄",
    scroll: "📜",
    newspaper: "📰",
    file_folder: "📁",
    open_file_folder: "📂",
    paperclip: "📎",
    chart_with_upwards_trend: "📈",
    chart_with_downwards_trend: "📉",
    bar_chart: "📊",
    label: "🏷️",
    round_pushpin: "📍",
    pushpin: "📌",
    link: "🔗",
    money_with_wings: "💸",
    credit_card: "💳",
    dollar: "💵",
    shield: "🛡️",
    crossed_swords: "⚔️",
    dart: "🎯",
    ghost: "👻",
    alien: "👽",
    robot: "🤖",
    hugs: "🤗",
    call_me_hand: "🤙",
    crossed_fingers: "🤞",
    sparkler: "🎇",
    fireworks: "🎆",
    balloon: "🎈",
  };

  /* HTML entity decoder — uses a textarea (DOMParser also works
   * but textarea is simpler and won't execute scripts since it
   * never enters a parsing state). Handles named (`&amp;`),
   * decimal (`&#39;`), and hex (`&#x27;`) entities in one pass. */
  let _entDecoder = null;
  function decodeEntities(s) {
    if (!s || s.indexOf("&") === -1) return s;
    if (!_entDecoder) _entDecoder = document.createElement("textarea");
    _entDecoder.innerHTML = s;
    return _entDecoder.value;
  }

  /* Convert :foo: shortcodes that match our map. Unknown shortcodes
   * are left alone (they're more readable than a literal `:foo:`
   * anyway, and dropping them risks eating timestamps like 12:30). */
  function emojify(s) {
    return s.replace(/:([a-z0-9_+-]+):/gi, (m, name) => {
      const u = EMOJI[name.toLowerCase()];
      return u || m;
    });
  }

  /* Strip the markdown leftovers most likely to appear in a one-
   * line description. Order matters: images before links (both use
   * `[…]` syntax), then formatting, then code, then headings. */
  function stripMarkdown(s) {
    return s
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1") // ![alt](url) → alt
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // [text](url) → text
      .replace(/(\*\*|__)(.+?)\1/g, "$2") // **bold** / __bold__
      .replace(/(?<!\w)([*_])([^*_\s][^*_]*?)\1(?!\w)/g, "$2") // *em*/_em_
      .replace(/~~(.+?)~~/g, "$1") // ~~strike~~
      .replace(/`([^`]+)`/g, "$1") // `inline code`
      .replace(/^#{1,6}\s+/gm, "") // # heading
      .replace(/^>\s+/gm, ""); // > quote
  }

  /* Remove residual HTML tags. Most descriptions are plain text but
   * scraped sources occasionally leave `<br>` or `<a>` in. We don't
   * try to render them — just strip. */
  function stripHtml(s) {
    return s.replace(/<\/?[a-z][^>]*>/gi, "");
  }

  /* Trim and normalise whitespace, including non-breaking spaces. */
  function normalizeSpace(s) {
    return s.replace(/[ \s]+/g, " ").trim();
  }

  /* Strip URLs, IPs, file refs, and the long unbroken artifact tokens
   * left over when scrapers eat the protocol slashes — patterns like
   * `SnakeVizhttpsgithub.comjiffyclubsnakeviz` or `127.0.0.13999http…`
   * Aggressive on purpose: descriptions don't need to be link-aware. */
  function stripUrlsAndArtifacts(s) {
    return (
      s
        // any http(s) token (with or without :// — covers glued cases)
        .replace(/https?:?\/?\/?[^\s]*/gi, " ")
        // IPv4 with optional port/path tail
        .replace(/\b\d{1,3}(?:\.\d{1,3}){3}[\w.:/-]*/g, " ")
        // file / image / asset orphans (.png, .html, .md, etc.)
        .replace(
          /\b[\w./-]+\.(?:png|jpe?g|gif|svg|webp|html?|md|txt|pdf|css|js|json|yaml|yml|toml|ini|sh|py|rs|go|ts|tsx)\b/gi,
          " ",
        )
        // standalone leading-dot tokens (.templates, .static, .gitignore)
        .replace(/(?:^|\s)\.[a-z][\w-]*\b/gi, " ")
        // any unbroken non-space run ≥ 28 chars — almost certainly junk
        // (real words top out around 20; URLs missing slashes go much longer)
        .replace(/\S{28,}/g, " ")
    );
  }

  /* Cap to the first complete sentence(s) under maxLen. Keeps the
   * first project's tagline, drops downstream README spillover. */
  function truncateAtSentence(s, maxLen = 320) {
    if (s.length <= maxLen) return s;
    const slice = s.slice(0, maxLen);
    const lastEnd = Math.max(
      slice.lastIndexOf(". "),
      slice.lastIndexOf("! "),
      slice.lastIndexOf("? "),
    );
    if (lastEnd > maxLen * 0.4) return slice.slice(0, lastEnd + 1).trim();
    const lastSpace = slice.lastIndexOf(" ");
    return (lastSpace > 0 ? slice.slice(0, lastSpace) : slice).trim() + "…";
  }

  function cleanDescription(text, limit = 320) {
    if (!text) return "";
    let s = String(text);
    s = decodeEntities(s);
    s = emojify(s);
    s = stripMarkdown(s);
    s = stripHtml(s);
    s = stripUrlsAndArtifacts(s);
    s = normalizeSpace(s);
    s = truncateAtSentence(s, limit);
    return s;
  }
  /* ── Auth + profile modal ─────────────────────────────────── */
  // `API_BASE` and `me` are hoisted near the top of the IIFE (search
  // for "let me = null") so the pre-paint boot `Promise.all` can
  // call `loadMe()` without tripping the TDZ on either of them.

  async function loadMe() {
    try {
      const r = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
      if (!r.ok) return null;
      return await r.json();
    } catch {
      return null;
    }
  }

  /* Theme toggle — flips the `data-theme` attribute on <html>
   * between "light" (default, the cream palette) and "dark" (the
   * editorial-midnight variant). The choice is persisted in
   * localStorage under the same key the welcome page uses, so the
   * preference rides along when the user navigates between the
   * grid and a search view.
   *
   * The synchronous boot script in search.html applies the saved
   * theme before first paint to avoid a flash; this handler only
   * has to swap the attribute and re-skin the icon. */
  const SUN_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M4.93 19.07l1.41-1.41"/><path d="M17.66 6.34l1.41-1.41"/></svg>';
  const MOON_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
  function syncThemeIcon() {
    const btn = $("themeToggle");
    if (!btn) return;
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    btn.innerHTML = isDark ? SUN_ICON : MOON_ICON;
    btn.setAttribute(
      "aria-label",
      isDark ? "Switch to light mode" : "Switch to dark mode",
    );
  }
  syncThemeIcon();
  // Twitter-style Post trigger — opens the compose dialog defined
  // below. The dialog accepts free-text + a URL anywhere in the
  // body; on submit it POSTs to /auth/me/documents/bulk and stars
  // the URL so it shows up in Favorites immediately. The simpler
  // `KnowledgeBookmark` dialog (in web/bookmark/) handles plain
  // URL-only bookmarks and stays the entry point on the welcome
  // page; compose is the richer authoring surface here.
  const postBtn = $("postTriggerBtn");
  if (postBtn) {
    postBtn.addEventListener("click", openComposeDialog);
  }

  /* Compose tag chips. Backing array is the source of truth; the
   * <input> only holds the *in-progress* tag (everything before the
   * user has committed it via comma / Enter / blur). renderComposeTags
   * paints the chips. */
  const _composeTags = [];
  function renderComposeTags() {
    const host = $("composeTagsChips");
    if (!host) return;
    host.innerHTML = _composeTags
      .map(
        (t, i) =>
          `<span class="compose-tag-chip" data-i="${i}">${escapeHtml(t)}<button type="button" class="x" data-remove="${i}" aria-label="Remove tag">×</button></span>`,
      )
      .join("");
    // Tag set is part of the draft — re-save whenever it changes.
    if (typeof saveComposeDraft === "function") saveComposeDraft();
  }
  function commitComposeTag(raw) {
    const t = (raw || "")
      .trim()
      .replace(/^,+|,+$/g, "")
      .trim();
    if (!t) return;
    const key = t.toLowerCase();
    if (_composeTags.some((x) => x.toLowerCase() === key)) return;
    _composeTags.push(t);
    renderComposeTags();
  }
  $("composeTagsInput")?.addEventListener("input", (e) => {
    const v = e.target.value;
    // Comma commits everything before the comma as a chip, leaves the
    // tail (what the user is still typing) in the input.
    if (v.includes(",")) {
      const parts = v.split(",");
      const tail = parts.pop();
      for (const p of parts) commitComposeTag(p);
      e.target.value = tail.trimStart();
    }
  });
  $("composeTagsInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const v = e.target.value;
      if (v.trim()) {
        commitComposeTag(v);
        e.target.value = "";
      }
    } else if (
      e.key === "Backspace" &&
      e.target.value === "" &&
      _composeTags.length
    ) {
      // Backspace on an empty input pops the last chip — same UX as
      // Twitter / Linear tag inputs.
      _composeTags.pop();
      renderComposeTags();
    }
  });
  $("composeTagsInput")?.addEventListener("blur", (e) => {
    if (e.target.value.trim()) {
      commitComposeTag(e.target.value);
      e.target.value = "";
    }
  });
  $("composeTagsChips")?.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-remove]");
    if (!btn) return;
    const i = +btn.dataset.remove;
    if (Number.isInteger(i)) {
      _composeTags.splice(i, 1);
      renderComposeTags();
    }
  });
  $("composeTagsField")?.addEventListener("click", (e) => {
    // Click anywhere in the field row (outside a chip) → focus the input.
    if (e.target.closest(".compose-tag-chip")) return;
    $("composeTagsInput")?.focus();
  });

  function openComposeDialog() {
    if (!me) {
      $("authBtn")?.click();
      return;
    }
    const back = $("composeBack");
    const root = $("compose");
    if (!root || !back) return;
    back.hidden = false;
    root.hidden = false;
    const av = $("composeAvatar");
    if (av) {
      if (me.avatar) {
        av.style.backgroundImage = `url('${String(me.avatar).replace(/['"\\]/g, encodeURIComponent)}')`;
        av.textContent = "";
      } else {
        av.style.backgroundImage = "";
        const initials = (me.name || me.slug || "?")
          .split(/\s+/)
          .slice(0, 2)
          .map((w) => (w[0] || "").toUpperCase())
          .join("");
        av.textContent = initials;
      }
    }
    const input = $("composeInput");
    input.value = "";
    const titleInput = $("composeTitleInput");
    if (titleInput) titleInput.value = "";
    const tagsInput = $("composeTagsInput");
    if (tagsInput) tagsInput.value = "";
    _composeTags.length = 0;
    renderComposeTags();
    // Restore any draft the user left behind on a prior close — title,
    // body, in-progress tag input, and committed tag chips.
    const draft = loadComposeDraft();
    if (draft) {
      if (titleInput) titleInput.value = draft.title || "";
      input.value = draft.body || "";
      if (tagsInput) tagsInput.value = draft.pendingTag || "";
      _composeTags.length = 0;
      for (const t of draft.tags || []) _composeTags.push(t);
      renderComposeTags();
    }
    setComposeHint("", "");
    syncComposeSubmitEnabled();
    syncComposeAutofillButton();
    // Forget the last URL the suggester saw so reopening with a
    // draft body that still contains a link re-fires the meta fetch.
    _composeUrlSeenLast = "";
    requestAnimationFrame(() => input.focus());
  }

  /* Compose draft persistence — keyed per signed-in user so two
   * accounts on the same machine don't trample each other. */
  const COMPOSE_DRAFT_PREFIX = "knowledge:compose-draft:";
  const _composeDraftKey = () => COMPOSE_DRAFT_PREFIX + (me?.slug || "anon");
  function saveComposeDraft() {
    if (!me) return;
    try {
      const title = $("composeTitleInput")?.value || "";
      const body = $("composeInput")?.value || "";
      const pendingTag = $("composeTagsInput")?.value || "";
      // If everything is empty, drop the entry rather than write a
      // useless {} marker.
      if (!title && !body && !pendingTag && _composeTags.length === 0) {
        localStorage.removeItem(_composeDraftKey());
        return;
      }
      localStorage.setItem(
        _composeDraftKey(),
        JSON.stringify({
          title,
          body,
          pendingTag,
          tags: _composeTags.slice(),
          at: Date.now(),
        }),
      );
    } catch {
      /* localStorage may be disabled — non-fatal */
    }
  }
  function loadComposeDraft() {
    if (!me) return null;
    try {
      const raw = localStorage.getItem(_composeDraftKey());
      if (!raw) return null;
      const obj = JSON.parse(raw);
      // Drop drafts older than a week — stale post-it notes are noise.
      if (obj && Date.now() - (obj.at || 0) < 7 * 24 * 3600 * 1000) {
        return obj;
      }
    } catch {}
    return null;
  }
  function clearComposeDraft() {
    if (!me) return;
    try {
      localStorage.removeItem(_composeDraftKey());
    } catch {}
  }
  // Persist on every input change. Cheap (synchronous localStorage
  // write) — text fields don't fire often enough to matter.
  for (const id of ["composeTitleInput", "composeInput", "composeTagsInput"]) {
    $(id)?.addEventListener("input", saveComposeDraft);
  }
  function closeComposeDialog() {
    const back = $("composeBack");
    const root = $("compose");
    if (back) back.hidden = true;
    if (root) root.hidden = true;
  }
  function setComposeHint(text, kind) {
    const h = $("composeHint");
    if (!h) return;
    h.textContent = text || "";
    h.classList.toggle("error", kind === "error");
    h.classList.toggle("ok", kind === "ok");
  }
  function extractUrl(text) {
    const m = (text || "").match(/https?:\/\/\S+/i);
    return m ? m[0].replace(/[),.;!?]+$/, "") : "";
  }
  function summaryFromText(text, url) {
    if (!text) return "";
    return text.replace(url || "", "").trim();
  }
  $("composeClose")?.addEventListener("click", closeComposeDialog);
  $("composeBack")?.addEventListener("click", closeComposeDialog);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("compose") && !$("compose").hidden) {
      closeComposeDialog();
    }
    if (
      (e.metaKey || e.ctrlKey) &&
      e.key === "Enter" &&
      $("compose") &&
      !$("compose").hidden
    ) {
      e.preventDefault();
      $("composeSubmit")?.click();
    }
  });
  function syncComposeSubmitEnabled() {
    const text = ($("composeInput")?.value || "").trim();
    const title = ($("composeTitleInput")?.value || "").trim();
    // Title is required; body is required too (we always need
    // SOMETHING to share — link or note).
    const btn = $("composeSubmit");
    if (btn) btn.disabled = !(text.length && title.length);
  }
  /* URL → metadata pre-fill. When the body contains a URL the user
   * hasn't auto-filled around yet, fetch its og:* metadata via the
   * existing `fetchPageMeta` proxy (or the public GitHub API for
   * github.com URLs — same shape as the github source module) and
   * gently suggest a title + summary for empty fields. Never
   * overwrites what the user typed.
   *
   * Wrapped end-to-end in try/catch so a network blip / parse error
   * stays silent. Hint messages are kept generic — we don't echo the
   * URL back into the dialog since that's been a source of confusion.
   */
  let _composeUrlSeenTimer = null;
  let _composeUrlSeenLast = "";

  function _parseGithubRepo(url) {
    try {
      const u = new URL(url);
      if (u.hostname !== "github.com" && u.hostname !== "www.github.com")
        return null;
      const parts = u.pathname.split("/").filter(Boolean);
      if (parts.length < 2) return null;
      return { owner: parts[0], repo: parts[1] };
    } catch {
      return null;
    }
  }
  const _githubMetaCache = new Map();
  async function _fetchGithubRepoMeta(owner, repo) {
    const key = `${owner}/${repo}`;
    if (_githubMetaCache.has(key)) return _githubMetaCache.get(key);
    try {
      const r = await fetch(`https://api.github.com/repos/${key}`, {
        headers: { Accept: "application/vnd.github+json" },
      });
      if (!r.ok) {
        _githubMetaCache.set(key, null);
        return null;
      }
      const j = await r.json();
      const meta = {
        title: j.full_name || key,
        description: j.description || "",
      };
      _githubMetaCache.set(key, meta);
      return meta;
    } catch {
      _githubMetaCache.set(key, null);
      return null;
    }
  }
  async function _composeMaybeSuggestFromUrl(opts = {}) {
    const force = !!opts.force;
    try {
      const body = $("composeInput")?.value || "";
      const url = extractUrl(body);
      if (!url) return;
      if (!force && url === _composeUrlSeenLast) return;
      _composeUrlSeenLast = url;
      setComposeHint("Fetching link metadata…", "");
      let meta = null;
      const gh = _parseGithubRepo(url);
      if (gh) meta = await _fetchGithubRepoMeta(gh.owner, gh.repo);
      if (!meta) meta = await fetchPageMeta(url);
      if (!meta) {
        setComposeHint("", "");
        return;
      }
      const titleEl = $("composeTitleInput");
      const bodyEl = $("composeInput");
      // `force` overwrites whatever the user typed; the auto path
      // (debounced on input) keeps user content intact.
      if (titleEl && meta.title && (force || !titleEl.value.trim())) {
        titleEl.value = String(meta.title).slice(0, 200);
        titleEl.dispatchEvent(new Event("input", { bubbles: true }));
      }
      if (bodyEl && meta.description) {
        const stripped = body.replace(url, "").trim();
        if (force || !stripped) {
          bodyEl.value = `${meta.description}\n\n${url}`;
          bodyEl.dispatchEvent(new Event("input", { bubbles: true }));
        }
      }
      setComposeHint("", "");
    } catch (e) {
      console.debug?.("[compose-suggest]", e);
      setComposeHint("", "");
    }
  }
  /* Toggle the Autofill button visibility from the body's content.
   * Visible whenever a URL is present — even if the auto path
   * already pre-filled, the user can re-trigger to overwrite. */
  function syncComposeAutofillButton() {
    const btn = $("composeAutofill");
    if (!btn) return;
    const body = $("composeInput")?.value || "";
    btn.hidden = !extractUrl(body);
  }
  $("composeAutofill")?.addEventListener("click", (e) => {
    e.preventDefault();
    _composeMaybeSuggestFromUrl({ force: true });
  });
  $("composeInput")?.addEventListener("input", (e) => {
    const text = e.target.value || "";
    const trimmed = text.trim();
    const url = extractUrl(text);
    syncComposeSubmitEnabled();
    syncComposeAutofillButton();
    if (!trimmed) {
      setComposeHint("", "");
    } else if (!($("composeTitleInput")?.value || "").trim()) {
      setComposeHint("Add a title to post.", "");
    } else if (url) {
      setComposeHint("Saving as a link.", "ok");
    } else {
      setComposeHint("Saving as a text note.", "ok");
    }
    // Debounced URL probe — runs after the user pauses typing so we
    // don't spam the proxy on every keystroke.
    if (url) {
      clearTimeout(_composeUrlSeenTimer);
      _composeUrlSeenTimer = setTimeout(_composeMaybeSuggestFromUrl, 350);
    }
  });
  $("composeTitleInput")?.addEventListener("input", () => {
    syncComposeSubmitEnabled();
    const text = ($("composeInput")?.value || "").trim();
    const title = ($("composeTitleInput")?.value || "").trim();
    if (text && !title) {
      setComposeHint("Add a title to post.", "");
    } else if (text && title) {
      const url = extractUrl(text);
      setComposeHint(
        url ? `Saving link: ${url}` : "Saving as a text note.",
        "ok",
      );
    }
  });
  $("composeSubmit")?.addEventListener("click", async () => {
    const text = $("composeInput").value || "";
    const trimmed = text.trim();
    if (!trimmed) return;
    const userTitleRaw = ($("composeTitleInput")?.value || "").trim();
    if (!userTitleRaw) {
      setComposeHint("Title is required.", "error");
      $("composeTitleInput")?.focus();
      return;
    }
    // Flush any in-progress tag that the user typed without confirming.
    const pendingTag = ($("composeTagsInput")?.value || "").trim();
    if (pendingTag) {
      commitComposeTag(pendingTag);
      $("composeTagsInput").value = "";
    }
    const url = extractUrl(text);
    // Body — the URL is stripped from the summary when the user is
    // sharing a link so we don't echo it. For a pure note the body
    // IS the summary.
    const bodyText = url ? summaryFromText(text, url) : trimmed;
    // Summary fallback: the user's own profile description, so a
    // bare-link post still has context attached. Body wins if the
    // user typed anything alongside the URL.
    const summary =
      bodyText && bodyText.length ? bodyText : (me?.description || "").trim();
    const btn = $("composeSubmit");
    btn.disabled = true;
    setComposeHint("Saving…", "");

    // For text-only posts (no URL pasted) we synthesise a stable
    // note URL keyed on the user + the post text hash. Keeps the
    // PG primary key `(user_id, url)` unique without needing a new
    // column.
    let docUrl = url;
    if (!url) {
      const h = await sha256Hex(`${me.slug}::${trimmed}::${Date.now()}`);
      docUrl = `knowledge://note/${me.slug}/${h.slice(0, 16)}`;
    }
    // Title is required (gated at submit time); truncate to fit the
    // backend column.
    const title = userTitleRaw.slice(0, 200);

    // Stamp today's date so the post shows up at the top of the feed
    // (the feed/timeline ORDER BY is date DESC). YYYY-MM-DD in local
    // time matches how the rest of the app formats dates.
    const today = (() => {
      const d = new Date();
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      return `${y}-${m}-${day}`;
    })();

    try {
      const r = await fetch(`${API_BASE}/auth/me/documents/bulk`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documents: [
            {
              url: docUrl,
              title,
              summary,
              date: today,
              source: url ? "bookmark" : "note",
              tags: _composeTags.slice(),
              // All posts are public by default — the audience toggle
              // was removed from the composer.
              public: true,
            },
          ],
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      // Posts no longer auto-star themselves; the user can favourite
      // them explicitly via the heart on the rendered card.
      setComposeHint("Posted ✓", "ok");
      // Wipe the persisted draft — the user's content is now in their
      // library, no need to keep it staged.
      clearComposeDraft();
      window.dispatchEvent(new CustomEvent("knowledge:bookmark-added"));
      setTimeout(closeComposeDialog, 700);
    } catch (err) {
      setComposeHint(`Failed: ${err.message || err}`, "error");
      btn.disabled = false;
    }
  });

  // SHA-256 hex helper — used to mint stable URLs for text-only
  // posts. WebCrypto subtle.digest is universally available.
  async function sha256Hex(s) {
    const data = new TextEncoder().encode(s);
    const buf = await crypto.subtle.digest("SHA-256", data);
    return [...new Uint8Array(buf)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }
  window.addEventListener("knowledge:bookmark-added", async () => {
    try {
      state.favorites = await K.getFavoriteUrls();
      state.favoriteOwners = new Set(await K.getFavoriteOwners());
      // Bust both feed and personal-page caches so the next refresh
      // re-fetches and surfaces the new / edited document
      // immediately, without a manual reload.
      _timelineCache.clear();
      window.KnowledgeSessionCache?.invalidatePrefix?.(_TIMELINE_SS_PREFIX);
      // Per-slug unindexed-docs cache (personal page browse). A fresh
      // post lives in PG before the ColBERT indexer picks it up, so
      // this cache is what makes the new doc visible at all.
      if (typeof K.invalidateUnindexed === "function") {
        for (const slug of state.libs) K.invalidateUnindexed(slug);
        if (me?.slug) K.invalidateUnindexed(me.slug);
      }
      // Same story for the personal-page browse cache (full PG
      // library, not just the unindexed tail) — without this the
      // 30s memo would shadow the new row until it expires.
      if (typeof K.invalidatePersonalDocs === "function") {
        for (const slug of state.libs) K.invalidatePersonalDocs(slug);
        if (me?.slug) K.invalidatePersonalDocs(me.slug);
      }
      // Same rule as the inline fav-click handler: pick the rebuilder
      // that matches the current view so the feed doesn't lose its rail.
      if (state.libs.size === 0) {
        await rebuildAllSourcesForFeed();
      } else {
        rebuildAllSources();
      }
      renderSrc();
      refresh();
    } catch {
      /* non-fatal */
    }
  });
  // Theme toggle button was removed from the search page — theme is
  // now changed from the Settings page (/profile). The listener
  // below is guarded so older HTML caches that still contain the
  // button keep working until they reload.
  $("themeToggle")?.addEventListener("click", () => {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const next = isDark ? "light" : "dark";
    if (next === "dark")
      document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
    try {
      localStorage.setItem("theme", next);
    } catch {}
    syncThemeIcon();
  });

  function renderAuthBtn() {
    const btn = $("authBtn");
    const postTrigger = $("postTriggerBtn");
    const settings = $("authSettings");
    const discover = $("discoverBtn");
    const feedLink = $("feedLink");
    if (postTrigger) postTrigger.hidden = !me;
    if (settings) settings.hidden = !me;
    // Discover is a signed-in-only utility — anonymous visitors see
    // the same heading text but it's inert. The `.is-discover` class
    // adds the cursor + hover affordance.
    if (discover) discover.classList.toggle("is-discover", !!me);
    // Following-only filter only makes sense once the user has a
    // follow graph to filter by — hidden for anonymous visitors.
    syncFollowingOnlyButton();
    if (feedLink) {
      // Show whenever we're NOT already on the feed. Logged-out users
      // get the global VIP timeline at `/`, logged-in users get their
      // followee feed — same button, same destination.
      feedLink.hidden = state.libs.size === 0;
      feedLink.title = me
        ? "Your feed — everyone you follow plus your own library"
        : "Feed — recent activity across featured libraries";
      const label = feedLink.querySelector("span");
      if (label) label.textContent = me ? "Back to your feed" : "Back to feed";
      feedLink.classList.remove("is-current");
    }
    const personalLink = $("personalLink");
    if (personalLink) {
      // Mirror of feedLink — only when signed in AND on the feed.
      personalLink.hidden = !me || state.libs.size !== 0;
      if (me) {
        personalLink.href = `/search?libs=${encodeURIComponent(me.slug)}`;
      }
    }
    if (me) {
      btn.classList.remove("anon");
      btn.classList.add("user-pill");
      btn.href = `/search?libs=${encodeURIComponent(me.slug)}`;
      btn.title = `Your library — @${me.slug}`;
      btn.setAttribute("aria-label", `Your library, @${me.slug}`);
      const initials = (me.name || me.slug || "?")
        .split(/\s+/)
        .slice(0, 2)
        .map((w) => (w[0] || "").toUpperCase())
        .join("");
      const avatarHtml = me.avatar
        ? `<img class="av" src="${escapeAttr(me.avatar)}" alt="" onerror="this.style.display='none'"/>`
        : `<span class="av av-fallback" aria-hidden="true">${escapeHtml(initials)}</span>`;
      btn.innerHTML = `
        ${avatarHtml}
        <span class="user-pill-name">${escapeHtml(me.name || me.slug)}</span>
      `;
      btn.onclick = null;
    } else {
      btn.classList.remove("user-pill");
      btn.classList.remove("avatar-pill");
      btn.classList.add("anon");
      btn.href = "#";
      btn.onclick = (e) => {
        e.preventDefault();
        window.KnowledgeAuth?.open("login");
      };
    }
  }

  /* Mobile chrome — bottom nav + Sources sheet trigger.
   *
   * Both surfaces are hidden by CSS on desktop; this routine only
   * wires the event handlers. Idempotent: a `data-wired` marker on
   * the trigger button stops a second call from binding twice.
   *
   * The exposed `_syncMobileChrome()` (also attached to window)
   * keeps the active-tab class + source-count badge in sync with
   * the URL state. It's invoked from `refresh()` so it tracks
   * every state change the rail / search bar pushes. */
  function wireMobileChrome() {
    const sheetSources = document.getElementById("mobileSourcesBtn");
    const nav = document.getElementById("mobileBottomNav");
    const backdrop = document.getElementById("mobileSheetBackdrop");
    if (!sheetSources || !nav || !backdrop) return;
    // ── Cupertino Pane wiring ─────────────────────────────────
    // Both bottom sheets (Sources & People) are rendered by the
    // Cupertino Pane library — a small vanilla-JS package that gives
    // an iOS / Twitter-style draggable sheet with snap breakpoints,
    // swipe-down-to-close, automatic visualViewport keyboard
    // tracking, and a frosted backdrop. We feed it the existing
    // `.rail` / `.people-rail` elements; the library positions and
    // animates them. Our prior CSS positioning rules for these
    // sheets have been removed so the library owns the layout.
    const panes = { sources: null, people: null, categories: null };

    function isOpen(kind) {
      return !!panes[kind]?.isPanePresented?.();
    }
    function anyOpen() {
      return isOpen("sources") || isOpen("people") || isOpen("categories");
    }

    function paneSelectorFor(kind) {
      if (kind === "sources") return ".rail";
      if (kind === "people") return ".people-rail";
      return ".category-rail";
    }

    // Build (or return the cached) CupertinoPane bound to a given
    // rail. The library tracks state on the underlying selector —
    // constructing a second instance with the same selector throws
    // "already in use". So we build once per kind and reuse the
    // instance across opens: each present()/destroy() cycle is
    // handled by the same pane object.
    function buildPane(kind) {
      if (typeof window.CupertinoPane === "undefined") return null;
      if (panes[kind]) return panes[kind];
      const selector = paneSelectorFor(kind);
      const el = document.querySelector(selector);
      if (!el) return null;
      // Move the rail into the body so Cupertino Pane can reposition
      // it as a top-level overlay — its desktop home (inside the
      // .layout grid) interferes with the fixed-position wrapper the
      // library creates.
      if (el.parentElement !== document.body) {
        el.dataset.originalParent = el.parentElement.id || "";
        document.body.appendChild(el);
      }
      // The rail starts with `display: none` on phone (mobile.css
      // hides the desktop rails). The library needs the element
      // measurable to compute breakpoints, so force it visible
      // before constructing the pane.
      el.style.display = "flex";
      const vh = window.innerHeight;
      // Three breakpoints:
      //   • middle — initial. ~75% of viewport, the default the user
      //     sees on open. Tall enough that 8–10 candidates show but
      //     the search bar of the feed underneath is still hinted
      //     through the frosted backdrop.
      //   • top    — almost full-screen (vh - 60). Auto-snapped to
      //     when the in-pane search input focuses, so the candidate
      //     list isn't occluded by the keyboard.
      //   • bottom — dismiss threshold (anything below closes with
      //     `bottomClose:true`).
      const topHeight = Math.max(520, vh - 60);
      const middleHeight = Math.max(440, Math.round(vh * 0.9));
      const pane = new window.CupertinoPane(selector, {
        parentElement: "body",
        backdrop: true,
        backdropOpacity: 0.45,
        bottomClose: true,
        fastSwipeClose: true,
        fastSwipeSensivity: 3,
        simulateTouch: true,
        cssClass: `kn-sheet kn-sheet-${kind}`,
        // A touch longer than the library default (300ms) — the
        // close slide stays visible through the middle of the
        // viewport instead of flashing past.
        animationDuration: 420,
        // Smooth cubic-bezier — slow-in, slow-out. Drops the
        // library's default overshoot which read as "snap" on
        // close.
        animationType: "cubic-bezier(0.22, 0.61, 0.36, 1)",
        // Show a subtle 36×5 handle bar at the top — the only
        // affordance we keep besides the backdrop tap.
        showDraggable: true,
        // Hide the library's default destroy button; we already have
        // the bottom-nav tab to flip back to the feed.
        buttonDestroy: false,
        breaks: {
          top: { enabled: true, height: topHeight, bounce: true },
          middle: { enabled: true, height: middleHeight, bounce: true },
          bottom: { enabled: true, height: 120 },
        },
        initialBreak: "middle",
        events: {
          onDidPresent: () => {
            window._syncMobileChrome?.();
          },
          // The library doesn't dismiss on backdrop tap by default —
          // wire it explicitly so a tap outside the pane closes it,
          // matching native iOS / Twitter behavior.
          onBackdropTap: () => {
            closeSheets();
          },
          onWillDismiss: () => {
            // Sync chrome immediately so the bottom-nav tab releases
            // its active state before the slide-out finishes.
            window._syncMobileChrome?.();
          },
          onDidDismiss: () => {
            // Don't null out panes[kind] — the same instance gets
            // re-presented on the next open. Just clear the inline
            // styles the library wrote on the rail so the next
            // present() starts from a clean slate.
            const el = document.querySelector(selector);
            if (el) {
              el.style.height = "";
              el.style.transition = "";
              el.style.overflow = "";
              el.style.overscrollBehavior = "";
            }
            // Drop the body class so the bottom-nav capsule reverts
            // from its flush docked state back to the floating
            // capsule. Critical for the swipe-down-to-close path:
            // closeSheets() only runs for backdrop tap / explicit
            // dismiss, but `bottomClose: true` lets the user drag
            // the pane off-screen and that route fires onDidDismiss
            // alone. Without this removal, the nav stays docked
            // flat against the viewport bottom after a swipe close.
            document.body.classList.remove("cupertino-pane-presented");
            window._syncMobileChrome?.();
          },
        },
      });
      panes[kind] = pane;
      return pane;
    }

    function openSheet(kind) {
      // Close any other open sheet first — only one bottom sheet at
      // a time. The People / Sources / Topics sheets are mutually
      // exclusive in the bottom-nav UX.
      for (const k of ["sources", "people", "categories"]) {
        if (k === kind) continue;
        if (panes[k]?.isPanePresented?.()) {
          panes[k].destroy({ animate: false });
        }
      }
      const pane = buildPane(kind);
      if (!pane) return;
      // Manually mark the body — the library does NOT add this
      // class itself in 1.5.4. The mobile.css ruleset uses it to
      // push sticky elements (the feed's top search bar, filter
      // strip, profile chrome) UNDER the frosted backdrop so the
      // whole page reads as one blurred surface behind the sheet.
      document.body.classList.add("cupertino-pane-presented");
      // Always start at the middle break — even if the pane was
      // previously dragged to top before close, we want a calm
      // half-screen default on each open.
      try {
        pane.present({ animate: true });
      } catch {
        /* already presented — defensive no-op. */
      }
      // Wire focus → moveToBreak('top') for the in-pane search
      // input. Runs once per rail, after the rail has been moved
      // into the wrapper. Subsequent opens keep the same listener
      // (the rail is reused), so the `expandWired` flag stops
      // duplicate attachments.
      setTimeout(() => {
        const inputSel =
          kind === "sources" ? "#srcFilter" : "#peopleRailFilter";
        const input = document.querySelector(`.kn-sheet ${inputSel}`);
        if (!input || input.dataset.expandWired === "1") return;
        input.dataset.expandWired = "1";
        input.addEventListener("focus", () => {
          const live = panes[kind];
          if (!live?.isPanePresented?.()) return;
          try {
            live.moveToBreak("top");
          } catch {
            /* break missing on some library versions — no-op. */
          }
        });
      }, 60);
    }
    function closeSheets() {
      ["sources", "people", "categories"].forEach((k) => {
        if (panes[k]?.isPanePresented?.()) {
          panes[k].destroy({ animate: true });
          // Keep the instance — buildPane returns the cached one
          // on the next open. Nulling it would force a new
          // construction and the library errors with "already in
          // use" on the same selector.
        }
      });
      document.body.classList.remove("cupertino-pane-presented");
    }
    function toggleSheet(kind) {
      if (isOpen(kind)) closeSheets();
      else openSheet(kind);
    }
    // Expose a one-shot close for the categories sheet so the
    // shared list-click handler (`onCatListClick` above) can dismiss
    // the sheet after the user picks a slug — without that handler
    // needing to know about Cupertino Pane internals. Routes through
    // closeSheets() so the body class + chrome sync happen on the
    // same code path the backdrop-tap uses.
    window._closeCategorySheet = () => {
      if (panes.categories?.isPanePresented?.()) {
        closeSheets();
      }
    };
    // Backdrop element from the previous custom sheet is now unused —
    // hide it so it never bleeds through if some stale state lingers.
    if (backdrop) backdrop.hidden = true;
    if (sheetSources.dataset.wired !== "1") {
      sheetSources.dataset.wired = "1";
      sheetSources.addEventListener("click", () => toggleSheet("sources"));
      // People sheet — tapping the tab while the sheet is already
      // open scrolls the candidate list back to the top (the same
      // affordance native iOS apps offer for their tab bars). Close
      // is still available via backdrop tap, swipe-down, or by
      // selecting a candidate.
      document.getElementById("mbnPeople")?.addEventListener("click", () => {
        if (isOpen("people")) {
          document
            .querySelector(".kn-sheet .people-rail-list")
            ?.scrollTo({ top: 0, behavior: "smooth" });
          return;
        }
        toggleSheet("people");
      });
      document
        .getElementById("mbnCategories")
        ?.addEventListener("click", () => {
          // Same scroll-to-top affordance as People when the sheet
          // is already open.
          if (isOpen("categories")) {
            document
              .querySelector(".kn-sheet .category-rail-list")
              ?.scrollTo({ top: 0, behavior: "smooth" });
            return;
          }
          // First open: warm the catalogue + render. Subsequent opens
          // re-use the same render — selection state is already in
          // sync with the URL.
          if (!_catalogueCache) {
            fetchCatalogue().then(() => renderCatPickerList(""));
          } else {
            renderCatPickerList(categoryRailFilter?.value || "");
          }
          toggleSheet("categories");
        });
      // Feed tab — when already on the feed and no sheet is open,
      // suppress the link's navigation and scroll the page to the
      // top instead. Same affordance as the floating back-to-top
      // button, routed through the tab the user already expects to
      // mean "where I am right now."
      document.getElementById("mbnFeed")?.addEventListener("click", (e) => {
        const onFeed = state.libs?.size === 0;
        if (onFeed && !anyOpen()) {
          e.preventDefault();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
      backdrop.addEventListener("click", closeSheets);
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && anyOpen()) {
          closeSheets();
        }
      });
      // The Personal tab routes to the signed-in user's library
      // (resolved at click time so an anonymous → signed-in
      // transition is reflected without a rebind). Anonymous taps
      // open the auth modal instead.
      document.getElementById("mbnPersonal")?.addEventListener("click", (e) => {
        if (!me?.slug) {
          e.preventDefault();
          window.KnowledgeAuth?.open("login");
          return;
        }
        // Already on the personal page → scroll to top instead of
        // re-navigating to the same URL (the browser would
        // otherwise either reload or no-op depending on the
        // navigation type).
        const onPersonal = state.libs?.size === 1 && state.libs.has(me.slug);
        if (onPersonal && !anyOpen()) {
          e.preventDefault();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
        // else fall through to native navigation. URL was synced in
        // _syncMobileChrome below.
      });
      // The in-bar Post button mirrors the legacy floating FAB:
      // anonymous → auth modal; signed-in → compose dialog.
      document.getElementById("mbnPostBtn")?.addEventListener("click", () => {
        if (!me) {
          window.KnowledgeAuth?.open("login");
          return;
        }
        document.getElementById("postTriggerBtn")?.click();
      });
    }
    // When the page is reached via a hash (#people / #sources /
    // #discover / #post) we pop the matching surface open and
    // strip the hash. This is how the Settings page's bottom-nav
    // action buttons reach back into the feed page's overlays.
    if (
      location.hash === "#people" ||
      location.hash === "#sources" ||
      location.hash === "#discover" ||
      location.hash === "#post"
    ) {
      const h = location.hash.slice(1);
      setTimeout(() => {
        if (h === "discover") {
          openDiscoverOverlay();
        } else if (h === "post") {
          document.getElementById("postTriggerBtn")?.click();
        } else {
          openSheet(h);
        }
        history.replaceState(null, "", location.pathname + location.search);
      }, 0);
    }
    /* Keep the URL + active state in sync. The four bottom-nav tabs
     * are mutually exclusive: when a sheet is open the corresponding
     * tab is the only one lit, otherwise we light the tab that
     * matches the current URL (Feed = no libs filter, Personal =
     * exactly the signed-in user). */
    function _sync() {
      const personal = document.getElementById("mbnPersonal");
      if (personal) {
        personal.href = me?.slug
          ? `/search?libs=${encodeURIComponent(me.slug)}`
          : "#";
      }
      sheetSources.classList.toggle("is-open", isOpen("sources"));

      const feed = document.getElementById("mbnFeed");
      const people = document.getElementById("mbnPeople");
      const categories = document.getElementById("mbnCategories");
      const settings = document.getElementById("mbnSettings");
      const onFeed = state.libs?.size === 0;
      const onPersonal =
        me?.slug && state.libs?.size === 1 && state.libs.has(me.slug);
      const peopleOpen = isOpen("people");
      const categoriesOpen = isOpen("categories");
      const anySheetOpen = peopleOpen || categoriesOpen;
      // Mutually exclusive: sheet beats URL state. The Topics tab
      // also lights up when state.category is set, even if the
      // sheet is closed — same convention sources uses to indicate
      // "filter is engaged".
      feed?.classList.toggle(
        "is-current",
        !anySheetOpen && !!onFeed && !onPersonal,
      );
      people?.classList.toggle("is-current", peopleOpen);
      categories?.classList.toggle(
        "is-current",
        categoriesOpen || (state.categories && state.categories.size > 0),
      );
      personal?.classList.toggle("is-current", !anySheetOpen && !!onPersonal);
      settings?.classList.remove("is-current");

      const realSources = [...(state.sources || [])].filter(
        (s) => s !== FAV_SOURCE_KEY,
      );
      const countEl = document.getElementById("mobileSourcesCount");
      if (countEl) {
        if (realSources.length) {
          countEl.textContent = String(realSources.length);
          countEl.hidden = false;
          sheetSources.classList.add("has-filter");
        } else {
          countEl.hidden = true;
          sheetSources.classList.remove("has-filter");
        }
      }
      // Mobile Topics-tab badge — mirrors the desktop catPickerCount
      // so a user navigating between phone and laptop sees a
      // consistent indicator that "some categories are picked".
      const catCount = document.getElementById("mobileCategoriesCount");
      const nCat = state.categories ? state.categories.size : 0;
      if (catCount) {
        if (nCat > 0) {
          catCount.textContent = String(nCat);
          catCount.hidden = false;
        } else {
          catCount.hidden = true;
        }
      }
    }
    window._syncMobileChrome = _sync;
    _sync();

    // ── Back-to-top button ──────────────────────────────────────
    // A small frosted-glass floater that fades in once the user has
    // scrolled past SHOW_AT pixels and smooth-scrolls them to the
    // top on tap. The element lives in the HTML hidden by default;
    // we toggle the `hidden` attribute on a passive scroll listener.
    // CSS handles the fade — we don't use display:none so the
    // transition stays visible.
    const backTop = document.getElementById("backToTop");
    if (backTop) {
      const SHOW_AT = 600;
      let lastVisible = null;
      const apply = () => {
        const visible = window.scrollY > SHOW_AT;
        if (visible === lastVisible) return;
        lastVisible = visible;
        if (visible) backTop.removeAttribute("hidden");
        else backTop.setAttribute("hidden", "");
      };
      window.addEventListener("scroll", apply, { passive: true });
      backTop.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      apply();
    }
  }

  /* Post-paint auth setup. `me` was resolved earlier in the boot
   * Promise.all so the first `refresh()` already painted the
   * correct fav-button state; nothing here needs to re-render the
   * result list. We just wire up the auth-dependent UI surfaces
   * (auth button, people rail, profile header, picker hydration,
   * source rail) and back-fill any missing personality metadata
   * the picker would otherwise show as a bare slug. */
  (async () => {
    renderAuthBtn();
    wireMobileChrome();
    wirePullRefresh();
    // Lazy-hydrate the people-to-follow rail once auth state is known.
    setupPeopleRail();
    showProfileHeader();
    if (!me && state.sources.has(FAV_SOURCE_KEY)) {
      state.sources.delete(FAV_SOURCE_KEY);
      writeUrl();
    }
    if (
      me &&
      me.slug &&
      me.slug !== state.hostSlug &&
      !state.allPersonalities.find((p) => p.slug === me.slug)
    ) {
      try {
        const myMeta = await K.getPersonality(me.slug);
        if (myMeta) {
          state.allPersonalities.push(myMeta);
          state.perSlugMeta[me.slug] = myMeta;
        }
      } catch {}
    }
    if (me) {
      try {
        state.personalityBookmarks = await K.getPersonalityBookmarks();
        const missing = [...state.personalityBookmarks].filter(
          (slug) => !state.allPersonalities.find((p) => p.slug === slug),
        );
        if (missing.length) {
          const fetched = await Promise.all(
            missing.map((s) => K.getPersonality(s).catch(() => null)),
          );
          for (const meta of fetched) {
            if (meta) {
              state.allPersonalities.push(meta);
              state.perSlugMeta[meta.slug] = meta;
            }
          }
        }
      } catch {}
    }
    rebuildAllSources();
    renderSrc();
    // Legacy `?profile=1` deep links — redirect to the standalone /profile.
    // Use `replace()` (not assign) so the deep-link entry is *replaced* in
    // history rather than stacked: pressing Back from /profile then jumps
    // to whatever the user was on before the deep link, instead of
    // bouncing them back into the same redirect.
    const params = new URLSearchParams(location.search);
    if (params.get("profile") === "1" && me && me.slug === state.hostSlug) {
      window.location.replace("/profile");
      return;
    }
  })();

  /* Hydrate state from the URL — must run after `state` exists and
   * before the first refresh().
   *
   * Source-list hydration trade-off: each non-host library named in
   * `?libs=` needs a GET on `/api/users/{slug}/sources` to populate
   * its source-filter chips. With small selections we await every
   * fetch so the rail is accurate on first paint; above the
   * ALL_INDEX_THRESHOLD we skip per-slug fanout entirely because
   * search routes through the `__all__` index anyway and the chips
   * can populate later from the result set via
   * renderResultSources(). This keeps a 100-lib selection from
   * blocking boot on 100 parallel GETs.
   */
  const extraLibsFromUrl = readUrl();
  // Re-sync the topbar controls now that the URL hydration has
  // populated state.dateSince / state.query / state.followingOnly.
  if ($("qSince")) $("qSince").value = state.dateSince || "";
  syncSinceFilterActive();
  syncFollowingOnlyButton();
  // The picker's initial label sync ran during the wire-up phase
  // before readUrl() populated state.category — re-fire now so the
  // button shows the selected category on a deep-link reload. If
  // the catalogue isn't cached yet the label falls back to the raw
  // slug; the upgrade to the human name happens on first popover
  // open (which warms the catalogue cache) and via the trailing
  // fetchCatalogue() below.
  syncCatPickerLabel();
  if (state.categories && state.categories.size) {
    fetchCatalogue().then(syncCatPickerLabel);
  }
  // Canonical points at the bare /search in the static HTML —
  // bring it in line with `?libs=<slug>` if we landed on a
  // personality page so Google indexes that view, not the root.
  syncCanonical();
  if (extraLibsFromUrl.length) {
    if (useAllOnly([...state.libs])) {
      // Large selection — skip per-slug source hydration entirely.
      // Search/browse routes through `__all__`; the source chips
      // populate from the result set itself via renderResultSources().
    } else {
      // Small selection — wait for source-list hydration so the rail
      // is accurate on the first paint.
      await Promise.all(extraLibsFromUrl.map((s) => ensureLibLoaded(s)));
      rebuildAllSources();
    }
  }
  writeQ();
  renderActiveTags();
  if (state.query) {
    $("sortRel")?.classList?.toggle("on", !state.sortByDate);
    $("sortDate")?.classList?.toggle("on", state.sortByDate);
  }

  renderLibs();
  renderSrc();
  refresh();
})();
