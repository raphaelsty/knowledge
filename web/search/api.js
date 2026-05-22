/* Shared helpers for the search page. One API client, one common doc
 * shape, in-flight coalescing + localStorage TTL caches so repeated
 * calls cost the server nothing.
 *
 * Loaded as a classic script before page.js — the page consumes the
 * exposed global `window.KnowledgeAPI`.
 */
(function () {
  "use strict";

  const API = window.KNOWLEDGE_API_BASE;
  const DEFAULT_SLUG = "max-halford";

  /* Credentials are only sent for endpoints that genuinely need the
   * session cookie (anything under `/auth/*` and the `/api/users/me`
   * profile endpoints). Public data endpoints respond with the
   * wildcard `Access-Control-Allow-Origin: *`, which the browser
   * refuses to expose to a credentialed request — so we MUST omit
   * credentials there. */
  function needsAuth(url) {
    return /\/auth\//.test(url) || /\/api\/users\/me(?:\/|$|\?)/.test(url);
  }
  async function fetchJson(url, opts = {}) {
    const credentials =
      opts.credentials || (needsAuth(url) ? "include" : "omit");
    const resp = await fetch(url, { ...opts, credentials });
    if (!resp.ok)
      throw new Error(`${resp.status} ${resp.statusText} on ${url}`);
    return resp.json();
  }

  /* ── Tiny localStorage TTL cache ─────────────────────────────
   * Production-ready stinginess on backend round-trips. Every
   * production user opens the profile modal multiple times per
   * session and re-types the same handles — without persisting,
   * we'd re-probe GitHub/Twitter/HF/etc. on every single mount.
   * 1h is short enough that revoked-key surfaces fast; 30s on
   * failures gives transient errors a chance to recover. */
  function _ttlGet(key) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj || typeof obj.exp !== "number") return null;
      if (Date.now() > obj.exp) {
        localStorage.removeItem(key);
        return null;
      }
      return obj.v;
    } catch {
      return null;
    }
  }
  function _ttlSet(key, value, ttlMs) {
    try {
      localStorage.setItem(
        key,
        JSON.stringify({ v: value, exp: Date.now() + ttlMs }),
      );
    } catch {}
  }
  function _ttlDel(key) {
    try {
      localStorage.removeItem(key);
    } catch {}
  }

  /* In-flight request coalescing: if N callers ask for the same
   * URL while a fetch is pending, they all share the same Promise
   * — one network hit, N consumers. */
  const _inflight = new Map();
  function _coalesce(key, fn) {
    if (_inflight.has(key)) return _inflight.get(key);
    const p = fn().finally(() => _inflight.delete(key));
    _inflight.set(key, p);
    return p;
  }

  /* listPersonalities — cached for 5 min. The personality grid is
   * effectively static on the timescale of a session. */
  const PERSONALITIES_TTL = 5 * 60_000;
  async function listPersonalities() {
    const cached = _ttlGet("k:personalities");
    if (cached) return cached;
    return _coalesce("listPersonalities", async () => {
      const list = await fetchJson(`${API}/api/users`);
      if (Array.isArray(list))
        _ttlSet("k:personalities", list, PERSONALITIES_TTL);
      return list;
    });
  }

  async function getPersonality(slug) {
    return _coalesce(`getPersonality:${slug}`, () =>
      fetchJson(`${API}/api/users/${encodeURIComponent(slug)}`),
    );
  }

  /* getSources — cached for 2 min per slug. Source counts shift only
   * when the pipeline runs; mid-session we can serve from cache. */
  const SOURCES_TTL = 2 * 60_000;
  async function getSources(slug) {
    const cached = _ttlGet(`k:sources:${slug}`);
    if (cached) return cached;
    return _coalesce(`getSources:${slug}`, async () => {
      try {
        const list = await fetchJson(
          `${API}/api/users/${encodeURIComponent(slug)}/sources`,
        );
        if (Array.isArray(list))
          _ttlSet(`k:sources:${slug}`, list, SOURCES_TTL);
        return list;
      } catch {
        return [];
      }
    });
  }

  async function getDocuments(slug) {
    try {
      return await fetchJson(
        `${API}/api/users/${encodeURIComponent(slug)}/documents`,
      );
    } catch {
      return {};
    }
  }

  /* Aggregate sources across every VIP user — one round-trip
   * alternative to fanning out `getSources(slug)` for each
   * selected library. Used by the search page when the user
   * crosses the `__all__` threshold and we want a complete
   * filter list without paying for N round-trips.
   *
   * Cached for 5 min: source counts shift only when the pipeline
   * runs, and a stale rail by a few minutes is fine. Returns the
   * full alphabet of sources — typically 4000+ distinct keys.
   */
  const ALL_SOURCES_TTL = 5 * 60_000;
  async function getAllSources() {
    const cached = _ttlGet("k:all-sources");
    if (cached) return cached;
    return _coalesce("getAllSources", async () => {
      try {
        const list = await fetchJson(`${API}/api/sources`);
        if (Array.isArray(list))
          _ttlSet("k:all-sources", list, ALL_SOURCES_TTL);
        return list;
      } catch {
        return [];
      }
    });
  }

  /* Per-(kind,value) probe cache. OK results live for 1h; bad ones
   * for 30s so transient API failures (rate-limit, network blip)
   * recover quickly. The page wraps `/api/profile/probe` calls
   * with these so a user re-opening the modal sees green checks
   * instantly and the server isn't asked again. */
  const PROBE_TTL_OK = 60 * 60_000; // 1h
  const PROBE_TTL_BAD = 30_000; // 30s
  function probeCacheKey(kind, value) {
    return `k:probe:${kind}:${value}`;
  }
  function probeCacheGet(kind, value) {
    return _ttlGet(probeCacheKey(kind, value));
  }
  function probeCacheSet(kind, value, state) {
    const ttl = state && state.status === "ok" ? PROBE_TTL_OK : PROBE_TTL_BAD;
    _ttlSet(probeCacheKey(kind, value), state, ttl);
  }
  /** Drop the cached personalities + the slug-specific source list
   * — call after a save that might have changed source counts. */
  function invalidateCaches(slug) {
    _ttlDel("k:personalities");
    if (slug) _ttlDel(`k:sources:${slug}`);
  }

  /** Fetch only the documents the pipeline hasn't embedded yet —
   * `indexed = false` rows in Postgres. The search index doesn't
   * know about these (they aren't searchable by similarity), so the
   * UI overlays them into the results list with a "not indexed yet"
   * badge until the next `make run` picks them up.
   *
   * Tag shape: the PG endpoint returns proper arrays under `tags`
   * and `extra-tags` (matching the on-disk database.json shape),
   * whereas the ColBERT index serialises both as comma-strings —
   * so we map directly here instead of going through
   * `transformMeta` (which expects the index shape). */
  // Memoised across the session so search-page refresh churn (typing,
  // sorting, filter toggles) doesn't re-hit `/api/users/.../documents`
  // on every keystroke. 30s TTL — short enough that a fresh sync
  // becomes visible quickly, long enough to absorb rapid filter changes.
  // Cache key is `${slug}|${sortedSources}|${sortedTags}` so distinct
  // filter combinations don't trample each other.
  const _unindexedCache = new Map(); // cacheKey → { at, list }
  const UNINDEXED_TTL = 30_000;
  function _unindexedKey(slug, sources, tags, urls, excludeSources) {
    const s = (sources || []).slice().sort().join(",");
    const t = (tags || []).slice().sort().join(",");
    const u = (urls || []).slice().sort().join(",");
    const x = (excludeSources || []).slice().sort().join(",");
    return `${slug}|${s}|${t}|${u}|${x}`;
  }
  /** Drop every cached unindexed list for a slug — used after a sync
   * lands so the next refresh sees the fresh rows. */
  function invalidateUnindexed(slug) {
    const prefix = `${slug}|`;
    for (const k of [..._unindexedCache.keys()]) {
      if (k === slug || k.startsWith(prefix)) _unindexedCache.delete(k);
    }
  }
  /** Fetch every doc in a user's library from Postgres (indexed + not).
   *
   * Personal-page browse mode uses this instead of the ColBERT
   * `latest` endpoint: the SQLite metadata sidecar that backs the
   * indexed pool drops `created_at` on the way through, so two docs
   * sharing the same `date` (very common — pipeline stamps `date`
   * to today on a lot of sources) end up ordered by SQLite
   * insertion order rather than by when the user actually saved
   * them. The PG endpoint orders rows by (date DESC, created_at
   * DESC) and returns `created_at` alongside every row, so the
   * tiebreaker is honored both server-side and (via reorderForBrowse)
   * client-side.
   *
   * Returned docs do NOT carry the `_unindexed` badge marker — that
   * flag is reserved for rows the search index doesn't yet know
   * about, surfaced via `getUnindexedDocuments` below.
   */
  // In-memory cache for the page session, mirrored into sessionStorage
  // via KnowledgeSessionCache so a fresh page load (back-button from
  // settings, etc.) repaints the personal page instantly instead of
  // running another full /api/users/.../documents round-trip.
  // The in-memory layer has a tight 30s TTL (so a recent compose-
  // dialog write becomes visible on the next refresh without an
  // explicit invalidate); the persistent layer carries a 10 min TTL
  // so cross-navigation reads still hit cache.
  const _personalDocsCache = new Map();
  // Short TTL on the persistent layer — long enough that bouncing
  // between feed and personal page reads from cache, short enough
  // that an active-ingestion workflow (twitter feeder running,
  // user reloads the target's personal page repeatedly to watch
  // new tweets land) doesn't show a 10-minute-old snapshot.
  const _PERSONAL_SS_TTL_MS = 2 * 60 * 1000; // 2 min
  const _PERSONAL_SS_PREFIX = "personal:";
  function invalidatePersonalDocs(slug) {
    const prefix = `${slug}|`;
    for (const k of [..._personalDocsCache.keys()]) {
      if (k === slug || k.startsWith(prefix)) _personalDocsCache.delete(k);
    }
    if (window.KnowledgeSessionCache) {
      window.KnowledgeSessionCache.invalidatePrefix(
        `${_PERSONAL_SS_PREFIX}${slug}|`,
      );
      window.KnowledgeSessionCache.invalidatePrefix(
        `${_PERSONAL_SS_PREFIX}${slug} `,
      );
    }
  }
  // Server-side cap on the personal-page payload. 300 fits the
  // initial 60-card paint plus several infinite-scroll slabs — past
  // that the user is typically searching anyway. Without the cap a
  // 4,000-row library shipped >3 MB of JSON for every personal-page
  // visit, dwarfing the feed's ~90 KB and slowing first paint.
  const PERSONAL_DOCS_LIMIT = 300;
  async function getPersonalPageDocuments(
    slug,
    {
      sources = null,
      tags = null,
      urls = null,
      excludeSources = null,
      categories = null,
      limit = PERSONAL_DOCS_LIMIT,
    } = {},
  ) {
    const sourcesArr = Array.isArray(sources) ? sources.filter(Boolean) : [];
    const tagsArr = Array.isArray(tags) ? tags.filter(Boolean) : [];
    const urlsArr = Array.isArray(urls) ? urls.filter(Boolean) : [];
    const excludeArr = Array.isArray(excludeSources)
      ? excludeSources.filter(Boolean)
      : [];
    const catsArr = Array.isArray(categories) ? categories.filter(Boolean) : [];
    // Bake the limit + categories into the cache key so callers that
    // narrow by topic get a separate cache slot (otherwise a Topics
    // toggle would surface the previous unfiltered slice from
    // cache).
    const key =
      _unindexedKey(slug, sourcesArr, tagsArr, urlsArr, excludeArr) +
      `|cat=${catsArr.slice().sort().join(",")}` +
      `|lim=${limit || ""}`;
    const hit = _personalDocsCache.get(key);
    if (hit && Date.now() - hit.at < UNINDEXED_TTL) return hit.list;
    // Cross-navigation fallback. The Map died on page reload but the
    // sessionStorage mirror may carry the last payload. Hydrate the
    // Map so subsequent reads in this page session skip the parse.
    if (window.KnowledgeSessionCache) {
      const persisted = window.KnowledgeSessionCache.get(
        _PERSONAL_SS_PREFIX + key,
        _PERSONAL_SS_TTL_MS,
      );
      if (Array.isArray(persisted)) {
        _personalDocsCache.set(key, { at: Date.now(), list: persisted });
        return persisted;
      }
    }
    return _coalesce(`personal-docs:${key}`, async () => {
      try {
        // Prod sets API="" (same-origin via Caddy). `new URL("/path")`
        // without a base throws, so pass `location.origin` as the
        // base — gives a fully-qualified URL in both prod and local
        // dev (where API is "http://localhost:8080").
        const u = new URL(
          `${API}/api/users/${encodeURIComponent(slug)}/documents`,
          location.origin,
        );
        // No `indexed` query param → endpoint returns everything,
        // ordered by (date DESC, created_at DESC).
        if (sourcesArr.length)
          u.searchParams.set("sources", sourcesArr.join(","));
        if (excludeArr.length)
          u.searchParams.set("exclude_sources", excludeArr.join(","));
        if (tagsArr.length) u.searchParams.set("tags", tagsArr.join(","));
        if (urlsArr.length) u.searchParams.set("urls", urlsArr.join(","));
        if (catsArr.length) u.searchParams.set("category", catsArr.join(","));
        if (limit && Number.isFinite(limit) && limit > 0) {
          u.searchParams.set("limit", String(Math.floor(limit)));
        }
        const obj = await fetchJson(u.toString());
        const list = Object.entries(obj).map(([url, m]) => {
          let linkedUrls = [];
          if (Array.isArray(m.linked_urls)) linkedUrls = m.linked_urls;
          else if (typeof m.linked_urls === "string" && m.linked_urls) {
            try {
              const parsed = JSON.parse(m.linked_urls);
              if (Array.isArray(parsed)) linkedUrls = parsed;
            } catch {
              /* ignore */
            }
          }
          const linkHosts = Array.isArray(m.link_hosts) ? m.link_hosts : [];
          return {
            url,
            title: m.title || "",
            summary: m.summary || "",
            date: m.date || "",
            createdAt: m.created_at || "",
            source: m.source || "",
            source_url: m.source_url || "",
            tags: Array.isArray(m.tags) ? m.tags : [],
            extraTags: Array.isArray(m["extra-tags"])
              ? m["extra-tags"]
              : Array.isArray(m.extra_tags)
                ? m.extra_tags
                : [],
            linkedUrls,
            linkHosts,
            // `_unindexed` only set for the rows where it's true,
            // so the not-indexed-yet badge still appears on those
            // (the renderer checks `d._unindexed`). The default
            // (undefined) is treated as "indexed" for badge
            // purposes by the existing renderer.
            ...(m.indexed === false ? { _unindexed: true } : {}),
          };
        });
        _personalDocsCache.set(key, { at: Date.now(), list });
        if (window.KnowledgeSessionCache) {
          window.KnowledgeSessionCache.set(_PERSONAL_SS_PREFIX + key, list);
        }
        return list;
      } catch {
        return [];
      }
    });
  }

  async function getUnindexedDocuments(
    slug,
    { sources = null, tags = null, urls = null, excludeSources = null } = {},
  ) {
    const sourcesArr = Array.isArray(sources) ? sources.filter(Boolean) : [];
    const tagsArr = Array.isArray(tags) ? tags.filter(Boolean) : [];
    const urlsArr = Array.isArray(urls) ? urls.filter(Boolean) : [];
    const excludeArr = Array.isArray(excludeSources)
      ? excludeSources.filter(Boolean)
      : [];
    const key = _unindexedKey(slug, sourcesArr, tagsArr, urlsArr, excludeArr);
    const hit = _unindexedCache.get(key);
    if (hit && Date.now() - hit.at < UNINDEXED_TTL) return hit.list;
    return _coalesce(`unindexed:${key}`, async () => {
      try {
        // Same `new URL` base trick as `getPersonalPageDocuments`:
        // prod's API="" makes the bare path argument throw without
        // a base; `location.origin` resolves both prod and local.
        const u = new URL(
          `${API}/api/users/${encodeURIComponent(slug)}/documents`,
          location.origin,
        );
        u.searchParams.set("indexed", "false");
        if (sourcesArr.length)
          u.searchParams.set("sources", sourcesArr.join(","));
        if (excludeArr.length)
          u.searchParams.set("exclude_sources", excludeArr.join(","));
        if (tagsArr.length) u.searchParams.set("tags", tagsArr.join(","));
        if (urlsArr.length) u.searchParams.set("urls", urlsArr.join(","));
        const obj = await fetchJson(u.toString());
        const list = Object.entries(obj).map(([url, m]) => ({
          url,
          title: m.title || "",
          summary: m.summary || "",
          date: m.date || "",
          // ISO-8601 UTC stamp from documents.created_at — used by
          // reorderForBrowse() to break ties when two docs share the
          // same date (typical right after the compose dialog).
          createdAt: m.created_at || "",
          source: m.source || "",
          source_url: m.source_url || "",
          tags: Array.isArray(m.tags) ? m.tags : [],
          extraTags: Array.isArray(m["extra-tags"])
            ? m["extra-tags"]
            : Array.isArray(m.extra_tags)
              ? m.extra_tags
              : [],
          _unindexed: true,
        }));
        _unindexedCache.set(key, { at: Date.now(), list });
        return list;
      } catch {
        return [];
      }
    });
  }

  /** Full-text search via the ColBERT API.
   *
   * `filter` (when set) pushes pre-search filtering down to the
   * index — the engine only scores docs matching the SQL WHERE
   * clause, so a query like `"embeddings"` with `source = "lighton"`
   * doesn't get crowded out of the top-K by github noise. Shape:
   *   { condition: "source IN (?,?)", parameters: ["github","huggingface"] }
   *
   * `subset` is the "narrow to a known set of document ids" knob.
   * Used by `findSimilar` and related flows that already hold a
   * list of `_subset_` ids and want the engine to score only those
   * candidates — cheaper than building an equivalent SQL filter. */
  /** SQL fallback for libraries whose ColBERT index is missing on
   * disk. Returns rows in the same shape the plaid endpoints emit,
   * so the per-call-site post-processing (transformMeta + similarity
   * spread) doesn't change. `subset`/`filter` are dropped — the
   * fallback intentionally ignores them because the docs the caller
   * is filtering against don't exist server-side without an index. */
  async function fallbackSearch({ indexName, query = "", topK = 60 }) {
    const qs = new URLSearchParams();
    if (query) qs.set("q", query);
    qs.set("limit", String(topK));
    const r = await fetch(
      `${API}/api/personalities/${encodeURIComponent(indexName)}/fallback?${qs}`,
    );
    if (!r.ok) return null; // null signals "fallback unavailable too"
    return r.json();
  }

  async function search({
    indexName,
    query,
    topK = 60,
    subset = null,
    filter = null,
  }) {
    const body = { queries: [query], params: { top_k: topK } };
    if (subset) body.subset = subset;
    let path = "search_with_encoding";
    if (filter && filter.condition) {
      body.filter_condition = filter.condition;
      body.filter_parameters = filter.parameters || [];
      path = "search/filtered_with_encoding";
    }
    const r = await fetch(`${API}/indices/${indexName}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // SQL fallback path. Only triggers on 404 (index not declared on
    // disk) — broken/error indices return 5xx and we let those
    // propagate, because the heal hook in run.py is what fixes them
    // and we don't want the fallback to mask a real failure. Subset
    // search isn't faithfully reproducible in SQL (the engine scores
    // candidates from a fixed id list), so we skip the fallback in
    // that case and let the empty result fall through.
    if (r.status === 404 && !subset) {
      const data = await fallbackSearch({ indexName, query, topK });
      if (!data) return [];
      const result = (data.results && data.results[0]) || {};
      const meta = result.metadata || [];
      const scores = result.scores || [];
      return meta.map((m, i) => ({
        ...transformMeta(m),
        similarity: scores[i] || 0,
      }));
    }
    const data = await r.json();
    const result = (data.results && data.results[0]) || {};
    const meta = result.metadata || [];
    const scores = result.scores || [];
    return meta.map((m, i) => ({
      ...transformMeta(m),
      similarity: scores[i] || 0,
    }));
  }

  /** Latest-by-date fallback when no query. Optionally filtered by condition.
   *
   * Sends `limit` to the server so the metadata blob is trimmed before
   * serialization — without this, `__all__` returned the full 22 MB
   * payload (50k rows) for every browse-mode refresh. The server-side
   * cap doesn't apply ORDER BY, so we still re-sort by date locally
   * to guarantee newest-first when the server's natural order isn't.
   * Client-side slice is now a safety net only.
   */
  async function latest({
    indexName,
    count = 60,
    condition = null,
    parameters = null,
  }) {
    // Over-fetch a bit (`count * 3`, capped at 600) so the local
    // sort still has room to reorder by date — the server returns
    // whatever rows match in storage order, which may not be date-
    // descending. The SQLite scan cost is the same with or without
    // the limit; only the transfer + parse get cheaper.
    const serverLimit = Math.min(600, Math.max(count * 3, 200));
    // order_by:"date" tells the API to sort by date before applying
    // `limit`. Without this, the server orders rows by `_subset_`
    // (SQLite insertion order) and truncates — so the most recently
    // indexed docs (highest subset id) get dropped before the client
    // gets a chance to sort by date. The client-side sort below is
    // now redundant but kept as a safety net.
    const body = condition
      ? {
          condition,
          parameters: parameters || [],
          limit: serverLimit,
          order_by: "date",
        }
      : {
          condition: "date != ?",
          parameters: [""],
          limit: serverLimit,
          order_by: "date",
        };
    const r = await fetch(`${API}/indices/${indexName}/metadata/get`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // SQL fallback for the browse-mode feed when the plaid index is
    // missing. Custom `condition`/`parameters` callers (e.g. trying
    // to enumerate by tag through metadata/get) get an empty feed
    // because the fallback's query model doesn't carry that — those
    // call sites already have to handle the missing-index case
    // elsewhere, and the user's request was scoped to the basic
    // browse/search flows.
    if (r.status === 404 && !condition) {
      const data = await fallbackSearch({ indexName, query: "", topK: count });
      if (!data) return [];
      const rows = (data.metadata || []).map(transformMeta);
      rows.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
      return rows.slice(0, count);
    }
    const data = await r.json();
    const rows = (data.metadata || []).map(transformMeta);
    rows.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    return rows.slice(0, count);
  }

  /* Cross-library intersection. Returns docs owned by ≥ 2 of the
   * supplied slugs, with each doc's `_owners` reflecting its actual
   * subset (not the input set). Server-side ordering is owner count
   * desc, then date desc, so a 3-way overlap from two years ago
   * outranks yesterday's 2-way overlap.
   *
   * Returns `[]` when fewer than 2 slugs are passed (no intersection
   * to compute) or when nothing is shared. */
  async function intersect(slugs, limit = 200) {
    if (!Array.isArray(slugs) || slugs.length < 2) return [];
    const qs = new URLSearchParams({
      slugs: slugs.join(","),
      limit: String(limit),
    });
    let payload;
    try {
      payload = await fetchJson(`${API}/api/users/intersect?${qs.toString()}`);
    } catch {
      return [];
    }
    const docs = (payload && payload.documents) || {};
    const fallbackOwners =
      payload && Array.isArray(payload.owners) ? payload.owners : slugs;
    const out = [];
    for (const [url, m] of Object.entries(docs)) {
      const docOwners =
        Array.isArray(m.owners) && m.owners.length
          ? m.owners
          : fallbackOwners.slice();
      out.push({
        url,
        title: m.title || "",
        summary: m.summary || "",
        date: m.date || "",
        source: m.source || "",
        source_url: m.source_url || "",
        tags: Array.isArray(m.tags) ? m.tags : [],
        extraTags: Array.isArray(m["extra-tags"]) ? m["extra-tags"] : [],
        _owners: docOwners.slice(),
        _unindexed: m.indexed === false,
      });
    }
    // Preserve server-side ordering (owner_count desc, then date desc)
    // — the diversity reorder on the page side already tiers by
    // _owners.length, but keeping this stable matters when the page
    // skips the reorder (e.g. tag-filtered browse).
    return out;
  }

  function transformMeta(m) {
    // Tags + extra_tags come back as comma-strings from the ColBERT
    // index (where metadata sits in SQLite) and as real arrays from
    // the PG fallback endpoint. Accept both — the cards downstream
    // expect a `tags` array regardless of source.
    const toTagArray = (v) => {
      if (Array.isArray(v)) {
        return v.map((t) => String(t).trim()).filter(Boolean);
      }
      if (typeof v === "string" && v) {
        return v
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean);
      }
      return [];
    };
    const tagsArr = toTagArray(m.tags);
    const extraArr = toTagArray(m.extra_tags);
    // linked_urls travels through the SQLite index as a JSON string
    // (PG → indexer → metadata sidecar) and through the PG endpoints
    // as a real array. Accept both, fail safe to an empty list — the
    // card renderer treats "no link cards" as a valid state.
    let linkedUrls = [];
    if (Array.isArray(m.linked_urls)) {
      linkedUrls = m.linked_urls;
    } else if (typeof m.linked_urls === "string" && m.linked_urls) {
      try {
        const parsed = JSON.parse(m.linked_urls);
        if (Array.isArray(parsed)) linkedUrls = parsed;
      } catch {
        /* malformed — treat as empty */
      }
    }
    // link_hosts mirrors the PG array column. Indexed-side carries a
    // comma-encoded string for parity with `tags` / `extra_tags`.
    let linkHosts = [];
    if (Array.isArray(m.link_hosts)) {
      linkHosts = m.link_hosts;
    } else if (typeof m.link_hosts === "string" && m.link_hosts) {
      linkHosts = m.link_hosts
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    }
    return {
      url: m.url,
      title: m.title || "",
      summary: m.summary || "",
      date: m.date || "",
      source: m.source || "",
      source_url: m.source_url || "",
      // `owner` is only set by the `__all__` cross-personality index,
      // where it carries the personality slug. Per-user indexes
      // don't write it; default to "" so callers don't need to
      // null-check.
      owner: m.owner || "",
      tags: tagsArr,
      extraTags: extraArr,
      linkedUrls,
      linkHosts,
    };
  }

  /* ── Favorites (optional, requires session) ─────────────────────────── */
  async function getFavoriteUrls() {
    try {
      const list = await fetchJson(`${API}/auth/me/favorite-docs`);
      return new Set(Array.isArray(list) ? list : []);
    } catch {
      return new Set();
    }
  }
  /* Distinct slugs owning at least one favorited URL — used by the
   * search page to broaden its per-lib fanout whenever the Favorites
   * filter is on, so a user's stars surface even when the owning lib
   * isn't currently selected. */
  async function getFavoriteOwners() {
    try {
      const list = await fetchJson(`${API}/auth/me/favorite-docs/owners`);
      return Array.isArray(list) ? list : [];
    } catch {
      return [];
    }
  }
  /* Hydrated favorite docs — bypasses the per-lib search fanout so
   * stars from libraries the user hasn't selected still surface when
   * the Favorites filter is on. Returns doc-shaped objects matching
   * the search-result shape (url/title/summary/date/tags/source). */
  async function getFavoriteDocs() {
    try {
      const list = await fetchJson(`${API}/auth/me/favorite-docs/full`);
      return Array.isArray(list)
        ? list.map((d) => ({
            url: d.url,
            title: d.title,
            summary: d.summary,
            date: d.date,
            tags: d.tags || [],
            extraTags: d["extra-tags"] || [],
            source: d.source,
            source_url: d.source_url || null,
          }))
        : [];
    } catch {
      return [];
    }
  }

  /* ── Personality bookmarks (cross-user "follow") ─────────────────────
   * The signed-in user's saved-people list. Surfaces in the library
   * picker as a dedicated "Bookmarks" section above the by-category
   * grouping.
   *
   * Backed by the `favorites` table — the single source of truth
   * for "user → user" starring across the whole product. Welcome
   * page rail, search-page picker, and profile all read from this
   * table via /auth/favorites, so toggling here is immediately
   * visible everywhere else.
   *
   * Cross-page live sync: every toggle posts a message to a shared
   * BroadcastChannel ("knowledge:personality-bookmarks") so any
   * other tab/page (welcome, another search tab) can patch its
   * local state immediately, without round-tripping the network. */
  const _bookmarkChannel =
    typeof BroadcastChannel === "function"
      ? new BroadcastChannel("knowledge:personality-bookmarks")
      : null;
  const _bookmarkSubscribers = new Set();
  if (_bookmarkChannel) {
    _bookmarkChannel.addEventListener("message", (ev) => {
      const { slug, bookmarked } = ev.data || {};
      if (typeof slug !== "string") return;
      for (const fn of _bookmarkSubscribers) {
        try {
          fn({ slug, bookmarked: !!bookmarked });
        } catch {}
      }
    });
  }
  function onPersonalityBookmarkChange(fn) {
    _bookmarkSubscribers.add(fn);
    return () => _bookmarkSubscribers.delete(fn);
  }
  async function getPersonalityBookmarks() {
    try {
      const list = await fetchJson(`${API}/auth/favorites`);
      return new Set(Array.isArray(list) ? list : []);
    } catch {
      return new Set();
    }
  }
  async function togglePersonalityBookmark(slug, already) {
    const init = {
      method: already ? "DELETE" : "PUT",
      credentials: "include",
    };
    const r = await fetch(
      `${API}/auth/favorites/${encodeURIComponent(slug)}`,
      init,
    );
    if (r.ok && _bookmarkChannel) {
      try {
        _bookmarkChannel.postMessage({ slug, bookmarked: !already });
      } catch {}
    }
    return r.ok;
  }

  async function toggleFavorite(url, already) {
    const init = {
      method: already ? "DELETE" : "POST",
      credentials: "include",
      headers: already ? {} : { "Content-Type": "application/json" },
      body: already ? null : JSON.stringify({ url }),
    };
    const endpoint = already
      ? `${API}/auth/me/favorite-docs?url=${encodeURIComponent(url)}`
      : `${API}/auth/me/favorite-docs`;
    const r = await fetch(endpoint, init);
    return r.ok;
  }

  /* ── Similar docs via re-search ─────────────────────────────────────── */
  async function findSimilar({ indexName, doc, topK = 8 }) {
    const parts = [doc.title];
    if (doc.tags.length) parts.push(doc.tags.join(" "));
    if (doc.summary)
      parts.push(doc.summary.split(/\s+/).slice(0, 20).join(" "));
    const q = parts.join(" ");
    const docs = await search({ indexName, query: q, topK: topK + 1 });
    return docs.filter((d) => d.url !== doc.url).slice(0, topK);
  }

  /* ── Screenshot helper ─────────────────────────────────────────────────
   * Thum.io is free, CORS-enabled, and gives a predictable PNG/JPEG URL
   * from any site. We cap width so the browser doesn't fetch megabytes
   * per card. If a site refuses to render (timeout, paywall), thum.io
   * returns a generic error image — callers swap to the favicon
   * fallback on load error.
   */
  function screenshotUrl(url, width = 600) {
    if (!url) return "";
    return `https://image.thum.io/get/width/${width}/noanimate/${url}`;
  }

  function faviconUrl(host) {
    if (!host) return "";
    return `https://icons.duckduckgo.com/ip3/${host}.ico`;
  }

  /* Map a source-key to its canonical hostname so the favicon CDN
   * resolves it. Source keys come in two flavours from the API:
   *   - hostnames (e.g. "lennysnewsletter.com")  → use as-is
   *   - category slugs (e.g. "github", "twitter") → map to host
   * Unknown slugs without a dot fall through to a generic favicon.
   * */
  const SOURCE_HOSTS = {
    github: "github.com",
    gist: "gist.github.com",
    twitter: "twitter.com",
    x: "x.com",
    hackernews: "news.ycombinator.com",
    hn: "news.ycombinator.com",
    reddit: "reddit.com",
    youtube: "youtube.com",
    youtube_search: "youtube.com",
    yt: "youtube.com",
    huggingface: "huggingface.co",
    hf: "huggingface.co",
    arxiv: "arxiv.org",
    scholar: "scholar.google.com",
    google_scholar: "scholar.google.com",
    semantic_scholar: "semanticscholar.org",
    semanticscholar: "semanticscholar.org",
    dblp: "dblp.org",
    stackoverflow: "stackoverflow.com",
    stackexchange: "stackexchange.com",
    so: "stackoverflow.com",
    wikipedia: "en.wikipedia.org",
    wiki: "en.wikipedia.org",
    medium: "medium.com",
    substack: "substack.com",
    zotero: "zotero.org",
    linkedin: "linkedin.com",
    podcast: "podcasts.apple.com",
    blog: "",
  };
  function sourceHost(key) {
    if (!key) return "";
    const k = key.toLowerCase();
    if (SOURCE_HOSTS.hasOwnProperty(k)) return SOURCE_HOSTS[k];
    // Anything with a dot is already a hostname-like key.
    if (k.includes(".")) return k.replace(/^www\./, "");
    return "";
  }
  /* Bundled brand icons under /icons/ — preferred over DuckDuckGo
   * favicons for well-known platforms because (a) they're already on
   * the same origin (no external CDN hop), (b) they're the SVGs the
   * rest of the app uses, so the rail visually matches the search
   * results' source labels, and (c) DuckDuckGo's favicon for
   * `news.ycombinator.com` is a tiny off-centre Y, hard to read at
   * 16px. */
  const BUNDLED_ICONS = {
    github: "/icons/github.png",
    gist: "/icons/github.png",
    twitter: "/icons/twitter.png",
    x: "/icons/twitter.png",
    hackernews: "/icons/hackernews.png",
    hn: "/icons/hackernews.png",
    reddit: "/icons/reddit.svg",
    huggingface: "/icons/huggingface.svg",
    hf: "/icons/huggingface.svg",
    arxiv: "/icons/arxiv.svg",
    scholar: "/icons/scholar.svg",
    google_scholar: "/icons/scholar.svg",
    stackoverflow: "/icons/stackoverflow.svg",
    so: "/icons/stackoverflow.svg",
    medium: "/icons/medium.svg",
    substack: "/icons/substack.svg",
  };
  function sourceIconUrl(key) {
    if (!key) return "";
    const k = key.toLowerCase();
    if (BUNDLED_ICONS.hasOwnProperty(k)) return BUNDLED_ICONS[k];
    // Fall back to DuckDuckGo favicon for raw hostnames
    // (e.g. "lighton.ai", "zeroentropy.dev").
    const host = sourceHost(key);
    if (!host) return "";
    return faviconUrl(host);
  }

  /* ── URL helpers ───────────────────────────────────────────────────── */
  function hostOf(url) {
    try {
      return new URL(url).hostname.toLowerCase().replace(/^www\./, "");
    } catch {
      return "";
    }
  }

  function formatDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  }

  function relativeDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const days = Math.floor((Date.now() - d.getTime()) / (1000 * 60 * 60 * 24));
    if (days < 1) return "today";
    if (days < 7) return `${days}d`;
    if (days < 30) return `${Math.floor(days / 7)}w`;
    if (days < 365) return `${Math.floor(days / 30)}mo`;
    return `${Math.floor(days / 365)}y`;
  }

  /* Weekly date label used on the feed (no library selected). The
   * server-side feed scorer buckets by 7-day windows, so the card
   * label matches that bucketing instead of day-level granularity.
   * Per-library and search views still call `relativeDate` because
   * they're date-sorted, not week-bucketed. */
  function feedRelativeDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const days = Math.floor((Date.now() - d.getTime()) / (1000 * 60 * 60 * 24));
    if (days < 7) return "This week";
    if (days < 14) return "A week ago";
    // Stretch the "N weeks ago" band to 30 days — at 28/29 days the
    // month bucket rounds to 0 and we'd render "0 months ago".
    if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
    if (days < 365) {
      const m = Math.floor(days / 30);
      return m === 1 ? "A month ago" : `${m} months ago`;
    }
    const y = Math.floor(days / 365);
    return y === 1 ? "A year ago" : `${y} years ago`;
  }

  /* ── Source bucketing ─────────────────────────────────────────────────
   * ~15 main categories + a long tail of per-hostname sources. The
   * helpers below let the UI group them however suits the layout.
   */
  const MAIN_SOURCES = new Set([
    "github",
    "twitter",
    "hackernews",
    "reddit",
    "youtube",
    "scholar",
    "arxiv",
    "huggingface",
    "wikipedia",
    "stackoverflow",
    "blog",
  ]);

  function groupSources(sources) {
    const main = [];
    const hosts = [];
    for (const s of sources || []) {
      if (!s || !s.key) continue;
      if (MAIN_SOURCES.has(s.key)) main.push(s);
      else hosts.push(s);
    }
    hosts.sort((a, b) => (b.count || 0) - (a.count || 0));
    main.sort((a, b) => (b.count || 0) - (a.count || 0));
    return { main, hosts };
  }

  window.KnowledgeAPI = {
    API,
    DEFAULT_SLUG,
    listPersonalities,
    getPersonality,
    getSources,
    getAllSources,
    getDocuments,
    getUnindexedDocuments,
    getPersonalPageDocuments,
    invalidatePersonalDocs,
    probeCacheGet,
    probeCacheSet,
    invalidateCaches,
    invalidateUnindexed,
    search,
    latest,
    intersect,
    getFavoriteUrls,
    getFavoriteDocs,
    getFavoriteOwners,
    toggleFavorite,
    getPersonalityBookmarks,
    togglePersonalityBookmark,
    onPersonalityBookmarkChange,
    findSimilar,
    screenshotUrl,
    faviconUrl,
    sourceIconUrl,
    sourceHost,
    hostOf,
    formatDate,
    relativeDate,
    feedRelativeDate,
    MAIN_SOURCES,
    groupSources,
  };
})();
