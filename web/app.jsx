const { useState, useEffect, useRef, useCallback, useMemo } = React;
const { createRoot, createPortal } = ReactDOM;

const IS_LOCAL = window.location.hostname === "localhost";
const API_BASE_URL = IS_LOCAL ? "http://localhost:8080" : "";
const DATA_API_URL = API_BASE_URL;
const EVENTS_API_URL = API_BASE_URL;
const INDEX_NAME = "knowledge";
const SEARCH_DEBOUNCE_MS = 400;
const SEARCH_SETTLE_MS = 2000;
const FLUSH_INTERVAL_MS = 10000;
const MAX_BUFFER_SIZE = 50;
const FETCH_COUNT = 300;
const DISPLAY_COUNT = 30;
const MAX_CANDIDATES = 1000;
const RERANK_INACTIVITY_MS = 1000;
const SUMMARY_TOKEN_LIMIT = 30;
const SIMILAR_COUNT = 10;

const syncURL = (params) => {
  const url = new URL(window.location);
  Object.entries(params).forEach(([k, v]) => {
    if (k === "source" || k === "tags") {
      if (v instanceof Set && v.size > 0)
        url.searchParams.set(k, [...v].join(","));
      else url.searchParams.delete(k);
    } else if (v) url.searchParams.set(k, v);
    else url.searchParams.delete(k);
  });
  const qs = url.searchParams.toString();
  window.history.pushState({}, null, qs ? `?${qs}` : window.location.pathname);
};

// --- Anonymous Analytics (RGPD-compliant, CNIL audience measurement exemption) ---

const analyticsOptedOut = () =>
  localStorage.getItem("_analytics_optout") === "1";

const getSessionId = () => {
  let id = sessionStorage.getItem("_sid");
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem("_sid", id);
  }
  return id;
};

const deviceType = window.innerWidth <= 768 ? "mobile" : "desktop";
const referrerDomain = (() => {
  try {
    return document.referrer ? new URL(document.referrer).hostname : "";
  } catch {
    return "";
  }
})();

const eventBuffer = [];
const clickedUrls = new Set();

const flushEvents = (useBeacon = false) => {
  if (eventBuffer.length === 0) return;
  const batch = eventBuffer.splice(0, MAX_BUFFER_SIZE);
  const body = JSON.stringify(batch);
  if (useBeacon && navigator.sendBeacon) {
    // sendBeacon with text/plain avoids CORS preflight on cross-origin
    navigator.sendBeacon(
      `${EVENTS_API_URL}/events`,
      new Blob([body], { type: "text/plain" }),
    );
  } else {
    fetch(`${EVENTS_API_URL}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {});
  }
};

const trackEvent = (eventType, payload) => {
  if (analyticsOptedOut()) return;
  if (eventBuffer.length >= MAX_BUFFER_SIZE) eventBuffer.shift();
  eventBuffer.push({
    session_id: getSessionId(),
    event_type: eventType,
    payload: { ...payload, device_type: deviceType },
  });
};

// Track page view once per session
if (!analyticsOptedOut() && !sessionStorage.getItem("_pv")) {
  sessionStorage.setItem("_pv", "1");
  trackEvent("page_view", {
    referrer_domain: referrerDomain,
  });
}

// Flush every 10s via fetch; use sendBeacon only on page unload
setInterval(flushEvents, FLUSH_INTERVAL_MS);
window.addEventListener("beforeunload", () => flushEvents(true));

/**
 * Transforms a metadata row from the API (comma-separated tags) into
 * the frontend document format (array tags).
 */
const transformMeta = (meta) => ({
  url: meta.url || "",
  title: meta.title || "",
  summary: meta.summary || "",
  date: meta.date || "",
  tags: meta.tags ? meta.tags.split(",").filter(Boolean) : [],
  "extra-tags": meta.extra_tags
    ? meta.extra_tags.split(",").filter(Boolean)
    : [],
});

/**
 * Builds a SQL WHERE condition + parameters for a source filter key.
 * Returns { condition, parameters } or null if no filter.
 */
const buildTagConditionExact = (tagFilters) => {
  if (!(tagFilters instanceof Set) || tagFilters.size === 0) return null;
  const clauses = [];
  const parameters = [];
  for (const tag of tagFilters) {
    // Comma-boundary match: ",python," won't match "cpython" or "python3".
    clauses.push(
      "(',' || tags || ',' LIKE ? OR ',' || extra_tags || ',' LIKE ?)",
    );
    parameters.push(`%,${tag},%`, `%,${tag},%`);
  }
  return { condition: clauses.join(" AND "), parameters };
};

const buildTagConditionLike = (tagFilters) => {
  if (!(tagFilters instanceof Set) || tagFilters.size === 0) return null;
  const clauses = [];
  const parameters = [];
  for (const tag of tagFilters) {
    clauses.push("(tags LIKE ? OR extra_tags LIKE ?)");
    parameters.push(`%${tag}%`, `%${tag}%`);
  }
  return { condition: clauses.join(" AND "), parameters };
};

const buildSourceCondition = (sourceSet) => {
  if (!(sourceSet instanceof Set) || sourceSet.size === 0) return null;
  // "other" is handled client-side only
  const keys = [...sourceSet].filter((k) => k !== "other");
  if (keys.length === 0) return null;

  const clauses = [];
  const parameters = [];
  for (const source of keys) {
    if (!source.includes(".")) {
      // Tag/title based sources (e.g. hackernews)
      clauses.push("(tags LIKE ? OR extra_tags LIKE ? OR title LIKE ?)");
      parameters.push(`%${source}%`, `%${source}%`, `%${source}%`);
    } else {
      // Domain-based sources — also match common aliases
      const patterns = [source];
      if (source === "twitter.com") patterns.push("x.com");
      const urlParts = patterns.map(() => "url LIKE ?");
      clauses.push(`(${urlParts.join(" OR ")})`);
      parameters.push(...patterns.map((p) => `%${p}%`));
    }
  }
  return {
    condition: clauses.join(" OR "),
    parameters,
  };
};

/**
 * Gets the subset of document IDs matching a source filter.
 * Returns an array of IDs, or null if no filter is active.
 */
const getFilteredSubset = async (source, tagCond = null) => {
  const sourceCond = buildSourceCondition(source);
  if (!sourceCond && !tagCond) return null;
  let condition, parameters;
  if (sourceCond && tagCond) {
    condition = `(${sourceCond.condition}) AND (${tagCond.condition})`;
    parameters = [...sourceCond.parameters, ...tagCond.parameters];
  } else if (sourceCond) {
    condition = sourceCond.condition;
    parameters = sourceCond.parameters;
  } else {
    condition = tagCond.condition;
    parameters = tagCond.parameters;
  }
  const resp = await fetch(
    `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ condition, parameters }),
    },
  );
  const data = await resp.json();
  return data.document_ids || null;
};

/**
 * Fetches the latest documents via metadata query.
 * When tagFilters is non-empty, runs two queries (exact then LIKE) and
 * concatenates results without duplicates, exact-match first.
 */
const apiLatest = async (count, source, tagFilters = null) => {
  const sourceCond = buildSourceCondition(source);
  const hasTagFilters = tagFilters instanceof Set && tagFilters.size > 0;

  const fetchMeta = async (tagCond) => {
    const parts = [];
    const params = [];
    if (sourceCond) {
      parts.push(`(${sourceCond.condition})`);
      params.push(...sourceCond.parameters);
    }
    if (tagCond) {
      parts.push(`(${tagCond.condition})`);
      params.push(...tagCond.parameters);
    }
    parts.push("date != ?");
    params.push("");
    const resp = await fetch(
      `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          condition: parts.join(" AND "),
          parameters: params,
        }),
      },
    );
    const data = await resp.json();
    return (data.metadata || []).map(transformMeta);
  };

  if (!hasTagFilters) {
    const rows = await fetchMeta(null);
    rows.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    return rows.slice(0, count);
  }

  // Two-stage: exact match first, then LIKE %tag%
  const [exactRows, likeRows] = await Promise.all([
    fetchMeta(buildTagConditionExact(tagFilters)),
    fetchMeta(buildTagConditionLike(tagFilters)),
  ]);
  exactRows.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  const exactUrls = new Set(exactRows.map((r) => r.url));
  const broadOnly = likeRows.filter((r) => !exactUrls.has(r.url));
  broadOnly.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  return [...exactRows, ...broadOnly].slice(0, count);
};

/**
 * Searches documents via ColBERT encoding and returns results.
 * When tagFilters is non-empty, uses a broad LIKE subset for the single
 * ColBERT search, then reorders results so exact-match docs come first.
 */
const apiSearch = async (
  query,
  sortByDate = false,
  source,
  tagFilters = null,
  topK = FETCH_COUNT,
) => {
  const hasTagFilters = tagFilters instanceof Set && tagFilters.size > 0;

  const runSearch = async (subset) => {
    const body = { queries: [query], params: { top_k: topK } };
    if (subset) body.subset = subset;
    const resp = await fetch(
      `${API_BASE_URL}/indices/${INDEX_NAME}/search_with_encoding`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    const data = await resp.json();
    const result = (data.results && data.results[0]) || {};
    const metadata = result.metadata || [];
    const scores = result.scores || [];
    return metadata.map((meta, i) => ({
      ...transformMeta(meta),
      similarity: scores[i] || 0,
    }));
  };

  if (!hasTagFilters) {
    const subset = await getFilteredSubset(source, null);
    let docs = await runSearch(subset);
    if (sortByDate)
      docs.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    return docs;
  }

  // Two-stage: get exact URLs and broad subset IDs in parallel, single ColBERT call.
  const sourceCond = buildSourceCondition(source);
  const exactTagCond = buildTagConditionExact(tagFilters);
  const likeTagCond = buildTagConditionLike(tagFilters);

  const buildMetaGetBody = (tagCond) => {
    const parts = [];
    const params = [];
    if (sourceCond) {
      parts.push(`(${sourceCond.condition})`);
      params.push(...sourceCond.parameters);
    }
    if (tagCond) {
      parts.push(`(${tagCond.condition})`);
      params.push(...tagCond.parameters);
    }
    return { condition: parts.join(" AND ") || "1=1", parameters: params };
  };

  const [exactMetaData, likeSubset] = await Promise.all([
    fetch(`${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildMetaGetBody(exactTagCond)),
    }).then((r) => r.json()),
    getFilteredSubset(source, likeTagCond),
  ]);

  const exactUrls = new Set((exactMetaData.metadata || []).map((m) => m.url));

  // Single ColBERT search restricted to broad (LIKE) subset
  const docs = await runSearch(likeSubset);

  // Reorder: exact-match docs first (preserving ColBERT score order within each group)
  const exactDocs = docs.filter((d) => exactUrls.has(d.url));
  const broadOnly = docs.filter((d) => !exactUrls.has(d.url));

  if (sortByDate) {
    exactDocs.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    broadOnly.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  }

  return [...exactDocs, ...broadOnly];
};

/**
 * Fetches documents similar to a given doc by searching with its content.
 */
const fetchSimilar = async (doc) => {
  const parts = [doc.title];
  const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
  if (allTags.length) parts.push(allTags.join(" "));
  if (doc.summary) parts.push(doc.summary.split(/\s+/).slice(0, 20).join(" "));
  const queryText = parts.join(" ");

  const resp = await fetch(
    `${API_BASE_URL}/indices/${INDEX_NAME}/search_with_encoding`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        queries: [queryText],
        params: { top_k: SIMILAR_COUNT + 1 },
      }),
    },
  );
  const data = await resp.json();
  const result = (data.results && data.results[0]) || {};
  const metadata = result.metadata || [];
  const scores = result.scores || [];
  return metadata
    .map((meta, i) => ({ ...transformMeta(meta), similarity: scores[i] || 0 }))
    .filter((d) => d.url !== doc.url)
    .slice(0, SIMILAR_COUNT);
};

// --- Custom Folders (localStorage) ---
const CUSTOM_FOLDERS_KEY = "finder-custom-folders";
const loadCustomFolders = () => {
  try {
    return JSON.parse(localStorage.getItem(CUSTOM_FOLDERS_KEY)) || [];
  } catch {
    return [];
  }
};
const saveCustomFolders = (folders) => {
  localStorage.setItem(CUSTOM_FOLDERS_KEY, JSON.stringify(folders));
};

const FINDER_DOC_LIMIT = 50;

const SOURCE_ICONS = {
  "github.com": "\uD83D\uDCBB",
  "twitter.com": "\uD835\uDD4F",
  "arxiv.org": "\uD83D\uDCDD",
  "huggingface.co": "\uD83E\uDD17",
  hackernews: "\uD83D\uDCF0",
  "superuser.com": "\uD83D\uDEE0\uFE0F",
  "ieeexplore.ieee.org": "\uD83C\uDF93",
};
const getFinderSourceIcon = (key) => SOURCE_ICONS[key] || "\uD83D\uDCC1";

/**
 * CreateFolderModal — overlay for creating a custom folder.
 */
// Count how many of the given tags a doc matches (exact comma-boundary)
const countTagMatches = (doc, tagList) => {
  const docTags = new Set([...(doc.tags || []), ...(doc["extra-tags"] || [])]);
  return tagList.filter((t) => docTags.has(t)).length;
};

// ── Custom-folder tree helpers ───────────────────────────────────────────────

const folderToItem = (f) => ({
  kind: "custom",
  id: f.id,
  label: f.name,
  filterType: f.filterType,
  folderData: f,
});

const findFolderById = (folders, id) => {
  for (const f of folders) {
    if (f.id === id) return f;
    const found = findFolderById(f.children || [], id);
    if (found) return found;
  }
  return null;
};

const addFolderToTree = (folders, parentId, newFolder) => {
  const node = { ...newFolder, children: newFolder.children || [] };
  if (!parentId) return [...folders, node]; // root level
  return folders.map((f) => {
    if (f.id === parentId)
      return { ...f, children: [...(f.children || []), node] };
    return {
      ...f,
      children: addFolderToTree(f.children || [], parentId, node),
    };
  });
};

const updateFolderInTree = (folders, updated) =>
  folders.map((f) => {
    if (f.id === updated.id) return { ...updated, children: f.children || [] }; // preserve children
    return { ...f, children: updateFolderInTree(f.children || [], updated) };
  });

const removeFolderFromTree = (folders, id) =>
  folders
    .filter((f) => f.id !== id)
    .map((f) => ({
      ...f,
      children: removeFolderFromTree(f.children || [], id),
    }));

const getSiblingNames = (folders, parentId) => {
  if (!parentId) return folders.map((f) => f.name);
  const parent = findFolderById(folders, parentId);
  return (parent?.children || []).map((f) => f.name);
};

// Returns the parentId of a folder by id (null = root, undefined = not found)
const findParentId = (folders, id, currentParent = null) => {
  for (const f of folders) {
    if (f.id === id) return currentParent;
    const result = findParentId(f.children || [], id, f.id);
    if (result !== undefined) return result;
  }
  return undefined;
};

// ─────────────────────────────────────────────────────────────────────────────

const CreateFolderModal = ({
  onClose,
  onCreate,
  initialFolder = null,
  existingNames = [],
}) => {
  const isEditing = initialFolder != null;
  const [name, setName] = useState(initialFolder?.name ?? "");
  const [filterType, setFilterType] = useState(
    initialFolder?.filterType ?? "none",
  );
  const [value, setValue] = useState(
    initialFolder?.filterType === "search"
      ? (initialFolder.searchQuery ?? "")
      : initialFolder?.filterType === "urls"
        ? (initialFolder.urls ?? []).join("\n")
        : "",
  );
  const [topK, setTopK] = useState(initialFolder?.topK ?? 50);
  const [live, setLive] = useState(initialFolder?.live ?? true);
  // tag multi-select
  const [allTags, setAllTags] = useState([]);
  const [selectedTags, setSelectedTags] = useState(
    new Set(
      Array.isArray(initialFolder?.tagFilter)
        ? initialFolder.tagFilter
        : initialFolder?.tagFilter
          ? [initialFolder.tagFilter]
          : [],
    ),
  );
  const [tagSearch, setTagSearch] = useState("");
  const [tagIntersect, setTagIntersect] = useState(
    initialFolder?.tagIntersect ?? false,
  );

  // Load tags from metadata API (guarantees every tag has at least one document)
  useEffect(() => {
    if (filterType !== "tag" || allTags.length > 0) return;
    fetch(`${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ condition: "1=1", parameters: [] }),
    })
      .then((r) => r.json())
      .then((data) => {
        const tagSet = new Set();
        for (const meta of data.metadata || []) {
          if (meta.tags)
            meta.tags
              .split(",")
              .filter(Boolean)
              .forEach((t) => tagSet.add(t.trim()));
          if (meta.extra_tags)
            meta.extra_tags
              .split(",")
              .filter(Boolean)
              .forEach((t) => tagSet.add(t.trim()));
        }
        setAllTags([...tagSet].sort());
      })
      .catch(() => {});
  }, [filterType]);

  const toggleTag = (tag) =>
    setSelectedTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      return next;
    });

  const filteredTags = tagSearch
    ? allTags.filter((t) => t.includes(tagSearch.toLowerCase()))
    : allTags;

  const [nameTouched, setNameTouched] = useState(false);

  const isDuplicateName =
    name.trim() &&
    existingNames
      .map((n) => n.toLowerCase())
      .includes(name.trim().toLowerCase());

  const showNameError = nameTouched && !name.trim();

  const isValid =
    name.trim() &&
    !isDuplicateName &&
    (filterType === "none" ||
      (filterType === "tag" ? selectedTags.size > 0 : value.trim()));

  const handleCreate = () => {
    if (!name.trim()) {
      setNameTouched(true);
      return;
    }
    if (!isValid) return;
    const folder = {
      id: initialFolder?.id ?? crypto.randomUUID(),
      createdAt: initialFolder?.createdAt ?? new Date().toISOString(),
      name: name.trim(),
      filterType,
      live: filterType === "urls" ? false : live,
      topK: filterType === "search" ? topK : undefined,
      searchQuery: filterType === "search" ? value.trim() : "",
      tagFilter: filterType === "tag" ? [...selectedTags] : [],
      tagIntersect: filterType === "tag" ? tagIntersect : false,
      urls:
        filterType === "urls"
          ? value
              .trim()
              .split("\n")
              .map((u) => u.trim())
              .filter(Boolean)
          : [],
    };
    onCreate(folder);
  };

  return (
    <div className="finder-modal-overlay" onClick={onClose}>
      <div className="finder-modal" onClick={(e) => e.stopPropagation()}>
        <div className="finder-modal-header">
          {isEditing ? "Edit Folder" : "New Folder"}
        </div>
        <div className="finder-modal-body">
          <div>
            <div className="finder-modal-label">
              Name <span className="finder-modal-required">*</span>
            </div>
            <input
              className={`finder-modal-input${showNameError ? " finder-modal-input--error" : ""}`}
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (nameTouched && e.target.value.trim()) setNameTouched(false);
              }}
              onBlur={() => setNameTouched(true)}
              placeholder="Give this folder a name…"
              autoFocus
            />
            {showNameError && (
              <div className="finder-modal-error">Name is required.</div>
            )}
            {isDuplicateName && (
              <div className="finder-modal-error">
                A folder with this name already exists.
              </div>
            )}
          </div>
          <div>
            <div className="finder-modal-label">Type</div>
            <div className="finder-modal-tabs">
              {[
                ["none", "Empty"],
                ["search", "Search"],
                ["tag", "Tag Filter"],
                ["urls", "Manual URLs"],
              ].map(([t, l]) => (
                <button
                  key={t}
                  className={`finder-modal-tab ${filterType === t ? "active" : ""}`}
                  onClick={() => {
                    setFilterType(t);
                    setValue("");
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          {filterType === "search" && (
            <>
              <div>
                <div className="finder-modal-label">Query</div>
                <input
                  className="finder-modal-input"
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  placeholder="e.g. machine learning"
                />
              </div>
              <div>
                <div className="finder-modal-label">Max results</div>
                <input
                  className="finder-modal-input"
                  type="number"
                  min={5}
                  max={500}
                  step={5}
                  value={topK}
                  onChange={(e) =>
                    setTopK(
                      Math.max(
                        5,
                        Math.min(500, parseInt(e.target.value) || 50),
                      ),
                    )
                  }
                  style={{ width: 90 }}
                />
              </div>
            </>
          )}

          {filterType === "tag" && (
            <div>
              <div className="finder-modal-label">
                Tags
                {selectedTags.size > 0 && (
                  <span className="finder-modal-tag-count">
                    {selectedTags.size} selected
                  </span>
                )}
              </div>
              {selectedTags.size > 0 && (
                <div className="finder-modal-tag-chips">
                  {[...selectedTags].map((t) => (
                    <span
                      key={t}
                      className="finder-modal-tag-chip"
                      onClick={() => toggleTag(t)}
                    >
                      {t} ×
                    </span>
                  ))}
                </div>
              )}
              <input
                className="finder-modal-input"
                value={tagSearch}
                onChange={(e) => setTagSearch(e.target.value)}
                placeholder="Search tags…"
                style={{ marginBottom: 6 }}
              />
              <div className="finder-modal-tag-list">
                {allTags.length === 0 ? (
                  <div className="finder-modal-tag-loading">Loading…</div>
                ) : filteredTags.length === 0 ? (
                  <div className="finder-modal-tag-loading">No matches</div>
                ) : (
                  filteredTags.map((t) => (
                    <div
                      key={t}
                      className={`finder-modal-tag-item${selectedTags.has(t) ? " selected" : ""}`}
                      onClick={() => toggleTag(t)}
                    >
                      {t}
                    </div>
                  ))
                )}
              </div>
              <button
                type="button"
                className={`finder-modal-intersect-btn${tagIntersect ? " active" : ""}`}
                onClick={() => setTagIntersect((v) => !v)}
              >
                <span className="finder-modal-intersect-icon">⋂</span>
                Intersection — all tags must match
              </button>
            </div>
          )}

          {filterType === "urls" && (
            <div>
              <div className="finder-modal-label">URLs (one per line)</div>
              <textarea
                className="finder-modal-input"
                style={{ height: 60, resize: "vertical", padding: "8px 12px" }}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="https://..."
              />
            </div>
          )}

          {filterType !== "urls" && filterType !== "none" && (
            <div className="finder-modal-toggle-row">
              <div className="finder-modal-toggle-info">
                <span
                  className="finder-modal-label"
                  style={{ marginBottom: 0 }}
                >
                  Auto-update
                </span>
                <span className="finder-modal-toggle-hint">
                  {live
                    ? "Re-query every time the folder is opened"
                    : "Snapshot taken once at creation"}
                </span>
              </div>
              <button
                className={`finder-modal-toggle${live ? " on" : ""}`}
                onClick={() => setLive((v) => !v)}
                type="button"
                aria-pressed={live}
              >
                <span className="finder-modal-toggle-thumb" />
              </button>
            </div>
          )}
        </div>
        <div className="finder-modal-footer">
          <button
            className="finder-modal-btn finder-modal-btn--cancel"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="finder-modal-btn finder-modal-btn--save"
            disabled={!isValid}
            onClick={handleCreate}
          >
            {isEditing ? "Save" : "Create"}
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * FinderBrowser — Apple Finder columns-view browser for the right panel.
 * Column 0: folder tree root + sources + custom folders
 * Column 1+: sub-categories / tags of the selected item
 * Docs column: documents in the selected tag / source / custom folder
 */
const COL_DEFAULT_WIDTH = 220;
const COL_MIN_WIDTH = 100;
const COL_MAX_WIDTH = 480;

const FinderBrowser = ({ sources, sourceKeys }) => {
  // columnStack[i] = { items: [...], selectedIdx: number|null }
  const [columnStack, setColumnStack] = useState([]);
  // docsColumn = null | { items: [...], loading: bool }
  const [docsColumn, setDocsColumn] = useState(null);
  const [filterQuery, setFilterQuery] = useState("");
  const [customFolders, setCustomFolders] = useState(loadCustomFolders);
  // null = closed; "" = open at root; "uuid" = open under that folder
  const [showCreateModal, setShowCreateModal] = useState(null);
  const [columnWidths, setColumnWidths] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("finder-col-widths")) || {};
    } catch {
      return {};
    }
  });
  const [showEditModal, setShowEditModal] = useState(null); // folder object | null
  const columnsRef = useRef(null);
  const dragRef = useRef(null);

  const startResize = useCallback((e, colIdx) => {
    e.preventDefault();
    e.stopPropagation();
    const colEl = e.currentTarget.parentElement;
    dragRef.current = {
      colIdx,
      startX: e.clientX,
      startWidth: colEl.offsetWidth,
    };
    e.currentTarget.classList.add("dragging");
    const handle = e.currentTarget;

    const onMouseMove = (ev) => {
      if (!dragRef.current) return;
      const { colIdx, startX, startWidth } = dragRef.current;
      const w = Math.max(
        COL_MIN_WIDTH,
        Math.min(COL_MAX_WIDTH, startWidth + ev.clientX - startX),
      );
      setColumnWidths((prev) => {
        const next = { ...prev, [colIdx]: w };
        localStorage.setItem("finder-col-widths", JSON.stringify(next));
        return next;
      });
    };

    const onMouseUp = () => {
      dragRef.current = null;
      handle.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  // Rebuild all columns when sources / custom folders change
  useEffect(() => {
    const rootItems = [
      ...sources.map((src) => ({
        kind: "source",
        key: src.key,
        label: src.label,
      })),
      ...customFolders.map(folderToItem),
    ];
    setColumnStack((prev) => {
      if (prev.length === 0)
        return [{ items: rootItems, selectedIdx: null, parentFolderId: null }];
      return prev.map((col) => {
        if (col.parentFolderId === null) return { ...col, items: rootItems };
        const parent = findFolderById(customFolders, col.parentFolderId);
        return {
          ...col,
          items: parent ? (parent.children || []).map(folderToItem) : [],
        };
      });
    });
  }, [sources, customFolders]);

  const handleCreateFolder = useCallback(async (folder, parentId) => {
    const targetParentId = parentId ?? null;
    setShowCreateModal(null);
    let finalFolder = folder;

    // Take a snapshot now if the folder is not live
    if (
      !folder.live &&
      folder.filterType !== "urls" &&
      folder.filterType !== "none"
    ) {
      try {
        let docs = [];
        if (folder.filterType === "search") {
          docs = await apiSearch(
            folder.searchQuery,
            false,
            null,
            null,
            folder.topK || 50,
          );
        } else if (folder.filterType === "tag") {
          const tagList = Array.isArray(folder.tagFilter)
            ? folder.tagFilter
            : folder.tagFilter
              ? [folder.tagFilter]
              : [];
          if (tagList.length > 0) {
            const clauses = tagList.map(
              () => "(tags LIKE ? OR extra_tags LIKE ?)",
            );
            const resp = await fetch(
              `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  condition: clauses.join(
                    folder.tagIntersect ? " AND " : " OR ",
                  ),
                  parameters: tagList.flatMap((t) => [`%${t}%`, `%${t}%`]),
                }),
              },
            );
            const data = await resp.json();
            docs = (data.metadata || []).map(transformMeta);
          }
        }
        finalFolder = { ...folder, urls: docs.map((d) => d.url) };
      } catch {
        // snapshot failed — fall back to live
        finalFolder = { ...folder, live: true };
      }
    }

    setCustomFolders((prev) => {
      const next = addFolderToTree(prev, targetParentId, finalFolder);
      saveCustomFolders(next);
      return next;
    });
  }, []);

  const handleEditFolder = useCallback(async (folder) => {
    setShowEditModal(null);
    let finalFolder = folder;

    if (
      !folder.live &&
      folder.filterType !== "urls" &&
      folder.filterType !== "none"
    ) {
      try {
        let docs = [];
        if (folder.filterType === "search") {
          docs = await apiSearch(
            folder.searchQuery,
            false,
            null,
            null,
            folder.topK || 50,
          );
        } else if (folder.filterType === "tag") {
          const tagList = Array.isArray(folder.tagFilter)
            ? folder.tagFilter
            : folder.tagFilter
              ? [folder.tagFilter]
              : [];
          if (tagList.length > 0) {
            const clauses = tagList.map(
              () => "(tags LIKE ? OR extra_tags LIKE ?)",
            );
            const resp = await fetch(
              `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  condition: clauses.join(
                    folder.tagIntersect ? " AND " : " OR ",
                  ),
                  parameters: tagList.flatMap((t) => [`%${t}%`, `%${t}%`]),
                }),
              },
            );
            const data = await resp.json();
            docs = (data.metadata || []).map(transformMeta);
          }
        }
        finalFolder = { ...folder, urls: docs.map((d) => d.url) };
      } catch {
        finalFolder = { ...folder, live: true };
      }
    }

    setCustomFolders((prev) => {
      const next = updateFolderInTree(prev, finalFolder);
      saveCustomFolders(next);
      return next;
    });
    // Close docs if this folder was selected so it reloads on next click
    setColumnStack((prev) => {
      const wasSelected = prev.some(
        (col) =>
          col.selectedIdx != null &&
          col.items[col.selectedIdx]?.id === finalFolder.id,
      );
      if (wasSelected) {
        setDocsColumn(null);
        return prev.length > 0
          ? [{ items: prev[0].items, selectedIdx: null }]
          : prev;
      }
      return prev;
    });
  }, []);

  const handleDeleteFolder = useCallback((id) => {
    setCustomFolders((prev) => {
      const next = removeFolderFromTree(prev, id);
      saveCustomFolders(next);
      return next;
    });
    // Collapse to root column if the deleted folder (or a parent of it) was open
    setDocsColumn(null);
    setColumnStack((prev) =>
      prev.length > 0
        ? [{ ...prev[0], selectedIdx: null, parentFolderId: null }]
        : prev,
    );
  }, []);

  // Load documents for a selected item (source or custom folder)
  const loadDocs = useCallback(async (item) => {
    if (item.kind === "source") {
      setDocsColumn({ items: [], loading: true });
      try {
        const docs = await apiLatest(FINDER_DOC_LIMIT, new Set([item.key]));
        setDocsColumn({ items: docs, loading: false });
      } catch {
        setDocsColumn({ items: [], loading: false });
      }
    } else if (item.kind === "custom") {
      const folder = item.folderData;
      if (folder.filterType === "none") {
        // Empty container folder — no docs to load
        setDocsColumn(null);
        return;
      }
      setDocsColumn({ items: [], loading: true });
      try {
        let docs = [];
        const useStoredUrls =
          folder.live === false || folder.filterType === "urls";
        if (useStoredUrls && folder.urls && folder.urls.length > 0) {
          const placeholders = folder.urls.map(() => "?").join(", ");
          const resp = await fetch(
            `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                condition: `url IN (${placeholders})`,
                parameters: folder.urls,
              }),
            },
          );
          const data = await resp.json();
          docs = (data.metadata || []).map(transformMeta);
        } else if (folder.filterType === "search") {
          docs = await apiSearch(
            folder.searchQuery,
            false,
            null,
            null,
            folder.topK || 50,
          );
        } else if (folder.filterType === "tag") {
          const tagList = Array.isArray(folder.tagFilter)
            ? folder.tagFilter
            : folder.tagFilter
              ? [folder.tagFilter]
              : [];
          if (tagList.length > 0) {
            const clauses = tagList.map(
              () => "(tags LIKE ? OR extra_tags LIKE ?)",
            );
            const resp = await fetch(
              `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  condition: clauses.join(
                    folder.tagIntersect ? " AND " : " OR ",
                  ),
                  parameters: tagList.flatMap((t) => [`%${t}%`, `%${t}%`]),
                }),
              },
            );
            const data = await resp.json();
            docs = (data.metadata || []).map(transformMeta);
            if (folder.tagIntersect) {
              docs.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
            } else {
              // OR mode: sort by number of matching tags (most matches first)
              docs.sort((a, b) => {
                const diff =
                  countTagMatches(b, tagList) - countTagMatches(a, tagList);
                return diff !== 0
                  ? diff
                  : (b.date || "").localeCompare(a.date || "");
              });
            }
          }
        }
        setDocsColumn({
          items: docs.slice(0, FINDER_DOC_LIMIT),
          loading: false,
        });
      } catch {
        setDocsColumn({ items: [], loading: false });
      }
    }
  }, []);

  // Handle item selection: mark selected, truncate deeper columns, load docs.
  // Clicking an already-selected item deselects it and closes the docs column.
  const selectItem = useCallback(
    (colIdx, itemIdx, item) => {
      const isDeselecting = columnStack[colIdx]?.selectedIdx === itemIdx;
      setFilterQuery("");
      setDocsColumn(null);

      if (isDeselecting) {
        setColumnStack((prev) =>
          prev
            .slice(0, colIdx + 1)
            .map((c, i) => (i === colIdx ? { ...c, selectedIdx: null } : c)),
        );
        return;
      }

      const base = (prev) =>
        prev
          .slice(0, colIdx + 1)
          .map((c, i) => (i === colIdx ? { ...c, selectedIdx: itemIdx } : c));

      if (item.kind === "custom") {
        const children = item.folderData.children || [];
        if (children.length > 0) {
          // Push sub-column; "New Subfolder" lives there
          setColumnStack((prev) => [
            ...base(prev),
            {
              items: children.map(folderToItem),
              selectedIdx: null,
              parentFolderId: item.id,
            },
          ]);
          if (
            item.folderData.filterType &&
            item.folderData.filterType !== "none"
          ) {
            loadDocs(item);
          } else {
            setDocsColumn(null);
          }
        } else {
          // No children: stay in the same column layout, load docs.
          // "New Subfolder" will appear inside the docs column.
          setColumnStack(base);
          if (
            item.folderData.filterType &&
            item.folderData.filterType !== "none"
          ) {
            loadDocs(item);
          } else {
            setDocsColumn({ items: [], loading: false });
          }
        }
        return;
      }

      setColumnStack(base);
      loadDocs(item);
    },
    [columnStack, loadDocs],
  );

  // Scroll to reveal the rightmost column when navigation changes
  useEffect(() => {
    if (columnsRef.current) {
      columnsRef.current.scrollLeft = columnsRef.current.scrollWidth;
    }
  }, [columnStack.length, docsColumn]);

  // The deepest selected custom folder with no children — used to show
  // "New Subfolder" inside the docs column instead of a separate empty column.
  const leafCustomFolder = (() => {
    for (let i = columnStack.length - 1; i >= 0; i--) {
      const col = columnStack[i];
      if (col.selectedIdx == null) continue;
      const item = col.items[col.selectedIdx];
      if (
        item?.kind === "custom" &&
        (item.folderData.children || []).length === 0
      )
        return item;
      break;
    }
    return null;
  })();

  const fq = filterQuery.toLowerCase();
  const filterItems = (items) =>
    fq
      ? items.filter((item) => (item.label || "").toLowerCase().includes(fq))
      : items;
  const filterDocs = (docs) =>
    fq ? docs.filter((d) => (d.title || "").toLowerCase().includes(fq)) : docs;

  return (
    <div className="finder">
      {/* Toolbar: filter input */}
      <div className="finder-toolbar">
        <div className="finder-search" style={{ flex: 1, width: "auto" }}>
          <span className="finder-search-icon">
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </span>
          <input
            className="finder-search-input"
            type="text"
            placeholder="Filter..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Columns container */}
      <div className="finder-columns" ref={columnsRef}>
        {/* Navigable columns */}
        {columnStack.map((col, colIdx) => (
          <div
            key={colIdx}
            className="finder-column"
            style={{ width: columnWidths[colIdx] ?? COL_DEFAULT_WIDTH }}
          >
            {filterItems(col.items).map((item) => {
              const origIdx = col.items.indexOf(item);
              const isSelected = col.selectedIdx === origIdx;
              const hasChildren =
                item.kind === "custom" &&
                (item.folderData.children || []).length > 0;
              return (
                <div
                  key={
                    item.kind === "source" ? `s-${item.key}` : `c-${item.id}`
                  }
                  className={`finder-row${isSelected ? " finder-row--selected" : ""}`}
                  onClick={() => selectItem(colIdx, origIdx, item)}
                >
                  <span className="finder-row-icon" style={{ fontSize: 13 }}>
                    {item.kind === "source"
                      ? getFinderSourceIcon(item.key)
                      : item.filterType === "search"
                        ? "\uD83D\uDD0D"
                        : item.filterType === "tag"
                          ? "\uD83C\uDFF7\uFE0F"
                          : item.filterType === "urls"
                            ? "\uD83D\uDD17"
                            : "\uD83D\uDCC1"}
                  </span>
                  <span className="finder-row-label">{item.label}</span>
                  {item.kind === "custom" && (
                    <>
                      {hasChildren && (
                        <span className="finder-row-chevron">›</span>
                      )}
                      <button
                        className="finder-row-edit"
                        title="Edit folder"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowEditModal(item.folderData);
                        }}
                      >
                        ✎
                      </button>
                      <button
                        className="finder-row-delete"
                        title="Delete folder"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteFolder(item.id);
                        }}
                      >
                        {"\u00D7"}
                      </button>
                    </>
                  )}
                </div>
              );
            })}
            {/* "New Folder" button — always shown, uses column's parentFolderId */}
            {col.parentFolderId !== undefined && (
              <button
                className="finder-add-row"
                onClick={() => setShowCreateModal(col.parentFolderId ?? "")}
              >
                <span className="finder-add-row-icon">+</span>
                <span className="finder-add-row-label">
                  {col.parentFolderId ? "New Subfolder" : "New Folder"}
                </span>
              </button>
            )}
            <div
              className="finder-col-resize-handle"
              onMouseDown={(e) => startResize(e, colIdx)}
            />
          </div>
        ))}

        {/* Documents column */}
        {docsColumn && (
          <div className="finder-column finder-column--docs">
            {leafCustomFolder && (
              <button
                className="finder-add-row finder-add-row--inline"
                onClick={() => setShowCreateModal(leafCustomFolder.id)}
              >
                <span className="finder-add-row-icon">+</span>
                <span className="finder-add-row-label">New Subfolder</span>
              </button>
            )}
            {docsColumn.loading ? (
              <div className="finder-loading">
                <span className="finder-loading-dot"></span>
                Loading...
              </div>
            ) : filterDocs(docsColumn.items).length === 0 ? (
              <div className="finder-empty">No documents</div>
            ) : (
              filterDocs(docsColumn.items).map((doc, i) => (
                <a
                  key={doc.url || i}
                  className="finder-row finder-row--doc"
                  href={doc.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className="finder-row-icon" style={{ fontSize: 13 }}>
                    {doc.url && doc.url.includes("github.com")
                      ? "\uD83D\uDCBB"
                      : doc.url &&
                          (doc.url.includes("twitter.com") ||
                            doc.url.includes("x.com"))
                        ? "\uD835\uDD4F"
                        : doc.url && doc.url.includes("arxiv.org")
                          ? "\uD83D\uDCDD"
                          : "\uD83D\uDCC4"}
                  </span>
                  <span className="finder-row-label">{doc.title}</span>
                  <span className="finder-row-meta">{doc.date}</span>
                </a>
              ))
            )}
          </div>
        )}
      </div>

      {/* Create folder modal */}
      {showCreateModal !== null && (
        <CreateFolderModal
          onClose={() => setShowCreateModal(null)}
          onCreate={(folder) =>
            handleCreateFolder(folder, showCreateModal || null)
          }
          existingNames={getSiblingNames(
            customFolders,
            showCreateModal || null,
          )}
        />
      )}

      {/* Edit folder modal */}
      {showEditModal && (
        <CreateFolderModal
          onClose={() => setShowEditModal(null)}
          onCreate={handleEditFolder}
          initialFolder={showEditModal}
          existingNames={getSiblingNames(
            removeFolderFromTree(customFolders, showEditModal.id),
            findParentId(customFolders, showEditModal.id) ?? null,
          )}
        />
      )}
    </div>
  );
};

/**
 * Pipeline progress bar — overlays on top of the folder view.
 */
function parsePipelineStep(output) {
  if (!output)
    return { label: "Starting", detail: "Initializing sources", pct: 1 };
  var last = { label: "Starting", detail: "Initializing sources", pct: 1 };
  var lines = output.split("\n");
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var start = line.indexOf("@@");
    var end = line.lastIndexOf("@@");
    if (start !== -1 && end > start + 2) {
      var inner = line.substring(start + 2, end);
      var parts = inner.split("|");
      if (parts.length >= 2) {
        last = {
          pct: parseInt(parts[0], 10) || 0,
          label: parts[1],
          detail: parts[2] || "",
        };
      }
    }
  }
  return last;
}

var PipelineBar = function (props) {
  var data = props.data;
  var onClose = props.onClose;
  var isRunning = data && data.status === "running";
  var lastRun = data && data.last_run;

  var output = isRunning
    ? (data && data.output) || ""
    : (lastRun && lastRun.output) || "";
  var parsed = parsePipelineStep(output);
  var pct = isRunning ? parsed.pct : lastRun ? 100 : 0;
  var label = isRunning
    ? parsed.label
    : lastRun
      ? lastRun.success
        ? "Sources updated"
        : "Update failed"
      : "Starting";
  var detail = isRunning
    ? parsed.detail
    : lastRun
      ? parsePipelineStep(lastRun.output).detail
      : "";
  var isDone = !isRunning && lastRun;
  var isFailed = isDone && !lastRun.success;

  return React.createElement(
    "div",
    {
      className:
        "pipe-bar" +
        (isDone ? (isFailed ? " pipe-bar-fail" : " pipe-bar-done") : ""),
    },
    React.createElement(
      "div",
      { className: "pipe-bar-inner" },
      React.createElement(
        "div",
        { className: "pipe-bar-info" },
        React.createElement(
          "div",
          { className: "pipe-bar-status" },
          React.createElement(
            "span",
            { className: "pipe-bar-label" },
            isRunning
              ? React.createElement("span", { className: "pipe-bar-dot" })
              : isDone
                ? React.createElement(
                    "span",
                    { className: "pipe-bar-icon" },
                    isFailed ? "\u2717" : "\u2713",
                  )
                : null,
            " ",
            label,
          ),
          detail
            ? React.createElement(
                "span",
                { className: "pipe-bar-detail" },
                detail,
              )
            : null,
        ),
        isRunning
          ? React.createElement(
              "span",
              { className: "pipe-bar-pct" },
              pct + "%",
            )
          : null,
        React.createElement(
          "button",
          {
            className: "pipe-bar-close",
            onClick: onClose,
            title: "Dismiss",
          },
          "\u00d7",
        ),
      ),
      React.createElement(
        "div",
        { className: "pipe-bar-track" },
        React.createElement("div", {
          className:
            "pipe-bar-fill" + (isRunning ? " pipe-bar-fill-active" : ""),
          style: { width: pct + "%" },
        }),
      ),
    ),
  );
};

/**
 * The main Search component.
 */
const Search = () => {
  // --- State Management ---
  const [query, setQuery] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [modelStatus, setModelStatus] = useState("Loading Script...");
  const [documents, setDocuments] = useState([]);
  const [isSortedByDate, setIsSortedByDate] = useState(false);
  const [resultsReranked, setResultsReranked] = useState(false);
  const [sourceFilter, setSourceFilter] = useState(new Set());
  const [sources, setSources] = useState([]);
  const [favorites, setFavorites] = useState(new Set());
  const [showFavorites, setShowFavorites] = useState(false);
  const [theme, setTheme] = useState(
    document.documentElement.getAttribute("data-theme") || "dark",
  );
  const [pipelineData, setPipelineData] = useState(null);
  const [pipelineOpen, setPipelineOpen] = useState(false);
  const [similarMap, setSimilarMap] = useState(new Map());
  const [tagFilters, setTagFilters] = useState(new Set());
  const [showFinder, setShowFinder] = useState(
    () => localStorage.getItem("finder-visible") !== "false",
  );
  const [displayedCount, setDisplayedCount] = useState(DISPLAY_COUNT);

  useEffect(() => {
    const panel = document.getElementById("folder-panel");
    const searchbox = document.getElementById("searchbox");
    if (panel) panel.classList.toggle("finder-panel--hidden", !showFinder);
    if (searchbox) searchbox.classList.toggle("finder-expanded", !showFinder);
    localStorage.setItem("finder-visible", String(showFinder));
  }, [showFinder]);

  // --- Refs ---
  const searchTimerRef = useRef(null);
  const workerRef = useRef(null);
  const latestQueryIdRef = useRef(0);
  const rerankTimerRef = useRef(null);
  const immediateRerankRef = useRef(false);
  const settleTimerRef = useRef(null);
  const lastSearchMeta = useRef(null);
  const pipelineIntervalRef = useRef(null);
  const similarMapRef = useRef(new Map());
  const sentinelRef = useRef(null);
  const topKRef = useRef(FETCH_COUNT);
  const isFetchingMoreRef = useRef(false);
  const documentsRef = useRef([]);

  // Keep documentsRef in sync for use inside async callbacks
  useEffect(() => {
    documentsRef.current = documents;
  }, [documents]);

  // --- Function Declarations ---

  /**
   * Fetches the latest documents from the backend.
   */
  const fetchLatest = useCallback(() => {
    setDisplayedCount(DISPLAY_COUNT);
    topKRef.current = FETCH_COUNT;
    apiLatest(FETCH_COUNT, sourceFilter, tagFilters)
      .then((docs) => setDocuments(docs))
      .catch((error) =>
        console.error("[APP] Failed to fetch latest documents:", error),
      );
  }, [sourceFilter, tagFilters]);

  /**
   * Fetches search results from the backend API.
   */
  const search = useCallback(
    (searchQuery, sortChronologically = false) => {
      if (!searchQuery.trim()) {
        fetchLatest();
        return;
      }
      const queryId = ++latestQueryIdRef.current;
      setDisplayedCount(DISPLAY_COUNT);
      topKRef.current = FETCH_COUNT;
      // Append tag filter to the query string if a node is selected
      const fullQuery = selectedNode
        ? `${searchQuery} ${selectedNode}`
        : searchQuery;

      const t0 = performance.now();
      apiSearch(fullQuery, sortChronologically, sourceFilter, tagFilters)
        .then((docs) => {
          const latencyMs = Math.round(performance.now() - t0);
          lastSearchMeta.current = { resultCount: docs.length, latencyMs };
          setDocuments(docs);
          setResultsReranked(false);
        })
        .catch((error) =>
          console.error(`[APP] Failed to fetch search results:`, error),
        );
    },
    [selectedNode, fetchLatest, sourceFilter, tagFilters],
  );

  /**
   * Resets the 2s settle timer. When the user stops typing for 2s,
   * logs a `search` analytics event with the settled query metadata.
   */
  const resetSettleTimer = useCallback(
    (searchQuery) => {
      clearTimeout(settleTimerRef.current);
      settleTimerRef.current = setTimeout(() => {
        const meta = lastSearchMeta.current;
        if (meta && searchQuery.trim()) {
          trackEvent("search", {
            query: searchQuery,
            result_count: meta.resultCount,
            latency_ms: meta.latencyMs,
            source_filter:
              sourceFilter.size > 0 ? [...sourceFilter].join(",") : "all",
            sort_mode: isSortedByDate ? "date" : "relevance",
          });
        }
      }, SEARCH_SETTLE_MS);
    },
    [sourceFilter, isSortedByDate],
  );

  /**
   * Runs search immediately, clearing any pending debounced calls.
   */
  const runNow = useCallback(
    (runQuery, sortChronologically = true) => {
      setResultsReranked(false);
      clearTimeout(rerankTimerRef.current);
      clearTimeout(searchTimerRef.current);
      search(runQuery, sortChronologically);
    },
    [search],
  );

  // --- Effects ---

  /**
   * Load folder tree on mount.
   */
  useEffect(() => {
    fetch(`${DATA_API_URL}/api/sources`)
      .then((res) => res.json())
      .then((data) => setSources(data))
      .catch(() => {
        fetch("data/sources.json")
          .then((res) => res.json())
          .then((data) => setSources(data))
          .catch((error) =>
            console.error("[APP] Failed to load sources:", error),
          );
      });
    fetch(`${DATA_API_URL}/api/favorites`)
      .then((res) => res.json())
      .then((urls) => setFavorites(new Set(urls)))
      .catch((error) =>
        console.error("[APP] Failed to load favorites:", error),
      );
  }, []);

  /**
   * Initializes the Web Worker on component mount.
   */
  useEffect(() => {
    const worker = new Worker("colbert.worker.js", { type: "module" });
    workerRef.current = worker;
    worker.postMessage({ type: "load" });

    worker.onmessage = (event) => {
      const { type, payload, queryId } = event.data;
      switch (type) {
        case "status":
          setModelStatus(payload);
          break;
        case "model-ready":
          setModelStatus("Model Ready");
          break;
        case "rank-update":
        case "rank-complete":
          if (queryId === latestQueryIdRef.current) {
            setDocuments(payload);
            setResultsReranked(true);
          }
          break;
        case "rank-similar-update":
        case "rank-similar-complete": {
          const simUrl = payload.sourceUrl;
          if (similarMapRef.current.has(simUrl)) {
            similarMapRef.current.set(simUrl, {
              loading: false,
              results: payload.results,
            });
            setSimilarMap(new Map(similarMapRef.current));
          }
          break;
        }
        case "error":
          setModelStatus(payload);
          console.error("[APP] Received error from worker:", payload);
          break;
      }
    };

    return () => worker.terminate();
  }, []);

  /**
   * Effect to handle initial page load.
   */
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlQuery = params.get("query") || "";
    const urlNode = params.get("node") || null;
    const urlSource = params.get("source") || "";
    const urlTags = params.get("tags") || "";
    if (urlSource) {
      setSourceFilter(new Set(urlSource.split(",").filter(Boolean)));
    }
    if (urlTags) {
      setTagFilters(new Set(urlTags.split(",").filter(Boolean)));
    }
    if (urlQuery) {
      setQuery(urlQuery);
      if (urlNode) {
        setSelectedNode(urlNode);
        runNow(`${urlQuery} ${urlNode}`);
      } else {
        runNow(urlQuery, true);
      }
    } else {
      fetchLatest();
    }
  }, []);

  /**
   * Re-fetch when source filter changes.
   */
  useEffect(() => {
    clearTimeout(rerankTimerRef.current);
    immediateRerankRef.current = true;
    if (query.trim()) {
      search(query, isSortedByDate);
    } else {
      fetchLatest();
    }
  }, [sourceFilter]);

  /**
   * Re-fetch when tag filters change.
   */
  useEffect(() => {
    syncURL({ query, source: sourceFilter, tags: tagFilters });
    clearTimeout(rerankTimerRef.current);
    immediateRerankRef.current = true;
    if (query.trim()) {
      search(query, isSortedByDate);
    } else {
      fetchLatest();
    }
  }, [tagFilters]);

  /**
   * Effect to handle window resizing for mobile detection.
   */
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  /**
   * Effect to automatically re-rank results after inactivity.
   */
  useEffect(() => {
    clearTimeout(rerankTimerRef.current);

    const shouldAttemptRerank =
      modelStatus === "Model Ready" &&
      !resultsReranked &&
      documents.length > 0 &&
      query.trim() &&
      !isSortedByDate &&
      tagFilters.size === 0;

    if (shouldAttemptRerank) {
      const delay = immediateRerankRef.current ? 0 : RERANK_INACTIVITY_MS;
      immediateRerankRef.current = false;
      rerankTimerRef.current = setTimeout(() => {
        workerRef.current.postMessage({
          type: "rank",
          payload: { query, documents, queryId: latestQueryIdRef.current },
        });
      }, delay);
    }

    return () => clearTimeout(rerankTimerRef.current);
  }, [
    documents,
    query,
    modelStatus,
    resultsReranked,
    isSortedByDate,
    tagFilters,
  ]);

  // --- Infinite scroll ---

  const fetchMoreCandidates = useCallback(async () => {
    if (isFetchingMoreRef.current) return;
    const newTopK = Math.min(topKRef.current + FETCH_COUNT, MAX_CANDIDATES);
    if (newTopK <= topKRef.current) return;
    isFetchingMoreRef.current = true;
    try {
      const more = query.trim()
        ? await apiSearch(
            query,
            isSortedByDate,
            sourceFilter,
            tagFilters,
            newTopK,
          )
        : await apiLatest(newTopK, sourceFilter, tagFilters);
      topKRef.current = newTopK;
      setDocuments((prev) => {
        const seen = new Set(prev.map((d) => d.url));
        const fresh = more.filter((d) => !seen.has(d.url));
        return fresh.length > 0 ? [...prev, ...fresh] : prev;
      });
    } finally {
      isFetchingMoreRef.current = false;
    }
  }, [query, isSortedByDate, sourceFilter, tagFilters]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    const canFetchMore =
      resultsReranked || tagFilters.size > 0 || !query.trim();
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setDisplayedCount((prev) => {
          const next = prev + DISPLAY_COUNT;
          if (next >= documentsRef.current.length && canFetchMore) {
            fetchMoreCandidates();
          }
          return next;
        });
      },
      { rootMargin: "200px" },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [resultsReranked, tagFilters, query, fetchMoreCandidates]);

  // --- Pipeline ---

  // On mount: check if pipeline is running, or auto-refresh if stale (> 3 min)
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/pipeline`)
      .then((r) => r.json())
      .then((data) => {
        setPipelineData(data);
        if (data.status === "running") {
          setPipelineOpen(true);
          return;
        }
        // Check last run time from DB to decide auto-refresh
        fetch(`${API_BASE_URL}/api/pipeline_run`)
          .then((r) => {
            if (r.ok) return r.json();
            throw new Error();
          })
          .then((run) => {
            if (run && run.finished_at) {
              var elapsed =
                (Date.now() - new Date(run.finished_at).getTime()) / 1000;
              if (elapsed > 180) {
                // Stale — auto-refresh
                fetch(`${API_BASE_URL}/api/pipeline`, { method: "POST" })
                  .then((r) => r.json())
                  .then((d) => {
                    if (
                      d.status === "started" ||
                      d.status === "already_running"
                    ) {
                      setPipelineData({
                        status: "running",
                        started_at: d.started_at,
                        output: "",
                        elapsed_secs: 0,
                      });
                      setPipelineOpen(true);
                    }
                  })
                  .catch(function () {});
              }
            }
          })
          .catch(function () {});
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!pipelineOpen || !(pipelineData && pipelineData.status === "running"))
      return;
    const poll = () =>
      fetch(`${API_BASE_URL}/api/pipeline`)
        .then((r) => r.json())
        .then((data) => setPipelineData(data))
        .catch(() => {});
    pipelineIntervalRef.current = setInterval(poll, 2000);
    return () => clearInterval(pipelineIntervalRef.current);
  }, [pipelineOpen, pipelineData && pipelineData.status]);

  // Auto-dismiss progress bar 6s after pipeline finishes
  useEffect(() => {
    if (!pipelineOpen) return;
    if (
      pipelineData &&
      pipelineData.status !== "running" &&
      pipelineData.last_run
    ) {
      var t = setTimeout(function () {
        setPipelineOpen(false);
      }, 6000);
      return function () {
        clearTimeout(t);
      };
    }
  }, [
    pipelineOpen,
    pipelineData && pipelineData.status,
    pipelineData && pipelineData.last_run,
  ]);

  const triggerPipeline = useCallback(() => {
    if (pipelineData && pipelineData.status === "running") {
      setPipelineOpen(true);
      return;
    }
    fetch(`${API_BASE_URL}/api/pipeline`, { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "started" || data.status === "already_running") {
          setPipelineData({
            status: "running",
            started_at: data.started_at,
            output: "",
            elapsed_secs: 0,
          });
          setPipelineOpen(true);
        }
      })
      .catch(() => {});
  }, [pipelineData && pipelineData.status]);

  // --- Event Handlers ---

  const handleChangeText = useCallback(
    (event) => {
      const newQuery = event.target.value.toLowerCase();
      setQuery(newQuery);
      clearTimeout(rerankTimerRef.current);
      clearTimeout(searchTimerRef.current);

      if (!newQuery.trim()) {
        // Reset all state when search is cleared
        setSelectedNode(null);
        setIsSortedByDate(false);
        setResultsReranked(false);
        syncURL({ source: sourceFilter, tags: tagFilters });
        fetchLatest();
        return;
      }

      setSelectedNode(null);
      setIsSortedByDate(false);
      setResultsReranked(false);
      syncURL({ query: newQuery, source: sourceFilter, tags: tagFilters });
      searchTimerRef.current = setTimeout(
        () => search(newQuery),
        SEARCH_DEBOUNCE_MS,
      );
      resetSettleTimer(newQuery);
    },
    [search, resetSettleTimer, sourceFilter, fetchLatest],
  );

  const handleClickDate = useCallback(() => {
    const newSortState = !isSortedByDate;
    setIsSortedByDate(newSortState);
    setResultsReranked(false);
    clearTimeout(rerankTimerRef.current);
    search(query, newSortState);
  }, [isSortedByDate, query, search]);

  const handleClickTag = useCallback((tag) => {
    setTagFilters((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      return next;
    });
    setResultsReranked(false);
  }, []);

  const updateSimilarMap = useCallback((fn) => {
    fn(similarMapRef.current);
    setSimilarMap(new Map(similarMapRef.current));
  }, []);

  const handleSimilar = useCallback(
    async (doc) => {
      const url = doc.url;
      if (similarMapRef.current.has(url)) {
        updateSimilarMap((m) => m.delete(url));
        return;
      }
      updateSimilarMap((m) => m.set(url, { loading: true, results: [] }));
      try {
        const results = await fetchSimilar(doc);
        if (!similarMapRef.current.has(url)) return;
        updateSimilarMap((m) => m.set(url, { loading: false, results }));
        trackEvent("find_similar", {
          doc_url: url,
          result_count: results.length,
        });
        // Dispatch to ColBERT worker for re-ranking against source doc content
        // Worker guards with if (!colbertModel) return, so always safe to post
        if (workerRef.current && results.length > 0) {
          const parts = [doc.title];
          const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
          if (allTags.length) parts.push(allTags.join(" "));
          if (doc.summary) parts.push(doc.summary);
          workerRef.current.postMessage({
            type: "rank-similar",
            payload: {
              sourceUrl: url,
              sourceText: parts.join(" "),
              results,
            },
          });
        }
      } catch (err) {
        console.error("[APP] Failed to fetch similar docs:", err);
        updateSimilarMap((m) => m.delete(url));
      }
    },
    [updateSimilarMap],
  );

  // --- UI Helper Functions ---

  const getDocumentSource = useCallback(
    (doc) => {
      const url = doc.url || "";
      const title = (doc.title || "").toLowerCase();
      const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
      const tagsLower = allTags.map((t) => (t || "").toLowerCase());

      // Check URL hostname against dynamic source keys
      try {
        const hostname = new URL(url).hostname.replace(/^www\./, "");
        for (const src of sources) {
          if (src.key === hostname || hostname.endsWith("." + src.key))
            return src.key;
          // Handle aliases: x.com -> twitter.com
          if (
            src.key === "twitter.com" &&
            (hostname === "x.com" || hostname.endsWith(".x.com"))
          )
            return src.key;
        }
      } catch {}

      // Tag/title-based detection for sources without distinct URLs
      for (const src of sources) {
        if (tagsLower.some((t) => t.includes(src.key.split(".")[0])))
          return src.key;
        if (title.includes(src.key.split(".")[0])) return src.key;
      }

      return "other";
    },
    [sources],
  );

  const getLinkLogo = (doc) => {
    const url = doc.url || "";
    const title = (doc.title || "").toLowerCase();
    const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
    const hasHackerNewsTag = allTags.some((tag) =>
      (tag || "").toLowerCase().includes("hackernews"),
    );

    if (url.includes("github.com")) {
      return <img src="icons/github.png" alt="GitHub Logo" />;
    }
    if (url.includes("twitter.com") || url.includes("x.com")) {
      return <img src="icons/twitter.png" alt="Twitter Logo" />;
    }
    if (title.includes("hackernews") || hasHackerNewsTag) {
      return <img src="icons/hackernews.png" alt="Hackernews Logo" />;
    }
    return "\uD83D\uDCC4";
  };

  const getIndicatorClass = (status) => {
    if (status === "Model Ready") return "status-ready";
    if (status.startsWith("Error")) return "status-error";
    return "status-loading";
  };

  const truncate = (text, limit) => {
    if (!text) return "";
    const tokens = text.trim().split(/\s+/);
    if (tokens.length <= limit) {
      return tokens.join(" ");
    }
    return tokens.slice(0, limit).join(" ") + "...";
  };

  const highlight = useCallback(
    (text) => {
      if (!text) return "";
      let keywords = query;
      if (selectedNode) keywords += ` ${selectedNode}`;
      const keywordSet = new Set(
        keywords
          .toLowerCase()
          .split(/\s+/)
          .filter((token) => token.length > 2),
      );
      if (keywordSet.size === 0) return <div className="inline">{text}</div>;

      const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const parts = text.split(
        new RegExp(`(${Array.from(keywordSet).map(escape).join("|")})`, "gi"),
      );
      return (
        <div className="inline">
          {parts.map((part, i) =>
            keywordSet.has(part.toLowerCase()) ? (
              <div className="highlight" key={i}>
                {part}
              </div>
            ) : (
              part
            ),
          )}
        </div>
      );
    },
    [query, selectedNode],
  );

  // --- Computed: filter by source + favorites then cap at displayedCount ---
  const displayedDocs = useMemo(() => {
    let filtered = (documents || []).filter(
      (doc) =>
        sourceFilter.size === 0 || sourceFilter.has(getDocumentSource(doc)),
    );
    if (showFavorites) {
      filtered = filtered.filter((doc) => favorites.has(doc.url));
    }
    return filtered.slice(0, displayedCount);
  }, [
    documents,
    sourceFilter,
    getDocumentSource,
    showFavorites,
    favorites,
    displayedCount,
  ]);

  // --- Render ---
  const folderPanel = document.getElementById("folder-panel");

  return (
    <React.Fragment>
      <div
        id="search-container"
        className={!showFinder ? "finder-expanded" : ""}
      >
        <input
          id="search"
          type="textarea"
          placeholder="Search"
          value={query}
          onChange={handleChangeText}
          autoFocus
        />
        <span
          className={`status-indicator ${getIndicatorClass(modelStatus)}`}
          title={modelStatus}
        ></span>
      </div>

      {!isMobile && (
        <button
          className={`finder-toggle${showFinder ? "" : " finder-toggle--closed"}`}
          onClick={() => setShowFinder((v) => !v)}
          title={showFinder ? "Hide sidebar" : "Show sidebar"}
        >
          <svg
            width="8"
            height="14"
            viewBox="0 0 8 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {showFinder ? (
              <polyline points="4,2 8,7 4,12" />
            ) : (
              <polyline points="6,2 2,7 6,12" />
            )}
          </svg>
        </button>
      )}

      <div id="controls-row">
        <div id="source-filter">
          {[{ key: "all", label: "All" }]
            .concat(sources)
            .concat([{ key: "other", label: "Other" }])
            .map(({ key, label }) => (
              <button
                key={key}
                className={`source-chip ${key === "all" ? (sourceFilter.size === 0 ? "active" : "") : sourceFilter.has(key) ? "active" : ""}`}
                onClick={() => {
                  let next;
                  if (key === "all") {
                    next = new Set();
                  } else {
                    next = new Set(sourceFilter);
                    if (next.has(key)) next.delete(key);
                    else next.add(key);
                  }
                  setSourceFilter(next);
                  syncURL({ query, source: next, tags: tagFilters });
                  trackEvent("filter_apply", {
                    source_key: next.size > 0 ? [...next].join(",") : "all",
                  });
                }}
              >
                {label}
              </button>
            ))}
        </div>
        <button
          className={`sort-date-toggle ${isSortedByDate ? "active" : ""}`}
          onClick={handleClickDate}
          title={isSortedByDate ? "Sort by relevance" : "Sort by date"}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
        </button>
        <button
          className="theme-toggle"
          onClick={() => {
            const next = theme === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            localStorage.setItem("theme", next);
            setTheme(next);
          }}
          aria-label="Toggle theme"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {theme === "dark" ? (
              <React.Fragment>
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </React.Fragment>
            ) : (
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            )}
          </svg>
        </button>
        <a
          className="github-star"
          href="https://github.com/raphaelsty/knowledge"
          target="_blank"
          rel="noopener noreferrer"
          title="Star on GitHub"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
          </svg>
        </a>
        <button
          className={
            "pipeline-toggle" +
            (pipelineData && pipelineData.status === "running" ? " active" : "")
          }
          onClick={() => {
            if (pipelineOpen) {
              setPipelineOpen(false);
            } else if (pipelineData && pipelineData.status === "running") {
              setPipelineOpen(true);
            } else {
              triggerPipeline();
            }
          }}
          title="Update sources"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={
              pipelineData && pipelineData.status === "running" ? "spin" : ""
            }
          >
            <path d="M21 2v6h-6" />
            <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
            <path d="M3 22v-6h6" />
            <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
          </svg>
        </button>
        <button
          className={`favorites-toggle ${showFavorites ? "active" : ""}`}
          onClick={() => setShowFavorites((v) => !v)}
          title={showFavorites ? "Show all" : "Show favorites"}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill={showFavorites ? "currentColor" : "none"}
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
          </svg>
          {favorites.size > 0 && (
            <span className="favorites-count">{favorites.size}</span>
          )}
        </button>
      </div>

      {tagFilters.size > 0 && (
        <div id="tag-filters-row">
          {[...tagFilters].map((tag) => (
            <div key={tag} className="tag-filter-chip">
              <span>{tag}</span>
              <button
                className="tag-filter-chip-remove"
                title={`Remove: ${tag}`}
                onClick={() =>
                  setTagFilters((prev) => {
                    const next = new Set(prev);
                    next.delete(tag);
                    return next;
                  })
                }
              >
                {"\u00D7"}
              </button>
            </div>
          ))}
          {tagFilters.size > 1 && (
            <button
              className="tag-filter-clear"
              onClick={() => setTagFilters(new Set())}
            >
              Clear all
            </button>
          )}
        </div>
      )}

      <div id="documents">
        {displayedDocs.map((doc, index) => (
          <React.Fragment key={doc.url || index}>
            <div
              className={`document${similarMap.has(doc.url) ? " document--similar-active" : ""}`}
            >
              <div className="title-wrapper">
                <span className="logo">{getLinkLogo(doc)}</span>
                <a
                  className="title"
                  href={doc.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => {
                    if (doc.url && !clickedUrls.has(doc.url)) {
                      clickedUrls.add(doc.url);
                      trackEvent("click", {
                        doc_url: doc.url,
                        position: index,
                        score: doc.colbertScore ?? doc.similarity ?? null,
                        query,
                      });
                    }
                  }}
                >
                  {highlight(doc.title)}
                </a>
              </div>
              <div className="date" onClick={handleClickDate}>
                {highlight(doc.date)}
              </div>
              <div className="summary">
                {highlight(truncate(doc.summary, SUMMARY_TOKEN_LIMIT))}
              </div>
              <div className="tags">
                {(doc.tags || [])
                  .concat(doc["extra-tags"] || [])
                  .map((tag, i) => (
                    <div
                      className={`tag${tagFilters.has(tag) ? " tag--active" : ""}`}
                      key={i}
                      onClick={() => handleClickTag(tag)}
                    >
                      {highlight(tag)}
                    </div>
                  ))}
                {typeof doc.colbertScore === "number" ? (
                  <span
                    className="score-badge reranker-score"
                    title="Re-ranker Score"
                  >
                    {doc.colbertScore.toFixed(3)}
                  </span>
                ) : typeof doc.similarity === "number" ? (
                  <span
                    className="score-badge retriever-score"
                    title="Retriever Score"
                  >
                    {doc.similarity.toFixed(3)}
                  </span>
                ) : null}
                <button
                  className={`similar-btn${similarMap.has(doc.url) ? " active" : ""}`}
                  title="Find similar documents"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSimilar(doc);
                  }}
                >
                  <svg
                    width="13"
                    height="13"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    <line x1="11" y1="8" x2="11" y2="14" />
                    <line x1="8" y1="11" x2="14" y2="11" />
                  </svg>
                  <span>Similar</span>
                </button>
                <button
                  className={`favorite-btn ${favorites.has(doc.url) ? "active" : ""}`}
                  title={
                    favorites.has(doc.url)
                      ? "Remove from favorites"
                      : "Add to favorites"
                  }
                  onClick={(e) => {
                    e.stopPropagation();
                    const url = doc.url;
                    setFavorites((prev) => {
                      const next = new Set(prev);
                      if (next.has(url)) next.delete(url);
                      else next.add(url);
                      return next;
                    });
                    fetch(`${DATA_API_URL}/api/favorites`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ url }),
                    }).catch(() => {
                      setFavorites((prev) => {
                        const rollback = new Set(prev);
                        if (rollback.has(url)) rollback.delete(url);
                        else rollback.add(url);
                        return rollback;
                      });
                    });
                  }}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill={favorites.has(doc.url) ? "currentColor" : "none"}
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                  </svg>
                </button>
              </div>
            </div>
            {similarMap.has(doc.url) &&
              (() => {
                const entry = similarMap.get(doc.url);
                return (
                  <div className="similar-results">
                    {entry.loading ? (
                      <div className="similar-loading">
                        <span className="similar-loading-dot"></span> Finding
                        similar...
                      </div>
                    ) : (
                      entry.results.map((sim, si) => (
                        <div
                          className="document similar-doc"
                          key={sim.url || si}
                          style={{ animationDelay: `${si * 0.04}s` }}
                        >
                          <div className="title-wrapper">
                            <span className="logo">{getLinkLogo(sim)}</span>
                            <a
                              className="title"
                              href={sim.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={() =>
                                trackEvent("click_similar", {
                                  source_url: doc.url,
                                  doc_url: sim.url,
                                  position: si,
                                  score: sim.similarity,
                                })
                              }
                            >
                              {highlight(sim.title)}
                            </a>
                          </div>
                          <div className="date">{highlight(sim.date)}</div>
                          <div className="summary">
                            {highlight(
                              truncate(sim.summary, SUMMARY_TOKEN_LIMIT),
                            )}
                          </div>
                          <div className="tags">
                            {(sim.tags || [])
                              .concat(sim["extra-tags"] || [])
                              .map((tag, ti) => (
                                <div
                                  className="tag"
                                  key={ti}
                                  onClick={() => handleClickTag(tag)}
                                >
                                  {highlight(tag)}
                                </div>
                              ))}
                            {typeof sim.colbertScore === "number" ? (
                              <span
                                className="score-badge reranker-score"
                                title="Re-ranker Score"
                              >
                                {sim.colbertScore.toFixed(3)}
                              </span>
                            ) : (
                              <span
                                className="score-badge retriever-score"
                                title="Similarity"
                              >
                                {sim.similarity.toFixed(3)}
                              </span>
                            )}
                            <button
                              className={`favorite-btn ${favorites.has(sim.url) ? "active" : ""}`}
                              title={
                                favorites.has(sim.url)
                                  ? "Remove from favorites"
                                  : "Add to favorites"
                              }
                              onClick={(e) => {
                                e.stopPropagation();
                                const url = sim.url;
                                setFavorites((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(url)) next.delete(url);
                                  else next.add(url);
                                  return next;
                                });
                                fetch(`${DATA_API_URL}/api/favorites`, {
                                  method: "POST",
                                  headers: {
                                    "Content-Type": "application/json",
                                  },
                                  body: JSON.stringify({ url }),
                                }).catch(() => {
                                  setFavorites((prev) => {
                                    const rollback = new Set(prev);
                                    if (rollback.has(url)) rollback.delete(url);
                                    else rollback.add(url);
                                    return rollback;
                                  });
                                });
                              }}
                            >
                              <svg
                                width="14"
                                height="14"
                                viewBox="0 0 24 24"
                                fill={
                                  favorites.has(sim.url)
                                    ? "currentColor"
                                    : "none"
                                }
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                              </svg>
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                );
              })()}
          </React.Fragment>
        ))}
        <div ref={sentinelRef} style={{ height: 1 }} />
      </div>

      {pipelineOpen && (
        <PipelineBar
          data={pipelineData}
          onClose={() => setPipelineOpen(false)}
        />
      )}

      {folderPanel &&
        !isMobile &&
        createPortal(
          <FinderBrowser
            sources={sources}
            sourceKeys={sources.map((s) => s.key)}
          />,
          folderPanel,
        )}
    </React.Fragment>
  );
};

// --- Mount Application ---
const container = document.getElementById("backsearch");
const root = createRoot(container);
root.render(<Search />);
