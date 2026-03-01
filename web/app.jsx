const { useState, useEffect, useRef, useCallback, useMemo } = React;
const { createRoot, createPortal } = ReactDOM;

const IS_LOCAL = window.location.hostname === "localhost";
const API_BASE_URL = IS_LOCAL ? "http://localhost:8080" : "";
const DATA_API_URL = IS_LOCAL ? "http://localhost:3001" : "";
const EVENTS_API_URL = IS_LOCAL ? "http://localhost:3002" : "";
const INDEX_NAME = "knowledge";
const SEARCH_DEBOUNCE_MS = 400;
const SEARCH_SETTLE_MS = 2000;
const FLUSH_INTERVAL_MS = 10000;
const MAX_BUFFER_SIZE = 50;
const FETCH_COUNT = 300;
const DISPLAY_COUNT = 30;
const RERANK_INACTIVITY_MS = 1000;
const SUMMARY_TOKEN_LIMIT = 30;

const syncURL = (params) => {
  const url = new URL(window.location);
  Object.entries(params).forEach(([k, v]) => {
    if (k === "source") {
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
const browsedFolders = new Set();

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
const getFilteredSubset = async (source) => {
  const cond = buildSourceCondition(source);
  if (!cond) return null;
  const resp = await fetch(
    `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cond),
    },
  );
  const data = await resp.json();
  return data.document_ids || null;
};

/**
 * Fetches the latest documents via metadata query.
 */
const apiLatest = async (count, source) => {
  const cond = buildSourceCondition(source);
  const body = cond
    ? {
        condition: `(${cond.condition}) AND date != ?`,
        parameters: [...cond.parameters, ""],
      }
    : { condition: "date IS NOT NULL AND date != ?", parameters: [""] };
  const resp = await fetch(
    `${API_BASE_URL}/indices/${INDEX_NAME}/metadata/get`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  const data = await resp.json();
  const rows = (data.metadata || []).map(transformMeta);
  rows.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  return rows.slice(0, count);
};

/**
 * Searches documents via ColBERT encoding and returns results.
 * When a source filter is active, restricts search to matching doc IDs.
 */
const apiSearch = async (query, sortByDate = false, source) => {
  const subset = await getFilteredSubset(source);
  const body = {
    queries: [query],
    params: { top_k: FETCH_COUNT },
  };
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
  let docs = metadata.map((meta, i) => ({
    ...transformMeta(meta),
    similarity: scores[i] || 0,
  }));
  if (sortByDate) {
    docs.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  }
  return docs;
};

/**
 * Recursively filters the tree to only include paths that contain matching tags.
 * Annotates each node with _bestRank (min document rank in subtree) for sorting.
 * Returns null if no match found in this subtree.
 */
const filterTree = (node, matchCounts, extraDocs, matchedUrls) => {
  let filteredChildren = [];
  let filteredTags = [];
  let totalMatch = 0;
  let bestRank = Infinity;

  if (node.c) {
    for (const child of node.c) {
      const fc = filterTree(child, matchCounts, extraDocs, matchedUrls);
      if (fc) {
        filteredChildren.push(fc);
        totalMatch += fc._matchCount || 0;
        if (fc._bestRank < bestRank) bestRank = fc._bestRank;
      }
    }
  }

  if (node.t) {
    for (const [tagName, tagCount, tagDocs] of node.t) {
      if (matchCounts[tagName]) {
        // Merge search result docs not already in tree tag
        let mergedDocs = tagDocs;
        const extra = extraDocs[tagName];
        if (extra && extra.length > 0) {
          const existing = new Set((tagDocs || []).map((d) => d.u));
          const toAdd = extra.filter((d) => !existing.has(d.u));
          if (toAdd.length > 0) {
            mergedDocs = [...(tagDocs || []), ...toAdd];
          }
        }
        // Compute best rank for this tag from its docs
        let tagBestRank = Infinity;
        for (const d of mergedDocs || []) {
          const rank = matchedUrls.get(d.u);
          if (rank !== undefined && rank < tagBestRank) tagBestRank = rank;
        }
        filteredTags.push([
          tagName,
          matchCounts[tagName],
          mergedDocs,
          tagBestRank,
        ]);
        totalMatch += matchCounts[tagName];
        if (tagBestRank < bestRank) bestRank = tagBestRank;
      }
    }
  }

  if (filteredChildren.length === 0 && filteredTags.length === 0) return null;

  return {
    name: node.name,
    n: node.n,
    c: filteredChildren.length > 0 ? filteredChildren : undefined,
    t: filteredTags.length > 0 ? filteredTags : undefined,
    _matchCount: totalMatch,
    _bestRank: bestRank,
  };
};

/**
 * Deduplicates docs across the tree: each document appears under only one tag
 * (the highest-ranked one). Walks in sorted render order (sub-folders by _bestRank,
 * then tags by best rank) so docs land in the most relevant tag first.
 */
const deduplicateTree = (nodes) => {
  const seen = new Set();
  const walk = (node) => {
    // Process sub-folders first (render before tags), sorted by _bestRank
    let children = node.c
      ? [...node.c]
          .sort((a, b) => (a._bestRank ?? Infinity) - (b._bestRank ?? Infinity))
          .map(walk)
          .filter(Boolean)
      : [];
    // Then tags sorted by best rank, dedup their docs
    let tags = node.t
      ? [...node.t]
          .sort((a, b) => (a[3] ?? Infinity) - (b[3] ?? Infinity))
          .map(([tagName, tagCount, tagDocs, tagBestRank]) => {
            const uniqueDocs = (tagDocs || []).filter((d) => {
              if (seen.has(d.u)) return false;
              seen.add(d.u);
              return true;
            });
            return [tagName, uniqueDocs.length, uniqueDocs, tagBestRank];
          })
          .filter(([, count]) => count > 0)
      : [];
    if (children.length === 0 && tags.length === 0) return null;
    return {
      ...node,
      c: children.length > 0 ? children : undefined,
      t: tags.length > 0 ? tags : undefined,
    };
  };
  return [...nodes]
    .sort((a, b) => (a._bestRank ?? Infinity) - (b._bestRank ?? Infinity))
    .map(walk)
    .filter(Boolean);
};

/**
 * Returns a small emoji logo based on URL domain.
 */
const getDocLogo = (url) => {
  if (!url) return "\uD83D\uDCC4";
  if (url.includes("github.com")) return "\uD83D\uDCBB";
  if (url.includes("twitter.com") || url.includes("x.com"))
    return "\uD835\uDD4F";
  if (url.includes("arxiv.org")) return "\uD83D\uDCDD";
  if (url.includes("huggingface.co")) return "\uD83E\uDD17";
  return "\uD83D\uDD17";
};

/**
 * Recursive tree node component for folders and tags.
 */
const TAG_DOC_LIMIT = 10;

/**
 * Infer source type from a document (compact tree format or full format).
 * Checks URL hostname, title, tags, and an optional parentTag (the tree tag
 * name the doc is grouped under) to match against known source keys.
 */
const getSourceFromDoc = (doc, sourceKeys, parentTag) => {
  const url = doc.u || doc.url || "";
  const title = (doc.t || doc.title || "").toLowerCase();
  try {
    const hostname = new URL(url).hostname.replace(/^www\./, "");
    for (const key of sourceKeys || []) {
      if (key === hostname || hostname.endsWith("." + key)) return key;
      if (
        key === "twitter.com" &&
        (hostname === "x.com" || hostname.endsWith(".x.com"))
      )
        return key;
    }
  } catch {}
  // Check tags if available (full doc format)
  const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
  const tagsLower = allTags.map((t) => (t || "").toLowerCase());
  for (const key of sourceKeys || []) {
    const prefix = key.split(".")[0];
    if (tagsLower.some((t) => t.includes(prefix))) return key;
    if (title.includes(prefix)) return key;
  }
  // Check parent tag name from tree context (compact docs lack tags)
  if (parentTag) {
    const ptLower = parentTag.toLowerCase();
    for (const key of sourceKeys || []) {
      const prefix = key.split(".")[0];
      if (ptLower.includes(prefix)) return key;
    }
  }
  return "other";
};

/**
 * Sorts docs in tiers, using ColBERT reranked position for matched docs:
 * 1. Matched + same source as filter (by rerank position)
 * 2. Matched + other source (by rerank position)
 * 3. Unmatched + same source (by date desc)
 * 4. Unmatched + other source (by date desc)
 * When no sourceFilter (or "all"), tiers collapse to: matched (by rank) first, then unmatched (by date).
 */
const sortTagDocs = (
  docs,
  matchedUrls,
  sourceFilter,
  sourceKeys,
  sortByDate,
  parentTag,
) => {
  return [...docs].sort((a, b) => {
    const aMatched = matchedUrls.has(a.u);
    const bMatched = matchedUrls.has(b.u);

    if (sortByDate) {
      // Matched first, then by date
      if (aMatched !== bMatched) return aMatched ? -1 : 1;
      return (b.d || "").localeCompare(a.d || "");
    }

    const hasFilter = sourceFilter instanceof Set && sourceFilter.size > 0;
    const aSameSource =
      hasFilter && sourceFilter.has(getSourceFromDoc(a, sourceKeys, parentTag));
    const bSameSource =
      hasFilter && sourceFilter.has(getSourceFromDoc(b, sourceKeys, parentTag));

    // Tier: matched+sameSource=0, matched+other=1, unmatched+sameSource=2, unmatched+other=3
    const aTier = (aMatched ? 0 : 2) + (aSameSource ? 0 : 1);
    const bTier = (bMatched ? 0 : 2) + (bSameSource ? 0 : 1);
    if (aTier !== bTier) return aTier - bTier;
    // Within matched tiers, sort by ColBERT rerank position
    if (aMatched && bMatched)
      return matchedUrls.get(a.u) - matchedUrls.get(b.u);
    return (b.d || "").localeCompare(a.d || "");
  });
};

/**
 * Recursive tree node component for folders and tags.
 */
const countMatchedDocs = (node, matchedUrls, sourceFilter, sourceKeys) => {
  let n = 0;
  for (const [tagName, , docs] of node.t || []) {
    for (const d of docs || []) {
      if (!matchedUrls.has(d.u)) continue;
      if (
        sourceFilter.size > 0 &&
        !sourceFilter.has(getSourceFromDoc(d, sourceKeys, tagName))
      )
        continue;
      n++;
    }
  }
  for (const ch of node.c || []) {
    n += countMatchedDocs(ch, matchedUrls, sourceFilter, sourceKeys);
  }
  return n;
};

const TreeNode = ({
  node,
  path,
  expanded,
  onToggle,
  onClickTag,
  isFiltered,
  expandedTags,
  onToggleTag,
  matchedUrls,
  showAllTags,
  onToggleShowAll,
  sourceFilter,
  sourceKeys,
  sortByDate,
}) => {
  const hasChildren =
    (node.c && node.c.length > 0) || (node.t && node.t.length > 0);
  const isExpanded = expanded.has(path);
  const displayCount = countMatchedDocs(
    node,
    matchedUrls,
    sourceFilter,
    sourceKeys,
  );

  if (displayCount === 0) return null;

  return (
    <div className="tree-node">
      <div
        className={`tree-header ${isFiltered && node._matchCount ? "matching" : ""}`}
        onClick={() => hasChildren && onToggle(path)}
      >
        <span className="tree-chevron">
          {hasChildren ? (isExpanded ? "\u25BE" : "\u25B8") : "\u2003"}
        </span>
        <span className="tree-icon">{"\uD83D\uDCC1"}</span>
        <span
          className="tree-name"
          onClick={(e) => {
            e.stopPropagation();
            if (!browsedFolders.has(node.name)) {
              browsedFolders.add(node.name);
              trackEvent("folder_browse", { folder_name: node.name });
            }
            if (hasChildren) onToggle(path);
          }}
        >
          {node.name}
        </span>
        <span className="tree-count">{displayCount}</span>
      </div>
      {isExpanded && hasChildren && (
        <div className="tree-children">
          {(node.c || []).map((child) => (
            <TreeNode
              key={child.name}
              node={child}
              path={`${path}/${child.name}`}
              expanded={expanded}
              onToggle={onToggle}
              onClickTag={onClickTag}
              isFiltered={isFiltered}
              expandedTags={expandedTags}
              onToggleTag={onToggleTag}
              matchedUrls={matchedUrls}
              showAllTags={showAllTags}
              onToggleShowAll={onToggleShowAll}
              sourceFilter={sourceFilter}
              sourceKeys={sourceKeys}
              sortByDate={sortByDate}
            />
          ))}
          {(node.t || []).map(([tagName, tagCount, tagDocs]) => {
            const tagPath = `${path}/#${tagName}`;
            const isTagExpanded = expandedTags.has(tagPath);
            // All docs for this tag, filtered by source if active
            let allTagDocs = tagDocs || [];
            if (sourceFilter.size > 0) {
              allTagDocs = allTagDocs.filter((d) =>
                sourceFilter.has(getSourceFromDoc(d, sourceKeys, tagName)),
              );
            }
            // Only show tag if at least one doc matches the search
            const hasMatchedDocs = allTagDocs.some((d) => matchedUrls.has(d.u));
            if (!hasMatchedDocs) return null;
            const sorted = sortTagDocs(
              allTagDocs,
              matchedUrls,
              sourceFilter,
              sourceKeys,
              sortByDate,
              tagName,
            );
            const showAll = showAllTags.has(tagPath);
            const visible = showAll ? sorted : sorted.slice(0, TAG_DOC_LIMIT);
            const hasMore = sorted.length > TAG_DOC_LIMIT;
            return (
              <div key={tagName} className="tree-tag-group">
                <div
                  className={`tree-leaf ${isFiltered ? "matching" : ""}`}
                  onClick={() => onToggleTag(tagPath)}
                >
                  <span className="tree-chevron">
                    {isTagExpanded ? "\u25BE" : "\u25B8"}
                  </span>
                  <span className="tree-icon">{"\uD83C\uDFF7\uFE0F"}</span>
                  <span
                    className="tree-name"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleTag(tagPath);
                    }}
                  >
                    {tagName}
                  </span>
                  <span className="tree-count">{allTagDocs.length}</span>
                </div>
                {isTagExpanded && (
                  <div className="tree-docs">
                    {visible.map((doc, i) => (
                      <a
                        key={doc.u || i}
                        className={`tree-doc ${matchedUrls.has(doc.u) ? "tree-doc-matched" : ""}`}
                        href={doc.u}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={() => {
                          if (doc.u && !clickedUrls.has(doc.u)) {
                            clickedUrls.add(doc.u);
                            trackEvent("click", {
                              doc_url: doc.u,
                              position: i,
                              score: null,
                              query: "",
                            });
                          }
                        }}
                      >
                        <span className="tree-doc-logo">
                          {getDocLogo(doc.u)}
                        </span>
                        <span className="tree-doc-title">{doc.t}</span>
                        <span className="tree-doc-date">{doc.d}</span>
                      </a>
                    ))}
                    {hasMore && (
                      <button
                        className="tree-doc-show-more"
                        onClick={() => onToggleShowAll(tagPath)}
                      >
                        {showAll
                          ? "Show less"
                          : `Show all ${sorted.length} documents`}
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

/**
 * FolderTree component: renders the full or filtered tree.
 */
const FolderTree = ({
  tree,
  documents,
  rankedDocs,
  query,
  onClickTag,
  sourceFilter,
  sourceKeys,
  sortByDate,
}) => {
  const [expandedTags, setExpandedTags] = useState(new Set());
  const [showAllTags, setShowAllTags] = useState(new Set());
  const [collapsedFolders, setCollapsedFolders] = useState(new Set());

  const handleToggleTag = useCallback((tagPath) => {
    setExpandedTags((prev) => {
      const next = new Set(prev);
      if (next.has(tagPath)) {
        next.delete(tagPath);
      } else {
        next.add(tagPath);
      }
      return next;
    });
  }, []);

  const handleToggleShowAll = useCallback((tagPath) => {
    setShowAllTags((prev) => {
      const next = new Set(prev);
      if (next.has(tagPath)) {
        next.delete(tagPath);
      } else {
        next.add(tagPath);
      }
      return next;
    });
  }, []);

  // Map of matched document URLs → rank position for relevancy sorting.
  // Built from rankedDocs (the same slice shown in the main view) so that
  // folder ordering exactly mirrors the main view, including after ColBERT reranking.
  const matchedUrls = useMemo(() => {
    const urls = new Map();
    const source = rankedDocs && rankedDocs.length > 0 ? rankedDocs : documents;
    for (let i = 0; i < source.length; i++) {
      if (source[i].url) urls.set(source[i].url, i);
    }
    return urls;
  }, [rankedDocs, documents]);

  // Build match counts from search result tags
  const matchCounts = useMemo(() => {
    if (!query.trim() || !documents.length) return {};
    const counts = {};
    for (const doc of documents) {
      const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
      for (const tag of allTags) {
        if (tag) counts[tag] = (counts[tag] || 0) + 1;
      }
    }
    return counts;
  }, [documents, query]);

  const hasMatches =
    query.trim().length > 0 && Object.keys(matchCounts).length > 0;

  // Build tag -> compact docs from search results for injection into tree
  const extraDocs = useMemo(() => {
    if (!hasMatches) return {};
    const byTag = {};
    for (const doc of documents) {
      const compact = {
        u: doc.url,
        t: doc.title || "",
        d: doc.date || "",
        tags: (doc.tags || []).concat(doc["extra-tags"] || []),
      };
      const allTags = (doc.tags || []).concat(doc["extra-tags"] || []);
      for (const tag of allTags) {
        if (tag && matchCounts[tag]) {
          if (!byTag[tag]) byTag[tag] = [];
          byTag[tag].push(compact);
        }
      }
    }
    return byTag;
  }, [documents, matchCounts, hasMatches]);

  // Filter tree, deduplicate docs (one tag per doc), sort by best rank
  const children = useMemo(() => {
    if (!tree || !hasMatches) return [];
    const filtered = filterTree(tree, matchCounts, extraDocs, matchedUrls);
    if (!filtered || !filtered.c) return [];
    return deduplicateTree(filtered.c);
  }, [tree, hasMatches, matchCounts, extraDocs, matchedUrls]);

  const handleToggleFolder = useCallback((path) => {
    setCollapsedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  // Auto-expand all folder paths and tag paths
  const { allFilteredPaths, autoExpandedTags } = useMemo(() => {
    const paths = new Set();
    const tagPaths = new Set();
    const walk = (node, path) => {
      paths.add(path);
      if (node.t) {
        for (const [tagName] of node.t) {
          tagPaths.add(`${path}/#${tagName}`);
        }
      }
      if (node.c) {
        for (const child of node.c) walk(child, `${path}/${child.name}`);
      }
    };
    for (const child of children) walk(child, `root/${child.name}`);
    return { allFilteredPaths: paths, autoExpandedTags: tagPaths };
  }, [children]);

  // Effective expanded folders: auto-expanded minus manually collapsed
  const effectiveExpandedFolders = useMemo(() => {
    const result = new Set(allFilteredPaths);
    for (const p of collapsedFolders) result.delete(p);
    return result;
  }, [allFilteredPaths, collapsedFolders]);

  // Effective expanded tags: auto-expanded minus manually collapsed
  const effectiveExpandedTags = useMemo(() => {
    const result = new Set(autoExpandedTags);
    for (const p of expandedTags) result.delete(p);
    return result;
  }, [autoExpandedTags, expandedTags]);

  if (children.length === 0) return null;

  return (
    <div className="folder-tree">
      <div className="folder-tree-header">
        <span className="folder-tree-title">Matching Folders</span>
        <span className="folder-tree-count">{children.length} folders</span>
      </div>
      {children.map((child) => (
        <TreeNode
          key={child.name}
          node={child}
          path={`root/${child.name}`}
          expanded={effectiveExpandedFolders}
          onToggle={handleToggleFolder}
          onClickTag={onClickTag}
          isFiltered={true}
          expandedTags={effectiveExpandedTags}
          onToggleTag={handleToggleTag}
          matchedUrls={matchedUrls}
          showAllTags={showAllTags}
          onToggleShowAll={handleToggleShowAll}
          sourceFilter={sourceFilter}
          sourceKeys={sourceKeys}
          sortByDate={sortByDate}
        />
      ))}
    </div>
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
  const [tree, setTree] = useState(null);
  const [sources, setSources] = useState([]);
  const [favorites, setFavorites] = useState(new Set());
  const [showFavorites, setShowFavorites] = useState(false);
  const [theme, setTheme] = useState(
    document.documentElement.getAttribute("data-theme") || "dark",
  );

  // --- Refs ---
  const searchTimerRef = useRef(null);
  const workerRef = useRef(null);
  const latestQueryIdRef = useRef(0);
  const rerankTimerRef = useRef(null);
  const immediateRerankRef = useRef(false);
  const settleTimerRef = useRef(null);
  const lastSearchMeta = useRef(null);

  // --- Function Declarations ---

  /**
   * Fetches the latest documents from the backend.
   */
  const fetchLatest = useCallback(() => {
    apiLatest(FETCH_COUNT, sourceFilter)
      .then((docs) => setDocuments(docs))
      .catch((error) =>
        console.error("[APP] Failed to fetch latest documents:", error),
      );
  }, [sourceFilter]);

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
      // Append tag filter to the query string if a node is selected
      const fullQuery = selectedNode
        ? `${searchQuery} ${selectedNode}`
        : searchQuery;

      const t0 = performance.now();
      apiSearch(fullQuery, sortChronologically, sourceFilter)
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
    [selectedNode, fetchLatest, sourceFilter],
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
    fetch(`${DATA_API_URL}/api/folder_tree`)
      .then((res) => res.json())
      .then((data) => setTree(data))
      .catch(() => {
        // Fallback to static JSON files if data API is unavailable
        fetch("data/folder_tree.json")
          .then((res) => res.json())
          .then((data) => setTree(data))
          .catch((error) =>
            console.error("[APP] Failed to load folder tree:", error),
          );
      });
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
    if (urlSource) {
      setSourceFilter(new Set(urlSource.split(",").filter(Boolean)));
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
      !isSortedByDate;

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
  }, [documents, query, modelStatus, resultsReranked, isSortedByDate]);

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
        syncURL({ source: sourceFilter });
        fetchLatest();
        return;
      }

      setSelectedNode(null);
      setIsSortedByDate(false);
      setResultsReranked(false);
      syncURL({ query: newQuery, source: sourceFilter });
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

  const handleClickTag = useCallback(
    (tag) => {
      const newQuery = `${query} ${tag}`.trim();
      setQuery(newQuery);
      setIsSortedByDate(false);
      setResultsReranked(false);
      const searchInput = document.getElementById("search");
      if (searchInput) searchInput.value = newQuery;
      syncURL({ query: newQuery, source: sourceFilter });
      runNow(newQuery, true);
    },
    [query, runNow, sourceFilter],
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

      const parts = text.split(
        new RegExp(`(${Array.from(keywordSet).join("|")})`, "gi"),
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

  // --- Computed: filter by source + favorites then cap at DISPLAY_COUNT ---
  const displayedDocs = useMemo(() => {
    let filtered = (documents || []).filter(
      (doc) =>
        sourceFilter.size === 0 || sourceFilter.has(getDocumentSource(doc)),
    );
    if (showFavorites) {
      filtered = filtered.filter((doc) => favorites.has(doc.url));
    }
    return filtered.slice(0, DISPLAY_COUNT);
  }, [documents, sourceFilter, getDocumentSource, showFavorites, favorites]);

  // --- Render ---
  const folderPanel = document.getElementById("folder-panel");

  return (
    <React.Fragment>
      <div id="search-container">
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
                  syncURL({ query, source: next });
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

      <div id="documents">
        {displayedDocs.map((doc, index) => (
          <div className="document" key={doc.url || index}>
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
                    className="tag"
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
        ))}
      </div>

      {folderPanel &&
        !isMobile &&
        createPortal(
          <FolderTree
            tree={tree}
            documents={documents}
            rankedDocs={displayedDocs}
            query={query}
            onClickTag={handleClickTag}
            sourceFilter={sourceFilter}
            sourceKeys={sources.map((s) => s.key)}
            sortByDate={isSortedByDate}
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
