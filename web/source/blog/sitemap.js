// Full-parity port of sources/blog/sitemap.py.
//
// Walks sitemap.xml (or sitemapindex, recursing up to depth 3 with a
// 50-child cap per index). Small sitemaps (≤ 100 candidate URLs) are
// "sampled": we fetch each page's HTML to pull the real <title> and
// <meta description> instead of deriving both from the URL slug.
// Concurrency is capped at 32 global with a per-host 4 in-flight limit
// so a single slow host can't stall the pipeline.

import { fetchText } from "../utils/http.js";
import {
  parseDate,
  coerceDate,
  titleFromUrl,
  unescapeHtml,
} from "./_helpers.js";

const MAX_URLS = 1000;
const INDEX_MAX_DEPTH = 3;
const INDEX_MAX_CHILDREN = 50;
const SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9";

const SAMPLE_PAGES_THRESHOLD = 100;
const SAMPLE_GLOBAL_WORKERS = 32;
const SAMPLE_PER_HOST_LIMIT = 4;
const SAMPLE_FETCH_TIMEOUT_MS = 8000;

const LIST_PAGE_SEGMENTS = new Set([
  "tag",
  "tags",
  "category",
  "categories",
  "author",
  "authors",
  "page",
  "archive",
  "archives",
  "label",
  "labels",
  "topic",
  "topics",
  "feed",
  "rss",
  "atom",
  "amp",
  "search",
]);

function looksLikeListPage(url) {
  try {
    const path = new URL(url).pathname.replace(/^\/|\/$/g, "").toLowerCase();
    if (!path) return false;
    return path.split("/").some((seg) => LIST_PAGE_SEGMENTS.has(seg));
  } catch {
    return false;
  }
}

function textNS(parent, name) {
  const el = parent.getElementsByTagNameNS(SM_NS, name)[0];
  if (el) return (el.textContent || "").trim();
  const plain = parent.getElementsByTagName(name)[0];
  return plain ? (plain.textContent || "").trim() : "";
}

// ─────────────────────────────────────────────────────────────────
// Page-meta sampler — fetch HTML, regex out <title> + <meta description>
// ─────────────────────────────────────────────────────────────────

const TITLE_RE = /<title[^>]*>([\s\S]*?)<\/title>/i;
// Python has two patterns (name|property first, content first). JS
// needs both because HTML attribute order is author-chosen.
const META_DESC_FORWARD_RE =
  /<meta\s+[^>]*?(?:name|property)\s*=\s*["']?(description|og:description|twitter:description)["']?[^>]*?content\s*=\s*["']([^"']+)["']/i;
const META_DESC_REVERSE_RE =
  /<meta\s+[^>]*?content\s*=\s*["']([^"']+)["'][^>]*?(?:name|property)\s*=\s*["']?(description|og:description|twitter:description)["']?/i;
const HTML_TAG_RE = /<[^>]+>/g;
const WHITESPACE_RE = /\s+/g;

// ── Date extraction patterns (mirror of sources/blog/sitemap.py) ──
// Order matters: most authoritative signals first.
const DATE_META_PATTERNS = [
  // <meta property="article:published_time" content="..."> (forward attr order)
  /<meta[^>]+(?:property|name)=["'](?:article:published_time|article:published|og:published_time|datePublished|date|publish[-_]?date|pubdate|publication_date|date\.published)["'][^>]+content=["']([^"']+)["']/i,
  // content first, property second
  /<meta[^>]+content=["']([^"']+)["'][^>]+(?:property|name)=["'](?:article:published_time|article:published|og:published_time|datePublished|date|publish[-_]?date|pubdate|publication_date|date\.published)["']/i,
  // JSON-LD: "datePublished":"2023-01-15T..."
  /"datePublished"\s*:\s*"([^"]+)"/,
  // <time datetime="…"> (first hit)
  /<time[^>]+datetime=["']([^"']+)["']/i,
  // microdata: itemprop="datePublished" content/datetime="..."
  /<[^>]+itemprop=["']datePublished["'][^>]*(?:content|datetime)=["']([^"']+)["']/i,
];
// "Published [on] January 1, 2023" / "Posted: 2024-03-15" / French
// "Publié le …". Strict label whitelist — bare "on" is excluded
// because it false-positives on template footers like "Released on:
// November 28, 2023" (Webflow Timothy Ricks template at the bottom
// of every lighton.ai blog post).
const INLINE_DATE_LABEL_RE =
  /(?:published|posted|publi[ée](?:e|ed)?|date)\s*(?:on\s+|le\s+|:\s*)?((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})/i;
// Last-ditch bare "Month Day, Year" anywhere in the body.
const BARE_DATE_RE =
  /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/i;

/* Extract a publication date from an HTML body. Best-effort, ranked
 * by signal confidence:
 *   1. <meta article:published_time> / OG / datePublished variants
 *   2. JSON-LD "datePublished"
 *   3. <time datetime="...">
 *   4. itemprop="datePublished"
 *   5. Labelled inline strings ("Published on Jan 1, 2023")
 *   6. Bare "Month Day, Year" anywhere in body
 *
 * Returns "" when nothing parseable is found. Reuses the existing
 * `parseDate` so each candidate goes through the full set of
 * supported input formats. */
function extractDateFromHtml(text) {
  for (const pat of DATE_META_PATTERNS) {
    const m = pat.exec(text);
    if (m) {
      const d = parseDate(m[1]);
      if (d) return d;
    }
  }
  const ml = INLINE_DATE_LABEL_RE.exec(text);
  if (ml) {
    const d = parseDate(ml[1]);
    if (d) return d;
  }
  const mb = BARE_DATE_RE.exec(text);
  if (mb) {
    const d = parseDate(mb[0]);
    if (d) return d;
  }
  return "";
}

/** Pull (title, description, date) out of an HTML body. The first
 * ~32 KB is enough for title+description (head only); date detection
 * widens to ~96 KB because templates often emit the publication
 * date in the body, not the head. */
function extractMeta(html) {
  const head = html.slice(0, 32_768);
  const body = html.slice(0, 96_000);
  let title = "";
  const tm = TITLE_RE.exec(head);
  if (tm) {
    title = tm[1].replace(HTML_TAG_RE, " ");
    title = unescapeHtml(title).replace(WHITESPACE_RE, " ").trim();
  }
  let desc = "";
  for (const pat of [META_DESC_FORWARD_RE, META_DESC_REVERSE_RE]) {
    const m = pat.exec(head);
    if (!m) continue;
    // Of the two captured groups, one is the attribute name (short),
    // one is the prose. Pick the longer.
    const candidates = [m[1], m[2]].filter(
      (g) =>
        g &&
        g.length > 4 &&
        !["description", "og:description", "twitter:description"].includes(
          g.toLowerCase(),
        ),
    );
    if (candidates.length > 0) {
      desc = unescapeHtml(candidates[0]).replace(WHITESPACE_RE, " ").trim();
      break;
    }
  }
  const date = extractDateFromHtml(body);
  return { title, description: desc, date };
}

/**
 * Fetch `url` with a timeout, return (title, description, date). Reads
 * the body once and runs regex over it — first 32 KB for title /
 * description, first 96 KB for date detection. Skips non-HTML
 * content types.
 */
async function fetchPageMeta(url) {
  const empty = { title: "", description: "", date: "" };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), SAMPLE_FETCH_TIMEOUT_MS);
  try {
    const resp = await fetch(url, {
      signal: ctrl.signal,
      headers: {
        Accept: "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
      },
    });
    if (!resp.ok) return empty;
    const ct = (resp.headers.get("Content-Type") || "").toLowerCase();
    if (ct && !ct.includes("html") && !ct.includes("xml")) {
      return empty;
    }
    const text = await resp.text();
    return extractMeta(text);
  } catch {
    return empty;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fan out `fetchPageMeta` across `urls` with two limits: a global
 * worker pool of 32, and a per-host semaphore of 4 in-flight. Matches
 * the thread-pool + host-semaphore setup in Python's `_sample_pages`.
 */
async function samplePages(urls) {
  const out = new Map();
  if (urls.length === 0) return out;

  let nextIndex = 0;
  const hostInflight = new Map(); // host → int
  const hostQueues = new Map(); // host → [resolve]

  function acquireHost(host) {
    const inflight = hostInflight.get(host) || 0;
    if (inflight < SAMPLE_PER_HOST_LIMIT) {
      hostInflight.set(host, inflight + 1);
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      if (!hostQueues.has(host)) hostQueues.set(host, []);
      hostQueues.get(host).push(resolve);
    });
  }

  function releaseHost(host) {
    const queue = hostQueues.get(host);
    if (queue && queue.length > 0) {
      queue.shift()();
      return;
    }
    const inflight = (hostInflight.get(host) || 1) - 1;
    if (inflight <= 0) hostInflight.delete(host);
    else hostInflight.set(host, inflight);
  }

  async function worker() {
    while (true) {
      const i = nextIndex++;
      if (i >= urls.length) return;
      const url = urls[i];
      let host = "";
      try {
        host = new URL(url).host.toLowerCase();
      } catch {
        out.set(url, { title: "", description: "" });
        continue;
      }
      await acquireHost(host);
      try {
        out.set(url, await fetchPageMeta(url));
      } finally {
        releaseHost(host);
      }
    }
  }

  const workers = [];
  for (let i = 0; i < Math.min(SAMPLE_GLOBAL_WORKERS, urls.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return out;
}

// ─────────────────────────────────────────────────────────────────
// Sitemap XML parser (with recursion into sitemapindex)
// ─────────────────────────────────────────────────────────────────

async function parseOne(xml, baseUrl, depth, seenIndexes, remaining) {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  if (!doc.documentElement || doc.getElementsByTagName("parsererror")[0]) {
    return [];
  }
  const root = doc.documentElement;
  const tag = root.tagName.toLowerCase();

  if (tag.endsWith("sitemapindex")) {
    if (depth >= INDEX_MAX_DEPTH) return [];
    const results = [];
    const children = root.getElementsByTagNameNS(SM_NS, "sitemap");
    const childList = children.length
      ? Array.from(children)
      : Array.from(root.getElementsByTagName("sitemap"));
    let visited = 0;
    for (const sm of childList) {
      if (visited >= INDEX_MAX_CHILDREN) break;
      const childUrl = textNS(sm, "loc");
      if (!childUrl || seenIndexes.has(childUrl)) continue;
      seenIndexes.add(childUrl);
      visited += 1;
      let childXml;
      try {
        childXml = await fetchText(childUrl, { timeoutMs: 30000 });
      } catch {
        continue;
      }
      const childResults = await parseOne(
        childXml,
        childUrl,
        depth + 1,
        seenIndexes,
        remaining == null ? null : Math.max(0, remaining - results.length),
      );
      results.push(...childResults);
      if (remaining != null && results.length >= remaining) break;
    }
    return results;
  }

  // urlset
  const urls = root.getElementsByTagNameNS(SM_NS, "url").length
    ? Array.from(root.getElementsByTagNameNS(SM_NS, "url"))
    : Array.from(root.getElementsByTagName("url"));
  const results = [];
  for (const urlEl of urls) {
    const loc = textNS(urlEl, "loc");
    if (!loc) continue;
    const lastmod = textNS(urlEl, "lastmod");
    results.push([loc, parseDate(lastmod)]);
    if (remaining != null && results.length >= remaining) break;
  }
  return results;
}

// ─────────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────────

export async function sitemap({
  sitemapUrl,
  tags = [],
  urlFilter = null,
  maxUrls = MAX_URLS,
  existingUrls = null,
}) {
  let xml;
  try {
    xml = await fetchText(sitemapUrl, { timeoutMs: 30000 });
  } catch (err) {
    console.warn(`[sync] sitemap fetch failed: ${sitemapUrl} — ${err.message}`);
    return {};
  }

  let entries;
  try {
    // Parse one past the cap so we can distinguish "exactly max_urls
    // articles" (keep) from "more than max_urls" (catalog — drop).
    entries = await parseOne(xml, sitemapUrl, 0, new Set(), maxUrls + 1);
  } catch (err) {
    console.warn("[sync] sitemap parse error:", err);
    return {};
  }

  const candidates = [];
  for (const [url, date] of entries) {
    if (urlFilter && !url.includes(urlFilter)) continue;
    if (looksLikeListPage(url)) continue;
    candidates.push([url, date]);
  }

  // Catalog guard (Python: silently drop if over the cap).
  if (candidates.length > maxUrls) return {};

  // Scope to NEW URLs before sampling so we don't refetch pages we
  // already own. Matches the Python behavior.
  const newCandidates = candidates.filter(
    ([url]) => !(existingUrls && existingUrls.has(url)),
  );

  let sampled = new Map();
  if (
    newCandidates.length > 0 &&
    newCandidates.length <= SAMPLE_PAGES_THRESHOLD
  ) {
    sampled = await samplePages(newCandidates.map(([u]) => u));
  }

  const out = {};
  for (let rank = 0; rank < newCandidates.length; rank++) {
    const [url, date] = newCandidates[rank];
    const slugTitle = titleFromUrl(url);
    const scraped = sampled.get(url) || {
      title: "",
      description: "",
      date: "",
    };
    // Date precedence: sitemap lastmod (if present) > scraped page
    // date > URL-embedded date > rank-based fallback. The scraped
    // date overrides URL year-only fallbacks so we get real
    // publication dates from `<meta article:published_time>` /
    // JSON-LD / `<time datetime=…>` whenever the page exposes them.
    const seedDate = date || scraped.date || "";
    const coercedDate = coerceDate(seedDate, url, rank);
    let title;
    let summary;
    if (scraped.title) {
      title = scraped.title;
      // Python note: leave summary empty when scraped desc is missing,
      // so downstream cleanup doesn't treat title==summary as a
      // single-token low-signal doc.
      summary = scraped.description;
    } else {
      title = slugTitle;
      summary = slugTitle !== url ? slugTitle : "";
    }
    out[url] = {
      title,
      summary,
      date: coercedDate,
      tags: [...tags],
    };
  }
  return out;
}
