// Full-parity port of sources/blog/_helpers.py.
//
// The Python helpers do a lot: 16-format date parser with weekday/TZ
// normalization, multi-pass HTML cleanup, title/summary noise
// stripping, URL-path tag hinting, and tracking-param scrubbing. We
// mirror it here so the browser sync and the server pipeline produce
// the same `{url, title, summary, date, tags}` shape for a given feed.
//
// One rule throughout: if Python and JS disagree on edge cases
// (e.g. strptime vs Date.parse), we bias toward Python's behavior.

// ─────────────────────────────────────────────────────────────────
// HTML cleanup
// ─────────────────────────────────────────────────────────────────

const SCRIPT_STYLE_RE =
  /<(?:script|style|noscript|template)\b[^>]*>[\s\S]*?<\/(?:script|style|noscript|template)>/gi;
const HTML_COMMENT_RE = /<!--[\s\S]*?-->/g;
const HTML_TAG_RE = /<[^>]+>/g;
const WHITESPACE_RE = /\s+/g;

/**
 * Full-fidelity HTML-entity decoder — mirror of Python's
 * `html.unescape`. We use the textarea innerHTML trick so the
 * browser's own parser handles every named entity (`&eacute;`,
 * `&trade;`, `&hellip;`, …) plus numeric (`&#8211;` / `&#x2014;`)
 * forms. Textarea is RCDATA content: tags stay literal, entities
 * decode. That matches `html.unescape` which doesn't touch tags.
 *
 * Sentinel swap protects against `</textarea>` in the input closing
 * our wrapper early (which would make the tail leak into the DOM as
 * real markup). We swap it to an unmistakable rune before decoding
 * and swap it back after. Using a Private Use Area codepoint that
 * never appears in real feed content.
 */
const _entityDecoder =
  typeof document !== "undefined" ? document.createElement("textarea") : null;
const _TEXTAREA_SENTINEL = "XTXA";
export function unescapeHtml(s) {
  if (!s) return "";
  if (!_entityDecoder) return s;
  const raw = String(s);
  const safe = raw.replace(/<\/textarea/gi, _TEXTAREA_SENTINEL);
  _entityDecoder.innerHTML = safe;
  const decoded = _entityDecoder.value;
  return decoded.includes(_TEXTAREA_SENTINEL)
    ? decoded.split(_TEXTAREA_SENTINEL).join("</textarea")
    : decoded;
}

/**
 * Remove HTML tags and normalize whitespace. Matches Python's
 * `_strip_html` exactly:
 *   1. Strip <script>/<style>/<noscript>/<template> blocks WITH
 *      their content so CSS/JS doesn't leak into summaries.
 *   2. Strip HTML comments.
 *   3. Unescape entities (CDATA-escaped markup like
 *      `&lt;script&gt;…&lt;/script&gt;` now shows up as tags).
 *   4. Run the script/style/comment strip AGAIN on the unescaped
 *      text — that's the whole reason for the two-pass design.
 *   5. Strip remaining tags, collapse whitespace.
 */
export function stripHtmlDeep(text) {
  if (!text) return "";
  let s = String(text);
  s = s.replace(SCRIPT_STYLE_RE, " ");
  s = s.replace(HTML_COMMENT_RE, " ");
  s = unescapeHtml(s);
  s = s.replace(SCRIPT_STYLE_RE, " ");
  s = s.replace(HTML_COMMENT_RE, " ");
  s = s.replace(HTML_TAG_RE, " ");
  s = s.replace(WHITESPACE_RE, " ");
  return s.trim();
}

// ─────────────────────────────────────────────────────────────────
// Summary / title cleanup
// ─────────────────────────────────────────────────────────────────

// Leading RFC-style date, e.g. "Thu, 09 Apr 2026 10:15:00 +0000 — <summary>".
const LEADING_RFC_DATE_RE = new RegExp(
  "^\\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\\s+\\d{1,2}\\s+" +
    "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+\\d{4}" +
    "(?:\\s+\\d{1,2}:\\d{2}(?::\\d{2})?(?:\\s*[+\\-]\\d{4}|\\s*[A-Z]{2,4})?)?",
  "i",
);

// Common feed-footer noise we'd rather not carry into the index.
const TRAILING_NOISE_RE = new RegExp(
  "\\s*(?:Continue reading.*|Read (?:more|the rest).*|" +
    "The post .*? appeared first on .+?\\.?|" +
    "\\[\\s*…\\s*\\]|" +
    "\\(more…\\))\\s*$",
  "is",
);

/**
 * Strip leading/trailing noise and truncate cleanly at a word
 * boundary. Matches Python's `_clean_summary`.
 */
export function cleanSummary(text, maxLen = 300) {
  if (!text) return "";
  let s = String(text).trim();
  s = s.replace(LEADING_RFC_DATE_RE, "");
  s = s.replace(/^[\s:\-|—]+/, "");
  s = s.replace(TRAILING_NOISE_RE, "");
  s = s.replace(WHITESPACE_RE, " ").trim();
  if (s.length > maxLen) {
    let cut = s.slice(0, maxLen - 1);
    if (cut.includes(" ")) {
      const idx = cut.lastIndexOf(" ");
      if (idx > 0) cut = cut.slice(0, idx);
    }
    s = cut.replace(/[\s,.;:\-—]+$/, "") + "…";
  }
  return s;
}

// ─────────────────────────────────────────────────────────────────
// Title cleanup
// ─────────────────────────────────────────────────────────────────

const PART_WHITELIST = new Set([
  "the sequel",
  "part 2",
  "part ii",
  "part one",
  "part two",
]);

/**
 * Clean a feed-entry title the way `_clean_title` does.
 *
 * - Strip HTML + whitespace.
 * - Drop trailing " | Site Name" / " – Blog Name" / " — Blog Name"
 *   suffixes that duplicate the feed title.
 * - Conservatively drop " - Site" suffix only when the prefix has
 *   ≥4 words and the suffix ≤3 words (and isn't a known "Part 2"
 *   style subtitle), so real subtitles are kept.
 * - If the title is a single all-lowercase token (Jekyll
 *   slug-as-title), capitalize its first letter or prefer a nicer
 *   URL-slug-derived title.
 */
export function cleanTitle(title, link = null) {
  if (!title) return "";
  let t = stripHtmlDeep(title);
  if (!t) return "";

  const pipeMatch = t.match(/^(.*?)\s+([|–—])\s+([^|–—]{2,40})$/);
  if (pipeMatch) {
    t = pipeMatch[1].trim();
  } else {
    const dashMatch = t.match(/^(.*?)\s+-\s+([^\-]{2,40})$/);
    if (dashMatch) {
      const prefix = dashMatch[1].trim();
      const suffix = dashMatch[2].trim();
      const prefixWords = prefix.split(/\s+/).length;
      const suffixWords = suffix.split(/\s+/).length;
      if (
        prefixWords >= 4 &&
        suffixWords <= 3 &&
        !PART_WHITELIST.has(suffix.toLowerCase())
      ) {
        t = prefix;
      }
    }
  }

  if (
    link &&
    t &&
    t === t.toLowerCase() &&
    !t.includes(" ") &&
    !t.includes("-") &&
    !t.includes("_")
  ) {
    const slugTitle = titleFromUrl(link);
    if (slugTitle && slugTitle.toLowerCase() !== t.toLowerCase()) {
      if (slugTitle.includes(" ")) return slugTitle;
    }
    return t.charAt(0).toUpperCase() + t.slice(1);
  }
  return t;
}

// ─────────────────────────────────────────────────────────────────
// Fallback summary with category-hint detection
// ─────────────────────────────────────────────────────────────────

export function fallbackSummary(summary, title, categoryHints = null) {
  if (summary) {
    const low = summary.trim().toLowerCase();
    if (Array.isArray(categoryHints) && low.length < 30) {
      const cats = new Set(
        categoryHints.map((c) => (c || "").trim().toLowerCase()),
      );
      if (cats.has(low)) return (title || "").trim();
    }
    return summary;
  }
  return (title || "").trim();
}

// ─────────────────────────────────────────────────────────────────
// Date parsing
// ─────────────────────────────────────────────────────────────────

const WEEKDAY_NORMALIZE = [
  [/^\s*(Monday|Mondy)/i, "Mon"],
  [/^\s*Tues(?:day)?/i, "Tue"],
  [/^\s*Wednes(?:day)?/i, "Wed"],
  [/^\s*Thurs(?:day)?/i, "Thu"],
  [/^\s*Frid(?:ay)?/i, "Fri"],
  [/^\s*Saturd(?:ay)?/i, "Sat"],
  [/^\s*Sund(?:ay)?/i, "Sun"],
];

const TZ_NORMALIZE = [
  [/\bUTC\b/g, "+0000"],
  [/\bGMT\s*\+?0+\b/g, "+0000"],
  [/\bEST\b/g, "-0500"],
  [/\bEDT\b/g, "-0400"],
  [/\bPST\b/g, "-0800"],
  [/\bPDT\b/g, "-0700"],
];

const MONTH_MAP = {
  jan: 1,
  feb: 2,
  mar: 3,
  apr: 4,
  may: 5,
  jun: 6,
  jul: 7,
  aug: 8,
  sep: 9,
  oct: 10,
  nov: 11,
  dec: 12,
  january: 1,
  february: 2,
  march: 3,
  april: 4,
  june: 6,
  july: 7,
  august: 8,
  september: 9,
  october: 10,
  november: 11,
  december: 12,
};

const MONTH_SHORT_MAP = {
  jan: "01",
  feb: "02",
  mar: "03",
  apr: "04",
  may: "05",
  jun: "06",
  jul: "07",
  aug: "08",
  sep: "09",
  oct: "10",
  nov: "11",
  dec: "12",
};

function pad2(n) {
  return String(n).padStart(2, "0");
}

/** Normalize weekday/TZ aliases so the fallback parsers recognise them. */
function normalizeDateString(s) {
  let out = s;
  // Weekday patterns are mutually exclusive (the string starts with
  // at most one weekday word); first match wins, mirroring Python's
  // `re.sub(..., count=1)` applied to each pattern.
  for (const [pat, repl] of WEEKDAY_NORMALIZE) {
    if (pat.test(out)) {
      out = out.replace(pat, repl);
      break;
    }
  }
  for (const [pat, repl] of TZ_NORMALIZE) out = out.replace(pat, repl);
  return out;
}

/**
 * Try to parse a variety of date strings into `YYYY-MM-DD`.
 *
 * Python's `_parse_date` uses strptime with 16 format strings.
 * Date.parse() in browsers accepts most ISO 8601 and RFC 2822
 * forms, which covers most of our fleet; we add explicit handling
 * for the extras Python parses (Jun 12, 2023 / 12 Jun 2023 /
 * YYYY/MM/DD / DD/MM/YYYY) and finish with a YYYY-MM-DD or
 * YYYY/MM/DD substring fallback exactly like Python does.
 */
export function parseDate(raw) {
  if (!raw) return "";
  const s0 = String(raw).trim();
  if (!s0) return "";
  const s = normalizeDateString(s0);

  // Fast path: the exact YYYY-MM-DD substring form, anywhere in the string.
  const isoSub = s.match(/(\d{4})-(\d{2})-(\d{2})/);
  if (isoSub) return `${isoSub[1]}-${isoSub[2]}-${isoSub[3]}`;

  // Date.parse handles: ISO 8601 (with/without ms/TZ), RFC 2822,
  // "Jan 2 2023", "January 2, 2023", "2 Jan 2023", etc.
  const t = Date.parse(s);
  if (!Number.isNaN(t)) {
    const d = new Date(t);
    return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
  }

  // Additional Python formats Date.parse commonly misses.
  // "Jun 12, 2023" / "June 12, 2023" / "12 Jun 2023" etc are handled
  // by Date.parse, but "DD/MM/YYYY" and "YYYY/MM/DD" are ambiguous.
  let m = s.match(/(\d{4})\/(\d{2})\/(\d{2})/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  m = s.match(/(\d{1,2})\/(\d{1,2})\/(\d{4})/);
  if (m) return `${m[3]}-${pad2(m[2])}-${pad2(m[1])}`;

  // Month-name variants we might see pre-Date.parse-fix.
  m = s.match(/\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})/);
  if (m) {
    const mm = MONTH_MAP[m[1].toLowerCase()];
    if (mm) return `${m[3]}-${pad2(mm)}-${pad2(parseInt(m[2], 10))}`;
  }
  m = s.match(/\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})/);
  if (m) {
    const mm = MONTH_MAP[m[2].toLowerCase()];
    if (mm) return `${m[3]}-${pad2(mm)}-${pad2(parseInt(m[1], 10))}`;
  }
  return "";
}

/** Extract a date from a URL path. Tries three patterns, in order
 * of confidence (mirrors the Python `_date_from_url`):
 *   1. `/YYYY/MM/DD/...`     → full date
 *   2. `/YYYY-MM-DD-...`     → full date (Hugo / Jekyll style)
 *   3. `/YYYY-...`           → year-only fallback (lands on Jan 1)
 *
 * The year-only fallback is what catches paths like
 * `/lighton-blogs/2023-the-rise-of-on-prem-llms`: the year matters
 * for date-sort and diversity bucketing, the day doesn't. */
const URL_DATE_RE = /\/(\d{4})\/([A-Za-z]{3}|\d{1,2})\/(\d{1,2})\//;
const URL_DATE_DASH_RE = /\/(\d{4})-(\d{2})-(\d{2})(?:[-/_.]|$)/;
const URL_YEAR_SLUG_RE = /\/(199\d|20\d{2})(?:-|$)/;
export function dateFromUrl(url) {
  const m = URL_DATE_RE.exec(url || "");
  if (m) {
    const [, year, monthStr, day] = m;
    const lower = monthStr.toLowerCase();
    let month = MONTH_SHORT_MAP[lower];
    if (!month) {
      month = monthStr.padStart(2, "0");
      if (!/^\d+$/.test(month)) return "";
    }
    return `${year}-${month}-${day.padStart(2, "0")}`;
  }
  const md = URL_DATE_DASH_RE.exec(url || "");
  if (md) return `${md[1]}-${md[2]}-${md[3]}`;
  const my = URL_YEAR_SLUG_RE.exec(url || "");
  if (my) return `${my[1]}-01-01`;
  return "";
}

export function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

/** Best-effort date: parse `dateStr`, else extract from URL, else
 * walk-backwards-from-today by `rank` days, else today. The `rank`
 * fallback prevents 420 sitemap URLs without a `lastmod` from all
 * collapsing onto the same day; pass an enumerate index from the
 * caller so each unknown-date entry within the same fetch gets a
 * distinct day, preserving the page's order. */
export function coerceDate(dateStr, fallbackUrl = null, rank = null) {
  const d = parseDate(dateStr);
  if (d) return d;
  if (fallbackUrl) {
    const fromUrl = dateFromUrl(fallbackUrl);
    if (fromUrl) return fromUrl;
  }
  if (typeof rank === "number" && rank > 0) {
    return new Date(Date.now() - rank * 86400000).toISOString().slice(0, 10);
  }
  return todayISO();
}

// ─────────────────────────────────────────────────────────────────
// URL handling
// ─────────────────────────────────────────────────────────────────

const TRACKING_PARAMS = new Set([
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_term",
  "utm_content",
  "utm_name",
  "utm_id",
  "utm_reader",
  "utm_referrer",
  "mc_cid",
  "mc_eid",
  "fbclid",
  "gclid",
  "gbraid",
  "wbraid",
  "igshid",
  "ref",
  "ref_src",
  "ref_url",
  "spm",
  "yclid",
  "_hsenc",
  "_hsmi",
]);

export function stripTracking(url) {
  try {
    const u = new URL(url);
    if (!u.search) return url;
    for (const k of [...u.searchParams.keys()]) {
      if (TRACKING_PARAMS.has(k.toLowerCase())) u.searchParams.delete(k);
    }
    u.hash = "";
    let s = u.toString();
    if (s.endsWith("?")) s = s.slice(0, -1);
    return s;
  } catch {
    return url;
  }
}

export function resolveUrl(base, href) {
  if (!href) return "";
  try {
    return stripTracking(new URL(href.trim(), base).toString());
  } catch {
    return "";
  }
}

// ─────────────────────────────────────────────────────────────────
// URL-derived helpers (tags, title fallback)
// ─────────────────────────────────────────────────────────────────

const BORING_PATH_SEGMENTS = new Set([
  "posts",
  "post",
  "blog",
  "blogs",
  "tag",
  "tags",
  "category",
  "categories",
  "articles",
  "article",
  "p",
  "entry",
  "essays",
  "essay",
  "notes",
  "note",
  "news",
  "jan",
  "feb",
  "mar",
  "apr",
  "may",
  "jun",
  "jul",
  "aug",
  "sep",
  "oct",
  "nov",
  "dec",
  "january",
  "february",
  "march",
  "april",
  "june",
  "july",
  "august",
  "september",
  "october",
  "november",
  "december",
]);
const DATE_SLUG_RE = /^\d{4}[-_]\d{1,2}([-_]\d{1,2})?([-_].+)?$/;

export function tagsFromUrl(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return [];
  }
  const parts = parsed.pathname.split("/").filter(Boolean);
  if (parts.length === 0) return [];
  const hints = [];
  // Exclude the final slug segment — that's the article itself.
  for (const p of parts.slice(0, -1)) {
    const slug = p.toLowerCase();
    if (BORING_PATH_SEGMENTS.has(slug)) continue;
    if (/^\d+$/.test(slug)) continue;
    if (DATE_SLUG_RE.test(slug)) continue;
    const hint = slug.replace(/[-_]/g, " ").trim();
    if (/\d/.test(hint)) continue;
    if (hint.length > 2 && hint.length < 40) hints.push(hint);
  }
  return hints;
}

/** URL-slug → sentence-case title, matching Python `_title_from_url`. */
export function titleFromUrl(url) {
  try {
    const p = new URL(url);
    const path = decodeURIComponent(p.pathname.replace(/\/$/, ""));
    const slug = path.includes("/") ? path.split("/").pop() : path;
    const t = slug.replace(/[-_]/g, " ").trim();
    if (!t) return url;
    return t.charAt(0).toUpperCase() + t.slice(1);
  } catch {
    return url;
  }
}
