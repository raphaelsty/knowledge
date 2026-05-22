// Full-parity port of sources/github/stars.py.
//
// Walks GitHub's `/users/{u}/starred` newest-first with the page-level
// early exit, then for every NEW repo fetches the README from
// raw.githubusercontent.com and extracts clean paragraph text until
// we have ≥50 tokens. raw.githubusercontent.com sets
// `Access-Control-Allow-Origin: *` for public files, so this works
// from the browser.

import { fetchJson, fetchText, sleep } from "../utils/http.js";

const DELAY_MS = 100;

// ─────────────────────────────────────────────────────────────────
// README extraction (mirror of `get_readme_text_by_token_count`)
// ─────────────────────────────────────────────────────────────────

const README_REPO_RE = /github\.com\/([^/]+)\/([^/]+)/;
const HTML_TAG_RE = /<[^>]+>/g;
const HR_RE = /^[-*_]{3,}$/;
// Python keeps [A-Za-z0-9 .,?!'-] only. Same here.
const CLEAN_CHARS_RE = /[^a-zA-Z0-9\s.,?!'\-]/g;
const WHITESPACE_RE = /\s+/g;

/** `https://github.com/USER/REPO` → `[USER, REPO]` or null. */
function parseRepoPath(githubUrl) {
  const m = README_REPO_RE.exec(githubUrl || "");
  if (!m) return null;
  return [m[1], m[2]];
}

/**
 * Fetch a repo's README.md from raw.githubusercontent.com, trying
 * `main` then `master`. Returns the raw text or null. 404s are fine
 * — lots of repos don't have a README at a top-level path.
 */
async function fetchReadmeText(user, repo) {
  for (const branch of ["main", "master"]) {
    const url = `https://raw.githubusercontent.com/${encodeURIComponent(
      user,
    )}/${encodeURIComponent(repo)}/${branch}/README.md`;
    try {
      return await fetchText(url, { timeoutMs: 15000 });
    } catch {
      // try next branch
    }
  }
  return null;
}

/**
 * Extract clean paragraph-ish text from a raw README markdown body.
 * Skips markdown artifacts (headings, list items, blockquotes,
 * image/link-only lines, horizontal rules) and stops once we've
 * collected ≥ `minTokens` words. Matches Python's
 * `get_readme_text_by_token_count` line-for-line.
 */
function extractReadmeText(rawReadme, minTokens = 50) {
  if (!rawReadme) return null;
  const noHtml = rawReadme.replace(HTML_TAG_RE, "");
  let collected = "";
  for (const line of noHtml.split(/\r?\n/)) {
    const stripped = line.trim();
    if (!stripped) continue;
    if (stripped.startsWith("#")) continue;
    if (
      stripped.startsWith("* ") ||
      stripped.startsWith("- ") ||
      stripped.startsWith("+ ")
    )
      continue;
    if (stripped.startsWith(">")) continue;
    // Lines that start with `[` and end with `)` are image-only or link-only markdown.
    if (stripped.startsWith("[") && stripped.endsWith(")")) continue;
    if (HR_RE.test(stripped)) continue;
    collected += stripped + " ";
    if (collected.split(/\s+/).filter(Boolean).length >= minTokens) break;
  }
  if (!collected) return null;
  const cleaned = collected.replace(CLEAN_CHARS_RE, "");
  return cleaned.replace(WHITESPACE_RE, " ").trim() || null;
}

async function readmeTextFor(githubUrl, minTokens = 50) {
  const parsed = parseRepoPath(githubUrl);
  if (!parsed) return null;
  const [user, repo] = parsed;
  const raw = await fetchReadmeText(user, repo);
  return extractReadmeText(raw, minTokens);
}

// ─────────────────────────────────────────────────────────────────
// Public fetcher
// ─────────────────────────────────────────────────────────────────

export async function stars({
  user,
  perPage = 100,
  limit = 100,
  existingUrls = null,
}) {
  // Phase 1: paginate starred repos newest-first with page-level
  // early exit. The `vnd.github.star+json` Accept header changes
  // the response shape from `[repo, …]` to `[{starred_at, repo}, …]`
  // so we get a per-star ISO-8601 timestamp — without this every
  // star ends up dated "today" and 1k+ docs collide on the same day
  // in the search panel. `limit` matches the Python fetcher: max
  // pages (each `per_page` wide) we'll walk.
  const pages = [];
  for (let page = 1; page <= limit; page++) {
    let items;
    try {
      items = await fetchJson(
        `https://api.github.com/users/${encodeURIComponent(user)}/starred?per_page=${perPage}&page=${page}`,
        { headers: { Accept: "application/vnd.github.star+json" } },
      );
    } catch (err) {
      console.warn("[sync] github stars error:", err);
      break;
    }
    if (!Array.isArray(items) || items.length === 0) break;
    // Normalise both shapes — if a CDN strips our Accept header we
    // still want to ingest the repos, just without star timestamps.
    const normalized = items.map((it) =>
      it && typeof it === "object" && it.repo && typeof it.repo === "object"
        ? it
        : { starred_at: null, repo: it },
    );
    pages.push(...normalized);
    if (existingUrls) {
      const newInPage = normalized.filter(
        (it) =>
          it.repo && it.repo.html_url && !existingUrls.has(it.repo.html_url),
      ).length;
      if (newInPage === 0) break;
    }
    await sleep(DELAY_MS);
  }

  // Phase 2: build docs only for URLs the user doesn't already
  // have. The earlier version pushed every starred repo into `out`
  // and only skipped the README fetch for known URLs — so the
  // fetcher reported "100 fetched" even when nothing was new and
  // the user always saw a full page on every sync.
  const todayMs = Date.now();
  const out = {};
  let rank = 0;
  for (const item of pages) {
    const repository = item && item.repo;
    if (!repository || !repository.url) continue;
    const url = repository.html_url;
    if (!url || out[url]) continue;
    if (existingUrls && existingUrls.has(url)) {
      // Still increment rank so the synthesized fallback below
      // stays monotonic even when most of the page is skipped.
      rank++;
      continue;
    }

    // Per-star date — prefer the API's `starred_at`, fall back to a
    // synthesized date that walks one day back per rank so even
    // synthesized values preserve newest-first order.
    const starredAt = item.starred_at;
    const date =
      typeof starredAt === "string" && starredAt.length >= 10
        ? starredAt.slice(0, 10)
        : new Date(todayMs - rank * 86400000).toISOString().slice(0, 10);
    rank++;

    const topics = (repository.topics || []).map((t) => t.toLowerCase());
    const lang = repository.language ? repository.language.toLowerCase() : null;
    const tags = Array.from(new Set(lang ? [...topics, lang] : topics));
    const description = repository.description || "";

    const readmeText = await readmeTextFor(url, 50);

    out[url] = {
      date,
      title: repository.name || "",
      summary: readmeText ? `${description} \n ${readmeText}` : description,
      tags,
      source: "github",
    };
  }
  return out;
}
