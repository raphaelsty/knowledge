// JS port of sources/huggingface/likes.py — same behaviour from the
// browser, so the in-page sync produces the rich summaries the Python
// pipeline does.
//
// Public Hub endpoints we hit (all CORS-friendly):
//   GET https://huggingface.co/api/users/{u}/likes
//   GET https://huggingface.co/api/{models|datasets|spaces}/{repo_id}
//        ?full=true   ← surfaces cardData (parsed YAML frontmatter)
//   GET https://huggingface.co/{repo_id}/raw/main/README.md      (model)
//   GET https://huggingface.co/datasets/{repo_id}/raw/main/README.md
//   GET https://huggingface.co/spaces/{repo_id}/raw/main/README.md
//
// The /api/repo endpoint returns `lastModified` ISO strings, the
// `cardData` object, `tags`, `pipeline_tag`, `library_name` —
// everything the Python flow uses to build a structured fact sheet.
//
// We mirror the Python summary recipe exactly:
//   1. "{Pipeline} {kind} by {org}, built with {library}, derived
//      from {base}, license: {id}." — already informative w/o prose.
//   2. + first substantive prose paragraph from the README, with
//      badges / images / HTML / blockquotes / boilerplate headings
//      stripped.
//   3. Sentence-aware truncation at SUMMARY_BUDGET chars.
//   4. Tags = ["huggingface", kind] + pipeline + library + Hub /
//      YAML topic tags, deduped, capped at 12.
//   5. Date = repo's `lastModified`, fallback to today.

import { fetchJson, fetchText } from "../utils/http.js";
import { isoDate } from "../utils/doc.js";

const SUMMARY_BUDGET = 700;
const TAG_CAP = 12;

// Lines we drop wholesale: badges, images, HTML, comments, raw URLs.
const BADGE_LINE_RE = /^\s*(\[!\[|!\[|<[a-z!/]|https?:\/\/|<!--)/i;
// Section headings whose entire text is just template boilerplate.
const BOILERPLATE_HEADING_RE =
  /^model\s+card(\s+for\s+.+)?$|^dataset\s+card(\s+for\s+.+)?$|^card\s+for\s+.+$/i;
// Inline markdown to clean inside a kept paragraph.
const INLINE_LINK_RE = /\[([^\]]+)\]\([^)]+\)/g;
const INLINE_IMG_RE = /!\[[^\]]*\]\([^)]+\)/g;
const INLINE_HTML_RE = /<[^>]+>/g;
// Tag categories that are filter scaffolding, not topical descriptors.
const NOISE_TAG_PREFIXES = [
  "license:",
  "region:",
  "arxiv:",
  "dataset:",
  "base_model:",
];

// ───────────────────────────────────────────────────────────────────
// Public entry
// ───────────────────────────────────────────────────────────────────

export async function likes({ username, existingUrls = null }) {
  const out = {};
  let items;
  try {
    items = await fetchJson(
      `https://huggingface.co/api/users/${encodeURIComponent(username)}/likes`,
    );
  } catch (err) {
    console.warn("[sync] hf likes error:", err);
    return out;
  }
  if (!Array.isArray(items)) return out;

  // Limit concurrency so a 200-like account doesn't fan out 600
  // simultaneous fetches (likes-list + repo-info + README per item).
  const CONCURRENCY = 6;
  const queue = items.slice();
  const workers = Array.from(
    { length: Math.min(CONCURRENCY, queue.length) },
    () => worker(),
  );
  await Promise.all(workers);

  return out;

  async function worker() {
    while (queue.length) {
      const item = queue.shift();
      const repo = item.repo || {};
      const repoId = repo.name || "";
      const kind = repo.type || "model"; // "model" | "dataset" | "space"
      if (!repoId) continue;
      const url = repoUrl(repoId, kind);
      if (existingUrls && existingUrls.has(url)) continue;
      try {
        const doc = await processEntry(repoId, kind, item);
        out[url] = doc;
      } catch (err) {
        // Don't let one bad repo poison the run — log and move on.
        console.warn(`[sync] hf ${kind} ${repoId} failed:`, err);
      }
    }
  }
}

async function processEntry(repoId, kind, likeItem) {
  const info = await fetchRepoInfo(repoId, kind);
  const cardData = pickCardData(info);
  const readme = await fetchReadme(repoId, kind);
  const repoShort = repoId.split("/").pop();
  return {
    title: `HuggingFace ${kind}: ${repoShort}`,
    summary: buildSummary(repoId, kind, info, cardData, readme),
    date: extractDate(info, likeItem),
    tags: buildTags(kind, info, cardData),
    source: "huggingface",
  };
}

function repoUrl(repoId, kind) {
  if (kind === "dataset") return `https://huggingface.co/datasets/${repoId}`;
  if (kind === "space") return `https://huggingface.co/spaces/${repoId}`;
  return `https://huggingface.co/${repoId}`;
}

function readmeUrl(repoId, kind) {
  if (kind === "dataset")
    return `https://huggingface.co/datasets/${repoId}/raw/main/README.md`;
  if (kind === "space")
    return `https://huggingface.co/spaces/${repoId}/raw/main/README.md`;
  return `https://huggingface.co/${repoId}/raw/main/README.md`;
}

// ───────────────────────────────────────────────────────────────────
// Hub API
// ───────────────────────────────────────────────────────────────────

async function fetchRepoInfo(repoId, kind) {
  const segment =
    kind === "dataset" ? "datasets" : kind === "space" ? "spaces" : "models";
  const apiUrl = `https://huggingface.co/api/${segment}/${repoId}?full=true`;
  try {
    return await fetchJson(apiUrl);
  } catch {
    return null;
  }
}

function pickCardData(info) {
  if (!info) return {};
  const cd = info.cardData || info.card_data;
  return cd && typeof cd === "object" ? cd : {};
}

async function fetchReadme(repoId, kind) {
  try {
    return await fetchText(readmeUrl(repoId, kind));
  } catch {
    return "";
  }
}

// ───────────────────────────────────────────────────────────────────
// README parsing — first substantive paragraph
// ───────────────────────────────────────────────────────────────────

function stripFrontmatter(content) {
  if (!content || !content.trimStart().startsWith("---")) return [content, {}];
  const m = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
  if (!m) return [content, {}];
  // Browsers don't ship a YAML parser. We don't need full YAML to
  // extract the handful of scalar/list fields the summary uses —
  // a lightweight key/value scanner over the frontmatter block is
  // enough for `pipeline_tag`, `library_name`, `license`,
  // `base_model`, and `tags`. Anything else falls back to API data.
  const fm = parseLightYaml(m[1]);
  return [content.slice(m[0].length), fm];
}

/**
 * Tiny "good-enough YAML" parser. Handles the shapes HF cards use:
 *   key: value
 *   key: [a, b, c]
 *   key:
 *     - a
 *     - b
 *
 * Returns an object with string or string[] values. Nested mappings
 * are skipped (not needed for the fields we care about).
 */
function parseLightYaml(block) {
  const out = {};
  const lines = block.split("\n");
  let listKey = null;
  let listAcc = null;
  for (const raw of lines) {
    if (!raw.trim()) continue;
    if (listKey && /^\s+-\s+/.test(raw)) {
      listAcc.push(raw.replace(/^\s+-\s+/, "").trim());
      continue;
    }
    listKey = null;
    listAcc = null;
    const m = raw.match(/^([A-Za-z0-9_\-]+)\s*:\s*(.*)$/);
    if (!m) continue;
    const key = m[1];
    const val = m[2].trim();
    if (val === "") {
      listKey = key;
      listAcc = [];
      out[key] = listAcc;
      continue;
    }
    if (val.startsWith("[") && val.endsWith("]")) {
      out[key] = val
        .slice(1, -1)
        .split(",")
        .map((s) => s.trim().replace(/^["']|["']$/g, ""))
        .filter(Boolean);
      continue;
    }
    out[key] = val.replace(/^["']|["']$/g, "");
  }
  return out;
}

function firstParagraph(body) {
  if (!body) return "";
  const lines = body.split("\n");
  let inCode = false;
  const paragraphs = [];
  let current = [];
  const flush = () => {
    if (current.length) {
      paragraphs.push(current);
      current = [];
    }
  };
  for (const raw of lines) {
    const stripped = raw.trim();
    if (stripped.startsWith("```")) {
      inCode = !inCode;
      flush();
      continue;
    }
    if (inCode) continue;
    if (!stripped) {
      flush();
      continue;
    }
    if (stripped.startsWith(">")) continue;
    if (BADGE_LINE_RE.test(stripped)) continue;
    if (stripped.startsWith("#")) {
      const headingText = stripped.replace(/^#+\s*/, "").trim();
      if (!headingText || BOILERPLATE_HEADING_RE.test(headingText)) continue;
      current.push(headingText);
      continue;
    }
    current.push(stripped);
  }
  flush();
  for (const para of paragraphs) {
    let text = para.join(" ");
    text = text.replace(INLINE_IMG_RE, "");
    text = text.replace(INLINE_LINK_RE, "$1");
    text = text.replace(INLINE_HTML_RE, "");
    text = text.replace(/\s+/g, " ").replace(/^[\s\-—–·]+|[\s\-—–·]+$/g, "");
    if (isSubstantive(text)) return text;
  }
  return "";
}

function isSubstantive(text) {
  if (!text) return false;
  const letters = text.match(/[A-Za-z]/g) || [];
  if (letters.length) {
    const upper = letters.filter((c) => c === c.toUpperCase()).length;
    if (upper / letters.length > 0.8) return false;
  }
  return text.split(/\s+/).filter(Boolean).length >= 6;
}

// ───────────────────────────────────────────────────────────────────
// Summary / tag composition
// ───────────────────────────────────────────────────────────────────

function buildSummary(repoId, kind, info, cardData, readme) {
  const [body, frontmatter] = stripFrontmatter(readme);
  const prose = firstParagraph(body);
  const org = repoId.includes("/") ? repoId.split("/")[0] : "";

  const pipeline =
    (info && info.pipeline_tag) ||
    cardData.pipeline_tag ||
    frontmatter.pipeline_tag;
  const library =
    (info && info.library_name) ||
    cardData.library_name ||
    frontmatter.library_name;
  const license = cardData.license || frontmatter.license;
  let baseModels = cardData.base_model || frontmatter.base_model || [];
  if (typeof baseModels === "string") baseModels = [baseModels];

  const descriptorBits = [];
  if (pipeline) descriptorBits.push(String(pipeline).replace(/-/g, " "));
  descriptorBits.push(kind);
  const descriptor = descriptorBits.join(" ");

  const metaClauses = [];
  if (org) metaClauses.push(`by ${org}`);
  if (library) metaClauses.push(`built with ${library}`);
  if (Array.isArray(baseModels) && baseModels.length) {
    const base = baseModels[0];
    if (typeof base === "string" && base && base !== repoId) {
      metaClauses.push(`derived from ${base}`);
    }
  }
  if (license) metaClauses.push(`license: ${license}`);

  let head = capitalize(descriptor);
  if (metaClauses.length) head += " " + metaClauses.join(", ");
  if (!head.endsWith(".")) head += ".";

  const joined = prose ? `${head} ${prose}` : head;
  return sentenceTruncate(joined, SUMMARY_BUDGET);
}

function capitalize(s) {
  if (!s) return "";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function sentenceTruncate(text, budget) {
  text = (text || "").replace(/\s+/g, " ").trim();
  if (text.length <= budget) return text;
  const cut = text.slice(0, budget);
  for (const stop of [". ", "! ", "? "]) {
    const i = cut.lastIndexOf(stop);
    if (i >= budget * 0.5) return cut.slice(0, i + 1).trimEnd();
  }
  return cut.replace(/[\s,;:\-—]+$/, "") + "…";
}

function buildTags(kind, info, cardData) {
  const tags = ["huggingface", kind];
  const pipeline = (info && info.pipeline_tag) || cardData.pipeline_tag;
  if (pipeline) tags.push(String(pipeline));
  const library = (info && info.library_name) || cardData.library_name;
  if (library) tags.push(String(library));

  const apiTags = (info && Array.isArray(info.tags) ? info.tags : []) || [];
  let yamlTags = cardData.tags || [];
  if (typeof yamlTags === "string") yamlTags = [yamlTags];
  if (!Array.isArray(yamlTags)) yamlTags = [];

  for (const raw of [...apiTags, ...yamlTags]) {
    if (typeof raw !== "string") continue;
    const t = raw.trim().toLowerCase();
    if (!t) continue;
    if (NOISE_TAG_PREFIXES.some((p) => t.startsWith(p))) continue;
    tags.push(t);
  }

  const seen = new Set();
  const out = [];
  for (const t of tags) {
    const tl = t.toLowerCase();
    if (seen.has(tl)) continue;
    seen.add(tl);
    out.push(tl);
    if (out.length >= TAG_CAP) break;
  }
  return out;
}

function extractDate(info, likeItem) {
  if (info) {
    const last = info.lastModified || info.last_modified;
    if (typeof last === "string" && last) return isoDate(last);
  }
  if (likeItem && likeItem.createdAt) return isoDate(likeItem.createdAt);
  return new Date().toISOString().slice(0, 10);
}
