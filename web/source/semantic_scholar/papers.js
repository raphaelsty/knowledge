// JS port of sources/semantic_scholar/papers.py. Uses the public
// Graph API — allows CORS, 100 req / 5 min without a key.

import { fetchJson, sleep } from "../utils/http.js";
import { truncate } from "../utils/doc.js";

const API = "https://api.semanticscholar.org/graph/v1";
const FIELDS =
  "title,year,citationCount,externalIds,abstract,url,venue,publicationTypes";
const DELAY_MS = 1000;
const PAGE_SIZE = 100;

const NON_PUBLICATION_TYPES = new Set(["Dataset", "Software", "Book"]);
const SOFTWARE_RELEASE_TITLE_RE = /^[\w.-]+\/[\w.-]+:\s*v?\d+(?:\.\d+)+/;

function isNoise(paper) {
  for (const t of paper.publicationTypes || []) {
    if (NON_PUBLICATION_TYPES.has(t)) return true;
  }
  const title = (paper.title || "").trim();
  if (SOFTWARE_RELEASE_TITLE_RE.test(title)) return true;
  return false;
}

// arXiv > DOI > Semantic Scholar page. Same precedence as Python.
function canonicalUrl(paper) {
  const ext = paper.externalIds || {};
  if (ext.ArXiv) return `https://arxiv.org/abs/${ext.ArXiv}`;
  if (ext.DOI) return `https://doi.org/${ext.DOI}`;
  return paper.url || "";
}

async function resolveAuthorId({ authorId, authorName }) {
  if (authorId) return authorId;
  if (!authorName) return null;
  try {
    const data = await fetchJson(
      `${API}/author/search?query=${encodeURIComponent(authorName)}&limit=1`,
    );
    const results = data.data || [];
    if (results.length > 0) return String(results[0].authorId);
  } catch (err) {
    console.warn("[sync] S2 author search failed:", err);
  }
  return null;
}

export async function papers({
  authorId = null,
  authorName = null,
  maxPapers = 300,
  minCitations = 0,
  existingUrls = null,
}) {
  const resolvedId = await resolveAuthorId({ authorId, authorName });
  if (!resolvedId) return {};

  const out = {};
  let offset = 0;
  while (offset < maxPapers) {
    let result;
    try {
      result = await fetchJson(
        `${API}/author/${resolvedId}/papers?fields=${FIELDS}&limit=${PAGE_SIZE}&offset=${offset}`,
      );
    } catch (err) {
      console.warn("[sync] S2 error at offset", offset, err);
      break;
    }
    const page = result.data || [];
    if (page.length === 0) break;

    for (const paper of page) {
      if (isNoise(paper)) continue;
      const citations = paper.citationCount || 0;
      if (minCitations > 0 && citations < minCitations) continue;
      const canonical = canonicalUrl(paper);
      if (!canonical) continue;
      if (existingUrls && existingUrls.has(canonical)) continue;
      if (out[canonical]) continue;
      const title = paper.title || "";
      if (!title) continue;
      const year = paper.year;
      const venue = paper.venue || "";
      const abstract = truncate((paper.abstract || "").trim(), 250);

      let summary = abstract;
      if (!summary) {
        const parts = [];
        if (venue) parts.push(venue);
        if (citations) parts.push(`${citations} citations`);
        summary = parts.join(". ");
      }

      out[canonical] = {
        title,
        summary,
        date: year ? `${year}-01-01` : "",
        tags: ["scholar"],
        source: "scholar",
      };
    }

    offset += page.length;
    if (page.length < PAGE_SIZE) break;
    await sleep(DELAY_MS);
  }
  return out;
}
