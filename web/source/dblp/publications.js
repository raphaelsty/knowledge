// JS port of sources/dblp/publications.py. DBLP's public search
// endpoint sets CORS and returns JSON when we pass `format=json`.

import { fetchJson, sleep } from "../utils/http.js";
import { truncate } from "../utils/doc.js";

const API = "https://dblp.org/search/publ/api";
const DELAY_MS = 1000;
const PAGE_SIZE = 100;
const ARXIV_RE = /(\d{4}\.\d{4,5})/;

function canonicalUrl(info) {
  const ee = info.ee || "";
  if (ee.includes("arxiv.org")) {
    const m = ee.match(ARXIV_RE);
    if (m) return `https://arxiv.org/abs/${m[1]}`;
  }
  if (ee.includes("doi.org")) return ee;
  return info.url || ee;
}

export async function publications({
  author,
  maxResults = 200,
  existingUrls = null,
}) {
  const out = {};
  let offset = 0;
  while (offset < maxResults) {
    const url = `${API}?q=author%3A${encodeURIComponent(
      author,
    )}&h=${PAGE_SIZE}&f=${offset}&format=json`;
    let result;
    try {
      result = await fetchJson(url);
    } catch (err) {
      console.warn("[sync] DBLP error:", err);
      break;
    }
    const hits = ((result.result || {}).hits || {}).hit || [];
    if (!Array.isArray(hits) || hits.length === 0) break;

    if (existingUrls) {
      const newInPage = hits.filter((h) => {
        const c = canonicalUrl(h.info || {});
        return c && !existingUrls.has(c);
      }).length;
      if (newInPage === 0) break;
    }

    for (const hit of hits) {
      const info = hit.info || {};
      // Python uses `.rstrip(".")` which strips a RUN of trailing
      // periods, not just one. Some DBLP titles end with "...".
      const title = (info.title || "").replace(/\.+$/, "");
      if (!title) continue;
      const canonical = canonicalUrl(info);
      if (!canonical) continue;
      if (existingUrls && existingUrls.has(canonical)) continue;
      if (out[canonical]) continue;

      const year = info.year || "";
      const venue = info.venue || "";
      let authorsRaw = (info.authors || {}).author || [];
      if (!Array.isArray(authorsRaw)) authorsRaw = [authorsRaw];
      // Python: `a.get("text", a) if isinstance(a, dict) else str(a)`.
      // When the dict lacks a "text" key, Python falls back to the
      // dict itself (later stringified into something like `{id: …}`).
      // We mirror that — real DBLP records always have "text", but
      // matching the edge-case behaviour keeps the ports byte-identical.
      const authors = authorsRaw
        .slice(0, 5)
        .map((a) => {
          if (a && typeof a === "object") {
            return a.text !== undefined ? String(a.text) : String(a);
          }
          return String(a);
        })
        .join(", ");

      const parts = [];
      if (authors) parts.push(authors);
      if (venue) parts.push(venue);
      const summary = truncate(parts.join(". "), 250);

      out[canonical] = {
        title,
        summary,
        date: year ? `${year}-01-01` : "",
        tags: ["scholar"],
        source: "scholar",
      };
    }

    offset += hits.length;
    if (hits.length < PAGE_SIZE) break;
    await sleep(DELAY_MS);
  }
  return out;
}
