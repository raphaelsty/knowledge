// JS port of sources/wikipedia/references.py.
//
// The MediaWiki action API is CORS-permissive when `origin=*` is
// passed on the query string, and doesn't need authentication for
// public pages.

import { fetchJson } from "../utils/http.js";

const API = "https://en.wikipedia.org/w/api.php";
const SKIP_DOMAINS = [
  "wikipedia.org",
  "wikimedia.org",
  "wikidata.org",
  "web.archive.org",
  "doi.org",
];

export async function references({ pages, existingUrls = null }) {
  const list = Array.isArray(pages) ? pages : [pages];
  const out = {};

  for (const pageTitle of list) {
    let result;
    try {
      const url = `${API}?action=query&titles=${encodeURIComponent(
        pageTitle,
      )}&prop=extlinks&ellimit=500&format=json&origin=*`;
      result = await fetchJson(url);
    } catch (err) {
      console.warn("[sync] wikipedia error:", err);
      continue;
    }
    const wpages = (result.query && result.query.pages) || {};
    for (const pageData of Object.values(wpages)) {
      if (pageData.missing !== undefined) continue;
      const extlinks = pageData.extlinks || [];
      for (const link of extlinks) {
        const extUrl = link["*"] || link.url || "";
        if (!extUrl) continue;
        let domain = "";
        try {
          domain = new URL(extUrl).hostname.toLowerCase();
        } catch {
          continue;
        }
        if (SKIP_DOMAINS.some((d) => domain.endsWith(d))) continue;
        if (existingUrls && existingUrls.has(extUrl)) continue;
        if (out[extUrl]) continue;
        const pretty = pageTitle.replace(/_/g, " ");
        out[extUrl] = {
          title: `Wikipedia: ${pretty}`,
          summary: `Referenced on the Wikipedia page for ${pretty}`,
          date: "",
          tags: ["wikipedia"],
          source: "wikipedia",
        };
      }
    }
  }
  return out;
}
