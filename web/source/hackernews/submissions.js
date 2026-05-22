// JS port of sources/hackernews/submissions.py.
//
// Walks the Algolia HN Search API for stories submitted by a
// username, newest-first, with the same page-level early exit.
//
// Upvotes (sources/hackernews/upvotes.py) is deliberately omitted —
// it needs the HN password that lives encrypted server-side.

import { fetchJson, sleep } from "../utils/http.js";
import { isoDate } from "../utils/doc.js";

const ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date";
const DELAY_MS = 500;

export async function submissions({
  username,
  maxItems = 500,
  existingUrls = null,
}) {
  const out = {};
  let page = 0;
  let total = 0;
  while (total < maxItems) {
    const url = `${ALGOLIA}?tags=story,author_${encodeURIComponent(
      username,
    )}&hitsPerPage=100&page=${page}`;
    let result;
    try {
      result = await fetchJson(url);
    } catch (err) {
      console.warn("[sync] HN submissions algolia error:", err);
      break;
    }
    const hits = result.hits || [];
    if (hits.length === 0) break;

    if (existingUrls) {
      const newInPage = hits.filter(
        (h) => h.url && !existingUrls.has(h.url),
      ).length;
      if (newInPage === 0) break;
    }

    for (const hit of hits) {
      const storyUrl = hit.url;
      if (!storyUrl) continue;
      if (existingUrls && existingUrls.has(storyUrl)) continue;
      if (out[storyUrl]) continue;
      const title = hit.title || "";
      const created = isoDate(hit.created_at);
      const points = hit.points || 0;
      const nComments = hit.num_comments || 0;
      const summary =
        points || nComments
          ? `${points} points, ${nComments} comments on HN`
          : "";
      out[storyUrl] = {
        title,
        summary,
        date: created,
        tags: ["hackernews"],
        source: "hackernews",
      };
    }

    total += hits.length;
    page += 1;
    if (page >= (result.nbPages || 0)) break;
    await sleep(DELAY_MS);
  }
  return out;
}
