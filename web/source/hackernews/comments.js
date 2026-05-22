// JS port of sources/hackernews/comments.py.
//
// Algolia HN Search for all comments by a user → parent story URL of
// each comment becomes a doc. The parent-story's title is used as the
// doc title; the comment text (stripped of HTML) becomes the summary.

import { fetchJson, sleep } from "../utils/http.js";
import {
  isoDate,
  stripHtml,
  collapseWhitespace,
  truncate,
} from "../utils/doc.js";

const ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search_by_date";
const DELAY_MS = 500;
const PER_PAGE = 100;

export async function comments({
  username,
  maxItems = 500,
  existingUrls = null,
}) {
  const out = {};
  const seenStories = new Set();
  let page = 0;
  let totalFetched = 0;

  while (totalFetched < maxItems) {
    const url = `${ALGOLIA_SEARCH}?tags=comment,author_${encodeURIComponent(
      username,
    )}&hitsPerPage=${PER_PAGE}&page=${page}`;
    let result;
    try {
      result = await fetchJson(url);
    } catch (err) {
      console.warn("[sync] HN comments algolia error:", err);
      break;
    }
    const hits = result.hits || [];
    if (hits.length === 0) break;

    if (existingUrls) {
      const newInPage = hits.filter(
        (h) => h.story_url && !existingUrls.has(h.story_url),
      ).length;
      if (newInPage === 0) break;
    }

    for (const hit of hits) {
      const storyId = hit.story_id;
      if (!storyId || seenStories.has(storyId)) continue;
      seenStories.add(storyId);
      const storyUrl = hit.story_url;
      if (!storyUrl) continue;
      if (existingUrls && existingUrls.has(storyUrl)) continue;

      const date = isoDate(hit.created_at);
      const commentText = hit.comment_text || "";
      const summary = truncate(collapseWhitespace(stripHtml(commentText)), 200);

      out[storyUrl] = {
        title: hit.story_title || "",
        summary,
        date,
        tags: ["hackernews"],
        source: "hackernews",
      };
    }

    totalFetched += hits.length;
    page += 1;
    if (page >= (result.nbPages || 0)) break;
    await sleep(DELAY_MS);
  }
  return out;
}
