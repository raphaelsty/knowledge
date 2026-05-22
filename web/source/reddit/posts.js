// JS port of sources/reddit/posts.py. Reddit's public JSON endpoint
// is frequently CORS-blocked from third-party origins; the fetch will
// reject and the orchestrator records the failure per-source.

import { fetchJson, sleep } from "../utils/http.js";
import { truncate } from "../utils/doc.js";

const DELAY_MS = 2000;
const REDDIT_DOMAINS = new Set([
  "reddit.com",
  "www.reddit.com",
  "old.reddit.com",
  "i.reddit.com",
  "v.redd.it",
  "i.redd.it",
  "preview.redd.it",
]);

function isRedditUrl(url) {
  try {
    return REDDIT_DOMAINS.has(
      new URL(url).hostname.toLowerCase().replace(/\.$/, ""),
    );
  } catch {
    return false;
  }
}

async function paginate(username, endpoint, maxPages) {
  const out = [];
  let after = null;
  for (let page = 0; page < maxPages; page++) {
    // `www.reddit.com` blanket-403s server-side fetches (which is
    // what the proxy fallback does) — `old.reddit.com` exposes the
    // exact same JSON without the bot blocker.
    let url = `https://old.reddit.com/user/${encodeURIComponent(
      username,
    )}/${endpoint}.json?limit=100&sort=new&raw_json=1`;
    if (after) url += `&after=${after}`;
    let result;
    try {
      result = await fetchJson(url);
    } catch (err) {
      console.warn("[sync] reddit error:", err);
      break;
    }
    const children = (result.data || {}).children || [];
    if (children.length === 0) break;
    for (const c of children) out.push(c.data);
    after = (result.data || {}).after;
    if (!after) break;
    await sleep(DELAY_MS);
  }
  return out;
}

export async function posts({
  username,
  maxPages = 5,
  includeComments = true,
  existingUrls = null,
}) {
  const out = {};

  const subs = await paginate(username, "submitted", maxPages);
  for (const item of subs) {
    const url = item.url || "";
    if (!url || isRedditUrl(url)) continue;
    if (existingUrls && existingUrls.has(url)) continue;
    if (out[url]) continue;
    const title = item.title || "";
    const subreddit = item.subreddit || "";
    const created = item.created_utc || 0;
    const date = created
      ? new Date(created * 1000).toISOString().slice(0, 10)
      : "";
    const selftext = truncate((item.selftext || "").trim(), 200);
    const summary =
      selftext || (subreddit ? `r/${subreddit}: ${title}` : title);
    out[url] = {
      title,
      summary,
      date,
      tags: ["reddit"],
      source: "reddit",
    };
  }

  if (includeComments) {
    const seenLinks = new Set();
    const cmts = await paginate(username, "comments", maxPages);
    for (const item of cmts) {
      const linkUrl = item.link_url || "";
      if (!linkUrl || isRedditUrl(linkUrl)) continue;
      if (seenLinks.has(linkUrl) || out[linkUrl]) continue;
      if (existingUrls && existingUrls.has(linkUrl)) continue;
      seenLinks.add(linkUrl);
      const title = item.link_title || "";
      const subreddit = item.subreddit || "";
      const created = item.created_utc || 0;
      const date = created
        ? new Date(created * 1000).toISOString().slice(0, 10)
        : "";
      // Python does `.strip()` + truncate; it does NOT collapse
      // internal whitespace, so multi-line comments keep their line
      // breaks through to the pipeline's cleaning pass.
      const body = truncate((item.body || "").trim(), 200);
      const summary = body || (subreddit ? `r/${subreddit}: ${title}` : title);
      out[linkUrl] = {
        title,
        summary,
        date,
        tags: ["reddit"],
        source: "reddit",
      };
    }
  }

  return out;
}
