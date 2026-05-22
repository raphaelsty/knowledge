// JS port of sources/stackoverflow/answers.py::Questions.

import { fetchJson, sleep } from "../utils/http.js";
import { stripHtml, collapseWhitespace, truncate } from "../utils/doc.js";

const API = "https://api.stackexchange.com/2.3";
const DELAY_MS = 2000;

async function resolveUserId({ userId, username, site }) {
  if (userId) return userId;
  if (!username) return null;
  try {
    const data = await fetchJson(
      `${API}/users?order=desc&sort=reputation&inname=${encodeURIComponent(
        username,
      )}&site=${site}&pagesize=1`,
    );
    const items = data.items || [];
    if (items.length > 0) return items[0].user_id;
  } catch (err) {
    console.warn("[sync] SE user search failed:", err);
  }
  return null;
}

export async function questions({
  userId = null,
  username = null,
  site = "stackoverflow",
  maxPages = 5,
  minScore = 0,
  existingUrls = null,
}) {
  const uid = await resolveUserId({ userId, username, site });
  if (!uid) return {};

  const out = {};
  for (let page = 1; page <= maxPages; page++) {
    let result;
    try {
      result = await fetchJson(
        `${API}/users/${uid}/questions?order=desc&sort=votes&site=${site}&pagesize=100&page=${page}&filter=withbody`,
      );
    } catch (err) {
      console.warn("[sync] SE questions error:", err);
      break;
    }
    for (const q of result.items || []) {
      const score = q.score || 0;
      if (score < minScore) continue;
      const link = q.link;
      if (!link) continue;
      if (existingUrls && existingUrls.has(link)) continue;
      if (out[link]) continue;
      const title = (q.title || "").trim();
      const body = q.body || "";
      const summary =
        truncate(collapseWhitespace(stripHtml(body)), 200) || title;
      const created = q.creation_date || 0;
      const date = created
        ? new Date(created * 1000).toISOString().slice(0, 10)
        : "";
      const tags = q.tags || [];
      out[link] = {
        title,
        summary,
        date,
        tags: ["stackoverflow", ...tags.slice(0, 3)],
        source: "stackoverflow",
      };
    }
    if (!result.has_more) break;
    await sleep(DELAY_MS);
  }
  return out;
}
