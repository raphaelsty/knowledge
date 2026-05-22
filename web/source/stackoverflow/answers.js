// JS port of sources/stackoverflow/answers.py (public Stack Exchange
// API). Browsers auto-decompress gzip, so the Python version's manual
// gzip.decompress fallback isn't needed here.
//
// Favorites aren't ported — they need the OAuth access_token that
// lives encrypted server-side.

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
      )}&site=${site}&pagesize=3`,
    );
    const items = data.items || [];
    if (items.length > 0) return items[0].user_id;
  } catch (err) {
    console.warn("[sync] SE user search failed:", err);
  }
  return null;
}

export async function answers({
  userId = null,
  username = null,
  site = "stackoverflow",
  maxPages = 5,
  minScore = 1,
  existingUrls = null,
}) {
  const uid = await resolveUserId({ userId, username, site });
  if (!uid) return {};

  const answersByQid = new Map();

  for (let page = 1; page <= maxPages; page++) {
    let result;
    try {
      result = await fetchJson(
        `${API}/users/${uid}/answers?order=desc&sort=votes&site=${site}&pagesize=100&page=${page}&filter=withbody`,
      );
    } catch (err) {
      console.warn("[sync] SE answers error:", err);
      break;
    }
    const items = result.items || [];
    if (items.length === 0) break;

    for (const a of items) {
      const score = a.score || 0;
      if (score < minScore) continue;
      const qid = a.question_id;
      if (!qid || answersByQid.has(qid)) continue;
      // Python hardcodes stackoverflow.com/q/{qid} regardless of the
      // `site` param (see sources/stackoverflow/answers.py:132). We
      // match that exactly — de-duping across Stack Exchange network
      // sites works as long as both paths use the same URL scheme.
      const url = `https://stackoverflow.com/q/${qid}`;
      if (existingUrls && existingUrls.has(url)) continue;
      const created = a.creation_date || 0;
      const date = created
        ? new Date(created * 1000).toISOString().slice(0, 10)
        : "";
      const summary = truncate(
        collapseWhitespace(stripHtml(a.body || "")),
        200,
      );
      answersByQid.set(qid, { url, date, summary });
    }

    if (!result.has_more) break;
    await sleep(DELAY_MS);
  }

  if (answersByQid.size === 0) return {};

  // Phase 2: question titles in batches of 100.
  const qids = [...answersByQid.keys()];
  for (let i = 0; i < qids.length; i += 100) {
    const batch = qids.slice(i, i + 100);
    const idsStr = batch.join(";");
    try {
      const result = await fetchJson(
        `${API}/questions/${idsStr}?site=${site}&filter=!nNPvSNOTRz&pagesize=100`,
      );
      for (const q of result.items || []) {
        const qid = q.question_id;
        if (answersByQid.has(qid)) {
          answersByQid.get(qid).title = q.title || "";
        }
      }
    } catch (err) {
      console.warn("[sync] SE question title fetch error:", err);
    }
    if (i + 100 < qids.length) await sleep(DELAY_MS);
  }

  const out = {};
  for (const [qid, info] of answersByQid) {
    out[info.url] = {
      title: info.title || `Stack Overflow #${qid}`,
      summary: info.summary,
      date: info.date,
      tags: ["stackoverflow"],
      source: "stackoverflow",
    };
  }
  return out;
}
