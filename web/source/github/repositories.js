// JS port of sources/github/repositories.py. Fetches a user's own
// non-fork public repos via `/users/{u}/repos?sort=updated`.

import { fetchJson, sleep } from "../utils/http.js";
import { isoDate, truncate } from "../utils/doc.js";

const DELAY_MS = 1000;

export async function repositories({ users, existingUrls = null }) {
  const list = Array.isArray(users) ? users : [users];
  const out = {};

  for (const user of list) {
    let page = 1;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      let repos;
      try {
        repos = await fetchJson(
          `https://api.github.com/users/${encodeURIComponent(user)}/repos?per_page=100&page=${page}&sort=updated&type=owner`,
          { headers: { Accept: "application/vnd.github.v3+json" } },
        );
      } catch (err) {
        console.warn("[sync] github repos error:", err);
        break;
      }
      if (!Array.isArray(repos) || repos.length === 0) break;

      if (existingUrls) {
        const newInPage = repos.filter(
          (r) => !r.fork && r.html_url && !existingUrls.has(r.html_url),
        ).length;
        if (newInPage === 0) break;
      }

      for (const r of repos) {
        if (r.fork) continue;
        const url = r.html_url;
        if (!url) continue;
        if (existingUrls && existingUrls.has(url)) continue;
        const desc = r.description || "";
        const topics = r.topics || [];
        const lang = r.language || "";
        const stars = r.stargazers_count || 0;
        const updated = isoDate(r.pushed_at);

        let summary = desc;
        if (stars > 0)
          summary += summary ? ` (${stars} stars)` : `${stars} stars`;
        summary = truncate(summary, 250);

        const tags = topics.map((t) => t.toLowerCase());
        const l = lang.toLowerCase();
        if (l && !tags.includes(l)) tags.push(l);

        out[url] = {
          title: r.name || "",
          summary,
          date: updated,
          tags,
          source: "github",
        };
      }
      page += 1;
      await sleep(DELAY_MS);
    }
  }
  return out;
}
