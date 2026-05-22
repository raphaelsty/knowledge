// JS port of sources/github/gists.py. One doc per public gist.

import { fetchJson, sleep } from "../utils/http.js";
import { isoDate, truncate } from "../utils/doc.js";

const DELAY_MS = 1000;

export async function gists({ users, existingUrls = null }) {
  const list = Array.isArray(users) ? users : [users];
  const out = {};

  for (const user of list) {
    let page = 1;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      let gistsPage;
      try {
        gistsPage = await fetchJson(
          `https://api.github.com/users/${encodeURIComponent(user)}/gists?per_page=100&page=${page}`,
          { headers: { Accept: "application/vnd.github.v3+json" } },
        );
      } catch (err) {
        console.warn("[sync] github gists error:", err);
        break;
      }
      if (!Array.isArray(gistsPage) || gistsPage.length === 0) break;

      if (existingUrls) {
        const newInPage = gistsPage.filter(
          (g) => g.html_url && !existingUrls.has(g.html_url),
        ).length;
        if (newInPage === 0) break;
      }

      for (const g of gistsPage) {
        const url = g.html_url;
        if (!url) continue;
        if (existingUrls && existingUrls.has(url)) continue;
        const desc = (g.description || "").trim();
        const files = g.files || {};
        const filenames = Object.keys(files);
        // Python: `desc or (filenames[0] if filenames else f"Gist …")`
        // — the first-filename branch is chosen by list non-emptiness,
        // not by the filename's truthiness. A truly empty filename
        // stays, rather than being replaced by the gist-id fallback.
        const fallback = filenames.length
          ? filenames[0]
          : `Gist ${(g.id || "").slice(0, 8)}`;
        const title = truncate(desc || fallback, 100);
        const created = isoDate(g.created_at);
        const langs = new Set();
        for (const f of Object.values(files)) {
          const l = f && f.language ? String(f.language).toLowerCase() : "";
          if (l) langs.add(l);
        }
        // Python: `desc if desc != title else ", ".join(filenames[:3])`
        // — note there's no falsy-check on desc. When desc is empty
        // (so title was taken from the first filename), desc != title
        // holds true and the summary stays empty.
        out[url] = {
          title,
          summary: desc !== title ? desc : filenames.slice(0, 3).join(", "),
          date: created,
          tags: Array.from(langs),
          source: "github",
        };
      }
      page += 1;
      await sleep(DELAY_MS);
    }
  }
  return out;
}
