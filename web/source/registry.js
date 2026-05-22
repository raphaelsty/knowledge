// Sync registry — maps a user's `sources` config (the same JSON the
// Python pipeline reads from users.sources) to the JS fetchers.
//
// A registry entry is `{ name, enabled(sources), run(sources, ctx) }`.
// `run` returns a flat `{ url: {title, summary, date, tags, source,
// source_url?} }` map. The orchestrator calls each entry in sequence
// so progress stays legible to the user; total runtime is typically
// dominated by GitHub stars + HN, both single-digit seconds.

import { submissions, comments } from "./hackernews/index.js";
import { stars, repositories, gists } from "./github/index.js";
import { likes as hfLikes } from "./huggingface/index.js";
import { references as wikiRefs } from "./wikipedia/index.js";
import { papers as s2Papers } from "./semantic_scholar/index.js";
import { publications as dblpPubs } from "./dblp/index.js";
import { papers as arxivPapers } from "./arxiv/index.js";
import {
  answers as soAnswers,
  questions as soQuestions,
} from "./stackoverflow/index.js";
import { posts as redditPosts } from "./reddit/index.js";
import { feed as blogFeed, sitemap as blogSitemap } from "./blog/index.js";
import { items as zoteroItems } from "./zotero/items.js";
import { hostnameSourceKey } from "./utils/hostname.js";

/** Stamp every doc with the given source key unless it already has one. */
function stamp(map, sourceKey) {
  for (const doc of Object.values(map)) {
    if (!doc.source) doc.source = sourceKey;
    // URL-level source overrides (matches Python
    // `_merge_and_track` short-circuits): when a URL lives on a
    // canonical brand domain, bucket it there regardless of how the
    // fetcher discovered it.
    // Done per-url by the caller; skipped here.
  }
  return map;
}

export const REGISTRY = [
  {
    key: "github.stars",
    label: "GitHub stars",
    enabled: (s) => Array.isArray(s.github) && s.github.length > 0,
    run: async (s, ctx) => {
      const users = s.github;
      const merged = {};
      for (const u of users) {
        const batch = await stars({
          user: u,
          existingUrls: ctx.existingUrls,
        });
        Object.assign(merged, batch);
      }
      return stamp(merged, "github");
    },
  },
  {
    key: "github.repositories",
    label: "GitHub own repos",
    enabled: (s) =>
      Array.isArray(s.github_repos) || typeof s.github_repos === "string",
    run: async (s, ctx) =>
      stamp(
        await repositories({
          users: s.github_repos,
          existingUrls: ctx.existingUrls,
        }),
        "github",
      ),
  },
  {
    key: "github.gists",
    label: "GitHub gists",
    enabled: (s) =>
      Array.isArray(s.github_gists) || typeof s.github_gists === "string",
    run: async (s, ctx) =>
      stamp(
        await gists({
          users: s.github_gists,
          existingUrls: ctx.existingUrls,
        }),
        "github",
      ),
  },
  {
    key: "hackernews.submissions",
    label: "HackerNews submissions",
    enabled: (s) => !!(s.hackernews && s.hackernews.username),
    run: async (s, ctx) =>
      stamp(
        await submissions({
          username: s.hackernews.username,
          existingUrls: ctx.existingUrls,
        }),
        "hackernews",
      ),
  },
  {
    key: "hackernews.comments",
    label: "HackerNews comments",
    enabled: (s) => !!(s.hackernews && s.hackernews.username),
    run: async (s, ctx) =>
      stamp(
        await comments({
          username: s.hackernews.username,
          existingUrls: ctx.existingUrls,
        }),
        "hackernews",
      ),
  },
  {
    key: "huggingface.likes",
    label: "HuggingFace likes",
    enabled: (s) =>
      typeof s.huggingface === "string" && s.huggingface.length > 0,
    run: async (s, ctx) =>
      stamp(
        await hfLikes({
          username: s.huggingface,
          existingUrls: ctx.existingUrls,
        }),
        "huggingface",
      ),
  },
  {
    key: "wikipedia",
    label: "Wikipedia references",
    enabled: (s) =>
      (Array.isArray(s.wikipedia) && s.wikipedia.length > 0) ||
      typeof s.wikipedia === "string",
    run: async (s, ctx) =>
      stamp(
        await wikiRefs({
          pages: s.wikipedia,
          existingUrls: ctx.existingUrls,
        }),
        "wikipedia",
      ),
  },
  {
    key: "semantic_scholar",
    label: "Semantic Scholar",
    enabled: (s) => !!s.semantic_scholar,
    run: async (s, ctx) => {
      const cfg = s.semantic_scholar;
      const params =
        typeof cfg === "string"
          ? { authorId: cfg }
          : {
              authorId: cfg.author_id,
              authorName: cfg.author_name,
              maxPapers: cfg.max_papers,
              minCitations: cfg.min_citations,
            };
      return stamp(
        await s2Papers({ ...params, existingUrls: ctx.existingUrls }),
        "scholar",
      );
    },
  },
  {
    key: "dblp",
    label: "DBLP",
    enabled: (s) => !!s.dblp,
    run: async (s, ctx) => {
      const cfg = s.dblp;
      const params =
        typeof cfg === "string"
          ? { author: cfg }
          : { author: cfg.author, maxResults: cfg.max_results };
      return stamp(
        await dblpPubs({ ...params, existingUrls: ctx.existingUrls }),
        "scholar",
      );
    },
  },
  {
    key: "arxiv",
    label: "arXiv",
    enabled: (s) => !!s.arxiv,
    run: async (s, ctx) => {
      const cfg = s.arxiv;
      const params =
        typeof cfg === "string"
          ? { author: cfg }
          : { author: cfg.author, maxResults: cfg.max_results };
      return stamp(
        await arxivPapers({ ...params, existingUrls: ctx.existingUrls }),
        "arxiv",
      );
    },
  },
  {
    key: "stackoverflow.answers",
    label: "Stack Overflow answers",
    enabled: (s) =>
      s.stackoverflow &&
      (s.stackoverflow.user_id ||
        s.stackoverflow.username ||
        typeof s.stackoverflow === "string"),
    run: async (s, ctx) => {
      const cfg = s.stackoverflow;
      const isObj = typeof cfg === "object";
      const userId = isObj ? cfg.user_id : null;
      const username = isObj ? cfg.username : String(cfg);
      // Mirror Python client.py: max_pages / min_score default to 5 /
      // 1, overridable via the users.sources JSON.
      const maxPages = isObj && cfg.max_pages != null ? cfg.max_pages : 5;
      const minScore = isObj && cfg.min_score != null ? cfg.min_score : 1;
      return stamp(
        await soAnswers({
          userId,
          username,
          site: "stackoverflow",
          maxPages,
          minScore,
          existingUrls: ctx.existingUrls,
        }),
        "stackoverflow",
      );
    },
  },
  {
    key: "stackoverflow.questions",
    label: "Stack Overflow questions",
    enabled: (s) =>
      s.stackoverflow &&
      (s.stackoverflow.user_id ||
        s.stackoverflow.username ||
        typeof s.stackoverflow === "string"),
    run: async (s, ctx) => {
      const cfg = s.stackoverflow;
      const isObj = typeof cfg === "object";
      const userId = isObj ? cfg.user_id : null;
      const username = isObj ? cfg.username : String(cfg);
      // Python hardcodes min_score=0 for Questions (every asked
      // question counts regardless of score) and reuses max_pages.
      const maxPages = isObj && cfg.max_pages != null ? cfg.max_pages : 5;
      return stamp(
        await soQuestions({
          userId,
          username,
          site: "stackoverflow",
          maxPages,
          minScore: 0,
          existingUrls: ctx.existingUrls,
        }),
        "stackoverflow",
      );
    },
  },
  {
    key: "reddit",
    label: "Reddit",
    enabled: (s) =>
      !!(s.reddit && (typeof s.reddit === "string" || s.reddit.username)),
    run: async (s, ctx) => {
      const cfg = s.reddit;
      const username = typeof cfg === "string" ? cfg : cfg.username;
      return stamp(
        await redditPosts({
          username,
          maxPages:
            typeof cfg === "object" && cfg.max_pages ? cfg.max_pages : 5,
          existingUrls: ctx.existingUrls,
        }),
        "reddit",
      );
    },
  },
  {
    // Zotero — runs through our `/auth/me/zotero/items` proxy because
    // the API key is encrypted at rest and the browser doesn't have
    // access to the encryption key. The proxy paginates personal +
    // group libraries server-side and returns ready-shaped docs.
    key: "zotero",
    label: "Zotero library",
    enabled: (s) => !!(s.zotero && s.zotero.api_key_enc && s.zotero.user_id),
    run: async (s, ctx) => {
      // Zotero items are stamped with their per-URL bucket inside
      // `applyUrlSourceOverrides` (sync.js): brand domains get
      // routed (arxiv / huggingface / github / youtube), the rest
      // fall back to hostname (aclanthology.org, dl.acm.org, …).
      // We pass an empty default so `stamp` is a no-op for sources
      // it can't classify; sync.js will fill them in.
      return await zoteroItems({
        apiBase: ctx.apiBase || "",
        existingUrls: ctx.existingUrls,
      });
    },
  },
  {
    key: "websites",
    label: "Websites (feeds + sitemaps)",
    enabled: (s) => Array.isArray(s.websites) && s.websites.length > 0,
    run: async (s, ctx) => {
      const merged = {};
      const errors = [];
      for (const ws of s.websites) {
        if (!ws || !ws.url) continue;
        const kind = ws.kind || "sitemap";
        const tags = ws.tags || ["blog"];
        const input = ws.input || ws.url;
        // No generic "blog" fallback — every site gets its own chip.
        // If hostnameSourceKey can't extract a host (very rare for
        // valid URLs), the doc still ships with `source = ""` and the
        // user_source_counts view filters it out of the panel.
        const sourceKey = hostnameSourceKey(input) || "";
        let batch = {};
        try {
          if (kind === "feed") {
            batch = await blogFeed({
              feedUrl: ws.url,
              tags,
              existingUrls: ctx.existingUrls,
            });
          } else {
            batch = await blogSitemap({
              sitemapUrl: ws.url,
              tags,
              urlFilter: ws.url_filter || null,
              existingUrls: ctx.existingUrls,
            });
          }
        } catch (err) {
          // CORS failures from cross-origin fetches surface as
          // TypeError("Failed to fetch") in browsers. Tag them
          // clearly so users understand it's the browser blocking
          // the response, not a bad sitemap on their end.
          const msg = String(err && err.message ? err.message : err);
          const isCors = /failed to fetch|network ?error|cors/i.test(msg);
          const friendly = isCors
            ? `${ws.url}: blocked by browser CORS — host doesn't allow cross-origin reads`
            : `${ws.url}: ${msg}`;
          errors.push(friendly);
          console.warn(`[sync] website ${friendly}`);
        }
        // Per-website source stamping (hostname bucket).
        for (const doc of Object.values(batch)) {
          if (!doc.source) doc.source = sourceKey;
        }
        Object.assign(merged, batch);
      }
      // If every website failed AND nothing was fetched, surface the
      // collated error so the sync log reports it instead of silently
      // succeeding with 0 docs.
      if (Object.keys(merged).length === 0 && errors.length > 0) {
        const e = new Error(errors.join(" · "));
        e.partial = true; // signal partial-failure rather than catastrophic
        throw e;
      }
      return merged;
    },
  },
];

/** Return the subset of REGISTRY that should run for this config. */
export function enabledFetchers(sourcesConfig) {
  const cfg = sourcesConfig || {};
  return REGISTRY.filter((e) => {
    try {
      return e.enabled(cfg);
    } catch {
      return false;
    }
  });
}
