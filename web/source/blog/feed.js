// Full-parity port of sources/blog/feed.py + the feed-parsing
// primitives in sources/blog/_helpers.py (_parse_atom, _parse_rss2,
// _parse_rdf, _parse_json_feed, _dispatch_feed).
//
// Parses Atom, RSS 2.0, RSS 1.0 (RDF), and JSON Feed. Feeds that
// don't set CORS headers will fail in the browser; the orchestrator
// records the per-site failure and the server pipeline still reaches
// them on the next run.

import { fetchText } from "../utils/http.js";
import {
  resolveUrl,
  stripHtmlDeep,
  cleanSummary,
  cleanTitle,
  coerceDate,
  fallbackSummary,
  tagsFromUrl,
} from "./_helpers.js";

const MAX_ENTRIES = 1000;

const NS = {
  atom: "http://www.w3.org/2005/Atom",
  dc: "http://purl.org/dc/elements/1.1/",
  content: "http://purl.org/rss/1.0/modules/content/",
  rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
  rss1: "http://purl.org/rss/1.0/",
};

/** Text of the first child tag `(ns, name)` or empty. */
function textNS(parent, ns, name) {
  const el = parent.getElementsByTagNameNS(ns, name)[0];
  return el ? (el.textContent || "").trim() : "";
}

/** Text of the first non-namespaced child tag `name` or empty. */
function textPlain(parent, name) {
  // DOMParser on "application/xml" puts every element into a namespace,
  // so `getElementsByTagName(name)` on a namespaced element returns
  // nothing. Walk the direct children by localName as a fallback.
  const direct = parent.getElementsByTagName(name);
  if (direct.length > 0) return (direct[0].textContent || "").trim();
  for (const child of parent.children) {
    if (child.localName === name) return (child.textContent || "").trim();
  }
  return "";
}

/** Walk text descendants of an atom:content type=xhtml element. */
function xhtmlContentText(contentEl) {
  // Python iterates DIRECT children of <content>, each stringified
  // to text, joined with a single space. Mirror that so adjacent
  // elements without whitespace between them (e.g. <p>A</p><p>B</p>)
  // keep a word boundary in the extracted summary.
  const parts = [];
  for (const child of contentEl.children) {
    const t = child.textContent;
    if (t) parts.push(t);
  }
  return parts.join(" ").trim();
}

// ─────────────────────────────────────────────────────────────────
// Atom
// ─────────────────────────────────────────────────────────────────

function parseAtom(root, baseTags, baseUrl) {
  const out = {};
  const entries = root.getElementsByTagNameNS(NS.atom, "entry");
  let rank = 0;
  for (const entry of entries) {
    const titleEl = entry.getElementsByTagNameNS(NS.atom, "title")[0];
    const rawTitle = titleEl ? titleEl.textContent || "" : "";

    // Pick the best alternate <link href="…"/>.
    let link = "";
    const links = entry.getElementsByTagNameNS(NS.atom, "link");
    for (const l of links) {
      const rel = l.getAttribute("rel") || "alternate";
      const href = l.getAttribute("href") || "";
      if (rel === "alternate" && href) {
        link = resolveUrl(baseUrl, href);
        break;
      }
      if (!link && href && !["self", "enclosure", "edit"].includes(rel)) {
        link = resolveUrl(baseUrl, href);
      }
    }

    const title = cleanTitle(rawTitle, link);
    if (!link || !title) continue;

    const published = textNS(entry, NS.atom, "published");
    const updated = textNS(entry, NS.atom, "updated");
    const dateStr = published || updated;

    // Content / summary. Python prefers <content>, then <summary>.
    // atom:content type="xhtml" must have its child text extracted.
    let summary = "";
    const contentEl = entry.getElementsByTagNameNS(NS.atom, "content")[0];
    const summaryEl = entry.getElementsByTagNameNS(NS.atom, "summary")[0];
    if (contentEl) {
      const ctype = (contentEl.getAttribute("type") || "text").toLowerCase();
      if (ctype === "xhtml") {
        summary = stripHtmlDeep(xhtmlContentText(contentEl));
      } else if (contentEl.textContent) {
        summary = stripHtmlDeep(contentEl.textContent);
      }
    }
    if (!summary && summaryEl && summaryEl.textContent) {
      summary = stripHtmlDeep(summaryEl.textContent);
    }
    summary = fallbackSummary(cleanSummary(summary), title);

    const entryTags = [...baseTags];
    const cats = entry.getElementsByTagNameNS(NS.atom, "category");
    for (const c of cats) {
      const term = (c.getAttribute("term") || c.getAttribute("label") || "")
        .trim()
        .toLowerCase();
      if (term && !entryTags.includes(term)) entryTags.push(term);
    }
    // No <category> at all → fall back to URL path hints.
    if (entryTags.length === baseTags.length) {
      for (const hint of tagsFromUrl(link)) {
        if (!entryTags.includes(hint)) entryTags.push(hint);
      }
    }

    out[link] = {
      title,
      summary,
      date: coerceDate(dateStr, link, rank),
      tags: entryTags,
    };
    rank++;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────
// RSS 2.0
// ─────────────────────────────────────────────────────────────────

function parseRss2(root, baseTags, baseUrl) {
  const out = {};
  const channels = root.getElementsByTagName("channel");
  if (channels.length === 0) return out;
  const channel = channels[0];

  const items = channel.getElementsByTagName("item");
  let rank = 0;
  for (const item of items) {
    const rawTitle = textPlain(item, "title");

    let link = textPlain(item, "link");
    // Atom-namespaced link inside an RSS 2.0 item (common on feedburner etc).
    if (!link) {
      const atomLink = item.getElementsByTagNameNS(NS.atom, "link")[0];
      if (atomLink) link = atomLink.getAttribute("href") || "";
    }
    // Blogger-style <guid isPermaLink="true"> fallback.
    if (!link) {
      const guid = item.getElementsByTagName("guid")[0];
      if (
        guid &&
        guid.textContent &&
        (guid.getAttribute("isPermaLink") || "true").toLowerCase() !== "false"
      ) {
        link = guid.textContent.trim();
      }
    }
    link = resolveUrl(baseUrl, link);
    const title = cleanTitle(rawTitle, link);
    if (!link || !title) continue;

    const pubDate = textPlain(item, "pubDate");
    const dcDate = textNS(item, NS.dc, "date");
    const dateStr = pubDate || dcDate;

    let summary = "";
    const encoded = item.getElementsByTagNameNS(NS.content, "encoded")[0];
    const description = item.getElementsByTagName("description")[0];
    if (encoded && encoded.textContent) {
      summary = stripHtmlDeep(encoded.textContent);
    } else if (description && description.textContent) {
      summary = stripHtmlDeep(description.textContent);
    }

    // Collect raw category terms BEFORE cleaning the summary so
    // `fallbackSummary` can detect category-only descriptions
    // (research.google emits only the category into <description>).
    const entryTags = [...baseTags];
    const rawCats = [];
    for (const c of item.getElementsByTagName("category")) {
      const term = (c.textContent || "").trim();
      if (term) {
        rawCats.push(term);
        const tl = term.toLowerCase();
        if (!entryTags.includes(tl)) entryTags.push(tl);
      }
    }
    for (const s of item.getElementsByTagNameNS(NS.dc, "subject")) {
      const term = (s.textContent || "").trim();
      if (term) {
        rawCats.push(term);
        const tl = term.toLowerCase();
        if (!entryTags.includes(tl)) entryTags.push(tl);
      }
    }
    summary = fallbackSummary(cleanSummary(summary), title, rawCats);

    if (entryTags.length === baseTags.length) {
      for (const hint of tagsFromUrl(link)) {
        if (!entryTags.includes(hint)) entryTags.push(hint);
      }
    }

    out[link] = {
      title,
      summary,
      date: coerceDate(dateStr, link, rank),
      tags: entryTags,
    };
    rank++;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────
// RSS 1.0 (RDF)
// ─────────────────────────────────────────────────────────────────

function parseRdf(root, baseTags, baseUrl) {
  const out = {};
  const items = root.getElementsByTagNameNS(NS.rss1, "item");
  let rank = 0;
  for (const item of items) {
    const rawTitle = textNS(item, NS.rss1, "title");
    let link = textNS(item, NS.rss1, "link");
    if (!link) link = item.getAttributeNS(NS.rdf, "about") || "";
    link = resolveUrl(baseUrl, link);
    const title = cleanTitle(rawTitle, link);
    if (!link || !title) continue;

    const dateStr = textNS(item, NS.dc, "date");
    let summary = "";
    const encoded = item.getElementsByTagNameNS(NS.content, "encoded")[0];
    const desc = item.getElementsByTagNameNS(NS.rss1, "description")[0];
    if (encoded && encoded.textContent) {
      summary = stripHtmlDeep(encoded.textContent);
    } else if (desc && desc.textContent) {
      summary = stripHtmlDeep(desc.textContent);
    }
    summary = fallbackSummary(cleanSummary(summary), title);

    const entryTags = [...baseTags];
    for (const s of item.getElementsByTagNameNS(NS.dc, "subject")) {
      const term = (s.textContent || "").trim().toLowerCase();
      if (term && !entryTags.includes(term)) entryTags.push(term);
    }
    if (entryTags.length === baseTags.length) {
      for (const hint of tagsFromUrl(link)) {
        if (!entryTags.includes(hint)) entryTags.push(hint);
      }
    }

    out[link] = {
      title,
      summary,
      date: coerceDate(dateStr, link, rank),
      tags: entryTags,
    };
    rank++;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────
// JSON Feed 1.1
// ─────────────────────────────────────────────────────────────────

function parseJsonFeed(text, baseTags, baseUrl) {
  let doc;
  try {
    doc = JSON.parse(text);
  } catch {
    return {};
  }
  if (!doc || typeof doc !== "object") return {};
  const out = {};
  let rank = 0;
  for (const item of doc.items || []) {
    if (!item || typeof item !== "object") continue;
    const rawLink = (item.url || item.external_url || "").trim();
    const link = resolveUrl(baseUrl, rawLink);
    const title = cleanTitle(item.title || "", link);
    if (!link || !title) continue;

    const dateStr = item.date_published || item.date_modified || "";
    let summary = "";
    if (item.summary) summary = String(item.summary).trim();
    else if (item.content_text) summary = String(item.content_text).trim();
    else if (item.content_html) summary = stripHtmlDeep(item.content_html);
    summary = fallbackSummary(cleanSummary(summary), title);

    const entryTags = [...baseTags];
    for (const t of item.tags || []) {
      const tag = (t || "").trim().toLowerCase();
      if (tag && !entryTags.includes(tag)) entryTags.push(tag);
    }
    if (entryTags.length === baseTags.length) {
      for (const hint of tagsFromUrl(link)) {
        if (!entryTags.includes(hint)) entryTags.push(hint);
      }
    }

    out[link] = {
      title,
      summary,
      date: coerceDate(dateStr, link, rank),
      tags: entryTags,
    };
    rank++;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────
// Dispatcher
// ─────────────────────────────────────────────────────────────────

function safeParseXml(text) {
  const parsed = new DOMParser().parseFromString(text, "application/xml");
  const root = parsed.documentElement;
  if (!root || parsed.getElementsByTagName("parsererror")[0]) {
    // Python falls back after stripping xml-stylesheet PIs + comments.
    const stripped = text
      .replace(/<\?xml-stylesheet[^>]*\?>/g, "")
      .replace(/<!--[\s\S]*?-->/g, "");
    const retry = new DOMParser().parseFromString(stripped, "application/xml");
    const retryRoot = retry.documentElement;
    if (!retryRoot || retry.getElementsByTagName("parsererror")[0]) return null;
    return retryRoot;
  }
  return root;
}

function dispatch(text, baseTags, baseUrl) {
  const stripped = text.replace(/^[﻿\s]+/, "");
  if (stripped.startsWith("{")) return parseJsonFeed(text, baseTags, baseUrl);
  const root = safeParseXml(text);
  if (!root) return {};
  const tag = root.tagName.toLowerCase();
  const ns = root.namespaceURI || "";
  if (ns === NS.atom || tag === "feed")
    return parseAtom(root, baseTags, baseUrl);
  if (tag === "rss") return parseRss2(root, baseTags, baseUrl);
  if (ns === NS.rdf || tag.endsWith(":rdf") || tag === "rdf")
    return parseRdf(root, baseTags, baseUrl);
  return {};
}

export async function feed({
  feedUrl,
  tags = [],
  maxEntries = MAX_ENTRIES,
  existingUrls = null,
}) {
  let text;
  try {
    text = await fetchText(feedUrl, { timeoutMs: 30000 });
  } catch (err) {
    console.warn(`[sync] feed fetch failed: ${feedUrl} — ${err.message}`);
    return {};
  }
  let out = dispatch(text, tags, feedUrl);
  if (existingUrls) {
    out = Object.fromEntries(
      Object.entries(out).filter(([url]) => !existingUrls.has(url)),
    );
  }
  // Python: feeds emit newest-first → slicing keeps the most recent.
  return Object.fromEntries(Object.entries(out).slice(0, maxEntries));
}
