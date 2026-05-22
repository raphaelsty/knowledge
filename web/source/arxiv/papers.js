// JS port of sources/arxiv/papers.py.
//
// arXiv's query API returns Atom XML. CORS support is flaky in
// practice (no Access-Control-Allow-Origin on all responses); when it
// blocks the fetch will reject and the orchestrator records a
// per-source error. The Python pipeline still handles arxiv so
// nothing is lost.

import { fetchText, sleep } from "../utils/http.js";
import { truncate } from "../utils/doc.js";

const API = "https://export.arxiv.org/api/query";
const DELAY_MS = 3000;
const PAGE_SIZE = 100;
const ARXIV_ID_RE = /\/abs\/(\d{4}\.\d{4,5})/;

function parseAtom(xmlText) {
  const doc = new DOMParser().parseFromString(xmlText, "application/xml");
  const err = doc.querySelector("parsererror");
  if (err) throw new Error("arxiv atom parse error");
  return Array.from(
    doc.getElementsByTagNameNS("http://www.w3.org/2005/Atom", "entry"),
  );
}

function textOf(entry, local, ns = "http://www.w3.org/2005/Atom") {
  const el = entry.getElementsByTagNameNS(ns, local)[0];
  return el ? (el.textContent || "").trim() : "";
}

function canonicalFromId(raw) {
  const m = raw.match(ARXIV_ID_RE);
  if (m) return `https://arxiv.org/abs/${m[1]}`;
  return raw.replace(/^http:/, "https:");
}

export async function papers({
  author,
  maxResults = 200,
  existingUrls = null,
}) {
  const out = {};
  let start = 0;

  while (start < maxResults) {
    const query = encodeURIComponent(`au:"${author}"`);
    const url = `${API}?search_query=${query}&start=${start}&max_results=${PAGE_SIZE}&sortBy=submittedDate&sortOrder=descending`;
    let xml;
    try {
      xml = await fetchText(url, { timeoutMs: 30000 });
    } catch (err) {
      console.warn("[sync] arxiv fetch error:", err);
      break;
    }
    let entries;
    try {
      entries = parseAtom(xml);
    } catch (err) {
      console.warn("[sync] arxiv parse error:", err);
      break;
    }
    if (entries.length === 0) break;

    if (existingUrls) {
      let newInPage = 0;
      for (const e of entries) {
        const raw = textOf(e, "id");
        if (!raw) continue;
        const canonical = canonicalFromId(raw);
        if (!existingUrls.has(canonical)) newInPage += 1;
      }
      if (newInPage === 0) break;
    }

    for (const entry of entries) {
      const raw = textOf(entry, "id");
      if (!raw) continue;
      const canonical = canonicalFromId(raw);
      if (existingUrls && existingUrls.has(canonical)) continue;
      if (out[canonical]) continue;

      // Python: `title.strip().replace("\n", " ")` — only newlines
      // get flattened, internal runs of spaces stay as-is. Same for
      // abstract. Avoid the blanket `\s+ → " "` collapse we had
      // before: it dropped significant whitespace (e.g. tabs between
      // math tokens) that Python preserved.
      const title = textOf(entry, "title").replace(/\n/g, " ");
      const abstract = truncate(
        textOf(entry, "summary").replace(/\n/g, " "),
        250,
      );
      const date = textOf(entry, "published").slice(0, 10);

      const categories = [];
      // Try the namespaced lookup first (what browsers resolve cleanly
      // from arxiv's `xmlns:arxiv="..."`). Some XML DOM implementations
      // don't walk prefix-bound namespaces, so fall back to the
      // prefixed tag name, which the arxiv feed always emits verbatim.
      let catEls = Array.from(
        entry.getElementsByTagNameNS(
          "http://arxiv.org/schemas/atom",
          "primary_category",
        ),
      );
      if (catEls.length === 0) {
        catEls = Array.from(
          entry.getElementsByTagName("arxiv:primary_category"),
        );
      }
      for (const c of catEls) {
        const term = c.getAttribute("term");
        if (term) categories.push(term);
      }

      out[canonical] = {
        title,
        summary: abstract,
        date,
        tags: ["arxiv", ...categories],
        source: "arxiv",
      };
    }

    start += entries.length;
    if (entries.length < PAGE_SIZE) break;
    await sleep(DELAY_MS);
  }
  return out;
}
