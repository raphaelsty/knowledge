// Helpers for assembling the doc objects each fetcher yields. Mirrors
// the `{title, summary, date, tags, source, source_url}` shape the
// Python pipeline uses (see sources/utils/client.py::_merge_and_track)
// minus the LLM-generated extra-tags, which the backend fills in
// later.
//
// A fetcher returns a plain object keyed by URL:
//   { "https://...": { title, summary, date, tags, source, source_url } }
// so it's cheap to de-dupe across fetchers within one sync pass.

/**
 * Clean a title/summary string the way sources/utils/cleaning.py does
 * at a minimum: collapse whitespace, strip control chars. The heavier
 * cleaning (quote normalization, ligature folding) runs server-side
 * before PG insert anyway, so we keep this light.
 */
export function collapseWhitespace(s) {
  if (!s) return "";
  return String(s).replace(/\s+/g, " ").trim();
}

/** Strip HTML tags from a string — enough for HN comment summaries. */
export function stripHtml(s) {
  if (!s) return "";
  return String(s).replace(/<[^>]+>/g, " ");
}

/** Truncate a string to at most N characters, adding an ellipsis. */
export function truncate(s, limit) {
  s = s || "";
  if (s.length <= limit) return s;
  return s.slice(0, limit - 3) + "...";
}

/** Take the date-part of an ISO timestamp (`YYYY-MM-DD`). */
export function isoDate(s) {
  if (!s) return "";
  return String(s).slice(0, 10);
}

/**
 * Merge `incoming` into `into`, skipping URLs already in `existing`
 * AND URLs already in `into`. Matches the Python
 * `merge_new_documents` pattern used across fetchers. Returns the
 * number of new entries added.
 */
export function mergeInto(into, incoming, existing) {
  let added = 0;
  for (const [url, doc] of Object.entries(incoming)) {
    if (!url) continue;
    if (existing && existing.has(url)) continue;
    if (into[url]) continue;
    into[url] = doc;
    added += 1;
  }
  return added;
}
