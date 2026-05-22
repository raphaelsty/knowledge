// Sync orchestrator. One public function: runSync(config, cb?).
//
//   - `config.sources`          — the signed-in user's `users.sources`
//                                 JSON (same shape the Python pipeline
//                                 reads from the DB).
//   - `config.existingUrls`     — Set<string> of URLs already in the
//                                 user's library, so fetchers can
//                                 early-exit. Pass the result of
//                                 `GET /auth/me/documents/urls`.
//   - `config.apiBase`          — e.g. "http://localhost:8080" or "".
//   - `config.batchSize`        — chunk size for the bulk POST (default 1000).
//   - `config.onProgress(evt)`  — optional callback; evts:
//       { type: "start",   steps }
//       { type: "step.start", key, label, index, total }
//       { type: "step.done",  key, label, index, total, fetched, error? }
//       { type: "upload",   received, inserted }
//       { type: "done",     totalFetched, totalInserted, errors }
//
// Never throws out of runSync — per-fetcher failures are captured in
// the progress stream so a broken source doesn't tank the whole run.

import { enabledFetchers } from "./registry.js";
import { hostnameSourceKey } from "./utils/hostname.js";

const MAX_BULK = 1000;

async function postBulk(apiBase, docs) {
  const body = {
    documents: docs.map((d) => ({
      url: d.url,
      title: d.title || "",
      summary: d.summary || "",
      date: d.date || "",
      source: d.source || "",
      source_url: d.source_url || null,
    })),
  };
  const resp = await fetch(`${apiBase}/auth/me/documents/bulk`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    throw new Error(`bulk upload HTTP ${resp.status}`);
  }
  return resp.json();
}

/* Route every doc to a bucket based on its URL.
 *
 * Three layers, applied in order:
 *
 *   1. Brand domains — arxiv, huggingface, github, youtube. A
 *      tweeted arXiv paper, a Zotero-saved HF model, and a Scholar-
 *      cited GitHub repo all bucket the same way regardless of
 *      which fetcher pulled them in.
 *   2. Sourceless docs (Zotero is the main case — its fetcher
 *      doesn't pre-stamp `source`) get the URL's hostname as their
 *      bucket. So an ACL paper from Zotero ends up at
 *      `aclanthology.org`, a YouTube link at `youtube`, an OpenReview
 *      submission at `openreview.net`, etc.
 *   3. Anything that already had a source (twitter, reddit, …)
 *      keeps it.
 *
 * Mirrors `_merge_and_track` in sources/utils/client.py — keep both
 * in sync. */
function applyUrlSourceOverrides(doc, url) {
  if (url.includes("arxiv.org")) doc.source = "arxiv";
  else if (url.includes("huggingface.co")) doc.source = "huggingface";
  else if (url.includes("github.com")) doc.source = "github";
  else if (url.includes("youtube.com") || url.includes("youtu.be"))
    doc.source = "youtube";
  else if (!doc.source) doc.source = hostnameSourceKey(url) || "";
}

export async function runSync({
  sources,
  existingUrls = new Set(),
  apiBase = "",
  batchSize = MAX_BULK,
  onProgress = () => {},
}) {
  const steps = enabledFetchers(sources);
  onProgress({
    type: "start",
    steps: steps.map((s) => ({ key: s.key, label: s.label })),
  });

  const runStartedAt = Date.now();
  // Open a live tracker row in pipeline_runs. Failure here just
  // means we won't be visible on the "what's running" dashboard —
  // the sync itself continues either way.
  let runId = 0;
  try {
    const r = await fetch(`${apiBase}/auth/me/sync/start`, {
      method: "POST",
      credentials: "include",
    });
    if (r.ok) {
      const body = await r.json();
      runId = body.run_id || 0;
    }
  } catch {
    /* tracker is best-effort — swallow */
  }

  const allDocs = {};
  const errors = [];

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    onProgress({
      type: "step.start",
      key: step.key,
      label: step.label,
      index: i,
      total: steps.length,
    });
    try {
      const docs = await step.run(sources, { existingUrls, apiBase });
      for (const [url, doc] of Object.entries(docs)) {
        applyUrlSourceOverrides(doc, url);
        // First writer wins — matches merge_new_documents semantics.
        if (!allDocs[url]) allDocs[url] = { ...doc, url };
      }
      onProgress({
        type: "step.done",
        key: step.key,
        label: step.label,
        index: i,
        total: steps.length,
        fetched: Object.keys(docs).length,
      });
    } catch (err) {
      errors.push({ key: step.key, error: String(err.message || err) });
      onProgress({
        type: "step.done",
        key: step.key,
        label: step.label,
        index: i,
        total: steps.length,
        fetched: 0,
        error: String(err.message || err),
      });
    }
  }

  // Dedupe vs existingUrls: extraction returned plenty of URLs we
  // already own, and the bulk endpoint is idempotent but we'd rather
  // not send them.
  const toUpload = Object.values(allDocs).filter(
    (d) => !existingUrls.has(d.url),
  );

  let totalInserted = 0;
  let totalReceived = 0;
  for (let i = 0; i < toUpload.length; i += batchSize) {
    const batch = toUpload.slice(i, i + batchSize);
    try {
      const r = await postBulk(apiBase, batch);
      totalInserted += r.inserted || 0;
      totalReceived += r.received || 0;
      onProgress({
        type: "upload",
        received: totalReceived,
        inserted: totalInserted,
      });
    } catch (err) {
      errors.push({ key: "upload", error: String(err.message || err) });
      break;
    }
  }

  const summary = {
    type: "done",
    totalFetched: Object.keys(allDocs).length,
    totalInserted,
    errors,
  };

  // Seal the tracker row. Success = "no per-step error AND no upload
  // error"; anything that landed in `errors` flips it to failed. Like
  // sync-start above, failure to call sync-end isn't fatal — it just
  // leaves a `running` row that the next Python run sweeps via
  // cleanup_stale_runs.
  if (runId > 0) {
    const success = errors.length === 0;
    const duration = (Date.now() - runStartedAt) / 1000;
    try {
      await fetch(`${apiBase}/auth/me/sync/end`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId,
          success,
          total_documents: Object.keys(allDocs).length,
          new_documents: totalInserted,
          duration_secs: duration,
          timings: steps.map((s) => ({
            step: s.key,
            label: s.label,
          })),
          error: success
            ? null
            : errors
                .map((e) => `${e.key}: ${e.error}`)
                .join("; ")
                .slice(0, 500),
        }),
      });
    } catch {
      /* see note above */
    }
  }

  onProgress(summary);
  return summary;
}
