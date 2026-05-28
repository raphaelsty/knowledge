/**
 * colbert.worker.js
 *
 * This script runs in a background Web Worker to handle all computationally
 * intensive tasks related to the ColBERT model. This prevents the main
 * browser UI thread from freezing during model loading and re-ranking.
 */

// ES module imports for the WASM-backed ColBERT model.
import init, { ColBERT } from "./pkg/pylate_rs.js";

// --- Constants ---

const CACHE_NAME = "colbert-model-cache-v1";
const MODEL_REPO = "lightonai/answerai-colbert-small-v1";
const MODEL_FILES = [
  "tokenizer.json",
  "model.safetensors",
  "config.json",
  "config_sentence_transformers.json",
  "1_Dense/model.safetensors",
  "1_Dense/config.json",
  "special_tokens_map.json",
];
const COLBERT_LATENCY_BUCKET = 32; // A ColBERT model parameter.
const MAX_DOCS_TO_RANK = 29; // We only re-rank the top N documents for performance.

// --- Hybrid scoring (ColBERT + BM25) ---
//
// The worker re-ranks the API's top-29 candidates locally. ColBERT alone
// loses rare-proper-noun queries — a doc with the exact token in its
// title can score below a semantically-broader doc that doesn't. Adding
// a per-query min-max normalized BM25 signal on top keeps lexical
// precision. Weights mirror paradigm-mission-control's offline-tuned
// linear-fusion recipe (w_dense = w_lex = 0.90, picked via grid search
// over 1,339 labeled queries). Equal weighting after normalization.
const W_COLBERT = 0.9;
const W_BM25 = 0.9;
// BM25 hyperparameters — Robertson-Spärck-Jones defaults; the corpus is
// only ~29 docs so tuning these further isn't worth it.
const BM25_K1 = 1.5;
const BM25_B = 0.75;

/** Tokenize for BM25: lowercase, split on non-alphanumeric, drop empties. */
const tokenize = (text) =>
  (text || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);

/** Min-max normalize a numeric array to [0, 1]. Returns zeros if all equal. */
const minmaxNormalize = (values) => {
  if (!values.length) return [];
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  if (span < 1e-9) return values.map(() => 0);
  return values.map((v) => (v - lo) / span);
};

/** BM25 score of a query (as token list) against a single doc's token list,
 *  given pre-computed corpus stats (avgDocLen, IDF map). */
const bm25Score = (queryTokens, docTokens, avgDocLen, idfByTerm) => {
  if (!queryTokens.length || !docTokens.length) return 0;
  const docLen = docTokens.length;
  const tf = new Map();
  for (const t of docTokens) tf.set(t, (tf.get(t) || 0) + 1);
  let score = 0;
  for (const qt of queryTokens) {
    const f = tf.get(qt) || 0;
    if (!f) continue;
    const idf = idfByTerm.get(qt) || 0;
    const numerator = f * (BM25_K1 + 1);
    const denominator =
      f + BM25_K1 * (1 - BM25_B + BM25_B * (docLen / avgDocLen));
    score += idf * (numerator / denominator);
  }
  return score;
};

/** Build per-query corpus stats (avgDocLen, IDF) over a list of doc-token
 *  arrays. IDF uses the Robertson-Spärck-Jones formula with a smoothing
 *  guard so a token present in 0 docs gets idf=0 (not negative infinity). */
const buildBm25Stats = (queryTokens, docTokensList) => {
  const N = docTokensList.length;
  const avgDocLen =
    docTokensList.reduce((s, d) => s + d.length, 0) / Math.max(N, 1);
  const df = new Map();
  for (const qt of queryTokens) {
    let count = 0;
    for (const tokens of docTokensList) if (tokens.includes(qt)) count++;
    df.set(qt, count);
  }
  const idfByTerm = new Map();
  for (const [term, dfi] of df) {
    // RSJ idf with the +1 smoothing inside the log keeps the value
    // non-negative for terms that appear in >half the corpus.
    const idf = Math.log(1 + (N - dfi + 0.5) / (dfi + 0.5));
    idfByTerm.set(term, Math.max(0, idf));
  }
  return { avgDocLen, idfByTerm };
};

// --- State ---

let colbertModel = null;
let latestQueryId = 0; // Track the ID of the latest 'rank' request.

// --- Helper Functions ---
// ... (No changes to sendStatus, sendError, getCachedOrFetch, loadModel) ...
const sendStatus = (message) => {
  console.log(`[WORKER][STATUS] ${message}`);
  self.postMessage({ type: "status", payload: message });
};
const sendError = (message, error) => {
  console.error(`[WORKER][ERROR] ${message}`, error);
  self.postMessage({
    type: "error",
    payload: `${message}${error ? `: ${error.message}` : ""}`,
  });
};
const getCachedOrFetch = async (url, displayName) => {
  sendStatus(`Downloading ${displayName}...`);
  const cache = await caches.open(CACHE_NAME);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    console.log(`[WORKER] Cache hit for ${displayName}. Using cached version.`);
    return cachedResponse;
  }
  console.log(
    `[WORKER] Cache miss for ${displayName}. Fetching from network...`,
  );
  const networkResponse = await fetch(url);
  if (!networkResponse.ok) {
    throw new Error(
      `Download failed for ${displayName}: ${networkResponse.statusText}`,
    );
  }
  await cache.put(url, networkResponse.clone());
  console.log(`[WORKER] Successfully fetched and cached ${displayName}.`);
  return networkResponse;
};
const loadModel = async () => {
  if (colbertModel) {
    self.postMessage({ type: "model-ready" });
    return;
  }
  try {
    sendStatus("Initializing WebAssembly module...");
    await init();
    const fileFetchPromises = MODEL_FILES.map((file) => {
      const url = `https://huggingface.co/${MODEL_REPO}/resolve/main/${file}`;
      const displayName = file.split("/").pop();
      return getCachedOrFetch(url, displayName);
    });
    const responses = await Promise.all(fileFetchPromises);
    sendStatus("Decoding model files...");
    const modelFileDataPromises = responses.map((response) =>
      response.arrayBuffer().then((buffer) => new Uint8Array(buffer)),
    );
    const modelFilesData = await Promise.all(modelFileDataPromises);
    const [
      tokenizerData,
      modelData,
      configData,
      sentenceTransformerConfigData,
      denseLayerData,
      denseLayerConfigData,
      specialTokensMapData,
    ] = modelFilesData;
    sendStatus("Instantiating ColBERT model...");
    colbertModel = new ColBERT(
      modelData,
      denseLayerData,
      tokenizerData,
      configData,
      sentenceTransformerConfigData,
      denseLayerConfigData,
      specialTokensMapData,
      COLBERT_LATENCY_BUCKET,
    );
    self.postMessage({ type: "model-ready" });
  } catch (error) {
    sendError("Fatal error during model loading", error);
  }
};

// --- Document Ranking ---

/**
 * Ranks documents asynchronously and cooperatively, allowing it to be interrupted.
 * @param {object} payload - The data for the ranking task.
 */
const rankDocuments = async (payload) => {
  if (!colbertModel) {
    return;
  }

  const { query, documents, queryId } = payload;

  if (queryId !== latestQueryId) {
    console.log(
      `[WORKER] Skipping stale ranking stream #${queryId}. Latest is #${latestQueryId}.`,
    );
    return;
  }

  console.log(
    `[WORKER] Ranking stream #${queryId}: Ranking top ${MAX_DOCS_TO_RANK} of ${documents.length} documents.`,
  );

  const docsToRank = documents.slice(0, MAX_DOCS_TO_RANK);
  const docsToPassThrough = documents.slice(MAX_DOCS_TO_RANK);

  // Pre-tokenize the query + every candidate doc once. BM25 corpus stats
  // (avgDocLen, IDF) are computed against this 29-doc local corpus so the
  // signal is sensitive to how distinctive a query term is *within the
  // candidate set the API surfaced* — which is what matters for the
  // re-rank decision.
  const queryTokens = tokenize(query);
  const docTexts = docsToRank.map((d) => {
    const title = d.title || "";
    const summary = d.summary || "";
    const allTags = (d.tags || []).concat(d["extra-tags"] || []).join(" ");
    return `${title} ${summary} ${allTags}`.trim();
  });
  const docTokensList = docTexts.map(tokenize);
  const { avgDocLen, idfByTerm } = buildBm25Stats(queryTokens, docTokensList);
  const bm25Scores = docTokensList.map((toks) =>
    bm25Score(queryTokens, toks, avgDocLen, idfByTerm),
  );

  /** Re-sort `rankedDocs` by the fused score (ColBERT min-max + BM25
   *  min-max, weighted). Recomputes normalization on every call so the
   *  partial-update stream stays consistent: as ColBERT scores arrive,
   *  the min/max shift and the fused order rebalances. */
  const fuseAndSort = () => {
    const colbertRaw = rankedDocs.map((d) => d.colbertScore);
    const bm25Raw = rankedDocs.map((d) => d._bm25Score);
    const colbertNorm = minmaxNormalize(colbertRaw);
    const bm25Norm = minmaxNormalize(bm25Raw);
    for (let i = 0; i < rankedDocs.length; i++) {
      rankedDocs[i].fusedScore =
        W_COLBERT * colbertNorm[i] + W_BM25 * bm25Norm[i];
    }
    rankedDocs.sort((a, b) => b.fusedScore - a.fusedScore);
  };

  const rankedDocs = [];
  const remainingForRanking = [...docsToRank];

  for (let i = 0; i < docsToRank.length; i++) {
    const document = docsToRank[i];
    // This line pauses the loop and allows the worker to process new messages.
    await new Promise((resolve) => setTimeout(resolve, 0));

    // After the pause, we check if a newer query has arrived. If so, abort.
    if (queryId !== latestQueryId) {
      console.log(
        `[WORKER] Aborting ranking stream #${queryId}. A newer query (#${latestQueryId}) has started.`,
      );
      return; // Exit the function early.
    }

    try {
      remainingForRanking.shift();

      const { data: scores } = colbertModel.similarity({
        queries: [query],
        documents: [docTexts[i]],
      });
      const score = scores[0][0];

      const scoredDocument = {
        ...document,
        colbertScore: score,
        _bm25Score: bm25Scores[i],
      };
      rankedDocs.push(scoredDocument);
      fuseAndSort();

      const partialResult = [
        ...rankedDocs,
        ...remainingForRanking,
        ...docsToPassThrough,
      ];
      self.postMessage({
        type: "rank-update",
        payload: partialResult,
        queryId: queryId,
      });
    } catch (error) {
      console.error(
        `[WORKER] Failed to rank document for query #${queryId}.`,
        error,
      );
      rankedDocs.push(document);
    }
  }

  if (queryId === latestQueryId) {
    const finalResult = [...rankedDocs, ...docsToPassThrough];
    self.postMessage({
      type: "rank-complete",
      payload: finalResult,
      queryId: queryId,
    });
  }
};

// --- Similar Document Re-ranking ---

/**
 * Re-ranks similar documents by scoring each against the source document's content.
 * The source doc's text acts as the "query" and each similar result is scored as a "document".
 */
const rankSimilar = async (payload) => {
  if (!colbertModel) return;

  const { sourceUrl, sourceText, results } = payload;
  const ranked = [];
  const remaining = [...results];

  for (const doc of results) {
    await new Promise((resolve) => setTimeout(resolve, 0));

    try {
      remaining.shift();
      const title = doc.title || "";
      const summary = doc.summary || "";
      const allTags = (doc.tags || [])
        .concat(doc["extra-tags"] || [])
        .join(" ");
      const combinedText = `${title} ${summary} ${allTags}`.trim();

      const { data: scores } = colbertModel.similarity({
        queries: [sourceText],
        documents: [combinedText],
      });

      ranked.push({ ...doc, colbertScore: scores[0][0] });
      ranked.sort((a, b) => b.colbertScore - a.colbertScore);

      self.postMessage({
        type: "rank-similar-update",
        payload: { sourceUrl, results: [...ranked, ...remaining] },
      });
    } catch (error) {
      console.error(
        `[WORKER] Failed to rank similar doc for ${sourceUrl}.`,
        error,
      );
      ranked.push(doc);
    }
  }

  self.postMessage({
    type: "rank-similar-complete",
    payload: { sourceUrl, results: [...ranked] },
  });
};

// --- Main Worker Entry Point ---

self.onmessage = async (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case "load":
      await loadModel();
      break;

    case "rank":
      latestQueryId = payload.queryId;
      rankDocuments(payload); // Fire and forget.
      break;

    case "rank-similar":
      rankSimilar(payload); // Fire and forget.
      break;

    default:
      console.warn(`[WORKER] Received unknown message type: '${type}'`);
      break;
  }
};
