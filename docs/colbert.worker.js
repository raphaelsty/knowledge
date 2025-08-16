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
    `[WORKER] Cache miss for ${displayName}. Fetching from network...`
  );
  const networkResponse = await fetch(url);
  if (!networkResponse.ok) {
    throw new Error(
      `Download failed for ${displayName}: ${networkResponse.statusText}`
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
      response.arrayBuffer().then((buffer) => new Uint8Array(buffer))
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
      COLBERT_LATENCY_BUCKET
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
      `[WORKER] Skipping stale ranking stream #${queryId}. Latest is #${latestQueryId}.`
    );
    return;
  }

  console.log(
    `[WORKER] Ranking stream #${queryId}: Ranking top ${MAX_DOCS_TO_RANK} of ${documents.length} documents.`
  );

  const docsToRank = documents.slice(0, MAX_DOCS_TO_RANK);
  const docsToPassThrough = documents.slice(MAX_DOCS_TO_RANK);
  const rankedDocs = [];
  const remainingForRanking = [...docsToRank];

  for (const document of docsToRank) {
    // This line pauses the loop and allows the worker to process new messages.
    await new Promise((resolve) => setTimeout(resolve, 0));

    // After the pause, we check if a newer query has arrived. If so, abort.
    if (queryId !== latestQueryId) {
      console.log(
        `[WORKER] Aborting ranking stream #${queryId}. A newer query (#${latestQueryId}) has started.`
      );
      return; // Exit the function early.
    }

    try {
      remainingForRanking.shift();

      const title = document.title || "";
      const summary = document.summary || "";
      const allTags = (document.tags || [])
        .concat(document["extra-tags"] || [])
        .join(" ");
      const combinedText = `${title} ${summary} ${allTags}`.trim();

      const { data: scores } = colbertModel.similarity({
        queries: [query],
        documents: [combinedText],
      });
      const score = scores[0][0];

      const scoredDocument = { ...document, colbertScore: score };
      rankedDocs.push(scoredDocument);
      rankedDocs.sort((a, b) => b.colbertScore - a.colbertScore);

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
        error
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

    default:
      console.warn(`[WORKER] Received unknown message type: '${type}'`);
      break;
  }
};
