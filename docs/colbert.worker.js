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
const MAX_DOCS_TO_RANK = 30; // We only re-rank the top N documents for performance.

// --- State ---

let colbertModel = null;

// --- Helper Functions ---

/**
 * Posts a status message back to the main thread.
 * @param {string} message The status message to send.
 */
const sendStatus = (message) => {
  console.log(`[WORKER][STATUS] ${message}`);
  self.postMessage({ type: "status", payload: message });
};

/**
 * Posts an error message back to the main thread.
 * @param {string} message The error message to send.
 * @param {Error} [error] The associated error object.
 */
const sendError = (message, error) => {
  console.error(`[WORKER][ERROR] ${message}`, error);
  self.postMessage({
    type: "error",
    payload: `${message}${error ? `: ${error.message}` : ""}`,
  });
};

/**
 * Fetches a file from the browser Cache API or from the network if not present.
 * Caching model files is crucial for fast subsequent loads.
 * @param {string} url - The URL of the file to fetch.
 * @param {string} displayName - A user-friendly name for the file for logging.
 * @returns {Promise<Response>} A promise that resolves to the Response object.
 */
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

  // Clone the response to put it in the cache, as a Response body can only be read once.
  await cache.put(url, networkResponse.clone());
  console.log(`[WORKER] Successfully fetched and cached ${displayName}.`);
  return networkResponse;
};

// --- Model Loading ---

/**
 * Initializes the WASM module and loads the ColBERT model files.
 */
const loadModel = async () => {
  if (colbertModel) {
    console.log("[WORKER] Model already loaded. Not re-initializing.");
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
 * Ranks a set of documents against a query using the ColBERT model.
 * This function streams updates back to the main thread as each document is scored.
 * @param {object} payload - The data for the ranking task.
 * @param {string} payload.query - The user's search query.
 * @param {Array<object>} payload.documents - The list of documents to rank.
 * @param {number|string} payload.queryId - An identifier for the specific query request.
 */
const rankDocuments = (payload) => {
  if (!colbertModel) {
    console.warn(
      "[WORKER] 'rank' message received, but the model is not ready. Ignoring."
    );
    return;
  }

  const { query, documents, queryId } = payload;
  console.log(
    `[WORKER] Ranking stream #${queryId}: Ranking top ${MAX_DOCS_TO_RANK} of ${documents.length} documents.`
  );

  const docsToRank = documents.slice(0, MAX_DOCS_TO_RANK);
  const docsToPassThrough = documents.slice(MAX_DOCS_TO_RANK);

  const rankedDocs = [];
  // A copy to track progress for UI updates. These are docs that are in the queue to be ranked.
  const remainingForRanking = [...docsToRank];

  for (const document of docsToRank) {
    try {
      remainingForRanking.shift();

      // Combine title, summary, and tags for a comprehensive document representation.
      const title = document.title || "";
      const summary = document.summary || "";
      const allTags = (document.tags || [])
        .concat(document["extra-tags"] || [])
        .join(" ");
      const combinedText = `${title} ${summary} ${allTags}`.trim();

      // The core similarity calculation.
      const { data: scores } = colbertModel.similarity({
        queries: [query],
        documents: [combinedText],
      });
      const score = scores[0][0];

      const scoredDocument = { ...document, colbertScore: score };
      rankedDocs.push(scoredDocument);

      // Sort the already-ranked documents to maintain a live-sorted list.
      rankedDocs.sort((a, b) => b.colbertScore - a.colbertScore);

      // Send a partial update to the main thread for a responsive UI.
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
      // If a single document fails, push the original doc so it's not lost from the final result.
      rankedDocs.push(document);
    }
  }

  // Final message with the fully sorted list.
  const finalResult = [...rankedDocs, ...docsToPassThrough];
  self.postMessage({
    type: "rank-complete",
    payload: finalResult,
    queryId: queryId,
  });
};

// --- Main Worker Entry Point ---

/**
 * Handles incoming messages from the main thread and routes them to the appropriate function.
 */
self.onmessage = async (event) => {
  const { type, payload } = event.data;
  const startTime = performance.now();

  console.log(`[WORKER] Received message of type: '${type}'`);

  switch (type) {
    case "load":
      await loadModel();
      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
      if (colbertModel) {
        console.log(
          `[WORKER] Model loading process finished in ${loadTime} seconds.`
        );
      }
      break;

    case "rank":
      rankDocuments(payload);
      const rankTime = ((performance.now() - startTime) / 1000).toFixed(2);
      console.log(
        `[WORKER] Finished ranking stream for queryId #${payload.queryId} in ${rankTime}s.`
      );
      break;

    default:
      console.warn(`[WORKER] Received unknown message type: '${type}'`);
      break;
  }
};
