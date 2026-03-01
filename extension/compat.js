// Cross-browser compatibility layer (Chrome, Edge, Firefox, Safari)
const api = globalThis.browser || globalThis.chrome;

// Safari does not support storage.sync — fall back to storage.local
const storage = api.storage.sync || api.storage.local;
