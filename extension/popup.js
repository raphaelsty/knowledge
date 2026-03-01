// api and storage are defined by compat.js (loaded before this script)

const titleInput = document.getElementById("title");
const urlInput = document.getElementById("url");
const summaryInput = document.getElementById("summary");
const tagsInput = document.getElementById("tags");
const saveBtn = document.getElementById("save-btn");
const statusEl = document.getElementById("status");
const optionsLink = document.getElementById("options-link");

function setStatus(message, type) {
  statusEl.textContent = message;
  statusEl.className = "status " + type;
}

// Open options page
optionsLink.addEventListener("click", (e) => {
  e.preventDefault();
  api.runtime.openOptionsPage();
});

// URL schemes where content script injection is not allowed
const RESTRICTED_SCHEMES = [
  "chrome://",
  "chrome-extension://",
  "about:",
  "edge://",
  "moz-extension://",
  "safari-web-extension://",
];

// On popup open: extract metadata from the active tab
(async () => {
  const tabs = await api.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!tab) return;

  urlInput.value = tab.url || "";
  titleInput.value = tab.title || "";

  // Skip content script injection for restricted pages
  if (!tab.url || RESTRICTED_SCHEMES.some((s) => tab.url.startsWith(s))) {
    return;
  }

  try {
    const results = await api.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });

    const meta = results?.[0]?.result;
    if (!meta) return;

    if (meta.title && !titleInput.value) {
      titleInput.value = meta.title;
    }
    if (meta.description) {
      summaryInput.value = meta.description;
    }
    if (meta.keywords) {
      tagsInput.value = meta.keywords;
    }
  } catch {
    // Content script injection may fail on some pages — that's fine
  }
})();

// Save bookmark
saveBtn.addEventListener("click", async () => {
  const url = urlInput.value.trim();
  const title = titleInput.value.trim();

  if (!url || !title) {
    setStatus("Title and URL are required.", "error");
    return;
  }

  const tags = tagsInput.value
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);

  const payload = {
    url,
    title,
    summary: summaryInput.value.trim(),
    tags,
    date: new Date().toISOString().split("T")[0],
  };

  saveBtn.disabled = true;
  setStatus("Saving...", "");

  try {
    const items = await storage.get({ apiUrl: "https://knowledge-web.org" });

    const response = await fetch(
      `${items.apiUrl.replace(/\/+$/, "")}/api/bookmark`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    setStatus("Saved!", "success");
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    saveBtn.disabled = false;
  }
});
