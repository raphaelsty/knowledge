// api and storage are defined by compat.js (loaded before this script)

const apiUrlInput = document.getElementById("api-url");
const saveBtn = document.getElementById("save");
const statusEl = document.getElementById("status");

(async () => {
  const items = await storage.get({ apiUrl: "https://knowledge-web.org" });
  apiUrlInput.value = items.apiUrl;
})();

saveBtn.addEventListener("click", async () => {
  const apiUrl = apiUrlInput.value.replace(/\/+$/, "");
  await storage.set({ apiUrl });
  statusEl.textContent = "Saved.";
  setTimeout(() => {
    statusEl.textContent = "";
  }, 2000);
});
