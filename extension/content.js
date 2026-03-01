(() => {
  const getMeta = (selectors) => {
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) {
        const value = (el.getAttribute("content") || "").trim();
        if (value) return value;
      }
    }
    return "";
  };

  return {
    title:
      getMeta(['meta[property="og:title"]', 'meta[name="twitter:title"]']) ||
      document.title ||
      "",
    description: getMeta([
      'meta[name="description"]',
      'meta[property="og:description"]',
      'meta[name="twitter:description"]',
    ]),
    keywords: getMeta(['meta[name="keywords"]']),
  };
})();
