<p align="center">
  <a href="https://knowledge-web.org"><img src="web/img/logo.svg" alt="Knowledge" width="460"></a>
</p>

<p align="center">
  <video src="web/img/demo.mp4" autoplay loop muted playsinline width="100%" style="border-radius:10px"></video>
</p>

Knowledge is a personal library. A place to keep what you found interesting on GitHub, on X, in a blog, in a paper, and to read it when you have time. A talk that mattered last week is still worth watching this week. A paper from two months ago still teaches you what you came for. Build your own library, or sit in someone else's for an hour. Spend that hour with Andrej Karpathy's bookmarks and you learn what fifteen years of ML looks like.

Tap the heart on any card and the doc lands in your library, indexed and searchable. Every contributor has a personal page that reads like a curated bookshelf: their tweets, their stars, the papers they wrote, the videos they show up in. Browse it the way you'd browse a friend's bookmarks folder.

Type a query. ColBERT searches the actual contents of every doc, not just titles, and ranks them by how well the words match. Search one library, several at once, or the whole shared corpus.

GitHub, X, Hacker News, Reddit, Stack Overflow, Hugging Face, arXiv, Google Scholar, DBLP, Zotero, YouTube, plus any blog with an RSS feed or sitemap.

The API exposes an MCP server at `/mcp` with fifteen tools: `search`, `latest`, `find_similar`, `intersect_documents`, `feed`, plus bearer-authed `my_library`, `my_timeline`, and `save_document`.

Knowledge has always been a showcase for the information retrieval tools I'm building. It started 4 years ago on a cherche backend and now runs on next-plaid and pylate-rs. The frontend is plain HTML and JS, with a ColBERT WASM worker for re-ranking. The API is a single Rust binary, the pipeline is Python. Everything runs on a single Hetzner VPS. So yes, when you type a query, a quantized ColBERT runs on the server's CPU against a next-plaid index — and then on your phone, an unquantized ColBERT running in WASM re-ranks the results.

## Citation

```bibtex
@software{sourty2026knowledge,
  author  = {Sourty, Raphaël},
  title   = {Knowledge: a library for the internet},
  year    = {2026},
  url     = {https://github.com/raphaelsty/knowledge},
  license = {PolyForm-Noncommercial-1.0.0}
}
```
