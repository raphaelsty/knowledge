<p align="center">
  <a href="https://knowledge-web.org"><img src="web/img/logo.svg" alt="Knowledge" width="460"></a>
</p>

<p align="center">
  <em>A library</em>
</p>

<p align="center">
  <a href="https://knowledge-web.org">knowledge-web.org</a>
</p>

<p align="center">
  <img src="web/img/demo.gif" alt="Demo" width="100%" style="border-radius:10px">
</p>

---

## What Knowledge is

Most software wants your attention right now. Knowledge wants you to come back next Sunday.

It's a personal library. A place to keep what you found interesting on GitHub, on X, in a blog, in a paper, and to read it when you have time. A talk that mattered last week is still worth watching this week. A paper from two months ago still teaches you what you came for.

Build your own library, or sit in someone else's for an hour. Spend that hour with Andrej Karpathy's bookmarks and you learn what fifteen years of ML looks like, the slow shape of what someone saves when no one is watching.

## What it's good at

### Save what you read

Tap the heart on any card and the doc lands in your library, indexed and searchable. The same heart shows you who else saved it, which is a quiet way to find other people reading similar things.

### Sit in someone's library

Every contributor has a personal page that reads like a curated bookshelf: their tweets, their stars, the papers they wrote, the videos they show up in. Browse it the way you'd browse a friend's bookmarks folder.

### Search across everything

Type a query. ColBERT searches the actual contents of every doc, not just titles, and ranks them by how well the words match. Search one library, several at once, or the whole shared corpus.

### Find your topic

Pick a few from the picker (178 in total: `semantic-search`, `ai-safety`, `chain-of-thought`, you get the idea). The feed, your personal page, and every other library narrow to that slice. The selection follows you across pages.

## Sources

GitHub, X, Hacker News, Reddit, Stack Overflow, Hugging Face, arXiv, Google Scholar, DBLP, Zotero, YouTube, plus any blog with an RSS feed or sitemap.

Wire your handles on the settings page. The pipeline runs nightly. Anything you bookmark in-app shows up immediately.

## Connect a model

The API exposes an MCP server at `/mcp` with twelve tools: `search`, `latest`, `find_similar`, `intersect_documents`, `feed`, plus bearer-authed `my_library`, `my_timeline`, and `save_document`.

```
claude mcp add knowledge --transport http https://knowledge-web.org/mcp \
  --header "Authorization: Bearer kn_..."
```

Mint a token at `/profile`.

---

## Technical bits

A Rust API (axum), a Python ingestion pipeline (uv-managed), and Postgres as the single source of truth. ColBERT v2 (small ONNX) handles search via pylate-rs. The index sits on disk and gets updated incrementally by a background scanner. Live in-app bookmarks land in the index within 30 seconds.

| Layer    | Choice                                                                                                                            |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Search   | ColBERT v2 small ONNX, [pylate-rs](https://github.com/lightonai/pylate-rs), [next-plaid](https://github.com/lightonai/next-plaid) |
| API      | Rust, axum, sqlx, tower-http                                                                                                      |
| Pipeline | Python, uv, psycopg, twikit, model2vec                                                                                            |
| DB       | PostgreSQL 16                                                                                                                     |
| Frontend | Plain HTML, esbuild-compiled JSX, ColBERT WASM worker for re-rank                                                                 |
| Edge     | Caddy (HTTPS + reverse proxy)                                                                                                     |
| Auth     | GitHub OAuth, signed session cookies                                                                                              |
| Hosting  | One Hetzner CX33 VPS (4 vCPU, 8 GB RAM)                                                                                           |

Run locally:

```bash
git clone https://github.com/raphaelsty/knowledge
cd knowledge
cp .env.example .env       # DB password + GitHub OAuth ids
make up                    # postgres + api + caddy via docker-compose
make run                   # fetch every source, build the index
open http://localhost:3000
```

Every operation lives in the Makefile. Run `make` with no arguments for the menu.

```
sources/    Python — fetchers, pipeline, SQL, soft-delete worker
api/        Rust API (single binary, single port)
web/        Static frontend + ColBERT WASM worker
clients/    Local helpers (twitter feeder, etc.)
run.py      Thin entrypoint
Makefile    Every operation
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Free to use, modify, and self-host for non-commercial purposes. Get in touch for anything else.

---

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
