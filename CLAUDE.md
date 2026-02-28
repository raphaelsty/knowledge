# Knowledge

Personal knowledge base: fetches bookmarks from GitHub, HackerNews, Zotero, HuggingFace, and X/Twitter, then serves them via a ColBERT search engine.

## Use the Makefile

Every operation goes through `make`. Do not run raw commands.

```
make install      # install prod dependencies
make install-dev  # install with dev tools (ruff, mypy, pre-commit)
make run          # fetch sources → generate tags → build tree → index
make index        # rebuild only the Rust search index
make serve        # start the API on :8080
make web          # serve the frontend on :3000
make lint         # ruff + mypy
make lint-fix     # auto-fix lint issues
make docker-build # build the Docker image
make launch       # build + run Docker container
make clean        # wipe caches and venv
```

## Project layout

- `sources/` — Python package: data fetchers, tag tree builder, and pipeline client
  - `sources/client.py` — main pipeline orchestrator (`from sources.client import main`)
  - `sources/taxonomy.py` — builds the folder tree from tag triples
- `embeddings/` — Rust binary that builds the ColBERT index
- `web/` — static frontend (index.html, app.jsx, CSS, WASM worker, images)
- `web/data/` — generated JSON files (database.json, sources.json, folder_tree.json, tree.json)
- `multi-vector-database/` — generated ColBERT index (committed for deploy)
- `run.py` — thin entry point (calls `sources.client.main()`)

## Key details

- Python package is `sources`, not `knowledge_database` (renamed)
- The API is `next-plaid-api` (Rust binary, installed via cargo or built in Docker)
- Frontend `API_BASE_URL` in `web/app.jsx` must match the API host
- The embeddings crate requires `libonnxruntime` — if it fails locally, the existing `multi-vector-database/` still works
