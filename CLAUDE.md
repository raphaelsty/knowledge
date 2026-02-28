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

- `sources/` — Python data fetchers (one subpackage per platform)
- `indexer/` — Rust binary that builds the ColBERT index
- `docs/` — static frontend (index.html, CSS, WASM worker, images)
- `database/` — generated JSON files (database.json, triples.json)
- `indices/` — generated ColBERT index (committed for deploy)
- `api/start.sh` — entrypoint for the Rust API server
- `run.py` — main pipeline script
- `build_tag_tree.py` — builds the folder tree from tag triples

## Key details

- Python package is `sources`, not `knowledge_database` (renamed)
- The API is `next-plaid-api` (Rust binary, installed via cargo or built in Docker)
- Frontend `API_BASE_URL` in `docs/index.html` must match the API host
- The indexer requires `libonnxruntime` — if it fails locally, the existing `indices/` still works
