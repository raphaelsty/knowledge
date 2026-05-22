"""Fully-local document categorization.

Each document is matched against the 178-row `document_categories`
catalogue by cosine similarity between sentence embeddings — both
the catalogue (computed once and cached) and the documents are
embedded with the same small `sentence-transformers` model. The
top-K most similar categories are assigned, with a confidence-gap
heuristic deciding how many of the K actually apply.

Design constraints
------------------
- Zero external API spend. Runs entirely on CPU.
- Must fit comfortably on the prod box (Hetzner CX33: 4 vCPU,
  8 GB RAM) alongside Postgres, the API, Caddy, and the clean
  daemon.
- Lives inside `run.py` so categorization happens automatically
  after every fetch + clean pass without a separate cron.

Model choice
------------
`minishlab/potion-base-32M` (~125 MB on disk, ~150 MB RSS at
runtime, 512-dim embeddings). Potion models are *static* — they
distil a sentence-transformer's vocabulary into a per-token lookup
table and a pooling step, so encoding has no neural-network
forward pass and runs at ~50k sentences/sec on a CPU. That makes
the categorization pass effectively free wall-clock-wise on the
4-vCPU / 8 GB Hetzner box (encoding 411k documents takes seconds,
not minutes).

Quality is below BGE-large for nuanced short text, which means
the assignment thresholds below have to do more of the work —
borderline matches stay uncategorized. The trade-off is worth it
for this pipeline: we want category assignments to be a
nice-to-have layered on top of the rest of `run.py`, not a step
that doubles the wall-clock of a fetch.

Assignment heuristic
--------------------
For each doc we rank all 178 categories, take the top 3, then:
1. Always keep rank 1 (most-similar category) → `is_primary = TRUE`.
2. Keep rank 2/3 if their cosine similarity is ≥ `KEEP_RATIO` of
   rank 1's score *and* exceeds `ABS_FLOOR`. This prevents a doc
   that's only weakly about one topic from picking up two extra
   noisy secondary categories.

The `score` column persists the raw cosine similarity (0..1 since
embeddings are L2-normalised) so the UI can sort or de-emphasise
low-confidence assignments later.

Usage
-----
    # Preview against 30 recent uncategorized docs:
    uv run python -m sources.utils.categorize --limit 30 --preview

    # Commit assignments for the next 5000 uncategorized docs:
    uv run python -m sources.utils.categorize --limit 5000 --commit

    # Inside the pipeline (called from run.py — no CLI):
    from sources.utils.categorize import categorize_all
    categorize_all(database_url)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import psycopg

LOG = logging.getLogger("categorize")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://knowledge:knowledge@127.0.0.1:5433/knowledge",
)

# Static Potion model. First call downloads to ~/.cache/huggingface;
# subsequent calls are warm. Static models don't need a query prefix
# — they're vocabulary lookups, no asymmetric encoder behaviour to
# steer.
EMBED_MODEL = os.environ.get("CATEGORIZE_EMBED_MODEL", "minishlab/potion-base-32M")

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "categorize"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Assignment heuristic: bias for precision over recall. A document
# that's genuinely multi-topic will earn its second/third slug
# naturally; a noisy short tweet stays uncategorized rather than
# getting a random label that pollutes the per-category feed.
MAX_CATS = 3
KEEP_RATIO = 0.95  # rank-N score must be ≥ this fraction of rank-1
ABS_FLOOR = 0.42  # raw cosine sim below this is not assigned at all

# Anchor-refinement parameters. After a description-only first pass
# we identify each category's most-confident top-1 matches, take
# their embedding centroid, and blend it back into the category
# prototype. The blended prototype then drives the final assignment
# pass — so every category gets pulled toward the actual cluster of
# documents that look like it, not just the language of its
# description. Static models like Potion are weak at zero-shot
# topical placement, but they're excellent at "which docs are
# nearest neighbours of these few anchors" — so this loop turns
# their weakness into a strength.
ANCHOR_MIN_SCORE = 0.55  # top-1 cosine to qualify as an anchor candidate
ANCHOR_MIN_GAP = 0.05  # top-1 must beat top-2 by at least this
ANCHOR_MIN_COUNT = 5  # below this many anchors, keep description as-is
ANCHOR_MAX_PER_CAT = 30  # cap to keep refinement robust to outliers
ANCHOR_BLEND = 0.65  # 0 = pure anchor centroid, 1 = pure description

# Batch sizes — encoding is fast on CPU; the bottleneck is the
# Postgres round-trip when --commit is on. 256 keeps memory low.
EMBED_BATCH = 256


# ── Schema bootstrap ─────────────────────────────────────────────────


def ensure_schema(conn: psycopg.Connection) -> None:
    """Idempotent create of the junction table. Safe to call before
    every run — the table may not yet exist on a fresh DB."""
    sql_path = Path(__file__).resolve().parents[1] / "sql" / "document_category_assignments.sql"
    if not sql_path.exists():
        LOG.warning("schema file missing: %s", sql_path)
        return
    with conn.cursor() as cur:
        cur.execute(sql_path.read_text())
    conn.commit()


# ── Model + catalogue ────────────────────────────────────────────────


class _ModelAdapter:
    """Wraps a model2vec StaticModel in a sentence-transformers-shaped
    `.encode(texts, ...) -> np.ndarray` API so the rest of the
    pipeline doesn't care which library is underneath. Embeddings
    are L2-normalised on the way out so downstream code can just
    matmul to get cosine similarity."""

    def __init__(self, model) -> None:
        self._model = model

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # model2vec.encode handles batching internally; we keep the
        # same kwarg surface for API parity.
        embeds = self._model.encode(texts, show_progress_bar=show_progress_bar).astype(np.float32)
        if normalize_embeddings:
            embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9
        return embeds


def _load_model() -> _ModelAdapter:
    """Lazy import so test/CI environments without model2vec can
    still import this module (`ensure_schema` works without it)."""
    from model2vec import StaticModel

    LOG.info("loading static embedding model %s", EMBED_MODEL)
    return _ModelAdapter(StaticModel.from_pretrained(EMBED_MODEL))


def fetch_categories(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, slug, name, group_name, description " "FROM document_categories ORDER BY sort_order")
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def category_embed_text(cat: dict) -> str:
    # Group name disambiguates near-duplicate slugs across groups
    # (e.g. "world-models" exists in Multimodal & Vision *and* in
    # RL & Robotics under similar phrasing).
    return f"{cat['name']} ({cat['group_name']}). {cat['description']}"


def category_embeddings(model, cats: list[dict]) -> np.ndarray:
    """L2-normalised (n_cats, dim) matrix; cached to disk."""
    cache_path = CACHE_DIR / f"cat_embeds_{EMBED_MODEL.replace('/', '_')}.npz"
    if cache_path.exists():
        try:
            z = np.load(cache_path, allow_pickle=False)
            if int(z["count"]) == len(cats):
                LOG.info("loaded %d cached category embeddings", len(cats))
                return z["embeds"]
        except Exception:
            LOG.warning("category embedding cache unreadable, recomputing")
    LOG.info("embedding %d categories", len(cats))
    inputs = [category_embed_text(c) for c in cats]
    embeds = model.encode(inputs, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.savez(cache_path, embeds=embeds, count=np.int64(len(cats)))
    return embeds


# ── Documents ────────────────────────────────────────────────────────


def doc_input_text(doc: dict) -> str:
    title = (doc.get("clean_title") or doc.get("title") or "").strip()
    summary = (doc.get("clean_summary") or doc.get("summary") or "").strip()
    # Raw tweets carry "@handle" as title — pure noise for similarity.
    if title.startswith("@") and " " not in title:
        title = ""
    text = (title + "\n\n" + summary).strip()
    # Potion handles long inputs by token-level pooling; capping at
    # 1500 chars keeps encoding deterministic and bounded.
    return text[:1500]


# Minimum input length before we bother categorizing. Tweets that
# the clean daemon couldn't extract meaningful body from (just an
# @handle, a URL, a media marker) collapse to <20 chars here — they
# produce noisy embeddings that frequently get bogus high-confidence
# matches against narrow categories purely on token-vector geometry.
# Filtering them out is more honest than assigning them a random
# slug.
DOC_INPUT_MIN_LEN = 25


def iter_uncategorized_batches(
    conn: psycopg.Connection,
    batch_size: int,
    sources: list[str] | None,
    recompute: bool,
    limit: int | None,
) -> Iterable[list[dict]]:
    """Stream docs in stable batches. We keyset-paginate by (date,
    url) so commits inside the loop don't cause skipping."""
    last_date = None
    last_url = None
    seen = 0
    sql_parts = [
        "SELECT d.user_id, d.url, d.source, d.title, d.summary,",
        "       d.clean_title, d.clean_summary, d.date",
        "FROM documents d",
    ]
    if not recompute:
        sql_parts.append("LEFT JOIN document_category_assignments a" " ON a.user_id = d.user_id AND a.url = d.url")
    where = ["COALESCE(NULLIF(d.clean_summary, ''), NULLIF(d.summary, '')) IS NOT NULL"]
    if not recompute:
        where.append("a.user_id IS NULL")
    if sources:
        where.append("d.source = ANY(%(sources)s)")

    while True:
        keyset = list(where)
        params: dict = {"limit": batch_size, "sources": sources}
        if last_date is not None:
            keyset.append("(d.date, d.url) < (%(last_date)s, %(last_url)s)")
            params["last_date"] = last_date
            params["last_url"] = last_url
        sql = (
            "\n".join(sql_parts)
            + "\nWHERE "
            + " AND ".join(keyset)
            + "\nORDER BY d.date DESC NULLS LAST, d.url DESC"
            + "\nLIMIT %(limit)s"
        )
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c.name for c in cur.description]
            rows = cur.fetchall()
        if not rows:
            return
        batch = [dict(zip(cols, r, strict=False)) for r in rows]
        yield batch
        last_date = batch[-1]["date"]
        last_url = batch[-1]["url"]
        seen += len(batch)
        if limit is not None and seen >= limit:
            return


# ── Assignment heuristic ─────────────────────────────────────────────


def pick_categories(sims_row: np.ndarray) -> list[tuple[int, float]]:
    """Return (idx, score) tuples for the chosen categories — at most
    MAX_CATS, ordered most-relevant first."""
    top = np.argpartition(-sims_row, MAX_CATS)[:MAX_CATS]
    top = top[np.argsort(-sims_row[top])]
    chosen: list[tuple[int, float]] = []
    top1 = float(sims_row[top[0]])
    if top1 < ABS_FLOOR:
        return []  # too weak — leave the doc uncategorized
    for rank, idx in enumerate(top):
        score = float(sims_row[idx])
        if rank == 0:
            chosen.append((int(idx), score))
            continue
        if score >= ABS_FLOOR and score >= KEEP_RATIO * top1:
            chosen.append((int(idx), score))
        else:
            break  # ranks are sorted descending; once one fails the rest will too
    return chosen


# ── Anchor-based prototype refinement ───────────────────────────────


def refine_prototypes(
    cat_embeds: np.ndarray,
    doc_embeds: np.ndarray,
    cats: list[dict],
) -> tuple[np.ndarray, dict[str, int]]:
    """Blend each category's description embedding with the centroid
    of its highest-confidence top-1 documents.

    Procedure
    ---------
    1. Compute cosine sims between every doc and every category.
    2. For each doc, take the top-1 category and its score, plus the
       top-2 score. The doc is an "anchor candidate" for its top-1
       category iff:
         - top1 score ≥ ANCHOR_MIN_SCORE (absolute strength)
         - top1 − top2 ≥ ANCHOR_MIN_GAP  (unambiguous)
    3. For each category with ≥ ANCHOR_MIN_COUNT candidates, keep
       the strongest ANCHOR_MAX_PER_CAT (sorted by top-1 score) and
       average their doc embeddings into a centroid.
    4. New prototype = ANCHOR_BLEND × description + (1−ANCHOR_BLEND)
       × centroid, L2-normalised. Categories below the anchor floor
       keep their original description embedding unchanged.

    The cap on anchors per category (ANCHOR_MAX_PER_CAT = 30) bounds
    the influence of any single big bucket — without it,
    `vision-language-action` (the worst sink in the bge run) would
    pull the centroid toward whatever noise made it a sink, not the
    other way around.
    """
    sims = doc_embeds @ cat_embeds.T  # (N_docs, N_cats)
    top1_idx = np.argmax(sims, axis=1)
    rows = np.arange(len(sims))
    top1_score = sims[rows, top1_idx]
    # Mask top-1 out to find top-2.
    masked = sims.copy()
    masked[rows, top1_idx] = -np.inf
    top2_score = masked.max(axis=1)
    del masked

    refined = cat_embeds.copy()
    anchor_counts: dict[str, int] = {}
    for ci in range(len(cats)):
        # Anchor candidates: docs whose top-1 is this category AND
        # pass the absolute + gap thresholds.
        candidate_mask = (
            (top1_idx == ci) & (top1_score >= ANCHOR_MIN_SCORE) & ((top1_score - top2_score) >= ANCHOR_MIN_GAP)
        )
        cand_idx = np.where(candidate_mask)[0]
        anchor_counts[cats[ci]["slug"]] = int(cand_idx.size)
        if cand_idx.size < ANCHOR_MIN_COUNT:
            continue
        # Keep only the strongest ANCHOR_MAX_PER_CAT anchors so a
        # sink category can't drown its prototype in noise.
        if cand_idx.size > ANCHOR_MAX_PER_CAT:
            order = np.argsort(-top1_score[cand_idx])[:ANCHOR_MAX_PER_CAT]
            cand_idx = cand_idx[order]
        centroid = doc_embeds[cand_idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        blended = ANCHOR_BLEND * cat_embeds[ci] + (1.0 - ANCHOR_BLEND) * centroid
        refined[ci] = blended / (np.linalg.norm(blended) + 1e-9)
    return refined, anchor_counts


# ── Persistence ──────────────────────────────────────────────────────


def persist_batch(
    conn: psycopg.Connection,
    rows: list[tuple[int, str, int, float, bool]],
) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO document_category_assignments
                (user_id, url, category_id, score, is_primary)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, url, category_id) DO UPDATE
              SET score = EXCLUDED.score,
                  is_primary = EXCLUDED.is_primary
            """,
            rows,
        )
    conn.commit()


# ── Public entry point (used by run.py) ──────────────────────────────


def _fetch_corpus_for_refinement(conn: psycopg.Connection, sources: list[str] | None) -> list[dict]:
    """Pull every assignable document in one shot. Static models
    make full-corpus encoding cheap (a few seconds for 400k docs at
    ~50k sentences/sec) so we just hold them in memory — no
    streaming gymnastics. The buyback is one tight, deterministic
    refinement pass per call."""
    sql = [
        "SELECT user_id, url, source, title, summary,",
        "       clean_title, clean_summary, date",
        "FROM documents",
        "WHERE COALESCE(NULLIF(clean_summary, ''), NULLIF(summary, '')) IS NOT NULL",
    ]
    params: list = []
    if sources:
        sql.append("AND source = ANY(%s)")
        params.append(sources)
    sql.append("ORDER BY date DESC NULLS LAST, url DESC")
    with conn.cursor() as cur:
        cur.execute("\n".join(sql), params)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def _existing_assignment_keys(
    conn: psycopg.Connection,
) -> set[tuple[int, str]]:
    """(user_id, url) tuples that already have ≥1 assignment row."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT user_id, url FROM document_category_assignments")
        return {(r[0], r[1]) for r in cur.fetchall()}


def categorize_all(
    database_url: str = DATABASE_URL,
    limit: int | None = None,
    sources: list[str] | None = None,
    recompute: bool = False,
) -> dict:
    """Two-stage categorization.

    Stage 1 — bootstrap embedding-only:
        Embed every assignable document once. Each category starts
        from its description embedding (cached on disk).

    Stage 2 — anchor refinement:
        For each category, find its most-confident top-1 docs and
        blend their embedding centroid into the category's
        prototype. See `refine_prototypes` for details.

    Stage 3 — final assignment:
        Re-score every document against the refined prototypes and
        upsert assignment rows. Documents that already have
        assignments are left alone unless `recompute=True`.

    Returns a small stats dict for the caller to log.
    """
    stats: dict[str, int] = {
        "corpus_docs": 0,
        "refined_categories": 0,
        "skipped_categories": 0,
        "docs_assigned": 0,
        "assignments_written": 0,
        "docs_skipped_weak": 0,
    }
    with psycopg.connect(database_url) as conn:
        ensure_schema(conn)
        cats = fetch_categories(conn)
        if not cats:
            LOG.warning("document_categories is empty — nothing to do")
            return stats

        model = _load_model()
        cat_embeds = category_embeddings(model, cats)

        # ── Stage 1: embed every assignable document.
        LOG.info("fetching corpus for refinement")
        docs = _fetch_corpus_for_refinement(conn, sources)
        stats["corpus_docs"] = len(docs)
        if not docs:
            LOG.info("no documents to categorize")
            return stats
        # Drop docs whose effective input collapsed to almost
        # nothing — encoding them produces high-variance noise that
        # surfaces as bogus high-confidence assignments.
        raw_docs = docs
        kept: list[dict] = []
        kept_inputs: list[str] = []
        skipped_too_short = 0
        for d in raw_docs:
            text = doc_input_text(d)
            if len(text) < DOC_INPUT_MIN_LEN:
                skipped_too_short += 1
                continue
            kept.append(d)
            kept_inputs.append(text)
        docs = kept
        if skipped_too_short:
            LOG.info(
                "skipping %d documents with effectively empty input",
                skipped_too_short,
            )
        LOG.info("encoding %d documents", len(docs))
        doc_embeds = model.encode(
            kept_inputs,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # ── Stage 2: anchor-refine the category prototypes.
        LOG.info("refining prototypes from anchor documents")
        refined_embeds, anchor_counts = refine_prototypes(cat_embeds, doc_embeds, cats)
        refined_n = sum(1 for n in anchor_counts.values() if n >= ANCHOR_MIN_COUNT)
        stats["refined_categories"] = refined_n
        stats["skipped_categories"] = len(cats) - refined_n
        # Surface the categories with the strongest anchor support so
        # the operator can sanity-check the loop has latched onto
        # something real.
        strongest = sorted(anchor_counts.items(), key=lambda kv: -kv[1])[:8]
        LOG.info(
            "refined %d/%d categories; top anchors: %s",
            refined_n,
            len(cats),
            ", ".join(f"{slug}={n}" for slug, n in strongest if n > 0),
        )

        # ── Stage 3: final assignment with refined prototypes.
        if recompute:
            existing: set[tuple[int, str]] = set()
        else:
            existing = _existing_assignment_keys(conn)

        sims = doc_embeds @ refined_embeds.T  # (N_docs, N_cats)
        write_rows: list[tuple[int, str, int, float, bool]] = []
        FLUSH_EVERY = 2000
        for doc, sims_row in zip(docs, sims, strict=False):
            key = (doc["user_id"], doc["url"])
            if key in existing:
                continue
            picks = pick_categories(sims_row)
            if not picks:
                stats["docs_skipped_weak"] += 1
                continue
            for rank, (idx, score) in enumerate(picks):
                write_rows.append(
                    (
                        doc["user_id"],
                        doc["url"],
                        cats[idx]["id"],
                        score,
                        rank == 0,
                    )
                )
            stats["docs_assigned"] += 1
            stats["assignments_written"] += len(picks)
            if len(write_rows) >= FLUSH_EVERY:
                persist_batch(conn, write_rows)
                write_rows.clear()
            if limit is not None and stats["docs_assigned"] >= limit:
                break
        if write_rows:
            persist_batch(conn, write_rows)
    return stats


# ── CLI ──────────────────────────────────────────────────────────────


def _preview(database_url: str, limit: int, sources: list[str] | None) -> None:
    """Like categorize_all but prints picks instead of writing."""
    with psycopg.connect(database_url) as conn:
        ensure_schema(conn)
        cats = fetch_categories(conn)
        model = _load_model()
        cat_embeds = category_embeddings(model, cats)
        # Single batch, just enough for preview.
        docs: list[dict] = []
        for batch in iter_uncategorized_batches(conn, limit, sources, recompute=False, limit=limit):
            docs.extend(batch)
            break
        if not docs:
            print("(no uncategorized docs)")
            return
        doc_embeds = model.encode(
            [doc_input_text(d) for d in docs],
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        sims = doc_embeds @ cat_embeds.T
        for doc, row in zip(docs, sims, strict=False):
            picks = pick_categories(row)
            head = (doc.get("clean_title") or doc.get("title") or "").strip()
            print(f"\n— [{doc['source']}] {doc['url']}")
            print(f"  title:  {head[:110]}")
            if not picks:
                print(f"  picked: (none) top1={float(row.max()):.2f}")
                continue
            chunks = [f"{cats[idx]['slug']}({score:.2f})" for idx, score in picks]
            print(f"  picked: {' · '.join(chunks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Preview mode runs on a small sample; commit mode defaults to
    # "everything uncategorized" (None → unlimited).
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Filter by source (repeatable, e.g. --source twitter --source arxiv).",
    )
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.preview and not args.commit:
        LOG.error("pass either --preview or --commit")
        sys.exit(2)
    if args.preview:
        _preview(DATABASE_URL, args.limit or 30, args.source)
        return
    stats = categorize_all(DATABASE_URL, limit=args.limit, sources=args.source, recompute=args.recompute)
    LOG.info("done: %s", stats)


if __name__ == "__main__":
    main()
