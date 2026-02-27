# Tag Tree Improvement Plan

## Constraints
- **No hardcoded tag names, patterns, or lists.** Every filter/rule must be data-driven and work on any set of documents/tags.
- Hub detection via clustering coefficient was too aggressive (removed python, deep-learning) — abandoned.
- Hardcoded SOURCE_META_PATTERNS regex was removed.

## Current State

- **7,230 documents** assigned across **529 folders**
- **34 top-level folders**, heavily unbalanced (`twitter`: 1,480 docs vs `metrics`: 10 docs)
- 0 document duplication (every doc placed exactly once)
- Assignment uses inverse document frequency (rarest matching tag wins)

## Diagnosed Weaknesses

### W1: Garbage tags pollute the graph
Common English words extracted from tweets (`time`, `no`, `us`, `was`, `now`, `find`, `make`, `hit`, `left`, `set`, `fun`, `edge`, `fit`, `force`) act as high-degree hub nodes. They create spurious co-occurrence edges that pull unrelated documents into the same clusters. Single-letter tags (`r`, `c`, `v`, `x`) and numeric tags (`2024`, `2025`, `2.0`) add noise.

**Evidence:** `argpartition` ended up in `fastapi/columnar-format/` because garbage tags bridged unrelated topics through the co-occurrence graph.

### W2: IDF-only assignment sends docs to irrelevant rare tags
A document with tags `[numpy, python, argpartition, topk]` gets assigned to `argpartition` (rarest tag) even though `numpy` is a much better semantic fit for any folder the doc lands in. The rarest tag might be a niche concept that clustered with unrelated niche concepts.

**Evidence:** `argpartition` clustered with `columnar-format` (both rare, weak co-occurrence link), pulling the numpy tutorial into fastapi.

### W3: Folder naming picks unrepresentative tags
One obscure tag from one document can become a folder name for a cluster of unrelated docs. Examples: `thoracic-diseases` (naming an ML folder), `coca-cola` (naming a Ukraine war folder), `greenland-shark`, `navier-stokes`, `mathematics-stackexchange`.

**Evidence:** The naming algorithm picks by within-cluster degree + containment + name quality, but when clusters are incoherent, even the "best" name is misleading.

### W4: No document duplication limits discoverability
A paper about "sentence embeddings with BERT" appears only under one tag, even though it's equally relevant to `bert`, `sentence-embeddings`, `natural-language-processing`, and `transformers`. Users browsing any of those topics won't find it.

### W5: Twitter content is a monolithic blob
1,480 docs (20% of the KB) live under `twitter` with minimal structure. 215 docs share only the tags `twitter`/`twitter-data`/`twitter-api`, making them impossible to organize by topic. Many tweets have meaningful extra-tags but the twitter-api cluster overwhelms.

### W6: Incoherent sub-clusters from bad graph edges
Since the co-occurrence graph has noise from garbage tags, Ward hierarchical clustering produces incoherent sub-folders. `fastapi` contains `linux-kernel` and `columnar-format`. `thoracic-diseases` contains `model-parallel`. `navier-stokes` contains `web3-is-going-great`.

### W7: Redundant/near-duplicate tags survive dedup
Tags like `distillation` vs `model-distillation` vs `knowledge-distillation`, `scene-text` vs `scene-text-recognition` vs `text-recognition`, and compound tags with semicolons (`snowflake;python;snowpark;machine-learning`) escape the current dedup pipeline.

### W8: French tags mixed with English equivalents
Tags like `discute-avec-raphael`, `dictionnaire`, `mathematiques`, `reseaux-bayesiens`, `rigolo`, `faire`, `coree` don't merge with their English equivalents and create orphan clusters.

---

## Improvement Steps

### Step 1: Tag Quality Filter (pre-graph)
**Goal:** Remove garbage tags before they create spurious edges.

**Algorithm:**
1. Build a stopword set dynamically: load NLTK English stopwords + detect single-char tags + detect pure-numeric tags + detect tags shorter than 3 chars
2. Detect "English common words": for every single-word tag (no hyphens), check if it appears in a frequency list (NLTK words corpus or similar). If it does AND it has no WordNet synset related to computing/science, flag it
3. Split compound semicolon tags (e.g., `snowflake;python;snowpark;machine-learning` → 4 separate tags) before any processing
4. Remove flagged tags from triples before graph construction
5. Apply to document tags too when generating doc-triples

**Metric:** Count of removed tags, verify no domain-specific tags are lost (spot-check sample).

**Status:** [x] DONE — KEPT

---

### Step 2: Improved Tag Deduplication
**Goal:** Catch remaining near-duplicates and merge French↔English variants.

**Algorithm:**
1. Add a pass for semicolon/comma-separated compound tags → split into individual tags
2. Add a pass for prefix-suffix dedup: if tag A is a prefix/suffix of tag B and they share >50% co-occurrence neighbors, merge (e.g., `scene-text` ← `scene-text-recognition`)
3. Add cross-language dedup: use the embedding model to find French↔English pairs (embed both, if cosine sim > 0.85 and one tag matches a known French pattern, merge into the English form)
4. Merge tags where one is a strict subset of words in the other AND they have high co-occurrence overlap (e.g., `distillation` ← `knowledge-distillation` when overlap > 0.6)

**Metric:** Count of additional merges. Spot-check 20 merged pairs for correctness.

**Status:** [x] DONE — KEPT (modest impact)

---

### Step 3: Semantic-Aware Document Assignment (replace pure IDF)
**Goal:** Assign each document to the tag that best represents it in context of where that tag lives in the tree.

**Algorithm:**
1. For each document, get all matching tree tags
2. Score each candidate tag using a weighted combination:
   - **IDF score** (current): `1 / log(1 + doc_count[tag])` — still prefer specific tags, but log-damped
   - **Semantic relevance**: cosine similarity between the document's tag-set centroid (embed all doc tags, average) and the candidate tag embedding
   - **Cluster coherence**: cosine similarity between the document's tag-set centroid and the centroid of the candidate tag's tree neighborhood (sibling tags in the same folder)
3. Final score = `0.3 * idf + 0.4 * semantic_relevance + 0.3 * cluster_coherence`
4. Assign to the highest-scoring tag

**Metric:** Sample 50 documents, manually verify placement quality vs current IDF-only.

**Status:** [x] DONE — REVERTED

---

### Step 4: Document Duplication (multi-placement)
**Goal:** Place documents in multiple relevant locations for better discoverability.

**Algorithm:**
1. After primary assignment (Step 3), for each document:
   - Compute relevance score for ALL its matching tree tags (same formula as Step 3)
   - Primary placement: highest score
   - Secondary placements: any tag with score > `0.7 * primary_score` AND the tag belongs to a DIFFERENT top-level folder
2. Cap at 3 placements per document to avoid excessive duplication
3. Mark duplicated docs in the JSON output so the UI can show a "also appears in..." hint

**Metric:** Average placements per doc, total duplication ratio, spot-check 20 multi-placed docs.

**Status:** [x] DONE — KEPT

---

### Step 5: Folder Naming Overhaul
**Goal:** Ensure every folder name accurately represents its content.

**Algorithm:**
1. After tree construction, for each folder:
   - Collect all document titles + tags in the subtree
   - Embed them and compute the subtree centroid
   - Score candidate names (current folder name + all tags in the subtree) by:
     - Cosine similarity to subtree centroid
     - Name quality score (current `_name_score` heuristics)
     - Penalty for tags that appear in many OTHER subtrees (prefer distinctive names)
   - Pick the highest-scoring name
2. Also try generating a 2-3 word descriptive name: embed the top-5 document titles, find the closest WordNet hypernym, use it if it scores better than any tag
3. Validate: no two sibling folders share the same name

**Metric:** Count of renamed folders, sample 20 renames for quality.

**Status:** [x] DONE — KEPT

---

### Step 6: Twitter Decomposition
**Goal:** Break the monolithic twitter cluster into topic-based sub-folders.

**Algorithm:**
1. For documents with ONLY generic twitter tags (`twitter`, `twitter-data`, `twitter-api`), attempt to extract meaningful topics:
   - Embed each document's title + summary (first 200 chars)
   - Run a mini-clustering (k-means or Louvain) on these embeddings
   - Name each mini-cluster using the same naming algorithm (Step 5)
2. For documents that DO have meaningful extra-tags beyond twitter, those tags should already route them correctly (Steps 1-3 handle this)
3. Consider not using `twitter`/`twitter-api` as tree tags at all — they're source metadata, not topic tags. Filter them out the same way we filter garbage tags in Step 1, but only from graph construction (keep them in document metadata).

**Metric:** Size of twitter cluster before/after, coherence of resulting sub-clusters.

**Status:** [x] DONE — KEPT

---

### Step 8: Meta-Tag Detection & Grab-Bag Dissolution
**Goal:** Prevent source/format/platform tags from forming top-level folders. Docs should cluster by topic.

**Algorithm:**
1. **Meta-tag detection** (two-pass, after embeddings loaded):
   - Pass 1 — Community spread: run Louvain on full graph, measure neighbor community concentration. Tags in bottom p25 = meta (catches python, twitter, typescript, hacktoberfest, etc.)
   - Pass 2 — Semantic mismatch: for remaining high-degree tags, check if tag embedding is distant from neighbors' embeddings. Tags in bottom p15 = meta (catches arxiv-doc type source tags)
2. **Title-based tag propagation**: group docs by exact title, propagate most common tags from tagged docs to untagged docs in same group (catches Twitter @username patterns)
3. **Grab-bag folder dissolution** (post-construction):
   - Criterion 1: folder name semantically distant from content centroid (bottom p25)
   - Criterion 2: folder name embeds >0.82 sim to any meta-tag
   - Dissolved sub-folders promoted to top level if ≥20 tags and not themselves meta-tags; otherwise merged into nearest keeper

**Metric:** % of docs in meta-tag folders: 53.6% → 1.5%

**Status:** [x] DONE — KEPT

---

### Step 7: Post-Construction Coherence Validation
**Goal:** Detect and fix remaining incoherent placements automatically.

**Algorithm:**
1. For each folder, compute the centroid of all document title/summary embeddings
2. For each document in the folder, compute its distance to the folder centroid
3. Flag outliers: documents whose distance to their folder centroid is > 2 standard deviations above the mean
4. For each outlier, find the folder whose centroid is closest → re-assign if significantly closer
5. After reassignment, recompute and re-check (iterate up to 3 times)

**Metric:** Number of outliers detected, number reassigned, before/after coherence scores.

**Status:** [x] DONE — KEPT (modest impact)

---

## Execution Order

1. **Step 1** (Tag Quality Filter) — foundational, fixes the graph that everything else depends on
2. **Step 2** (Improved Dedup) — reduces noise further, depends on Step 1 output
3. **Step 6** (Twitter Decomposition) — can run after Steps 1-2, breaks the biggest problem cluster
4. **Step 3** (Semantic Assignment) — needs clean graph from Steps 1-2
5. **Step 4** (Document Duplication) — builds on Step 3's scoring
6. **Step 5** (Folder Naming) — runs after tree is built with Steps 3-4
7. **Step 7** (Coherence Validation) — final cleanup pass

## Results Log

| Step | Date | Outcome | Notes |
|------|------|---------|-------|
| Step 1: Tag Quality Filter | 2026-02-25 | KEPT | Removed 56 garbage tags (stopwords, single-char, numeric). Split compound semicolon tags. `argpartition` moved out of fastapi. `thoracic-diseases` gone. fastapi reorganized into rest-api. 32 top-level folders (was 34), 7220 docs (was 7230), 525 folders (was 530). Hub detection (clustering coeff) tried and REVERTED — it removed legitimate high-value tags (python, deep-learning, etc.). Hardcoded SOURCE_META_PATTERNS also removed per user requirement. |
| Step 2: Improved Dedup | 2026-02-25 | KEPT | Added Pass 3 (word-subset dedup): 120 additional merges. `time-series` ← `time-series-analysis/classification/segmentation/database`. First attempt (168 merges) chained `reinforcement-learning` into `machine-learning` via union-find — fixed with frequency guard (top-100 tags excluded from subset merging). 29 top-level folders (was 32), 505 subfolders (was 525). |
| Step 6: Twitter Decomp | 2026-02-25 | KEPT (modest) | Implemented generic `enrich_undertagged_docs`: for docs with no useful tags after filtering, embed title+summary and assign top-3 nearest tree tags. 52 docs enriched, 7272 assigned (was 7220). Twitter cluster still large (1963 docs) — most undertagged tweets lack title/summary content. Approach is data-driven, not twitter-specific. |
| Step 3: Semantic Assign | 2026-02-25 | REVERTED | Tried weighted IDF+semantic scoring (0.4/0.6 and 0.7/0.3). Both pushed MORE docs into twitter (2468 and 2067 vs 1963 with pure IDF) because tweet content embeddings are semantically close to "twitter" tag. Semantic scoring favors generic category tags over specific ones — opposite of what we want. Pure IDF remains better for this use case. |
| Step 4: Doc Duplication | 2026-02-25 | KEPT | Secondary placements in tags with ≤5 docs (very specific). 9,607 placements from 7,272 docs (1.3 avg). 1,478 docs multi-placed (621 in 2 places, 857 in 3). Twitter only +84 docs (4% increase). Earlier attempts with threshold ≤20 and ≤10 amplified twitter too much (2,521 and 3,465). |
| Step 5: Folder Naming | 2026-02-25 | KEPT | Post-hoc rename using content centroid + name quality score. Only depth >= 2 (top-level preserved). 283 subfolders renamed. Eliminated `coca-cola`, `thoracic-diseases`, `navier-stokes` as folder names. First attempt renamed 552 (too aggressive at depth=0) — restricted to depth ≥ 2 and tightened threshold (current_sim < 0.35 AND improvement > 0.10). Some incoherent clusters (greenland-shark) remain because no better name exists for mixed content. |
| Step 7: Coherence Valid. | 2026-02-25 | KEPT (modest) | Outlier detection (>2 stddev from folder centroid) + reassignment if >30% closer to another folder. 22 documents moved in 1 iteration. Conservative threshold keeps this safe. |
| Step 8: Meta-tag Removal | 2026-02-25 | KEPT | Three-part approach: (1) `detect_meta_tags()` uses community-spread (Louvain partition) + semantic-mismatch to find 182 meta-tags. Removes them from community detection graph but keeps for doc assignment. (2) `propagate_tags_by_title()` enriches 398 undertagged docs via same-title groups. (3) `dissolve_grab_bag_folders()` dissolves 15 incoherent top-level folders post-construction — promotes large sub-topics to top level, merges small ones. Blocks meta-tag sub-folders from promotion. Reduced meta-folder docs from 53.6% to 1.5%. Failed approaches: Gini coefficient (all weights=1), embedding coherence (overlapping scores), meta-word overlap (too many false positives). |

## Final State vs Baseline

| Metric | Baseline | Step 1-7 | Step 8 (meta-tag removal) | Change |
|--------|----------|----------|---------------------------|--------|
| Top-level folders | 34 | 28 | 33 | -1 (topic-based) |
| Total folders | 530 | 594 | 576 | +46 |
| Total docs (unique) | 7,230 | 7,272 | 7,272 | +42 |
| Total placements | 7,230 | 9,607 | 9,610 | +2,380 |
| Avg placements/doc | 1.0 | 1.3 | 1.3 | +0.3 |
| Docs in meta folders | ~4,000 (53.6%) | 2,141 (22%) | 137 (1.5%) | -97% |
| Meta top-level folders | twitter, python, typescript, arxiv-doc, github-actions, open-source, etc. | twitter (2141) | parser-library (137, borderline) | eliminated |
| Bad folder names | many (coca-cola, thoracic-diseases, navier-stokes) | greenland-shark only | minor (phares-des-baleines, ikuya-yamada) | user-specific content |

### What worked
- **Tag quality filter** (Step 1): highest impact, removing garbage stopwords/numeric tags cleaned the graph fundamentally
- **Word-subset dedup** (Step 2): merged 120 redundant tag variants, consolidating scattered content
- **Folder renaming** (Step 5): eliminated obviously misleading folder names using content centroids
- **Document duplication** (Step 4): improved discoverability with controlled duplication into rare tags only
- **Meta-tag detection** (Step 8): two-pass approach (community concentration + semantic mismatch) removed 182 meta-tags from graph construction. Reduced meta-folder docs from 53.6% to 1.5%.
- **Title-based tag propagation** (Step 8): 398 undertagged documents enriched by propagating tags from same-title groups (e.g., tweets from same author)
- **Grab-bag folder dissolution** (Step 8): dissolved 15 incoherent top-level folders. Large sub-topics promoted to top level, small ones merged into nearest keeper. Meta-tag sub-folders blocked from promotion.

### What didn't work
- **Semantic document assignment** (Step 3): semantic similarity between doc content and tag names favors generic tags over specific ones — counterproductive
- **Hub detection via clustering coefficient**: removed legitimate high-value tags (python, ML)
- **Hardcoded source-metadata patterns**: violated the no-hardcoding constraint
- **Embedding-based neighbor coherence for meta-tag detection**: coherence scores overlapped heavily between meta-tags and topic tags (both ~0.04-0.06). Abandoned in favor of community-spread approach.
- **Gini coefficient for meta-tag detection**: all edge weights = 1, so Gini = 0 for everything. Completely failed.
- **Meta-word overlap dissolution**: words like "data", "learning", "language" appear in both meta-tags AND topic tags, causing massive false positives (dissolved data-science, reinforcement-learning, etc.). Replaced with embedding similarity (>0.82) to meta-tags.

### Remaining weaknesses
- Some promoted sub-folders have person names (ikuya-yamada, nils-reimers) or conference names (thewebconf-2021) — these are user-specific content, not algorithmic failures
- Sub-folder coherence is still mixed within some topic clusters (chrome-extension contains neovim-plugin, etc.) — inherited from underlying Louvain communities
- ~275 tweets with ONLY twitter tags and no summaries remain unclassifiable
- The tree structure is fundamentally limited by tag quality: garbage-in, garbage-out
