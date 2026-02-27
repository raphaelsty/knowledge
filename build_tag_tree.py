#!/usr/bin/env python3
"""
Build a beautiful hierarchical folder tree from tag co-occurrence triples.

Input:  JSON array of {"head": "tag-a", "tail": "tag-b"} co-occurrence pairs
Output: A nested folder tree on disk + tag_tree.json + tag_tree.txt

Fully data-driven — no hardcoded categories. Works on any domain.

Algorithm:
  1. Build co-occurrence graph
  2. Embed all tags with model2vec
  3. Top-level: Louvain community detection on co-occurrence graph
     → natural topic clusters that respect graph structure
  4. Merge tiny communities into nearest neighbor (by embedding centroid)
  5. Name folders: best representative tag + optional WordNet hypernym
  6. Sub-cluster within each folder using hybrid distance (semantic + cooccurrence)
  7. Post-process: redistribute orphans, absorb tiny, flatten, sort

Requirements: pip install model2vec scipy numpy nltk networkx python-louvain

Usage:
  python build_tag_tree.py <triples.json> <output_dir> [database.json]
"""

import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

import community as community_louvain
import networkx as nx
import nltk
import numpy as np
from model2vec import StaticModel
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as nltk_stopwords
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)


# ─── Config ───


class Config:
    MODEL_NAME = "minishlab/potion-base-8M"
    MIN_TAG_DEGREE = 2
    MAX_DEPTH = 4
    MIN_FOLDER_SIZE = 5
    TARGET_CHILDREN = (5, 16)
    # Sub-clustering weights (embeddings are useful within a community)
    SEMANTIC_WEIGHT = 0.50
    COOCCURRENCE_WEIGHT = 0.50
    # Louvain community detection
    LOUVAIN_RESOLUTION = 8.0  # higher = more communities
    MAX_TOP_LEVEL = 120
    MIN_TOP_CLUSTER = 5
    MAX_TOP_CLUSTER_RATIO = 0.03  # max fraction of total tags in one cluster
    # WordNet
    USE_WORDNET = True  # set False to skip WordNet hypernym naming
    # Documents
    MIN_DOCS_FOR_FOLDER = 3  # subtrees with fewer docs get collapsed into parent

    @classmethod
    def auto_tune(cls, n_tags, n_docs):
        """Return a Config instance with parameters scaled to the dataset size.
        Default values are tuned for ~3500 tags / ~7000 docs. This method
        scales them proportionally for smaller or larger datasets."""
        import math
        cfg = cls()
        # Reference point: 3500 tags
        ref_tags = 3500
        ratio = n_tags / ref_tags if ref_tags else 1.0
        cfg.LOUVAIN_RESOLUTION = 8.0 * math.sqrt(ratio)
        cfg.MIN_TAG_DEGREE = 2  # universal
        cfg.MIN_FOLDER_SIZE = max(3, n_tags // 500)
        cfg.MAX_TOP_CLUSTER_RATIO = min(0.10, 0.03 / max(ratio, 0.1))
        cfg.MIN_TOP_CLUSTER = max(3, n_tags // 400)
        return cfg


# ─── Data loading ───


def load_triples(path):
    with open(path) as f:
        return json.load(f)


def build_graph(triples):
    adj = defaultdict(Counter)
    for t in triples:
        h, tl = t["head"], t["tail"]
        adj[h][tl] += 1
        adj[tl][h] += 1
    return adj


def get_all_tags(adj, min_degree=2):
    return sorted(t for t, nbrs in adj.items() if len(nbrs) >= min_degree)


# ─── Tag quality filter ───


def _detect_stopword_languages(tag_set, min_overlap=0.05):
    """Auto-detect which NLTK stopword languages have significant overlap with the tag set.
    Returns list of language names. Always includes English, adds others if >=min_overlap
    fraction of their stopwords appear in the tags."""
    languages = ["english"]  # always include English
    try:
        available = nltk_stopwords.fileids()
    except LookupError:
        return languages

    tag_lower = {t.lower() for t in tag_set}
    for lang in available:
        if lang == "english":
            continue
        try:
            sw = set(nltk_stopwords.words(lang))
            overlap = len(sw & tag_lower) / max(len(sw), 1)
            if overlap >= min_overlap:
                languages.append(lang)
        except LookupError:
            continue
    return languages


def _build_garbage_tag_set(tag_set=None):
    """Build a set of tags that should be filtered from graph construction.
    Uses NLTK stopwords with auto-detected languages. If tag_set is provided,
    detects which languages have significant stopword overlap with the tags.
    No hardcoded tag lists — fully automated."""
    garbage = set()

    if tag_set:
        languages = _detect_stopword_languages(tag_set)
    else:
        languages = ["english"]

    for lang in languages:
        try:
            garbage.update(nltk_stopwords.words(lang))
        except LookupError:
            pass

    return garbage


def is_garbage_tag(tag, stopwords_set):
    """Determine if a tag is garbage (non-informative) based on generic rules.
    Returns True if the tag should be filtered from graph construction."""
    # Single-char or empty
    if len(tag) <= 1:
        return True
    # Pure numeric (years, version fragments like "2.0")
    if re.match(r"^[\d.]+$", tag):
        return True
    # Bare stopwords (exact match, no hyphens) — catches 2-char stopwords like
    # "is", "an", "or", "no", "us" while keeping meaningful tags like "ml", "go", "cv"
    if "-" not in tag and tag.lower() in stopwords_set:
        return True
    return False


def split_compound_tags(triples):
    """Split semicolon/comma-separated compound tags into individual tags.
    e.g. 'snowflake;python;snowpark' → 3 separate tags, creating edges between them."""
    new_triples = []
    for t in triples:
        head_parts = re.split(r"[;,]", t["head"])
        tail_parts = re.split(r"[;,]", t["tail"])
        head_parts = [p.strip() for p in head_parts if p.strip()]
        tail_parts = [p.strip() for p in tail_parts if p.strip()]
        # Create edges between all head parts × all tail parts
        for h in head_parts:
            for tl in tail_parts:
                if h != tl:
                    new_triples.append({"head": h, "tail": tl})
        # Also create edges within compound tags (parts are related)
        for i in range(len(head_parts)):
            for j in range(i + 1, len(head_parts)):
                new_triples.append({"head": head_parts[i], "tail": head_parts[j]})
        for i in range(len(tail_parts)):
            for j in range(i + 1, len(tail_parts)):
                new_triples.append({"head": tail_parts[i], "tail": tail_parts[j]})
    return new_triples


def detect_meta_tags(adj, tags, tag_emb, tag_idx, verbose=True):
    """Detect meta-tags using two complementary signals.

    Data-driven, two-pass approach:
    Pass 1 — Community spread: Run Louvain, then measure how each high-degree
    tag's neighbors spread across communities. Meta-tags like "python" and
    "twitter" have neighbors in many communities (low concentration).

    Pass 2 — Semantic mismatch: For high-degree tags NOT caught by Pass 1,
    check if the tag embedding is semantically distant from its neighbors'
    centroid. Source/format tags like "arxiv-doc" may have concentrated
    neighbors (all ML) but the tag name is unrelated to the topic.
    """
    if not tags or tag_emb is None:
        return set()

    tag_set = set(tags)

    # Build networkx graph for Louvain (sorted for determinism)
    G = nx.Graph()
    sorted_tags = sorted(tags)
    for t in sorted_tags:
        G.add_node(t)
    for t in sorted_tags:
        for nbr in sorted(adj.get(t, {}).keys()):
            if nbr in tag_set and nbr != t:
                w = adj[t][nbr]
                if G.has_edge(t, nbr):
                    G[t][nbr]["weight"] += w
                else:
                    G.add_edge(t, nbr, weight=w)

    if G.number_of_edges() == 0:
        return set()

    # Run Louvain to get natural communities
    partition = community_louvain.best_partition(G, resolution=1.5, random_state=42)

    # For each tag, compute community concentration of its neighbors
    degrees = {t: len(adj.get(t, {})) for t in tags}
    degree_vals = sorted(degrees.values(), reverse=True)
    # Only analyze top 10% by degree — low-degree tags can't form big grab-bag folders
    degree_p90 = degree_vals[max(0, min(int(len(degree_vals) * 0.10), len(degree_vals) - 1))]
    high_degree_tags = [t for t in tags if degrees[t] >= degree_p90 and t in tag_idx]

    if not high_degree_tags:
        return set()

    tag_concentration = {}
    tag_n_communities = {}
    tag_neighbor_sim = {}
    for tag in high_degree_tags:
        neighbors = [n for n in adj.get(tag, {}).keys() if n in partition and n in tag_idx]
        if len(neighbors) < 5:
            continue

        # --- Pass 1 metric: community concentration ---
        comm_counts = Counter(partition[n] for n in neighbors)
        total = sum(comm_counts.values())
        max_count = max(comm_counts.values())
        concentration = max_count / total
        n_comms = len(comm_counts)

        tag_concentration[tag] = concentration
        tag_n_communities[tag] = n_comms

        # --- Pass 2 metric: tag-to-neighbor semantic similarity ---
        sample = neighbors
        if len(sample) > 50:
            rng = np.random.RandomState(42)
            sample = list(rng.choice(neighbors, 50, replace=False))
        tag_vec = tag_emb[tag_idx[tag]]
        nbr_vecs = tag_emb[[tag_idx[n] for n in sample]]
        avg_sim = float((nbr_vecs @ tag_vec).mean())
        tag_neighbor_sim[tag] = avg_sim

    if not tag_concentration:
        return set()

    # Pass 1: low community concentration → meta-tag
    conc_vals = sorted(tag_concentration.values())
    conc_p25 = conc_vals[len(conc_vals) // 4]

    meta_tags = set()
    for tag in tag_concentration:
        if tag_concentration[tag] <= conc_p25:
            meta_tags.add(tag)

    # Pass 2: for remaining high-degree tags, check semantic mismatch
    # Tags whose name is semantically unrelated to their neighbors are source/format tags
    sim_vals = sorted(tag_neighbor_sim.values())
    sim_p15 = sim_vals[max(0, int(len(sim_vals) * 0.15))]

    pass2_added = set()
    for tag in tag_concentration:
        if tag in meta_tags:
            continue
        if tag_neighbor_sim.get(tag, 1.0) <= sim_p15:
            meta_tags.add(tag)
            pass2_added.add(tag)

    if verbose:
        print(f"  High-degree tags analyzed: {len(tag_concentration)}")
        print(f"  Pass 1 — concentration threshold (p25): {conc_p25:.3f}")
        print(f"  Pass 1 meta-tags: {len(meta_tags) - len(pass2_added)}")
        print(f"  Pass 2 — semantic sim threshold (p15): {sim_p15:.3f}")
        print(f"  Pass 2 additional meta-tags: {len(pass2_added)}")
        print(f"  Total meta-tags: {len(meta_tags)}")
        if meta_tags:
            by_degree = sorted(meta_tags, key=lambda t: -degrees[t])
            for t in by_degree[:30]:
                src = "P2" if t in pass2_added else "P1"
                print(
                    f"    [{src}] {t}: degree={degrees[t]}, "
                    f"conc={tag_concentration.get(t, 0):.3f}, "
                    f"sim={tag_neighbor_sim.get(t, 0):.3f}, "
                    f"comms={tag_n_communities.get(t, 0)}"
                )

    return meta_tags


def filter_garbage_tags(triples, verbose=True):
    """Remove garbage tags from triples. Returns (cleaned_triples, garbage_tags).
    Fully data-driven: uses NLTK stopwords for basic filtering.
    Meta-tag detection happens later (after embeddings are loaded)."""
    # Collect all tags first so we can auto-detect languages
    all_tags = set()
    for t in triples:
        all_tags.add(t["head"])
        all_tags.add(t["tail"])

    stopwords_set = _build_garbage_tag_set(tag_set=all_tags)

    garbage_tags = set()
    for tag in all_tags:
        if is_garbage_tag(tag, stopwords_set):
            garbage_tags.add(tag)

    # Filter triples
    cleaned = [
        t for t in triples
        if t["head"] not in garbage_tags and t["tail"] not in garbage_tags
    ]

    if verbose:
        print(f"  Tags before filter: {len(all_tags)}")
        print(f"  Garbage tags removed: {len(garbage_tags)}")
        if garbage_tags:
            sample = sorted(garbage_tags)[:20]
            print(f"    Sample: {', '.join(sample)}")
        print(f"  Triples: {len(triples)} → {len(cleaned)}")

    return cleaned, garbage_tags


# ─── Tag deduplication ───


def deduplicate_triples(triples, verbose=True):
    """
    Merge near-duplicate tags: separator variants, plurals, typos.
    Fully automated pre-processing — no hardcoded word lists.

    Pass 1: Normalize separators (hyphens/spaces/underscores) + depluralization
    Pass 2: Levenshtein edit distance for typos/misspellings (blocked by prefix+suffix)

    Returns (cleaned_triples, merge_map).
    """
    # Count how often each tag appears across all triples
    tag_freq = Counter()
    for t in triples:
        tag_freq[t["head"]] += 1
        tag_freq[t["tail"]] += 1
    all_tags = sorted(tag_freq.keys())
    print(f"  {len(all_tags)} unique tags")

    # --- Normalization helpers ---

    def _depluralize(n):
        """Conservative English depluralization."""
        if len(n) <= 4:
            return n
        if n.endswith("ies") and len(n) > 5:
            return n[:-3] + "y"  # libraries → library
        if n.endswith("ss") or n.endswith("us") or n.endswith("is"):
            return n  # loss, status, analysis — keep
        if n.endswith("s"):
            return n[:-1]  # models → model
        return n

    def _normalize(tag):
        """Canonical form: lowercase, strip separators, depluralize."""
        n = re.sub(r"[-_\s]+", "", tag.lower())
        return _depluralize(n)

    # --- Union-Find ---

    parent = {t: t for t in all_tags}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        # Most frequent tag becomes canonical
        fa, fb = tag_freq[ra], tag_freq[rb]
        if fa < fb:
            parent[ra] = rb
        elif fa == fb:
            # Tiebreak: prefer hyphenated form (more readable), then longer
            if ("-" in rb and "-" not in ra) or (len(rb) > len(ra)):
                parent[ra] = rb
            else:
                parent[rb] = ra
        else:
            parent[rb] = ra
        return True

    # --- Pass 1: Normalization (separators + plurals) ---

    print("  Pass 1: Normalization (separators + plurals)...")
    norm_groups = defaultdict(list)
    for tag in all_tags:
        norm_groups[_normalize(tag)].append(tag)

    pass1_merges = 0
    for norm, group in norm_groups.items():
        if len(group) > 1:
            for tag in group[1:]:
                if union(group[0], tag):
                    pass1_merges += 1

    print(f"    {pass1_merges} merges")

    # --- Pass 2: Levenshtein edit distance (typos) ---
    # Safety: require minimum length + co-occurrence neighborhood overlap
    # to avoid merging genuinely different entities (python≠cython, llama≠llava)

    print("  Pass 2: Edit distance (typos)...")

    # Build preliminary co-occurrence graph for neighbor overlap checks
    raw_adj = build_graph(triples)

    # For each canonical tag, gather neighbors from all its merged variants
    pass1_groups = defaultdict(list)
    for tag in all_tags:
        pass1_groups[find(tag)].append(tag)

    canon_neighbors = {}
    for canonical, variants in pass1_groups.items():
        nbrs = set()
        for v in variants:
            nbrs.update(raw_adj.get(v, {}).keys())
        # Resolve neighbors to their canonical forms
        nbrs_canonical = set()
        for n in nbrs:
            nc = find(n)
            if nc != canonical:
                nbrs_canonical.add(nc)
        canon_neighbors[canonical] = nbrs_canonical

    def _overlap_coeff(t1, t2):
        """Overlap coefficient: fraction of the SMALLER set that overlaps.
        Unlike Jaccard, this isn't penalized when one tag is much more popular.
        A typo shares nearly all its neighbors with the correct spelling."""
        n1 = canon_neighbors.get(t1, set())
        n2 = canon_neighbors.get(t2, set())
        smaller = min(len(n1), len(n2))
        if smaller == 0:
            return 0.0
        return len(n1 & n2) / smaller

    # Get current canonical representatives after pass 1
    canonicals = sorted(set(find(t) for t in all_tags))
    canon_norms = {t: _normalize(t) for t in canonicals}

    MIN_NORM_LEN = 7  # skip very short tags where 1-char edits create different words

    def _lev(s1, s2, max_d):
        """Levenshtein distance with early termination."""
        len1, len2 = len(s1), len(s2)
        if abs(len1 - len2) > max_d:
            return max_d + 1
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        prev = list(range(len2 + 1))
        for i in range(len1):
            curr = [i + 1]
            mn = curr[0]
            for j in range(len2):
                cost = 0 if s1[i] == s2[j] else 1
                val = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
                curr.append(val)
                if val < mn:
                    mn = val
            if mn > max_d:
                return max_d + 1
            prev = curr
        return prev[len2]

    # Block by prefix (first 3 chars) and suffix (last 3 chars) for efficiency.
    # Two blocking schemes catch typos regardless of position.
    prefix_blocks = defaultdict(list)
    suffix_blocks = defaultdict(list)
    for t in canonicals:
        n = canon_norms[t]
        if len(n) >= MIN_NORM_LEN:
            prefix_blocks[n[:3]].append(t)
            suffix_blocks[n[-3:]].append(t)

    pass2_merges = 0
    pass2_rejected = 0
    checked = set()

    def _check_block(block_tags):
        nonlocal pass2_merges, pass2_rejected
        for i in range(len(block_tags)):
            t1 = block_tags[i]
            n1 = canon_norms[t1]
            for j in range(i + 1, len(block_tags)):
                t2 = block_tags[j]
                pair = (min(t1, t2), max(t1, t2))
                if pair in checked:
                    continue
                checked.add(pair)
                n2 = canon_norms[t2]
                # Two-tier edit distance:
                #   ed ≤ 1 for all lengths (safe: catches most typos)
                #   ed ≤ 2 only for very long tags (>15 chars) where
                #   double-typos are more plausible (artifiical, knwoledge)
                min_len = min(len(n1), len(n2))
                max_d = 1 if min_len <= 15 else 2
                if abs(len(n1) - len(n2)) > max_d:
                    continue
                d = _lev(n1, n2, max_d)
                if d <= max_d:
                    # Safety check 1: word-part count must match.
                    # "deep-learning" (2 parts) ≠ "deep-q-learning" (3 parts)
                    # "multimodal" (1 part) ≠ "multi-model" (2 parts)
                    # But "transformer" (1) == "transfomer" (1) → OK
                    p1 = re.split(r"[-_\s]+", t1.lower())
                    p2 = re.split(r"[-_\s]+", t2.lower())
                    if len(p1) != len(p2):
                        pass2_rejected += 1
                        continue
                    # Safety check 2: co-occurrence neighborhood overlap.
                    # Typos share context with the correct spelling.
                    # Stricter threshold for ed=2 (more likely to be false positive)
                    oc = _overlap_coeff(find(t1), find(t2))
                    min_oc = 0.18 if d == 1 else 0.50
                    if oc < min_oc:
                        pass2_rejected += 1
                        continue
                    if union(t1, t2):
                        pass2_merges += 1

    for block_tags in prefix_blocks.values():
        _check_block(block_tags)
    for block_tags in suffix_blocks.values():
        _check_block(block_tags)

    print(f"    {pass2_merges} merges ({pass2_rejected} rejected by co-occurrence check)")

    # --- Pass 3: Word-subset dedup ---
    # If tag A's words are a strict subset of tag B's words AND they have
    # high co-occurrence overlap, merge B into A (the shorter, broader tag).
    # e.g. "distillation" ← "knowledge-distillation", "scene-text" ← "scene-text-recognition"

    print("  Pass 3: Word-subset dedup...")
    # Refresh canonicals after pass 2
    canonicals = sorted(set(find(t) for t in all_tags))

    # Re-collect neighbors for current canonical groups
    pass2_groups = defaultdict(list)
    for tag in all_tags:
        pass2_groups[find(tag)].append(tag)
    for canonical, variants in pass2_groups.items():
        nbrs = set()
        for v in variants:
            nbrs.update(raw_adj.get(v, {}).keys())
        nbrs_canonical = set()
        for n in nbrs:
            nc = find(n)
            if nc != canonical:
                nbrs_canonical.add(nc)
        canon_neighbors[canonical] = nbrs_canonical

    word_sets = {}
    for t in canonicals:
        word_sets[t] = set(re.split(r"[-_\s]+", t.lower()))

    pass3_merges = 0
    pass3_rejected = 0
    # Don't merge high-frequency tags — they are distinct concepts even if
    # one's words are a subset of the other (e.g. "deep-learning" ≠ "deep-reinforcement-learning")
    freq_threshold = sorted(tag_freq.values(), reverse=True)[min(100, len(tag_freq) - 1)]
    for t1 in canonicals:
        w1 = word_sets[t1]
        # Only merge compound tags (2+ words) — single-word subset matching
        # is too aggressive (e.g. "python" ⊂ "python-tutorial" chains endlessly)
        if len(w1) < 2 or len(w1) > 4:
            continue
        for t2 in canonicals:
            if t1 == t2 or find(t1) == find(t2):
                continue
            w2 = word_sets[t2]
            # t1's words must be a strict subset of t2's words
            if not (w1 < w2):
                continue
            # Extra words should be exactly 1 (tight match)
            if len(w2) - len(w1) != 1:
                continue
            # Don't merge if either tag is high-frequency (distinct concept)
            if tag_freq.get(find(t1), 0) >= freq_threshold or tag_freq.get(find(t2), 0) >= freq_threshold:
                pass3_rejected += 1
                continue
            # Require high co-occurrence overlap
            oc = _overlap_coeff(find(t1), find(t2))
            if oc >= 0.50:
                if union(t1, t2):
                    pass3_merges += 1
            else:
                pass3_rejected += 1

    print(f"    {pass3_merges} merges")

    # --- Build final merge map ---

    merge_map = {}
    merge_groups = defaultdict(list)
    for tag in all_tags:
        root = find(tag)
        merge_groups[root].append(tag)
        if root != tag:
            merge_map[tag] = root

    multi_groups = {k: sorted(v, key=lambda t: -tag_freq[t]) for k, v in merge_groups.items() if len(v) > 1}

    n_affected = sum(len(v) for v in multi_groups.values())
    print(f"  Result: {n_affected} tags in {len(multi_groups)} groups → {len(multi_groups)} canonical tags")

    if verbose:
        examples = sorted(multi_groups.items(), key=lambda kv: -sum(tag_freq[t] for t in kv[1]))
        for canonical, group in examples[:30]:
            others = [t for t in group if t != canonical]
            if others:
                parts = [f"{canonical}({tag_freq[canonical]})"]
                parts.extend(f"{t}({tag_freq[t]})" for t in others[:5])
                if len(others) > 5:
                    parts.append(f"... +{len(others) - 5} more")
                print(f"    {' ← '.join(parts)}")

    # --- Normalize spaces to hyphens in all tags ---

    def _hyphenate(tag):
        return re.sub(r"\s+", "-", tag.strip())

    # Update merge map: every tag maps to a hyphenated canonical
    for tag in all_tags:
        canonical = merge_map.get(tag, tag)
        hyphenated = _hyphenate(canonical)
        merge_map[tag] = hyphenated

    # --- Apply merge map to triples ---

    cleaned = []
    removed = 0
    for t in triples:
        h = merge_map.get(t["head"], _hyphenate(t["head"]))
        tl = merge_map.get(t["tail"], _hyphenate(t["tail"]))
        if h != tl:
            cleaned.append({"head": h, "tail": tl})
        else:
            removed += 1

    print(f"  Triples: {len(triples)} → {len(cleaned)} ({removed} self-loops removed)")

    return cleaned, merge_map


# ─── Document loading & assignment ───


def load_database(path):
    with open(path) as f:
        return json.load(f)


def triples_from_documents(database, garbage_tags=None):
    """
    Generate co-occurrence triples from document tags.
    Each pair of tags on the same document creates an edge,
    so document-tag relationships directly influence the tree structure.
    """
    if garbage_tags is None:
        garbage_tags = set()
    triples = []
    for url, doc in database.items():
        all_tags = list(doc.get("tags", [])) + list(doc.get("extra-tags", []))
        # Normalize: lowercase, spaces→hyphens, split compounds
        raw = set()
        for t in all_tags:
            t = t.strip()
            if not t:
                continue
            # Split compound semicolon/comma tags
            parts = re.split(r"[;,]", t)
            for p in parts:
                p = re.sub(r"\s+", "-", p.strip().lower())
                if p and p not in garbage_tags:
                    raw.add(p)
        tags = sorted(raw)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                triples.append({"head": tags[i], "tail": tags[j]})
    return triples


def propagate_tags_by_title(database, tree_tags, merge_map, garbage_tags, top_k=3):
    """Propagate tags from tagged docs to untagged docs that share the same title.

    Data-driven: if documents share the same title (e.g., tweets from the same author),
    the tagged ones provide topic signals for the untagged ones. The most common tags
    from the tagged group are assigned to untagged members.
    """
    tree_tag_set = set(tree_tags)

    def _get_useful_tags(doc):
        all_tags = list(doc.get("tags", [])) + list(doc.get("extra-tags", []))
        useful = set()
        for t in all_tags:
            t_clean = re.sub(r"\s+", "-", t.strip().lower())
            for part in re.split(r"[;,]", t_clean):
                part = part.strip()
                if part and part not in garbage_tags:
                    merged = merge_map.get(part, part)
                    if merged in tree_tag_set:
                        useful.add(merged)
        return useful

    # Group docs by exact title
    title_groups = defaultdict(list)
    for url, doc in database.items():
        title = doc.get("title", "")
        if title:
            title_groups[title].append(url)

    enriched = 0
    for title, urls in title_groups.items():
        if len(urls) < 2:
            continue

        # Collect tags from all tagged docs in this group
        group_tags = Counter()
        untagged_urls = []
        for url in urls:
            tags = _get_useful_tags(database[url])
            if tags:
                for t in tags:
                    group_tags[t] += 1
            else:
                untagged_urls.append(url)

        if not group_tags or not untagged_urls:
            continue

        # Pick the top-k most common tags (must appear in >1 doc or be >25% of group)
        min_count = max(2, len(urls) // 4)
        propagated = [t for t, c in group_tags.most_common(top_k) if c >= min_count]
        if not propagated:
            # Fallback: just use the top tag if it appears in >1 doc
            propagated = [t for t, c in group_tags.most_common(1) if c >= 2]
        if not propagated:
            continue

        for url in untagged_urls:
            existing = list(database[url].get("extra-tags", []))
            database[url]["extra-tags"] = existing + propagated
            enriched += 1

    return enriched


def enrich_undertagged_docs(database, tree_tags, merge_map, garbage_tags, model, top_k=3):
    """For documents with no useful tags after filtering, assign synthetic tags
    by embedding the document title+summary and finding the nearest tree tags.

    This is fully data-driven — works on any dataset. It prevents large clusters
    of undifferentiated documents from forming around generic source-metadata tags.
    """
    tree_tag_list = sorted(tree_tags)
    if not tree_tag_list:
        return 0

    # Embed all tree tags
    tag_texts = [re.sub(r"[-_]+", " ", t) for t in tree_tag_list]
    tag_embs = model.encode(tag_texts)
    norms = np.linalg.norm(tag_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tag_embs = tag_embs / norms

    enriched = 0
    batch_texts = []
    batch_urls = []

    for url, doc in database.items():
        all_tags = list(doc.get("tags", [])) + list(doc.get("extra-tags", []))
        # Normalize and check if any useful tags remain
        useful = set()
        for t in all_tags:
            t_clean = re.sub(r"\s+", "-", t.strip().lower())
            for part in re.split(r"[;,]", t_clean):
                part = part.strip()
                if part and part not in garbage_tags:
                    merged = merge_map.get(part, part)
                    if merged in tree_tags:
                        useful.add(merged)
        if useful:
            continue  # Already has meaningful tags

        # This doc has no useful tags — build a text representation
        title = doc.get("title", "")
        summary = doc.get("summary", "")[:300]
        text = f"{title}. {summary}".strip()
        if len(text) < 10:
            continue

        batch_texts.append(text)
        batch_urls.append(url)

    if not batch_texts:
        return 0

    # Embed all undertagged doc texts
    doc_embs = model.encode(batch_texts)
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    doc_embs = doc_embs / norms

    # Find nearest tree tags for each doc
    sims = doc_embs @ tag_embs.T  # (n_docs, n_tags)

    for i, url in enumerate(batch_urls):
        top_indices = np.argsort(sims[i])[-top_k:][::-1]
        synthetic_tags = [tree_tag_list[j] for j in top_indices if sims[i][j] > 0.15]
        if synthetic_tags:
            # Add synthetic tags to the document's extra-tags
            existing = list(database[url].get("extra-tags", []))
            database[url]["extra-tags"] = existing + synthetic_tags
            enriched += 1

    return enriched


def assign_documents(database, merge_map, tree_tags):
    """
    Assign each document to ONE primary tag using inverse document frequency.
    Returns (tag_to_docs, assigned_docs).
    """
    tree_tag_set = set(tree_tags)

    def _hyphenate(tag):
        return re.sub(r"\s+", "-", tag.strip().lower())

    # Build tag_to_docs: normalized tag → list of doc urls
    tag_to_docs = defaultdict(list)
    doc_tags_map = {}  # url → set of normalized tags that exist in tree

    for url, doc in database.items():
        all_tags = list(doc.get("tags", [])) + list(doc.get("extra-tags", []))
        if not all_tags:
            continue

        normalized = set()
        for t in all_tags:
            h = _hyphenate(t)
            merged = merge_map.get(t, merge_map.get(h, h))
            normalized.add(merged)

        # Filter to tags that exist in the tree
        matching = normalized & tree_tag_set
        if not matching:
            continue

        doc_tags_map[url] = matching
        for tag in matching:
            tag_to_docs[tag].append(url)

    # Assign each doc: primary tag (rarest) + secondary placements for discoverability
    # A doc can appear in up to MAX_PLACEMENTS locations if it has multiple relevant rare tags
    MAX_PLACEMENTS = 3

    assigned_docs = defaultdict(list)
    assigned_count = 0
    total_placements = 0

    for url, matching_tags in doc_tags_map.items():
        doc = database[url]
        all_doc_tags = list(doc.get("tags", [])) + list(doc.get("extra-tags", []))
        doc_entry = {
            "url": url,
            "title": doc.get("title", ""),
            "date": doc.get("date", ""),
            "tags": [t for t in all_doc_tags if t.strip()],
            "summary": doc.get("summary", "")[:500],
        }

        # Score by inverse document frequency: prefer rare tags
        sorted_tags = sorted(matching_tags, key=lambda t: len(tag_to_docs[t]))
        best_count = len(tag_to_docs[sorted_tags[0]])

        # Primary placement: always the rarest tag
        assigned_docs[sorted_tags[0]].append(doc_entry)
        placed = {sorted_tags[0]}

        # Secondary placements: other rare tags with similar specificity
        # Only place in tags that are rare enough to be meaningful (not hubs)
        # and similar in specificity to the primary tag
        # Only duplicate into tags that are very specific (few docs)
        # This avoids amplifying large clusters while improving discoverability
        # in niche areas where a doc is genuinely relevant
        SECONDARY_MAX_DOCS = 5
        for tag in sorted_tags[1:]:
            if len(placed) >= MAX_PLACEMENTS:
                break
            tag_count = len(tag_to_docs[tag])
            if tag_count <= SECONDARY_MAX_DOCS:
                assigned_docs[tag].append(doc_entry)
                placed.add(tag)

        assigned_count += 1
        total_placements += len(placed)

    avg_placements = total_placements / max(assigned_count, 1)
    print(f"  Documents: {len(database)} total, {assigned_count} assigned to {len(assigned_docs)} tags")
    print(f"  Placements: {total_placements} total ({avg_placements:.1f} avg per doc)")
    print(f"  Skipped: {len(database) - assigned_count} (no matching tree tags)")

    # Sort docs within each tag by date (newest first)
    for tag in assigned_docs:
        assigned_docs[tag].sort(key=lambda d: d.get("date", ""), reverse=True)

    return dict(tag_to_docs), dict(assigned_docs)


# ─── Attach documents to tree ───


def attach_documents(tree, assigned_docs):
    """
    Walk tree recursively, converting leaf strings to objects with documents
    and adding documents to folder nodes.
    """
    # Convert leaves from strings to objects with documents
    new_leaves = []
    for leaf in tree.get("leaves", []):
        tag = leaf if isinstance(leaf, str) else leaf["tag"]
        docs = assigned_docs.get(tag, [])
        new_leaves.append({"tag": tag, "documents": docs})
    tree["leaves"] = new_leaves

    # Attach documents to folder node itself (by its name tag)
    name = tree.get("name")
    if name:
        tree["documents"] = assigned_docs.get(name, [])

    # Recurse into children
    for ch in tree.get("children", []):
        attach_documents(ch, assigned_docs)

    return tree


def _count_docs(tree):
    """Count total documents in a tree node and all descendants."""
    n = len(tree.get("documents", []))
    for leaf in tree.get("leaves", []):
        if isinstance(leaf, dict):
            n += len(leaf.get("documents", []))
    for ch in tree.get("children", []):
        n += _count_docs(ch)
    return n


def _collect_leaves_deep(tree):
    """Collect all leaves (with docs) from a subtree, flattening it."""
    leaves = list(tree.get("leaves", []))
    # The folder-name tag itself becomes a leaf
    name = tree.get("name")
    if name:
        leaves.append({"tag": name, "documents": tree.get("documents", [])})
    for ch in tree.get("children", []):
        leaves.extend(_collect_leaves_deep(ch))
    return leaves


def collapse_thin_subtrees(tree, min_docs):
    """
    Collapse child subtrees that have fewer than min_docs total documents.
    Their leaves get absorbed into the parent node.
    """
    # Recurse first so we collapse bottom-up
    for ch in tree.get("children", []):
        collapse_thin_subtrees(ch, min_docs)

    new_children = []
    absorbed_leaves = []
    for ch in tree.get("children", []):
        if _count_docs(ch) < min_docs:
            # Absorb this subtree's leaves into parent
            absorbed_leaves.extend(_collect_leaves_deep(ch))
        else:
            new_children.append(ch)

    tree["children"] = new_children
    if absorbed_leaves:
        tree["leaves"] = tree.get("leaves", []) + absorbed_leaves

    return tree


# ─── Embedding ───


def load_model(model_name):
    print(f"  Loading model {model_name}...")
    return StaticModel.from_pretrained(model_name)


def embed(model, texts):
    e = model.encode(texts)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1
    return e / n


def embed_tags(model, tags):
    clean = [re.sub(r"[-_]+", " ", t) for t in tags]
    print(f"  Encoding {len(tags)} tags...")
    return embed(model, clean)


# ─── Top-level: Louvain community detection ───


def build_nx_graph(adj, tag_set):
    """Build a networkx graph from the adjacency dict, restricted to tag_set."""
    G = nx.Graph()
    # Sort nodes and edges for deterministic graph construction
    sorted_tags = sorted(tag_set)
    for t in sorted_tags:
        G.add_node(t)
    for t in sorted_tags:
        for nbr in sorted(adj.get(t, {}).keys()):
            if nbr in tag_set and nbr > t:  # avoid duplicate edges
                G.add_edge(t, nbr, weight=adj[t][nbr])
    return G


def detect_communities(G, cfg, n_tags):
    """
    Use Louvain to detect communities. Auto-increase resolution if
    the largest community exceeds the max size ratio.
    """
    resolution = cfg.LOUVAIN_RESOLUTION
    max_size = int(n_tags * cfg.MAX_TOP_CLUSTER_RATIO)

    for attempt in range(8):
        partition = community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=42)
        communities = defaultdict(list)
        for node, comm in partition.items():
            communities[comm].append(node)

        sizes = sorted([len(v) for v in communities.values()], reverse=True)
        print(
            f"  Resolution {resolution:.2f}: {len(communities)} communities, biggest={sizes[0]}, smallest={sizes[-1]}"
        )

        if sizes[0] <= max_size:
            break
        resolution *= 1.2

    return communities


def louvain_top_level(adj, tags, emb, model, cfg):
    """
    Detect top-level folders using Louvain on the co-occurrence graph.
    Merge small communities, name each one.
    """
    tag_set = set(tags)
    tag_idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)

    # Build graph and detect communities
    print("  Building networkx graph...")
    G = build_nx_graph(adj, tag_set)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("  Louvain community detection...")
    communities = detect_communities(G, cfg, n)

    # Sort by size
    comm_list = sorted(communities.items(), key=lambda x: -len(x[1]))
    print(f"  Found {len(comm_list)} communities")

    def _centroid(grp):
        idxs = [tag_idx[t] for t in grp if t in tag_idx]
        if not idxs:
            return np.zeros(emb.shape[1])
        return emb[idxs].mean(axis=0)

    # Merge small communities into nearest big neighbor
    big = {}
    small_pool = []
    for comm_id, grp in comm_list:
        if len(grp) >= cfg.MIN_TOP_CLUSTER:
            big[comm_id] = grp
        else:
            small_pool.extend(grp)

    if big and small_pool:
        big_ids = sorted(big.keys())
        big_cents = np.array([_centroid(big[cid]) for cid in big_ids])
        # Assign each orphan tag to nearest big community
        for tag in small_pool:
            if tag not in tag_idx:
                continue
            v = emb[tag_idx[tag]]
            dists = np.linalg.norm(big_cents - v, axis=1)
            nearest = big_ids[int(np.argmin(dists))]
            big[nearest].append(tag)
        print(f"  After merging small: {len(big)} communities ({len(small_pool)} orphans redistributed)")
    elif not big:
        big = dict(comm_list)

    # If still too many, merge closest pairs
    while len(big) > cfg.MAX_TOP_LEVEL:
        ids = sorted(big.keys())
        cents = np.array([_centroid(big[cid]) for cid in ids])
        min_dist, mi, mj = float("inf"), 0, 1
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = float(np.linalg.norm(cents[i] - cents[j]))
                # Prefer merging smaller clusters
                size = len(big[ids[i]]) + len(big[ids[j]])
                score = d + size * 0.0005
                if score < min_dist:
                    min_dist, mi, mj = score, i, j
        big[ids[mi]].extend(big[ids[mj]])
        del big[ids[mj]]

    # Reassignment pass: move tags closer to another community.
    # Uses cosine similarity with a conservative margin to avoid destabilizing.
    print("  Reassignment pass...")
    cluster_tags = {k: list(v) for k, v in big.items()}
    label_list = sorted(cluster_tags.keys())
    moved = 0

    for _round in range(5):
        centroids = np.array([_centroid(cluster_tags[l]) for l in label_list])
        to_move = []

        for li, label in enumerate(label_list):
            for t in list(cluster_tags[label]):
                if t not in tag_idx:
                    continue
                v = emb[tag_idx[t]]
                dists = np.linalg.norm(centroids - v, axis=1)
                own_d = dists[li]
                dists[li] = float("inf")
                best_j = int(np.argmin(dists))
                if dists[best_j] < own_d * 0.80:
                    to_move.append((t, label, label_list[best_j]))

        round_moved = 0
        for t, old_l, new_l in to_move:
            if t in cluster_tags[old_l] and len(cluster_tags[old_l]) > cfg.MIN_TOP_CLUSTER:
                cluster_tags[old_l].remove(t)
                cluster_tags[new_l].append(t)
                round_moved += 1

        moved += round_moved
        if round_moved == 0:
            break

    print(f"  Reassigned {moved} tags over {_round + 1} rounds")

    # Name each community
    print("  Naming communities...")
    result = []
    used_names = set()
    for label in sorted(cluster_tags.keys(), key=lambda l: -len(cluster_tags[l])):
        grp = cluster_tags[label]
        if not grp:
            continue

        tag_name = _get_representative(grp, emb, tag_idx, adj)
        wn_name = find_cluster_wordnet_name(model, grp, adj) if cfg.USE_WORDNET else None

        folder_name = tag_name
        if wn_name and wn_name.lower() not in used_names:
            wn_vec = embed(model, [wn_name])[0]
            tag_vec = embed(model, [tag_name.replace("-", " ")])[0]
            centroid = _centroid(grp)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            wn_sim = float(wn_vec @ centroid)
            tag_sim = float(tag_vec @ centroid)
            if wn_sim > tag_sim * 1.05 and len(wn_name) <= 25:
                folder_name = wn_name

        used_names.add(folder_name.lower())
        result.append((folder_name, grp))
        wn_label = f" (wn: {wn_name})" if wn_name and wn_name != folder_name else ""
        print(f"    📁 {folder_name}{wn_label} ({len(grp)} tags)")

    # Dedup pass 1: exact normalization (spaces/hyphens/underscores, plurals)
    def _normalize_name(name):
        n = re.sub(r"[-_ ]+", " ", name.lower()).strip()
        # Strip trailing 's' for basic plural handling
        if n.endswith("s") and len(n) > 3:
            n = n[:-1]
        return n

    merged = []
    seen_normalized = {}
    for name, grp in result:
        norm = _normalize_name(name)
        if norm in seen_normalized:
            idx = seen_normalized[norm]
            merged[idx] = (merged[idx][0], merged[idx][1] + grp)
        else:
            seen_normalized[norm] = len(merged)
            merged.append((name, grp))

    # Dedup pass 2: embedding similarity (catch semantic near-duplicates)
    if len(merged) > 1:
        names = [m[0] for m in merged]
        name_embs = embed(model, [n.replace("-", " ") for n in names])
        sims = name_embs @ name_embs.T
        to_merge = []
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if sims[i][j] > 0.92:  # very high similarity threshold
                    to_merge.append((i, j))
        # Merge (keep the bigger cluster's name)
        absorbed = set()
        for i, j in sorted(to_merge, key=lambda x: x[1], reverse=True):
            if j in absorbed or i in absorbed:
                continue
            # Merge j into i (i appears first = likely bigger)
            merged[i] = (merged[i][0], merged[i][1] + merged[j][1])
            absorbed.add(j)
        if absorbed:
            merged = [m for idx, m in enumerate(merged) if idx not in absorbed]

    if len(merged) < len(result):
        print(f"  Deduped: {len(result)} → {len(merged)} folders")

    return merged


# ─── String containment ───


# ─── WordNet naming ───


def _is_too_generic(synset, max_hyponyms=200):
    """
    A hypernym is too generic if it has too many sub-types in WordNet.
    Terms like 'entity' (74k), 'object' (29k), 'artifact' (10k) are
    clearly too abstract, while 'algorithm' (2), 'compiler' (5),
    'database' (98) are specific enough.
    Uses WordNet's own taxonomy structure — no hardcoded word list.
    """
    n_hypos = len(list(synset.closure(lambda s: s.hyponyms())))
    return n_hypos > max_hyponyms


def _best_synset(model, tag, adj):
    """Pick WordNet synset using graph-neighbor context for disambiguation."""
    clean = tag.replace("-", "_").replace(" ", "_")
    synsets = wn.synsets(clean)
    words = re.split(r"[-_ ]+", tag)
    if not synsets and len(words) > 1:
        synsets = wn.synsets(words[-1])
    if not synsets:
        synsets = wn.synsets(words[0])
    if not synsets:
        return None

    nbrs = sorted(adj.get(tag, {}).items(), key=lambda x: -x[1])[:8]
    context = " ".join([tag.replace("-", " ")] + [n.replace("-", " ") for n, _ in nbrs])
    ctx_vec = embed(model, [context])[0]

    best_sim, best = -1, None
    for syn in synsets:
        gloss = syn.definition() + " " + " ".join(syn.examples())
        gv = embed(model, [gloss])[0]
        sim = float(ctx_vec @ gv)
        if sim > best_sim:
            best_sim, best = sim, syn
    return best


def find_cluster_wordnet_name(model, tag_list, adj):
    """Find a WordNet hypernym name for a cluster. Returns name or None."""
    sample = tag_list[:30] if len(tag_list) > 30 else tag_list
    hypernym_counts = Counter()

    for tag in sample:
        syn = _best_synset(model, tag, adj)
        if syn is None:
            continue
        cur = syn
        for depth in range(3):
            hyps = cur.hypernyms()
            if not hyps:
                break
            cur = hyps[0]
            # Skip hypernyms that are too close to WordNet's root
            if _is_too_generic(cur):
                continue
            name = cur.lemma_names()[0].replace("_", " ")
            if depth < 2:
                hypernym_counts[name] += 1

    if not hypernym_counts:
        return None

    best_name, best_count = hypernym_counts.most_common(1)[0]
    if best_count >= max(3, len(sample) * 0.25):
        return best_name
    return None


# ─── Naming ───


def _name_score(tag, adj, all_tags=None):
    """Score how good a tag is as a folder name. Fully generic, no hardcoded words."""
    length_penalty = max(0, len(tag) - 25) * 0.02
    words = re.split(r"[-_ ]+", tag)
    word_bonus = 0.1 if len(words) <= 3 else -0.05 * (len(words) - 3)
    if re.match(r"^[\d.]+$", tag):
        return -1.0
    # Penalize single-char or very short tags (ambiguous as folder names)
    if len(tag) <= 2:
        return -0.5
    # Penalize single generic words: if a tag is 1 word and has very high degree
    # relative to others, it's likely a hub/meta-tag, not a good category name.
    # Prefer 2-3 word compound tags that are more specific.
    if len(words) == 1 and len(tag) <= 6:
        word_bonus -= 0.1  # short single words are often ambiguous
    if "-" in tag and 2 <= len(words) <= 3:
        word_bonus += 0.05  # compound tags tend to be more descriptive
    # Tags with spaces (phrases) are less typical as category names
    if " " in tag and len(words) >= 3:
        word_bonus -= 0.1
    degree = len(adj.get(tag, {}))
    # Penalize niche tags (low degree) — they are too specific to be good folder names.
    # A well-connected tag (degree 100+) is a natural category; a tag with degree < 30
    # is likely a specific tool/paper/library name, not a topic.
    niche_penalty = max(0, (30 - degree) * 0.01)  # up to 0.30 for degree 0
    return np.log1p(degree) * 0.08 + word_bonus - length_penalty - niche_penalty


def _get_representative(tag_list, emb, tag_idx, adj):
    """Pick the best tag to name a cluster.
    Uses graph importance + name quality + containment (is it a parent of many?)."""
    if not tag_list:
        return None
    if len(tag_list) == 1:
        return tag_list[0]

    tag_set = set(tag_list)

    # Within-cluster degree
    cluster_degree = {}
    for t in tag_list:
        nbrs = set(adj.get(t, {}).keys()) & tag_set
        cluster_degree[t] = len(nbrs)

    # Containment score: how many other cluster tags contain this tag's words?
    containment_score = {}
    normalized = {}
    for t in tag_list:
        normalized[t] = set(re.split(r"[-_ ]+", t.lower()))

    for t in tag_list:
        t_words = normalized[t]
        if not t_words:
            containment_score[t] = 0
            continue
        count = sum(1 for other in tag_list if other != t and t_words < normalized[other])
        containment_score[t] = count

    # Stage 1: score all tags by combined metric
    max_deg = max(cluster_degree.values()) if cluster_degree else 1
    max_deg = max(max_deg, 1)
    max_cont = max(containment_score.values()) if containment_score else 1
    max_cont = max(max_cont, 1)

    scores = {}
    for t in tag_list:
        deg_norm = cluster_degree.get(t, 0) / max_deg
        cont_norm = containment_score.get(t, 0) / max_cont
        scores[t] = 0.5 * deg_norm + 0.3 * cont_norm + 0.2 * _name_score(t, adj)

    # Stage 2: take top 15% candidates, pick best name score
    n_candidates = max(5, len(tag_list) // 7)
    ranked = sorted(tag_list, key=lambda t: -scores[t])
    candidates = ranked[:n_candidates]

    best_tag = max(candidates, key=lambda t: _name_score(t, adj))
    return best_tag


# ─── Sub-clustering ───


def build_subtree(tag_list, emb, tags, adj, tag_idx, cfg, depth=1):
    """Recursively cluster tags within a folder using hybrid distance."""
    if len(tag_list) <= cfg.MIN_FOLDER_SIZE or depth >= cfg.MAX_DEPTH:
        return {"name": None, "children": [], "leaves": sorted(tag_list)}

    lo, hi = cfg.TARGET_CHILDREN
    k = max(lo, min(hi, len(tag_list) // 10))
    k = min(k, len(tag_list) // 2)
    if k < 2:
        return {"name": None, "children": [], "leaves": sorted(tag_list)}

    indices = [tag_idx[t] for t in tag_list if t in tag_idx]
    valid_tags = [t for t in tag_list if t in tag_idx]
    if len(valid_tags) < k * 2:
        return {"name": None, "children": [], "leaves": sorted(tag_list)}

    vecs = emb[indices]

    # Semantic distance
    sem = pdist(vecs, metric="cosine")
    sem = np.nan_to_num(sem, nan=1.0)
    mx = sem.max()
    if mx > 0:
        sem /= mx

    # Co-occurrence Jaccard distance
    tag_set = set(tags)
    nbr_sets = [set(adj.get(t, {}).keys()) & tag_set for t in valid_tags]
    n = len(valid_tags)
    cooc = np.ones(n * (n - 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n):
        a = nbr_sets[i]
        for j in range(i + 1, n):
            b = nbr_sets[j]
            u = len(a | b)
            if u > 0:
                cooc[idx] = 1.0 - len(a & b) / u
            idx += 1

    hybrid = cfg.SEMANTIC_WEIGHT * sem + cfg.COOCCURRENCE_WEIGHT * cooc
    hybrid = np.clip(hybrid, 0, None)

    try:
        Z = linkage(hybrid, method="ward")
    except Exception:
        return {"name": None, "children": [], "leaves": sorted(tag_list)}

    labels = fcluster(Z, t=k, criterion="maxclust")

    groups = defaultdict(list)
    for tag, label in zip(valid_tags, labels, strict=False):
        groups[label].append(tag)

    children = []
    leftover = []
    for _, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(grp) < cfg.MIN_FOLDER_SIZE:
            leftover.extend(grp)
            continue
        rep = _get_representative(grp, emb, tag_idx, adj)
        remaining = [t for t in grp if t != rep]
        subtree = build_subtree(remaining, emb, tags, adj, tag_idx, cfg, depth + 1)
        subtree["name"] = rep
        children.append(subtree)

    children.sort(key=lambda c: -_count_all(c))
    return {"name": None, "children": children, "leaves": sorted(leftover)}


# ─── Tree utils ───


def _count_all(t):
    n = len(t.get("leaves", []))
    if t.get("name"):
        n += 1
    for c in t.get("children", []):
        n += _count_all(c)
    return n


def _collect_all(t):
    tags = list(t.get("leaves", []))
    if t.get("name"):
        tags.append(t["name"])
    for c in t.get("children", []):
        tags.extend(_collect_all(c))
    return tags


# ─── Post-processing ───


def flatten_single_child(tree):
    tree["children"] = [flatten_single_child(c) for c in tree.get("children", [])]
    if len(tree.get("children", [])) == 1 and not tree.get("leaves") and tree.get("name"):
        child = tree["children"][0]
        tree["children"] = child.get("children", [])
        tree["leaves"] = child.get("leaves", [])
    return tree


def absorb_tiny(tree, min_size=5):
    tree["children"] = [absorb_tiny(c, min_size) for c in tree.get("children", [])]
    new_ch, new_lv = [], list(tree.get("leaves", []))
    for ch in tree.get("children", []):
        if _count_all(ch) < min_size:
            new_lv.extend(_collect_all(ch))
        else:
            new_ch.append(ch)
    tree["children"] = new_ch
    tree["leaves"] = sorted(set(new_lv))
    return tree


def redistribute_orphans(tree, emb, tags):
    """Push root-level leaves into the nearest child cluster."""
    tag_idx = {t: i for i, t in enumerate(tags)}
    leaves = tree.get("leaves", [])
    children = tree.get("children", [])
    if not children or not leaves:
        return tree

    centroids = []
    for ch in children:
        idxs = [tag_idx[t] for t in _collect_all(ch) if t in tag_idx]
        centroids.append(emb[idxs].mean(axis=0) if idxs else np.zeros(emb.shape[1]))
    centroids = np.array(centroids)

    for leaf in leaves:
        if leaf not in tag_idx:
            continue
        dists = np.linalg.norm(centroids - emb[tag_idx[leaf]], axis=1)
        children[int(np.argmin(dists))]["leaves"].append(leaf)

    for ch in children:
        ch["leaves"] = sorted(set(ch["leaves"]))
    tree["leaves"] = []
    return tree


def rename_folders_by_content(tree, model, adj, meta_tags=None, vocab_tags=None):
    """Post-processing: rename folders whose name doesn't match their content.

    For each folder, collect all tags in the subtree, compute centroid embedding,
    and check if a different tag would be a better representative. Rename if a
    significantly better candidate is found.

    Fully data-driven — uses embedding similarity, no hardcoded names.
    Meta-tags are excluded from rename candidates to prevent generic/source tags
    from becoming folder names.

    vocab_tags: optional list of high-degree tags from the full graph to expand
    the candidate vocabulary for top-level folders. This allows folders to get
    broad, well-known names even if those tags aren't in the subtree.
    """
    renames = 0
    excluded_names = meta_tags or set()

    # Pre-compute vocab embeddings for expanded vocabulary
    vocab_embs = None
    vocab_list = None
    if vocab_tags:
        vocab_list = [t for t in vocab_tags if t not in excluded_names]
        if vocab_list:
            vtexts = [re.sub(r"[-_]+", " ", t) for t in vocab_list]
            vocab_embs = model.encode(vtexts)
            norms = np.linalg.norm(vocab_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vocab_embs = vocab_embs / norms

    # Collect existing top-level names to avoid duplicates
    existing_names = set()
    for ch in tree.get("children", []):
        if ch.get("name"):
            existing_names.add(ch["name"])

    def _collect_subtree_tags(t):
        """Get all tag names from a subtree (leaves + children names)."""
        tags = []
        for leaf in t.get("leaves", []):
            if isinstance(leaf, dict):
                tags.append(leaf["tag"])
            elif isinstance(leaf, str):
                tags.append(leaf)
        for ch in t.get("children", []):
            name = ch.get("name")
            if name:
                tags.append(name)
            tags.extend(_collect_subtree_tags(ch))
        return tags

    def _rename(t, depth=0):
        nonlocal renames
        # Recurse into children first
        for ch in t.get("children", []):
            _rename(ch, depth + 1)

        name = t.get("name")
        if not name or depth == 0:
            return  # Don't rename root

        subtree_tags = _collect_subtree_tags(t)
        if len(subtree_tags) < 2:
            return

        # All candidates: current name + all tags in subtree.
        # For top-level folders, allow meta-tags (they make great broad folder names).
        # For deeper folders, exclude meta-tags to keep names specific.
        if depth == 1:
            candidates = list(set([name] + subtree_tags))
        else:
            candidates = list(set([name] + [t for t in subtree_tags if t not in excluded_names]))
        candidate_texts = [re.sub(r"[-_]+", " ", c) for c in candidates]
        candidate_embs = model.encode(candidate_texts)
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        candidate_embs = candidate_embs / norms

        # For large top-level folders, use direct children names for centroid
        # (the full subtree centroid is meaningless for grab-bags with 100+ tags).
        # Use ONLY children names (not root leaves) to avoid biasing the centroid
        # with residual tags like the folder's own name.
        if depth == 1 and len(subtree_tags) > 30:
            children_names = []
            for ch in t.get("children", []):
                ch_name = ch.get("name")
                if ch_name and ch_name != name:  # exclude self-named children
                    children_names.append(ch_name)
            if len(children_names) >= 2:
                centroid_tags = children_names
            else:
                # Fallback: use all direct tags if few children
                direct_tags = list(children_names)
                for leaf in t.get("leaves", []):
                    tag = leaf["tag"] if isinstance(leaf, dict) else leaf
                    direct_tags.append(tag)
                centroid_tags = direct_tags if len(direct_tags) >= 2 else subtree_tags
        else:
            centroid_tags = subtree_tags

        # Centroid (excluding current name to avoid bias)
        centroid_texts = [re.sub(r"[-_]+", " ", t) for t in centroid_tags]
        centroid_embs_arr = model.encode(centroid_texts)
        norms = np.linalg.norm(centroid_embs_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroid_embs_arr = centroid_embs_arr / norms
        centroid = centroid_embs_arr.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm

        # For top-level folders, expand candidates with high-degree vocab tags
        current_degree = len(adj.get(name, {}))
        if depth == 1 and vocab_embs is not None:
            # Find vocab tags that are close to the folder centroid
            vocab_sims = vocab_embs @ centroid
            # Add top-K most similar vocab tags as extra candidates.
            # Use a lower similarity threshold for niche names (degree < 30)
            # to cast a wider net for better alternatives.
            vocab_sim_thresh = 0.25 if current_degree < 30 else 0.40
            top_k = min(30 if current_degree < 30 else 20, len(vocab_list))
            top_indices = np.argsort(vocab_sims)[-top_k:][::-1]
            for vi in top_indices:
                vt = vocab_list[vi]
                if vt not in candidates and float(vocab_sims[vi]) > vocab_sim_thresh:
                    candidates.append(vt)
                    candidate_embs = np.vstack([candidate_embs, vocab_embs[vi:vi+1]])

        # Score each candidate — weight name quality higher for top-level folders
        # to prefer broad, well-known terms over random specific tags.
        current_idx = candidates.index(name)
        current_sim = float(candidate_embs[current_idx] @ centroid)

        nq_weight = 0.5 if depth == 1 else 0.3
        sim_weight = 1.0 - nq_weight

        best_idx = current_idx
        best_score = -1
        for i, cand in enumerate(candidates):
            sim = float(candidate_embs[i] @ centroid)
            nq = _name_score(cand, adj)
            score = sim_weight * sim + nq_weight * nq
            # Penalize (don't hard-block) names already used by another top-level folder.
            # Use lighter penalty for niche names (degree < 30): a duplicate good name
            # is better than keeping a cryptic niche name.
            if depth == 1 and cand != name and cand in existing_names:
                score -= 0.05 if current_degree < 30 else 0.15
            if score > best_score:
                best_score = score
                best_idx = i

        best_cand = candidates[best_idx]
        best_sim = float(candidate_embs[best_idx] @ centroid)
        current_score = sim_weight * current_sim + nq_weight * _name_score(name, adj)

        # Depth-aware rename thresholds:
        # Top-level: aggressively rename bad names (most visible to user)
        #   Use combined score margin (accounts for both similarity AND name quality).
        # Deeper: moderate cleanup
        current_degree = len(adj.get(name, {}))
        if depth == 1:
            # For niche names (low degree), bypass sim_threshold entirely —
            # a tag with degree < 30 naming a top-level folder is almost always wrong.
            if current_degree < 30:
                sim_threshold = 1.0   # always consider rename
                score_margin = 0.0    # any improvement is worth it
            else:
                sim_threshold = 0.80
                score_margin = 0.02
        else:
            sim_threshold = 0.50
            score_margin = 0.05

        # For top-level niche names: also force rename when the current name has
        # poor name quality (nq < 0.15), even if no candidate "beats" it by score.
        # This handles cases where the centroid is dominated by niche content
        # that artificially inflates the current name's similarity.
        current_nq = _name_score(name, adj)
        force_rename = (depth == 1 and current_degree < 30 and current_nq < 0.15
                        and best_cand != name)


        if (best_cand != name and current_sim < sim_threshold
                and best_score > current_score + score_margin) or force_rename:
            if depth == 1:
                existing_names.discard(name)
                existing_names.add(best_cand)
            t["name"] = best_cand
            renames += 1

    _rename(tree)
    return tree, renames


def validate_coherence(tree, model, max_iterations=3):
    """Post-processing: detect documents that are far from their folder centroid
    and reassign them to a better-matching folder.

    For each folder with documents, embed all doc titles, compute centroid,
    find outliers (> 2 stddev from centroid), and move them to a closer folder.
    """
    # Step 1: Build a map of all folders and their document centroids
    folder_centroids = {}  # id(folder_node) → centroid embedding
    folder_docs = {}  # id(folder_node) → list of (leaf_obj, doc_idx) pairs
    folder_nodes = {}  # id(folder_node) → folder node ref

    def _collect_folders(t, path=""):
        fid = id(t)
        folder_nodes[fid] = t
        name = t.get("name", "")
        current_path = f"{path}/{name}" if name else path

        all_texts = []
        doc_refs = []

        # Docs on the folder itself
        for i, doc in enumerate(t.get("documents", [])):
            title = doc.get("title", "")
            summary = doc.get("summary", "")[:200]
            text = f"{title}. {summary}".strip()
            if len(text) > 5:
                all_texts.append(text)
                doc_refs.append(("folder", fid, i))

        # Docs on leaves
        for li, leaf in enumerate(t.get("leaves", [])):
            if isinstance(leaf, dict):
                for di, doc in enumerate(leaf.get("documents", [])):
                    title = doc.get("title", "")
                    summary = doc.get("summary", "")[:200]
                    text = f"{title}. {summary}".strip()
                    if len(text) > 5:
                        all_texts.append(text)
                        doc_refs.append(("leaf", fid, li, di))

        if all_texts:
            embs = model.encode(all_texts)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embs = embs / norms
            centroid = embs.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            folder_centroids[fid] = centroid

            # Compute distances
            dists = [float(np.linalg.norm(embs[i] - centroid)) for i in range(len(all_texts))]
            folder_docs[fid] = list(zip(doc_refs, dists, all_texts))

        for ch in t.get("children", []):
            _collect_folders(ch, current_path)

    _collect_folders(tree)

    if not folder_centroids:
        return tree, 0

    # Step 2: Find outliers and reassign
    total_moved = 0
    centroid_list = list(folder_centroids.keys())
    centroid_matrix = np.array([folder_centroids[fid] for fid in centroid_list])

    for iteration in range(max_iterations):
        moved = 0
        for fid, doc_entries in list(folder_docs.items()):
            if len(doc_entries) < 3:
                continue  # Need enough docs to compute stats
            dists = [d for _, d, _ in doc_entries]
            mean_d = np.mean(dists)
            std_d = np.std(dists)
            if std_d < 0.01:
                continue  # All docs are equally close

            threshold = mean_d + 2 * std_d
            for ref, dist, text in doc_entries:
                if dist <= threshold:
                    continue
                # This doc is an outlier — find a better folder
                doc_emb = model.encode([text])[0]
                doc_emb = doc_emb / max(np.linalg.norm(doc_emb), 1e-8)
                # Find nearest folder centroid
                sims = centroid_matrix @ doc_emb
                best_fid_idx = int(np.argmax(sims))
                best_fid = centroid_list[best_fid_idx]
                if best_fid == fid:
                    continue  # Already in best folder
                # Check if the best folder is significantly closer
                best_dist = float(np.linalg.norm(doc_emb - folder_centroids[best_fid]))
                if best_dist < dist * 0.7:  # Must be 30%+ closer
                    # Move doc: remove from current, add to target's folder-level docs
                    if ref[0] == "folder":
                        _, _, doc_idx = ref
                        node = folder_nodes[fid]
                        if doc_idx < len(node.get("documents", [])):
                            doc_obj = node["documents"].pop(doc_idx)
                            target = folder_nodes[best_fid]
                            if "documents" not in target:
                                target["documents"] = []
                            target["documents"].append(doc_obj)
                            moved += 1
                    elif ref[0] == "leaf":
                        _, _, leaf_idx, doc_idx = ref
                        node = folder_nodes[fid]
                        leaves = node.get("leaves", [])
                        if leaf_idx < len(leaves):
                            leaf = leaves[leaf_idx]
                            if isinstance(leaf, dict) and doc_idx < len(leaf.get("documents", [])):
                                doc_obj = leaf["documents"].pop(doc_idx)
                                target = folder_nodes[best_fid]
                                if "documents" not in target:
                                    target["documents"] = []
                                target["documents"].append(doc_obj)
                                moved += 1

        total_moved += moved
        if moved == 0:
            break
        # Recompute centroids for next iteration
        folder_docs.clear()
        folder_centroids.clear()
        _collect_folders(tree)
        centroid_list = list(folder_centroids.keys())
        if centroid_list:
            centroid_matrix = np.array([folder_centroids[fid] for fid in centroid_list])

    return tree, total_moved


def _leaf_sort_key(leaf):
    """Sort leaves: by doc count descending, then alphabetically."""
    if isinstance(leaf, dict):
        return (-len(leaf.get("documents", [])), leaf.get("tag", ""))
    return (0, leaf)


def sort_tree(tree):
    tree["children"] = [sort_tree(c) for c in tree.get("children", [])]
    tree["children"].sort(key=lambda c: (-_count_all(c), c.get("name") or ""))
    tree["leaves"] = sorted(tree.get("leaves", []), key=_leaf_sort_key)
    return tree


# ─── Filesystem ───


def sanitize(name):
    name = name.replace("/", "-").replace("\\", "-").replace(":", "-").replace("\0", "").strip(". ")
    return name[:80] if name else "_unnamed_"



def _write_docs_json(folder_path, docs):
    """Write a single documents.json for a list of documents. Returns 1 if written, 0 otherwise."""
    if not docs:
        return 0
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "documents.json"), "w") as f:
        json.dump(docs, f, indent=2)
    return 1


def _collect_all_docs(tree):
    """Collect every document from a tree node and all descendants into a flat list."""
    docs = list(tree.get("documents", []))
    for leaf in tree.get("leaves", []):
        if isinstance(leaf, dict):
            docs.extend(leaf.get("documents", []))
    for ch in tree.get("children", []):
        docs.extend(_collect_all_docs(ch))
    return docs


def _doc_has_tag(doc, tag_name):
    """Check if a document has a given tag (case-insensitive, hyphen-normalized)."""
    norm = tag_name.lower().replace("_", "-").replace(" ", "-")
    for t in doc.get("tags", []):
        if t.lower().replace("_", "-").replace(" ", "-") == norm:
            return True
    return False


def create_folders(tree, base, is_root=False):
    d, f = 0, 0
    folder_name = tree.get("name") or "_group_"
    cur = base if is_root else os.path.join(base, sanitize(folder_name))
    os.makedirs(cur, exist_ok=True)
    d += 1

    # Collect all docs from this node (folder-node docs + leaf docs + collapsed children)
    all_docs = list(tree.get("documents", []))
    for leaf in tree.get("leaves", []):
        if isinstance(leaf, dict):
            all_docs.extend(leaf.get("documents", []))

    for ch in tree.get("children", []):
        child_docs = _count_docs(ch)
        if child_docs == 0:
            continue
        if child_docs <= 2:
            # Tiny subtree: collapse docs into current folder
            all_docs.extend(_collect_all_docs(ch))
        else:
            dd, ff = create_folders(ch, cur)
            d += dd
            f += ff

    # Filter: only keep docs whose tags contain this folder's name
    if not is_root and all_docs:
        all_docs = [doc for doc in all_docs if _doc_has_tag(doc, folder_name)]

    if all_docs:
        _write_docs_json(cur, all_docs)
        f += 1

    return d, f


def build_folder_tree(tree):
    """Convert tag_tree format to compact {name, n, c, t} format for the frontend."""
    global_seen = set()  # Each document appears in only one folder

    def compact_docs(docs):
        result = []
        for doc in docs:
            url = doc.get("url", "")
            if url in global_seen:
                continue
            global_seen.add(url)
            result.append({"u": url, "t": doc.get("title", ""), "d": doc.get("date", "")})
        return result

    def convert_node(node):
        name = node.get("name")

        # Build tag entries from leaves
        tag_merged = {}
        for leaf in node.get("leaves", []):
            if not isinstance(leaf, dict):
                continue
            tag_name = leaf.get("tag", "")
            docs = leaf.get("documents", [])
            if not docs:
                continue
            cdocs = compact_docs(docs)
            if tag_name in tag_merged:
                existing = tag_merged[tag_name]
                existing_urls = {d["u"] for d in existing[2]}
                for d in cdocs:
                    if d["u"] not in existing_urls:
                        existing[2].append(d)
                existing[1] = len(existing[2])
            else:
                tag_merged[tag_name] = [tag_name, len(cdocs), cdocs]

        # Add folder's own documents as a tag entry
        folder_docs = node.get("documents", [])
        if folder_docs and name:
            cdocs = compact_docs(folder_docs)
            if name in tag_merged:
                existing = tag_merged[name]
                existing_urls = {d["u"] for d in existing[2]}
                for d in cdocs:
                    if d["u"] not in existing_urls:
                        existing[2].append(d)
                existing[1] = len(existing[2])
            else:
                tag_merged[name] = [name, len(cdocs), cdocs]

        tags = list(tag_merged.values())

        # Convert children recursively, merging duplicates by name
        child_merged = {}
        for ch in node.get("children", []):
            child = convert_node(ch)
            cname = child["name"]
            if cname in child_merged:
                child_merged[cname]["t"].extend(child.get("t", []))
                child_merged[cname]["c"].extend(child.get("c", []))
                child_merged[cname]["n"] += child["n"]
            else:
                child_merged[cname] = child
        children = list(child_merged.values())

        # Absorb children whose name matches the parent: pull their tags/children up
        absorbed = []
        for child in children:
            if child["name"] == name:
                tags.extend(child.get("t", []))
                absorbed.extend(child.get("c", []))
            else:
                absorbed.append(child)
        children = absorbed

        n = sum(t[1] for t in tags) + sum(c["n"] for c in children)
        return {"name": name, "n": n, "c": children, "t": tags}

    return convert_node(tree)


# ─── Display ───


def print_tree(tree, prefix="", is_last=True, max_depth=4, depth=0, file=sys.stdout):
    name = tree.get("name") or ("." if depth == 0 else "_group_")
    leaves, children = tree.get("leaves", []), tree.get("children", [])
    total = _count_all(tree)
    doc_total = _count_docs(tree)

    # Split leaves into those with docs and those without
    with_docs = []
    without_docs = []
    for leaf in leaves:
        if isinstance(leaf, dict) and leaf.get("documents"):
            with_docs.append(leaf)
        else:
            without_docs.append(leaf)

    parts = []
    if children:
        parts.append(f"{len(children)} subfolders")
    if with_docs:
        parts.append(f"{len(with_docs)} tags")
    if doc_total:
        parts.append(f"{doc_total} docs")
    detail = f" ({', '.join(parts)})" if parts else ""

    label = f"📁 {name}{detail}"
    conn = "└── " if is_last else "├── "
    print(label if depth == 0 else f"{prefix}{conn}{label}", file=file)

    if depth >= max_depth and (children or with_docs):
        cp = prefix + ("    " if is_last else "│   ")
        print(f"{cp}└── ... ({total} items)", file=file)
        return

    cp = prefix + ("    " if is_last else "│   ")
    show = 6

    # Show children first, then doc-rich leaves, then summarize empty tags
    items = [(True, c) for c in children] + [(False, l) for l in with_docs[:show]]
    extra_with_docs = max(0, len(with_docs) - show)
    has_trailing = extra_with_docs > 0 or without_docs

    for i, (is_ch, item) in enumerate(items):
        last = (i == len(items) - 1) and not has_trailing
        if is_ch:
            print_tree(item, cp, last, max_depth, depth + 1, file=file)
        else:
            tag = item["tag"] if isinstance(item, dict) else item
            ndocs = len(item.get("documents", [])) if isinstance(item, dict) else 0
            doc_label = f" ({ndocs})" if ndocs else ""
            print(f"{cp}{'└── ' if last else '├── '}🏷️  {tag}{doc_label}", file=file)

    if extra_with_docs > 0 and not without_docs:
        print(f"{cp}└── ... +{extra_with_docs} more tags with docs", file=file)
    elif extra_with_docs > 0:
        print(f"{cp}{'├── ' if without_docs else '└── '}... +{extra_with_docs} more tags with docs", file=file)
    if without_docs:
        names = [l["tag"] if isinstance(l, dict) else l for l in without_docs]
        if len(names) <= 4:
            print(f"{cp}└── 📋 {', '.join(names)}", file=file)
        else:
            print(f"{cp}└── 📋 {len(names)} more tags: {', '.join(names[:4])}...", file=file)


def dissolve_grab_bag_folders(tree, model, tag_emb, tag_idx, meta_tags=None, verbose=True):
    """Dissolve top-level folders that are grab-bags (incoherent topic mixtures).

    A folder is dissolved if EITHER:
    1. Low intra-cluster coherence: its sub-folder names are semantically
       unrelated to each other (the folder mixes unrelated topics), OR
    2. The folder name is very similar to a known meta-tag

    Dissolved folders have their sub-folders promoted to top level or
    redistributed to the nearest coherent folder.
    """
    if meta_tags is None:
        meta_tags = set()
    children = tree.get("children", [])
    if len(children) < 3:
        return tree, 0

    # Compute content centroids for each top-level folder
    folder_centroids = {}
    folder_tags_list = {}
    folder_subfolder_names = {}
    for child in children:
        name = child.get("name", "")
        subtree_tags = []
        for leaf in child.get("leaves", []):
            tag = leaf["tag"] if isinstance(leaf, dict) else leaf
            subtree_tags.append(tag)
        sub_names = []
        for sub in child.get("children", []):
            if sub.get("name"):
                subtree_tags.append(sub["name"])
                sub_names.append(sub["name"])
            for leaf in sub.get("leaves", []):
                tag = leaf["tag"] if isinstance(leaf, dict) else leaf
                subtree_tags.append(tag)
        folder_tags_list[name] = subtree_tags
        folder_subfolder_names[name] = sub_names

        idxs = [tag_idx[t] for t in subtree_tags if t in tag_idx]
        if idxs:
            folder_centroids[name] = tag_emb[idxs].mean(axis=0)
        else:
            folder_centroids[name] = None

    # Criterion 1: Intra-cluster coherence.
    # For each folder with ≥3 sub-folders, compute avg pairwise cosine similarity
    # between sub-folder name embeddings. Low coherence = grab-bag.
    folder_coherence = {}
    for name, sub_names in folder_subfolder_names.items():
        if len(sub_names) < 3:
            folder_coherence[name] = 1.0  # too few sub-folders to judge reliably
            continue
        sub_embs = model.encode([s.replace("-", " ") for s in sub_names])
        norms = np.linalg.norm(sub_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        sub_embs = sub_embs / norms
        sims = sub_embs @ sub_embs.T
        n = len(sub_names)
        # Average of off-diagonal elements
        avg_coherence = (sims.sum() - n) / (n * (n - 1))
        folder_coherence[name] = float(avg_coherence)

    coherence_threshold = 0.05

    # Folder name embeddings and name-content similarity
    all_folder_names = list(folder_centroids.keys())
    name_embs = model.encode([n.replace("-", " ") for n in all_folder_names])
    norms = np.linalg.norm(name_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    name_embs = name_embs / norms
    name_emb_map = dict(zip(all_folder_names, name_embs))

    # Compute name-content similarity for each folder
    folder_name_sim = {}
    for name, centroid in folder_centroids.items():
        if centroid is None:
            folder_name_sim[name] = 0.0
            continue
        c = centroid.copy()
        c_norm = np.linalg.norm(c)
        if c_norm > 0:
            c = c / c_norm
        folder_name_sim[name] = float(name_emb_map[name] @ c)

    # Criterion 2: meta-tag similarity
    meta_tag_list = sorted(meta_tags)
    meta_embs = None
    if meta_tag_list:
        meta_embs = model.encode([t.replace("-", " ") for t in meta_tag_list])
        m_norms = np.linalg.norm(meta_embs, axis=1, keepdims=True)
        m_norms[m_norms == 0] = 1
        meta_embs = meta_embs / m_norms

    to_dissolve = []
    dissolve_reasons = {}
    for name in all_folder_names:
        coh = folder_coherence[name]
        name_sim = folder_name_sim.get(name, 0)
        n_subs = len(folder_subfolder_names.get(name, []))

        # Criterion 1a: low intra-cluster coherence (subs unrelated to each other)
        if coh <= coherence_threshold and coh < 1.0:
            to_dissolve.append(name)
            dissolve_reasons[name] = f"low coherence ({coh:.3f})"
            continue

        # Criterion 1b: large folder with low coherence AND low name-content sim.
        n_tags = len(folder_tags_list.get(name, []))
        if n_tags > 80 and n_subs >= 3 and coh < 0.08 and name_sim < 0.20:
            to_dissolve.append(name)
            dissolve_reasons[name] = f"low name-sim ({name_sim:.3f}) + low coherence ({coh:.3f})"
            continue

        # Criterion 2: folder name is very similar to a meta-tag
        if meta_embs is not None:
            n_emb = name_emb_map[name]
            meta_sims = meta_embs @ n_emb
            max_meta_sim = float(meta_sims.max())
            best_meta = meta_tag_list[int(meta_sims.argmax())]
            if max_meta_sim > 0.82:
                to_dissolve.append(name)
                dissolve_reasons[name] = f"similar to meta-tag '{best_meta}' (sim={max_meta_sim:.2f})"

    if verbose:
        # Print coherence stats for ALL folders with enough subs to evaluate
        all_stats = [(n, folder_coherence[n], len(folder_tags_list.get(n, [])), folder_name_sim.get(n, 0))
                     for n in all_folder_names if folder_coherence[n] < 1.0]  # exclude <5 subs (set to 1.0)
        all_stats.sort(key=lambda x: x[1])
        print(f"  Coherence of {len(all_stats)} folders with >=5 subs:")
        for n, coh, nt, ns in all_stats:
            dissolved_marker = " *** DISSOLVE" if n in to_dissolve else ""
            print(f"    {n}: coh={coh:.3f}, name_sim={ns:.3f}, tags={nt}, subs={len(folder_subfolder_names.get(n, []))}{dissolved_marker}")

    if not to_dissolve:
        return tree, 0

    if verbose:
        print(f"  Coherence threshold (p25): {coherence_threshold:.3f}")
        print(f"  Dissolving {len(to_dissolve)} grab-bag folders:")
        for name in to_dissolve:
            n_subs = len(folder_subfolder_names.get(name, []))
            coh = folder_coherence.get(name, 0)
            print(f"    {name}: coherence={coh:.3f}, subs={n_subs}, reason={dissolve_reasons[name]}")

    # Promote dissolved folders' sub-items to top level or merge into keepers.
    dissolved_set = set(to_dissolve)
    keeper_children = [c for c in children if c.get("name") not in dissolved_set]
    dissolved_children = [c for c in children if c.get("name") in dissolved_set]

    # Build centroids for keeper folders
    keeper_centroids = []
    for child in keeper_children:
        name = child.get("name", "")
        c = folder_centroids.get(name)
        if c is not None:
            n = np.linalg.norm(c)
            keeper_centroids.append(c / n if n > 0 else c)
        else:
            keeper_centroids.append(np.zeros(tag_emb.shape[1]))
    keeper_centroids = np.array(keeper_centroids) if keeper_centroids else None

    total_promoted = 0
    root_leaves = list(tree.get("leaves", []))

    def _count_subtree(node):
        n = len(node.get("leaves", []))
        for ch in node.get("children", []):
            n += _count_subtree(ch)
        return n

    min_promote_size = 8
    # Only merge into a keeper if the sub-folder is semantically close enough
    min_merge_sim = 0.30

    def _find_best_keeper(name):
        """Find the best keeper for a sub-folder, returning (index, similarity)."""
        if keeper_centroids is None or len(keeper_children) == 0:
            return -1, 0.0
        if name in tag_idx:
            v = tag_emb[tag_idx[name]]
        else:
            v = model.encode([name.replace("-", " ")])[0]
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            v = v / v_norm
        sims = keeper_centroids @ v
        best = int(np.argmax(sims))
        return best, float(sims[best])

    for dissolved in dissolved_children:
        for sub in dissolved.get("children", []):
            sub_name = sub.get("name", "")
            sub_size = _count_subtree(sub)
            is_meta_sub = sub_name in meta_tags
            if sub_size >= min_promote_size and not is_meta_sub:
                # Large enough: promote to top level
                keeper_children.append(sub)
                total_promoted += 1
            else:
                # Small: merge into nearest keeper ONLY if good match
                best, sim = _find_best_keeper(sub_name)
                if best >= 0 and sim >= min_merge_sim:
                    keeper_children[best].setdefault("children", []).append(sub)
                else:
                    # No good match: promote anyway
                    keeper_children.append(sub)
                total_promoted += 1
        for leaf in dissolved.get("leaves", []):
            # Merge leaves into nearest keeper if good match
            tag = leaf["tag"] if isinstance(leaf, dict) else leaf
            best, sim = _find_best_keeper(tag)
            if best >= 0 and sim >= min_merge_sim:
                keeper_children[best].setdefault("leaves", []).append(leaf)
            else:
                root_leaves.append(leaf)
            total_promoted += 1

    tree["children"] = keeper_children
    tree["leaves"] = root_leaves
    return tree, total_promoted


def split_grab_bags(tree, model, tag_emb, tag_idx, adj, meta_tags=None, verbose=True, already_split=None):
    """Split large incoherent top-level folders into coherent sub-groups.

    Instead of dissolving grab-bags and promoting sub-folders (which creates
    new grab-bags), this function SPLITS each grab-bag into 2-5 coherent groups
    using hierarchical clustering on sub-folder embeddings.

    Each group becomes a new top-level folder named by its best representative.
    """
    if meta_tags is None:
        meta_tags = set()
    if already_split is None:
        already_split = set()

    children = tree.get("children", [])
    if len(children) < 3:
        return tree, 0

    def _collect_all_tags(node):
        tags = []
        for leaf in node.get("leaves", []):
            tag = leaf["tag"] if isinstance(leaf, dict) else leaf
            tags.append(tag)
        for ch in node.get("children", []):
            if ch.get("name"):
                tags.append(ch["name"])
            tags.extend(_collect_all_tags(ch))
        return tags

    # Compute spread + name-content similarity for each folder
    folder_stats = {}
    for child in children:
        name = child.get("name", "")
        all_tags = _collect_all_tags(child)
        n_tags = len(all_tags)

        if n_tags < 8:
            continue

        idxs = [tag_idx[t] for t in all_tags if t in tag_idx]
        if len(idxs) < 5:
            continue

        embs = tag_emb[idxs]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs_normed = embs / norms
        centroid = embs_normed.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm

        sims_to_centroid = embs_normed @ centroid
        spread = 1.0 - float(sims_to_centroid.mean())

        name_emb = model.encode([name.replace("-", " ")])[0]
        n_norm = np.linalg.norm(name_emb)
        if n_norm > 0:
            name_emb = name_emb / n_norm
        name_sim = float(name_emb @ centroid)

        score = spread * (1.0 - name_sim)
        folder_stats[name] = {
            "spread": spread, "name_sim": name_sim, "n_tags": n_tags,
            "n_subs": len(child.get("children", [])), "score": score,
        }

    if not folder_stats:
        return tree, 0

    # Identify grab-bags to split (skip already-split folders to prevent cascading)
    to_split = []
    for name, stats in folder_stats.items():
        if name in already_split:
            continue
        n_tags = stats["n_tags"]
        score = stats["score"]
        # Split scattered folders with poor name coverage
        if n_tags >= 80 and score > 0.33:
            to_split.append(name)
        elif n_tags >= 40 and score > 0.35:
            to_split.append(name)
        elif n_tags >= 15 and score > 0.38:
            to_split.append(name)

    if verbose:
        sorted_stats = sorted(folder_stats.items(), key=lambda x: -x[1]["score"])
        print(f"  Content spread of {len(folder_stats)} large folders (top 20):")
        for name, stats in sorted_stats[:20]:
            marker = " *** SPLIT" if name in to_split else ""
            print(f"    {name}: spread={stats['spread']:.3f}, name_sim={stats['name_sim']:.3f}, "
                  f"score={stats['score']:.3f}, tags={stats['n_tags']}, subs={stats['n_subs']}{marker}")

    if not to_split:
        return tree, 0

    # Split each grab-bag into coherent groups
    split_set = set(to_split)
    new_children = []
    total_splits = 0

    for child in children:
        name = child.get("name", "")
        if name not in split_set:
            new_children.append(child)
            continue

        # Gather all items (sub-folders + leaves) with their embeddings
        items = []  # (embedding, node_or_leaf, is_child)
        for sub in child.get("children", []):
            sub_name = sub.get("name", "")
            sub_tags = _collect_all_tags(sub)
            if sub_name:
                sub_tags.append(sub_name)
            sub_idxs = [tag_idx[t] for t in sub_tags if t in tag_idx]
            if sub_idxs:
                emb = tag_emb[sub_idxs].mean(axis=0)
            else:
                emb = model.encode([sub_name.replace("-", " ")])[0]
            items.append((emb, sub, True))
        for leaf in child.get("leaves", []):
            tag = leaf["tag"] if isinstance(leaf, dict) else leaf
            if tag in tag_idx:
                emb = tag_emb[tag_idx[tag]]
            else:
                emb = model.encode([tag.replace("-", " ")])[0]
            items.append((emb, leaf, False))

        if len(items) < 4:
            new_children.append(child)
            continue

        # Hierarchical clustering to split into coherent groups
        embs_mat = np.array([it[0] for it in items])
        norms = np.linalg.norm(embs_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs_mat = embs_mat / norms

        # Use cosine distance
        dists = pdist(embs_mat, metric="cosine")
        Z = linkage(dists, method="ward")

        # Target 2-5 groups depending on folder size
        n_items = len(items)
        if n_items >= 30:
            target_groups = 4
        elif n_items >= 15:
            target_groups = 3
        else:
            target_groups = 2

        labels = fcluster(Z, t=target_groups, criterion="maxclust")
        groups = defaultdict(list)
        for i, (emb, item, is_child) in enumerate(items):
            groups[labels[i]].append((item, is_child))

        # Sort groups by size (largest first)
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        # Merge tiny groups (< 3 items) into the largest group
        main_groups = []
        for g in sorted_groups:
            if len(g) >= 3 or not main_groups:
                main_groups.append(g)
            else:
                # Merge into the largest (first) group
                main_groups[0].extend(g)

        # Create a new top-level folder for each group
        for gi, group_items in enumerate(main_groups):
            group_children = [it for it, is_ch in group_items if is_ch]
            group_leaves = [it for it, is_ch in group_items if not is_ch]

            # Name the group by the candidate closest to the group's centroid
            # Collect all candidate names and their embeddings
            candidate_names = []
            candidate_embs_list = []
            for ch in group_children:
                ch_name = ch.get("name", "")
                if ch_name and ch_name not in (meta_tags or set()):
                    ch_emb = model.encode([ch_name.replace("-", " ")])[0]
                    candidate_names.append(ch_name)
                    candidate_embs_list.append(ch_emb)
            for lf in group_leaves:
                lf_tag = lf["tag"] if isinstance(lf, dict) else lf
                if lf_tag and lf_tag not in (meta_tags or set()):
                    lf_emb = model.encode([lf_tag.replace("-", " ")])[0]
                    candidate_names.append(lf_tag)
                    candidate_embs_list.append(lf_emb)

            if not candidate_names:
                if group_children:
                    best_name = group_children[0].get("name", "unnamed")
                elif group_leaves:
                    best_name = group_leaves[0]["tag"] if isinstance(group_leaves[0], dict) else group_leaves[0]
                else:
                    continue
            else:
                # Compute group centroid from all item embeddings
                group_emb_indices = []
                for it, is_ch in group_items:
                    if is_ch:
                        sub_tags = _collect_all_tags(it)
                        if it.get("name"):
                            sub_tags.append(it["name"])
                        for st in sub_tags:
                            if st in tag_idx:
                                group_emb_indices.append(tag_idx[st])
                    else:
                        tag = it["tag"] if isinstance(it, dict) else it
                        if tag in tag_idx:
                            group_emb_indices.append(tag_idx[tag])
                if group_emb_indices:
                    group_centroid = tag_emb[group_emb_indices].mean(axis=0)
                    gc_norm = np.linalg.norm(group_centroid)
                    if gc_norm > 0:
                        group_centroid = group_centroid / gc_norm
                else:
                    group_centroid = np.array(candidate_embs_list).mean(axis=0)
                    gc_norm = np.linalg.norm(group_centroid)
                    if gc_norm > 0:
                        group_centroid = group_centroid / gc_norm

                # Score candidates: centroid similarity + name quality
                best_name = candidate_names[0]
                best_score = -1
                for cn, ce in zip(candidate_names, candidate_embs_list):
                    ce_norm = np.linalg.norm(ce)
                    if ce_norm > 0:
                        ce = ce / ce_norm
                    sim = float(ce @ group_centroid)
                    nq = _name_score(cn, adj)
                    score = 0.5 * sim + 0.5 * nq
                    if score > best_score:
                        best_score = score
                        best_name = cn

            new_folder = {
                "name": best_name,
                "children": group_children,
                "leaves": group_leaves,
            }
            if child.get("documents"):
                new_folder["documents"] = []
            new_children.append(new_folder)

        total_splits += 1
        already_split.add(name)
        # Also mark all new group names as already-split to prevent re-splitting
        for ch in new_children[-len(main_groups):]:
            already_split.add(ch.get("name", ""))
        if verbose:
            group_sizes = [len(g) for g in groups.values()]
            new_names = [ch.get("name", "?") for ch in new_children[-len(main_groups):]]
            print(f"    Split {name} ({len(items)} items) → {len(main_groups)} groups: {group_sizes} as {new_names}")

    tree["children"] = new_children
    tree["leaves"] = tree.get("leaves", [])

    if verbose and total_splits:
        print(f"  Split {total_splits} grab-bag folders")

    return tree, total_splits


def absorb_tiny_root_folders(tree, model, tag_emb, tag_idx, min_tags=8, verbose=True):
    """Merge very small root folders into their nearest larger neighbor.

    Folders with fewer than min_tags tags are absorbed into the most
    semantically similar larger folder. This prevents cluttering the
    root level with folders like 'actuarial' (3 tags) or 'helm' (3 tags).
    """
    children = tree.get("children", [])
    if len(children) < 5:
        return tree

    def _count_tags(node):
        n = len(node.get("leaves", []))
        for ch in node.get("children", []):
            n += _count_tags(ch)
        return n

    def _collect_all_tags(node):
        tags = []
        for leaf in node.get("leaves", []):
            tag = leaf["tag"] if isinstance(leaf, dict) else leaf
            tags.append(tag)
        for ch in node.get("children", []):
            if ch.get("name"):
                tags.append(ch["name"])
            tags.extend(_collect_all_tags(ch))
        return tags

    # Compute centroid for each root folder
    folder_centroids = {}
    folder_sizes = {}
    for child in children:
        name = child.get("name", "")
        all_tags = _collect_all_tags(child)
        folder_sizes[name] = len(all_tags)
        idxs = [tag_idx[t] for t in all_tags if t in tag_idx]
        if idxs:
            embs = tag_emb[idxs]
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            centroid = (embs / norms).mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            folder_centroids[name] = centroid
        else:
            emb = model.encode([name.replace("-", " ")])[0]
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            folder_centroids[name] = emb

    # Identify tiny folders and their best targets
    tiny = []
    large = []
    for child in children:
        name = child.get("name", "")
        n_tags = folder_sizes.get(name, 0)
        if n_tags < min_tags:
            tiny.append(child)
        else:
            large.append(child)

    if not tiny or not large:
        return tree

    # Build centroid matrix for large folders
    large_names = [ch.get("name", "") for ch in large]
    large_centroids = np.array([folder_centroids[n] for n in large_names])

    absorbed = 0
    for child in tiny:
        name = child.get("name", "")
        centroid = folder_centroids.get(name)
        if centroid is None:
            large.append(child)
            continue

        # Find nearest large folder by cosine similarity
        sims = large_centroids @ centroid
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        # Use a lower similarity threshold for very tiny folders (< 5 tags) —
        # they are almost certainly noise that should be absorbed somewhere.
        n_tags = folder_sizes.get(name, 0)
        sim_thresh = 0.10 if n_tags < 5 else 0.20
        if best_sim < sim_thresh:
            # Too dissimilar, keep as-is
            large.append(child)
            continue

        target = large[best_idx]
        # Absorb: add tiny folder's children and leaves into target
        target_children = target.get("children", [])
        target_leaves = target.get("leaves", [])

        # If the tiny folder has sub-structure, add it as a child of the target
        if child.get("children"):
            target_children.append(child)
        else:
            # Just merge leaves
            target_leaves.extend(child.get("leaves", []))

        target["children"] = target_children
        target["leaves"] = target_leaves
        absorbed += 1

        if verbose:
            print(f"    Absorbed '{name}' ({folder_sizes.get(name, 0)} tags) → "
                  f"'{target.get('name', '')}' (sim={best_sim:.3f})")

    tree["children"] = large
    if verbose and absorbed:
        print(f"  Absorbed {absorbed} tiny root folders (< {min_tags} tags)")

    return tree


def merge_duplicate_folders(tree, model, adj, meta_tags=None, vocab_tags=None, verbose=True):
    """Merge root folders that share the same name.

    When multiple top-level folders have identical names, keep the largest one
    and merge the others' children/leaves into it. Then rename the smaller
    duplicates to unique names if possible, or merge them into the largest.
    """
    children = tree.get("children", [])
    if not children:
        return tree

    # Group by name
    from collections import defaultdict as _dd
    name_groups = _dd(list)
    for i, ch in enumerate(children):
        name_groups[ch.get("name", "")].append(i)

    duplicates = {name: idxs for name, idxs in name_groups.items() if len(idxs) > 1}
    if not duplicates:
        return tree

    merged_count = 0
    to_remove = set()

    for name, idxs in duplicates.items():
        # Sort by size (largest first)
        sized = [(i, _count_all(children[i])) for i in idxs]
        sized.sort(key=lambda x: -x[1])
        target_idx = sized[0][0]
        target = children[target_idx]

        for donor_idx, _ in sized[1:]:
            donor = children[donor_idx]
            # Merge donor's children and leaves into target
            target_children = target.get("children", [])
            target_leaves = target.get("leaves", [])

            # Add donor as a sub-folder of target (preserving its structure)
            if donor.get("children") or len(donor.get("leaves", [])) > 5:
                target_children.append(donor)
            else:
                target_leaves.extend(donor.get("leaves", []))

            target["children"] = target_children
            target["leaves"] = target_leaves
            to_remove.add(donor_idx)
            merged_count += 1

            if verbose:
                print(f"    Merged duplicate '{name}' into main folder")

    if to_remove:
        tree["children"] = [ch for i, ch in enumerate(children) if i not in to_remove]

    if verbose and merged_count:
        print(f"  Merged {merged_count} duplicate folders")

    return tree


# ─── Main ───


def main(triples_path=None, output_dir=None, database_path=None):
    if triples_path is None:
        triples_path = sys.argv[1] if len(sys.argv) >= 2 else "database/triples.json"
    if output_dir is None:
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "tree"
    if database_path is None:
        database_path = sys.argv[3] if len(sys.argv) >= 4 else "database/database.json"

    # Fix global random seed for deterministic results
    np.random.seed(42)
    import random
    random.seed(42)

    print("─── Loading ───")
    triples = load_triples(triples_path)
    print(f"  {len(triples)} triples from co-occurrence data")

    database = load_database(database_path)
    print(f"  {len(database)} documents from database")

    print("\n─── Splitting compound tags ───")
    triples = split_compound_tags(triples)
    print(f"  {len(triples)} triples after splitting compounds")

    print("\n─── Filtering garbage tags ───")
    triples, garbage_tags = filter_garbage_tags(triples)

    print("\n─── Generating document tag co-occurrences ───")
    doc_triples = triples_from_documents(database, garbage_tags=garbage_tags)
    print(f"  {len(doc_triples)} triples from document tags")
    triples = triples + doc_triples
    print(f"  {len(triples)} total triples (merged)")

    print("\n─── Deduplicating tags ───")
    triples, merge_map = deduplicate_triples(triples)

    adj = build_graph(triples)
    # Auto-tune config based on dataset size
    preliminary_tags = get_all_tags(adj, Config.MIN_TAG_DEGREE)
    cfg = Config.auto_tune(len(preliminary_tags), len(database))
    tags = get_all_tags(adj, cfg.MIN_TAG_DEGREE)
    print(f"\n  Auto-tuned config for {len(tags)} tags, {len(database)} docs")
    print(f"    LOUVAIN_RESOLUTION={cfg.LOUVAIN_RESOLUTION:.2f}, MIN_FOLDER_SIZE={cfg.MIN_FOLDER_SIZE}, "
          f"MIN_TOP_CLUSTER={cfg.MIN_TOP_CLUSTER}, MAX_TOP_CLUSTER_RATIO={cfg.MAX_TOP_CLUSTER_RATIO:.3f}")
    print(f"  After dedup: {len(triples)} triples, {len(tags)} tags (degree >= {cfg.MIN_TAG_DEGREE})")

    print("\n─── Embedding ───")
    model = load_model(cfg.MODEL_NAME)
    tag_emb = embed_tags(model, tags)
    tag_idx = {t: i for i, t in enumerate(tags)}

    # Auto-disable WordNet if <30% of tags get hits (non-English / technical domains)
    if cfg.USE_WORDNET:
        sample = tags[:min(100, len(tags))]
        hits = sum(1 for t in sample if wn.synsets(t.replace("-", "_")))
        hit_rate = hits / max(len(sample), 1)
        if hit_rate < 0.30:
            cfg.USE_WORDNET = False
            print(f"  WordNet auto-disabled: only {hit_rate:.0%} of tags have synsets")
        else:
            print(f"  WordNet enabled: {hit_rate:.0%} of tags have synsets")

    print("\n─── Detecting meta-tags ───")
    meta_tags = detect_meta_tags(adj, tags, tag_emb, tag_idx)
    excluded_from_graph = garbage_tags | meta_tags

    # Rebuild graph excluding meta-tags from community detection
    # Meta-tags stay as valid tags for document assignment, but are removed
    # from the graph that Louvain uses so they don't form top-level folders.
    community_triples = [
        t for t in triples
        if t["head"] not in meta_tags and t["tail"] not in meta_tags
    ]
    community_adj = build_graph(community_triples)
    community_tags = get_all_tags(community_adj, cfg.MIN_TAG_DEGREE)
    community_emb = embed_tags(model, community_tags)
    community_idx = {t: i for i, t in enumerate(community_tags)}
    print(f"  Community graph: {len(community_tags)} tags (excluded {len(tags) - len(community_tags)} meta-tags)")

    print("\n─── Propagating tags by title ───")
    tree_tag_set = set(tags)
    n_propagated = propagate_tags_by_title(database, tree_tag_set, merge_map, excluded_from_graph)
    print(f"  {n_propagated} documents enriched via title-group propagation")

    print("\n─── Enriching remaining undertagged documents ───")
    n_enriched = enrich_undertagged_docs(database, tree_tag_set, merge_map, excluded_from_graph, model)
    print(f"  {n_enriched} documents enriched with synthetic tags")

    print("\n─── Assigning documents to tags ───")
    tag_to_docs, assigned_docs = assign_documents(database, merge_map, tags)

    print("\n─── Top-level: Louvain community detection ───")
    top_clusters = louvain_top_level(community_adj, community_tags, community_emb, model, cfg)

    # Redistribute meta-tags into the nearest community cluster as leaves.
    # This ensures meta-tags become leaf nodes inside topic folders, not top-level folders.
    if meta_tags:
        cluster_centroids = []
        for _, tag_list in top_clusters:
            idxs = [tag_idx[t] for t in tag_list if t in tag_idx]
            if idxs:
                cluster_centroids.append(tag_emb[idxs].mean(axis=0))
            else:
                cluster_centroids.append(np.zeros(tag_emb.shape[1]))
        cluster_centroids = np.array(cluster_centroids)

        placed_meta = 0
        for mt in meta_tags:
            if mt not in tag_idx:
                continue
            v = tag_emb[tag_idx[mt]]
            dists = np.linalg.norm(cluster_centroids - v, axis=1)
            best = int(np.argmin(dists))
            top_clusters[best][1].append(mt)
            placed_meta += 1
        print(f"  Redistributed {placed_meta} meta-tags into topic clusters as leaves")

    print(f"\n─── Building sub-trees ({len(top_clusters)} top-level folders) ───")
    children = []
    for folder_name, tag_list in top_clusters:
        remaining = [t for t in tag_list if t != folder_name]
        subtree = build_subtree(remaining, tag_emb, tags, adj, tag_idx, cfg, depth=1)
        subtree["name"] = folder_name
        children.append(subtree)

    tree = {"name": None, "children": children, "leaves": []}

    print("\n─── Post-processing ───")
    tree = redistribute_orphans(tree, tag_emb, tags)
    tree = absorb_tiny(tree, 5)
    tree = flatten_single_child(tree)
    tree = sort_tree(tree)

    print("\n─── Attaching documents ───")
    tree = attach_documents(tree, assigned_docs)
    total_docs = _count_docs(tree)
    print(f"  {total_docs} documents attached to tree")

    print("\n─── Dissolving grab-bag folders (iterative) ───")
    for dissolve_round in range(5):
        tree, n_dissolved = dissolve_grab_bag_folders(tree, model, tag_emb, tag_idx, meta_tags=meta_tags)
        print(f"  Round {dissolve_round + 1}: {n_dissolved} items redistributed")
        if n_dissolved == 0:
            break

    print(f"\n─── Collapsing thin subtrees (min {cfg.MIN_DOCS_FOR_FOLDER} docs) ───")
    tree = collapse_thin_subtrees(tree, cfg.MIN_DOCS_FOR_FOLDER)

    # Build expanded vocabulary: high-degree tags that make good category names
    vocab_tags = [t for t in tags if len(adj.get(t, {})) >= 15]
    print(f"\n─── Renaming folders by content ({len(vocab_tags)} vocab tags) ───")
    tree, n_renames = rename_folders_by_content(tree, model, adj, meta_tags=meta_tags, vocab_tags=vocab_tags)
    print(f"  {n_renames} folders renamed")

    print("\n─── Splitting grab-bag folders (post-rename) ───")
    split_tracking = set()
    for split_round in range(3):
        tree, n_split = split_grab_bags(tree, model, tag_emb, tag_idx, adj, meta_tags=meta_tags, already_split=split_tracking)
        if n_split == 0:
            break
        print(f"  Round {split_round + 1}: {n_split} folders split")
        # Rename the newly created split folders
        tree, nr = rename_folders_by_content(tree, model, adj, meta_tags=meta_tags, vocab_tags=vocab_tags)
        if nr:
            print(f"  {nr} folders re-renamed")

    print("\n─── Absorbing tiny root folders ───")
    tree = absorb_tiny_root_folders(tree, model, tag_emb, tag_idx, min_tags=10, verbose=True)

    print("\n─── Final rename pass ───")
    tree, n_final_renames = rename_folders_by_content(tree, model, adj, meta_tags=meta_tags, vocab_tags=vocab_tags)
    print(f"  {n_final_renames} folders renamed in final pass")

    print("\n─── Merging duplicate folders ───")
    tree = merge_duplicate_folders(tree, model, adj, meta_tags=meta_tags, vocab_tags=vocab_tags)
    # Rename again after merge — the merged sub-folders may need new names
    tree, n_post_merge = rename_folders_by_content(tree, model, adj, meta_tags=meta_tags, vocab_tags=vocab_tags)
    if n_post_merge:
        print(f"  {n_post_merge} folders renamed after merge")

    print("\n─── Coherence validation ───")
    tree, n_moved = validate_coherence(tree, model)
    print(f"  {n_moved} documents reassigned to better folders")

    tree = sort_tree(tree)  # Re-sort so doc-rich leaves/folders surface first

    def nf(t):
        return len(t.get("children", [])) + sum(nf(c) for c in t.get("children", []))

    folders = nf(tree)
    total = _count_all(tree)

    print(f"\n{'=' * 50}")
    print(f"  Top-level:     {len(tree['children'])} folders")
    print(f"  Total folders: {folders}")
    print(f"  Total tags:    {total}")
    print(f"  Tags/folder:   {total / max(folders, 1):.1f}")
    print(f"  Total docs:    {total_docs}")
    print(f"{'=' * 50}")

    print("\n─── Tree ───\n")
    print_tree(tree, max_depth=3)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, "tag_tree.json"), "w") as f:
        json.dump(tree, f, indent=2)
    print(f"\n  Written: {output_dir}/tag_tree.json")

    folder_tree = build_folder_tree(tree)
    folder_tree_path = os.path.join("docs", "folder_tree.json")
    with open(folder_tree_path, "w") as f:
        json.dump(folder_tree, f)
    print(f"  Written: {folder_tree_path}")


if __name__ == "__main__":
    main()
