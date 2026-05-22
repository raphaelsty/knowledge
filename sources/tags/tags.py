"""
Tags module for generating tag relationships and automatic tagging.

This module provides utilities for building a tag co-occurrence graph and
automatically discovering additional relevant tags for documents based on
their content.
"""

import collections
import itertools
import re

from flashtext import KeywordProcessor

__all__ = ["get_extra_tags", "get_tags_triples"]


# ---------------------------------------------------------------------------
# "Weak" tag filter
# ---------------------------------------------------------------------------
#
# We only use flashtext for extra-tag matching, which is whole-word exact
# only — so the surface area for false positives is far smaller than under
# the old TF-IDF char-n-gram retriever. We still drop a tiny set of
# obvious junk candidates:
#   - length ≤ 2 characters (unless hyphenated → compound tag)
#   - purely numeric (years, "2.0", …)
#   - bare English stopwords
#
# The stopword list mirrors NLTK's English set with no runtime dependency,
# so tags.py stays import-light. Tags containing a hyphen or digit are
# trusted (domain-specific compounds like `fine-tuning` or `v2`).

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_NUMERIC_RE = re.compile(r"^[\d.]+$")


def _is_weak_tag(tag: str) -> bool:
    """Return True if `tag` is too generic / empty to carry meaning."""
    if not tag:
        return True
    t = tag.strip()
    if not t:
        return True
    # Compound tags (hyphenated) are trusted — keep e.g. `fine-tuning`, `nlp-task`.
    if "-" in t:
        return False
    if len(t) <= 2:
        return True
    if _NUMERIC_RE.match(t):
        return True
    if t.lower() in _STOPWORDS:
        return True
    return False


def get_tags_triples(
    data: dict[str, dict],
    excluded_tags: dict[str, bool] | None = None,
) -> list[dict]:
    """
    Build a graph of tag co-occurrence relationships.

    Creates edges between tags that appear together in the same document,
    enabling visualization of knowledge domain relationships.

    Parameters
    ----------
    data : dict[str, dict]
        Dictionary mapping URLs to document metadata. Each document should
        contain 'tags' and 'extra-tags' lists.
    excluded_tags : dict[str, bool], optional
        Tags to exclude from the graph (e.g., generic source tags like 'github').

    Returns
    -------
    list[dict]
        List of edge dictionaries with 'head' and 'tail' keys representing
        connected tags. Each edge appears only once (undirected graph).

    Example
    -------
    >>> documents = {
    ...     "url1": {"tags": ["python", "ml"], "extra-tags": ["pytorch"]},
    ...     "url2": {"tags": ["python", "web"], "extra-tags": []},
    ... }
    >>> triples = get_tags_triples(documents)
    >>> # Creates edges: python-ml, python-pytorch, ml-pytorch, python-web
    """
    excluded_tags = {} if excluded_tags is None else excluded_tags
    triples = []

    # Track seen edges to avoid duplicates (undirected graph)
    seen: dict[str, dict[str, bool]] = collections.defaultdict(dict)

    for _, document in data.items():
        all_tags = document["tags"] + document["extra-tags"]

        # Create edges for all tag pairs within the document
        for head, tail in itertools.combinations(all_tags, 2):
            # Skip excluded tags
            if head in excluded_tags or tail in excluded_tags:
                continue

            # Skip if edge already exists (either direction)
            if head in seen[tail] or tail in seen[head]:
                continue

            triples.append({"head": head, "tail": tail})
            seen[head][tail] = True
            seen[tail][head] = True

    return triples


def _build_keyword_processor(vocab: list[str]) -> KeywordProcessor:
    """Build a flashtext keyword processor that handles compound tags.

    Multi-word tags in our vocabulary are stored hyphenated (e.g.
    ``information-retrieval``, ``unstructured-data``) but documents
    almost always write them with spaces (``information retrieval``).
    For each hyphenated tag we register both forms with the same
    ``clean_name`` (the canonical hyphenated tag) so flashtext returns
    the canonical form regardless of which variant the text used.

    flashtext does whole-word matching with case-insensitive lookup —
    no false positives from substrings (``ai`` no longer matches
    ``pairing``) and no per-doc TF-IDF noise.
    """
    kp = KeywordProcessor(case_sensitive=False)
    for tag in vocab:
        if not tag:
            continue
        kp.add_keyword(tag, clean_name=tag)
        if "-" in tag:
            spaced = tag.replace("-", " ")
            # add_keyword is idempotent for the same (text, clean_name)
            # pair, so we don't have to track what we've already added.
            kp.add_keyword(spaced, clean_name=tag)
    return kp


def get_extra_tags(
    data: dict[str, dict],
    shared_tags: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, dict]:
    """Discover additional relevant tags via exact-keyword matching.

    Drives flashtext over each doc's title + summary, looking for
    occurrences of any tag in the (per-personality + shared) vocabulary.
    A tag is added when:

    1. it appears as a whole-word match in the document text,
    2. it isn't already on the document, and
    3. it isn't a weak/stopword candidate.

    Compound tags like ``information-retrieval`` match either
    ``information-retrieval`` or ``information retrieval`` in the text;
    output is always the canonical hyphenated form.

    Parameters
    ----------
    data
        ``{url: {title, summary, tags, …}}``. Mutated only via the
        returned dict.
    shared_tags
        Cross-personality vocabulary. When supplied, every personality
        draws from the same pool, so a tag coined elsewhere can still
        get attached here.
    top_k
        Cap on extra-tags per document. Ranking is by match frequency
        (most-mentioned tag first), with stable tie-breaking by tag name.

    Returns
    -------
    dict[str, dict]
        New dict with ``extra-tags`` populated on every document.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    # 1. Collect the existing tag universe. Per-doc tag sets feed the
    #    "don't resuggest" filter; the global vocabulary is what
    #    flashtext gets to match against.
    vocab: dict[str, bool] = {}
    tagged: dict[str, set[str]] = collections.defaultdict(set)
    for url, document in data.items():
        for tag in document.get("tags", []):
            tagged[url].add(tag)
            if not _is_weak_tag(tag):
                vocab[tag] = True
    if shared_tags:
        for tag in shared_tags:
            if tag and not _is_weak_tag(tag):
                vocab[tag] = True

    kp = _build_keyword_processor(list(vocab))

    # 2. For each doc, count keyword occurrences in title + summary
    #    and pick the most-mentioned ones, skipping anything already
    #    on the document or filtered as weak.
    extra_tags: dict[str, list[str]] = {}
    for url, document in data.items():
        text = (document.get("title", "") or "") + " " + (document.get("summary", "") or "")
        if not text.strip():
            extra_tags[url] = []
            continue
        hits = kp.extract_keywords(text)
        if not hits:
            extra_tags[url] = []
            continue
        # Counter preserves insertion order for ties → reproducible runs.
        ranked = [
            tag
            for tag, _count in collections.Counter(hits).most_common()
            if tag not in tagged[url] and not _is_weak_tag(tag)
        ]
        picked = ranked[:top_k]
        assert len(picked) <= top_k, f"{url}: {len(picked)} > top_k={top_k}"
        assert all(not _is_weak_tag(t) for t in picked), f"{url}: weak tag leaked: {picked!r}"
        extra_tags[url] = picked

    # Merge: spread the loaded document FIRST so any pre-existing
    # `extra-tags` (carried over from PG by `load_documents`) are
    # *replaced* by what we just computed. The previous order
    # (``{**{"extra-tags": new}, **document}``) silently let the old
    # value win — making `KNOWLEDGE_RETAG` a no-op for any doc that
    # had ever been tagged before.
    return {url: {**document, "extra-tags": extra_tags[url]} for url, document in data.items()}
