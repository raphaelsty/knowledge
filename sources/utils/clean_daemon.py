"""Pedagogical title / summary cleaner for VIP documents.

A long-lived daemon (Docker compose service on prod) that produces
a pedagogical rewrite of selected VIP documents and writes the
result back into the `clean_title` and `clean_summary` columns on
the `documents` table. The raw `title` and `summary` columns are
left untouched so search (which indexes the raw values) is not
affected.

Scope alignment with the global feed
------------------------------------
The OpenAI bill is the dominant operating cost of this daemon, so
the selection mirrors *exactly* the documents that can appear in
the anonymous `/api/feed` (`handlers::users::build_feed_payload`).
Same WHERE clauses, plus a couple of extras:

  * `d.date IS NOT NULL AND d.deleted = FALSE`
    Identical to the feed query — the universe of cleanable docs.

  * `u.vip = TRUE`
    Only VIP-owned documents. Non-VIP users' libraries don't get
    boosted in the feed score, so cleaning them spends model
    tokens that few people will ever read.

  * `d.date >= now() - INTERVAL '21 days'`
    The feed's recency bonus tops out at ~5 weeks and decays to
    zero past that — anything older than 3 weeks is unlikely to
    surface in the top-N regardless. 3 weeks is the budget the
    operator chose.

  * `lower(d.source) IN (tweets ∪ papers)`
    The two surfaces with worthwhile rewrites:
       - tweets (twitter, x): peel marketing framing
       - papers (arxiv, scholar, dblp, openreview, semantic
         scholar, paperswithcode): distil abstract into
         pedagogical summary
    HuggingFace cards used to be in scope; dropped to keep cost
    down — they're mostly skeletal anyway.

  * `d.cleaned = FALSE`
    Idempotence; resets only when an operator explicitly flips the
    flag (e.g. after a prompt change).

Routing by source (post-selection):
  * Academic papers → keep title verbatim, rewrite summary only.
    The paper's own title is the canonical reference; rewriting it
    would defeat the citation surface.
  * Tweets → rewrite both title and summary.

CPU footprint: the work is I/O-bound on the OpenAI API. We sleep
`CLEAN_SLEEP_S` (default 1.5 s) between docs so wall-clock CPU
stays well under the 20 % budget on the production box.

Usage:

  python -m sources.utils.clean_daemon              # run forever
  python -m sources.utils.clean_daemon --preview 5  # print 5
                                                     # cleaned docs
                                                     # without
                                                     # writing back

Environment variables:

  DATABASE_URL          required, Postgres DSN
  OPENAI_API_KEY        required
  OPENAI_CLEAN_MODEL    default "gpt-4o-mini"
  CLEAN_SLEEP_S         default 1.5  (inter-doc pause)
  CLEAN_IDLE_SLEEP_S    default 600  (sleep when no docs left)
  CLEAN_WINDOW_DAYS     default 21   (matches the feed's effective
                                      recency horizon — 3 weeks)
  CLEAN_BATCH_SIZE      default 10   (rows pulled per loop)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

import psycopg

# Regex matching a URL we want to keep visible to the reader. Same
# pattern used in the SQL back-fill on the documents.urls column.
_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)

# Twitter / X attachment URLs the pipeline glues into the raw text
# as media markers. These are NOT user-facing content URLs and the
# preview-rendering code never wants to surface them as plain
# anchors — they're the inlined "📷 https://pbs.twimg.com/..." or
# "🎬 https://video.twimg.com/..." prefixes from the tweet
# scraper. Filter them out at extraction time.
_MEDIA_URL_HOSTS = (
    "pbs.twimg.com",
    "video.twimg.com",
    "ton.twimg.com",
)


def _extract_urls(text: str, linked: list | None) -> list[str]:
    """Return the deduped, order-preserving list of every URL the
    user-facing post referenced. Pulls from the raw text via regex
    and unions with `linked_urls[].url` (the OG-cluster). Drops
    Twitter media-attachment URLs so the result is content links
    only — the images themselves render through `linked_urls` /
    `renderTweetSummary` already."""
    seen: set[str] = set()
    out: list[str] = []
    for u in _URL_RE.findall(text or ""):
        # Peel trailing punctuation (sentence end, paren).
        while u and u[-1] in ".,;:!?)":
            u = u[:-1]
        if not u or u in seen:
            continue
        if any(h in u for h in _MEDIA_URL_HOSTS):
            continue
        seen.add(u)
        out.append(u)
    if isinstance(linked, list):
        for entry in linked:
            if not isinstance(entry, dict):
                continue
            u = (entry.get("url") or "").strip()
            if not u or u in seen:
                continue
            if any(h in u for h in _MEDIA_URL_HOSTS):
                continue
            seen.add(u)
            out.append(u)
    return out


# Belt-and-braces emoji strip for the cleaned title. The prompt
# already forbids emojis but gpt-4o-mini occasionally lets one
# slip through. This regex covers the BMP emoji ranges that show
# up in tweets (faces, hands, hearts, decorative symbols, flags,
# rockets, etc.). Applied to clean_title only — summaries keep
# whatever escaped the prompt's filter, since they go through the
# light-edit path which the model is more careful about.
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001ffff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]",
    flags=re.UNICODE,
)


def _strip_emoji(s: str) -> str:
    return _EMOJI_RE.sub("", s).strip()


# ── Configuration ───────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_CLEAN_MODEL", "gpt-4o-mini")

INTER_DOC_SLEEP_S = float(os.environ.get("CLEAN_SLEEP_S", "1.5"))
IDLE_SLEEP_S = float(os.environ.get("CLEAN_IDLE_SLEEP_S", "600"))
# 21 days = the feed's effective recency horizon. The feed scores
# bottom out at 5 weeks but the docs that actually surface to most
# viewers cluster in the last 2-3 weeks, so cleaning more than that
# burns tokens on docs no one will see.
WINDOW_DAYS = int(os.environ.get("CLEAN_WINDOW_DAYS", "21"))
# Small batch so memory stays low and the loop iterates quickly
# enough to see fresh inserts.
BATCH_SIZE = int(os.environ.get("CLEAN_BATCH_SIZE", "10"))

# Source whitelist (lowercased). Two modes handled by the prompt:
#   - Tweets (twitter, x): light-edit. Preserve the author's words,
#     fix typos, expand casual abbreviations, reformat any Quoting
#     block, drop emojis/media URLs, append a one-sentence context
#     paragraph for technical posts.
#   - Academic papers (arxiv, scholar, dblp, openreview,
#     semanticscholar, paperswithcode): KEEP the title verbatim and
#     distil the abstract into a clear pedagogical summary
#     organised around problem / method / result / takeaway.
#
# HuggingFace cards (huggingface, hf) used to be rewritten too —
# dropped to cut OpenAI cost. Most HF cards are skeletal and the
# rewrite barely changed them.
ACADEMIC_SOURCES = {
    "arxiv",
    "scholar",
    "dblp",
    "openreview",
    "semanticscholar",
    "semantic_scholar",
    "paperswithcode",
}
REWRITE_SOURCES = {
    "twitter",
    "x",
}
ALL_SOURCES = sorted(ACADEMIC_SOURCES | REWRITE_SOURCES)


# ── Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You produce a clean, pedagogical version of a
document's title and summary. Your behaviour depends on the source.

THREE MODES
1. Source is 'arxiv', 'scholar', 'dblp', 'openreview',
   'semanticscholar', or 'paperswithcode' — ACADEMIC PAPER MODE.
   Keep the title verbatim. Set clean_title to the raw title
   exactly, character for character. Treat the raw summary as the
   paper's abstract and distil it (see ACADEMIC SUMMARY below).

2. Source is 'twitter' or 'x' — TWEET MODE.
   Light-edit the body to make it comfortable to read. Preserve
   the author's words and ideas. Apply the rules under TWEET /
   HUGGINGFACE EDITING below.

3. Source is 'huggingface' or 'hf' — HUGGINGFACE MODE.
   Same light-edit rules as TWEET MODE. Most HF cards are
   skeletal — when there is no substance to rewrite, return an
   empty clean_summary.

------------------------------------------------------------------
ACADEMIC SUMMARY (papers)

You read the paper's abstract and write a clear, pedagogical
summary that a curious non-expert can follow. The summary is the
content, not the description of the content — talk about the
problem and the result, not about the paper.

Structure the summary around four key elements, in this order, as
flowing prose (no labels, no bullet list):

  1. Problem — what question or limitation does this paper take
     on. What was hard or unresolved before this work.
  2. Method — what does the paper actually do. The architecture,
     the loss, the dataset, the trick. Name technical concepts
     directly; the reader can look up what they don't know.
  3. Result — what does the paper find. The metric and the
     comparison if the abstract gives them. The number matters;
     do not strip it.
  4. Takeaway — one sentence on the practical implication or
     what the result enables.

Length: as long as needed to be informative, no shorter. Do not
inflate the length with hedging; do not trim if the abstract is
rich. Two or three short paragraphs is typical, separated by a
blank line. Plain present tense. No first person.

Avoid AI cliches: 'leverages', 'delves into', 'groundbreaking',
'robust', 'cutting-edge', 'comprehensive', 'in essence',
'underscores', 'pivotal', 'crucial', 'seamlessly', 'empowers',
'streamlines', 'it is worth noting', 'in today's world',
'navigating the complexities of', 'at its core', 'stands out',
'harnesses the power of'.

Never use these meta-frames in academic mode:
  - 'This paper presents / introduces / proposes / explores ...'
  - 'The authors show / argue / demonstrate ...'
  - 'The work focuses on ...'
  - 'In this paper, we ...'
Talk about the thing itself. Replace 'The authors propose X' with
'X is ...', 'The paper shows Y' with 'Y'.

If the raw abstract is empty, missing, or just a placeholder,
return an empty clean_summary. Never invent results, numbers, or
methods. Examples of "not a real abstract":
  - "Abstract page for arXiv paper 2605.05701: <paper title>"
  - A bibliographic reference: "[References — Transformer (deep
    learning architecture)] Author, X.; Author, Y. (2024). Paper
    Title. Proceedings of ..., arXiv: 1234.5678, doi: 10.../...,
    ISBN: ..." — this is a citation block scraped from Wikipedia,
    not the paper's abstract.
  - Any text that names the paper, its authors, its venue, and its
    DOI without describing the work itself.

If you can name only the title and not the actual contribution,
the correct answer is an empty summary. Do not summarise a paper
you have not been given the abstract of.

------------------------------------------------------------------
TWEET / HUGGINGFACE EDITING

You lightly edit the body so it reads as comfortable prose. The
author's words and ideas are preserved. You do not paraphrase, you
do not summarise. Most of the words in the output appear in the
input.

What you DO:
  - Remove emojis and decorative symbols.
  - Remove media-attachment lines: 'PHOTO https://...',
    'VIDEO https://...', and bare URLs to twimg / pbs / video.twimg.
  - Fix typos and obvious spelling mistakes.
  - Fix capitalisation at sentence starts and after periods.
  - Add missing punctuation (periods at end of sentences, commas
    where the sentence demands one).
  - Add a paragraph break where the author already broke the line
    AND the new line starts a new idea.
  - Expand non-technical abbreviations only when it improves
    readability: 'rn' to 'right now', 'tbh' to 'to be honest', 'idk'
    to 'I don\\'t know', 'imo' to 'in my opinion'.
  - Keep technical abbreviations and proper nouns verbatim: LLM,
    RAG, RLHF, transformer, JEPA, ColBERT, GGUF, MoE, KV cache,
    @handle, repository names, paper titles, model names.

URL HANDLING
The frontend renders the document with a separate preview-card
panel for any URL the post linked to, fed from `linked_urls`. The
clean_summary itself should NOT try to mention those URLs. Strict
rules:

  - NEVER write Markdown link syntax. No `[label](url)`. No
    `[text][ref]`. No reference-style links. The frontend escapes
    HTML and renders the cleaned text inside a `<p>` with
    `white-space: pre-line`, so any Markdown shows up as literal
    bracket-paren noise.
  - NEVER invent a URL. Do not synthesise an arXiv id, a
    GeoCodeBench host, or a github path. If the URL is not
    visible in the raw input, do not add one.
  - Plain bare URLs visible in the raw text are fine — leave
    them as-is, character for character. The frontend turns them
    into clickable links via the renderer's existing URL detector.
  - When the raw text has a trailing label without content
    (`Paper:`, `Project:`, `Code:`), drop the label entirely.
    The URL itself is already in `linked_urls` and will render
    as a tile below the card. Repeating "Paper:" with no body
    just leaves an empty hanging label.
  - When the raw mentions a paper or project by NAME (without
    URL), keep the name as plain text. No brackets, no parens.

What you DO NOT do:
  - You do not paraphrase. You do not summarise.
  - You do not change the meaning. You do not add information that
    is not in the source.
  - You do not invent context.
  - You do not add transitions like 'furthermore', 'moreover',
    'in addition' if the author did not use them.

QUOTING STRUCTURE
Tweets often contain a 'Quoting @handle' marker followed (sometimes)
by the quoted tweet's text. Reformat this as a separated quote at
the end:

  [editor's note: the cleaned main text comes first]

  [empty line]

  @handle: "[the cleaned quoted text]"

Rules for the quote:
  - The @handle keeps its '@' prefix.
  - The quoted text goes inside straight double quotes.
  - Apply the same cleaning rules to the quoted text (emojis off,
    typos fixed, etc).
  - If 'Quoting @handle' has NO quoted text after it (the marker
    sits at the end of the tweet with nothing following), DROP the
    line entirely. It is just attribution metadata.
  - If there are multiple quotes (rare), separate each with a blank
    line, each in the '@handle: "..."' shape.

CONTEXT PARAGRAPH (almost always add for technical posts)
After the cleaned body (and any quoted block), add ONE sentence in
a new paragraph that gives the curious non-expert reader the
background they need to understand what is going on. Default to
adding it. The aim is to teach.

The context sentence should explain a BACKGROUND CONCEPT or stake
that the tweet assumes the reader already knows. It should NOT
paraphrase the body. Useful angles:
  - What is the underlying technique (RAG, JEPA, MoE, RL-from-AI-
    feedback) and what does it do.
  - What problem does it address.
  - Why does the result matter; what was the previous state of the
    art or the natural baseline.
  - What the linked paper claims (use the linked_urls block when it
    is present).

Add the context paragraph in all of these cases:
  - Any tweet about a method, paper, model, benchmark, dataset,
    result, tool, infrastructure detail, architectural choice, or
    technical critique.
  - Any tweet that mentions a named system, model family, or
    benchmark a beginner might not know (e.g. Composer, Manificus
    Humanitas, Laguna, NanoGPT-Bench, JEPA, FlashAttention,
    Qwen3.6, etc).
  - Any tweet that references a paper through linked_urls.

DO NOT add the context paragraph when:
  - The tweet is pure mood content with no technical anchor:
    travel notes ("on our way to ..."), event reminders ("see you
    tomorrow"), personal anecdotes about non-technical life, jokes,
    or emotional reactions detached from any concept.
  - The cleaned body is just two or three words and there is
    nothing to anchor a context sentence to.
  - You genuinely cannot say something true without inventing
    facts. Silence is better than slop.

When a paper, model, or repository is referenced AND the user
message includes a 'linked_urls' block describing it (title +
summary), USE THAT METADATA to ground your context sentence. Quote
or paraphrase the linked summary's key claim. Do not invent paper
titles, author names, or numerical results that are not present in
the linked_urls block or the tweet itself.

Format: a blank line, then one sentence, period at the end. No
'Context:' label, no 'TL;DR:', no 'In other words:'. Just the
sentence, written as a calm informational aside.

BAD context paragraphs (do NOT do this)
  - "The linked article discusses the impact of these changes."
    -> says nothing. If you don't have a concrete claim, skip it.
  - "This post highlights an important issue."
    -> meta-frame slop. The whole point of context is to add
       NEW information, not to describe the tweet.
  - "It is important to understand the implications of AI."
    -> empty platitude. Skip.
  - "The author is making a point about X."
    -> redundant with the body. Skip.

Good context paragraphs share these traits:
  - They name a concrete concept, system, or fact the reader
    might not know.
  - They could appear on their own as a Wikipedia-style aside,
    independent of this specific tweet.
  - They do not start with 'The author', 'The post', 'The
    article', 'This work', 'The paper introduces' (unless paired
    with the actual paper's claim from linked_urls).

TITLE (tweets / huggingface only — academic keeps raw title)
The clean_title is an INFORMATIVE headline for the post. The
goal is to tell the reader what the post is about in one
glance, not to make them want to click. Read the whole body
(and the quoted tweet if present), identify the most
substantive point, and write a short factual title that names
it directly.

Concretely:
  - The title states the thing itself. If the post is about a
    new model, name the model. If it is about a benchmark
    result, name the benchmark and the result. If it is an
    opinion, name what the opinion is about.
  - Short and dense — typically 8 to 14 words, around 50 to 80
    characters.
  - Sentence case, not Title Case. No emojis, no hashtags, no
    @handles (yours or anyone else's), no exclamation marks.
    Question marks only when the post itself is genuinely asking
    a question whose answer is in the body.
  - The author's exact words are not sacred at the title level —
    paraphrase if a tighter wording captures the point.

AVOID CLICKBAIT
The headline is informative, not enticing. Banned patterns:
  - Curiosity gaps that withhold the point. NO "Why X did Y"
    titles unless the answer is right there in the title. NO
    "Here's what happened when ..."
  - Suspense framing. NO "You won't believe ...", NO "The
    surprising reason ..."
  - Vague superlatives. NO "The best ...", NO "The most powerful
    ...", NO "A game-changer", NO "A breakthrough".
  - Hype adjectives. NO "fascinating", "stunning", "shocking",
    "remarkable", "incredible", "wild", "insane".
  - Promotional verbs. NO "unleashes", "revolutionizes",
    "shatters", "destroys", "obliterates", "redefines".
  - Boilerplate openers. NO "Introducing ...", NO "Excited to
    share ...", NO "Today we ship ...", NO "Just dropped ...".
    Just state the thing.
  - AI / marketing cliches. NO "leverages", "delves into",
    "groundbreaking", "cutting-edge", "robust", "seamlessly",
    "harnesses".

If you would not put the headline on the front of a research
notes page or in a textbook's references, it is too clickbaity.

Examples of the informative style (paired with their input bodies):

  body: "Really clean approach. Do cross entropy loss on the
        environment feedback. This allows the model to get
        supervision even on failed rollouts..."
  → "Cross-entropy on environment feedback as RL supervision signal"

  body: "Composer 2.5 is very good. It's good at doing more than
        just quick iterations of front-end now. I will probably
        use it over Claude in Cursor"
  → "Author switches from Claude to Composer 2.5 inside Cursor"

  body: "#CVPR2026 Can frontier LLMs write PhD-level 3D vision
        code? We introduce GeoCodeBench... Best result so far:
        GPT-5 reaches only 36.6%."
  → "GeoCodeBench: GPT-5 reaches 36.6% on 3D geometric vision coding"

  body: "On our way to I/O 2026. See you at 10am PT tomorrow!"
  → "Author attending Google I/O 2026"

  body: "Aurora farming" (+ quote with no body)
  → "Aurora farming"

If the body has no real content (just a media link, just a couple
of words with a quote attached), the clean_title is a brief
literal description of what the post is and clean_summary stays
empty.

HUGGINGFACE CARDS
Apply the same conservative cleaning. Many HF cards are skeletal
("Model by X, derived from Y. Recommended way to run this model:")
— in those cases clean_summary stays empty.

OUTPUT FORMAT
Return strict JSON with exactly two keys:
  {"clean_title": "...", "clean_summary": "..."}
Newlines inside clean_summary are encoded as the JSON escape '\\n'.

LANGUAGE
Preserve the source language. English in, English out. French in,
French out.

EMPTY CASE — IMPORTANT
If the input is too thin to summarise honestly, return an empty
clean_summary ("") and a minimal clean_title. Examples of "too
thin":
  - A bare URL.
  - A one-line model card that only says "Model by X, derived
    from Y. Recommended way to run this model:" without any
    description of what the model does.
  - A HuggingFace space description that only says "Space by X,
    license: mit. Check out the configuration reference at".
  - A tweet text that is just an emoji and a link.

NEVER pad a thin input with generic claims like "designed for
specific applications", "improves performance on its target task",
"focuses on AI techniques". If the raw says nothing concrete,
clean_summary stays empty. Empty is better than slop.

REFERENCE EXAMPLES — these illustrate the two modes. Academic
papers get a pedagogical distillation organised around problem /
method / result / takeaway. Tweets get a light edit that preserves
the author's words.

# Academic paper — title kept verbatim, summary is a distilled
# pedagogical version of the abstract. Notice the structure:
# first paragraph names the problem and the method, second
# paragraph reports the result and what it enables. No meta-
# framing ('the paper introduces ...'), no AI clichés.
INPUT
source: arxiv
title: Scaling Laws for Mixture Pretraining Under Data Constraints
summary: Modern large-scale language model pretraining is increasingly bottlenecked by data rather than compute: the unique tokens available for training are finite, and repeating tokens degrades quality. We study how to allocate a fixed token budget across a mixture of data sources when more tokens cannot be obtained. We derive scaling laws describing how the validation loss responds to the relative weighting of each source and show that the optimal mixture shifts predictably with model size. Experiments on dense and mixture-of-experts models at scales up to 8B parameters confirm the predictions and yield an order-of-magnitude reduction in the number of ablations needed to set mixture weights.
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Scaling Laws for Mixture Pretraining Under Data Constraints",
  "clean_summary": "Large language model pretraining is increasingly bottlenecked by data rather than compute. The unique tokens available are finite, and repeating tokens degrades quality, so the question is how to allocate a fixed token budget across a mixture of data sources.\\n\\nThe work derives scaling laws describing how validation loss responds to the relative weighting of each source. The optimal mixture shifts predictably with model size, and experiments on dense and MoE models up to 8B parameters confirm the predictions while cutting the number of ablations needed to set mixture weights by an order of magnitude. The practical upshot is a way to pick mixture weights from first principles instead of grid search."
}

# Twitter — multi-paragraph thread with a trailing 'Quoting' tag
# that has no quoted text. Drop the empty quote line entirely.
INPUT
source: twitter
title: Dimitris Papailiopoulos (@DimitrisPapail)
summary: nice work by @DimitrisPapail and @VaishShrivas!
this work is reinforcing a recent trend that tries to make foundation models jointly predict future states (aka 'world models') and actions instead of actions alone.
we're seeing it in different forms, like World Action Models in embodied agents, or implicit world modeling in Early Experience ( also some interesting link to on-policy self-distillation.
shared learning here is, there's still rich supervision signals that are underexplored. such signals were hard to exploit in classic ML, but foundation models have made it possible, potentially creating a recursive self-improvement loop.
📷 https://pbs.twimg.com/media/HIpVYh0awAAC-KX.jpg
Quoting @DimitrisPapail
GOOD OUTPUT
{
  "clean_title": "Foundation models jointly predicting future states and actions, not just actions",
  "clean_summary": "Nice work by @DimitrisPapail and @VaishShrivas. This work is reinforcing a recent trend that tries to make foundation models jointly predict future states (aka 'world models') and actions, instead of actions alone.\\n\\nWe're seeing it in different forms, like World Action Models in embodied agents, or implicit world modeling in Early Experience, and also some interesting link to on-policy self-distillation.\\n\\nThe shared learning here is that there are still rich supervision signals that are underexplored. Such signals were hard to exploit in classic ML, but foundation models have made it possible, potentially creating a recursive self-improvement loop."
}

# Twitter — short technical take with a trailing 'Quoting' that has
# no body. Drop the line. CONTEXT PARAGRAPH ADDED because the body
# assumes the reader knows what RL rollouts are.
INPUT
source: twitter
title: Cody Blakeney (@code_star)
summary: Really clean approach.
Do cross entropy loss on the environment feedback. This allows the model to get supervision even on failed rollouts and helps form a sort of pseudo world model!
📷 https://pbs.twimg.com/media/HIpeBb6bIAA-Wv8.jpg
Quoting @DimitrisPapail
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Cross entropy loss on environment feedback as supervision for failed rollouts",
  "clean_summary": "Really clean approach. Do cross entropy loss on the environment feedback. This allows the model to get supervision even on failed rollouts and helps form a sort of pseudo world model.\\n\\nIn reinforcement learning, a rollout is one trajectory of the agent acting in its environment; usually only successful rollouts that reach the reward provide a clear learning signal, so getting supervision from failed ones too is a way to use data that would otherwise be wasted."
}

# Twitter — mood content with no technical anchor. NO context paragraph.
INPUT
source: twitter
title: Cody Blakeney (@code_star)
summary: I found out there is a library hotel in Tokyo. I'm thinking of booking it.
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Considering booking a library hotel in Tokyo",
  "clean_summary": "I found out there is a library hotel in Tokyo. I'm thinking of booking it."
}

# Twitter — references a paper via linked_urls. CONTEXT PARAGRAPH
# uses the linked abstract to ground the explanation.
INPUT
source: twitter
title: Some account (@whoever)
summary: Really interesting result from @bclavie's new paper. Looks like a ColBERT-style late interaction model can match dense retrieval at a fraction of the index size when paired with a proper compression scheme.
linked_urls:
- host=arxiv.org; title=Reason-ModernColBERT: a late-interaction model with learned token compression; summary=We present Reason-ModernColBERT, a 149M-parameter late-interaction retriever trained with a learned compression head over token embeddings. On BrowseComp-Plus the model matches dense retrievers 50x larger while reducing the index by an order of magnitude.
GOOD OUTPUT
{
  "clean_title": "ColBERT-style late interaction matching dense retrieval with a smaller index",
  "clean_summary": "Really interesting result from @bclavie's new paper. Looks like a ColBERT-style late interaction model can match dense retrieval at a fraction of the index size when paired with a proper compression scheme.\\n\\nThe paper introduces Reason-ModernColBERT, a 149M-parameter late-interaction retriever that uses a learned compression head over token embeddings to match dense retrievers fifty times larger on BrowseComp-Plus while shrinking the index by an order of magnitude."
}

# Twitter — quote tweet WITH quoted body. Format the quote as
# '@handle: "..."' at the end.
INPUT
source: twitter
title: Some account (@whoever)
summary: A fascinating reality check for AI coding agents. The new NanoGPT-Bench reveals that current agents (e.g., Claude Code and Codex) only recover 9.3% of human progress on AI R&amp;D tasks.
Quoting @IntologyAI
Can coding agents do research?
We release NanoGPT-Bench, an internal eval we’ve used to test agents on an AI R&D problem with months of human progress
Codex, Claude Code, Autoresearch recover only 9.3% of human progress, mostly tuning hyperparams & ignoring algorithmic research
📷 https://pbs.twimg.com/media/HIsVXgCaQAAYkZc.jpg
GOOD OUTPUT
{
  "clean_title": "A reality check for AI coding agents on NanoGPT-Bench",
  "clean_summary": "A fascinating reality check for AI coding agents. The new NanoGPT-Bench reveals that current agents (e.g. Claude Code and Codex) only recover 9.3% of human progress on AI R&D tasks.\\n\\n@IntologyAI: \\"Can coding agents do research? We release NanoGPT-Bench, an internal eval we've used to test agents on an AI R&D problem with months of human progress. Codex, Claude Code, and Autoresearch recover only 9.3% of human progress, mostly tuning hyperparams and ignoring algorithmic research.\\""
}

# Twitter — body is just two words plus a 'Quoting' marker with no
# quoted body. Title carries the literal text, summary stays empty.
INPUT
source: twitter
title: Cody Blakeney (@code_star)
summary: Aurora farming
Quoting @PrimeIntellect
📷 https://pbs.twimg.com/media/HInECrlWsAE-x1v.jpg
GOOD OUTPUT
{
  "clean_title": "Aurora farming",
  "clean_summary": ""
}

# Twitter — main body is ONLY a media attachment (no real text);
# the quoted tweet carries all the substance. In that case the
# cleaned summary is just the quote, on its own.
INPUT
source: twitter
title: Edward Grefenstette (@egrefen)
summary: 🎬 https://pbs.twimg.com/tweet_video_thumb/HIqz_uBWcAAXoTu.jpg | https://video.twimg.com/tweet_video/HIqz_uBWcAAXoTu.mp4
Quoting @pmddomingos
If the transformers paper was written by one of my students, I wouldn’t let him graduate until he did a better job.
GOOD OUTPUT
{
  "clean_title": "Pedro Domingos on the writing quality of the transformers paper",
  "clean_summary": "@pmddomingos: \\"If the transformers paper was written by one of my students, I wouldn't let him graduate until he did a better job.\\""
}

# Twitter — typo + abbreviation + emoji in a short personal take.
INPUT
source: twitter
title: Federico Cassano (@ellev3n11)
summary: Composer 2.5 is very good 🔥
It's good at doing more than just quick iterations of front-end now
I will probably use it over Claude in Cursor tbh
GOOD OUTPUT
{
  "clean_title": "Composer 2.5 is now useful beyond quick front-end iterations",
  "clean_summary": "Composer 2.5 is very good. It's good at doing more than just quick iterations of front-end now. I will probably use it over Claude in Cursor, to be honest."
}

# HuggingFace — skeletal model card. Empty summary is correct.
INPUT
source: huggingface
title: HuggingFace model: Qwen3.6-35B-A3B-MTP-GGUF
summary: Model by ggml-org, derived from Qwen/Qwen3.6-35B-A3B.
Recommended way to run this model:
GOOD OUTPUT
{
  "clean_title": "GGUF build of Qwen3.6-35B-A3B by ggml-org",
  "clean_summary": ""
}

# HuggingFace — informative space card. Light edit only.
INPUT
source: huggingface
title: HuggingFace space: carbon-demo
summary: Space by HuggingFaceBio. A streaming demo for the
`hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1` model. Enter a DNA
sequence prefix and watch the model continue it.
GOOD OUTPUT
{
  "clean_title": "Streaming demo for the carbon-3B DNA sequence model",
  "clean_summary": "A streaming demo for the hf-carbon/carbon-3B-hybrid-loss-1T-mix2-v1 model. Enter a DNA sequence prefix and watch the model continue it."
}

Notice in every example: the cleaned output preserves the author's
words. The only changes are removing emojis and media URLs,
capitalising sentence starts, fixing spelling, expanding casual
abbreviations, splitting paragraphs at natural boundaries, and
reformatting any 'Quoting @handle' tail into an explicit
'@handle: "..."' quote.
"""


USER_TEMPLATE = """source: {source}

title: {title}

summary: {summary}

linked_urls: {linked_urls}"""


def _format_linked_urls(linked_urls) -> str:
    """Compact, model-friendly serialisation of the linked_urls JSONB.

    Each entry is a dict with {url, host, title, summary, image}.
    Drop the image (we don't pass image data) and cap the summary at
    400 chars so a single tweet with five paper links doesn't blow
    the context. If empty, return 'none' so the prompt's template
    stays valid and the model knows there is no extra context.
    """
    if not isinstance(linked_urls, list) or not linked_urls:
        return "none"
    lines = []
    for entry in linked_urls[:5]:
        if not isinstance(entry, dict):
            continue
        host = (entry.get("host") or "").strip()
        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or "").strip()
        if len(summary) > 400:
            summary = summary[:400].rsplit(" ", 1)[0] + "..."
        line = f"- host={host}; title={title}"
        if summary:
            line += f"; summary={summary}"
        lines.append(line)
    return "\n".join(lines) if lines else "none"


# ── Database helpers ────────────────────────────────────────────────


def fetch_batch(conn: psycopg.Connection, limit: int) -> list[dict]:
    """Pull the next batch of unprocessed VIP docs.

    Scope = (feed-candidate set) ∩ (VIP) ∩ (tweet|paper) ∩ (last
    CLEAN_WINDOW_DAYS) ∩ (not yet cleaned). Mirrors the WHERE
    clause in `handlers::users::build_feed_payload`:

      - `d.deleted = FALSE`   — same as the feed
      - `d.date IS NOT NULL`  — same as the feed (also implied by
                                the `>=` cutoff below, but kept
                                explicit so the alignment with the
                                feed is obvious in the SQL)
      - `u.vip = TRUE`        — only personalities that get
                                surfaced in the feed; cleaning
                                non-VIP docs would burn tokens on
                                content the recency+sharer score
                                buries anyway.
      - `d.date >= now() - CLEAN_WINDOW_DAYS`
                              — the feed's recency bonus decays to
                                zero past ~5 weeks, so 3 weeks is
                                where the cost/benefit tips.
      - `lower(d.source) = ANY(ALL_SOURCES)`
                              — tweets + papers only. HF dropped.
      - `d.cleaned = FALSE`   — idempotence guard. Resets only on
                                operator intervention.
    """
    sql = """
        SELECT d.user_id, d.url, d.title, d.summary, d.source,
               d.linked_urls, d.urls
          FROM documents d
          JOIN users     u ON u.id = d.user_id
         WHERE u.vip = TRUE
           AND d.deleted = FALSE
           AND d.date IS NOT NULL
           AND d.date >= (now() - make_interval(days => %s))::date
           AND lower(d.source) = ANY(%s)
           AND d.cleaned = FALSE
         -- Newest first, unambiguously. Three levels of ordering,
         -- each one settling docs that tied on the previous:
         --   1. d.date DESC — the publication date carried by the
         --      doc (tweet date, paper publish date).
         --   2. d.created_at DESC — when we ingested the row, used
         --      to break ties within the same date.
         --   3. d.url DESC — within a single ingestion second
         --      (common during a sync), tweet URLs contain
         --      monotonically-increasing snowflake status ids, so
         --      url DESC orders newest-status-first. Works as a
         --      sensible final tiebreaker for arxiv too (higher
         --      arXiv numbers are newer papers).
         ORDER BY d.date DESC NULLS LAST,
                  d.created_at DESC NULLS LAST,
                  d.url DESC
         LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (WINDOW_DAYS, ALL_SOURCES, limit))
        rows = cur.fetchall()
    return [
        {
            "user_id": r[0],
            "url": r[1],
            "title": r[2] or "",
            "summary": r[3] or "",
            "source": (r[4] or "").lower(),
            "linked_urls": r[5] or [],
            "urls": list(r[6] or []),
        }
        for r in rows
    ]


def write_back(
    conn: psycopg.Connection,
    doc: dict,
    clean_title: str,
    clean_summary: str,
    urls: list[str],
) -> None:
    sql = """
        UPDATE documents
           SET clean_title   = %s,
               clean_summary = %s,
               -- Refresh the flat URL list at the same time so any
               -- URL the raw post referenced is recorded, even if
               -- the cleaned summary drops the label that wrapped
               -- it. Idempotent: re-running with the same input
               -- produces the same array.
               urls          = %s,
               cleaned       = TRUE,
               updated_at    = now()
         WHERE user_id = %s AND url = %s
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (clean_title, clean_summary, urls, doc["user_id"], doc["url"]),
        )
    conn.commit()


# ── OpenAI client ───────────────────────────────────────────────────


def call_openai(client, doc: dict) -> tuple[str, str]:
    user_msg = USER_TEMPLATE.format(
        source=doc["source"],
        title=doc["title"],
        summary=doc["summary"],
        linked_urls=_format_linked_urls(doc.get("linked_urls")),
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        # Low temp keeps the rewrite predictable across runs and
        # cuts the chance of cliche injection on the long tail.
        temperature=0.3,
    )
    payload = resp.choices[0].message.content or "{}"
    parsed = json.loads(payload)
    clean_title = (parsed.get("clean_title") or "").strip()
    clean_summary = (parsed.get("clean_summary") or "").strip()
    # Belt-and-braces emoji strip on the title. The prompt forbids
    # emojis but gpt-4o-mini occasionally lets one slip through;
    # `_strip_emoji` removes any U+1Fxxx / U+26xx / regional-
    # indicator codepoints from the cleaned title.
    clean_title = _strip_emoji(clean_title)
    # Academic papers: force clean_title to the raw title verbatim.
    # The paper's title is the canonical citation surface.
    if doc["source"] in ACADEMIC_SOURCES:
        clean_title = doc["title"]
    return clean_title, clean_summary


# ── Modes ───────────────────────────────────────────────────────────


def fetch_preview_mix(conn: psycopg.Connection, n: int) -> list[dict]:
    """Pull a balanced sample of docs across sources.

    `fetch_batch` orders by date desc which can return a long run of
    the same source (a user's recent HuggingFace dump, say). For the
    preview we want a mix so the prompt is exercised across tweets,
    HF, and academic papers in one shot.
    """
    sql = """
        SELECT user_id, url, title, summary, source, linked_urls FROM (
          SELECT
            d.user_id, d.url, d.title, d.summary, d.source, d.linked_urls,
            ROW_NUMBER() OVER (PARTITION BY d.source ORDER BY d.date DESC) AS rn
          FROM documents d
          JOIN users     u ON u.id = d.user_id
         WHERE u.vip = TRUE
           AND d.deleted = FALSE
           AND d.date IS NOT NULL
           AND d.date >= (now() - make_interval(days => %s))::date
           AND lower(d.source) = ANY(%s)
           AND d.cleaned = FALSE
        ) s
        WHERE rn <= %s
        ORDER BY source, rn
    """
    per_source = max(1, n // 3)
    with conn.cursor() as cur:
        cur.execute(sql, (WINDOW_DAYS, ALL_SOURCES, per_source))
        rows = cur.fetchall()
    return [
        {
            "user_id": r[0],
            "url": r[1],
            "title": r[2] or "",
            "summary": r[3] or "",
            "source": (r[4] or "").lower(),
            "linked_urls": r[5] or [],
        }
        for r in rows
    ][:n]


def preview(n: int) -> None:
    """Pull `n` docs and print before/after without writing back.

    Used to eyeball the prompt's output before flipping the daemon on
    in production.
    """
    if not DATABASE_URL:
        sys.exit("DATABASE_URL is required")
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is required")
    # Lazy import so an environment without the package can still
    # use the rest of the module.
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    with psycopg.connect(DATABASE_URL) as conn:
        docs = fetch_preview_mix(conn, n=n)
    if not docs:
        print("no candidate docs in the window")
        return
    for i, doc in enumerate(docs, 1):
        try:
            ct, cs = call_openai(client, doc)
        except Exception as e:
            print(f"[{i}] FAILED on {doc['url']}: {e}")
            continue
        print(f"\n{'=' * 78}")
        print(f"[{i}/{len(docs)}] source={doc['source']}")
        print(f"url:   {doc['url']}")
        print(f"\n--- RAW title ---\n{doc['title']}")
        print(f"\n--- RAW summary ---\n{doc['summary']}")
        print(f"\n--- CLEAN title ---\n{ct}")
        print(f"\n--- CLEAN summary ---\n{cs}")


def _is_out_of_credit(exc: Exception) -> bool:
    """Return True if `exc` looks like an OpenAI 'no credit / quota
    exhausted' response. OpenAI raises `RateLimitError` for BOTH
    transient rate limits and account-level quota exhaustion; the
    distinguishing signal is the inner `code` field
    ('insufficient_quota') or the message body. We match a few
    spellings defensively so a future SDK rename doesn't quietly
    stop us from sleeping."""
    msg = str(exc).lower()
    code = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            code = (err.get("code") or "").lower()
    if not code:
        code = (getattr(exc, "code", "") or "").lower()
    if code in {"insufficient_quota", "billing_hard_limit_reached"}:
        return True
    return "insufficient_quota" in msg or "exceeded your current quota" in msg or "billing" in msg and "quota" in msg


# How long to sleep when the API reports the account is out of
# credit. The clean daemon is non-essential, and burning the
# 10 %-CPU quota retrying every batch is wasteful — wait an hour
# so the operator has time to top up the balance.
OUT_OF_CREDIT_SLEEP_S = 3600.0


def run_forever() -> None:
    if not DATABASE_URL:
        sys.exit("DATABASE_URL is required")
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is required")
    from openai import OpenAI

    log = logging.getLogger("clean-daemon")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Yield CPU to the rest of the box. The work is I/O-bound on
    # OpenAI, so this is mostly belt-and-braces.
    try:
        os.nice(19)
    except OSError:
        pass

    client = OpenAI(api_key=OPENAI_API_KEY)
    log.info(
        "clean-daemon up: model=%s window=%dd batch=%d sleep=%.1fs",
        OPENAI_MODEL,
        WINDOW_DAYS,
        BATCH_SIZE,
        INTER_DOC_SLEEP_S,
    )

    while True:
        # Process one batch with the PG connection held open ONLY
        # for the duration of that batch. Sleeps happen outside the
        # `with` so we never sit on an idle-in-transaction backend
        # (the earlier design did, and a 1 h out-of-credit nap
        # locked the documents table for an hour at a time — see
        # `pg_stat_activity` postmortem).
        out_of_credit = False
        no_work = False
        try:
            with psycopg.connect(DATABASE_URL) as conn:
                batch = fetch_batch(conn, limit=BATCH_SIZE)
                if not batch:
                    no_work = True
                else:
                    log.info("processing %d docs", len(batch))
                    for doc in batch:
                        try:
                            ct, cs = call_openai(client, doc)
                        except Exception as e:
                            if _is_out_of_credit(e):
                                log.warning(
                                    "openai out of credit (%s) — sleeping %.0fs before retry",
                                    e,
                                    OUT_OF_CREDIT_SLEEP_S,
                                )
                                out_of_credit = True
                                break
                            # Transient OpenAI error — log and continue.
                            # The doc stays untouched and will be
                            # picked up again next loop.
                            log.warning(
                                "openai failed on %s: %s",
                                doc["url"][:80],
                                e,
                            )
                            time.sleep(min(30.0, INTER_DOC_SLEEP_S * 4))
                            continue
                        urls = _extract_urls(doc.get("summary", ""), doc.get("linked_urls"))
                        try:
                            write_back(conn, doc, ct, cs, urls)
                        except Exception as e:
                            log.exception(
                                "db write failed on %s: %s",
                                doc["url"][:80],
                                e,
                            )
                            continue
                        log.info(
                            "cleaned %s | %s",
                            doc["source"],
                            doc["url"][:80],
                        )
                        time.sleep(INTER_DOC_SLEEP_S)
        except Exception as e:
            log.exception("loop iteration failed, sleeping 30s: %s", e)
            time.sleep(30)
            continue
        # Connection released. Now it's safe to sleep for minutes /
        # hours without blocking other readers / writers of the
        # documents table.
        if out_of_credit:
            time.sleep(OUT_OF_CREDIT_SLEEP_S)
        elif no_work:
            log.info("no docs to clean, sleeping %.0fs", IDLE_SLEEP_S)
            time.sleep(IDLE_SLEEP_S)


def main() -> None:
    p = argparse.ArgumentParser(description="Pedagogical clean daemon")
    p.add_argument(
        "--preview",
        type=int,
        default=0,
        help="print N cleaned docs and exit (no DB write)",
    )
    args = p.parse_args()
    if args.preview > 0:
        preview(args.preview)
    else:
        run_forever()


if __name__ == "__main__":
    main()
