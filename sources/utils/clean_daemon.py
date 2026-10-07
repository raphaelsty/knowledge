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
       - tweets (twitter, x): explain the idea behind the post
       - papers (arxiv, scholar, dblp, openreview, semantic
         scholar, paperswithcode): turn the abstract into
         an explainer
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
  OPENAI_CLEAN_MODEL    default "gpt-4.1-mini"
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
OPENAI_MODEL = os.environ.get("OPENAI_CLEAN_MODEL", "gpt-4.1-mini")

INTER_DOC_SLEEP_S = float(os.environ.get("CLEAN_SLEEP_S", "1.5"))
IDLE_SLEEP_S = float(os.environ.get("CLEAN_IDLE_SLEEP_S", "600"))
# 21 days = the feed's effective recency horizon. The feed scores
# bottom out at 5 weeks but the docs that actually surface to most
# viewers cluster in the last 2-3 weeks, so cleaning more than that
# burns tokens on docs no one will see.
WINDOW_DAYS = int(os.environ.get("CLEAN_WINDOW_DAYS", "21"))
# Hard ceiling on the docs the daemon will ever rewrite per refresh
# cycle. We only clean docs that are in the top-N of
# `feed_snapshot.score` — the only docs the feed actually shows.
# Cleaning a rank-50 000 long-tail tweet burns OpenAI tokens on
# content that nobody will scroll to. 3000 covers the visible page
# (≤ 200 cards) + a generous pagination tail.
CLEAN_FEED_TOP_N = int(os.environ.get("CLEAN_FEED_TOP_N", "3000"))
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

SYSTEM_PROMPT = """You turn a saved document (a tweet, a thread, a model
card or a paper abstract) into a short explainer card that teaches a
curious reader something. Write the way The Rust Programming Language
book explains things: warm, direct and plain.

VOICE — RUST BOOK EXPLAINER
- Address the reader as "you", and use "we" when walking through an
  idea together.
- Explain why before how. Start from something the reader already
  knows (the problem, the cost, the usual way of doing it), then show
  the idea and what it buys you.
- Introduce one concept at a time. Define a term in passing the first
  time it appears ("a critic, a model that estimates how good a
  partial answer is, ...").
- Short, concrete sentences. Use the post's own example or number to
  make the idea tangible; keep the key number when there is one.
- Two short paragraphs, 90 to 150 words in total, separated by a
  blank line. The first sets up the problem; the second shows the idea
  and its result or consequence.
- When the post is someone's opinion or experience, attribute it to
  them by name ("Federico Cassano finds ...") rather than stating it
  as fact. Never write "the author", "this thread", "this paper",
  "the post".
- Get attribution right. The title names the post's author, and every
  part of a thread ("[2/3] ...") is theirs, even when it starts with
  @mentions (those are the people being replied to, not speakers).
  Only the text after "Quoting @handle" belongs to @handle.

GROUNDING — the most important rule
- Every fact about THIS work (what it does, numbers, names, results,
  comparisons, dates, availability) comes from the input: the title,
  the body, any quoted tweet, or the linked_urls metadata. Never add
  one.
- You may explain a general background concept the post relies on,
  but only with textbook-level facts that are true independently of
  this post. Never attach new numbers, names, benchmarks or claims to
  it. If you are not sure a definition is textbook-correct, leave it
  out.
- Never expand an acronym or name a method's full form unless the
  input spells it out (write "GRPO", not a guessed expansion).
- If the input is too thin to teach anything (a bare link, an emoji, a
  few words over a quote with no text, a skeletal model card, a
  placeholder abstract such as "Abstract page for arXiv paper ..." or
  a citation block), return an empty clean_summary. Empty is better
  than invented.
- A personal, non-technical post (travel, an event reminder, a joke)
  has nothing to teach: write one or two plain sentences that say
  what it is, in the same voice, without inventing context.

FORM
- No emojis, no hashtags, no Markdown, no bullet lists, no headings,
  no labels such as "TL;DR" or "Context:". @handles only when
  attributing a quoted tweet.
- No hype or AI cliches: groundbreaking, cutting-edge, leverages,
  delves, robust, seamless, game-changer, revolutionizes, pivotal,
  crucial, underscores, harnesses, unleashes, in essence, at its core.
- URLs: never invent one and never write Markdown links. The
  interface shows the post's linked pages as cards under the text, so
  don't repeat them; drop dangling labels such as "Paper:" or "Code:".
  Never mention media attachments (photo / video lines).
- Same language as the input. English in, English out; French in,
  French out.

TITLE
- Papers (source arxiv, scholar, dblp, openreview, semanticscholar,
  paperswithcode): clean_title is the raw title, verbatim.
- Everything else: an informative headline that names the thing
  itself (the method, the model, the result, the opinion's subject).
  6 to 14 words, sentence case. No emojis, hashtags, @handles or
  exclamation marks.
- No clickbait: no curiosity gaps ("Why X did Y", "Here's what ..."),
  no "Introducing", "Excited to share", "Just dropped", no hype
  adjectives (fascinating, stunning, remarkable, incredible, wild), no
  promotional verbs (unleashes, revolutionizes, redefines). If it
  wouldn't fit in a textbook's references, rewrite it.

OUTPUT
Strict JSON with exactly two keys:
  {"clean_title": "...", "clean_summary": "..."}
Paragraph breaks inside clean_summary are encoded as "\\n\\n".

REFERENCE EXAMPLES

INPUT
source: arxiv
title: Scaling Laws for Mixture Pretraining Under Data Constraints
summary: Modern large-scale language model pretraining is increasingly bottlenecked by data rather than compute: the unique tokens available for training are finite, and repeating tokens degrades quality. We study how to allocate a fixed token budget across a mixture of data sources when more tokens cannot be obtained. We derive scaling laws describing how the validation loss responds to the relative weighting of each source and show that the optimal mixture shifts predictably with model size. Experiments on dense and mixture-of-experts models at scales up to 8B parameters confirm the predictions and yield an order-of-magnitude reduction in the number of ablations needed to set mixture weights.
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Scaling Laws for Mixture Pretraining Under Data Constraints",
  "clean_summary": "When you pretrain a language model, you usually mix several data sources: web pages, code, books. Compute is no longer the main limit; data is. Each source has a finite number of unique tokens, and repeating them makes the model worse, so you have to decide how much of each source goes into a fixed budget.\\n\\nThe usual answer is a grid search over mixture weights, which costs many training runs. Here we get scaling laws that predict how validation loss responds to each source's weight, and the best mixture turns out to shift predictably with model size. On dense and mixture-of-experts models up to 8B parameters the predictions hold, cutting the ablations needed to set the weights by an order of magnitude."
}

INPUT
source: twitter
title: Some account (@whoever)
summary: Really interesting result from @bclavie's new paper. Looks like a ColBERT-style late interaction model can match dense retrieval at a fraction of the index size when paired with a proper compression scheme.
linked_urls:
- host=arxiv.org; title=Reason-ModernColBERT: a late-interaction model with learned token compression; summary=We present Reason-ModernColBERT, a 149M-parameter late-interaction retriever trained with a learned compression head over token embeddings. On BrowseComp-Plus the model matches dense retrievers 50x larger while reducing the index by an order of magnitude.
GOOD OUTPUT
{
  "clean_title": "Late-interaction retrieval matching dense models with a smaller index",
  "clean_summary": "A dense retriever squeezes each document into a single vector. A late-interaction model like ColBERT keeps one vector per token instead and compares query and document token by token, which captures more detail but makes the index much bigger.\\n\\nReason-ModernColBERT attacks that cost with a learned compression head over the token embeddings. At 149M parameters it matches dense retrievers 50 times larger on BrowseComp-Plus while shrinking the index by an order of magnitude, so you no longer have to trade accuracy for storage."
}

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
  "clean_title": "NanoGPT-Bench: coding agents recover 9.3% of human research progress",
  "clean_summary": "Coding agents are good at writing and fixing code, but can they do research, where progress comes from new ideas rather than tuning? To find out, you need a problem where humans already made months of measurable progress, and then you check how much of it an agent can reproduce.\\n\\nNanoGPT-Bench is such a test, released by @IntologyAI on an AI R&D problem. Codex, Claude Code and Autoresearch recover only 9.3% of the human progress, and they get there mostly by tuning hyperparameters while ignoring algorithmic ideas."
}

INPUT
source: twitter
title: Federico Cassano (@ellev3n11)
summary: Composer 2.5 is very good 🔥
It's good at doing more than just quick iterations of front-end now
I will probably use it over Claude in Cursor tbh
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Composer 2.5 is now useful beyond quick front-end iterations",
  "clean_summary": "Federico Cassano finds Composer 2.5 very good, and no longer only for quick front-end iterations. He expects to use it instead of Claude inside Cursor."
}

INPUT
source: twitter
title: Cody Blakeney (@code_star)
summary: Aurora farming
Quoting @PrimeIntellect
📷 https://pbs.twimg.com/media/HInECrlWsAE-x1v.jpg
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Aurora farming",
  "clean_summary": ""
}

INPUT
source: arxiv
title: Some Paper Title
summary: Abstract page for arXiv paper 2605.05701: Some Paper Title
linked_urls: none
GOOD OUTPUT
{
  "clean_title": "Some Paper Title",
  "clean_summary": ""
}
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
    """Pull the next batch of unprocessed VIP docs, prioritised by
    where they actually rank in the feed.

    Scope = (top-N `feed_snapshot` rows by score)
            ∩ (VIP) ∩ (tweet|paper) ∩ (last CLEAN_WINDOW_DAYS)
            ∩ (not yet cleaned).

    Why join `feed_snapshot`:
      The feed is the only surface where `clean_title`/`clean_summary`
      gets read. Cleaning a rank-50 000 doc that no one will ever
      scroll to wastes OpenAI tokens. So we read the same ranking
      the timeline reads from — `feed_snapshot.score DESC` — and
      cap the candidate set to the top `CLEAN_FEED_TOP_N` rows.
      Below that score band the daemon does nothing.

    Why VIP + source + window filters stay:
      `feed_snapshot` already only contains VIP-anchored rows
      from the 180-day window, so those filters are redundant
      strictly speaking. We keep them for belt-and-braces (if
      the snapshot is mid-refresh and missing rows, the daemon
      still won't pick up non-VIP / old / unsupported-source docs)
      and so the SQL still works on a fresh deploy before the
      snapshot has been built for the first time.

    URL-level dedup: same as before. A single arxiv paper saved
    by 10 VIPs is rewritten once; `write_back` propagates the
    result to every row sharing that URL.
    """
    sql = """
        WITH top_feed AS (
            -- The N highest-scored anchors in the snapshot —
            -- exactly the rows the timeline can surface for the
            -- visible page + a generous pagination tail.
            SELECT url, score
              FROM feed_snapshot
             ORDER BY score DESC
             LIMIT %s
        ),
        candidates AS (
            SELECT DISTINCT ON (d.url)
                   d.user_id, d.url, d.title, d.summary, d.source,
                   d.linked_urls, d.urls, d.date, d.created_at,
                   tf.score AS feed_score
              FROM documents d
              JOIN users     u  ON u.id  = d.user_id
              JOIN top_feed  tf ON tf.url = d.url
             WHERE u.vip = TRUE
               AND d.deleted = FALSE
               AND d.date IS NOT NULL
               AND d.date >= (now() - make_interval(days => %s))::date
               AND lower(d.source) = ANY(%s)
               AND d.cleaned = FALSE
             ORDER BY d.url, d.date DESC NULLS LAST,
                      d.created_at DESC NULLS LAST
        )
        SELECT user_id, url, title, summary, source, linked_urls, urls
          FROM candidates
         -- Process highest-priority docs first — exactly mirrors
         -- the order the feed will render them in. Date breaks
         -- score ties so two same-score docs come out
         -- newest-first.
         ORDER BY feed_score DESC,
                  date DESC NULLS LAST,
                  created_at DESC NULLS LAST,
                  url DESC
         LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (CLEAN_FEED_TOP_N, WINDOW_DAYS, ALL_SOURCES, limit))
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
    """Propagate the cleaned title/summary to every row sharing this URL.

    The selection step deduped by URL (matching the feed), so one
    OpenAI call covers a single logical document. The same URL can
    live in N personal libraries — write the cleaned values into
    every one of them so that:

      * the feed and the per-user personal pages stay in sync (the
        feed picks one row per URL, the personal pages pick the
        owner's row; they should display the same cleaned text),
      * the daemon's `cleaned = TRUE` guard fires for every copy,
        so the next loop's `fetch_batch` skips them.

    No `user_id` clause in the UPDATE — `WHERE url = %s` matches
    every owner. PG's (user_id, url) primary key still scopes each
    update to one row per user; we're just hitting all of them at
    once.
    """
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
         WHERE url = %s
           AND deleted = FALSE
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (clean_title, clean_summary, urls, doc["url"]),
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
