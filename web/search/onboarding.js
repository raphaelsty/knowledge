/* Follow-graph onboarding.
 *
 * Self-contained module that takes over the empty-state on a
 * signed-in user's own personal page when they have zero follows,
 * and also powers the "Discover Peoples" overlay.
 *
 * UX:
 *   - The picker is the first thing the user sees (no intro slide).
 *   - Two slides: pick (15 category cards) → affinity (current picks
 *     + recommendations). Step transitions animate the new slide in
 *     from the right.
 *   - Selection is per-category: tapping a card adds every member of
 *     that bucket to the follow list. Tapping again removes them.
 *     Each card shows the top-6 most-followed members so the user
 *     knows who they're signing up for; the rest are still followed.
 *
 * Public entry point:
 *
 *   window.KnowledgeOnboarding.open({
 *     personalities,   // array — typically state.allPersonalities
 *     apiBase,         // e.g. "http://localhost:8080" or "" for prod
 *     onSkip,          // () => void — caller handles the dismiss path
 *     host,            // optional element to render into. Defaults to
 *                      // `#empty`. The Discover overlay passes
 *                      // `#discoverBody`.
 *     mode,            // "onboard" | "discover" — only changes the
 *                      // commit-button label.
 *   })
 */
(function () {
  "use strict";

  // ── Ontology (mirrors sources/sql/categories.sql) ──────────────────
  //
  // Two kinds:
  //   - topic   → fine-grained research / role buckets (LLM Research,
  //               Computer Vision, AI Safety, …)
  //   - company → institutional affiliations (OpenAI, Anthropic, …).
  // The picker renders each kind as its own section so the 35-card
  // grid stays scannable.
  const LABELS = {
    "llm-research": "LLM Research",
    "nlp-retrieval": "NLP & Retrieval",
    "computer-vision": "Computer Vision",
    "generative-media": "Generative Media",
    multimodal: "Multimodal AI",
    "rl-robotics": "Reinforcement Learning & Robotics",
    "ai-safety": "AI Safety & Alignment",
    "ml-theory": "ML Theory & Foundations",
    "ml-infra": "ML Infra & Systems",
    "efficient-inference": "Efficient Inference",
    "oss-tools": "Open-Source ML Tools",
    founders: "AI Founders & Builders",
    "lab-leaders": "AI Lab Leadership",
    educators: "Educators & Bloggers",
    pioneers: "Pioneers & Laureates",
    // Companies & labs
    openai: "OpenAI",
    anthropic: "Anthropic",
    "google-deepmind": "Google DeepMind",
    "meta-ai": "Meta AI",
    "mistral-ai": "Mistral AI",
    "hugging-face": "Hugging Face",
    "nvidia-research": "NVIDIA Research",
    "microsoft-research": "Microsoft Research",
    "apple-ml": "Apple ML",
    xai: "xAI",
    cohere: "Cohere",
    "stability-ai": "Stability AI",
    perplexity: "Perplexity",
    "allen-ai": "Allen AI",
    "eleuther-ai": "EleutherAI",
    lighton: "LightOn",
    "stanford-ai": "Stanford AI",
    "berkeley-ai": "UC Berkeley AI",
    "mit-csail": "MIT CSAIL",
    "cmu-ai": "CMU AI",
  };
  const DESCRIPTIONS = {
    "llm-research": "Building or studying frontier language models.",
    "nlp-retrieval": "Tokenization, embeddings, RAG, search.",
    "computer-vision": "Image models, segmentation, vision-language.",
    "generative-media": "Diffusion, image/video/audio generation.",
    multimodal: "Vision-language, speech, cross-modal.",
    "rl-robotics": "RL theory plus embodied AI.",
    "ai-safety": "Interpretability, RLHF, alignment.",
    "ml-theory": "Optimization, scaling laws, learning theory.",
    "ml-infra": "Training stacks, distributed compute.",
    "efficient-inference": "Quantization, distillation, on-device.",
    "oss-tools": "Library and framework maintainers.",
    founders: "Startup CEOs and hands-on builders.",
    "lab-leaders": "Directors at OpenAI / DeepMind / Anthropic.",
    educators: "Course creators, explainers, writers.",
    pioneers: "Turing-award and field-defining figures.",
    // Companies & labs descriptions kept compact — the name says
    // most of what the reader needs to know.
    openai: "Researchers and engineers at OpenAI.",
    anthropic: "Researchers and engineers at Anthropic.",
    "google-deepmind": "Across DeepMind, Brain, and Google Research.",
    "meta-ai": "Researchers at Meta AI / FAIR.",
    "mistral-ai": "Mistral AI team.",
    "hugging-face": "Hugging Face engineers and researchers.",
    "nvidia-research": "NVIDIA research scientists.",
    "microsoft-research": "Researchers at MSR and Microsoft AI.",
    "apple-ml": "Machine learning teams at Apple.",
    xai: "xAI researchers and engineers.",
    cohere: "Cohere team.",
    "stability-ai": "Stability AI past and present researchers.",
    perplexity: "Perplexity engineers and researchers.",
    "allen-ai": "Allen Institute for AI researchers.",
    "eleuther-ai": "EleutherAI open research collective.",
    lighton: "LightOn researchers and engineers.",
    "stanford-ai": "Stanford AI faculty, CRFM, HAI.",
    "berkeley-ai": "UC Berkeley AI Research (BAIR).",
    "mit-csail": "MIT CSAIL and AI faculty.",
    "cmu-ai": "Carnegie Mellon University AI / ML faculty.",
  };
  const TOPIC_ORDER = [
    "llm-research",
    "nlp-retrieval",
    "computer-vision",
    "generative-media",
    "multimodal",
    "rl-robotics",
    "ai-safety",
    "ml-theory",
    "ml-infra",
    "efficient-inference",
    "oss-tools",
    "founders",
    "lab-leaders",
    "educators",
    "pioneers",
  ];
  const COMPANY_ORDER = [
    "openai",
    "anthropic",
    "google-deepmind",
    "meta-ai",
    "mistral-ai",
    "hugging-face",
    "nvidia-research",
    "microsoft-research",
    "apple-ml",
    "xai",
    "cohere",
    "stability-ai",
    "perplexity",
    "allen-ai",
    "eleuther-ai",
    "lighton",
    "stanford-ai",
    "berkeley-ai",
    "mit-csail",
    "cmu-ai",
  ];
  // Used by the affinity recommender and any code that scans the
  // full ontology — keeps that pass kind-agnostic.
  const ORDER = [...TOPIC_ORDER, ...COMPANY_ORDER];

  // ── Tuning knobs ───────────────────────────────────────────────────
  // Named people previewed inside each card. The rest of the bucket
  // is still followed when the card is selected — the preview is just
  // a "who's in this group" hint.
  const PREVIEW_PER_CARD = 6;

  // ── Tiny DOM helpers ───────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  // escapeHtml comes from /lib/utils.js
  const escapeAttr = escapeHtml;

  // ── Module state ───────────────────────────────────────────────────
  let host = null;
  let cfg = null;
  let catIndex = null; // slug → [vip, vip, …] sorted by reach desc
  let shell = null;
  const selected = new Set();
  let step = "pick"; // "pick" | "affinity"

  function buildCategoryIndex(personalities) {
    const byCat = new Map();
    for (const p of personalities || []) {
      if (!p || !p.vip) continue;
      for (const c of p.categories || []) {
        if (!byCat.has(c)) byCat.set(c, []);
        byCat.get(c).push(p);
      }
    }
    // Sort by social reach so recognisable names lead the preview.
    const reach = (p) =>
      (p.twitterFollowers || 0) * 1 + (p.githubFollowers || 0) * 0.1;
    for (const arr of byCat.values()) {
      arr.sort(
        (a, b) =>
          reach(b) - reach(a) ||
          (b.documentCount || 0) - (a.documentCount || 0),
      );
    }
    return byCat;
  }

  /* Distinct VIPs the user would follow if they committed right now.
   * Deduped across categories. */
  function selectedPeople() {
    const out = new Set();
    for (const cat of selected) {
      for (const p of catIndex.get(cat) || []) out.add(p.slug);
    }
    return out;
  }

  /* Recommend follow-up categories for the affinity step.
   *   - high: top-4 by VIP overlap with the picked set (size-
   *           normalised so a niche overlap of 3 beats a popular
   *           overlap of 5).
   *   - tail: bottom-2 by overlap (forced diversity). */
  function recommendNextCategories() {
    const pickedPeople = selectedPeople();
    const scored = [];
    for (const slug of ORDER) {
      if (selected.has(slug)) continue;
      const members = catIndex.get(slug) || [];
      if (!members.length) continue;
      let overlap = 0;
      for (const p of members) if (pickedPeople.has(p.slug)) overlap++;
      const score = overlap / Math.sqrt(members.length);
      scored.push({ slug, overlap, score, size: members.length });
    }
    scored.sort((a, b) => b.score - a.score || b.overlap - a.overlap);
    const high = scored.slice(0, 4).map((x) => x.slug);
    const tail = scored
      .slice()
      .reverse()
      .filter((x) => !high.includes(x.slug))
      .slice(0, 2)
      .map((x) => x.slug);
    return { high, tail };
  }

  // ── Card markup ────────────────────────────────────────────────────
  function previewPersonHtml(p) {
    const initial = (p.name || p.slug || "?").trim()[0]?.toUpperCase() || "?";
    const avatar = p.avatar
      ? `<img class="onb-mini-av" src="${escapeAttr(p.avatar)}" alt="" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'onb-mini-av onb-mini-av-fb',textContent:'${initial}'}))"/>`
      : `<span class="onb-mini-av onb-mini-av-fb">${escapeHtml(initial)}</span>`;
    return `<li><span class="onb-person-row">${avatar}<span class="onb-mini-name">${escapeHtml(p.name || p.slug)}</span></span></li>`;
  }

  function cardHtml(slug) {
    const members = catIndex.get(slug) || [];
    const preview = members.slice(0, PREVIEW_PER_CARD);
    const extra = Math.max(0, members.length - preview.length);
    const peopleHtml = preview.map(previewPersonHtml).join("");
    const isSelected = selected.has(slug);
    const moreHtml = extra > 0 ? `<p class="onb-more">+ ${extra} more</p>` : "";
    return `
      <button type="button"
              class="onb-card ${isSelected ? "is-selected" : ""}"
              data-onb-cat="${escapeAttr(slug)}"
              aria-pressed="${isSelected ? "true" : "false"}">
        <header class="onb-card-head">
          <h3>${escapeHtml(LABELS[slug] || slug)}</h3>
          <span class="onb-count">${members.length}</span>
        </header>
        <p class="onb-desc">${escapeHtml(DESCRIPTIONS[slug] || "")}</p>
        <ul class="onb-card-people">${peopleHtml}</ul>
        ${moreHtml}
        <span class="onb-check" aria-hidden="true">✓</span>
      </button>`;
  }

  // ── Render (once) ──────────────────────────────────────────────────
  function render() {
    if (!host) return;
    host.style.display = "";
    host.classList.add("follow-onboarding");

    // Render each kind as its own section so the picker has a clear
    // information hierarchy. Empty buckets (cards with 0 members)
    // are skipped — they'd read as confusing placeholders.
    const hasMembers = (slug) => (catIndex.get(slug) || []).length > 0;
    const topicCards = TOPIC_ORDER.filter(hasMembers).map(cardHtml).join("");
    const companyCards = COMPANY_ORDER.filter(hasMembers)
      .map(cardHtml)
      .join("");
    host.innerHTML = `
      <div class="onb-shell" data-step="${step}">
        <header class="onb-top">
          <div class="onb-progress" aria-hidden="true">
            <span class="onb-progress-fill"></span>
          </div>
        </header>

        <div class="onb-slides">

          <section class="onb-slide" data-slide="pick">
            <header class="onb-slide-head">
              <h2>Pick a few topics</h2>
              <p>We'll set up your feed with the people working on each.</p>
            </header>
            <div class="onb-grid">${topicCards}</div>
            ${
              companyCards
                ? `
            <header class="onb-section-head">
              <h3>Companies &amp; labs</h3>
              <p>Follow everyone at a specific organisation.</p>
            </header>
            <div class="onb-grid onb-grid--companies">${companyCards}</div>`
                : ""
            }
          </section>

          <section class="onb-slide" data-slide="affinity">
            <div data-affinity-host></div>
          </section>

        </div>

        <footer class="onb-foot">
          <span class="onb-counter">
            <strong data-counter-people>0</strong> people to follow
          </span>
          <div class="onb-foot-actions">
            <button type="button" class="onb-back" data-onb-back hidden>Back</button>
            <button type="button" class="onb-skip" data-onb-dismiss>Skip for now</button>
            <button type="button" class="onb-next" data-onb-next>Continue</button>
          </div>
        </footer>
      </div>`;

    shell = host.querySelector(".onb-shell");
    wire();
    syncFooter();
    syncProgress();
  }

  // ── Targeted DOM updates ───────────────────────────────────────────
  function syncFooter() {
    if (!shell) return;
    const peopleEl = shell.querySelector("[data-counter-people]");
    if (peopleEl) peopleEl.textContent = String(selectedPeople().size);

    const back = shell.querySelector("[data-onb-back]");
    const next = shell.querySelector("[data-onb-next]");
    const onAffinity = step === "affinity";

    if (back) back.hidden = !onAffinity;
    if (next) {
      next.disabled = selected.size === 0;
      next.textContent = onAffinity
        ? cfg.mode === "discover"
          ? "Follow these people"
          : "Finish · follow these people"
        : "Continue →";
    }
  }

  function syncProgress() {
    if (!shell) return;
    shell.setAttribute("data-step", step);
  }

  function toggleCard(btn) {
    const slug = btn.dataset.onbCat;
    const wasSelected = selected.has(slug);
    if (wasSelected) selected.delete(slug);
    else selected.add(slug);
    // Update every clone of this slug on the picker slide. Card
    // classes flip in-place — no full re-render needed.
    if (shell) {
      shell
        .querySelectorAll(
          `.onb-slide[data-slide="pick"] [data-onb-cat="${CSS.escape(slug)}"]`,
        )
        .forEach((node) => {
          node.classList.toggle("is-selected", !wasSelected);
          node.setAttribute("aria-pressed", !wasSelected ? "true" : "false");
        });
    }
    // On the affinity step the chip strip + recommendation cards
    // need to reflect the new state — including removing the chip
    // when a topic is deselected. Rebuild the slide so everything
    // (live count, chips, recommendations) re-renders consistently.
    if (step === "affinity") buildAffinity();
    syncFooter();
  }

  // ── Affinity slide (built on demand) ───────────────────────────────
  function pickedChipsHtml() {
    return [...selected]
      .map(
        (slug) =>
          `<button type="button"
                   class="onb-chip"
                   data-onb-cat="${escapeAttr(slug)}"
                   title="Remove ${escapeHtml(LABELS[slug] || slug)}">
             ${escapeHtml(LABELS[slug] || slug)}
             <span class="onb-chip-x" aria-hidden="true">×</span>
           </button>`,
      )
      .join("");
  }

  function buildAffinity() {
    if (!shell) return;
    const recHost = shell.querySelector("[data-affinity-host]");
    if (!recHost) return;
    const recs = recommendNextCategories();
    const total = selectedPeople().size;
    const headline = total
      ? `<strong>${total}</strong> ${total === 1 ? "person" : "people"} in your feed so far`
      : "Pick at least one more topic";

    const summaryHtml = `
      <header class="onb-aff-head">
        <h2>Round out your feed</h2>
        <p class="onb-aff-sub">${headline} · add a few more topics to broaden it.</p>
        ${
          selected.size
            ? `<div class="onb-chips">${pickedChipsHtml()}</div>`
            : ""
        }
      </header>`;

    const highHtml = recs.high.map(cardHtml).join("");
    const tailHtml = recs.tail.map(cardHtml).join("");

    recHost.innerHTML = `
      ${summaryHtml}
      ${
        highHtml
          ? `
        <section class="onb-aff-section">
          <header class="onb-aff-sec-head">
            <span class="onb-aff-sec-num">1</span>
            <div>
              <h3>More like the topics you picked</h3>
              <p>These groups share the most people with your selection.</p>
            </div>
          </header>
          <div class="onb-grid">${highHtml}</div>
        </section>`
          : ""
      }
      ${
        tailHtml
          ? `
        <section class="onb-aff-section">
          <header class="onb-aff-sec-head">
            <span class="onb-aff-sec-num">2</span>
            <div>
              <h3>Branch out</h3>
              <p>A few directions that barely overlap with your picks, in case you want variety.</p>
            </div>
          </header>
          <div class="onb-grid">${tailHtml}</div>
        </section>`
          : ""
      }`;

    wireCards(recHost);
  }

  // ── Step transitions ───────────────────────────────────────────────
  function goTo(nextStep) {
    if (nextStep === step) return;
    step = nextStep;
    if (step === "affinity") buildAffinity();
    if (step === "pick" && shell) {
      // Re-sync card states across the picker on the way back.
      shell
        .querySelectorAll('.onb-slide[data-slide="pick"] [data-onb-cat]')
        .forEach((btn) => {
          const sel = selected.has(btn.dataset.onbCat);
          btn.classList.toggle("is-selected", sel);
          btn.setAttribute("aria-pressed", sel ? "true" : "false");
        });
    }
    syncProgress();
    syncFooter();
    if (host && host.scrollTo) host.scrollTo({ top: 0, behavior: "smooth" });
    else window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function back() {
    if (step === "affinity") goTo("pick");
  }

  async function next() {
    if (selected.size === 0) return;
    if (step === "pick") return goTo("affinity");
    // Commit
    const slugs = [...selectedPeople()];
    const btn = shell?.querySelector("[data-onb-next]");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Setting up your feed…";
    }
    try {
      await fetch(`${cfg.apiBase}/api/me/follow/bulk`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slugs }),
      });
    } catch (e) {
      console.warn("[onboarding] bulk follow failed", e);
    }
    window.location.href = "/";
  }

  // ── Wiring ─────────────────────────────────────────────────────────
  function wireCards(scope) {
    scope.querySelectorAll("[data-onb-cat]").forEach((btn) => {
      btn.addEventListener("click", () => toggleCard(btn));
    });
  }
  function wire() {
    if (!shell) return;
    wireCards(shell);
    shell.querySelector("[data-onb-next]")?.addEventListener("click", next);
    shell.querySelector("[data-onb-back]")?.addEventListener("click", back);
    shell.querySelector("[data-onb-dismiss]")?.addEventListener("click", () => {
      selected.clear();
      step = "pick";
      shell = null;
      host.classList.remove("follow-onboarding");
      host.innerHTML = "";
      if (typeof cfg.onSkip === "function") cfg.onSkip();
    });
  }

  // ── Public API ─────────────────────────────────────────────────────
  function open(options) {
    cfg = Object.assign(
      {
        personalities: [],
        apiBase: "",
        onSkip: () => {},
        host: null,
        mode: "onboard",
      },
      options || {},
    );
    host = cfg.host || $("empty");
    if (!host) {
      console.warn("[onboarding] no host element");
      return;
    }
    catIndex = buildCategoryIndex(cfg.personalities);
    selected.clear();
    step = "pick";
    render();
  }

  window.KnowledgeOnboarding = { open };
})();
