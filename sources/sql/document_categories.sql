-- Fine-grained topical ontology applied to documents.
--
-- Distinct from `categories` (which classifies USERS for the
-- onboarding "what kinds of people to follow" picker) and from
-- `topics` (high-level interest filter the user picks once). This
-- table is the catalogue the clean daemon picks from when it
-- assigns a single category to each document, so the feed can be
-- filtered to a single fine-grained slice ("Quantization this
-- week", "Protein Design this week", …).
--
-- The clean daemon receives the full table (slug + name +
-- description) as part of its system prompt and is asked to emit
-- one slug. Descriptions therefore double as the prompt's
-- disambiguation hints — keep them tight and contrastive.

CREATE TABLE IF NOT EXISTS document_categories (
    id          BIGSERIAL   PRIMARY KEY,
    slug        TEXT        NOT NULL UNIQUE,
    name        TEXT        NOT NULL,
    -- Coarse grouping for UI navigation (sidebar headers,
    -- collapsible sections). NOT the assignment target — the
    -- daemon writes the leaf slug.
    group_name  TEXT        NOT NULL,
    description TEXT        NOT NULL DEFAULT '',
    sort_order  INTEGER     NOT NULL DEFAULT 1000,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_document_categories_group
    ON document_categories(group_name, sort_order);

COMMENT ON TABLE  document_categories IS 'Per-document topical ontology consumed by the clean daemon to assign one category per doc.';
COMMENT ON COLUMN document_categories.slug        IS 'Stable machine identifier (URL-safe kebab-case).';
COMMENT ON COLUMN document_categories.name        IS 'Human-friendly display label.';
COMMENT ON COLUMN document_categories.group_name  IS 'Coarse UI grouping; only used for navigation, not for assignment.';
COMMENT ON COLUMN document_categories.description IS 'One-line contrastive hint; doubles as disambiguation cue in the daemon prompt.';

-- ── Seed (idempotent: ON CONFLICT DO UPDATE) ─────────────────────────
INSERT INTO document_categories (slug, name, group_name, description, sort_order) VALUES
    -- Pretraining & Architecture
    ('pretraining-recipes',          'Pretraining Recipes',          'Pretraining & Architecture', 'Recipes for pretraining base language models from scratch — optimizer choice (AdamW, Lion, Shampoo, Muon), learning-rate schedules (cosine, WSD), warmup, batch-size schedules, gradient clipping, loss spikes.', 1010),
    ('scaling-laws',                 'Scaling Laws',                 'Pretraining & Architecture', 'Empirical scaling laws relating compute, parameter count, and data size to validation loss — Chinchilla, Hoffmann-optimal, isoFLOP curves, compute-optimal training. Specifically the laws and their fits.', 1020),
    ('data-mixtures',                'Data Mixtures for Pretraining','Pretraining & Architecture', 'Blending and weighting pretraining corpora — domain reweighting, data-mixture optimization (DoReMi, RegMix), web/code/math ratio ablations, mixture schedules over training.', 1030),
    ('tokenization',                 'Tokenization',                 'Pretraining & Architecture', 'Tokenizer design for language models — BPE, SentencePiece, Unigram, tiktoken, byte-level tokenizers, multilingual tokenizers, vocabulary size choice, tokenizer-free models like ByT5.', 1040),
    ('mixture-of-experts',           'Mixture of Experts',           'Pretraining & Architecture', 'Sparsely-activated mixture-of-experts language models — expert routing, top-k gating, load balancing, GShard, Switch, DeepSeek-MoE, Mixtral, expert specialization, fine-grained MoE.', 1050),
    ('long-context',                 'Long Context',                 'Pretraining & Architecture', 'Training language models to handle long context windows — RoPE scaling, YaRN, NTK-aware scaling, position interpolation, context-extension training, needle-in-haystack design. Training-side, not inference-side.', 1060),
    ('sparse-attention',             'Sparse Attention',             'Pretraining & Architecture', 'Attention patterns that subsample the quadratic interaction — local/sliding window attention, blockwise attention, BigBird, Longformer, native sparse attention, learned sparse patterns.', 1070),
    ('state-space-models',           'State-Space Models',           'Pretraining & Architecture', 'Sub-quadratic sequence models that replace attention with SSMs — Mamba, Mamba-2, S4, S5, RWKV, linear attention, RetNet, recurrent transformers, RWKV-v7.', 1080),
    ('hybrid-architectures',         'Hybrid Architectures',         'Pretraining & Architecture', 'Models that mix transformer attention with SSM, MoE, or convolutional layers — Jamba, Zamba, Samba, Griffin, hybrid attention/SSM stacks aiming for the best of both.', 1090),
    ('small-language-models',        'Small Language Models',        'Pretraining & Architecture', 'Compact language models designed to run on consumer hardware or edge devices — Phi, SmolLM, TinyLlama, MobileLLM, Qwen-0.5B, Gemma-2B, distilled small models, <3B parameter base models.', 1100),

    -- Post-Training
    ('instruction-tuning',           'Instruction Tuning',           'Post-Training', 'Supervised fine-tuning on instruction-following datasets — SFT, FLAN, Alpaca-style data, instruction-tuned variants of base models, multi-task instruction blends.', 1210),
    ('preference-optimization',      'Preference Optimization',      'Post-Training', 'Aligning LMs to human preferences with offline pair-loss objectives — DPO, IPO, KTO, SimPO, ORPO, NLHF. Preference-pair training without an explicit reward model.', 1220),
    ('rlhf-and-reward-modeling',     'RLHF and Reward Modeling',     'Post-Training', 'Classic RLHF post-training — PPO on LM policies, reward model training from human preferences, KL constraints, RLHF infrastructure. Specifically the on-policy RL post-training pipeline.', 1230),
    ('reasoning-rl',                 'Reasoning RL',                 'Post-Training', 'RL post-training with verifiable rewards on math and code — RLVR, GRPO, RLOO, REINFORCE++, DeepSeek-R1-Zero, o1-style reasoning RL, process reward models.', 1240),
    ('constitutional-ai',            'Constitutional AI & Self-Critique','Post-Training', 'Training LMs to follow written principles via self-critique — Constitutional AI (Anthropic), RLAIF, principle-guided self-improvement, AI-feedback loops.', 1250),
    ('synthetic-training-data',      'Synthetic Data for Training',  'Post-Training', 'Generating SFT/RL training data with LLMs — distillation corpora, evol-instruct, self-instruct, persona generation, synthetic reasoning traces, OpenHermes/Magpie-style data, data flywheels.', 1260),
    ('distillation',                 'Distillation',                 'Post-Training', 'Compressing a large teacher model into a smaller student — knowledge distillation, logit matching, hidden-state distillation, MiniLM-style, distilled small LMs.', 1270),
    ('self-play',                    'Self-Play and Self-Improvement','Post-Training', 'Models that improve themselves through self-generated training data — self-rewarding LMs, self-play tournaments, recursive self-improvement, STaR, ReST, weak-to-strong self-bootstrap.', 1280),
    ('continual-pretraining',        'Continual Pretraining',        'Post-Training', 'Continuing pretraining of an existing base model on new domain data — code-CPT, math-CPT, multilingual-CPT, replay buffers to avoid catastrophic forgetting.', 1290),
    ('curriculum-learning',          'Curriculum Learning',          'Post-Training', 'Ordering training examples from easy to hard during training — curriculum schedules, difficulty estimation, anti-curriculum, automatic curricula.', 1300),

    -- Capabilities & Inference
    ('reasoning-models',             'Reasoning Models',             'Capabilities & Inference', 'LLMs that generate explicit step-by-step reasoning traces before answering — OpenAI o1 / o3, DeepSeek R1, Claude with extended thinking, Gemini Deep Think, internal CoT models, thinking-token models.', 1410),
    ('chain-of-thought',             'Chain-of-Thought',             'Capabilities & Inference', 'Prompting techniques that elicit step-by-step reasoning at inference time — CoT prompting, self-consistency, scratchpads, "let''s think step by step", least-to-most prompting.', 1420),
    ('test-time-compute',            'Test-Time Compute',            'Capabilities & Inference', 'Scaling inference compute to improve answers — best-of-N sampling, majority voting, tree-of-thoughts, MCTS search at inference, process-reward search, scaling test-time compute curves.', 1430),
    ('speculative-decoding',         'Speculative Decoding',         'Capabilities & Inference', 'Draft-and-verify decoding to accelerate LLM inference — Medusa, EAGLE, speculative sampling, n-gram drafts, self-speculative, lookahead decoding.', 1440),
    ('quantization',                 'Quantization',                 'Capabilities & Inference', 'Reducing weight precision for cheaper LLM inference — INT8, INT4, FP8, FP4 quantization, GGUF, GPTQ, AWQ, bitsandbytes, llama.cpp quants, post-training quantization, QAT. NOT distillation.', 1450),
    ('inference-engines',            'Inference Engines',            'Capabilities & Inference', 'LLM serving runtimes — vLLM, SGLang, TensorRT-LLM, llama.cpp, ggml, MLC-LLM, exllama, ollama backend, continuous batching, paged attention engines. Serving frameworks, NOT model training.', 1460),
    ('kv-cache-optimization',        'KV-Cache Optimization',        'Capabilities & Inference', 'Reducing KV-cache memory for long-context LLM serving — paged attention, KV compression, KV eviction, prefix sharing, attention sink, H2O, StreamingLLM.', 1470),
    ('long-context-inference',       'Long-Context Inference',       'Capabilities & Inference', 'Serving very long contexts in production — attention sinks, KV eviction, streaming inference, context caching, RAG vs long-context tradeoffs at serve time. Inference-side, not training-side.', 1480),
    ('tool-use',                     'Tool Use and Function Calling','Capabilities & Inference', 'LLMs invoking external tools and APIs — function calling, tool-use formats, ReAct prompting, JSON tool calls, OpenAI function calling, native tool use in Claude/Gemini.', 1490),
    ('structured-outputs',           'Structured Outputs',           'Capabilities & Inference', 'Forcing LLMs to produce schema-valid structured data — JSON mode, JSON Schema constrained decoding, grammar-constrained generation, outlines, Instructor library, BAML, XML/regex constraints.', 1500),
    ('prompting-techniques',         'Prompting Techniques',         'Capabilities & Inference', 'Practical prompt engineering for LLMs — system prompts, few-shot examples, persona prompting, format scaffolding, prompt chaining, prompt patterns. The craft of writing prompts, NOT chain-of-thought.', 1510),
    ('hallucination-research',       'Hallucination Research',       'Capabilities & Inference', 'Measuring and reducing LLM hallucinations — factuality benchmarks, hallucination detection, calibration, abstention training, hallucination eval suites, TruthfulQA, FActScore.', 1520),

    -- Agents & Retrieval
    ('coding-agents',                'Coding Agents',                'Agents & Retrieval', 'AI agents that write code autonomously — Cursor, Aider, Devin, Claude Code, Codex CLI, SWE-agent, OpenHands, Cline, swe-bench solvers, multi-file editing agents.', 1610),
    ('browser-agents',               'Browser Agents',               'Agents & Retrieval', 'AI agents that navigate the web through a browser — Playwright/Selenium-driven agents, web shopping agents, WebArena, Mind2Web, browser-use, multi-step web navigation.', 1620),
    ('computer-use-agents',          'Computer-Use Agents',          'Agents & Retrieval', 'AI agents that control desktop GUIs at pixel level — Anthropic Computer Use, OpenAI Operator, GUI grounding, screenshot-based agents, OSWorld, ScreenAgent, mobile UI agents.', 1630),
    ('agentic-frameworks',           'Agentic Frameworks',           'Agents & Retrieval', 'Software frameworks for building LLM agents — LangGraph, LangChain agents, AutoGen, CrewAI, LlamaIndex agents, MetaGPT, agent orchestration scaffolds. The frameworks themselves, not specific deployed agents.', 1640),
    ('multi-agent-systems',          'Multi-Agent Systems',          'Agents & Retrieval', 'Multiple LLM agents collaborating, debating, or competing — multi-agent debate, agent societies, ChatDev, AutoGen GroupChat, hierarchical agent crews, agent role specialization.', 1650),
    ('memory-systems',               'Memory Systems',               'Agents & Retrieval', 'Long-term and episodic memory for LLM agents — MemGPT, Letta, Mem0, semantic memory stores for agents, conversation summarization, memory reflection loops.', 1660),
    ('retrieval-augmented-generation','Retrieval-Augmented Generation','Agents & Retrieval', 'Augmenting LLM generation with retrieved documents — RAG pipelines, chunking strategies, hybrid retrieval, reranking, query rewriting, contextual retrieval, citation grounding, retrieval-then-read.', 1670),
    ('vector-databases',             'Vector Databases',             'Agents & Retrieval', 'Specialized stores for nearest-neighbour search over embedding vectors — Pinecone, Weaviate, Qdrant, Milvus, pgvector, FAISS, ChromaDB, HNSW indexes, ANN benchmarks. NOT relational databases or general data stores.', 1680),
    ('embedding-models',             'Embedding Models',             'Agents & Retrieval', 'Dense and late-interaction text embedding models — OpenAI text-embedding, BGE, E5, GTE, mxbai-embed, jina-embed, Cohere embed, ColBERT, NV-Embed, embedding model training and evaluation (MTEB).', 1690),
    ('semantic-search',              'Semantic Search',              'Agents & Retrieval', 'Information retrieval ranking systems — BM25, dense retrieval, hybrid search, learning-to-rank, cross-encoder rerankers, ColBERT late interaction, BEIR benchmark, search quality. NOT web crawling.', 1700),

    -- Releases
    ('open-weight-releases',         'Open-Weight Model Releases',   'Releases', 'New open-weight LLM checkpoints with public weights — Llama, Qwen, Mistral open variants, Gemma, Phi, Yi, Falcon, OLMo, base+chat releases, license discussion. NOT closed frontier APIs.', 1810),
    ('frontier-model-releases',      'Frontier Model Releases',      'Releases', 'Closed flagship model launches from frontier labs — GPT-5, Claude Opus, Gemini Ultra, Grok-3, top-of-the-line API-only models. Specifically the release event of a closed flagship.', 1820),
    ('chinese-ai-labs',              'Chinese AI Labs',              'Releases', 'News and releases from China-headquartered AI labs — DeepSeek, Qwen (Alibaba), Moonshot/Kimi, Zhipu/GLM, MiniMax, Baichuan, 01.AI Yi, Tencent ARC, ByteDance Seed/Doubao. Specifically Chinese-lab news.', 1830),
    ('openai-news',                  'OpenAI News',                  'Releases', 'OpenAI corporate news — leadership changes, product launches (ChatGPT features, GPT-5, Sora), partnerships, board drama, OpenAI policy positions. The company OpenAI specifically.', 1840),
    ('anthropic-news',               'Anthropic News',               'Releases', 'Anthropic corporate news — Claude releases, Anthropic product features (Projects, Artifacts), funding rounds, hiring, policy positions, Claude API updates. The company Anthropic specifically.', 1850),
    ('google-deepmind-news',         'Google DeepMind News',         'Releases', 'Google DeepMind announcements — Gemini family launches, AlphaFold/AlphaProof/AlphaGeometry, Veo, Imagen, DeepMind research output, Google I/O AI keynote. Google and DeepMind specifically.', 1860),
    ('meta-ai-news',                 'Meta AI News',                 'Releases', 'Meta AI announcements — Llama model releases, FAIR research output, Meta product AI (Meta AI assistant, Ray-Ban), Yann LeCun statements, Meta superintelligence team. The company Meta specifically.', 1870),
    ('mistral-news',                 'Mistral News',                 'Releases', 'Mistral AI announcements — Mistral Small/Medium/Large, Mixtral, Codestral, Le Chat product, partnerships with Microsoft/IBM, Mistral team output. The company Mistral specifically.', 1880),
    ('xai-news',                     'xAI News',                     'Releases', 'xAI announcements — Grok releases, Colossus supercluster, xAI hiring, Elon Musk AI statements, xAI product launches. The company xAI specifically.', 1890),
    ('other-frontier-labs',          'Other Frontier Labs',          'Releases', 'Smaller and emerging frontier labs — Cohere, Reka, AI21 Jamba, Adept, Inflection, Character.AI, Liquid AI, SSI (Safe Superintelligence), Magic, Imbue, Poolside, MultiOn. Lab-level news from these.', 1900),

    -- Evaluation
    ('benchmarks-leaderboards',      'Benchmarks and Leaderboards',  'Evaluation', 'New ML benchmarks and leaderboards — MMLU, GPQA, SWE-bench, HumanEval, GSM8K, MATH, Big-Bench, leaderboard updates, eval-suite releases. Specifically new benchmarks or rankings.', 2010),
    ('evaluation-methodology',       'Evaluation Methodology',       'Evaluation', 'How to do ML evaluation well — eval design, statistics of evaluation, common pitfalls, evaluation rigor, principled benchmarking. The methodology, not specific benchmarks.', 2020),
    ('eval-hygiene',                 'Contamination and Eval Hygiene','Evaluation', 'Detecting and avoiding eval contamination — train/test overlap, web-scrape leakage, contamination detection methods, eval-set decontamination, canary strings.', 2030),
    ('human-evaluation',             'Human Evaluation',             'Evaluation', 'Human-in-the-loop evaluation — LMSYS Arena, Chatbot Arena ELO, crowdworker rating studies, preference collection, side-by-side human eval, MT-Bench.', 2040),
    ('capability-forecasting',       'Capability Forecasting',       'Evaluation', 'Predicting model capabilities from training inputs — predicting downstream task performance from FLOPs, predictability of capability emergence, BIG-Bench progress measures.', 2050),
    ('red-team-evals',               'Red-Team Evaluations',         'Evaluation', 'Adversarial evaluation for dangerous capabilities — dangerous capability evals, CBRN evals, persuasion evals, autonomous replication, METR-style evals.', 2060),

    -- Multimodal & Vision
    ('vision-language-models',       'Vision-Language Models',       'Multimodal & Vision', 'Multimodal LLMs that take images as input — GPT-4V, Claude vision, Gemini, LLaVA, Qwen-VL, InternVL, MiniCPM-V, vision encoders fused with LLMs, image-understanding benchmarks (MMBench, MMMU).', 2210),
    ('image-generation',             'Image Generation',             'Multimodal & Vision', 'Generating images from text or other conditions — Stable Diffusion, FLUX, SDXL, DALL-E, Midjourney, Imagen, latent diffusion, flow matching, autoregressive image transformers, T2I/I2I models.', 2220),
    ('image-editing',                'Image Editing',                'Multimodal & Vision', 'Editing existing images via instructions or controls — instruct-pix2pix, inpainting, outpainting, ControlNet, IP-Adapter, image-to-image diffusion, removing watermarks, image restoration, super-resolution.', 2230),
    ('video-generation',             'Video Generation',             'Multimodal & Vision', 'Generating synthetic video — Sora, Veo, Kling, Runway Gen-3, Pika, video diffusion models, T2V, I2V, motion priors, long-video generation, video VAEs.', 2240),
    ('world-models',                 'World Models',                 'Multimodal & Vision', 'Generative video models used as physical world simulators — Genie, Sora-as-world-model, action-conditioned video, neural game engines, generative simulators for robotics training.', 2250),
    ('3d-reconstruction',            '3D Reconstruction',            'Multimodal & Vision', 'Recovering 3D geometry from images — Structure-from-Motion (SfM), MVS, COLMAP, deep stereo, multi-view 3D, photogrammetry pipelines, learned MVS, dense 3D reconstruction.', 2260),
    ('gaussian-splatting',           'Gaussian Splatting',           'Multimodal & Vision', '3D Gaussian Splatting for novel view synthesis — 3DGS, 4D-GS for dynamic scenes, real-time splatting renderers, editable splats, mip-splatting.', 2270),
    ('nerf-implicit',                'NeRF and Implicit Surfaces',   'Multimodal & Vision', 'Implicit-function 3D representations — Neural Radiance Fields (NeRF), Instant-NGP, signed distance fields, occupancy networks, neural surfaces, mip-NeRF.', 2280),
    ('detection-segmentation',       'Detection and Segmentation',   'Multimodal & Vision', 'Object detection and segmentation — YOLO, DETR, Grounding DINO, panoptic and instance segmentation, SAM/SAM2, semantic segmentation models, open-vocabulary detection.', 2290),
    ('ocr-document-ai',              'OCR and Document AI',          'Multimodal & Vision', 'Document parsing and OCR — Tesseract, PaddleOCR, Donut, Nougat, table extraction, document layout analysis, form parsing, document understanding benchmarks.', 2300),
    ('tracking-motion',              'Tracking and Motion',          'Multimodal & Vision', 'Object tracking and motion analysis — multi-object tracking, optical flow (RAFT, FlowFormer), pose estimation, action recognition. Specifically tracking and motion, not 3D scene reconstruction.', 2310),
    ('visual-tokenizers',            'Visual Tokenizers',            'Multimodal & Vision', 'Discrete tokenizers that turn images into token sequences for transformers — VQ-VAE, VQ-GAN, FSQ, MoVQ, MaskGIT, finite-scalar quantization, image codebooks. Used IN multimodal models, NOT generation itself.', 2320),

    -- Speech & Audio
    ('text-to-speech',               'Text-to-Speech',               'Speech & Audio', 'TTS systems and synthesized voice — ElevenLabs, OpenAI TTS, Kokoro, XTTS, F5-TTS, neural vocoders, expressive speech synthesis, voice cloning for TTS.', 2410),
    ('speech-recognition',           'Speech Recognition',           'Speech & Audio', 'Automatic speech recognition (ASR) — Whisper, Whisper-large-v3, Distil-Whisper, NVIDIA Canary, Conformer, streaming ASR, multilingual ASR.', 2420),
    ('music-generation',             'Music Generation',             'Speech & Audio', 'Music and audio generation models — Suno, Udio, MusicLM, MusicGen, AudioLDM, Stable Audio, audio diffusion. Specifically generating music or general audio.', 2430),
    ('voice-cloning',                'Voice Cloning',                'Speech & Audio', 'Cloning a specific voice from a short sample — zero-shot voice cloning, few-shot voice cloning, speaker adaptation, deepfake voice concerns.', 2440),
    ('audio-language-models',        'Audio Language Models',        'Speech & Audio', 'End-to-end audio LLMs — AudioLM, SoundStream, Moshi, Kyutai speech-to-speech, audio-token language models, real-time multimodal speech.', 2450),

    -- RL & Robotics
    ('policy-optimization',          'Policy Optimization',          'RL & Robotics', 'On-policy policy-gradient RL algorithms — PPO, TRPO, A2C, A3C, importance sampling, advantage estimation, on-policy actor-critic methods.', 2610),
    ('exploration-curiosity',        'Exploration and Curiosity',    'RL & Robotics', 'Exploration strategies in RL — intrinsic motivation, curiosity-driven exploration, count-based bonuses, RND, never-give-up, novelty rewards.', 2620),
    ('off-policy-learning',          'Off-Policy Learning',          'RL & Robotics', 'Off-policy RL — Q-learning, DQN, SAC, TD3, replay buffers, target networks, value-based methods, offline RL on logged data.', 2630),
    ('model-based-rl',               'Model-Based RL',               'RL & Robotics', 'Learning dynamics models for planning — Dreamer, MuZero, world-model RL, planning with learned models, Dyna, MBPO.', 2640),
    ('manipulation',                 'Manipulation',                 'RL & Robotics', 'Robot manipulation — robotic grasping, dexterous in-hand manipulation, bi-manual coordination, manipulation benchmarks (LIBERO, RoboCasa), tool use by robots.', 2650),
    ('locomotion',                   'Locomotion',                   'RL & Robotics', 'Robot locomotion — quadruped and bipedal walking, humanoid control, ANYmal, Unitree, parkour policies, dynamic locomotion, terrain traversal.', 2660),
    ('vision-language-action',       'Vision-Language-Action Models','RL & Robotics', 'End-to-end robot policies that map image observations and language commands to motor actions — RT-2, OpenVLA, Octo, π0, RDT, Figure-class VLA, manipulator policies driven by VLMs. NOT general VLMs.', 2670),
    ('sim2real',                     'Sim2Real Transfer',            'RL & Robotics', 'Transferring policies from simulation to real robots — domain randomization, real-to-sim, system identification, Isaac Lab, MuJoCo-to-real transfer, sim2real benchmarks.', 2680),
    ('game-playing-agents',          'Game-Playing Agents',          'RL & Robotics', 'RL agents that play games — AlphaGo, AlphaZero, MuZero, OpenAI Five Dota, Atari agents, StarCraft AlphaStar, poker bots, video-game RL.', 2690),
    ('open-ended-learning',          'Open-Ended Learning',          'RL & Robotics', 'Open-ended evolution of agents and environments — POET, PAIRED, ACCEL, evolving curricula, NetHack environments, generally-capable open-ended agents.', 2700),

    -- Theory & Math of ML
    ('generalization-theory',        'Generalization Theory',        'Theory & Math of ML', 'Theoretical research on why over-parameterized neural nets generalize — implicit regularization, double descent, lottery ticket hypothesis, generalization bounds for deep nets, benign overfitting.', 2810),
    ('optimization-theory',          'Optimization Theory',          'Theory & Math of ML', 'Theory of training optimizers — convergence proofs for SGD/Adam, second-order optimization theory (Shampoo, K-FAC), adaptive learning rate theory.', 2820),
    ('loss-landscape',               'Loss Landscape Analysis',      'Theory & Math of ML', 'Geometric analysis of neural-net loss surfaces — mode connectivity, sharpness/flatness, basin structure, linear-mode connectivity, loss-landscape visualization.', 2830),
    ('neural-scaling-theory',        'Neural Scaling Theory',        'Theory & Math of ML', 'Theory underlying neural scaling laws — infinite-width limits, neural tangent kernel theory, maximal update parameterization (muP), feature learning theory, Tensor Programs.', 2840),
    ('statistical-learning',         'Statistical Learning',         'Theory & Math of ML', 'Classical statistical learning theory — PAC learning, VC dimension, Rademacher complexity, uniform convergence bounds, kernel methods theory.', 2850),
    ('causal-inference',             'Causal Inference',             'Theory & Math of ML', 'Causal ML and causal inference — do-calculus, instrumental variables, propensity scores, double machine learning, causal effect estimation, Judea Pearl-style methods.', 2860),
    ('bayesian-deep-learning',       'Bayesian Deep Learning',       'Theory & Math of ML', 'Probabilistic deep learning that quantifies uncertainty — Bayesian neural networks, variational inference for deep nets, MC dropout, deep ensembles as posterior, Stan/Pyro/NumPyro.', 2870),
    ('information-theory-ml',        'Information Theory in ML',     'Theory & Math of ML', 'Information-theoretic analyses of ML — information bottleneck, mutual information estimation, MINE, entropy and ML, rate-distortion in deep learning.', 2880),

    -- Interpretability & Safety
    ('mechanistic-interpretability', 'Mechanistic Interpretability', 'Interpretability & Safety', 'Reverse-engineering computations inside neural networks — transformer circuits, induction heads, attention head analysis, MLP feature decomposition, model internals reverse engineering.', 3010),
    ('sparse-autoencoders',          'Sparse Autoencoders',          'Interpretability & Safety', 'Sparse autoencoders for feature discovery on model activations — SAE training, Top-K SAEs, Anthropic dictionary learning, monosemantic features, SAE feature visualization.', 3020),
    ('circuit-discovery',            'Circuit Discovery',            'Interpretability & Safety', 'Discovering computation circuits in transformers — ACDC, attribution patching, edge attribution, circuit-finding tooling, automated circuit discovery.', 3030),
    ('feature-attribution',          'Feature Attribution',          'Interpretability & Safety', 'Attribution methods for model decisions — saliency maps, integrated gradients, SHAP, LIME, GradCAM, attention rollout, attribution evaluation.', 3040),
    ('jailbreaks-prompt-injection',  'Jailbreaks and Prompt Injection','Interpretability & Safety', 'Adversarial inputs against LLMs — jailbreaks, prompt injection (direct and indirect), DAN-style attacks, suffix attacks, GCG, prompt extraction attacks.', 3050),
    ('red-teaming',                  'Red-Teaming',                  'Interpretability & Safety', 'Structured adversarial testing of AI systems — red-team frameworks, capability evals for harm, automated red-teaming, harmbench, dangerous-capability discovery.', 3060),
    ('deceptive-alignment',          'Deceptive Alignment',          'Interpretability & Safety', 'Research on deceptive AI behavior — sleeper agents, alignment faking, deceptive alignment theory, scheming behaviors, hidden goals in models.', 3070),
    ('scalable-oversight',           'Scalable Oversight',           'Interpretability & Safety', 'Methods for overseeing AIs more capable than humans — debate, IDA (iterated distillation and amplification), weak-to-strong generalization, recursive reward modeling.', 3080),
    ('ai-policy',                    'AI Governance and Policy',     'Interpretability & Safety', 'AI regulation and policy — EU AI Act, US AI executive orders, compute governance, export controls, AI safety institutes, policy briefs. Government and regulatory side.', 3090),
    ('ai-ethics-bias',               'AI Ethics and Bias',           'Interpretability & Safety', 'Social impact, bias, and fairness of deployed AI — algorithmic discrimination, fairness metrics, bias audits, Stochastic Parrots-style critique, AI ethics frameworks.', 3100),

    -- Hardware & Systems
    ('ai-chips',                     'AI Chips and Accelerators',    'Hardware & Systems', 'AI accelerator startups and silicon — Cerebras WSE, Groq LPU, Tenstorrent, Lightmatter, Etched Sohu, Rain AI, SambaNova, AI-specific ASICs and accelerators. NOT NVIDIA GPUs, which have their own category.', 3210),
    ('gpu-architecture',             'GPU Architecture',             'Hardware & Systems', 'NVIDIA and AMD GPU internals — Hopper, Blackwell, MI300, MI325, SM architecture, tensor cores, GPU memory hierarchies, HBM, GPU microarchitecture analysis.', 3220),
    ('custom-silicon',               'Custom Silicon',               'Hardware & Systems', 'In-house silicon by hyperscalers and labs — Google TPU v5/v6/Trillium, AWS Trainium/Inferentia, Microsoft Maia, Meta MTIA, Apple Neural Engine, in-house lab chips.', 3230),
    ('datacenter-economics',         'Datacenter Economics',         'Hardware & Systems', 'AI compute economics — training-cluster spend, GPU pricing, power and cooling costs, SemiAnalysis-style hardware economics, datacenter buildouts, Stargate, capex tracking.', 3240),
    ('networking-interconnects',     'Networking and Interconnects', 'Hardware & Systems', 'Cluster networking for AI training — NVLink, NVSwitch, InfiniBand, RoCE, all-reduce, collective communication (NCCL), high-bandwidth interconnects.', 3250),
    ('distributed-training',         'Distributed Training',         'Hardware & Systems', 'Multi-node LLM training systems — data parallelism, pipeline parallelism, gradient accumulation, fault tolerance at scale, large-cluster training operations.', 3260),
    ('sharding-fsdp',                'Sharding and FSDP',            'Hardware & Systems', 'Parameter sharding for large-model training — FSDP, ZeRO-3, DTensor, tensor parallelism, sequence parallelism, context parallelism, sharded optimizer states.', 3270),
    ('training-infra',               'Training Infrastructure',      'Hardware & Systems', 'LLM training stacks and tooling — Megatron-LM, NeMo, Composer/MosaicML, MaxText, fairscale, training failure handling, checkpoint rotation.', 3280),
    ('cuda-triton-kernels',          'CUDA and Triton Kernels',      'Hardware & Systems', 'Hand-written GPU kernels — CUDA, Triton language, ThunderKittens, FlashAttention, kernel fusion, low-level GPU optimization, PyTorch CUDA extensions.', 3290),
    ('compilers-runtimes',           'Compilers and Runtimes',       'Hardware & Systems', 'ML compilers and runtimes — XLA, MLIR, torch.compile / Inductor, IREE, TVM, ONNX Runtime, compiler-driven ML optimization.', 3300),

    -- Data Engineering
    ('web-crawling-datasets',        'Web Crawling and Datasets',    'Data Engineering', 'Web-scale training dataset releases — Common Crawl, FineWeb, FineWeb-Edu, RefinedWeb, RedPajama, SlimPajama, dataset construction at scale.', 3410),
    ('data-curation',                'Data Curation and Filtering',  'Data Engineering', 'Quality filtering of training data — classifier-based filtering, MinHash dedup as quality signal, perplexity filtering, fastText filters, education-quality classifiers.', 3420),
    ('deduplication',                'Deduplication',                'Data Engineering', 'Removing duplicates from training corpora — MinHash, SimHash, semantic deduplication, near-duplicate detection, deduplication ablations.', 3430),
    ('synthetic-data-pipelines',     'Synthetic Data Pipelines',     'Data Engineering', 'Pipelines that generate training data with LLMs — Magpie, persona-based synthetic data, self-instruct pipelines, distillation-corpus engineering, data flywheels.', 3440),
    ('data-licensing',               'Data Licensing',               'Data Engineering', 'Training-data copyright and licensing issues — NYT vs OpenAI, copyright lawsuits about training, data-licensing deals, fair use of training data.', 3450),

    -- MLOps
    ('experiment-tracking',          'Experiment Tracking',          'MLOps', 'Tools for tracking ML experiments — Weights & Biases, MLflow, Aim, Neptune, TensorBoard, run logging, experiment comparison dashboards.', 3510),
    ('model-serving',                'Model Serving',                'MLOps', 'Production serving infrastructure for models — endpoint autoscaling, batching policies for serving, Modal, Replicate, Baseten, BentoML, KServe.', 3520),
    ('evaluation-harnesses',         'Evaluation Harnesses',         'MLOps', 'Frameworks for running evals at scale — lm-eval-harness, HELM, OpenCompass, BIG-bench infrastructure, Open LLM Leaderboard tooling, eval automation.', 3530),
    ('versioning-registries',        'Versioning and Registries',    'MLOps', 'Versioning of models and datasets — model registries, dataset versioning, DVC, lineage tracking, reproducibility tooling.', 3540),

    -- AI for Science
    ('protein-structure',            'Protein Structure Prediction', 'AI for Science', 'Predicting 3D protein structure from sequence — AlphaFold 2/3, ESMFold, OpenFold, RoseTTAFold, structure-prediction benchmarks. Predicting structure of existing proteins. NOT de novo design.', 3610),
    ('protein-design',               'Protein Design',               'AI for Science', 'De novo design of new proteins — ProteinMPNN, RFdiffusion, Chroma, RFAA, designing novel binders/enzymes/scaffolds. Designing new proteins, NOT predicting existing structures.', 3620),
    ('molecular-property',           'Molecular Property Prediction','AI for Science', 'Predicting properties of small molecules from structure — QSAR, GNNs on molecules, MoleculeNet, ADMET property prediction, OGB molecule benchmarks.', 3630),
    ('drug-discovery',               'Drug Discovery',               'AI for Science', 'AI for drug discovery pipelines — virtual screening, drug-target interaction, generative chemistry for drugs, Insilico Medicine, Recursion, Isomorphic Labs.', 3640),
    ('genomics-dna',                 'Genomics and DNA Models',      'AI for Science', 'DNA foundation models and genomics ML — Evo, Evo-2, HyenaDNA, Carbon, Nucleotide Transformer, DNA language models, variant effect prediction.', 3650),
    ('single-cell-biology',          'Single-Cell Biology',          'AI for Science', 'ML for single-cell transcriptomics — scRNA-seq foundation models, Geneformer, scGPT, scBERT, single-cell atlas analysis, cell-type classification.', 3660),
    ('medical-imaging',              'Medical Imaging',              'AI for Science', 'Medical image analysis — radiology AI, pathology, ophthalmology, X-ray, CT, MRI deep learning, segmentation in medical images. Specifically medical/health imaging.', 3670),
    ('clinical-nlp',                 'Clinical NLP',                 'AI for Science', 'NLP for clinical text — EHR processing, clinical notes summarization, medical entity extraction, MedQA, clinical concept normalization, ICD coding.', 3680),
    ('healthcare-llms',              'Healthcare LLMs',              'AI for Science', 'LLMs tuned for medical use — Med-PaLM, MedGemma, Meditron, clinical chatbots, patient triage, medical reasoning benchmarks.', 3690),
    ('chemistry-foundation-models',  'Chemistry Foundation Models',  'AI for Science', 'Foundation models for chemistry — molecular language models, reaction prediction, retrosynthesis, ChemBERTa, MoLFormer, chemistry-specific transformers.', 3700),
    ('materials-discovery',          'Materials Discovery',          'AI for Science', 'AI-driven materials discovery — GNoME, crystal structure prediction, materials property prediction, battery materials, catalyst discovery, MatterGen.', 3710),
    ('weather-climate',              'Weather and Climate Modeling', 'AI for Science', 'ML weather and climate forecasting — GraphCast, Pangu-Weather, FourCastNet, neural climate emulators, AI weather prediction systems.', 3720),
    ('physics-simulation',           'Physics and Simulation',       'AI for Science', 'Neural surrogates for physical simulation — PINNs, neural ODE/PDE solvers, computational fluid dynamics with ML, learned simulators for physics, JAX-based scientific computing.', 3730),
    ('math-theorem-proving',         'Mathematics and Theorem Proving','AI for Science', 'AI for formal mathematics — AlphaProof, AlphaGeometry, Lean theorem prover, miniF2F, MathArena, mathematical reasoning at Olympiad level, formal proof systems.', 3740),

    -- Neuroscience Adjacent
    ('computational-neuroscience',   'Computational Neuroscience',   'Neuroscience Adjacent', 'Computational models of biological neural systems — spiking neural networks, dendritic computation, predictive coding theory, biologically plausible learning rules, computational psychiatry. Modeling brains, not artificial neural nets.', 3810),
    ('brain-computer-interfaces',    'Brain-Computer Interfaces',    'Neuroscience Adjacent', 'BCIs and neural decoding — Neuralink, electrode arrays, decoding speech from neural activity, motor BCI, EEG decoding, brain-computer interface hardware and methods.', 3820),
    ('cognitive-science-ai',         'Cognitive Science of AI',      'Neuroscience Adjacent', 'Empirical comparisons between human and LLM cognition — behavioural studies of LLM reasoning, theory-of-mind in models, LLMs as cognitive test subjects, model psychology. Research treating models as objects of study.', 3830),

    -- ML Dev Ecosystem
    ('pytorch',                      'PyTorch',                      'ML Dev Ecosystem', 'The PyTorch deep-learning framework specifically — torch.nn, torch.compile, TorchDynamo, distributed PyTorch (FSDP/DTensor), torchvision/torchaudio, PyTorch releases. NOT generic ML library news.', 3910),
    ('jax',                          'JAX',                          'ML Dev Ecosystem', 'The JAX framework specifically — Flax, Haiku, Equinox, pjit/pmap, jit compilation, XLA-via-JAX, Pallas kernels, JAX-based research libraries. Only when JAX is the actual topic.', 3920),
    ('huggingface-ecosystem',        'Hugging Face Ecosystem',       'ML Dev Ecosystem', 'Hugging Face libraries and Hub — transformers, datasets, accelerate, peft, diffusers, trl, hub releases, HF Spaces, dataset releases on HF. Specifically the Hugging Face ecosystem.', 3930),
    ('scikit-learn',                 'Scikit-Learn and Classical ML','ML Dev Ecosystem', 'Classical ML libraries and methods — scikit-learn, XGBoost, LightGBM, CatBoost, gradient boosting, random forests, classical algorithms applied today.', 3940),
    ('notebooks-visualization',      'Notebooks and Visualization',  'ML Dev Ecosystem', 'Computational notebooks and scientific plotting — Jupyter Lab, Marimo, observable, matplotlib, seaborn, plotly, Altair, vega-lite. Tools for interactive data exploration and ML viz.', 3950),

    -- Web & Frontend
    ('react-next',                   'React and Next.js',            'Web & Frontend', 'React UI library and Next.js framework — React Server Components, Next.js app router, Vercel platform, React Compiler, React Native, Suspense, server actions. Specifically React/Next ecosystem.', 4010),
    ('other-web-frameworks',         'Other Web Frameworks',         'Web & Frontend', 'Non-React web frameworks — Vue, Svelte/SvelteKit, Solid, Astro, Qwik, Remix, Nuxt. Specifically these alternatives to React.', 4020),
    ('web-performance',              'Web Performance',              'Web & Frontend', 'Web page performance — Core Web Vitals (LCP, INP, CLS), JS bundle size, lazy loading, image optimization, render-blocking optimization, web performance tooling.', 4030),
    ('browser-internals',            'Browser Internals',            'Web & Frontend', 'Web browser engine internals — V8, JavaScriptCore, SpiderMonkey, Blink, WebKit, Gecko, browser performance work, browser feature implementation.', 4040),
    ('css-animation',                'CSS and Frontend Animation',   'Web & Frontend', 'CSS language and animation — CSS Working Group proposals, container queries, anchor positioning, view transitions, CSS animation libraries, design systems CSS, Tailwind, modern CSS features.', 4050),
    ('backend-infrastructure',       'Backend Infrastructure',       'Web & Frontend', 'Web backend systems — API design (REST, GraphQL, gRPC), microservices, message queues, web framework backends (Django, FastAPI, Rails, Phoenix). Server-side web infrastructure.', 4060),

    -- Languages & Tools
    ('rust',                         'Rust',                         'Languages & Tools', 'The Rust systems programming language — Rust compiler, cargo, tokio async runtime, Rust crates, ownership and borrowing discussions, Rust for systems programming.', 4110),
    ('mojo',                         'Mojo',                         'Languages & Tools', 'Modular''s Mojo programming language — Mojo language design, MAX runtime, Mojo for AI inference, Mojo std library.', 4120),
    ('swift',                        'Swift',                        'Languages & Tools', 'The Swift programming language — Swift evolution proposals, server-side Swift (Vapor), Swift for AI/ML (Swift for TensorFlow lineage), SwiftUI as a language feature.', 4130),
    ('python-language',              'Python Language and Ecosystem','Languages & Tools', 'Python language internals and tooling ecosystem — CPython, PEP discussions, type hints (mypy, pyright), packaging (uv, poetry, pip, hatch), async/asyncio, Python releases. The language itself, NOT ML libraries written in Python.', 4140),
    ('terminal-cli',                 'Terminal and CLI Tools',       'Languages & Tools', 'Terminal emulators and command-line tools — Ghostty, WezTerm, Kitty, Alacritty, modern shells (fish, nushell), CLI utilities (ripgrep, fzf, bat, eza), shell prompts.', 4150),
    ('editors-ides',                 'Editors and IDEs',             'Languages & Tools', 'Code editors and IDEs — VS Code, Cursor, Zed, Neovim/Vim, JetBrains products, Emacs, Helix, editor configuration, IDE features. NOT coding agents.', 4160),
    ('git-code-review',              'Git and Code Review',          'Languages & Tools', 'Git version control and code review tooling — git porcelain commands, GitHub features, pull request workflows, Graphite, Gerrit, Reviewable, code review practice.', 4170),

    -- Databases
    ('postgres-sql',                 'Postgres and SQL Engines',     'Databases', 'Relational databases and SQL — PostgreSQL, MySQL, SQLite, DuckDB, query engines, SQL standards, database internals, query planners. NOT vector databases.', 4210),
    ('redis-inmemory',               'Redis and In-Memory',          'Databases', 'In-memory data stores — Redis, Valkey, KeyDB, Dragonfly, KVRocks, memcached, in-memory cache systems.', 4220),
    ('distributed-data',             'Distributed Data Systems',     'Databases', 'Distributed analytics and data processing — Apache Spark, Ray Data, Trino, Apache Iceberg, Delta Lake, lakehouse architectures, Polars, DataFusion.', 4230),

    -- Industry
    ('startups-funding',             'AI Startups and Funding',      'Industry', 'AI startup funding announcements — seed/Series A/B/C rounds in AI startups, valuations, new AI startup launches, fundraises by Mistral/Anthropic/OpenAI/etc.', 4310),
    ('mergers-acquisitions',         'Mergers and Acquisitions',     'Industry', 'M&A activity in AI — Microsoft acquiring Inflection talent, Google''s Character.AI deal, lab acquisitions, acquihires, M&A rumors.', 4320),
    ('hiring-job-moves',             'Hiring and Job Moves',         'Industry', '"I''m joining X" announcements, "we''re hiring" posts, AI researcher career moves between labs, faculty appointments, hiring threads.', 4330),
    ('layoffs-restructuring',        'Layoffs and Restructuring',    'Industry', 'AI lab layoffs and team restructurings — Meta superintelligence reshuffle, OpenAI departures, Stability AI restructuring, team dissolutions, talent migration.', 4340),
    ('conferences-workshops',        'Conferences and Workshops',    'Industry', 'AI/ML academic conferences — NeurIPS, ICML, ICLR, CVPR, ECCV, ACL, EMNLP, COLM, oral presentations, poster sessions, workshop calls, conference attendance.', 4350),
    ('awards-prizes',                'Awards and Prizes',            'Industry', 'AI/ML awards — Nobel Prize for ML (Hinton, Hassabis), Turing Award, Test of Time awards, Best Paper awards at conferences, ICML/NeurIPS Outstanding Paper.', 4360),
    ('productivity-workflow',        'Productivity and Workflow',    'Industry', 'How researchers/engineers work — productivity tools, daily workflow, habit threads, time management for technical work, deep-work essays.', 4370),

    -- Discourse & Learning
    ('hot-takes',                    'Hot Takes and Discourse',      'Discourse & Learning', 'Public AI/ML opinion threads and debates — "is RL fake", "AGI when", "X is overhyped" takes, ML Twitter discourse, contrarian arguments. Opinion pieces without technical content.', 4410),
    ('agi-timelines',                'AGI Timelines and Predictions','Discourse & Learning', 'AGI capability forecasting and timelines — predicted dates of AGI/superintelligence, capability roadmaps, scaling extrapolations to AGI, "what AGI will look like" essays.', 4420),
    ('paper-walkthroughs',           'Paper Walkthroughs',           'Discourse & Learning', 'Long-form Twitter threads or blog posts walking through a specific paper in depth — "I read this paper so you don''t have to", paper summary threads, detailed paper breakdowns.', 4430),
    ('course-material',              'Course Material',              'Discourse & Learning', 'Online courses and university course material — Stanford CS classes, MIT OCW, fast.ai courses, Karpathy''s nanoGPT tutorials, deep-learning courses, syllabi releases.', 4440),
    ('tutorial-threads',             'Tutorial Threads',             'Discourse & Learning', 'Short-form how-to content — Twitter threads explaining "how X works", code-walkthrough tutorials, technique explanations in thread form, technical mini-tutorials.', 4450),
    ('books-longform',               'Books and Long-Form',          'Discourse & Learning', 'Book releases, long-form blog posts and essays — Hundred-Page ML Book, Deep Learning textbook, gwern essays, Lil''Log, in-depth multi-thousand-word writeups.', 4460),
    ('research-methodology',         'Research Methodology',         'Discourse & Learning', 'How to do ML research — ablation design, peer review process, reproducibility, statistical hygiene, research-taste essays, advice on doing good experiments.', 4470),
    ('career-advice',                'Career Advice',                'Discourse & Learning', 'Career guidance for ML/AI practitioners — PhD advice, how to break into ML research, mentorship threads, picking a lab, junior-vs-senior expectations, interview prep advice.', 4480),

    -- Personal & Off-Topic
    ('personal-updates',             'Personal Updates',             'Personal & Off-Topic', 'Personal life updates — moving cities, getting married, having a baby, vacation photos, personal milestones unrelated to work.', 4510),
    ('memes-humor',                  'Memes and Humor',              'Personal & Off-Topic', 'ML memes, jokes, screenshots posted for laughs, lighthearted tech humor, ironic posts, satirical takes — content with no informational payload.', 4520),
    ('society-geopolitics',          'Society and Geopolitics',      'Personal & Off-Topic', 'Politics, geopolitics, war coverage, election commentary, general society discourse unrelated to AI/ML research or industry.', 4530)
ON CONFLICT (slug) DO UPDATE
    SET name        = EXCLUDED.name,
        group_name  = EXCLUDED.group_name,
        description = EXCLUDED.description,
        sort_order  = EXCLUDED.sort_order;
