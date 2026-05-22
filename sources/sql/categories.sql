-- Ontology of fine-grained topical categories.
--
-- Replaces the legacy single-string `users.category` column. Each VIP
-- can belong to 2–4 categories (e.g. Karpathy → llm-research +
-- educators + pioneers), so the onboarding flow can show "which kinds
-- of people you want to follow" with usable granularity.
--
-- `slug` is the stable machine identifier (used in URLs / API
-- payloads). `name` is the human label. `sort_order` controls display
-- order — onboarding wants broad/researchy buckets first, niche ones
-- last. Lower number = earlier.

CREATE TABLE IF NOT EXISTS categories (
    id          BIGSERIAL PRIMARY KEY,
    slug        TEXT        NOT NULL UNIQUE,
    name        TEXT        NOT NULL,
    description TEXT        NOT NULL DEFAULT '',
    sort_order  INTEGER     NOT NULL DEFAULT 0,
    -- 'topic'    → fine-grained research/role buckets (LLM Research,
    --              Computer Vision, AI Safety, …)
    -- 'company'  → institutional affiliations (OpenAI, Anthropic,
    --              Google DeepMind, Hugging Face, …)
    -- Onboarding renders the two kinds as separate sections so the
    -- 35-card grid stays scannable.
    kind        TEXT        NOT NULL DEFAULT 'topic',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE categories ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'topic';

-- Many-to-many: a user belongs to N categories. PK on the pair
-- enforces dedup and gives us a free lookup index by user_id; the
-- secondary index supports "all users in category X" queries used by
-- the onboarding flow.
CREATE TABLE IF NOT EXISTS user_categories (
    user_id     BIGINT NOT NULL REFERENCES users(id)      ON DELETE CASCADE,
    category_id BIGINT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, category_id)
);
CREATE INDEX IF NOT EXISTS idx_user_categories_category
    ON user_categories(category_id);

COMMENT ON TABLE categories IS
    'Topical ontology — fine-grained buckets the onboarding flow uses to ask "which kinds of people do you want to follow?". Each VIP can belong to multiple categories via user_categories.';
COMMENT ON COLUMN categories.slug       IS 'Stable machine identifier (URL-safe lower-kebab).';
COMMENT ON COLUMN categories.name       IS 'Human-friendly display label.';
COMMENT ON COLUMN categories.description IS 'Short blurb shown next to the category in the onboarding picker.';
COMMENT ON COLUMN categories.sort_order IS 'Display order (lower = earlier). Hand-curated.';

COMMENT ON TABLE user_categories IS
    'Many-to-many: a user belongs to one or more categories. Replaces the legacy single-string `users.category` column.';

-- ── Seed the 15-category ontology ─────────────────────────────────────
-- Idempotent: ON CONFLICT DO UPDATE refreshes the name/description
-- so editing this file is the canonical way to evolve the labels.
INSERT INTO categories (slug, name, description, sort_order) VALUES
    ('llm-research',       'LLM Research',                   'People building or studying frontier language models',         10),
    ('nlp-retrieval',      'NLP & Retrieval',                'Tokenization, embeddings, RAG, search systems',                20),
    ('computer-vision',    'Computer Vision',                'Image models, segmentation, vision-language',                  30),
    ('generative-media',   'Generative Media',               'Diffusion, image/video/audio generation',                      40),
    ('multimodal',         'Multimodal AI',                  'Vision-language, speech, cross-modal models',                  50),
    ('rl-robotics',        'Reinforcement Learning & Robotics', 'RL theory plus embodied AI',                                60),
    ('ai-safety',          'AI Safety & Alignment',          'Interpretability, RLHF, alignment research',                   70),
    ('ml-theory',          'ML Theory & Foundations',        'Optimization, scaling laws, learning theory',                  80),
    ('ml-infra',           'ML Infra & Systems',             'Training stacks, distributed compute, inference systems',      90),
    ('efficient-inference','Efficient Inference',            'Quantization, distillation, on-device models',                100),
    ('oss-tools',          'Open-Source ML Tools',           'Library and framework maintainers',                           110),
    ('founders',           'AI Founders & Builders',         'Startup CEOs and hands-on company builders',                  120),
    ('lab-leaders',        'AI Lab Leadership',              'Directors at OpenAI, DeepMind, Anthropic, Meta AI, Mistral…', 130),
    ('educators',          'Educators & Bloggers',           'Course creators, explainers, prolific writers',               140),
    ('pioneers',           'Pioneers & Laureates',           'Turing-award and field-defining figures',                     150)
ON CONFLICT (slug) DO UPDATE
    SET name        = EXCLUDED.name,
        description = EXCLUDED.description,
        sort_order  = EXCLUDED.sort_order;

-- ── Companies & labs ─────────────────────────────────────────────────
-- Institutional affiliations. Sort order leaves a 1000-gap from the
-- topic ontology so anything that needs to come between fits.
INSERT INTO categories (slug, name, description, sort_order, kind) VALUES
    ('openai',          'OpenAI',           'Researchers and engineers at OpenAI',                                 1010, 'company'),
    ('anthropic',       'Anthropic',        'Researchers and engineers at Anthropic',                              1020, 'company'),
    ('google-deepmind', 'Google DeepMind',  'Researchers across Google DeepMind, Brain, and Research',             1030, 'company'),
    ('meta-ai',         'Meta AI',          'Researchers at Meta AI / FAIR',                                       1040, 'company'),
    ('mistral-ai',      'Mistral AI',       'Mistral AI team',                                                     1050, 'company'),
    ('hugging-face',    'Hugging Face',     'Hugging Face engineers and researchers',                              1060, 'company'),
    ('nvidia-research', 'NVIDIA Research',  'NVIDIA research scientists and engineers',                            1070, 'company'),
    ('microsoft-research','Microsoft Research','Researchers at MSR and Microsoft AI',                              1080, 'company'),
    ('apple-ml',        'Apple ML',         'Machine learning teams at Apple',                                     1090, 'company'),
    ('xai',             'xAI',              'xAI researchers and engineers',                                       1100, 'company'),
    ('cohere',          'Cohere',           'Cohere team',                                                         1110, 'company'),
    ('stability-ai',    'Stability AI',     'Stability AI past and present researchers',                           1120, 'company'),
    ('perplexity',      'Perplexity',       'Perplexity engineers and researchers',                                1130, 'company'),
    ('allen-ai',        'Allen AI',         'Allen Institute for AI researchers',                                  1140, 'company'),
    ('eleuther-ai',     'EleutherAI',       'EleutherAI open research collective',                                 1150, 'company'),
    ('lighton',         'LightOn',          'LightOn researchers and engineers',                                   1160, 'company'),
    ('stanford-ai',     'Stanford AI',      'Stanford AI faculty, CRFM, HAI',                                      1210, 'company'),
    ('berkeley-ai',     'UC Berkeley AI',   'UC Berkeley AI Research (BAIR)',                                      1220, 'company'),
    ('mit-csail',       'MIT CSAIL',        'MIT CSAIL and AI faculty',                                            1230, 'company'),
    ('cmu-ai',          'CMU AI',           'Carnegie Mellon University AI / ML faculty',                          1240, 'company')
ON CONFLICT (slug) DO UPDATE
    SET name        = EXCLUDED.name,
        description = EXCLUDED.description,
        sort_order  = EXCLUDED.sort_order,
        kind        = EXCLUDED.kind;
