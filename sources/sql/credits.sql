-- Credit-billing schema (v1 — top-up packs only).
--
-- Source of truth is `credit_events` (a ledger). Every change to a
-- user's balance is a row: top-ups land as positive `delta`,
-- operations debit with negative `delta`. The denormalised
-- `balance_after` column on every row gives an O(1) lookup for the
-- current balance without scanning the whole ledger.
--
-- Polar.sh (https://polar.sh) handles the checkout / merchant-of-
-- record side. We never touch payment instruments — the user is
-- redirected to a Polar-hosted checkout, then Polar fires a webhook
-- (`checkout.updated` / `order.created`) which we verify and turn
-- into a positive `credit_events` row.
--
-- v1 does NOT debit anything on operations. The schema + plumbing
-- are in place so adding debits is a single function call.

CREATE TABLE IF NOT EXISTS credit_events (
    id              BIGSERIAL   PRIMARY KEY,
    user_id         BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Signed delta. Positive = credit added (top-up, refund, manual
    -- adjustment); negative = credit spent (twitter fetch, storage,
    -- pipeline run, etc.).
    delta           INTEGER     NOT NULL,
    -- Denormalised running balance immediately after this event,
    -- maintained by the helper functions / atomic SQL. Trades a
    -- couple of bytes per row for instant "current balance" reads
    -- and easy auditing.
    balance_after   INTEGER     NOT NULL CHECK (balance_after >= 0),
    -- Free-text identifier for the kind of event:
    --   top_up                — Polar order webhook
    --   refund                — Polar refund webhook
    --   manual_adjustment     — admin grant / clawback
    --   debit:twitter-api     — Twitter API page fetched (per page)
    --   debit:storage         — storage rent for a billing period
    --   debit:pipeline-run    — one full pipeline cycle
    kind            TEXT        NOT NULL,
    -- Polar event id for top-ups / refunds. UNIQUE so a re-delivered
    -- webhook can't double-credit the same user (idempotency key).
    polar_event_id  TEXT        UNIQUE,
    -- Arbitrary JSON: e.g.
    --   {"polar_order_id": "...", "amount_minor": 1000, "currency": "EUR"}
    --   {"pages": 4, "endpoint": "/twitter/user/last_tweets"}
    meta            JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_credit_events_user_created
    ON credit_events (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_credit_events_kind
    ON credit_events (kind);

COMMENT ON TABLE  credit_events IS
    'Ledger of every credit movement. Balance is SUM(delta) per user; balance_after is denormalised for O(1) reads.';
COMMENT ON COLUMN credit_events.delta         IS 'Signed integer. Positive = credit added; negative = credit spent.';
COMMENT ON COLUMN credit_events.balance_after IS 'Running balance immediately after this event. CHECK constraint guards against debits below zero.';
COMMENT ON COLUMN credit_events.kind          IS 'Event taxonomy. See sources/sql/credits.sql header for the supported values.';
COMMENT ON COLUMN credit_events.polar_event_id IS 'Polar webhook event id; UNIQUE so re-deliveries cannot double-credit.';

-- ── Mapping users ↔ Polar customers ─────────────────────────────────
-- The first time a user starts a checkout, we create a Polar
-- customer for them via the Polar API and store the returned id here
-- so subsequent purchases reuse the same customer record.
CREATE TABLE IF NOT EXISTS polar_customers (
    user_id           BIGINT     PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    polar_customer_id TEXT       NOT NULL UNIQUE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE polar_customers IS
    'Maps an internal user_id to its Polar customer id. Populated on the first checkout request.';

-- ── Atomic balance helpers ──────────────────────────────────────────
-- Append-only ledger functions. Both run inside a transaction and
-- use SELECT FOR UPDATE on the user row to serialise concurrent
-- credit changes — debits issued in parallel won't oversell the
-- balance.

-- Current balance for a user (0 when no events yet).
CREATE OR REPLACE FUNCTION credits_balance(p_user_id BIGINT)
RETURNS INTEGER
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(
        (SELECT balance_after
           FROM credit_events
          WHERE user_id = p_user_id
          ORDER BY id DESC
          LIMIT 1),
        0
    );
$$;

-- Append a top-up event. Caller is expected to dedupe by
-- `polar_event_id` BEFORE calling (the UNIQUE constraint will also
-- reject duplicates as a safety net). Returns the new balance.
CREATE OR REPLACE FUNCTION credits_top_up(
    p_user_id        BIGINT,
    p_amount         INTEGER,
    p_kind           TEXT,
    p_polar_event_id TEXT,
    p_meta           JSONB
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    current_balance INTEGER;
    new_balance     INTEGER;
BEGIN
    IF p_amount <= 0 THEN
        RAISE EXCEPTION 'top-up amount must be positive, got %', p_amount;
    END IF;
    -- Lock the user row so concurrent debits/top-ups serialise.
    PERFORM 1 FROM users WHERE id = p_user_id FOR UPDATE;
    current_balance := credits_balance(p_user_id);
    new_balance     := current_balance + p_amount;
    INSERT INTO credit_events (user_id, delta, balance_after, kind, polar_event_id, meta)
    VALUES (p_user_id, p_amount, new_balance, p_kind, p_polar_event_id, COALESCE(p_meta, '{}'::jsonb));
    RETURN new_balance;
END;
$$;

-- Append a debit event. Returns the new balance, or NULL when the
-- user has insufficient credits (no row inserted in that case).
CREATE OR REPLACE FUNCTION credits_debit(
    p_user_id BIGINT,
    p_amount  INTEGER,
    p_kind    TEXT,
    p_meta    JSONB
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    current_balance INTEGER;
    new_balance     INTEGER;
BEGIN
    IF p_amount <= 0 THEN
        RAISE EXCEPTION 'debit amount must be positive, got %', p_amount;
    END IF;
    PERFORM 1 FROM users WHERE id = p_user_id FOR UPDATE;
    current_balance := credits_balance(p_user_id);
    IF current_balance < p_amount THEN
        RETURN NULL;
    END IF;
    new_balance := current_balance - p_amount;
    INSERT INTO credit_events (user_id, delta, balance_after, kind, meta)
    VALUES (p_user_id, -p_amount, new_balance, p_kind, COALESCE(p_meta, '{}'::jsonb));
    RETURN new_balance;
END;
$$;
