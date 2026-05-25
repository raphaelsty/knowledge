-- Database-wide safety nets. Idempotent (`ALTER DATABASE` overwrites
-- the prior value silently); applied on every API boot via the
-- include_str! migration list.
--
-- All settings take effect on NEW connections after this runs — the
-- current migration session inherits the old defaults, which is fine.
--
-- ────────────────────────────────────────────────────────────────────
-- Problem this prevents
-- ────────────────────────────────────────────────────────────────────
--
-- May 2026: a pipeline daemon's INSERT INTO documents transaction
-- was open when Dokploy killed the old API container. The daemon's
-- TCP socket went dead, but PG sat for 44 minutes waiting for the
-- next query (state = `ClientRead`, holding a row-exclusive lock on
-- `documents`). The new API container's boot-time migration tried
-- to acquire the same lock and hung indefinitely, never reaching
-- the `listen` step. Healthcheck never passed → Dokploy never
-- attached the container to its router network → Traefik 404'd
-- every request for 50 minutes.
--
-- The three settings below catch the failure at three different
-- layers so we never see this again.
-- ────────────────────────────────────────────────────────────────────

-- (1) Auto-terminate sessions that hold a transaction open without
-- doing anything for > 5 minutes. Real pipeline work (a sweep over
-- all VIPs, the feed-snapshot rebuild) completes well inside this
-- window; an orphan transaction caused by a dropped client crosses
-- it almost immediately. PG-15+ name; safe on PG 16.
ALTER DATABASE knowledge
    SET idle_in_transaction_session_timeout = '5min';

-- (2) TCP keepalives on the server's side of every PG connection.
-- Default `tcp_keepalives_idle = 0` defers to the kernel (Linux
-- default 7200 s = 2 hours), which is far too lax for a service that
-- restarts its clients on every deploy. With these settings PG sends
-- the first probe after 60 s of silence, retries every 10 s up to 3
-- times, and tears the connection down after ~90 s if the peer is
-- unreachable. The held locks roll back at that point.
ALTER DATABASE knowledge SET tcp_keepalives_idle      = 60;
ALTER DATABASE knowledge SET tcp_keepalives_interval  = 10;
ALTER DATABASE knowledge SET tcp_keepalives_count     = 3;
