-- Tag vocabulary contributed by VIP users.
--
-- Returns the alphabetical union of `tags` (curated, not auto-generated
-- `extra_tags`) across every VIP user's documents. Non-VIP users are
-- excluded by design — their tags should not seed other people's tag
-- vocabularies. Each personality's own tags are unioned in separately
-- by the pipeline (see `get_user_tags`).

SELECT DISTINCT t
  FROM documents d
  JOIN users u ON u.id = d.user_id
       AND u.vip = TRUE,
       unnest(d.tags) AS t
 WHERE t <> ''
 ORDER BY t;
