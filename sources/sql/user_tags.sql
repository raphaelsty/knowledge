-- Tag vocabulary for a single user.
--
-- Returns the alphabetical set of `tags` (curated, not auto-generated
-- `extra_tags`) attached to one user's documents. The pipeline unions
-- this with `get_vip_tags` so a non-VIP personality still draws from
-- the cross-user VIP pool plus their own personal tags.

SELECT DISTINCT t
  FROM documents,
       unnest(tags) AS t
 WHERE user_id = %s
   AND t <> ''
 ORDER BY t;
