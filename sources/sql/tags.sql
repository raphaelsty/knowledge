-- Shared tag vocabulary across every user's documents.
--
-- Returns the alphabetical union of `tags` (the curated/user-tagged set —
-- `extra_tags` are excluded so we don't amplify a previous run's TF-IDF
-- suggestions). Used by the pipeline to seed `get_extra_tags` so each
-- personality draws auto-tags from a site-wide universe rather than
-- their own local one.

SELECT DISTINCT t
  FROM documents,
       unnest(tags) AS t
 WHERE t <> ''
 ORDER BY t;
