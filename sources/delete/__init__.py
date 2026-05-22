"""Offline purge jobs.

Soft-deleted rows are flagged ``documents.to_delete = TRUE`` by the API
(when a user removes an originating source from their profile). The
modules in this package walk those tombstones, drop the matching
entries from each user's ColBERT index, and delete the rows from PG.
"""
