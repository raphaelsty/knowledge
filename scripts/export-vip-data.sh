#!/usr/bin/env bash
#
# Export VIP users + their documents from prod PG to sorted JSONL files.
# Called once a day by the `backup-vip-data` GitHub Action.
#
# Every query below uses an EXPLICIT column allow-list — never SELECT *.
# Sensitive columns (email, password_hash, OAuth tokens, sources JSON
# which can carry Twitter cookies / HN passwords / Zotero keys, all
# email-verification + password-reset tokens) are deliberately NOT
# listed and therefore NEVER leave prod.
#
# Output is sorted by primary key so consecutive daily commits produce
# small git pack-file deltas.
#
# Env:
#   SSH_HOST   — `user@host`. Default: root@65.21.111.133 (Hetzner CX33)
#   SSH_KEY    — path to private key. Default: ~/.ssh/id_ed25519
#   OUT_DIR    — where to drop the .jsonl files. Default: ./data
#   PG_CTR     — prod postgres container name. Default discovered via
#                `docker ps`.

set -euo pipefail

SSH_HOST=${SSH_HOST:-root@65.21.111.133}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
OUT_DIR=${OUT_DIR:-./data}
PG_CTR=${PG_CTR:-knowledge-prod-gjqqg2-postgres-1}

mkdir -p "$OUT_DIR"

# Run a SQL query against prod PG via docker exec on the host.
# - `-A` unaligned output (no padding spaces)
# - `-t` tuples only (no header / row count)
# - `-q` quiet (no NOTICE noise)
# - `-X` skip .psqlrc
# - `-v ON_ERROR_STOP=1` fail fast if the query is malformed
# Each `SELECT json_build_object(...)` emits one row per line — perfect
# JSONL when piped to a file.
run_sql() {
    local sql=$1
    ssh -i "$SSH_KEY" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "$SSH_HOST" \
        "docker exec -i $PG_CTR psql -U knowledge -d knowledge -X -A -t -q -v ON_ERROR_STOP=1" \
        <<EOF
$sql
EOF
}

echo "→ exporting users.jsonl"
run_sql "
SELECT json_build_object(
  'slug',              u.username,
  'name',              u.name,
  'description',       u.description,
  'avatar',            u.avatar,
  'links',             u.links,
  'twitter_followers', u.twitter_followers,
  'github_followers',  u.github_followers,
  'citations',         u.citations,
  'tweet_newest_date', to_char(u.tweet_newest_date, 'YYYY-MM-DD'),
  'tweet_oldest_date', to_char(u.tweet_oldest_date, 'YYYY-MM-DD'),
  'interest_topics',   u.interest_topics,
  'created_at',        to_char(u.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SSZ')
)::text
  FROM users u
 WHERE u.vip = TRUE
 ORDER BY u.username;
" > "$OUT_DIR/users.jsonl"

echo "→ exporting documents/ (one file per VIP slug)"
# Documents per-slug rather than one giant file:
#   * Each VIP's library is its own .jsonl, so daily commits diff only
#     the slugs that gained new docs — pack-file deltas stay tiny.
#   * No single file approaches GitHub's 100 MB per-file warning even
#     for the biggest libraries (Karpathy-tier ≈ 30 MB).
# Wipe the dir first so VIPs that lose their last doc disappear cleanly.
rm -rf "$OUT_DIR/documents"
mkdir -p "$OUT_DIR/documents"

# Emit `<slug>\t<json-line>` on a single SSH round-trip, then split
# locally by the leading slug column. awk handles the fan-out in one
# pass — no per-VIP query.
run_sql "
SELECT u.username || E'\\t' || json_build_object(
  'url',                d.url,
  'canonical_url',      d.canonical_url,
  'title',              d.title,
  'clean_title',        d.clean_title,
  'summary',            d.summary,
  'clean_summary',      d.clean_summary,
  'date',               to_char(d.date, 'YYYY-MM-DD'),
  'tags',               d.tags,
  'extra_tags',         d.extra_tags,
  'source',             d.source,
  'source_url',         d.source_url,
  'urls',               d.urls,
  'linked_urls',        d.linked_urls,
  'link_hosts',         d.link_hosts,
  'citation_count',     d.citation_count,
  'twitter_likes',      d.twitter_likes,
  'twitter_retweets',   d.twitter_retweets,
  'twitter_replies',    d.twitter_replies,
  'twitter_quotes',     d.twitter_quotes,
  'twitter_views',      d.twitter_views,
  'twitter_bookmarks',  d.twitter_bookmarks,
  'created_at',         to_char(d.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SSZ')
)::text
  FROM documents d
  JOIN users u ON u.id = d.user_id
 WHERE u.vip = TRUE
   AND d.deleted = FALSE
   AND d.public  = TRUE
 ORDER BY u.username, d.url;
" | awk -F'\t' -v dir="$OUT_DIR/documents" '
    { slug = $1; sub(/^[^\t]*\t/, ""); print > (dir "/" slug ".jsonl") }
'

echo "→ exporting favorites.jsonl"
run_sql "
SELECT json_build_object(
  'user_slug',  u.username,
  'url',        f.url,
  'created_at', to_char(f.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SSZ')
)::text
  FROM favorite_documents f
  JOIN users u ON u.id = f.user_id
 WHERE u.vip = TRUE
 ORDER BY u.username, f.url;
" > "$OUT_DIR/favorites.jsonl"

echo "→ exporting follows.jsonl"
run_sql "
SELECT json_build_object(
  'follower_slug', f_user.username,
  'followed_slug', t_user.username,
  'created_at',    to_char(fo.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SSZ')
)::text
  FROM follows fo
  JOIN users f_user ON f_user.id = fo.follower_id
  JOIN users t_user ON t_user.id = fo.followed_id
 WHERE f_user.vip = TRUE
   AND t_user.vip = TRUE
 ORDER BY f_user.username, t_user.username;
" > "$OUT_DIR/follows.jsonl"

echo "→ done"
wc -l "$OUT_DIR"/*.jsonl
du -h "$OUT_DIR"/*.jsonl
