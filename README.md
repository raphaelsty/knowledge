# Knowledge — VIP data snapshots

Daily snapshot of all **VIP** users and their public documents. Maintained
automatically by [`.github/workflows/backup-vip-data.yml`](https://github.com/raphaelsty/knowledge/blob/main/.github/workflows/backup-vip-data.yml)
on the `main` branch.

The orphan `data` branch has no shared history with `main`. Code lives on
`main`; only snapshots live here. Clone just this branch:

```bash
git clone --single-branch --branch data https://github.com/raphaelsty/knowledge.git
```

## Layout

```
data/users.jsonl       # one VIP per line, sorted by slug
data/documents.jsonl   # one VIP-owned doc per line, sorted by (slug, url)
data/favorites.jsonl   # VIP upvotes
data/follows.jsonl     # VIP → VIP follow edges
```

Each file is regenerated in full on every run; Git's pack-file delta
compression keeps the repo growth at a few hundred KB per day. To restore
state at any past date:

```bash
git checkout <yyyy-mm-dd-tag>   # or browse commit history
```

## What is exported

Only columns that are already publicly visible on
[knowledge-web.org](https://knowledge-web.org). Specifically **NOT**
exported:

- Any non-VIP user (`users.vip = FALSE`)
- Passwords, session tokens, OAuth identities, email-verification tokens
- Private settings (`users.sources` may contain credentials → excluded)
- Encrypted Twitter cookies, HackerNews passwords, Zotero API keys
- `events`, `auth_sessions`, `api_tokens`, anything in `*_secrets`
- Soft-deleted rows (`documents.deleted = TRUE`)

The exporter uses an explicit allow-list of columns in
[`scripts/export-vip-data.sh`](https://github.com/raphaelsty/knowledge/blob/main/scripts/export-vip-data.sh) — never `SELECT *`.

## Cadence

One commit per day at 03:00 UTC via GitHub Actions.
