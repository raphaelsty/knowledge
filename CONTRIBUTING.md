# Contributing

Thanks for taking the time. Knowledge is a small project, so the rules are
short.

## Quick start

```bash
git clone https://github.com/raphaelsty/knowledge && cd knowledge
make install-dev           # creates .venv, installs ruff/mypy/pre-commit
uvx pre-commit install     # arms the git hook (only needed once)
make up                    # starts postgres locally via docker compose
```

You're good to go. `make serve` runs the Rust API on `:8080`,
`make web` serves the static frontend on `:3000`, and `make run` walks the
pipeline (fetch → clean → tag → write) for whatever's in your local PG.

## How code style is enforced

There is exactly one source of truth: `.pre-commit-config.yaml`.

- **Locally**: the pre-commit hook runs on every `git commit`. If it fails,
  fix what it tells you and commit again. Don't bypass with `--no-verify`.
- **In CI**: GitHub Actions runs the *same* hooks across the whole tree
  (`uvx pre-commit run --all-files`). What passes locally passes in CI.

If you want to bump a tool version, change it in `.pre-commit-config.yaml`
only — CI picks it up automatically. There are no other lint jobs to keep
in sync.

What runs:
- `ruff` (lint + format) for Python
- `mypy` for Python type-checks
- `prettier` for `web/` JS / CSS / HTML
- `cargo fmt` + `cargo clippy -D warnings` for `api/`
- `pre-commit-hooks` for trailing whitespace, line endings, large files,
  YAML syntax

## Commits

- Lowercase imperative subjects, scoped where it helps:
  `caddy: bake web/ into a custom image`.
- Explain *why* in the body, not what — the diff already shows what.
- No `Co-Authored-By: Claude` trailers.
- Don't amend or force-push to `main` unless you really mean it.

## Deploys

`main` is the live branch. Pushing to `origin/main` triggers Dokploy on the
Hetzner box, which rebuilds and redeploys via the GitHub webhook
(~1-2 min). Watch the build in the Dokploy UI at
[dokploy.knowledge-web.org](https://dokploy.knowledge-web.org). Rollbacks
happen there too — every push leaves a deployment record.

For changes that touch the prod stack (compose / Caddy / migrations),
verify locally with `make up` first.

## License

Contributions are licensed under [PolyForm Noncommercial 1.0.0](LICENSE).
By opening a pull request you're agreeing your changes ship under the
same license. Commercial use of the project (including any contributed
code) requires explicit permission from the maintainer.

## Reporting security issues

Found a vulnerability? Email `raphael.sourty@lighton.ai` — please don't
open a public issue. See [`/.well-known/security.txt`](https://knowledge-web.org/.well-known/security.txt).
