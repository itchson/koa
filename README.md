# Koa Runtime

Koa is a Rust-first, CLI-first agentic runtime. The v0.0.1 implementation is intentionally strict: it does not ship mock inference, mock tools, or silent fallbacks.

The first native slice contains:

- Koa workspace/session state.
- A vault with raw/wiki/schema/log structure.
- A unified toolbox manifest and audit log.
- Dynamic skill and subagent specs.
- Native Gemma 4 E2B-it model asset validation.
- WSL2/Linux capsule readiness checks and capsule metadata.

`koa chat` requires a passing native inference gate. Until pinned Gemma 4 E2B-it assets and the native executor are present, it fails loudly instead of fabricating a response.

## Quick Start

```powershell
cargo run -p koa-cli -- init
cargo run -p koa-cli -- doctor
cargo run -p koa-cli -- model doctor
cargo run -p koa-cli -- capsule doctor
```

Useful commands:

```powershell
cargo run -p koa-cli -- model manifest --revision <exact-40-character-hugging-face-commit-sha>
cargo run -p koa-cli -- model verify
cargo run -p koa-cli -- session list
cargo run -p koa-cli -- vault ingest .\README.md
cargo run -p koa-cli -- vault search Koa
cargo run -p koa-cli -- skill create research.notes "Research Notes" "Capture sourced task research" "Use approved toolbox tools and record provenance."
cargo run -p koa-cli -- agent spawn reviewer "Reviewer" "Review a plan or patch" --output-contract "result.summary, findings, risks, next_actions"
```

Architecture notes live in `docs/architecture.md`. Model asset requirements live in `docs/model-assets.md`. Capsule notes live in `docs/capsules.md`.

## Repository

Remote target: <https://github.com/itchson/koa>
