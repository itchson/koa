# Koa Runtime v0.0.1 Architecture

Koa v0.0.1 is a native-first runtime substrate. It deliberately fails closed when required native pieces are missing.

## Runtime Loop

1. `koa init` creates `.koa`, the vault, toolbox manifest, session registry, capsule metadata, and audit log.
2. `koa doctor` verifies the native inference gate, capsule substrate, vault, and toolbox.
3. `koa chat` refuses to run unless the native inference gate passes. It never emits canned or fallback model text.
4. Session, vault, toolbox, skill, and agent commands operate on real local artifacts and append audit events.

## Crates

- `koa-cli`: command-line surface.
- `koa-core`: workspace state, command orchestration, audit, and persistence glue.
- `koa-infer`: Gemma 4 E2B-it asset validation and native executor boundary.
- `koa-capsule`: Linux namespace, overlayfs, cgroup, and seccomp capsule primitives.
- `koa-context`: filesystem vault ingest/search/lint.
- `koa-store`: SQLite + FTS5 document store.
- `koa-toolbox`: tool specs, policy, and audit primitives.
- `koa-skill`: non-executable dynamic skill artifacts.
- `koa-agent`: agent specs and lifecycle registry.

## Hard-Fail Policy

Koa does not use mock inference, mock search, mock MCP servers, Docker fallback, Ollama fallback, or llama.cpp fallback. Missing substrate produces diagnostics through `koa doctor` or command errors.
