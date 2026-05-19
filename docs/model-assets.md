# Native Gemma 4 E2B-it Assets

`koa-infer` expects pinned local model assets under `.koa/models/gemma-4-E2B-it` by default.

Required files:

- `koa-model.toml`
- `config.json`
- `tokenizer.json`
- one or more `.safetensors` weight files

Example `koa-model.toml` shape:

```toml
model_id = "google/gemma-4-E2B-it"
revision = "exact-hugging-face-commit-sha"

[[files]]
path = "config.json"
sha256 = "..."

[[files]]
path = "tokenizer.json"
sha256 = "..."

[[files]]
path = "model-00001-of-00002.safetensors"
sha256 = "..."
```

Floating revisions such as `main`, `latest`, or `pin-required` are rejected. If any file is missing or the checksum differs, Koa fails the inference gate.
