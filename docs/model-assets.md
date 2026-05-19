# Native Gemma 4 E2B-it Assets

`koa-infer` expects pinned local model assets under `.koa/models/gemma-4-E2B-it` by default.

Required files:

- `koa-model.toml`
- `config.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `chat_template.jinja`
- `generation_config.json`
- `processor_config.json`
- `model.safetensors`

The current live Hugging Face shape for `google/gemma-4-E2B-it` uses a single `model.safetensors` file, not sharded weights. As of the latest checked Hub state, `main` pointed at commit `905e84b50c4d2a365ebde34e685027578e6728db`; Koa still requires an explicit pinned 40-character commit SHA in your local manifest.

Example `koa-model.toml` shape:

```toml
model_id = "google/gemma-4-E2B-it"
revision = "0123456789abcdef0123456789abcdef01234567"

[[files]]
path = "config.json"
sha256 = "..."

[[files]]
path = "tokenizer_config.json"
sha256 = "..."

[[files]]
path = "tokenizer.json"
sha256 = "..."

[[files]]
path = "chat_template.jinja"
sha256 = "..."

[[files]]
path = "generation_config.json"
sha256 = "..."

[[files]]
path = "processor_config.json"
sha256 = "..."

[[files]]
path = "model.safetensors"
sha256 = "..."
```

Floating revisions such as `main`, `latest`, tags, short SHAs, or `pin-required` are rejected. If any file is missing or the checksum differs, Koa fails the inference gate.

After placing the required files in the model directory, write a pinned manifest with:

```powershell
cargo run -p koa-cli -- model manifest --revision <exact-40-character-hugging-face-commit-sha>
```

Verify the manifest and checksums with:

```powershell
cargo run -p koa-cli -- model verify
```

`koa model verify` exits with status 2 when the model directory, manifest, pinned revision, required file declarations, checksums, or safetensors files are invalid.
