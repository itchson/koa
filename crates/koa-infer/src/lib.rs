use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::Tokenizer;

pub const REQUIRED_MODEL_ID: &str = "google/gemma-4-E2B-it";

#[derive(Debug, Error)]
pub enum InferError {
    #[error("native Gemma 4 executor is not complete; Koa refuses to fabricate model output")]
    ExecutorUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub revision: String,
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceDoctor {
    pub model_dir: PathBuf,
    pub model_id_ok: bool,
    pub revision_pinned: bool,
    pub manifest_ok: bool,
    pub config_ok: bool,
    pub tokenizer_ok: bool,
    pub safetensors_ok: bool,
    pub cuda_requested: bool,
    pub cuda_compiled: bool,
    pub problems: Vec<String>,
}

impl InferenceDoctor {
    pub fn healthy(&self) -> bool {
        self.problems.is_empty()
            && self.model_id_ok
            && self.revision_pinned
            && self.manifest_ok
            && self.config_ok
            && self.tokenizer_ok
            && self.safetensors_ok
            && (!self.cuda_requested || self.cuda_compiled)
    }
}

pub struct NativeGemmaEngine {
    model_dir: PathBuf,
    manifest: ModelManifest,
    tokenizer: Tokenizer,
}

impl NativeGemmaEngine {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self> {
        let model_dir = model_dir.into();
        let manifest = load_manifest(&model_dir)?;
        validate_manifest(&model_dir, &manifest)?;
        validate_config(&model_dir)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {err}",
                tokenizer_path.display()
            )
        })?;
        validate_safetensors(&model_dir, &manifest)?;
        Ok(Self {
            model_dir,
            manifest,
            tokenizer,
        })
    }

    pub fn doctor(model_dir: impl Into<PathBuf>, require_cuda: bool) -> InferenceDoctor {
        let model_dir = model_dir.into();
        let mut doctor = InferenceDoctor {
            model_dir: model_dir.clone(),
            model_id_ok: false,
            revision_pinned: false,
            manifest_ok: false,
            config_ok: false,
            tokenizer_ok: false,
            safetensors_ok: false,
            cuda_requested: require_cuda,
            cuda_compiled: cfg!(feature = "cuda"),
            problems: Vec::new(),
        };

        match load_manifest(&model_dir) {
            Ok(manifest) => {
                doctor.manifest_ok = true;
                doctor.model_id_ok = manifest.model_id == REQUIRED_MODEL_ID;
                doctor.revision_pinned = is_pinned_revision(&manifest.revision);
                if !doctor.model_id_ok {
                    doctor.problems.push(format!(
                        "model id must be {REQUIRED_MODEL_ID}, found {}",
                        manifest.model_id
                    ));
                }
                if !doctor.revision_pinned {
                    doctor.problems.push(
                        "model revision must be an exact pinned revision, not `main`, `latest`, or `pin-required`"
                            .to_string(),
                    );
                }
                if let Err(err) = validate_manifest(&model_dir, &manifest) {
                    doctor.problems.push(err.to_string());
                }
                if let Err(err) = validate_safetensors(&model_dir, &manifest) {
                    doctor.problems.push(err.to_string());
                } else {
                    doctor.safetensors_ok = true;
                }
            }
            Err(err) => doctor.problems.push(err.to_string()),
        }

        if let Err(err) = validate_config(&model_dir) {
            doctor.problems.push(err.to_string());
        } else {
            doctor.config_ok = true;
        }

        match Tokenizer::from_file(model_dir.join("tokenizer.json")) {
            Ok(_) => doctor.tokenizer_ok = true,
            Err(err) => doctor
                .problems
                .push(format!("tokenizer check failed: {err}")),
        }

        if require_cuda && !doctor.cuda_compiled {
            doctor
                .problems
                .push("koa-infer was not compiled with the `cuda` feature".to_string());
        }
        doctor
    }

    pub fn generate(&mut self, request: InferenceRequest) -> Result<InferenceResponse> {
        if request.prompt.trim().is_empty() {
            bail!("prompt cannot be empty");
        }
        if request.max_new_tokens == 0 {
            bail!("max_new_tokens must be greater than zero");
        }
        let encoding = self
            .tokenizer
            .encode(request.prompt, true)
            .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;
        let _prompt_tokens = encoding.get_ids().len();
        let _ = &self.model_dir;
        let _ = &self.manifest;
        Err(InferError::ExecutorUnavailable.into())
    }
}

pub fn load_manifest(model_dir: &Path) -> Result<ModelManifest> {
    let path = model_dir.join("koa-model.toml");
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let manifest: ModelManifest =
        toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    if manifest.files.is_empty() {
        bail!("model manifest must include at least one checked file");
    }
    Ok(manifest)
}

fn validate_manifest(model_dir: &Path, manifest: &ModelManifest) -> Result<()> {
    if manifest.model_id != REQUIRED_MODEL_ID {
        bail!(
            "model manifest id must be {REQUIRED_MODEL_ID}, found {}",
            manifest.model_id
        );
    }
    if !is_pinned_revision(&manifest.revision) {
        bail!("model manifest revision must be pinned to an exact immutable revision");
    }
    for file in &manifest.files {
        let path = model_dir.join(&file.path);
        let actual = sha256_file(&path)?;
        if actual != file.sha256 {
            bail!(
                "checksum mismatch for {}: expected {}, got {}",
                path.display(),
                file.sha256,
                actual
            );
        }
    }
    Ok(())
}

fn validate_config(model_dir: &Path) -> Result<()> {
    let path = model_dir.join("config.json");
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let value: Value = serde_json::from_str(&text)?;
    let model_type = value
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model_type != "gemma4" {
        bail!("config.json model_type must be `gemma4`, found `{model_type}`");
    }
    let text_type = value
        .get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if text_type != "gemma4_text" {
        bail!("config.json text_config.model_type must be `gemma4_text`, found `{text_type}`");
    }
    Ok(())
}

fn validate_safetensors(model_dir: &Path, manifest: &ModelManifest) -> Result<()> {
    let mut found = false;
    for file in &manifest.files {
        if !file.path.ends_with(".safetensors") {
            continue;
        }
        found = true;
        let path = model_dir.join(&file.path);
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        SafeTensors::deserialize(&bytes)
            .with_context(|| format!("invalid safetensors file {}", path.display()))?;
    }
    if !found {
        bail!("model manifest does not include any .safetensors weight files");
    }
    Ok(())
}

fn is_pinned_revision(revision: &str) -> bool {
    let revision = revision.trim();
    !revision.is_empty()
        && revision != "main"
        && revision != "latest"
        && revision != "pin-required"
        && revision.len() >= 12
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_revision_is_rejected() {
        assert!(!is_pinned_revision("main"));
        assert!(!is_pinned_revision("pin-required"));
        assert!(is_pinned_revision("abc123def456"));
    }
}
