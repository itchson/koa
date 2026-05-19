use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::Tokenizer;

pub const REQUIRED_MODEL_ID: &str = "google/gemma-4-E2B-it";
pub const MODEL_MANIFEST_FILE: &str = "koa-model.toml";
pub const REQUIRED_CONFIG_FILE: &str = "config.json";
pub const REQUIRED_TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
pub const REQUIRED_TOKENIZER_FILE: &str = "tokenizer.json";
pub const REQUIRED_CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";
pub const REQUIRED_GENERATION_CONFIG_FILE: &str = "generation_config.json";
pub const REQUIRED_PROCESSOR_CONFIG_FILE: &str = "processor_config.json";
pub const REQUIRED_WEIGHTS_FILE: &str = "model.safetensors";
pub const REQUIRED_MODEL_ASSET_FILES: &[&str] = &[
    REQUIRED_CONFIG_FILE,
    REQUIRED_TOKENIZER_CONFIG_FILE,
    REQUIRED_TOKENIZER_FILE,
    REQUIRED_CHAT_TEMPLATE_FILE,
    REQUIRED_GENERATION_CONFIG_FILE,
    REQUIRED_PROCESSOR_CONFIG_FILE,
    REQUIRED_WEIGHTS_FILE,
];

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
pub struct ModelAssetReport {
    pub model_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest_ok: bool,
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub revision_pinned: bool,
    pub required_files_declared: bool,
    pub weight_files_declared: bool,
    pub safetensors_ok: bool,
    pub checked_files: Vec<ModelAssetCheck>,
    pub problems: Vec<String>,
}

impl ModelAssetReport {
    pub fn ok(&self) -> bool {
        self.manifest_ok
            && self.revision_pinned
            && self.required_files_declared
            && self.weight_files_declared
            && self.safetensors_ok
            && self.checked_files.iter().all(|file| file.ok)
            && self.problems.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssetCheck {
    pub path: String,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
    pub ok: bool,
    pub problem: Option<String>,
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

pub fn discover_model_files(model_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let model_dir = model_dir.as_ref();
    if !model_dir.is_dir() {
        bail!("model directory {} does not exist", model_dir.display());
    }

    let mut files = Vec::new();
    for required in REQUIRED_MODEL_ASSET_FILES {
        let path = model_dir.join(required);
        if !path.is_file() {
            bail!(
                "cannot create model manifest: required file `{required}` is missing from {}",
                model_dir.display()
            );
        }
        files.push(PathBuf::from(required));
    }

    let mut weights = Vec::new();
    for entry in fs::read_dir(model_dir)
        .with_context(|| format!("failed to read model directory {}", model_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            let name = path
                .file_name()
                .with_context(|| format!("failed to determine file name for {}", path.display()))?;
            if name != REQUIRED_WEIGHTS_FILE {
                weights.push(PathBuf::from(name));
            }
        }
    }
    weights.sort();
    files.extend(weights);
    Ok(files)
}

pub fn create_pinned_manifest(
    model_dir: impl AsRef<Path>,
    revision: &str,
    files: &[PathBuf],
) -> Result<ModelManifest> {
    let model_dir = model_dir.as_ref();
    if !model_dir.is_dir() {
        bail!("model directory {} does not exist", model_dir.display());
    }
    if !is_pinned_revision(revision) {
        bail!(
            "model revision must be pinned to an exact 40-character Hugging Face commit SHA, found `{}`",
            revision.trim()
        );
    }

    let files = if files.is_empty() {
        discover_model_files(model_dir)?
    } else {
        files.to_vec()
    };

    let mut seen = BTreeSet::new();
    let mut manifest_files = Vec::new();
    for file in files {
        let relative = normalize_model_file_path(model_dir, &file)?;
        if !seen.insert(relative.clone()) {
            bail!("duplicate model manifest file `{relative}`");
        }
        let path = model_dir.join(&relative);
        if !path.is_file() {
            bail!(
                "model manifest file `{relative}` does not exist at {}",
                path.display()
            );
        }
        manifest_files.push(ModelFile {
            path: relative,
            sha256: sha256_file(&path)?,
        });
    }

    let manifest = ModelManifest {
        model_id: REQUIRED_MODEL_ID.to_string(),
        revision: revision.trim().to_string(),
        files: manifest_files,
    };
    ensure_manifest_shape(&manifest)?;
    Ok(manifest)
}

pub fn write_pinned_manifest(
    model_dir: impl AsRef<Path>,
    revision: &str,
    files: &[PathBuf],
) -> Result<ModelManifest> {
    let model_dir = model_dir.as_ref();
    let manifest = create_pinned_manifest(model_dir, revision, files)?;
    let path = model_dir.join(MODEL_MANIFEST_FILE);
    fs::write(&path, toml::to_string_pretty(&manifest)?)
        .with_context(|| format!("failed to write model manifest {}", path.display()))?;
    Ok(manifest)
}

pub fn verify_model_assets(model_dir: impl Into<PathBuf>) -> ModelAssetReport {
    let model_dir = model_dir.into();
    let manifest_path = model_dir.join(MODEL_MANIFEST_FILE);
    let mut report = ModelAssetReport {
        model_dir: model_dir.clone(),
        manifest_path,
        manifest_ok: false,
        model_id: None,
        revision: None,
        revision_pinned: false,
        required_files_declared: false,
        weight_files_declared: false,
        safetensors_ok: false,
        checked_files: Vec::new(),
        problems: Vec::new(),
    };

    if !model_dir.is_dir() {
        report.problems.push(format!(
            "model directory {} does not exist",
            model_dir.display()
        ));
        return report;
    }

    let manifest = match load_manifest(&model_dir) {
        Ok(manifest) => manifest,
        Err(err) => {
            report.problems.push(err.to_string());
            return report;
        }
    };

    report.model_id = Some(manifest.model_id.clone());
    report.revision = Some(manifest.revision.clone());
    report.revision_pinned = is_pinned_revision(&manifest.revision);
    report.required_files_declared = REQUIRED_MODEL_ASSET_FILES
        .iter()
        .all(|file| manifest_declares_file(&manifest, file));
    report.weight_files_declared = manifest
        .files
        .iter()
        .any(|file| file.path.ends_with(".safetensors"));

    let shape_problems = manifest_shape_problems(&manifest);
    report.manifest_ok = shape_problems.is_empty();
    report.problems.extend(shape_problems);

    for file in &manifest.files {
        let check = checksum_model_file(&model_dir, file);
        if let Some(problem) = &check.problem {
            report.problems.push(problem.clone());
        }
        report.checked_files.push(check);
    }
    if report.manifest_ok && report.checked_files.iter().all(|file| file.ok) {
        match validate_safetensors(&model_dir, &manifest) {
            Ok(()) => report.safetensors_ok = true,
            Err(err) => report.problems.push(err.to_string()),
        }
    }

    report
}

pub fn ensure_model_assets(model_dir: impl Into<PathBuf>) -> Result<ModelAssetReport> {
    let report = verify_model_assets(model_dir);
    if !report.ok() {
        let problems = if report.problems.is_empty() {
            "unknown model asset verification failure".to_string()
        } else {
            report.problems.join("; ")
        };
        bail!(
            "model assets failed verification for {}: {problems}",
            report.model_dir.display()
        );
    }
    Ok(report)
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

        let assets = verify_model_assets(model_dir.clone());
        doctor.model_id_ok = assets.model_id.as_deref() == Some(REQUIRED_MODEL_ID);
        doctor.revision_pinned = assets.revision_pinned;
        doctor.manifest_ok = assets.manifest_ok && assets.checked_files.iter().all(|file| file.ok);
        doctor.safetensors_ok = assets.safetensors_ok;
        doctor.problems.extend(assets.problems);

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
    let path = model_dir.join(MODEL_MANIFEST_FILE);
    if !path.is_file() {
        bail!("model manifest {} is missing", path.display());
    }
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read model manifest {}", path.display()))?;
    let manifest: ModelManifest = toml::from_str(&text)
        .with_context(|| format!("failed to parse model manifest {}", path.display()))?;
    if manifest.files.is_empty() {
        bail!("model manifest must include at least one checked file");
    }
    Ok(manifest)
}

fn validate_manifest(model_dir: &Path, manifest: &ModelManifest) -> Result<()> {
    ensure_manifest_shape(manifest)?;
    for file in &manifest.files {
        let check = checksum_model_file(model_dir, file);
        if !check.ok {
            bail!(
                "{}",
                check
                    .problem
                    .unwrap_or_else(|| format!("checksum check failed for `{}`", check.path))
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
        let file =
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        // Mapping avoids loading multi-GB Gemma weight files into heap memory just to validate
        // the safetensors header and offsets.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        SafeTensors::deserialize(&mmap)
            .with_context(|| format!("invalid safetensors file {}", path.display()))?;
    }
    if !found {
        bail!("model manifest does not include any .safetensors weight files");
    }
    Ok(())
}

pub fn is_pinned_revision(revision: &str) -> bool {
    let revision = revision.trim();
    revision.len() == 40 && revision.chars().all(|ch| ch.is_ascii_hexdigit())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn ensure_manifest_shape(manifest: &ModelManifest) -> Result<()> {
    let problems = manifest_shape_problems(manifest);
    if !problems.is_empty() {
        bail!("{}", problems.join("; "));
    }
    Ok(())
}

fn manifest_shape_problems(manifest: &ModelManifest) -> Vec<String> {
    let mut problems = Vec::new();
    if manifest.model_id != REQUIRED_MODEL_ID {
        problems.push(format!(
            "model manifest id must be {REQUIRED_MODEL_ID}, found {}",
            manifest.model_id
        ));
    }
    if !is_pinned_revision(&manifest.revision) {
        problems.push(format!(
            "model manifest revision must be pinned to an exact 40-character Hugging Face commit SHA, found `{}`",
            manifest.revision.trim()
        ));
    }
    if manifest.files.is_empty() {
        problems.push("model manifest must include at least one checked file".to_string());
    }

    let mut seen = BTreeSet::new();
    for file in &manifest.files {
        if let Err(err) = validate_manifest_file_path(&file.path) {
            problems.push(err.to_string());
        } else if !seen.insert(file.path.clone()) {
            problems.push(format!("duplicate model manifest file `{}`", file.path));
        }
        if !is_sha256_hex(&file.sha256) {
            problems.push(format!(
                "sha256 for `{}` must be 64 hexadecimal characters",
                file.path
            ));
        }
    }

    for required in REQUIRED_MODEL_ASSET_FILES {
        if !manifest_declares_file(manifest, required) {
            problems.push(format!("model manifest must include `{required}`"));
        }
    }
    if !manifest
        .files
        .iter()
        .any(|file| file.path.ends_with(".safetensors"))
    {
        problems
            .push("model manifest must include at least one .safetensors weight file".to_string());
    }

    problems
}

fn manifest_declares_file(manifest: &ModelManifest, path: &str) -> bool {
    manifest.files.iter().any(|file| file.path == path)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn checksum_model_file(model_dir: &Path, file: &ModelFile) -> ModelAssetCheck {
    let mut check = ModelAssetCheck {
        path: file.path.clone(),
        expected_sha256: file.sha256.clone(),
        actual_sha256: None,
        ok: false,
        problem: None,
    };

    if let Err(err) = validate_manifest_file_path(&file.path) {
        check.problem = Some(err.to_string());
        return check;
    }
    if !is_sha256_hex(&file.sha256) {
        check.problem = Some(format!(
            "sha256 for `{}` must be 64 hexadecimal characters",
            file.path
        ));
        return check;
    }

    match sha256_file(&model_dir.join(&file.path)) {
        Ok(actual) => {
            check.actual_sha256 = Some(actual.clone());
            if actual.eq_ignore_ascii_case(&file.sha256) {
                check.ok = true;
            } else {
                check.problem = Some(format!(
                    "checksum mismatch for `{}`: expected {}, got {}",
                    file.path, file.sha256, actual
                ));
            }
        }
        Err(err) => {
            check.problem = Some(format!("failed to checksum `{}`: {err}", file.path));
        }
    }
    check
}

fn validate_manifest_file_path(path: &str) -> Result<()> {
    if path.trim().is_empty() {
        bail!("model manifest file path cannot be empty");
    }
    let path = Path::new(path);
    if path.is_absolute() {
        bail!(
            "model manifest file path `{}` must be relative to the model directory",
            path.display()
        );
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => bail!(
                "model manifest file path `{}` must be relative and cannot contain `.` or `..`",
                path.display()
            ),
        }
    }
    Ok(())
}

fn normalize_model_file_path(model_dir: &Path, path: &Path) -> Result<String> {
    let relative = if path.is_absolute() {
        path.strip_prefix(model_dir).with_context(|| {
            format!(
                "model file {} must be inside {}",
                path.display(),
                model_dir.display()
            )
        })?
    } else {
        path
    };

    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => {
                let part = part.to_str().with_context(|| {
                    format!("model file path {} is not valid UTF-8", path.display())
                })?;
                parts.push(part.to_string());
            }
            _ => bail!(
                "model file path {} must be relative and cannot contain `.` or `..`",
                path.display()
            ),
        }
    }
    if parts.is_empty() {
        bail!("model file path cannot be empty");
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_revision_is_rejected() {
        assert!(!is_pinned_revision("main"));
        assert!(!is_pinned_revision("latest"));
        assert!(!is_pinned_revision("pin-required"));
        assert!(!is_pinned_revision("abc123def456"));
        assert!(!is_pinned_revision("release-2026a"));
        assert!(!is_pinned_revision(
            "0123456789abcdef0123456789abcdef0123456"
        ));
        assert!(!is_pinned_revision(
            "0123456789abcdef0123456789abcdef012345678"
        ));
        assert!(!is_pinned_revision(
            "0123456789abcdef0123456789abcdef0123456g"
        ));
        assert!(is_pinned_revision(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(is_pinned_revision(
            "0123456789ABCDEF0123456789ABCDEF01234567"
        ));
    }

    #[test]
    fn pinned_manifest_can_be_written_and_verified() {
        let temp = tempfile::tempdir().expect("tempdir");
        seed_model_dir(temp.path());

        let revision = "0123456789abcdef0123456789abcdef01234567";
        let manifest = write_pinned_manifest(temp.path(), revision, &[]).expect("manifest");
        assert_eq!(manifest.model_id, REQUIRED_MODEL_ID);
        assert_eq!(manifest.revision, revision);
        assert_eq!(
            manifest
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>(),
            vec![
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "chat_template.jinja",
                "generation_config.json",
                "processor_config.json",
                "model.safetensors"
            ]
        );

        let report = verify_model_assets(temp.path());
        assert!(report.ok(), "{report:?}");
        assert!(report.manifest_ok);
        assert!(report.safetensors_ok);
        assert_eq!(report.checked_files.len(), 7);
    }

    #[test]
    fn manifest_writer_requires_complete_live_asset_set() {
        let temp = tempfile::tempdir().expect("tempdir");
        fs::write(
            temp.path().join(REQUIRED_CONFIG_FILE),
            br#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}"#,
        )
        .expect("config");
        fs::write(
            temp.path().join(REQUIRED_TOKENIZER_FILE),
            br#"{"version":"1.0"}"#,
        )
        .expect("tokenizer");
        write_test_safetensors(&temp.path().join(REQUIRED_WEIGHTS_FILE));

        let err =
            create_pinned_manifest(temp.path(), "0123456789abcdef0123456789abcdef01234567", &[])
                .expect_err("incomplete asset set must fail");
        assert!(err.to_string().contains(REQUIRED_TOKENIZER_CONFIG_FILE));
    }

    #[test]
    fn checksum_mismatch_is_reported_clearly() {
        let temp = tempfile::tempdir().expect("tempdir");
        seed_model_dir(temp.path());
        write_pinned_manifest(temp.path(), "0123456789abcdef0123456789abcdef01234567", &[])
            .expect("manifest");
        fs::write(temp.path().join(REQUIRED_TOKENIZER_FILE), b"changed").expect("mutate");

        let report = verify_model_assets(temp.path());
        assert!(!report.ok());
        assert!(report.problems.iter().any(|problem| {
            problem.contains("checksum mismatch for `tokenizer.json`")
                && problem.contains("expected")
                && problem.contains("got")
        }));
        let tokenizer = report
            .checked_files
            .iter()
            .find(|file| file.path == REQUIRED_TOKENIZER_FILE)
            .expect("tokenizer check");
        assert!(!tokenizer.ok);
        assert!(tokenizer.actual_sha256.is_some());
    }

    #[test]
    fn manifest_writer_requires_pinned_revision() {
        let temp = tempfile::tempdir().expect("tempdir");
        seed_model_dir(temp.path());

        let err = create_pinned_manifest(temp.path(), "main", &[])
            .expect_err("floating revisions must fail");
        assert!(err.to_string().contains("revision must be pinned"));
    }

    #[test]
    fn manifest_paths_must_stay_inside_model_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        fs::write(
            temp.path().join(MODEL_MANIFEST_FILE),
            format!(
                r#"model_id = "{REQUIRED_MODEL_ID}"
revision = "0123456789abcdef0123456789abcdef01234567"

[[files]]
path = "../escape.safetensors"
sha256 = "{}"
"#,
                "0".repeat(64)
            ),
        )
        .expect("write manifest");

        let report = verify_model_assets(temp.path());
        assert!(!report.ok());
        assert!(
            report
                .problems
                .iter()
                .any(|problem| { problem.contains("must be relative") && problem.contains("..") })
        );
    }

    fn seed_model_dir(model_dir: &Path) {
        fs::write(
            model_dir.join(REQUIRED_CONFIG_FILE),
            br#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}"#,
        )
        .expect("config");
        fs::write(
            model_dir.join(REQUIRED_TOKENIZER_FILE),
            br#"{"version":"1.0"}"#,
        )
        .expect("tokenizer");
        fs::write(
            model_dir.join(REQUIRED_TOKENIZER_CONFIG_FILE),
            br#"{"chat_template":"{{ messages }}"}"#,
        )
        .expect("tokenizer config");
        fs::write(
            model_dir.join(REQUIRED_CHAT_TEMPLATE_FILE),
            br#"{% for message in messages %}{{ message.content }}{% endfor %}"#,
        )
        .expect("chat template");
        fs::write(
            model_dir.join(REQUIRED_GENERATION_CONFIG_FILE),
            br#"{"max_new_tokens":128}"#,
        )
        .expect("generation config");
        fs::write(
            model_dir.join(REQUIRED_PROCESSOR_CONFIG_FILE),
            br#"{"processor_class":"Gemma4Processor"}"#,
        )
        .expect("processor config");
        write_test_safetensors(&model_dir.join(REQUIRED_WEIGHTS_FILE));
    }

    fn write_test_safetensors(path: &Path) {
        let data = 1.0f32.to_le_bytes();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data)
            .expect("tensor view");
        let bytes =
            safetensors::serialize([("weight".to_string(), view)], None).expect("safetensors");
        fs::write(path, bytes).expect("weights");
    }
}
