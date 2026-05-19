use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use koa_agent::{AgentId, AgentSpec, SkillRef};
use koa_capsule::{Capsule, CapsuleConfig, CapsuleId, Doctor as CapsuleDoctor};
use koa_context::{IngestReport, LintReport, Vault};
use koa_infer::{
    InferenceDoctor, InferenceRequest, ModelAssetReport, ModelManifest, NativeGemmaEngine,
    verify_model_assets as verify_model_assets_in_dir, write_pinned_manifest,
};
use koa_skill::{SkillArtifactKind, SkillBuilder, SkillId};
use koa_toolbox::{JsonSchema, ToolCall, ToolId, ToolPolicy, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use walkdir::WalkDir;

pub const KOA_DIR: &str = ".koa";

#[derive(Debug, Clone)]
pub struct KoaPaths {
    pub root: PathBuf,
    pub state: PathBuf,
    pub config: PathBuf,
    pub vault: PathBuf,
    pub sessions: PathBuf,
    pub capsules: PathBuf,
    pub models: PathBuf,
    pub toolbox: PathBuf,
    pub skills: PathBuf,
    pub agents: PathBuf,
    pub audit: PathBuf,
}

impl KoaPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let state = root.join(KOA_DIR);
        Self {
            config: state.join("config.toml"),
            vault: state.join("vault"),
            sessions: state.join("sessions"),
            capsules: state.join("capsules"),
            models: state.join("models"),
            toolbox: state.join("toolbox"),
            skills: state.join("skills"),
            agents: state.join("agents"),
            audit: state.join("audit.jsonl"),
            state,
            root,
        }
    }

    pub fn ensure(&self) -> Result<()> {
        for path in [
            &self.state,
            &self.vault,
            &self.sessions,
            &self.capsules,
            &self.models,
            &self.toolbox,
            &self.skills,
            &self.agents,
        ] {
            fs::create_dir_all(path)
                .with_context(|| format!("failed to create {}", path.display()))?;
        }
        fs::create_dir_all(self.vault.join("raw"))?;
        fs::create_dir_all(self.vault.join("wiki"))?;
        fs::create_dir_all(self.vault.join("schema"))?;
        fs::create_dir_all(self.skills.join("session"))?;
        fs::create_dir_all(self.skills.join("global"))?;
        fs::create_dir_all(self.agents.join("session"))?;
        fs::create_dir_all(self.agents.join("global"))?;
        write_if_missing(
            &self.vault.join("index.md"),
            "# Koa Vault Index\n\nOnly real ingested notes and files belong here.\n",
        )?;
        write_if_missing(&self.vault.join("log.md"), "# Koa Vault Log\n\n")?;
        write_if_missing(
            &self.vault.join("schema").join("AGENTS.md"),
            "# Agent Context Schema\n\nGenerated agents must record role, goal, tools, skills, lifecycle, and provenance.\n",
        )?;
        write_if_missing(&self.toolbox.join("tools.json"), "[]\n")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoaConfig {
    pub version: String,
    pub active_session: Option<String>,
    pub inference: InferenceConfig,
    pub capsule: CapsuleRuntimeConfig,
}

impl Default for KoaConfig {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            active_session: None,
            inference: InferenceConfig::default(),
            capsule: CapsuleRuntimeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub model_id: String,
    pub revision: String,
    pub model_dir: String,
    pub max_context_tokens: usize,
    pub require_cuda: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_id: "google/gemma-4-E2B-it".to_string(),
            revision: "pin-required".to_string(),
            model_dir: ".koa/models/gemma-4-E2B-it".to_string(),
            max_context_tokens: 16_384,
            require_cuda: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleRuntimeConfig {
    pub base_rootfs: String,
    pub memory_limit_mb: u64,
    pub pids_limit: u64,
    pub cpu_quota_percent: u64,
}

impl Default for CapsuleRuntimeConfig {
    fn default() -> Self {
        Self {
            base_rootfs: ".koa/capsules/base-rootfs".to_string(),
            memory_limit_mb: 4096,
            pids_limit: 256,
            cpu_quota_percent: 200,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    pub name: String,
    pub workspace: PathBuf,
    pub created_at_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub workspace_ok: bool,
    pub inference_ok: bool,
    pub capsule_ok: bool,
    pub vault_ok: bool,
    pub toolbox_ok: bool,
    pub problems: Vec<String>,
}

impl DoctorReport {
    pub fn healthy(&self) -> bool {
        self.workspace_ok
            && self.inference_ok
            && self.capsule_ok
            && self.vault_ok
            && self.toolbox_ok
            && self.problems.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleHostDoctor {
    pub mode: String,
    pub ok: bool,
    pub checks: Vec<String>,
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    pub spec: ToolSpec,
    pub output_schema: Value,
    pub provider: String,
    pub trust_level: String,
    pub permissions: Vec<String>,
    pub timeout_ms: u64,
    pub audit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentFile {
    spec: AgentSpec,
    lifecycle: String,
    context_budget_tokens: usize,
    output_contract: String,
    created_at_unix: u64,
}

pub struct KoaRuntime {
    paths: KoaPaths,
}

impl KoaRuntime {
    pub fn init(root: impl Into<PathBuf>) -> Result<Self> {
        let runtime = Self {
            paths: KoaPaths::new(root),
        };
        runtime.paths.ensure()?;
        if !runtime.paths.config.exists() {
            runtime.save_config(&KoaConfig::default())?;
        }
        if runtime.list_sessions()?.is_empty() {
            let session = runtime.create_session("main")?;
            let mut config = runtime.load_config()?;
            config.active_session = Some(session.id);
            runtime.save_config(&config)?;
        }
        runtime.audit(
            "runtime.init",
            json!({"version": env!("CARGO_PKG_VERSION")}),
        )?;
        Ok(runtime)
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let runtime = Self {
            paths: KoaPaths::new(root),
        };
        if !runtime.paths.state.is_dir() {
            bail!(
                "{} does not exist; run `koa init` first",
                runtime.paths.state.display()
            );
        }
        Ok(runtime)
    }

    pub fn paths(&self) -> &KoaPaths {
        &self.paths
    }

    pub fn verify_model_assets(&self) -> Result<ModelAssetReport> {
        let config = self.load_config()?;
        let mut report =
            verify_model_assets_in_dir(self.resolve_workspace_path(&config.inference.model_dir));
        verify_config_matches_model_report(&config, &mut report);
        Ok(report)
    }

    pub fn write_model_manifest(&self, revision: &str, files: &[PathBuf]) -> Result<ModelManifest> {
        let mut config = self.load_config()?;
        let model_dir = self.resolve_workspace_path(&config.inference.model_dir);
        let manifest = write_pinned_manifest(&model_dir, revision, files)?;
        config.inference.model_id = manifest.model_id.clone();
        config.inference.revision = manifest.revision.clone();
        self.save_config(&config)?;
        self.audit(
            "model.manifest.write",
            json!({
                "model_dir": model_dir,
                "model_id": &manifest.model_id,
                "revision": &manifest.revision,
                "files": &manifest.files,
            }),
        )?;
        Ok(manifest)
    }

    pub fn model_doctor(&self) -> Result<InferenceDoctor> {
        let config = self.load_config()?;
        let mut doctor = NativeGemmaEngine::doctor(
            self.resolve_workspace_path(&config.inference.model_dir),
            config.inference.require_cuda,
        );
        verify_config_matches_inference_doctor(&config, &mut doctor);
        Ok(doctor)
    }

    pub fn capsule_doctor(&self) -> Result<CapsuleHostDoctor> {
        let config = self.load_config()?;
        Ok(capsule_runtime_doctor(
            self.resolve_workspace_path(&config.capsule.base_rootfs),
        ))
    }

    pub fn doctor(&self) -> DoctorReport {
        let mut report = DoctorReport {
            workspace_ok: self.paths.state.is_dir() && self.paths.config.is_file(),
            inference_ok: false,
            capsule_ok: false,
            vault_ok: false,
            toolbox_ok: false,
            problems: Vec::new(),
        };
        let infer = match self.model_doctor() {
            Ok(infer) => infer,
            Err(err) => {
                report.problems.push(format!("inference: {err}"));
                return report;
            }
        };
        report.inference_ok = infer.healthy();
        report.problems.extend(
            infer
                .problems
                .into_iter()
                .map(|problem| format!("inference: {problem}")),
        );

        let capsule = match self.capsule_doctor() {
            Ok(capsule) => capsule,
            Err(err) => {
                report.problems.push(format!("capsule: {err}"));
                return report;
            }
        };
        report.capsule_ok = capsule.ok;
        report.problems.extend(
            capsule
                .problems
                .into_iter()
                .map(|problem| format!("capsule: {problem}")),
        );

        match Vault::open(&self.paths.vault).and_then(|vault| vault.lint()) {
            Ok(lint) => {
                report.vault_ok = lint.errors().next().is_none();
                report.problems.extend(
                    lint.issues
                        .into_iter()
                        .filter(|issue| matches!(issue.severity, koa_context::LintSeverity::Error))
                        .map(|issue| format!("vault: {} - {}", issue.code, issue.message)),
                );
            }
            Err(err) => report.problems.push(format!("vault: {err}")),
        }

        match self.read_tool_manifests() {
            Ok(_) => report.toolbox_ok = true,
            Err(err) => report.problems.push(format!("toolbox: {err}")),
        }
        report
    }

    pub fn chat(&self, prompt: &str) -> Result<String> {
        self.chat_with_options(prompt, ChatOptions::default())
    }

    pub fn chat_with_options(&self, prompt: &str, options: ChatOptions) -> Result<String> {
        if prompt.trim().is_empty() {
            bail!("prompt cannot be empty");
        }
        if options.max_new_tokens == 0 {
            bail!("max_new_tokens must be greater than zero");
        }
        if !options.temperature.is_finite() || options.temperature < 0.0 {
            bail!("temperature must be finite and greater than or equal to zero");
        }
        if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
            bail!("top_p must be finite and within (0, 1]");
        }
        let report = self.doctor();
        if !report.inference_ok {
            bail!(
                "native inference gate failed; refusing to fabricate a chat response. Problems: {}",
                report.problems.join("; ")
            );
        }
        let config = self.load_config()?;
        let mut engine =
            NativeGemmaEngine::load(self.resolve_workspace_path(&config.inference.model_dir))?;
        let response = engine.generate(InferenceRequest {
            prompt: prompt.to_string(),
            max_new_tokens: options.max_new_tokens,
            temperature: options.temperature,
            top_p: options.top_p,
            seed: options.seed,
        })?;
        self.audit(
            "runtime.chat",
            json!({
                "prompt_tokens": response.prompt_tokens,
                "generated_tokens": response.generated_tokens,
                "backend": response.backend,
            }),
        )?;
        Ok(response.text)
    }

    pub fn create_session(&self, name: &str) -> Result<SessionRecord> {
        let name = require_name("session name", name)?;
        let id = Uuid::new_v4().to_string();
        let workspace = self.paths.sessions.join(&id).join("workspace");
        fs::create_dir_all(&workspace)?;
        let record = SessionRecord {
            id: id.clone(),
            name: name.to_string(),
            workspace,
            created_at_unix: unix_secs(),
        };
        fs::write(
            self.paths.sessions.join(&id).join("session.json"),
            serde_json::to_vec_pretty(&record)?,
        )?;
        self.create_capsule_metadata(&record)?;
        self.audit("session.create", json!({"session_id": id, "name": name}))?;
        Ok(record)
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionRecord>> {
        if !self.paths.sessions.is_dir() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in fs::read_dir(&self.paths.sessions)? {
            let entry = entry?;
            let path = entry.path().join("session.json");
            if path.is_file() {
                sessions.push(serde_json::from_slice(&fs::read(&path)?)?);
            }
        }
        sessions.sort_by(|a: &SessionRecord, b| a.created_at_unix.cmp(&b.created_at_unix));
        Ok(sessions)
    }

    pub fn switch_session(&self, id: &str) -> Result<()> {
        if self.session(id)?.is_none() {
            bail!("unknown session `{id}`");
        }
        let mut config = self.load_config()?;
        config.active_session = Some(id.to_string());
        self.save_config(&config)?;
        self.audit("session.switch", json!({"session_id": id}))?;
        Ok(())
    }

    pub fn destroy_session(&self, id: &str) -> Result<()> {
        let session_dir = self.paths.sessions.join(id);
        if !session_dir.is_dir() {
            bail!("unknown session `{id}`");
        }
        fs::remove_dir_all(&session_dir)?;
        let capsule_dir = self.paths.capsules.join("state").join(id);
        if capsule_dir.is_dir() {
            fs::remove_dir_all(capsule_dir)?;
        }
        let mut config = self.load_config()?;
        if config.active_session.as_deref() == Some(id) {
            config.active_session = self
                .list_sessions()?
                .first()
                .map(|session| session.id.clone());
            self.save_config(&config)?;
        }
        self.audit("session.destroy", json!({"session_id": id}))?;
        Ok(())
    }

    pub fn snapshot_active_session(&self, name: &str) -> Result<PathBuf> {
        let session = self.active_session()?;
        let mut capsule = self.open_session_capsule(&session)?;
        let snapshot = capsule.snapshot(name)?;
        self.audit(
            "session.snapshot",
            json!({"session_id": session.id, "snapshot": snapshot.name}),
        )?;
        Ok(snapshot.upper_dir)
    }

    pub fn restore_active_session(&self, name: &str) -> Result<()> {
        let session = self.active_session()?;
        let mut capsule = self.open_session_capsule(&session)?;
        capsule.restore(name)?;
        self.audit(
            "session.restore",
            json!({"session_id": session.id, "snapshot": name}),
        )?;
        Ok(())
    }

    pub fn vault_ingest(&self, source: impl AsRef<Path>) -> Result<IngestReport> {
        let source = source.as_ref();
        if !source.exists() {
            bail!("cannot ingest missing path {}", source.display());
        }
        copy_into_vault_raw(source, &self.paths.vault.join("raw"))?;
        let vault = Vault::open(&self.paths.vault)?;
        let report = vault.ingest()?;
        self.audit("vault.ingest", json!({"source": source, "report": report}))?;
        Ok(report)
    }

    pub fn vault_search(&self, query: &str, limit: usize) -> Result<Vec<koa_store::SearchHit>> {
        Ok(Vault::open(&self.paths.vault)?.search(query, limit)?)
    }

    pub fn vault_lint(&self) -> Result<LintReport> {
        Ok(Vault::open(&self.paths.vault)?.lint()?)
    }

    pub fn register_tool(&self, manifest: ToolManifest) -> Result<()> {
        if manifest.timeout_ms == 0 {
            bail!("tool timeout must be greater than zero");
        }
        if !manifest.audit {
            bail!("Koa v0.0.1 requires audited tools");
        }
        let mut tools = self.read_tool_manifests()?;
        if tools.iter().any(|tool| tool.spec.id == manifest.spec.id) {
            bail!("tool `{}` already registered", manifest.spec.id);
        }
        tools.push(manifest);
        self.write_tool_manifests(&tools)
    }

    pub fn register_tool_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let manifest: ToolManifest = serde_json::from_slice(&fs::read(path.as_ref())?)?;
        self.register_tool(manifest)
    }

    pub fn list_tools(&self) -> Result<Vec<ToolManifest>> {
        self.read_tool_manifests()
    }

    pub fn call_tool(&self, id: &str, input: Value) -> Result<Value> {
        let tool_id = ToolId::new(id).map_err(|err| anyhow::anyhow!(err))?;
        let call = ToolCall::new(
            format!("call-{}", unix_secs()),
            tool_id.clone(),
            input.clone(),
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        let tools = self.read_tool_manifests()?;
        let Some(tool) = tools.iter().find(|tool| tool.spec.id == tool_id) else {
            self.audit(
                "tool.call.denied",
                json!({"tool_id": id, "reason": "tool is not registered", "input": input}),
            )?;
            bail!("tool `{id}` is not registered");
        };
        let evaluation = ToolPolicy::default_deny().evaluate(&call);
        self.audit(
            "tool.call.denied",
            json!({
                "tool_id": tool.spec.id.as_str(),
                "reason": evaluation.reason,
                "input": input
            }),
        )?;
        bail!(
            "tool `{}` is registered but has no approved runtime executor wired in v0.0.1",
            tool.spec.id
        )
    }

    pub fn register_mcp(&self, name: &str, command: &str) -> Result<PathBuf> {
        let name = require_name("MCP server name", name)?;
        let command = require_name("MCP command", command)?;
        let path = self.paths.toolbox.join("mcp-servers.json");
        let mut servers = if path.is_file() {
            serde_json::from_slice::<Vec<Value>>(&fs::read(&path)?)?
        } else {
            Vec::new()
        };
        if servers
            .iter()
            .any(|server| server.get("name").and_then(Value::as_str) == Some(name))
        {
            bail!("MCP server `{name}` is already registered");
        }
        servers.push(json!({"name": name, "command": command, "registered_at_unix": unix_secs()}));
        fs::write(&path, serde_json::to_vec_pretty(&servers)?)?;
        Ok(path)
    }

    pub fn create_skill(
        &self,
        id: &str,
        name: &str,
        description: &str,
        body: &str,
        global: bool,
    ) -> Result<PathBuf> {
        reject_executable_skill_body(body)?;
        let skill_id = SkillId::new(id).map_err(|err| anyhow::anyhow!(err))?;
        let package = SkillBuilder::new(skill_id, env!("CARGO_PKG_VERSION"))
            .display_name(name)
            .description(description)
            .policy(ToolPolicy::default_deny())
            .artifact_bytes(
                "SKILL.md",
                SkillArtifactKind::Instructions,
                "text/markdown",
                body.as_bytes().to_vec(),
            )
            .map_err(|err| anyhow::anyhow!(err))?
            .build()
            .map_err(|err| anyhow::anyhow!(err))?;
        let scope = if global { "global" } else { "session" };
        let dir = self.paths.skills.join(scope).join(id);
        if dir.exists() {
            bail!("skill `{id}` already exists");
        }
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join("skill.json"),
            serde_json::to_vec_pretty(&package.manifest)?,
        )?;
        package
            .materialize_artifacts(&dir)
            .map_err(|err| anyhow::anyhow!(err))?;
        Ok(dir)
    }

    pub fn list_skills(&self) -> Result<Vec<Value>> {
        read_json_files_under(&self.paths.skills)
    }

    pub fn delete_skill(&self, id: &str) -> Result<()> {
        remove_named_dir(&self.paths.skills, id)
    }

    pub fn spawn_agent(
        &self,
        id: &str,
        role: &str,
        goal: &str,
        output_contract: &str,
        global: bool,
    ) -> Result<PathBuf> {
        let agent_id = AgentId::new(id).map_err(|err| anyhow::anyhow!(err))?;
        let spec = AgentSpec::new(
            agent_id,
            env!("CARGO_PKG_VERSION"),
            role,
            goal,
            Vec::<SkillRef>::new(),
            Vec::<ToolSpec>::new(),
            ToolPolicy::default_deny(),
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        let scope = if global { "global" } else { "session" };
        let path = self.paths.agents.join(scope).join(format!("{id}.json"));
        if path.exists() {
            bail!("agent `{id}` already exists");
        }
        let file = AgentFile {
            spec,
            lifecycle: scope.to_string(),
            context_budget_tokens: 4096,
            output_contract: output_contract.to_string(),
            created_at_unix: unix_secs(),
        };
        fs::write(&path, serde_json::to_vec_pretty(&file)?)?;
        Ok(path)
    }

    pub fn list_agents(&self) -> Result<Vec<Value>> {
        read_json_files_under(&self.paths.agents)
    }

    pub fn delete_agent(&self, id: &str) -> Result<()> {
        remove_named_file(&self.paths.agents, id)
    }

    fn load_config(&self) -> Result<KoaConfig> {
        let text = fs::read_to_string(&self.paths.config)
            .with_context(|| format!("failed to read {}", self.paths.config.display()))?;
        Ok(toml::from_str(&text)?)
    }

    fn save_config(&self, config: &KoaConfig) -> Result<()> {
        fs::write(&self.paths.config, toml::to_string_pretty(config)?)?;
        Ok(())
    }

    fn active_session(&self) -> Result<SessionRecord> {
        let config = self.load_config()?;
        let Some(id) = config.active_session else {
            bail!("no active session configured");
        };
        self.session(&id)?
            .with_context(|| format!("active session `{id}` does not exist"))
    }

    fn session(&self, id: &str) -> Result<Option<SessionRecord>> {
        let path = self.paths.sessions.join(id).join("session.json");
        if !path.is_file() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
    }

    fn create_capsule_metadata(&self, session: &SessionRecord) -> Result<()> {
        let config = self.load_config()?;
        let lower = self.resolve_workspace_path(&config.capsule.base_rootfs);
        let state = self.paths.capsules.join("state");
        let capsule_id = CapsuleId::new(&session.id).map_err(|err| anyhow::anyhow!(err))?;
        Capsule::create(
            CapsuleConfig::new(lower, state)
                .with_id(capsule_id)
                .with_label("session_id", &session.id)
                .with_label("network", "offline"),
        )
        .map_err(|err| anyhow::anyhow!(err))?;
        Ok(())
    }

    fn open_session_capsule(&self, session: &SessionRecord) -> Result<Capsule> {
        Capsule::open(self.paths.capsules.join("state").join(&session.id))
            .map_err(|err| anyhow::anyhow!(err))
    }

    fn read_tool_manifests(&self) -> Result<Vec<ToolManifest>> {
        let path = self.paths.toolbox.join("tools.json");
        if !path.is_file() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    fn write_tool_manifests(&self, tools: &[ToolManifest]) -> Result<()> {
        fs::write(
            self.paths.toolbox.join("tools.json"),
            serde_json::to_vec_pretty(tools)?,
        )?;
        Ok(())
    }

    fn resolve_workspace_path(&self, value: &str) -> PathBuf {
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            self.paths.root.join(path)
        }
    }

    fn audit(&self, action: &str, payload: Value) -> Result<()> {
        let event = json!({
            "timestamp_unix": unix_secs(),
            "action": action,
            "payload": payload,
        });
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.paths.audit)?;
        writeln!(file, "{}", serde_json::to_string(&event)?)?;
        Ok(())
    }
}

fn copy_into_vault_raw(source: &Path, raw: &Path) -> Result<()> {
    fs::create_dir_all(raw)?;
    if source.is_file() {
        let name = source
            .file_name()
            .with_context(|| format!("cannot determine file name for {}", source.display()))?;
        fs::copy(source, raw.join(name))?;
        return Ok(());
    }
    for entry in WalkDir::new(source)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let relative = entry.path().strip_prefix(source)?;
        let target = raw.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(entry.path(), target)?;
    }
    Ok(())
}

fn read_json_files_under(root: &Path) -> Result<Vec<Value>> {
    let mut values = Vec::new();
    if !root.is_dir() {
        return Ok(values);
    }
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        if entry.path().extension().and_then(|ext| ext.to_str()) == Some("json") {
            values.push(serde_json::from_slice(&fs::read(entry.path())?)?);
        }
    }
    Ok(values)
}

fn remove_named_dir(root: &Path, id: &str) -> Result<()> {
    for scope in ["session", "global"] {
        let path = root.join(scope).join(id);
        if path.is_dir() {
            fs::remove_dir_all(path)?;
            return Ok(());
        }
    }
    bail!("unknown item `{id}`");
}

fn remove_named_file(root: &Path, id: &str) -> Result<()> {
    for scope in ["session", "global"] {
        let path = root.join(scope).join(format!("{id}.json"));
        if path.is_file() {
            fs::remove_file(path)?;
            return Ok(());
        }
    }
    bail!("unknown item `{id}`");
}

fn require_name<'a>(label: &str, value: &'a str) -> Result<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{label} cannot be empty");
    }
    Ok(value)
}

fn reject_executable_skill_body(body: &str) -> Result<()> {
    for marker in ["```bash", "```sh", "```powershell", "```ps1", "exec("] {
        if body.contains(marker) {
            bail!("dynamic skills may not include executable code blocks in v0.0.1");
        }
    }
    Ok(())
}

fn write_if_missing(path: &Path, body: &str) -> Result<()> {
    if !path.exists() {
        fs::write(path, body)?;
    }
    Ok(())
}

fn capsule_host_doctor() -> CapsuleHostDoctor {
    let native = CapsuleDoctor::run();
    if native.required_failures().next().is_none() {
        return CapsuleHostDoctor {
            mode: "native-linux".to_string(),
            ok: true,
            checks: native
                .checks
                .into_iter()
                .map(|check| format!("{}: {:?}", check.id, check.status))
                .collect(),
            problems: Vec::new(),
        };
    }

    #[cfg(windows)]
    {
        wsl_capsule_doctor("Ubuntu-24.04")
    }

    #[cfg(not(windows))]
    {
        CapsuleHostDoctor {
            mode: "native-linux".to_string(),
            ok: false,
            checks: native
                .checks
                .iter()
                .map(|check| format!("{}: {:?}", check.id, check.status))
                .collect(),
            problems: native
                .required_failures()
                .map(|check| format!("{} - {}", check.id, check.details))
                .collect(),
        }
    }
}

fn capsule_runtime_doctor(base_rootfs: PathBuf) -> CapsuleHostDoctor {
    let mut report = capsule_host_doctor();
    let mut rootfs_ok = true;

    if base_rootfs.is_dir() {
        report
            .checks
            .push(format!("base_rootfs: pass ({})", base_rootfs.display()));
    } else {
        rootfs_ok = false;
        report
            .checks
            .push(format!("base_rootfs: fail ({})", base_rootfs.display()));
        report.problems.push(format!(
            "base rootfs {} is missing; Koa will not synthesize a fake rootfs",
            base_rootfs.display()
        ));
    }

    let shell = base_rootfs.join("bin").join("sh");
    if shell.is_file() {
        report
            .checks
            .push(format!("rootfs_shell: pass ({})", shell.display()));
    } else {
        rootfs_ok = false;
        report
            .checks
            .push(format!("rootfs_shell: fail ({})", shell.display()));
        report.problems.push(format!(
            "base rootfs must contain a real /bin/sh at {}",
            shell.display()
        ));
    }

    let proc_dir = base_rootfs.join("proc");
    if proc_dir.is_dir() {
        report
            .checks
            .push(format!("rootfs_proc: pass ({})", proc_dir.display()));
    } else {
        rootfs_ok = false;
        report
            .checks
            .push(format!("rootfs_proc: fail ({})", proc_dir.display()));
        report.problems.push(format!(
            "base rootfs must contain a /proc mount point directory at {}",
            proc_dir.display()
        ));
    }

    report.ok = report.ok && rootfs_ok;
    report
}

#[cfg(windows)]
fn wsl_capsule_doctor(distribution: &str) -> CapsuleHostDoctor {
    let probes = [
        (
            "wsl2",
            "uname -a | grep -qi microsoft",
            "WSL2 kernel was not detected",
        ),
        (
            "user_namespace",
            "unshare -Ur /bin/true",
            "unprivileged user namespace probe failed",
        ),
        (
            "pid_namespace",
            "unshare -Urpf --mount-proc /bin/true",
            "PID namespace probe failed",
        ),
        (
            "network_namespace",
            "unshare -Urn /bin/sh -c 'true'",
            "network namespace probe failed",
        ),
        (
            "overlayfs",
            "tmp=\"$(mktemp -d)\"; mkdir -p \"$tmp/l\" \"$tmp/u\" \"$tmp/w\" \"$tmp/m\"; echo base > \"$tmp/l/base.txt\"; unshare -Urmpf /bin/sh -c \"mount -t overlay overlay -o lowerdir=$tmp/l,upperdir=$tmp/u,workdir=$tmp/w $tmp/m && test \\\"$(cat $tmp/m/base.txt)\\\" = base && umount $tmp/m\"; rc=$?; rm -rf \"$tmp\"; exit $rc",
            "overlayfs probe failed",
        ),
        (
            "cgroup",
            "test -r /sys/fs/cgroup/cgroup.controllers || test -d /sys/fs/cgroup/pids",
            "cgroup controls were not visible",
        ),
        (
            "seccomp",
            "grep -q '^Seccomp:' /proc/self/status && grep -q '^NoNewPrivs:' /proc/self/status",
            "seccomp/no-new-privs status was not visible",
        ),
    ];

    let mut checks = Vec::new();
    let mut problems = Vec::new();
    for (id, script, failure) in probes {
        match run_wsl_probe(distribution, script) {
            Ok(()) => checks.push(format!("{id}: pass")),
            Err(err) => {
                checks.push(format!("{id}: fail"));
                problems.push(format!("{failure}: {err}"));
            }
        }
    }
    CapsuleHostDoctor {
        mode: format!("wsl2:{distribution}"),
        ok: problems.is_empty(),
        checks,
        problems,
    }
}

#[cfg(windows)]
fn run_wsl_probe(distribution: &str, script: &str) -> Result<()> {
    let output = Command::new("wsl")
        .args([
            "-d",
            distribution,
            "--",
            "bash",
            "--noprofile",
            "--norc",
            "-lc",
            script,
        ])
        .output()
        .with_context(|| "failed to spawn wsl")?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!("{}{}", stdout.trim(), stderr.trim())
    }
}

fn verify_config_matches_model_report(config: &KoaConfig, report: &mut ModelAssetReport) {
    if let Some(model_id) = &report.model_id {
        if model_id != &config.inference.model_id {
            report.problems.push(format!(
                "config inference.model_id `{}` does not match model manifest id `{model_id}`",
                config.inference.model_id
            ));
        }
    }
    if let Some(revision) = &report.revision {
        if revision != &config.inference.revision {
            report.problems.push(format!(
                "config inference.revision `{}` does not match model manifest revision `{revision}`",
                config.inference.revision
            ));
        }
    }
}

fn verify_config_matches_inference_doctor(config: &KoaConfig, doctor: &mut InferenceDoctor) {
    if let Some(model_id) = &doctor.model_id {
        if model_id != &config.inference.model_id {
            doctor.problems.push(format!(
                "config inference.model_id `{}` does not match model manifest id `{model_id}`",
                config.inference.model_id
            ));
        }
    }
    if let Some(revision) = &doctor.revision {
        if revision != &config.inference.revision {
            doctor.problems.push(format!(
                "config inference.revision `{}` does not match model manifest revision `{revision}`",
                config.inference.revision
            ));
        }
    }
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

pub fn empty_schema() -> JsonSchema {
    JsonSchema::object()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_creates_state_and_session() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        assert!(runtime.paths().config.is_file());
        assert_eq!(runtime.list_sessions().expect("sessions").len(), 1);
    }

    #[test]
    fn init_does_not_synthesize_base_rootfs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        assert!(!runtime.paths().capsules.join("base-rootfs").exists());
    }

    #[test]
    fn capsule_doctor_reports_missing_base_rootfs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        let report = runtime.capsule_doctor().expect("capsule doctor");

        assert!(!report.ok);
        assert!(
            report
                .problems
                .iter()
                .any(|problem| problem.contains("base rootfs") && problem.contains("missing"))
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.starts_with("base_rootfs: fail"))
        );
    }

    #[test]
    fn capsule_doctor_checks_real_rootfs_markers() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        let rootfs = runtime.paths().capsules.join("base-rootfs");
        fs::create_dir_all(rootfs.join("bin")).expect("bin");
        fs::write(rootfs.join("bin").join("sh"), b"#!/bin/sh\n").expect("shell");
        fs::create_dir_all(rootfs.join("proc")).expect("proc");

        let report = runtime.capsule_doctor().expect("capsule doctor");
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.starts_with("base_rootfs: pass"))
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.starts_with("rootfs_shell: pass"))
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.starts_with("rootfs_proc: pass"))
        );
    }

    #[test]
    fn write_model_manifest_updates_config_revision() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        seed_model_dir(&runtime.paths().models.join("gemma-4-E2B-it"));

        let revision = "0123456789abcdef0123456789abcdef01234567";
        runtime
            .write_model_manifest(revision, &[])
            .expect("manifest");

        let config = runtime.load_config().expect("config");
        assert_eq!(config.inference.model_id, koa_infer::REQUIRED_MODEL_ID);
        assert_eq!(config.inference.revision, revision);
        let report = runtime.verify_model_assets().expect("verify");
        assert!(
            report
                .problems
                .iter()
                .all(|problem| !problem.contains("does not match"))
        );
    }

    #[test]
    fn model_doctor_reports_config_manifest_revision_mismatch() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        seed_model_dir(&runtime.paths().models.join("gemma-4-E2B-it"));

        runtime
            .write_model_manifest("0123456789abcdef0123456789abcdef01234567", &[])
            .expect("manifest");
        let mut config = runtime.load_config().expect("config");
        config.inference.revision = "ffffffffffffffffffffffffffffffffffffffff".to_string();
        runtime.save_config(&config).expect("save config");

        let doctor = runtime.model_doctor().expect("doctor");
        assert!(!doctor.healthy());
        assert!(doctor.problems.iter().any(|problem| {
            problem.contains("config inference.revision")
                && problem.contains("does not match model manifest revision")
        }));
    }

    #[test]
    fn chat_rejects_blank_prompt_before_runtime_doctor() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = KoaRuntime::init(temp.path()).expect("init");
        let err = runtime.chat("   ").expect_err("blank prompt");
        assert_eq!(err.to_string(), "prompt cannot be empty");
    }

    fn seed_model_dir(model_dir: &Path) {
        fs::create_dir_all(model_dir).expect("model dir");
        fs::write(
            model_dir.join("config.json"),
            br#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}"#,
        )
        .expect("config");
        fs::write(
            model_dir.join("tokenizer_config.json"),
            br#"{"chat_template":"{{ messages }}"}"#,
        )
        .expect("tokenizer config");
        fs::write(model_dir.join("tokenizer.json"), br#"{"version":"1.0"}"#).expect("tokenizer");
        fs::write(
            model_dir.join("chat_template.jinja"),
            br#"{% for message in messages %}{{ message.content }}{% endfor %}"#,
        )
        .expect("chat template");
        fs::write(
            model_dir.join("generation_config.json"),
            br#"{"max_new_tokens":128}"#,
        )
        .expect("generation config");
        fs::write(
            model_dir.join("processor_config.json"),
            br#"{"processor_class":"Gemma4Processor"}"#,
        )
        .expect("processor config");
        write_test_safetensors(&model_dir.join("model.safetensors"));
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
