use crate::artifact::{DynamicSkillArtifact, SkillArtifactKind};
use crate::error::SkillError;
use crate::manifest::SkillManifest;
use crate::types::SkillId;
use koa_toolbox::{ToolPolicy, ToolSpec};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SkillPackage {
    pub manifest: SkillManifest,
    dynamic_artifacts: Vec<DynamicSkillArtifact>,
}

impl SkillPackage {
    pub fn dynamic_artifacts(&self) -> &[DynamicSkillArtifact] {
        &self.dynamic_artifacts
    }

    pub fn materialize_artifacts(
        &self,
        root: impl AsRef<Path>,
    ) -> Result<Vec<PathBuf>, SkillError> {
        self.dynamic_artifacts
            .iter()
            .map(|artifact| artifact.materialize(root.as_ref()))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SkillBuilder {
    id: SkillId,
    version: String,
    display_name: String,
    description: String,
    tools: Vec<ToolSpec>,
    dynamic_artifacts: Vec<DynamicSkillArtifact>,
    policy: ToolPolicy,
}

impl SkillBuilder {
    pub fn new(id: SkillId, version: impl Into<String>) -> Self {
        Self {
            id,
            version: version.into(),
            display_name: String::new(),
            description: String::new(),
            tools: Vec::new(),
            dynamic_artifacts: Vec::new(),
            policy: ToolPolicy::default_deny(),
        }
    }

    pub fn display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = display_name.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn tool(mut self, tool: ToolSpec) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn policy(mut self, policy: ToolPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn artifact_bytes(
        mut self,
        name: impl Into<String>,
        kind: SkillArtifactKind,
        media_type: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<Self, SkillError> {
        self.dynamic_artifacts
            .push(DynamicSkillArtifact::new(name, kind, media_type, bytes)?);
        Ok(self)
    }

    pub fn build(self) -> Result<SkillPackage, SkillError> {
        let artifacts = self
            .dynamic_artifacts
            .iter()
            .map(|artifact| artifact.manifest().clone())
            .collect();
        let manifest = SkillManifest::new(
            self.id,
            self.version,
            self.display_name,
            self.description,
            self.tools,
            artifacts,
            self.policy,
        )?;

        Ok(SkillPackage {
            manifest,
            dynamic_artifacts: self.dynamic_artifacts,
        })
    }
}
