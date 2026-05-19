use crate::artifact::SkillArtifact;
use crate::error::{SkillError, require_non_empty};
use crate::types::SkillId;
use koa_toolbox::{ToolId, ToolPolicy, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillManifest {
    pub id: SkillId,
    pub version: String,
    pub display_name: String,
    pub description: String,
    pub tools: Vec<ToolSpec>,
    pub artifacts: Vec<SkillArtifact>,
    pub policy: ToolPolicy,
}

impl SkillManifest {
    pub fn new(
        id: SkillId,
        version: impl Into<String>,
        display_name: impl Into<String>,
        description: impl Into<String>,
        tools: Vec<ToolSpec>,
        artifacts: Vec<SkillArtifact>,
        policy: ToolPolicy,
    ) -> Result<Self, SkillError> {
        let version = require_non_empty("version", version.into())?;
        let display_name = require_non_empty("display_name", display_name.into())?;
        let description = require_non_empty("description", description.into())?;
        ensure_unique_tools(&tools)?;
        ensure_unique_artifacts(&artifacts)?;

        Ok(Self {
            id,
            version,
            display_name,
            description,
            tools,
            artifacts,
            policy,
        })
    }

    pub fn declares_tool(&self, tool_id: &ToolId) -> bool {
        self.tools.iter().any(|tool| tool.id == *tool_id)
    }
}

fn ensure_unique_tools(tools: &[ToolSpec]) -> Result<(), SkillError> {
    let mut seen = BTreeSet::new();
    for tool in tools {
        if !seen.insert(tool.id.as_str().to_owned()) {
            return Err(SkillError::DuplicateTool(tool.id.to_string()));
        }
    }
    Ok(())
}

fn ensure_unique_artifacts(artifacts: &[SkillArtifact]) -> Result<(), SkillError> {
    let mut seen = BTreeSet::new();
    for artifact in artifacts {
        if !seen.insert(artifact.name.clone()) {
            return Err(SkillError::DuplicateArtifact(artifact.name.clone()));
        }
    }
    Ok(())
}
