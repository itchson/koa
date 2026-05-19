use crate::error::{AgentError, require_non_empty, validate_identifier};
use koa_skill::SkillId;
use koa_toolbox::{ToolPolicy, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId(String);

impl AgentId {
    pub fn new(value: impl Into<String>) -> Result<Self, AgentError> {
        let value = value.into();
        validate_identifier("agent_id", &value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl AsRef<str> for AgentId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SkillRef {
    pub skill_id: SkillId,
    pub required: bool,
}

impl SkillRef {
    pub fn required(skill_id: SkillId) -> Self {
        Self {
            skill_id,
            required: true,
        }
    }

    pub fn optional(skill_id: SkillId) -> Self {
        Self {
            skill_id,
            required: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub id: AgentId,
    pub version: String,
    pub display_name: String,
    pub description: String,
    pub skills: Vec<SkillRef>,
    pub tools: Vec<ToolSpec>,
    pub policy: ToolPolicy,
}

impl AgentSpec {
    pub fn new(
        id: AgentId,
        version: impl Into<String>,
        display_name: impl Into<String>,
        description: impl Into<String>,
        skills: Vec<SkillRef>,
        tools: Vec<ToolSpec>,
        policy: ToolPolicy,
    ) -> Result<Self, AgentError> {
        let version = require_non_empty("version", version.into())?;
        let display_name = require_non_empty("display_name", display_name.into())?;
        let description = require_non_empty("description", description.into())?;
        ensure_unique_skills(&skills)?;
        ensure_unique_tools(&tools)?;

        Ok(Self {
            id,
            version,
            display_name,
            description,
            skills,
            tools,
            policy,
        })
    }
}

fn ensure_unique_skills(skills: &[SkillRef]) -> Result<(), AgentError> {
    let mut seen = BTreeSet::new();
    for skill in skills {
        if !seen.insert(skill.skill_id.as_str().to_owned()) {
            return Err(AgentError::DuplicateSkill(skill.skill_id.to_string()));
        }
    }
    Ok(())
}

fn ensure_unique_tools(tools: &[ToolSpec]) -> Result<(), AgentError> {
    let mut seen = BTreeSet::new();
    for tool in tools {
        if !seen.insert(tool.id.as_str().to_owned()) {
            return Err(AgentError::DuplicateTool(tool.id.to_string()));
        }
    }
    Ok(())
}
