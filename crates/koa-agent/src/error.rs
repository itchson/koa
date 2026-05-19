use crate::registry::AgentLifecycleState;
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum AgentError {
    EmptyField(&'static str),
    InvalidIdentifier {
        field: &'static str,
        value: String,
    },
    DuplicateSkill(String),
    DuplicateTool(String),
    AlreadyRegistered(String),
    MissingAgent(String),
    InvalidTransition {
        id: String,
        from: AgentLifecycleState,
        to: AgentLifecycleState,
    },
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "`{field}` cannot be empty"),
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "`{field}` has an invalid identifier: `{value}`")
            }
            Self::DuplicateSkill(skill) => write!(formatter, "duplicate agent skill: `{skill}`"),
            Self::DuplicateTool(tool) => write!(formatter, "duplicate agent tool: `{tool}`"),
            Self::AlreadyRegistered(id) => write!(formatter, "agent already registered: `{id}`"),
            Self::MissingAgent(id) => write!(formatter, "agent is not registered: `{id}`"),
            Self::InvalidTransition { id, from, to } => write!(
                formatter,
                "agent `{id}` cannot transition from `{from:?}` to `{to:?}`"
            ),
        }
    }
}

impl std::error::Error for AgentError {}

pub(crate) fn require_non_empty(field: &'static str, value: String) -> Result<String, AgentError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(AgentError::EmptyField(field))
    } else {
        Ok(trimmed.to_owned())
    }
}

pub(crate) fn validate_identifier(field: &'static str, value: &str) -> Result<(), AgentError> {
    if value.is_empty() {
        return Err(AgentError::EmptyField(field));
    }

    let mut chars = value.chars();
    let first = chars.next().expect("checked non-empty");
    let valid_first = first.is_ascii_alphanumeric();
    let valid_rest = chars.all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | ':' | '@')
    });

    if valid_first && valid_rest {
        Ok(())
    } else {
        Err(AgentError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        })
    }
}
