use std::fmt;

#[derive(Debug)]
pub enum SkillError {
    EmptyField(&'static str),
    InvalidIdentifier { field: &'static str, value: String },
    InvalidArtifactName(String),
    ExecutableArtifactRejected(String),
    DuplicateArtifact(String),
    DuplicateTool(String),
    Io(std::io::Error),
}

impl fmt::Display for SkillError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "`{field}` cannot be empty"),
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "`{field}` has an invalid identifier: `{value}`")
            }
            Self::InvalidArtifactName(name) => {
                write!(formatter, "invalid skill artifact name: `{name}`")
            }
            Self::ExecutableArtifactRejected(name) => {
                write!(formatter, "executable skill artifact rejected: `{name}`")
            }
            Self::DuplicateArtifact(name) => {
                write!(formatter, "duplicate skill artifact: `{name}`")
            }
            Self::DuplicateTool(tool) => write!(formatter, "duplicate skill tool: `{tool}`"),
            Self::Io(error) => write!(formatter, "skill I/O error: {error}"),
        }
    }
}

impl std::error::Error for SkillError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SkillError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub(crate) fn require_non_empty(field: &'static str, value: String) -> Result<String, SkillError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(SkillError::EmptyField(field))
    } else {
        Ok(trimmed.to_owned())
    }
}

pub(crate) fn validate_identifier(field: &'static str, value: &str) -> Result<(), SkillError> {
    if value.is_empty() {
        return Err(SkillError::EmptyField(field));
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
        Err(SkillError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        })
    }
}
