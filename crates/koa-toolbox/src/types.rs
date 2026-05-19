use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KoaToolboxError {
    EmptyField(&'static str),
    InvalidIdentifier { field: &'static str, value: String },
    InvalidJsonSchema(String),
}

impl fmt::Display for KoaToolboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "`{field}` cannot be empty"),
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "`{field}` has an invalid identifier: `{value}`")
            }
            Self::InvalidJsonSchema(reason) => write!(formatter, "invalid JSON schema: {reason}"),
        }
    }
}

impl std::error::Error for KoaToolboxError {}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolId(String);

impl ToolId {
    pub fn new(value: impl Into<String>) -> Result<Self, KoaToolboxError> {
        let value = value.into();
        validate_identifier("tool_id", &value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl AsRef<str> for ToolId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JsonSchema(Value);

impl JsonSchema {
    pub fn new(value: Value) -> Result<Self, KoaToolboxError> {
        match value {
            Value::Object(_) => Ok(Self(value)),
            other => Err(KoaToolboxError::InvalidJsonSchema(format!(
                "schema root must be an object, got {other}"
            ))),
        }
    }

    pub fn object() -> Self {
        Self(Value::Object(Default::default()))
    }

    pub fn value(&self) -> &Value {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub id: ToolId,
    pub display_name: String,
    pub description: String,
    pub input_schema: JsonSchema,
}

impl ToolSpec {
    pub fn new(
        id: ToolId,
        display_name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonSchema,
    ) -> Result<Self, KoaToolboxError> {
        let display_name = require_non_empty("display_name", display_name.into())?;
        let description = require_non_empty("description", description.into())?;

        Ok(Self {
            id,
            display_name,
            description,
            input_schema,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    pub tool_id: ToolId,
    pub input: Value,
}

impl ToolCall {
    pub fn new(
        call_id: impl Into<String>,
        tool_id: ToolId,
        input: Value,
    ) -> Result<Self, KoaToolboxError> {
        let call_id = call_id.into();
        validate_identifier("call_id", &call_id)?;

        Ok(Self {
            call_id,
            tool_id,
            input,
        })
    }
}

pub(crate) fn require_non_empty(
    field: &'static str,
    value: String,
) -> Result<String, KoaToolboxError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(KoaToolboxError::EmptyField(field))
    } else {
        Ok(trimmed.to_owned())
    }
}

pub(crate) fn validate_identifier(field: &'static str, value: &str) -> Result<(), KoaToolboxError> {
    if value.is_empty() {
        return Err(KoaToolboxError::EmptyField(field));
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
        Err(KoaToolboxError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        })
    }
}
