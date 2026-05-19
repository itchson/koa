use crate::audit::{AuditEvent, AuditOutcome, AuditSink};
use crate::types::{KoaToolboxError, ToolCall, ToolId, require_non_empty, validate_identifier};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyEffect {
    Allow,
    Deny,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ToolMatcher {
    Exact(ToolId),
    Prefix(String),
}

impl ToolMatcher {
    pub fn prefix(prefix: impl Into<String>) -> Result<Self, KoaToolboxError> {
        let prefix = prefix.into();
        validate_identifier("tool_prefix", &prefix)?;
        Ok(Self::Prefix(prefix))
    }

    pub fn matches(&self, tool_id: &ToolId) -> bool {
        match self {
            Self::Exact(expected) => expected == tool_id,
            Self::Prefix(prefix) => tool_id.as_str().starts_with(prefix),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolPolicyRule {
    pub id: String,
    pub matcher: ToolMatcher,
    pub effect: PolicyEffect,
    pub reason: String,
}

impl ToolPolicyRule {
    pub fn new(
        id: impl Into<String>,
        matcher: ToolMatcher,
        effect: PolicyEffect,
        reason: impl Into<String>,
    ) -> Result<Self, KoaToolboxError> {
        let id = id.into();
        validate_identifier("rule_id", &id)?;
        let reason = require_non_empty("reason", reason.into())?;

        Ok(Self {
            id,
            matcher,
            effect,
            reason,
        })
    }

    pub fn allow_exact(
        id: impl Into<String>,
        tool_id: ToolId,
        reason: impl Into<String>,
    ) -> Result<Self, KoaToolboxError> {
        Self::new(id, ToolMatcher::Exact(tool_id), PolicyEffect::Allow, reason)
    }

    pub fn deny_exact(
        id: impl Into<String>,
        tool_id: ToolId,
        reason: impl Into<String>,
    ) -> Result<Self, KoaToolboxError> {
        Self::new(id, ToolMatcher::Exact(tool_id), PolicyEffect::Deny, reason)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolPolicy {
    pub default_effect: PolicyEffect,
    pub rules: Vec<ToolPolicyRule>,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self::default_deny()
    }
}

impl ToolPolicy {
    pub fn default_deny() -> Self {
        Self {
            default_effect: PolicyEffect::Deny,
            rules: Vec::new(),
        }
    }

    pub fn default_allow() -> Self {
        Self {
            default_effect: PolicyEffect::Allow,
            rules: Vec::new(),
        }
    }

    pub fn with_rule(mut self, rule: ToolPolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn evaluate(&self, call: &ToolCall) -> PolicyEvaluation {
        let matching_rules = self
            .rules
            .iter()
            .filter(|rule| rule.matcher.matches(&call.tool_id));

        let mut first_allow = None;
        for rule in matching_rules {
            match rule.effect {
                PolicyEffect::Deny => {
                    return PolicyEvaluation::new(
                        call.tool_id.clone(),
                        PolicyEffect::Deny,
                        Some(rule.id.clone()),
                        rule.reason.clone(),
                    );
                }
                PolicyEffect::Allow => {
                    first_allow.get_or_insert(rule);
                }
            }
        }

        if let Some(rule) = first_allow {
            return PolicyEvaluation::new(
                call.tool_id.clone(),
                PolicyEffect::Allow,
                Some(rule.id.clone()),
                rule.reason.clone(),
            );
        }

        PolicyEvaluation::new(
            call.tool_id.clone(),
            self.default_effect.clone(),
            None,
            match self.default_effect {
                PolicyEffect::Allow => "allowed by default policy",
                PolicyEffect::Deny => "denied by default policy",
            },
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub tool_id: ToolId,
    pub effect: PolicyEffect,
    pub rule_id: Option<String>,
    pub reason: String,
}

impl PolicyEvaluation {
    pub fn new(
        tool_id: ToolId,
        effect: PolicyEffect,
        rule_id: Option<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            tool_id,
            effect,
            rule_id,
            reason: reason.into(),
        }
    }

    pub fn is_allowed(&self) -> bool {
        self.effect == PolicyEffect::Allow
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyDenied {
    pub evaluation: PolicyEvaluation,
}

impl PolicyDenied {
    pub fn new(evaluation: PolicyEvaluation) -> Self {
        Self { evaluation }
    }
}

impl fmt::Display for PolicyDenied {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "tool `{}` denied: {}",
            self.evaluation.tool_id, self.evaluation.reason
        )
    }
}

impl std::error::Error for PolicyDenied {}

#[derive(Clone)]
pub struct ToolPolicyAuditor<S: AuditSink> {
    policy: ToolPolicy,
    audit: S,
}

impl<S: AuditSink> ToolPolicyAuditor<S> {
    pub fn new(policy: ToolPolicy, audit: S) -> Self {
        Self { policy, audit }
    }

    pub fn evaluate(&self, actor: &str, call: &ToolCall) -> PolicyEvaluation {
        let evaluation = self.policy.evaluate(call);
        let outcome = if evaluation.is_allowed() {
            AuditOutcome::Allowed
        } else {
            AuditOutcome::Denied
        };

        self.audit.record(
            AuditEvent::new(
                actor,
                "tool.policy.evaluate",
                call.tool_id.as_str(),
                outcome,
                evaluation.reason.clone(),
            )
            .with_metadata("call_id", call.call_id.clone())
            .with_metadata("tool_id", call.tool_id.as_str()),
        );

        evaluation
    }

    pub fn enforce(&self, actor: &str, call: &ToolCall) -> Result<PolicyEvaluation, PolicyDenied> {
        let evaluation = self.evaluate(actor, call);
        if evaluation.is_allowed() {
            Ok(evaluation)
        } else {
            Err(PolicyDenied::new(evaluation))
        }
    }

    pub fn policy(&self) -> &ToolPolicy {
        &self.policy
    }

    pub fn audit(&self) -> &S {
        &self.audit
    }
}
