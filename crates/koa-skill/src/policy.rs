use crate::manifest::SkillManifest;
use koa_toolbox::{
    AuditEvent, AuditOutcome, AuditSink, PolicyDenied, PolicyEffect, PolicyEvaluation, ToolCall,
    ToolPolicyAuditor,
};

#[derive(Clone)]
pub struct SkillToolGate<S: AuditSink> {
    manifest: SkillManifest,
    auditor: ToolPolicyAuditor<S>,
    audit: S,
}

impl<S: AuditSink> SkillToolGate<S> {
    pub fn new(manifest: SkillManifest, audit: S) -> Self {
        let auditor = ToolPolicyAuditor::new(manifest.policy.clone(), audit.clone());
        Self {
            manifest,
            auditor,
            audit,
        }
    }

    pub fn evaluate(&self, actor: &str, call: &ToolCall) -> PolicyEvaluation {
        if !self.manifest.declares_tool(&call.tool_id) {
            let evaluation = PolicyEvaluation::new(
                call.tool_id.clone(),
                PolicyEffect::Deny,
                None,
                format!(
                    "tool `{}` is not declared by skill `{}`",
                    call.tool_id, self.manifest.id
                ),
            );

            self.audit.record(
                AuditEvent::new(
                    actor,
                    "skill.tool.policy.evaluate",
                    call.tool_id.as_str(),
                    AuditOutcome::Denied,
                    evaluation.reason.clone(),
                )
                .with_metadata("call_id", call.call_id.clone())
                .with_metadata("skill_id", self.manifest.id.as_str()),
            );

            return evaluation;
        }

        self.auditor.evaluate(actor, call)
    }

    pub fn enforce(&self, actor: &str, call: &ToolCall) -> Result<PolicyEvaluation, PolicyDenied> {
        let evaluation = self.evaluate(actor, call);
        if evaluation.is_allowed() {
            Ok(evaluation)
        } else {
            Err(PolicyDenied::new(evaluation))
        }
    }

    pub fn manifest(&self) -> &SkillManifest {
        &self.manifest
    }
}
