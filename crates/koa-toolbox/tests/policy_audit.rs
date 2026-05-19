use koa_toolbox::{
    InMemoryAuditLog, PolicyEffect, ToolCall, ToolId, ToolPolicy, ToolPolicyAuditor, ToolPolicyRule,
};
use serde_json::json;

#[test]
fn default_policy_denies_and_records_audit() {
    let audit = InMemoryAuditLog::default();
    let auditor = ToolPolicyAuditor::new(ToolPolicy::default_deny(), audit.clone());
    let call = ToolCall::new("call_1", ToolId::new("koa.echo").unwrap(), json!({})).unwrap();

    let error = auditor.enforce("agent:test", &call).unwrap_err();

    assert_eq!(error.evaluation.effect, PolicyEffect::Deny);
    assert_eq!(error.evaluation.rule_id, None);
    let events = audit.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].actor, "agent:test");
    assert_eq!(events[0].resource, "koa.echo");
    assert_eq!(events[0].metadata.get("call_id").unwrap(), "call_1");
}

#[test]
fn explicit_allow_authorizes_and_audits() {
    let audit = InMemoryAuditLog::default();
    let tool_id = ToolId::new("koa.echo").unwrap();
    let policy = ToolPolicy::default_deny().with_rule(
        ToolPolicyRule::allow_exact("allow_echo", tool_id.clone(), "echo is safe").unwrap(),
    );
    let auditor = ToolPolicyAuditor::new(policy, audit.clone());
    let call = ToolCall::new("call_2", tool_id, json!({"message": "hello"})).unwrap();

    let evaluation = auditor.enforce("agent:test", &call).unwrap();

    assert_eq!(evaluation.effect, PolicyEffect::Allow);
    assert_eq!(evaluation.rule_id.as_deref(), Some("allow_echo"));
    assert_eq!(
        audit.events()[0].outcome,
        koa_toolbox::AuditOutcome::Allowed
    );
}

#[test]
fn explicit_deny_takes_precedence_over_allow() {
    let audit = InMemoryAuditLog::default();
    let tool_id = ToolId::new("koa.network").unwrap();
    let policy = ToolPolicy::default_deny()
        .with_rule(
            ToolPolicyRule::allow_exact("allow_network", tool_id.clone(), "temporarily allowed")
                .unwrap(),
        )
        .with_rule(
            ToolPolicyRule::deny_exact("deny_network", tool_id.clone(), "network disabled")
                .unwrap(),
        );
    let auditor = ToolPolicyAuditor::new(policy, audit);
    let call = ToolCall::new("call_3", tool_id, json!({})).unwrap();

    let error = auditor.enforce("agent:test", &call).unwrap_err();

    assert_eq!(error.evaluation.effect, PolicyEffect::Deny);
    assert_eq!(error.evaluation.rule_id.as_deref(), Some("deny_network"));
}
