use koa_skill::{DynamicSkillArtifact, SkillArtifactKind, SkillBuilder, SkillError, SkillId};
use koa_toolbox::{
    InMemoryAuditLog, JsonSchema, PolicyEffect, ToolCall, ToolId, ToolPolicy, ToolPolicyRule,
    ToolSpec,
};
use serde_json::json;

#[test]
fn dynamic_artifact_materializes_with_digest() {
    let artifact = DynamicSkillArtifact::new(
        "instructions.md",
        SkillArtifactKind::Instructions,
        "text/markdown",
        b"Use the declared echo tool only.".to_vec(),
    )
    .unwrap();

    assert_eq!(artifact.manifest().size_bytes, 32);
    assert_eq!(artifact.manifest().sha256.len(), 64);

    let directory = tempfile::tempdir().unwrap();
    let path = artifact.materialize(directory.path()).unwrap();

    assert_eq!(
        std::fs::read(path).unwrap(),
        b"Use the declared echo tool only."
    );
}

#[test]
fn executable_or_traversing_artifacts_are_rejected() {
    let shell = DynamicSkillArtifact::new(
        "run.sh",
        SkillArtifactKind::Instructions,
        "text/markdown",
        b"echo unsafe".to_vec(),
    )
    .unwrap_err();
    assert!(matches!(shell, SkillError::ExecutableArtifactRejected(_)));

    let traversal = DynamicSkillArtifact::new(
        "../instructions.md",
        SkillArtifactKind::Instructions,
        "text/markdown",
        b"unsafe".to_vec(),
    )
    .unwrap_err();
    assert!(matches!(traversal, SkillError::InvalidArtifactName(_)));
}

#[test]
fn skill_gate_allows_declared_policy_and_denies_undeclared_tools() {
    let tool_id = ToolId::new("koa.echo").unwrap();
    let tool = ToolSpec::new(
        tool_id.clone(),
        "Echo",
        "Echo a message.",
        JsonSchema::object(),
    )
    .unwrap();
    let policy = ToolPolicy::default_deny().with_rule(
        ToolPolicyRule::allow_exact("allow_echo", tool_id.clone(), "declared echo only").unwrap(),
    );
    let package = SkillBuilder::new(SkillId::new("skill.echo").unwrap(), "0.1.0")
        .display_name("Echo Skill")
        .description("Provides a constrained echo tool.")
        .tool(tool)
        .policy(policy)
        .artifact_bytes(
            "instructions.md",
            SkillArtifactKind::Instructions,
            "text/markdown",
            b"Use echo only.".to_vec(),
        )
        .unwrap()
        .build()
        .unwrap();
    let audit = InMemoryAuditLog::default();
    let gate = koa_skill::SkillToolGate::new(package.manifest, audit.clone());

    let allowed = ToolCall::new("call_1", tool_id, json!({"message": "hello"})).unwrap();
    assert_eq!(
        gate.enforce("agent:test", &allowed).unwrap().effect,
        PolicyEffect::Allow
    );

    let undeclared = ToolCall::new(
        "call_2",
        ToolId::new("koa.network").unwrap(),
        json!({"url": "https://example.invalid"}),
    )
    .unwrap();
    let error = gate.enforce("agent:test", &undeclared).unwrap_err();

    assert_eq!(error.evaluation.effect, PolicyEffect::Deny);
    assert!(error.evaluation.reason.contains("not declared"));
    assert_eq!(audit.events().len(), 2);
}
