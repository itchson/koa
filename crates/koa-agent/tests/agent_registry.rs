use koa_agent::{AgentError, AgentId, AgentLifecycleState, AgentRegistry, AgentSpec, SkillRef};
use koa_skill::SkillId;
use koa_toolbox::{InMemoryAuditLog, JsonSchema, ToolId, ToolPolicy, ToolSpec};

fn agent_spec(id: &str) -> AgentSpec {
    let tool = ToolSpec::new(
        ToolId::new("koa.echo").unwrap(),
        "Echo",
        "Echo a message.",
        JsonSchema::object(),
    )
    .unwrap();

    AgentSpec::new(
        AgentId::new(id).unwrap(),
        "0.1.0",
        "Echo Agent",
        "Uses the echo skill and no undeclared fallback data.",
        vec![SkillRef::required(SkillId::new("skill.echo").unwrap())],
        vec![tool],
        ToolPolicy::default_deny(),
    )
    .unwrap()
}

#[test]
fn registry_tracks_agent_lifecycle_and_audit_events() {
    let audit = InMemoryAuditLog::default();
    let mut registry = AgentRegistry::with_audit(audit.clone());
    let id = AgentId::new("agent.echo").unwrap();

    registry
        .register("system", agent_spec("agent.echo"), "initial registration")
        .unwrap();
    assert_eq!(
        registry.get(&id).unwrap().state,
        AgentLifecycleState::Registered
    );

    registry.activate("system", &id, "ready").unwrap();
    registry.suspend("operator", &id, "maintenance").unwrap();
    let record = registry.retire("operator", &id, "decommissioned").unwrap();

    assert_eq!(record.state, AgentLifecycleState::Retired);
    assert_eq!(record.revision, 4);
    assert_eq!(record.transitions.len(), 4);

    let events = audit.events();
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].metadata.get("to").unwrap(), "registered");
    assert_eq!(events[1].metadata.get("to").unwrap(), "active");
    assert_eq!(events[2].metadata.get("from").unwrap(), "active");
    assert_eq!(events[3].metadata.get("to").unwrap(), "retired");
}

#[test]
fn registry_rejects_duplicate_registration_and_invalid_transition() {
    let mut registry = AgentRegistry::new();
    let id = AgentId::new("agent.echo").unwrap();

    registry
        .register("system", agent_spec("agent.echo"), "initial registration")
        .unwrap();
    let duplicate = registry
        .register("system", agent_spec("agent.echo"), "duplicate")
        .unwrap_err();
    assert!(matches!(duplicate, AgentError::AlreadyRegistered(_)));

    let invalid = registry.suspend("system", &id, "not active").unwrap_err();
    assert!(matches!(invalid, AgentError::InvalidTransition { .. }));
}

#[test]
fn registry_returns_missing_instead_of_fallback_record() {
    let mut registry = AgentRegistry::new();
    let id = AgentId::new("agent.missing").unwrap();

    assert!(registry.get(&id).is_none());

    let missing = registry.activate("system", &id, "activate").unwrap_err();
    assert!(matches!(missing, AgentError::MissingAgent(_)));
}
