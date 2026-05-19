use crate::error::AgentError;
use crate::types::{AgentId, AgentSpec};
use koa_toolbox::{AuditEvent, AuditOutcome, AuditSink, InMemoryAuditLog};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentLifecycleState {
    Registered,
    Active,
    Suspended,
    Retired,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LifecycleTransition {
    pub from: Option<AgentLifecycleState>,
    pub to: AgentLifecycleState,
    pub actor: String,
    pub reason: String,
    pub timestamp_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentRecord {
    pub spec: AgentSpec,
    pub state: AgentLifecycleState,
    pub revision: u64,
    pub transitions: Vec<LifecycleTransition>,
}

#[derive(Clone)]
pub struct AgentRegistry<S: AuditSink = InMemoryAuditLog> {
    records: BTreeMap<AgentId, AgentRecord>,
    audit: S,
}

impl AgentRegistry<InMemoryAuditLog> {
    pub fn new() -> Self {
        Self::with_audit(InMemoryAuditLog::default())
    }
}

impl Default for AgentRegistry<InMemoryAuditLog> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: AuditSink> AgentRegistry<S> {
    pub fn with_audit(audit: S) -> Self {
        Self {
            records: BTreeMap::new(),
            audit,
        }
    }

    pub fn register(
        &mut self,
        actor: impl Into<String>,
        spec: AgentSpec,
        reason: impl Into<String>,
    ) -> Result<&AgentRecord, AgentError> {
        if self.records.contains_key(&spec.id) {
            return Err(AgentError::AlreadyRegistered(spec.id.to_string()));
        }

        let actor = actor.into();
        let reason = reason.into();
        let id = spec.id.clone();
        let transition = LifecycleTransition {
            from: None,
            to: AgentLifecycleState::Registered,
            actor: actor.clone(),
            reason: reason.clone(),
            timestamp_ms: timestamp_millis(),
        };
        let record = AgentRecord {
            spec,
            state: AgentLifecycleState::Registered,
            revision: 1,
            transitions: vec![transition],
        };

        self.records.insert(id.clone(), record);
        self.record_transition_audit(
            &actor,
            &id,
            None,
            &AgentLifecycleState::Registered,
            &reason,
            1,
        );
        Ok(self
            .records
            .get(&id)
            .expect("record was inserted immediately before lookup"))
    }

    pub fn activate(
        &mut self,
        actor: impl Into<String>,
        id: &AgentId,
        reason: impl Into<String>,
    ) -> Result<&AgentRecord, AgentError> {
        self.transition(
            actor,
            id,
            AgentLifecycleState::Active,
            reason,
            &[
                AgentLifecycleState::Registered,
                AgentLifecycleState::Suspended,
            ],
        )
    }

    pub fn suspend(
        &mut self,
        actor: impl Into<String>,
        id: &AgentId,
        reason: impl Into<String>,
    ) -> Result<&AgentRecord, AgentError> {
        self.transition(
            actor,
            id,
            AgentLifecycleState::Suspended,
            reason,
            &[AgentLifecycleState::Active],
        )
    }

    pub fn retire(
        &mut self,
        actor: impl Into<String>,
        id: &AgentId,
        reason: impl Into<String>,
    ) -> Result<&AgentRecord, AgentError> {
        self.transition(
            actor,
            id,
            AgentLifecycleState::Retired,
            reason,
            &[
                AgentLifecycleState::Registered,
                AgentLifecycleState::Active,
                AgentLifecycleState::Suspended,
            ],
        )
    }

    pub fn get(&self, id: &AgentId) -> Option<&AgentRecord> {
        self.records.get(id)
    }

    pub fn list(&self) -> Vec<&AgentRecord> {
        self.records.values().collect()
    }

    pub fn audit(&self) -> &S {
        &self.audit
    }

    fn transition(
        &mut self,
        actor: impl Into<String>,
        id: &AgentId,
        to: AgentLifecycleState,
        reason: impl Into<String>,
        allowed_from: &[AgentLifecycleState],
    ) -> Result<&AgentRecord, AgentError> {
        let actor = actor.into();
        let reason = reason.into();
        let (from, revision) = {
            let record = self
                .records
                .get_mut(id)
                .ok_or_else(|| AgentError::MissingAgent(id.to_string()))?;
            let from = record.state.clone();
            if !allowed_from.contains(&from) {
                return Err(AgentError::InvalidTransition {
                    id: id.to_string(),
                    from,
                    to,
                });
            }

            record.state = to.clone();
            record.revision += 1;
            record.transitions.push(LifecycleTransition {
                from: Some(from.clone()),
                to: to.clone(),
                actor: actor.clone(),
                reason: reason.clone(),
                timestamp_ms: timestamp_millis(),
            });
            (from, record.revision)
        };

        self.record_transition_audit(&actor, id, Some(&from), &to, &reason, revision);
        Ok(self
            .records
            .get(id)
            .expect("record was updated immediately before lookup"))
    }

    fn record_transition_audit(
        &self,
        actor: &str,
        id: &AgentId,
        from: Option<&AgentLifecycleState>,
        to: &AgentLifecycleState,
        reason: &str,
        revision: u64,
    ) {
        let mut event = AuditEvent::new(
            actor,
            "agent.lifecycle.transition",
            id.as_str(),
            AuditOutcome::Changed,
            reason,
        )
        .with_metadata("to", format!("{to:?}").to_ascii_lowercase())
        .with_metadata("revision", revision.to_string());

        if let Some(from) = from {
            event = event.with_metadata("from", format!("{from:?}").to_ascii_lowercase());
        }

        self.audit.record(event);
    }
}

fn timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
