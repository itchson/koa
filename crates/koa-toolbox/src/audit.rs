use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

/// The normalized outcome captured for a policy or lifecycle audit event.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Allowed,
    Denied,
    Changed,
    Failed,
}

/// A single immutable audit event.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: u64,
    pub timestamp_ms: u64,
    pub actor: String,
    pub action: String,
    pub resource: String,
    pub outcome: AuditOutcome,
    pub reason: String,
    pub metadata: BTreeMap<String, String>,
}

impl AuditEvent {
    pub fn new(
        actor: impl Into<String>,
        action: impl Into<String>,
        resource: impl Into<String>,
        outcome: AuditOutcome,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            event_id: NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed),
            timestamp_ms: timestamp_millis(),
            actor: actor.into(),
            action: action.into(),
            resource: resource.into(),
            outcome,
            reason: reason.into(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A sink for append-only audit events.
pub trait AuditSink: Clone + Send + Sync + 'static {
    fn record(&self, event: AuditEvent);
}

/// Thread-safe in-memory audit storage used by local registries and tests.
#[derive(Clone, Default)]
pub struct InMemoryAuditLog {
    events: Arc<Mutex<Vec<AuditEvent>>>,
}

impl InMemoryAuditLog {
    pub fn events(&self) -> Vec<AuditEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn clear(&self) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }
}

impl AuditSink for InMemoryAuditLog {
    fn record(&self, event: AuditEvent) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event);
    }
}

pub(crate) fn timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
