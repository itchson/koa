//! Agent specifications and lifecycle registry for Project Koa.

pub mod error;
pub mod registry;
pub mod types;

pub use error::AgentError;
pub use registry::{AgentLifecycleState, AgentRecord, AgentRegistry, LifecycleTransition};
pub use types::{AgentId, AgentSpec, SkillRef};
