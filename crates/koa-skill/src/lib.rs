//! Skill manifests and dynamic, non-executable skill artifacts for Project Koa.

pub mod artifact;
pub mod builder;
pub mod error;
pub mod manifest;
pub mod policy;
pub mod types;

pub use artifact::{DynamicSkillArtifact, SkillArtifact, SkillArtifactKind};
pub use builder::{SkillBuilder, SkillPackage};
pub use error::SkillError;
pub use manifest::SkillManifest;
pub use policy::SkillToolGate;
pub use types::SkillId;
