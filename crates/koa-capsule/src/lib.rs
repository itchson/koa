//! Koa capsule primitives.
//!
//! This crate intentionally uses Linux primitives directly: namespace syscalls,
//! overlayfs mounts, cgroup v2 files, and seccomp filters. There is no Docker
//! integration and no mock execution fallback. On unsupported hosts, doctor and
//! execution APIs fail closed with explicit prerequisite errors.

mod capsule;
mod doctor;
mod error;
mod exec;
mod metadata;
mod snapshot;
mod sys;

pub use capsule::{Capsule, CapsuleConfig};
pub use doctor::{CheckStatus, Doctor, DoctorCheck, DoctorReport};
pub use error::{CapsuleError, Result};
pub use exec::{CapsuleExit, CgroupLimits, CpuLimit, ExecRequest};
pub use metadata::{CapsuleId, CapsuleMetadata, SnapshotMetadata};
