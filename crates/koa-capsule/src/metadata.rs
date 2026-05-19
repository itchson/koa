use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{CapsuleError, IoContext, Result};

const CAPSULE_ID_MAX: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CapsuleId(String);

impl CapsuleId {
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        validate_name(&id, CAPSULE_ID_MAX)
            .map_err(|_| CapsuleError::InvalidCapsuleId(id.clone()))?;
        Ok(Self(id))
    }

    pub fn generate() -> Self {
        let nanos = unix_nanos();
        let pid = std::process::id();
        Self(format!("capsule-{pid:x}-{nanos:x}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CapsuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for CapsuleId {
    type Error = CapsuleError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl From<CapsuleId> for String {
    fn from(value: CapsuleId) -> Self {
        value.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub name: String,
    pub created_at_unix: u64,
    pub upper_dir: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapsuleMetadata {
    pub schema_version: u32,
    pub id: CapsuleId,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    pub state_dir: PathBuf,
    pub lower_dir: PathBuf,
    pub upper_dir: PathBuf,
    pub work_dir: PathBuf,
    pub merged_dir: PathBuf,
    pub labels: BTreeMap<String, String>,
    pub snapshots: BTreeMap<String, SnapshotMetadata>,
}

impl CapsuleMetadata {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn new(
        id: CapsuleId,
        state_dir: PathBuf,
        lower_dir: PathBuf,
        labels: BTreeMap<String, String>,
    ) -> Self {
        let now = unix_secs();
        Self {
            schema_version: Self::SCHEMA_VERSION,
            id,
            created_at_unix: now,
            updated_at_unix: now,
            upper_dir: state_dir.join("upper"),
            work_dir: state_dir.join("work"),
            merged_dir: state_dir.join("rootfs"),
            state_dir,
            lower_dir,
            labels,
            snapshots: BTreeMap::new(),
        }
    }

    pub fn touch(&mut self) {
        self.updated_at_unix = unix_secs();
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_path("reading", path)?;
        serde_json::from_slice(&bytes).map_err(|source| CapsuleError::Json {
            op: "parsing",
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).with_path("creating", parent)?;
        let tmp = parent.join(format!(".{}.tmp", file_name(path)));
        let bytes = serde_json::to_vec_pretty(self).map_err(|source| CapsuleError::Json {
            op: "serializing",
            path: path.to_path_buf(),
            source,
        })?;
        fs::write(&tmp, bytes).with_path("writing", &tmp)?;
        fs::rename(&tmp, path).with_path("renaming", path)?;
        Ok(())
    }
}

pub(crate) fn validate_snapshot_name(name: &str) -> Result<()> {
    validate_name(name, CAPSULE_ID_MAX)
        .map_err(|_| CapsuleError::InvalidSnapshotName(name.to_owned()))
}

pub(crate) fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn validate_name(value: &str, max: usize) -> std::result::Result<(), ()> {
    if value.is_empty() || value.len() > max || value == "." || value == ".." {
        return Err(());
    }
    if value.starts_with('.') || value.ends_with('.') {
        return Err(());
    }
    if value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(())
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("metadata.json")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_capsule_ids() {
        for value in ["", ".", "..", "../x", ".hidden", "has/slash", "has space"] {
            assert!(CapsuleId::new(value).is_err(), "{value}");
        }
    }

    #[test]
    fn accepts_portable_capsule_ids() {
        let id = CapsuleId::new("koa_capsule-01.test").unwrap();
        assert_eq!(id.as_str(), "koa_capsule-01.test");
    }

    #[test]
    fn metadata_roundtrips_json() {
        let temp = tempfile::tempdir().unwrap();
        let meta_path = temp.path().join("metadata.json");
        let metadata = CapsuleMetadata::new(
            CapsuleId::new("roundtrip").unwrap(),
            temp.path().join("state"),
            temp.path().join("lower"),
            BTreeMap::new(),
        );

        metadata.save_atomic(&meta_path).unwrap();
        assert_eq!(CapsuleMetadata::load(&meta_path).unwrap(), metadata);
    }
}
