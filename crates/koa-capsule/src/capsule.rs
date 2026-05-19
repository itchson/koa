use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{CapsuleError, IoContext, Result};
use crate::exec::{CapsuleExit, ExecRequest};
use crate::metadata::{
    CapsuleId, CapsuleMetadata, SnapshotMetadata, unix_secs, validate_snapshot_name,
};
use crate::snapshot::{copy_tree, ensure_relative_path, replace_tree};
use crate::sys;

#[derive(Clone, Debug)]
pub struct CapsuleConfig {
    pub id: Option<CapsuleId>,
    pub lower_dir: PathBuf,
    pub state_root: PathBuf,
    pub labels: BTreeMap<String, String>,
}

impl CapsuleConfig {
    pub fn new(lower_dir: impl Into<PathBuf>, state_root: impl Into<PathBuf>) -> Self {
        Self {
            id: None,
            lower_dir: lower_dir.into(),
            state_root: state_root.into(),
            labels: BTreeMap::new(),
        }
    }

    pub fn with_id(mut self, id: CapsuleId) -> Self {
        self.id = Some(id);
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

#[derive(Clone, Debug)]
pub struct Capsule {
    metadata_path: PathBuf,
    metadata: CapsuleMetadata,
}

impl Capsule {
    pub const METADATA_FILE: &'static str = "metadata.json";

    pub fn create(config: CapsuleConfig) -> Result<Self> {
        let id = config.id.unwrap_or_else(CapsuleId::generate);
        let state_dir = config.state_root.join(id.as_str());
        let metadata_path = state_dir.join(Self::METADATA_FILE);
        let metadata = CapsuleMetadata::new(id, state_dir, config.lower_dir, config.labels);

        fs::create_dir_all(&metadata.upper_dir).with_path("creating", &metadata.upper_dir)?;
        fs::create_dir_all(&metadata.work_dir).with_path("creating", &metadata.work_dir)?;
        fs::create_dir_all(&metadata.merged_dir).with_path("creating", &metadata.merged_dir)?;
        fs::create_dir_all(metadata.state_dir.join("snapshots"))
            .with_path("creating", metadata.state_dir.join("snapshots"))?;
        metadata.save_atomic(&metadata_path)?;

        Ok(Self {
            metadata_path,
            metadata,
        })
    }

    pub fn open(state_dir: impl AsRef<Path>) -> Result<Self> {
        let state_dir = state_dir.as_ref();
        let metadata_path = state_dir.join(Self::METADATA_FILE);
        let metadata = CapsuleMetadata::load(&metadata_path)?;
        validate_state_paths(state_dir, &metadata)?;
        Ok(Self {
            metadata_path,
            metadata,
        })
    }

    pub fn metadata(&self) -> &CapsuleMetadata {
        &self.metadata
    }

    pub fn metadata_path(&self) -> &Path {
        &self.metadata_path
    }

    pub fn mount_overlay(&self) -> Result<()> {
        sys::mount_overlay(
            &self.metadata.lower_dir,
            &self.metadata.upper_dir,
            &self.metadata.work_dir,
            &self.metadata.merged_dir,
        )
    }

    pub fn unmount_overlay(&self) -> Result<()> {
        sys::unmount(&self.metadata.merged_dir)
    }

    pub fn is_mounted(&self) -> Result<bool> {
        sys::is_mount_point(&self.metadata.merged_dir)
    }

    pub fn snapshot(&mut self, name: &str) -> Result<SnapshotMetadata> {
        validate_snapshot_name(name)?;
        if self.metadata.snapshots.contains_key(name) {
            return Err(CapsuleError::SnapshotExists(name.to_owned()));
        }

        let relative_upper = PathBuf::from("snapshots").join(name).join("upper");
        ensure_relative_path(&relative_upper)?;
        let snapshot_upper = self.metadata.state_dir.join(relative_upper);
        copy_tree(&self.metadata.upper_dir, &snapshot_upper)?;

        let snapshot = SnapshotMetadata {
            name: name.to_owned(),
            created_at_unix: unix_secs(),
            upper_dir: snapshot_upper,
        };
        self.metadata
            .snapshots
            .insert(name.to_owned(), snapshot.clone());
        self.metadata.touch();
        self.persist()?;
        Ok(snapshot)
    }

    pub fn restore(&mut self, name: &str) -> Result<()> {
        validate_snapshot_name(name)?;
        if self.is_mounted()? {
            return Err(CapsuleError::MountedRoot(self.metadata.merged_dir.clone()));
        }
        let snapshot = self
            .metadata
            .snapshots
            .get(name)
            .cloned()
            .ok_or_else(|| CapsuleError::SnapshotNotFound(name.to_owned()))?;

        ensure_snapshot_inside_state(&self.metadata.state_dir, &snapshot.upper_dir)?;
        replace_tree(&snapshot.upper_dir, &self.metadata.upper_dir)?;
        fs::create_dir_all(&self.metadata.work_dir)
            .with_path("creating", &self.metadata.work_dir)?;
        fs::create_dir_all(&self.metadata.merged_dir)
            .with_path("creating", &self.metadata.merged_dir)?;
        self.metadata.touch();
        self.persist()
    }

    pub fn exec(&self, request: ExecRequest) -> Result<CapsuleExit> {
        request.validate()?;
        request.doctor_report().ensure_prerequisites()?;
        self.mount_overlay()?;
        sys::exec_capsule(&self.metadata, &request)
    }

    fn persist(&self) -> Result<()> {
        self.metadata.save_atomic(&self.metadata_path)
    }
}

fn validate_state_paths(state_dir: &Path, metadata: &CapsuleMetadata) -> Result<()> {
    for path in [
        &metadata.upper_dir,
        &metadata.work_dir,
        &metadata.merged_dir,
    ] {
        ensure_snapshot_inside_state(state_dir, path)?;
    }
    Ok(())
}

fn ensure_snapshot_inside_state(state_dir: &Path, path: &Path) -> Result<()> {
    let base = normalize_existing_or_parent(state_dir)?;
    let candidate = normalize_existing_or_parent(path)?;
    if candidate.starts_with(&base) {
        Ok(())
    } else {
        Err(CapsuleError::PathEscapesCapsule {
            path: path.to_path_buf(),
            base,
        })
    }
}

fn normalize_existing_or_parent(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return path.canonicalize().with_path("canonicalizing", path);
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let canonical_parent = if parent.exists() {
        parent.canonicalize().with_path("canonicalizing", parent)?
    } else {
        normalize_existing_or_parent(parent)?
    };
    Ok(canonical_parent.join(path.file_name().unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_capsule_state_and_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let lower = temp.path().join("lower");
        fs::create_dir(&lower).unwrap();

        let capsule = Capsule::create(
            CapsuleConfig::new(&lower, temp.path().join("state"))
                .with_id(CapsuleId::new("capsule-a").unwrap()),
        )
        .unwrap();

        assert!(capsule.metadata().upper_dir.is_dir());
        assert!(capsule.metadata().work_dir.is_dir());
        assert!(capsule.metadata().merged_dir.is_dir());
        assert!(capsule.metadata_path().is_file());
    }

    #[test]
    fn snapshots_and_restores_upper_without_root() {
        let temp = tempfile::tempdir().unwrap();
        let lower = temp.path().join("lower");
        fs::create_dir(&lower).unwrap();

        let mut capsule = Capsule::create(
            CapsuleConfig::new(&lower, temp.path().join("state"))
                .with_id(CapsuleId::new("capsule-b").unwrap()),
        )
        .unwrap();

        let file = capsule.metadata().upper_dir.join("etc").join("koa.conf");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "before").unwrap();
        capsule.snapshot("clean").unwrap();
        fs::write(&file, "after").unwrap();

        capsule.restore("clean").unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "before");
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn exec_fails_closed_when_linux_primitives_are_missing() {
        let temp = tempfile::tempdir().unwrap();
        let lower = temp.path().join("lower");
        fs::create_dir(&lower).unwrap();
        let capsule = Capsule::create(
            CapsuleConfig::new(&lower, temp.path().join("state"))
                .with_id(CapsuleId::new("capsule-c").unwrap()),
        )
        .unwrap();

        let err = capsule.exec(ExecRequest::new("sh")).unwrap_err();
        assert!(err.to_string().contains("doctor prerequisites failed"));
    }
}
