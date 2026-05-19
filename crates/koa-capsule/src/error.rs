use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CapsuleError {
    #[error("{feature} is not supported on this platform")]
    UnsupportedPlatform { feature: &'static str },

    #[error("invalid capsule id `{0}`")]
    InvalidCapsuleId(String),

    #[error("invalid snapshot name `{0}`")]
    InvalidSnapshotName(String),

    #[error("invalid command: {0}")]
    InvalidCommand(String),

    #[error("path `{path}` must remain inside `{base}`")]
    PathEscapesCapsule { path: PathBuf, base: PathBuf },

    #[error("snapshot `{0}` already exists")]
    SnapshotExists(String),

    #[error("snapshot `{0}` was not found")]
    SnapshotNotFound(String),

    #[error("cannot restore while `{0}` is mounted")]
    MountedRoot(PathBuf),

    #[error("doctor prerequisites failed: {0}")]
    DoctorFailed(String),

    #[error("capsule exec setup failed at {stage}: {message}")]
    ExecSetup {
        stage: &'static str,
        message: String,
    },

    #[error("io error while {op} `{path}`: {source}")]
    Io {
        op: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("json error while {op} `{path}`: {source}")]
    Json {
        op: &'static str,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

pub type Result<T> = std::result::Result<T, CapsuleError>;

pub(crate) trait IoContext<T> {
    fn with_path(self, op: &'static str, path: impl Into<PathBuf>) -> Result<T>;
}

impl<T> IoContext<T> for std::io::Result<T> {
    fn with_path(self, op: &'static str, path: impl Into<PathBuf>) -> Result<T> {
        self.map_err(|source| CapsuleError::Io {
            op,
            path: path.into(),
            source,
        })
    }
}
