use crate::error::SkillError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillArtifactKind {
    Instructions,
    Data,
    Template,
    Asset,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SkillArtifact {
    pub name: String,
    pub kind: SkillArtifactKind,
    pub media_type: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicSkillArtifact {
    manifest: SkillArtifact,
    bytes: Vec<u8>,
}

impl DynamicSkillArtifact {
    pub fn new(
        name: impl Into<String>,
        kind: SkillArtifactKind,
        media_type: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<Self, SkillError> {
        let name = name.into();
        validate_artifact_name(&name)?;
        let media_type = media_type.into();
        if is_executable_media_type(&media_type) {
            return Err(SkillError::ExecutableArtifactRejected(name));
        }

        let manifest = SkillArtifact {
            name,
            kind,
            media_type,
            size_bytes: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        };

        Ok(Self { manifest, bytes })
    }

    pub fn manifest(&self) -> &SkillArtifact {
        &self.manifest
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn materialize(&self, root: impl AsRef<Path>) -> Result<PathBuf, SkillError> {
        fs::create_dir_all(root.as_ref())?;
        let path = root.as_ref().join(&self.manifest.name);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&self.bytes)?;
        Ok(path)
    }
}

fn validate_artifact_name(name: &str) -> Result<(), SkillError> {
    if name.trim().is_empty()
        || name != name.trim()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains(':')
        || name.chars().any(char::is_control)
        || Path::new(name).is_absolute()
    {
        return Err(SkillError::InvalidArtifactName(name.to_owned()));
    }

    if is_executable_name(name) {
        return Err(SkillError::ExecutableArtifactRejected(name.to_owned()));
    }

    Ok(())
}

fn is_executable_name(name: &str) -> bool {
    const EXECUTABLE_EXTENSIONS: &[&str] = &[
        "bat", "bash", "cmd", "com", "cjs", "dll", "dylib", "exe", "fish", "jar", "js", "mjs",
        "msi", "pl", "ps1", "py", "rb", "scr", "sh", "so", "ts", "vbs", "zsh",
    ];

    Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            EXECUTABLE_EXTENSIONS
                .iter()
                .any(|blocked| extension.eq_ignore_ascii_case(blocked))
        })
        .unwrap_or(false)
}

fn is_executable_media_type(media_type: &str) -> bool {
    const EXECUTABLE_MEDIA_TYPES: &[&str] = &[
        "application/javascript",
        "application/x-dosexec",
        "application/x-executable",
        "application/x-msdownload",
        "application/x-powershell",
        "application/x-sh",
        "text/javascript",
        "text/x-python",
        "text/x-shellscript",
    ];

    let media_type = media_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    EXECUTABLE_MEDIA_TYPES
        .iter()
        .any(|blocked| media_type == *blocked)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(hex_digit(byte >> 4));
        output.push(hex_digit(byte & 0x0f));
    }
    output
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => char::from(b'0' + nibble),
        10..=15 => char::from(b'a' + nibble - 10),
        _ => unreachable!("nibble is masked"),
    }
}
