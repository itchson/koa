use std::fs;
use std::path::{Component, Path};

use crate::error::{CapsuleError, IoContext, Result};

pub(crate) fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    if !from.exists() {
        fs::create_dir_all(to).with_path("creating", to)?;
        return Ok(());
    }
    let metadata = fs::symlink_metadata(from).with_path("reading metadata for", from)?;
    if metadata.file_type().is_symlink() {
        copy_symlink(from, to)?;
        return Ok(());
    }
    if metadata.is_file() {
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent).with_path("creating", parent)?;
        }
        fs::copy(from, to).with_path("copying", to)?;
        fs::set_permissions(to, metadata.permissions()).with_path("setting permissions on", to)?;
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    fs::create_dir_all(to).with_path("creating", to)?;
    fs::set_permissions(to, metadata.permissions()).with_path("setting permissions on", to)?;
    for entry in fs::read_dir(from).with_path("reading", from)? {
        let entry = entry.with_path("reading entry in", from)?;
        let name = entry.file_name();
        copy_tree(&entry.path(), &to.join(name))?;
    }
    Ok(())
}

pub(crate) fn replace_tree(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        fs::remove_dir_all(to).with_path("removing", to)?;
    }
    copy_tree(from, to)
}

pub(crate) fn ensure_relative_path(path: &Path) -> Result<()> {
    if path.is_absolute() {
        return Err(CapsuleError::PathEscapesCapsule {
            path: path.to_path_buf(),
            base: Path::new(".").to_path_buf(),
        });
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(CapsuleError::PathEscapesCapsule {
            path: path.to_path_buf(),
            base: Path::new(".").to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(unix)]
fn copy_symlink(from: &Path, to: &Path) -> Result<()> {
    use std::os::unix::fs::symlink;

    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).with_path("creating", parent)?;
    }
    let target = fs::read_link(from).with_path("reading symlink", from)?;
    symlink(&target, to).with_path("creating symlink", to)
}

#[cfg(windows)]
fn copy_symlink(from: &Path, to: &Path) -> Result<()> {
    use std::os::windows::fs::{symlink_dir, symlink_file};

    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).with_path("creating", parent)?;
    }
    let target = fs::read_link(from).with_path("reading symlink", from)?;
    let target_metadata = fs::metadata(from);
    if target_metadata.map(|meta| meta.is_dir()).unwrap_or(false) {
        symlink_dir(&target, to).with_path("creating symlink", to)
    } else {
        symlink_file(&target, to).with_path("creating symlink", to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_path_guard_rejects_parent_components() {
        assert!(ensure_relative_path(Path::new("../escape")).is_err());
        assert!(ensure_relative_path(Path::new("ok/path")).is_ok());
    }
}
