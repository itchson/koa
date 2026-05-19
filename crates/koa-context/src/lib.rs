use koa_store::{NewDocument, SearchHit, Store};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ContextError>;

pub const KOA_DIR: &str = ".koa";
pub const STORE_FILE: &str = "koa.sqlite3";

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("store error: {0}")]
    Store(#[from] koa_store::StoreError),

    #[error("path {path} is outside vault root {root}")]
    PathOutsideVault { path: PathBuf, root: PathBuf },

    #[error("unsupported path inside vault: {0}")]
    UnsupportedPath(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultLayout {
    root: PathBuf,
    koa_dir: PathBuf,
    store_path: PathBuf,
}

impl VaultLayout {
    pub fn for_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let koa_dir = root.join(KOA_DIR);
        let store_path = koa_dir.join(STORE_FILE);

        Self {
            root,
            koa_dir,
            store_path,
        }
    }

    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(&self.koa_dir)?;
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn koa_dir(&self) -> &Path {
        &self.koa_dir
    }

    pub fn store_path(&self) -> &Path {
        &self.store_path
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestOptions {
    extensions: BTreeSet<String>,
}

impl IngestOptions {
    pub fn new(extensions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let extensions = extensions
            .into_iter()
            .map(|extension| normalize_extension(&extension.into()))
            .filter(|extension| !extension.is_empty())
            .collect();

        Self { extensions }
    }

    pub fn extensions(&self) -> &BTreeSet<String> {
        &self.extensions
    }

    pub fn accepts(&self, path: &Path) -> bool {
        path.extension()
            .and_then(OsStr::to_str)
            .map(normalize_extension)
            .is_some_and(|extension| self.extensions.contains(&extension))
    }
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self::new(["md", "markdown", "txt"])
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReport {
    pub scanned: usize,
    pub ingested: usize,
    pub skipped: usize,
    pub deleted: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LintSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintIssue {
    pub severity: LintSeverity,
    pub code: String,
    pub path: Option<PathBuf>,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintReport {
    pub issues: Vec<LintIssue>,
}

impl LintReport {
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    pub fn errors(&self) -> impl Iterator<Item = &LintIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == LintSeverity::Error)
    }

    pub fn warnings(&self) -> impl Iterator<Item = &LintIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == LintSeverity::Warning)
    }

    fn push(
        &mut self,
        severity: LintSeverity,
        code: impl Into<String>,
        path: Option<PathBuf>,
        message: impl Into<String>,
    ) {
        self.issues.push(LintIssue {
            severity,
            code: code.into(),
            path,
            message: message.into(),
        });
    }
}

#[derive(Clone)]
pub struct Vault {
    layout: VaultLayout,
    store: Store,
    vault_id: String,
}

impl Vault {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(root.as_ref())?;
        let root = root.as_ref().canonicalize()?;
        let layout = VaultLayout::for_root(root);
        layout.ensure()?;
        let store = Store::open(layout.store_path())?;
        let vault_id = path_id(layout.root())?;

        Ok(Self {
            layout,
            store,
            vault_id,
        })
    }

    pub fn layout(&self) -> &VaultLayout {
        &self.layout
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn vault_id(&self) -> &str {
        &self.vault_id
    }

    pub fn ingest(&self) -> Result<IngestReport> {
        self.ingest_with_options(&IngestOptions::default())
    }

    pub fn ingest_with_options(&self, options: &IngestOptions) -> Result<IngestReport> {
        let mut report = IngestReport::default();
        let mut files = Vec::new();
        collect_source_files(&self.layout, options, &mut report, &mut files)?;

        let mut known_relative_paths = Vec::with_capacity(files.len());
        for path in files {
            let source = self.read_source(&path)?;
            known_relative_paths.push(source.relative_path.clone());
            self.store
                .upsert_document(source.into_document(&self.vault_id))?;
            report.ingested += 1;
        }

        report.deleted = self
            .store
            .delete_missing(&self.vault_id, &known_relative_paths)?;

        Ok(report)
    }

    pub fn search(&self, query: impl AsRef<str>, limit: usize) -> Result<Vec<SearchHit>> {
        self.store
            .search_in_vault(&self.vault_id, query.as_ref(), limit)
            .map_err(ContextError::from)
    }

    pub fn lint(&self) -> Result<LintReport> {
        self.lint_with_options(&IngestOptions::default())
    }

    pub fn lint_with_options(&self, options: &IngestOptions) -> Result<LintReport> {
        let mut report = LintReport::default();

        if !self.layout.root().is_dir() {
            report.push(
                LintSeverity::Error,
                "vault.root_missing",
                Some(self.layout.root().to_path_buf()),
                "vault root is missing or is not a directory",
            );
        }

        if !self.layout.koa_dir().is_dir() {
            report.push(
                LintSeverity::Error,
                "vault.koa_dir_missing",
                Some(self.layout.koa_dir().to_path_buf()),
                "vault metadata directory is missing or is not a directory",
            );
        }

        if !self.layout.store_path().is_file() {
            report.push(
                LintSeverity::Error,
                "vault.store_missing",
                Some(self.layout.store_path().to_path_buf()),
                "vault sqlite store is missing",
            );
        }

        let stored_documents = self.store.list_documents_for_vault(&self.vault_id)?;
        let stored_paths: HashSet<&str> = stored_documents
            .iter()
            .map(|document| document.relative_path.as_str())
            .collect();

        let mut scan_report = IngestReport::default();
        let mut files = Vec::new();
        collect_source_files(&self.layout, options, &mut scan_report, &mut files)?;

        for path in files {
            let relative_path = relative_path(self.layout.root(), &path)?;
            if !stored_paths.contains(relative_path.as_str()) {
                report.push(
                    LintSeverity::Warning,
                    "source.not_ingested",
                    Some(path),
                    format!("source file {relative_path} has not been ingested"),
                );
            }
        }

        for document in stored_documents {
            let source_path = source_path(self.layout.root(), &document.relative_path)?;
            if !source_path.is_file() {
                report.push(
                    LintSeverity::Error,
                    "source.missing",
                    Some(source_path),
                    format!(
                        "stored document {} has no source file",
                        document.relative_path
                    ),
                );
                continue;
            }

            match self.read_source(&source_path) {
                Ok(source) => {
                    if source.content_hash != document.content_hash {
                        report.push(
                            LintSeverity::Warning,
                            "source.stale",
                            Some(source_path.clone()),
                            format!("stored document {} is stale", document.relative_path),
                        );
                    }

                    if source.body.trim().is_empty() {
                        report.push(
                            LintSeverity::Warning,
                            "source.empty",
                            Some(source_path),
                            format!("source file {} is empty", document.relative_path),
                        );
                    }
                }
                Err(err) => {
                    report.push(
                        LintSeverity::Error,
                        "source.unreadable",
                        Some(source_path),
                        format!(
                            "source file {} could not be read: {err}",
                            document.relative_path
                        ),
                    );
                }
            }
        }

        Ok(report)
    }

    fn read_source(&self, path: &Path) -> Result<SourceDocument> {
        let bytes = fs::read(path)?;
        let body = String::from_utf8(bytes.clone()).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("source file is not valid UTF-8: {err}"),
            )
        })?;
        let metadata = fs::metadata(path)?;
        let modified_ms = modified_ms(&metadata)?;
        let relative_path = relative_path(self.layout.root(), path)?;
        let title = markdown_title(&body);
        let extension = path
            .extension()
            .and_then(OsStr::to_str)
            .map(normalize_extension)
            .unwrap_or_default();
        let byte_len = bytes.len();
        let content_hash = sha256_hex(&bytes);

        Ok(SourceDocument {
            relative_path,
            title,
            body,
            content_hash,
            modified_ms,
            metadata: json!({
                "bytes": byte_len,
                "extension": extension,
            }),
        })
    }
}

#[derive(Debug)]
struct SourceDocument {
    relative_path: String,
    title: Option<String>,
    body: String,
    content_hash: String,
    modified_ms: i64,
    metadata: serde_json::Value,
}

impl SourceDocument {
    fn into_document(self, vault_id: &str) -> NewDocument {
        NewDocument {
            vault_path: vault_id.to_string(),
            relative_path: self.relative_path,
            title: self.title,
            body: self.body,
            content_hash: self.content_hash,
            modified_ms: self.modified_ms,
            metadata: self.metadata,
        }
    }
}

fn collect_source_files(
    layout: &VaultLayout,
    options: &IngestOptions,
    report: &mut IngestReport,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    collect_source_files_from(layout.root(), layout, options, report, files)?;
    files.sort();
    Ok(())
}

fn collect_source_files_from(
    dir: &Path,
    layout: &VaultLayout,
    options: &IngestOptions,
    report: &mut IngestReport,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    let mut entries = fs::read_dir(dir)?.collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            if path == layout.koa_dir() {
                continue;
            }

            collect_source_files_from(&path, layout, options, report, files)?;
            continue;
        }

        if !file_type.is_file() {
            continue;
        }

        report.scanned += 1;
        if options.accepts(&path) {
            files.push(path);
        } else {
            report.skipped += 1;
        }
    }

    Ok(())
}

fn relative_path(root: &Path, path: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| ContextError::PathOutsideVault {
            path: path.to_path_buf(),
            root: root.to_path_buf(),
        })?;

    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| ContextError::UnsupportedPath(path.to_path_buf()))?;
                parts.push(part.to_string());
            }
            _ => return Err(ContextError::UnsupportedPath(path.to_path_buf())),
        }
    }

    if parts.is_empty() {
        return Err(ContextError::UnsupportedPath(path.to_path_buf()));
    }

    Ok(parts.join("/"))
}

fn source_path(root: &Path, relative_path: &str) -> Result<PathBuf> {
    let mut path = root.to_path_buf();

    for part in relative_path.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return Err(ContextError::UnsupportedPath(PathBuf::from(relative_path)));
        }

        path.push(part);
    }

    Ok(path)
}

fn markdown_title(body: &str) -> Option<String> {
    body.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix("# ")
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .map(str::to_string)
    })
}

fn normalize_extension(extension: &str) -> String {
    extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase()
}

fn modified_ms(metadata: &fs::Metadata) -> Result<i64> {
    let duration = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("file modified time is before unix epoch: {err}"),
            )
        })?;

    i64::try_from(duration.as_millis()).map_err(|_| {
        ContextError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "file modified time does not fit in i64",
        ))
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();

    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }

    encoded
}

fn path_id(path: &Path) -> Result<String> {
    Ok(path
        .to_str()
        .ok_or_else(|| ContextError::UnsupportedPath(path.to_path_buf()))?
        .replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn opens_vault_ingests_sources_and_searches() {
        let temp = tempdir().unwrap();
        let note = temp.path().join("notes").join("alpha.md");
        fs::create_dir_all(note.parent().unwrap()).unwrap();
        fs::write(&note, "# Alpha\n\nRust context search lives here.").unwrap();
        fs::write(temp.path().join("ignored.json"), "{}").unwrap();

        let vault = Vault::open(temp.path()).unwrap();
        let report = vault.ingest().unwrap();

        assert_eq!(report.scanned, 2);
        assert_eq!(report.ingested, 1);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.deleted, 0);
        assert!(vault.layout().store_path().is_file());

        let lint = vault.lint().unwrap();
        assert!(lint.is_clean(), "{lint:?}");

        let hits = vault.search("rust context", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].relative_path, "notes/alpha.md");
        assert_eq!(hits[0].title.as_deref(), Some("Alpha"));
    }

    #[test]
    fn lint_reports_stale_sources_until_reingested() {
        let temp = tempdir().unwrap();
        let note = temp.path().join("alpha.md");
        fs::write(&note, "# Alpha\n\nOriginal text.").unwrap();

        let vault = Vault::open(temp.path()).unwrap();
        vault.ingest().unwrap();

        fs::write(&note, "# Alpha\n\nUpdated search token.").unwrap();

        let lint = vault.lint().unwrap();
        assert!(lint.warnings().any(|issue| issue.code == "source.stale"));

        vault.ingest().unwrap();
        let lint = vault.lint().unwrap();
        assert!(lint.is_clean(), "{lint:?}");

        let hits = vault.search("updated", 5).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn ingest_prunes_deleted_sources() {
        let temp = tempdir().unwrap();
        let note = temp.path().join("alpha.md");
        fs::write(&note, "# Alpha\n\nDelete me.").unwrap();

        let vault = Vault::open(temp.path()).unwrap();
        vault.ingest().unwrap();
        fs::remove_file(&note).unwrap();

        let lint = vault.lint().unwrap();
        assert!(lint.errors().any(|issue| issue.code == "source.missing"));

        let report = vault.ingest().unwrap();
        assert_eq!(report.deleted, 1);
        assert!(vault.search("delete", 5).unwrap().is_empty());
    }

    #[test]
    fn lint_reports_uningested_supported_sources() {
        let temp = tempdir().unwrap();
        fs::write(temp.path().join("alpha.md"), "# Alpha\n\nReady.").unwrap();

        let vault = Vault::open(temp.path()).unwrap();
        let lint = vault.lint().unwrap();

        assert!(
            lint.warnings()
                .any(|issue| issue.code == "source.not_ingested")
        );
    }
}
