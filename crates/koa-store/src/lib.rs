use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, StoreError>;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    title TEXT,
    body TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    modified_ms INTEGER NOT NULL,
    ingested_ms INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(vault_path, relative_path)
);

CREATE INDEX IF NOT EXISTS idx_documents_vault_path
    ON documents(vault_path);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
USING fts5(
    title,
    body,
    content='documents',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS documents_ai
AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, body)
    VALUES (new.id, new.title, new.body);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad
AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
END;

CREATE TRIGGER IF NOT EXISTS documents_au
AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
    INSERT INTO documents_fts(rowid, title, body)
    VALUES (new.id, new.title, new.body);
END;
"#;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("store connection lock poisoned")]
    LockPoisoned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewDocument {
    pub vault_path: String,
    pub relative_path: String,
    pub title: Option<String>,
    pub body: String,
    pub content_hash: String,
    pub modified_ms: i64,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub vault_path: String,
    pub relative_path: String,
    pub title: Option<String>,
    pub body: String,
    pub content_hash: String,
    pub modified_ms: i64,
    pub ingested_ms: i64,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: i64,
    pub vault_path: String,
    pub relative_path: String,
    pub title: Option<String>,
    pub snippet: String,
    pub score: f64,
    pub content_hash: String,
    pub modified_ms: i64,
    pub ingested_ms: i64,
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreStats {
    pub document_count: usize,
    pub vault_count: usize,
}

#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        Self::from_connection(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn upsert_document(&self, document: NewDocument) -> Result<Document> {
        validate_document(&document)?;

        let metadata_json = serde_json::to_string(&document.metadata)?;
        let ingested_ms = now_ms()?;
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;

        tx.execute(
            r#"
            INSERT INTO documents (
                vault_path,
                relative_path,
                title,
                body,
                content_hash,
                modified_ms,
                ingested_ms,
                metadata_json
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(vault_path, relative_path) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                content_hash = excluded.content_hash,
                modified_ms = excluded.modified_ms,
                ingested_ms = excluded.ingested_ms,
                metadata_json = excluded.metadata_json
            "#,
            params![
                &document.vault_path,
                &document.relative_path,
                document.title.as_deref(),
                &document.body,
                &document.content_hash,
                document.modified_ms,
                ingested_ms,
                &metadata_json,
            ],
        )?;

        let stored = tx.query_row(
            r#"
            SELECT id, vault_path, relative_path, title, body, content_hash,
                   modified_ms, ingested_ms, metadata_json
            FROM documents
            WHERE vault_path = ?1 AND relative_path = ?2
            "#,
            params![&document.vault_path, &document.relative_path],
            document_from_row,
        )?;

        tx.commit()?;
        Ok(stored)
    }

    pub fn get_document(
        &self,
        vault_path: impl AsRef<str>,
        relative_path: impl AsRef<str>,
    ) -> Result<Option<Document>> {
        let conn = self.conn()?;
        let document = conn
            .query_row(
                r#"
                SELECT id, vault_path, relative_path, title, body, content_hash,
                       modified_ms, ingested_ms, metadata_json
                FROM documents
                WHERE vault_path = ?1 AND relative_path = ?2
                "#,
                params![vault_path.as_ref(), relative_path.as_ref()],
                document_from_row,
            )
            .optional()?;

        Ok(document)
    }

    pub fn list_documents(&self) -> Result<Vec<Document>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, vault_path, relative_path, title, body, content_hash,
                   modified_ms, ingested_ms, metadata_json
            FROM documents
            ORDER BY vault_path, relative_path
            "#,
        )?;

        let rows = stmt.query_map([], document_from_row)?;
        collect_rows(rows)
    }

    pub fn list_documents_for_vault(&self, vault_path: impl AsRef<str>) -> Result<Vec<Document>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT id, vault_path, relative_path, title, body, content_hash,
                   modified_ms, ingested_ms, metadata_json
            FROM documents
            WHERE vault_path = ?1
            ORDER BY relative_path
            "#,
        )?;

        let rows = stmt.query_map(params![vault_path.as_ref()], document_from_row)?;
        collect_rows(rows)
    }

    pub fn remove_document(
        &self,
        vault_path: impl AsRef<str>,
        relative_path: impl AsRef<str>,
    ) -> Result<bool> {
        let conn = self.conn()?;
        let changed = conn.execute(
            "DELETE FROM documents WHERE vault_path = ?1 AND relative_path = ?2",
            params![vault_path.as_ref(), relative_path.as_ref()],
        )?;

        Ok(changed > 0)
    }

    pub fn delete_missing(
        &self,
        vault_path: impl AsRef<str>,
        known_relative_paths: &[String],
    ) -> Result<usize> {
        let vault_path = vault_path.as_ref();
        let known: HashSet<&str> = known_relative_paths.iter().map(String::as_str).collect();
        let mut deleted = 0;
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;

        if known.is_empty() {
            deleted = tx.execute(
                "DELETE FROM documents WHERE vault_path = ?1",
                params![vault_path],
            )?;
            tx.commit()?;
            return Ok(deleted);
        }

        let existing = {
            let mut stmt = tx.prepare(
                "SELECT relative_path FROM documents WHERE vault_path = ?1 ORDER BY relative_path",
            )?;
            let rows = stmt.query_map(params![vault_path], |row| row.get::<_, String>(0))?;
            collect_rows(rows)?
        };

        for relative_path in existing {
            if !known.contains(relative_path.as_str()) {
                deleted += tx.execute(
                    "DELETE FROM documents WHERE vault_path = ?1 AND relative_path = ?2",
                    params![vault_path, relative_path],
                )?;
            }
        }

        tx.commit()?;
        Ok(deleted)
    }

    pub fn search(&self, query: impl AsRef<str>, limit: usize) -> Result<Vec<SearchHit>> {
        self.search_scoped(None, query.as_ref(), limit)
    }

    pub fn search_in_vault(
        &self,
        vault_path: impl AsRef<str>,
        query: impl AsRef<str>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        self.search_scoped(Some(vault_path.as_ref()), query.as_ref(), limit)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let conn = self.conn()?;
        let document_count = conn.query_row("SELECT COUNT(*) FROM documents", [], |row| {
            row.get::<_, i64>(0)
        })?;
        let vault_count = conn.query_row(
            "SELECT COUNT(DISTINCT vault_path) FROM documents",
            [],
            |row| row.get::<_, i64>(0),
        )?;

        Ok(StoreStats {
            document_count: usize::try_from(document_count).unwrap_or(usize::MAX),
            vault_count: usize::try_from(vault_count).unwrap_or(usize::MAX),
        })
    }

    fn search_scoped(
        &self,
        vault_path: Option<&str>,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let Some(match_query) = fts_query(query) else {
            return Ok(Vec::new());
        };

        let conn = self.conn()?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);

        match vault_path {
            Some(vault_path) => {
                let mut stmt = conn.prepare(
                    r#"
                    SELECT d.id,
                           d.vault_path,
                           d.relative_path,
                           d.title,
                           snippet(documents_fts, 1, '[', ']', '...', 18) AS snippet,
                           bm25(documents_fts) AS score,
                           d.content_hash,
                           d.modified_ms,
                           d.ingested_ms,
                           d.metadata_json
                    FROM documents_fts
                    JOIN documents d ON d.id = documents_fts.rowid
                    WHERE documents_fts MATCH ?1 AND d.vault_path = ?2
                    ORDER BY score, d.relative_path
                    LIMIT ?3
                    "#,
                )?;
                let rows = stmt.query_map(params![match_query, vault_path, limit], hit_from_row)?;
                collect_rows(rows)
            }
            None => {
                let mut stmt = conn.prepare(
                    r#"
                    SELECT d.id,
                           d.vault_path,
                           d.relative_path,
                           d.title,
                           snippet(documents_fts, 1, '[', ']', '...', 18) AS snippet,
                           bm25(documents_fts) AS score,
                           d.content_hash,
                           d.modified_ms,
                           d.ingested_ms,
                           d.metadata_json
                    FROM documents_fts
                    JOIN documents d ON d.id = documents_fts.rowid
                    WHERE documents_fts MATCH ?1
                    ORDER BY score, d.vault_path, d.relative_path
                    LIMIT ?2
                    "#,
                )?;
                let rows = stmt.query_map(params![match_query, limit], hit_from_row)?;
                collect_rows(rows)
            }
        }
    }

    fn conn(&self) -> Result<MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|_| StoreError::LockPoisoned)
    }
}

fn validate_document(document: &NewDocument) -> Result<()> {
    if document.vault_path.trim().is_empty() {
        return Err(StoreError::InvalidInput(
            "document vault_path cannot be empty".to_string(),
        ));
    }

    if document.relative_path.trim().is_empty() {
        return Err(StoreError::InvalidInput(
            "document relative_path cannot be empty".to_string(),
        ));
    }

    if document.content_hash.trim().is_empty() {
        return Err(StoreError::InvalidInput(
            "document content_hash cannot be empty".to_string(),
        ));
    }

    Ok(())
}

fn now_ms() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| {
            StoreError::InvalidInput(format!("system clock before unix epoch: {err}"))
        })?;

    i64::try_from(duration.as_millis())
        .map_err(|_| StoreError::InvalidInput("current time does not fit in i64".to_string()))
}

fn fts_query(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '-'))
        .filter_map(|raw| {
            let term = raw.trim();
            (!term.is_empty()).then(|| format!("\"{}\"", term.replace('"', "\"\"")))
        })
        .collect();

    (!terms.is_empty()).then(|| terms.join(" AND "))
}

fn document_from_row(row: &Row<'_>) -> rusqlite::Result<Document> {
    let metadata_json: String = row.get(8)?;

    Ok(Document {
        id: row.get(0)?,
        vault_path: row.get(1)?,
        relative_path: row.get(2)?,
        title: row.get(3)?,
        body: row.get(4)?,
        content_hash: row.get(5)?,
        modified_ms: row.get(6)?,
        ingested_ms: row.get(7)?,
        metadata: parse_metadata(metadata_json, 8)?,
    })
}

fn hit_from_row(row: &Row<'_>) -> rusqlite::Result<SearchHit> {
    let metadata_json: String = row.get(9)?;

    Ok(SearchHit {
        id: row.get(0)?,
        vault_path: row.get(1)?,
        relative_path: row.get(2)?,
        title: row.get(3)?,
        snippet: row.get(4)?,
        score: row.get(5)?,
        content_hash: row.get(6)?,
        modified_ms: row.get(7)?,
        ingested_ms: row.get(8)?,
        metadata: parse_metadata(metadata_json, 9)?,
    })
}

fn parse_metadata(raw: String, column: usize) -> rusqlite::Result<Value> {
    serde_json::from_str(&raw)
        .map_err(|err| rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(err)))
}

fn collect_rows<T>(
    rows: impl Iterator<Item = rusqlite::Result<T>>,
) -> std::result::Result<Vec<T>, StoreError> {
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(StoreError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn note(relative_path: &str, body: &str, hash: &str) -> NewDocument {
        NewDocument {
            vault_path: "vault-a".to_string(),
            relative_path: relative_path.to_string(),
            title: Some(relative_path.to_string()),
            body: body.to_string(),
            content_hash: hash.to_string(),
            modified_ms: 1,
            metadata: json!({ "kind": "test-note" }),
        }
    }

    #[test]
    fn upserts_searches_and_updates_fts_index() {
        let store = Store::open_in_memory().unwrap();

        store
            .upsert_document(note("a.md", "Rust context storage", "hash-a"))
            .unwrap();
        store
            .upsert_document(note("b.md", "SQLite vault search", "hash-b"))
            .unwrap();

        let hits = store.search_in_vault("vault-a", "rust", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].relative_path, "a.md");

        store
            .upsert_document(note("a.md", "Context storage without that term", "hash-c"))
            .unwrap();

        let hits = store.search_in_vault("vault-a", "rust", 10).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn delete_missing_prunes_stale_documents() {
        let store = Store::open_in_memory().unwrap();

        store
            .upsert_document(note("a.md", "Alpha", "hash-a"))
            .unwrap();
        store
            .upsert_document(note("b.md", "Beta", "hash-b"))
            .unwrap();

        let deleted = store
            .delete_missing("vault-a", &["a.md".to_string()])
            .unwrap();
        assert_eq!(deleted, 1);

        let docs = store.list_documents_for_vault("vault-a").unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].relative_path, "a.md");
    }

    #[test]
    fn blank_searches_do_not_query_fts() {
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_document(note("a.md", "Alpha", "hash-a"))
            .unwrap();

        assert!(store.search("   ", 10).unwrap().is_empty());
        assert!(store.search("!!!", 10).unwrap().is_empty());
        assert!(store.search("alpha", 0).unwrap().is_empty());
    }
}
