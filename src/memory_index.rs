use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Once},
};

use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, ffi::sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use tokio::sync::Mutex;

static SQLITE_VEC_INIT: Once = Once::new();

#[derive(Debug, Clone)]
pub struct IndexedMemoryRow {
    pub memory_id: String,
    pub search_text: String,
    pub category: String,
    pub source: String,
    pub stage: Option<String>,
    pub importance: u8,
    pub content_hash: String,
}

#[derive(Debug, Clone, Default)]
pub struct MemorySearchSignals {
    pub vector_rank_score: f32,
    pub lexical_rank_score: f32,
}

#[derive(Debug, Clone)]
pub struct MemorySearchQuery {
    pub guild_id: u64,
    pub embedding: Vec<f32>,
    pub fts_query: Option<String>,
    pub vector_limit: usize,
    pub lexical_limit: usize,
}

#[derive(Clone)]
pub struct MemoryIndex {
    conn: Arc<Mutex<Connection>>,
    embedding_dimensions: usize,
}

impl MemoryIndex {
    pub fn open(path: PathBuf, embedding_dimensions: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create memory db dir {}", parent.display()))?;
        }

        register_sqlite_vec();
        let conn = Connection::open(path).context("failed to open memory sqlite database")?;
        conn.execute_batch(&format!(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                search_text TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                stage TEXT,
                importance INTEGER NOT NULL,
                content_hash TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_memories_guild_id
                ON memories(guild_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                guild_id UNINDEXED,
                content
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                memory_id TEXT PRIMARY KEY,
                guild_id INTEGER PARTITION KEY,
                embedding FLOAT[{embedding_dimensions}] DISTANCE_METRIC=COSINE,
                category TEXT,
                source TEXT,
                stage TEXT,
                importance INTEGER
            );
            "#,
        ))
        .context("failed to initialize memory sqlite schema")?;
        ensure_content_hash_column(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedding_dimensions: embedding_dimensions.max(1),
        })
    }

    pub async fn stale_memory_ids(
        &self,
        guild_id: u64,
        rows: &[IndexedMemoryRow],
    ) -> Result<Vec<String>> {
        let guild_id_i64 = i64::try_from(guild_id).context("guild id overflow for sqlite")?;
        let conn = self.conn.lock().await;
        let existing = load_guild_hashes(&conn, guild_id_i64)?;

        Ok(rows
            .iter()
            .filter(|row| existing.get(&row.memory_id) != Some(&row.content_hash))
            .map(|row| row.memory_id.clone())
            .collect())
    }

    pub async fn sync_guild(
        &self,
        guild_id: u64,
        rows: &[IndexedMemoryRow],
        embeddings: &HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        let guild_id_i64 = i64::try_from(guild_id).context("guild id overflow for sqlite")?;
        let mut conn = self.conn.lock().await;
        let existing = load_guild_hashes(&conn, guild_id_i64)?;
        let desired_ids = rows
            .iter()
            .map(|row| row.memory_id.clone())
            .collect::<HashSet<_>>();
        let delete_ids = existing
            .keys()
            .filter(|memory_id| !desired_ids.contains(*memory_id))
            .cloned()
            .collect::<Vec<_>>();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("failed to start memory sync transaction")?;

        for memory_id in delete_ids {
            tx.execute(
                "DELETE FROM memories WHERE memory_id = ?1",
                [memory_id.as_str()],
            )
            .context("failed to delete stale memory row")?;
            tx.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?1",
                [memory_id.as_str()],
            )
            .context("failed to delete stale memory fts row")?;
            tx.execute(
                "DELETE FROM memories_vec WHERE memory_id = ?1",
                [memory_id.as_str()],
            )
            .context("failed to delete stale memory vector row")?;
        }

        for row in rows {
            if existing.get(&row.memory_id) == Some(&row.content_hash) {
                continue;
            }

            let embedding = embeddings
                .get(&row.memory_id)
                .ok_or_else(|| anyhow!("missing embedding for changed memory {}", row.memory_id))?;
            if embedding.len() != self.embedding_dimensions {
                return Err(anyhow!(
                    "embedding dimension mismatch for memory {}: expected {}, got {}",
                    row.memory_id,
                    self.embedding_dimensions,
                    embedding.len()
                ));
            }

            tx.execute(
                "INSERT INTO memories (memory_id, guild_id, search_text, category, source, stage, importance, content_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(memory_id) DO UPDATE SET
                    guild_id = excluded.guild_id,
                    search_text = excluded.search_text,
                    category = excluded.category,
                    source = excluded.source,
                    stage = excluded.stage,
                    importance = excluded.importance,
                    content_hash = excluded.content_hash",
                (
                    row.memory_id.as_str(),
                    guild_id_i64,
                    row.search_text.as_str(),
                    row.category.as_str(),
                    row.source.as_str(),
                    row.stage.as_deref(),
                    i64::from(row.importance),
                    row.content_hash.as_str(),
                ),
            )
            .context("failed to upsert memory row")?;
            tx.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?1",
                [row.memory_id.as_str()],
            )
            .context("failed to replace memory fts row")?;
            tx.execute(
                "INSERT INTO memories_fts (memory_id, guild_id, content) VALUES (?1, ?2, ?3)",
                (
                    row.memory_id.as_str(),
                    guild_id_i64,
                    row.search_text.as_str(),
                ),
            )
            .context("failed to insert memory fts row")?;
            tx.execute(
                "DELETE FROM memories_vec WHERE memory_id = ?1",
                [row.memory_id.as_str()],
            )
            .context("failed to replace memory vector row")?;
            tx.execute(
                "INSERT INTO memories_vec (memory_id, guild_id, embedding, category, source, stage, importance)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                (
                    row.memory_id.as_str(),
                    guild_id_i64,
                    embedding_json(embedding).as_str(),
                    row.category.as_str(),
                    row.source.as_str(),
                    row.stage.as_deref().unwrap_or(""),
                    i64::from(row.importance),
                ),
            )
            .context("failed to insert memory vector row")?;
        }

        tx.commit()
            .context("failed to commit memory sync transaction")?;
        Ok(())
    }

    pub async fn search(
        &self,
        query: &MemorySearchQuery,
    ) -> Result<HashMap<String, MemorySearchSignals>> {
        let guild_id_i64 = i64::try_from(query.guild_id).context("guild id overflow for sqlite")?;
        let mut signals = HashMap::<String, MemorySearchSignals>::new();
        let conn = self.conn.lock().await;

        let memory_count: Option<i64> = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE guild_id = ?1",
                [guild_id_i64],
                |row| row.get(0),
            )
            .optional()
            .context("failed counting guild memories")?;
        if memory_count.unwrap_or(0) == 0 {
            return Ok(signals);
        }

        if query.embedding.len() == self.embedding_dimensions {
            let embedding = embedding_json(&query.embedding);
            let mut stmt = conn
                .prepare(
                    "SELECT memory_id, distance
                     FROM memories_vec
                     WHERE guild_id = ?1
                       AND embedding MATCH ?2
                       AND k = ?3
                     ORDER BY distance",
                )
                .context("failed preparing vector search")?;
            let rows = stmt
                .query_map(
                    (guild_id_i64, embedding.as_str(), query.vector_limit as i64),
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?)),
                )
                .context("failed running vector search")?;
            for (index, row) in rows.enumerate() {
                let (memory_id, distance) = row.context("invalid vector search row")?;
                let entry = signals.entry(memory_id).or_default();
                entry.vector_rank_score =
                    (1.0 / ((index + 1) as f32)) + (1.0 / (1.0 + distance.max(0.0)));
            }
        }

        if let Some(fts_query) = query
            .fts_query
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            let mut stmt = conn
                .prepare(
                    "SELECT memory_id
                     FROM memories_fts
                     WHERE memories_fts MATCH ?1
                       AND guild_id = ?2
                     LIMIT ?3",
                )
                .context("failed preparing fts search")?;
            let rows = stmt
                .query_map(
                    (fts_query, guild_id_i64, query.lexical_limit as i64),
                    |row| row.get::<_, String>(0),
                )
                .context("failed running fts search")?;
            for (index, row) in rows.enumerate() {
                let memory_id = row.context("invalid fts search row")?;
                let entry = signals.entry(memory_id).or_default();
                entry.lexical_rank_score = 1.5 / ((index + 1) as f32);
            }
        }

        Ok(signals)
    }
}

fn register_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    });
}

fn ensure_content_hash_column(conn: &Connection) -> Result<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(memories)")
        .context("failed to inspect memory schema")?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .context("failed to inspect memory columns")?;
    let mut has_content_hash = false;
    for row in rows {
        if row.context("invalid memory schema row")? == "content_hash" {
            has_content_hash = true;
            break;
        }
    }

    if !has_content_hash {
        conn.execute(
            "ALTER TABLE memories ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
            [],
        )
        .context("failed to add content_hash column to memories")?;
    }

    Ok(())
}

fn load_guild_hashes(conn: &Connection, guild_id_i64: i64) -> Result<HashMap<String, String>> {
    let mut stmt = conn
        .prepare("SELECT memory_id, content_hash FROM memories WHERE guild_id = ?1")
        .context("failed preparing guild memory hash query")?;
    let rows = stmt
        .query_map([guild_id_i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .context("failed querying guild memory hashes")?;

    let mut hashes = HashMap::new();
    for row in rows {
        let (memory_id, content_hash) = row.context("invalid guild memory hash row")?;
        hashes.insert(memory_id, content_hash);
    }
    Ok(hashes)
}

pub fn hashed_embedding(input: &str, dimensions: usize) -> Vec<f32> {
    let target_dimensions = dimensions.max(1);
    let mut values = vec![0.0f32; target_dimensions];
    let tokens = input
        .split_whitespace()
        .map(|token| token.trim())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return values;
    }

    for token in tokens {
        let hash = fxhash(token.as_bytes());
        let index = (hash as usize) % target_dimensions;
        let sign = if ((hash >> 8) & 1) == 0 { 1.0 } else { -1.0 };
        let magnitude = 1.0 + ((token.len().min(12) as f32) / 12.0);
        values[index] += sign * magnitude;
    }

    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut values {
            *value /= norm;
        }
    }
    values
}

fn embedding_json(values: &[f32]) -> String {
    let rendered = values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rendered}]")
}

fn fxhash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use super::{IndexedMemoryRow, MemoryIndex, MemorySearchQuery, hashed_embedding};

    #[tokio::test]
    async fn sqlite_memory_index_searches_fts_and_vectors() {
        let path = PathBuf::from(format!(
            "{}/memory-index-test-{}.sqlite3",
            std::env::temp_dir().display(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let index = MemoryIndex::open(path, 128).unwrap();
        let rows = vec![
            IndexedMemoryRow {
                memory_id: "a".to_string(),
                search_text: "dwayne prefers hbd payouts".to_string(),
                category: "preference".to_string(),
                source: "user_note".to_string(),
                stage: None,
                importance: 4,
                content_hash: "hash-a".to_string(),
            },
            IndexedMemoryRow {
                memory_id: "b".to_string(),
                search_text: "karaoke incident while cooked".to_string(),
                category: "incident".to_string(),
                source: "system_event".to_string(),
                stage: Some("cooked".to_string()),
                importance: 5,
                content_hash: "hash-b".to_string(),
            },
        ];
        let stale = index.stale_memory_ids(1, &rows).await.unwrap();
        let embeddings = rows
            .iter()
            .filter(|row| stale.contains(&row.memory_id))
            .map(|row| {
                (
                    row.memory_id.clone(),
                    hashed_embedding(&row.search_text, 128),
                )
            })
            .collect::<HashMap<_, _>>();
        index.sync_guild(1, &rows, &embeddings).await.unwrap();

        let signals = index
            .search(&MemorySearchQuery {
                guild_id: 1,
                embedding: hashed_embedding("who prefers hbd", 128),
                fts_query: Some("prefers OR hbd".to_string()),
                vector_limit: 8,
                lexical_limit: 8,
            })
            .await
            .unwrap();

        assert!(signals.contains_key("a"));
        assert!(signals["a"].lexical_rank_score > 0.0 || signals["a"].vector_rank_score > 0.0);
    }

    #[tokio::test]
    async fn sqlite_memory_index_can_resync_unchanged_rows_without_new_embeddings() {
        let path = PathBuf::from(format!(
            "{}/memory-index-test-{}.sqlite3",
            std::env::temp_dir().display(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let index = MemoryIndex::open(path, 128).unwrap();
        let rows = vec![IndexedMemoryRow {
            memory_id: "a".to_string(),
            search_text: "dwayne prefers hbd payouts".to_string(),
            category: "preference".to_string(),
            source: "user_note".to_string(),
            stage: None,
            importance: 4,
            content_hash: "hash-a".to_string(),
        }];
        let embeddings = rows
            .iter()
            .map(|row| {
                (
                    row.memory_id.clone(),
                    hashed_embedding(&row.search_text, 128),
                )
            })
            .collect::<HashMap<_, _>>();

        index.sync_guild(1, &rows, &embeddings).await.unwrap();
        index.sync_guild(1, &rows, &HashMap::new()).await.unwrap();

        let signals = index
            .search(&MemorySearchQuery {
                guild_id: 1,
                embedding: hashed_embedding("who prefers hbd", 128),
                fts_query: Some("prefers OR hbd".to_string()),
                vector_limit: 8,
                lexical_limit: 8,
            })
            .await
            .unwrap();

        assert!(signals.contains_key("a"));
    }
}
