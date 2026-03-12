use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::{fs, sync::Mutex};

use crate::model::GuildState;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedState {
    pub guilds: BTreeMap<String, GuildState>,
}

#[derive(Clone)]
pub struct JsonStore {
    path: PathBuf,
    state: Arc<Mutex<PersistedState>>,
}

impl JsonStore {
    pub async fn load(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create data dir {}", parent.display()))?;
        }

        let initial = match fs::read_to_string(&path).await {
            Ok(contents) => serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse {}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => PersistedState::default(),
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        };

        Ok(Self {
            path,
            state: Arc::new(Mutex::new(initial)),
        })
    }

    pub async fn snapshot(&self) -> PersistedState {
        self.state.lock().await.clone()
    }

    pub async fn mutate_state<F, R>(&self, mutate: F) -> Result<R>
    where
        F: FnOnce(&mut PersistedState) -> Result<R>,
    {
        let mut guard = self.state.lock().await;
        let result = mutate(&mut guard)?;
        self.persist_locked(&guard).await?;
        Ok(result)
    }

    pub async fn upsert_guild<F, R>(
        &self,
        guild_id: u64,
        init: impl FnOnce() -> GuildState,
        mutate: F,
    ) -> Result<R>
    where
        F: FnOnce(&mut GuildState) -> Result<R>,
    {
        self.mutate_state(|state| {
            let key = guild_id.to_string();
            let guild = state.guilds.entry(key).or_insert_with(init);
            mutate(guild)
        })
        .await
    }

    async fn persist_locked(&self, state: &PersistedState) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(state).context("failed to serialize store")?;
        let temp_path = self.path.with_extension("json.tmp");
        fs::write(&temp_path, bytes)
            .await
            .with_context(|| format!("failed to write {}", temp_path.display()))?;
        fs::rename(&temp_path, &self.path)
            .await
            .with_context(|| format!("failed to persist {}", self.path.display()))?;
        Ok(())
    }
}
