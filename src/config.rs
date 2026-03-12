use std::{env, path::PathBuf};

use anyhow::{Context, Result};

use crate::{
    llm::{AnthropicConfig, GoogleConfig, LlmConfig, OllamaConfig, OpenAiConfig, ProviderKind},
    model::ThemeSkin,
};

#[derive(Debug, Clone)]
pub struct PaymentRuntimeConfig {
    pub hive_api_url: String,
    pub hive_engine_api_url: String,
    pub history_batch_size: u32,
    pub timeout_secs: u64,
    pub poll_interval_secs: u64,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub discord_token: String,
    pub storage_path: PathBuf,
    pub default_bot_name: String,
    pub default_theme: ThemeSkin,
    pub llm: LlmConfig,
    pub payments: PaymentRuntimeConfig,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let discord_token =
            env::var("DISCORD_TOKEN").context("DISCORD_TOKEN is required for the bot")?;
        let storage_path = PathBuf::from(
            env::var("STORAGE_PATH").unwrap_or_else(|_| "data/tavern-state.json".to_string()),
        );
        let default_bot_name =
            env::var("DEFAULT_BOT_NAME").unwrap_or_else(|_| "Baron Botley".to_string());
        let default_theme = env::var("DEFAULT_THEME")
            .ok()
            .and_then(|value| ThemeSkin::parse(&value))
            .unwrap_or(ThemeSkin::Tavern);

        Ok(Self {
            discord_token,
            storage_path,
            default_bot_name,
            default_theme,
            llm: build_llm_config(),
            payments: PaymentRuntimeConfig {
                hive_api_url: env::var("HIVE_API_URL")
                    .unwrap_or_else(|_| "https://api.hive.blog".to_string()),
                hive_engine_api_url: env::var("HIVE_ENGINE_API_URL")
                    .unwrap_or_else(|_| "https://api.hive-engine.com/rpc/blockchain".to_string()),
                history_batch_size: env::var("HIVE_HISTORY_BATCH_SIZE")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(100),
                timeout_secs: env::var("PAYMENT_TIMEOUT_SECS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(20),
                poll_interval_secs: env::var("PAYMENT_POLL_INTERVAL_SECS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(15),
            },
        })
    }
}

fn build_llm_config() -> LlmConfig {
    let openai_model = env::var("OPENAI_MODEL").ok();
    let anthropic_model = env::var("ANTHROPIC_MODEL").ok();
    let google_model = env::var("GOOGLE_MODEL").ok();
    let ollama_model = env::var("OLLAMA_MODEL").ok();

    let provider = env::var("DEFAULT_LLM_PROVIDER")
        .ok()
        .and_then(|value| parse_provider(&value))
        .or_else(|| {
            if env::var("OPENAI_API_KEY").is_ok() && openai_model.is_some() {
                Some(ProviderKind::OpenAi)
            } else if env::var("ANTHROPIC_API_KEY").is_ok() && anthropic_model.is_some() {
                Some(ProviderKind::Anthropic)
            } else if env::var("GOOGLE_API_KEY").is_ok() && google_model.is_some() {
                Some(ProviderKind::Google)
            } else if ollama_model.is_some() {
                Some(ProviderKind::Ollama)
            } else {
                Some(ProviderKind::Ollama)
            }
        })
        .unwrap_or(ProviderKind::Ollama);

    let default_model = env::var("DEFAULT_LLM_MODEL")
        .ok()
        .or_else(|| openai_model.clone())
        .or_else(|| anthropic_model.clone())
        .or_else(|| google_model.clone())
        .or_else(|| ollama_model.clone())
        .unwrap_or_else(|| "qwen3:4b".to_string());

    LlmConfig {
        provider,
        default_model,
        temperature: env::var("LLM_TEMPERATURE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.9),
        max_tokens: env::var("LLM_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(240),
        timeout_secs: env::var("LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(30),
        openai: Some(OpenAiConfig {
            api_key: env::var("OPENAI_API_KEY").ok(),
            base_url: env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            organization: env::var("OPENAI_ORGANIZATION").ok(),
            project: env::var("OPENAI_PROJECT").ok(),
        }),
        anthropic: Some(AnthropicConfig {
            api_key: env::var("ANTHROPIC_API_KEY").ok(),
            base_url: env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1".to_string()),
            version: env::var("ANTHROPIC_VERSION").unwrap_or_else(|_| "2023-06-01".to_string()),
        }),
        google: Some(GoogleConfig {
            api_key: env::var("GOOGLE_API_KEY").ok(),
            base_url: env::var("GOOGLE_BASE_URL")
                .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta".to_string()),
        }),
        ollama: Some(OllamaConfig {
            base_url: env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            keep_alive: env::var("OLLAMA_KEEP_ALIVE").ok(),
        }),
    }
}

pub fn parse_provider(input: &str) -> Option<ProviderKind> {
    match input.trim().to_ascii_lowercase().as_str() {
        "offline" => Some(ProviderKind::Offline),
        "openai" | "open_ai" => Some(ProviderKind::OpenAi),
        "anthropic" => Some(ProviderKind::Anthropic),
        "google" | "gemini" => Some(ProviderKind::Google),
        "ollama" => Some(ProviderKind::Ollama),
        _ => None,
    }
}
