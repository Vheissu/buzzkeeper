use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Offline,
    OpenAi,
    Anthropic,
    Google,
    Ollama,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Offline => "offline",
            ProviderKind::OpenAi => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Google => "google",
            ProviderKind::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: ProviderKind,
    pub default_model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_secs: u64,
    pub openai: Option<OpenAiConfig>,
    pub anthropic: Option<AnthropicConfig>,
    pub google: Option<GoogleConfig>,
    pub ollama: Option<OllamaConfig>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: ProviderKind::Offline,
            default_model: "tavern-gremlin".to_string(),
            temperature: 0.8,
            max_tokens: 480,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            openai: Some(OpenAiConfig::default()),
            anthropic: Some(AnthropicConfig::default()),
            google: Some(GoogleConfig::default()),
            ollama: Some(OllamaConfig::default()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub organization: Option<String>,
    pub project: Option<String>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_OPENAI_BASE_URL.to_string(),
            organization: None,
            project: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub version: String,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_ANTHROPIC_BASE_URL.to_string(),
            version: DEFAULT_ANTHROPIC_VERSION.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleConfig {
    pub api_key: Option<String>,
    pub base_url: String,
}

impl Default for GoogleConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_GOOGLE_BASE_URL.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub base_url: String,
    pub keep_alive: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_BASE_URL.to_string(),
            keep_alive: Some("10m".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl GenerateRequest {
    #[allow(dead_code)]
    pub fn new(system_prompt: impl Into<String>, user_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            user_prompt: user_prompt.into(),
            model: None,
            temperature: None,
            max_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub provider: ProviderKind,
    pub model: String,
    pub text: String,
}

#[allow(dead_code)]
#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn default_model(&self) -> &str;
    async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse>;
}

pub fn build_provider(config: &LlmConfig) -> Result<Box<dyn LlmProvider>> {
    match config.provider {
        ProviderKind::Offline => Ok(Box::new(OfflineProvider::new(config.clone()))),
        ProviderKind::OpenAi => Ok(Box::new(OpenAiProvider::new(config)?)),
        ProviderKind::Anthropic => Ok(Box::new(AnthropicProvider::new(config)?)),
        ProviderKind::Google => Ok(Box::new(GoogleProvider::new(config)?)),
        ProviderKind::Ollama => Ok(Box::new(OllamaProvider::new(config)?)),
    }
}

pub async fn embed_inputs(
    config: &LlmConfig,
    model: Option<&str>,
    dimensions: usize,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let target_dimensions = dimensions.max(1);
    let embeddings = match config.provider {
        ProviderKind::Offline => bail!("offline provider does not support embeddings"),
        ProviderKind::OpenAi => embed_with_openai(config, model, target_dimensions, inputs).await?,
        ProviderKind::Ollama => embed_with_ollama(config, model, inputs).await?,
        ProviderKind::Anthropic => bail!("anthropic provider does not expose embeddings here"),
        ProviderKind::Google => bail!("google embedding support is not wired in yet"),
    };

    Ok(embeddings
        .into_iter()
        .map(|embedding| fit_embedding(&embedding, target_dimensions))
        .collect())
}

#[derive(Debug, Clone)]
struct RuntimeOptions {
    model: String,
    temperature: f32,
    max_tokens: u32,
}

impl RuntimeOptions {
    fn from_config(config: &LlmConfig, request: &GenerateRequest) -> Self {
        Self {
            model: request
                .model
                .clone()
                .unwrap_or_else(|| config.default_model.clone()),
            temperature: request.temperature.unwrap_or(config.temperature),
            max_tokens: request.max_tokens.unwrap_or(config.max_tokens),
        }
    }
}

#[derive(Clone)]
pub struct OfflineProvider {
    config: LlmConfig,
}

impl OfflineProvider {
    pub fn new(config: LlmConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl LlmProvider for OfflineProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Offline
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let options = RuntimeOptions::from_config(&self.config, request);
        let prompt = request.user_prompt.trim();
        let summary = if prompt.is_empty() {
            "The tavern bot is awake, but nobody has said anything yet.".to_string()
        } else {
            let clipped = prompt.chars().take(180).collect::<String>();
            format!(
                "Offline mode only. The tavern keeps a note instead of calling a remote model: {clipped}"
            )
        };

        Ok(GenerateResponse {
            provider: ProviderKind::Offline,
            model: options.model,
            text: summary,
        })
    }
}

#[derive(Clone)]
pub struct OpenAiProvider {
    client: Client,
    config: LlmConfig,
    provider: OpenAiConfig,
}

impl OpenAiProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let provider = config.openai.clone().unwrap_or_default();
        if provider.api_key.as_deref().unwrap_or("").is_empty() {
            bail!("openai provider requires an api key");
        }

        Ok(Self {
            client: build_http_client(config.timeout_secs)?,
            config: config.clone(),
            provider,
        })
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let options = RuntimeOptions::from_config(&self.config, request);
        let url = join_url(&self.provider.base_url, "responses")?;
        let body = json!({
            "model": options.model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "temperature": options.temperature,
            "max_output_tokens": options.max_tokens,
        });

        let mut builder = self
            .client
            .post(url)
            .bearer_auth(
                self.provider
                    .api_key
                    .as_deref()
                    .ok_or_else(|| anyhow!("missing openai api key"))?,
            )
            .json(&body);

        if let Some(org) = &self.provider.organization {
            builder = builder.header("OpenAI-Organization", org);
        }

        if let Some(project) = &self.provider.project {
            builder = builder.header("OpenAI-Project", project);
        }

        let value = send_json(builder).await?;
        let text = extract_openai_text(&value)?;

        Ok(GenerateResponse {
            provider: ProviderKind::OpenAi,
            model: options.model,
            text,
        })
    }
}

#[derive(Clone)]
pub struct AnthropicProvider {
    client: Client,
    config: LlmConfig,
    provider: AnthropicConfig,
}

impl AnthropicProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let provider = config.anthropic.clone().unwrap_or_default();
        if provider.api_key.as_deref().unwrap_or("").is_empty() {
            bail!("anthropic provider requires an api key");
        }

        Ok(Self {
            client: build_http_client(config.timeout_secs)?,
            config: config.clone(),
            provider,
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let options = RuntimeOptions::from_config(&self.config, request);
        let url = join_url(&self.provider.base_url, "messages")?;
        let body = json!({
            "model": options.model,
            "system": request.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": request.user_prompt,
                }
            ],
            "temperature": options.temperature,
            "max_tokens": options.max_tokens,
        });

        let value = send_json(
            self.client
                .post(url)
                .header(
                    "x-api-key",
                    self.provider
                        .api_key
                        .as_deref()
                        .ok_or_else(|| anyhow!("missing anthropic api key"))?,
                )
                .header("anthropic-version", &self.provider.version)
                .json(&body),
        )
        .await?;

        let text = extract_anthropic_text(&value)?;

        Ok(GenerateResponse {
            provider: ProviderKind::Anthropic,
            model: options.model,
            text,
        })
    }
}

#[derive(Clone)]
pub struct GoogleProvider {
    client: Client,
    config: LlmConfig,
    provider: GoogleConfig,
}

impl GoogleProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        let provider = config.google.clone().unwrap_or_default();
        if provider.api_key.as_deref().unwrap_or("").is_empty() {
            bail!("google provider requires an api key");
        }

        Ok(Self {
            client: build_http_client(config.timeout_secs)?,
            config: config.clone(),
            provider,
        })
    }
}

#[async_trait]
impl LlmProvider for GoogleProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Google
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let options = RuntimeOptions::from_config(&self.config, request);
        let mut url = join_url(
            &self.provider.base_url,
            &format!("models/{}:generateContent", options.model),
        )?;
        url.query_pairs_mut().append_pair(
            "key",
            self.provider
                .api_key
                .as_deref()
                .ok_or_else(|| anyhow!("missing google api key"))?,
        );

        let body = json!({
            "system_instruction": {
                "parts": [
                    {
                        "text": request.system_prompt,
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": request.user_prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": options.temperature,
                "maxOutputTokens": options.max_tokens,
            }
        });

        let value = send_json(self.client.post(url).json(&body)).await?;
        let text = extract_google_text(&value)?;

        Ok(GenerateResponse {
            provider: ProviderKind::Google,
            model: options.model,
            text,
        })
    }
}

#[derive(Clone)]
pub struct OllamaProvider {
    client: Client,
    config: LlmConfig,
    provider: OllamaConfig,
}

impl OllamaProvider {
    pub fn new(config: &LlmConfig) -> Result<Self> {
        Ok(Self {
            client: build_http_client(config.timeout_secs)?,
            config: config.clone(),
            provider: config.ollama.clone().unwrap_or_default(),
        })
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let options = RuntimeOptions::from_config(&self.config, request);
        let url = join_url(&self.provider.base_url, "api/chat")?;
        let system_prompt = ollama_system_prompt(&options.model, &request.system_prompt);
        let body = json!({
            "model": options.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": request.user_prompt,
                }
            ],
            "stream": false,
            "think": false,
            "format": ollama_reply_schema(),
            "keep_alive": self.provider.keep_alive,
            "options": {
                "temperature": ollama_temperature(&options.model, options.temperature),
                "num_predict": options.max_tokens,
            }
        });

        let value = send_json(self.client.post(url).json(&body)).await?;
        let text = extract_ollama_text(&value)?;

        Ok(GenerateResponse {
            provider: ProviderKind::Ollama,
            model: options.model,
            text,
        })
    }
}

fn build_http_client(timeout_secs: u64) -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(timeout_secs.max(1)))
        .build()
        .context("failed to build llm http client")
}

async fn embed_with_openai(
    config: &LlmConfig,
    model: Option<&str>,
    dimensions: usize,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let provider = config.openai.clone().unwrap_or_default();
    let api_key = provider
        .api_key
        .as_deref()
        .ok_or_else(|| anyhow!("missing openai api key"))?;
    let selected_model = model.unwrap_or("text-embedding-3-small");
    let url = join_url(&provider.base_url, "embeddings")?;
    let mut body = json!({
        "model": selected_model,
        "input": inputs,
    });
    body["dimensions"] = json!(dimensions);

    let mut builder = build_http_client(config.timeout_secs)?
        .post(url)
        .bearer_auth(api_key)
        .json(&body);

    if let Some(org) = &provider.organization {
        builder = builder.header("OpenAI-Organization", org);
    }
    if let Some(project) = &provider.project {
        builder = builder.header("OpenAI-Project", project);
    }

    let value = send_json(builder).await?;
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("openai embeddings response missing data"))?;

    let mut rows = Vec::with_capacity(data.len());
    for item in data {
        let embedding = item
            .get("embedding")
            .ok_or_else(|| anyhow!("openai embeddings response missing embedding"))?;
        rows.push(parse_embedding(embedding)?);
    }
    Ok(rows)
}

async fn embed_with_ollama(
    config: &LlmConfig,
    model: Option<&str>,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let provider = config.ollama.clone().unwrap_or_default();
    let selected_model = model.unwrap_or(&config.default_model);
    let url = join_url(&provider.base_url, "api/embed")?;
    let body = json!({
        "model": selected_model,
        "input": inputs,
        "keep_alive": provider.keep_alive,
    });

    let value = send_json(
        build_http_client(config.timeout_secs)?
            .post(url)
            .json(&body),
    )
    .await?;
    if let Some(items) = value.get("embeddings").and_then(Value::as_array) {
        return items.iter().map(parse_embedding).collect();
    }

    if let Some(single) = value.get("embedding") {
        return Ok(vec![parse_embedding(single)?]);
    }

    bail!("ollama embedding response did not contain embeddings")
}

fn parse_embedding(value: &Value) -> Result<Vec<f32>> {
    let items = value
        .as_array()
        .ok_or_else(|| anyhow!("embedding payload was not an array"))?;
    let mut embedding = Vec::with_capacity(items.len());
    for item in items {
        let number = item
            .as_f64()
            .ok_or_else(|| anyhow!("embedding value was not numeric"))?;
        embedding.push(number as f32);
    }
    if embedding.is_empty() {
        bail!("embedding payload was empty");
    }
    Ok(embedding)
}

fn fit_embedding(values: &[f32], dimensions: usize) -> Vec<f32> {
    let target_dimensions = dimensions.max(1);
    if values.is_empty() {
        return vec![0.0; target_dimensions];
    }

    let mut projected = vec![0.0f32; target_dimensions];
    for (index, value) in values.iter().enumerate() {
        let bucket = index % target_dimensions;
        projected[bucket] += *value;
    }

    let norm = projected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut projected {
            *value /= norm;
        }
    }

    projected
}

fn join_url(base_url: &str, path: &str) -> Result<Url> {
    let mut normalized = base_url.to_string();
    if !normalized.ends_with('/') {
        normalized.push('/');
    }

    Url::parse(&normalized)
        .with_context(|| format!("invalid base url: {base_url}"))?
        .join(path)
        .with_context(|| format!("failed to join base url {base_url} with path {path}"))
}

async fn send_json(builder: reqwest::RequestBuilder) -> Result<Value> {
    let response = builder.send().await.context("provider request failed")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read provider response")?;

    if !status.is_success() {
        bail!("provider request failed with status {status}: {body}");
    }

    serde_json::from_str(&body).with_context(|| format!("invalid provider json: {body}"))
}

fn extract_openai_text(value: &Value) -> Result<String> {
    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        return Ok(text.trim().to_string());
    }

    if let Some(output) = value.get("output").and_then(Value::as_array) {
        let mut parts = Vec::new();
        for item in output {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for chunk in content {
                    if let Some(text) = chunk.get("text").and_then(Value::as_str) {
                        parts.push(text.trim().to_string());
                    }
                }
            }
        }

        if !parts.is_empty() {
            return Ok(parts.join("\n").trim().to_string());
        }
    }

    extract_text_deep(value).ok_or_else(|| anyhow!("openai response did not contain text"))
}

fn extract_anthropic_text(value: &Value) -> Result<String> {
    if let Some(content) = value.get("content").and_then(Value::as_array) {
        let mut parts = Vec::new();
        for block in content {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                parts.push(text.trim().to_string());
            }
        }

        if !parts.is_empty() {
            return Ok(parts.join("\n").trim().to_string());
        }
    }

    extract_text_deep(value).ok_or_else(|| anyhow!("anthropic response did not contain text"))
}

fn extract_google_text(value: &Value) -> Result<String> {
    if let Some(candidates) = value.get("candidates").and_then(Value::as_array) {
        for candidate in candidates {
            if let Some(parts) = candidate
                .get("content")
                .and_then(|content| content.get("parts"))
                .and_then(Value::as_array)
            {
                let texts = parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .map(|text| text.trim().to_string())
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>();

                if !texts.is_empty() {
                    return Ok(texts.join("\n"));
                }
            }
        }
    }

    extract_text_deep(value).ok_or_else(|| anyhow!("google response did not contain text"))
}

fn extract_ollama_text(value: &Value) -> Result<String> {
    if let Some(text) = value
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
    {
        let trimmed = parse_ollama_reply_text(text).trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    if let Some(text) = value.get("response").and_then(Value::as_str) {
        let trimmed = parse_ollama_reply_text(text).trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    if value.get("thinking").is_some() {
        bail!("ollama returned reasoning but no final response text");
    }

    extract_text_deep(value).ok_or_else(|| anyhow!("ollama response did not contain text"))
}

fn ollama_system_prompt(model: &str, system_prompt: &str) -> String {
    let json_contract = r#"Return only valid JSON matching this schema: {"reply": "string"}."#;
    if model.to_ascii_lowercase().contains("qwen3") {
        format!("/no_think\n{system_prompt}\n{json_contract}")
    } else {
        format!("{system_prompt}\n{json_contract}")
    }
}

fn ollama_reply_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "reply": {
                "type": "string"
            }
        },
        "required": ["reply"]
    })
}

fn ollama_temperature(model: &str, requested: f32) -> f32 {
    if model.to_ascii_lowercase().contains("qwen3") {
        requested.min(0.35)
    } else {
        requested
    }
}

fn strip_think_blocks(input: &str) -> String {
    if let Some(index) = input.rfind("</think>") {
        return input[index + "</think>".len()..].trim().to_string();
    }

    let mut remaining = input;
    let mut output = String::new();

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];
        if let Some(end) = after_start.find("</think>") {
            remaining = &after_start[end + "</think>".len()..];
        } else {
            break;
        }
    }

    output.trim().to_string()
}

fn parse_ollama_reply_text(input: &str) -> String {
    let cleaned = strip_think_blocks(input);
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
        if let Some(reply) = value.get("reply").and_then(Value::as_str) {
            return reply.trim().to_string();
        }
    }
    cleaned.trim().to_string()
}

fn extract_text_deep(value: &Value) -> Option<String> {
    let mut parts = Vec::new();
    collect_text_fragments(value, &mut parts);
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n").trim().to_string())
    }
}

fn collect_text_fragments(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let text_key = matches!(
                    key.as_str(),
                    "text" | "output_text" | "response" | "content"
                );

                if text_key {
                    match child {
                        Value::String(text) => {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                parts.push(trimmed.to_string());
                            }
                        }
                        _ => collect_text_fragments(child, parts),
                    }
                } else {
                    collect_text_fragments(child, parts);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_text_fragments(item, parts);
            }
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        extract_ollama_text, ollama_system_prompt, ollama_temperature, parse_ollama_reply_text,
        strip_think_blocks,
    };

    #[test]
    fn strip_think_blocks_keeps_only_final_answer_after_closing_tag() {
        let input = "analysis text\n</think>\n\nHello from the tavern.";
        assert_eq!(strip_think_blocks(input), "Hello from the tavern.");
    }

    #[test]
    fn extract_ollama_text_prefers_chat_content_and_drops_reasoning() {
        let payload = json!({
            "message": {
                "role": "assistant",
                "content": "reasoning here\n</think>\n\nZero. The tavern's been sober since dawn.",
                "thinking": "hidden"
            }
        });

        let text = extract_ollama_text(&payload).unwrap();
        assert_eq!(text, "Zero. The tavern's been sober since dawn.");
    }

    #[test]
    fn parse_ollama_reply_text_extracts_reply_from_json_content() {
        let input = "{\n  \"reply\": \"Hive? I know enough to keep the mugs moving.\"\n}";
        assert_eq!(
            parse_ollama_reply_text(input),
            "Hive? I know enough to keep the mugs moving."
        );
    }

    #[test]
    fn extract_ollama_text_prefers_structured_reply_field() {
        let payload = json!({
            "message": {
                "role": "assistant",
                "content": "{\n  \"reply\": \"Keep your HIVE handy and your stories cleaner.\"\n}"
            }
        });

        let text = extract_ollama_text(&payload).unwrap();
        assert_eq!(text, "Keep your HIVE handy and your stories cleaner.");
    }

    #[test]
    fn ollama_system_prompt_adds_json_contract() {
        let prompt = ollama_system_prompt("qwen3:4b", "You are a tavern bot.");
        assert!(prompt.contains("/no_think"));
        assert!(prompt.contains("Return only valid JSON"));
    }

    #[test]
    fn ollama_temperature_clamps_qwen3() {
        assert_eq!(ollama_temperature("qwen3:4b", 0.9), 0.35);
        assert_eq!(ollama_temperature("llama3.2", 0.9), 0.9);
    }
}
