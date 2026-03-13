#![allow(dead_code)]

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::model::PricedItem;

const DEFAULT_HIVE_API_URL: &str = "https://api.hive.blog";
const DEFAULT_HIVE_ENGINE_API_URL: &str = "https://api.hive-engine.com/rpc/blockchain";
const DEFAULT_TIMEOUT_SECS: u64 = 20;
const DEFAULT_HISTORY_BATCH_SIZE: u32 = 100;
const HIVE_NAI: &str = "@@000000021";
const HBD_NAI: &str = "@@000000013";
const TRANSIENT_RETRY_DELAYS_MS: [u64; 2] = [100, 250];

fn token_issuer_cache() -> &'static Mutex<HashMap<String, Option<String>>> {
    static TOKEN_ISSUER_CACHE: OnceLock<Mutex<HashMap<String, Option<String>>>> = OnceLock::new();
    TOKEN_ISSUER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TipReceipt {
    pub tx_ref: Option<String>,
    pub payer: Option<String>,
    pub price: PricedItem,
    pub recorded_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PaymentChain {
    Manual,
    Hive,
    HiveEngine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IncomingPayment {
    pub chain: PaymentChain,
    pub tx_id: String,
    pub tx_ref: Option<String>,
    pub sender: String,
    pub recipient: String,
    pub symbol: String,
    pub issuer: Option<String>,
    pub amount: String,
    pub memo: Option<String>,
    pub block_height: u64,
    pub timestamp: DateTime<Utc>,
    pub raw_payload: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PaymentCursor {
    pub hive: Option<HiveHistoryCursor>,
    pub hive_engine: Option<HiveEngineCursor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveHistoryCursor {
    pub account: String,
    pub last_sequence: u64,
    pub last_block: u64,
    pub last_tx_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveEngineCursor {
    pub account: String,
    pub last_block: u64,
    pub last_ref_hive_block: Option<u64>,
    pub last_tx_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentPollOutcome {
    pub cursor: PaymentCursor,
    pub payments: Vec<IncomingPayment>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentIngestConfig {
    pub bot_account: String,
    pub hive_api_url: String,
    pub hive_engine_api_url: String,
    pub history_batch_size: u32,
    pub timeout_secs: u64,
}

impl PaymentIngestConfig {
    pub fn new(bot_account: impl Into<String>) -> Self {
        Self {
            bot_account: bot_account.into(),
            hive_api_url: DEFAULT_HIVE_API_URL.to_string(),
            hive_engine_api_url: DEFAULT_HIVE_ENGINE_API_URL.to_string(),
            history_batch_size: DEFAULT_HISTORY_BATCH_SIZE,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }

    pub fn http_client(&self) -> Result<Client> {
        Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs.max(1)))
            .build()
            .context("failed to build payment http client")
    }

    pub fn hive_engine_contracts_api_url(&self) -> String {
        derive_hive_engine_contracts_api_url(&self.hive_engine_api_url)
    }
}

impl Default for PaymentIngestConfig {
    fn default() -> Self {
        Self::new("")
    }
}

#[allow(dead_code)]
#[async_trait]
pub trait PaymentAdapter: Send + Sync {
    fn name(&self) -> &'static str;
    async fn validate(&self, receipt: &TipReceipt) -> Result<bool>;
    async fn bootstrap_cursor(&self) -> Result<PaymentCursor>;
    async fn poll_incoming(&self, cursor: Option<&PaymentCursor>) -> Result<PaymentPollOutcome>;
}

#[derive(Debug, Clone, Default)]
pub struct ManualPaymentAdapter;

#[async_trait]
impl PaymentAdapter for ManualPaymentAdapter {
    fn name(&self) -> &'static str {
        "manual"
    }

    async fn validate(&self, _receipt: &TipReceipt) -> Result<bool> {
        Ok(true)
    }

    async fn bootstrap_cursor(&self) -> Result<PaymentCursor> {
        Ok(PaymentCursor::default())
    }

    async fn poll_incoming(&self, cursor: Option<&PaymentCursor>) -> Result<PaymentPollOutcome> {
        Ok(PaymentPollOutcome {
            cursor: cursor.cloned().unwrap_or_default(),
            payments: Vec::new(),
            warnings: Vec::new(),
        })
    }
}

#[derive(Clone)]
pub struct HivePaymentAdapter {
    config: PaymentIngestConfig,
    client: Client,
}

impl HivePaymentAdapter {
    pub fn new(config: PaymentIngestConfig) -> Result<Self> {
        if config.bot_account.trim().is_empty() {
            bail!("payment ingestion requires a configured bot account");
        }

        Ok(Self {
            client: config.http_client()?,
            config,
        })
    }

    pub async fn bootstrap(&self) -> Result<PaymentCursor> {
        let hive = HiveClient::new(self.client.clone(), &self.config)
            .bootstrap_cursor(&self.config.bot_account)
            .await?;
        let hive_engine = HiveEngineClient::new(self.client.clone(), &self.config)
            .bootstrap_cursor(&self.config.bot_account)
            .await?;

        Ok(PaymentCursor {
            hive: Some(hive),
            hive_engine: Some(hive_engine),
        })
    }

    pub async fn poll(&self, cursor: Option<&PaymentCursor>) -> Result<PaymentPollOutcome> {
        let hive_client = HiveClient::new(self.client.clone(), &self.config);
        let hive_engine_client = HiveEngineClient::new(self.client.clone(), &self.config);
        let seed = cursor.cloned().unwrap_or_default();
        let mut warnings = Vec::new();

        let (hive_cursor, mut payments) = if let Some(hive_cursor) = seed.hive.as_ref() {
            match hive_client
                .poll_incoming(&self.config.bot_account, hive_cursor)
                .await
            {
                Ok((cursor, payments)) => (Some(cursor), payments),
                Err(error) => {
                    warnings.push(summarize_payment_poll_error("Hive", &error));
                    (Some(hive_cursor.clone()), Vec::new())
                }
            }
        } else {
            match hive_client.bootstrap_cursor(&self.config.bot_account).await {
                Ok(cursor) => (Some(cursor), Vec::new()),
                Err(error) => {
                    warnings.push(summarize_payment_poll_error("Hive", &error));
                    (None, Vec::new())
                }
            }
        };

        let (hive_engine_cursor, mut hive_engine_payments) =
            if let Some(hive_engine_cursor) = seed.hive_engine.as_ref() {
                match hive_engine_client
                    .poll_incoming(&self.config.bot_account, hive_engine_cursor)
                    .await
                {
                    Ok((cursor, payments)) => (Some(cursor), payments),
                    Err(error) => {
                        warnings.push(summarize_payment_poll_error("Hive Engine", &error));
                        (Some(hive_engine_cursor.clone()), Vec::new())
                    }
                }
            } else {
                match hive_engine_client
                    .bootstrap_cursor(&self.config.bot_account)
                    .await
                {
                    Ok(cursor) => (Some(cursor), Vec::new()),
                    Err(error) => {
                        warnings.push(summarize_payment_poll_error("Hive Engine", &error));
                        (None, Vec::new())
                    }
                }
            };

        payments.append(&mut hive_engine_payments);
        payments.sort_by_key(|payment| (payment.block_height, payment.timestamp));

        Ok(PaymentPollOutcome {
            cursor: PaymentCursor {
                hive: hive_cursor,
                hive_engine: hive_engine_cursor,
            },
            payments,
            warnings,
        })
    }
}

#[async_trait]
impl PaymentAdapter for HivePaymentAdapter {
    fn name(&self) -> &'static str {
        "hive"
    }

    async fn validate(&self, receipt: &TipReceipt) -> Result<bool> {
        Ok(receipt
            .tx_ref
            .as_deref()
            .is_some_and(|tx| !tx.trim().is_empty()))
    }

    async fn bootstrap_cursor(&self) -> Result<PaymentCursor> {
        self.bootstrap().await
    }

    async fn poll_incoming(&self, cursor: Option<&PaymentCursor>) -> Result<PaymentPollOutcome> {
        self.poll(cursor).await
    }
}

#[derive(Clone)]
pub struct HiveClient {
    client: Client,
    endpoint: String,
    batch_size: u32,
}

impl HiveClient {
    pub fn new(client: Client, config: &PaymentIngestConfig) -> Self {
        Self {
            client,
            endpoint: config.hive_api_url.clone(),
            batch_size: config.history_batch_size.max(1),
        }
    }

    pub async fn bootstrap_cursor(&self, account: &str) -> Result<HiveHistoryCursor> {
        let latest = self.fetch_history(account, -1, 1).await?.into_iter().last();

        let (last_sequence, last_block, last_tx_id) = latest
            .map(|entry| {
                (
                    entry.sequence,
                    entry.block,
                    normalize_zero_tx_id(entry.tx_id),
                )
            })
            .unwrap_or((0, 0, None));

        Ok(HiveHistoryCursor {
            account: account.to_string(),
            last_sequence,
            last_block,
            last_tx_id,
        })
    }

    pub async fn poll_incoming(
        &self,
        account: &str,
        cursor: &HiveHistoryCursor,
    ) -> Result<(HiveHistoryCursor, Vec<IncomingPayment>)> {
        let latest = self.fetch_history(account, -1, 1).await?.into_iter().last();

        let Some(latest) = latest else {
            return Ok((cursor.clone(), Vec::new()));
        };

        if latest.sequence <= cursor.last_sequence {
            return Ok((
                HiveHistoryCursor {
                    account: account.to_string(),
                    last_sequence: latest.sequence,
                    last_block: latest.block,
                    last_tx_id: normalize_zero_tx_id(latest.tx_id),
                },
                Vec::new(),
            ));
        }

        let mut next_sequence = cursor.last_sequence.saturating_add(1);
        let mut payments = Vec::new();
        let latest_sequence = latest.sequence;
        let mut newest_cursor = cursor.clone();

        while next_sequence <= latest_sequence {
            let remaining = latest_sequence - next_sequence + 1;
            let chunk_size = remaining.min(u64::from(self.batch_size)) as u32;
            let chunk_end = next_sequence + u64::from(chunk_size) - 1;
            let start = i64::try_from(chunk_end).context("hive history sequence overflow")?;
            let entries = self.fetch_history(account, start, chunk_size).await?;

            for entry in entries {
                if entry.sequence < next_sequence {
                    continue;
                }

                newest_cursor.last_sequence = entry.sequence;
                newest_cursor.last_block = entry.block;
                newest_cursor.last_tx_id = normalize_zero_tx_id(entry.tx_id.clone());

                if let Some(payment) = parse_hive_history_entry(account, entry)? {
                    payments.push(payment);
                }
            }

            next_sequence = chunk_end.saturating_add(1);
        }

        newest_cursor.account = account.to_string();
        Ok((newest_cursor, payments))
    }

    async fn fetch_history(
        &self,
        account: &str,
        start: i64,
        limit: u32,
    ) -> Result<Vec<HiveHistoryEntry>> {
        let response: HiveAccountHistoryResponse = post_json(
            &self.client,
            &self.endpoint,
            "account_history_api.get_account_history",
            &json!({
                "jsonrpc": "2.0",
                "method": "account_history_api.get_account_history",
                "params": {
                    "account": account,
                    "start": start,
                    "limit": limit,
                },
                "id": 1,
            }),
        )
        .await?;

        response
            .result
            .history
            .into_iter()
            .map(HiveHistoryEntry::try_from)
            .collect()
    }
}

#[derive(Clone)]
pub struct HiveEngineClient {
    client: Client,
    endpoint: String,
    contracts_endpoint: String,
}

impl HiveEngineClient {
    pub fn new(client: Client, config: &PaymentIngestConfig) -> Self {
        Self {
            client,
            endpoint: config.hive_engine_api_url.clone(),
            contracts_endpoint: config.hive_engine_contracts_api_url(),
        }
    }

    pub async fn bootstrap_cursor(&self, account: &str) -> Result<HiveEngineCursor> {
        let latest = self.latest_block().await?;
        Ok(HiveEngineCursor {
            account: account.to_string(),
            last_block: latest.block_number,
            last_ref_hive_block: Some(latest.ref_hive_block_number),
            last_tx_id: latest
                .transactions
                .last()
                .map(|tx| tx.transaction_id.clone()),
        })
    }

    pub async fn poll_incoming(
        &self,
        account: &str,
        cursor: &HiveEngineCursor,
    ) -> Result<(HiveEngineCursor, Vec<IncomingPayment>)> {
        let latest = self.latest_block().await?;
        if latest.block_number <= cursor.last_block {
            return Ok((
                HiveEngineCursor {
                    account: account.to_string(),
                    last_block: latest.block_number,
                    last_ref_hive_block: Some(latest.ref_hive_block_number),
                    last_tx_id: latest
                        .transactions
                        .last()
                        .map(|tx| tx.transaction_id.clone()),
                },
                Vec::new(),
            ));
        }

        let mut payments = Vec::new();
        let mut newest_cursor = cursor.clone();

        let mut issuer_cache = HashMap::<String, Option<String>>::new();

        for block_number in cursor.last_block.saturating_add(1)..=latest.block_number {
            let block = self.block_info(block_number).await?;
            newest_cursor.last_block = block.block_number;
            newest_cursor.last_ref_hive_block = Some(block.ref_hive_block_number);

            for tx in &block.transactions {
                newest_cursor.last_tx_id = Some(tx.transaction_id.clone());
                let mut events = parse_hive_engine_logs(tx.logs.as_deref())?;
                for event in events.drain(..) {
                    if !event.contract.eq_ignore_ascii_case("tokens") {
                        continue;
                    }

                    if !matches!(
                        event.event.as_str(),
                        "transfer" | "transferToContract" | "transferFromContract"
                    ) {
                        continue;
                    }

                    let Some(to) = event.data.get("to").and_then(Value::as_str) else {
                        continue;
                    };
                    if !account_matches(to, account) {
                        continue;
                    }

                    let sender = event
                        .data
                        .get("from")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string();
                    let symbol = event
                        .data
                        .get("symbol")
                        .and_then(Value::as_str)
                        .unwrap_or("UNKNOWN")
                        .to_string();
                    let issuer = match issuer_cache.get(&symbol) {
                        Some(value) => value.clone(),
                        None => {
                            let value = self.token_issuer(&symbol).await?;
                            issuer_cache.insert(symbol.clone(), value.clone());
                            value
                        }
                    };
                    let amount = event
                        .data
                        .get("quantity")
                        .and_then(Value::as_str)
                        .unwrap_or("0")
                        .to_string();
                    let memo = event
                        .data
                        .get("memo")
                        .and_then(Value::as_str)
                        .map(str::to_string);

                    payments.push(IncomingPayment {
                        chain: PaymentChain::HiveEngine,
                        tx_id: tx.transaction_id.clone(),
                        tx_ref: Some(tx.transaction_id.clone()),
                        sender,
                        recipient: to.to_string(),
                        symbol,
                        issuer,
                        amount,
                        memo,
                        block_height: block.block_number,
                        timestamp: parse_hive_timestamp(&block.timestamp)?,
                        raw_payload: Some(tx.payload.clone()),
                    });
                }
            }
        }

        newest_cursor.account = account.to_string();
        Ok((newest_cursor, payments))
    }

    async fn latest_block(&self) -> Result<HiveEngineBlockInfo> {
        let response: HiveEngineBlockResponse = post_json(
            &self.client,
            &self.endpoint,
            "getLatestBlockInfo",
            &json!({
                "jsonrpc": "2.0",
                "method": "getLatestBlockInfo",
                "params": {},
                "id": 1,
            }),
        )
        .await?;

        Ok(response.result)
    }

    async fn block_info(&self, block_number: u64) -> Result<HiveEngineBlockInfo> {
        let response: HiveEngineBlockResponse = post_json(
            &self.client,
            &self.endpoint,
            "getBlockInfo",
            &json!({
                "jsonrpc": "2.0",
                "method": "getBlockInfo",
                "params": {
                    "blockNumber": block_number,
                },
                "id": 1,
            }),
        )
        .await?;

        Ok(response.result)
    }

    async fn token_issuer(&self, symbol: &str) -> Result<Option<String>> {
        let cache_key = format!(
            "{}::{}",
            self.contracts_endpoint,
            symbol.trim().to_ascii_uppercase()
        );
        if let Some(value) = token_issuer_cache()
            .lock()
            .ok()
            .and_then(|cache| cache.get(&cache_key).cloned())
        {
            return Ok(value);
        }

        let response: HiveEngineFindOneResponse<HiveEngineTokenRecord> = post_json(
            &self.client,
            &self.contracts_endpoint,
            "token issuer lookup",
            &json!({
                "jsonrpc": "2.0",
                "method": "findOne",
                "params": {
                    "contract": "tokens",
                    "table": "tokens",
                    "query": {
                        "symbol": symbol,
                    },
                },
                "id": 1,
            }),
        )
        .await?;

        let issuer = response
            .result
            .and_then(|record| normalize_token_issuer(record.issuer));
        if let Ok(mut cache) = token_issuer_cache().lock() {
            cache.insert(cache_key, issuer.clone());
        }
        Ok(issuer)
    }
}

fn parse_hive_history_entry(
    expected_recipient: &str,
    entry: HiveHistoryEntry,
) -> Result<Option<IncomingPayment>> {
    if !entry
        .operation_type
        .eq_ignore_ascii_case("transfer_operation")
    {
        return Ok(None);
    }

    let value = entry
        .operation_value
        .as_object()
        .ok_or_else(|| anyhow!("hive transfer operation payload was not an object"))?;

    let Some(recipient) = value.get("to").and_then(Value::as_str) else {
        return Ok(None);
    };
    if !account_matches(recipient, expected_recipient) {
        return Ok(None);
    }

    let sender = value
        .get("from")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let memo = value
        .get("memo")
        .and_then(Value::as_str)
        .map(str::to_string);
    let (amount, symbol) = parse_hive_asset(
        value
            .get("amount")
            .ok_or_else(|| anyhow!("hive transfer missing amount"))?,
    )?;

    Ok(Some(IncomingPayment {
        chain: PaymentChain::Hive,
        tx_id: entry
            .tx_id
            .clone()
            .unwrap_or_else(|| format!("virtual-{}-{}", entry.block, entry.sequence)),
        tx_ref: entry.tx_id,
        sender,
        recipient: recipient.to_string(),
        symbol,
        issuer: None,
        amount,
        memo,
        block_height: entry.block,
        timestamp: parse_hive_timestamp(&entry.timestamp)?,
        raw_payload: None,
    }))
}

fn parse_hive_asset(value: &Value) -> Result<(String, String)> {
    if let Some(asset) = value.as_str() {
        let mut parts = asset.split_whitespace();
        let amount = parts
            .next()
            .ok_or_else(|| anyhow!("invalid legacy hive asset: {asset}"))?;
        let symbol = parts
            .next()
            .ok_or_else(|| anyhow!("invalid legacy hive asset: {asset}"))?;
        return Ok((amount.to_string(), symbol.to_string()));
    }

    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("unsupported hive asset payload: {value}"))?;
    let raw_amount = object
        .get("amount")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("hive asset object missing amount"))?;
    let precision = object
        .get("precision")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("hive asset object missing precision"))?
        as usize;
    let symbol = match object.get("nai").and_then(Value::as_str) {
        Some(HIVE_NAI) => "HIVE",
        Some(HBD_NAI) => "HBD",
        Some(other) => other,
        None => "UNKNOWN",
    };

    Ok((
        format_scaled_amount(raw_amount, precision),
        symbol.to_string(),
    ))
}

fn format_scaled_amount(raw_amount: &str, precision: usize) -> String {
    if precision == 0 {
        return raw_amount.to_string();
    }

    let negative = raw_amount.starts_with('-');
    let digits = raw_amount.trim_start_matches('-');
    let padded = if digits.len() <= precision {
        format!("{:0>width$}", digits, width = precision + 1)
    } else {
        digits.to_string()
    };
    let split_index = padded.len() - precision;
    let (whole, fraction) = padded.split_at(split_index);
    let prefix = if negative { "-" } else { "" };
    format!("{prefix}{whole}.{fraction}")
}

fn parse_hive_timestamp(input: &str) -> Result<DateTime<Utc>> {
    let naive = NaiveDateTime::parse_from_str(input, "%Y-%m-%dT%H:%M:%S")
        .with_context(|| format!("invalid hive timestamp: {input}"))?;
    Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

fn parse_hive_engine_logs(logs: Option<&str>) -> Result<Vec<HiveEngineLogEvent>> {
    let Some(logs) = logs else {
        return Ok(Vec::new());
    };
    if logs.trim().is_empty() {
        return Ok(Vec::new());
    }

    let parsed: HiveEngineLogsEnvelope =
        serde_json::from_str(logs).with_context(|| format!("invalid hive engine logs: {logs}"))?;
    Ok(parsed.events)
}

fn account_matches(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

fn normalize_token_issuer(input: String) -> Option<String> {
    let trimmed = input.trim().trim_start_matches('@').to_ascii_lowercase();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn summarize_payment_poll_error(chain: &str, error: &anyhow::Error) -> String {
    let text = format!("{error:#}");
    let request_name = extract_request_name(&text);
    let detail = if text.contains("status 503 Service Unavailable") {
        describe_request_issue(request_name, "returned 503 Service Unavailable")
    } else if text.contains("status 502 Bad Gateway") {
        describe_request_issue(request_name, "returned 502 Bad Gateway")
    } else if text.contains("status 504 Gateway Timeout") {
        describe_request_issue(request_name, "returned 504 Gateway Timeout")
    } else if text.to_ascii_lowercase().contains("timed out") {
        describe_request_issue(request_name, "timed out")
    } else if text.to_ascii_lowercase().contains("connection refused") {
        describe_request_issue(request_name, "connection was refused")
    } else {
        text.lines()
            .next()
            .unwrap_or("request failed")
            .trim()
            .to_string()
    };

    format!("{chain} payment RPC issue: {detail}. Keeping the previous cursor and retrying later.")
}

fn extract_request_name(text: &str) -> Option<&str> {
    text.split(" request to ")
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn describe_request_issue(request_name: Option<&str>, description: &str) -> String {
    request_name
        .map(|name| format!("{name} {description}"))
        .unwrap_or_else(|| format!("upstream {description}"))
}

fn derive_hive_engine_contracts_api_url(blockchain_url: &str) -> String {
    if let Some(prefix) = blockchain_url.strip_suffix("/blockchain") {
        return format!("{prefix}/contracts");
    }
    if let Some(prefix) = blockchain_url.strip_suffix("blockchain") {
        return format!("{prefix}contracts");
    }
    format!("{}/contracts", blockchain_url.trim_end_matches('/'))
}

fn normalize_zero_tx_id(tx_id: Option<String>) -> Option<String> {
    tx_id.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().all(|ch| ch == '0') {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

async fn post_json<T>(
    client: &Client,
    endpoint: &str,
    request_name: &str,
    body: &Value,
) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    for delay_ms in [
        Some(TRANSIENT_RETRY_DELAYS_MS[0]),
        Some(TRANSIENT_RETRY_DELAYS_MS[1]),
        None,
    ]
    .into_iter()
    {
        let response = match client.post(endpoint).json(body).send().await {
            Ok(response) => response,
            Err(error) => {
                if attempt_is_retryable_transport(delay_ms, &error) {
                    sleep(Duration::from_millis(delay_ms.unwrap())).await;
                    continue;
                }
                return Err(error)
                    .with_context(|| format!("{request_name} request to {endpoint} failed"));
            }
        };
        let status = response.status();
        let text = response
            .text()
            .await
            .with_context(|| format!("failed reading response from {endpoint}"))?;

        if !status.is_success() {
            if attempt_is_retryable_status(delay_ms, status) {
                sleep(Duration::from_millis(delay_ms.unwrap())).await;
                continue;
            }
            bail!("{request_name} request to {endpoint} failed with status {status}: {text}");
        }

        return serde_json::from_str(&text)
            .with_context(|| format!("failed to decode response from {endpoint}: {text}"));
    }

    unreachable!("retry loop always returns or bails")
}

fn attempt_is_retryable_transport(delay_ms: Option<u64>, error: &reqwest::Error) -> bool {
    delay_ms.is_some() && (error.is_timeout() || error.is_connect())
}

fn attempt_is_retryable_status(delay_ms: Option<u64>, status: StatusCode) -> bool {
    delay_ms.is_some()
        && matches!(
            status,
            StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT
        )
}

#[derive(Debug, Deserialize)]
struct HiveAccountHistoryResponse {
    result: HiveAccountHistoryResult,
}

#[derive(Debug, Deserialize)]
struct HiveAccountHistoryResult {
    history: Vec<(u64, HiveHistoryItem)>,
}

#[derive(Debug)]
struct HiveHistoryEntry {
    sequence: u64,
    block: u64,
    operation_type: String,
    operation_value: Value,
    timestamp: String,
    tx_id: Option<String>,
}

impl TryFrom<(u64, HiveHistoryItem)> for HiveHistoryEntry {
    type Error = anyhow::Error;

    fn try_from(value: (u64, HiveHistoryItem)) -> Result<Self> {
        Ok(Self {
            sequence: value.0,
            block: value.1.block,
            operation_type: value.1.op.operation_type,
            operation_value: value.1.op.value,
            timestamp: value.1.timestamp,
            tx_id: normalize_zero_tx_id(Some(value.1.trx_id)),
        })
    }
}

#[derive(Debug, Deserialize)]
struct HiveHistoryItem {
    block: u64,
    op: HiveOperation,
    timestamp: String,
    trx_id: String,
}

#[derive(Debug, Deserialize)]
struct HiveOperation {
    #[serde(rename = "type")]
    operation_type: String,
    value: Value,
}

#[derive(Debug, Deserialize)]
struct HiveEngineBlockResponse {
    result: HiveEngineBlockInfo,
}

#[derive(Debug, Deserialize)]
struct HiveEngineFindOneResponse<T> {
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
struct HiveEngineBlockInfo {
    #[serde(rename = "blockNumber")]
    block_number: u64,
    #[serde(rename = "refHiveBlockNumber")]
    ref_hive_block_number: u64,
    timestamp: String,
    #[serde(default)]
    transactions: Vec<HiveEngineTransaction>,
}

#[derive(Debug, Deserialize)]
struct HiveEngineTransaction {
    #[serde(rename = "transactionId")]
    transaction_id: String,
    payload: String,
    #[serde(default)]
    logs: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HiveEngineLogsEnvelope {
    #[serde(default)]
    events: Vec<HiveEngineLogEvent>,
}

#[derive(Debug, Deserialize)]
struct HiveEngineLogEvent {
    contract: String,
    event: String,
    #[serde(default)]
    data: Value,
}

#[derive(Debug, Deserialize)]
struct HiveEngineTokenRecord {
    issuer: String,
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use anyhow::anyhow;
    use reqwest::Client;
    use serde_json::json;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        task::JoinHandle,
    };

    use super::{
        HiveEngineClient, PaymentIngestConfig, derive_hive_engine_contracts_api_url, post_json,
        summarize_payment_poll_error,
    };

    #[test]
    fn derives_contracts_endpoint_from_blockchain_endpoint() {
        assert_eq!(
            derive_hive_engine_contracts_api_url("https://api.hive-engine.com/rpc/blockchain"),
            "https://api.hive-engine.com/rpc/contracts"
        );
        assert_eq!(
            derive_hive_engine_contracts_api_url("https://example.com/custom/blockchain"),
            "https://example.com/custom/contracts"
        );
    }

    #[test]
    fn summarizes_upstream_503_without_dumping_html() {
        let summary = summarize_payment_poll_error(
            "Hive Engine",
            &anyhow!(
                "getLatestBlockInfo request to https://api.hive-engine.com/rpc/blockchain failed with status 503 Service Unavailable: <html>...</html>"
            ),
        );

        assert_eq!(
            summary,
            "Hive Engine payment RPC issue: getLatestBlockInfo returned 503 Service Unavailable. Keeping the previous cursor and retrying later."
        );
    }

    #[tokio::test]
    async fn post_json_retries_transient_503s() {
        let responses = Arc::new([
            http_response(503, "<html>busy</html>", "text/html"),
            http_response(503, "<html>busy again</html>", "text/html"),
            http_response(200, r#"{"ok":true}"#, "application/json"),
        ]);
        let hits = Arc::new(AtomicUsize::new(0));
        let (endpoint, server) = spawn_test_server(responses.clone(), hits.clone()).await;

        let value: serde_json::Value = post_json(
            &Client::new(),
            &endpoint,
            "getLatestBlockInfo",
            &json!({"ping": true}),
        )
        .await
        .unwrap();

        assert_eq!(value, json!({"ok": true}));
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn token_issuer_lookup_uses_process_cache() {
        let responses = Arc::new([http_response(
            200,
            r#"{"jsonrpc":"2.0","id":1,"result":{"issuer":"leo.tokens"}}"#,
            "application/json",
        )]);
        let hits = Arc::new(AtomicUsize::new(0));
        let (endpoint, server) = spawn_test_server(responses.clone(), hits.clone()).await;

        let config = PaymentIngestConfig {
            bot_account: "buzzkeeper.bot".to_string(),
            hive_api_url: "https://api.hive.blog".to_string(),
            hive_engine_api_url: format!("{endpoint}/blockchain"),
            history_batch_size: 100,
            timeout_secs: 5,
        };
        let client = HiveEngineClient::new(config.http_client().unwrap(), &config);

        let first = client.token_issuer("LEO").await.unwrap();
        let second = client.token_issuer("LEO").await.unwrap();

        assert_eq!(first, Some("leo.tokens".to_string()));
        assert_eq!(second, Some("leo.tokens".to_string()));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    async fn spawn_test_server(
        responses: Arc<[String]>,
        hits: Arc<AtomicUsize>,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for response in responses.iter() {
                let (mut stream, _) = listener.accept().await.unwrap();
                hits.fetch_add(1, Ordering::SeqCst);
                read_http_request(&mut stream).await;
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        (format!("http://{addr}"), server)
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) {
        let mut buffer = [0u8; 4096];
        let mut request = Vec::new();
        loop {
            let count = stream.read(&mut buffer).await.unwrap();
            if count == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..count]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
    }

    fn http_response(status: u16, body: &str, content_type: &str) -> String {
        let reason = match status {
            200 => "OK",
            503 => "Service Unavailable",
            _ => "Test Response",
        };
        format!(
            "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }
}
