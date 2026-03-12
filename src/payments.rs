#![allow(dead_code)]

use std::{collections::HashMap, time::Duration};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::model::PricedItem;

const DEFAULT_HIVE_API_URL: &str = "https://api.hive.blog";
const DEFAULT_HIVE_ENGINE_API_URL: &str = "https://api.hive-engine.com/rpc/blockchain";
const DEFAULT_TIMEOUT_SECS: u64 = 20;
const DEFAULT_HISTORY_BATCH_SIZE: u32 = 100;
const HIVE_NAI: &str = "@@000000021";
const HBD_NAI: &str = "@@000000013";

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
        let seed = match cursor {
            Some(cursor) => cursor.clone(),
            None => {
                return Ok(PaymentPollOutcome {
                    cursor: self.bootstrap().await?,
                    payments: Vec::new(),
                });
            }
        };

        let hive_client = HiveClient::new(self.client.clone(), &self.config);
        let hive_engine_client = HiveEngineClient::new(self.client.clone(), &self.config);

        let (hive_cursor, mut payments) = hive_client
            .poll_incoming(
                &self.config.bot_account,
                seed.hive.as_ref().unwrap_or(&HiveHistoryCursor {
                    account: self.config.bot_account.clone(),
                    last_sequence: 0,
                    last_block: 0,
                    last_tx_id: None,
                }),
            )
            .await?;

        let (hive_engine_cursor, mut hive_engine_payments) = hive_engine_client
            .poll_incoming(
                &self.config.bot_account,
                seed.hive_engine.as_ref().unwrap_or(&HiveEngineCursor {
                    account: self.config.bot_account.clone(),
                    last_block: 0,
                    last_ref_hive_block: None,
                    last_tx_id: None,
                }),
            )
            .await?;

        payments.append(&mut hive_engine_payments);
        payments.sort_by_key(|payment| (payment.block_height, payment.timestamp));

        Ok(PaymentPollOutcome {
            cursor: PaymentCursor {
                hive: Some(hive_cursor),
                hive_engine: Some(hive_engine_cursor),
            },
            payments,
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
        let response: HiveEngineFindOneResponse<HiveEngineTokenRecord> = post_json(
            &self.client,
            &self.contracts_endpoint,
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

        Ok(response
            .result
            .and_then(|record| normalize_token_issuer(record.issuer)))
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

async fn post_json<T>(client: &Client, endpoint: &str, body: &Value) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let response = client
        .post(endpoint)
        .json(body)
        .send()
        .await
        .with_context(|| format!("request to {endpoint} failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("failed reading response from {endpoint}"))?;

    if !status.is_success() {
        bail!("request to {endpoint} failed with status {status}: {text}");
    }

    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode response from {endpoint}: {text}"))
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
    use super::derive_hive_engine_contracts_api_url;

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
}
