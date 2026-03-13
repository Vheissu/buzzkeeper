use std::collections::BTreeMap;

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};

use crate::payments::PaymentCursor;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThemeSkin {
    Tavern,
    Coffeehouse,
    PotionShop,
    CyberDive,
}

impl ThemeSkin {
    pub fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "tavern" | "pub" => Some(Self::Tavern),
            "coffee" | "coffeehouse" | "cafe" => Some(Self::Coffeehouse),
            "potion" | "potions" | "potion_shop" | "potion-shop" => Some(Self::PotionShop),
            "cyber" | "cyberdive" | "cyber-dive" => Some(Self::CyberDive),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tavern => "tavern",
            Self::Coffeehouse => "coffeehouse",
            Self::PotionShop => "potion_shop",
            Self::CyberDive => "cyber_dive",
        }
    }

    pub fn flavor(&self) -> &'static str {
        match self {
            Self::Tavern => "rowdy fantasy tavern",
            Self::Coffeehouse => "over-caffeinated late-night cafe",
            Self::PotionShop => "chaotic potion bar",
            Self::CyberDive => "neon market dive bar",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(rename_all = "snake_case")]
pub enum IntoxicationStage {
    #[default]
    Sober,
    Warm,
    Tipsy,
    Buzzing,
    Cooked,
    Gone,
    Hungover,
}

impl IntoxicationStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sober => "sober",
            Self::Warm => "warm",
            Self::Tipsy => "tipsy",
            Self::Buzzing => "buzzing",
            Self::Cooked => "cooked",
            Self::Gone => "gone",
            Self::Hungover => "hungover",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Sober => "Sober",
            Self::Warm => "Warm",
            Self::Tipsy => "Tipsy",
            Self::Buzzing => "Buzzing",
            Self::Cooked => "Cooked",
            Self::Gone => "Absolutely Gone",
            Self::Hungover => "Hungover",
        }
    }

    pub fn default_persona(&self) -> &'static str {
        match self {
            Self::Sober => "watchful bartender",
            Self::Warm => "confident host",
            Self::Tipsy => "chaos sommelier",
            Self::Buzzing => "karaoke gremlin",
            Self::Cooked => "unhinged market analyst",
            Self::Gone => "lore catastrophe",
            Self::Hungover => "apology machine",
        }
    }

    pub fn from_points(points: i32) -> Self {
        match points {
            i if i <= 0 => Self::Sober,
            1..=14 => Self::Warm,
            15..=34 => Self::Tipsy,
            35..=59 => Self::Buzzing,
            60..=89 => Self::Cooked,
            _ => Self::Gone,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssetLedger {
    Hive,
    Hbd,
    HiveEngine,
}

impl AssetLedger {
    pub fn parse(input: &str) -> Option<Self> {
        match input.trim().to_ascii_lowercase().as_str() {
            "hive" => Some(Self::Hive),
            "hbd" => Some(Self::Hbd),
            "hive-engine" | "hive_engine" | "engine" => Some(Self::HiveEngine),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hive => "hive",
            Self::Hbd => "hbd",
            Self::HiveEngine => "hive_engine",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenSpec {
    pub ledger: AssetLedger,
    pub symbol: String,
    pub issuer: Option<String>,
}

impl TokenSpec {
    pub fn display(&self) -> String {
        match &self.issuer {
            Some(issuer) => format!(
                "{} on {} ({issuer})",
                self.symbol.to_ascii_uppercase(),
                self.ledger.as_str()
            ),
            None => format!(
                "{} on {}",
                self.symbol.to_ascii_uppercase(),
                self.ledger.as_str()
            ),
        }
    }

    pub fn key(&self) -> String {
        format!(
            "{}:{}",
            self.ledger.as_str(),
            self.symbol.to_ascii_uppercase()
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PricedItem {
    pub token: TokenSpec,
    pub amount: f64,
}

impl PricedItem {
    pub fn display(&self) -> String {
        format!(
            "{:.3} {}",
            self.amount,
            self.token.symbol.to_ascii_uppercase()
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DrinkCategory {
    Drink,
    Recovery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrinkDefinition {
    pub slug: String,
    pub name: String,
    pub description: String,
    pub category: DrinkCategory,
    pub price: PricedItem,
    pub party_points: u32,
    pub intoxication_delta: i32,
    pub flavor_line: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDefinition {
    pub slug: String,
    pub name: String,
    pub description: String,
    pub price: PricedItem,
    pub minimum_stage: IntoxicationStage,
    pub flavor: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    Preference,
    Lore,
    Incident,
    SelfReflection,
}

impl Default for MemoryCategory {
    fn default() -> Self {
        Self::Lore
    }
}

impl MemoryCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Preference => "preference",
            Self::Lore => "lore",
            Self::Incident => "incident",
            Self::SelfReflection => "self_reflection",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    UserNote,
    BotUtterance,
    SystemEvent,
}

impl Default for MemorySource {
    fn default() -> Self {
        Self::UserNote
    }
}

impl MemorySource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserNote => "user_note",
            Self::BotUtterance => "bot_utterance",
            Self::SystemEvent => "system_event",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub at: DateTime<Utc>,
    pub user_id: String,
    pub user_name: String,
    pub text: String,
    #[serde(default)]
    pub category: MemoryCategory,
    #[serde(default)]
    pub source: MemorySource,
    #[serde(default)]
    pub bot_stage: Option<IntoxicationStage>,
    #[serde(default)]
    pub bot_persona: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default = "default_memory_importance")]
    pub importance: u8,
    #[serde(default)]
    pub recall_count: u32,
    #[serde(default)]
    pub last_recalled_at: Option<DateTime<Utc>>,
}

fn default_memory_importance() -> u8 {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegularRecord {
    pub display_name: String,
    pub total_spend: BTreeMap<String, f64>,
    pub drinks_bought: u32,
    pub actions_triggered: u32,
    pub recovery_actions: u32,
    pub chaos_score: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TavernEvent {
    pub at: DateTime<Utc>,
    pub kind: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationSpeaker {
    User,
    Bot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub at: DateTime<Utc>,
    pub speaker: ConversationSpeaker,
    pub speaker_name: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationSession {
    pub turns: Vec<ConversationTurn>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuildPermissions {
    pub allowed_channel_ids: Vec<u64>,
    pub admin_role_ids: Vec<u64>,
    pub payment_channel_id: Option<u64>,
}

impl GuildPermissions {
    pub fn allows_channel(&self, channel_id: u64) -> bool {
        self.allowed_channel_ids.is_empty() || self.allowed_channel_ids.contains(&channel_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHours {
    pub enabled: bool,
    pub start_hour_utc: u8,
    pub end_hour_utc: u8,
}

impl Default for QuietHours {
    fn default() -> Self {
        Self {
            enabled: false,
            start_hour_utc: 22,
            end_hour_utc: 8,
        }
    }
}

impl QuietHours {
    pub fn is_active(&self, now: DateTime<Utc>) -> bool {
        if !self.enabled {
            return false;
        }

        let hour = now.hour() as u8;
        if self.start_hour_utc == self.end_hour_utc {
            return true;
        }
        if self.start_hour_utc < self.end_hour_utc {
            hour >= self.start_hour_utc && hour < self.end_hour_utc
        } else {
            hour >= self.start_hour_utc || hour < self.end_hour_utc
        }
    }
}

fn default_public_tavern_enabled() -> bool {
    false
}

fn default_ambient_reply_chance_pct() -> u8 {
    18
}

fn default_ambient_cooldown_secs() -> u64 {
    12
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuildConfig {
    pub guild_id: u64,
    pub bot_name: String,
    pub theme: ThemeSkin,
    pub house_currency: TokenSpec,
    pub bot_hive_account: Option<String>,
    pub llm_provider: String,
    pub llm_model: Option<String>,
    pub quiet_hours: QuietHours,
    pub mention_replies_enabled: bool,
    pub mention_cooldown_secs: u64,
    pub chat_cooldown_secs: u64,
    #[serde(default = "default_public_tavern_enabled")]
    pub public_tavern_enabled: bool,
    #[serde(default = "default_ambient_reply_chance_pct")]
    pub ambient_reply_chance_pct: u8,
    #[serde(default = "default_ambient_cooldown_secs")]
    pub ambient_cooldown_secs: u64,
    pub roast_mode_enabled: bool,
    pub custom_system_prompt: Option<String>,
    pub permissions: GuildPermissions,
    pub drinks: Vec<DrinkDefinition>,
    pub actions: Vec<ActionDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuildState {
    pub config: GuildConfig,
    pub party_meter: u32,
    pub intoxication_points: i32,
    pub hangover_until: Option<DateTime<Utc>>,
    pub persona: String,
    pub memories: Vec<MemoryEntry>,
    pub regulars: BTreeMap<String, RegularRecord>,
    pub recent_events: Vec<TavernEvent>,
    #[serde(default)]
    pub conversation_sessions: BTreeMap<String, ConversationSession>,
    pub payment_cursor: Option<PaymentCursor>,
    pub recent_payment_ids: Vec<String>,
    #[serde(default)]
    pub chat_reply_at_by_session: BTreeMap<String, DateTime<Utc>>,
    #[serde(default)]
    pub mention_reply_at_by_session: BTreeMap<String, DateTime<Utc>>,
    #[serde(default)]
    pub ambient_reply_at_by_channel: BTreeMap<u64, DateTime<Utc>>,
    #[serde(default)]
    pub last_chat_reply_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_mention_reply_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl GuildState {
    pub fn new(
        guild_id: u64,
        bot_name: String,
        theme: ThemeSkin,
        house_currency: TokenSpec,
        llm_provider: String,
        llm_model: Option<String>,
        now: DateTime<Utc>,
    ) -> Self {
        let config = GuildConfig {
            guild_id,
            bot_name,
            theme: theme.clone(),
            house_currency: house_currency.clone(),
            bot_hive_account: None,
            llm_provider,
            llm_model,
            quiet_hours: QuietHours::default(),
            mention_replies_enabled: true,
            mention_cooldown_secs: 3,
            chat_cooldown_secs: 2,
            public_tavern_enabled: false,
            ambient_reply_chance_pct: 18,
            ambient_cooldown_secs: 12,
            roast_mode_enabled: true,
            custom_system_prompt: None,
            permissions: GuildPermissions::default(),
            drinks: default_drinks(&house_currency, &theme),
            actions: default_actions(&house_currency, &theme),
        };
        let persona = IntoxicationStage::Sober.default_persona().to_string();

        Self {
            config,
            party_meter: 0,
            intoxication_points: 0,
            hangover_until: None,
            persona,
            memories: Vec::new(),
            regulars: BTreeMap::new(),
            recent_events: Vec::new(),
            conversation_sessions: BTreeMap::new(),
            payment_cursor: None,
            recent_payment_ids: Vec::new(),
            chat_reply_at_by_session: BTreeMap::new(),
            mention_reply_at_by_session: BTreeMap::new(),
            ambient_reply_at_by_channel: BTreeMap::new(),
            last_chat_reply_at: None,
            last_mention_reply_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn stage_at(&self, now: DateTime<Utc>) -> IntoxicationStage {
        if self.hangover_until.is_some_and(|until| now < until) {
            IntoxicationStage::Hungover
        } else {
            IntoxicationStage::from_points(self.intoxication_points)
        }
    }
}

pub fn default_house_currency() -> TokenSpec {
    TokenSpec {
        ledger: AssetLedger::Hive,
        symbol: "HIVE".to_string(),
        issuer: None,
    }
}

pub fn default_drinks(token: &TokenSpec, theme: &ThemeSkin) -> Vec<DrinkDefinition> {
    let (drink1, drink2, drink3, recovery1, recovery2) = match theme {
        ThemeSkin::Tavern => (
            ("First Round", "Gets the room moving.", 1.0, 10, 8),
            ("Goblin Grog", "Reckless confidence in a mug.", 2.5, 20, 18),
            ("Blackout Barrel", "This never ends well.", 5.0, 35, 34),
            (
                "Greasy Breakfast",
                "Soaks up shame and poor decisions.",
                1.5,
                0,
                -18,
            ),
            (
                "Espresso Shot",
                "Harsh reality in concentrated form.",
                0.75,
                0,
                -10,
            ),
        ),
        ThemeSkin::Coffeehouse => (
            (
                "Cold Brew IV",
                "The bot starts talking too quickly.",
                1.0,
                10,
                7,
            ),
            (
                "Quad Espresso",
                "The room acquires dangerous momentum.",
                2.0,
                18,
                15,
            ),
            (
                "Nitro Overclock",
                "This is how legends get banned.",
                4.0,
                30,
                28,
            ),
            (
                "Oat Milk Reset",
                "A softer landing than deserved.",
                1.2,
                0,
                -16,
            ),
            (
                "Hydration Break",
                "A glass of water and a reality check.",
                0.5,
                0,
                -9,
            ),
        ),
        ThemeSkin::PotionShop => (
            (
                "Mana Fizz",
                "A bright potion with suspicious sparkle.",
                1.0,
                12,
                8,
            ),
            (
                "Chaos Tonic",
                "The cork pops and so does decorum.",
                2.0,
                22,
                16,
            ),
            (
                "Eldritch Elixir",
                "Probably not cleared by regulators.",
                4.5,
                36,
                30,
            ),
            (
                "Antidote Soup",
                "Tastes awful, works immediately.",
                1.4,
                0,
                -17,
            ),
            (
                "Clarity Draught",
                "A temporary restoration of judgment.",
                0.8,
                0,
                -10,
            ),
        ),
        ThemeSkin::CyberDive => (
            ("Patch Cable", "Small buzz, clean signal.", 1.0, 11, 7),
            (
                "Neon Surge",
                "Packet loss begins in the frontal lobe.",
                2.4,
                21,
                17,
            ),
            (
                "Kernel Panic",
                "The whole tavern is on unsafe clock speed.",
                4.8,
                36,
                31,
            ),
            (
                "Reboot Noodles",
                "The recovery ramen of champions.",
                1.4,
                0,
                -16,
            ),
            ("Ice Water Ping", "Latency down, clarity up.", 0.6, 0, -9),
        ),
    };

    vec![
        build_drink(token, drink1, DrinkCategory::Drink),
        build_drink(token, drink2, DrinkCategory::Drink),
        build_drink(token, drink3, DrinkCategory::Drink),
        build_drink(token, recovery1, DrinkCategory::Recovery),
        build_drink(token, recovery2, DrinkCategory::Recovery),
    ]
}

pub fn default_actions(token: &TokenSpec, theme: &ThemeSkin) -> Vec<ActionDefinition> {
    let prices = match theme {
        ThemeSkin::Tavern => [0.75, 1.2, 1.4, 2.0, 2.5, 3.5],
        ThemeSkin::Coffeehouse => [0.6, 1.0, 1.2, 1.8, 2.2, 3.0],
        ThemeSkin::PotionShop => [0.7, 1.1, 1.3, 2.0, 2.6, 3.4],
        ThemeSkin::CyberDive => [0.7, 1.1, 1.4, 2.1, 2.7, 3.6],
    };

    vec![
        ActionDefinition {
            slug: "toast".to_string(),
            name: "Toast".to_string(),
            description: "Raise a glass and celebrate the channel.".to_string(),
            price: priced(token, prices[0]),
            minimum_stage: IntoxicationStage::Warm,
            flavor: "A quick public cheer with a little swagger.".to_string(),
        },
        ActionDefinition {
            slug: "compliment".to_string(),
            name: "Compliment".to_string(),
            description: "Lavish the room or a target with suspiciously intense praise."
                .to_string(),
            price: priced(token, prices[1]),
            minimum_stage: IntoxicationStage::Warm,
            flavor: "Positive reinforcement, possibly overcooked.".to_string(),
        },
        ActionDefinition {
            slug: "roast".to_string(),
            name: "Light Roast".to_string(),
            description: "A moderation-safe jab at a target or the whole tavern.".to_string(),
            price: priced(token, prices[2]),
            minimum_stage: IntoxicationStage::Tipsy,
            flavor: "Friendly fire only.".to_string(),
        },
        ActionDefinition {
            slug: "karaoke".to_string(),
            name: "Karaoke".to_string(),
            description: "Break into a dramatic musical number.".to_string(),
            price: priced(token, prices[3]),
            minimum_stage: IntoxicationStage::Buzzing,
            flavor: "The room did not approve this acoustically.".to_string(),
        },
        ActionDefinition {
            slug: "sports-cast".to_string(),
            name: "Sports Cast".to_string(),
            description: "Narrate current chat energy like a live match.".to_string(),
            price: priced(token, prices[4]),
            minimum_stage: IntoxicationStage::Cooked,
            flavor: "Commentary volume exceeds skill level.".to_string(),
        },
        ActionDefinition {
            slug: "prophecy".to_string(),
            name: "Prophecy".to_string(),
            description: "Deliver a fake-deep tavern prophecy.".to_string(),
            price: priced(token, prices[5]),
            minimum_stage: IntoxicationStage::Cooked,
            flavor: "Absolutely not financial advice.".to_string(),
        },
    ]
}

fn build_drink(
    token: &TokenSpec,
    tuple: (&str, &str, f64, u32, i32),
    category: DrinkCategory,
) -> DrinkDefinition {
    let slug = slugify(tuple.0);
    DrinkDefinition {
        slug,
        name: tuple.0.to_string(),
        description: tuple.1.to_string(),
        category,
        price: priced(token, tuple.2),
        party_points: tuple.3,
        intoxication_delta: tuple.4,
        flavor_line: tuple.1.to_string(),
    }
}

fn priced(token: &TokenSpec, amount: f64) -> PricedItem {
    PricedItem {
        token: token.clone(),
        amount,
    }
}

pub fn slugify(input: &str) -> String {
    let mut slug = String::with_capacity(input.len());
    let mut last_dash = false;

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }

    slug.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use serde_json::json;

    use super::{GuildState, QuietHours};

    fn dt(input: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(input)
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn quiet_hours_disabled_never_blocks() {
        let quiet = QuietHours::default();
        assert!(!quiet.is_active(dt("2026-03-12T23:00:00Z")));
    }

    #[test]
    fn quiet_hours_wrap_across_midnight() {
        let quiet = QuietHours {
            enabled: true,
            start_hour_utc: 22,
            end_hour_utc: 8,
        };
        assert!(quiet.is_active(dt("2026-03-12T23:00:00Z")));
        assert!(quiet.is_active(dt("2026-03-12T07:00:00Z")));
        assert!(!quiet.is_active(dt("2026-03-12T12:00:00Z")));
    }

    #[test]
    fn quiet_hours_same_start_end_means_always_active() {
        let quiet = QuietHours {
            enabled: true,
            start_hour_utc: 0,
            end_hour_utc: 0,
        };
        assert!(quiet.is_active(dt("2026-03-12T12:00:00Z")));
    }

    #[test]
    fn guild_state_deserializes_legacy_config_without_public_tavern_fields() {
        let payload = json!({
            "config": {
                "guild_id": 1,
                "bot_name": "Baron Botley",
                "theme": "tavern",
                "house_currency": {
                    "ledger": "hive",
                    "symbol": "HIVE",
                    "issuer": null
                },
                "bot_hive_account": null,
                "llm_provider": "ollama",
                "llm_model": "qwen3:4b",
                "quiet_hours": {
                    "enabled": false,
                    "start_hour_utc": 22,
                    "end_hour_utc": 8
                },
                "mention_replies_enabled": true,
                "mention_cooldown_secs": 45,
                "chat_cooldown_secs": 10,
                "roast_mode_enabled": true,
                "custom_system_prompt": null,
                "permissions": {
                    "allowed_channel_ids": [],
                    "admin_role_ids": [],
                    "payment_channel_id": null
                },
                "drinks": [],
                "actions": []
            },
            "party_meter": 0,
            "intoxication_points": 0,
            "hangover_until": null,
            "persona": "watchful bartender",
            "memories": [],
            "regulars": {},
            "recent_events": [],
            "payment_cursor": null,
            "recent_payment_ids": [],
            "last_chat_reply_at": null,
            "last_mention_reply_at": null,
            "created_at": "2026-03-12T00:00:00Z",
            "updated_at": "2026-03-12T00:00:00Z"
        });

        let state: GuildState = serde_json::from_value(payload).unwrap();
        assert!(!state.config.public_tavern_enabled);
        assert_eq!(state.config.ambient_reply_chance_pct, 18);
        assert_eq!(state.config.ambient_cooldown_secs, 12);
    }
}
