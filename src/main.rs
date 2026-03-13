mod app;
mod config;
mod llm;
mod memory_index;
mod model;
mod payments;
mod store;

use std::{sync::Arc, time::Duration};

use anyhow::{Context as AnyhowContext, Result, anyhow, bail};
use app::{
    ActorRef, BotApp, DispatchMessage, MentionReplyOutcome, PublicReplyOutcome, SetupRequest,
};
use config::AppConfig;
use model::{AssetLedger, GuildState, IntoxicationStage, ThemeSkin};
use poise::serenity_prelude as serenity;
use store::JsonStore;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

type Error = anyhow::Error;
type Context<'a> = poise::Context<'a, Data, Error>;

struct Data {
    app: Arc<BotApp>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,serenity=warn,poise=warn")),
        )
        .init();

    let config = AppConfig::from_env()?;
    let poll_interval_secs = config.payments.poll_interval_secs;
    let store = JsonStore::load(config.storage_path.clone()).await?;
    let app = Arc::new(BotApp::new(config.clone(), store)?);
    let pruned_memories = app.sanitize_memory_corpus().await?;
    if pruned_memories > 0 {
        info!("pruned {pruned_memories} bad bot memory entries from persistent state");
    }
    app.sync_memory_indexes().await?;

    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![
                setup(),
                status(),
                catalog(),
                policy(),
                set_quiet_hours(),
                set_reply_behavior(),
                set_public_tavern(),
                set_system_prompt(),
                clear_system_prompt(),
                set_payment_account(),
                set_payment_channel(),
                allow_channel(),
                disallow_channel(),
                clear_channels(),
                add_admin_role(),
                remove_admin_role(),
                sync_payments(),
                drink(),
                action(),
                set_stage(),
                clear_context(),
                debug_memory(),
                remember(),
                sync_commands(),
                chat(),
            ],
            prefix_options: poise::PrefixFrameworkOptions {
                mention_as_prefix: false,
                ..Default::default()
            },
            event_handler: |ctx, event, framework, data| {
                Box::pin(event_handler(ctx, event, framework, data))
            },
            ..Default::default()
        })
        .setup(move |ctx, ready, framework| {
            let app = Arc::clone(&app);
            Box::pin(async move {
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                for guild in &ready.guilds {
                    if let Err(error) = poise::builtins::register_in_guild(
                        ctx,
                        &framework.options().commands,
                        guild.id,
                    )
                    .await
                    {
                        warn!(
                            guild_id = guild.id.get(),
                            "failed to register guild commands on startup: {error:#}"
                        );
                    }
                }

                let poll_app = Arc::clone(&app);
                let http = ctx.http.clone();
                tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(Duration::from_secs(poll_interval_secs.max(5)));
                    loop {
                        interval.tick().await;
                        match poll_app.poll_payments().await {
                            Ok(summary) => {
                                if let Err(error) =
                                    flush_dispatches_http(http.as_ref(), &summary.messages).await
                                {
                                    warn!("failed to flush payment dispatches: {error:#}");
                                }
                            }
                            Err(error) => warn!("payment poll loop failed: {error:#}"),
                        }
                    }
                });

                Ok(Data { app })
            })
        })
        .build();

    let intents =
        serenity::GatewayIntents::non_privileged() | serenity::GatewayIntents::MESSAGE_CONTENT;
    let mut client = serenity::ClientBuilder::new(config.discord_token, intents)
        .framework(framework)
        .await
        .context("failed to build discord client")?;

    client.start().await.context("discord client exited")?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn setup(
    ctx: Context<'_>,
    #[description = "Bot character name"] bot_name: Option<String>,
    #[description = "tavern | coffeehouse | potion_shop | cyber_dive"] theme: Option<String>,
    #[description = "openai | anthropic | google | ollama | offline"] llm_provider: Option<String>,
    #[description = "Model override for the chosen provider"] llm_model: Option<String>,
    #[description = "hive | hbd | hive-engine"] asset_ledger: Option<String>,
    #[description = "Token symbol, for example HIVE, HBD, LEO"] asset_symbol: Option<String>,
    #[description = "Required for Hive Engine assets: issuer or token namespace"]
    asset_issuer: Option<String>,
    #[description = "Hive account to watch for incoming payments"] payment_account: Option<String>,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let request = SetupRequest {
        guild_id: guild_id.get(),
        setup_channel_id: ctx.channel_id().get(),
        bot_name,
        theme: theme.as_deref().and_then(ThemeSkin::parse),
        llm_provider,
        llm_model,
        asset_ledger: asset_ledger.as_deref().and_then(AssetLedger::parse),
        asset_symbol,
        asset_issuer,
        payment_account,
    };
    let state = ctx.data().app.configure_guild(request).await?;

    ctx.say(format!(
        "**{}** is ready.\nTheme: **{}**\nHouse asset: **{}**\nPayment account: **{}**\nPayment channel: **{}**\nLLM: **{}** / **{}**",
        state.config.bot_name,
        state.config.theme.as_str(),
        state.config.house_currency.display(),
        state
            .config
            .bot_hive_account
            .clone()
            .unwrap_or_else(|| "not set".to_string()),
        state
            .config
            .permissions
            .payment_channel_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not set".to_string()),
        state.config.llm_provider,
        state
            .config
            .llm_model
            .clone()
            .unwrap_or_else(|| "provider default".to_string())
    ))
    .await?;

    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn status(ctx: Context<'_>) -> Result<()> {
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx.data().app.status(guild_id.get()).await?;
    let stage = state.stage_at(chrono::Utc::now());
    let regulars = top_regulars_line(&state);
    let latest = state
        .recent_events
        .last()
        .map(|event| event.summary.clone())
        .unwrap_or_else(|| "No incidents yet.".to_string());

    ctx.say(format!(
        "**{}**\nStage: **{}**\nPersona: **{}**\nTheme: **{}**\nParty meter: **{}**\nHouse asset: **{}**\nPayment account: **{}**\nPayment channel: **{}**\nPublic tavern: **{}**\nAmbient cooldown: **{}s**\nAmbient chance: **{}%**\nRegulars: **{}**\nLatest event: {}",
        state.config.bot_name,
        stage.label(),
        state.persona,
        state.config.theme.as_str(),
        state.party_meter,
        state.config.house_currency.display(),
        state
            .config
            .bot_hive_account
            .clone()
            .unwrap_or_else(|| "not set".to_string()),
        state
            .config
            .permissions
            .payment_channel_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not set".to_string()),
        state.config.public_tavern_enabled,
        state.config.ambient_cooldown_secs,
        state.config.ambient_reply_chance_pct,
        regulars,
        latest
    ))
    .await?;

    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn catalog(ctx: Context<'_>) -> Result<()> {
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx.data().app.catalog(guild_id.get()).await?;

    let drinks = state
        .config
        .drinks
        .iter()
        .map(|drink| {
            format!(
                "`{}`: {} ({}, {})",
                drink.slug,
                drink.name,
                drink.price.display(),
                drink.description
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let actions = state
        .config
        .actions
        .iter()
        .map(|action| {
            format!(
                "`{}`: {} ({}, min stage: {})",
                action.slug,
                action.name,
                action.price.display(),
                action.minimum_stage.label()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    ctx.say(format!(
        "**Drinks**\n{drinks}\n\n**Actions**\n{actions}\n\nOn-chain memo format: `drink:<slug>` or `action:<slug> [target]`."
    ))
    .await?;

    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn policy(ctx: Context<'_>) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx.data().app.policy(guild_id.get()).await?;

    let allowed_channels = list_ids(&state.config.permissions.allowed_channel_ids);
    let admin_roles = list_ids(&state.config.permissions.admin_role_ids);
    let cursor_summary = state
        .payment_cursor
        .as_ref()
        .map(|cursor| {
            format!(
                "native_cursor={}, engine_cursor={}",
                cursor
                    .hive
                    .as_ref()
                    .map(|value| value.last_sequence.to_string())
                    .unwrap_or_else(|| "unset".to_string()),
                cursor
                    .hive_engine
                    .as_ref()
                    .map(|value| value.last_block.to_string())
                    .unwrap_or_else(|| "unset".to_string())
            )
        })
        .unwrap_or_else(|| "unset".to_string());

    ctx.say(format!(
        "**Policy**\nPayment account: **{}**\nPayment channel: **{}**\nAllowed channels: **{}**\nAdmin roles: **{}**\nMentions enabled: **{}**\nDirect mentions: **always reply when enabled**\nChat cooldown: **{}s**\nPublic tavern: **{}**\nAmbient chance: **{}%**\nAmbient cooldown: **{}s**\nQuiet hours: **{}**\nCursor: **{}**",
        state
            .config
            .bot_hive_account
            .clone()
            .unwrap_or_else(|| "not set".to_string()),
        state
            .config
            .permissions
            .payment_channel_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not set".to_string()),
        allowed_channels,
        admin_roles,
        state.config.mention_replies_enabled,
        state.config.chat_cooldown_secs,
        state.config.public_tavern_enabled,
        state.config.ambient_reply_chance_pct,
        state.config.ambient_cooldown_secs,
        if state.config.quiet_hours.enabled {
            format!(
                "{}:00-{}:00 UTC",
                state.config.quiet_hours.start_hour_utc, state.config.quiet_hours.end_hour_utc
            )
        } else {
            "disabled".to_string()
        },
        cursor_summary
    ))
    .await?;

    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_payment_account(
    ctx: Context<'_>,
    #[description = "Hive account name for incoming payments"] account: String,
    #[description = "Optional raw or mentioned channel id for announcements"]
    payment_channel: Option<String>,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let payment_channel_id = payment_channel
        .as_deref()
        .map(parse_snowflake)
        .transpose()?;
    let state = ctx
        .data()
        .app
        .set_payment_account(guild_id.get(), account, payment_channel_id)
        .await?;
    ctx.say(format!(
        "Watching **{}** for on-chain payments. Announcement channel: **{}**.",
        state
            .config
            .bot_hive_account
            .clone()
            .unwrap_or_else(|| "not set".to_string()),
        state
            .config
            .permissions
            .payment_channel_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not set".to_string())
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_quiet_hours(
    ctx: Context<'_>,
    #[description = "Enable quiet hours"] enabled: bool,
    #[description = "UTC start hour 0-23"] start_hour_utc: u8,
    #[description = "UTC end hour 0-23"] end_hour_utc: u8,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx
        .data()
        .app
        .set_quiet_hours(guild_id.get(), enabled, start_hour_utc, end_hour_utc)
        .await?;
    ctx.say(format!(
        "Quiet hours {}. Window: **{}:00-{}:00 UTC**.",
        if state.config.quiet_hours.enabled {
            "enabled"
        } else {
            "disabled"
        },
        state.config.quiet_hours.start_hour_utc,
        state.config.quiet_hours.end_hour_utc
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_reply_behavior(
    ctx: Context<'_>,
    #[description = "Whether direct @mentions should trigger replies"] mention_enabled: bool,
    #[description = "Stored for compatibility only, direct mentions now bypass cooldown"]
    mention_cooldown_secs: u64,
    #[description = "Cooldown for /chat replies in seconds"] chat_cooldown_secs: u64,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx
        .data()
        .app
        .set_reply_behavior(
            guild_id.get(),
            mention_enabled,
            mention_cooldown_secs,
            chat_cooldown_secs,
        )
        .await?;
    ctx.say(format!(
        "Mention replies: **{}**. Direct mentions bypass cooldown. Chat cooldown: **{}s**.",
        state.config.mention_replies_enabled, state.config.chat_cooldown_secs
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_public_tavern(
    ctx: Context<'_>,
    #[description = "Whether the bot can join public channel chatter"] enabled: bool,
    #[description = "Chance 0-100 for ambient replies when a message looks tavern-relevant"]
    ambient_reply_chance_pct: u8,
    #[description = "Per-channel cooldown for ambient replies in seconds"]
    ambient_cooldown_secs: u64,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx
        .data()
        .app
        .set_public_tavern_behavior(
            guild_id.get(),
            enabled,
            ambient_reply_chance_pct,
            ambient_cooldown_secs,
        )
        .await?;
    ctx.say(format!(
        "Public tavern: **{}**. Ambient reply chance: **{}%**. Ambient cooldown: **{}s**.",
        state.config.public_tavern_enabled,
        state.config.ambient_reply_chance_pct,
        state.config.ambient_cooldown_secs
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_system_prompt(
    ctx: Context<'_>,
    #[description = "Additional system prompt instructions"] prompt: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    ctx.data()
        .app
        .set_system_prompt_override(guild_id.get(), Some(prompt))
        .await?;
    ctx.say("Custom system prompt updated.").await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn clear_system_prompt(ctx: Context<'_>) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    ctx.data()
        .app
        .set_system_prompt_override(guild_id.get(), None)
        .await?;
    ctx.say("Custom system prompt cleared; built-in tavern persona prompt is active.")
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_payment_channel(
    ctx: Context<'_>,
    #[description = "Raw channel id or channel mention"] channel: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let channel_id = parse_snowflake(&channel)?;
    ctx.data()
        .app
        .set_payment_channel(guild_id.get(), channel_id)
        .await?;
    ctx.say(format!(
        "Payment announcements will go to **{}**.",
        channel_id
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn allow_channel(
    ctx: Context<'_>,
    #[description = "Raw channel id or channel mention"] channel: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let channel_id = parse_snowflake(&channel)?;
    ctx.data()
        .app
        .add_allowed_channel(guild_id.get(), channel_id)
        .await?;
    ctx.say(format!(
        "Allowed command channel added: **{}**.",
        channel_id
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn disallow_channel(
    ctx: Context<'_>,
    #[description = "Raw channel id or channel mention"] channel: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let channel_id = parse_snowflake(&channel)?;
    ctx.data()
        .app
        .remove_allowed_channel(guild_id.get(), channel_id)
        .await?;
    ctx.say(format!(
        "Allowed command channel removed: **{}**.",
        channel_id
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn clear_channels(ctx: Context<'_>) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    ctx.data()
        .app
        .clear_allowed_channels(guild_id.get())
        .await?;
    ctx.say("Allowed channel list cleared. Commands are available in all channels again.")
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn add_admin_role(
    ctx: Context<'_>,
    #[description = "Raw role id or role mention"] role: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let role_id = parse_snowflake(&role)?;
    ctx.data()
        .app
        .add_admin_role(guild_id.get(), role_id)
        .await?;
    ctx.say(format!("Admin role added: **{}**.", role_id))
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn remove_admin_role(
    ctx: Context<'_>,
    #[description = "Raw role id or role mention"] role: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let role_id = parse_snowflake(&role)?;
    ctx.data()
        .app
        .remove_admin_role(guild_id.get(), role_id)
        .await?;
    ctx.say(format!("Admin role removed: **{}**.", role_id))
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn sync_payments(ctx: Context<'_>) -> Result<()> {
    let _deferred = ctx.defer_or_broadcast().await?;
    ensure_admin(ctx).await?;
    let summary = ctx.data().app.poll_payments().await?;
    flush_dispatches_http(ctx.serenity_context().http.as_ref(), &summary.messages).await?;
    let warning_suffix = if summary.warnings.is_empty() {
        String::new()
    } else {
        format!(
            " Warnings: **{}** upstream issue(s).",
            summary.warnings.len()
        )
    };
    ctx.say(format!(
        "Payment sync completed. Processed **{}** incoming payment events.{}",
        summary.processed, warning_suffix
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn drink(
    ctx: Context<'_>,
    #[description = "Drink slug from /catalog"] item: String,
    #[description = "Optional tx reference or memo when mirroring a paid action"] tx_ref: Option<
        String,
    >,
) -> Result<()> {
    let _ = tx_ref;
    ensure_admin(ctx).await?;
    ensure_allowed_channel(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let outcome = ctx
        .data()
        .app
        .buy_test_drink(guild_id.get(), actor_from_ctx(ctx), &item)
        .await?;

    ctx.say(format!("**{}**\n{}", outcome.headline, outcome.body))
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn action(
    ctx: Context<'_>,
    #[description = "Action slug from /catalog"] item: String,
    #[description = "Optional target name"] target: Option<String>,
) -> Result<()> {
    ensure_admin(ctx).await?;
    ensure_allowed_channel(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let outcome = ctx
        .data()
        .app
        .trigger_test_action(guild_id.get(), actor_from_ctx(ctx), &item, target)
        .await?;

    ctx.say(format!("**{}**\n{}", outcome.headline, outcome.body))
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn set_stage(
    ctx: Context<'_>,
    #[description = "sober | warm | tipsy | buzzing | cooked | gone | hungover"] stage: String,
) -> Result<()> {
    ensure_admin(ctx).await?;
    ensure_allowed_channel(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let stage = parse_stage(&stage)?;
    let outcome = ctx
        .data()
        .app
        .set_stage(guild_id.get(), actor_from_ctx(ctx), stage)
        .await?;

    ctx.say(format!("**{}**\n{}", outcome.headline, outcome.body))
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn clear_context(ctx: Context<'_>) -> Result<()> {
    ensure_admin(ctx).await?;
    ensure_allowed_channel(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let outcome = ctx
        .data()
        .app
        .clear_context(guild_id.get(), ctx.channel_id().get(), actor_from_ctx(ctx))
        .await?;

    ctx.say(format!("**{}**\n{}", outcome.headline, outcome.body))
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn debug_memory(
    ctx: Context<'_>,
    #[description = "Prompt to inspect against the memory system"] prompt: String,
) -> Result<()> {
    let _deferred = ctx.defer_or_broadcast().await?;
    ensure_admin(ctx).await?;
    ensure_allowed_channel(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let report = ctx
        .data()
        .app
        .debug_memory_report(
            guild_id.get(),
            ctx.channel_id().get(),
            actor_from_ctx(ctx),
            &prompt,
        )
        .await?;

    ctx.say(report).await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn remember(
    ctx: Context<'_>,
    #[description = "Fact, running joke, or lore worth keeping"] fact: String,
) -> Result<()> {
    let _deferred = ctx.defer_or_broadcast().await?;
    ensure_allowed_channel(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx
        .data()
        .app
        .remember(guild_id.get(), actor_from_ctx(ctx), &fact)
        .await?;

    ctx.say(format!(
        "{} pockets the new memory. Stored memories: **{}**.",
        state.config.bot_name,
        state.memories.len()
    ))
    .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn sync_commands(ctx: Context<'_>) -> Result<()> {
    let _deferred = ctx.defer_or_broadcast().await?;
    ensure_admin(ctx).await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    poise::builtins::register_in_guild(
        ctx.serenity_context(),
        &ctx.framework().options().commands,
        guild_id,
    )
    .await?;
    ctx.say("Re-registered slash commands in this guild. Guild commands should appear almost immediately.")
        .await?;
    Ok(())
}

#[poise::command(slash_command, guild_only)]
async fn chat(
    ctx: Context<'_>,
    #[description = "Message to the tavern bot"] prompt: String,
) -> Result<()> {
    ensure_allowed_channel(ctx).await?;
    let _deferred = ctx.defer_or_broadcast().await?;
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let reply = ctx
        .data()
        .app
        .speak(
            guild_id.get(),
            ctx.channel_id().get(),
            actor_from_ctx(ctx),
            &prompt,
        )
        .await?;

    ctx.say(reply).await?;
    Ok(())
}

async fn event_handler(
    ctx: &serenity::Context,
    event: &serenity::FullEvent,
    framework: poise::FrameworkContext<'_, Data, Error>,
    data: &Data,
) -> Result<()> {
    match event {
        serenity::FullEvent::GuildCreate { guild, .. } => {
            if let Err(error) =
                poise::builtins::register_in_guild(ctx, &framework.options().commands, guild.id)
                    .await
            {
                warn!(
                    guild_id = guild.id.get(),
                    "failed to register guild commands on guild create: {error:#}"
                );
            }
        }
        serenity::FullEvent::Message { new_message } => {
            if new_message.author.bot {
                return Ok(());
            }
            let Some(guild_id) = new_message.guild_id else {
                return Ok(());
            };

            let current_user_id = ctx.cache.current_user().id;
            let is_mention = new_message
                .mentions
                .iter()
                .any(|user| user.id == current_user_id);
            let is_reply_to_bot = new_message
                .referenced_message
                .as_ref()
                .is_some_and(|message| message.author.id == current_user_id);
            let raw_prompt = if is_mention {
                strip_bot_mentions(&new_message.content, current_user_id)
            } else {
                new_message.content.trim().to_string()
            };
            if raw_prompt.trim().is_empty() || raw_prompt.starts_with('/') {
                return Ok(());
            }

            let actor = ActorRef {
                user_key: format!("discord:{}", new_message.author.id.get()),
                user_name: new_message.author.name.clone(),
            };
            if is_mention {
                let _typing = new_message.channel_id.start_typing(&ctx.http);
                match data
                    .app
                    .speak_from_mention(
                        guild_id.get(),
                        new_message.channel_id.get(),
                        actor,
                        &raw_prompt,
                    )
                    .await?
                {
                    MentionReplyOutcome::Reply(reply) => {
                        new_message.reply(ctx, reply).await?;
                    }
                    MentionReplyOutcome::Suppressed(reason) => {
                        info!(
                            guild_id = guild_id.get(),
                            channel_id = new_message.channel_id.get(),
                            user_id = new_message.author.id.get(),
                            "mention reply suppressed: {reason}"
                        );
                    }
                }
            } else {
                let looks_publicly_relevant = raw_prompt.contains('?')
                    || raw_prompt.to_ascii_lowercase().contains("drink")
                    || raw_prompt.to_ascii_lowercase().contains("story")
                    || raw_prompt.to_ascii_lowercase().contains("coin")
                    || is_reply_to_bot;
                let _typing =
                    looks_publicly_relevant.then(|| new_message.channel_id.start_typing(&ctx.http));
                match data
                    .app
                    .speak_public(
                        guild_id.get(),
                        new_message.channel_id.get(),
                        actor,
                        &raw_prompt,
                        &new_message.id.get().to_string(),
                        is_reply_to_bot,
                    )
                    .await?
                {
                    PublicReplyOutcome::Reply(reply) => {
                        new_message.reply(ctx, reply).await?;
                    }
                    PublicReplyOutcome::Suppressed(reason) => {
                        info!(
                            guild_id = guild_id.get(),
                            channel_id = new_message.channel_id.get(),
                            user_id = new_message.author.id.get(),
                            "public reply suppressed: {reason}"
                        );
                    }
                }
            }
        }
        _ => {}
    }

    Ok(())
}

fn actor_from_ctx(ctx: Context<'_>) -> ActorRef {
    ActorRef {
        user_key: format!("discord:{}", ctx.author().id.get()),
        user_name: ctx.author().name.clone(),
    }
}

async fn ensure_admin(ctx: Context<'_>) -> Result<()> {
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx.data().app.policy(guild_id.get()).await?;
    if has_admin_access(ctx, &state).await? {
        Ok(())
    } else {
        bail!("admin permission required for this command")
    }
}

async fn has_admin_access(ctx: Context<'_>, state: &GuildState) -> Result<bool> {
    let channel = match ctx.channel_id().to_channel(ctx).await? {
        serenity::Channel::Guild(channel) => channel,
        _ => return Ok(false),
    };
    let member_binding = ctx
        .author_member()
        .await
        .ok_or_else(|| anyhow!("guild member not available"))?;
    let member = member_binding.as_ref();

    let has_role = member.roles.iter().any(|role_id| {
        state
            .config
            .permissions
            .admin_role_ids
            .contains(&role_id.get())
    });
    let guild = ctx
        .guild()
        .ok_or_else(|| anyhow!("guild cache not available"))?;
    let permissions = guild.user_permissions_in(&channel, member);
    let is_owner = guild.owner_id == ctx.author().id;

    Ok(is_owner || permissions.administrator() || permissions.manage_guild() || has_role)
}

async fn ensure_allowed_channel(ctx: Context<'_>) -> Result<()> {
    let guild_id = ctx
        .guild_id()
        .ok_or_else(|| anyhow!("command must run in a guild"))?;
    let state = ctx.data().app.status(guild_id.get()).await?;
    if has_admin_access(ctx, &state).await? {
        return Ok(());
    }
    if ctx
        .data()
        .app
        .command_allowed_in_channel(&state, ctx.channel_id().get())
    {
        Ok(())
    } else {
        bail!(
            "this command is disabled in this channel; allowed channels: {}",
            list_ids(&state.config.permissions.allowed_channel_ids)
        )
    }
}

fn parse_stage(input: &str) -> Result<IntoxicationStage> {
    match input.trim().to_ascii_lowercase().as_str() {
        "sober" => Ok(IntoxicationStage::Sober),
        "warm" => Ok(IntoxicationStage::Warm),
        "tipsy" => Ok(IntoxicationStage::Tipsy),
        "buzzing" | "buzzed" => Ok(IntoxicationStage::Buzzing),
        "cooked" => Ok(IntoxicationStage::Cooked),
        "gone" => Ok(IntoxicationStage::Gone),
        "hungover" | "hangover" => Ok(IntoxicationStage::Hungover),
        _ => bail!("unknown stage; use sober, warm, tipsy, buzzing, cooked, gone, or hungover"),
    }
}

async fn flush_dispatches_http(http: &serenity::Http, messages: &[DispatchMessage]) -> Result<()> {
    for message in messages {
        if let Err(error) = serenity::ChannelId::new(message.channel_id)
            .say(http, &message.content)
            .await
        {
            error!(
                "failed to send dispatch to channel {}: {error:#}",
                message.channel_id
            );
        }
    }
    Ok(())
}

fn parse_snowflake(input: &str) -> Result<u64> {
    let digits = input
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        bail!("expected a channel or role id");
    }
    digits
        .parse::<u64>()
        .with_context(|| format!("invalid snowflake `{input}`"))
}

fn list_ids(values: &[u64]) -> String {
    if values.is_empty() {
        "all / none configured".to_string()
    } else {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn strip_bot_mentions(content: &str, bot_user_id: serenity::UserId) -> String {
    let mention_a = format!("<@{}>", bot_user_id.get());
    let mention_b = format!("<@!{}>", bot_user_id.get());
    content
        .replace(&mention_a, "")
        .replace(&mention_b, "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::strip_bot_mentions;
    use poise::serenity_prelude as serenity;

    #[test]
    fn mention_stripping_removes_both_mention_forms() {
        let user_id = serenity::UserId::new(42);
        assert_eq!(
            strip_bot_mentions("<@42> tell me a joke", user_id),
            "tell me a joke"
        );
        assert_eq!(
            strip_bot_mentions("<@!42> tavern status?", user_id),
            "tavern status?"
        );
    }
}

fn top_regulars_line(state: &GuildState) -> String {
    let mut items = state.regulars.values().collect::<Vec<_>>();
    items.sort_by(|a, b| b.chaos_score.cmp(&a.chaos_score));
    if items.is_empty() {
        "none yet".to_string()
    } else {
        items
            .into_iter()
            .take(3)
            .map(|entry| entry.display_name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
