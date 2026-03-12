# Buzzkeeper

Buzzkeeper is a Rust Discord bot for Hive communities. It acts like a tavern character inside a server: people can talk to it, tip it, buy it drinks with HIVE, HBD, or Hive Engine tokens, push it into different mood states, and turn a Discord channel into a noisy on-chain pub.

Repository: [github.com/Vheissu/buzzkeeper](https://github.com/Vheissu/buzzkeeper)

## What Buzzkeeper Does

- Runs as a Discord bot using `poise` and `serenity`
- Stores persistent guild state in a local JSON file
- Tracks tavern mood, intoxication, hangovers, party meter, regulars, and memories
- Supports local and hosted LLM backends
- Defaults to Ollama with `qwen3:4b`
- Watches a Hive account for incoming HIVE, HBD, and Hive Engine transfers
- Turns recognized payments into drinks, actions, or explicit tips
- Supports public tavern mode so the bot can act like a participating server member
- Supports admin roles, channel policy, quiet hours, cooldowns, and payment announcement channels

## Current Architecture

Buzzkeeper is currently a single-process bot with a single JSON state file.

That means:

- run one bot instance per deployed environment
- mount persistent storage for `STORAGE_PATH`
- do not horizontally scale multiple replicas against the same JSON file

If you want multi-instance scaling later, the storage layer should move from JSON to a real shared database.

## Requirements

- Rust 1.85+ with Cargo
- A Discord bot application and token
- A Hive account to receive transfers
- An LLM backend

Recommended local stack:

- Ollama
- `qwen3:4b`

## Quick Start

1. Clone the repo:

```bash
git clone https://github.com/Vheissu/buzzkeeper.git
cd buzzkeeper
```

2. Copy the example environment file:

```bash
cp .env.example .env
```

3. Fill in `DISCORD_TOKEN`

4. Install the default local model:

```bash
ollama pull qwen3:4b
```

5. Run the test suite:

```bash
cargo test
```

6. Start the bot:

```bash
cargo run
```

## Discord Setup

### 1. Create the Discord Application

Go to the [Discord Developer Portal](https://discord.com/developers/applications), create a new application, then create a bot user for it.

### 2. Copy the Bot Token

In the `Bot` tab:

- create the bot if you have not already
- copy the token
- put it in `.env` as `DISCORD_TOKEN`

### 3. Enable Required Intents

Buzzkeeper uses:

- standard non-privileged intents
- `Message Content Intent`

`Message Content Intent` is required if you want:

- public tavern mode
- non-mention chatter detection
- reply-chain continuation without explicit mention every time

In the Discord Developer Portal:

1. open your application
2. go to `Bot`
3. enable `Message Content Intent`
4. save

### 4. Invite the Bot to Your Server

In `OAuth2` -> `URL Generator`:

- select scopes:
  - `bot`
  - `applications.commands`
- select permissions:
  - `View Channels`
  - `Send Messages`
  - `Read Message History`
  - `Use Slash Commands`

Optional but useful:

- `Embed Links`
- `Attach Files`

Not recommended unless you truly need them:

- `Mention Everyone`
- `Administrator`

### 5. Wait for Slash Commands to Appear

Buzzkeeper currently registers commands globally on startup. New or changed slash commands can take a little while to show up in Discord.

If you restart the bot after pulling updates, give Discord a minute or two to reflect the latest command set.

## Hive Account and Key Setup

### Important: Buzzkeeper Does Not Currently Need Your Hive Private Key

Buzzkeeper does not sign on-chain transactions in the current version.

It only:

- watches inbound transfers to a Hive account
- parses matching transfers
- reacts inside Discord

So today you do **not** paste a Hive private key into `.env` or into Discord.

### What You Actually Need

You need a dedicated Hive account that can receive:

- HIVE
- HBD
- Hive Engine tokens

Recommended approach:

- create a dedicated account just for Buzzkeeper
- do not reuse your personal account if you are testing apps and transfers constantly
- manage that account's keys in Hive Keychain or your normal Hive wallet flow
- keep the active key secure and offline from the bot until you build outbound transfer support

### Suggested Key Management Practice

Use a dedicated Hive account and store its keys in a wallet or key manager you trust. The bot only needs the account name today.

Example:

- account name: `buzzkeeper.bot`
- configure Buzzkeeper to watch that account
- send users' tavern payments to that account

### What Happens With Unknown Transfers

Buzzkeeper now ignores unknown memos and unrelated transfer noise.

It will react only to:

- recognized memos like `drink:first-round`
- recognized memos like `action:karaoke`
- explicit `tip` / `donation` / `house`
- exact price matches when there is no memo

Random app memos and unrelated tiny transfers are ignored.

## LLM Setup

### Default: Ollama + Qwen 3 4B

Buzzkeeper defaults to:

- provider: `ollama`
- model: `qwen3:4b`

That makes local development cheap and predictable.

Install and run:

```bash
ollama pull qwen3:4b
```

Buzzkeeper talks to:

```text
http://127.0.0.1:11434
```

by default.

### Supported Providers

- Ollama
- OpenAI
- Anthropic
- Google Gemini
- Offline fallback

### Provider Environment Variables

See [.env.example](.env.example).

Common examples:

#### Ollama

```env
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

#### OpenAI

```env
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

#### Anthropic

```env
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

#### Google Gemini

```env
DEFAULT_LLM_PROVIDER=google
DEFAULT_LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-2.5-flash
```

## Local Development

### Environment

Minimum `.env` for local Discord + local Ollama:

```env
DISCORD_TOKEN=your_discord_bot_token
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
STORAGE_PATH=data/tavern-state.json
```

### Local Commands

Format:

```bash
cargo fmt
```

Run tests:

```bash
cargo test
```

Build check:

```bash
cargo check
```

Run bot:

```bash
cargo run
```

### Local Ollama Smoke Test

If you want to verify the model path before running the bot:

```bash
curl -s http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3:4b",
    "messages":[
      {"role":"system","content":"You are a tavern bot. Return only valid JSON with a reply field."},
      {"role":"user","content":"Say hello from the tavern in one sentence."}
    ],
    "stream":false,
    "think":false,
    "format":{
      "type":"object",
      "properties":{"reply":{"type":"string"}},
      "required":["reply"]
    }
  }'
```

### Local Discord Testing

Use a private test server first.

Recommended test flow:

1. `/setup`
2. `/status`
3. `/catalog`
4. `/set_reply_behavior`
5. `/set_public_tavern`
6. `@Buzzkeeper hello`
7. reply directly to the bot
8. test `/chat`
9. test a transfer and `/sync_payments`

## Payment Setup

Buzzkeeper watches one Hive account per guild.

Users trigger tavern behavior by sending the configured asset to that account.

### House Asset

The configured house asset can be:

- HIVE
- HBD
- a Hive Engine token symbol

You set that asset with `/setup`.

### Recommended Setup Example

For HIVE:

```text
/setup bot_name:Buzzkeeper theme:tavern llm_provider:ollama llm_model:qwen3:4b asset_ledger:hive asset_symbol:HIVE payment_account:buzzkeeper.bot
```

For a Hive Engine token:

```text
/setup bot_name:Buzzkeeper theme:tavern llm_provider:ollama llm_model:qwen3:4b asset_ledger:hive-engine asset_symbol:LEO asset_issuer:leofinance payment_account:buzzkeeper.bot
```

Then set the Discord channel where payment reactions should be posted:

```text
/set_payment_channel channel:#buzzkeeper-bar
```

Or do it in one shot:

```text
/set_payment_account account:buzzkeeper.bot payment_channel:#buzzkeeper-bar
```

### Payment Memos

Supported memo formats:

- `drink:<slug>`
- `action:<slug>`
- `action:<slug> <target>`
- `tip`
- `donation`
- `house`

Examples:

- `drink:first-round`
- `drink:goblin-grog`
- `action:karaoke`
- `action:roast @beggars`
- `tip`

### No-Memo Behavior

If a payment has **no memo**:

- Buzzkeeper checks whether the amount exactly matches a drink price
- then checks whether it exactly matches an action price
- otherwise it ignores the transfer

### Unknown Memo Behavior

If a payment has an unknown memo:

- Buzzkeeper ignores it
- no Discord message is posted

This is intentional so shared app/testing accounts do not spam your server.

### Payment Polling

Buzzkeeper polls for:

- native Hive transfers
- Hive Engine transfers

You can force an immediate refresh with:

```text
/sync_payments
```

### Current Hive Limitations

Be aware of current behavior:

- Buzzkeeper only ingests incoming transfers
- it does not broadcast outbound on-chain operations
- it does not use your Hive private keys today
- Hive Engine matching is currently based on ledger + symbol; issuer storage exists, but strict issuer enforcement is not yet implemented in the matching logic

## Public Tavern Mode

Buzzkeeper has two conversation modes:

### Mention / Direct Mode

People use:

- `@Buzzkeeper ...`
- `/chat ...`

### Public Tavern Mode

Buzzkeeper can also behave like a public participating member in allowed channels.

When enabled, it can:

- react to normal channel chatter
- continue conversations when users reply to one of its messages
- use shared channel context instead of treating every message as a standalone request

Enable it:

```text
/set_public_tavern enabled:true ambient_reply_chance_pct:25 ambient_cooldown_secs:8
```

Recommended companion settings:

```text
/set_reply_behavior mention_enabled:true mention_cooldown_secs:2 chat_cooldown_secs:1
```

### Channel Scope

Public tavern mode is guild-wide, but actual usable channels are controlled by the allowed-channel policy.

To keep Buzzkeeper only in one channel:

```text
/clear_channels
/allow_channel channel:#buzzkeeper-bar
```

If the allowed-channel list is empty, Buzzkeeper treats that as all channels allowed.

## Admin and Moderation Model

Admin commands require one of:

- Discord `Manage Server`
- a configured admin role in Buzzkeeper

Grant a bot admin role:

```text
/add_admin_role role:@Mods
```

Restrict command channels:

```text
/allow_channel channel:#buzzkeeper-bar
/disallow_channel channel:#off-topic
/clear_channels
```

Set quiet hours:

```text
/set_quiet_hours enabled:true start_hour_utc:22 end_hour_utc:8
```

## Command Reference

### Setup and Status

- `/setup`
  - configures bot name, theme, provider, model, house asset, and optional payment account
- `/status`
  - shows stage, persona, party meter, asset, latest event, and public tavern state
- `/catalog`
  - lists drinks and actions for the current guild
- `/policy`
  - shows admin, payment, cooldown, public tavern, and channel policy

### Payment and Economy

- `/set_payment_account`
  - sets the watched Hive account and optional payment announcement channel
- `/set_payment_channel`
  - sets the Discord channel for payment reactions
- `/sync_payments`
  - forces an immediate poll of Hive and Hive Engine transfers

### Conversation and Personality

- `/set_reply_behavior`
  - controls mention replies and `/chat` cooldowns
- `/set_public_tavern`
  - enables or disables public tavern mode, ambient chance, and ambient cooldown
- `/set_system_prompt`
  - appends additional guild-specific persona instructions
- `/clear_system_prompt`
  - removes custom guild prompt additions
- `/chat`
  - direct slash-command conversation with the bot
- `/remember`
  - saves a lore note or recurring joke into persistent memory

### Channel and Admin Policy

- `/allow_channel`
  - add an allowed channel
- `/disallow_channel`
  - remove an allowed channel
- `/clear_channels`
  - clear the channel restriction list
- `/add_admin_role`
  - add a role that can administer the bot
- `/remove_admin_role`
  - remove a role from the bot admin list
- `/set_quiet_hours`
  - define UTC quiet hours

### Direct Tavern Actions

- `/drink`
  - simulate buying the bot a drink in Discord
- `/action`
  - trigger a tavern action without waiting for on-chain payment

## Storage

Buzzkeeper stores state in one JSON file.

Default:

```text
data/tavern-state.json
```

Override it with:

```env
STORAGE_PATH=/absolute/path/to/tavern-state.json
```

Recommended production path:

```env
STORAGE_PATH=/data/tavern-state.json
```

## Deploying Buzzkeeper

Deployment guidance lives in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

Short version:

- deploy one instance only
- mount persistent storage
- set `STORAGE_PATH` to that mounted path
- keep outbound internet access enabled
- enable `Message Content Intent` in Discord

This repo now includes:

- [Dockerfile](Dockerfile)
- [.dockerignore](.dockerignore)

## Troubleshooting

### `DisallowedGatewayIntents`

Enable `Message Content Intent` in the Discord Developer Portal and restart the bot.

### Slash Commands Not Appearing

Buzzkeeper registers commands globally on startup. Wait a minute or two after restart, or reopen Discord.

### Old State File Fails to Load

Pull the latest version and restart. Backward-compatible defaults now exist for the public tavern fields.

### Bot Feels Silent

Check:

- `/policy`
- allowed channels
- quiet hours
- public tavern mode
- mention cooldown
- ambient cooldown

### No Payment Events

Check:

- watched Hive account is correct
- house asset is configured correctly
- memo matches expected format
- payment channel is set
- `/sync_payments`

## Roadmap

- stricter Hive Engine issuer enforcement
- custom admin-defined drinks and actions
- timezone-aware quiet hours
- richer server events and quests
- outbound Hive operations if the project later needs them

## License and Contribution

If you are open sourcing Buzzkeeper, add the license file and contribution rules you want before publishing broadly. The codebase is ready for that next step, but the repository currently does not include a license file in this workspace.
