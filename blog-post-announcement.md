# Introducing Buzzkeeper: A Hive-Powered Tavern Bot for Discord

We are open sourcing **Buzzkeeper**, a Discord bot built for Hive communities that want something more alive than a static tip bot.

Buzzkeeper is a tavern character. You can talk to it, push it into different moods, buy it drinks, trigger server actions, feed it memories, and wire all of that to **HIVE**, **HBD**, or **Hive Engine tokens**. The goal was simple: make a bot that feels like part of the server instead of a vending machine with slash commands.

The repository is now live at:

[https://github.com/Vheissu/buzzkeeper](https://github.com/Vheissu/buzzkeeper)

## Why We Built It

Most crypto bots inside Discord are transactional. They move numbers around, maybe show balances, maybe do one or two utility actions, and that is about it.

That was never the interesting version.

The interesting version was a bot that becomes a social object in the room.

Something that:

- people can bait into funny responses
- communities can theme around their own token
- remembers regulars and recurring jokes
- reacts when the whole server piles on
- turns on-chain payments into visible social moments

That is the niche Buzzkeeper is trying to fill.

## What Buzzkeeper Does

Buzzkeeper is a Rust Discord bot with:

- persistent guild state
- tavern mood and intoxication stages
- configurable house asset using HIVE, HBD, or Hive Engine tokens
- on-chain transfer polling for Hive and Hive Engine
- drink and action catalogs
- memory and regular tracking
- admin role and channel controls
- quiet hours, cooldowns, and public tavern mode
- support for OpenAI, Anthropic, Google, Ollama, and offline fallback

By default, Buzzkeeper runs well with **Ollama + qwen3:4b**, which keeps local hosting cheap and easy.

## On-Chain Interaction

One of the core ideas behind Buzzkeeper is that social interaction should connect back to Hive without turning into a gambling product or a compliance mess.

Users can send a transfer to the configured bot account with memos like:

- `drink:first-round`
- `action:karaoke`
- `action:roast @someone`
- `tip`

Buzzkeeper watches the account, ingests recognized transfers, and turns them into public Discord events.

That means a community can make its own token part of the server culture instead of just a number on a market page.

## Designed for Server Personality

Buzzkeeper is not locked to one exact vibe.

Servers can shape it through:

- bot naming
- theme skin
- chosen house asset
- custom system prompt additions
- public tavern mode
- channel policy

You can run it as:

- a fantasy tavern
- a potion shop
- a late-night coffeehouse
- a cyber dive bar

Same mechanics, different identity.

## Public Tavern Mode

The latest iteration moves Buzzkeeper closer to the original idea: a **public participating server member**.

Instead of only replying to slash commands, Buzzkeeper can now:

- reply to direct mentions
- continue conversations when users reply to one of its messages
- join public channel chatter when public tavern mode is enabled
- maintain short conversation context in-channel

That matters because a social bot dies quickly if it feels like a kiosk.

The point is not just to answer prompts. The point is to feel present.

## What We Learned While Building It

Running local models inside a Discord bot sounds straightforward until you actually do it.

In practice, a lot of the hard work is not the Discord integration or the payment polling. It is output control.

Small local models can:

- leak reasoning
- repeat system instructions
- echo the user back
- drift into meta-analysis instead of speaking in character

Buzzkeeper now defends against that by:

- using Ollama chat mode
- forcing structured output for Ollama responses
- extracting only the final reply payload
- rejecting obvious prompt leaks and parroting behavior

That work mattered more than expected.

## Current State

Buzzkeeper is already useful, but it is still early.

The current version is best described as:

**a strong open-source MVP with real personality, real on-chain ingestion, and clear room to evolve**

What it already does well:

- local development with Ollama
- Discord command/admin flow
- payment-based drinks and actions
- persistent server personality
- public tavern interactions

What still needs more work over time:

- stricter Hive Engine token issuer enforcement
- more custom drink/action authoring
- richer ambient behaviors
- better long-term memory tooling
- a database-backed storage layer for multi-instance deployments

## Who This Is For

Buzzkeeper makes the most sense for:

- Hive communities that already live in Discord
- tokenized projects that want a social layer
- communities that want a bot with actual server character
- builders who want a Rust base for a Hive-native Discord bot

If you want a perfectly neutral financial utility bot, this is probably not the right project.

If you want a server character that people can tip, tease, shape, and turn into part of the culture, this is exactly the direction.

## Open Source

The code is available now:

[https://github.com/Vheissu/buzzkeeper](https://github.com/Vheissu/buzzkeeper)

If you want to run it yourself, the repository includes:

- local setup instructions
- Discord bot setup guidance
- payment setup guidance
- Docker support
- cloud deployment notes

## Final Thought

Hive has always been strongest when it feels social, not just financial.

Buzzkeeper leans into that.

It turns a Discord server into a place where:

- transfers become events
- inside jokes become memory
- regulars become part of the lore
- the bot becomes a participant instead of a utility endpoint

That is the real experiment here.

If you run it, fork it, or build on top of it, I want to see what kind of taverns people make.
