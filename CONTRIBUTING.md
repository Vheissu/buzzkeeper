# Contributing to Buzzkeeper

Thanks for contributing to Buzzkeeper.

Buzzkeeper is an open-source Rust Discord bot for Hive communities. Contributions are welcome across code, docs, tests, bug fixes, performance improvements, deployment support, and new tavern features.

## Ground Rules

- Keep changes practical and shippable.
- Prefer small, focused pull requests over broad refactors.
- Do not break existing bot behavior without a clear reason.
- If you add or change behavior, add or update tests where it makes sense.
- If you change commands, config, deployment, or setup flow, update the docs too.

## AI-Assisted Contributions

AI-generated or AI-assisted contributions are welcome.

That includes code written with tools like Codex, ChatGPT, Claude, Gemini, Copilot, or similar systems.

The bar is simple:

- the change must be understandable
- the code must fit the project
- the tests must pass
- the feature or fix must actually work

If you use AI, review the output like a human maintainer. Do not submit generated code blindly.

## Before You Start

Make sure you can build and test the project locally.

Typical local workflow:

```bash
cargo fmt
cargo test
cargo run
```

If you are working on local LLM behavior, Ollama with `qwen3:4b` is the default setup used by the project.

## Development Setup

1. Fork the repository.
2. Clone your fork.
3. Copy `.env.example` to `.env`.
4. Fill in the values you need for local development.
5. Run the bot locally with Rust tooling.

Example:

```bash
cp .env.example .env
cargo run
```

Useful local settings:

```env
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

If you want to exercise Discord behavior, you will also need:

```env
DISCORD_TOKEN=your_bot_token
```

For payment polling and chain integration, configure the Hive account and other environment values described in the main [README](/Users/dwayne/Code/hive-discord-bot/README.md).

## What to Contribute

Good contribution areas include:

- bug fixes
- test coverage improvements
- payment ingestion improvements
- Hive Engine handling improvements
- moderation and admin controls
- public tavern behavior
- better docs
- deployment improvements
- memory and conversation quality
- model/provider integrations

## Coding Expectations

- Follow the existing code style.
- Run `cargo fmt` before opening a PR.
- Run `cargo test` and make sure it passes.
- Prefer straightforward code over clever code.
- Keep dependencies justified.
- Avoid unrelated drive-by changes in the same PR.

If you touch persistence or config models, preserve backward compatibility where possible. Existing local state files should not break without a migration path.

## Tests

Tests matter here because the bot has several fragile edges:

- model output sanitization
- payment memo parsing
- transfer ingestion
- cooldowns and quiet hours
- persistent state compatibility

At minimum:

- add tests for new parsing or sanitization logic
- add tests for bug fixes when practical
- do not remove coverage without a reason

Before opening a PR, run:

```bash
cargo fmt --check
cargo test
```

If you changed something that affects compile-time behavior or features, also run:

```bash
cargo check
```

## Pull Requests

When opening a PR:

- explain what changed
- explain why it changed
- note any behavior changes
- note any new config or env vars
- note any docs updates
- include test coverage details

If the PR fixes a bug, include:

- how to reproduce the issue
- what the fix does
- how you verified it

Screenshots or logs are helpful for Discord-facing behavior.

## Commit Messages

Keep commit messages clear and direct.

Good examples:

- `Fix Qwen reply sanitization for echoed prompts`
- `Ignore unknown incoming transfers`
- `Add deployment guide for Railway and Fly`

## Documentation Changes

If your change affects users or operators, update docs in the same PR.

This includes changes to:

- commands
- environment variables
- Discord setup
- payment setup
- deployment flow
- moderation behavior
- public tavern behavior

## Things to Avoid

- submitting untested AI output
- adding large dependencies without need
- mixing formatting-only churn with real logic changes
- breaking state compatibility casually
- changing public behavior without documenting it

## Reporting Bugs

If you are not ready to open a PR, bug reports are still useful.

A good report includes:

- what you expected
- what happened instead
- steps to reproduce
- logs or screenshots
- config details if relevant

## Feature Ideas

Feature proposals are welcome, especially around:

- custom drinks and actions
- better long-term memory
- richer public tavern behavior
- safer moderation controls
- better Hive/Hive Engine UX
- deployment and observability

If the feature is large, open an issue first so the shape of the work can be discussed before implementation.

## Code of Conduct

Be respectful and useful.

Good contributions come from clear communication, solid verification, and a willingness to improve the result when review finds problems.
