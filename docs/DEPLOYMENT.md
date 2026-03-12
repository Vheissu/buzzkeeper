# Deployment Guide

This guide covers how to deploy Buzzkeeper locally, with Docker, and on common cloud platforms.

## Before You Deploy

Buzzkeeper is currently a single-process application with JSON file storage.

That means production deployments should follow these rules:

- run exactly one replica
- mount persistent storage
- point `STORAGE_PATH` at that persistent location
- do not run multiple instances against separate local files and expect shared state

Recommended production path:

```env
STORAGE_PATH=/data/tavern-state.json
```

## Required Environment Variables

At minimum:

```env
DISCORD_TOKEN=...
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen3:4b
STORAGE_PATH=/data/tavern-state.json
```

If using local Ollama on the same machine:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

If using a remote provider instead of Ollama, configure the matching API key and model in `.env`.

## Discord Production Checklist

Before deploying:

1. create the bot in the Discord Developer Portal
2. copy the bot token into `DISCORD_TOKEN`
3. enable `Message Content Intent`
4. invite the bot with:
   - `View Channels`
   - `Send Messages`
   - `Read Message History`
   - `Use Slash Commands`

## Docker

This repo includes a production-oriented [Dockerfile](../Dockerfile).

### Build

```bash
docker build -t buzzkeeper:latest .
```

### Run

```bash
docker run --rm \
  --name buzzkeeper \
  --env-file .env \
  -e STORAGE_PATH=/data/tavern-state.json \
  -v $(pwd)/data:/data \
  buzzkeeper:latest
```

If you are using Ollama on the same host, you need the container to reach it. The exact networking depends on your OS:

- on macOS/Windows Docker Desktop, `host.docker.internal` is usually easiest
- on Linux, use host networking or expose Ollama on a reachable interface

Example:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## Local VM / VPS

Buzzkeeper works well on a small Linux VM.

Recommended:

- 1 vCPU
- 1 to 2 GB RAM if using a remote LLM
- more RAM if running Ollama on the same machine
- persistent disk mounted at `/data`

Typical flow:

1. install Rust or use Docker
2. clone the repo
3. copy `.env.example` to `.env`
4. set `STORAGE_PATH=/data/tavern-state.json`
5. run with a process supervisor

Example using `tmux`:

```bash
tmux new -s buzzkeeper
cargo run --release
```

For a cleaner service deployment, use `systemd`.

## Railway

Railway is a reasonable option if you want a simple single-service deploy.

Recommended Railway setup:

- deploy from this repo using the included Dockerfile
- add a persistent volume mounted at `/data`
- set `STORAGE_PATH=/data/tavern-state.json`
- run only one replica

Environment variables to set in Railway:

- `DISCORD_TOKEN`
- `DEFAULT_LLM_PROVIDER`
- `DEFAULT_LLM_MODEL`
- `STORAGE_PATH`
- whichever provider-specific variables you use

Two deployment patterns:

### Railway + Remote Provider

Use OpenAI, Anthropic, or Gemini as the model backend. This is the simplest hosted setup.

### Railway + External Ollama

Run Ollama elsewhere and point Railway at it with:

```env
OLLAMA_BASE_URL=https://your-ollama-endpoint
```

Do not expect a standard Railway container to run `qwen3:4b` comfortably unless you have explicitly provisioned the right machine shape and understand the tradeoffs.

## Fly.io

Fly.io is also a good fit for a single always-on bot.

Recommended Fly setup:

- deploy the Docker image
- attach a volume mounted at `/data`
- set `STORAGE_PATH=/data/tavern-state.json`
- keep instance count at 1

If you use Ollama remotely, set `OLLAMA_BASE_URL` to the remote endpoint.

If you use a hosted provider, just configure the provider API key and model.

## Render

Render can also host Buzzkeeper, but the same rule applies:

- one instance
- persistent disk
- `STORAGE_PATH` on that disk

If persistent storage is not available on the plan you choose, use a different platform or move state into a real database first.

## Recommended Production Configs

### Cheapest Practical Hosted Setup

- Platform: Railway or Fly.io
- LLM: OpenAI / Anthropic / Gemini
- Storage: mounted persistent disk
- `STORAGE_PATH=/data/tavern-state.json`

### Cheapest Local-Control Setup

- Platform: small VPS
- LLM: local Ollama with `qwen3:4b`
- Storage: local disk
- process manager: `systemd` or Docker

### Most Stable Current Setup

- Platform: single VPS or single Fly.io machine
- LLM: hosted provider or a separate Ollama box
- Storage: mounted disk
- one replica only

## Upgrades and Restarts

Because Buzzkeeper stores state in JSON, safe upgrade flow is:

1. stop the process
2. back up the JSON state file
3. pull latest code
4. run `cargo test`
5. start the new version

If you use Docker:

1. keep the volume mounted
2. rebuild image
3. replace the container

## Operational Notes

### Payment Polling

Buzzkeeper polls Hive and Hive Engine at the configured interval:

```env
PAYMENT_POLL_INTERVAL_SECS=15
```

Lower values mean faster reactions but more API traffic.

### Logging

Set:

```env
RUST_LOG=info
```

or, for more noise:

```env
RUST_LOG=debug
```

### Message Content Intent

If public tavern mode is enabled but the bot never responds to normal messages, check Discord first. Missing `Message Content Intent` is the most common cause.

## Things Buzzkeeper Does Not Yet Do

- sign outbound Hive transactions
- hold or use Hive private keys
- support safe multi-instance shared state
- enforce Hive Engine issuer at ingest time

If you add outbound chain operations later, revisit the deployment model and key-management docs before going live.
