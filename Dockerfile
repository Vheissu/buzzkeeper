FROM rust:1.85-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN useradd --create-home --uid 10001 buzzkeeper

WORKDIR /app

COPY --from=builder /app/target/release/hive-discord-bot /usr/local/bin/buzzkeeper

RUN mkdir -p /data && chown -R buzzkeeper:buzzkeeper /data /app

USER buzzkeeper

ENV STORAGE_PATH=/data/tavern-state.json
ENV RUST_LOG=info

CMD ["buzzkeeper"]
