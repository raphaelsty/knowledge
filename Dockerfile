# Stage 1: Build Rust binaries
FROM rust:1.82-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo install next-plaid-api --features model
COPY indexer /build/indexer
RUN cargo build --release --manifest-path /build/indexer/Cargo.toml

# Stage 2: Runtime (serves the pre-built index)
FROM debian:bookworm-slim

COPY --from=builder /usr/local/cargo/bin/next-plaid-api /usr/local/bin/

WORKDIR /code

COPY indices /code/indices
COPY docs /code/docs

CMD ["next-plaid-api", "--index-dir", "indices", "--model", "lightonai/answerai-colbert-small-v1-onnx", "--int8", "--port", "8080"]
