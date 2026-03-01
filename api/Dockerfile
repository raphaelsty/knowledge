# =============================================================================
# Knowledge API Dockerfile
# =============================================================================
# Unified API: search + data + events + ingest in a single binary.
#
# Build:
#   docker build -t knowledge-api api/
#
# Run:
#   docker run -p 8080:8080 \
#     -e DATABASE_URL=postgresql://knowledge:knowledge@host:5432/knowledge \
#     -v ./multi-vector-database/knowledge:/data/indices/knowledge \
#     -v models:/models \
#     knowledge-api --model lightonai/answerai-colbert-small-v1-onnx --int8
# =============================================================================

# =============================================================================
# Chef stage — install cargo-chef for dependency caching
# =============================================================================
FROM debian:bookworm-slim AS chef

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and cargo-chef
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install cargo-chef

# Download ONNX Runtime CPU (architecture-aware)
ARG ORT_VERSION=1.23.0
RUN mkdir -p /opt/ort_cpu && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
    ORT_ARCH="aarch64"; \
    else \
    ORT_ARCH="x64"; \
    fi && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz && \
    cp -r onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}/lib/* /opt/ort_cpu/ && \
    rm -rf onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION} onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz

ENV ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.${ORT_VERSION}
ENV LD_LIBRARY_PATH=/opt/ort_cpu:${LD_LIBRARY_PATH}

WORKDIR /app

# =============================================================================
# Planner stage — create dependency recipe
# =============================================================================
FROM chef AS planner

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo chef prepare --recipe-path recipe.json

# =============================================================================
# Builder stage — build dependencies (cached) then application
# =============================================================================
FROM chef AS builder

# Build dependencies first (cached if recipe.json unchanged)
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json --features "openblas,model"

# Build the application
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --features "openblas,model"

# =============================================================================
# Runtime stage
# =============================================================================
FROM debian:bookworm-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash knowledge

# Create directories
RUN mkdir -p /data/indices /models /opt/ort_cpu && chown -R knowledge:knowledge /data /models

# Copy ONNX Runtime libraries from builder
COPY --from=builder /opt/ort_cpu /opt/ort_cpu

# Copy binary from builder
COPY --from=builder /app/target/release/knowledge-api /usr/local/bin/knowledge-api

# Copy entrypoint script
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Set library paths
ARG ORT_VERSION=1.23.0
ENV ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.${ORT_VERSION}
ENV LD_LIBRARY_PATH=/opt/ort_cpu:${LD_LIBRARY_PATH}

USER knowledge

EXPOSE 8080

ENV RUST_LOG=info
ENV OPENBLAS_NUM_THREADS=1

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f --max-time 5 http://localhost:8080/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices", "--parallel", "8", "--batch-size", "2", "--model-pool-size", "1"]
