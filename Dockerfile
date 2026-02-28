# =============================================================================
# Knowledge Search API Dockerfile
# =============================================================================
# Based on next-plaid-api/Dockerfile from lategrep.
# Uses cargo install from crates.io instead of building from the workspace.
#
# Build:   docker compose up -d search-api
# Run:     docker run -p 8080:8080 -v ./multi-vector-database:/data/indices/knowledge knowledge-search-api
# =============================================================================

# =============================================================================
# Builder stage - Install dependencies and build from crates.io
# =============================================================================
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

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

# Install next-plaid-api from GitHub
RUN cargo install next-plaid-api --git https://github.com/lightonai/next-plaid.git --features "openblas,model"

# =============================================================================
# Runtime stage - CPU with model support
# =============================================================================
FROM debian:bookworm-slim AS runtime-cpu

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash nextplaid

# Create directories for indices, models, and ONNX Runtime
RUN mkdir -p /data/indices /models /opt/ort_cpu && chown -R nextplaid:nextplaid /data /models

# Copy ONNX Runtime CPU libraries from builder
COPY --from=builder /opt/ort_cpu /opt/ort_cpu

# Copy binary from builder
COPY --from=builder /root/.cargo/bin/next-plaid-api /usr/local/bin/next-plaid-api

# Copy entrypoint script
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Set library paths for ONNX Runtime
ARG ORT_VERSION=1.23.0
ENV ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.${ORT_VERSION}
ENV LD_LIBRARY_PATH=/opt/ort_cpu:${LD_LIBRARY_PATH}

# Switch to non-root user
USER nextplaid

# Expose API port
EXPOSE 8080

# Default environment variables
ENV RUST_LOG=info
ENV INDEX_DIR=/data/indices
# Prevent OpenBLAS thread explosion when used alongside rayon parallelism
ENV OPENBLAS_NUM_THREADS=1

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f --max-time 5 http://localhost:8080/health || exit 1

# Run the API via entrypoint script (handles model download)
ENTRYPOINT ["docker-entrypoint.sh"]
# CPU defaults: 16 parallel sessions, batch size 4, 1 model pool worker
CMD ["--host", "0.0.0.0", "--port", "8080", "--index-dir", "/data/indices", "--parallel", "16", "--batch-size", "4", "--model-pool-size", "1"]
