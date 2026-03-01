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

WORKDIR /build
COPY . .
RUN cargo build --release --features openblas

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime libraries
COPY --from=builder /opt/ort_cpu /opt/ort_cpu
ARG ORT_VERSION=1.23.0
ENV ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.${ORT_VERSION}
ENV LD_LIBRARY_PATH=/opt/ort_cpu:${LD_LIBRARY_PATH}

COPY --from=builder /build/target/release/ingest-api /usr/local/bin/ingest-api

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=2 \
    CMD curl -f --max-time 5 http://localhost:3003/health || exit 1

CMD ["ingest-api"]
