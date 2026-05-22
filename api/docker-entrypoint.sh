#!/bin/bash
# =============================================================================
# Knowledge API Docker Entrypoint
# =============================================================================
# Handles HuggingFace model downloads and passes arguments to the API.
#
# If --model contains a HuggingFace model ID (org/model format), it will be
# automatically downloaded to /models/ before starting the API.
#
# Environment variables:
#   HF_TOKEN     HuggingFace token for private models
#   MODELS_DIR   Directory to store downloaded models (default: /models)
#   DATABASE_URL PostgreSQL connection string (enables data/events/ingest)
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

is_hf_model_id() {
    [[ "$1" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]
}

download_hf_file() {
    local repo_id="$1" filename="$2" dest_path="$3" token="$4"
    local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"

    log_info "Downloading ${filename}..."

    local curl_opts="-fSL --progress-bar"
    [ -n "$token" ] && curl_opts="$curl_opts -H \"Authorization: Bearer ${token}\""

    if eval curl $curl_opts -o "${dest_path}/${filename}" "$url"; then
        log_info "Downloaded ${filename}"
        return 0
    else
        log_error "Failed to download ${filename}"
        return 1
    fi
}

download_model() {
    local model_path="$1" repo_id="$2" token="${3:-}" use_int8="${4:-false}"

    log_info "Downloading model from HuggingFace: ${repo_id}"
    mkdir -p "$model_path"

    local files=("tokenizer.json" "config_sentence_transformers.json" "onnx_config.json")
    if [ "$use_int8" = "true" ]; then
        files+=("model_int8.onnx")
        log_info "Downloading INT8 quantized model"
    else
        files+=("model.onnx")
        log_info "Downloading FP32 model"
    fi

    local failed=0
    for file in "${files[@]}"; do
        download_hf_file "$repo_id" "$file" "$model_path" "$token" || failed=1
    done

    [ $failed -eq 1 ] && { log_error "Some files failed to download"; return 1; }
    log_info "Model download complete"
}

model_exists() {
    local model_path="$1" use_int8="${2:-false}"

    [ ! -d "$model_path" ] && return 1
    [ ! -f "${model_path}/tokenizer.json" ] && return 1

    if [ "$use_int8" = "true" ]; then
        [ ! -f "${model_path}/model_int8.onnx" ] && return 1
    else
        [ ! -f "${model_path}/model.onnx" ] && return 1
    fi
    return 0
}

main() {
    log_info "Starting Knowledge API..."

    local args=("$@")
    local final_args=()
    local models_dir="${MODELS_DIR:-/models}"
    local use_int8="false"

    # First pass: check for --int8 flag
    for arg in "${args[@]}"; do
        [ "$arg" = "--int8" ] && use_int8="true"
    done

    # Second pass: process arguments, handling HF model download
    local i=0
    while [ $i -lt ${#args[@]} ]; do
        local arg="${args[$i]}"

        if [ "$arg" = "--model" ] || [ "$arg" = "-m" ]; then
            local model_id="${args[$((i+1))]}"

            if is_hf_model_id "$model_id"; then
                local model_name="${model_id#*/}"
                local local_path="${models_dir}/${model_name}"

                log_info "Detected HuggingFace model: ${model_id}"

                if ! model_exists "$local_path" "$use_int8"; then
                    log_info "Model not found locally, downloading..."
                    download_model "$local_path" "$model_id" "${HF_TOKEN:-}" "$use_int8" || exit 1
                else
                    log_info "Model already exists at ${local_path}"
                fi

                final_args+=("--model" "$local_path")
            else
                if [ ! -d "$model_id" ]; then
                    log_warn "Model path ${model_id} does not exist"
                fi
                final_args+=("--model" "$model_id")
            fi
            i=$((i + 2))
        else
            final_args+=("$arg")
            i=$((i + 1))
        fi
    done

    log_info "Executing: knowledge-api ${final_args[*]}"

    exec knowledge-api "${final_args[@]}"
}

main "$@"
