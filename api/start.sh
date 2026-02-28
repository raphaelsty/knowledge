#!/bin/bash
exec next-plaid-api \
  --index-dir "${INDEX_DIR:-indices}" \
  --model "${MODEL:-lightonai/answerai-colbert-small-v1-onnx}" \
  --int8 \
  --port "${PORT:-8080}"
