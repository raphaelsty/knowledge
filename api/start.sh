#!/bin/bash
exec next-plaid-api \
  --index-dir indices \
  --model lightonai/answerai-colbert-small-v1-onnx \
  --int8 \
  --port 8080
