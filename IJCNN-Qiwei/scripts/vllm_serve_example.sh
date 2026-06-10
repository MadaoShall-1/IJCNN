#!/usr/bin/env bash
set -euo pipefail

# EXACT requires every LLM component to be served by vLLM or a compatible
# OpenAI-style server. Use one <=8B open-source model at a time.
MODEL_NAME="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

vllm serve "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${MODEL_NAME}"
