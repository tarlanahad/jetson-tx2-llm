#!/usr/bin/env bash
# run.sh – GPU-accelerated inference wrapper for Jetson TX2.
#
# A lightweight alternative to `ollama run` that uses the CUDA-10.2-compiled
# llama-server directly, bypassing Ollama's runner compatibility issues.
#
# Usage:
#   ./scripts/run.sh <model-blob-path> "your prompt here"
#   ./scripts/run.sh                             # interactive mode with default model
#
# Environment variables:
#   GPU_LAYERS   Number of layers to offload (default: 99 = all layers)
#   CTX_SIZE     Context window in tokens    (default: 2048)
#   THREADS      CPU threads                 (default: 4)

set -euo pipefail

LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-2048}"
THREADS="${THREADS:-4}"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "ERROR: llama-cli not found at $LLAMA_CLI"
  echo "Run install.sh first."
  exit 1
fi

# Auto-detect model if not provided
if [[ -n "${1:-}" ]]; then
  MODEL="$1"
  shift
else
  MODEL=$(find /usr/share/ollama/.ollama/models/blobs -size +500M -size -3G 2>/dev/null \
          | xargs ls -s 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
  if [[ -z "$MODEL" ]]; then
    echo "ERROR: No model found. Pull one first: ollama pull qwen2.5:3b"
    exit 1
  fi
fi

PROMPT="${1:-}"

if [[ -z "$PROMPT" ]]; then
  # Interactive mode
  echo "Model: $MODEL  |  GPU layers: $GPU_LAYERS  |  Context: $CTX_SIZE"
  echo "Type your prompt and press Enter (Ctrl-C to quit):"
  while IFS= read -r -p "> " PROMPT; do
    FORMATTED="<|im_start|>user\n${PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    "$LLAMA_CLI" -m "$MODEL" -p "$FORMATTED" \
      --n-gpu-layers "$GPU_LAYERS" -c "$CTX_SIZE" -t "$THREADS" \
      -n 512 --log-disable 2>/dev/null
    echo ""
  done
else
  FORMATTED="<|im_start|>user\n${PROMPT}<|im_end|>\n<|im_start|>assistant\n"
  "$LLAMA_CLI" -m "$MODEL" -p "$FORMATTED" \
    --n-gpu-layers "$GPU_LAYERS" -c "$CTX_SIZE" -t "$THREADS" \
    -n 512 --log-disable 2>/dev/null
fi
