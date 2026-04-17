#!/usr/bin/env bash
# benchmark.sh – Compare CPU-only vs full-GPU inference speed on Jetson TX2.
#
# Usage:
#   ./scripts/benchmark.sh [model-blob-path]
#
# If no path is given, the script tries to locate the smallest model blob
# under /usr/share/ollama/.ollama/models/blobs (i.e. the 3 B model).

set -euo pipefail

LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "ERROR: llama-cli not found at $LLAMA_CLI"
  echo "Run install.sh first."
  exit 1
fi

# Auto-detect model blob if not provided
if [[ -n "${1:-}" ]]; then
  MODEL="$1"
else
  MODEL=$(find /usr/share/ollama/.ollama/models/blobs -size +500M -size -3G 2>/dev/null \
          | xargs ls -s 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
fi

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "ERROR: Could not find a suitable model blob."
  echo "Usage: $0 /path/to/model.gguf"
  exit 1
fi

echo "Model: $MODEL"
echo "Size:  $(du -sh "$MODEL" | cut -f1)"
echo ""

PROMPT="<|im_start|>user\nExplain what machine learning is in 3 sentences.<|im_end|>\n<|im_start|>assistant\n"
COMMON_ARGS=(-m "$MODEL" -p "$PROMPT" -n 80 -c 2048 -t 4)

run_bench() {
  local label="$1"; shift
  echo "--- $label ---"
  "$LLAMA_CLI" "${COMMON_ARGS[@]}" "$@" 2>&1 \
    | grep -E "prompt eval time|eval time" \
    | awk '{
        if (/prompt eval/) printf "  Prompt eval : %s tok/s\n", $NF
        else               printf "  Generation  : %s tok/s\n", $NF
      }'
  echo ""
}

run_bench "CPU only"            --n-gpu-layers 0
run_bench "GPU full (all layers)" --n-gpu-layers 99
