#!/usr/bin/env bash
# replace-runner.sh
#
# Ollama extracts its own bundled runner binaries to a fresh /tmp/ollamaXXX
# directory on every start.  The bundled CUDA runner targets CUDA 11, which
# the Jetson TX2 (JetPack 4.6, CUDA 10.2) does not support.
#
# This script locates the extracted cuda_v11 runner and replaces it with the
# CUDA-10.2-compatible llama-server binary we built from source.
#
# Run once after every `sudo systemctl restart ollama`, BEFORE issuing the
# first `ollama run` command.  The install.sh wires this up automatically via
# a systemd drop-in override.
#
# Usage:
#   ./scripts/replace-runner.sh [path-to-llama-server]
#
# The default llama-server path is ~/llama.cpp/build/bin/llama-server.

set -euo pipefail

LLAMA_SERVER="${1:-$HOME/llama.cpp/build/bin/llama-server}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: llama-server not found at $LLAMA_SERVER"
  echo "Run install.sh first, or pass the correct path as the first argument."
  exit 1
fi

echo "Waiting for Ollama to extract its runners..."
for i in $(seq 1 30); do
  RUNNER=$(find /tmp/ollama*/runners/cuda_v11/ollama_llama_server 2>/dev/null | head -1)
  if [[ -n "$RUNNER" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "$RUNNER" ]]; then
  echo "ERROR: Could not find Ollama's extracted runner in /tmp/ollama*/."
  echo "Make sure Ollama is running: sudo systemctl status ollama"
  exit 1
fi

echo "Found runner: $RUNNER"
echo "Replacing with CUDA 10.2 binary: $LLAMA_SERVER"
sudo cp "$LLAMA_SERVER" "$RUNNER"
sudo chmod 755 "$RUNNER"

echo "Done. Verify CUDA linkage:"
ldd "$RUNNER" | grep -E "cuda|cublas" || true
echo ""
echo "Runner is ready. You can now run: ollama run <model>"
