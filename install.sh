#!/usr/bin/env bash
# install.sh – One-shot GPU LLM setup for Jetson TX2 (JetPack 4.6, Ubuntu 18.04)
#
# What this script does:
#   1. Installs CUDA 10.2 toolkit (already in Jetson apt repos)
#   2. Installs GCC 8 and CMake 3.25 (required to compile llama.cpp)
#   3. Clones llama.cpp b3278 and applies CUDA-10.2 compatibility patches
#   4. Builds llama-cli and llama-server with CUDA support
#   5. Installs Ollama v0.1.48 (last version compatible with GLIBC 2.27)
#   6. Configures the Ollama systemd service for unified-memory GPU use
#   7. Installs a systemd drop-in that auto-replaces Ollama's CUDA 11 runner
#      with our CUDA 10.2 build on every service start
#
# Usage:
#   chmod +x install.sh && sudo ./install.sh
#
# Requirements:
#   - Jetson TX2 with JetPack 4.6.x (Ubuntu 18.04, kernel 4.9-tegra)
#   - Internet access
#   - ~10 GB free disk space (build artifacts + model)

set -euo pipefail

# ── helpers ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[[ $EUID -ne 0 ]] && error "Please run as root: sudo ./install.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_USER="${SUDO_USER:-$(logname 2>/dev/null || echo tarlan)}"
BUILD_HOME=$(eval echo "~$BUILD_USER")

info "Installing for user: $BUILD_USER (home: $BUILD_HOME)"

# ── 1. System packages ─────────────────────────────────────────────────────────
info "Step 1/7 – Installing system packages"

# CUDA 10.2 toolkit (available in the Jetson apt sources out-of-the-box)
apt-get install -y cuda-toolkit-10-2 zstd

# GCC 8: the only version accepted by CUDA 10.2's nvcc (rejects GCC ≥ 9).
# GCC 8 lacks vld1q_u8_x4/vld1q_s8_x4 natively — the ggml-impl.h.patch adds them.
apt-get install -y gcc-8 g++-8

# Build tools
apt-get install -y git build-essential

# CMake 3.25 from Kitware (Ubuntu 18.04 ships only 3.10, llama.cpp needs ≥ 3.14)
apt-get install -y apt-transport-https ca-certificates gnupg
curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
  -o /tmp/kitware.asc
apt-key add /tmp/kitware.asc
bash -c 'echo "deb https://apt.kitware.com/ubuntu/ bionic main" \
  > /etc/apt/sources.list.d/kitware.list'
apt-get update -qq
apt-get install -y cmake

info "cmake version: $(cmake --version | head -1)"
info "nvcc version:  $(PATH=/usr/local/cuda/bin:$PATH nvcc --version | grep release)"

# ── 2. cuda_bf16.h stub ────────────────────────────────────────────────────────
info "Step 2/7 – Installing cuda_bf16.h stub for CUDA 10.2"
# CUDA 10.2 does not ship cuda_bf16.h; llama.cpp unconditionally includes it.
# The stub satisfies the compiler; no BF16 ops are actually called on this HW.
cp "$SCRIPT_DIR/stubs/cuda_bf16.h" /usr/local/cuda-10.2/include/cuda_bf16.h
info "Stub installed to /usr/local/cuda-10.2/include/cuda_bf16.h"

# ── 3. Build llama.cpp ─────────────────────────────────────────────────────────
info "Step 3/7 – Cloning llama.cpp b3278"
LLAMA_DIR="$BUILD_HOME/llama.cpp"

if [[ -d "$LLAMA_DIR" ]]; then
  warn "llama.cpp directory already exists at $LLAMA_DIR — skipping clone."
else
  sudo -u "$BUILD_USER" git clone --depth=1 --branch b3278 \
    https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

info "Step 4/7 – Applying CUDA 10.2 compatibility patches"
cd "$LLAMA_DIR"

# Apply all patches (idempotent: patch will skip already-applied hunks)
for PATCH in "$SCRIPT_DIR/patches"/*.patch; do
  info "  Applying $(basename "$PATCH")"
  sudo -u "$BUILD_USER" patch -p1 --forward --reject-file=/tmp/patch-rejects \
    < "$PATCH" || {
      # patch returns non-zero when hunks are already applied; that's fine.
      warn "  $(basename "$PATCH") may already be applied — continuing."
    }
done

info "Step 5/7 – Building llama.cpp with CUDA 10.2 (compute_62, ~15 min)"
BUILD_DIR="$LLAMA_DIR/build"
sudo -u "$BUILD_USER" mkdir -p "$BUILD_DIR"

sudo -u "$BUILD_USER" cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES="62" \
  -DCMAKE_C_COMPILER=gcc-8 \
  -DCMAKE_CXX_COMPILER=g++-8 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-8

sudo -u "$BUILD_USER" cmake --build "$BUILD_DIR" \
  --target llama-cli llama-server \
  --parallel 4

info "Build complete: $BUILD_DIR/bin/llama-cli"
info "Build complete: $BUILD_DIR/bin/llama-server"

# ── 4. Install Ollama ──────────────────────────────────────────────────────────
info "Step 6/7 – Installing Ollama"

# Ollama v0.1.48 is the last release whose binary only requires GLIBC 2.17.
# Newer releases require GLIBC 2.28, which Ubuntu 18.04 does not provide.
# Always pin to v0.1.48 regardless of what may already be installed.
if ! command -v ollama &>/dev/null; then
  # First install: use the official script to set up the ollama user, group,
  # and systemd service unit. The binary it downloads may require GLIBC 2.28
  # and fail to run — that is expected; we replace it immediately below.
  apt-get install -y zstd
  curl -fsSL https://ollama.com/install.sh -o /tmp/ollama_install.sh
  bash /tmp/ollama_install.sh || true
fi

info "Pinning Ollama to v0.1.48 (GLIBC 2.17 compatible)"
TMPBIN=$(mktemp)
curl -fsSL \
  "https://github.com/ollama/ollama/releases/download/v0.1.48/ollama-linux-arm64" \
  -o "$TMPBIN"
mv "$TMPBIN" /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama
info "Ollama version: $(ollama --version 2>&1 | head -1 || true)"

# ── 5. Configure Ollama systemd service ───────────────────────────────────────
info "Step 7/7 – Configuring Ollama systemd service"

cp "$SCRIPT_DIR/config/ollama.service" /etc/systemd/system/ollama.service

# Drop-in that runs replace-runner.sh automatically after Ollama starts.
# This overwrites the bundled CUDA 11 runner with our CUDA 10.2 binary so
# that `ollama run` uses the GPU without any manual steps.
mkdir -p /etc/systemd/system/ollama.service.d
LLAMA_SERVER="$BUILD_DIR/bin/llama-server"
cat > /etc/systemd/system/ollama.service.d/replace-runner.conf << EOF
[Service]
ExecStartPost=/bin/bash -c ' \
  for i in \$(seq 1 30); do \
    R=\$(find /tmp/ollama*/runners/cuda_v11/ollama_llama_server 2>/dev/null | head -1); \
    [ -n "\$R" ] && break; sleep 1; \
  done; \
  [ -n "\$R" ] && cp ${LLAMA_SERVER} "\$R" && chmod 755 "\$R" && \
  echo "Runner replaced with CUDA 10.2 binary" || \
  echo "WARNING: Could not find Ollama runner to replace"'
EOF

systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama

# Wait for Ollama to be ready
info "Waiting for Ollama API..."
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:11434/ &>/dev/null && break
  sleep 2
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Pull the recommended model (1.8 GB, fits 100% on GPU):"
echo "    ollama pull qwen2.5:3b"
echo ""
echo "  Run GPU inference via llama-cli:"
echo "    $BUILD_DIR/bin/llama-cli \\"
echo "      -m \$(find /usr/share/ollama/.ollama/models/blobs -size +1G | head -1) \\"
echo "      --n-gpu-layers 99 -c 2048 -t 4 \\"
echo "      -p 'Hello, who are you?'"
echo ""
echo "  Run the CPU vs GPU benchmark:"
echo "    bash $SCRIPT_DIR/scripts/benchmark.sh"
echo ""
