# LLM on Jetson TX2 with GPU Acceleration

Run large language models on the **NVIDIA Jetson TX2** with full GPU acceleration.
This repository solves the non-obvious compatibility issues between the TX2's JetPack 4.6
software stack and modern LLM tooling.

Out of the box, neither Ollama nor llama.cpp compile or run correctly on the TX2.
This repo provides a one-shot installer, all required patches, and a clear explanation
of every obstacle so you understand what is happening and why.

---

## Hardware & Software

| Component | Value |
|-----------|-------|
| Board | NVIDIA Jetson TX2 |
| JetPack | 4.6.x (R32.7.6) |
| OS | Ubuntu 18.04.6 LTS (Bionic) |
| Kernel | 4.9-tegra (aarch64) |
| GPU | 256-core Pascal, compute capability 6.2 |
| Memory | 8 GB unified LPDDR4 (CPU + GPU share the same pool) |
| CUDA | 10.2 (driver + toolkit) |
| GLIBC | 2.27 |

---

## Performance Results

All numbers measured on the TX2 with `qwen2.5:3b` (Q4_K_M, 1.8 GB).

| Mode | Prompt eval | Generation | Notes |
|------|------------|------------|-------|
| CPU only | 1.86 tok/s | **0.84 tok/s** | All 37 layers on CPU |
| **GPU full** | **12.60 tok/s** | **5.13 tok/s** | All 37 layers on Tegra GPU |

**6× speedup** when the entire model fits in GPU memory.

For `qwen2.5:7b` (4.4 GB) with partial offload (20/29 layers on GPU):

| Mode | Generation |
|------|------------|
| CPU only | 0.48 tok/s |
| GPU partial (20 layers) | 0.97 tok/s |

---

## Quick Start

```bash
git clone https://github.com/<your-username>/jetson-tx2-llm.git
cd jetson-tx2-llm
chmod +x install.sh
sudo ./install.sh
```

After installation (~15–20 min, mostly compile time):

```bash
# Pull the recommended model
ollama pull qwen2.5:3b

# Run GPU-accelerated inference
./scripts/run.sh "What is the capital of France?"

# Benchmark CPU vs GPU
./scripts/benchmark.sh
```

---

## Why This Is Needed

### Problem 1 — Ollama requires GLIBC 2.28, TX2 has 2.27

Ollama's official installer downloads the latest release, which is compiled against
GLIBC 2.28. Ubuntu 18.04 ships GLIBC 2.27 and upgrading it is not safe on a Jetson.

**Fix:** Download Ollama **v0.1.48** specifically. It is the last release compiled
against GLIBC 2.17, which works on any modern Linux.

```
/usr/local/bin/ollama: /lib/aarch64-linux-gnu/libc.so.6:
  version 'GLIBC_2.28' not found
```

---

### Problem 2 — Ollama's bundled CUDA runner requires CUDA 11

Ollama ships a pre-built `ollama_llama_server` binary that links against
`libcudart.so.11.0`. The TX2 only has CUDA 10.2. Even though the GPU is
detected correctly, every inference falls back to CPU silently.

**Fix:** Build `llama-server` from source against CUDA 10.2 and replace
Ollama's extracted runner binary after each service start. A systemd drop-in
automates this.

---

### Problem 3 — Modern llama.cpp does not compile with CUDA 10.2

The latest llama.cpp uses C++17 and CUDA 11+ features in its CUDA kernel code.
CUDA 10.2's `nvcc` rejects several of them. We target **llama.cpp b3278**
(August 2024) and apply five small patches:

#### Patch 1 — `cuda_bf16.h` missing (`stubs/cuda_bf16.h`)
`cuda_bf16.h` (bfloat16 support) was added in CUDA 11. llama.cpp includes it
unconditionally. We provide a minimal stub that defines the types without
implementing anything — they are never called on CUDA 10.2 code paths.

#### Patch 2 — `std::is_same_v` fold expression (`patches/common.cuh.patch`)
```cpp
// Before (C++17 fold expression — nvcc 10.2 rejects this)
template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

// After (C++14-compatible recursive template)
template <typename T, typename U>
struct is_same_any_multi { ... };
template <typename T, typename... Ts>
inline constexpr bool is_any = is_same_any_multi<T, Ts...>::value;
```

#### Patch 3 — `constexpr __device__` variable (`patches/common.cuh.patch`)
CUDA 10.2 does not allow `__device__` variables to be `constexpr`.
```cpp
// Before
static constexpr __device__ int8_t kvalues_iq4nl[16] = { ... };
// After
static __device__ int8_t kvalues_iq4nl[16] = { ... };
```

#### Patch 4 — `__builtin_assume` in device code (`patches/fattn-*.patch`)
`__builtin_assume` is a GCC/Clang built-in that nvcc 10.2 does not recognise.
```cpp
// Before
__builtin_assume(tid < D);
// After (removed; it was a hint only, not functional)
```

#### Patch 5 — ARM NEON `vld1q_*_x4` intrinsics (`patches/ggml-impl.h.patch`)
The `vld1q_u8_x4` / `vld1q_s8_x4` "load four vectors" intrinsics were added
to GCC's `arm_neon.h` in GCC 9. The TX2 build must use GCC 8 (CUDA 10.2 rejects
GCC ≥ 9). The types `uint8x16x4_t` / `int8x16x4_t` exist in GCC 8 but the load
functions do not.

```c
// After (inline fallback using scalar vld1q_u8 calls)
inline static uint8x16x4_t ggml_vld1q_u8_x4(const uint8_t *ptr) {
    uint8x16x4_t res;
    res.val[0] = vld1q_u8(ptr +  0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);
    return res;
}
```

---

### Problem 4 — Unified memory not recognised as GPU VRAM

The TX2 uses a unified memory architecture: CPU and GPU share the same physical
DRAM. Ollama queries the CUDA driver for free GPU memory and sees only ~779 MiB
(the current free pages from the GPU's perspective). It assumes the model won't
fit and silently falls back to CPU.

**Fix:** Set `OLLAMA_MAX_VRAM=6000000000` in the systemd service environment.
This overrides the free-memory check and allows Ollama to schedule layers onto
the GPU using the full unified pool.

---

### Problem 5 — Default context window causes OOM

Ollama and llama.cpp default to a 32 768-token context window. For `qwen2.5:7b`
this allocates a **1.8 GB KV cache** in addition to the 4.4 GB model weights,
exceeding 8 GB and triggering the Linux OOM killer.

**Fix:** Always set `num_ctx 2048` (or `-c 2048` with llama-cli) for inference
on the TX2. This reduces the KV cache to ~112 MB.

---

## GCC Version Matrix

| GCC | NEON `vld1q_*_x4` | CUDA 10.2 | Use? |
|-----|-------------------|-----------|------|
| 7.x | ❌ Not available | ✅ | ❌ |
| **8.x** | **✅ Available** | **✅** | **✅** |
| 9.x | ✅ Available | ❌ Rejected | ❌ |

GCC 8 is the only version that satisfies both constraints simultaneously.

---

## Recommended Models

| Model | Size | GPU layers | Generation speed | Notes |
|-------|------|-----------|-----------------|-------|
| `qwen2.5:3b` | 1.8 GB | 37/37 (full) | **~5 tok/s** | Best balance — recommended |
| `qwen2.5:7b` | 4.4 GB | 20/29 (partial) | ~1 tok/s | Smarter, slower |
| `llama3.2:3b` | 2.0 GB | all | ~4 tok/s | Good alternative |

Pull a model with Ollama (model management only):
```bash
ollama pull qwen2.5:3b
```

---

## Repository Structure

```
jetson-tx2-llm/
├── install.sh                      # One-shot installer
├── patches/
│   ├── common.cuh.patch            # is_same_v, constexpr device, __builtin_assume
│   ├── fattn-common.cuh.patch      # __builtin_assume
│   ├── fattn-vec-f16.cuh.patch     # __builtin_assume
│   ├── fattn-vec-f32.cuh.patch     # __builtin_assume
│   └── ggml-impl.h.patch           # ARM NEON vld1q_*_x4 fallbacks
├── stubs/
│   └── cuda_bf16.h                 # BF16 stub for CUDA 10.2
├── config/
│   ├── ollama.service              # Systemd unit with OLLAMA_MAX_VRAM + LD_LIBRARY_PATH
│   └── Modelfile.example           # num_ctx + num_gpu for safe TX2 inference
└── scripts/
    ├── install.sh                  # (same as root install.sh)
    ├── replace-runner.sh           # Swap Ollama's CUDA 11 runner post-start
    ├── benchmark.sh                # CPU vs GPU tok/s comparison
    └── run.sh                      # GPU inference wrapper (alternative to ollama run)
```

---

## Manual Build (without install.sh)

If you prefer to follow the steps manually:

```bash
# 1. Install dependencies
sudo apt-get install -y cuda-toolkit-10-2 gcc-8 g++-8 git zstd
# CMake 3.25 from Kitware
sudo bash -c 'echo "deb https://apt.kitware.com/ubuntu/ bionic main" \
  > /etc/apt/sources.list.d/kitware.list'
curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y cmake

# 2. BF16 stub
sudo cp stubs/cuda_bf16.h /usr/local/cuda-10.2/include/cuda_bf16.h

# 3. Clone and patch llama.cpp
git clone --depth=1 --branch b3278 \
  https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
for p in ../patches/*.patch; do patch -p1 < "$p"; done

# 4. Build
mkdir build && cd build
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES="62" \
  -DCMAKE_C_COMPILER=gcc-8 \
  -DCMAKE_CXX_COMPILER=g++-8 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-8
make -j4 llama-cli llama-server

# 5. Install Ollama v0.1.48
curl -fsSL https://ollama.com/install.sh | sudo sh
curl -fsSL https://github.com/ollama/ollama/releases/download/v0.1.48/ollama-linux-arm64 \
  -o /tmp/ollama-0.1.48
sudo mv /tmp/ollama-0.1.48 /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama

# 6. Configure systemd service
sudo cp config/ollama.service /etc/systemd/system/ollama.service
sudo systemctl daemon-reload && sudo systemctl restart ollama

# 7. Replace Ollama's CUDA 11 runner with ours
./scripts/replace-runner.sh
```

---

## How the Runner Replacement Works

Ollama embeds pre-built runner binaries in its own binary and extracts them to
`/tmp/ollamaXXXXXX/runners/` at startup. The `cuda_v11` runner links against
`libcudart.so.11.0`, which does not exist on CUDA 10.2.

After each Ollama restart the `replace-runner.sh` script (automated via a
systemd `ExecStartPost` drop-in) waits for extraction to complete and then
overwrites `runners/cuda_v11/ollama_llama_server` with our CUDA-10.2-compiled
`llama-server` binary. Since `LD_LIBRARY_PATH` in the service unit points to
the CUDA 10.2 libraries, the runner finds the correct shared objects at runtime.

---

## Troubleshooting

**`GLIBC_2.28 not found`**
You have a newer Ollama binary. Re-run `install.sh` or manually download v0.1.48:
```bash
curl -fsSL https://github.com/ollama/ollama/releases/download/v0.1.48/ollama-linux-arm64 \
  -o /tmp/ollama && sudo mv /tmp/ollama /usr/local/bin/ollama && sudo chmod +x /usr/local/bin/ollama
```

**OOM killer terminates the process**
Reduce the context window: add `-c 2048` to llama-cli or set `num_ctx 2048` in
your Modelfile.

**GPU not detected (`inference compute` missing from logs)**
Check that `OLLAMA_MAX_VRAM` is set: `sudo systemctl cat ollama | grep VRAM`

**Model loads on CPU despite GPU being available**
The runner may not have been replaced. Run:
```bash
./scripts/replace-runner.sh
```
Then verify: `ldd $(sudo find /tmp/ollama*/runners/cuda_v11 -name ollama_llama_server) | grep cuda`
Expected output should show `libcudart.so.10.2`, not `libcudart.so.11.0`.

**`nvcc fatal: Unsupported gpu architecture 'compute_80'`**
You are using a newer llama.cpp version or did not set `CMAKE_CUDA_ARCHITECTURES=62`.
CUDA 10.2 only supports compute capabilities up to 7.5.

---

## License

MIT — see [LICENSE](LICENSE).

Patches are derived from [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT) and [Ollama](https://github.com/ollama/ollama) (MIT).
