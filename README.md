# kintsugi-zig

Dedicated bare-metal **Zig `0.17.1` + native `sm_89` CUDA** inference runtime for **Kintsugi-27B** (`Qwen3.8-27B` Escha ~2.5-bit + `DFlash2` 5-layer speculative drafter), tailored specifically for **NVIDIA RTX 4070 Ti (12 GB, Ada Lovelace, 60 SMs, 48 MiB L2)**.

Zero `ggml`, zero `libcudart`, zero `libstdc++`, zero external HTTP dependencies — compiles into a single self-contained native binary embedding all 6 `sm_89` CUBIN modules via `@embedFile`.

## Verified Results vs `llama-server` (RTX 4070 Ti 12 GB, Same Prompt & Weights)

Benchmark: 256-token greedy generation on `"Write a concise Python implementation of quicksort and explain its time complexity in detail."` (exact greedy match on `France` 32/32, `Fibonacci` 32/32, and `Quicksort` prefix).

| Metric | `llama-server` (Kintsugi) | `kintsugi-zig` (Native `sm_89`) | Delta |
| :--- | ---: | ---: | ---: |
| **HTTP Decode Throughput** | `123.89 tok/s` | **`139.77 tok/s`** | **+12.8%** |
| **CLI Decode Throughput** (`bench-spec`) | `123.89 tok/s` | **`139.19 tok/s`** | **+12.3%** |
| **Step Latency** (`M=8` verify + `M=7` draft) | `40.36 ms/step` | **`35.23 ms/step`** | **-12.7%** (1.15× faster) |
| **Prompt Prefill / TTFT** (16 tok) | `188.25 ms` (`85.0 tok/s`) | **`56.81 ms`** (`281.6 tok/s`) | **3.31× faster** |
| **VRAM Q4_0 KV Cache Pool** | `69,632 cells` | **`118,784 cells`** | **+70.6% context** |
| **Cold Load Time** (27B + DFlash2 5L) | — | **`772 ms`** | Single-arena async DMA |
| **Greedy Parity (`M=1 == Spec == Ref`)** | Baseline | **100% Exact Match** | 0 local register spills |

### Per-Step Stage Latency Breakdown (`35.23 ms/step`)
- **Target 64L Trunk (`M=8`, CUDA Graph)**: `27.35 ms`
- **Target 2-Stage LM Head (`248,320` vocab)**: `2.09 ms` (GPU `Q4_0 × Q8_1` LDG.128 + Bitonic Top-64: `1.86 ms` | CPU `Q8_0` exact rescore from host mmap: `0.22 ms`, zero VRAM)
- **DFlash2 5L Draft (`M=8` + Selector + LM Head `M=7`, CUDA Graph)**: `4.76 ms` (5L Trunk + Selector: `2.90 ms` | LM Head `M=7`: `1.86 ms`)
- **GDN Speculative Commit (`n=5`, CUDA Graph)**: `0.88 ms`
- **DFlash2 KV Ring Inject (`n=5`, CUDA Graph)**: `0.32 ms`

## Architecture
- **Host runtime (`src/`)**: Pure Zig `0.17.1` calling `libcuda.so.1` directly via the CUDA Driver API (`src/cuda.zig`), zero-copy `mmap` GGUF v3 loader (`src/gguf.zig`), Qwen3.5 byte-level BPE tokenizer (`src/tokenizer.zig`), 64-layer hybrid GDN/FullAttn engine (`src/engine.zig`), DFlash2 5-layer speculative drafter (`src/dflash.zig`), and zero-dependency OpenAI HTTP/1.1 + SSE server (`src/server.zig`).
- **Native `sm_89` kernels (`kernels/`)**: Compiled directly to `sm_89` CUBINs via CUDA 12.8 NVRTC (`tools/compile_cubin.py`), **0 bytes local register spill** across all kernels:
  - `kernels/sm89_core.cu`: Fused residual + bias + RMSNorm + 2×/3× 128-pt Hadamard (`FP16`), DFlash2 hidden-state capture, GPU `Q8_0 → Q4_0` LM-head transcoder.
  - `kernels/escha_sm89.cu`: 2-bit & 3-bit Escha MCG Trellis `m16n8k16` Tensor Core GEMV with 60-SM wave-saturated `narrow16` geometry.
  - `kernels/gdn_sm89.cu`: 48-layer Gated DeltaNet Conv1D+SiLU+L2Norm, α/β gates, register-resident FP32 recurrence with speculative verify/commit snapshot, fused RMSNorm×SiLU(z)+Had128.
  - `kernels/attn_sm89.cu`: 16-layer Full Attention Q/K RMSNorm + MRoPE (`freq_base=1e7`) + Q4_0 KV append, `GQA6×4` shared-memory tiled Split-K FlashAttention, device-pointer `_dpos` wrappers for static CUDA Graph capture.
  - `kernels/lmhead_sm89.cu`: Fused SwiGLU+Had128, vectorized LDG.128 shared-memory `Q4_0 × Q8_1` `__dp4a` GEMV (`M=1,7,8`), parallel-bitonic exact GPU Top-64.
  - `kernels/dflash_sm89.cu`: DFlash2 5-layer RMSNorm, dynamic grouped causal Conv2, QK-Norm + NeoX RoPE(128), FP16 SWA-2048 ring buffer KV inject, non-causal GQA (32:8) FlashAttention, fused SwiGLU→Q8_1.
- **Zero-VRAM Host-Mmap Offload**: `token_embd.weight` (`1.26 GiB`), `output.weight` Q8_0 Stage-2 rescore (`1.26 GiB`), and DFlash2 `selector_predecessor`/`selector_successor` (`68.2 MiB`) stay out of VRAM via zero-copy `mmap`, freeing **+49,152 additional Q4_0 KV cells** (`118,784` total) in 12 GB VRAM.
- **Static CUDA Graph Capture**: Target `M=8` verify (3 sequence-length buckets), GDN `commitSpeculative[1..8]`, DFlash2 `encodeAndInject[1..8]`, and DFlash2 `draft` (including Stage-1 LM Head) all replay from cached `CUgraphExec` handles with device-side `pos` indirection.

## Build & Run

```bash
export PATH="$HOME/.local/toolchains/zig-0.17.1:$PATH"

# Build & run unit tests (no GPU required for tests)
zig build -Doptimize=ReleaseFast
zig build test

# Hardware probe, GGUF VRAM plan & kernel benchmarks
./zig-out/bin/kintsugi-zig probe
./zig-out/bin/kintsugi-zig info
./zig-out/bin/kintsugi-zig bench-escha
./zig-out/bin/kintsugi-zig bench-ops

# End-to-end verification & speculative decoding benchmark
./zig-out/bin/kintsugi-zig verify-e2e
./zig-out/bin/kintsugi-zig bench-spec

# CLI text generation (DFlash2 speculative decoding on by default; pass --no-spec for M=1)
./zig-out/bin/kintsugi-zig generate --prompt "The capital of France is" --max-tokens 64 --temp 0.0

# OpenAI-compatible HTTP/1.1 + SSE streaming server
./zig-out/bin/kintsugi-zig serve --host 127.0.0.1 --port 8080
```

### HTTP Server Endpoints
- `GET  /health` — liveness, speculative flag, allocated VRAM KV cells
- `GET  /v1/models` — OpenAI models list (`Kintsugi-27B`)
- `POST /tokenize` — `{"content": "..."}` → `{"tokens": [...]}`
- `POST /v1/completions` — OpenAI completions (`prompt`, `max_tokens`, `temperature`, `top_k`, `top_p`, `seed`, `stop`, `stream: true|false`, plus `timings`)
- `POST /v1/chat/completions` — OpenAI chat completions with Qwen3.5 ChatML formatting, UTF-8-safe SSE streaming, and optional `"enable_thinking": false`
