# kintsugi-zig

Dedicated bare-metal **Zig + native `sm_89` CUDA** inference runtime for **Kintsugi-27B** (`Qwen3.8-27B` Escha ~2.5-bit + `DFlash2`) tailored specifically for **NVIDIA RTX 4070 Ti (12 GB, Ada Lovelace)**.

## Why a Dedicated Runtime?
Measured on RTX 4070 Ti (`437.2 GB/s` D2D VRAM bandwidth):
- Physical bandwidth floor for 1 DFlash2 draft + 1 target 8-row verify step (`10.336 GB`): **`23.64 ms/step`** (`~154 tok/s` @ `mean_len=3.65`).
- Tuned `llama.cpp` Kintsugi server: **`39.85 ms/step`** (`91.5 tok/s`, 59.3% BW utilization, `69,632` VRAM KV cells).
- `kintsugi-zig` eliminates `ggml`'s 4,825-node graph scheduling, `~350–500 MiB` framework buffer overhead, and host round-trips per step.

## Architecture
- **Host runtime (`src/`)**: Pure Zig `0.17.1`, statically linking zero C++ runtime (`libcudart` / `libstdc++` not needed), calling `libcuda.so.1` directly via CUDA Driver API + CUDA VMM (`cuMemMap`).
- **GPU kernels (`kernels/`)**: Native `sm_89` CUBINs compiled directly via CUDA 12.8 NVRTC (`tools/compile_cubin.py`) and embedded into the single binary via `@embedFile`.

## Roadmap
- **Phase 0 (Done)**: Repo + Git + NVRTC `sm_89` CUBIN build pipeline + CUDA Driver API bindings + zero-copy `mmap` GGUF v3 auditor + `fused_add_rmsnorm_had128_f16` kernel verified against CPU golden reference (`5.68 µs/call`).
- **Phase 1 Step 1 (Done)**: Standalone `sm_89` Escha 2-bit & 3-bit Trellis GEMV kernels (`kernels/escha_sm89.cu`) verified against independent CPU decoder (`src/escha_ref.zig`) on real `Kintsugi-27B-12G.gguf` weights (`cos_sim = 0.9999999`, 0 local register spills), plus RTX 4070 Ti 60-SM wave-saturation geometry (`narrow16` `OC/32 + hadOut`: **2.70×** on `OC=1024`, **1.16×** on `OC=5120`, `5.96e-8` max diff vs Kintsugi baseline).
- **Phase 1 Step 2 (Next)**: Port GDN (Gated DeltaNet) linear-attention + Conv1D, RoPE + Q4_0 FlashAttention, and Q4_0/Q8_0 LM head kernels; wire full 64-layer forward pass (`M=1` & `M=8`) verified against `llama-server`.
- **Phase 2**: Static CUDA VMM tiered KV cache (`~106K` VRAM KV cells), GPU-resident DFlash2 loop + GPU sampling, persistent `sm_89` fused kernels, and minimal OpenAI SSE server.

## Build & Run
```bash
export PATH="$HOME/.local/toolchains/zig-0.17.1:$PATH"
zig build -Doptimize=ReleaseFast
zig build test
./zig-out/bin/kintsugi-zig probe
./zig-out/bin/kintsugi-zig info
./zig-out/bin/kintsugi-zig bench-escha
```
