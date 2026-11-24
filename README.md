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
- **Phase 1 Step 1 (Done)**: Standalone `sm_89` Escha 2-bit & 3-bit Trellis GEMV kernels (`kernels/escha_sm89.cu`) verified against independent CPU decoder (`src/escha_ref.zig`) on real `Kintsugi-27B-12G.gguf` weights (`cos_sim = 0.9999999`, 0 local register spills), plus RTX 4070 Ti 60-SM wave-saturation geometry (`narrow16` `OC/32 + hadOut`: **2.71×** on `OC=1024`, **1.21×** on `OC=5120`, `5.96e-8` max diff vs Kintsugi baseline).
- **Phase 1 Step 2 (Done)**: Complete non-Escha `sm_89` operator suite (15 kernels across `kernels/gdn_sm89.cu`, `kernels/attn_sm89.cu`, `kernels/lmhead_sm89.cu`, all **0 B local register spill** and CPU golden-reference verified on real `Kintsugi-27B-12G.gguf` weights):
  - **48-layer GDN (Gated DeltaNet)**: Conv1D+SiLU+L2Norm, α/β gates, fused FP32 recurrence with speculative `n_commit` snapshot, RMSNorm+SiLU(z)+Had128 → **`26.90 µs/layer` (`1.29 ms × 48`, `cos=1.0000000`)**.
  - **16-layer Full Attention + Q4_0 KV Cache**: IEEE-exact Q/K RMSNorm + MRoPE (`freq_base=1e7`) + Q4_0 append, `GQA6×4` shared-memory tiled Split-K FlashAttention (`half4` bank-conflict-free + `HFMA2` + `MUFU.EX2` 4-token blocked softmax, **2.66× faster** than 1-warp baseline: `205.34 → 77.10 µs`), fused Split-K reduce + `sigmoid(gate)` + Had128 → **`84.37 µs/layer` (`1.35 ms × 16`, `cos=0.9999997`)**.
  - **FFN & 2-Stage LM Head**: `fused_swiglu_had128_f16` (`2.82 µs/layer` = **`0.18 ms × 64`**), `__dp4a` `Q4_0 × Q8_1` GEMV (`288.0 GB/s`), 128-thread parallel-bitonic exact GPU Top-64 over `248,320 × 8` (**`6.58×`** speedup: `694.52 → 105.60 µs`, `512/512` exact), and sparse exact `Q8_0` Top-64 rescore (`3.86 µs`, `cos=0.9999959`).
- **Phase 1 Step 3 (Next)**: Wire full 64-layer `M=1` & `M=8` forward pass + DFlash2 5-layer drafter + BPE tokenizer, and verify end-to-end logits/tokens against `llama-server`.
- **Phase 2**: Static CUDA VMM tiered KV cache (`~106K` VRAM KV cells), GPU-resident DFlash2 loop + GPU sampling, persistent `sm_89` fused kernels, and minimal OpenAI SSE server.

## Build & Run
```bash
export PATH="$HOME/.local/toolchains/zig-0.17.1:$PATH"
zig build -Doptimize=ReleaseFast
zig build test
./zig-out/bin/kintsugi-zig probe
./zig-out/bin/kintsugi-zig info
./zig-out/bin/kintsugi-zig bench-escha
./zig-out/bin/kintsugi-zig bench-ops
```
