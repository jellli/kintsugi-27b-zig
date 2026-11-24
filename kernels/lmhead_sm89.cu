// Native sm_89 (RTX 4070 Ti, Ada Lovelace) FFN SwiGLU + 2-Stage LM Head & Q4_0/Q8_0 kernels
// for Kintsugi-27B + DFlash2.
//
// Kernels:
//   1. fused_swiglu_had128_f16   : SiLU(gate+b) * (up+b) * rin + Had128 -> FP16 (direct into ffn_down!)
//   2. quantize_f32_to_q8_1_sm89 : Warp-level F32 -> ggml block_q8_1 (36 bytes/32 elems)
//   3. q4_0_q8_1_gemv_sm89       : __dp4a INT8-dot Q4_0 x Q8_1 GEMV (M=1..8, Stage-1 LM head + DFlash2!)
//   4. topk64_stage1_sm89        : Exact threshold-filtered block Top-64 over N logits (single streaming pass)
//   5. topk64_stage2_sm89        : Exact final merge of block Top-64 -> cand_ids[M, 64]
//   6. q8_0_sparse_rescore_sm89  : Stage-2 LM Head exact Q8_0 x Q8_1 __dp4a rescore on Top-64 candidates

#include <cuda_fp16.h>

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define INV_SQRT128 0.08838834764831845f
#define Q4_0_BYTES 18   // 2B f16 d + 16B qs
#define Q8_0_BYTES 34   // 2B f16 d + 32B int8 qs
#define Q8_1_BYTES 36   // 4B half2(d, s) + 32B int8 qs
#define TOPK_K 64
#define TOPK_BLOCKS 64

struct __align__(8) half4 {
    half2 x;
    half2 y;
};

static __device__ __forceinline__ float warp_sum_32(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

static __device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ void shuffle_had_f2x32(float& x, float& y, int lane) {
    #pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
        float px = __shfl_xor_sync(0xffffffffu, x, stride);
        float py = __shfl_xor_sync(0xffffffffu, y, stride);
        if (lane & stride) {
            x = px - x;
            y = py - y;
        } else {
            x = x + px;
            y = y + py;
        }
    }
}

static __device__ __forceinline__ void had_lane_128(float4& v, int lane) {
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    v = make_float4(s0 + s1, d0 + d1, s0 - s1, d0 - d1);
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
}

static __device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
    int res;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

// =============================================================================
// 1. Fused FFN SwiGLU + Bias + escha_rin + Had128 -> FP16 (direct input to ffn_down!)
//    Grid : (ceil(n_tokens * (d_ff / 128) / 8), 1, 1) — 8 warps/block, 1 warp = 128 dims
//    Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) fused_swiglu_had128_f16(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ gate_bias,
    const float* __restrict__ up_bias,
    const __half* __restrict__ rin,
    __half* __restrict__ xh_out,
    int d_ff,
    int n_tokens
) {
    const int nb = d_ff >> 7;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w >= n_tokens * nb) return;

    const int row = w / nb;
    const int blk = w % nb;
    const int lane = threadIdx.x & 31;

    const size_t off = (size_t)row * d_ff + (size_t)(blk << 7);
    const size_t b_off = (size_t)(blk << 7);

    float4 g = ((const float4*)(gate + off))[lane];
    float4 u = ((const float4*)(up + off))[lane];
    if (gate_bias) {
        const float4 gb = ((const float4*)(gate_bias + b_off))[lane];
        g.x += gb.x; g.y += gb.y; g.z += gb.z; g.w += gb.w;
    }
    if (up_bias) {
        const float4 ub = ((const float4*)(up_bias + b_off))[lane];
        u.x += ub.x; u.y += ub.y; u.z += ub.z; u.w += ub.w;
    }
    const half4 r = ((const half4*)(rin + b_off))[lane];

    float4 v;
    v.x = silu_f32(g.x) * u.x * __low2float(r.x);
    v.y = silu_f32(g.y) * u.y * __high2float(r.x);
    v.z = silu_f32(g.z) * u.z * __low2float(r.y);
    v.w = silu_f32(g.w) * u.w * __high2float(r.y);

    had_lane_128(v, lane);

    half4 out_h;
    out_h.x = __floats2half2_rn(v.x * INV_SQRT128, v.y * INV_SQRT128);
    out_h.y = __floats2half2_rn(v.z * INV_SQRT128, v.w * INV_SQRT128);
    ((half4*)(xh_out + off))[lane] = out_h;
}

// =============================================================================
// 2. Quantize F32 [n_tokens, K] -> ggml block_q8_1 [n_tokens, K/32] (36 bytes/block)
//    Layout per block_q8_1 (4-byte aligned!):
//      bytes 0..3   : half2(d, s) where d = max_abs / 127.0f, s = d * sum(qs)
//      bytes 4..35  : int8_t qs[32]
//    Grid : (ceil(n_tokens * (K/32) / 8), 1, 1), Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) quantize_f32_to_q8_1_sm89(
    const float* __restrict__ x,
    uint8_t* __restrict__ dst_q8_1,
    int K,
    int n_tokens
) {
    const int nb = K >> 5;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w >= n_tokens * nb) return;

    const int lane = threadIdx.x & 31;
    const float val = x[(size_t)w * 32 + lane];

    float amax = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    const float d = amax * (1.0f / 127.0f);
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    int qi = __float2int_rn(val * id);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;

    int qsum = qi;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        qsum += __shfl_xor_sync(0xffffffffu, qsum, mask);
    }

    uint8_t* blk_ptr = dst_q8_1 + (size_t)w * Q8_1_BYTES;
    if (lane == 0) {
        half2 ds = __floats2half2_rn(d, d * (float)qsum);
        *reinterpret_cast<half2*>(blk_ptr) = ds;
    }
    ((int8_t*)(blk_ptr + 4))[lane] = (int8_t)qi;
}

// =============================================================================
// 3. __dp4a Q4_0 x Q8_1 GEMV (M = 1..8): Y[M, N] = X_q8_1[M, K] @ W_q4_0[N, K]^T
//    Amortizes each Q4_0 weight block load across all M tokens simultaneously!
//    Grid : (ceil(N / 4), 1, 1) — 4 warps/block = 4 output rows/block
//    Block: (128, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 4) q4_0_q8_1_gemv_sm89(
    const uint8_t* __restrict__ W_q4_0,  // [N, K/32 * 18]
    const uint8_t* __restrict__ X_q8_1,  // [M, K/32 * 36]
    float* __restrict__ Y,               // [M, N]
    int M,
    int K,
    int N
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const int nb = K >> 5;
    const size_t w_row_bytes = (size_t)nb * Q4_0_BYTES;
    const size_t x_row_bytes = (size_t)nb * Q8_1_BYTES;
    const uint8_t* w_row = W_q4_0 + (size_t)row * w_row_bytes;

    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;

    for (int b = lane; b < nb; b += 32) {
        const uint8_t* wb = w_row + (size_t)b * Q4_0_BYTES;
        const uint16_t dw_bits = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
        const float dw = __half2float(__ushort_as_half(dw_bits));

        // Load 16 nibble bytes (2-byte aligned) into 4 uint32 words
        const uint16_t* w16 = reinterpret_cast<const uint16_t*>(wb + 2);
        const uint32_t u0 = (uint32_t)w16[0] | ((uint32_t)w16[1] << 16);
        const uint32_t u1 = (uint32_t)w16[2] | ((uint32_t)w16[3] << 16);
        const uint32_t u2 = (uint32_t)w16[4] | ((uint32_t)w16[5] << 16);
        const uint32_t u3 = (uint32_t)w16[6] | ((uint32_t)w16[7] << 16);

        const int wl0 = (int)(u0 & 0x0f0f0f0fu);
        const int wl1 = (int)(u1 & 0x0f0f0f0fu);
        const int wl2 = (int)(u2 & 0x0f0f0f0fu);
        const int wl3 = (int)(u3 & 0x0f0f0f0fu);
        const int wh0 = (int)((u0 >> 4) & 0x0f0f0f0fu);
        const int wh1 = (int)((u1 >> 4) & 0x0f0f0f0fu);
        const int wh2 = (int)((u2 >> 4) & 0x0f0f0f0fu);
        const int wh3 = (int)((u3 >> 4) & 0x0f0f0f0fu);

        for (int m = 0; m < M; ++m) {
            const uint8_t* xb = X_q8_1 + (size_t)m * x_row_bytes + (size_t)b * Q8_1_BYTES;
            const half2 ds = *reinterpret_cast<const half2*>(xb);
            const int* xq = reinterpret_cast<const int*>(xb + 4);

            int sumi = 0;
            sumi = dp4a_i8(wl0, xq[0], sumi);
            sumi = dp4a_i8(wl1, xq[1], sumi);
            sumi = dp4a_i8(wl2, xq[2], sumi);
            sumi = dp4a_i8(wl3, xq[3], sumi);
            sumi = dp4a_i8(wh0, xq[4], sumi);
            sumi = dp4a_i8(wh1, xq[5], sumi);
            sumi = dp4a_i8(wh2, xq[6], sumi);
            sumi = dp4a_i8(wh3, xq[7], sumi);

            acc[m] += dw * (__low2float(ds) * (float)sumi - 8.0f * __high2float(ds));
        }
    }

    for (int m = 0; m < M; ++m) {
        const float tot = warp_sum_32(acc[m]);
        if (lane == 0) {
            Y[(size_t)m * N + row] = tot;
        }
    }
}

// =============================================================================
// 3b. Vectorized LDG.128 + Shared-Memory-Broadcast + Register-Pinned Q4_0 x Q8_1 GEMV
//     Requires K % 1024 == 0 (nb % 32 == 0, satisfied by 4096, 5120, 17408, 25600).
//     Grid : (ceil(N / 8), 1, 1) — 8 warps/block = 8 output rows sharing X_q8_1 in smem!
//     Block: (256, 1, 1)
// =============================================================================
template <int M_FIXED>
static __device__ __forceinline__ void q4_0_q8_1_gemv_fast_impl(
    const uint8_t* __restrict__ W_q4_0,  // [N, K/32 * 18], 16-byte aligned per row & 32-blk tile
    const uint8_t* __restrict__ X_q8_1,  // [M_FIXED, K/32 * 36], 16-byte aligned per row & 32-blk tile
    float* __restrict__ Y,               // [M_FIXED, N]
    int K,
    int N
) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;  // 0..7
    const int lane = tid & 31;  // 0..31
    const int row  = blockIdx.x * 8 + warp;
    const bool valid_row = (row < N);

    const int nb = K >> 5;
    const size_t w_row_bytes = (size_t)nb * Q4_0_BYTES;
    const size_t x_row_bytes = (size_t)nb * Q8_1_BYTES;
    const uint8_t* w_row = W_q4_0 + (size_t)(valid_row ? row : 0) * w_row_bytes;

    // 32 Q8_1 blocks = 1152 bytes = 72 uint4 per token; 32 Q4_0 blocks = 576 bytes = 36 uint4 per warp
    __shared__ uint4 s_x[M_FIXED][72];
    __shared__ uint4 s_w[8][36];

    float acc[M_FIXED];
    #pragma unroll
    for (int m = 0; m < M_FIXED; ++m) acc[m] = 0.0f;

    for (int b0 = 0; b0 < nb; b0 += 32) {
        // 1. Collaboratively load X_q8_1[0..M_FIXED-1, b0..b0+31] via coalesced 128-bit loads
        const int total_x_u4 = M_FIXED * 72;
        for (int idx = tid; idx < total_x_u4; idx += 256) {
            const int m = idx / 72;
            const int q = idx - m * 72;
            const uint4* x_src = reinterpret_cast<const uint4*>(
                X_q8_1 + (size_t)m * x_row_bytes + (size_t)b0 * Q8_1_BYTES
            );
            s_x[m][q] = __ldg(x_src + q);
        }

        // 2. Each warp loads its 32 Q4_0 blocks (36 uint4 = 576 bytes) via coalesced 128-bit loads!
        if (valid_row) {
            const uint4* w_src = reinterpret_cast<const uint4*>(w_row + (size_t)b0 * Q4_0_BYTES);
            s_w[warp][lane] = __ldg(w_src + lane);
            if (lane < 4) {
                s_w[warp][32 + lane] = __ldg(w_src + 32 + lane);
            }
        }
        __syncthreads();

        if (valid_row) {
            // Unpack 1 Q4_0 block (18 bytes) for this lane from shared memory
            const uint8_t* wb = reinterpret_cast<const uint8_t*>(s_w[warp]) + lane * Q4_0_BYTES;
            const uint16_t dw_bits = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
            const float dw = __half2float(__ushort_as_half(dw_bits));

            const uint16_t* w16 = reinterpret_cast<const uint16_t*>(wb + 2);
            const uint32_t u0 = (uint32_t)w16[0] | ((uint32_t)w16[1] << 16);
            const uint32_t u1 = (uint32_t)w16[2] | ((uint32_t)w16[3] << 16);
            const uint32_t u2 = (uint32_t)w16[4] | ((uint32_t)w16[5] << 16);
            const uint32_t u3 = (uint32_t)w16[6] | ((uint32_t)w16[7] << 16);

            const int wl0 = (int)(u0 & 0x0f0f0f0fu);
            const int wl1 = (int)(u1 & 0x0f0f0f0fu);
            const int wl2 = (int)(u2 & 0x0f0f0f0fu);
            const int wl3 = (int)(u3 & 0x0f0f0f0fu);
            const int wh0 = (int)((u0 >> 4) & 0x0f0f0f0fu);
            const int wh1 = (int)((u1 >> 4) & 0x0f0f0f0fu);
            const int wh2 = (int)((u2 >> 4) & 0x0f0f0f0fu);
            const int wh3 = (int)((u3 >> 4) & 0x0f0f0f0fu);

            #pragma unroll
            for (int m = 0; m < M_FIXED; ++m) {
                const uint32_t* xb32 = reinterpret_cast<const uint32_t*>(s_x[m]) + lane * 9;
                const half2 ds = *reinterpret_cast<const half2*>(xb32);
                const int* xq = reinterpret_cast<const int*>(xb32 + 1);

                int sumi = 0;
                sumi = dp4a_i8(wl0, xq[0], sumi);
                sumi = dp4a_i8(wl1, xq[1], sumi);
                sumi = dp4a_i8(wl2, xq[2], sumi);
                sumi = dp4a_i8(wl3, xq[3], sumi);
                sumi = dp4a_i8(wh0, xq[4], sumi);
                sumi = dp4a_i8(wh1, xq[5], sumi);
                sumi = dp4a_i8(wh2, xq[6], sumi);
                sumi = dp4a_i8(wh3, xq[7], sumi);

                acc[m] += dw * (__low2float(ds) * (float)sumi - 8.0f * __high2float(ds));
            }
        }
        __syncthreads();
    }

    if (valid_row) {
        #pragma unroll
        for (int m = 0; m < M_FIXED; ++m) {
            const float tot = warp_sum_32(acc[m]);
            if (lane == 0) {
                Y[(size_t)m * N + row] = tot;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) q4_0_q8_1_gemv_m1_sm89(
    const uint8_t* __restrict__ W_q4_0, const uint8_t* __restrict__ X_q8_1, float* __restrict__ Y, int K, int N
) { q4_0_q8_1_gemv_fast_impl<1>(W_q4_0, X_q8_1, Y, K, N); }

extern "C" __global__ void __launch_bounds__(256, 4) q4_0_q8_1_gemv_m7_sm89(
    const uint8_t* __restrict__ W_q4_0, const uint8_t* __restrict__ X_q8_1, float* __restrict__ Y, int K, int N
) { q4_0_q8_1_gemv_fast_impl<7>(W_q4_0, X_q8_1, Y, K, N); }

extern "C" __global__ void __launch_bounds__(256, 4) q4_0_q8_1_gemv_m8_sm89(
    const uint8_t* __restrict__ W_q4_0, const uint8_t* __restrict__ X_q8_1, float* __restrict__ Y, int K, int N
) { q4_0_q8_1_gemv_fast_impl<8>(W_q4_0, X_q8_1, Y, K, N); }

// =============================================================================
// 4 & 5. Exact Parallel-Bitonic Threshold-Filtered Top-64 Selection (2-stage GPU)
//    Stage 1: Grid=(64, M, 1), Block=(128, 1, 1) -> part_val[M, 64*64], part_idx[M, 64*64]
//    Stage 2: Grid=(1,  M, 1), Block=(128, 1, 1) -> out_val[M, 64],     out_idx[M, 64]
//
//    Zero serial thread-0 bottleneck:
//      - Initial 256 elements sorted in parallel via 128-thread bitonic network (~0.14 us)
//      - Subsequent 128-element chunks use single-instruction __syncthreads_or(pass) skip
//      - Passing candidates batch in sh[64..255] and flush via bitonic_sort_256_desc only when cnt >= 64
// =============================================================================
static __device__ __forceinline__ bool topk_better(float va, int ia, float vb, int ib) {
    return (va > vb) || (va == vb && ((ib < 0) || (ia >= 0 && ia < ib)));
}

static __device__ __forceinline__ void bitonic_sort_256_desc(
    float* __restrict__ sh_val,
    int* __restrict__ sh_idx,
    int tid
) {
    #pragma unroll
    for (int k = 2; k <= 256; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int a = ((tid & ~(j - 1)) << 1) | (tid & (j - 1));
            const int b = a | j;
            const float va = sh_val[a];
            const int ia = sh_idx[a];
            const float vb = sh_val[b];
            const int ib = sh_idx[b];
            const bool desc = ((a & k) == 0);
            const bool do_swap = desc ? topk_better(vb, ib, va, ia) : topk_better(va, ia, vb, ib);
            if (do_swap) {
                sh_val[a] = vb; sh_idx[a] = ib;
                sh_val[b] = va; sh_idx[b] = ia;
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 8) topk64_stage1_sm89(
    const float* __restrict__ logits, // [M, N]
    float* __restrict__ part_val,     // [M, TOPK_BLOCKS * TOPK_K]
    int* __restrict__ part_idx,       // [M, TOPK_BLOCKS * TOPK_K]
    int N
) {
    const int blk = blockIdx.x; // 0..63
    const int m = blockIdx.y;   // 0..M-1
    const int tid = threadIdx.x; // 0..127

    const int chunk = (N + TOPK_BLOCKS - 1) / TOPK_BLOCKS;
    const int start = blk * chunk;
    const int end = (start + chunk < N) ? (start + chunk) : N;
    const float* row = logits + (size_t)m * N;

    __shared__ float sh_val[256];
    __shared__ int sh_idx[256];
    __shared__ int sh_cnt;

    // Seed sh[0..255] with first 256 elements and bitonic-sort to establish a tight threshold immediately
    const int i0 = start + tid;
    const int i1 = start + tid + 128;
    sh_val[tid] = (i0 < end) ? row[i0] : -3.4028234e38f;
    sh_idx[tid] = (i0 < end) ? i0 : -1;
    sh_val[tid + 128] = (i1 < end) ? row[i1] : -3.4028234e38f;
    sh_idx[tid + 128] = (i1 < end) ? i1 : -1;
    if (tid == 0) sh_cnt = 0;
    __syncthreads();

    bitonic_sort_256_desc(sh_val, sh_idx, tid);

    for (int base = start + 256; base < end; base += 128) {
        const float thresh = sh_val[TOPK_K - 1];
        const int thresh_id = sh_idx[TOPK_K - 1];
        const int idx = base + tid;
        float v = -3.4028234e38f;
        bool pass = false;
        if (idx < end) {
            v = row[idx];
            pass = topk_better(v, idx, thresh, thresh_id);
        }
        if (!__syncthreads_or(pass)) continue;

        if (pass) {
            const int slot = atomicAdd(&sh_cnt, 1);
            sh_val[TOPK_K + slot] = v;
            sh_idx[TOPK_K + slot] = idx;
        }
        __syncthreads();

        const int cnt = sh_cnt;
        if (cnt >= 64) {
            for (int p = TOPK_K + tid; p < 256; p += 128) {
                if (p >= TOPK_K + cnt) {
                    sh_val[p] = -3.4028234e38f;
                    sh_idx[p] = -1;
                }
            }
            __syncthreads();
            bitonic_sort_256_desc(sh_val, sh_idx, tid);
            if (tid == 0) sh_cnt = 0;
            __syncthreads();
        }
    }

    const int rem = sh_cnt;
    if (rem > 0) {
        for (int p = TOPK_K + tid; p < 256; p += 128) {
            if (p >= TOPK_K + rem) {
                sh_val[p] = -3.4028234e38f;
                sh_idx[p] = -1;
            }
        }
        __syncthreads();
        bitonic_sort_256_desc(sh_val, sh_idx, tid);
    }

    if (tid < TOPK_K) {
        const size_t out_off = ((size_t)m * TOPK_BLOCKS + blk) * TOPK_K + tid;
        part_val[out_off] = sh_val[tid];
        part_idx[out_off] = sh_idx[tid];
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) topk64_stage2_sm89(
    const float* __restrict__ part_val, // [M, TOPK_BLOCKS * TOPK_K]
    const int* __restrict__ part_idx,   // [M, TOPK_BLOCKS * TOPK_K]
    float* __restrict__ out_val,        // [M, TOPK_K]
    int* __restrict__ out_idx           // [M, TOPK_K]
) {
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    const int total = TOPK_BLOCKS * TOPK_K; // 4096
    const float* pval = part_val + (size_t)m * total;
    const int* pidx = part_idx + (size_t)m * total;

    __shared__ float sh_val[256];
    __shared__ int sh_idx[256];
    __shared__ int sh_cnt;

    const int id0 = pidx[tid];
    const int id1 = pidx[tid + 128];
    sh_val[tid] = (id0 >= 0) ? pval[tid] : -3.4028234e38f;
    sh_idx[tid] = id0;
    sh_val[tid + 128] = (id1 >= 0) ? pval[tid + 128] : -3.4028234e38f;
    sh_idx[tid + 128] = id1;
    if (tid == 0) sh_cnt = 0;
    __syncthreads();

    bitonic_sort_256_desc(sh_val, sh_idx, tid);

    for (int base = 256; base < total; base += 128) {
        const float thresh = sh_val[TOPK_K - 1];
        const int thresh_id = sh_idx[TOPK_K - 1];
        const int i = base + tid;
        const int id = pidx[i];
        float v = -3.4028234e38f;
        bool pass = false;
        if (id >= 0) {
            v = pval[i];
            pass = topk_better(v, id, thresh, thresh_id);
        }
        if (!__syncthreads_or(pass)) continue;

        if (pass) {
            const int slot = atomicAdd(&sh_cnt, 1);
            sh_val[TOPK_K + slot] = v;
            sh_idx[TOPK_K + slot] = id;
        }
        __syncthreads();

        const int cnt = sh_cnt;
        if (cnt >= 64) {
            for (int p = TOPK_K + tid; p < 256; p += 128) {
                if (p >= TOPK_K + cnt) {
                    sh_val[p] = -3.4028234e38f;
                    sh_idx[p] = -1;
                }
            }
            __syncthreads();
            bitonic_sort_256_desc(sh_val, sh_idx, tid);
            if (tid == 0) sh_cnt = 0;
            __syncthreads();
        }
    }

    const int rem = sh_cnt;
    if (rem > 0) {
        for (int p = TOPK_K + tid; p < 256; p += 128) {
            if (p >= TOPK_K + rem) {
                sh_val[p] = -3.4028234e38f;
                sh_idx[p] = -1;
            }
        }
        __syncthreads();
        bitonic_sort_256_desc(sh_val, sh_idx, tid);
    }

    if (tid < TOPK_K) {
        out_val[(size_t)m * TOPK_K + tid] = sh_val[tid];
        out_idx[(size_t)m * TOPK_K + tid] = sh_idx[tid];
    }
}

// =============================================================================
// 6. Stage-2 Sparse Exact LM Head Rescore: Q8_0 (output.weight) x Q8_1 (X) via __dp4a
//    Rescores ONLY the K_cand candidate rows (e.g., 64 rows = 348 KiB vs 1.35 GiB!)
//    Grid : (K_cand, M, 1) — 1 warp computes exact dot product for (m, cand_ids[m, k])
//    Block: (32, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(32, 16) q8_0_sparse_rescore_sm89(
    const uint8_t* __restrict__ W_q8_0,  // [vocab, K/32 * 34]
    const uint8_t* __restrict__ X_q8_1,  // [M, K/32 * 36]
    const int* __restrict__ cand_ids,    // [M, K_cand]
    float* __restrict__ cand_logits,     // [M, K_cand]
    int K,
    int K_cand
) {
    const int ck = blockIdx.x;
    const int m = blockIdx.y;
    const int lane = threadIdx.x;

    const int row_id = cand_ids[(size_t)m * K_cand + ck];
    if (row_id < 0) {
        if (lane == 0) cand_logits[(size_t)m * K_cand + ck] = -3.4028234e38f;
        return;
    }

    const int nb = K >> 5;
    const uint8_t* w_row = W_q8_0 + (size_t)row_id * ((size_t)nb * Q8_0_BYTES);
    const uint8_t* x_row = X_q8_1 + (size_t)m * ((size_t)nb * Q8_1_BYTES);

    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* wb = w_row + (size_t)b * Q8_0_BYTES;
        const uint16_t dw_bits = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
        const float dw = __half2float(__ushort_as_half(dw_bits));

        // Load 32 int8 weights (2-byte aligned -> 16 uint16 loads -> 8 int32 words)
        const uint16_t* w16 = reinterpret_cast<const uint16_t*>(wb + 2);
        const uint8_t* xb = x_row + (size_t)b * Q8_1_BYTES;
        const half2 ds = *reinterpret_cast<const half2*>(xb);
        const int* xq = reinterpret_cast<const int*>(xb + 4);

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int wword = (int)((uint32_t)w16[2 * j] | ((uint32_t)w16[2 * j + 1] << 16));
            sumi = dp4a_i8(wword, xq[j], sumi);
        }
        acc += dw * __low2float(ds) * (float)sumi;
    }

    const float tot = warp_sum_32(acc);
    if (lane == 0) {
        cand_logits[(size_t)m * K_cand + ck] = tot;
    }
}
