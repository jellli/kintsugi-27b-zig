// Native sm_89 (RTX 4070 Ti, Ada Lovelace) Escha ~2.5-bit Trellis GEMV kernels.
//
// Derived from exllamav3 (Copyright (c) 2025 Turboderp, MIT) and Kintsugi-27B /
// llama.cpp-escha (MIT). Compiled directly to sm_89 CUBIN via NVRTC with zero
// ggml / cooperative_groups / host C++ runtime dependencies.
//
// Computes:  y[m, oc] = T128( T128(x[m, :] * rin[:]) @ decode_mcg(code) ) * rout[oc]
// Layout of code: [IC/16][OC/16][16*K] uint16 (K in {2, 3}), identical to GGUF escha_code.

#include <cuda_fp16.h>

typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ESCHA_GEMV_RS 0.088388347648f // 1/sqrt(128)
#define EXL3_GEMV_MAX_M 8
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))

struct __align__(8) half4 {
    half2 x;
    half2 y;
};

template <typename T, int N>
struct Vec {
    T elems[N];
    __device__ __forceinline__ T& operator[](int i) { return elems[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return elems[i]; }
};

using FragB = Vec<half2, 2>;
using FragC_h = Vec<half2, 2>;

static __device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int shift) {
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

static __device__ __forceinline__ uint32_t bfe16_imm(uint32_t src, int pos) {
    return (src >> pos) & 0xffffu;
}

// MCG codebook (cb = 1): x -> ((x * 0xCBAC1FED) & 0x8fff8fff) ^ 0x3b603b60, sum low+high fp16
static __device__ __forceinline__ half2 decode_mcg_pair(uint32_t w0, uint32_t w1) {
    uint32_t x0 = w0 * 0xCBAC1FEDu;
    uint32_t x1 = w1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    half2 h0 = *reinterpret_cast<const half2*>(&x0);
    half2 h1 = *reinterpret_cast<const half2*>(&x1);
    return __hadd2(__lows2half2(h0, h1), __highs2half2(h0, h1));
}

static __device__ __forceinline__ void decode8_mcg(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
    FragB& f0, FragB& f1
) {
    f0[0] = decode_mcg_pair(w0, w1);
    f0[1] = decode_mcg_pair(w2, w3);
    f1[0] = decode_mcg_pair(w4, w5);
    f1[1] = decode_mcg_pair(w6, w7);
}

static __device__ __forceinline__ void dq8_regs_2bits_mcg(
    uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1
) {
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    uint32_t w7 = b & 0xffffu;
    uint32_t w6 = bfe16_imm(b, 2);
    uint32_t w5 = bfe16_imm(b, 4);
    uint32_t w4 = bfe16_imm(b, 6);
    uint32_t w3 = bfe16_imm(b, 8);
    uint32_t w2 = bfe16_imm(b, 10);
    uint32_t w1 = bfe16_imm(b, 12);
    uint32_t w0 = bfe16_imm(b, 14);
    decode8_mcg(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

static __device__ __forceinline__ void dq8_regs_3bits_mcg(
    uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1
) {
    uint32_t w7 = fshift(b, a, s2);
    uint32_t w6 = w7 >> 3;
    uint32_t w5 = w6 >> 3;
    uint32_t w4 = w5 >> 3;
    uint32_t w3 = fshift(b, a, s2 + 12);
    uint32_t w2 = w3 >> 3;
    uint32_t w1 = w2 >> 3;
    uint32_t w0 = w1 >> 3;
    decode8_mcg(
        w0 & 0xffffu, w1 & 0xffffu, w2 & 0xffffu, w3 & 0xffffu,
        w4 & 0xffffu, w5 & 0xffffu, w6 & 0xffffu, w7 & 0xffffu,
        f0, f1
    );
}

// Ada / Ampere m16n8k16 FP16 Tensor Core MMA (FP16 accumulate, periodically folded to FP32)
static __device__ __forceinline__ void mma_ab_h(
    const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c
) {
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
          "r"(bb[0]), "r"(bb[1])
    );
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

// -----------------------------------------------------------------------------
// Input Hadamard rotation: xh[m, IC] = f16(T128(x[m, IC] * rin[IC]))
// Grid: ceil(n_rows * (IC/128) / 8), Block: 256
// -----------------------------------------------------------------------------
extern "C" __global__ void escha_had_in_f32_f16(
    const float* __restrict__ x,
    const __half* __restrict__ rin,
    __half* __restrict__ xh,
    int IC,
    int n_rows
) {
    const int nb = IC >> 7;
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= n_rows * nb) return;
    const int row = w / nb;
    const int blk = w % nb;
    const int lane = threadIdx.x & 31;

    float4 v = ((const float4*)(x + (size_t)row * IC + (blk << 7)))[lane];
    const half4 r = ((const half4*)(rin + (blk << 7)))[lane];
    v.x *= __low2float(r.x);
    v.y *= __high2float(r.x);
    v.z *= __low2float(r.y);
    v.w *= __high2float(r.y);

    had_lane_128(v, lane);

    half4 h;
    h.x = __floats2half2_rn(v.x * ESCHA_GEMV_RS, v.y * ESCHA_GEMV_RS);
    h.y = __floats2half2_rn(v.z * ESCHA_GEMV_RS, v.w * ESCHA_GEMV_RS);
    ((half4*)(xh + (size_t)row * IC + (blk << 7)))[lane] = h;
}

// -----------------------------------------------------------------------------
// Output Hadamard rotation: dst[m, OC] = T128(mid[m, OC]) * rout[OC]
// Grid: ceil(n_rows * (OC/128) / 8), Block: 256
// -----------------------------------------------------------------------------
extern "C" __global__ void escha_had_out_f32(
    const float* __restrict__ mid,
    const __half* __restrict__ rout,
    float* __restrict__ dst,
    int OC,
    int n_rows
) {
    const int nb = OC >> 7;
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= n_rows * nb) return;
    const int row = w / nb;
    const int blk = w % nb;
    const int lane = threadIdx.x & 31;

    float4 v = ((const float4*)(mid + (size_t)row * OC + (blk << 7)))[lane];
    had_lane_128(v, lane);
    const half4 r = ((const half4*)(rout + (blk << 7)))[lane];
    v.x = v.x * ESCHA_GEMV_RS * __low2float(r.x);
    v.y = v.y * ESCHA_GEMV_RS * __high2float(r.x);
    v.z = v.z * ESCHA_GEMV_RS * __low2float(r.y);
    v.w = v.w * ESCHA_GEMV_RS * __high2float(r.y);

    ((float4*)(dst + (size_t)row * OC + (blk << 7)))[lane] = v;
}

// -----------------------------------------------------------------------------
// Core parameterized Escha Trellis GEMV (contiguous row-major A[M, IC] -> C[M, OC])
//   bits  in {2, 3}
//   MMODE 0 => M==1 fast path, MMODE 1 => 2 <= M <= 8
//   WNT   in {2, 4}: 16-col tiles per warp
//   S     in {4, 8, 16}: K-split warps per column group
//   FUSE  true => G=8/WNT groups (128 cols/block), applies T128 + rout in smem epilogue
// -----------------------------------------------------------------------------
template <int bits, int MMODE, int WNT, int S, bool FUSE, int FOLD_OVERRIDE = 0>
static __device__ __forceinline__ void escha_gemv_impl(
    const __half* __restrict__ A,
    const uint16_t* __restrict__ B,
    const __half* __restrict__ rout,
    float* __restrict__ C,
    int size_m,
    int size_k,
    int size_n
) {
    constexpr int G = FUSE ? (8 / WNT) : 1;
    constexpr int PF = (WNT <= 2) ? 4 : 2;
    constexpr int FOLD = (FOLD_OVERRIDE > 0) ? FOLD_OVERRIDE : ((WNT <= 2) ? 4 : 2);
    constexpr int THREADS = 32 * S * G;
    constexpr int ROWS = (MMODE == 0) ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;
    constexpr int BCOLS = G * COLS;

    constexpr int TWORDS = 8 * bits;
    constexpr bool TWO_PER_LOAD = (bits == 2);
    constexpr int LOADS = TWO_PER_LOAD ? (WNT / 2) : WNT;
    constexpr int LSTRIDE = TWO_PER_LOAD ? (2 * TWORDS) : TWORDS;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int grp = warp / S;
    const int ksp = warp % S;

    const int ntiles = size_n >> 4;
    const int kslices = size_k >> 4;

    const int chunk = CEIL_DIVIDE(kslices, S);
    const int ks0 = ksp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
    const size_t slice_stride = (size_t)ntiles * TWORDS;
    const half2* A2 = reinterpret_cast<const half2*>(A);
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t)r0 * (size_k >> 1);
    const bool r0_ok = (MMODE == 0) ? (lane < 4) : (r0 < size_m);

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        const int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ __align__(16) float sh_red[S][ROWS][BCOLS];

    const int group = blockIdx.x * G + grp;
    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + (size_t)group * WNT * TWORDS + lane;

    auto ld_b = [&](int i, int l) -> uint32_t {
        if constexpr (LSTRIDE < 32) {
            return (lane < LSTRIDE) ? __ldcs(bp + (size_t)i * slice_stride + l * LSTRIDE) : 0u;
        } else {
            return __ldcs(bp + (size_t)i * slice_stride + l * LSTRIDE);
        }
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d) {
        if (d < myn) {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);
        }
    }

    FragC_h ch[WNT][2];
    float2 acc0[WNT][2];
    #pragma unroll
    for (int t = 0; t < WNT; ++t) {
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            ch[t][f][0] = hzero;
            ch[t][f][1] = hzero;
            acc0[t][f] = make_float2(0.0f, 0.0f);
        }
    }

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    const uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    const uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    dq8_regs_2bits_mcg(awv, bwv, lane << 3, f0, f1);
                } else {
                    const uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    const uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    dq8_regs_3bits_mcg(awv, bwv, x_s2, f0, f1);
                }
                mma_ab_h(a01, a23, f0, ch[t][0]);
                mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((i + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
                }
            }
        }
    }

    // Store per-split accumulators into shared memory
    {
        const int c0 = grp * COLS + 2 * (lane & 3);
        const bool store0 = (MMODE == 0) ? (lane < 4) : (r0 < ROWS);
        const int sr0 = (MMODE == 0) ? 0 : r0;
        if (store0) {
            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + c0;
                    sh_red[ksp][sr0][col + 0] = acc0[t][f].x;
                    sh_red[ksp][sr0][col + 1] = acc0[t][f].y;
                }
            }
        }
    }
    __syncthreads();

    const int rows_out = (MMODE == 0) ? 1 : min(size_m, ROWS);
    for (int idx = threadIdx.x; idx < BCOLS * rows_out; idx += THREADS) {
        const int r = idx / BCOLS;
        const int c = idx % BCOLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < S; ++j) sum += sh_red[j][r][c];
        if constexpr (FUSE) {
            sh_red[0][r][c] = sum;
        } else {
            C[(size_t)r * size_n + blockIdx.x * BCOLS + c] = sum;
        }
    }

    if constexpr (FUSE) {
        __syncthreads();
        if (warp < rows_out) {
            float4 v = ((const float4*)sh_red[0][warp])[lane];
            had_lane_128(v, lane);
            const half4 r = ((const half4*)(rout + (size_t)blockIdx.x * 128))[lane];
            v.x = v.x * ESCHA_GEMV_RS * __low2float(r.x);
            v.y = v.y * ESCHA_GEMV_RS * __high2float(r.x);
            v.z = v.z * ESCHA_GEMV_RS * __low2float(r.y);
            v.w = v.w * ESCHA_GEMV_RS * __high2float(r.y);
            ((float4*)(C + (size_t)warp * size_n + (size_t)blockIdx.x * 128))[lane] = v;
        }
    }
}

// =============================================================================
// Exported C entrypoints (Zig calls directly via cuLaunchKernel)
// =============================================================================

// 1. Kintsugi reference fused geometry: WNT=2, S=8, G=4 -> 1024 threads, grid OC/128
extern "C" __global__ void __launch_bounds__(1024, 1) escha_gemv_k2_m1_base(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 0, 2, 8, true>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(1024, 1) escha_gemv_k2_m8_base(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 2, 8, true>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(1024, 1) escha_gemv_k3_m1_base(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 0, 2, 8, true>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(1024, 1) escha_gemv_k3_m8_base(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 2, 8, true>(A, B, rout, C, m, k, n); }

// 2. RTX 4070 Ti (sm_89) fused 2-block/SM geometry: WNT=4, S=8, G=2 -> 512 threads, grid OC/128
//    Note: keeps S=8 K-split count! With FOLD=4 (matching WNT=2 fold cadence), reduction is bit-identical!
extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k2_m8_sm89_w4s8(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 4, 8, true, 4>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k3_m8_sm89_w4s8(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 4, 8, true, 4>(A, B, rout, C, m, k, n); }

// 3. RTX 4070 Ti (sm_89) compact fused geometry: WNT=4, S=4, G=2 -> 256 threads, grid OC/128
extern "C" __global__ void __launch_bounds__(256, 4) escha_gemv_k2_m8_sm89_w4s4(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 4, 4, true>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(256, 4) escha_gemv_k3_m8_sm89_w4s4(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 4, 4, true>(A, B, rout, C, m, k, n); }

// 4. High-SM-saturation narrow unfused: WNT=2, S=16, G=1 -> 512 threads, grid OC/32 (160 blocks @ OC=5120)
extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k2_m8_narrow_s16(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 2, 16, false>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k3_m8_narrow_s16(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 2, 16, false>(A, B, rout, C, m, k, n); }

// 5. Single-wave RTX 4070 Ti (60 SMs) narrow unfused: WNT=2, S=8, G=1 -> 256 threads, 8 KiB smem, grid OC/32
//    Up to 4 blocks/SM = 240 blocks/wave -> OC=5120 (160 blocks) fits in ONE wave! Also keeps S=8, FOLD=4 (bit-exact!).
extern "C" __global__ void __launch_bounds__(256, 4) escha_gemv_k2_m8_narrow_s8(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 2, 8, false>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(256, 4) escha_gemv_k3_m8_narrow_s8(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 2, 8, false>(A, B, rout, C, m, k, n); }

// 6. Ultra-compact 128-thread narrow unfused: WNT=2, S=4, G=1 -> 128 threads, 4 KiB smem, grid OC/32
extern "C" __global__ void __launch_bounds__(128, 8) escha_gemv_k2_m8_narrow_s4(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 2, 4, false>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(128, 8) escha_gemv_k3_m8_narrow_s4(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 2, 4, false>(A, B, rout, C, m, k, n); }

// 7. Full-60-SM saturation for narrow K=3 projections (OC=1024 attn_k/v): WNT=1, S=16, G=1 -> 512t, grid OC/16 (64 blocks)
extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k3_m8_narrow_w1s16(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 1, 16, false>(A, B, rout, C, m, k, n); }

// 8. M=1 single-token high-SM-saturation narrow unfused: WNT=2, S=16, G=1 -> 512 threads, grid OC/32
extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k2_m1_narrow_s16(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 0, 2, 16, false>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k3_m1_narrow_s16(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 0, 2, 16, false>(A, B, rout, C, m, k, n); }
