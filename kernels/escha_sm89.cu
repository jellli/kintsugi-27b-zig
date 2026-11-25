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

// 9. Fused 512t wide-OC geometry (OC>=10240: ffn_gate/up, attn_qkv, attn_q): WNT=2, S=4, G=4 -> 512t, grid OC/128 (zero hadOut!)
extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k2_m8_fused_w2s4(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 2, 4, true>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(512, 2) escha_gemv_k3_m8_fused_w2s4(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 2, 4, true>(A, B, rout, C, m, k, n); }

// 10. 3-block/SM Ada single-wave geometry for OC=5120 (160 <= 180 blks/wave): WNT=2, S=12, G=1 -> 384t, grid OC/32
extern "C" __global__ void __launch_bounds__(384, 3) escha_gemv_k2_m8_narrow_s12(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<2, 1, 2, 12, false>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(384, 3) escha_gemv_k3_m8_narrow_s12(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemv_impl<3, 1, 2, 12, false>(A, B, rout, C, m, k, n); }

// =============================================================================
// 11. Batched Tensor Core Escha Prefill GEMM (64x32 output tile, arbitrary M via grid.y)
//     - WNT=2 (32 cols/block, full 32-lane 128B coalesced B loads for 2-bit, 2x A reuse!)
//     - MT=4 (64 rows/block -> reads 7.24 GB weights ONCE per 64 tokens!)
//     - S=4 K-splits (128 threads/block, 8 KiB smem, 0 spills, ZERO inner barriers!)
//     - Full m16n8k16 FP32-acc Tensor Core MMAs (all 16 rows active, no FOLD overhead!)
//       with uniform active-tile skip when m < 64.
//     Grid : (OC / 32, ceil(M / 64), 1), Block: (128, 1, 1)
// =============================================================================
static __device__ __forceinline__ void mma_ab_f32(
    const FragB& a01, const FragB& a23, const FragB& b, float4& c
) {
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
          "r"(bb[0]), "r"(bb[1])
    );
}

template <int bits, int WNT, int S, int MT>
static __device__ __forceinline__ void escha_gemm_impl(
    const __half* __restrict__ A,
    const uint16_t* __restrict__ B,
    const __half* __restrict__ rout,
    float* __restrict__ C,
    int size_m,
    int size_k,
    int size_n
) {
    (void)rout;
    constexpr int PF = 2;
    constexpr int THREADS = 32 * S;
    constexpr int COLS = WNT * 16; // 32
    constexpr int TWORDS = 8 * bits;
    constexpr bool TWO_PER_LOAD = (bits == 2);
    constexpr int LOADS = TWO_PER_LOAD ? (WNT / 2) : WNT;
    constexpr int LSTRIDE = TWO_PER_LOAD ? (2 * TWORDS) : TWORDS;

    const int ksp = threadIdx.x >> 5; // 0..S-1
    const int lane = threadIdx.x & 31;

    const int m_base = blockIdx.y * (MT * 16);
    const int ntiles = size_n >> 4;
    const int kslices = size_k >> 4;

    const int chunk = CEIL_DIVIDE(kslices, S);
    const int ks0 = ksp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
    const size_t slice_stride = (size_t)ntiles * TWORDS;
    const half2* A2 = reinterpret_cast<const half2*>(A);
    const half2 hzero = __half2half2(__ushort_as_half(0));
    const size_t a_stride = (size_t)(size_k >> 1);

    const int r0 = lane >> 2; // 0..7
    const int r1 = r0 + 8;    // 8..15

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

    const uint32_t* bp = B32 + (size_t)ks0 * slice_stride + (size_t)blockIdx.x * WNT * TWORDS + lane;

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

    float4 acc[MT][WNT][2];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            acc[mt][t][0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            acc[mt][t][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

            // Decode WNT weight tiles ONCE and reuse across all active MT*16 rows!
            FragB wf[WNT][2];
            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    const uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    const uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    dq8_regs_2bits_mcg(awv, bwv, lane << 3, wf[t][0], wf[t][1]);
                } else {
                    const uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    const uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    dq8_regs_3bits_mcg(awv, bwv, x_s2, wf[t][0], wf[t][1]);
                }
            }

            const size_t a_col = (size_t)(ks0 + i) * 8 + (lane & 3);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int tile_row0 = m_base + mt * 16;
                if (tile_row0 >= size_m) break;
                const int row0 = tile_row0 + r0;
                const int row1 = tile_row0 + r1;
                FragB a01, a23;
                if (row0 < size_m) {
                    const size_t off0 = (size_t)row0 * a_stride + a_col;
                    a01[0] = A2[off0];
                    a23[0] = A2[off0 + 4];
                } else {
                    a01[0] = hzero;
                    a23[0] = hzero;
                }
                if (row1 < size_m) {
                    const size_t off1 = (size_t)row1 * a_stride + a_col;
                    a01[1] = A2[off1];
                    a23[1] = A2[off1 + 4];
                } else {
                    a01[1] = hzero;
                    a23[1] = hzero;
                }
                #pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    mma_ab_f32(a01, a23, wf[t][0], acc[mt][t][0]);
                    mma_ab_f32(a01, a23, wf[t][1], acc[mt][t][1]);
                }
            }
        }
    }

    // Reduce S K-splits one 16x32 tile at a time (only S * 16 * 32 * 4 = 8 KiB smem!)
    __shared__ __align__(16) float sh_red[S][16][COLS];
    const int c0 = 2 * (lane & 3);
    const size_t col_base = (size_t)blockIdx.x * COLS;

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int tile_row0 = m_base + mt * 16;
        if (tile_row0 >= size_m) break;

        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + c0;
                sh_red[ksp][r0][col + 0] = acc[mt][t][f].x;
                sh_red[ksp][r0][col + 1] = acc[mt][t][f].y;
                sh_red[ksp][r1][col + 0] = acc[mt][t][f].z;
                sh_red[ksp][r1][col + 1] = acc[mt][t][f].w;
            }
        }
        __syncthreads();

        const int rows_here = min(16, size_m - tile_row0);
        for (int idx = threadIdx.x; idx < rows_here * COLS; idx += THREADS) {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < S; ++j) sum += sh_red[j][r][c];
            C[(size_t)(tile_row0 + r) * size_n + col_base + c] = sum;
        }
        if (mt + 1 < MT) __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) escha_gemm_k2_m64(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemm_impl<2, 2, 4, 4>(A, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(128, 4) escha_gemm_k3_m64(
    const __half* A, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemm_impl<3, 2, 4, 4>(A, B, rout, C, m, k, n); }

// =============================================================================
// 12. Native sm_89 316-TOPS INT8 Tensor Core (m16n8k32) Escha Prefill GEMM
//     - Exhaustive property of MCG codebook: ALL 65,536 codes lie in [-3.9502, +3.9502].
//       Scaling by exact power-of-2 32.0 maps every weight into [-126, +126] (zero clipping,
//       zero per-block/per-channel weight scales, cos = 0.999974!).
//     - Per-row symmetric INT8 quantization of Hadamard-rotated FP16 activations A_had
//       (Gaussian, no outliers -> cos > 0.9999).
//     - mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 runs at 315.9 TOPS on RTX 4070 Ti
//       (4x FP32-acc, 2x FP16-acc!) with ZERO folding overhead (exact INT32 accumulate).
//     - Two geometries (both 0 bytes local register spill, bit-identical outputs):
//       * m128n64 (BM=128, BN=64, BK=64, 30 KiB smem -> 3 blks/SM): best for M<=128 or OC<6144
//       * m256n64 (BM=256, BN=64, BK=64, 42 KiB swizzled smem -> 2 blks/SM): best for M>128 & OC>=6144
// =============================================================================
static __device__ __forceinline__ void escha_cp_async_16_zfill(
    void* smem_ptr, const void* glob_ptr, int src_size
) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(glob_ptr), "r"(src_size));
}

static __device__ __forceinline__ void escha_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

static __device__ __forceinline__ void escha_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

static __device__ __forceinline__ void escha_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr
) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}
extern "C" __global__ void __launch_bounds__(256, 4) escha_quant_f16_to_i8(
    const __half* __restrict__ A_f16, // [M, K]
    signed char* __restrict__ A_i8,   // [M, K]
    float* __restrict__ scale_a,      // [M]
    int size_k
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x; // 0..255
    const int k8 = size_k >> 3;  // chunks of 8 halfs (16B)

    const uint4* src = reinterpret_cast<const uint4*>(A_f16 + (size_t)row * size_k);
    float amax = 0.0f;
    for (int i = tid; i < k8; i += 256) {
        const uint4 v = src[i];
        const half2* h = reinterpret_cast<const half2*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = fabsf(__low2float(h[j]));
            const float b = fabsf(__high2float(h[j]));
            amax = fmaxf(amax, fmaxf(a, b));
        }
    }

    // Warp + block max reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    __shared__ float sh_max[8];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) sh_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float bm = (lane < 8) ? sh_max[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off));
        }
        if (lane == 0) sh_max[0] = bm;
    }
    __syncthreads();

    const float row_max = sh_max[0];
    const float sc = (row_max > 0.0f) ? (row_max * (1.0f / 127.0f)) : 1.0f;
    const float inv_sc = (row_max > 0.0f) ? (127.0f / row_max) : 0.0f;
    if (tid == 0) scale_a[row] = sc;

    uint2* dst = reinterpret_cast<uint2*>(A_i8 + (size_t)row * size_k);
    for (int i = tid; i < k8; i += 256) {
        const uint4 v = src[i];
        const half2* h = reinterpret_cast<const half2*>(&v);
        int q[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[2 * j + 0] = __float2int_rn(__low2float(h[j]) * inv_sc);
            q[2 * j + 1] = __float2int_rn(__high2float(h[j]) * inv_sc);
        }
        const uint32_t p0 = (q[0] & 0xffu) | ((q[1] & 0xffu) << 8) | ((q[2] & 0xffu) << 16) | ((q[3] & 0xffu) << 24);
        const uint32_t p1 = (q[4] & 0xffu) | ((q[5] & 0xffu) << 8) | ((q[6] & 0xffu) << 16) | ((q[7] & 0xffu) << 24);
        dst[i] = make_uint2(p0, p1);
    }
}

static __device__ __forceinline__ void escha_mma_s8_u32(
    int4& d,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d.x), "+r"(d.y), "+r"(d.z), "+r"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

static __device__ __forceinline__ uint16_t half2_to_i8x2_s32(half2 h) {
    const half2 hs = __hmul2(h, __floats2half2_rn(32.0f, 32.0f));
    const int lo = __half2int_rn(__low2half(hs));
    const int hi = __half2int_rn(__high2half(hs));
    return static_cast<uint16_t>((lo & 0xff) | ((hi & 0xff) << 8));
}

template <int bits>
static __device__ __forceinline__ void escha_gemm_i8_m128n64_impl(
    const signed char* __restrict__ A_i8,
    const float* __restrict__ scale_a,
    const uint16_t* __restrict__ B,
    const __half* __restrict__ rout,
    float* __restrict__ C,
    int size_m,
    int size_k,
    int size_n
) {
    (void)rout;
    constexpr int TWORDS = 8 * bits; // 16 for 2-bit, 24 for 3-bit
    constexpr int LD_B = 80;         // 64 + 16 bytes (20 banks/row -> 0% bank conflict for ldmatrix.x4!)

    const int tid = threadIdx.x;     // 0..255
    const int warp = tid >> 5;       // 0..7
    const int lane = tid & 31;       // 0..31

    const int m_base = blockIdx.y * 128;
    const int n_base = blockIdx.x * 64;
    const int ntiles = size_n >> 4;
    const int k_iters = size_k >> 6; // number of BK=64 iterations (K is multiple of 128)

    const int warp_m = warp >> 1;    // 0..3 (32 rows/warp)
    const int warp_n = warp & 1;     // 0..1 (32 cols/warp)
    const int wm_base = warp_m * 32;
    const int wn_base = warp_n * 32;

    // 8 warps cooperatively decode 4 K-slices x 4 N-tiles = 16 weight tiles (2 tiles/warp)!
    const int b_ks_pair = (warp >> 2) * 2; // 0 (ks 0,1) or 2 (ks 2,3)
    const int b_nt_loc = warp & 3;         // 0..3
    const int b_nt_glob = blockIdx.x * 4 + b_nt_loc;

    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
    const size_t slice_stride = (size_t)ntiles * TWORDS;

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

    // Shared memory: 20 KiB (sh_a) + 10 KiB (sh_b) = 30 KiB -> 2-3 blocks/SM
    __shared__ __align__(16) signed char sh_a[2][128][LD_B];
    __shared__ __align__(16) signed char sh_b[2][64][LD_B];

    // Precompute cp.async A-tile coordinates (256 threads x 2 chunks of 16B = 128 x 64 bytes)
    const int cp_r0 = tid >> 2;         // 0..63
    const int cp_r1 = cp_r0 + 64;       // 64..127
    const int cp_c = (tid & 3) * 16;    // 0, 16, 32, 48 bytes
    const int g_r0 = m_base + cp_r0;
    const int g_r1 = m_base + cp_r1;
    const int sz0 = (g_r0 < size_m) ? 16 : 0;
    const int sz1 = (g_r1 < size_m) ? 16 : 0;
    const signed char* a_row_ptr0 = (g_r0 < size_m) ? (A_i8 + (size_t)g_r0 * size_k + cp_c) : A_i8;
    const signed char* a_row_ptr1 = (g_r1 < size_m) ? (A_i8 + (size_t)g_r1 * size_k + cp_c) : A_i8;

    auto issue_a = [&](int ki, int stage) {
        const size_t k_off = (size_t)ki * 64;
        escha_cp_async_16_zfill(&sh_a[stage][cp_r0][cp_c], (sz0 ? (a_row_ptr0 + k_off) : A_i8), sz0);
        escha_cp_async_16_zfill(&sh_a[stage][cp_r1][cp_c], (sz1 ? (a_row_ptr1 + k_off) : A_i8), sz1);
        escha_cp_async_commit();
    };

    const uint32_t* b_warp_base0 = B32 + (size_t)(b_ks_pair + 0) * slice_stride + (size_t)b_nt_glob * TWORDS + lane;
    const uint32_t* b_warp_base1 = B32 + (size_t)(b_ks_pair + 1) * slice_stride + (size_t)b_nt_glob * TWORDS + lane;
    auto ld_b_pair = [&](int ki, uint32_t& w0, uint32_t& w1) {
        const size_t off = (size_t)(ki * 4) * slice_stride;
        if (lane < TWORDS) {
            w0 = __ldcs(b_warp_base0 + off);
            w1 = __ldcs(b_warp_base1 + off);
        } else {
            w0 = 0u;
            w1 = 0u;
        }
    };

    const int bn0 = b_nt_loc * 16 + (lane >> 2);
    const int bk_lane = (lane & 3) * 2;
    auto decode_one_to_smem = [&](uint32_t w, int stage, int ks_loc) {
        FragB wf0, wf1;
        if constexpr (bits == 2) {
            const uint32_t bwv = __shfl_sync(0xffffffffu, w, x_src_b);
            const uint32_t awv = __shfl_sync(0xffffffffu, w, x_src_a);
            dq8_regs_2bits_mcg(awv, bwv, lane << 3, wf0, wf1);
        } else {
            const uint32_t awv = __shfl_sync(0xffffffffu, w, x_src_a);
            const uint32_t bwv = __shfl_sync(0xffffffffu, w, x_src_b);
            dq8_regs_3bits_mcg(awv, bwv, x_s2, wf0, wf1);
        }
        const int bk0 = ks_loc * 16 + bk_lane;
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0    ][bk0    ]) = half2_to_i8x2_s32(wf0[0]);
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0    ][bk0 + 8]) = half2_to_i8x2_s32(wf0[1]);
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0 + 8][bk0    ]) = half2_to_i8x2_s32(wf1[0]);
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0 + 8][bk0 + 8]) = half2_to_i8x2_s32(wf1[1]);
    };

    // Precompute ldmatrix.x4 byte offsets (identical geometry to FP16!)
    const int a_ld_row = lane & 15;
    const int a_ld_col = (lane >> 4) * 16; // 0 or 16 bytes
    const int mat = lane >> 3;
    const int b_ld_row = ((mat >> 1) * 8) + (lane & 7);
    const int b_ld_col = (mat & 1) * 16;   // 0 or 16 bytes

    int4 acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc[mi][ni] = make_int4(0, 0, 0, 0);
        }
    }

    if (k_iters > 0) {
        issue_a(0, 0);
        uint32_t w0, w1;
        ld_b_pair(0, w0, w1);
        decode_one_to_smem(w0, 0, b_ks_pair + 0);
        decode_one_to_smem(w1, 0, b_ks_pair + 1);
        escha_cp_async_wait_all();
        __syncthreads();
    }

    for (int ki = 0; ki < k_iters; ++ki) {
        const int cur = ki & 1;
        const int nxt = cur ^ 1;

        uint32_t wn0 = 0u, wn1 = 0u;
        if (ki + 1 < k_iters) {
            issue_a(ki + 1, nxt);
            ld_b_pair(ki + 1, wn0, wn1);
        }

        // Compute 16 INT8 m16n8k32 MMAs across BK=64 (2 k-steps of 32 bytes)
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 32;
            uint32_t a0[4], a1[4];
            escha_ldmatrix_x4(a0[0], a0[1], a0[2], a0[3], &sh_a[cur][wm_base +  0 + a_ld_row][k_off + a_ld_col]);
            escha_ldmatrix_x4(a1[0], a1[1], a1[2], a1[3], &sh_a[cur][wm_base + 16 + a_ld_row][k_off + a_ld_col]);

            uint32_t b0[4], b1[4];
            escha_ldmatrix_x4(b0[0], b0[1], b0[2], b0[3], &sh_b[cur][wn_base +  0 + b_ld_row][k_off + b_ld_col]);
            escha_ldmatrix_x4(b1[0], b1[1], b1[2], b1[3], &sh_b[cur][wn_base + 16 + b_ld_row][k_off + b_ld_col]);

            escha_mma_s8_u32(acc[0][0], a0[0], a0[1], a0[2], a0[3], b0[0], b0[1]);
            escha_mma_s8_u32(acc[0][1], a0[0], a0[1], a0[2], a0[3], b0[2], b0[3]);
            escha_mma_s8_u32(acc[0][2], a0[0], a0[1], a0[2], a0[3], b1[0], b1[1]);
            escha_mma_s8_u32(acc[0][3], a0[0], a0[1], a0[2], a0[3], b1[2], b1[3]);

            escha_mma_s8_u32(acc[1][0], a1[0], a1[1], a1[2], a1[3], b0[0], b0[1]);
            escha_mma_s8_u32(acc[1][1], a1[0], a1[1], a1[2], a1[3], b0[2], b0[3]);
            escha_mma_s8_u32(acc[1][2], a1[0], a1[1], a1[2], a1[3], b1[0], b1[1]);
            escha_mma_s8_u32(acc[1][3], a1[0], a1[1], a1[2], a1[3], b1[2], b1[3]);
        }

        if (ki + 1 < k_iters) {
            decode_one_to_smem(wn0, nxt, b_ks_pair + 0);
            decode_one_to_smem(wn1, nxt, b_ks_pair + 1);
            escha_cp_async_wait_all();
        }
        __syncthreads();
    }

    // Epilogue: multiply INT32 accumulators by (scale_a[row] * 0.03125f) -> FP32 global output
    const int r_in = lane >> 2;
    const int c_in = (lane & 3) * 2;
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int row0 = m_base + wm_base + mi * 16 + r_in;
        const int row1 = row0 + 8;
        if (row0 < size_m) {
            const float s0 = scale_a[row0] * 0.03125f;
            float* dst0 = C + (size_t)row0 * size_n + n_base + wn_base + c_in;
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                *reinterpret_cast<float2*>(dst0 + ni * 8) = make_float2(
                    static_cast<float>(acc[mi][ni].x) * s0,
                    static_cast<float>(acc[mi][ni].y) * s0
                );
            }
        }
        if (row1 < size_m) {
            const float s1 = scale_a[row1] * 0.03125f;
            float* dst1 = C + (size_t)row1 * size_n + n_base + wn_base + c_in;
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                *reinterpret_cast<float2*>(dst1 + ni * 8) = make_float2(
                    static_cast<float>(acc[mi][ni].z) * s1,
                    static_cast<float>(acc[mi][ni].w) * s1
                );
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3) escha_gemm_k2_i8_m128n64(
    const signed char* A_i8, const float* scale_a, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemm_i8_m128n64_impl<2>(A_i8, scale_a, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(256, 3) escha_gemm_k3_i8_m128n64(
    const signed char* A_i8, const float* scale_a, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemm_i8_m128n64_impl<3>(A_i8, scale_a, B, rout, C, m, k, n); }

template <int bits>
static __device__ __forceinline__ void escha_gemm_i8_m256n64_impl(
    const signed char* __restrict__ A_i8,
    const float* __restrict__ scale_a,
    const uint16_t* __restrict__ B,
    const __half* __restrict__ rout,
    float* __restrict__ C,
    int size_m,
    int size_k,
    int size_n
) {
    (void)rout;
    constexpr int TWORDS = 8 * bits; // 16 for 2-bit, 24 for 3-bit
    constexpr int LD_B = 80;         // 64 + 16 bytes (0% bank conflict for ldmatrix.x4)

    const int tid = threadIdx.x;     // 0..255
    const int warp = tid >> 5;       // 0..7
    const int lane = tid & 31;       // 0..31

    const int m_base = blockIdx.y * 256;
    const int n_base = blockIdx.x * 64;
    const int ntiles = size_n >> 4;
    const int k_iters = size_k >> 6; // number of BK=64 iterations

    const int warp_m = warp >> 1;    // 0..3 (64 rows/warp)
    const int warp_n = warp & 1;     // 0..1 (32 cols/warp)
    const int wm_base = warp_m * 64;
    const int wn_base = warp_n * 32;

    const int b_ks_pair = (warp >> 2) * 2;
    const int b_nt_loc = warp & 3;
    const int b_nt_glob = blockIdx.x * 4 + b_nt_loc;

    const uint32_t* B32 = reinterpret_cast<const uint32_t*>(B);
    const size_t slice_stride = (size_t)ntiles * TWORDS;

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

    // Swizzled sh_a[2][256][64] (32 KiB, 0% bank conflicts!) + sh_b[2][64][80] (10 KiB) = 42 KiB -> 2 blks/SM
    __shared__ __align__(16) signed char sh_a[2][256][64];
    __shared__ __align__(16) signed char sh_b[2][64][LD_B];

    // 256 threads x 4 chunks of 16B = 1024 x 16B = 256 x 64 bytes
    const int cp_r0 = tid >> 2;          // 0..63
    const int cp_c_raw = tid & 3;        // 0..3 (16B chunk index)
    const int cp_c_swiz = (cp_c_raw ^ ((cp_r0 >> 1) & 3)) * 16;
    const int cp_c_glob = cp_c_raw * 16;

    const int gr0 = m_base + cp_r0;
    const int gr1 = gr0 + 64;
    const int gr2 = gr0 + 128;
    const int gr3 = gr0 + 192;
    const int sz0 = (gr0 < size_m) ? 16 : 0;
    const int sz1 = (gr1 < size_m) ? 16 : 0;
    const int sz2 = (gr2 < size_m) ? 16 : 0;
    const int sz3 = (gr3 < size_m) ? 16 : 0;
    const signed char* ap0 = sz0 ? (A_i8 + (size_t)gr0 * size_k + cp_c_glob) : A_i8;
    const signed char* ap1 = sz1 ? (A_i8 + (size_t)gr1 * size_k + cp_c_glob) : A_i8;
    const signed char* ap2 = sz2 ? (A_i8 + (size_t)gr2 * size_k + cp_c_glob) : A_i8;
    const signed char* ap3 = sz3 ? (A_i8 + (size_t)gr3 * size_k + cp_c_glob) : A_i8;

    auto issue_a = [&](int ki, int stage) {
        const size_t k_off = (size_t)ki * 64;
        escha_cp_async_16_zfill(&sh_a[stage][cp_r0 +   0][cp_c_swiz], sz0 ? (ap0 + k_off) : A_i8, sz0);
        escha_cp_async_16_zfill(&sh_a[stage][cp_r0 +  64][cp_c_swiz], sz1 ? (ap1 + k_off) : A_i8, sz1);
        escha_cp_async_16_zfill(&sh_a[stage][cp_r0 + 128][cp_c_swiz], sz2 ? (ap2 + k_off) : A_i8, sz2);
        escha_cp_async_16_zfill(&sh_a[stage][cp_r0 + 192][cp_c_swiz], sz3 ? (ap3 + k_off) : A_i8, sz3);
        escha_cp_async_commit();
    };

    const uint32_t* b_warp_base0 = B32 + (size_t)(b_ks_pair + 0) * slice_stride + (size_t)b_nt_glob * TWORDS + lane;
    const uint32_t* b_warp_base1 = B32 + (size_t)(b_ks_pair + 1) * slice_stride + (size_t)b_nt_glob * TWORDS + lane;
    auto ld_b_pair = [&](int ki, uint32_t& w0, uint32_t& w1) {
        const size_t off = (size_t)(ki * 4) * slice_stride;
        if (lane < TWORDS) {
            w0 = __ldcs(b_warp_base0 + off);
            w1 = __ldcs(b_warp_base1 + off);
        } else {
            w0 = 0u;
            w1 = 0u;
        }
    };

    const int bn0 = b_nt_loc * 16 + (lane >> 2);
    const int bk_lane = (lane & 3) * 2;
    auto decode_one_to_smem = [&](uint32_t w, int stage, int ks_loc) {
        FragB wf0, wf1;
        if constexpr (bits == 2) {
            const uint32_t bwv = __shfl_sync(0xffffffffu, w, x_src_b);
            const uint32_t awv = __shfl_sync(0xffffffffu, w, x_src_a);
            dq8_regs_2bits_mcg(awv, bwv, lane << 3, wf0, wf1);
        } else {
            const uint32_t awv = __shfl_sync(0xffffffffu, w, x_src_a);
            const uint32_t bwv = __shfl_sync(0xffffffffu, w, x_src_b);
            dq8_regs_3bits_mcg(awv, bwv, x_s2, wf0, wf1);
        }
        const int bk0 = ks_loc * 16 + bk_lane;
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0    ][bk0    ]) = half2_to_i8x2_s32(wf0[0]);
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0    ][bk0 + 8]) = half2_to_i8x2_s32(wf0[1]);
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0 + 8][bk0    ]) = half2_to_i8x2_s32(wf1[0]);
        *reinterpret_cast<uint16_t*>(&sh_b[stage][bn0 + 8][bk0 + 8]) = half2_to_i8x2_s32(wf1[1]);
    };

    const int a_ld_row = lane & 15;
    const int a_swiz_k0 = ((lane >> 4) ^ ((a_ld_row >> 1) & 3)) * 16;
    const int a_swiz_k1 = ((2 + (lane >> 4)) ^ ((a_ld_row >> 1) & 3)) * 16;
    const int mat = lane >> 3;
    const int b_ld_row = ((mat >> 1) * 8) + (lane & 7);
    const int b_ld_col = (mat & 1) * 16;

    int4 acc[4][4];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc[mi][ni] = make_int4(0, 0, 0, 0);
        }
    }

    if (k_iters > 0) {
        issue_a(0, 0);
        uint32_t w0, w1;
        ld_b_pair(0, w0, w1);
        decode_one_to_smem(w0, 0, b_ks_pair + 0);
        decode_one_to_smem(w1, 0, b_ks_pair + 1);
        escha_cp_async_wait_all();
        __syncthreads();
    }

    for (int ki = 0; ki < k_iters; ++ki) {
        const int cur = ki & 1;
        const int nxt = cur ^ 1;

        uint32_t wn0 = 0u, wn1 = 0u;
        if (ki + 1 < k_iters) {
            issue_a(ki + 1, nxt);
            ld_b_pair(ki + 1, wn0, wn1);
        }

        // 32 INT8 m16n8k32 MMAs per warp (WM=64, WN=32, BK=64)
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int a_col = (kk == 0) ? a_swiz_k0 : a_swiz_k1;
            const int b_off = kk * 32;
            uint32_t b0[4], b1[4];
            escha_ldmatrix_x4(b0[0], b0[1], b0[2], b0[3], &sh_b[cur][wn_base +  0 + b_ld_row][b_off + b_ld_col]);
            escha_ldmatrix_x4(b1[0], b1[1], b1[2], b1[3], &sh_b[cur][wn_base + 16 + b_ld_row][b_off + b_ld_col]);

            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                uint32_t a[4];
                escha_ldmatrix_x4(a[0], a[1], a[2], a[3], &sh_a[cur][wm_base + mi * 16 + a_ld_row][a_col]);
                escha_mma_s8_u32(acc[mi][0], a[0], a[1], a[2], a[3], b0[0], b0[1]);
                escha_mma_s8_u32(acc[mi][1], a[0], a[1], a[2], a[3], b0[2], b0[3]);
                escha_mma_s8_u32(acc[mi][2], a[0], a[1], a[2], a[3], b1[0], b1[1]);
                escha_mma_s8_u32(acc[mi][3], a[0], a[1], a[2], a[3], b1[2], b1[3]);
            }
        }

        if (ki + 1 < k_iters) {
            decode_one_to_smem(wn0, nxt, b_ks_pair + 0);
            decode_one_to_smem(wn1, nxt, b_ks_pair + 1);
            escha_cp_async_wait_all();
        }
        __syncthreads();
    }

    const int r_in = lane >> 2;
    const int c_in = (lane & 3) * 2;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int row0 = m_base + wm_base + mi * 16 + r_in;
        const int row1 = row0 + 8;
        if (row0 < size_m) {
            const float s0 = scale_a[row0] * 0.03125f;
            float* dst0 = C + (size_t)row0 * size_n + n_base + wn_base + c_in;
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                *reinterpret_cast<float2*>(dst0 + ni * 8) = make_float2(
                    static_cast<float>(acc[mi][ni].x) * s0,
                    static_cast<float>(acc[mi][ni].y) * s0
                );
            }
        }
        if (row1 < size_m) {
            const float s1 = scale_a[row1] * 0.03125f;
            float* dst1 = C + (size_t)row1 * size_n + n_base + wn_base + c_in;
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                *reinterpret_cast<float2*>(dst1 + ni * 8) = make_float2(
                    static_cast<float>(acc[mi][ni].z) * s1,
                    static_cast<float>(acc[mi][ni].w) * s1
                );
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) escha_gemm_k2_i8_m256n64(
    const signed char* A_i8, const float* scale_a, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemm_i8_m256n64_impl<2>(A_i8, scale_a, B, rout, C, m, k, n); }

extern "C" __global__ void __launch_bounds__(256, 2) escha_gemm_k3_i8_m256n64(
    const signed char* A_i8, const float* scale_a, const uint16_t* B, const __half* rout, float* C, int m, int k, int n
) { escha_gemm_i8_m256n64_impl<3>(A_i8, scale_a, B, rout, C, m, k, n); }


