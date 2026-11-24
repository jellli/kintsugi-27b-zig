#include <cuda_fp16.h>

// Native sm_89 (RTX 4070 Ti, Ada Lovelace) core kernels for kintsugi-zig.
// Compiled directly to sm_89 CUBIN via NVRTC and loaded by Zig through CUDA Driver API.

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

// -----------------------------------------------------------------------------
// 1. RMSNorm (f32 -> f32) specialized for Qwen3.8-27B (d_model = 5120, eps = 1e-6)
//    Note: Qwen3.5/3.8 RMSNorm weights are stored directly (or +1 depending on layer;
//    caller selects unit-offset vs direct). Here y[i] = x[i] * rsqrt(mean(x^2)+eps) * w[i].
//    Grid: (n_rows), Block: (256)
// -----------------------------------------------------------------------------
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = xr[i];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float smem[8];
    if (lane == 0) smem[warp] = sum_sq;
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        total = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane == 0) smem[0] = rsqrtf(total / (float)dim + eps);
    }
    __syncthreads();

    const float scale = smem[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        yr[i] = xr[i] * scale * w[i];
    }
}

// -----------------------------------------------------------------------------
// 2. Fused Residual Add + RMSNorm + Escha 128-pt Hadamard input rotation (f32 -> f16)
//    Eliminates 3 global memory round-trips per layer compared to separate ggml nodes:
//      h[i]   = residual[i] + delta[i]
//      xh[i]  = f16( T128( rmsnorm(h, w) * rin ) )
//    One warp handles each 128-channel block (4 elements per lane -> 32 lanes = 128).
//    Grid: (n_rows), Block: (256)
// -----------------------------------------------------------------------------
static __device__ __forceinline__ void fwht128_warp(float v[4]) {
    // Stage 1-2: in-thread 4-pt WHT
    float a0 = v[0] + v[1];
    float a1 = v[0] - v[1];
    float a2 = v[2] + v[3];
    float a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;

    // Stage 3-7: across 32 warp lanes (5 butterfly stages: 1, 2, 4, 8, 16)
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
        const float sign = (lane & stride) ? -1.0f : 1.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float peer = __shfl_xor_sync(0xffffffffu, v[k], stride);
            v[k] = (lane & stride) ? (peer - v[k]) : (v[k] + peer);
            (void)sign;
        }
    }
    // Orthonormal 1/sqrt(128) = 0.08838834764831845f
    const float inv_sqrt128 = 0.08838834764831845f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) v[k] *= inv_sqrt128;
}

extern "C" __global__ void fused_add_rmsnorm_had128_f16(
    float* __restrict__ residual,         // in-out [n_rows, dim]
    const float* __restrict__ delta,      // in     [n_rows, dim] (nullable: pass residual for no-add)
    const float* __restrict__ norm_w,     // in     [dim]
    const __half* __restrict__ rin,       // in     [dim] Escha Hadamard input sign/scale
    __half* __restrict__ xh_out,          // out    [n_rows, dim] rotated f16 activation
    int dim,
    float eps,
    int has_delta
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockDim.x >> 5;

    float* res_r = residual + (size_t)row * dim;
    const float* del_r = has_delta ? (delta + (size_t)row * dim) : nullptr;
    __half* out_r = xh_out + (size_t)row * dim;

    // Pass 1: residual add + sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = res_r[i];
        if (has_delta) {
            v += del_r[i];
            res_r[i] = v;
        }
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float smem[8];
    if (lane == 0) smem[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        float tot = (lane < n_warps) ? smem[lane] : 0.0f;
        tot = warp_reduce_sum(tot);
        if (lane == 0) smem[0] = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

    // Pass 2: each warp processes 128-element blocks -> rmsnorm * rin -> FWHT128 -> f16
    const int n_blocks128 = dim >> 7;
    for (int blk = warp; blk < n_blocks128; blk += n_warps) {
        const int base = (blk << 7) + (lane << 2);
        float v[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int idx = base + k;
            v[k] = res_r[idx] * inv_rms * norm_w[idx] * __half2float(rin[idx]);
        }
        fwht128_warp(v);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            out_r[base + k] = __float2half(v[k]);
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Fused SwiGLU: out[i] = silu(gate[i]) * up[i]
// -----------------------------------------------------------------------------
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float s = g / (1.0f + __expf(-g));
        out[i] = s * up[i];
    }
}
