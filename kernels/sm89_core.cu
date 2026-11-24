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

// -----------------------------------------------------------------------------
// 4. Fused Residual Add (+ optional delta + optional bias) + RMSNorm + 2x Had128 -> FP16
//    Used for GDN pre-attn (also writes x_norm_f32 for ssm_alpha/ssm_beta!) and FFN pre-mlp
//    (ffn_gate + ffn_up with independent escha_rin!).
//    Grid: (n_rows), Block: (256)
// -----------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256, 4) fused_add_bias_rmsnorm_had128_2x_f16(
    float* __restrict__ residual,         // in-out [n_rows, dim]
    const float* __restrict__ delta,      // in     [n_rows, dim] (ignored if has_delta == 0)
    const float* __restrict__ delta_bias, // in     [dim]         (ignored if has_delta == 0)
    const float* __restrict__ norm_w,     // in     [dim]
    const __half* __restrict__ rin0,      // in     [dim]
    const __half* __restrict__ rin1,      // in     [dim]
    __half* __restrict__ xh0_out,         // out    [n_rows, dim]
    __half* __restrict__ xh1_out,         // out    [n_rows, dim]
    float* __restrict__ x_norm_out,       // out    [n_rows, dim] (optional, pass nullptr if unused)
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
    __half* out0 = xh0_out + (size_t)row * dim;
    __half* out1 = xh1_out + (size_t)row * dim;
    float* xnorm_r = (x_norm_out != nullptr) ? (x_norm_out + (size_t)row * dim) : nullptr;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = res_r[i];
        if (has_delta) {
            v += del_r[i] + delta_bias[i];
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

    const int n_blocks128 = dim >> 7;
    for (int blk = warp; blk < n_blocks128; blk += n_warps) {
        const int base = (blk << 7) + (lane << 2);
        float xn[4];
        float v0[4];
        float v1[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = base + k;
            xn[k] = res_r[idx] * inv_rms * norm_w[idx];
            if (xnorm_r != nullptr) xnorm_r[idx] = xn[k];
            v0[k] = xn[k] * __half2float(rin0[idx]);
            v1[k] = xn[k] * __half2float(rin1[idx]);
        }
        fwht128_warp(v0);
        fwht128_warp(v1);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            out0[base + k] = __float2half(v0[k]);
            out1[base + k] = __float2half(v1[k]);
        }
    }
}

// -----------------------------------------------------------------------------
// 5. Fused Residual Add (+ delta + bias) + RMSNorm + 3x Had128 -> FP16
//    Used for FullAttn pre-attn (attn_q, attn_k, attn_v with independent escha_rin!).
//    Grid: (n_rows), Block: (256)
// -----------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256, 4) fused_add_bias_rmsnorm_had128_3x_f16(
    float* __restrict__ residual,         // in-out [n_rows, dim]
    const float* __restrict__ delta,      // in     [n_rows, dim]
    const float* __restrict__ delta_bias, // in     [dim]
    const float* __restrict__ norm_w,     // in     [dim]
    const __half* __restrict__ rin0,      // in     [dim]
    const __half* __restrict__ rin1,      // in     [dim]
    const __half* __restrict__ rin2,      // in     [dim]
    __half* __restrict__ xh0_out,         // out    [n_rows, dim]
    __half* __restrict__ xh1_out,         // out    [n_rows, dim]
    __half* __restrict__ xh2_out,         // out    [n_rows, dim]
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
    __half* out0 = xh0_out + (size_t)row * dim;
    __half* out1 = xh1_out + (size_t)row * dim;
    __half* out2 = xh2_out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = res_r[i];
        if (has_delta) {
            v += del_r[i] + delta_bias[i];
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

    const int n_blocks128 = dim >> 7;
    for (int blk = warp; blk < n_blocks128; blk += n_warps) {
        const int base = (blk << 7) + (lane << 2);
        float v0[4], v1[4], v2[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = base + k;
            const float xn = res_r[idx] * inv_rms * norm_w[idx];
            v0[k] = xn * __half2float(rin0[idx]);
            v1[k] = xn * __half2float(rin1[idx]);
            v2[k] = xn * __half2float(rin2[idx]);
        }
        fwht128_warp(v0);
        fwht128_warp(v1);
        fwht128_warp(v2);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            out0[base + k] = __float2half(v0[k]);
            out1[base + k] = __float2half(v1[k]);
            out2[base + k] = __float2half(v2[k]);
        }
    }
}

// -----------------------------------------------------------------------------
// 6. Fused Residual Add (+ delta + bias) + RMSNorm -> FP32 (for Layer 63 -> output_norm!)
//    Grid: (n_rows), Block: (256)
// -----------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256, 4) fused_add_bias_rmsnorm_f32(
    float* __restrict__ residual,
    const float* __restrict__ delta,
    const float* __restrict__ delta_bias,
    const float* __restrict__ norm_w,
    float* __restrict__ y_out,
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
    float* yr = y_out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = res_r[i];
        if (has_delta) {
            v += del_r[i] + delta_bias[i];
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

    for (int i = tid; i < dim; i += blockDim.x) {
        yr[i] = res_r[i] * inv_rms * norm_w[i];
    }
}

// -----------------------------------------------------------------------------
// 7. Capture residual + delta + bias into DFlash2 target-feature buffer [M, 5 * 5120]
//    Note: DFlash2 captures t_layer_inp[il] for il in {6, 20, 34, 48, 62}, which is
//    EXACTLY `residual + prev_ffn_down + prev_ffn_down_bias` BEFORE layer `il` runs!
//    Grid: (M, 20, 1), Block: (256)
// -----------------------------------------------------------------------------
extern "C" __global__ void capture_layer_inp_f32(
    const float* __restrict__ residual,   // [M, 5120] (already updated by fused_add_bias_rmsnorm!)
    float* __restrict__ cap_buf,          // [M, 5 * 5120]
    int slot,                             // 0..4
    int dim                               // 5120
) {
    const int row = blockIdx.x;
    const int idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (idx < dim) {
        cap_buf[(size_t)row * (5 * (size_t)dim) + (size_t)slot * dim + idx] = residual[(size_t)row * dim + idx];
    }
}

// -----------------------------------------------------------------------------
// 8. Fast GPU Q8_0 -> Q4_0 converter (builds 682 MiB Stage-1 LM Head in < 20 ms!)
//    1 warp converts 1 block (32 elements): 34 bytes Q8_0 -> 18 bytes Q4_0.
//    Grid: (ceil(n_blocks / 8)), Block: (256)
// -----------------------------------------------------------------------------
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

extern "C" __global__ void __launch_bounds__(256, 4) convert_q8_0_to_q4_0_sm89(
    const uint8_t* __restrict__ src_q8_0,
    uint8_t* __restrict__ dst_q4_0,
    unsigned long long n_blocks
) {
    const unsigned long long blk = ((unsigned long long)blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (blk >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* sb = src_q8_0 + blk * 34ull;
    const uint16_t d8_bits = (uint16_t)sb[0] | ((uint16_t)sb[1] << 8);
    const float d8 = __half2float(__ushort_as_half(d8_bits));
    const float val = d8 * (float)((const int8_t*)(sb + 2))[lane];

    // Match ggml quantize_row_q4_0_ref: signed max-magnitude element determines d = max_signed / -8.0f
    float amax = fabsf(val);
    float smax = val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const float other_a = __shfl_xor_sync(0xffffffffu, amax, mask);
        const float other_s = __shfl_xor_sync(0xffffffffu, smax, mask);
        if (other_a > amax) {
            amax = other_a;
            smax = other_s;
        }
    }
    smax = __shfl_sync(0xffffffffu, smax, 0);
    const float d4 = smax * (-0.125f);
    const float id4 = (d4 != 0.0f) ? (1.0f / d4) : 0.0f;
    int q = (int)(val * id4 + 8.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    const int q_hi = __shfl_down_sync(0xffffffffu, q, 16);
    uint8_t* db = dst_q4_0 + blk * 18ull;
    if (lane == 0) {
        const uint16_t d4_bits = __half_as_ushort(__float2half(d4));
        db[0] = (uint8_t)(d4_bits & 0xffu);
        db[1] = (uint8_t)(d4_bits >> 8);
    }
    if (lane < 16) {
        db[2 + lane] = (uint8_t)((q & 0x0f) | ((q_hi & 0x0f) << 4));
    }
}
