// Native sm_89 (RTX 4070 Ti, Ada Lovelace) DFlash2 5-layer Speculative Drafter kernels.
//
// Kernels:
//   1. dflash_rmsnorm_f32            : Standard RMSNorm(5120) -> F32 (M=1..8)
//   2. dflash_add_rmsnorm_f32        : Fused residual += delta + RMSNorm(5120) -> F32
//   3. dflash_conv2_sm89             : DFlash2 dynamic grouped causal conv (k=2, g=16, side=0/1)
//   4. dflash_qk_norm_rope128_f32    : Per-head RMSNorm(128) + full 128-dim NeoX RoPE (freq_base=1e7)
//   5. dflash_kv_inject_f16_sm89     : Fused K-RMSNorm(128) + NeoX RoPE(128) + FP16 ring-buffer KV write
//   6. dflash_noncausal_attn_f16_sm89: Non-causal GQA (32:8, D=128) SWA-2048 FlashAttention over prefix ring + noise block
//   7. dflash_swiglu_q8_1_sm89       : Fused SiLU(gate)*up -> ggml block_q8_1 (direct input to Q4_0 ffn_down!)

#include <cuda_fp16.h>

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

#define DFLASH_SWA_MASK 2047
#define DFLASH_KV_HEADS 8
#define DFLASH_Q_HEADS 32
#define DFLASH_HEAD_DIM 128
#define DFLASH_ATTN_SCALE 0.08838834764831845f // 1 / sqrt(128)
#define Q8_1_BYTES 36

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

// =============================================================================
// 1. RMSNorm(dim) -> F32 (out may alias x)
//    Grid: (M, 1, 1), Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) dflash_rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float* xr = x + (size_t)row * dim;
    float* yr = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        const float v = xr[i];
        sum_sq += v * v;
    }
    sum_sq = warp_sum_32(sum_sq);

    __shared__ float s_warp[8];
    if (lane == 0) s_warp[warp] = sum_sq;
    __syncthreads();

    __shared__ float s_inv;
    if (warp == 0) {
        float tot = (lane < 8) ? s_warp[lane] : 0.0f;
        tot = warp_sum_32(tot);
        if (lane == 0) s_inv = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    for (int i = tid; i < dim; i += 256) {
        yr[i] = xr[i] * inv * w[i];
    }
}

// =============================================================================
// 2. Fused residual += delta + RMSNorm(dim) -> F32
//    Grid: (M, 1, 1), Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) dflash_add_rmsnorm_f32(
    float* __restrict__ residual,
    const float* __restrict__ delta,
    const float* __restrict__ w,
    float* __restrict__ out,
    int dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float* rr = residual + (size_t)row * dim;
    const float* dr = delta + (size_t)row * dim;
    float* yr = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        const float v = rr[i] + dr[i];
        rr[i] = v;
        sum_sq += v * v;
    }
    sum_sq = warp_sum_32(sum_sq);

    __shared__ float s_warp[8];
    if (lane == 0) s_warp[warp] = sum_sq;
    __syncthreads();

    __shared__ float s_inv;
    if (warp == 0) {
        float tot = (lane < 8) ? s_warp[lane] : 0.0f;
        tot = warp_sum_32(tot);
        if (lane == 0) s_inv = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    for (int i = tid; i < dim; i += 256) {
        yr[i] = rr[i] * inv * w[i];
    }
}

// =============================================================================
// 3. DFlash2 Dynamic Grouped Causal Conv (kernel_size=2, group_size=16)
//    hidden  : [M, dim]                (dim = 5120)
//    dynamic : [M, 2, 2, n_groups]     (n_groups = dim/16 = 320; stride 1280/tok)
//              layout within token: side(0..1) * 640 + tap(0..1) * 320 + group(0..319)
//    base    : [2, 2, dim]             layout: side(0..1) * 2*dim + tap(0..1) * dim + d
//    out     : [M, dim]                (must NOT alias hidden!)
//    Grid: (M, ceil(dim / 256), 1), Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) dflash_conv2_sm89(
    const float* __restrict__ hidden,
    const float* __restrict__ dynamic,
    const float* __restrict__ base,
    float* __restrict__ out,
    int side,
    int dim
) {
    const int m = blockIdx.x;
    const int d = blockIdx.y * 256 + threadIdx.x;
    if (d >= dim) return;

    const int n_groups = dim >> 4;
    const int g = d >> 4;

    const float* base_side = base + (size_t)side * (2 * (size_t)dim);
    const float* dyn_side  = dynamic + (size_t)m * (4 * (size_t)n_groups) + (size_t)side * (2 * (size_t)n_groups);

    const float w0 = base_side[d] + dyn_side[g];
    float acc = w0 * hidden[(size_t)m * dim + d];

    if (m > 0) {
        const float w1 = base_side[dim + d] + dyn_side[n_groups + g];
        acc += w1 * hidden[(size_t)(m - 1) * dim + d];
    }
    out[(size_t)m * dim + d] = acc;
}

// =============================================================================
// 4. Per-head RMSNorm(128) + Full 128-dim NeoX RoPE (in-place on F32 [M, H, 128])
//    Grid: (M, n_heads, 1), Block: (128, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 8) dflash_qk_norm_rope128_f32(
    float* __restrict__ qk,
    const float* __restrict__ w,
    int n_heads,
    const int* __restrict__ pos_ptr,
    float eps,
    float freq_base
) {
    const int m = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x; // 0..127
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int pos_base = *pos_ptr;

    float* ptr = qk + ((size_t)m * n_heads + h) * 128;
    const float v = ptr[tid];

    float ss = warp_sum_32(v * v);
    __shared__ float s_w[4];
    __shared__ float s_x[128];
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();

    float tot = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    float inv = rsqrtf(tot * (1.0f / 128.0f) + eps);
    s_x[tid] = v * inv * w[tid];
    __syncthreads();

    if (tid < 64) {
        const float pos = (float)(pos_base + m);
        const float theta = pos * powf(freq_base, -(2.0f * (float)tid) * (1.0f / 128.0f));
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float x0 = s_x[tid];
        const float x1 = s_x[tid + 64];
        ptr[tid]      = x0 * cs - x1 * sn;
        ptr[tid + 64] = x0 * sn + x1 * cs;
    }
}

// =============================================================================
// 5. Fused K-RMSNorm(128) + NeoX RoPE(128) + FP16 SWA-2048 Ring Buffer KV Inject
//    k_in, v_in : [M, 8, 128] F32
//    kv_k_ring, kv_v_ring : [2048, 8, 128] FP16 (slot = (*pos_ptr + m) & 2047)
//    Grid: (M, 8, 1), Block: (128, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 8) dflash_kv_inject_f16_sm89(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ k_norm_w,
    __half* __restrict__ kv_k_ring,
    __half* __restrict__ kv_v_ring,
    const int* __restrict__ pos_ptr,
    float eps,
    float freq_base
) {
    const int m = blockIdx.x;
    const int h = blockIdx.y; // 0..7
    const int tid = threadIdx.x; // 0..127
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int pos_base = *pos_ptr;

    const size_t src_off = ((size_t)m * DFLASH_KV_HEADS + h) * 128;
    const float kv = k_in[src_off + tid];

    float ss = warp_sum_32(kv * kv);
    __shared__ float s_w[4];
    __shared__ float s_k[128];
    if (lane == 0) s_w[warp] = ss;
    __syncthreads();

    const float tot = s_w[0] + s_w[1] + s_w[2] + s_w[3];
    const float inv = rsqrtf(tot * (1.0f / 128.0f) + eps);
    s_k[tid] = kv * inv * k_norm_w[tid];
    __syncthreads();

    const int pos = pos_base + m;
    const int slot = pos & DFLASH_SWA_MASK;
    const size_t dst_off = ((size_t)slot * DFLASH_KV_HEADS + h) * 128;

    // Write V directly to FP16 ring buffer
    kv_v_ring[dst_off + tid] = __float2half_rn(v_in[src_off + tid]);

    // Apply NeoX RoPE(128) and write K to FP16 ring buffer
    if (tid < 64) {
        const float theta = (float)pos * powf(freq_base, -(2.0f * (float)tid) * (1.0f / 128.0f));
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float x0 = s_k[tid];
        const float x1 = s_k[tid + 64];
        kv_k_ring[dst_off + tid]      = __float2half_rn(x0 * cs - x1 * sn);
        kv_k_ring[dst_off + tid + 64] = __float2half_rn(x0 * sn + x1 * cs);
    }
}

// =============================================================================
// 6. Non-Causal GQA (32:8, D=128) SWA-2048 FlashAttention
//    Q         : [M, 32, 128] F32 (post Q-norm + RoPE)
//    K_cur     : [M,  8, 128] F32 (post K-norm + RoPE, current noise block)
//    V_cur     : [M,  8, 128] F32 (current noise block)
//    kv_k_ring : [2048, 8, 128] FP16 (injected target prefix positions 0..n_past-1)
//    kv_v_ring : [2048, 8, 128] FP16
//    out       : [M, 32, 128] F32
//    Grid: (32, M, 1), Block: (128, 1, 1) — 4 warps split KV positions
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 4) dflash_noncausal_attn_f16_sm89(
    const float* __restrict__ Q,
    const float* __restrict__ K_cur,
    const float* __restrict__ V_cur,
    const __half* __restrict__ kv_k_ring,
    const __half* __restrict__ kv_v_ring,
    float* __restrict__ out,
    const int* __restrict__ n_past_ptr,
    int M
) {
    const int qh = blockIdx.x;  // 0..31
    const int qi = blockIdx.y;  // 0..M-1
    const int kvh = qh >> 2;    // GQA 32:8
    const int warp = threadIdx.x >> 5; // 0..3
    const int lane = threadIdx.x & 31; // 0..31 (4 dims per lane = 128 dims!)
    const int n_past = *n_past_ptr;

    const float* q_ptr = Q + ((size_t)qi * DFLASH_Q_HEADS + qh) * 128;
    const float4 qv = ((const float4*)q_ptr)[lane];

    const int pos_q = n_past + qi;
    const int p_start = (pos_q > DFLASH_SWA_MASK) ? (pos_q - DFLASH_SWA_MASK) : 0;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    // 1. Attend over injected prefix positions [p_start .. n_past - 1] in FP16 ring buffer
    for (int p = p_start + warp; p < n_past; p += 4) {
        const int slot = p & DFLASH_SWA_MASK;
        const size_t kv_off = ((size_t)slot * DFLASH_KV_HEADS + kvh) * 128;
        const half4 kh = ((const half4*)(kv_k_ring + kv_off))[lane];
        const half4 vh = ((const half4*)(kv_v_ring + kv_off))[lane];

        const float2 k01 = __half22float2(kh.x);
        const float2 k23 = __half22float2(kh.y);
        float dot = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
        dot = warp_sum_32(dot) * DFLASH_ATTN_SCALE;

        const float m_new = fmaxf(m_max, dot);
        const float alpha = expf(m_max - m_new);
        const float beta  = expf(dot - m_new);
        l_sum = l_sum * alpha + beta;

        const float2 v01 = __half22float2(vh.x);
        const float2 v23 = __half22float2(vh.y);
        o0 = o0 * alpha + beta * v01.x;
        o1 = o1 * alpha + beta * v01.y;
        o2 = o2 * alpha + beta * v23.x;
        o3 = o3 * alpha + beta * v23.y;
        m_max = m_new;
    }

    // 2. Attend non-causally over all M noise-block tokens [0 .. M - 1]
    for (int j = warp; j < M; j += 4) {
        const size_t cur_off = ((size_t)j * DFLASH_KV_HEADS + kvh) * 128;
        const float4 kf = ((const float4*)(K_cur + cur_off))[lane];
        const float4 vf = ((const float4*)(V_cur + cur_off))[lane];

        float dot = qv.x * kf.x + qv.y * kf.y + qv.z * kf.z + qv.w * kf.w;
        dot = warp_sum_32(dot) * DFLASH_ATTN_SCALE;

        const float m_new = fmaxf(m_max, dot);
        const float alpha = expf(m_max - m_new);
        const float beta  = expf(dot - m_new);
        l_sum = l_sum * alpha + beta;

        o0 = o0 * alpha + beta * vf.x;
        o1 = o1 * alpha + beta * vf.y;
        o2 = o2 * alpha + beta * vf.z;
        o3 = o3 * alpha + beta * vf.w;
        m_max = m_new;
    }

    // 3. Merge 4 warps via shared memory
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float4 s_o[4][32];

    if (lane == 0) {
        s_m[warp] = m_max;
        s_l[warp] = l_sum;
    }
    s_o[warp][lane] = make_float4(o0, o1, o2, o3);
    __syncthreads();

    if (warp == 0) {
        float mg = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
        float w0 = (s_l[0] > 0.0f) ? expf(s_m[0] - mg) : 0.0f;
        float w1 = (s_l[1] > 0.0f) ? expf(s_m[1] - mg) : 0.0f;
        float w2 = (s_l[2] > 0.0f) ? expf(s_m[2] - mg) : 0.0f;
        float w3 = (s_l[3] > 0.0f) ? expf(s_m[3] - mg) : 0.0f;

        float l_tot = w0 * s_l[0] + w1 * s_l[1] + w2 * s_l[2] + w3 * s_l[3];
        float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;

        float4 a0 = s_o[0][lane];
        float4 a1 = s_o[1][lane];
        float4 a2 = s_o[2][lane];
        float4 a3 = s_o[3][lane];

        float4 res;
        res.x = (w0 * a0.x + w1 * a1.x + w2 * a2.x + w3 * a3.x) * inv_l;
        res.y = (w0 * a0.y + w1 * a1.y + w2 * a2.y + w3 * a3.y) * inv_l;
        res.z = (w0 * a0.z + w1 * a1.z + w2 * a2.z + w3 * a3.z) * inv_l;
        res.w = (w0 * a0.w + w1 * a1.w + w2 * a2.w + w3 * a3.w) * inv_l;

        float* out_ptr = out + ((size_t)qi * DFLASH_Q_HEADS + qh) * 128;
        ((float4*)out_ptr)[lane] = res;
    }
}

// =============================================================================
// 7. Fused SwiGLU(gate, up) -> ggml block_q8_1 (direct input to Q4_0 ffn_down!)
//    Grid: (ceil(M * (d_ff / 32) / 8), 1, 1), Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) dflash_swiglu_q8_1_sm89(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    uint8_t* __restrict__ dst_q8_1,
    int d_ff,
    int M
) {
    const int nb = d_ff >> 5;
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w >= M * nb) return;

    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)w * 32 + lane;
    const float val = silu_f32(gate[idx]) * up[idx];

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
        *reinterpret_cast<half2*>(blk_ptr) = __floats2half2_rn(d, d * (float)qsum);
    }
    ((int8_t*)(blk_ptr + 4))[lane] = (int8_t)qi;
}
