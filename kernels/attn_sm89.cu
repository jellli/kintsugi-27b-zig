// Native sm_89 (RTX 4070 Ti, Ada Lovelace) Full Attention & Q4_0 KV Cache kernels
// for Kintsugi-27B (Qwen3.8-27B, 16 FullAttn layers: 24 Q heads, 4 KV heads, D=256, RoPE=64).
//
// Fusions vs ggml:
//   1. attn_q_norm_mrope_split_f32       : Q+gate bias + split + RMSNorm(256) + partial RoPE(64) in 1 kernel
//   2. attn_k_norm_mrope_kv_q4_0_store   : K/V bias + K RMSNorm(256) + partial RoPE(64) + Q4_0 KV quant+store in 1 kernel
//   3. attn_decode_q4_0_splitk_sm89      : Register-resident Split-K causal FlashAttention (Q4_0 KV, 24Q/4KV, D=256, M=1..8)
//   4. attn_splitk_reduce_gate_had128_f16: Split-K softmax merge + sigmoid(gate) + escha_rin + Had128 -> FP16 in 1 kernel

#include <cuda_fp16.h>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ATTN_HQ 24
#define ATTN_HKV 4
#define ATTN_D 256
#define ATTN_ROPE_D 64
#define ATTN_SCALE 0.0625f         // 1 / sqrt(256)
#define ATTN_INV_SQRT128 0.08838834764831845f
#define Q4_0_BLOCK_BYTES 18        // 2B f16 scale + 16B nibbles (32 elems)
#define Q4_0_HEAD_BYTES (8 * 18)   // 144 bytes per 256-dim head

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

// Quantize 32 floats (1 float per lane across a warp) into 18-byte ggml block_q4_0.
// Matches ggml_quantize_q4_0: d = signed_amax / -8.0f, qs[j] = lo(j) | (hi(j+16) << 4).
static __device__ __forceinline__ void quantize_q4_0_warp(float val, uint8_t* dst18, int lane) {
    float amax = val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, amax, mask);
        if (fabsf(other) > fabsf(amax)) amax = other;
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = amax * (-0.125f); // amax / -8.0f
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    float qf = val * id + 8.5f;
    int q = (int)qf;
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    // Pair lane j (0..15) with lane j+16 (16..31)
    int q_hi = __shfl_down_sync(0xffffffffu, q, 16);
    if (lane == 0) {
        uint16_t d_bits = __half_as_ushort(__float2half_rn(d));
        dst18[0] = (uint8_t)(d_bits & 0xffu);
        dst18[1] = (uint8_t)(d_bits >> 8);
    }
    if (lane < 16) {
        dst18[2 + lane] = (uint8_t)((q & 0x0f) | ((q_hi & 0x0f) << 4));
    }
}

// =============================================================================
// 1. Fused Q+Gate Bias + Split + RMSNorm(256) + Partial NeoX/MRoPE(64)
//    q_full : [n_tokens, 24, 512] (each head: [0..255]=Q, [256..511]=gate)
//    q_bias : [12288] or nullptr
//    q_norm : [256]
//    q_out  : [n_tokens, 24, 256]
//    gate_o : [n_tokens, 24, 256]
//    Grid   : (24, n_tokens, 1) — 1 block = 1 head (256 threads = 8 warps)
//    Block  : (256, 1, 1)
// =============================================================================
static __device__ __forceinline__ void attn_q_norm_mrope_split_f32_impl(
    const float* __restrict__ q_full,
    const float* __restrict__ q_bias,
    const float* __restrict__ q_norm,
    float* __restrict__ q_out,
    float* __restrict__ gate_out,
    int pos_base,
    float rope_freq_base,
    float eps
) {
    const int h = blockIdx.x;    // 0..23
    const int t = blockIdx.y;    // 0..M-1
    const int tid = threadIdx.x; // 0..255

    const size_t full_base = ((size_t)t * ATTN_HQ + h) * (2 * ATTN_D);
    const size_t bias_base = (size_t)h * (2 * ATTN_D);
    const size_t out_base = ((size_t)t * ATTN_HQ + h) * ATTN_D;

    float qv = q_full[full_base + tid] + (q_bias ? q_bias[bias_base + tid] : 0.0f);
    float gv = q_full[full_base + ATTN_D + tid] + (q_bias ? q_bias[bias_base + ATTN_D + tid] : 0.0f);
    gate_out[out_base + tid] = gv;

    // RMSNorm(256)
    __shared__ float sh_sq[8];
    __shared__ float sh_q[ATTN_ROPE_D];
    float sq = warp_sum_32(qv * qv);
    if ((tid & 31) == 0) sh_sq[tid >> 5] = sq;
    __syncthreads();
    float tot_sq = sh_sq[0] + sh_sq[1] + sh_sq[2] + sh_sq[3]
                 + sh_sq[4] + sh_sq[5] + sh_sq[6] + sh_sq[7];
    float inv_rms = rsqrtf(tot_sq * (1.0f / 256.0f) + eps);
    qv = qv * inv_rms * q_norm[tid];

    if (tid < ATTN_ROPE_D) {
        sh_q[tid] = qv;
    }
    __syncthreads();

    if (tid < (ATTN_ROPE_D / 2)) {
        const float pos = (float)(pos_base + t);
        const float inv_freq = powf(rope_freq_base, -((float)(2 * tid) / (float)ATTN_ROPE_D));
        const float theta = pos * inv_freq;
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = sh_q[tid];
        const float x1 = sh_q[tid + (ATTN_ROPE_D / 2)];
        q_out[out_base + tid] = x0 * c - x1 * s;
        q_out[out_base + tid + (ATTN_ROPE_D / 2)] = x0 * s + x1 * c;
    } else if (tid >= ATTN_ROPE_D) {
        q_out[out_base + tid] = qv;
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) attn_q_norm_mrope_split_f32(
    const float* __restrict__ q_full, const float* __restrict__ q_bias, const float* __restrict__ q_norm,
    float* __restrict__ q_out, float* __restrict__ gate_out, int pos_base, float rope_freq_base, float eps
) { attn_q_norm_mrope_split_f32_impl(q_full, q_bias, q_norm, q_out, gate_out, pos_base, rope_freq_base, eps); }

extern "C" __global__ void __launch_bounds__(256, 4) attn_q_norm_mrope_split_f32_dpos(
    const float* __restrict__ q_full, const float* __restrict__ q_bias, const float* __restrict__ q_norm,
    float* __restrict__ q_out, float* __restrict__ gate_out, const int* __restrict__ pos_ptr, float rope_freq_base, float eps
) { attn_q_norm_mrope_split_f32_impl(q_full, q_bias, q_norm, q_out, gate_out, *pos_ptr, rope_freq_base, eps); }

// =============================================================================
// 2. Fused K/V Bias + K RMSNorm(256) + Partial RoPE(64) + Q4_0 KV Cache Quantize & Store
//    k_in, v_in     : [n_tokens, 4, 256] f32
//    k_bias, v_bias : [1024] or nullptr
//    k_norm         : [256]
//    k_cache_q4     : [max_cells, 4, 144] uint8 (Q4_0)
//    v_cache_q4     : [max_cells, 4, 144] uint8 (Q4_0)
//    Grid           : (4, n_tokens, 2) — z=0 processes K (norm+rope+quant), z=1 processes V (quant)
//    Block          : (256, 1, 1) — 8 warps = 8 Q4_0 blocks of 32 elements
// =============================================================================
static __device__ __forceinline__ void attn_k_norm_mrope_kv_q4_0_store_impl(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    const float* __restrict__ k_bias,
    const float* __restrict__ v_bias,
    const float* __restrict__ k_norm,
    uint8_t* __restrict__ k_cache_q4,
    uint8_t* __restrict__ v_cache_q4,
    int pos_base,
    float rope_freq_base,
    float eps
) {
    const int hkv = blockIdx.x;   // 0..3
    const int t = blockIdx.y;     // 0..M-1
    const bool is_v = (blockIdx.z == 1);
    const int tid = threadIdx.x;  // 0..255
    const int warp = tid >> 5;    // 0..7 (Q4_0 block index)
    const int lane = tid & 31;    // 0..31 (element inside Q4_0 block)

    const size_t in_off = ((size_t)t * ATTN_HKV + hkv) * ATTN_D + tid;
    const size_t bias_off = (size_t)hkv * ATTN_D + tid;
    const size_t cache_slot = (size_t)(pos_base + t);
    const size_t dst_head_off = (cache_slot * ATTN_HKV + hkv) * Q4_0_HEAD_BYTES;

    float val;
    if (is_v) {
        val = v_in[in_off] + (v_bias ? v_bias[bias_off] : 0.0f);
        quantize_q4_0_warp(val, v_cache_q4 + dst_head_off + warp * Q4_0_BLOCK_BYTES, lane);
    } else {
        float kv = k_in[in_off] + (k_bias ? k_bias[bias_off] : 0.0f);
        __shared__ float sh_sq[8];
        __shared__ float sh_k[ATTN_ROPE_D];
        float sq = warp_sum_32(kv * kv);
        if (lane == 0) sh_sq[warp] = sq;
        __syncthreads();
        float tot_sq = sh_sq[0] + sh_sq[1] + sh_sq[2] + sh_sq[3]
                     + sh_sq[4] + sh_sq[5] + sh_sq[6] + sh_sq[7];
        float inv_rms = rsqrtf(tot_sq * (1.0f / 256.0f) + eps);
        kv = kv * inv_rms * k_norm[tid];

        if (tid < ATTN_ROPE_D) sh_k[tid] = kv;
        __syncthreads();

        if (tid < (ATTN_ROPE_D / 2)) {
            const float pos = (float)(pos_base + t);
            const float inv_freq = powf(rope_freq_base, -((float)(2 * tid) / (float)ATTN_ROPE_D));
            const float theta = pos * inv_freq;
            float s, c;
            sincosf(theta, &s, &c);
            val = sh_k[tid] * c - sh_k[tid + (ATTN_ROPE_D / 2)] * s;
        } else if (tid < ATTN_ROPE_D) {
            const int pair = tid - (ATTN_ROPE_D / 2);
            const float pos = (float)(pos_base + t);
            const float inv_freq = powf(rope_freq_base, -((float)(2 * pair) / (float)ATTN_ROPE_D));
            const float theta = pos * inv_freq;
            float s, c;
            sincosf(theta, &s, &c);
            val = sh_k[pair] * s + sh_k[tid] * c;
        } else {
            val = kv;
        }
        quantize_q4_0_warp(val, k_cache_q4 + dst_head_off + warp * Q4_0_BLOCK_BYTES, lane);
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) attn_k_norm_mrope_kv_q4_0_store(
    const float* __restrict__ k_in, const float* __restrict__ v_in,
    const float* __restrict__ k_bias, const float* __restrict__ v_bias,
    const float* __restrict__ k_norm, uint8_t* __restrict__ k_cache_q4, uint8_t* __restrict__ v_cache_q4,
    int pos_base, float rope_freq_base, float eps
) { attn_k_norm_mrope_kv_q4_0_store_impl(k_in, v_in, k_bias, v_bias, k_norm, k_cache_q4, v_cache_q4, pos_base, rope_freq_base, eps); }

extern "C" __global__ void __launch_bounds__(256, 4) attn_k_norm_mrope_kv_q4_0_store_dpos(
    const float* __restrict__ k_in, const float* __restrict__ v_in,
    const float* __restrict__ k_bias, const float* __restrict__ v_bias,
    const float* __restrict__ k_norm, uint8_t* __restrict__ k_cache_q4, uint8_t* __restrict__ v_cache_q4,
    const int* __restrict__ pos_ptr, float rope_freq_base, float eps
) { attn_k_norm_mrope_kv_q4_0_store_impl(k_in, v_in, k_bias, v_bias, k_norm, k_cache_q4, v_cache_q4, *pos_ptr, rope_freq_base, eps); }

// Helper: load 8 elements (4 low + 4 high) from one Q4_0 block for lane's (blk, sub)
static __device__ __forceinline__ void load_q4_0_8(
    const uint8_t* head_ptr,
    int blk,
    int sub,
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7
) {
    const uint8_t* b = head_ptr + blk * Q4_0_BLOCK_BYTES;
    const uint16_t d_bits = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    const float d = __half2float(__ushort_as_half(d_bits));
    const uint8_t* q = b + 2 + sub * 4;
    const uint8_t b0 = q[0];
    const uint8_t b1 = q[1];
    const uint8_t b2 = q[2];
    const uint8_t b3 = q[3];
    v0 = ((int)(b0 & 0x0fu) - 8) * d;
    v1 = ((int)(b1 & 0x0fu) - 8) * d;
    v2 = ((int)(b2 & 0x0fu) - 8) * d;
    v3 = ((int)(b3 & 0x0fu) - 8) * d;
    v4 = ((int)(b0 >> 4) - 8) * d;
    v5 = ((int)(b1 >> 4) - 8) * d;
    v6 = ((int)(b2 >> 4) - 8) * d;
    v7 = ((int)(b3 >> 4) - 8) * d;
}

// =============================================================================
// 3. Split-K Causal FlashAttention Decode on Q4_0 KV Cache (24Q / 4KV, D=256, M=1..8)
//    Grid : (24, n_tokens, n_splits)
//    Block: (32, 1, 1) — 1 warp owns all 256 dims (8 elems/lane, ZERO shared memory!)
//
//    Each query token t (0..M-1) attends causally to KV positions [0 .. pos_base + t].
//    Partial outputs written to:
//      part_o : [n_tokens, 24, n_splits, 256]
//      part_m : [n_tokens, 24, n_splits]
//      part_l : [n_tokens, 24, n_splits]
// =============================================================================
static __device__ __forceinline__ void attn_decode_q4_0_splitk_sm89_impl(
    const float* __restrict__ q_rot,        // [n_tokens, 24, 256]
    const uint8_t* __restrict__ k_cache_q4, // [max_cells, 4, 144]
    const uint8_t* __restrict__ v_cache_q4, // [max_cells, 4, 144]
    float* __restrict__ part_o,             // [n_tokens, 24, n_splits, 256]
    float* __restrict__ part_m,             // [n_tokens, 24, n_splits]
    float* __restrict__ part_l,             // [n_tokens, 24, n_splits]
    int pos_base,
    int n_splits
) {
    const int hq = blockIdx.x;    // 0..23
    const int hkv = hq / 6;       // 24Q / 4KV = 6:1 GQA
    const int t = blockIdx.y;     // 0..M-1
    const int sp = blockIdx.z;    // 0..n_splits-1
    const int lane = threadIdx.x; // 0..31

    const int blk = lane >> 2;    // 0..7
    const int sub = lane & 3;     // 0..3
    const int idx_lo = blk * 32 + sub * 4;
    const int idx_hi = idx_lo + 16;

    const int seq_len = pos_base + t + 1;
    const int chunk = (seq_len + n_splits - 1) / n_splits;
    const int p_start = sp * chunk;
    const int p_end = (p_start + chunk < seq_len) ? (p_start + chunk) : seq_len;

    const size_t meta_idx = ((size_t)t * ATTN_HQ + hq) * n_splits + sp;
    float* dst_o = part_o + meta_idx * ATTN_D;

    if (p_start >= p_end) {
        if (lane == 0) {
            part_m[meta_idx] = -1.0e30f;
            part_l[meta_idx] = 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst_o[idx_lo + j] = 0.0f;
            dst_o[idx_hi + j] = 0.0f;
        }
        return;
    }

    const float* q_ptr = q_rot + ((size_t)t * ATTN_HQ + hq) * ATTN_D;
    const float q0 = q_ptr[idx_lo + 0], q1 = q_ptr[idx_lo + 1], q2 = q_ptr[idx_lo + 2], q3 = q_ptr[idx_lo + 3];
    const float q4 = q_ptr[idx_hi + 0], q5 = q_ptr[idx_hi + 1], q6 = q_ptr[idx_hi + 2], q7 = q_ptr[idx_hi + 3];

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;
    float m_run = -1.0e30f;
    float l_run = 0.0f;

    const size_t kv_stride = (size_t)ATTN_HKV * Q4_0_HEAD_BYTES;
    const uint8_t* k_base = k_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;
    const uint8_t* v_base = v_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;

    for (int p = p_start; p < p_end; ++p) {
        float k0, k1, k2, k3, k4, k5, k6, k7;
        load_q4_0_8(k_base + (size_t)p * kv_stride, blk, sub, k0, k1, k2, k3, k4, k5, k6, k7);

        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3
                  + q4 * k4 + q5 * k5 + q6 * k6 + q7 * k7;
        const float score = warp_sum_32(dot) * ATTN_SCALE;

        const float m_new = fmaxf(m_run, score);
        const float alpha = expf(m_run - m_new);
        const float beta = expf(score - m_new);
        l_run = l_run * alpha + beta;
        m_run = m_new;

        float v0, v1, v2, v3, v4, v5, v6, v7;
        load_q4_0_8(v_base + (size_t)p * kv_stride, blk, sub, v0, v1, v2, v3, v4, v5, v6, v7);

        o0 = o0 * alpha + beta * v0;
        o1 = o1 * alpha + beta * v1;
        o2 = o2 * alpha + beta * v2;
        o3 = o3 * alpha + beta * v3;
        o4 = o4 * alpha + beta * v4;
        o5 = o5 * alpha + beta * v5;
        o6 = o6 * alpha + beta * v6;
        o7 = o7 * alpha + beta * v7;
    }

    if (lane == 0) {
        part_m[meta_idx] = m_run;
        part_l[meta_idx] = l_run;
    }
    dst_o[idx_lo + 0] = o0; dst_o[idx_lo + 1] = o1; dst_o[idx_lo + 2] = o2; dst_o[idx_lo + 3] = o3;
    dst_o[idx_hi + 0] = o4; dst_o[idx_hi + 1] = o5; dst_o[idx_hi + 2] = o6; dst_o[idx_hi + 3] = o7;
}

extern "C" __global__ void __launch_bounds__(32, 16) attn_decode_q4_0_splitk_sm89(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int pos_base, int n_splits
) { attn_decode_q4_0_splitk_sm89_impl(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, pos_base, n_splits); }

extern "C" __global__ void __launch_bounds__(32, 16) attn_decode_q4_0_splitk_sm89_dpos(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_splits
) { attn_decode_q4_0_splitk_sm89_impl(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_splits); }

// =============================================================================
// 4. Fused Split-K Reduce + Sigmoid(gate) + escha_rin + Had128 -> FP16
//    Merges partial (m, l, o) across n_splits, applies Qwen3.5 sigmoid(gate),
//    and outputs FP16 Hadamard-rotated activation directly for attn_output Escha GEMV!
//
//    Grid : (48, n_tokens, 1) — 24 heads x 2 halves of 128 dims = 48 warps/token
//    Block: (32, 1, 1)        — 1 warp handles 128 contiguous dims (4 floats/lane)
// =============================================================================
extern "C" __global__ void __launch_bounds__(32, 16) attn_splitk_reduce_gate_had128_f16(
    const float* __restrict__ part_o, // [n_tokens, 24, n_splits, 256]
    const float* __restrict__ part_m, // [n_tokens, 24, n_splits]
    const float* __restrict__ part_l, // [n_tokens, 24, n_splits]
    const float* __restrict__ gate,   // [n_tokens, 24, 256]
    const __half* __restrict__ rin,   // [6144]
    __half* __restrict__ xh_out,      // [n_tokens, 6144]
    int n_splits
) {
    const int blk128 = blockIdx.x;    // 0..47 (head = blk128 >> 1, half = blk128 & 1)
    const int hq = blk128 >> 1;       // 0..23
    const int half_off = (blk128 & 1) * 128;
    const int t = blockIdx.y;         // 0..M-1
    const int lane = threadIdx.x;     // 0..31

    const size_t meta_base = ((size_t)t * ATTN_HQ + hq) * n_splits;

    // 1. Find global max m_max across n_splits
    float m_max = -1.0e30f;
    for (int s = 0; s < n_splits; ++s) {
        const float ms = part_m[meta_base + s];
        if (ms > m_max) m_max = ms;
    }

    // 2. Accumulate rescaled partial o and l
    float l_tot = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < n_splits; ++s) {
        const float ls = part_l[meta_base + s];
        if (ls <= 0.0f) continue;
        const float w = expf(part_m[meta_base + s] - m_max);
        l_tot += ls * w;
        const float* os = part_o + (meta_base + s) * ATTN_D + half_off;
        const float4 ov = ((const float4*)os)[lane];
        acc.x += ov.x * w;
        acc.y += ov.y * w;
        acc.z += ov.z * w;
        acc.w += ov.w * w;
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    const size_t flat_off = (size_t)t * 6144 + (size_t)blk128 * 128;
    const size_t rin_off = (size_t)blk128 * 128;

    const float4 g = ((const float4*)(gate + flat_off))[lane];
    const half4 r = ((const half4*)(rin + rin_off))[lane];

    float4 v;
    v.x = (acc.x * inv_l) * (1.0f / (1.0f + expf(-g.x))) * __low2float(r.x);
    v.y = (acc.y * inv_l) * (1.0f / (1.0f + expf(-g.y))) * __high2float(r.x);
    v.z = (acc.z * inv_l) * (1.0f / (1.0f + expf(-g.z))) * __low2float(r.y);
    v.w = (acc.w * inv_l) * (1.0f / (1.0f + expf(-g.w))) * __high2float(r.y);

    had_lane_128(v, lane);

    half4 out_h;
    out_h.x = __floats2half2_rn(v.x * ATTN_INV_SQRT128, v.y * ATTN_INV_SQRT128);
    out_h.y = __floats2half2_rn(v.z * ATTN_INV_SQRT128, v.w * ATTN_INV_SQRT128);
    ((half4*)(xh_out + flat_off))[lane] = out_h;
}

// =============================================================================
// 5. High-Throughput GQA6x4 Shared-Memory Tiled Q4_0 Split-K FlashAttention (sm_89)
//    1 Block = 1 KV head x 4 draft tokens x ALL 6 GQA Q-heads = 24 warps (768 threads).
//    Loads each 16-token Q4_0 K/V tile ONCE into 16 KiB FP16 shared memory and reuses
//    it across all 24 (q_head, token) warps -> 24x DRAM bandwidth reduction vs 1-warp kernel!
//
//    Grid : (4, ceil(n_tokens / 4), n_splits)
//    Block: (768, 1, 1) — 24 warps, 16 KiB shared memory
// =============================================================================
#define FA_TILE_KV 16

static __device__ __forceinline__ void attn_decode_q4_0_gqa6_sm89_impl(
    const float* __restrict__ q_rot,        // [n_tokens, 24, 256]
    const uint8_t* __restrict__ k_cache_q4, // [max_cells, 4, 144]
    const uint8_t* __restrict__ v_cache_q4, // [max_cells, 4, 144]
    float* __restrict__ part_o,             // [n_tokens, 24, n_splits, 256]
    float* __restrict__ part_m,             // [n_tokens, 24, n_splits]
    float* __restrict__ part_l,             // [n_tokens, 24, n_splits]
    int pos_base,
    int n_tokens,
    int n_splits
) {
    const int hkv = blockIdx.x;        // 0..3
    const int t_base = blockIdx.y * 4; // 0, 4, ...
    const int sp = blockIdx.z;         // 0..n_splits-1
    const int tid = threadIdx.x;       // 0..767
    const int warp = tid >> 5;         // 0..23
    const int lane = tid & 31;         // 0..31

    const int q_sub = warp >> 2;       // 0..5 (6 GQA heads per KV head)
    const int t_sub = warp & 3;        // 0..3 (4 draft tokens per block)
    const int hq = hkv * 6 + q_sub;
    const int t = t_base + t_sub;
    const bool t_valid = (t < n_tokens);

    const int blk = lane >> 2;         // 0..7
    const int sub = lane & 3;          // 0..3
    const int idx_lo = blk * 32 + sub * 4;
    const int idx_hi = idx_lo + 16;

    const int max_t_in_blk = (t_base + 4 < n_tokens) ? (t_base + 4) : n_tokens;
    const int max_seq_len = pos_base + max_t_in_blk;
    const int chunk = (max_seq_len + n_splits - 1) / n_splits;
    const int p_start = sp * chunk;
    const int p_end = (p_start + chunk < max_seq_len) ? (p_start + chunk) : max_seq_len;

    half4 qlo, qhi;
    qlo.x = __floats2half2_rn(0.0f, 0.0f); qlo.y = qlo.x;
    qhi.x = qlo.x; qhi.y = qlo.x;
    if (t_valid) {
        const float* q_ptr = q_rot + ((size_t)t * ATTN_HQ + hq) * ATTN_D;
        qlo.x = __floats2half2_rn(q_ptr[idx_lo + 0], q_ptr[idx_lo + 1]);
        qlo.y = __floats2half2_rn(q_ptr[idx_lo + 2], q_ptr[idx_lo + 3]);
        qhi.x = __floats2half2_rn(q_ptr[idx_hi + 0], q_ptr[idx_hi + 1]);
        qhi.y = __floats2half2_rn(q_ptr[idx_hi + 2], q_ptr[idx_hi + 3]);
    }

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float o4 = 0.0f, o5 = 0.0f, o6 = 0.0f, o7 = 0.0f;
    float m_run = -1.0e30f;
    float l_run = 0.0f;

    __shared__ __align__(16) half4 sh_k_lo[FA_TILE_KV][32];
    __shared__ __align__(16) half4 sh_k_hi[FA_TILE_KV][32];
    __shared__ __align__(16) half4 sh_v_lo[FA_TILE_KV][32];
    __shared__ __align__(16) half4 sh_v_hi[FA_TILE_KV][32];

    const size_t kv_stride = (size_t)ATTN_HKV * Q4_0_HEAD_BYTES;
    const uint8_t* k_base = k_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;
    const uint8_t* v_base = v_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;
    const int causal_limit = pos_base + t;

    for (int p_tile = p_start; p_tile < p_end; p_tile += FA_TILE_KV) {
        const int tile_len = (p_tile + FA_TILE_KV < p_end) ? FA_TILE_KV : (p_end - p_tile);

        // First 16 warps (512 threads) cooperatively load & dequantize 16 K + 16 V heads
        if (warp < tile_len) {
            const int p = p_tile + warp;
            float k0, k1, k2, k3, k4, k5, k6, k7;
            load_q4_0_8(k_base + (size_t)p * kv_stride, blk, sub, k0, k1, k2, k3, k4, k5, k6, k7);
            half4 klo, khi;
            klo.x = __floats2half2_rn(k0, k1);
            klo.y = __floats2half2_rn(k2, k3);
            khi.x = __floats2half2_rn(k4, k5);
            khi.y = __floats2half2_rn(k6, k7);
            sh_k_lo[warp][lane] = klo;
            sh_k_hi[warp][lane] = khi;

            float v0, v1, v2, v3, v4, v5, v6, v7;
            load_q4_0_8(v_base + (size_t)p * kv_stride, blk, sub, v0, v1, v2, v3, v4, v5, v6, v7);
            half4 vlo, vhi;
            vlo.x = __floats2half2_rn(v0, v1);
            vlo.y = __floats2half2_rn(v2, v3);
            vhi.x = __floats2half2_rn(v4, v5);
            vhi.y = __floats2half2_rn(v6, v7);
            sh_v_lo[warp][lane] = vlo;
            sh_v_hi[warp][lane] = vhi;
        }
        __syncthreads();

        if (t_valid) {
            const int j_max = (p_tile + tile_len - 1 <= causal_limit)
                ? tile_len
                : (causal_limit - p_tile + 1);
            // Fast 4-token blocked online softmax: 1 rescale per 4 tokens + MUFU.EX2 (__expf) + HFMA2 QK!
            int j = 0;
            for (; j + 3 < j_max; j += 4) {
                float sc[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const half4 klo = sh_k_lo[j + u][lane];
                    const half4 khi = sh_k_hi[j + u][lane];
                    half2 acc2 = __hmul2(qlo.x, klo.x);
                    acc2 = __hfma2(qlo.y, klo.y, acc2);
                    acc2 = __hfma2(qhi.x, khi.x, acc2);
                    acc2 = __hfma2(qhi.y, khi.y, acc2);
                    const float dot = __low2float(acc2) + __high2float(acc2);
                    sc[u] = warp_sum_32(dot) * ATTN_SCALE;
                }
                const float m_blk = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
                const float m_new = fmaxf(m_run, m_blk);
                const float alpha = __expf(m_run - m_new);
                m_run = m_new;
                l_run *= alpha;
                o0 *= alpha; o1 *= alpha; o2 *= alpha; o3 *= alpha;
                o4 *= alpha; o5 *= alpha; o6 *= alpha; o7 *= alpha;

                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float p = __expf(sc[u] - m_new);
                    l_run += p;
                    const half4 vlo = sh_v_lo[j + u][lane];
                    const half4 vhi = sh_v_hi[j + u][lane];
                    o0 = fmaf(p, __low2float(vlo.x), o0);
                    o1 = fmaf(p, __high2float(vlo.x), o1);
                    o2 = fmaf(p, __low2float(vlo.y), o2);
                    o3 = fmaf(p, __high2float(vlo.y), o3);
                    o4 = fmaf(p, __low2float(vhi.x), o4);
                    o5 = fmaf(p, __high2float(vhi.x), o5);
                    o6 = fmaf(p, __low2float(vhi.y), o6);
                    o7 = fmaf(p, __high2float(vhi.y), o7);
                }
            }
            // Tail (0..3 remaining causal tokens)
            for (; j < j_max; ++j) {
                const half4 klo = sh_k_lo[j][lane];
                const half4 khi = sh_k_hi[j][lane];
                half2 acc2 = __hmul2(qlo.x, klo.x);
                acc2 = __hfma2(qlo.y, klo.y, acc2);
                acc2 = __hfma2(qhi.x, khi.x, acc2);
                acc2 = __hfma2(qhi.y, khi.y, acc2);
                const float dot = __low2float(acc2) + __high2float(acc2);
                const float score = warp_sum_32(dot) * ATTN_SCALE;

                const float m_new = fmaxf(m_run, score);
                const float alpha = __expf(m_run - m_new);
                const float beta = __expf(score - m_new);
                l_run = l_run * alpha + beta;
                m_run = m_new;

                const half4 vlo = sh_v_lo[j][lane];
                const half4 vhi = sh_v_hi[j][lane];
                o0 = fmaf(beta, __low2float(vlo.x), o0 * alpha);
                o1 = fmaf(beta, __high2float(vlo.x), o1 * alpha);
                o2 = fmaf(beta, __low2float(vlo.y), o2 * alpha);
                o3 = fmaf(beta, __high2float(vlo.y), o3 * alpha);
                o4 = fmaf(beta, __low2float(vhi.x), o4 * alpha);
                o5 = fmaf(beta, __high2float(vhi.x), o5 * alpha);
                o6 = fmaf(beta, __low2float(vhi.y), o6 * alpha);
                o7 = fmaf(beta, __high2float(vhi.y), o7 * alpha);
            }
        }
        __syncthreads();
    }

    if (t_valid) {
        const size_t meta_idx = ((size_t)t * ATTN_HQ + hq) * n_splits + sp;
        if (lane == 0) {
            part_m[meta_idx] = m_run;
            part_l[meta_idx] = l_run;
        }
        float* dst_o = part_o + meta_idx * ATTN_D;
        dst_o[idx_lo + 0] = o0; dst_o[idx_lo + 1] = o1; dst_o[idx_lo + 2] = o2; dst_o[idx_lo + 3] = o3;
        dst_o[idx_hi + 0] = o4; dst_o[idx_hi + 1] = o5; dst_o[idx_hi + 2] = o6; dst_o[idx_hi + 3] = o7;
    }
}

extern "C" __global__ void __launch_bounds__(768, 1) attn_decode_q4_0_gqa6_sm89(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int pos_base, int n_tokens, int n_splits
) { attn_decode_q4_0_gqa6_sm89_impl(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, pos_base, n_tokens, n_splits); }

extern "C" __global__ void __launch_bounds__(768, 1) attn_decode_q4_0_gqa6_sm89_dpos(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_tokens, int n_splits
) { attn_decode_q4_0_gqa6_sm89_impl(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_tokens, n_splits); }
