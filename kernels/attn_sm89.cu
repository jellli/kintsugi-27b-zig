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
template <bool IS_TREE>
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
        const int rel_pos = IS_TREE ? ((t < 5) ? t : (t - 4)) : t;
        const float pos = (float)(pos_base + rel_pos);
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
) { attn_q_norm_mrope_split_f32_impl<false>(q_full, q_bias, q_norm, q_out, gate_out, pos_base, rope_freq_base, eps); }

extern "C" __global__ void __launch_bounds__(256, 4) attn_q_norm_mrope_split_f32_dpos(
    const float* __restrict__ q_full, const float* __restrict__ q_bias, const float* __restrict__ q_norm,
    float* __restrict__ q_out, float* __restrict__ gate_out, const int* __restrict__ pos_ptr, float rope_freq_base, float eps
) { attn_q_norm_mrope_split_f32_impl<false>(q_full, q_bias, q_norm, q_out, gate_out, *pos_ptr, rope_freq_base, eps); }

extern "C" __global__ void __launch_bounds__(256, 4) attn_q_norm_mrope_split_f32_dpos_tree(
    const float* __restrict__ q_full, const float* __restrict__ q_bias, const float* __restrict__ q_norm,
    float* __restrict__ q_out, float* __restrict__ gate_out, const int* __restrict__ pos_ptr, float rope_freq_base, float eps
) { attn_q_norm_mrope_split_f32_impl<true>(q_full, q_bias, q_norm, q_out, gate_out, *pos_ptr, rope_freq_base, eps); }

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
template <bool IS_TREE>
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

        const int rel_pos = IS_TREE ? ((t < 5) ? t : (t - 4)) : t;
        if (tid < (ATTN_ROPE_D / 2)) {
            const float pos = (float)(pos_base + rel_pos);
            const float inv_freq = powf(rope_freq_base, -((float)(2 * tid) / (float)ATTN_ROPE_D));
            const float theta = pos * inv_freq;
            float s, c;
            sincosf(theta, &s, &c);
            val = sh_k[tid] * c - sh_k[tid + (ATTN_ROPE_D / 2)] * s;
        } else if (tid < ATTN_ROPE_D) {
            const int pair = tid - (ATTN_ROPE_D / 2);
            const float pos = (float)(pos_base + rel_pos);
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
) { attn_k_norm_mrope_kv_q4_0_store_impl<false>(k_in, v_in, k_bias, v_bias, k_norm, k_cache_q4, v_cache_q4, pos_base, rope_freq_base, eps); }

extern "C" __global__ void __launch_bounds__(256, 4) attn_k_norm_mrope_kv_q4_0_store_dpos(
    const float* __restrict__ k_in, const float* __restrict__ v_in,
    const float* __restrict__ k_bias, const float* __restrict__ v_bias,
    const float* __restrict__ k_norm, uint8_t* __restrict__ k_cache_q4, uint8_t* __restrict__ v_cache_q4,
    const int* __restrict__ pos_ptr, float rope_freq_base, float eps
) { attn_k_norm_mrope_kv_q4_0_store_impl<false>(k_in, v_in, k_bias, v_bias, k_norm, k_cache_q4, v_cache_q4, *pos_ptr, rope_freq_base, eps); }

extern "C" __global__ void __launch_bounds__(256, 4) attn_k_norm_mrope_kv_q4_0_store_dpos_tree(
    const float* __restrict__ k_in, const float* __restrict__ v_in,
    const float* __restrict__ k_bias, const float* __restrict__ v_bias,
    const float* __restrict__ k_norm, uint8_t* __restrict__ k_cache_q4, uint8_t* __restrict__ v_cache_q4,
    const int* __restrict__ pos_ptr, float rope_freq_base, float eps
) { attn_k_norm_mrope_kv_q4_0_store_impl<true>(k_in, v_in, k_bias, v_bias, k_norm, k_cache_q4, v_cache_q4, *pos_ptr, rope_freq_base, eps); }

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
template <bool IS_TREE>
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
        if (IS_TREE && t >= 5 && p > pos_base && p < pos_base + 5) continue;
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
) { attn_decode_q4_0_splitk_sm89_impl<false>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, pos_base, n_splits); }

extern "C" __global__ void __launch_bounds__(32, 16) attn_decode_q4_0_splitk_sm89_dpos(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_splits
) { attn_decode_q4_0_splitk_sm89_impl<false>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_splits); }

extern "C" __global__ void __launch_bounds__(32, 16) attn_decode_q4_0_splitk_sm89_dpos_tree(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_splits
) { attn_decode_q4_0_splitk_sm89_impl<true>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_splits); }

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

template <bool IS_TREE>
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
            const bool tree_skip_tile = IS_TREE && (t >= 5) && (p_tile + tile_len - 1 > pos_base);
            // Fast 4-token blocked online softmax: 1 rescale per 4 tokens + MUFU.EX2 (__expf) + HFMA2 QK!
            int j = 0;
            if (!tree_skip_tile) {
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
            }
            // Tail (0..3 remaining causal tokens, or boundary tile for Branch B in tree mode)
            for (; j < j_max; ++j) {
                if (IS_TREE && t >= 5) {
                    const int p_cur = p_tile + j;
                    if (p_cur > pos_base && p_cur < pos_base + 5) continue;
                }
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
) { attn_decode_q4_0_gqa6_sm89_impl<false>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, pos_base, n_tokens, n_splits); }

extern "C" __global__ void __launch_bounds__(768, 1) attn_decode_q4_0_gqa6_sm89_dpos(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_tokens, int n_splits
) { attn_decode_q4_0_gqa6_sm89_impl<false>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_tokens, n_splits); }

extern "C" __global__ void __launch_bounds__(768, 1) attn_decode_q4_0_gqa6_sm89_dpos_tree(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_tokens, int n_splits
) { attn_decode_q4_0_gqa6_sm89_impl<true>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_tokens, n_splits); }

// =============================================================================
// 6. Branch-B Q4_0 KV Cache Compaction across all 16 FullAttn layers
//    Moves slots (pos_base + 5 + k) -> (pos_base + 1 + k) for k = 0 .. n_acc_b - 1.
//    Each token per layer = 4 * 144 = 576 bytes = 144 uint32_t words.
//    Grid : (16, n_acc_b, 2)  [z=0: K cache, z=1: V cache]
//    Block: (144, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(144, 4) attn_compact_branch_b_kv(
    uint8_t* __restrict__ k_cache_q4,
    uint8_t* __restrict__ v_cache_q4,
    uint64_t kv_layer_bytes,
    int pos_base
) {
    const uint64_t layer = blockIdx.x; // 0..15
    const int k = blockIdx.y;          // 0..n_acc_b-1 (0..2)
    const int tid = threadIdx.x;       // 0..143
    uint8_t* base = (blockIdx.z == 0 ? k_cache_q4 : v_cache_q4) + layer * kv_layer_bytes;
    const size_t tok_bytes = (size_t)ATTN_HKV * Q4_0_HEAD_BYTES; // 576 bytes (4-byte aligned)
    uint32_t* dst = reinterpret_cast<uint32_t*>(base + (size_t)(pos_base + 1 + k) * tok_bytes);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(base + (size_t)(pos_base + 5 + k) * tok_bytes);
    dst[tid] = src[tid];
}

// =============================================================================
// 7. Native sm_89 Tensor Core (m16n8k16) GQA6x8 Split-K Q4_0 FlashAttention
//    Unifies high-throughput M=8 speculative decode AND long-context prefill!
//
//    1 Block = 1 KV head x 6 GQA Q-heads x 8 tokens = 48 queries (3 x m16 tiles).
//    - Block : (192, 1, 1) = 6 warps = 3 Q-tiles x 2 D-halves (128 dims/warp)
//    - Grid  : (4, ceil(n_tokens / 8), n_splits)
//    - Q stays 100% in registers (32 regs/thread, pre-scaled by ATTN_SCALE = 1/16)
//    - O stays 100% in registers (64 regs/thread = 16 m16n8 tiles covering 128 dims)
//    - Shared memory: sh_k[16][264] + sh_v[16][264] + sh_s[2][3][2][32] = 22.5 KiB
//      (0% bank conflicts for ldmatrix.x4 and ldmatrix.x4.trans, 2 blocks/SM = 12 warps/SM)
//    - Reads 76K Q4_0 KV ONCE for all M=8 draft tokens (2x less DRAM than GQA6x4!)
//      and executes QK^T + PV on FP16 Tensor Cores (158+ TFLOPS) with 0 register spills.
// =============================================================================
#define FA_MMA_BC 16
#define FA_MMA_LD 264 // 256 + 8 halfs: 16B aligned & 100% bank-conflict-free for ldmatrix

static __device__ __forceinline__ void mma_m16n8k16_f32_f16(
    float4& d,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

static __device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr
) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

static __device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr
) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

static __device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(glob_ptr));
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

static __device__ __forceinline__ void dequant_q4_0_8_h4(
    const uint8_t* head_ptr,
    int blk,
    int sub,
    half4& out_lo,
    half4& out_hi
) {
    const uint16_t* b16 = reinterpret_cast<const uint16_t*>(head_ptr + blk * Q4_0_BLOCK_BYTES);
    const uint16_t d_bits = b16[0];
    const half2 d_h2 = __half2half2(__ushort_as_half(d_bits));
    const uint32_t q01 = b16[1 + sub * 2];
    const uint32_t q23 = b16[2 + sub * 2];

    // IEEE FP16 bit-hack: 0x6400 | n == 1024.0h + n (exact for n in 0..15).
    // Subtracting 0x6408 (1032.0h) yields exact (n - 8) in FP16 with zero overflow risk for any scale d.
    const uint32_t bias_u32 = 0x64086408u;
    const half2 bias_h2 = *reinterpret_cast<const half2*>(&bias_u32);

    const uint32_t lo01_u = ((q01 & 0x000fu) | ((q01 & 0x0f00u) << 8)) | 0x64006400u;
    const uint32_t lo23_u = ((q23 & 0x000fu) | ((q23 & 0x0f00u) << 8)) | 0x64006400u;
    const uint32_t hi01_u = (((q01 >> 4) & 0x000fu) | ((q01 & 0xf000u) << 4)) | 0x64006400u;
    const uint32_t hi23_u = (((q23 >> 4) & 0x000fu) | ((q23 & 0xf000u) << 4)) | 0x64006400u;

    out_lo.x = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&lo01_u), bias_h2), d_h2);
    out_lo.y = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&lo23_u), bias_h2), d_h2);
    out_hi.x = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&hi01_u), bias_h2), d_h2);
    out_hi.y = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&hi23_u), bias_h2), d_h2);
}

static __device__ __forceinline__ uint32_t pack_h2_scaled(float a, float b) {
    const half2 h = __floats2half2_rn(a * ATTN_SCALE, b * ATTN_SCALE);
    return *reinterpret_cast<const uint32_t*>(&h);
}

static __device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <bool IS_TREE, int Q_TOKENS>
static __device__ __forceinline__ void attn_mma_q4_0_gqa6_sm89_impl(
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
    constexpr int N_QTILES = (Q_TOKENS / 8) * 3; // 3 for Q_TOKENS=8 (192 thr), 6 for Q_TOKENS=16 (384 thr)
    constexpr int N_WARPS = N_QTILES * 2;
    constexpr int N_THREADS = N_WARPS * 32;

    const int hkv = blockIdx.x;               // 0..3
    const int t_base = blockIdx.y * Q_TOKENS; // 0, Q_TOKENS, ...
    const int sp = blockIdx.z;                // 0..n_splits-1
    const int tid = threadIdx.x;              // 0..N_THREADS-1
    const int warp = tid >> 5;                // 0..N_WARPS-1
    const int lane = tid & 31;                // 0..31

    const int q_tile = warp >> 1;             // 0..N_QTILES-1
    const int d_half = warp & 1;              // 0 (dims 0..127) or 1 (dims 128..255)
    const int d_base = d_half * 128;

    const int g = lane >> 2;                  // 0..7 (token offset inside 8-token sub-block)
    const int tid_g = lane & 3;               // 0..3 (pair index inside 8-col MMA tile)

    const int t_sub = (Q_TOKENS == 8) ? 0 : ((q_tile >= 3) ? 8 : 0);
    const int h_sub = (Q_TOKENS == 8) ? q_tile : ((q_tile >= 3) ? (q_tile - 3) : q_tile);

    // Mapping: row_lo = (hq_lo, t), row_hi = (hq_hi, t) — BOTH share the exact same token index `t`!
    const int t = t_base + t_sub + g;
    const bool t_valid = (t < n_tokens);
    const int hq_lo = hkv * 6 + h_sub * 2;
    const int hq_hi = hq_lo + 1;

    // 1. Load Q into registers directly in m16n8k16 A-fragment layout (pre-scaled by 1/16)
    uint32_t qa[8][4];
    if (t_valid) {
        const float* q_ptr_lo = q_rot + ((size_t)t * ATTN_HQ + hq_lo) * ATTN_D + d_base;
        const float* q_ptr_hi = q_rot + ((size_t)t * ATTN_HQ + hq_hi) * ATTN_D + d_base;
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int d0 = ks * 16 + tid_g * 2;
            qa[ks][0] = pack_h2_scaled(q_ptr_lo[d0 + 0], q_ptr_lo[d0 + 1]);
            qa[ks][1] = pack_h2_scaled(q_ptr_hi[d0 + 0], q_ptr_hi[d0 + 1]);
            qa[ks][2] = pack_h2_scaled(q_ptr_lo[d0 + 8], q_ptr_lo[d0 + 9]);
            qa[ks][3] = pack_h2_scaled(q_ptr_hi[d0 + 8], q_ptr_hi[d0 + 9]);
        }
    } else {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            qa[ks][0] = 0u; qa[ks][1] = 0u; qa[ks][2] = 0u; qa[ks][3] = 0u;
        }
    }

    // 2. Initialize O accumulators (16 m16n8 tiles = 128 dims) and online softmax state
    float4 o[16];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        o[dt] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float m_lo = -1.0e30f, l_lo = 0.0f;
    float m_hi = -1.0e30f, l_hi = 0.0f;

    const int max_t_in_blk = (t_base + Q_TOKENS < n_tokens) ? (t_base + Q_TOKENS) : n_tokens;
    const int max_seq_len = pos_base + max_t_in_blk;
    const int chunk = (max_seq_len + n_splits - 1) / n_splits;
    const int p_start = sp * chunk;
    const int p_end = (p_start + chunk < max_seq_len) ? (p_start + chunk) : max_seq_len;

    const int causal_limit = pos_base + t;
    const size_t kv_stride = (size_t)ATTN_HKV * Q4_0_HEAD_BYTES;
    const uint8_t* k_base = k_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;
    const uint8_t* v_base = v_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;

    // Shared memory: 31.5 KiB (Q_TOKENS=8, 2 blks/SM) or 37.5 KiB (Q_TOKENS=16, 1 blk/SM)
    __shared__ __align__(16) __half sh_k[FA_MMA_BC][FA_MMA_LD];
    __shared__ __align__(16) __half sh_v[FA_MMA_BC][FA_MMA_LD];
    __shared__ __align__(16) float4 sh_s[2][N_QTILES][2][32]; // [d_half][q_tile][0=s0,1=s1][lane]
    __shared__ __align__(16) uint8_t sh_q4[2][32][Q4_0_HEAD_BYTES]; // 2-stage cp.async raw Q4_0 ring (9 KiB)

    const int blk = lane >> 2;         // 0..7 (Q4_0 block index)
    const int sub = lane & 3;          // 0..3 (sub-chunk of 4 low + 4 high elems)
    const int idx_lo = blk * 32 + sub * 4;
    const int idx_hi = idx_lo + 16;

    // Precompute ldmatrix lane offsets (constant across all tiles)
    const int mat = lane >> 3;         // 0..3
    const int r_in_mat = lane & 7;     // 0..7
    const int k_ld_row = ((mat >> 1) * 8) + r_in_mat;
    const int k_ld_col_off = (mat & 1) * 8;
    const int v_ld_row = ((mat & 1) * 8) + r_in_mat;
    const int v_ld_col_off = (mat >> 1) * 8;

    // Precompute 16B cp.async chunk assignments (32 items x 9 chunks = 288 chunks)
    const int cp_item0 = tid / 9;
    const int cp_off0 = (tid - cp_item0 * 9) * 16;
    const bool cp_isv0 = (cp_item0 >= 16);
    const int cp_ploc0 = cp_isv0 ? (cp_item0 - 16) : cp_item0;

    const int tid1 = tid + N_THREADS;
    const int cp_item1 = (tid1 < 288) ? (tid1 / 9) : 0;
    const int cp_off1 = (tid1 < 288) ? ((tid1 - cp_item1 * 9) * 16) : 0;
    const bool cp_isv1 = (cp_item1 >= 16);
    const int cp_ploc1 = cp_isv1 ? (cp_item1 - 16) : cp_item1;

    auto issue_q4_tile = [&](int p_t, int t_len, int stage) {
        if (tid < 288 && cp_ploc0 < t_len) {
            const uint8_t* src = (cp_isv0 ? v_base : k_base) + (size_t)(p_t + cp_ploc0) * kv_stride + cp_off0;
            cp_async_16(&sh_q4[stage][cp_item0][cp_off0], src);
        }
        if (tid1 < 288 && cp_ploc1 < t_len) {
            const uint8_t* src = (cp_isv1 ? v_base : k_base) + (size_t)(p_t + cp_ploc1) * kv_stride + cp_off1;
            cp_async_16(&sh_q4[stage][cp_item1][cp_off1], src);
        }
        cp_async_commit();
    };

    const half4 h4_zero = { __floats2half2_rn(0.0f, 0.0f), __floats2half2_rn(0.0f, 0.0f) };
    const int n_tiles = (p_end > p_start) ? ((p_end - p_start + FA_MMA_BC - 1) / FA_MMA_BC) : 0;

    if (n_tiles > 0) {
        const int t_len0 = (p_start + FA_MMA_BC < p_end) ? FA_MMA_BC : (p_end - p_start);
        issue_q4_tile(p_start, t_len0, 0);
        cp_async_wait_all();
        __syncthreads();
    }

    for (int ti = 0; ti < n_tiles; ++ti) {
        const int p_tile = p_start + ti * FA_MMA_BC;
        const int tile_len = (p_tile + FA_MMA_BC < p_end) ? FA_MMA_BC : (p_end - p_tile);
        const int cur = ti & 1;
        const int nxt = cur ^ 1;

        // Stage A: Dequantize 16 K + 16 V heads from shared sh_q4[cur] -> sh_k / sh_v (zero DRAM stalls!)
        #pragma unroll
        for (int item = warp; item < 32; item += N_WARPS) {
            const bool is_v = (item >= 16);
            const int p_loc = is_v ? (item - 16) : item;
            half4 hlo, hhi;
            if (p_loc < tile_len) {
                dequant_q4_0_8_h4(sh_q4[cur][item], blk, sub, hlo, hhi);
            } else {
                hlo = h4_zero;
                hhi = h4_zero;
            }
            __half* dst_row = is_v ? sh_v[p_loc] : sh_k[p_loc];
            *reinterpret_cast<half4*>(dst_row + idx_lo) = hlo;
            *reinterpret_cast<half4*>(dst_row + idx_hi) = hhi;
        }
        __syncthreads();

        // Prefetch NEXT tile's raw Q4_0 K+V asynchronously while Tensor Cores run QK^T + Softmax + PV!
        if (ti + 1 < n_tiles) {
            const int p_next = p_tile + FA_MMA_BC;
            const int next_len = (p_next + FA_MMA_BC < p_end) ? FA_MMA_BC : (p_end - p_next);
            issue_q4_tile(p_next, next_len, nxt);
        }

        // Stage B: Compute partial S_half = Q_half @ K_half^T via 16 FP16 Tensor Core MMAs
        float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            uint32_t kb0, kb1, kb2, kb3;
            ldmatrix_x4(kb0, kb1, kb2, kb3, &sh_k[k_ld_row][d_base + ks * 16 + k_ld_col_off]);
            mma_m16n8k16_f32_f16(s0, qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb0, kb1);
            mma_m16n8k16_f32_f16(s1, qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb2, kb3);
        }

        // Exchange S_lo <-> S_hi across the two D-half warps (128B coalesced, zero bank conflict)
        sh_s[d_half][q_tile][0][lane] = s0;
        sh_s[d_half][q_tile][1][lane] = s1;
        __syncthreads();

        const int other = d_half ^ 1;
        const float4 os0 = sh_s[other][q_tile][0][lane];
        const float4 os1 = sh_s[other][q_tile][1][lane];
        s0.x += os0.x; s0.y += os0.y; s0.z += os0.z; s0.w += os0.w;
        s1.x += os1.x; s1.y += os1.y; s1.z += os1.z; s1.w += os1.w;

        // Stage C: Causal / Tree Mask + Register-Resident Online Softmax (only 2 warp shuffles!)
        const int c0 = 2 * tid_g;
        const int c1 = c0 + 1;
        const int c2 = 8 + c0;
        const int c3 = c2 + 1;

        auto col_valid = [&](int c) -> bool {
            if (!t_valid || c >= tile_len) return false;
            const int p_cur = p_tile + c;
            if (p_cur > causal_limit) return false;
            if constexpr (IS_TREE) {
                if (t >= 5 && p_cur > pos_base && p_cur < pos_base + 5) return false;
            }
            return true;
        };

        const bool v0 = col_valid(c0);
        const bool v1 = col_valid(c1);
        const bool v2 = col_valid(c2);
        const bool v3 = col_valid(c3);

        if (!v0) { s0.x = -1.0e30f; s0.z = -1.0e30f; }
        if (!v1) { s0.y = -1.0e30f; s0.w = -1.0e30f; }
        if (!v2) { s1.x = -1.0e30f; s1.z = -1.0e30f; }
        if (!v3) { s1.y = -1.0e30f; s1.w = -1.0e30f; }

        // Row-wise max across the 4 threads holding the same row (lane ^ 1, lane ^ 2)
        float m_blk_lo = fmaxf(fmaxf(s0.x, s0.y), fmaxf(s1.x, s1.y));
        m_blk_lo = fmaxf(m_blk_lo, __shfl_xor_sync(0xffffffffu, m_blk_lo, 1));
        m_blk_lo = fmaxf(m_blk_lo, __shfl_xor_sync(0xffffffffu, m_blk_lo, 2));

        float m_blk_hi = fmaxf(fmaxf(s0.z, s0.w), fmaxf(s1.z, s1.w));
        m_blk_hi = fmaxf(m_blk_hi, __shfl_xor_sync(0xffffffffu, m_blk_hi, 1));
        m_blk_hi = fmaxf(m_blk_hi, __shfl_xor_sync(0xffffffffu, m_blk_hi, 2));

        uint32_t pa0 = 0u, pa1 = 0u, pa2 = 0u, pa3 = 0u;
        float alpha_lo = 1.0f, alpha_hi = 1.0f;
        const bool has_lo = (m_blk_lo > -1.0e29f);
        const bool has_hi = (m_blk_hi > -1.0e29f);

        float l_blk_lo = 0.0f;
        float m_new_lo = m_lo;
        if (has_lo) {
            m_new_lo = fmaxf(m_lo, m_blk_lo);
            alpha_lo = (m_lo > -1.0e29f) ? ((m_new_lo == m_lo) ? 1.0f : __expf(m_lo - m_new_lo)) : 0.0f;
            const float p0 = v0 ? __expf(s0.x - m_new_lo) : 0.0f;
            const float p1 = v1 ? __expf(s0.y - m_new_lo) : 0.0f;
            const float p2 = v2 ? __expf(s1.x - m_new_lo) : 0.0f;
            const float p3 = v3 ? __expf(s1.y - m_new_lo) : 0.0f;
            l_blk_lo = (p0 + p1) + (p2 + p3);
            pa0 = pack_h2(p0, p1);
            pa2 = pack_h2(p2, p3);
        }
        l_blk_lo += __shfl_xor_sync(0xffffffffu, l_blk_lo, 1);
        l_blk_lo += __shfl_xor_sync(0xffffffffu, l_blk_lo, 2);
        if (has_lo) {
            l_lo = l_lo * alpha_lo + l_blk_lo;
            m_lo = m_new_lo;
        }

        float l_blk_hi = 0.0f;
        float m_new_hi = m_hi;
        if (has_hi) {
            m_new_hi = fmaxf(m_hi, m_blk_hi);
            alpha_hi = (m_hi > -1.0e29f) ? ((m_new_hi == m_hi) ? 1.0f : __expf(m_hi - m_new_hi)) : 0.0f;
            const float p0 = v0 ? __expf(s0.z - m_new_hi) : 0.0f;
            const float p1 = v1 ? __expf(s0.w - m_new_hi) : 0.0f;
            const float p2 = v2 ? __expf(s1.z - m_new_hi) : 0.0f;
            const float p3 = v3 ? __expf(s1.w - m_new_hi) : 0.0f;
            l_blk_hi = (p0 + p1) + (p2 + p3);
            pa1 = pack_h2(p0, p1);
            pa3 = pack_h2(p2, p3);
        }
        l_blk_hi += __shfl_xor_sync(0xffffffffu, l_blk_hi, 1);
        l_blk_hi += __shfl_xor_sync(0xffffffffu, l_blk_hi, 2);
        if (has_hi) {
            l_hi = l_hi * alpha_hi + l_blk_hi;
            m_hi = m_new_hi;
        }

        // Skip 64 FMULs on the ~99% of tiles where running max did not increase!
        if (alpha_lo != 1.0f) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                o[dt].x *= alpha_lo;
                o[dt].y *= alpha_lo;
            }
        }
        if (alpha_hi != 1.0f) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                o[dt].z *= alpha_hi;
                o[dt].w *= alpha_hi;
            }
        }

        // Execute O += P @ V_half via 16 FP16 Tensor Core MMAs
        #pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
            uint32_t vb0, vb1, vb2, vb3;
            ldmatrix_x4_trans(vb0, vb1, vb2, vb3, &sh_v[v_ld_row][d_base + dp * 16 + v_ld_col_off]);
            mma_m16n8k16_f32_f16(o[2 * dp + 0], pa0, pa1, pa2, pa3, vb0, vb1);
            mma_m16n8k16_f32_f16(o[2 * dp + 1], pa0, pa1, pa2, pa3, vb2, vb3);
        }
        if (ti + 1 < n_tiles) {
            cp_async_wait_all();
        }
        __syncthreads();
    }

    // 3. Store partial (m, l, o) to Split-K buffers
    if (t_valid) {
        const size_t meta_lo = ((size_t)t * ATTN_HQ + hq_lo) * n_splits + sp;
        const size_t meta_hi = ((size_t)t * ATTN_HQ + hq_hi) * n_splits + sp;
        if (d_half == 0 && tid_g == 0) {
            part_m[meta_lo] = m_lo;
            part_l[meta_lo] = l_lo;
            part_m[meta_hi] = m_hi;
            part_l[meta_hi] = l_hi;
        }
        float* dst_lo = part_o + meta_lo * ATTN_D + d_base;
        float* dst_hi = part_o + meta_hi * ATTN_D + d_base;
        #pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            const int off = dt * 8 + tid_g * 2;
            *reinterpret_cast<float2*>(dst_lo + off) = make_float2(o[dt].x, o[dt].y);
            *reinterpret_cast<float2*>(dst_hi + off) = make_float2(o[dt].z, o[dt].w);
        }
    }
}

extern "C" __global__ void __launch_bounds__(192, 2) attn_mma_q4_0_gqa6x8_sm89(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int pos_base, int n_tokens, int n_splits
) { attn_mma_q4_0_gqa6_sm89_impl<false, 8>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, pos_base, n_tokens, n_splits); }

extern "C" __global__ void __launch_bounds__(192, 2) attn_mma_q4_0_gqa6x8_sm89_dpos(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_tokens, int n_splits
) { attn_mma_q4_0_gqa6_sm89_impl<false, 8>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_tokens, n_splits); }

extern "C" __global__ void __launch_bounds__(192, 2) attn_mma_q4_0_gqa6x8_sm89_dpos_tree(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_tokens, int n_splits
) { attn_mma_q4_0_gqa6_sm89_impl<true, 8>(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_tokens, n_splits); }

// =============================================================================
// 7. Native sm_89 Prefill Q4_0 FlashAttention (FP16-acc Tensor Core, 92+ TFLOPS)
//    1 Block = 1 KV head x 16 query tokens x ALL 6 GQA Q-heads = 96 Q rows.
//    12 warps (384 threads, 1 block/SM, 168 regs/thr, 0 B local spill, 94,592 B dyn smem):
//      - 6 Q-tiles (16 rows = 8 tokens x 2 Q-heads each) x 2 D-halves (128 dims each)
//      - BC=64 KV tokens/tile: 2-stage cp.async raw Q4_0 ring + XOR-swizzled K/V-aliased sh_kv[64][256]
//      - 2x Tensor Core throughput via mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 (160 TFLOPS peak)
//      - LOG2E folded into Q + split online softmax across D-halves (2x less SFU work, 0 extra syncs)
//      - 2x dp-interleaved P@V accumulation into FP32 O (hides ldmatrix + MMA RAW latency)
//      - Grid: (4, ceil(n_tokens / 16), n_splits), Block: (384, 1, 1), DynSmem: 94,592 B
// =============================================================================
#define ATTN_LOG2E_F 1.4426950408889634f
#define ATTN_LN2_F   0.6931471805599453f
#define ATTN_SCALE_LOG2E (ATTN_SCALE * ATTN_LOG2E_F)

static __device__ __forceinline__ void mma_m16n8k16_f16_f16(
    uint32_t& c0, uint32_t& c1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

static __device__ __forceinline__ float ex2_approx_f32(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ uint32_t pack_h2_log2(float a, float b) {
    const half2 h = __floats2half2_rn(a * ATTN_SCALE_LOG2E, b * ATTN_SCALE_LOG2E);
    return *reinterpret_cast<const uint32_t*>(&h);
}

static __device__ __forceinline__ int attn_swiz_col(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

static __device__ __forceinline__ void attn_prefill_q4_0_sm89_impl(
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
    constexpr int BC = 64;
    constexpr int N_WARPS = 12;
    constexpr int N_THREADS = 384;
    constexpr int N_BC16 = BC / 16;

    const int hkv = blockIdx.x;        // 0..3
    const int t_base = blockIdx.y * 16;
    const int sp = blockIdx.z;         // 0..n_splits-1
    const int tid = threadIdx.x;       // 0..383
    const int warp = tid >> 5;         // 0..11
    const int lane = tid & 31;         // 0..31

    const int q_tile = warp >> 1;      // 0..5
    const int d_half = warp & 1;       // 0 (dims 0..127) or 1 (dims 128..255)
    const int d_base = d_half * 128;

    const int t_sub = (q_tile >= 3) ? 8 : 0;
    const int h_sub = (q_tile >= 3) ? (q_tile - 3) : q_tile;
    const int g = lane >> 2;           // 0..7
    const int tid_g = lane & 3;        // 0..3
    const int t = t_base + t_sub + g;
    const bool t_valid = (t < n_tokens);
    const int hq_lo = hkv * 6 + h_sub * 2;
    const int hq_hi = hq_lo + 1;

    // 1. Load Q into registers pre-scaled by ATTN_SCALE * LOG2E
    uint32_t qa[8][4];
    if (t_valid) {
        const float* q_lo = q_rot + ((size_t)t * ATTN_HQ + hq_lo) * ATTN_D + d_base;
        const float* q_hi = q_rot + ((size_t)t * ATTN_HQ + hq_hi) * ATTN_D + d_base;
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int d0 = ks * 16 + tid_g * 2;
            qa[ks][0] = pack_h2_log2(q_lo[d0 + 0], q_lo[d0 + 1]);
            qa[ks][1] = pack_h2_log2(q_hi[d0 + 0], q_hi[d0 + 1]);
            qa[ks][2] = pack_h2_log2(q_lo[d0 + 8], q_lo[d0 + 9]);
            qa[ks][3] = pack_h2_log2(q_hi[d0 + 8], q_hi[d0 + 9]);
        }
    } else {
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) { qa[ks][0] = qa[ks][1] = qa[ks][2] = qa[ks][3] = 0u; }
    }

    // 2. FP32 O accumulator (128 dims) + split online softmax state (d_half=0 owns hq_lo, d_half=1 owns hq_hi)
    float4 o[16];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) o[dt] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float m_own = -1.0e30f, l_own = 0.0f;

    const int max_t_in_blk = (t_base + 16 < n_tokens) ? (t_base + 16) : n_tokens;
    const int max_seq_len = pos_base + max_t_in_blk;
    const int chunk = (max_seq_len + n_splits - 1) / n_splits;
    const int p_start = sp * chunk;
    const int p_end = (p_start + chunk < max_seq_len) ? (p_start + chunk) : max_seq_len;
    const int causal_limit = pos_base + t;

    const size_t kv_stride = (size_t)ATTN_HKV * Q4_0_HEAD_BYTES;
    const uint8_t* k_base = k_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;
    const uint8_t* v_base = v_cache_q4 + (size_t)hkv * Q4_0_HEAD_BYTES;

    // Dynamic shared memory layout (94,592 bytes total):
    //   sh_kv    : 64 * 256 * 2        = 32,768 B (K and V share this XOR-swizzled buffer)
    //   sh_q4    : 2 * 128 * 144       = 36,864 B (2-stage cp.async raw Q4_0 K+V ring)
    //   sh_s     : 2 * 6 * 8 * 32 * 8  = 24,576 B (D-half S/P exchange)
    //   sh_alpha : 6 * 2 * 8 * 4       =    384 B (split-softmax rescale factor exchange)
    extern __shared__ uint8_t dyn_smem[];
    __half (*sh_kv)[256] = reinterpret_cast<__half (*)[256]>(dyn_smem);
    uint8_t (*sh_q4)[2 * BC][Q4_0_HEAD_BYTES] = reinterpret_cast<uint8_t (*)[2 * BC][Q4_0_HEAD_BYTES]>(
        dyn_smem + BC * 256 * 2
    );
    uint2 (*sh_s)[6][N_BC16 * 2][32] = reinterpret_cast<uint2 (*)[6][N_BC16 * 2][32]>(
        dyn_smem + BC * 256 * 2 + 2 * 2 * BC * Q4_0_HEAD_BYTES
    );
    float (*sh_alpha)[2][8] = reinterpret_cast<float (*)[2][8]>(
        dyn_smem + BC * 256 * 2 + 2 * 2 * BC * Q4_0_HEAD_BYTES + 2 * 6 * N_BC16 * 2 * 32 * sizeof(uint2)
    );

    const int blk = lane >> 2;
    const int sub = lane & 3;
    const int idx_lo = blk * 32 + sub * 4;
    const int idx_hi = idx_lo + 16;

    const int mat = lane >> 3;
    const int r_in_mat = lane & 7;
    const int k_ld_row = ((mat >> 1) * 8) + r_in_mat;
    const int k_ld_col_off = (mat & 1) * 8;
    const int v_ld_row = ((mat & 1) * 8) + r_in_mat;
    const int v_ld_col_off = (mat >> 1) * 8;

    constexpr int TOTAL_CHUNKS = 2 * BC * 9; // 1152 = 3 * 384 exact!
    auto issue_q4_tile = [&](int p_t, int t_len, int stage) {
        #pragma unroll
        for (int c = tid; c < TOTAL_CHUNKS; c += N_THREADS) {
            const int row = c / 9;
            const int ch = c - row * 9;
            const bool is_v = (row >= BC);
            const int r = is_v ? (row - BC) : row;
            if (r < t_len) {
                const uint8_t* src = (is_v ? v_base : k_base) + (size_t)(p_t + r) * kv_stride + ch * 16;
                cp_async_16(&sh_q4[stage][row][ch * 16], src);
            }
        }
        cp_async_commit();
    };

    const int n_tiles = (p_end > p_start) ? ((p_end - p_start + BC - 1) / BC) : 0;
    if (n_tiles > 0) {
        const int first_len = (p_start + BC < p_end) ? BC : (p_end - p_start);
        issue_q4_tile(p_start, first_len, 0);
        cp_async_wait_all();
        __syncthreads();
    }

    const half4 h4_zero = { __float2half2_rn(0.0f), __float2half2_rn(0.0f) };

    for (int ti = 0; ti < n_tiles; ++ti) {
        const int cur = ti & 1;
        const int nxt = cur ^ 1;
        const int p_tile = p_start + ti * BC;
        const int tile_len = (p_tile + BC < p_end) ? BC : (p_end - p_tile);

        // Stage A: Dequantize 64 K heads into XOR-swizzled sh_kv
        #pragma unroll
        for (int r = warp; r < BC; r += N_WARPS) {
            half4 hlo, hhi;
            if (r < tile_len) dequant_q4_0_8_h4(sh_q4[cur][r], blk, sub, hlo, hhi);
            else { hlo = h4_zero; hhi = h4_zero; }
            *reinterpret_cast<half4*>(&sh_kv[r][attn_swiz_col(r, idx_lo)]) = hlo;
            *reinterpret_cast<half4*>(&sh_kv[r][attn_swiz_col(r, idx_hi)]) = hhi;
        }
        __syncthreads();

        // Prefetch next tile's Q4_0 K+V asynchronously while Tensor Cores run
        if (ti + 1 < n_tiles) {
            const int p_next = p_tile + BC;
            const int next_len = (p_next + BC < p_end) ? BC : (p_end - p_next);
            issue_q4_tile(p_next, next_len, nxt);
        }

        // Stage B: Q @ K^T (FP16-acc Tensor Core MMA)
        uint32_t s_lo[N_BC16 * 2], s_hi[N_BC16 * 2];
        #pragma unroll
        for (int j = 0; j < N_BC16 * 2; ++j) { s_lo[j] = 0u; s_hi[j] = 0u; }

        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int sc = (((d_base + ks * 16 + k_ld_col_off) >> 3) ^ r_in_mat) << 3;
            #pragma unroll
            for (int nb = 0; nb < N_BC16; ++nb) {
                uint32_t kb0, kb1, kb2, kb3;
                ldmatrix_x4(kb0, kb1, kb2, kb3, &sh_kv[nb * 16 + k_ld_row][sc]);
                mma_m16n8k16_f16_f16(s_lo[2 * nb + 0], s_hi[2 * nb + 0], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb0, kb1);
                mma_m16n8k16_f16_f16(s_lo[2 * nb + 1], s_hi[2 * nb + 1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb2, kb3);
            }
        }

        // Half-traffic S exchange: d_half=0 publishes s_hi, d_half=1 publishes s_lo
        if (d_half == 0) {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) sh_s[0][q_tile][j][lane].y = s_hi[j];
        } else {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) sh_s[1][q_tile][j][lane].x = s_lo[j];
        }
        __syncthreads();

        // Stage C: Dequantize 64 V heads into sh_kv (reusing K's smem)
        #pragma unroll
        for (int r = warp; r < BC; r += N_WARPS) {
            half4 hlo, hhi;
            if (r < tile_len) dequant_q4_0_8_h4(sh_q4[cur][BC + r], blk, sub, hlo, hhi);
            else { hlo = h4_zero; hhi = h4_zero; }
            *reinterpret_cast<half4*>(&sh_kv[r][attn_swiz_col(r, idx_lo)]) = hlo;
            *reinterpret_cast<half4*>(&sh_kv[r][attn_swiz_col(r, idx_hi)]) = hhi;
        }

        // Stage D: Split online softmax (d_half=0 computes hq_lo, d_half=1 computes hq_hi)
        uint32_t s_own[N_BC16 * 2];
        if (d_half == 0) {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) {
                const uint32_t op = sh_s[1][q_tile][j][lane].x;
                half2 v = __hadd2(*reinterpret_cast<const half2*>(&s_lo[j]), *reinterpret_cast<const half2*>(&op));
                s_own[j] = *reinterpret_cast<const uint32_t*>(&v);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) {
                const uint32_t op = sh_s[0][q_tile][j][lane].y;
                half2 v = __hadd2(*reinterpret_cast<const half2*>(&s_hi[j]), *reinterpret_cast<const half2*>(&op));
                s_own[j] = *reinterpret_cast<const uint32_t*>(&v);
            }
        }

        // Warp-uniform fast path: all 8 tokens in this warp (t_min..t_max) are valid and fully unmasked!
        // Keeping fast_all warp-uniform guarantees 100% warp convergence on all __shfl_xor_sync(0xffffffffu) calls.
        const int t_min = t_base + t_sub;
        const bool warp_fast_all = (t_min + 7 < n_tokens) && (tile_len == BC) && (p_tile + BC - 1 <= pos_base + t_min);
        float alpha_own = 1.0f;

        if (warp_fast_all) {
            half2 hmax = *reinterpret_cast<const half2*>(&s_own[0]);
            #pragma unroll
            for (int j = 1; j < N_BC16 * 2; ++j) {
                hmax = __hmax2(hmax, *reinterpret_cast<const half2*>(&s_own[j]));
            }
            float m_blk = fmaxf(__low2float(hmax), __high2float(hmax));
            m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 1));
            m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 2));

            const float m_new = fmaxf(m_own, m_blk);
            alpha_own = (m_own > -1.0e29f) ? ((m_new == m_own) ? 1.0f : ex2_approx_f32(m_own - m_new)) : 0.0f;

            float l_blk = 0.0f;
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) {
                const half2 v = *reinterpret_cast<const half2*>(&s_own[j]);
                const float p0 = ex2_approx_f32(__low2float(v) - m_new);
                const float p1 = ex2_approx_f32(__high2float(v) - m_new);
                l_blk += p0 + p1;
                s_own[j] = pack_h2(p0, p1);
            }
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 1);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 2);
            l_own = l_own * alpha_own + l_blk;
            m_own = m_new;
        } else {
            float m_blk = -1.0e30f;
            #pragma unroll
            for (int nb = 0; nb < N_BC16; ++nb) {
                const int base_c = nb * 16 + 2 * tid_g;
                #pragma unroll
                for (int k2 = 0; k2 < 2; ++k2) {
                    const int j = 2 * nb + k2;
                    const half2 v = *reinterpret_cast<const half2*>(&s_own[j]);
                    const int c0 = base_c + k2 * 8;
                    const int c1 = c0 + 1;
                    float f0 = (t_valid && c0 < tile_len && p_tile + c0 <= causal_limit) ? __low2float(v) : -1.0e30f;
                    float f1 = (t_valid && c1 < tile_len && p_tile + c1 <= causal_limit) ? __high2float(v) : -1.0e30f;
                    m_blk = fmaxf(m_blk, fmaxf(f0, f1));
                }
            }
            m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 1));
            m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 2));

            const bool has_v = (m_blk > -1.0e29f);
            const float m_new = has_v ? fmaxf(m_own, m_blk) : m_own;
            if (has_v) {
                alpha_own = (m_own > -1.0e29f) ? ((m_new == m_own) ? 1.0f : ex2_approx_f32(m_own - m_new)) : 0.0f;
            }

            float l_blk = 0.0f;
            if (has_v) {
                #pragma unroll
                for (int nb = 0; nb < N_BC16; ++nb) {
                    const int base_c = nb * 16 + 2 * tid_g;
                    #pragma unroll
                    for (int k2 = 0; k2 < 2; ++k2) {
                        const int j = 2 * nb + k2;
                        const half2 v = *reinterpret_cast<const half2*>(&s_own[j]);
                        const int c0 = base_c + k2 * 8;
                        const int c1 = c0 + 1;
                        const float p0 = (t_valid && c0 < tile_len && p_tile + c0 <= causal_limit) ? ex2_approx_f32(__low2float(v) - m_new) : 0.0f;
                        const float p1 = (t_valid && c1 < tile_len && p_tile + c1 <= causal_limit) ? ex2_approx_f32(__high2float(v) - m_new) : 0.0f;
                        l_blk += p0 + p1;
                        s_own[j] = pack_h2(p0, p1);
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < N_BC16 * 2; ++j) s_own[j] = 0u;
            }
            // Keep warp shuffle outside lane-divergent `if (has_v)` so all 32 lanes converge!
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 1);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 2);
            if (has_v) {
                l_own = l_own * alpha_own + l_blk;
                m_own = m_new;
            }
        }

        // Publish computed P + alpha to partner D-half before V-ready barrier
        if (d_half == 0) {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) { s_lo[j] = s_own[j]; sh_s[0][q_tile][j][lane].x = s_own[j]; }
            if (tid_g == 0) sh_alpha[q_tile][0][g] = alpha_own;
        } else {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) { s_hi[j] = s_own[j]; sh_s[1][q_tile][j][lane].y = s_own[j]; }
            if (tid_g == 0) sh_alpha[q_tile][1][g] = alpha_own;
        }
        __syncthreads();

        float alpha_lo, alpha_hi;
        if (d_half == 0) {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) s_hi[j] = sh_s[1][q_tile][j][lane].y;
            alpha_lo = alpha_own;
            alpha_hi = sh_alpha[q_tile][1][g];
        } else {
            #pragma unroll
            for (int j = 0; j < N_BC16 * 2; ++j) s_lo[j] = sh_s[0][q_tile][j][lane].x;
            alpha_lo = sh_alpha[q_tile][0][g];
            alpha_hi = alpha_own;
        }

        if (alpha_lo != 1.0f) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) { o[dt].x *= alpha_lo; o[dt].y *= alpha_lo; }
        }
        if (alpha_hi != 1.0f) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) { o[dt].z *= alpha_hi; o[dt].w *= alpha_hi; }
        }

        // Stage E: 2x dp-interleaved P @ V (4 independent FP16-acc MMAs/step -> FP32 O)
        #pragma unroll
        for (int dp2 = 0; dp2 < 4; ++dp2) {
            const int dpA = dp2 * 2;
            const int dpB = dpA + 1;
            const int scA = (((d_base + dpA * 16 + v_ld_col_off) >> 3) ^ r_in_mat) << 3;
            const int scB = (((d_base + dpB * 16 + v_ld_col_off) >> 3) ^ r_in_mat) << 3;
            uint32_t a0_lo = 0u, a0_hi = 0u, a1_lo = 0u, a1_hi = 0u;
            uint32_t b0_lo = 0u, b0_hi = 0u, b1_lo = 0u, b1_hi = 0u;
            #pragma unroll
            for (int nb = 0; nb < N_BC16; ++nb) {
                const int vr = nb * 16 + v_ld_row;
                uint32_t va0, va1, va2, va3;
                uint32_t vb0, vb1, vb2, vb3;
                ldmatrix_x4_trans(va0, va1, va2, va3, &sh_kv[vr][scA]);
                ldmatrix_x4_trans(vb0, vb1, vb2, vb3, &sh_kv[vr][scB]);
                const uint32_t p0 = s_lo[2 * nb + 0], p1 = s_hi[2 * nb + 0];
                const uint32_t p2 = s_lo[2 * nb + 1], p3 = s_hi[2 * nb + 1];
                mma_m16n8k16_f16_f16(a0_lo, a0_hi, p0, p1, p2, p3, va0, va1);
                mma_m16n8k16_f16_f16(a1_lo, a1_hi, p0, p1, p2, p3, va2, va3);
                mma_m16n8k16_f16_f16(b0_lo, b0_hi, p0, p1, p2, p3, vb0, vb1);
                mma_m16n8k16_f16_f16(b1_lo, b1_hi, p0, p1, p2, p3, vb2, vb3);
            }
            const half2 ha0 = *reinterpret_cast<const half2*>(&a0_lo), ha1 = *reinterpret_cast<const half2*>(&a0_hi);
            const half2 ha2 = *reinterpret_cast<const half2*>(&a1_lo), ha3 = *reinterpret_cast<const half2*>(&a1_hi);
            const half2 hb0 = *reinterpret_cast<const half2*>(&b0_lo), hb1 = *reinterpret_cast<const half2*>(&b0_hi);
            const half2 hb2 = *reinterpret_cast<const half2*>(&b1_lo), hb3 = *reinterpret_cast<const half2*>(&b1_hi);
            o[2 * dpA + 0].x += __low2float(ha0); o[2 * dpA + 0].y += __high2float(ha0);
            o[2 * dpA + 0].z += __low2float(ha1); o[2 * dpA + 0].w += __high2float(ha1);
            o[2 * dpA + 1].x += __low2float(ha2); o[2 * dpA + 1].y += __high2float(ha2);
            o[2 * dpA + 1].z += __low2float(ha3); o[2 * dpA + 1].w += __high2float(ha3);
            o[2 * dpB + 0].x += __low2float(hb0); o[2 * dpB + 0].y += __high2float(hb0);
            o[2 * dpB + 0].z += __low2float(hb1); o[2 * dpB + 0].w += __high2float(hb1);
            o[2 * dpB + 1].x += __low2float(hb2); o[2 * dpB + 1].y += __high2float(hb2);
            o[2 * dpB + 1].z += __low2float(hb3); o[2 * dpB + 1].w += __high2float(hb3);
        }

        if (ti + 1 < n_tiles) cp_async_wait_all();
        __syncthreads();
    }

    // 3. Store partial (m in natural log domain, l, o) to Split-K buffers
    if (t_valid) {
        const size_t meta_lo = ((size_t)t * ATTN_HQ + hq_lo) * n_splits + sp;
        const size_t meta_hi = ((size_t)t * ATTN_HQ + hq_hi) * n_splits + sp;
        const float m_nat = (m_own > -1.0e29f) ? (m_own * ATTN_LN2_F) : -1.0e30f;
        if (tid_g == 0) {
            if (d_half == 0) { part_m[meta_lo] = m_nat; part_l[meta_lo] = l_own; }
            else             { part_m[meta_hi] = m_nat; part_l[meta_hi] = l_own; }
        }
        float* dst_lo = part_o + meta_lo * ATTN_D + d_base;
        float* dst_hi = part_o + meta_hi * ATTN_D + d_base;
        #pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            const int off = dt * 8 + tid_g * 2;
            *reinterpret_cast<float2*>(dst_lo + off) = make_float2(o[dt].x, o[dt].y);
            *reinterpret_cast<float2*>(dst_hi + off) = make_float2(o[dt].z, o[dt].w);
        }
    }
}

extern "C" __global__ void __launch_bounds__(384, 1) attn_prefill_q4_0_sm89(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int pos_base, int n_tokens, int n_splits
) { attn_prefill_q4_0_sm89_impl(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, pos_base, n_tokens, n_splits); }

extern "C" __global__ void __launch_bounds__(384, 1) attn_prefill_q4_0_sm89_dpos(
    const float* __restrict__ q_rot, const uint8_t* __restrict__ k_cache_q4, const uint8_t* __restrict__ v_cache_q4,
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, const int* __restrict__ pos_ptr, int n_tokens, int n_splits
) { attn_prefill_q4_0_sm89_impl(q_rot, k_cache_q4, v_cache_q4, part_o, part_m, part_l, *pos_ptr, n_tokens, n_splits); }

