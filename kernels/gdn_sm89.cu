// Native sm_89 (RTX 4070 Ti, Ada Lovelace) Gated DeltaNet (GDN) fused kernels
// for Kintsugi-27B (Qwen3.8-27B, 48 GDN layers: H_k=16, H_v=48, D=128, d_conv=4).
//
// Fusions vs ggml:
//   1. gdn_conv1d_silu_l2norm_f32 : Causal Conv1D(4) + optional Escha QKV bias + SiLU
//                                   + Q/K L2-norm in ONE kernel (eliminates 3 VRAM passes)
//   2. gdn_alpha_beta_f16_f32     : FP16 alpha/beta GEMV + dt_bias + softplus*a + exp + sigmoid
//                                   in ONE kernel (eliminates 4 kernels)
//   3. gdn_recurrence_sm89        : Register-resident S[128][128] recurrence (4 cols/warp,
//                                   4 rows/lane = 16 regs/thread, 384 blocks across 60 SMs)
//   4. gdn_norm_gate_had128_f16   : RMSNorm(128) * SiLU(z) * escha_rin + Had128 -> FP16
//                                   in ONE zero-smem warp-shuffle kernel (eliminates 3 kernels)

#include <cuda_fp16.h>

typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define GDN_D 128
#define GDN_HK 16
#define GDN_HV 48
#define GDN_QKV_DIM 10240 // 16*128 + 16*128 + 48*128
#define GDN_INV_SQRT128 0.08838834764831845f

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

// =============================================================================
// 1. Fused Causal Conv1D (k=4) + optional QKV bias + SiLU + Q/K L2-Norm
//    Grid : (80, 1, 1)  — 80 head-blocks of 128 channels (0..15=Q, 16..31=K, 32..79=V)
//    Block: (128, 1, 1) — 1 thread per channel in the 128-dim head
//
//    conv_state_in  : [3, 10240] (oldest=row 0, newest=row 2)
//    conv_state_out : [3, 10240] (state after `n_commit` tokens; if <=0, uses n_tokens)
//    qkv_in         : [n_tokens, 10240]
//    qkv_bias       : [10240] or nullptr
//    conv_w         : [10240, 4] (GGUF dims=(4, 10240): tap 0..3 contiguous per channel)
//    q_out          : [n_tokens, 16, 128]
//    k_out          : [n_tokens, 16, 128]
//    v_out          : [n_tokens, 48, 128]
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 4) gdn_conv1d_silu_l2norm_f32(
    const float* __restrict__ qkv_in,
    const float* __restrict__ qkv_bias,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_state_in,
    float* __restrict__ conv_state_out,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int n_tokens,
    int n_commit,
    float eps
) {
    const int head_blk = blockIdx.x; // 0..79
    const int tid = threadIdx.x;     // 0..127
    const int ch = head_blk * GDN_D + tid;

    const float b = qkv_bias ? qkv_bias[ch] : 0.0f;
    const float4 w = ((const float4*)conv_w)[ch]; // w.x=tap0(oldest), w.y=tap1, w.z=tap2, w.w=tap3(curr)

    // Load 3-element history for this channel
    float h0 = conv_state_in[0 * GDN_QKV_DIM + ch];
    float h1 = conv_state_in[1 * GDN_QKV_DIM + ch];
    float h2 = conv_state_in[2 * GDN_QKV_DIM + ch];

    const int commit_at = (n_commit > 0 && n_commit <= n_tokens) ? n_commit : n_tokens;
    const bool is_qk = (head_blk < 2 * GDN_HK);
    __shared__ float sh_warp_sq[4];

    for (int t = 0; t < n_tokens; ++t) {
        const float x_cur = qkv_in[(size_t)t * GDN_QKV_DIM + ch] + b;
        const float conv = h0 * w.x + h1 * w.y + h2 * w.z + x_cur * w.w;
        float val = silu_f32(conv);

        // Advance rolling state
        h0 = h1;
        h1 = h2;
        h2 = x_cur;

        if (conv_state_out && (t + 1) == commit_at) {
            conv_state_out[0 * GDN_QKV_DIM + ch] = h0;
            conv_state_out[1 * GDN_QKV_DIM + ch] = h1;
            conv_state_out[2 * GDN_QKV_DIM + ch] = h2;
        }

        if (is_qk) {
            // Per-128-dim-head L2 norm: scale = rsqrt(max(sum(val^2), eps^2))
            float sq = warp_sum_32(val * val);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            if (lane == 0) sh_warp_sq[warp] = sq;
            __syncthreads();
            float tot = sh_warp_sq[0] + sh_warp_sq[1] + sh_warp_sq[2] + sh_warp_sq[3];
            float inv_norm = rsqrtf(fmaxf(tot, eps * eps));
            val *= inv_norm;
            __syncthreads();

            if (head_blk < GDN_HK) {
                q_out[((size_t)t * GDN_HK + head_blk) * GDN_D + tid] = val;
            } else {
                k_out[((size_t)t * GDN_HK + (head_blk - GDN_HK)) * GDN_D + tid] = val;
            }
        } else {
            v_out[((size_t)t * GDN_HV + (head_blk - 2 * GDN_HK)) * GDN_D + tid] = val;
        }
    }
}

// =============================================================================
// 2. Fused FP16 Alpha/Beta Projection + Gate Activation
//    x        : [n_tokens, 5120] f32 (pre-normed hidden state)
//    w_alpha  : [48, 5120] f16 (GGUF dims=(5120, 48))
//    w_beta   : [48, 5120] f16 (GGUF dims=(5120, 48))
//    dt_bias  : [48] f32
//    ssm_a    : [48] f32
//    g_exp    : [n_tokens, 48] f32 = exp(ssm_a[h] * softplus(alpha + dt_bias[h]))
//    beta_sig : [n_tokens, 48] f32 = sigmoid(beta)
//
//    Grid : (96, 1, 1) — blocks 0..47 compute g_exp[0..M-1, h], 48..95 compute beta_sig[0..M-1, h]
//    Block: (256, 1, 1) — 8 warps reduce K=5120 (20 float4/thread)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) gdn_alpha_beta_f16_f32(
    const float* __restrict__ x,
    const __half* __restrict__ w_alpha,
    const __half* __restrict__ w_beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ g_exp,
    float* __restrict__ beta_sig,
    int n_tokens,
    int d_model
) {
    const bool is_beta = (blockIdx.x >= GDN_HV);
    const int h = is_beta ? (blockIdx.x - GDN_HV) : blockIdx.x;
    const int t_base = blockIdx.y * 8;
    if (t_base >= n_tokens) return;
    const int t_cnt = (n_tokens - t_base < 8) ? (n_tokens - t_base) : 8;
    const __half* w_row = (is_beta ? w_beta : w_alpha) + (size_t)h * d_model;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_vec4 = d_model >> 2;

    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;

    const half4* w4 = reinterpret_cast<const half4*>(w_row);
    for (int i = tid; i < n_vec4; i += 256) {
        const half4 wh = w4[i];
        const float w0 = __low2float(wh.x);
        const float w1 = __high2float(wh.x);
        const float w2 = __low2float(wh.y);
        const float w3 = __high2float(wh.y);
        for (int t = 0; t < t_cnt; ++t) {
            const float4 xv = ((const float4*)(x + (size_t)(t_base + t) * d_model))[i];
            acc[t] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
        }
    }

    __shared__ float sh_red[8][8]; // [t][warp]
    for (int t = 0; t < t_cnt; ++t) {
        float s = warp_sum_32(acc[t]);
        if (lane == 0) sh_red[t][warp] = s;
    }
    __syncthreads();

    if (warp == 0 && lane < t_cnt) {
        const int t = t_base + lane;
        float tot = sh_red[lane][0] + sh_red[lane][1] + sh_red[lane][2] + sh_red[lane][3]
                  + sh_red[lane][4] + sh_red[lane][5] + sh_red[lane][6] + sh_red[lane][7];
        if (is_beta) {
            beta_sig[(size_t)t * GDN_HV + h] = 1.0f / (1.0f + expf(-tot));
        } else {
            const float u = tot + dt_bias[h];
            const float sp = (u > 20.0f) ? u : logf(1.0f + expf(u));
            g_exp[(size_t)t * GDN_HV + h] = expf(ssm_a[h] * sp);
        }
    }
}

// =============================================================================
// 3. Register-Resident GDN State Recurrence (sm_89, 384 blocks across 60 SMs)
//    State layout: S[48, 128, 128] stored transposed as S[h][col][row] (row contiguous).
//    Each warp owns 4 columns x 128 rows (4 rows/lane = 16 floats/thread in registers).
//    Block: (32, 4, 1) = 128 threads = 16 columns/block.
//    Grid : (48, 8, 1) = 384 blocks (48 heads x 8 column tiles of 16).
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 4) gdn_recurrence_sm89(
    const float* __restrict__ q,        // [n_tokens, 16, 128]
    const float* __restrict__ k,        // [n_tokens, 16, 128]
    const float* __restrict__ v,        // [n_tokens, 48, 128]
    const float* __restrict__ g_exp,    // [n_tokens, 48]
    const float* __restrict__ beta_sig, // [n_tokens, 48]
    const float* __restrict__ state_in, // [48, 128, 128]
    float* __restrict__ state_out,      // [48, 128, 128] (may alias state_in)
    float* __restrict__ attn_out,       // [n_tokens, 48, 128]
    int n_tokens,
    int n_commit
) {
    const int h = blockIdx.x;                 // 0..47
    const int hk = h & 15;                    // h % 16 (Kintsugi Escha tiled head order)
    const int lane = threadIdx.x;             // 0..31
    const int col0 = (blockIdx.y * 4 + threadIdx.y) * 4; // 4 cols per warp

    const float* s_in_h = state_in + (size_t)h * (GDN_D * GDN_D);
    float* s_out_h = state_out ? (state_out + (size_t)h * (GDN_D * GDN_D)) : nullptr;

    float s[4][4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            s[c][r] = s_in_h[(col0 + c) * GDN_D + r * 32 + lane];
        }
    }

    const int commit_at = (n_commit > 0 && n_commit <= n_tokens) ? n_commit : n_tokens;

    for (int t = 0; t < n_tokens; ++t) {
        const float* qt = attn_out ? (q + ((size_t)t * GDN_HK + hk) * GDN_D) : nullptr;
        const float* kt = k + ((size_t)t * GDN_HK + hk) * GDN_D;
        const float* vt = v + ((size_t)t * GDN_HV + h) * GDN_D;
        const float gv = g_exp[(size_t)t * GDN_HV + h];
        const float bv = beta_sig[(size_t)t * GDN_HV + h];

        float kr[4], qr[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = r * 32 + lane;
            kr[r] = kt[idx];
            qr[r] = qt ? qt[idx] : 0.0f;
        }

        float* out_t = attn_out ? (attn_out + ((size_t)t * GDN_HV + h) * GDN_D) : nullptr;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float kv_part = s[c][0] * kr[0] + s[c][1] * kr[1] + s[c][2] * kr[2] + s[c][3] * kr[3];
            const float kv_col = warp_sum_32(kv_part);
            const float delta = (vt[col0 + c] - gv * kv_col) * bv;

            float o_part = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float ns = gv * s[c][r] + kr[r] * delta;
                s[c][r] = ns;
                if (out_t) o_part += ns * qr[r];
            }
            if (out_t) {
                const float o_col = warp_sum_32(o_part);
                if (lane == 0) {
                    out_t[col0 + c] = o_col * GDN_INV_SQRT128;
                }
            }
        }

        if (s_out_h && (t + 1) == commit_at) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    s_out_h[(col0 + c) * GDN_D + r * 32 + lane] = s[c][r];
                }
            }
        }
    }
}

// =============================================================================
// 3b. Advance Conv1D state [3, 10240] in-place by n_commit tokens from saved qkv_in + bias
//     Grid: (80, 1, 1), Block: (128, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 4) gdn_commit_conv_state_f32(
    const float* __restrict__ qkv_in,
    const float* __restrict__ qkv_bias,
    float* __restrict__ conv_state, // in-out [3, 10240]
    int n_commit
) {
    const int ch = blockIdx.x * GDN_D + threadIdx.x;
    const float b = qkv_bias ? qkv_bias[ch] : 0.0f;
    float h0 = conv_state[0 * GDN_QKV_DIM + ch];
    float h1 = conv_state[1 * GDN_QKV_DIM + ch];
    float h2 = conv_state[2 * GDN_QKV_DIM + ch];
    for (int t = 0; t < n_commit; ++t) {
        h0 = h1;
        h1 = h2;
        h2 = qkv_in[(size_t)t * GDN_QKV_DIM + ch] + b;
    }
    conv_state[0 * GDN_QKV_DIM + ch] = h0;
    conv_state[1 * GDN_QKV_DIM + ch] = h1;
    conv_state[2 * GDN_QKV_DIM + ch] = h2;
}

// =============================================================================
// 4. Fused GDN Output Norm(128) * SiLU(z + z_bias) * escha_rin + Had128 -> FP16
//    Feeds directly into ssm_out Escha GEMV with ZERO intermediate VRAM round-trip!
//    Grid : (ceil(n_tokens * 48 / 8), 1, 1) — 8 warps/block, 1 warp = 1 head of 128 dims
//    Block: (256, 1, 1)
// =============================================================================
extern "C" __global__ void __launch_bounds__(256, 4) gdn_norm_gate_had128_f16(
    const float* __restrict__ attn_out, // [n_tokens, 6144]
    const float* __restrict__ z_gate,   // [n_tokens, 6144]
    const float* __restrict__ z_bias,   // [6144] or nullptr
    const float* __restrict__ ssm_norm, // [128]
    const __half* __restrict__ rin,     // [6144]
    __half* __restrict__ xh_out,        // [n_tokens, 6144]
    int n_tokens,
    float eps
) {
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int total_heads = n_tokens * GDN_HV;
    if (w >= total_heads) return;

    const int lane = threadIdx.x & 31;
    const int t = w / GDN_HV;
    const int h = w % GDN_HV;
    const size_t base = ((size_t)t * GDN_HV + h) * GDN_D;
    const size_t h_off = (size_t)h * GDN_D;

    float4 a = ((const float4*)(attn_out + base))[lane];
    float sq = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    sq = warp_sum_32(sq);
    const float inv_rms = rsqrtf(sq * (1.0f / 128.0f) + eps);

    const float4 nw = ((const float4*)ssm_norm)[lane];
    float4 z = ((const float4*)(z_gate + base))[lane];
    if (z_bias) {
        const float4 zb = ((const float4*)(z_bias + h_off))[lane];
        z.x += zb.x;
        z.y += zb.y;
        z.z += zb.z;
        z.w += zb.w;
    }
    const half4 r = ((const half4*)(rin + h_off))[lane];

    float4 v;
    v.x = (a.x * inv_rms * nw.x) * silu_f32(z.x) * __low2float(r.x);
    v.y = (a.y * inv_rms * nw.y) * silu_f32(z.y) * __high2float(r.x);
    v.z = (a.z * inv_rms * nw.z) * silu_f32(z.z) * __low2float(r.y);
    v.w = (a.w * inv_rms * nw.w) * silu_f32(z.w) * __high2float(r.y);

    had_lane_128(v, lane);

    half4 out_h;
    out_h.x = __floats2half2_rn(v.x * GDN_INV_SQRT128, v.y * GDN_INV_SQRT128);
    out_h.y = __floats2half2_rn(v.z * GDN_INV_SQRT128, v.w * GDN_INV_SQRT128);
    ((half4*)(xh_out + base))[lane] = out_h;
}

// =============================================================================
// 5. Top-2 Tree Speculation (M=8: 0 -> Branch A [1..4] | Branch B [5..7])
//    5a. Tree Conv1D + SiLU + L2-Norm (saves h0..h2 after t=0, restores before t=5)
//    5b. Tree Register-Resident Recurrence (saves s[4][4] after t=0 in 8 KiB smem -> 0 spills!)
//    5c. Branch-B Save Compaction (copies slots 5..5+n_acc_b-1 -> 1..n_acc_b across all 48 layers)
// =============================================================================
extern "C" __global__ void __launch_bounds__(128, 4) gdn_conv1d_silu_l2norm_tree_f32(
    const float* __restrict__ qkv_in,
    const float* __restrict__ qkv_bias,
    const float* __restrict__ conv_w,
    const float* __restrict__ conv_state_in,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    float eps
) {
    const int head_blk = blockIdx.x; // 0..79
    const int tid = threadIdx.x;     // 0..127
    const int ch = head_blk * GDN_D + tid;

    const float b = qkv_bias ? qkv_bias[ch] : 0.0f;
    const float4 w = ((const float4*)conv_w)[ch];

    float h0 = conv_state_in[0 * GDN_QKV_DIM + ch];
    float h1 = conv_state_in[1 * GDN_QKV_DIM + ch];
    float h2 = conv_state_in[2 * GDN_QKV_DIM + ch];
    float s0 = h0, s1 = h1, s2 = h2;

    const bool is_qk = (head_blk < 2 * GDN_HK);
    __shared__ float sh_warp_sq[4];

    for (int t = 0; t < 8; ++t) {
        if (t == 5) {
            h0 = s0;
            h1 = s1;
            h2 = s2;
        }
        const float x_cur = qkv_in[(size_t)t * GDN_QKV_DIM + ch] + b;
        const float conv = h0 * w.x + h1 * w.y + h2 * w.z + x_cur * w.w;
        float val = silu_f32(conv);

        h0 = h1;
        h1 = h2;
        h2 = x_cur;

        if (t == 0) {
            s0 = h0;
            s1 = h1;
            s2 = h2;
        }

        if (is_qk) {
            float sq = warp_sum_32(val * val);
            const int lane = tid & 31;
            const int warp = tid >> 5;
            if (lane == 0) sh_warp_sq[warp] = sq;
            __syncthreads();
            float tot = sh_warp_sq[0] + sh_warp_sq[1] + sh_warp_sq[2] + sh_warp_sq[3];
            float inv_norm = rsqrtf(fmaxf(tot, eps * eps));
            val *= inv_norm;
            __syncthreads();

            if (head_blk < GDN_HK) {
                q_out[((size_t)t * GDN_HK + head_blk) * GDN_D + tid] = val;
            } else {
                k_out[((size_t)t * GDN_HK + (head_blk - GDN_HK)) * GDN_D + tid] = val;
            }
        } else {
            v_out[((size_t)t * GDN_HV + (head_blk - 2 * GDN_HK)) * GDN_D + tid] = val;
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) gdn_recurrence_tree_sm89(
    const float* __restrict__ q,        // [8, 16, 128]
    const float* __restrict__ k,        // [8, 16, 128]
    const float* __restrict__ v,        // [8, 48, 128]
    const float* __restrict__ g_exp,    // [8, 48]
    const float* __restrict__ beta_sig, // [8, 48]
    const float* __restrict__ state_in, // [48, 128, 128]
    float* __restrict__ attn_out        // [8, 48, 128]
) {
    const int h = blockIdx.x;                 // 0..47
    const int hk = h & 15;                    // h % 16
    const int lane = threadIdx.x;             // 0..31
    const int wy = threadIdx.y;               // 0..3
    const int col0 = (blockIdx.y * 4 + wy) * 4;

    const float* s_in_h = state_in + (size_t)h * (GDN_D * GDN_D);

    // 8 KiB bank-conflict-free shared-memory snapshot of state after t=0 (0 extra regs -> 0 spills!)
    __shared__ float sh_s0[4][4][4][32];

    float s[4][4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            s[c][r] = s_in_h[(col0 + c) * GDN_D + r * 32 + lane];
        }
    }

    for (int t = 0; t < 8; ++t) {
        if (t == 5) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    s[c][r] = sh_s0[wy][c][r][lane];
                }
            }
        }

        const float* qt = q + ((size_t)t * GDN_HK + hk) * GDN_D;
        const float* kt = k + ((size_t)t * GDN_HK + hk) * GDN_D;
        const float* vt = v + ((size_t)t * GDN_HV + h) * GDN_D;
        const float gv = g_exp[(size_t)t * GDN_HV + h];
        const float bv = beta_sig[(size_t)t * GDN_HV + h];

        float kr[4], qr[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = r * 32 + lane;
            kr[r] = kt[idx];
            qr[r] = qt[idx];
        }

        float* out_t = attn_out + ((size_t)t * GDN_HV + h) * GDN_D;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float kv_part = s[c][0] * kr[0] + s[c][1] * kr[1] + s[c][2] * kr[2] + s[c][3] * kr[3];
            const float kv_col = warp_sum_32(kv_part);
            const float delta = (vt[col0 + c] - gv * kv_col) * bv;

            float o_part = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float ns = gv * s[c][r] + kr[r] * delta;
                s[c][r] = ns;
                o_part += ns * qr[r];
            }
            const float o_col = warp_sum_32(o_part);
            if (lane == 0) {
                out_t[col0 + c] = o_col * GDN_INV_SQRT128;
            }
        }

        if (t == 0) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    sh_s0[wy][c][r][lane] = s[c][r];
                }
            }
        }
    }
}

// Grid: (48, n_acc_b, 1), Block: (256, 1, 1)
// Copies Branch-B saved GDN inputs from slot (5 + k) -> slot (1 + k) across all 48 GDN layers.
extern "C" __global__ void __launch_bounds__(256, 4) gdn_compact_branch_b_save(
    float* __restrict__ save_qkv,  // [48, 8, 10240]
    float* __restrict__ save_k,    // [48, 8, 2048]
    float* __restrict__ save_v,    // [48, 8, 6144]
    float* __restrict__ save_g,    // [48, 8, 48]
    float* __restrict__ save_beta  // [48, 8, 48]
) {
    const size_t g_idx = blockIdx.x; // 0..47
    const size_t k = blockIdx.y;     // 0..n_acc_b-1 (0..2)
    const int tid = threadIdx.x;     // 0..255
    const size_t dst_slot = 1 + k;
    const size_t src_slot = 5 + k;

    float4* qkv_dst = reinterpret_cast<float4*>(save_qkv + (g_idx * 8 + dst_slot) * GDN_QKV_DIM);
    const float4* qkv_src = reinterpret_cast<const float4*>(save_qkv + (g_idx * 8 + src_slot) * GDN_QKV_DIM);
    for (int i = tid; i < (GDN_QKV_DIM / 4); i += 256) {
        qkv_dst[i] = qkv_src[i];
    }

    float4* k_dst = reinterpret_cast<float4*>(save_k + (g_idx * 8 + dst_slot) * 2048);
    const float4* k_src = reinterpret_cast<const float4*>(save_k + (g_idx * 8 + src_slot) * 2048);
    for (int i = tid; i < (2048 / 4); i += 256) {
        k_dst[i] = k_src[i];
    }

    float4* v_dst = reinterpret_cast<float4*>(save_v + (g_idx * 8 + dst_slot) * 6144);
    const float4* v_src = reinterpret_cast<const float4*>(save_v + (g_idx * 8 + src_slot) * 6144);
    for (int i = tid; i < (6144 / 4); i += 256) {
        v_dst[i] = v_src[i];
    }

    if (tid < GDN_HV) {
        save_g[(g_idx * 8 + dst_slot) * GDN_HV + tid] = save_g[(g_idx * 8 + src_slot) * GDN_HV + tid];
        save_beta[(g_idx * 8 + dst_slot) * GDN_HV + tid] = save_beta[(g_idx * 8 + src_slot) * GDN_HV + tid];
    }
}
