//! Complete 64-layer Kintsugi-27B (Qwen3.8-27B) sm_89 Inference Engine (M=1..8).
//! Runs 48 GDN + 16 Full-Attention (Q4_0 KV cache) + 64 SwiGLU FFN + 2-Stage LM Head
//! with zero ggml graph overhead, narrow16 60-SM wave-saturated Escha GEMV,
//! speculative GDN state verify/commit, and DFlash2 hidden-state capture.

const std = @import("std");
const config = @import("config.zig");
const cuda = @import("cuda.zig");
const model_mod = @import("model.zig");

const Spec = config.ModelSpec;

const sm89_core_cubin align(8) = @embedFile("sm89_core.cubin");
const escha_sm89_cubin align(8) = @embedFile("escha_sm89.cubin");
const gdn_sm89_cubin align(8) = @embedFile("gdn_sm89.cubin");
const attn_sm89_cubin align(8) = @embedFile("attn_sm89.cubin");
const lmhead_sm89_cubin align(8) = @embedFile("lmhead_sm89.cubin");

pub const MAX_M: usize = 8;
pub const PREFILL_M: usize = 240; // 15*16 -> 4*15*1=60 blocks (exact 1.0 wave for Prefill FA on 60 SMs with n_splits=1!)
pub const ATTN_PREFILL_SMEM_BYTES: u32 = 94592; // sh_kv(32768) + sh_q4(36864) + sh_s(24576) + sh_alpha(384)
pub const TOPK_K: usize = 64;
pub const TOPK_BLOCKS: usize = 64;
pub const MAX_SPLITS: usize = 30;
pub const MAX_PART_SLOTS: usize = 256; // max (m * n_splits), 256 * 24 * 256 * 4 = 6 MiB
pub const NUM_ATTN_BUCKETS: usize = 5;
pub const Q4_HEAD_BYTES: usize = 144; // 256 / 32 * 18
pub const GDN_QKV_DIM: usize = 10240;
pub const GDN_STATE_BYTES: usize = 48 * 48 * 128 * 128 * 4; // 144 MiB
pub const GDN_CONV_BYTES: usize = 48 * 3 * GDN_QKV_DIM * 4; // 5.625 MiB

pub const Kernels = struct {
    mod_core: cuda.Module,
    mod_escha: cuda.Module,
    mod_gdn: cuda.Module,
    mod_attn: cuda.Module,
    mod_lm: cuda.Module,

    // Core
    fn_norm_2x: cuda.CUfunction,
    fn_norm_3x: cuda.CUfunction,
    fn_norm_f32: cuda.CUfunction,
    fn_capture: cuda.CUfunction,
    fn_q8_to_q4: cuda.CUfunction,

    // Escha
    fn_escha_k2_m1: cuda.CUfunction,
    fn_escha_k3_m1: cuda.CUfunction,
    fn_escha_k2_m8: cuda.CUfunction,
    fn_escha_k3_m8: cuda.CUfunction,
    fn_escha_k3_m8_w1s16: cuda.CUfunction,
    fn_escha_k2_m64: cuda.CUfunction,
    fn_escha_k3_m64: cuda.CUfunction,
    fn_escha_quant_i8: cuda.CUfunction,
    fn_escha_k2_i8_m128: cuda.CUfunction,
    fn_escha_k3_i8_m128: cuda.CUfunction,
    fn_escha_k2_i8_m256: cuda.CUfunction,
    fn_escha_k3_i8_m256: cuda.CUfunction,
    fn_escha_had_out: cuda.CUfunction,

    // GDN
    fn_gdn_conv: cuda.CUfunction,
    fn_gdn_conv_tree: cuda.CUfunction,
    fn_gdn_ab: cuda.CUfunction,
    fn_gdn_rec: cuda.CUfunction,
    fn_gdn_rec_tree: cuda.CUfunction,
    fn_gdn_commit_conv: cuda.CUfunction,
    fn_gdn_compact_b: cuda.CUfunction,
    fn_gdn_post: cuda.CUfunction,

    // FullAttn
    fn_attn_q: cuda.CUfunction,
    fn_attn_q_tree: cuda.CUfunction,
    fn_attn_kv: cuda.CUfunction,
    fn_attn_kv_tree: cuda.CUfunction,
    fn_attn_fa_naive: cuda.CUfunction,
    fn_attn_fa_naive_tree: cuda.CUfunction,
    fn_attn_fa_gqa6: cuda.CUfunction,
    fn_attn_fa_gqa6_tree: cuda.CUfunction,
    fn_attn_fa_prefill: cuda.CUfunction,
    fn_attn_red: cuda.CUfunction,
    fn_attn_compact_b: cuda.CUfunction,

    // FFN + 2-Stage LM Head
    fn_swiglu: cuda.CUfunction,
    fn_q81: cuda.CUfunction,
    fn_q4_gemv: cuda.CUfunction,
    fn_q4_gemv_m1: cuda.CUfunction,
    fn_q4_gemv_m7: cuda.CUfunction,
    fn_q4_gemv_m8: cuda.CUfunction,
    fn_topk1: cuda.CUfunction,
    fn_topk2: cuda.CUfunction,

    pub fn load() !Kernels {
        var mod_core = try cuda.Module.loadData(sm89_core_cubin);
        errdefer mod_core.deinit();
        var mod_escha = try cuda.Module.loadData(escha_sm89_cubin);
        errdefer mod_escha.deinit();
        var mod_gdn = try cuda.Module.loadData(gdn_sm89_cubin);
        errdefer mod_gdn.deinit();
        var mod_attn = try cuda.Module.loadData(attn_sm89_cubin);
        errdefer mod_attn.deinit();
        var mod_lm = try cuda.Module.loadData(lmhead_sm89_cubin);
        errdefer mod_lm.deinit();

        const fn_attn_fa_prefill = try mod_attn.getFunction("attn_prefill_q4_0_sm89_dpos");
        try cuda.setMaxDynamicSharedSize(fn_attn_fa_prefill, ATTN_PREFILL_SMEM_BYTES);

        return .{
            .mod_core = mod_core,
            .mod_escha = mod_escha,
            .mod_gdn = mod_gdn,
            .mod_attn = mod_attn,
            .mod_lm = mod_lm,

            .fn_norm_2x = try mod_core.getFunction("fused_add_bias_rmsnorm_had128_2x_f16"),
            .fn_norm_3x = try mod_core.getFunction("fused_add_bias_rmsnorm_had128_3x_f16"),
            .fn_norm_f32 = try mod_core.getFunction("fused_add_bias_rmsnorm_f32"),
            .fn_capture = try mod_core.getFunction("capture_layer_inp_f32"),
            .fn_q8_to_q4 = try mod_core.getFunction("convert_q8_0_to_q4_0_sm89"),

            .fn_escha_k2_m1 = try mod_escha.getFunction("escha_gemv_k2_m1_narrow_s16"),
            .fn_escha_k3_m1 = try mod_escha.getFunction("escha_gemv_k3_m1_narrow_s16"),
            .fn_escha_k2_m8 = try mod_escha.getFunction("escha_gemv_k2_m8_narrow_s16"),
            .fn_escha_k3_m8 = try mod_escha.getFunction("escha_gemv_k3_m8_narrow_s16"),
            .fn_escha_k3_m8_w1s16 = try mod_escha.getFunction("escha_gemv_k3_m8_narrow_w1s16"),
            .fn_escha_k2_m64 = try mod_escha.getFunction("escha_gemm_k2_m64"),
            .fn_escha_k3_m64 = try mod_escha.getFunction("escha_gemm_k3_m64"),
            .fn_escha_quant_i8 = try mod_escha.getFunction("escha_quant_f16_to_i8"),
            .fn_escha_k2_i8_m128 = try mod_escha.getFunction("escha_gemm_k2_i8_m128n64"),
            .fn_escha_k3_i8_m128 = try mod_escha.getFunction("escha_gemm_k3_i8_m128n64"),
            .fn_escha_k2_i8_m256 = try mod_escha.getFunction("escha_gemm_k2_i8_m256n64"),
            .fn_escha_k3_i8_m256 = try mod_escha.getFunction("escha_gemm_k3_i8_m256n64"),
            .fn_escha_had_out = try mod_escha.getFunction("escha_had_out_f32"),

            .fn_gdn_conv = try mod_gdn.getFunction("gdn_conv1d_silu_l2norm_f32"),
            .fn_gdn_conv_tree = try mod_gdn.getFunction("gdn_conv1d_silu_l2norm_tree_f32"),
            .fn_gdn_ab = try mod_gdn.getFunction("gdn_alpha_beta_f16_f32"),
            .fn_gdn_rec = try mod_gdn.getFunction("gdn_recurrence_sm89"),
            .fn_gdn_rec_tree = try mod_gdn.getFunction("gdn_recurrence_tree_sm89"),
            .fn_gdn_commit_conv = try mod_gdn.getFunction("gdn_commit_conv_state_f32"),
            .fn_gdn_compact_b = try mod_gdn.getFunction("gdn_compact_branch_b_save"),
            .fn_gdn_post = try mod_gdn.getFunction("gdn_norm_gate_had128_f16"),

            .fn_attn_q = try mod_attn.getFunction("attn_q_norm_mrope_split_f32_dpos"),
            .fn_attn_q_tree = try mod_attn.getFunction("attn_q_norm_mrope_split_f32_dpos_tree"),
            .fn_attn_kv = try mod_attn.getFunction("attn_k_norm_mrope_kv_q4_0_store_dpos"),
            .fn_attn_kv_tree = try mod_attn.getFunction("attn_k_norm_mrope_kv_q4_0_store_dpos_tree"),
            .fn_attn_fa_naive = try mod_attn.getFunction("attn_decode_q4_0_splitk_sm89_dpos"),
            .fn_attn_fa_naive_tree = try mod_attn.getFunction("attn_decode_q4_0_splitk_sm89_dpos_tree"),
            .fn_attn_fa_gqa6 = try mod_attn.getFunction("attn_mma_q4_0_gqa6x8_sm89_dpos"),
            .fn_attn_fa_gqa6_tree = try mod_attn.getFunction("attn_mma_q4_0_gqa6x8_sm89_dpos_tree"),
            .fn_attn_fa_prefill = fn_attn_fa_prefill,
            .fn_attn_red = try mod_attn.getFunction("attn_splitk_reduce_gate_had128_f16"),
            .fn_attn_compact_b = try mod_attn.getFunction("attn_compact_branch_b_kv"),

            .fn_swiglu = try mod_lm.getFunction("fused_swiglu_had128_f16"),
            .fn_q81 = try mod_lm.getFunction("quantize_f32_to_q8_1_sm89"),
            .fn_q4_gemv = try mod_lm.getFunction("q4_0_q8_1_gemv_sm89"),
            .fn_q4_gemv_m1 = try mod_lm.getFunction("q4_0_q8_1_gemv_m1_sm89"),
            .fn_q4_gemv_m7 = try mod_lm.getFunction("q4_0_q8_1_gemv_m7_sm89"),
            .fn_q4_gemv_m8 = try mod_lm.getFunction("q4_0_q8_1_gemv_m8_sm89"),
            .fn_topk1 = try mod_lm.getFunction("topk64_stage1_sm89"),
            .fn_topk2 = try mod_lm.getFunction("topk64_stage2_sm89"),
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.mod_lm.deinit();
        self.mod_attn.deinit();
        self.mod_gdn.deinit();
        self.mod_escha.deinit();
        self.mod_core.deinit();
    }
};

pub const CandidateList = struct {
    ids: [TOPK_K]i32,
    logits: [TOPK_K]f32,
};

pub const StepOutput = struct {
    tokens: [MAX_M]u32 = @splat(0),
    top_logits: [MAX_M]f32 = @splat(0),
    cands: [MAX_M]CandidateList = undefined,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    ctx: *const cuda.Context,
    ks: *const Kernels,
    model: *const model_mod.TargetModel,

    // GDN recurrent state (48 layers)
    d_gdn_state: cuda.CUdeviceptr, // [48, 48, 128, 128] f32 = 144 MiB
    d_gdn_conv: cuda.CUdeviceptr,  // [48, 3, 10240] f32 = 5.625 MiB

    // Speculative M>1 GDN verify save buffers (48 layers, ~28.5 MiB)
    d_gdn_save_qkv: cuda.CUdeviceptr,  // [48, 8, 10240] f32 = 15 MiB
    d_gdn_save_k: cuda.CUdeviceptr,    // [48, 8, 2048] f32  = 3 MiB
    d_gdn_save_v: cuda.CUdeviceptr,    // [48, 8, 6144] f32  = 9 MiB
    d_gdn_save_g: cuda.CUdeviceptr,    // [48, 8, 48] f32
    d_gdn_save_beta: cuda.CUdeviceptr, // [48, 8, 48] f32

    // Q4_0 KV cache (16 FullAttn layers)
    d_kv_k: cuda.CUdeviceptr, // [16, max_kv_cells, 4, 144] u8
    d_kv_v: cuda.CUdeviceptr, // [16, max_kv_cells, 4, 144] u8
    max_kv_cells: u32,
    kv_layer_bytes: usize,

    // Static decode scratchpad (~14 MiB)
    d_scratch: cuda.CUdeviceptr,
    d_residual: cuda.CUdeviceptr,  // [8, 5120] f32
    d_delta: cuda.CUdeviceptr,     // [8, 5120] f32 (ssm_out / attn_output / ffn_down output)
    d_xnorm: cuda.CUdeviceptr,     // [8, 5120] f32 (GDN alpha/beta input & final output_norm)
    d_xh0: cuda.CUdeviceptr,       // [8, 17408] f16 (sized for ffn_down too!)
    d_xh1: cuda.CUdeviceptr,       // [8, 5120] f16
    d_xh2: cuda.CUdeviceptr,       // [8, 5120] f16
    d_buf_a: cuda.CUdeviceptr,     // [8, 17408] f32 (ffn_gate / attn_q[12288] / attn_qkv[10240])
    d_buf_b: cuda.CUdeviceptr,     // [8, 17408] f32 (ffn_up / attn_gate[6144] / q_rot[6144])
    d_buf_c: cuda.CUdeviceptr,     // [8, 6144] f32  (gdn_attn_out / attn_gate_split / k_in+v_in)
    d_gdn_q: cuda.CUdeviceptr,     // [8, 2048] f32
    d_gdn_k: cuda.CUdeviceptr,     // [8, 2048] f32
    d_gdn_v: cuda.CUdeviceptr,     // [8, 6144] f32
    d_gdn_g: cuda.CUdeviceptr,     // [8, 48] f32
    d_gdn_beta: cuda.CUdeviceptr,  // [8, 48] f32
    d_part_o: cuda.CUdeviceptr,    // [8, 24, 8, 256] f32 = 1.5 MiB
    d_part_m: cuda.CUdeviceptr,    // [8, 24, 8] f32
    d_part_l: cuda.CUdeviceptr,    // [8, 24, 8] f32
    d_xq81: cuda.CUdeviceptr,      // [8, 160 * 36] u8
    d_logits: cuda.CUdeviceptr,    // [8, 248320] f32 = 7.58 MiB
    d_topk_pval: cuda.CUdeviceptr, // [8, 64 * 64] f32
    d_topk_pidx: cuda.CUdeviceptr, // [8, 64 * 64] i32
    d_topk_oval: cuda.CUdeviceptr, // [8, 64] f32
    d_topk_oidx: cuda.CUdeviceptr, // [8, 64] i32
    d_cap_buf: cuda.CUdeviceptr,   // [PREFILL_M, 5 * 5120] f32 (DFlash2 captured target hidden states!)
    d_escha_i8: cuda.CUdeviceptr,  // [PREFILL_M, 17408] i8 (INT8 Tensor Core prefill activations)
    d_escha_scale: cuda.CUdeviceptr, // [PREFILL_M] f32 (per-row INT8 quantization scales)
    d_meta_pos: cuda.CUdeviceptr,  // [1] i32 (device-side pos for static CUDA Graph capture!)

    // Host staging buffers
    h_emb_buf: []f32,     // [PREFILL_M * 5120]
    h_out_norm: []f32,    // [8 * 5120]
    h_cand_ids: []i32,    // [8 * 64]
    h_cand_vals: []f32,   // [8 * 64]
    h_meta_pos: i32 = 0,

    // Cached static CUDA Graphs (eliminates ~1,550 kernel launches/step!)
    g_verify8: [NUM_ATTN_BUCKETS]?cuda.GraphExec = @splat(null),
    g_verify8_tree: [NUM_ATTN_BUCKETS]?cuda.GraphExec = @splat(null),
    g_commit: [MAX_M]?cuda.GraphExec = @splat(null),

    // DFlash2 capture config (default: [6, 20, 34, 48, 62])
    capture_enabled: bool = false,
    capture_layers: [5]u32 = .{ 6, 20, 34, 48, 62 },

    // Sequence state
    pos: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *const cuda.Context,
        ks: *const Kernels,
        model: *const model_mod.TargetModel,
        reserve_vram_mib: u64,
    ) !Engine {
        // 1. GDN states (48 layers)
        const gdn_state_bytes: usize = 48 * 48 * 128 * 128 * 4;
        const gdn_conv_bytes: usize = 48 * 3 * GDN_QKV_DIM * 4;
        const d_gdn_state = try cuda.Context.alloc(gdn_state_bytes);
        errdefer cuda.Context.free(d_gdn_state);
        const d_gdn_conv = try cuda.Context.alloc(gdn_conv_bytes);
        errdefer cuda.Context.free(d_gdn_conv);

        // 2. Speculative GDN verify buffers
        const d_gdn_save_qkv = try cuda.Context.alloc(48 * MAX_M * GDN_QKV_DIM * 4);
        errdefer cuda.Context.free(d_gdn_save_qkv);
        const d_gdn_save_k = try cuda.Context.alloc(48 * MAX_M * 2048 * 4);
        errdefer cuda.Context.free(d_gdn_save_k);
        const d_gdn_save_v = try cuda.Context.alloc(48 * MAX_M * 6144 * 4);
        errdefer cuda.Context.free(d_gdn_save_v);
        const d_gdn_save_g = try cuda.Context.alloc(48 * MAX_M * 48 * 4);
        errdefer cuda.Context.free(d_gdn_save_g);
        const d_gdn_save_beta = try cuda.Context.alloc(48 * MAX_M * 48 * 4);
        errdefer cuda.Context.free(d_gdn_save_beta);

        // 3. Static scratch arena (activation buffers sized for PREFILL_M; LM head stays MAX_M=8)
        //    Non-overlapping sub-arena aliasing saves 31.3 MiB so 114,688 KV cells fit at PREFILL_M=240:
        //    - d_xh1/d_xh2 (5120+5120 f16) live in upper d_xh0[5120..15360) (<= 17408 = d_ff),
        //      because d_xh0 only uses [0..17408) during ffn_down when d_xh1/d_xh2 are dead.
        //    - d_gdn_q/k/v (2048+2048+6144=10240 f32) live in upper d_buf_b[6144..16384) (<= 17408 = d_ff),
        //      because d_buf_b only uses [0..6144) during GDN (attn_gate) & FullAttn (q_rot).
        //    - d_escha_i8 + d_escha_scale (4.0 MiB) alias d_part_o (6.0 MiB), which is ONLY live
        //      between FullAttn FA and Split-K reduce (never during any launchEscha call).
        //    - d_logits (MAX_M*n_vocab f32 = 7.6 MiB) aliases d_buf_a (15.9 MiB), only used after layer 63.
        comptime {
            std.debug.assert(3 * Spec.d_model <= Spec.d_ff);
            std.debug.assert(6144 + 2048 + 2048 + 6144 <= Spec.d_ff);
            std.debug.assert(PREFILL_M * Spec.d_ff + 256 + PREFILL_M * 4 <= MAX_PART_SLOTS * Spec.n_heads * Spec.head_dim * 4);
            std.debug.assert(MAX_M * Spec.n_vocab * 4 <= PREFILL_M * Spec.d_ff * 4);
        }
        var sc_off: usize = 0;
        const off_residual = bump(&sc_off, PREFILL_M * Spec.d_model * 4);
        const off_delta = bump(&sc_off, PREFILL_M * Spec.d_model * 4);
        const off_xnorm = bump(&sc_off, PREFILL_M * Spec.d_model * 4);
        const off_xh0 = bump(&sc_off, PREFILL_M * Spec.d_ff * 2);
        const off_xh1 = off_xh0 + PREFILL_M * Spec.d_model * 2;
        const off_xh2 = off_xh0 + PREFILL_M * (2 * Spec.d_model) * 2;
        const off_buf_a = bump(&sc_off, PREFILL_M * Spec.d_ff * 4);
        const off_buf_b = bump(&sc_off, PREFILL_M * Spec.d_ff * 4);
        const off_gdn_q = off_buf_b + PREFILL_M * 6144 * 4;
        const off_gdn_k = off_buf_b + PREFILL_M * 8192 * 4;
        const off_gdn_v = off_buf_b + PREFILL_M * 10240 * 4;
        const off_gdn_g = bump(&sc_off, PREFILL_M * 48 * 4);
        const off_gdn_beta = bump(&sc_off, PREFILL_M * 48 * 4);
        const off_part_o = bump(&sc_off, MAX_PART_SLOTS * Spec.n_heads * Spec.head_dim * 4);
        const off_part_m = bump(&sc_off, MAX_PART_SLOTS * Spec.n_heads * 4);
        const off_part_l = bump(&sc_off, MAX_PART_SLOTS * Spec.n_heads * 4);
        const off_xq81 = bump(&sc_off, MAX_M * (Spec.d_model / 32) * 36);
        const off_logits = off_buf_a;
        const off_tk_pval = bump(&sc_off, MAX_M * TOPK_BLOCKS * TOPK_K * 4);
        const off_tk_pidx = bump(&sc_off, MAX_M * TOPK_BLOCKS * TOPK_K * 4);
        const off_tk_oval = bump(&sc_off, MAX_M * TOPK_K * 4);
        const off_tk_oidx = bump(&sc_off, MAX_M * TOPK_K * 4);
        const off_cap_buf = bump(&sc_off, PREFILL_M * 5 * Spec.d_model * 4);
        const off_escha_i8 = off_part_o;
        const off_escha_scale = off_part_o + std.mem.alignForward(usize, PREFILL_M * Spec.d_ff, 256);
        const off_meta_pos = bump(&sc_off, 16);

        const d_scratch = try cuda.Context.alloc(sc_off);
        errdefer cuda.Context.free(d_scratch);

        // 4. Query remaining free VRAM and maximize static Q4_0 KV cache cells!
        const dev_info = try ctx.queryInfo();
        const reserve_bytes = (reserve_vram_mib + 160) * 1024 * 1024; // reserve for DFlash2 + driver headroom
        const kv_budget: u64 = if (dev_info.free_vram > reserve_bytes) dev_info.free_vram - reserve_bytes else 64 * 1024 * 1024;
        const raw_cells = kv_budget / Spec.kv_bytes_per_token;
        const aligned_cells = @min(@as(u64, Spec.max_ctx), @max(@as(u64, 4096), (raw_cells / 4096) * 4096));
        const max_kv_cells: u32 = @intCast(aligned_cells);
        const kv_layer_bytes: usize = @as(usize, max_kv_cells) * Spec.n_kv_heads * Q4_HEAD_BYTES;
        const kv_half_total: usize = Spec.n_full_attn_layers * kv_layer_bytes;

        const d_kv_k = try cuda.Context.alloc(kv_half_total);
        errdefer cuda.Context.free(d_kv_k);
        const d_kv_v = try cuda.Context.alloc(kv_half_total);
        errdefer cuda.Context.free(d_kv_v);

        const h_emb_buf = try allocator.alloc(f32, PREFILL_M * Spec.d_model);
        errdefer allocator.free(h_emb_buf);
        const h_out_norm = try allocator.alloc(f32, MAX_M * Spec.d_model);
        errdefer allocator.free(h_out_norm);
        const h_cand_ids = try allocator.alloc(i32, MAX_M * TOPK_K);
        errdefer allocator.free(h_cand_ids);
        const h_cand_vals = try allocator.alloc(f32, MAX_M * TOPK_K);
        errdefer allocator.free(h_cand_vals);

        var eng: Engine = .{
            .allocator = allocator,
            .ctx = ctx,
            .ks = ks,
            .model = model,
            .d_gdn_state = d_gdn_state,
            .d_gdn_conv = d_gdn_conv,
            .d_gdn_save_qkv = d_gdn_save_qkv,
            .d_gdn_save_k = d_gdn_save_k,
            .d_gdn_save_v = d_gdn_save_v,
            .d_gdn_save_g = d_gdn_save_g,
            .d_gdn_save_beta = d_gdn_save_beta,
            .d_kv_k = d_kv_k,
            .d_kv_v = d_kv_v,
            .max_kv_cells = max_kv_cells,
            .kv_layer_bytes = kv_layer_bytes,
            .d_scratch = d_scratch,
            .d_residual = d_scratch + off_residual,
            .d_delta = d_scratch + off_delta,
            .d_xnorm = d_scratch + off_xnorm,
            .d_xh0 = d_scratch + off_xh0,
            .d_xh1 = d_scratch + off_xh1,
            .d_xh2 = d_scratch + off_xh2,
            .d_buf_a = d_scratch + off_buf_a,
            .d_buf_b = d_scratch + off_buf_b,
            .d_buf_c = d_scratch + off_gdn_v, // FullAttn gate aliases unused d_gdn_v; GDN aout uses d_buf_a
            .d_gdn_q = d_scratch + off_gdn_q,
            .d_gdn_k = d_scratch + off_gdn_k,
            .d_gdn_v = d_scratch + off_gdn_v,
            .d_gdn_g = d_scratch + off_gdn_g,
            .d_gdn_beta = d_scratch + off_gdn_beta,
            .d_part_o = d_scratch + off_part_o,
            .d_part_m = d_scratch + off_part_m,
            .d_part_l = d_scratch + off_part_l,
            .d_xq81 = d_scratch + off_xq81,
            .d_logits = d_scratch + off_logits,
            .d_topk_pval = d_scratch + off_tk_pval,
            .d_topk_pidx = d_scratch + off_tk_pidx,
            .d_topk_oval = d_scratch + off_tk_oval,
            .d_topk_oidx = d_scratch + off_tk_oidx,
            .d_cap_buf = d_scratch + off_cap_buf,
            .d_escha_i8 = d_scratch + off_escha_i8,
            .d_escha_scale = d_scratch + off_escha_scale,
            .d_meta_pos = d_scratch + off_meta_pos,
            .h_emb_buf = h_emb_buf,
            .h_out_norm = h_out_norm,
            .h_cand_ids = h_cand_ids,
            .h_cand_vals = h_cand_vals,
        };
        try eng.reset();
        return eng;
    }

    pub fn deinit(self: *Engine) void {
        for (&self.g_verify8) |*g| {
            if (g.*) |*exec| exec.deinit();
        }
        for (&self.g_verify8_tree) |*g| {
            if (g.*) |*exec| exec.deinit();
        }
        for (&self.g_commit) |*g| {
            if (g.*) |*exec| exec.deinit();
        }
        self.allocator.free(self.h_cand_vals);
        self.allocator.free(self.h_cand_ids);
        self.allocator.free(self.h_out_norm);
        self.allocator.free(self.h_emb_buf);
        cuda.Context.free(self.d_kv_v);
        cuda.Context.free(self.d_kv_k);
        cuda.Context.free(self.d_scratch);
        cuda.Context.free(self.d_gdn_save_beta);
        cuda.Context.free(self.d_gdn_save_g);
        cuda.Context.free(self.d_gdn_save_v);
        cuda.Context.free(self.d_gdn_save_k);
        cuda.Context.free(self.d_gdn_save_qkv);
        cuda.Context.free(self.d_gdn_conv);
        cuda.Context.free(self.d_gdn_state);
    }

    pub fn reset(self: *Engine) !void {
        self.pos = 0;
        try self.ctx.memsetZeroAsync(self.d_gdn_state, GDN_STATE_BYTES);
        try self.ctx.memsetZeroAsync(self.d_gdn_conv, GDN_CONV_BYTES);
        try self.ctx.sync();
    }

    /// Snapshot all 48 GDN recurrent + conv1d states to Host RAM (~150 MiB, ~8-12 ms D2H).
    pub fn saveGdnStateToHost(self: *const Engine, dst_state: []u8, dst_conv: []u8) !void {
        std.debug.assert(dst_state.len == GDN_STATE_BYTES);
        std.debug.assert(dst_conv.len == GDN_CONV_BYTES);
        try self.ctx.copyDtoHAsync(dst_state, self.d_gdn_state);
        try self.ctx.copyDtoHAsync(dst_conv, self.d_gdn_conv);
        try self.ctx.sync();
    }

    /// Restore all 48 GDN recurrent + conv1d states from Host RAM and set `self.pos`.
    pub fn loadGdnStateFromHost(self: *Engine, src_state: []const u8, src_conv: []const u8, pos: u32) !void {
        std.debug.assert(src_state.len == GDN_STATE_BYTES);
        std.debug.assert(src_conv.len == GDN_CONV_BYTES);
        try self.ctx.copyHtoDAsync(self.d_gdn_state, src_state);
        try self.ctx.copyHtoDAsync(self.d_gdn_conv, src_conv);
        self.pos = pos;
    }

    fn launchEscha(
        self: *const Engine,
        proj: model_mod.EschaProj,
        d_xh: cuda.CUdeviceptr,
        d_out: cuda.CUdeviceptr,
        m: u32,
    ) !void {
        var p_a = d_xh;
        var p_b = proj.code;
        var p_rout = proj.rout;
        var p_c = d_out;
        var p_m: c_int = @intCast(m);
        var p_k: c_int = @intCast(proj.ic);
        var p_n: c_int = @intCast(proj.oc);
        var params = [_]?*anyopaque{
            @ptrCast(&p_a),
            @ptrCast(&p_b),
            @ptrCast(&p_rout),
            @ptrCast(&p_c),
            @ptrCast(&p_m),
            @ptrCast(&p_k),
            @ptrCast(&p_n),
        };
        if (m > 32) {
            // Native sm_89 316-TOPS INT8 Tensor Core Escha Prefill GEMM:
            // 1. Per-row symmetric INT8 quantization of Hadamard-rotated FP16 A -> d_escha_i8 + d_escha_scale
            var p_i8 = self.d_escha_i8;
            var p_sc = self.d_escha_scale;
            var p_q = [_]?*anyopaque{
                @ptrCast(&p_a),
                @ptrCast(&p_i8),
                @ptrCast(&p_sc),
                @ptrCast(&p_k),
            };
            try cuda.launch(self.ks.fn_escha_quant_i8, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_q);

            // 2. Hybrid tile dispatch (both bit-exact to each other, 0 B spill):
            //    - m256n64 (BM=256, BN=64, 42 KiB swizzled): 159-184 TOPS on OC >= 6144 when m > 128
            //    - m128n64 (BM=128, BN=64, 30 KiB, 3 blks/SM): better 60-SM balance on OC < 6144 or m <= 128
            var p_i8_gemm = [_]?*anyopaque{
                @ptrCast(&p_i8),
                @ptrCast(&p_sc),
                @ptrCast(&p_b),
                @ptrCast(&p_rout),
                @ptrCast(&p_c),
                @ptrCast(&p_m),
                @ptrCast(&p_k),
                @ptrCast(&p_n),
            };
            const gx_gemm: u32 = proj.oc / 64;
            if (m > 128 and proj.oc >= 6144) {
                const fn_i8 = if (proj.k_bits == 2) self.ks.fn_escha_k2_i8_m256 else self.ks.fn_escha_k3_i8_m256;
                const gy: u32 = (m + 255) / 256;
                try cuda.launch(fn_i8, .{ gx_gemm, gy, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_i8_gemm);
            } else {
                const fn_i8 = if (proj.k_bits == 2) self.ks.fn_escha_k2_i8_m128 else self.ks.fn_escha_k3_i8_m128;
                const gy: u32 = (m + 127) / 128;
                try cuda.launch(fn_i8, .{ gx_gemm, gy, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_i8_gemm);
            }
        } else if (m > MAX_M) {
            const fn_gemm = if (proj.k_bits == 2) self.ks.fn_escha_k2_m64 else self.ks.fn_escha_k3_m64;
            const gx_gemm: u32 = proj.oc / 32;
            const gy: u32 = (m + 63) / 64;
            try cuda.launch(fn_gemm, .{ gx_gemm, gy, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &params);
        } else {
            const gx: u32 = proj.oc / 32;
            const fn_gemv = if (m == 1)
                (if (proj.k_bits == 2) self.ks.fn_escha_k2_m1 else self.ks.fn_escha_k3_m1)
            else
                (if (proj.k_bits == 2) self.ks.fn_escha_k2_m8 else self.ks.fn_escha_k3_m8);
            try cuda.launch(fn_gemv, .{ gx, 1, 1 }, .{ 512, 1, 1 }, 0, self.ctx.stream, &params);
        }

        // In-place 128-pt output Hadamard + escha_rout scaling (~1.5 us)
        var p_oc: c_int = @intCast(proj.oc);
        var p_rows: c_int = @intCast(m);
        var p_had = [_]?*anyopaque{
            @ptrCast(&p_c),
            @ptrCast(&p_rout),
            @ptrCast(&p_c),
            @ptrCast(&p_oc),
            @ptrCast(&p_rows),
        };
        const n_blks: u32 = (m * (proj.oc / 128) + 7) / 8;
        try cuda.launch(self.ks.fn_escha_had_out, .{ n_blks, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_had);
    }

    /// Run a forward pass on `tokens[0..m]` starting at `self.pos`.
    /// - If `commit_immediately == true`: `1 <= m <= PREFILL_M` (64). Advances GDN recurrent & conv state in-place
    ///   for all `m` tokens and sets `self.pos += m`. When `compute_logits == true`, runs LM head ONLY on the final
    ///   token `m - 1` and places its prediction in `out.tokens[0]` / `out.cands[0]`.
    /// - If `commit_immediately == false` (speculative verify): `1 <= m <= MAX_M` (8). Keeps GDN state untouched at
    ///   `self.pos`, saves per-layer GDN step inputs for `commitSpeculative(n_commit)`, and computes logits for all `m` tokens.
    pub fn forward(
        self: *Engine,
        tokens: []const u32,
        commit_immediately: bool,
        compute_logits: bool,
    ) !StepOutput {
        const m: u32 = @intCast(tokens.len);
        std.debug.assert(m >= 1 and m <= (if (commit_immediately) PREFILL_M else MAX_M));
        if (self.pos + m > self.max_kv_cells) return error.ContextFull;

        // 1. Embed tokens on CPU from host mmap Q8_0 -> H2D + upload pos
        for (tokens, 0..) |tok, r| {
            self.model.embedToken(tok, self.h_emb_buf[r * Spec.d_model ..][0..Spec.d_model]);
        }
        try self.ctx.copyHtoDAsync(self.d_residual, std.mem.sliceAsBytes(self.h_emb_buf[0 .. m * Spec.d_model]));
        self.h_meta_pos = @intCast(self.pos);
        try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.asBytes(&self.h_meta_pos));

        const total_seq = self.pos + m;
        const attn_bucket: usize = if (total_seq >= 16384)
            4
        else if (total_seq >= 4096)
            3
        else if (total_seq >= 1024)
            2
        else if (total_seq >= 256)
            1
        else
            0;

        // 2. Execute 64L Trunk (+ Stage-1 GPU LM Head if compute_logits)
        //    Hot speculative verify path (m==8, !commit_immediately, compute_logits, capture_enabled)
        //    uses a cached static CUDA Graph (eliminates ~1,300 kernel launches/step!).
        if (m == 8 and !commit_immediately and compute_logits and self.capture_enabled) {
            if (self.g_verify8[attn_bucket]) |exec| {
                try exec.launch(self.ctx.stream);
            } else {
                try self.ctx.beginCapture();
                try self.launchForwardBody(8, false, true, attn_bucket, false);
                const exec = try self.ctx.endCaptureExec();
                self.g_verify8[attn_bucket] = exec;
                try exec.launch(self.ctx.stream);
            }
        } else {
            try self.launchForwardBody(m, commit_immediately, compute_logits, attn_bucket, false);
        }

        if (commit_immediately) {
            self.pos += m;
        }

        var out: StepOutput = .{};
        if (!compute_logits) {
            return out;
        }

        // 3. Stage-2 Exact Q8_0 rescore of Top-64 candidates on CPU (zero VRAM!)
        if (commit_immediately) {
            const d_last = self.d_xnorm + @as(usize, m - 1) * Spec.d_model * 4;
            try self.finishStage2LmHeadCpu(d_last, 1, &out);
        } else {
            try self.finishStage2LmHeadCpu(self.d_xnorm, m, &out);
        }
        return out;
    }

    /// Run a speculative M=8 Top-2 Tree verification pass:
    ///   Slot 0: `cur` at `pos_base + 0`
    ///   Branch A (slots 1..4): `pos_base + 1..4` (parent chain 0 -> 1 -> 2 -> 3 -> 4)
    ///   Branch B (slots 5..7): `pos_base + 1..3` (parent chain 0 -> 5 -> 6 -> 7)
    /// Keeps GDN state untouched at `self.pos`, saves per-layer GDN/KV/capture buffers for all 8 slots,
    /// and returns Stage-2 exact Top-64 candidates for all 8 slots.
    pub fn forwardTree8(self: *Engine, tokens: *const [MAX_M]u32) !StepOutput {
        const m: u32 = MAX_M;
        if (self.pos + m > self.max_kv_cells) return error.ContextFull;

        for (tokens, 0..) |tok, r| {
            self.model.embedToken(tok, self.h_emb_buf[r * Spec.d_model ..][0..Spec.d_model]);
        }
        try self.ctx.copyHtoDAsync(self.d_residual, std.mem.sliceAsBytes(self.h_emb_buf[0 .. m * Spec.d_model]));
        self.h_meta_pos = @intCast(self.pos);
        try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.asBytes(&self.h_meta_pos));

        const total_seq = self.pos + m;
        const attn_bucket: usize = if (total_seq >= 16384)
            4
        else if (total_seq >= 4096)
            3
        else if (total_seq >= 1024)
            2
        else if (total_seq >= 256)
            1
        else
            0;

        if (self.capture_enabled) {
            if (self.g_verify8_tree[attn_bucket]) |exec| {
                try exec.launch(self.ctx.stream);
            } else {
                try self.ctx.beginCapture();
                try self.launchForwardBody(8, false, true, attn_bucket, true);
                const exec = try self.ctx.endCaptureExec();
                self.g_verify8_tree[attn_bucket] = exec;
                try exec.launch(self.ctx.stream);
            }
        } else {
            try self.launchForwardBody(8, false, true, attn_bucket, true);
        }

        var out: StepOutput = .{};
        try self.finishStage2LmHeadCpu(self.d_xnorm, m, &out);
        return out;
    }

    fn launchForwardBody(
        self: *const Engine,
        m: u32,
        commit_immediately: bool,
        compute_logits: bool,
        attn_bucket: usize,
        is_tree: bool,
    ) !void {
        var prev_bias: cuda.CUdeviceptr = 0;
        var has_delta: c_int = 0;
        var p_dim: c_int = Spec.d_model;
        var p_eps: f32 = Spec.rms_eps;
        var p_m: c_int = @intCast(m);
        var p_dpos = self.d_meta_pos;
        var p_fb: f32 = Spec.rope_theta;

        // Keep m * n_splits <= MAX_PART_SLOTS (256) so d_part_o/m/l never overflow!
        // MMA6x8 launches (4, ceil(m/8), n_splits) at 2 blocks/SM (120 blocks/wave on 60 SMs):
        // - m=1..8 decode: n_splits=15 (60 blks = 1 blk/SM at 1K..4K), n_splits=30 (120 blks = 2 blks/SM at 4K+).
        // - m=240 prefill: n_splits=1 (4*30*1 = 120 blks = exact 1.0 full wave, zero split-K reduce overhead!).
        const n_splits: u32 = if (m <= MAX_M)
            switch (attn_bucket) {
                0 => 1,
                1 => 4,
                2 => 15,
                else => 30,
            }
        else if (m <= 16)
            switch (attn_bucket) {
                0 => 1,
                1 => 4,
                2 => 7,
                else => 15,
            }
        else if (m <= 32)
            switch (attn_bucket) {
                0 => 1,
                1 => 4,
                else => 7,
            }
        else if (m <= 64)
            switch (attn_bucket) {
                0 => 1,
                1 => 2,
                else => 3,
            }
        else if (m <= 128)
            switch (attn_bucket) {
                0 => 1,
                else => 2,
            }
        else
            1;
        std.debug.assert(@as(usize, m) * @as(usize, n_splits) <= MAX_PART_SLOTS);
        const use_gqa6 = (attn_bucket >= 1) or (m > MAX_M);

        // Execute all 64 layers!
        for (0..Spec.n_layers) |l| {
            // Check if this layer's input should be captured for DFlash2
            var cap_slot: ?c_int = null;
            if (self.capture_enabled) {
                for (self.capture_layers, 0..) |cl, sidx| {
                    if (cl == l) {
                        cap_slot = @intCast(sidx);
                        break;
                    }
                }
            }

            switch (self.model.layers[l]) {
                .gdn => |gl| {
                    // 2A. Fused Residual(+prev_ffn_down+bias) + RMSNorm + 2x Had128 (qkv, gate) + x_norm_f32
                    var p_res = self.d_residual;
                    var p_del = self.d_delta;
                    var p_db = prev_bias;
                    var p_nw = gl.attn_norm;
                    var p_r0 = gl.attn_qkv.rin;
                    var p_r1 = gl.attn_gate.rin;
                    var p_x0 = self.d_xh0;
                    var p_x1 = self.d_xh1;
                    var p_xn = self.d_xnorm;
                    var p_n2 = [_]?*anyopaque{
                        @ptrCast(&p_res), @ptrCast(&p_del), @ptrCast(&p_db), @ptrCast(&p_nw),
                        @ptrCast(&p_r0),  @ptrCast(&p_r1),  @ptrCast(&p_x0), @ptrCast(&p_x1),
                        @ptrCast(&p_xn),  @ptrCast(&p_dim), @ptrCast(&p_eps), @ptrCast(&has_delta),
                    };
                    try cuda.launch(self.ks.fn_norm_2x, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_n2);

                    if (cap_slot) |slot| {
                        try self.launchCapture(m, slot);
                    }

                    // 2B. Escha projections: attn_qkv (10240) and attn_gate (6144)
                    const g_idx: usize = gl.gdn_idx;
                    const qkv_dst = if (commit_immediately)
                        self.d_buf_a
                    else
                        self.d_gdn_save_qkv + g_idx * MAX_M * GDN_QKV_DIM * 4;
                    try self.launchEscha(gl.attn_qkv, self.d_xh0, qkv_dst, m);
                    try self.launchEscha(gl.attn_gate, self.d_xh1, self.d_buf_b, m);

                    // 2C. GDN Conv1D + SiLU + L2-Norm
                    const d_k_out = if (commit_immediately) self.d_gdn_k else self.d_gdn_save_k + g_idx * MAX_M * 2048 * 4;
                    const d_v_out = if (commit_immediately) self.d_gdn_v else self.d_gdn_save_v + g_idx * MAX_M * 6144 * 4;
                    const d_conv_st = self.d_gdn_conv + g_idx * 3 * GDN_QKV_DIM * 4;
                    var p_qkv = qkv_dst;
                    var p_qb = gl.attn_qkv.bias;
                    var p_cw = gl.ssm_conv1d;
                    var p_cin = d_conv_st;
                    var p_cout: cuda.CUdeviceptr = if (commit_immediately) d_conv_st else 0;
                    var p_qo = self.d_gdn_q;
                    var p_ko = d_k_out;
                    var p_vo = d_v_out;
                    var p_nc: c_int = @intCast(m);
                    if (is_tree) {
                        var p_conv_t = [_]?*anyopaque{
                            @ptrCast(&p_qkv), @ptrCast(&p_qb), @ptrCast(&p_cw),
                            @ptrCast(&p_cin), @ptrCast(&p_qo), @ptrCast(&p_ko),
                            @ptrCast(&p_vo),  @ptrCast(&p_eps),
                        };
                        try cuda.launch(self.ks.fn_gdn_conv_tree, .{ 80, 1, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_conv_t);
                    } else {
                        var p_conv = [_]?*anyopaque{
                            @ptrCast(&p_qkv), @ptrCast(&p_qb),  @ptrCast(&p_cw),
                            @ptrCast(&p_cin), @ptrCast(&p_cout), @ptrCast(&p_qo),
                            @ptrCast(&p_ko),  @ptrCast(&p_vo),  @ptrCast(&p_m),
                            @ptrCast(&p_nc), @ptrCast(&p_eps),
                        };
                        try cuda.launch(self.ks.fn_gdn_conv, .{ 80, 1, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_conv);
                    }

                    // 2D. GDN alpha + beta FP16 GEMV + exp/sigmoid gates (tiled along y for m > 8)
                    const d_g_out = if (commit_immediately) self.d_gdn_g else self.d_gdn_save_g + g_idx * MAX_M * 48 * 4;
                    const d_beta_out = if (commit_immediately) self.d_gdn_beta else self.d_gdn_save_beta + g_idx * MAX_M * 48 * 4;
                    var p_aw = gl.ssm_alpha;
                    var p_bw = gl.ssm_beta;
                    var p_dtb = gl.ssm_dt_bias;
                    var p_sa = gl.ssm_a;
                    var p_go = d_g_out;
                    var p_bo = d_beta_out;
                    var p_ab = [_]?*anyopaque{
                        @ptrCast(&p_xn), @ptrCast(&p_aw), @ptrCast(&p_bw),
                        @ptrCast(&p_dtb), @ptrCast(&p_sa), @ptrCast(&p_go),
                        @ptrCast(&p_bo), @ptrCast(&p_m), @ptrCast(&p_dim),
                    };
                    try cuda.launch(self.ks.fn_gdn_ab, .{ 96, (m + 7) / 8, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_ab);

                    // 2E. GDN register-resident recurrence (commit in-place if commit_immediately, else read-only verify)
                    const d_st = self.d_gdn_state + g_idx * 48 * 128 * 128 * 4;
                    var p_sin = d_st;
                    var p_sout: cuda.CUdeviceptr = if (commit_immediately) d_st else 0;
                    var p_aout = self.d_buf_a; // d_buf_a is dead after 2C (d_buf_c aliases d_gdn_v!)
                    if (is_tree) {
                        var p_rec_t = [_]?*anyopaque{
                            @ptrCast(&p_qo),  @ptrCast(&p_ko),  @ptrCast(&p_vo),
                            @ptrCast(&p_go),  @ptrCast(&p_bo),  @ptrCast(&p_sin),
                            @ptrCast(&p_aout),
                        };
                        try cuda.launch(self.ks.fn_gdn_rec_tree, .{ 48, 8, 1 }, .{ 32, 4, 1 }, 0, self.ctx.stream, &p_rec_t);
                    } else {
                        var p_rec = [_]?*anyopaque{
                            @ptrCast(&p_qo), @ptrCast(&p_ko), @ptrCast(&p_vo),
                            @ptrCast(&p_go), @ptrCast(&p_bo), @ptrCast(&p_sin),
                            @ptrCast(&p_sout), @ptrCast(&p_aout), @ptrCast(&p_m),
                            @ptrCast(&p_nc),
                        };
                        try cuda.launch(self.ks.fn_gdn_rec, .{ 48, 8, 1 }, .{ 32, 4, 1 }, 0, self.ctx.stream, &p_rec);
                    }

                    // 2F. Fused GDN RMSNorm(128) * SiLU(z + z_bias) * rin + Had128 -> FP16
                    var p_z = self.d_buf_b;
                    var p_zb = gl.attn_gate.bias;
                    var p_snw = gl.ssm_norm;
                    var p_srin = gl.ssm_out.rin;
                    var p_sxh = self.d_xh0;
                    var p_post = [_]?*anyopaque{
                        @ptrCast(&p_aout), @ptrCast(&p_z), @ptrCast(&p_zb),
                        @ptrCast(&p_snw),  @ptrCast(&p_srin), @ptrCast(&p_sxh),
                        @ptrCast(&p_m),    @ptrCast(&p_eps),
                    };
                    try cuda.launch(self.ks.fn_gdn_post, .{ (m * 48 + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_post);

                    // 2G. ssm_out Escha GEMV -> d_delta
                    try self.launchEscha(gl.ssm_out, self.d_xh0, self.d_delta, m);

                    // 2H. FFN block (residual += ssm_out + ssm_out.bias, post_attn_norm, gate/up/SwiGLU/down)
                    try self.runFfnBlock(gl.post_attn_norm, gl.ssm_out.bias, gl.ffn_gate, gl.ffn_up, gl.ffn_down, m);
                    prev_bias = gl.ffn_down.bias;
                    has_delta = 1;
                },

                .full_attn => |al| {
                    // 2A. Fused Residual(+prev_ffn_down+bias) + RMSNorm + 3x Had128 (q, k, v)
                    var p_res = self.d_residual;
                    var p_del = self.d_delta;
                    var p_db = prev_bias;
                    var p_nw = al.attn_norm;
                    var p_r0 = al.attn_q.rin;
                    var p_r1 = al.attn_k.rin;
                    var p_r2 = al.attn_v.rin;
                    var p_x0 = self.d_xh0;
                    var p_x1 = self.d_xh1;
                    var p_x2 = self.d_xh2;
                    var p_n3 = [_]?*anyopaque{
                        @ptrCast(&p_res), @ptrCast(&p_del), @ptrCast(&p_db), @ptrCast(&p_nw),
                        @ptrCast(&p_r0),  @ptrCast(&p_r1),  @ptrCast(&p_r2), @ptrCast(&p_x0),
                        @ptrCast(&p_x1),  @ptrCast(&p_x2),  @ptrCast(&p_dim), @ptrCast(&p_eps),
                        @ptrCast(&has_delta),
                    };
                    try cuda.launch(self.ks.fn_norm_3x, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_n3);

                    if (cap_slot) |slot| {
                        try self.launchCapture(m, slot);
                    }

                    // 2B. Escha projections: attn_q (12288), attn_k (1024), attn_v (1024)
                    const d_kin = self.d_gdn_q; // [8, 1024] fits easily in d_gdn_q [8, 2048]
                    const d_vin = self.d_gdn_k; // [8, 1024] fits easily in d_gdn_k [8, 2048]
                    try self.launchEscha(al.attn_q, self.d_xh0, self.d_buf_a, m);
                    try self.launchEscha(al.attn_k, self.d_xh1, d_kin, m);
                    try self.launchEscha(al.attn_v, self.d_xh2, d_vin, m);

                    // 2C. Q RMSNorm + MRoPE + [Q, gate] split
                    var p_qfull = self.d_buf_a;
                    var p_qb = al.attn_q.bias;
                    var p_qn = al.attn_q_norm;
                    var p_qrot = self.d_buf_b; // [8, 6144]
                    var p_gate = self.d_buf_c; // [8, 6144]
                    var p_q = [_]?*anyopaque{
                        @ptrCast(&p_qfull), @ptrCast(&p_qb), @ptrCast(&p_qn),
                        @ptrCast(&p_qrot),  @ptrCast(&p_gate), @ptrCast(&p_dpos),
                        @ptrCast(&p_fb),    @ptrCast(&p_eps),
                    };
                    const fn_q = if (is_tree) self.ks.fn_attn_q_tree else self.ks.fn_attn_q;
                    try cuda.launch(fn_q, .{ Spec.n_heads, m, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_q);

                    // 2D. K RMSNorm + MRoPE + Q4_0 KV Cache Append at [*dpos .. *dpos+m)
                    const d_kc = self.d_kv_k + @as(usize, al.attn_idx) * self.kv_layer_bytes;
                    const d_vc = self.d_kv_v + @as(usize, al.attn_idx) * self.kv_layer_bytes;
                    var p_ki = d_kin;
                    var p_vi = d_vin;
                    var p_kb = al.attn_k.bias;
                    var p_vb = al.attn_v.bias;
                    var p_kn = al.attn_k_norm;
                    var p_kcp = d_kc;
                    var p_vcp = d_vc;
                    var p_kv = [_]?*anyopaque{
                        @ptrCast(&p_ki),  @ptrCast(&p_vi),  @ptrCast(&p_kb), @ptrCast(&p_vb),
                        @ptrCast(&p_kn),  @ptrCast(&p_kcp), @ptrCast(&p_vcp), @ptrCast(&p_dpos),
                        @ptrCast(&p_fb),  @ptrCast(&p_eps),
                    };
                    const fn_kv = if (is_tree) self.ks.fn_attn_kv_tree else self.ks.fn_attn_kv;
                    try cuda.launch(fn_kv, .{ Spec.n_kv_heads, m, 2 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_kv);

                    // 2E. Split-K Q4_0 FlashAttention (adaptive: FP16-acc Prefill FA for m>32, MMA6x8 for decode/verify, 1-warp for short seq)
                    var p_po = self.d_part_o;
                    var p_pm = self.d_part_m;
                    var p_pl = self.d_part_l;
                    var p_ns: c_int = @intCast(n_splits);

                    if (!is_tree and m > 32) {
                        var p_fa = [_]?*anyopaque{
                            @ptrCast(&p_qrot), @ptrCast(&p_kcp), @ptrCast(&p_vcp),
                            @ptrCast(&p_po),   @ptrCast(&p_pm),  @ptrCast(&p_pl),
                            @ptrCast(&p_dpos), @ptrCast(&p_m),   @ptrCast(&p_ns),
                        };
                        const prefill_ty: u32 = (m + 15) / 16;
                        try cuda.launch(
                            self.ks.fn_attn_fa_prefill,
                            .{ Spec.n_kv_heads, prefill_ty, n_splits },
                            .{ 384, 1, 1 },
                            ATTN_PREFILL_SMEM_BYTES,
                            self.ctx.stream,
                            &p_fa,
                        );
                    } else if (use_gqa6) {
                        var p_fa = [_]?*anyopaque{
                            @ptrCast(&p_qrot), @ptrCast(&p_kcp), @ptrCast(&p_vcp),
                            @ptrCast(&p_po),   @ptrCast(&p_pm),  @ptrCast(&p_pl),
                            @ptrCast(&p_dpos), @ptrCast(&p_m),   @ptrCast(&p_ns),
                        };
                        const gqa6_ty: u32 = (m + 7) / 8;
                        const fn_gqa6 = if (is_tree) self.ks.fn_attn_fa_gqa6_tree else self.ks.fn_attn_fa_gqa6;
                        try cuda.launch(fn_gqa6, .{ Spec.n_kv_heads, gqa6_ty, n_splits }, .{ 192, 1, 1 }, 0, self.ctx.stream, &p_fa);
                    } else {
                        var p_fa = [_]?*anyopaque{
                            @ptrCast(&p_qrot), @ptrCast(&p_kcp), @ptrCast(&p_vcp),
                            @ptrCast(&p_po),   @ptrCast(&p_pm),  @ptrCast(&p_pl),
                            @ptrCast(&p_dpos), @ptrCast(&p_ns),
                        };
                        const fn_naive = if (is_tree) self.ks.fn_attn_fa_naive_tree else self.ks.fn_attn_fa_naive;
                        try cuda.launch(fn_naive, .{ Spec.n_heads, m, n_splits }, .{ 32, 1, 1 }, 0, self.ctx.stream, &p_fa);
                    }

                    // 2F. Fused Split-K Reduce + sigmoid(gate) + rin + Had128 -> FP16
                    var p_orin = al.attn_output.rin;
                    var p_oxh = self.d_xh0;
                    var p_red = [_]?*anyopaque{
                        @ptrCast(&p_po),   @ptrCast(&p_pm),   @ptrCast(&p_pl),
                        @ptrCast(&p_gate), @ptrCast(&p_orin), @ptrCast(&p_oxh),
                        @ptrCast(&p_ns),
                    };
                    try cuda.launch(self.ks.fn_attn_red, .{ 48, m, 1 }, .{ 32, 1, 1 }, 0, self.ctx.stream, &p_red);

                    // 2G. attn_output Escha GEMV -> d_delta
                    try self.launchEscha(al.attn_output, self.d_xh0, self.d_delta, m);

                    // 2H. FFN block
                    try self.runFfnBlock(al.post_attn_norm, al.attn_output.bias, al.ffn_gate, al.ffn_up, al.ffn_down, m);
                    prev_bias = al.ffn_down.bias;
                    has_delta = 1;
                },
            }
        }

        // 3. Final residual add (+ blk.63.ffn_down.bias) + output_norm -> d_xnorm [m, 5120]
        {
            var p_res = self.d_residual;
            var p_del = self.d_delta;
            var p_db = prev_bias;
            var p_nw = self.model.output_norm;
            var p_yo = self.d_xnorm;
            var p_fin = [_]?*anyopaque{
                @ptrCast(&p_res), @ptrCast(&p_del), @ptrCast(&p_db), @ptrCast(&p_nw),
                @ptrCast(&p_yo),  @ptrCast(&p_dim), @ptrCast(&p_eps), @ptrCast(&has_delta),
            };
            try cuda.launch(self.ks.fn_norm_f32, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_fin);
        }

        // 4. Stage-1 GPU LM Head + Exact Top-64 (captured inside CUDA Graph when verifying!)
        if (compute_logits) {
            if (commit_immediately) {
                const d_last = self.d_xnorm + @as(usize, m - 1) * Spec.d_model * 4;
                try self.launchStage1LmHeadGpu(d_last, 1);
            } else {
                try self.launchStage1LmHeadGpu(self.d_xnorm, m);
            }
        }
    }

    fn runFfnBlock(
        self: *const Engine,
        post_norm: cuda.CUdeviceptr,
        attn_out_bias: cuda.CUdeviceptr,
        ffn_gate: model_mod.EschaProj,
        ffn_up: model_mod.EschaProj,
        ffn_down: model_mod.EschaProj,
        m: u32,
    ) !void {
        var p_res = self.d_residual;
        var p_del = self.d_delta;
        var p_db = attn_out_bias;
        var p_nw = post_norm;
        var p_r0 = ffn_gate.rin;
        var p_r1 = ffn_up.rin;
        var p_x0 = self.d_xh0;
        var p_x1 = self.d_xh1;
        var p_null: cuda.CUdeviceptr = 0;
        var p_dim: c_int = Spec.d_model;
        var p_eps: f32 = Spec.rms_eps;
        var p_hd: c_int = 1;
        var p_n2 = [_]?*anyopaque{
            @ptrCast(&p_res), @ptrCast(&p_del), @ptrCast(&p_db),  @ptrCast(&p_nw),
            @ptrCast(&p_r0),  @ptrCast(&p_r1),  @ptrCast(&p_x0),  @ptrCast(&p_x1),
            @ptrCast(&p_null), @ptrCast(&p_dim), @ptrCast(&p_eps), @ptrCast(&p_hd),
        };
        try cuda.launch(self.ks.fn_norm_2x, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_n2);

        try self.launchEscha(ffn_gate, self.d_xh0, self.d_buf_a, m);
        try self.launchEscha(ffn_up, self.d_xh1, self.d_buf_b, m);

        var p_g = self.d_buf_a;
        var p_u = self.d_buf_b;
        var p_gb = ffn_gate.bias;
        var p_ub = ffn_up.bias;
        var p_drin = ffn_down.rin;
        var p_dxh = self.d_xh0;
        var p_dff: c_int = Spec.d_ff;
        var p_m: c_int = @intCast(m);
        var p_sw = [_]?*anyopaque{
            @ptrCast(&p_g),  @ptrCast(&p_u),    @ptrCast(&p_gb),  @ptrCast(&p_ub),
            @ptrCast(&p_drin), @ptrCast(&p_dxh), @ptrCast(&p_dff), @ptrCast(&p_m),
        };
        const gx: u32 = (m * (Spec.d_ff / 128) + 7) / 8;
        try cuda.launch(self.ks.fn_swiglu, .{ gx, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_sw);

        try self.launchEscha(ffn_down, self.d_xh0, self.d_delta, m);
    }

    fn launchCapture(self: *const Engine, m: u32, slot: c_int) !void {
        var p_res = self.d_residual;
        var p_cap = self.d_cap_buf;
        var p_slot = slot;
        var p_dim: c_int = Spec.d_model;
        var params = [_]?*anyopaque{ @ptrCast(&p_res), @ptrCast(&p_cap), @ptrCast(&p_slot), @ptrCast(&p_dim) };
        try cuda.launch(self.ks.fn_capture, .{ m, 20, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &params);
    }

    pub fn launchCommitBody(self: *const Engine, n_commit: u32) !void {
        return self.launchCommitBodyOnStream(n_commit, self.ctx.stream);
    }

    pub fn launchCommitBodyOnStream(self: *const Engine, n_commit: u32, stream: cuda.CUstream) !void {
        var p_nc: c_int = @intCast(n_commit);
        var p_null: cuda.CUdeviceptr = 0;

        for (0..Spec.n_layers) |l| {
            switch (self.model.layers[l]) {
                .gdn => |gl| {
                    const g_idx: usize = gl.gdn_idx;
                    const d_qkv = self.d_gdn_save_qkv + g_idx * MAX_M * GDN_QKV_DIM * 4;
                    const d_conv = self.d_gdn_conv + g_idx * 3 * GDN_QKV_DIM * 4;
                    var p_qkv = d_qkv;
                    var p_qb = gl.attn_qkv.bias;
                    var p_cs = d_conv;
                    var p_cc = [_]?*anyopaque{ @ptrCast(&p_qkv), @ptrCast(&p_qb), @ptrCast(&p_cs), @ptrCast(&p_nc) };
                    try cuda.launch(self.ks.fn_gdn_commit_conv, .{ 80, 1, 1 }, .{ 128, 1, 1 }, 0, stream, &p_cc);

                    var p_k = self.d_gdn_save_k + g_idx * MAX_M * 2048 * 4;
                    var p_v = self.d_gdn_save_v + g_idx * MAX_M * 6144 * 4;
                    var p_g = self.d_gdn_save_g + g_idx * MAX_M * 48 * 4;
                    var p_b = self.d_gdn_save_beta + g_idx * MAX_M * 48 * 4;
                    var p_st = self.d_gdn_state + g_idx * 48 * 128 * 128 * 4;
                    var p_rec = [_]?*anyopaque{
                        @ptrCast(&p_null), @ptrCast(&p_k),  @ptrCast(&p_v),
                        @ptrCast(&p_g),    @ptrCast(&p_b),  @ptrCast(&p_st),
                        @ptrCast(&p_st),   @ptrCast(&p_null), @ptrCast(&p_nc),
                        @ptrCast(&p_nc),
                    };
                    try cuda.launch(self.ks.fn_gdn_rec, .{ 48, 8, 1 }, .{ 32, 4, 1 }, 0, stream, &p_rec);
                },
                .full_attn => {},
            }
        }
    }

    /// Commit exact `n_commit` (`1..8`) tokens into all 48 GDN states after speculative verification!
    pub fn commitSpeculative(self: *Engine, n_commit: u32) !void {
        std.debug.assert(n_commit >= 1 and n_commit <= MAX_M);
        const idx: usize = n_commit - 1;
        if (self.g_commit[idx]) |exec| {
            try exec.launch(self.ctx.stream);
        } else {
            try self.ctx.beginCapture();
            try self.launchCommitBody(n_commit);
            const exec = try self.ctx.endCaptureExec();
            self.g_commit[idx] = exec;
            try exec.launch(self.ctx.stream);
        }
        self.pos += n_commit;
    }

    /// Compact Branch-B accepted slots `5 .. 5 + n_acc_b - 1` into contiguous slots `1 .. n_acc_b`
    /// across `d_cap_buf`, all 16 FullAttn Q4_0 KV caches, and all 48 GDN saved state buffers (~25 us).
    /// After this call, standard `commitSpeculative` / `commitAndInject` / `commitInjectAndDraftSampled`
    /// works identically for `n_commit = 1 + n_acc_b`!
    pub fn compactBranchB(self: *const Engine, n_acc_b: u32) !void {
        std.debug.assert(n_acc_b >= 1 and n_acc_b <= 3);

        // 1. Compact DFlash2 captured hidden states: slots 5..5+n_acc_b -> 1..1+n_acc_b (non-overlapping!)
        const cap_row_bytes: usize = 5 * Spec.d_model * 4;
        try self.ctx.copyDtoDAsync(
            self.d_cap_buf + 1 * cap_row_bytes,
            self.d_cap_buf + 5 * cap_row_bytes,
            @as(usize, n_acc_b) * cap_row_bytes,
        );

        // 2. Compact 16-layer FullAttn Q4_0 K/V cache: pos_base + 5..5+n_acc_b -> pos_base + 1..1+n_acc_b
        var p_kk = self.d_kv_k;
        var p_kv = self.d_kv_v;
        var p_lbytes: u64 = @intCast(self.kv_layer_bytes);
        var p_pos: c_int = @intCast(self.pos);
        var p_kv_args = [_]?*anyopaque{ @ptrCast(&p_kk), @ptrCast(&p_kv), @ptrCast(&p_lbytes), @ptrCast(&p_pos) };
        try cuda.launch(self.ks.fn_attn_compact_b, .{ 16, n_acc_b, 2 }, .{ 144, 1, 1 }, 0, self.ctx.stream, &p_kv_args);

        // 3. Compact 48-layer GDN speculative save buffers: slots 5..5+n_acc_b -> 1..1+n_acc_b
        var p_qkv = self.d_gdn_save_qkv;
        var p_k = self.d_gdn_save_k;
        var p_v = self.d_gdn_save_v;
        var p_g = self.d_gdn_save_g;
        var p_b = self.d_gdn_save_beta;
        var p_gdn_args = [_]?*anyopaque{ @ptrCast(&p_qkv), @ptrCast(&p_k), @ptrCast(&p_v), @ptrCast(&p_g), @ptrCast(&p_b) };
        try cuda.launch(self.ks.fn_gdn_compact_b, .{ 48, n_acc_b, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_gdn_args);
    }

    /// Launch Stage-1 Q4_0 LM head + Exact Top-64 on GPU (capturable inside a CUDA Graph!).
    pub fn launchStage1LmHeadGpu(self: *const Engine, d_hidden: cuda.CUdeviceptr, m: u32) !void {
        var p_x = d_hidden;
        var p_xq81 = self.d_xq81;
        var p_k: c_int = Spec.d_model;
        var p_m: c_int = @intCast(m);
        var p_nv: c_int = Spec.n_vocab;

        // 1. Quantize d_hidden [m, 5120] -> Q8_1
        var p_q81 = [_]?*anyopaque{ @ptrCast(&p_x), @ptrCast(&p_xq81), @ptrCast(&p_k), @ptrCast(&p_m) };
        const nb = Spec.d_model / 32;
        try cuda.launch(self.ks.fn_q81, .{ (m * nb + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_q81);

        // 2. Stage-1 Q4_0 x Q8_1 __dp4a GEMV over full 248,320 vocab (LDG.128 shmem fast path for M=1,7,8)
        var p_w = self.model.head_q4_0;
        var p_y = self.d_logits;
        if (m == 1 or m == 7 or m == 8) {
            const fn_fast = if (m == 1) self.ks.fn_q4_gemv_m1 else if (m == 7) self.ks.fn_q4_gemv_m7 else self.ks.fn_q4_gemv_m8;
            var p_fast = [_]?*anyopaque{ @ptrCast(&p_w), @ptrCast(&p_xq81), @ptrCast(&p_y), @ptrCast(&p_k), @ptrCast(&p_nv) };
            try cuda.launch(fn_fast, .{ (Spec.n_vocab + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_fast);
        } else {
            var p_gemv = [_]?*anyopaque{
                @ptrCast(&p_w), @ptrCast(&p_xq81), @ptrCast(&p_y),
                @ptrCast(&p_m), @ptrCast(&p_k),    @ptrCast(&p_nv),
            };
            try cuda.launch(self.ks.fn_q4_gemv, .{ (Spec.n_vocab + 3) / 4, 1, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_gemv);
        }

        // 3. Exact GPU Parallel-Bitonic Top-64
        var p_pv = self.d_topk_pval;
        var p_pi = self.d_topk_pidx;
        var p_ov = self.d_topk_oval;
        var p_oi = self.d_topk_oidx;
        var p_tk1 = [_]?*anyopaque{ @ptrCast(&p_y), @ptrCast(&p_pv), @ptrCast(&p_pi), @ptrCast(&p_nv) };
        var p_tk2 = [_]?*anyopaque{ @ptrCast(&p_pv), @ptrCast(&p_pi), @ptrCast(&p_ov), @ptrCast(&p_oi) };
        try cuda.launch(self.ks.fn_topk1, .{ TOPK_BLOCKS, m, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_tk1);
        try cuda.launch(self.ks.fn_topk2, .{ 1, m, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_tk2);
    }

    /// D2H Top-64 IDs + hidden states, sync, and run exact Stage-2 Q8_0 rescore on CPU (zero VRAM!).
    pub fn finishStage2LmHeadCpu(self: *Engine, d_hidden: cuda.CUdeviceptr, m: u32, out: *StepOutput) !void {
        // 4. Tiny D2H: cand_ids[m, 64] (<= 2 KiB) + hidden[m, 5120] (<= 160 KiB)
        try self.ctx.copyDtoHAsync(std.mem.sliceAsBytes(self.h_cand_ids[0 .. m * TOPK_K]), self.d_topk_oidx);
        try self.ctx.copyDtoHAsync(std.mem.sliceAsBytes(self.h_out_norm[0 .. m * Spec.d_model]), d_hidden);
        try self.ctx.sync();

        // 5. Stage-2 Exact Q8_0 rescore of Top-64 candidates directly from host mmap (ZERO VRAM, ~20 us/tok!)
        for (0..m) |r| {
            const x_row = self.h_out_norm[r * Spec.d_model ..][0..Spec.d_model];
            var best_id: u32 = 0;
            var best_val: f32 = -3.4028234e38;
            for (0..TOPK_K) |k| {
                const cid = self.h_cand_ids[r * TOPK_K + k];
                out.cands[r].ids[k] = cid;
                if (cid >= 0) {
                    const score = self.model.rescoreQ8_0Row(@intCast(cid), x_row);
                    out.cands[r].logits[k] = score;
                    if (score > best_val) {
                        best_val = score;
                        best_id = @intCast(cid);
                    }
                } else {
                    out.cands[r].logits[k] = -3.4028234e38;
                }
            }
            out.tokens[r] = best_id;
            out.top_logits[r] = best_val;
        }
    }

    /// Run Stage-1 Q4_0 LM head + Top-64 on GPU, then exact Q8_0 Stage-2 rescore on CPU (zero VRAM!)
    pub fn runTwoStageLmHead(self: *Engine, m: u32, out: *StepOutput) !void {
        return self.runTwoStageLmHeadFrom(self.d_xnorm, m, out);
    }

    pub fn runTwoStageLmHeadFrom(self: *Engine, d_hidden: cuda.CUdeviceptr, m: u32, out: *StepOutput) !void {
        try self.launchStage1LmHeadGpu(d_hidden, m);
        try self.finishStage2LmHeadCpu(d_hidden, m, out);
    }

    /// Fast batched prefill of `tokens` starting from current `self.pos` in chunks of PREFILL_M (64).
    /// When `need_logits == true`, returns the final token's prediction at `out.tokens[0]` / `out.cands[0]`.
    pub fn prefillRange(self: *Engine, tokens: []const u32, need_logits: bool) !StepOutput {
        std.debug.assert(tokens.len > 0);
        var offset: usize = 0;
        while (offset < tokens.len) {
            const chunk = @min(PREFILL_M, tokens.len - offset);
            const is_last = (offset + chunk == tokens.len);
            const res = try self.forward(tokens[offset .. offset + chunk], true, need_logits and is_last);
            if (is_last) return res;
            offset += chunk;
        }
        unreachable;
    }

    /// Fast prefill of arbitrary prompt length in chunks of PREFILL_M (64).
    /// Returns the StepOutput with the final token's prediction at index 0 (`tokens[0]` / `cands[0]`).
    pub fn prefill(self: *Engine, prompt_tokens: []const u32) !StepOutput {
        return self.prefillRange(prompt_tokens, true);
    }
};

fn bump(off: *usize, bytes: usize) usize {
    const aligned = (off.* + 255) & ~@as(usize, 255);
    off.* = aligned + bytes;
    return aligned;
}

pub fn isGreedySampling(temp: f32, top_k: u32, top_p: f32) bool {
    return !std.math.isFinite(temp) or temp <= 0.0 or top_k == 1 or (!std.math.isNan(top_p) and top_p <= 0.0);
}

pub fn isGreedySamplingEx(temp: f32, top_k: u32, top_p: f32, min_p: f32) bool {
    return isGreedySampling(temp, top_k, top_p) or (!std.math.isNan(min_p) and min_p >= 1.0);
}

/// Normalized discrete probability distribution over up to `TOPK_K` (64) candidate tokens.
pub const CandidateDist = struct {
    ids: [TOPK_K]u32 = @splat(0),
    probs: [TOPK_K]f32 = @splat(0),
    len: usize = 0,

    pub fn pointMass(id: u32) CandidateDist {
        var d: CandidateDist = .{};
        d.ids[0] = id;
        d.probs[0] = 1.0;
        d.len = 1;
        return d;
    }

    pub fn probOf(self: *const CandidateDist, id: u32) f32 {
        for (0..self.len) |i| {
            if (self.ids[i] == id) return self.probs[i];
        }
        return 0.0;
    }

    pub fn sample(self: *const CandidateDist, rng: *std.Random.DefaultPrng) u32 {
        if (self.len == 0) return 0;
        if (self.len == 1) return self.ids[0];
        const r = rng.random().float(f32);
        var acc: f32 = 0.0;
        for (0..self.len) |i| {
            acc += self.probs[i];
            if (r < acc) return self.ids[i];
        }
        return self.ids[self.len - 1];
    }

    /// Return the conditional distribution `P(X = x | X != excluded_id)` (renormalized).
    /// Returns an empty distribution (`len == 0`) if all probability mass was on `excluded_id`.
    pub fn conditionExcluding(self: *const CandidateDist, excluded_id: u32) CandidateDist {
        var out: CandidateDist = .{};
        var sum: f32 = 0.0;
        for (0..self.len) |i| {
            if (self.ids[i] != excluded_id and self.probs[i] > 0.0) {
                out.ids[out.len] = self.ids[i];
                out.probs[out.len] = self.probs[i];
                sum += self.probs[i];
                out.len += 1;
            }
        }
        if (out.len == 0 or !(sum > 0.0) or !std.math.isFinite(sum)) return .{};
        const inv = 1.0 / sum;
        for (0..out.len) |i| out.probs[i] *= inv;
        return out;
    }
};

/// Apply OpenAI-compatible `frequency_penalty` and `presence_penalty` to a `CandidateList`
/// based on token occurrence counts in the current completion (`logit -= freq*count + pres`).
pub fn applyTokenPenalties(
    cands: *const CandidateList,
    counts: *const std.AutoHashMapUnmanaged(u32, u32),
    freq_penalty: f32,
    pres_penalty: f32,
) CandidateList {
    var out = cands.*;
    const has_freq = std.math.isFinite(freq_penalty) and freq_penalty != 0.0;
    const has_pres = std.math.isFinite(pres_penalty) and pres_penalty != 0.0;
    if ((!has_freq and !has_pres) or counts.count() == 0) return out;
    for (0..TOPK_K) |i| {
        const cid = out.ids[i];
        if (cid < 0 or !std.math.isFinite(out.logits[i])) continue;
        if (counts.get(@intCast(cid))) |cnt| {
            if (cnt > 0) {
                var pen: f32 = 0.0;
                if (has_freq) pen += freq_penalty * @as(f32, @floatFromInt(cnt));
                if (has_pres) pen += pres_penalty;
                out.logits[i] -= pen;
            }
        }
    }
    return out;
}

/// Build the normalized post-(temperature, top_k, min_p, top_p) probability distribution from arbitrary
/// `(ids, logits)` slices (up to `TOPK_K` valid entries).
pub fn buildDistFromLogitsEx(
    ids_in: []const i32,
    logits_in: []const f32,
    temp: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
) CandidateDist {
    const Entry = struct { id: u32, val: f32 };
    var entries: [TOPK_K]Entry = undefined;
    var n: usize = 0;
    const count = @min(@min(ids_in.len, logits_in.len), TOPK_K);
    for (0..count) |i| {
        if (ids_in[i] >= 0 and !std.math.isNan(logits_in[i])) {
            entries[n] = .{ .id = @intCast(ids_in[i]), .val = logits_in[i] };
            n += 1;
        }
    }
    if (n == 0) return .{};

    // Greedy fast path (also handles non-finite temp, top_k == 1, non-positive top_p, or min_p >= 1)
    if (isGreedySampling(temp, top_k, top_p) or (!std.math.isNan(min_p) and min_p >= 1.0)) {
        var best = entries[0];
        for (entries[1..n]) |e| {
            if (e.val > best.val) best = e;
        }
        return CandidateDist.pointMass(best.id);
    }

    std.mem.sort(Entry, entries[0..n], {}, struct {
        fn gt(_: void, a: Entry, b: Entry) bool {
            return a.val > b.val or (a.val == b.val and a.id < b.id);
        }
    }.gt);

    const max_v = entries[0].val;
    if (!std.math.isFinite(max_v)) return CandidateDist.pointMass(entries[0].id);

    const k_lim = if (top_k > 0 and top_k < n) @as(usize, top_k) else n;
    const inv_t: f32 = 1.0 / temp;
    var probs: [TOPK_K]f32 = undefined;
    var sum: f32 = 0.0;
    for (0..k_lim) |i| {
        const diff = entries[i].val - max_v;
        const p = if (std.math.isFinite(diff)) @exp(diff * inv_t) else 0.0;
        probs[i] = p;
        sum += p;
    }
    if (!(sum > 0.0) or !std.math.isFinite(sum)) return CandidateDist.pointMass(entries[0].id);

    // Optional min_p filter (relative to top-1 probability probs[0] = 1.0 before dividing by sum)
    var eff_k: usize = k_lim;
    if (!std.math.isNan(min_p) and min_p > 0.0) {
        const thresh = probs[0] * min_p;
        var kept: usize = 1;
        var minp_sum: f32 = probs[0];
        for (1..k_lim) |i| {
            if (probs[i] >= thresh) {
                kept = i + 1;
                minp_sum += probs[i];
            } else {
                break; // sorted descending
            }
        }
        eff_k = kept;
        sum = minp_sum;
    }

    const eff_top_p: f32 = if (std.math.isNan(top_p) or top_p > 1.0) 1.0 else top_p;
    const inv_sum = 1.0 / sum;
    var cum: f32 = 0.0;
    var cut: usize = eff_k;
    for (0..eff_k) |i| {
        probs[i] *= inv_sum;
        cum += probs[i];
        if (cum >= eff_top_p) {
            cut = i + 1;
            break;
        }
    }

    var out: CandidateDist = .{};
    const inv_cum = if (cum > 0.0 and std.math.isFinite(cum)) 1.0 / cum else 1.0;
    for (0..cut) |i| {
        out.ids[i] = entries[i].id;
        out.probs[i] = probs[i] * inv_cum;
    }
    out.len = cut;
    return out;
}

pub fn buildDistFromLogits(
    ids_in: []const i32,
    logits_in: []const f32,
    temp: f32,
    top_k: u32,
    top_p: f32,
) CandidateDist {
    return buildDistFromLogitsEx(ids_in, logits_in, temp, top_k, top_p, 0.0);
}

pub fn buildCandidateDistEx(cands: *const CandidateList, temp: f32, top_k: u32, top_p: f32, min_p: f32) CandidateDist {
    return buildDistFromLogitsEx(&cands.ids, &cands.logits, temp, top_k, top_p, min_p);
}

pub fn buildCandidateDist(cands: *const CandidateList, temp: f32, top_k: u32, top_p: f32) CandidateDist {
    return buildDistFromLogitsEx(&cands.ids, &cands.logits, temp, top_k, top_p, 0.0);
}

/// Top-k + min_p + top-p + temperature sampler on the exact Stage-2 Top-64 candidates!
pub fn sampleFromCandidatesEx(cands: *const CandidateList, temp: f32, top_k: u32, top_p: f32, min_p: f32, rng: *std.Random.DefaultPrng) u32 {
    const dist = buildCandidateDistEx(cands, temp, top_k, top_p, min_p);
    return dist.sample(rng);
}

pub fn sampleFromCandidates(cands: *const CandidateList, temp: f32, top_k: u32, top_p: f32, rng: *std.Random.DefaultPrng) u32 {
    const dist = buildCandidateDist(cands, temp, top_k, top_p);
    return dist.sample(rng);
}

pub const VerifyResult = struct {
    accepted: bool,
    token: u32,
};

/// Exact lossless speculative rejection sampling (Leviathan et al. 2023 / Chen et al. 2023):
///   Given target distribution `p` and drafter proposal distribution `q` that produced `draft_tok`:
///   1. Accept `draft_tok` with probability `min(1, p(draft_tok) / q(draft_tok))`.
///   2. On rejection, sample replacement token from the residual distribution
///      `p'(x) = max(0, p(x) - q(x)) / sum_y max(0, p(y) - q(y))`.
///   The resulting token's marginal distribution equals `p` identically for every valid `q`.
pub fn verifySpeculativeToken(
    target_dist: *const CandidateDist,
    draft_dist: *const CandidateDist,
    draft_tok: u32,
    rng: *std.Random.DefaultPrng,
) VerifyResult {
    if (target_dist.len == 0) return .{ .accepted = false, .token = 0 };

    const p_d = target_dist.probOf(draft_tok);
    const q_d = draft_dist.probOf(draft_tok);

    if (p_d > 0.0) {
        if (q_d <= 0.0 or p_d >= q_d) {
            return .{ .accepted = true, .token = draft_tok };
        }
        if (rng.random().float(f32) < p_d / q_d) {
            return .{ .accepted = true, .token = draft_tok };
        }
    }

    // Rejected: draw from residual p'(x) proportional to max(0, p(x) - q(x))
    var res_probs: [TOPK_K]f32 = undefined;
    var res_sum: f32 = 0.0;
    for (0..target_dist.len) |i| {
        const diff = target_dist.probs[i] - draft_dist.probOf(target_dist.ids[i]);
        const rp: f32 = if (diff > 0.0) diff else 0.0;
        res_probs[i] = rp;
        res_sum += rp;
    }

    if (res_sum > 0.0 and std.math.isFinite(res_sum)) {
        const u = rng.random().float(f32) * res_sum;
        var acc: f32 = 0.0;
        var last_pos: usize = 0;
        for (0..target_dist.len) |i| {
            if (res_probs[i] > 0.0) last_pos = i;
            acc += res_probs[i];
            if (u < acc) {
                return .{ .accepted = false, .token = target_dist.ids[i] };
            }
        }
        return .{ .accepted = false, .token = target_dist.ids[last_pos] };
    }

    return .{ .accepted = false, .token = target_dist.sample(rng) };
}

pub const ForkBranch = enum {
    branch_a,
    branch_b,
    rejected,
};

pub const ForkVerifyResult = struct {
    branch: ForkBranch,
    token: u32,
};

/// Mathematically exact lossless Top-2 fork verification (SpecInfer / Sequoia):
///   Given target distribution `p`, Branch-A draft `d_a ~ q_a`, and Branch-B draft `d_b ~ q_b`:
///   1. Try `d_a` against `p` with accept probability `min(1, p(d_a) / q_a(d_a))`.
///   2. If rejected, form the normalized residual target `p_1(x) proportional to max(0, p(x) - q_a(x))`
///      and run standard lossless `verifySpeculativeToken(&p_1, q_b, d_b, rng)`.
///   Marginal distribution of `result.token` equals `p` identically, even when `q_b = q_a(. | x != d_a)`!
pub fn verifySpeculativeFork(
    target_dist: *const CandidateDist,
    draft_a_dist: *const CandidateDist,
    draft_a_tok: u32,
    draft_b_dist: *const CandidateDist,
    draft_b_tok: u32,
    rng: *std.Random.DefaultPrng,
) ForkVerifyResult {
    if (target_dist.len == 0) return .{ .branch = .rejected, .token = 0 };

    // 1. Verify Branch A against target p
    const p_a = target_dist.probOf(draft_a_tok);
    const q_a = draft_a_dist.probOf(draft_a_tok);
    if (p_a > 0.0) {
        if (q_a <= 0.0 or p_a >= q_a) {
            return .{ .branch = .branch_a, .token = draft_a_tok };
        }
        if (rng.random().float(f32) < p_a / q_a) {
            return .{ .branch = .branch_a, .token = draft_a_tok };
        }
    }

    // 2. Rejected Branch A -> construct normalized residual target p_1(x) proportional to max(0, p(x) - q_a(x))
    var p1: CandidateDist = .{};
    var res_sum: f32 = 0.0;
    for (0..target_dist.len) |i| {
        const diff = target_dist.probs[i] - draft_a_dist.probOf(target_dist.ids[i]);
        if (diff > 0.0) {
            p1.ids[p1.len] = target_dist.ids[i];
            p1.probs[p1.len] = diff;
            res_sum += diff;
            p1.len += 1;
        }
    }
    if (p1.len == 0 or !(res_sum > 0.0) or !std.math.isFinite(res_sum)) {
        return .{ .branch = .rejected, .token = target_dist.sample(rng) };
    }
    const inv_sum = 1.0 / res_sum;
    for (0..p1.len) |i| p1.probs[i] *= inv_sum;

    // 3. Verify Branch B against residual target p_1
    const vr_b = verifySpeculativeToken(&p1, draft_b_dist, draft_b_tok, rng);
    return .{
        .branch = if (vr_b.accepted) .branch_b else .rejected,
        .token = vr_b.token,
    };
}

test "sampleFromCandidates greedy ignores invalid slots, NaN, and handles non-finite temp" {
    var cands: CandidateList = .{
        .logits = @splat(-std.math.inf(f32)),
        .ids = @splat(-1),
    };
    // Slot 0 invalid with huge logit, slot 1 NaN, slot 2 valid (1.5), slot 3 valid (4.2)
    cands.ids[0] = -1;
    cands.logits[0] = 999.0;
    cands.ids[1] = 10;
    cands.logits[1] = std.math.nan(f32);
    cands.ids[2] = 20;
    cands.logits[2] = 1.5;
    cands.ids[3] = 30;
    cands.logits[3] = 4.2;

    var rng = std.Random.DefaultPrng.init(42);
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidates(&cands, 0.0, 64, 0.95, &rng));
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidates(&cands, -1.0, 64, 0.95, &rng));
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidates(&cands, 0.8, 1, 0.95, &rng));
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidates(&cands, 0.8, 64, 0.0, &rng));
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidates(&cands, std.math.nan(f32), 64, 0.95, &rng));
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidates(&cands, std.math.inf(f32), 64, 0.95, &rng));

    // +Inf logit must fall back cleanly to that candidate instead of NaN softmax
    cands.ids[4] = 40;
    cands.logits[4] = std.math.inf(f32);
    try std.testing.expectEqual(@as(u32, 40), sampleFromCandidates(&cands, 0.7, 64, 0.95, &rng));

    // All invalid / NaN returns 0 safely
    var empty: CandidateList = .{
        .logits = @splat(std.math.nan(f32)),
        .ids = @splat(-1),
    };
    try std.testing.expectEqual(@as(u32, 0), sampleFromCandidates(&empty, 0.7, 64, 0.95, &rng));
}

test "sampleFromCandidates respects top_k, top_p, top_k=0, and seed determinism" {
    var cands: CandidateList = .{
        .logits = @splat(-std.math.inf(f32)),
        .ids = @splat(-1),
    };
    // Unsorted order on purpose
    cands.ids[0] = 100;
    cands.logits[0] = 1.0;
    cands.ids[1] = 200;
    cands.logits[1] = 10.0; // rank 1 (~0.731 at temp=1 with rank 2 = 9.0)
    cands.ids[2] = 300;
    cands.logits[2] = 9.0; // rank 2 (~0.269)
    cands.ids[3] = 400;
    cands.logits[3] = 8.9; // rank 3

    // top_k = 2 must ONLY draw from {200, 300}
    var rng1 = std.Random.DefaultPrng.init(12345);
    var rng2 = std.Random.DefaultPrng.init(12345);
    var seen_200 = false;
    var seen_300 = false;
    for (0..100) |_| {
        const a = sampleFromCandidates(&cands, 1.0, 2, 1.0, &rng1);
        const b = sampleFromCandidates(&cands, 1.0, 2, 1.0, &rng2);
        try std.testing.expectEqual(a, b); // deterministic under same seed
        try std.testing.expect(a == 200 or a == 300);
        if (a == 200) seen_200 = true;
        if (a == 300) seen_300 = true;
    }
    try std.testing.expect(seen_200 and seen_300);

    // top_k = 0 means no top-k cap (all valid candidates eligible); tight top_p = 0.5 forces top-1 (200)
    for (0..50) |_| {
        const tok = sampleFromCandidates(&cands, 1.0, 0, 0.5, &rng1);
        try std.testing.expectEqual(@as(u32, 200), tok);
    }

    // With high temperature and top_k=0, top_p=1.0, even rank-4 (100) is reachable
    var seen_400 = false;
    for (0..100) |_| {
        const tok = sampleFromCandidates(&cands, 2.0, 0, 1.0, &rng1);
        if (tok == 400) seen_400 = true;
    }
    try std.testing.expect(seen_400);
}

test "verifySpeculativeToken preserves exact target marginal distribution and optimal acceptance rate" {
    // Target p over {10, 20, 30, 40}: [0.40, 0.30, 0.20, 0.10]
    var target: CandidateDist = .{};
    target.ids[0] = 10;
    target.probs[0] = 0.40;
    target.ids[1] = 20;
    target.probs[1] = 0.30;
    target.ids[2] = 30;
    target.probs[2] = 0.20;
    target.ids[3] = 40;
    target.probs[3] = 0.10;
    target.len = 4;

    // 1. Identical proposal q == p must achieve 100% acceptance!
    var rng = std.Random.DefaultPrng.init(2026);
    for (0..2000) |_| {
        const d = target.sample(&rng);
        const vr = verifySpeculativeToken(&target, &target, d, &rng);
        try std.testing.expect(vr.accepted);
        try std.testing.expectEqual(d, vr.token);
    }

    // 2. Mismatched proposal q with partial overlap + out-of-support token 99:
    //    q over {10, 20, 30, 99}: [0.55, 0.15, 0.20, 0.10]
    //    Theoretical acceptance = sum_x min(p(x), q(x)) = 0.40 + 0.15 + 0.20 + 0.00 = 0.75
    var draft: CandidateDist = .{};
    draft.ids[0] = 10;
    draft.probs[0] = 0.55;
    draft.ids[1] = 20;
    draft.probs[1] = 0.15;
    draft.ids[2] = 30;
    draft.probs[2] = 0.20;
    draft.ids[3] = 99; // not in target!
    draft.probs[3] = 0.10;
    draft.len = 4;

    const trials: usize = 200_000;
    var counts: [4]usize = .{ 0, 0, 0, 0 };
    var accepted: usize = 0;
    for (0..trials) |_| {
        const d = draft.sample(&rng);
        const vr = verifySpeculativeToken(&target, &draft, d, &rng);
        if (vr.accepted) accepted += 1;
        switch (vr.token) {
            10 => counts[0] += 1,
            20 => counts[1] += 1,
            30 => counts[2] += 1,
            40 => counts[3] += 1,
            else => return error.UnexpectedOutOfSupportToken,
        }
    }

    const inv_n: f64 = 1.0 / @as(f64, @floatFromInt(trials));
    const acc_rate = @as(f64, @floatFromInt(accepted)) * inv_n;
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), acc_rate, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.40), @as(f64, @floatFromInt(counts[0])) * inv_n, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.30), @as(f64, @floatFromInt(counts[1])) * inv_n, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.20), @as(f64, @floatFromInt(counts[2])) * inv_n, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.10), @as(f64, @floatFromInt(counts[3])) * inv_n, 0.006);

    // 3. Edge cases: out-of-support draft always rejected; empty target returns 0 safely; point masses match greedy
    const vr_oos = verifySpeculativeToken(&target, &draft, 99, &rng);
    try std.testing.expect(!vr_oos.accepted);
    try std.testing.expect(vr_oos.token == 20 or vr_oos.token == 40); // only tokens with p(x) > q(x)

    const empty_dist: CandidateDist = .{};
    const vr_empty = verifySpeculativeToken(&empty_dist, &draft, 10, &rng);
    try std.testing.expect(!vr_empty.accepted);
    try std.testing.expectEqual(@as(u32, 0), vr_empty.token);

    const pm_a = CandidateDist.pointMass(77);
    const pm_b = CandidateDist.pointMass(88);
    try std.testing.expectEqual(VerifyResult{ .accepted = true, .token = 77 }, verifySpeculativeToken(&pm_a, &pm_a, 77, &rng));
    try std.testing.expectEqual(VerifyResult{ .accepted = false, .token = 77 }, verifySpeculativeToken(&pm_a, &pm_b, 88, &rng));
}

test "buildDistFromLogitsEx applies min_p truncation cleanly" {
    // Logits [10.0, 9.0, 7.0, 5.0] at temp=1.0 -> relative exp: [1.0, 0.3679, 0.0498, 0.0067]
    const ids = [_]i32{ 1, 2, 3, 4 };
    const logits = [_]f32{ 10.0, 9.0, 7.0, 5.0 };

    // min_p = 0.10 keeps only {1, 2} (0.3679 >= 0.10 > 0.0498)
    const d1 = buildDistFromLogitsEx(&ids, &logits, 1.0, 64, 1.0, 0.10);
    try std.testing.expectEqual(@as(usize, 2), d1.len);
    try std.testing.expectEqual(@as(u32, 1), d1.ids[0]);
    try std.testing.expectEqual(@as(u32, 2), d1.ids[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), d1.probs[0] + d1.probs[1], 1e-5);

    // min_p = 0.01 keeps {1, 2, 3} (0.0498 >= 0.01 > 0.0067)
    const d2 = buildDistFromLogitsEx(&ids, &logits, 1.0, 64, 1.0, 0.01);
    try std.testing.expectEqual(@as(usize, 3), d2.len);

    // min_p >= 1.0 collapses to greedy top-1
    const d3 = buildDistFromLogitsEx(&ids, &logits, 1.0, 64, 1.0, 1.0);
    try std.testing.expectEqual(@as(usize, 1), d3.len);
    try std.testing.expectEqual(@as(u32, 1), d3.ids[0]);
    try std.testing.expect(isGreedySamplingEx(1.0, 64, 1.0, 1.0));
}

test "applyTokenPenalties modifies logits by frequency and presence counts" {
    var cands: CandidateList = .{
        .logits = @splat(-std.math.inf(f32)),
        .ids = @splat(-1),
    };
    cands.ids[0] = 10;
    cands.logits[0] = 5.0;
    cands.ids[1] = 20;
    cands.logits[1] = 4.5;
    cands.ids[2] = 30;
    cands.logits[2] = 4.0;

    var counts: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer counts.deinit(std.testing.allocator);
    try counts.put(std.testing.allocator, 10, 3); // pen = 0.5*3 + 0.4 = 1.9 -> 3.1
    try counts.put(std.testing.allocator, 20, 1); // pen = 0.5*1 + 0.4 = 0.9 -> 3.6

    const pen = applyTokenPenalties(&cands, &counts, 0.5, 0.4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.1), pen.logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.6), pen.logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), pen.logits[2], 1e-5);

    // Greedy pick after penalties switches from 10 -> 30
    var rng = std.Random.DefaultPrng.init(1);
    try std.testing.expectEqual(@as(u32, 30), sampleFromCandidatesEx(&pen, 0.0, 64, 1.0, 0.0, &rng));
}

test "verifySpeculativeFork preserves exact target marginal distribution with without-replacement Top-2 drafts" {
    // Target p over {10, 20, 30, 40}: [0.35, 0.30, 0.25, 0.10]
    var target: CandidateDist = .{};
    target.ids[0] = 10;
    target.probs[0] = 0.35;
    target.ids[1] = 20;
    target.probs[1] = 0.30;
    target.ids[2] = 30;
    target.probs[2] = 0.25;
    target.ids[3] = 40;
    target.probs[3] = 0.10;
    target.len = 4;

    // Mismatched drafter q_a over {10, 20, 30, 99}: [0.55, 0.15, 0.20, 0.10]
    // Single-draft acceptance = 0.35 + 0.15 + 0.20 = 0.70.
    // With Branch B drawn from q_b = q_a(. | x != d_a), total acceptance strictly exceeds 0.70!
    var draft_a: CandidateDist = .{};
    draft_a.ids[0] = 10;
    draft_a.probs[0] = 0.55;
    draft_a.ids[1] = 20;
    draft_a.probs[1] = 0.15;
    draft_a.ids[2] = 30;
    draft_a.probs[2] = 0.20;
    draft_a.ids[3] = 99;
    draft_a.probs[3] = 0.10;
    draft_a.len = 4;

    var rng = std.Random.DefaultPrng.init(20261125);
    const trials: usize = 200_000;
    var counts: [4]usize = .{ 0, 0, 0, 0 };
    var acc_a: usize = 0;
    var acc_b: usize = 0;

    for (0..trials) |_| {
        const d_a = draft_a.sample(&rng);
        const draft_b = draft_a.conditionExcluding(d_a);
        const d_b = draft_b.sample(&rng);
        try std.testing.expect(d_a != d_b);

        const fr = verifySpeculativeFork(&target, &draft_a, d_a, &draft_b, d_b, &rng);
        switch (fr.branch) {
            .branch_a => acc_a += 1,
            .branch_b => acc_b += 1,
            .rejected => {},
        }
        switch (fr.token) {
            10 => counts[0] += 1,
            20 => counts[1] += 1,
            30 => counts[2] += 1,
            40 => counts[3] += 1,
            else => return error.UnexpectedOutOfSupportToken,
        }
    }

    const inv_n: f64 = 1.0 / @as(f64, @floatFromInt(trials));
    const rate_a = @as(f64, @floatFromInt(acc_a)) * inv_n;
    const rate_b = @as(f64, @floatFromInt(acc_b)) * inv_n;
    try std.testing.expectApproxEqAbs(@as(f64, 0.70), rate_a, 0.006);
    try std.testing.expect(rate_b > 0.12); // Branch B lifts pos-0 acceptance substantially!
    // Exact target marginal distribution preserved across all 4 tokens:
    try std.testing.expectApproxEqAbs(@as(f64, 0.35), @as(f64, @floatFromInt(counts[0])) * inv_n, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.30), @as(f64, @floatFromInt(counts[1])) * inv_n, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), @as(f64, @floatFromInt(counts[2])) * inv_n, 0.006);
    try std.testing.expectApproxEqAbs(@as(f64, 0.10), @as(f64, @floatFromInt(counts[3])) * inv_n, 0.006);

    // Greedy point-mass case: accepts Branch B when argmax == d_b != d_a
    const pm_t = CandidateDist.pointMass(20);
    const pm_a = CandidateDist.pointMass(10);
    const pm_b = CandidateDist.pointMass(20);
    const gr_b = verifySpeculativeFork(&pm_t, &pm_a, 10, &pm_b, 20, &rng);
    try std.testing.expectEqual(ForkBranch.branch_b, gr_b.branch);
    try std.testing.expectEqual(@as(u32, 20), gr_b.token);
}
