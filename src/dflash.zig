//! Native sm_89 DFlash2 5-layer Speculative Drafter for Kintsugi-27B.
//!
//! Architecture:
//!   - Encoder: fc [25600 -> 5120] Q4_0 + enc.output_norm -> projects 5 captured target layers
//!     [6, 20, 34, 48, 62] directly from target GPU VRAM (`d_cap_buf`, zero host round-trip!)
//!   - KV Inject: projects encoded context through each layer's wk/wv + k_norm + NeoX RoPE(128)
//!     into a 5-layer FP16 SWA-2048 ring buffer (40 MiB total VRAM, zero rollback needed!)
//!   - Decoder: 5 layers of [RMSNorm -> Dynamic Conv2(in) -> Non-Causal GQA(32:8) -> Conv2(out)
//!     -> Residual -> RMSNorm -> Dynamic Conv2(in) -> SwiGLU FFN -> Conv2(out) -> Residual]
//!   - Selector: shares Target 2-stage LM Head (682 MiB Q4_0 Stage-1 + zero-VRAM Q8_0 Stage-2)
//!     for Top-16 candidates + zero-VRAM host-mmap selector_predecessor/selector_successor walk
//!     (saves 68.2 MiB VRAM, executes in ~2 us on CPU, bit-exact to llama.cpp's lattice walk!).

const std = @import("std");
const config = @import("config.zig");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");
const engine_mod = @import("engine.zig");

const dflash_sm89_cubin align(8) = @embedFile("dflash_sm89.cubin");

pub const N_LAYERS: usize = 5;
pub const D_MODEL: usize = 5120;
pub const D_ENC_IN: usize = 5 * D_MODEL; // 25600
pub const D_FF: usize = 17408;
pub const N_Q_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const Q_DIM: usize = N_Q_HEADS * HEAD_DIM; // 4096
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 1024
pub const CONV_DYN_DIM: usize = 1280; // (5120 / 16) * 2 * 2
pub const BLOCK_SIZE: usize = 8;
pub const N_DRAFT: usize = BLOCK_SIZE - 1; // 7
pub const SWA_WINDOW: usize = 2048;
pub const KV_RING_BYTES: usize = N_LAYERS * 2 * SWA_WINDOW * N_KV_HEADS * HEAD_DIM * 2; // 40 MiB
pub const SELECTOR_RANK: usize = 256;
pub const SELECTOR_TOP_K: usize = 16;
pub const ROPE_FREQ_BASE: f32 = 10_000_000.0;
pub const RMS_EPS: f32 = 1e-6;
pub const Q4_0_BYTES: usize = 18;
pub const Q8_1_BYTES: usize = 36;
pub const DEFAULT_MASK_TOKEN_ID: u32 = 248070;

pub const DFlashKernels = struct {
    mod: cuda.Module,
    fn_rmsnorm: cuda.CUfunction,
    fn_add_rmsnorm: cuda.CUfunction,
    fn_conv2: cuda.CUfunction,
    fn_qk_norm_rope: cuda.CUfunction,
    fn_kv_inject: cuda.CUfunction,
    fn_attn: cuda.CUfunction,
    fn_swiglu_q81: cuda.CUfunction,

    pub fn load() !DFlashKernels {
        var mod = try cuda.Module.loadData(dflash_sm89_cubin);
        errdefer mod.deinit();
        return .{
            .mod = mod,
            .fn_rmsnorm = try mod.getFunction("dflash_rmsnorm_f32"),
            .fn_add_rmsnorm = try mod.getFunction("dflash_add_rmsnorm_f32"),
            .fn_conv2 = try mod.getFunction("dflash_conv2_sm89"),
            .fn_qk_norm_rope = try mod.getFunction("dflash_qk_norm_rope128_f32"),
            .fn_kv_inject = try mod.getFunction("dflash_kv_inject_f16_sm89"),
            .fn_attn = try mod.getFunction("dflash_noncausal_attn_f16_sm89"),
            .fn_swiglu_q81 = try mod.getFunction("dflash_swiglu_q8_1_sm89"),
        };
    }

    pub fn deinit(self: *DFlashKernels) void {
        self.mod.deinit();
    }
};

pub const DFlashLayer = struct {
    attn_norm: cuda.CUdeviceptr,
    attn_conv_base: cuda.CUdeviceptr,
    attn_conv_proj: cuda.CUdeviceptr,
    attn_q: cuda.CUdeviceptr,
    attn_k: cuda.CUdeviceptr,
    attn_v: cuda.CUdeviceptr,
    attn_q_norm: cuda.CUdeviceptr,
    attn_k_norm: cuda.CUdeviceptr,
    attn_output: cuda.CUdeviceptr,
    ffn_norm: cuda.CUdeviceptr,
    ffn_conv_base: cuda.CUdeviceptr,
    ffn_conv_proj: cuda.CUdeviceptr,
    ffn_gate: cuda.CUdeviceptr,
    ffn_up: cuda.CUdeviceptr,
    ffn_down: cuda.CUdeviceptr,
};

pub const DFlashEngine = struct {
    allocator: std.mem.Allocator,
    ctx: *const cuda.Context,
    gguf_file: gguf.GgufFile,
    ks: DFlashKernels,

    mask_token_id: u32,
    vram_bytes: usize,

    // Single GPU weight arena (~965 MiB)
    d_weights: cuda.CUdeviceptr,
    fc_weight: cuda.CUdeviceptr,
    enc_output_norm: cuda.CUdeviceptr,
    output_norm: cuda.CUdeviceptr,
    selector_hidden: cuda.CUdeviceptr,
    layers: [N_LAYERS]DFlashLayer,

    // Zero-VRAM host-mmap selector embeddings (saves 68.2 MiB VRAM!)
    sel_prev_q4_mmap: []const u8, // [n_vocab, 256/32 * 18]
    sel_next_q4_mmap: []const u8, // [n_vocab, 256/32 * 18]

    // 5-layer FP16 SWA-2048 ring buffer KV cache (40.0 MiB)
    d_kv_ring: cuda.CUdeviceptr,
    kv_layer_half_bytes: usize, // 2048 * 8 * 128 * 2 = 4 MiB

    // Static decode scratchpad (~2.5 MiB)
    d_scratch: cuda.CUdeviceptr,
    d_residual: cuda.CUdeviceptr, // [8, 5120] f32
    d_norm: cuda.CUdeviceptr,     // [8, 5120] f32
    d_conv_out: cuda.CUdeviceptr, // [8, 5120] f32
    d_dyn: cuda.CUdeviceptr,      // [8, 1280] f32
    d_q: cuda.CUdeviceptr,        // [8, 4096] f32
    d_k: cuda.CUdeviceptr,        // [8, 1024] f32
    d_v: cuda.CUdeviceptr,        // [8, 1024] f32
    d_attn_out: cuda.CUdeviceptr, // [8, 4096] f32
    d_delta: cuda.CUdeviceptr,    // [8, 5120] f32
    d_ffn_gate: cuda.CUdeviceptr, // [8, 17408] f32
    d_ffn_up: cuda.CUdeviceptr,   // [8, 17408] f32
    d_xq81: cuda.CUdeviceptr,     // [8, (25600/32) * 36] u8
    d_sel_gate: cuda.CUdeviceptr, // [8, 256] f32
    d_meta_pos: cuda.CUdeviceptr, // [1] i32 (device-side pos/n_past for static CUDA Graph!)

    // Host staging buffers
    h_emb_buf: []f32,    // [8, 5120]
    h_mask_embed: []f32, // [5120] pre-dequantized once!
    h_sel_gate: []f32,   // [7, 256]
    h_meta_pos: i32 = 0,
    h_meta_pos_pair: [2]i32 = .{ 0, 0 },

    // Cached static CUDA Graphs
    g_draft: ?cuda.GraphExec = null,
    g_inject: [BLOCK_SIZE]?cuda.GraphExec = @splat(null),
    g_commit_inject: [BLOCK_SIZE]?cuda.GraphExec = @splat(null),
    g_commit_inject_draft: [BLOCK_SIZE]?cuda.GraphExec = @splat(null),

    pub fn load(
        allocator: std.mem.Allocator,
        ctx: *const cuda.Context,
        gguf_path: []const u8,
        target: *const engine_mod.Engine,
    ) !DFlashEngine {
        var path_buf: [4096]u8 = undefined;
        if (gguf_path.len >= path_buf.len) return error.PathTooLong;
        @memcpy(path_buf[0..gguf_path.len], gguf_path);
        path_buf[gguf_path.len] = 0;
        var gf = try gguf.GgufFile.open(allocator, path_buf[0..gguf_path.len :0]);
        errdefer gf.deinit(allocator);

        var ks = try DFlashKernels.load();
        errdefer ks.deinit();

        const mask_id: u32 = DEFAULT_MASK_TOKEN_ID;

        const sel_prev_info = gf.find("selector_predecessor.weight") orelse return error.MissingTensor;
        const sel_next_info = gf.find("selector_successor.weight") orelse return error.MissingTensor;
        const sel_prev_mmap = gf.tensorBytes(sel_prev_info);
        const sel_next_mmap = gf.tensorBytes(sel_next_info);

        // Pass 1: compute total single-arena GPU weight bytes
        var arena_size: usize = 0;
        const plan = struct {
            fn add(file: *const gguf.GgufFile, name: []const u8, total: *usize) !usize {
                const info = file.find(name) orelse return error.MissingTensor;
                const off = (total.* + 255) & ~@as(usize, 255);
                total.* = off + @as(usize, @intCast(info.byteSize()));
                return off;
            }
            fn addLayer(file: *const gguf.GgufFile, l: usize, suffix: []const u8, total: *usize) !usize {
                var buf: [96]u8 = undefined;
                const name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ l, suffix }) catch unreachable;
                return add(file, name, total);
            }
        };

        const off_fc = try plan.add(&gf, "fc.weight", &arena_size);
        const off_enc_norm = try plan.add(&gf, "enc.output_norm.weight", &arena_size);
        const off_out_norm = try plan.add(&gf, "output_norm.weight", &arena_size);
        const off_sel_hid = try plan.add(&gf, "selector_hidden.weight", &arena_size);

        var l_offs: [N_LAYERS]struct {
            attn_norm: usize,
            attn_conv_base: usize,
            attn_conv_proj: usize,
            attn_q: usize,
            attn_k: usize,
            attn_v: usize,
            attn_q_norm: usize,
            attn_k_norm: usize,
            attn_output: usize,
            ffn_norm: usize,
            ffn_conv_base: usize,
            ffn_conv_proj: usize,
            ffn_gate: usize,
            ffn_up: usize,
            ffn_down: usize,
        } = undefined;

        for (0..N_LAYERS) |l| {
            l_offs[l] = .{
                .attn_norm = try plan.addLayer(&gf, l, "attn_norm.weight", &arena_size),
                .attn_conv_base = try plan.addLayer(&gf, l, "attn_conv_base", &arena_size),
                .attn_conv_proj = try plan.addLayer(&gf, l, "attn_conv_proj.weight", &arena_size),
                .attn_q = try plan.addLayer(&gf, l, "attn_q.weight", &arena_size),
                .attn_k = try plan.addLayer(&gf, l, "attn_k.weight", &arena_size),
                .attn_v = try plan.addLayer(&gf, l, "attn_v.weight", &arena_size),
                .attn_q_norm = try plan.addLayer(&gf, l, "attn_q_norm.weight", &arena_size),
                .attn_k_norm = try plan.addLayer(&gf, l, "attn_k_norm.weight", &arena_size),
                .attn_output = try plan.addLayer(&gf, l, "attn_output.weight", &arena_size),
                .ffn_norm = try plan.addLayer(&gf, l, "ffn_norm.weight", &arena_size),
                .ffn_conv_base = try plan.addLayer(&gf, l, "ffn_conv_base", &arena_size),
                .ffn_conv_proj = try plan.addLayer(&gf, l, "ffn_conv_proj.weight", &arena_size),
                .ffn_gate = try plan.addLayer(&gf, l, "ffn_gate.weight", &arena_size),
                .ffn_up = try plan.addLayer(&gf, l, "ffn_up.weight", &arena_size),
                .ffn_down = try plan.addLayer(&gf, l, "ffn_down.weight", &arena_size),
            };
        }

        // Pass 2: allocate single weight arena & upload async
        const d_weights = try cuda.Context.alloc(arena_size);
        errdefer cuda.Context.free(d_weights);

        const upload = struct {
            fn put(c: *const cuda.Context, file: *const gguf.GgufFile, base: cuda.CUdeviceptr, off: usize, name: []const u8) !cuda.CUdeviceptr {
                const info = file.find(name) orelse return error.MissingTensor;
                const dst = base + off;
                try c.copyHtoDAsync(dst, file.tensorBytes(info));
                return dst;
            }
            fn putLayer(c: *const cuda.Context, file: *const gguf.GgufFile, base: cuda.CUdeviceptr, off: usize, l: usize, suffix: []const u8) !cuda.CUdeviceptr {
                var buf: [96]u8 = undefined;
                const name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ l, suffix }) catch unreachable;
                return put(c, file, base, off, name);
            }
        };

        const fc_weight = try upload.put(ctx, &gf, d_weights, off_fc, "fc.weight");
        const enc_output_norm = try upload.put(ctx, &gf, d_weights, off_enc_norm, "enc.output_norm.weight");
        const output_norm = try upload.put(ctx, &gf, d_weights, off_out_norm, "output_norm.weight");
        const selector_hidden = try upload.put(ctx, &gf, d_weights, off_sel_hid, "selector_hidden.weight");

        var layers: [N_LAYERS]DFlashLayer = undefined;
        for (0..N_LAYERS) |l| {
            layers[l] = .{
                .attn_norm = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_norm, l, "attn_norm.weight"),
                .attn_conv_base = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_conv_base, l, "attn_conv_base"),
                .attn_conv_proj = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_conv_proj, l, "attn_conv_proj.weight"),
                .attn_q = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_q, l, "attn_q.weight"),
                .attn_k = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_k, l, "attn_k.weight"),
                .attn_v = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_v, l, "attn_v.weight"),
                .attn_q_norm = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_q_norm, l, "attn_q_norm.weight"),
                .attn_k_norm = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_k_norm, l, "attn_k_norm.weight"),
                .attn_output = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].attn_output, l, "attn_output.weight"),
                .ffn_norm = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].ffn_norm, l, "ffn_norm.weight"),
                .ffn_conv_base = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].ffn_conv_base, l, "ffn_conv_base"),
                .ffn_conv_proj = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].ffn_conv_proj, l, "ffn_conv_proj.weight"),
                .ffn_gate = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].ffn_gate, l, "ffn_gate.weight"),
                .ffn_up = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].ffn_up, l, "ffn_up.weight"),
                .ffn_down = try upload.putLayer(ctx, &gf, d_weights, l_offs[l].ffn_down, l, "ffn_down.weight"),
            };
        }

        // 3. Allocate 5-layer FP16 SWA-2048 ring buffer (40.0 MiB)
        const kv_layer_half_bytes: usize = SWA_WINDOW * N_KV_HEADS * HEAD_DIM * 2;
        const kv_total_bytes: usize = N_LAYERS * 2 * kv_layer_half_bytes;
        const d_kv_ring = try cuda.Context.alloc(kv_total_bytes);
        errdefer cuda.Context.free(d_kv_ring);

        // 4. Allocate static scratchpad (~2.5 MiB)
        var sc_off: usize = 0;
        const off_res = bump(&sc_off, BLOCK_SIZE * D_MODEL * 4);
        const off_norm = bump(&sc_off, BLOCK_SIZE * D_MODEL * 4);
        const off_conv = bump(&sc_off, BLOCK_SIZE * D_MODEL * 4);
        const off_dyn = bump(&sc_off, BLOCK_SIZE * CONV_DYN_DIM * 4);
        const off_q = bump(&sc_off, BLOCK_SIZE * Q_DIM * 4);
        const off_k = bump(&sc_off, BLOCK_SIZE * KV_DIM * 4);
        const off_v = bump(&sc_off, BLOCK_SIZE * KV_DIM * 4);
        const off_ao = bump(&sc_off, BLOCK_SIZE * Q_DIM * 4);
        const off_del = bump(&sc_off, BLOCK_SIZE * D_MODEL * 4);
        const off_fg = bump(&sc_off, BLOCK_SIZE * D_FF * 4);
        const off_fu = bump(&sc_off, BLOCK_SIZE * D_FF * 4);
        const off_xq81 = bump(&sc_off, BLOCK_SIZE * (D_ENC_IN / 32) * Q8_1_BYTES);
        const off_sg = bump(&sc_off, BLOCK_SIZE * SELECTOR_RANK * 4);
        const off_meta_pos = bump(&sc_off, 16);

        const d_scratch = try cuda.Context.alloc(sc_off);
        errdefer cuda.Context.free(d_scratch);

        // 5. Host buffers + pre-dequantize MASK embedding once!
        const h_emb_buf = try allocator.alloc(f32, BLOCK_SIZE * D_MODEL);
        errdefer allocator.free(h_emb_buf);
        const h_mask_embed = try allocator.alloc(f32, D_MODEL);
        errdefer allocator.free(h_mask_embed);
        const h_sel_gate = try allocator.alloc(f32, N_DRAFT * SELECTOR_RANK);
        errdefer allocator.free(h_sel_gate);

        target.model.embedToken(mask_id, h_mask_embed);
        for (1..BLOCK_SIZE) |i| {
            @memcpy(h_emb_buf[i * D_MODEL ..][0..D_MODEL], h_mask_embed);
        }

        try ctx.sync();

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .gguf_file = gf,
            .ks = ks,
            .mask_token_id = mask_id,
            .vram_bytes = arena_size + kv_total_bytes + sc_off,
            .d_weights = d_weights,
            .fc_weight = fc_weight,
            .enc_output_norm = enc_output_norm,
            .output_norm = output_norm,
            .selector_hidden = selector_hidden,
            .layers = layers,
            .sel_prev_q4_mmap = sel_prev_mmap,
            .sel_next_q4_mmap = sel_next_mmap,
            .d_kv_ring = d_kv_ring,
            .kv_layer_half_bytes = kv_layer_half_bytes,
            .d_scratch = d_scratch,
            .d_residual = d_scratch + off_res,
            .d_norm = d_scratch + off_norm,
            .d_conv_out = d_scratch + off_conv,
            .d_dyn = d_scratch + off_dyn,
            .d_q = d_scratch + off_q,
            .d_k = d_scratch + off_k,
            .d_v = d_scratch + off_v,
            .d_attn_out = d_scratch + off_ao,
            .d_delta = d_scratch + off_del,
            .d_ffn_gate = d_scratch + off_fg,
            .d_ffn_up = d_scratch + off_fu,
            .d_xq81 = d_scratch + off_xq81,
            .d_sel_gate = d_scratch + off_sg,
            .d_meta_pos = d_scratch + off_meta_pos,
            .h_emb_buf = h_emb_buf,
            .h_mask_embed = h_mask_embed,
            .h_sel_gate = h_sel_gate,
        };
    }

    pub fn deinit(self: *DFlashEngine) void {
        if (self.g_draft) |*exec| exec.deinit();
        for (&self.g_inject) |*g| {
            if (g.*) |*exec| exec.deinit();
        }
        for (&self.g_commit_inject) |*g| {
            if (g.*) |*exec| exec.deinit();
        }
        for (&self.g_commit_inject_draft) |*g| {
            if (g.*) |*exec| exec.deinit();
        }
        self.allocator.free(self.h_sel_gate);
        self.allocator.free(self.h_mask_embed);
        self.allocator.free(self.h_emb_buf);
        cuda.Context.free(self.d_scratch);
        cuda.Context.free(self.d_kv_ring);
        cuda.Context.free(self.d_weights);
        self.ks.deinit();
        self.gguf_file.deinit(self.allocator);
    }

    inline fn kvKPtr(self: *const DFlashEngine, layer: usize) cuda.CUdeviceptr {
        return self.d_kv_ring + (layer * 2) * self.kv_layer_half_bytes;
    }

    inline fn kvVPtr(self: *const DFlashEngine, layer: usize) cuda.CUdeviceptr {
        return self.d_kv_ring + (layer * 2 + 1) * self.kv_layer_half_bytes;
    }

    fn launchQuantQ8_1(self: *const DFlashEngine, target: *const engine_mod.Engine, src_f32: cuda.CUdeviceptr, k: u32, m: u32) !void {
        var p_x = src_f32;
        var p_dst = self.d_xq81;
        var p_k: c_int = @intCast(k);
        var p_m: c_int = @intCast(m);
        var params = [_]?*anyopaque{ @ptrCast(&p_x), @ptrCast(&p_dst), @ptrCast(&p_k), @ptrCast(&p_m) };
        const nb = k / 32;
        try cuda.launch(target.ks.fn_q81, .{ (m * nb + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &params);
    }

    fn launchQ4Gemv(self: *const DFlashEngine, target: *const engine_mod.Engine, w_q4: cuda.CUdeviceptr, dst_f32: cuda.CUdeviceptr, m: u32, k: u32, n: u32) !void {
        var p_w = w_q4;
        var p_x = self.d_xq81;
        var p_y = dst_f32;
        var p_m: c_int = @intCast(m);
        var p_k: c_int = @intCast(k);
        var p_n: c_int = @intCast(n);
        if ((k & 1023) == 0 and (m == 1 or m == 7 or m == 8)) {
            const fn_fast = if (m == 1) target.ks.fn_q4_gemv_m1 else if (m == 7) target.ks.fn_q4_gemv_m7 else target.ks.fn_q4_gemv_m8;
            var p_fast = [_]?*anyopaque{ @ptrCast(&p_w), @ptrCast(&p_x), @ptrCast(&p_y), @ptrCast(&p_k), @ptrCast(&p_n) };
            try cuda.launch(fn_fast, .{ (n + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_fast);
        } else {
            var params = [_]?*anyopaque{
                @ptrCast(&p_w), @ptrCast(&p_x), @ptrCast(&p_y),
                @ptrCast(&p_m), @ptrCast(&p_k), @ptrCast(&p_n),
            };
            try cuda.launch(target.ks.fn_q4_gemv, .{ (n + 3) / 4, 1, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &params);
        }
    }

    /// Snapshot the 5-layer FP16 SWA-2048 KV ring buffer to Host RAM (40 MiB).
    pub fn saveKvRingToHost(self: *const DFlashEngine, dst: []u8) !void {
        std.debug.assert(dst.len == KV_RING_BYTES);
        try self.ctx.copyDtoHAsync(dst, self.d_kv_ring);
        try self.ctx.sync();
    }

    /// Restore the 5-layer FP16 SWA-2048 KV ring buffer from Host RAM (40 MiB).
    pub fn loadKvRingFromHost(self: *DFlashEngine, src: []const u8) !void {
        std.debug.assert(src.len == KV_RING_BYTES);
        try self.ctx.copyHtoDAsync(self.d_kv_ring, src);
    }

    fn launchEncodeAndInjectBodyFrom(
        self: *const DFlashEngine,
        target: *const engine_mod.Engine,
        d_cap_src: cuda.CUdeviceptr,
        n_tokens: u32,
    ) !void {
        // 1. Encoder: fc(25600 -> 5120) + enc.output_norm (uses d_delta so d_residual stays intact for fused draft!)
        try self.launchQuantQ8_1(target, d_cap_src, D_ENC_IN, n_tokens);
        try self.launchQ4Gemv(target, self.fc_weight, self.d_delta, n_tokens, D_ENC_IN, D_MODEL);

        var p_x = self.d_delta;
        var p_w = self.enc_output_norm;
        var p_o = self.d_norm;
        var p_dim: c_int = D_MODEL;
        var p_eps: f32 = RMS_EPS;
        var p_rn = [_]?*anyopaque{ @ptrCast(&p_x), @ptrCast(&p_w), @ptrCast(&p_o), @ptrCast(&p_dim), @ptrCast(&p_eps) };
        try cuda.launch(self.ks.fn_rmsnorm, .{ n_tokens, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_rn);

        // 2. Quantize encoded context once, project wk/wv and inject K/V for all 5 layers
        try self.launchQuantQ8_1(target, self.d_norm, D_MODEL, n_tokens);

        var p_dpos = self.d_meta_pos;
        var p_fb: f32 = ROPE_FREQ_BASE;
        for (0..N_LAYERS) |l| {
            try self.launchQ4Gemv(target, self.layers[l].attn_k, self.d_k, n_tokens, D_MODEL, KV_DIM);
            try self.launchQ4Gemv(target, self.layers[l].attn_v, self.d_v, n_tokens, D_MODEL, KV_DIM);

            var p_ki = self.d_k;
            var p_vi = self.d_v;
            var p_kn = self.layers[l].attn_k_norm;
            var p_kr = self.kvKPtr(l);
            var p_vr = self.kvVPtr(l);
            var p_inj = [_]?*anyopaque{
                @ptrCast(&p_ki), @ptrCast(&p_vi), @ptrCast(&p_kn),
                @ptrCast(&p_kr), @ptrCast(&p_vr), @ptrCast(&p_dpos),
                @ptrCast(&p_eps), @ptrCast(&p_fb),
            };
            try cuda.launch(self.ks.fn_kv_inject, .{ n_tokens, N_KV_HEADS, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_inj);
        }
    }

    /// Encode the first `n_tokens` (`1..8`) captured target hidden states directly from
    /// `target.d_cap_buf` and inject their K/V into the 5-layer FP16 SWA-2048 ring buffer
    /// at positions `pos_base .. pos_base + n_tokens - 1`. Completely async on GPU!
    pub fn encodeAndInject(self: *DFlashEngine, target: *const engine_mod.Engine, n_tokens: u32, pos_base: u32) !void {
        std.debug.assert(n_tokens >= 1 and n_tokens <= BLOCK_SIZE);
        self.h_meta_pos = @intCast(pos_base);
        try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.asBytes(&self.h_meta_pos));

        const idx: usize = n_tokens - 1;
        if (self.g_inject[idx]) |exec| {
            try exec.launch(self.ctx.stream);
        } else {
            try self.ctx.beginCapture();
            try self.launchEncodeAndInjectBodyFrom(target, target.d_cap_buf, n_tokens);
            const exec = try self.ctx.endCaptureExec();
            self.g_inject[idx] = exec;
            try exec.launch(self.ctx.stream);
        }
    }

    /// Fused single-CUDA-Graph target GDN commit + DFlash2 KV encode & inject (`1..8` tokens).
    /// Eliminates inter-graph host/driver launch overhead and advances `target.pos += n_commit`.
    pub fn commitAndInject(self: *DFlashEngine, target: *engine_mod.Engine, n_commit: u32) !void {
        std.debug.assert(n_commit >= 1 and n_commit <= BLOCK_SIZE);
        self.h_meta_pos = @intCast(target.pos);
        try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.asBytes(&self.h_meta_pos));

        const idx: usize = n_commit - 1;
        if (self.g_commit_inject[idx]) |exec| {
            try exec.launch(self.ctx.stream);
        } else {
            try self.ctx.beginCapture();
            try self.ctx.forkAux();
            try target.launchCommitBodyOnStream(n_commit, self.ctx.aux_stream);
            try self.launchEncodeAndInjectBodyFrom(target, target.d_cap_buf, n_commit);
            try self.ctx.joinAux();
            const exec = try self.ctx.endCaptureExec();
            self.g_commit_inject[idx] = exec;
            try exec.launch(self.ctx.stream);
        }
        target.pos += n_commit;
    }

    /// Encode and inject an arbitrary prefill chunk (`1..PREFILL_M`) from `target.d_cap_buf`
    /// in slices of BLOCK_SIZE (8).
    pub fn encodeAndInjectChunk(self: *DFlashEngine, target: *const engine_mod.Engine, n_tokens: u32, pos_base: u32) !void {
        var sub: u32 = 0;
        while (sub < n_tokens) {
            const take: u32 = @min(@as(u32, BLOCK_SIZE), n_tokens - sub);
            if (sub == 0) {
                try self.encodeAndInject(target, take, pos_base);
            } else {
                self.h_meta_pos = @intCast(pos_base + sub);
                try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.asBytes(&self.h_meta_pos));
                const d_src = target.d_cap_buf + @as(usize, sub) * D_ENC_IN * 4;
                try self.launchEncodeAndInjectBodyFrom(target, d_src, take);
            }
            sub += take;
        }
    }

    /// Batched prefill of `tokens` (`PREFILL_M=64` chunks) starting from `target.pos`.
    /// Skips DFlash2 capture & KV injection for early tokens strictly outside the final
    /// `SWA_WINDOW` (2048) tokens of this segment (they would be overwritten in the ring anyway!).
    /// When `need_logits == true`, returns the final token's prediction at index 0.
    pub fn prefillRange(
        self: *DFlashEngine,
        target: *engine_mod.Engine,
        tokens: []const u32,
        need_logits: bool,
    ) !engine_mod.StepOutput {
        std.debug.assert(tokens.len > 0);
        errdefer target.capture_enabled = true;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const chunk: u32 = @intCast(@min(engine_mod.PREFILL_M, tokens.len - offset));
            const is_last = (offset + chunk == tokens.len);
            const in_swa_tail = (offset + chunk + SWA_WINDOW > tokens.len);
            target.capture_enabled = in_swa_tail;
            const pos_before = target.pos;
            const res = try target.forward(tokens[offset .. offset + chunk], true, need_logits and is_last);
            if (in_swa_tail) {
                try self.encodeAndInjectChunk(target, chunk, pos_before);
            }
            if (is_last) {
                target.capture_enabled = true;
                return res;
            }
            offset += chunk;
        }
        unreachable;
    }

    /// Prefill prompt on target engine with DFlash2 capture + KV injection for the SWA-2048 tail!
    /// Returns the StepOutput with the final token's prediction at index 0 (`tokens[0]` / `cands[0]`).
    pub fn prefill(self: *DFlashEngine, target: *engine_mod.Engine, prompt_tokens: []const u32) !engine_mod.StepOutput {
        return self.prefillRange(target, prompt_tokens, true);
    }

    fn launchDraftBodyWithPos(self: *const DFlashEngine, target: *const engine_mod.Engine, d_pos: cuda.CUdeviceptr) !void {
        const m: u32 = BLOCK_SIZE;
        var p_dim: c_int = D_MODEL;
        var p_eps: f32 = RMS_EPS;
        var p_fb: f32 = ROPE_FREQ_BASE;
        var p_dpos = d_pos;
        var p_m: c_int = @intCast(m);
        var p_qh: c_int = N_Q_HEADS;
        var p_kvh: c_int = N_KV_HEADS;
        var p_side0: c_int = 0;
        var p_side1: c_int = 1;
        var p_dff: c_int = D_FF;

        // 2. Run all 5 DFlash2 layers
        for (0..N_LAYERS) |l| {
            const ly = self.layers[l];

            // 2A. (Residual add from prev layer +) attn_norm
            if (l == 0) {
                var p_x = self.d_residual;
                var p_w = ly.attn_norm;
                var p_o = self.d_norm;
                var p_rn = [_]?*anyopaque{ @ptrCast(&p_x), @ptrCast(&p_w), @ptrCast(&p_o), @ptrCast(&p_dim), @ptrCast(&p_eps) };
                try cuda.launch(self.ks.fn_rmsnorm, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_rn);
            } else {
                var p_r = self.d_residual;
                var p_d = self.d_delta;
                var p_w = ly.attn_norm;
                var p_o = self.d_norm;
                var p_ar = [_]?*anyopaque{ @ptrCast(&p_r), @ptrCast(&p_d), @ptrCast(&p_w), @ptrCast(&p_o), @ptrCast(&p_dim), @ptrCast(&p_eps) };
                try cuda.launch(self.ks.fn_add_rmsnorm, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_ar);
            }

            // 2B. attn_conv_proj -> dynamic conv2 (side=0)
            try self.launchQuantQ8_1(target, self.d_norm, D_MODEL, m);
            try self.launchQ4Gemv(target, ly.attn_conv_proj, self.d_dyn, m, D_MODEL, CONV_DYN_DIM);
            {
                var p_h = self.d_norm;
                var p_dy = self.d_dyn;
                var p_b = ly.attn_conv_base;
                var p_o = self.d_conv_out;
                var p_c0 = [_]?*anyopaque{ @ptrCast(&p_h), @ptrCast(&p_dy), @ptrCast(&p_b), @ptrCast(&p_o), @ptrCast(&p_side0), @ptrCast(&p_dim) };
                try cuda.launch(self.ks.fn_conv2, .{ m, D_MODEL / 256, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_c0);
            }

            // 2C. Q, K, V projections + QK-norm + NeoX RoPE(128)
            try self.launchQuantQ8_1(target, self.d_conv_out, D_MODEL, m);
            try self.launchQ4Gemv(target, ly.attn_q, self.d_q, m, D_MODEL, Q_DIM);
            try self.launchQ4Gemv(target, ly.attn_k, self.d_k, m, D_MODEL, KV_DIM);
            try self.launchQ4Gemv(target, ly.attn_v, self.d_v, m, D_MODEL, KV_DIM);
            {
                var p_q = self.d_q;
                var p_qn = ly.attn_q_norm;
                var p_rq = [_]?*anyopaque{ @ptrCast(&p_q), @ptrCast(&p_qn), @ptrCast(&p_qh), @ptrCast(&p_dpos), @ptrCast(&p_eps), @ptrCast(&p_fb) };
                try cuda.launch(self.ks.fn_qk_norm_rope, .{ m, N_Q_HEADS, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_rq);

                var p_k = self.d_k;
                var p_kn = ly.attn_k_norm;
                var p_rk = [_]?*anyopaque{ @ptrCast(&p_k), @ptrCast(&p_kn), @ptrCast(&p_kvh), @ptrCast(&p_dpos), @ptrCast(&p_eps), @ptrCast(&p_fb) };
                try cuda.launch(self.ks.fn_qk_norm_rope, .{ m, N_KV_HEADS, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_rk);
            }

            // 2D. Non-causal GQA (32:8) SWA-2048 FlashAttention over prefix ring + noise block
            {
                var p_q = self.d_q;
                var p_k = self.d_k;
                var p_v = self.d_v;
                var p_kr = self.kvKPtr(l);
                var p_vr = self.kvVPtr(l);
                var p_ao = self.d_attn_out;
                var p_fa = [_]?*anyopaque{
                    @ptrCast(&p_q),    @ptrCast(&p_k),  @ptrCast(&p_v),
                    @ptrCast(&p_kr),   @ptrCast(&p_vr), @ptrCast(&p_ao),
                    @ptrCast(&p_dpos), @ptrCast(&p_m),
                };
                try cuda.launch(self.ks.fn_attn, .{ N_Q_HEADS, m, 1 }, .{ 128, 1, 1 }, 0, self.ctx.stream, &p_fa);
            }

            // 2E. attn_output -> dynamic conv2 (side=1) -> fused residual + ffn_norm
            try self.launchQuantQ8_1(target, self.d_attn_out, Q_DIM, m);
            try self.launchQ4Gemv(target, ly.attn_output, self.d_norm, m, Q_DIM, D_MODEL);
            {
                var p_h = self.d_norm;
                var p_dy = self.d_dyn;
                var p_b = ly.attn_conv_base;
                var p_o = self.d_delta;
                var p_c1 = [_]?*anyopaque{ @ptrCast(&p_h), @ptrCast(&p_dy), @ptrCast(&p_b), @ptrCast(&p_o), @ptrCast(&p_side1), @ptrCast(&p_dim) };
                try cuda.launch(self.ks.fn_conv2, .{ m, D_MODEL / 256, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_c1);
            }
            {
                var p_r = self.d_residual;
                var p_d = self.d_delta;
                var p_w = ly.ffn_norm;
                var p_o = self.d_norm;
                var p_ar = [_]?*anyopaque{ @ptrCast(&p_r), @ptrCast(&p_d), @ptrCast(&p_w), @ptrCast(&p_o), @ptrCast(&p_dim), @ptrCast(&p_eps) };
                try cuda.launch(self.ks.fn_add_rmsnorm, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_ar);
            }

            // 2F. ffn_conv_proj -> dynamic conv2 (side=0) -> gate/up -> fused SwiGLU->Q8_1 -> down -> conv2 (side=1)
            try self.launchQuantQ8_1(target, self.d_norm, D_MODEL, m);
            try self.launchQ4Gemv(target, ly.ffn_conv_proj, self.d_dyn, m, D_MODEL, CONV_DYN_DIM);
            {
                var p_h = self.d_norm;
                var p_dy = self.d_dyn;
                var p_b = ly.ffn_conv_base;
                var p_o = self.d_conv_out;
                var p_c0 = [_]?*anyopaque{ @ptrCast(&p_h), @ptrCast(&p_dy), @ptrCast(&p_b), @ptrCast(&p_o), @ptrCast(&p_side0), @ptrCast(&p_dim) };
                try cuda.launch(self.ks.fn_conv2, .{ m, D_MODEL / 256, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_c0);
            }

            try self.launchQuantQ8_1(target, self.d_conv_out, D_MODEL, m);
            try self.launchQ4Gemv(target, ly.ffn_gate, self.d_ffn_gate, m, D_MODEL, D_FF);
            try self.launchQ4Gemv(target, ly.ffn_up, self.d_ffn_up, m, D_MODEL, D_FF);
            {
                var p_g = self.d_ffn_gate;
                var p_u = self.d_ffn_up;
                var p_q = self.d_xq81;
                var p_sw = [_]?*anyopaque{ @ptrCast(&p_g), @ptrCast(&p_u), @ptrCast(&p_q), @ptrCast(&p_dff), @ptrCast(&p_m) };
                const gx: u32 = (m * (D_FF / 32) + 7) / 8;
                try cuda.launch(self.ks.fn_swiglu_q81, .{ gx, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_sw);
            }
            try self.launchQ4Gemv(target, ly.ffn_down, self.d_norm, m, D_FF, D_MODEL);
            {
                var p_h = self.d_norm;
                var p_dy = self.d_dyn;
                var p_b = ly.ffn_conv_base;
                var p_o = self.d_delta;
                var p_c1 = [_]?*anyopaque{ @ptrCast(&p_h), @ptrCast(&p_dy), @ptrCast(&p_b), @ptrCast(&p_o), @ptrCast(&p_side1), @ptrCast(&p_dim) };
                try cuda.launch(self.ks.fn_conv2, .{ m, D_MODEL / 256, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_c1);
            }
        }

        // 3. Final residual add + output_norm -> d_norm [8, 5120]
        {
            var p_r = self.d_residual;
            var p_d = self.d_delta;
            var p_w = self.output_norm;
            var p_o = self.d_norm;
            var p_ar = [_]?*anyopaque{ @ptrCast(&p_r), @ptrCast(&p_d), @ptrCast(&p_w), @ptrCast(&p_o), @ptrCast(&p_dim), @ptrCast(&p_eps) };
            try cuda.launch(self.ks.fn_add_rmsnorm, .{ m, 1, 1 }, .{ 256, 1, 1 }, 0, self.ctx.stream, &p_ar);
        }

        // 4. Selector hidden gate [7, 256] + Target Stage-1 GPU LM Head (all inside g_draft!)
        const d_draft_hidden = self.d_norm + D_MODEL * 4;
        try self.launchQuantQ8_1(target, d_draft_hidden, D_MODEL, N_DRAFT);
        try self.launchQ4Gemv(target, self.selector_hidden, self.d_sel_gate, N_DRAFT, D_MODEL, SELECTOR_RANK);
        try target.launchStage1LmHeadGpu(d_draft_hidden, N_DRAFT);
    }

    pub const DraftSampling = struct {
        temp: f32 = 0.0,
        top_k: u32 = 64,
        top_p: f32 = 1.0,
        min_p: f32 = 0.0,
        rng: *std.Random.DefaultPrng,
    };

    /// Run DFlash2 5-layer noise-block diffusion on `[id_last, MASK * 7]` at positions `n_past..n_past+7`,
    /// score through Target 2-stage LM Head, and run the exact DFlash2 selector walk on CPU (greedy)!
    /// Returns number of drafted tokens (`0..7`, always `7` when `p_min == 0.0`).
    pub fn draft(
        self: *DFlashEngine,
        target: *engine_mod.Engine,
        id_last: u32,
        n_past: u32,
        p_min: f32,
        draft_out: *[N_DRAFT]u32,
    ) !u32 {
        return self.draftSampled(target, id_last, n_past, p_min, null, draft_out, null);
    }

    /// Run DFlash2 drafting with optional stochastic sampling (`DraftSampling`).
    /// When `sampling` is non-greedy, each selector step samples `draft_out[i] ~ q_i` (applying the
    /// same temperature/top_k/top_p to the Top-16 selector scores) and writes `q_i` into `dists_out[i]`
    /// so the caller can perform mathematically exact lossless speculative rejection sampling.
    pub fn draftSampled(
        self: *DFlashEngine,
        target: *engine_mod.Engine,
        id_last: u32,
        n_past: u32,
        p_min: f32,
        sampling: ?DraftSampling,
        draft_out: *[N_DRAFT]u32,
        dists_out: ?*[N_DRAFT]engine_mod.CandidateDist,
    ) !u32 {
        // 1. Update anchor row 0 embedding (rows 1..7 are pre-populated with MASK!) + n_past async
        target.model.embedToken(id_last, self.h_emb_buf[0..D_MODEL]);
        try self.ctx.copyHtoDAsync(self.d_residual, std.mem.sliceAsBytes(self.h_emb_buf));
        self.h_meta_pos = @intCast(n_past);
        try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.asBytes(&self.h_meta_pos));

        // 2. Replay static CUDA Graph (5 layers + output_norm + selector_hidden + Target Stage-1 LM Head!)
        if (self.g_draft) |exec| {
            try exec.launch(self.ctx.stream);
        } else {
            try self.ctx.beginCapture();
            try self.launchDraftBodyWithPos(target, self.d_meta_pos);
            const exec = try self.ctx.endCaptureExec();
            self.g_draft = exec;
            try exec.launch(self.ctx.stream);
        }

        return self.finishDraftSelectorCpu(target, id_last, p_min, sampling, draft_out, dists_out);
    }

    /// Fused single-CUDA-Graph `target.commit(n_commit)` + `dflash.inject(n_commit)` + `dflash.draft(id_last)`!
    /// Eliminates 2 inter-graph CUDA driver launches per speculative decode step and advances `target.pos += n_commit`.
    pub fn commitInjectAndDraftSampled(
        self: *DFlashEngine,
        target: *engine_mod.Engine,
        n_commit: u32,
        id_last: u32,
        p_min: f32,
        sampling: ?DraftSampling,
        draft_out: *[N_DRAFT]u32,
        dists_out: ?*[N_DRAFT]engine_mod.CandidateDist,
    ) !u32 {
        std.debug.assert(n_commit >= 1 and n_commit <= BLOCK_SIZE);
        const pos_before = target.pos;
        const pos_after = pos_before + n_commit;

        // 1. Upload next draft anchor embedding + [pos_before, pos_after] pair async
        //    (Safe before inject because launchEncodeAndInjectBodyFrom uses d_delta, not d_residual!)
        target.model.embedToken(id_last, self.h_emb_buf[0..D_MODEL]);
        try self.ctx.copyHtoDAsync(self.d_residual, std.mem.sliceAsBytes(self.h_emb_buf));
        self.h_meta_pos_pair = .{ @intCast(pos_before), @intCast(pos_after) };
        try self.ctx.copyHtoDAsync(self.d_meta_pos, std.mem.sliceAsBytes(&self.h_meta_pos_pair));

        // 2. Launch single fused CUDA Graph: GDN Commit -> DFlash2 KV Inject -> DFlash2 Draft + Target Stage-1 LM Head!
        const idx: usize = n_commit - 1;
        if (self.g_commit_inject_draft[idx]) |exec| {
            try exec.launch(self.ctx.stream);
        } else {
            try self.ctx.beginCapture();
            try self.ctx.forkAux();
            try target.launchCommitBodyOnStream(n_commit, self.ctx.aux_stream);
            try self.launchEncodeAndInjectBodyFrom(target, target.d_cap_buf, n_commit);
            try self.launchDraftBodyWithPos(target, self.d_meta_pos + 4);
            try self.ctx.joinAux();
            const exec = try self.ctx.endCaptureExec();
            self.g_commit_inject_draft[idx] = exec;
            try exec.launch(self.ctx.stream);
        }
        target.pos = pos_after;

        return self.finishDraftSelectorCpu(target, id_last, p_min, sampling, draft_out, dists_out);
    }

    pub fn commitInjectAndDraft(
        self: *DFlashEngine,
        target: *engine_mod.Engine,
        n_commit: u32,
        id_last: u32,
        p_min: f32,
        draft_out: *[N_DRAFT]u32,
    ) !u32 {
        return self.commitInjectAndDraftSampled(target, n_commit, id_last, p_min, null, draft_out, null);
    }

    fn finishDraftSelectorCpu(
        self: *DFlashEngine,
        target: *engine_mod.Engine,
        id_last: u32,
        p_min: f32,
        sampling: ?DraftSampling,
        draft_out: *[N_DRAFT]u32,
        dists_out: ?*[N_DRAFT]engine_mod.CandidateDist,
    ) !u32 {
        // 3. Single D2H + sync for selector gate + Stage-2 exact Q8_0 rescore on CPU
        const d_draft_hidden = self.d_norm + D_MODEL * 4;
        try self.ctx.copyDtoHAsync(std.mem.sliceAsBytes(self.h_sel_gate), self.d_sel_gate);
        var lm_out: engine_mod.StepOutput = .{};
        try target.finishStage2LmHeadCpu(d_draft_hidden, N_DRAFT, &lm_out);

        // 5. Exact DFlash2 Selector autoregressive walk on CPU (~2 us, zero VRAM!)
        //    Matches llama.cpp build_dflash2_selector + common/speculative.cpp verbatim when greedy,
        //    and draws from normalized selector scores when stochastic sampling is enabled.
        const use_stochastic = if (sampling) |s| !engine_mod.isGreedySamplingEx(s.temp, s.top_k, s.top_p, s.min_p) else false;
        var pred_tok: u32 = id_last;
        var prev_vec: [SELECTOR_RANK]f32 = undefined;
        var cond_vec: [SELECTOR_RANK]f32 = undefined;
        var n_drafted: u32 = 0;

        for (0..N_DRAFT) |pos_i| {
            // Sort exact Stage-2 rescored Top-64 candidates descending and take Top-16
            var order: [engine_mod.TOPK_K]usize = undefined;
            for (0..engine_mod.TOPK_K) |k| order[k] = k;
            const cands = &lm_out.cands[pos_i];
            // Partial selection sort for top 16 (64*16 = 1024 comparisons = < 1 us!)
            for (0..SELECTOR_TOP_K) |k| {
                var best_j = k;
                var best_v = cands.logits[order[k]];
                var best_id = cands.ids[order[k]];
                for (k + 1..engine_mod.TOPK_K) |j| {
                    const vj = cands.logits[order[j]];
                    const idj = cands.ids[order[j]];
                    if (vj > best_v or (vj == best_v and idj >= 0 and (best_id < 0 or idj < best_id))) {
                        best_j = j;
                        best_v = vj;
                        best_id = idj;
                    }
                }
                const tmp = order[k];
                order[k] = order[best_j];
                order[best_j] = tmp;
            }

            // Dequantize predecessor embedding [256] and gate elementwise
            dequantQ4_0Row256(self.sel_prev_q4_mmap, pred_tok, &prev_vec);
            const gate_row = self.h_sel_gate[pos_i * SELECTOR_RANK ..][0..SELECTOR_RANK];
            for (0..SELECTOR_RANK) |r| {
                cond_vec[r] = prev_vec[r] * gate_row[r];
            }

            // Score Top-16 candidates: score[k] = unary[k] + dot(successor[cand_k], cond)
            var sel_ids: [SELECTOR_TOP_K]i32 = @splat(-1);
            var scores: [SELECTOR_TOP_K]f32 = undefined;
            var best_k: usize = 0;
            var best_score: f32 = -3.4028234e38;
            for (0..SELECTOR_TOP_K) |k| {
                const idx = order[k];
                const cid = cands.ids[idx];
                sel_ids[k] = cid;
                if (cid < 0) {
                    scores[k] = -3.4028234e38;
                    continue;
                }
                const pairwise = dotQ4_0Row256(self.sel_next_q4_mmap, @intCast(cid), &cond_vec);
                const sc = cands.logits[idx] + pairwise;
                scores[k] = sc;
                if (sc > best_score) {
                    best_score = sc;
                    best_k = k;
                }
            }

            if (p_min > 0.0) {
                var exp_sum: f32 = 0.0;
                for (0..SELECTOR_TOP_K) |k| {
                    exp_sum += @exp(scores[k] - best_score);
                }
                if (1.0 / exp_sum < p_min) break;
            }

            const chosen_id: u32 = if (use_stochastic) blk: {
                const s = sampling.?;
                const q_dist = engine_mod.buildDistFromLogitsEx(&sel_ids, &scores, s.temp, s.top_k, s.top_p, s.min_p);
                if (dists_out) |do| do[n_drafted] = q_dist;
                break :blk q_dist.sample(s.rng);
            } else blk: {
                const cid = sel_ids[best_k];
                const id: u32 = if (cid >= 0) @intCast(cid) else 0;
                if (dists_out) |do| do[n_drafted] = engine_mod.CandidateDist.pointMass(id);
                break :blk id;
            };
            draft_out[n_drafted] = chosen_id;
            n_drafted += 1;
            pred_tok = chosen_id;
        }

        return n_drafted;
    }
};

fn bump(off: *usize, bytes: usize) usize {
    const aligned = (off.* + 255) & ~@as(usize, 255);
    off.* = aligned + bytes;
    return aligned;
}

inline fn dequantQ4_0Row256(mmap: []const u8, row_id: u32, dst: *[SELECTOR_RANK]f32) void {
    const n_blk: usize = SELECTOR_RANK / 32; // 8
    const row_bytes: usize = n_blk * Q4_0_BYTES; // 144
    const base = @as(usize, row_id) * row_bytes;
    for (0..n_blk) |b| {
        const off = base + b * Q4_0_BYTES;
        const d_bits = std.mem.readInt(u16, mmap[off..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = mmap[off + 2 ..][0..16];
        const out_base = b * 32;
        for (0..16) |j| {
            const byte = qs[j];
            const lo: i32 = @as(i32, byte & 0x0f) - 8;
            const hi: i32 = @as(i32, byte >> 4) - 8;
            dst[out_base + j] = d * @as(f32, @floatFromInt(lo));
            dst[out_base + 16 + j] = d * @as(f32, @floatFromInt(hi));
        }
    }
}

inline fn dotQ4_0Row256(mmap: []const u8, row_id: u32, x: *const [SELECTOR_RANK]f32) f32 {
    const n_blk: usize = SELECTOR_RANK / 32; // 8
    const row_bytes: usize = n_blk * Q4_0_BYTES; // 144
    const base = @as(usize, row_id) * row_bytes;
    var acc: f32 = 0.0;
    for (0..n_blk) |b| {
        const off = base + b * Q4_0_BYTES;
        const d_bits = std.mem.readInt(u16, mmap[off..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = mmap[off + 2 ..][0..16];
        const x_base = b * 32;
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        for (0..16) |j| {
            const byte = qs[j];
            const lo: i32 = @as(i32, byte & 0x0f) - 8;
            const hi: i32 = @as(i32, byte >> 4) - 8;
            s0 += @as(f32, @floatFromInt(lo)) * x[x_base + j];
            s1 += @as(f32, @floatFromInt(hi)) * x[x_base + 16 + j];
        }
        acc += d * (s0 + s1);
    }
    return acc;
}

test "dflash Q4_0 row dequant and dot identity" {
    // 8 blocks of 18 bytes = 144 bytes, d = 1.0 (0x3c00), qs = 0x97 (lo=7->-1, hi=9->+1)
    var buf: [144]u8 = undefined;
    for (0..8) |b| {
        std.mem.writeInt(u16, buf[b * 18 ..][0..2], 0x3c00, .little);
        @memset(buf[b * 18 + 2 ..][0..16], 0x97);
    }
    var vec: [SELECTOR_RANK]f32 = undefined;
    dequantQ4_0Row256(&buf, 0, &vec);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), vec[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vec[16], 1e-6);
    const dot = dotQ4_0Row256(&buf, 0, &vec);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), dot, 1e-4);
}
