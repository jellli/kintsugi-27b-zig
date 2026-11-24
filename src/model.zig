//! Static single-arena GPU weight loader and VRAM allocator for Kintsugi-27B (Qwen3.8-27B).
//! Eliminates ggml's ~350-500 MiB framework/graph buffer overhead so RTX 4070 Ti (12 GB) fits
//! 106K+ Q4_0 VRAM KV cells alongside the full 27B trunk, Stage-1 Q4_0 LM head, and DFlash2.

const std = @import("std");
const config = @import("config.zig");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");
const escha_ref = @import("escha_ref.zig");

const Spec = config.ModelSpec;

pub const EschaProj = struct {
    code: cuda.CUdeviceptr,
    rin: cuda.CUdeviceptr,
    rout: cuda.CUdeviceptr,
    bias: cuda.CUdeviceptr,
    k_bits: u8, // 2 or 3
    ic: u32,
    oc: u32,
};

pub const GdnLayer = struct {
    attn_norm: cuda.CUdeviceptr,
    post_attn_norm: cuda.CUdeviceptr,
    ssm_a: cuda.CUdeviceptr,
    ssm_dt_bias: cuda.CUdeviceptr,
    ssm_conv1d: cuda.CUdeviceptr,
    ssm_norm: cuda.CUdeviceptr,
    ssm_alpha: cuda.CUdeviceptr,
    ssm_beta: cuda.CUdeviceptr,
    attn_qkv: EschaProj,
    attn_gate: EschaProj,
    ssm_out: EschaProj,
    ffn_gate: EschaProj,
    ffn_up: EschaProj,
    ffn_down: EschaProj,
    gdn_idx: u32, // 0..47
};

pub const FullAttnLayer = struct {
    attn_norm: cuda.CUdeviceptr,
    post_attn_norm: cuda.CUdeviceptr,
    attn_q_norm: cuda.CUdeviceptr,
    attn_k_norm: cuda.CUdeviceptr,
    attn_q: EschaProj,
    attn_k: EschaProj,
    attn_v: EschaProj,
    attn_output: EschaProj,
    ffn_gate: EschaProj,
    ffn_up: EschaProj,
    ffn_down: EschaProj,
    attn_idx: u32, // 0..15
};

pub const Layer = union(enum) {
    gdn: GdnLayer,
    full_attn: FullAttnLayer,
};

pub const TargetModel = struct {
    allocator: std.mem.Allocator,
    gguf_file: gguf.GgufFile,
    trunk_arena: cuda.CUdeviceptr,
    trunk_bytes: usize,
    head_q4_0: cuda.CUdeviceptr,
    head_q4_bytes: usize,
    output_norm: cuda.CUdeviceptr,
    layers: [Spec.n_layers]Layer,
    /// Host mmap slices (ZERO VRAM!)
    token_embd_q8: []const u8,
    output_w_q8: []const u8,

    pub fn load(
        allocator: std.mem.Allocator,
        ctx: *const cuda.Context,
        fn_q8_to_q4: cuda.CUfunction,
        model_path: []const u8,
    ) !TargetModel {
        const path_z = try allocator.allocSentinel(u8, model_path.len, 0);
        defer allocator.free(path_z);
        @memcpy(path_z, model_path);

        var gf = try gguf.GgufFile.open(allocator, path_z);
        errdefer gf.deinit(allocator);

        const emb_info = gf.find("token_embd.weight") orelse return error.MissingTokenEmbd;
        const out_info = gf.find("output.weight") orelse return error.MissingOutputWeight;
        const token_embd_q8 = gf.tensorBytes(emb_info);
        const output_w_q8 = gf.tensorBytes(out_info);

        // Pass 1: compute total aligned trunk VRAM bytes
        var total_bytes: usize = 0;
        var name_buf: [96]u8 = undefined;

        const addTensorSize = struct {
            fn f(g: *const gguf.GgufFile, name: []const u8, acc: *usize) !void {
                const info = g.find(name) orelse {
                    std.log.err("Missing required tensor: {s}", .{name});
                    return error.MissingTensor;
                };
                acc.* = alignUp256(acc.*) + @as(usize, @intCast(info.byteSize()));
            }
        }.f;

        try addTensorSize(&gf, "output_norm.weight", &total_bytes);
        for (0..Spec.n_layers) |l| {
            try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.attn_norm.weight", .{l}), &total_bytes);
            try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.post_attention_norm.weight", .{l}), &total_bytes);
            if (Spec.isFullAttnLayer(@intCast(l))) {
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.attn_q_norm.weight", .{l}), &total_bytes);
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.attn_k_norm.weight", .{l}), &total_bytes);
                for ([_][]const u8{ "attn_q", "attn_k", "attn_v", "attn_output" }) |p| {
                    try addEschaSize(&gf, l, p, &total_bytes);
                }
            } else {
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.ssm_a", .{l}), &total_bytes);
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.ssm_dt.bias", .{l}), &total_bytes);
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.ssm_conv1d.weight", .{l}), &total_bytes);
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.ssm_norm.weight", .{l}), &total_bytes);
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.ssm_alpha.weight", .{l}), &total_bytes);
                try addTensorSize(&gf, fmt(&name_buf, "blk.{d}.ssm_beta.weight", .{l}), &total_bytes);
                for ([_][]const u8{ "attn_qkv", "attn_gate", "ssm_out" }) |p| {
                    try addEschaSize(&gf, l, p, &total_bytes);
                }
            }
            for ([_][]const u8{ "ffn_gate", "ffn_up", "ffn_down" }) |p| {
                try addEschaSize(&gf, l, p, &total_bytes);
            }
        }
        total_bytes = alignUp256(total_bytes);

        // Pass 2: allocate single contiguous VRAM trunk arena and upload all 64 layers
        const arena = try cuda.Context.alloc(total_bytes);
        errdefer cuda.Context.free(arena);

        var cursor: usize = 0;
        const output_norm = try uploadTensor(&gf, arena, &cursor, "output_norm.weight");

        var layers: [Spec.n_layers]Layer = undefined;
        var gdn_idx: u32 = 0;
        var attn_idx: u32 = 0;

        for (0..Spec.n_layers) |l| {
            const attn_norm = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.attn_norm.weight", .{l}));
            const post_norm = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.post_attention_norm.weight", .{l}));
            if (Spec.isFullAttnLayer(@intCast(l))) {
                const q_norm = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.attn_q_norm.weight", .{l}));
                const k_norm = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.attn_k_norm.weight", .{l}));
                layers[l] = .{
                    .full_attn = .{
                        .attn_norm = attn_norm,
                        .post_attn_norm = post_norm,
                        .attn_q_norm = q_norm,
                        .attn_k_norm = k_norm,
                        .attn_q = try uploadEscha(&gf, arena, &cursor, l, "attn_q", Spec.d_model, Spec.n_heads * 2 * Spec.head_dim),
                        .attn_k = try uploadEscha(&gf, arena, &cursor, l, "attn_k", Spec.d_model, Spec.n_kv_heads * Spec.head_dim),
                        .attn_v = try uploadEscha(&gf, arena, &cursor, l, "attn_v", Spec.d_model, Spec.n_kv_heads * Spec.head_dim),
                        .attn_output = try uploadEscha(&gf, arena, &cursor, l, "attn_output", Spec.n_heads * Spec.head_dim, Spec.d_model),
                        .ffn_gate = try uploadEscha(&gf, arena, &cursor, l, "ffn_gate", Spec.d_model, Spec.d_ff),
                        .ffn_up = try uploadEscha(&gf, arena, &cursor, l, "ffn_up", Spec.d_model, Spec.d_ff),
                        .ffn_down = try uploadEscha(&gf, arena, &cursor, l, "ffn_down", Spec.d_ff, Spec.d_model),
                        .attn_idx = attn_idx,
                    },
                };
                attn_idx += 1;
            } else {
                layers[l] = .{
                    .gdn = .{
                        .attn_norm = attn_norm,
                        .post_attn_norm = post_norm,
                        .ssm_a = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.ssm_a", .{l})),
                        .ssm_dt_bias = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.ssm_dt.bias", .{l})),
                        .ssm_conv1d = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.ssm_conv1d.weight", .{l})),
                        .ssm_norm = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.ssm_norm.weight", .{l})),
                        .ssm_alpha = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.ssm_alpha.weight", .{l})),
                        .ssm_beta = try uploadTensor(&gf, arena, &cursor, fmt(&name_buf, "blk.{d}.ssm_beta.weight", .{l})),
                        .attn_qkv = try uploadEscha(&gf, arena, &cursor, l, "attn_qkv", Spec.d_model, 10240),
                        .attn_gate = try uploadEscha(&gf, arena, &cursor, l, "attn_gate", Spec.d_model, Spec.ssm_inner_size),
                        .ssm_out = try uploadEscha(&gf, arena, &cursor, l, "ssm_out", Spec.ssm_inner_size, Spec.d_model),
                        .ffn_gate = try uploadEscha(&gf, arena, &cursor, l, "ffn_gate", Spec.d_model, Spec.d_ff),
                        .ffn_up = try uploadEscha(&gf, arena, &cursor, l, "ffn_up", Spec.d_model, Spec.d_ff),
                        .ffn_down = try uploadEscha(&gf, arena, &cursor, l, "ffn_down", Spec.d_ff, Spec.d_model),
                        .gdn_idx = gdn_idx,
                    },
                };
                gdn_idx += 1;
            }
        }

        // Pass 3: Build 682 MiB Stage-1 Q4_0 LM head directly on GPU from host mmap output.weight (Q8_0)
        const total_blocks: u64 = (@as(u64, Spec.n_vocab) * @as(u64, Spec.d_model)) / 32;
        const head_q4_bytes: usize = @intCast(total_blocks * 18);
        const head_q4_0 = try cuda.Context.alloc(head_q4_bytes);
        errdefer cuda.Context.free(head_q4_0);

        const chunk_blocks: u64 = 256 * 1024; // 256K blocks = 8.5 MiB Q8_0 staging
        const d_stage = try cuda.Context.alloc(@intCast(chunk_blocks * 34));
        defer cuda.Context.free(d_stage);

        var done_blocks: u64 = 0;
        while (done_blocks < total_blocks) {
            const n_blk: u64 = @min(chunk_blocks, total_blocks - done_blocks);
            const src_off: usize = @intCast(done_blocks * 34);
            const src_len: usize = @intCast(n_blk * 34);
            try ctx.copyHtoDAsync(d_stage, output_w_q8[src_off .. src_off + src_len]);

            var p_src = d_stage;
            var p_dst = head_q4_0 + done_blocks * 18;
            var p_n: u64 = n_blk;
            var params = [_]?*anyopaque{ @ptrCast(&p_src), @ptrCast(&p_dst), @ptrCast(&p_n) };
            const grid_x: u32 = @intCast((n_blk + 7) / 8);
            try cuda.launch(fn_q8_to_q4, .{ grid_x, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
            try ctx.sync();
            done_blocks += n_blk;
        }

        return .{
            .allocator = allocator,
            .gguf_file = gf,
            .trunk_arena = arena,
            .trunk_bytes = total_bytes,
            .head_q4_0 = head_q4_0,
            .head_q4_bytes = head_q4_bytes,
            .output_norm = output_norm,
            .layers = layers,
            .token_embd_q8 = token_embd_q8,
            .output_w_q8 = output_w_q8,
        };
    }

    pub fn deinit(self: *TargetModel) void {
        cuda.Context.free(self.head_q4_0);
        cuda.Context.free(self.trunk_arena);
        self.gguf_file.deinit(self.allocator);
    }

    /// Fast CPU Q8_0 row embedding lookup from host mmap (ZERO VRAM for 1.29 GiB token_embd!)
    pub fn embedToken(self: *const TargetModel, token_id: u32, dst: []f32) void {
        const nb: usize = Spec.d_model / 32; // 160
        const row_off: usize = @as(usize, token_id) * nb * 34;
        const row = self.token_embd_q8[row_off .. row_off + nb * 34];
        for (0..nb) |b| {
            const blk = row[b * 34 ..][0..34];
            const d_bits = std.mem.readInt(u16, blk[0..2], .little);
            const d = escha_ref.f16BitsToF32(d_bits);
            const d_vec: @Vector(8, f32) = @splat(d);
            const qs: *const [32]i8 = @ptrCast(blk[2..34]);
            inline for (0..4) |g| {
                var qf: @Vector(8, f32) = undefined;
                inline for (0..8) |j| {
                    qf[j] = @floatFromInt(qs[g * 8 + j]);
                }
                const res = qf * d_vec;
                dst[b * 32 + g * 8 ..][0..8].* = res;
            }
        }
    }

    /// Exact Stage-2 sparse Q8_0 dot product directly from host mmap output.weight (ZERO VRAM!)
    /// SIMD @Vector(8, f32): 64 candidates x 5120 takes ~20 us per token on modern x86_64!
    pub fn rescoreQ8_0Row(self: *const TargetModel, row_id: u32, x_norm: []const f32) f32 {
        const nb: usize = Spec.d_model / 32; // 160
        const row_off: usize = @as(usize, row_id) * nb * 34;
        const row = self.output_w_q8[row_off .. row_off + nb * 34];
        var acc: @Vector(8, f32) = @splat(0.0);
        for (0..nb) |b| {
            const blk = row[b * 34 ..][0..34];
            const d = escha_ref.f16BitsToF32(std.mem.readInt(u16, blk[0..2], .little));
            const qs: *const [32]i8 = @ptrCast(blk[2..34]);
            var blk_acc: @Vector(8, f32) = @splat(0.0);
            inline for (0..4) |g| {
                const xv: @Vector(8, f32) = x_norm[b * 32 + g * 8 ..][0..8].*;
                var qv: @Vector(8, f32) = undefined;
                inline for (0..8) |j| {
                    qv[j] = @floatFromInt(qs[g * 8 + j]);
                }
                blk_acc += xv * qv;
            }
            const d_vec: @Vector(8, f32) = @splat(d);
            acc += blk_acc * d_vec;
        }
        return @reduce(.Add, acc);
    }
};

fn alignUp256(v: usize) usize {
    return (v + 255) & ~@as(usize, 255);
}

fn fmt(buf: []u8, comptime fstr: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fstr, args) catch unreachable;
}

fn addEschaSize(gf: *const gguf.GgufFile, l: usize, proj: []const u8, acc: *usize) !void {
    var b: [96]u8 = undefined;
    for ([_][]const u8{ "escha_code", "escha_rin", "escha_rout", "bias" }) |suf| {
        const nm = fmt(&b, "blk.{d}.{s}.{s}", .{ l, proj, suf });
        const info = gf.find(nm) orelse {
            std.log.err("Missing Escha tensor: {s}", .{nm});
            return error.MissingTensor;
        };
        acc.* = alignUp256(acc.*) + @as(usize, @intCast(info.byteSize()));
    }
}

fn uploadTensor(gf: *const gguf.GgufFile, arena: cuda.CUdeviceptr, cursor: *usize, name: []const u8) !cuda.CUdeviceptr {
    const info = gf.find(name) orelse return error.MissingTensor;
    const bytes = gf.tensorBytes(info);
    cursor.* = alignUp256(cursor.*);
    const dst = arena + cursor.*;
    try cuda.Context.copyHtoD(dst, bytes);
    cursor.* += bytes.len;
    return dst;
}

fn uploadEscha(
    gf: *const gguf.GgufFile,
    arena: cuda.CUdeviceptr,
    cursor: *usize,
    l: usize,
    proj: []const u8,
    ic: u32,
    oc: u32,
) !EschaProj {
    var b: [96]u8 = undefined;
    const code_nm = fmt(&b, "blk.{d}.{s}.escha_code", .{ l, proj });
    const code_info = gf.find(code_nm) orelse return error.MissingTensor;
    const k_bits: u8 = @intCast(code_info.dims[0] / 16);

    const code = try uploadTensor(gf, arena, cursor, code_nm);
    const rin = try uploadTensor(gf, arena, cursor, fmt(&b, "blk.{d}.{s}.escha_rin", .{ l, proj }));
    const rout = try uploadTensor(gf, arena, cursor, fmt(&b, "blk.{d}.{s}.escha_rout", .{ l, proj }));
    const bias = try uploadTensor(gf, arena, cursor, fmt(&b, "blk.{d}.{s}.bias", .{ l, proj }));

    return .{
        .code = code,
        .rin = rin,
        .rout = rout,
        .bias = bias,
        .k_bits = k_bits,
        .ic = ic,
        .oc = oc,
    };
}
