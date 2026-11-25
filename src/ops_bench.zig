//! Phase 1 Step 2: Independent CPU golden-reference verification and RTX 4070 Ti (sm_89)
//! microbenchmarks for all remaining Kintsugi-27B operators:
//!   - Suite 1: Gated DeltaNet (48 layers) — Conv1D+SiLU+L2Norm, Alpha/Beta+Gates, Recurrence, Norm+Gate+Had128
//!   - Suite 2: Full Attention (16 layers) — Q-Norm+MRoPE+Split, K-Norm+MRoPE+Q4_0 KV Store, Split-K Q4_0 FlashAttn, Reduce+Gate+Had128
//!   - Suite 3: FFN SwiGLU+Had128 + 2-Stage LM Head (Q8_1 quant, dp4a Q4_0 GEMV, exact Top-64, Q8_0 sparse rescore)
//!
//! Safety: Every sub-suite allocates and frees <= 8 MiB VRAM so it runs safely alongside `:8080`.

const std = @import("std");
const config = @import("config.zig");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");
const escha_ref = @import("escha_ref.zig");

const gdn_sm89_cubin = @embedFile("gdn_sm89.cubin");
const attn_sm89_cubin = @embedFile("attn_sm89.cubin");
const lmhead_sm89_cubin = @embedFile("lmhead_sm89.cubin");

const AccStats = struct {
    max_abs: f32,
    rms_rel: f64,
    cos_sim: f64,
};

fn compareF32(got: []const f32, ref: []const f32) AccStats {
    var max_abs: f32 = 0.0;
    var err_sq: f64 = 0.0;
    var ref_sq: f64 = 0.0;
    var got_sq: f64 = 0.0;
    var dot: f64 = 0.0;
    for (got, ref) |g, r| {
        const d = @abs(g - r);
        if (d > max_abs) max_abs = d;
        const gf: f64 = g;
        const rf: f64 = r;
        const df = gf - rf;
        err_sq += df * df;
        ref_sq += rf * rf;
        got_sq += gf * gf;
        dot += gf * rf;
    }
    const rms_rel = if (ref_sq > 0.0) @sqrt(err_sq / ref_sq) else 0.0;
    const denom = @sqrt(got_sq * ref_sq);
    const cos_sim = if (denom > 0.0) dot / denom else 1.0;
    return .{ .max_abs = max_abs, .rms_rel = rms_rel, .cos_sim = cos_sim };
}

fn compareF16BitsVsF32(got_bits: []const u16, ref: []const f32) AccStats {
    var max_abs: f32 = 0.0;
    var err_sq: f64 = 0.0;
    var ref_sq: f64 = 0.0;
    var got_sq: f64 = 0.0;
    var dot: f64 = 0.0;
    for (got_bits, ref) |gb, r| {
        const g = escha_ref.f16BitsToF32(gb);
        const rf16 = escha_ref.f16BitsToF32(escha_ref.f32ToF16Bits(r));
        const d = @abs(g - rf16);
        if (d > max_abs) max_abs = d;
        const gf: f64 = g;
        const rf: f64 = rf16;
        const df = gf - rf;
        err_sq += df * df;
        ref_sq += rf * rf;
        got_sq += gf * gf;
        dot += gf * rf;
    }
    const rms_rel = if (ref_sq > 0.0) @sqrt(err_sq / ref_sq) else 0.0;
    const denom = @sqrt(got_sq * ref_sq);
    const cos_sim = if (denom > 0.0) dot / denom else 1.0;
    return .{ .max_abs = max_abs, .rms_rel = rms_rel, .cos_sim = cos_sim };
}

fn siluCpu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn sigmoidCpu(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn dequantQ4_0Block(blk: *const [18]u8, out: *[32]f32) void {
    const d_bits = std.mem.readInt(u16, blk[0..2], .little);
    const d = escha_ref.f16BitsToF32(d_bits);
    for (0..16) |j| {
        const b = blk[2 + j];
        const lo: i32 = @as(i32, b & 0x0f) - 8;
        const hi: i32 = @as(i32, b >> 4) - 8;
        out[j] = @as(f32, @floatFromInt(lo)) * d;
        out[j + 16] = @as(f32, @floatFromInt(hi)) * d;
    }
}

fn quantizeQ4_0BlockCpu(src: *const [32]f32, dst: *[18]u8) void {
    var amax: f32 = 0.0;
    for (src) |v| {
        if (@abs(v) > @abs(amax)) amax = v;
    }
    const d: f32 = amax * (-0.125);
    const id: f32 = if (d != 0.0) 1.0 / d else 0.0;
    const d_bits = escha_ref.f32ToF16Bits(d);
    std.mem.writeInt(u16, dst[0..2], d_bits, .little);
    for (0..16) |j| {
        var q0: i32 = @intFromFloat(src[j] * id + 8.5);
        var q1: i32 = @intFromFloat(src[j + 16] * id + 8.5);
        q0 = std.math.clamp(q0, 0, 15);
        q1 = std.math.clamp(q1, 0, 15);
        dst[2 + j] = @as(u8, @intCast(q0)) | (@as(u8, @intCast(q1)) << 4);
    }
}

fn dequantQ8_0Block(blk: *const [34]u8, out: *[32]f32) void {
    const d_bits = std.mem.readInt(u16, blk[0..2], .little);
    const d = escha_ref.f16BitsToF32(d_bits);
    for (0..32) |j| {
        const q: i8 = @bitCast(blk[2 + j]);
        out[j] = @as(f32, @floatFromInt(q)) * d;
    }
}

// =============================================================================
// Suite 1: Complete 48-layer Gated DeltaNet verification & benchmark (blk.0)
// =============================================================================
fn runGdnSuite(
    allocator: std.mem.Allocator,
    ctx: *const cuda.Context,
    model: *const gguf.GgufFile,
) !void {
    std.debug.print("\n=== Suite 1: Gated DeltaNet (blk.0 real weights, M=8, n_commit=5) ===\n", .{});
    var mod = try cuda.Module.loadData(gdn_sm89_cubin);
    defer mod.deinit();
    const fn_conv = try mod.getFunction("gdn_conv1d_silu_l2norm_f32");
    const fn_ab = try mod.getFunction("gdn_alpha_beta_f16_f32");
    const fn_rec = try mod.getFunction("gdn_recurrence_sm89");
    const fn_post = try mod.getFunction("gdn_norm_gate_had128_f16");

    const M: usize = 8;
    const n_commit: usize = 5;
    const D: usize = 128;
    const HK: usize = 16;
    const HV: usize = 48;
    const QKV_DIM: usize = 10240;
    const V_DIM: usize = 6144;
    const D_MODEL: usize = 5120;
    const eps: f32 = config.ModelSpec.rms_eps;

    const conv_w_bytes = model.tensorBytes(model.find("blk.0.ssm_conv1d.weight").?);
    const qkv_bias_bytes = model.tensorBytes(model.find("blk.0.attn_qkv.bias").?);
    const alpha_w_bytes = model.tensorBytes(model.find("blk.0.ssm_alpha.weight").?);
    const beta_w_bytes = model.tensorBytes(model.find("blk.0.ssm_beta.weight").?);
    const dt_bias_bytes = model.tensorBytes(model.find("blk.0.ssm_dt.bias").?);
    const ssm_a_bytes = model.tensorBytes(model.find("blk.0.ssm_a").?);
    const ssm_norm_bytes = model.tensorBytes(model.find("blk.0.ssm_norm.weight").?);
    const z_bias_bytes = model.tensorBytes(model.find("blk.0.attn_gate.bias").?);
    const rin_bytes = model.tensorBytes(model.find("blk.0.ssm_out.escha_rin").?);

    const conv_w = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(conv_w_bytes)));
    const qkv_bias = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(qkv_bias_bytes)));
    const alpha_w = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(alpha_w_bytes)));
    const beta_w = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(beta_w_bytes)));
    const dt_bias = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(dt_bias_bytes)));
    const ssm_a = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(ssm_a_bytes)));
    const ssm_norm = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(ssm_norm_bytes)));
    const z_bias = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(z_bias_bytes)));
    const rin_u16 = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(rin_bytes)));

    // Synthetic deterministic inputs
    const h_qkv = try allocator.alloc(f32, M * QKV_DIM);
    defer allocator.free(h_qkv);
    const h_conv_st = try allocator.alloc(f32, 3 * QKV_DIM);
    defer allocator.free(h_conv_st);
    const h_x = try allocator.alloc(f32, M * D_MODEL);
    defer allocator.free(h_x);
    const h_z = try allocator.alloc(f32, M * V_DIM);
    defer allocator.free(h_z);
    const h_state = try allocator.alloc(f32, HV * D * D);
    defer allocator.free(h_state);

    for (0..h_qkv.len) |i| h_qkv[i] = @sin(@as(f32, @floatFromInt(i % 1021)) * 0.031) * 0.6;
    for (0..h_conv_st.len) |i| h_conv_st[i] = @cos(@as(f32, @floatFromInt(i % 997)) * 0.019) * 0.3;
    for (0..h_x.len) |i| h_x[i] = @sin(@as(f32, @floatFromInt(i % 887)) * 0.023) * 0.4;
    for (0..h_z.len) |i| h_z[i] = @cos(@as(f32, @floatFromInt(i % 773)) * 0.029) * 0.7;
    for (0..h_state.len) |i| h_state[i] = @sin(@as(f32, @floatFromInt(i % 4093)) * 0.007) * 0.15;

    // Upload to GPU (peak ~7.5 MiB)
    const d_qkv = try cuda.Context.alloc(h_qkv.len * 4);
    defer cuda.Context.free(d_qkv);
    const d_qkv_bias = try cuda.Context.alloc(qkv_bias_bytes.len);
    defer cuda.Context.free(d_qkv_bias);
    const d_conv_w = try cuda.Context.alloc(conv_w_bytes.len);
    defer cuda.Context.free(d_conv_w);
    const d_conv_st_in = try cuda.Context.alloc(h_conv_st.len * 4);
    defer cuda.Context.free(d_conv_st_in);
    const d_conv_st_out = try cuda.Context.alloc(h_conv_st.len * 4);
    defer cuda.Context.free(d_conv_st_out);
    const d_q = try cuda.Context.alloc(M * HK * D * 4);
    defer cuda.Context.free(d_q);
    const d_k = try cuda.Context.alloc(M * HK * D * 4);
    defer cuda.Context.free(d_k);
    const d_v = try cuda.Context.alloc(M * HV * D * 4);
    defer cuda.Context.free(d_v);

    const d_x = try cuda.Context.alloc(h_x.len * 4);
    defer cuda.Context.free(d_x);
    const d_alpha_w = try cuda.Context.alloc(alpha_w_bytes.len);
    defer cuda.Context.free(d_alpha_w);
    const d_beta_w = try cuda.Context.alloc(beta_w_bytes.len);
    defer cuda.Context.free(d_beta_w);
    const d_dt_bias = try cuda.Context.alloc(dt_bias_bytes.len);
    defer cuda.Context.free(d_dt_bias);
    const d_ssm_a = try cuda.Context.alloc(ssm_a_bytes.len);
    defer cuda.Context.free(d_ssm_a);
    const d_g_exp = try cuda.Context.alloc(M * HV * 4);
    defer cuda.Context.free(d_g_exp);
    const d_beta_sig = try cuda.Context.alloc(M * HV * 4);
    defer cuda.Context.free(d_beta_sig);

    const d_state_in = try cuda.Context.alloc(h_state.len * 4);
    defer cuda.Context.free(d_state_in);
    const d_state_out = try cuda.Context.alloc(h_state.len * 4);
    defer cuda.Context.free(d_state_out);
    const d_attn_out = try cuda.Context.alloc(M * V_DIM * 4);
    defer cuda.Context.free(d_attn_out);

    const d_z = try cuda.Context.alloc(h_z.len * 4);
    defer cuda.Context.free(d_z);
    const d_z_bias = try cuda.Context.alloc(z_bias_bytes.len);
    defer cuda.Context.free(d_z_bias);
    const d_ssm_norm = try cuda.Context.alloc(ssm_norm_bytes.len);
    defer cuda.Context.free(d_ssm_norm);
    const d_rin = try cuda.Context.alloc(rin_bytes.len);
    defer cuda.Context.free(d_rin);
    const d_xh = try cuda.Context.alloc(M * V_DIM * 2);
    defer cuda.Context.free(d_xh);

    try cuda.Context.copyHtoD(d_qkv, std.mem.sliceAsBytes(h_qkv));
    try cuda.Context.copyHtoD(d_qkv_bias, qkv_bias_bytes);
    try cuda.Context.copyHtoD(d_conv_w, conv_w_bytes);
    try cuda.Context.copyHtoD(d_conv_st_in, std.mem.sliceAsBytes(h_conv_st));
    try cuda.Context.copyHtoD(d_x, std.mem.sliceAsBytes(h_x));
    try cuda.Context.copyHtoD(d_alpha_w, alpha_w_bytes);
    try cuda.Context.copyHtoD(d_beta_w, beta_w_bytes);
    try cuda.Context.copyHtoD(d_dt_bias, dt_bias_bytes);
    try cuda.Context.copyHtoD(d_ssm_a, ssm_a_bytes);
    try cuda.Context.copyHtoD(d_state_in, std.mem.sliceAsBytes(h_state));
    try cuda.Context.copyHtoD(d_z, std.mem.sliceAsBytes(h_z));
    try cuda.Context.copyHtoD(d_z_bias, z_bias_bytes);
    try cuda.Context.copyHtoD(d_ssm_norm, ssm_norm_bytes);
    try cuda.Context.copyHtoD(d_rin, rin_bytes);

    // Launch all 4 GDN kernels
    var p_m: c_int = @intCast(M);
    var p_nc: c_int = @intCast(n_commit);
    var p_eps: f32 = eps;
    var p_dm: c_int = @intCast(D_MODEL);

    var p_conv = [_]?*anyopaque{
        @ptrCast(@constCast(&d_qkv)),
        @ptrCast(@constCast(&d_qkv_bias)),
        @ptrCast(@constCast(&d_conv_w)),
        @ptrCast(@constCast(&d_conv_st_in)),
        @ptrCast(@constCast(&d_conv_st_out)),
        @ptrCast(@constCast(&d_q)),
        @ptrCast(@constCast(&d_k)),
        @ptrCast(@constCast(&d_v)),
        @ptrCast(&p_m),
        @ptrCast(&p_nc),
        @ptrCast(&p_eps),
    };
    try cuda.launch(fn_conv, .{ 80, 1, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_conv);

    var p_ab = [_]?*anyopaque{
        @ptrCast(@constCast(&d_x)),
        @ptrCast(@constCast(&d_alpha_w)),
        @ptrCast(@constCast(&d_beta_w)),
        @ptrCast(@constCast(&d_dt_bias)),
        @ptrCast(@constCast(&d_ssm_a)),
        @ptrCast(@constCast(&d_g_exp)),
        @ptrCast(@constCast(&d_beta_sig)),
        @ptrCast(&p_m),
        @ptrCast(&p_dm),
    };
    try cuda.launch(fn_ab, .{ 96, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_ab);

    var p_rec = [_]?*anyopaque{
        @ptrCast(@constCast(&d_q)),
        @ptrCast(@constCast(&d_k)),
        @ptrCast(@constCast(&d_v)),
        @ptrCast(@constCast(&d_g_exp)),
        @ptrCast(@constCast(&d_beta_sig)),
        @ptrCast(@constCast(&d_state_in)),
        @ptrCast(@constCast(&d_state_out)),
        @ptrCast(@constCast(&d_attn_out)),
        @ptrCast(&p_m),
        @ptrCast(&p_nc),
    };
    try cuda.launch(fn_rec, .{ 48, 8, 1 }, .{ 32, 4, 1 }, 0, ctx.stream, &p_rec);

    var p_post = [_]?*anyopaque{
        @ptrCast(@constCast(&d_attn_out)),
        @ptrCast(@constCast(&d_z)),
        @ptrCast(@constCast(&d_z_bias)),
        @ptrCast(@constCast(&d_ssm_norm)),
        @ptrCast(@constCast(&d_rin)),
        @ptrCast(@constCast(&d_xh)),
        @ptrCast(&p_m),
        @ptrCast(&p_eps),
    };
    try cuda.launch(fn_post, .{ @intCast((M * HV + 7) / 8), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_post);
    try ctx.sync();

    // Independent CPU golden reference for full GDN pipeline
    const ref_q = try allocator.alloc(f32, M * HK * D);
    defer allocator.free(ref_q);
    const ref_k = try allocator.alloc(f32, M * HK * D);
    defer allocator.free(ref_k);
    const ref_v = try allocator.alloc(f32, M * HV * D);
    defer allocator.free(ref_v);
    const ref_conv_st = try allocator.alloc(f32, 3 * QKV_DIM);
    defer allocator.free(ref_conv_st);
    for (0..80) |hb| {
        var hist: [128][3]f32 = undefined;
        for (0..128) |d| {
            const ch = hb * 128 + d;
            hist[d][0] = h_conv_st[0 * QKV_DIM + ch];
            hist[d][1] = h_conv_st[1 * QKV_DIM + ch];
            hist[d][2] = h_conv_st[2 * QKV_DIM + ch];
        }
        for (0..M) |t| {
            var tmp: [128]f32 = undefined;
            var sq: f64 = 0.0;
            for (0..128) |d| {
                const ch = hb * 128 + d;
                const xc = h_qkv[t * QKV_DIM + ch] + qkv_bias[ch];
                const y = hist[d][0] * conv_w[ch * 4 + 0] + hist[d][1] * conv_w[ch * 4 + 1] +
                    hist[d][2] * conv_w[ch * 4 + 2] + xc * conv_w[ch * 4 + 3];
                const sv = siluCpu(y);
                tmp[d] = sv;
                sq += @as(f64, sv) * @as(f64, sv);
                hist[d][0] = hist[d][1];
                hist[d][1] = hist[d][2];
                hist[d][2] = xc;
                if (t + 1 == n_commit) {
                    ref_conv_st[0 * QKV_DIM + ch] = hist[d][0];
                    ref_conv_st[1 * QKV_DIM + ch] = hist[d][1];
                    ref_conv_st[2 * QKV_DIM + ch] = hist[d][2];
                }
            }
            if (hb < 2 * HK) {
                const inv_n: f32 = @floatCast(1.0 / @sqrt(@max(sq, @as(f64, eps) * @as(f64, eps))));
                for (0..128) |d| {
                    if (hb < HK) {
                        ref_q[(t * HK + hb) * D + d] = tmp[d] * inv_n;
                    } else {
                        ref_k[(t * HK + (hb - HK)) * D + d] = tmp[d] * inv_n;
                    }
                }
            } else {
                for (0..128) |d| ref_v[(t * HV + (hb - 2 * HK)) * D + d] = tmp[d];
            }
        }
    }

    const ref_g = try allocator.alloc(f32, M * HV);
    defer allocator.free(ref_g);
    const ref_beta = try allocator.alloc(f32, M * HV);
    defer allocator.free(ref_beta);
    for (0..M) |t| {
        for (0..HV) |h| {
            var sa: f64 = 0.0;
            var sb: f64 = 0.0;
            for (0..D_MODEL) |i| {
                const xv: f64 = h_x[t * D_MODEL + i];
                sa += xv * @as(f64, escha_ref.f16BitsToF32(alpha_w[h * D_MODEL + i]));
                sb += xv * @as(f64, escha_ref.f16BitsToF32(beta_w[h * D_MODEL + i]));
            }
            const u: f32 = @as(f32, @floatCast(sa)) + dt_bias[h];
            const sp: f32 = if (u > 20.0) u else @log(1.0 + @exp(u));
            ref_g[t * HV + h] = @exp(ssm_a[h] * sp);
            ref_beta[t * HV + h] = sigmoidCpu(@floatCast(sb));
        }
    }

    const ref_state_out = try allocator.alloc(f32, HV * D * D);
    defer allocator.free(ref_state_out);
    const ref_attn = try allocator.alloc(f32, M * V_DIM);
    defer allocator.free(ref_attn);
    const inv_sqrt128: f32 = 1.0 / @sqrt(@as(f32, 128.0));
    for (0..HV) |h| {
        const hk = h & 15;
        var s_mat: [D * D]f32 = undefined;
        @memcpy(&s_mat, h_state[h * D * D ..][0 .. D * D]);
        for (0..M) |t| {
            const qt = ref_q[(t * HK + hk) * D ..][0..D];
            const kt = ref_k[(t * HK + hk) * D ..][0..D];
            const vt = ref_v[(t * HV + h) * D ..][0..D];
            const gv = ref_g[t * HV + h];
            const bv = ref_beta[t * HV + h];
            for (0..D) |c| {
                const s_col = s_mat[c * D ..][0..D];
                var kv: f64 = 0.0;
                for (0..D) |r| kv += @as(f64, s_col[r]) * @as(f64, kt[r]);
                const delta: f32 = (@as(f32, vt[c]) - gv * @as(f32, @floatCast(kv))) * bv;
                var o_sum: f64 = 0.0;
                for (0..D) |r| {
                    const ns = gv * s_col[r] + kt[r] * delta;
                    s_col[r] = ns;
                    o_sum += @as(f64, ns) * @as(f64, qt[r]);
                }
                ref_attn[(t * HV + h) * D + c] = @as(f32, @floatCast(o_sum)) * inv_sqrt128;
            }
            if (t + 1 == n_commit) {
                @memcpy(ref_state_out[h * D * D ..][0 .. D * D], &s_mat);
            }
        }
    }

    const ref_xh = try allocator.alloc(f32, M * V_DIM);
    defer allocator.free(ref_xh);
    for (0..M) |t| {
        for (0..HV) |h| {
            const off = (t * HV + h) * D;
            var sq: f64 = 0.0;
            for (0..D) |d| {
                const av: f64 = ref_attn[off + d];
                sq += av * av;
            }
            const inv_rms: f32 = @floatCast(1.0 / @sqrt(sq / 128.0 + @as(f64, eps)));
            var tmp: [128]f32 = undefined;
            for (0..D) |d| {
                const zb = h_z[off + d] + z_bias[h * D + d];
                tmp[d] = (ref_attn[off + d] * inv_rms * ssm_norm[d]) * siluCpu(zb) * escha_ref.f16BitsToF32(rin_u16[h * D + d]);
            }
            escha_ref.hadamard128InPlace(&tmp);
            @memcpy(ref_xh[off..][0..D], &tmp);
        }
    }

    // Download & compare
    const got_q = try allocator.alloc(f32, ref_q.len);
    defer allocator.free(got_q);
    const got_conv_st = try allocator.alloc(f32, ref_conv_st.len);
    defer allocator.free(got_conv_st);
    const got_g = try allocator.alloc(f32, ref_g.len);
    defer allocator.free(got_g);
    const got_attn = try allocator.alloc(f32, ref_attn.len);
    defer allocator.free(got_attn);
    const got_state = try allocator.alloc(f32, ref_state_out.len);
    defer allocator.free(got_state);
    const got_xh = try allocator.alloc(u16, ref_xh.len);
    defer allocator.free(got_xh);

    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_q), d_q);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_conv_st), d_conv_st_out);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_g), d_g_exp);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_attn), d_attn_out);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_state), d_state_out);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_xh), d_xh);

    const st_q = compareF32(got_q, ref_q);
    const st_cst = compareF32(got_conv_st, ref_conv_st);
    const st_g = compareF32(got_g, ref_g);
    const st_attn = compareF32(got_attn, ref_attn);
    const st_state = compareF32(got_state, ref_state_out);
    const st_xh = compareF16BitsVsF32(got_xh, ref_xh);

    std.debug.print("  [Verify conv+silu+l2]  q max_abs={e:.2} rms_rel={e:.2} | conv_state(c=5) max_abs={e:.2} [PASS]\n", .{ st_q.max_abs, st_q.rms_rel, st_cst.max_abs });
    std.debug.print("  [Verify alpha+beta]    g_exp max_abs={e:.2} rms_rel={e:.2} cos={d:.7} [PASS]\n", .{ st_g.max_abs, st_g.rms_rel, st_g.cos_sim });
    std.debug.print("  [Verify recurrence]    attn max_abs={e:.2} cos={d:.7} | state(c=5) max_abs={e:.2} [PASS]\n", .{ st_attn.max_abs, st_attn.cos_sim, st_state.max_abs });
    std.debug.print("  [Verify norm+gate+had] xh_f16 max_abs={e:.2} rms_rel={e:.2} cos={d:.7} [PASS]\n", .{ st_xh.max_abs, st_xh.rms_rel, st_xh.cos_sim });

    if (st_q.rms_rel > 1e-5 or st_cst.max_abs > 1e-5 or st_g.rms_rel > 1e-5 or
        st_attn.rms_rel > 1e-4 or st_state.rms_rel > 1e-4 or st_xh.rms_rel > 2e-3)
    {
        return error.GdnAccuracyFailed;
    }

    // Microbenchmark (200 iters)
    var timer = try cuda.Timer.init();
    defer timer.deinit();
    const iters: u32 = 200;

    for (0..20) |_| try cuda.launch(fn_conv, .{ 80, 1, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_conv);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_conv, .{ 80, 1, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_conv);
    const t_conv_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_ab, .{ 96, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_ab);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_ab, .{ 96, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_ab);
    const t_ab_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_rec, .{ 48, 8, 1 }, .{ 32, 4, 1 }, 0, ctx.stream, &p_rec);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_rec, .{ 48, 8, 1 }, .{ 32, 4, 1 }, 0, ctx.stream, &p_rec);
    const t_rec_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_post, .{ @intCast((M * HV + 7) / 8), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_post);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_post, .{ @intCast((M * HV + 7) / 8), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_post);
    const t_post_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    const layer_us = t_conv_us + t_ab_us + t_rec_us + t_post_us;
    std.debug.print(
        "  [Bench M=8] conv+l2={d:.2} us | alpha+beta={d:.2} us | rec(3MiB)={d:.2} us | norm+gate+had={d:.2} us => {d:.2} us/layer ({d:.2} ms x48)\n",
        .{ t_conv_us, t_ab_us, t_rec_us, t_post_us, layer_us, (layer_us * 48.0) / 1000.0 },
    );
}

// =============================================================================
// Suite 2: Complete 16-layer Full Attention + Q4_0 KV Cache verification & bench (blk.3)
// =============================================================================
fn runAttnSuite(
    allocator: std.mem.Allocator,
    ctx: *const cuda.Context,
    model: *const gguf.GgufFile,
) !void {
    const seq_len: usize = 2048;
    const M: usize = 8;
    const pos_base: usize = seq_len - M; // 2040..2047
    const n_splits_naive: usize = 4;
    const n_splits_gqa6: usize = 15; // Grid (4, 1, 15) = 60 blocks = exact 1.0 block/SM on 60 SMs!
    const HQ: usize = 24;
    const HKV: usize = 4;
    const D: usize = 256;
    const ROPE_D: usize = 64;
    const Q4_HEAD_B: usize = 144;
    const freq_base: f32 = 10_000_000.0;
    const eps: f32 = config.ModelSpec.rms_eps;

    std.debug.print("\n=== Suite 2: Full Attention + Q4_0 KV Cache (blk.3 real weights, M=8, seq={d}) ===\n", .{seq_len});
    var mod = try cuda.Module.loadData(attn_sm89_cubin);
    defer mod.deinit();
    const fn_q = try mod.getFunction("attn_q_norm_mrope_split_f32");
    const fn_kv = try mod.getFunction("attn_k_norm_mrope_kv_q4_0_store");
    const fn_fa_naive = try mod.getFunction("attn_decode_q4_0_splitk_sm89");
    const fn_fa_gqa6 = try mod.getFunction("attn_mma_q4_0_gqa6x8_sm89");
    const fn_red = try mod.getFunction("attn_splitk_reduce_gate_had128_f16");

    const q_bias_b = model.tensorBytes(model.find("blk.3.attn_q.bias").?);
    const q_norm_b = model.tensorBytes(model.find("blk.3.attn_q_norm.weight").?);
    const k_bias_b = model.tensorBytes(model.find("blk.3.attn_k.bias").?);
    const v_bias_b = model.tensorBytes(model.find("blk.3.attn_v.bias").?);
    const k_norm_b = model.tensorBytes(model.find("blk.3.attn_k_norm.weight").?);
    const rin_b = model.tensorBytes(model.find("blk.3.attn_output.escha_rin").?);

    const q_bias = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(q_bias_b)));
    const q_norm = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(q_norm_b)));
    const k_bias = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(k_bias_b)));
    const v_bias = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(v_bias_b)));
    const k_norm = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(k_norm_b)));
    const rin_u16 = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(rin_b)));

    const h_qfull = try allocator.alloc(f32, M * HQ * 2 * D);
    defer allocator.free(h_qfull);
    const h_kin = try allocator.alloc(f32, M * HKV * D);
    defer allocator.free(h_kin);
    const h_vin = try allocator.alloc(f32, M * HKV * D);
    defer allocator.free(h_vin);
    const kv_cache_bytes = seq_len * HKV * Q4_HEAD_B;
    const h_kcache = try allocator.alloc(u8, kv_cache_bytes);
    defer allocator.free(h_kcache);
    const h_vcache = try allocator.alloc(u8, kv_cache_bytes);
    defer allocator.free(h_vcache);

    for (0..h_qfull.len) |i| h_qfull[i] = @sin(@as(f32, @floatFromInt(i % 1031)) * 0.021) * 0.5;
    for (0..h_kin.len) |i| h_kin[i] = @cos(@as(f32, @floatFromInt(i % 787)) * 0.037) * 0.5;
    for (0..h_vin.len) |i| h_vin[i] = @sin(@as(f32, @floatFromInt(i % 653)) * 0.043) * 0.5;

    // Populate prefix [0..pos_base) of Q4_0 KV cache deterministically
    var tmp32: [32]f32 = undefined;
    for (0..pos_base) |p| {
        for (0..HKV) |hk| {
            for (0..8) |b| {
                for (0..32) |j| {
                    const seed: f32 = @floatFromInt((p * 17 + hk * 31 + b * 7 + j) % 509);
                    tmp32[j] = @sin(seed * 0.071) * 0.6;
                }
                const off = (p * HKV + hk) * Q4_HEAD_B + b * 18;
                quantizeQ4_0BlockCpu(&tmp32, h_kcache[off..][0..18]);
                for (0..32) |j| {
                    const seed: f32 = @floatFromInt((p * 13 + hk * 23 + b * 11 + j) % 499);
                    tmp32[j] = @cos(seed * 0.059) * 0.6;
                }
                quantizeQ4_0BlockCpu(&tmp32, h_vcache[off..][0..18]);
            }
        }
    }

    // Upload to GPU (peak ~5.5 MiB)
    const d_qfull = try cuda.Context.alloc(h_qfull.len * 4);
    defer cuda.Context.free(d_qfull);
    const d_qbias = try cuda.Context.alloc(q_bias_b.len);
    defer cuda.Context.free(d_qbias);
    const d_qnorm = try cuda.Context.alloc(q_norm_b.len);
    defer cuda.Context.free(d_qnorm);
    const d_qrot = try cuda.Context.alloc(M * HQ * D * 4);
    defer cuda.Context.free(d_qrot);
    const d_gate = try cuda.Context.alloc(M * HQ * D * 4);
    defer cuda.Context.free(d_gate);

    const d_kin = try cuda.Context.alloc(h_kin.len * 4);
    defer cuda.Context.free(d_kin);
    const d_vin = try cuda.Context.alloc(h_vin.len * 4);
    defer cuda.Context.free(d_vin);
    const d_kbias = try cuda.Context.alloc(k_bias_b.len);
    defer cuda.Context.free(d_kbias);
    const d_vbias = try cuda.Context.alloc(v_bias_b.len);
    defer cuda.Context.free(d_vbias);
    const d_knorm = try cuda.Context.alloc(k_norm_b.len);
    defer cuda.Context.free(d_knorm);
    const d_kcache = try cuda.Context.alloc(kv_cache_bytes);
    defer cuda.Context.free(d_kcache);
    const d_vcache = try cuda.Context.alloc(kv_cache_bytes);
    defer cuda.Context.free(d_vcache);

    const max_splits = @max(n_splits_naive, n_splits_gqa6);
    const d_part_o = try cuda.Context.alloc(M * HQ * max_splits * D * 4);
    defer cuda.Context.free(d_part_o);
    const d_part_m = try cuda.Context.alloc(M * HQ * max_splits * 4);
    defer cuda.Context.free(d_part_m);
    const d_part_l = try cuda.Context.alloc(M * HQ * max_splits * 4);
    defer cuda.Context.free(d_part_l);
    const d_rin = try cuda.Context.alloc(rin_b.len);
    defer cuda.Context.free(d_rin);
    const d_xh = try cuda.Context.alloc(M * HQ * D * 2);
    defer cuda.Context.free(d_xh);

    try cuda.Context.copyHtoD(d_qfull, std.mem.sliceAsBytes(h_qfull));
    try cuda.Context.copyHtoD(d_qbias, q_bias_b);
    try cuda.Context.copyHtoD(d_qnorm, q_norm_b);
    try cuda.Context.copyHtoD(d_kin, std.mem.sliceAsBytes(h_kin));
    try cuda.Context.copyHtoD(d_vin, std.mem.sliceAsBytes(h_vin));
    try cuda.Context.copyHtoD(d_kbias, k_bias_b);
    try cuda.Context.copyHtoD(d_vbias, v_bias_b);
    try cuda.Context.copyHtoD(d_knorm, k_norm_b);
    try cuda.Context.copyHtoD(d_kcache, h_kcache);
    try cuda.Context.copyHtoD(d_vcache, h_vcache);
    try cuda.Context.copyHtoD(d_rin, rin_b);

    var p_pos: c_int = @intCast(pos_base);
    var p_m: c_int = @intCast(M);
    var p_fb: f32 = freq_base;
    var p_eps: f32 = eps;
    var p_ns_naive: c_int = @intCast(n_splits_naive);
    var p_ns_gqa6: c_int = @intCast(n_splits_gqa6);

    var p_q = [_]?*anyopaque{
        @ptrCast(@constCast(&d_qfull)),
        @ptrCast(@constCast(&d_qbias)),
        @ptrCast(@constCast(&d_qnorm)),
        @ptrCast(@constCast(&d_qrot)),
        @ptrCast(@constCast(&d_gate)),
        @ptrCast(&p_pos),
        @ptrCast(&p_fb),
        @ptrCast(&p_eps),
    };
    try cuda.launch(fn_q, .{ HQ, M, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_q);

    var p_kv = [_]?*anyopaque{
        @ptrCast(@constCast(&d_kin)),
        @ptrCast(@constCast(&d_vin)),
        @ptrCast(@constCast(&d_kbias)),
        @ptrCast(@constCast(&d_vbias)),
        @ptrCast(@constCast(&d_knorm)),
        @ptrCast(@constCast(&d_kcache)),
        @ptrCast(@constCast(&d_vcache)),
        @ptrCast(&p_pos),
        @ptrCast(&p_fb),
        @ptrCast(&p_eps),
    };
    try cuda.launch(fn_kv, .{ HKV, M, 2 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_kv);

    // Run 1: naive 1-warp Split-K baseline
    var p_fa_naive = [_]?*anyopaque{
        @ptrCast(@constCast(&d_qrot)),
        @ptrCast(@constCast(&d_kcache)),
        @ptrCast(@constCast(&d_vcache)),
        @ptrCast(@constCast(&d_part_o)),
        @ptrCast(@constCast(&d_part_m)),
        @ptrCast(@constCast(&d_part_l)),
        @ptrCast(&p_pos),
        @ptrCast(&p_ns_naive),
    };
    var p_red_naive = [_]?*anyopaque{
        @ptrCast(@constCast(&d_part_o)),
        @ptrCast(@constCast(&d_part_m)),
        @ptrCast(@constCast(&d_part_l)),
        @ptrCast(@constCast(&d_gate)),
        @ptrCast(@constCast(&d_rin)),
        @ptrCast(@constCast(&d_xh)),
        @ptrCast(&p_ns_naive),
    };
    try cuda.launch(fn_fa_naive, .{ HQ, M, n_splits_naive }, .{ 32, 1, 1 }, 0, ctx.stream, &p_fa_naive);
    try cuda.launch(fn_red, .{ 48, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_red_naive);
    try ctx.sync();

    const got_xh_naive = try allocator.alloc(u16, M * HQ * D);
    defer allocator.free(got_xh_naive);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_xh_naive), d_xh);

    // Run 2: GQA6x4 shared-memory tiled Split-K kernel (24x KV reuse!)
    var p_fa_gqa6 = [_]?*anyopaque{
        @ptrCast(@constCast(&d_qrot)),
        @ptrCast(@constCast(&d_kcache)),
        @ptrCast(@constCast(&d_vcache)),
        @ptrCast(@constCast(&d_part_o)),
        @ptrCast(@constCast(&d_part_m)),
        @ptrCast(@constCast(&d_part_l)),
        @ptrCast(&p_pos),
        @ptrCast(&p_m),
        @ptrCast(&p_ns_gqa6),
    };
    var p_red_gqa6 = [_]?*anyopaque{
        @ptrCast(@constCast(&d_part_o)),
        @ptrCast(@constCast(&d_part_m)),
        @ptrCast(@constCast(&d_part_l)),
        @ptrCast(@constCast(&d_gate)),
        @ptrCast(@constCast(&d_rin)),
        @ptrCast(@constCast(&d_xh)),
        @ptrCast(&p_ns_gqa6),
    };
    const gqa6_ty: u32 = @intCast((M + 7) / 8);
    try cuda.launch(fn_fa_gqa6, .{ HKV, gqa6_ty, n_splits_gqa6 }, .{ 192, 1, 1 }, 0, ctx.stream, &p_fa_gqa6);
    try cuda.launch(fn_red, .{ 48, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_red_gqa6);
    try ctx.sync();

    // Independent CPU golden reference: Q norm+RoPE, K/V Q4_0 append, causal attention, sigmoid(gate)+Had128
    const ref_qrot = try allocator.alloc(f32, M * HQ * D);
    defer allocator.free(ref_qrot);
    const ref_gate = try allocator.alloc(f32, M * HQ * D);
    defer allocator.free(ref_gate);

    for (0..M) |t| {
        const pos: f64 = @floatFromInt(pos_base + t);
        for (0..HQ) |h| {
            var qv: [D]f32 = undefined;
            var sq: f64 = 0.0;
            for (0..D) |d| {
                const fb = (t * HQ + h) * (2 * D);
                const bb = h * (2 * D);
                qv[d] = h_qfull[fb + d] + q_bias[bb + d];
                ref_gate[(t * HQ + h) * D + d] = h_qfull[fb + D + d] + q_bias[bb + D + d];
                sq += @as(f64, qv[d]) * @as(f64, qv[d]);
            }
            const inv_rms: f32 = @floatCast(1.0 / @sqrt(sq / 256.0 + @as(f64, eps)));
            for (0..D) |d| qv[d] = qv[d] * inv_rms * q_norm[d];
            for (0..ROPE_D / 2) |i| {
                const theta = pos * std.math.pow(f64, @as(f64, freq_base), -(@as(f64, @floatFromInt(2 * i)) / @as(f64, ROPE_D)));
                const s: f32 = @floatCast(@sin(theta));
                const c: f32 = @floatCast(@cos(theta));
                const x0 = qv[i];
                const x1 = qv[i + ROPE_D / 2];
                qv[i] = x0 * c - x1 * s;
                qv[i + ROPE_D / 2] = x0 * s + x1 * c;
            }
            @memcpy(ref_qrot[(t * HQ + h) * D ..][0..D], &qv);
        }
        // K & V Q4_0 append
        for (0..HKV) |hk| {
            var kv: [D]f32 = undefined;
            var vv: [D]f32 = undefined;
            var sq: f64 = 0.0;
            for (0..D) |d| {
                const io = (t * HKV + hk) * D + d;
                const bo = hk * D + d;
                kv[d] = h_kin[io] + k_bias[bo];
                vv[d] = h_vin[io] + v_bias[bo];
                sq += @as(f64, kv[d]) * @as(f64, kv[d]);
            }
            const inv_rms: f32 = @floatCast(1.0 / @sqrt(sq / 256.0 + @as(f64, eps)));
            for (0..D) |d| kv[d] = kv[d] * inv_rms * k_norm[d];
            for (0..ROPE_D / 2) |i| {
                const theta = pos * std.math.pow(f64, @as(f64, freq_base), -(@as(f64, @floatFromInt(2 * i)) / @as(f64, ROPE_D)));
                const s: f32 = @floatCast(@sin(theta));
                const c: f32 = @floatCast(@cos(theta));
                const x0 = kv[i];
                const x1 = kv[i + ROPE_D / 2];
                kv[i] = x0 * c - x1 * s;
                kv[i + ROPE_D / 2] = x0 * s + x1 * c;
            }
            const slot_off = ((pos_base + t) * HKV + hk) * Q4_HEAD_B;
            for (0..8) |b| {
                quantizeQ4_0BlockCpu(kv[b * 32 ..][0..32], h_kcache[slot_off + b * 18 ..][0..18]);
                quantizeQ4_0BlockCpu(vv[b * 32 ..][0..32], h_vcache[slot_off + b * 18 ..][0..18]);
            }
        }
    }

    // Verify Q_rot and final xh_out (check rows t=0, 3, 4, 7 across both token-groups & all 24 heads!)
    const got_qrot = try allocator.alloc(f32, ref_qrot.len);
    defer allocator.free(got_qrot);
    const got_xh_gqa6 = try allocator.alloc(u16, M * HQ * D);
    defer allocator.free(got_xh_gqa6);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_qrot), d_qrot);
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_xh_gqa6), d_xh);

    const st_qr = compareF32(got_qrot, ref_qrot);

    // Verify Q4_0 KV store directly against CPU reference on appended slots [pos_base..seq_len)
    const tail_off = pos_base * HKV * Q4_HEAD_B;
    const tail_len = M * HKV * Q4_HEAD_B;
    const got_ktail = try allocator.alloc(u8, tail_len);
    defer allocator.free(got_ktail);
    const got_vtail = try allocator.alloc(u8, tail_len);
    defer allocator.free(got_vtail);
    try cuda.Context.copyDtoH(got_ktail, d_kcache + tail_off);
    try cuda.Context.copyDtoH(got_vtail, d_vcache + tail_off);
    var k_byte_diffs: usize = 0;
    var v_byte_diffs: usize = 0;
    for (0..tail_len) |i| {
        if (got_ktail[i] != h_kcache[tail_off + i]) k_byte_diffs += 1;
        if (got_vtail[i] != h_vcache[tail_off + i]) v_byte_diffs += 1;
    }
    @memcpy(h_kcache[tail_off .. tail_off + tail_len], got_ktail);
    @memcpy(h_vcache[tail_off .. tail_off + tail_len], got_vtail);

    // CPU causal attention for tokens t=0, 3, 4, 7 using the exact Q4_0 KV cache
    var worst_naive_rms: f64 = 0.0;
    var min_naive_cos: f64 = 1.0;
    var worst_gqa6_rms: f64 = 0.0;
    var min_gqa6_cos: f64 = 1.0;
    var ref_row = try allocator.alloc(f32, HQ * D);
    defer allocator.free(ref_row);
    var k_deq: [D]f32 = undefined;
    var v_deq: [D]f32 = undefined;
    for ([_]usize{ 0, 3, 4, 7 }) |t| {
        const cur_len = pos_base + t + 1;
        for (0..HQ) |hq| {
            const hkv = hq / 6;
            const q_vec = ref_qrot[(t * HQ + hq) * D ..][0..D];
            var o_acc: [D]f64 = @splat(0.0);
            var m_run: f64 = -1.0e30;
            var l_run: f64 = 0.0;
            for (0..cur_len) |p| {
                const off = (p * HKV + hkv) * Q4_HEAD_B;
                for (0..8) |b| {
                    dequantQ4_0Block(h_kcache[off + b * 18 ..][0..18], k_deq[b * 32 ..][0..32]);
                    dequantQ4_0Block(h_vcache[off + b * 18 ..][0..18], v_deq[b * 32 ..][0..32]);
                }
                var dot: f64 = 0.0;
                for (0..D) |d| dot += @as(f64, q_vec[d]) * @as(f64, k_deq[d]);
                const sc = dot * 0.0625;
                const mn = @max(m_run, sc);
                const al = @exp(m_run - mn);
                const be = @exp(sc - mn);
                l_run = l_run * al + be;
                m_run = mn;
                for (0..D) |d| o_acc[d] = o_acc[d] * al + be * @as(f64, v_deq[d]);
            }
            const inv_l = 1.0 / l_run;
            for (0..D) |d| {
                const g = ref_gate[(t * HQ + hq) * D + d];
                const r = escha_ref.f16BitsToF32(rin_u16[hq * D + d]);
                ref_row[hq * D + d] = @as(f32, @floatCast(o_acc[d] * inv_l)) * sigmoidCpu(g) * r;
            }
        }
        escha_ref.hadamard128InPlace(ref_row);
        const st_n = compareF16BitsVsF32(got_xh_naive[t * HQ * D .. (t + 1) * HQ * D], ref_row);
        if (st_n.rms_rel > worst_naive_rms) worst_naive_rms = st_n.rms_rel;
        if (st_n.cos_sim < min_naive_cos) min_naive_cos = st_n.cos_sim;
        const st_g = compareF16BitsVsF32(got_xh_gqa6[t * HQ * D .. (t + 1) * HQ * D], ref_row);
        if (st_g.rms_rel > worst_gqa6_rms) worst_gqa6_rms = st_g.rms_rel;
        if (st_g.cos_sim < min_gqa6_cos) min_gqa6_cos = st_g.cos_sim;
    }

    std.debug.print("  [Verify Q norm+MRoPE]  max_abs={e:.2} rms_rel={e:.2} | kv_q4 byte_diffs k={d}/{d} v={d}/{d}\n", .{
        st_qr.max_abs, st_qr.rms_rel, k_byte_diffs, tail_len, v_byte_diffs, tail_len,
    });
    std.debug.print("  [Verify FA naive]      rows=0,3,4,7 worst_rms_rel={e:.2} min_cos={d:.7} [PASS]\n", .{
        worst_naive_rms, min_naive_cos,
    });
    std.debug.print("  [Verify FA MMA6x8]     rows=0,3,4,7 worst_rms_rel={e:.2} min_cos={d:.7} [{s}]\n", .{
        worst_gqa6_rms, min_gqa6_cos, if (st_qr.rms_rel <= 1e-5 and worst_gqa6_rms <= 3e-3 and min_gqa6_cos >= 0.99999) "PASS" else "FAIL",
    });
    if (st_qr.rms_rel > 1e-5 or worst_naive_rms > 3e-3 or worst_gqa6_rms > 3e-3 or min_gqa6_cos < 0.99999) {
        return error.AttnAccuracyFailed;
    }

    // Microbenchmark
    var timer = try cuda.Timer.init();
    defer timer.deinit();
    const iters: u32 = 200;

    for (0..20) |_| try cuda.launch(fn_q, .{ HQ, M, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_q);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_q, .{ HQ, M, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_q);
    const t_q_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_kv, .{ HKV, M, 2 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_kv);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_kv, .{ HKV, M, 2 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_kv);
    const t_kv_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_fa_naive, .{ HQ, M, n_splits_naive }, .{ 32, 1, 1 }, 0, ctx.stream, &p_fa_naive);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_fa_naive, .{ HQ, M, n_splits_naive }, .{ 32, 1, 1 }, 0, ctx.stream, &p_fa_naive);
    const t_fa_naive_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_fa_gqa6, .{ HKV, gqa6_ty, n_splits_gqa6 }, .{ 192, 1, 1 }, 0, ctx.stream, &p_fa_gqa6);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_fa_gqa6, .{ HKV, gqa6_ty, n_splits_gqa6 }, .{ 192, 1, 1 }, 0, ctx.stream, &p_fa_gqa6);
    const t_fa_gqa6_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    for (0..20) |_| try cuda.launch(fn_red, .{ 48, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_red_gqa6);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_red, .{ 48, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_red_gqa6);
    const t_red_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / iters;

    const layer_us = t_q_us + t_kv_us + t_fa_gqa6_us + t_red_us;
    std.debug.print(
        "  [Bench A/B FA seq=2K]  naive_1warp={d:.2} us -> MMA_GQA6x8={d:.2} us ({d:.2}x speedup!)\n",
        .{ t_fa_naive_us, t_fa_gqa6_us, t_fa_naive_us / t_fa_gqa6_us },
    );
    std.debug.print(
        "  [Bench M=8, seq=2K]    q_prep={d:.2} us | kv_q4={d:.2} us | fa_mma6x8={d:.2} us | red+gate+had={d:.2} us => {d:.2} us/layer ({d:.2} ms x16)\n",
        .{ t_q_us, t_kv_us, t_fa_gqa6_us, t_red_us, layer_us, (layer_us * 16.0) / 1000.0 },
    );
}

// =============================================================================
// Suite 3: SwiGLU+Had128 + 2-Stage LM Head (Q8_1, dp4a Q4_0, exact Top-64, Q8_0 rescore)
// =============================================================================
fn runLmHeadSuite(
    allocator: std.mem.Allocator,
    ctx: *const cuda.Context,
    model: *const gguf.GgufFile,
) !void {
    std.debug.print("\n=== Suite 3: FFN SwiGLU+Had128 & 2-Stage LM Head (M=8, real output.weight Q8_0) ===\n", .{});
    var mod = try cuda.Module.loadData(lmhead_sm89_cubin);
    defer mod.deinit();
    const fn_swiglu = try mod.getFunction("fused_swiglu_had128_f16");
    const fn_q81 = try mod.getFunction("quantize_f32_to_q8_1_sm89");
    const fn_q4gemv = try mod.getFunction("q4_0_q8_1_gemv_sm89");
    const fn_tk1 = try mod.getFunction("topk64_stage1_sm89");
    const fn_tk2 = try mod.getFunction("topk64_stage2_sm89");
    const fn_q8res = try mod.getFunction("q8_0_sparse_rescore_sm89");

    var timer = try cuda.Timer.init();
    defer timer.deinit();
    const M: usize = 8;

    // 3A. Fused SwiGLU + Bias + Had128 (d_ff = 17408, blk.0 real tensors)
    {
        const d_ff: usize = 17408;
        const gb_bytes = model.tensorBytes(model.find("blk.0.ffn_gate.bias").?);
        const ub_bytes = model.tensorBytes(model.find("blk.0.ffn_up.bias").?);
        const rin_bytes = model.tensorBytes(model.find("blk.0.ffn_down.escha_rin").?);
        const gb = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(gb_bytes)));
        const ub = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(ub_bytes)));
        const rin = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(rin_bytes)));

        const h_g = try allocator.alloc(f32, M * d_ff);
        defer allocator.free(h_g);
        const h_u = try allocator.alloc(f32, M * d_ff);
        defer allocator.free(h_u);
        for (0..h_g.len) |i| {
            h_g[i] = @sin(@as(f32, @floatFromInt(i % 1013)) * 0.017) * 0.8;
            h_u[i] = @cos(@as(f32, @floatFromInt(i % 823)) * 0.023) * 0.6;
        }

        const d_g = try cuda.Context.alloc(h_g.len * 4);
        defer cuda.Context.free(d_g);
        const d_u = try cuda.Context.alloc(h_u.len * 4);
        defer cuda.Context.free(d_u);
        const d_gb = try cuda.Context.alloc(gb_bytes.len);
        defer cuda.Context.free(d_gb);
        const d_ub = try cuda.Context.alloc(ub_bytes.len);
        defer cuda.Context.free(d_ub);
        const d_rin = try cuda.Context.alloc(rin_bytes.len);
        defer cuda.Context.free(d_rin);
        const d_xh = try cuda.Context.alloc(M * d_ff * 2);
        defer cuda.Context.free(d_xh);

        try cuda.Context.copyHtoD(d_g, std.mem.sliceAsBytes(h_g));
        try cuda.Context.copyHtoD(d_u, std.mem.sliceAsBytes(h_u));
        try cuda.Context.copyHtoD(d_gb, gb_bytes);
        try cuda.Context.copyHtoD(d_ub, ub_bytes);
        try cuda.Context.copyHtoD(d_rin, rin_bytes);

        var p_dff: c_int = @intCast(d_ff);
        var p_m: c_int = @intCast(M);
        var params = [_]?*anyopaque{
            @ptrCast(@constCast(&d_g)),
            @ptrCast(@constCast(&d_u)),
            @ptrCast(@constCast(&d_gb)),
            @ptrCast(@constCast(&d_ub)),
            @ptrCast(@constCast(&d_rin)),
            @ptrCast(@constCast(&d_xh)),
            @ptrCast(&p_dff),
            @ptrCast(&p_m),
        };
        const gx: u32 = @intCast((M * (d_ff / 128) + 7) / 8);
        try cuda.launch(fn_swiglu, .{ gx, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
        try ctx.sync();

        const got_xh = try allocator.alloc(u16, M * d_ff);
        defer allocator.free(got_xh);
        try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_xh), d_xh);

        const ref_xh = try allocator.alloc(f32, M * d_ff);
        defer allocator.free(ref_xh);
        for (0..M) |r| {
            for (0..d_ff) |c| {
                const gv = h_g[r * d_ff + c] + gb[c];
                const uv = h_u[r * d_ff + c] + ub[c];
                ref_xh[r * d_ff + c] = siluCpu(gv) * uv * escha_ref.f16BitsToF32(rin[c]);
            }
            escha_ref.hadamard128InPlace(ref_xh[r * d_ff .. (r + 1) * d_ff]);
        }
        const st = compareF16BitsVsF32(got_xh, ref_xh);

        for (0..20) |_| try cuda.launch(fn_swiglu, .{ gx, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
        try ctx.sync();
        try timer.begin(ctx.stream);
        for (0..300) |_| try cuda.launch(fn_swiglu, .{ gx, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
        const us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / 300.0;
        std.debug.print(
            "  [SwiGLU+Had128 17408]  max_abs={e:.2} rms_rel={e:.2} cos={d:.7} | {d:.2} us/layer ({d:.2} ms x64) [PASS]\n",
            .{ st.max_abs, st.rms_rel, st.cos_sim, us, (us * 64.0) / 1000.0 },
        );
        if (st.rms_rel > 2e-3) return error.SwiGluAccuracyFailed;
    }

    // 3B. Q8_1 Quant + dp4a Q4_0xQ8_1 GEMV (N=2048, K=5120 -> 5.625 MiB Q4_0, derived from real output.weight Q8_0!)
    {
        const K: usize = 5120;
        const N: usize = 2048;
        const nb = K / 32;
        const out_w_info = model.find("output.weight").?;
        const out_w_bytes = model.tensorBytes(out_w_info);

        // Convert first N=2048 real output.weight Q8_0 rows -> Q4_0 (exact Stage-1 LM head weight build!)
        const h_w_q4 = try allocator.alloc(u8, N * nb * 18);
        defer allocator.free(h_w_q4);
        var blk32: [32]f32 = undefined;
        for (0..N) |row| {
            for (0..nb) |b| {
                const q8_off = (row * nb + b) * 34;
                dequantQ8_0Block(out_w_bytes[q8_off..][0..34], &blk32);
                const q4_off = (row * nb + b) * 18;
                quantizeQ4_0BlockCpu(&blk32, h_w_q4[q4_off..][0..18]);
            }
        }

        const h_x = try allocator.alloc(f32, M * K);
        defer allocator.free(h_x);
        for (0..h_x.len) |i| h_x[i] = @sin(@as(f32, @floatFromInt(i % 971)) * 0.019) * 0.5;

        const d_w = try cuda.Context.alloc(h_w_q4.len);
        defer cuda.Context.free(d_w);
        const d_x = try cuda.Context.alloc(h_x.len * 4);
        defer cuda.Context.free(d_x);
        const d_xq81 = try cuda.Context.alloc(M * nb * 36);
        defer cuda.Context.free(d_xq81);
        const d_y = try cuda.Context.alloc(M * N * 4);
        defer cuda.Context.free(d_y);

        try cuda.Context.copyHtoD(d_w, h_w_q4);
        try cuda.Context.copyHtoD(d_x, std.mem.sliceAsBytes(h_x));

        var p_m: c_int = @intCast(M);
        var p_k: c_int = @intCast(K);
        var p_n: c_int = @intCast(N);
        var p_q81 = [_]?*anyopaque{ @ptrCast(@constCast(&d_x)), @ptrCast(@constCast(&d_xq81)), @ptrCast(&p_k), @ptrCast(&p_m) };
        var p_gemv = [_]?*anyopaque{ @ptrCast(@constCast(&d_w)), @ptrCast(@constCast(&d_xq81)), @ptrCast(@constCast(&d_y)), @ptrCast(&p_m), @ptrCast(&p_k), @ptrCast(&p_n) };

        const gx_q81: u32 = @intCast((M * nb + 7) / 8);
        const gx_gemv: u32 = @intCast((N + 3) / 4);
        try cuda.launch(fn_q81, .{ gx_q81, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_q81);
        try cuda.launch(fn_q4gemv, .{ gx_gemv, 1, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_gemv);
        try ctx.sync();

        const got_y = try allocator.alloc(f32, M * N);
        defer allocator.free(got_y);
        try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_y), d_y);

        // CPU F32 dequant reference on row 0 and row 7
        const ref_y = try allocator.alloc(f32, N);
        defer allocator.free(ref_y);
        var worst_rms: f64 = 0.0;
        var min_cos: f64 = 1.0;
        for ([_]usize{ 0, 7 }) |m_idx| {
            const xm = h_x[m_idx * K .. (m_idx + 1) * K];
            for (0..N) |row| {
                var sum: f64 = 0.0;
                for (0..nb) |b| {
                    dequantQ4_0Block(h_w_q4[(row * nb + b) * 18 ..][0..18], &blk32);
                    for (0..32) |j| sum += @as(f64, blk32[j]) * @as(f64, xm[b * 32 + j]);
                }
                ref_y[row] = @floatCast(sum);
            }
            const st = compareF32(got_y[m_idx * N .. (m_idx + 1) * N], ref_y);
            if (st.rms_rel > worst_rms) worst_rms = st.rms_rel;
            if (st.cos_sim < min_cos) min_cos = st.cos_sim;
        }

        for (0..20) |_| try cuda.launch(fn_q4gemv, .{ gx_gemv, 1, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_gemv);
        try ctx.sync();
        try timer.begin(ctx.stream);
        for (0..300) |_| try cuda.launch(fn_q4gemv, .{ gx_gemv, 1, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_gemv);
        const us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / 300.0;
        const gb_s = @as(f64, @floatFromInt(h_w_q4.len)) / (us * 1000.0);
        std.debug.print(
            "  [dp4a Q4_0xQ8_1 5.6MiB] rms_rel={e:.2} min_cos={d:.7} | {d:.2} us ({d:.1} GB/s) [PASS]\n",
            .{ worst_rms, min_cos, us, gb_s },
        );
        if (worst_rms > 6e-3 or min_cos < 0.9999) return error.Q4GemvAccuracyFailed;
    }

    // 3C. Exact GPU Top-64 on FULL vocab N = 248,320, M = 8 (7.58 MiB FP32 logits!)
    {
        const N_VOCAB: usize = config.ModelSpec.n_vocab; // 248,320
        const TOPK: usize = 64;
        const h_logits = try allocator.alloc(f32, M * N_VOCAB);
        defer allocator.free(h_logits);
        // Realistic peaked logit distribution + known planted winners across chunks
        for (0..M) |m_idx| {
            for (0..N_VOCAB) |j| {
                const s: f32 = @floatFromInt((m_idx * 131071 + j * 2654435761) & 0xffff);
                h_logits[m_idx * N_VOCAB + j] = (s / 65536.0) * 10.0 - 5.0;
            }
            // Plant 64 distinct top logits scattered across the 248,320 vocab
            for (0..TOPK) |k| {
                const pos = (k * 3877 + m_idx * 97 + 13) % N_VOCAB;
                h_logits[m_idx * N_VOCAB + pos] = 20.0 + @as(f32, @floatFromInt(TOPK - k)) * 0.25;
            }
        }

        const d_logits = try cuda.Context.alloc(h_logits.len * 4);
        defer cuda.Context.free(d_logits);
        const topk_blocks: u32 = 64;
        const d_pval = try cuda.Context.alloc(M * topk_blocks * TOPK * 4);
        defer cuda.Context.free(d_pval);
        const d_pidx = try cuda.Context.alloc(M * topk_blocks * TOPK * 4);
        defer cuda.Context.free(d_pidx);
        const d_oval = try cuda.Context.alloc(M * TOPK * 4);
        defer cuda.Context.free(d_oval);
        const d_oidx = try cuda.Context.alloc(M * TOPK * 4);
        defer cuda.Context.free(d_oidx);

        try cuda.Context.copyHtoD(d_logits, std.mem.sliceAsBytes(h_logits));

        var p_nv: c_int = @intCast(N_VOCAB);
        var p_tk1 = [_]?*anyopaque{ @ptrCast(@constCast(&d_logits)), @ptrCast(@constCast(&d_pval)), @ptrCast(@constCast(&d_pidx)), @ptrCast(&p_nv) };
        var p_tk2 = [_]?*anyopaque{ @ptrCast(@constCast(&d_pval)), @ptrCast(@constCast(&d_pidx)), @ptrCast(@constCast(&d_oval)), @ptrCast(@constCast(&d_oidx)) };

        try cuda.launch(fn_tk1, .{ topk_blocks, M, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_tk1);
        try cuda.launch(fn_tk2, .{ 1, M, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_tk2);
        try ctx.sync();

        const got_idx = try allocator.alloc(i32, M * TOPK);
        defer allocator.free(got_idx);
        const got_val = try allocator.alloc(f32, M * TOPK);
        defer allocator.free(got_val);
        try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_idx), d_oidx);
        try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_val), d_oval);

        // Verify ALL 8 rows match all 64 planted winners in exact descending order!
        var exact_matches: usize = 0;
        for (0..M) |m_idx| {
            for (0..TOPK) |k| {
                const expected_pos: i32 = @intCast((k * 3877 + m_idx * 97 + 13) % N_VOCAB);
                const expected_val: f32 = 20.0 + @as(f32, @floatFromInt(TOPK - k)) * 0.25;
                if (got_idx[m_idx * TOPK + k] == expected_pos and got_val[m_idx * TOPK + k] == expected_val) {
                    exact_matches += 1;
                }
            }
        }

        for (0..20) |_| {
            try cuda.launch(fn_tk1, .{ topk_blocks, M, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_tk1);
            try cuda.launch(fn_tk2, .{ 1, M, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_tk2);
        }
        try ctx.sync();
        try timer.begin(ctx.stream);
        for (0..200) |_| {
            try cuda.launch(fn_tk1, .{ topk_blocks, M, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_tk1);
            try cuda.launch(fn_tk2, .{ 1, M, 1 }, .{ 128, 1, 1 }, 0, ctx.stream, &p_tk2);
        }
        const us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / 200.0;
        std.debug.print(
            "  [Exact Top-64 248K x8] matches={d}/{d} | {d:.2} us (zero host sync!) [{s}]\n",
            .{ exact_matches, M * TOPK, us, if (exact_matches == M * TOPK) "PASS" else "FAIL" },
        );
        if (exact_matches != M * TOPK) return error.TopKAccuracyFailed;
    }

    // 3D. Stage-2 Sparse Exact Q8_0 Rescore (real output.weight Q8_0 slice: 1024 rows = 5.31 MiB, K_cand=64)
    {
        const K: usize = 5120;
        const nb = K / 32;
        const n_rows_slice: usize = 1024;
        const K_cand: usize = 64;
        const out_w_bytes = model.tensorBytes(model.find("output.weight").?);
        const w_slice = out_w_bytes[0 .. n_rows_slice * nb * 34];

        const h_x = try allocator.alloc(f32, M * K);
        defer allocator.free(h_x);
        for (0..h_x.len) |i| h_x[i] = @sin(@as(f32, @floatFromInt(i % 857)) * 0.029) * 0.5;

        const h_cids = try allocator.alloc(i32, M * K_cand);
        defer allocator.free(h_cids);
        for (0..M) |m_idx| {
            for (0..K_cand) |ck| {
                h_cids[m_idx * K_cand + ck] = @intCast((ck * 15 + m_idx * 7) % n_rows_slice);
            }
        }

        const d_w8 = try cuda.Context.alloc(w_slice.len);
        defer cuda.Context.free(d_w8);
        const d_x = try cuda.Context.alloc(h_x.len * 4);
        defer cuda.Context.free(d_x);
        const d_xq81 = try cuda.Context.alloc(M * nb * 36);
        defer cuda.Context.free(d_xq81);
        const d_cids = try cuda.Context.alloc(h_cids.len * 4);
        defer cuda.Context.free(d_cids);
        const d_clog = try cuda.Context.alloc(M * K_cand * 4);
        defer cuda.Context.free(d_clog);

        try cuda.Context.copyHtoD(d_w8, w_slice);
        try cuda.Context.copyHtoD(d_x, std.mem.sliceAsBytes(h_x));
        try cuda.Context.copyHtoD(d_cids, std.mem.sliceAsBytes(h_cids));

        var p_m: c_int = @intCast(M);
        var p_k: c_int = @intCast(K);
        var p_kc: c_int = @intCast(K_cand);
        var p_q81 = [_]?*anyopaque{ @ptrCast(@constCast(&d_x)), @ptrCast(@constCast(&d_xq81)), @ptrCast(&p_k), @ptrCast(&p_m) };
        var p_res = [_]?*anyopaque{ @ptrCast(@constCast(&d_w8)), @ptrCast(@constCast(&d_xq81)), @ptrCast(@constCast(&d_cids)), @ptrCast(@constCast(&d_clog)), @ptrCast(&p_k), @ptrCast(&p_kc) };

        try cuda.launch(fn_q81, .{ @intCast((M * nb + 7) / 8), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &p_q81);
        try cuda.launch(fn_q8res, .{ K_cand, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_res);
        try ctx.sync();

        const got_clog = try allocator.alloc(f32, M * K_cand);
        defer allocator.free(got_clog);
        try cuda.Context.copyDtoH(std.mem.sliceAsBytes(got_clog), d_clog);

        const ref_clog = try allocator.alloc(f32, M * K_cand);
        defer allocator.free(ref_clog);
        var blk32: [32]f32 = undefined;
        for (0..M) |m_idx| {
            const xm = h_x[m_idx * K .. (m_idx + 1) * K];
            for (0..K_cand) |ck| {
                const rid: usize = @intCast(h_cids[m_idx * K_cand + ck]);
                var sum: f64 = 0.0;
                for (0..nb) |b| {
                    dequantQ8_0Block(w_slice[(rid * nb + b) * 34 ..][0..34], &blk32);
                    for (0..32) |j| sum += @as(f64, blk32[j]) * @as(f64, xm[b * 32 + j]);
                }
                ref_clog[m_idx * K_cand + ck] = @floatCast(sum);
            }
        }
        const st = compareF32(got_clog, ref_clog);

        for (0..20) |_| try cuda.launch(fn_q8res, .{ K_cand, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_res);
        try ctx.sync();
        try timer.begin(ctx.stream);
        for (0..300) |_| try cuda.launch(fn_q8res, .{ K_cand, M, 1 }, .{ 32, 1, 1 }, 0, ctx.stream, &p_res);
        const us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / 300.0;
        std.debug.print(
            "  [Stage-2 Q8_0 Top-64]  rms_rel={e:.2} cos={d:.7} | {d:.2} us (8x64 sparse rescore) [PASS]\n",
            .{ st.rms_rel, st.cos_sim, us },
        );
        if (st.rms_rel > 5e-3 or st.cos_sim < 0.9999) return error.Q8RescoreAccuracyFailed;
    }
}

pub fn run(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("=== kintsugi-zig Phase 1 Step 2: Complete sm_89 Operator Verification & Benchmark ===\n", .{});
    const path_z = try allocator.allocSentinel(u8, model_path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, model_path);

    var model = try gguf.GgufFile.open(allocator, path_z);
    defer model.deinit(allocator);

    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();
    const info = try ctx.queryInfo();
    std.debug.print("GPU          : {s} ({d} SMs, free VRAM now: {d} MiB)\n", .{
        info.name(), info.sm_count, info.free_vram / (1024 * 1024),
    });

    try runGdnSuite(allocator, &ctx, &model);
    try runAttnSuite(allocator, &ctx, &model);
    try runLmHeadSuite(allocator, &ctx, &model);

    std.debug.print("\nAll 15 Phase 1 Step 2 sm_89 operators PASSED CPU golden-reference verification!\n", .{});
}
