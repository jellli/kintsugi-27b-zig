//! Real-weight verification and RTX 4070 Ti (sm_89) microbenchmark for native
//! Escha ~2.5-bit Trellis GEMV kernels (`kernels/escha_sm89.cu`).
//!
//! Safety note: Allocates and frees one GGUF weight tensor at a time (<= 12 MiB)
//! so this benchmark can run cleanly even alongside a live production server.

const std = @import("std");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");
const escha_ref = @import("escha_ref.zig");

const escha_sm89_cubin = @embedFile("escha_sm89.cubin");

const KernelSet = struct {
    had_in: cuda.CUfunction,
    had_out: cuda.CUfunction,
    k2_m1_base: cuda.CUfunction,
    k2_m8_base: cuda.CUfunction,
    k3_m1_base: cuda.CUfunction,
    k3_m8_base: cuda.CUfunction,
    k2_m8_w4s8: cuda.CUfunction,
    k3_m8_w4s8: cuda.CUfunction,
    k2_m8_w4s4: cuda.CUfunction,
    k3_m8_w4s4: cuda.CUfunction,
    k2_m8_narrow_s16: cuda.CUfunction,
    k3_m8_narrow_s16: cuda.CUfunction,
    k2_m8_narrow_s8: cuda.CUfunction,
    k3_m8_narrow_s8: cuda.CUfunction,
    k2_m8_narrow_s4: cuda.CUfunction,
    k3_m8_narrow_s4: cuda.CUfunction,
    k3_m8_narrow_w1s16: cuda.CUfunction,
    k2_m8_fused_w2s4: cuda.CUfunction,
    k3_m8_fused_w2s4: cuda.CUfunction,
    k2_m8_narrow_s12: cuda.CUfunction,
    k3_m8_narrow_s12: cuda.CUfunction,

    fn load(mod: *const cuda.Module) !KernelSet {
        return .{
            .had_in = try mod.getFunction("escha_had_in_f32_f16"),
            .had_out = try mod.getFunction("escha_had_out_f32"),
            .k2_m1_base = try mod.getFunction("escha_gemv_k2_m1_base"),
            .k2_m8_base = try mod.getFunction("escha_gemv_k2_m8_base"),
            .k3_m1_base = try mod.getFunction("escha_gemv_k3_m1_base"),
            .k3_m8_base = try mod.getFunction("escha_gemv_k3_m8_base"),
            .k2_m8_w4s8 = try mod.getFunction("escha_gemv_k2_m8_sm89_w4s8"),
            .k3_m8_w4s8 = try mod.getFunction("escha_gemv_k3_m8_sm89_w4s8"),
            .k2_m8_w4s4 = try mod.getFunction("escha_gemv_k2_m8_sm89_w4s4"),
            .k3_m8_w4s4 = try mod.getFunction("escha_gemv_k3_m8_sm89_w4s4"),
            .k2_m8_narrow_s16 = try mod.getFunction("escha_gemv_k2_m8_narrow_s16"),
            .k3_m8_narrow_s16 = try mod.getFunction("escha_gemv_k3_m8_narrow_s16"),
            .k2_m8_narrow_s8 = try mod.getFunction("escha_gemv_k2_m8_narrow_s8"),
            .k3_m8_narrow_s8 = try mod.getFunction("escha_gemv_k3_m8_narrow_s8"),
            .k2_m8_narrow_s4 = try mod.getFunction("escha_gemv_k2_m8_narrow_s4"),
            .k3_m8_narrow_s4 = try mod.getFunction("escha_gemv_k3_m8_narrow_s4"),
            .k3_m8_narrow_w1s16 = try mod.getFunction("escha_gemv_k3_m8_narrow_w1s16"),
            .k2_m8_fused_w2s4 = try mod.getFunction("escha_gemv_k2_m8_fused_w2s4"),
            .k3_m8_fused_w2s4 = try mod.getFunction("escha_gemv_k3_m8_fused_w2s4"),
            .k2_m8_narrow_s12 = try mod.getFunction("escha_gemv_k2_m8_narrow_s12"),
            .k3_m8_narrow_s12 = try mod.getFunction("escha_gemv_k3_m8_narrow_s12"),
        };
    }
};

const AccuracyStats = struct {
    max_abs: f32,
    rms_rel: f64,
    cos_sim: f64,
};

fn compareVectors(got: []const f32, ref: []const f32) AccuracyStats {
    var max_abs: f32 = 0.0;
    var sum_err_sq: f64 = 0.0;
    var sum_ref_sq: f64 = 0.0;
    var sum_got_sq: f64 = 0.0;
    var dot: f64 = 0.0;
    for (got, ref) |g, r| {
        const d = @abs(g - r);
        if (d > max_abs) max_abs = d;
        const gf: f64 = g;
        const rf: f64 = r;
        const df = gf - rf;
        sum_err_sq += df * df;
        sum_ref_sq += rf * rf;
        sum_got_sq += gf * gf;
        dot += gf * rf;
    }
    const rms_rel = if (sum_ref_sq > 0.0) @sqrt(sum_err_sq / sum_ref_sq) else 0.0;
    const denom = @sqrt(sum_got_sq * sum_ref_sq);
    const cos_sim = if (denom > 0.0) dot / denom else 1.0;
    return .{ .max_abs = max_abs, .rms_rel = rms_rel, .cos_sim = cos_sim };
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0.0;
    for (a, b) |x, y| {
        const d = @abs(x - y);
        if (d > m) m = d;
    }
    return m;
}

fn launchHadIn(
    ctx: *const cuda.Context,
    fn_had_in: cuda.CUfunction,
    d_x: cuda.CUdeviceptr,
    d_rin: cuda.CUdeviceptr,
    d_xh: cuda.CUdeviceptr,
    ic: usize,
    n_rows: usize,
) !void {
    const nb = ic >> 7;
    const total_warps = n_rows * nb;
    const grid_x: u32 = @intCast((total_warps + 7) / 8);
    var p_x = d_x;
    var p_rin = d_rin;
    var p_xh = d_xh;
    var p_ic: c_int = @intCast(ic);
    var p_rows: c_int = @intCast(n_rows);
    var params = [_]?*anyopaque{
        @ptrCast(&p_x),
        @ptrCast(&p_rin),
        @ptrCast(&p_xh),
        @ptrCast(&p_ic),
        @ptrCast(&p_rows),
    };
    try cuda.launch(fn_had_in, .{ grid_x, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
}

fn launchHadOut(
    ctx: *const cuda.Context,
    fn_had_out: cuda.CUfunction,
    d_mid: cuda.CUdeviceptr,
    d_rout: cuda.CUdeviceptr,
    d_dst: cuda.CUdeviceptr,
    oc: usize,
    n_rows: usize,
) !void {
    const nb = oc >> 7;
    const total_warps = n_rows * nb;
    const grid_x: u32 = @intCast((total_warps + 7) / 8);
    var p_mid = d_mid;
    var p_rout = d_rout;
    var p_dst = d_dst;
    var p_oc: c_int = @intCast(oc);
    var p_rows: c_int = @intCast(n_rows);
    var params = [_]?*anyopaque{
        @ptrCast(&p_mid),
        @ptrCast(&p_rout),
        @ptrCast(&p_dst),
        @ptrCast(&p_oc),
        @ptrCast(&p_rows),
    };
    try cuda.launch(fn_had_out, .{ grid_x, 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
}

fn launchGemv(
    ctx: *const cuda.Context,
    func: cuda.CUfunction,
    grid_x: u32,
    block_x: u32,
    d_xh: cuda.CUdeviceptr,
    d_code: cuda.CUdeviceptr,
    d_rout: cuda.CUdeviceptr,
    d_out: cuda.CUdeviceptr,
    m: usize,
    ic: usize,
    oc: usize,
) !void {
    var p_a = d_xh;
    var p_b = d_code;
    var p_rout = d_rout;
    var p_c = d_out;
    var p_m: c_int = @intCast(m);
    var p_k: c_int = @intCast(ic);
    var p_n: c_int = @intCast(oc);
    var params = [_]?*anyopaque{
        @ptrCast(&p_a),
        @ptrCast(&p_b),
        @ptrCast(&p_rout),
        @ptrCast(&p_c),
        @ptrCast(&p_m),
        @ptrCast(&p_k),
        @ptrCast(&p_n),
    };
    try cuda.launch(func, .{ grid_x, 1, 1 }, .{ block_x, 1, 1 }, 0, ctx.stream, &params);
}

const BenchVariant = struct {
    label: []const u8,
    func: cuda.CUfunction,
    grid_div: usize,
    block_x: u32,
    unfused: bool,
};

fn testAndBenchTensor(
    allocator: std.mem.Allocator,
    ctx: *const cuda.Context,
    ks: *const KernelSet,
    model: *const gguf.GgufFile,
    dep_k2: []const i16,
    dep_k3: []const i16,
    prefix: []const u8,
    verify_all_rows: bool,
    run_m8_bench: bool,
) !void {
    var name_buf: [128]u8 = undefined;
    const code_name = try std.fmt.bufPrint(&name_buf, "{s}.escha_code", .{prefix});
    const code_info = model.find(code_name) orelse return error.TensorNotFound;
    var rin_buf: [128]u8 = undefined;
    const rin_name = try std.fmt.bufPrint(&rin_buf, "{s}.escha_rin", .{prefix});
    const rin_info = model.find(rin_name) orelse return error.TensorNotFound;
    var rout_buf: [128]u8 = undefined;
    const rout_name = try std.fmt.bufPrint(&rout_buf, "{s}.escha_rout", .{prefix});
    const rout_info = model.find(rout_name) orelse return error.TensorNotFound;

    const k_bits: usize = @intCast(code_info.dims[0] / 16);
    const oc: usize = @intCast(code_info.dims[1] * 16);
    const ic: usize = @intCast(code_info.dims[2] * 16);
    const code_bytes = model.tensorBytes(code_info);
    const rin_bytes = model.tensorBytes(rin_info);
    const rout_bytes = model.tensorBytes(rout_info);
    const rin_u16: []const u16 = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(rin_bytes)));
    const rout_u16: []const u16 = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(rout_bytes)));
    const dep = if (k_bits == 2) dep_k2 else dep_k3;

    const code_mib = @as(f64, @floatFromInt(code_bytes.len)) / (1024.0 * 1024.0);
    std.debug.print("\n--- Tensor: {s} (K={d}-bit, IC={d} -> OC={d}, code={d:.2} MiB) ---\n", .{
        prefix, k_bits, ic, oc, code_mib,
    });

    const max_m: usize = 8;
    const needed_vram = code_bytes.len + rin_bytes.len + rout_bytes.len +
        max_m * ic * (@sizeOf(f32) + @sizeOf(u16)) +
        max_m * oc * @sizeOf(f32) * 2;
    const gpu_now = try ctx.queryInfo();
    const safety_margin: usize = 12 * 1024 * 1024;
    if (gpu_now.free_vram < needed_vram + safety_margin) {
        std.debug.print(
            "  [SKIPPED] Needs {d:.2} MiB + 12 MiB margin, only {d:.2} MiB free VRAM (live server active)\n",
            .{
                @as(f64, @floatFromInt(needed_vram)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(gpu_now.free_vram)) / (1024.0 * 1024.0),
            },
        );
        return;
    }

    // Build deterministic non-trivial input X[8, IC]
    const h_x = try allocator.alloc(f32, max_m * ic);
    defer allocator.free(h_x);
    for (0..max_m) |r| {
        for (0..ic) |c| {
            const rf: f32 = @floatFromInt(r + 1);
            const cf: f32 = @floatFromInt(c);
            h_x[r * ic + c] = @sin(cf * 0.017 * rf + rf * 0.7) * 0.4 + @cos(cf * 0.0053 - rf * 0.3) * 0.2;
        }
    }

    // Upload tensor & working buffers (freed at end of scope via defer)
    const d_code = try cuda.Context.alloc(code_bytes.len);
    defer cuda.Context.free(d_code);
    const d_rin = try cuda.Context.alloc(rin_bytes.len);
    defer cuda.Context.free(d_rin);
    const d_rout = try cuda.Context.alloc(rout_bytes.len);
    defer cuda.Context.free(d_rout);
    const d_x = try cuda.Context.alloc(max_m * ic * @sizeOf(f32));
    defer cuda.Context.free(d_x);
    const d_xh = try cuda.Context.alloc(max_m * ic * @sizeOf(u16));
    defer cuda.Context.free(d_xh);
    const d_mid = try cuda.Context.alloc(max_m * oc * @sizeOf(f32));
    defer cuda.Context.free(d_mid);
    const d_y = try cuda.Context.alloc(max_m * oc * @sizeOf(f32));
    defer cuda.Context.free(d_y);

    try cuda.Context.copyHtoD(d_code, code_bytes);
    try cuda.Context.copyHtoD(d_rin, rin_bytes);
    try cuda.Context.copyHtoD(d_rout, rout_bytes);
    try cuda.Context.copyHtoD(d_x, std.mem.sliceAsBytes(h_x));

    const h_y_ref = try allocator.alloc(f32, oc);
    defer allocator.free(h_y_ref);
    const h_y_gpu = try allocator.alloc(f32, max_m * oc);
    defer allocator.free(h_y_gpu);
    const h_y_base = try allocator.alloc(f32, max_m * oc);
    defer allocator.free(h_y_base);

    // 1. Verify M=1 fast path against independent CPU ground truth (row 0)
    try launchHadIn(ctx, ks.had_in, d_x, d_rin, d_xh, ic, 1);
    const fn_m1 = if (k_bits == 2) ks.k2_m1_base else ks.k3_m1_base;
    try launchGemv(ctx, fn_m1, @intCast(oc / 128), 1024, d_xh, d_code, d_rout, d_y, 1, ic, oc);
    try ctx.sync();
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(h_y_gpu[0..oc]), d_y);

    try escha_ref.eschaMatVecCpuRef(allocator, code_bytes, rin_u16, rout_u16, dep, h_x[0..ic], h_y_ref, k_bits, ic, oc);
    const st_m1 = compareVectors(h_y_gpu[0..oc], h_y_ref);
    std.debug.print("  [Verify M=1 base vs CPU]   max_abs={e:.3}  rms_rel={e:.3}  cos_sim={d:.7} [{s}]\n", .{
        st_m1.max_abs, st_m1.rms_rel, st_m1.cos_sim, if (st_m1.cos_sim > 0.99999 and st_m1.rms_rel < 5.0e-3) "PASS" else "FAIL",
    });
    if (st_m1.cos_sim <= 0.99999 or st_m1.rms_rel >= 5.0e-3) return error.EschaAccuracyFailed;

    // 2. Verify M=8 path against independent CPU ground truth
    try launchHadIn(ctx, ks.had_in, d_x, d_rin, d_xh, ic, max_m);
    const fn_m8_base = if (k_bits == 2) ks.k2_m8_base else ks.k3_m8_base;
    try launchGemv(ctx, fn_m8_base, @intCast(oc / 128), 1024, d_xh, d_code, d_rout, d_y, max_m, ic, oc);
    try ctx.sync();
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(h_y_base), d_y);

    // Row 0 of M=8 vs CPU ref (also check row 0 M=8 vs M=1!)
    const m1_vs_m8_row0 = maxAbsDiff(h_y_gpu[0..oc], h_y_base[0..oc]);
    var worst_rms_rel: f64 = 0.0;
    var worst_cos_sim: f64 = 1.0;
    var worst_max_abs: f32 = 0.0;
    const rows_to_check: []const usize = if (verify_all_rows) &.{ 0, 1, 2, 3, 4, 5, 6, 7 } else &.{ 0, 7 };
    for (rows_to_check) |r| {
        if (r != 0) {
            try escha_ref.eschaMatVecCpuRef(
                allocator,
                code_bytes,
                rin_u16,
                rout_u16,
                dep,
                h_x[r * ic .. (r + 1) * ic],
                h_y_ref,
                k_bits,
                ic,
                oc,
            );
        }
        const st = compareVectors(h_y_base[r * oc .. (r + 1) * oc], h_y_ref);
        if (st.rms_rel > worst_rms_rel) worst_rms_rel = st.rms_rel;
        if (st.cos_sim < worst_cos_sim) worst_cos_sim = st.cos_sim;
        if (st.max_abs > worst_max_abs) worst_max_abs = st.max_abs;
    }
    std.debug.print("  [Verify M=8 base vs CPU]   rows={d} worst_abs={e:.3}  worst_rms_rel={e:.3}  min_cos={d:.7}  (M1-M8 r0={e:.2}) [{s}]\n", .{
        rows_to_check.len,
        worst_max_abs,
        worst_rms_rel,
        worst_cos_sim,
        m1_vs_m8_row0,
        if (worst_cos_sim > 0.99999 and worst_rms_rel < 5.0e-3) "PASS" else "FAIL",
    });
    if (worst_cos_sim <= 0.99999 or worst_rms_rel >= 5.0e-3) return error.EschaAccuracyFailed;

    if (!run_m8_bench) return;

    // 3. Benchmark M=8 geometries on RTX 4070 Ti (same pre-rotated d_xh)
    var variants_buf: [9]BenchVariant = undefined;
    var n_variants: usize = 0;
    variants_buf[n_variants] = .{
        .label = "base      WNT=2 S=8  1024t (OC/128 fused)",
        .func = if (k_bits == 2) ks.k2_m8_base else ks.k3_m8_base,
        .grid_div = 128,
        .block_x = 1024,
        .unfused = false,
    };
    n_variants += 1;
    variants_buf[n_variants] = .{
        .label = "fused2s4  WNT=2 S=4   512t (OC/128 fused)",
        .func = if (k_bits == 2) ks.k2_m8_fused_w2s4 else ks.k3_m8_fused_w2s4,
        .grid_div = 128,
        .block_x = 512,
        .unfused = false,
    };
    n_variants += 1;
    variants_buf[n_variants] = .{
        .label = "narrow16  WNT=2 S=16  512t (OC/32 +hadOut)",
        .func = if (k_bits == 2) ks.k2_m8_narrow_s16 else ks.k3_m8_narrow_s16,
        .grid_div = 32,
        .block_x = 512,
        .unfused = true,
    };
    n_variants += 1;
    variants_buf[n_variants] = .{
        .label = "narrow12  WNT=2 S=12  384t (OC/32 +hadOut)",
        .func = if (k_bits == 2) ks.k2_m8_narrow_s12 else ks.k3_m8_narrow_s12,
        .grid_div = 32,
        .block_x = 384,
        .unfused = true,
    };
    n_variants += 1;
    variants_buf[n_variants] = .{
        .label = "narrow8   WNT=2 S=8   256t (OC/32 +hadOut)",
        .func = if (k_bits == 2) ks.k2_m8_narrow_s8 else ks.k3_m8_narrow_s8,
        .grid_div = 32,
        .block_x = 256,
        .unfused = true,
    };
    n_variants += 1;
    variants_buf[n_variants] = .{
        .label = "narrow4   WNT=2 S=4   128t (OC/32 +hadOut)",
        .func = if (k_bits == 2) ks.k2_m8_narrow_s4 else ks.k3_m8_narrow_s4,
        .grid_div = 32,
        .block_x = 128,
        .unfused = true,
    };
    n_variants += 1;

    var timer = try cuda.Timer.init();
    defer timer.deinit();

    var base_us: f64 = 0.0;
    const warmup_iters: u32 = 30;
    const bench_iters: u32 = 300;

    // Standalone hadOut overhead measurement
    for (0..warmup_iters) |_| try launchHadOut(ctx, ks.had_out, d_mid, d_rout, d_y, oc, max_m);
    try ctx.sync();
    try timer.begin(ctx.stream);
    for (0..bench_iters) |_| try launchHadOut(ctx, ks.had_out, d_mid, d_rout, d_y, oc, max_m);
    const had_out_us = (@as(f64, try timer.endMs(ctx.stream)) * 1000.0) / @as(f64, @floatFromInt(bench_iters));

    std.debug.print("  [M=8 Bench ({d} iters | standalone hadOut={d:.2} us)]\n", .{ bench_iters, had_out_us });
    for (variants_buf[0..n_variants], 0..) |v, idx| {
        const grid_x: u32 = @intCast(oc / v.grid_div);
        // Correctness vs base
        if (v.unfused) {
            try launchGemv(ctx, v.func, grid_x, v.block_x, d_xh, d_code, d_rout, d_mid, max_m, ic, oc);
            try launchHadOut(ctx, ks.had_out, d_mid, d_rout, d_y, oc, max_m);
        } else {
            try launchGemv(ctx, v.func, grid_x, v.block_x, d_xh, d_code, d_rout, d_y, max_m, ic, oc);
        }
        try ctx.sync();
        try cuda.Context.copyDtoH(std.mem.sliceAsBytes(h_y_gpu), d_y);
        const diff_vs_base = maxAbsDiff(h_y_gpu, h_y_base);

        // Warmup
        for (0..warmup_iters) |_| {
            if (v.unfused) {
                try launchGemv(ctx, v.func, grid_x, v.block_x, d_xh, d_code, d_rout, d_mid, max_m, ic, oc);
                try launchHadOut(ctx, ks.had_out, d_mid, d_rout, d_y, oc, max_m);
            } else {
                try launchGemv(ctx, v.func, grid_x, v.block_x, d_xh, d_code, d_rout, d_y, max_m, ic, oc);
            }
        }
        try ctx.sync();

        // Timed run
        try timer.begin(ctx.stream);
        for (0..bench_iters) |_| {
            if (v.unfused) {
                try launchGemv(ctx, v.func, grid_x, v.block_x, d_xh, d_code, d_rout, d_mid, max_m, ic, oc);
                try launchHadOut(ctx, ks.had_out, d_mid, d_rout, d_y, oc, max_m);
            } else {
                try launchGemv(ctx, v.func, grid_x, v.block_x, d_xh, d_code, d_rout, d_y, max_m, ic, oc);
            }
        }
        const total_ms = try timer.endMs(ctx.stream);
        const us = (@as(f64, total_ms) * 1000.0) / @as(f64, @floatFromInt(bench_iters));
        if (idx == 0) base_us = us;
        const gb_s = @as(f64, @floatFromInt(code_bytes.len)) / (us * 1000.0);
        const speedup = base_us / us;

        std.debug.print("    {s} | grid={d:3} | {d:6.2} us | {d:6.1} GB/s | {d:4.2}x | diff_vs_base={e:.2}\n", .{
            v.label, grid_x, us, gb_s, speedup, diff_vs_base,
        });
    }
}

pub fn run(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("=== kintsugi-zig Phase 1 Step 1: Real-Weight Escha sm_89 Verification & Benchmark ===\n", .{});

    const path_z = try allocator.allocSentinel(u8, model_path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, model_path);

    var model = try gguf.GgufFile.open(allocator, path_z);
    defer model.deinit(allocator);

    const dep2_info = model.find("escha_dep_k2") orelse return error.MissingDepK2;
    const dep3_info = model.find("escha_dep_k3") orelse return error.MissingDepK3;
    const dep_k2: []const i16 = std.mem.bytesAsSlice(i16, @as([]align(2) const u8, @alignCast(model.tensorBytes(dep2_info))));
    const dep_k3: []const i16 = std.mem.bytesAsSlice(i16, @as([]align(2) const u8, @alignCast(model.tensorBytes(dep3_info))));

    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();
    const info = try ctx.queryInfo();
    std.debug.print("GPU          : {s} ({d} SMs, free VRAM now: {d} MiB)\n", .{
        info.name(), info.sm_count, info.free_vram / (1024 * 1024),
    });

    var mod = try cuda.Module.loadData(escha_sm89_cubin);
    defer mod.deinit();
    const ks = try KernelSet.load(&mod);
    std.debug.print("CUBIN loaded : escha_sm89.cubin ({d} bytes embedded)\n", .{escha_sm89_cubin.len});

    // 1. 3-bit small tensor (1.88 MiB) — verify ALL 8 rows against CPU ground truth
    try testAndBenchTensor(allocator, &ctx, &ks, &model, dep_k2, dep_k3, "blk.3.attn_k", true, true);

    // 2. 2-bit OC=5120 tensor (7.50 MiB) — exact 60-SM underutilization geometry
    try testAndBenchTensor(allocator, &ctx, &ks, &model, dep_k2, dep_k3, "blk.3.attn_output", false, true);

    // 3. 3-bit OC=5120 tensor (11.25 MiB)
    try testAndBenchTensor(allocator, &ctx, &ks, &model, dep_k2, dep_k3, "blk.0.ssm_out", false, true);

    // 4. Wide-OC projections (attn_qkv OC=10240, ffn_gate OC=17408)
    try testAndBenchTensor(allocator, &ctx, &ks, &model, dep_k2, dep_k3, "blk.0.attn_qkv", false, true);
    try testAndBenchTensor(allocator, &ctx, &ks, &model, dep_k2, dep_k3, "blk.0.ffn_gate", false, true);

    // 5. 3-bit FFN down tensor (31.88 MiB, 25% of model!)
    try testAndBenchTensor(allocator, &ctx, &ks, &model, dep_k2, dep_k3, "blk.0.ffn_down", false, true);

    std.debug.print("\nAll runnable Escha sm_89 real-weight checks PASSED.\n", .{});
}
