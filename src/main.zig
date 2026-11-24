const std = @import("std");
const config = @import("config.zig");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");

const sm89_core_cubin = @embedFile("sm89_core.cubin");

const default_model_path = "/home/jili/personal/kintsugi-27b/Kintsugi-27B/models/Kintsugi-27B-12G.gguf";
const default_drafter_path = "/home/jili/personal/kintsugi-27b/Kintsugi-27B/models/dflash-Qwen3.8-27B-Q4_0.gguf";

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    const cmd = if (args.len >= 2) args[1] else "probe";
    if (std.mem.eql(u8, cmd, "probe")) {
        try runProbe(allocator);
    } else if (std.mem.eql(u8, cmd, "info")) {
        const model_path = if (args.len >= 3) args[2] else default_model_path;
        const drafter_path = if (args.len >= 4) args[3] else default_drafter_path;
        try runInfo(allocator, model_path, drafter_path);
    } else {
        std.debug.print(
            \\kintsugi-zig — dedicated sm_89 CUDA runtime for Kintsugi-27B + DFlash2
            \\
            \\Commands:
            \\  probe                         Verify RTX 4070 Ti CUDA Driver API, sm_89 CUBIN kernels & D2D bandwidth
            \\  info [model.gguf] [draft.gguf] Zero-copy mmap GGUF inspection & static VRAM KV budget plan
            \\
        , .{});
    }
}

fn f32ToF16Bits(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

fn f16BitsToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

fn fwht128Cpu(v: *[128]f32) void {
    var len: usize = 1;
    while (len < 128) : (len <<= 1) {
        var i: usize = 0;
        while (i < 128) : (i += len * 2) {
            for (0..len) |j| {
                const a = v[i + j];
                const b = v[i + j + len];
                v[i + j] = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    const scale: f32 = 1.0 / @sqrt(@as(f32, 128.0));
    for (v) |*x| x.* *= scale;
}

fn runProbe(allocator: std.mem.Allocator) !void {
    std.debug.print("=== kintsugi-zig Phase 0: RTX 4070 Ti (sm_89) Bare-Metal Probe ===\n", .{});
    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();

    const info = try ctx.queryInfo();
    std.debug.print(
        "GPU          : {s} (cc {d}.{d}, {d} SMs, L2 {d} MiB, opt-in smem {d} KiB)\n",
        .{
            info.name(),
            info.cc_major,
            info.cc_minor,
            info.sm_count,
            @divTrunc(info.l2_cache_bytes, 1024 * 1024),
            @divTrunc(info.max_smem_optin, 1024),
        },
    );
    std.debug.print(
        "VRAM / VMM   : free {d} MiB / total {d} MiB | CUDA Driver {d} | VMM={any} (granularity {d} MiB)\n",
        .{
            info.free_vram / (1024 * 1024),
            info.total_vram / (1024 * 1024),
            info.driver_version,
            info.vmm_supported,
            info.vmm_min_granularity / (1024 * 1024),
        },
    );

    // 1. Load embedded native sm_89 CUBIN
    var mod = try cuda.Module.loadData(sm89_core_cubin);
    defer mod.deinit();
    const fn_fused = try mod.getFunction("fused_add_rmsnorm_had128_f16");
    std.debug.print("CUBIN loaded : sm89_core.cubin ({d} bytes embedded)\n", .{sm89_core_cubin.len});

    // 2. Numerical accuracy self-test for fused_add_rmsnorm_had128_f16 at exact Kintsugi M=8, D=5120
    const m: usize = config.ModelSpec.dflash_block_size; // 8
    const d: usize = config.ModelSpec.d_model; // 5120
    const eps: f32 = config.ModelSpec.rms_eps;
    const total_elems = m * d;

    const h_res = try allocator.alloc(f32, total_elems);
    const h_del = try allocator.alloc(f32, total_elems);
    const h_w = try allocator.alloc(f32, d);
    const h_rin_f16 = try allocator.alloc(u16, d);
    const h_out_gpu = try allocator.alloc(u16, total_elems);
    const h_out_ref = try allocator.alloc(f32, total_elems);

    for (0..total_elems) |i| {
        const fi: f32 = @floatFromInt(i % 997);
        h_res[i] = @sin(fi * 0.013) * 0.5;
        h_del[i] = @cos(fi * 0.029) * 0.25;
    }
    for (0..d) |j| {
        const fj: f32 = @floatFromInt(j);
        h_w[j] = 0.8 + 0.4 * @sin(fj * 0.011);
        const sign: f32 = if ((j & 3) == 1 or (j & 3) == 2) -1.0 else 1.0;
        h_rin_f16[j] = f32ToF16Bits(sign * (0.9 + 0.1 * @cos(fj * 0.007)));
    }

    // CPU golden reference
    for (0..m) |row| {
        const base = row * d;
        var sum_sq: f64 = 0.0;
        for (0..d) |j| {
            const v = h_res[base + j] + h_del[base + j];
            sum_sq += @as(f64, v) * @as(f64, v);
        }
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(d)) + @as(f64, eps)));
        var blk: usize = 0;
        while (blk < d) : (blk += 128) {
            var tmp: [128]f32 = undefined;
            for (0..128) |k| {
                const j = blk + k;
                const v = h_res[base + j] + h_del[base + j];
                tmp[k] = v * inv_rms * h_w[j] * f16BitsToF32(h_rin_f16[j]);
            }
            fwht128Cpu(&tmp);
            for (0..128) |k| {
                h_out_ref[base + blk + k] = tmp[k];
            }
        }
    }

    // Upload to GPU
    const d_res = try cuda.Context.alloc(total_elems * @sizeOf(f32));
    defer cuda.Context.free(d_res);
    const d_del = try cuda.Context.alloc(total_elems * @sizeOf(f32));
    defer cuda.Context.free(d_del);
    const d_w = try cuda.Context.alloc(d * @sizeOf(f32));
    defer cuda.Context.free(d_w);
    const d_rin = try cuda.Context.alloc(d * @sizeOf(u16));
    defer cuda.Context.free(d_rin);
    const d_out = try cuda.Context.alloc(total_elems * @sizeOf(u16));
    defer cuda.Context.free(d_out);

    try cuda.Context.copyHtoD(d_res, std.mem.sliceAsBytes(h_res));
    try cuda.Context.copyHtoD(d_del, std.mem.sliceAsBytes(h_del));
    try cuda.Context.copyHtoD(d_w, std.mem.sliceAsBytes(h_w));
    try cuda.Context.copyHtoD(d_rin, std.mem.sliceAsBytes(h_rin_f16));

    var p_res = d_res;
    var p_del = d_del;
    var p_w = d_w;
    var p_rin = d_rin;
    var p_out = d_out;
    var p_dim: c_int = @intCast(d);
    var p_eps: f32 = eps;
    var p_has_delta: c_int = 1;
    var params = [_]?*anyopaque{
        @ptrCast(&p_res),
        @ptrCast(&p_del),
        @ptrCast(&p_w),
        @ptrCast(&p_rin),
        @ptrCast(&p_out),
        @ptrCast(&p_dim),
        @ptrCast(&p_eps),
        @ptrCast(&p_has_delta),
    };

    try cuda.launch(fn_fused, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
    try ctx.sync();
    try cuda.Context.copyDtoH(std.mem.sliceAsBytes(h_out_gpu), d_out);

    var max_abs_err: f32 = 0.0;
    var max_rel_err: f32 = 0.0;
    for (0..total_elems) |i| {
        const got = f16BitsToF32(h_out_gpu[i]);
        const ref_f16 = f16BitsToF32(f32ToF16Bits(h_out_ref[i]));
        const diff = @abs(got - ref_f16);
        const rel = diff / (@abs(ref_f16) + 1.0e-3);
        if (diff > max_abs_err) max_abs_err = diff;
        if (rel > max_rel_err) max_rel_err = rel;
    }
    std.debug.print(
        "Kernel test  : fused_add_rmsnorm_had128_f16 (M={d}, D={d}) max_abs_f16={e:.3}, max_rel={e:.3} [{s}]\n",
        .{ m, d, max_abs_err, max_rel_err, if (max_rel_err < 2.0e-3) "PASS" else "FAIL" },
    );
    if (max_rel_err >= 2.0e-3) return error.KernelAccuracyFailed;

    // 3. Microbenchmark fused kernel (1000 launches) + D2D bandwidth (auto-sized to live free VRAM)
    var timer = try cuda.Timer.init();
    defer timer.deinit();

    // Warmup
    for (0..20) |_| try cuda.launch(fn_fused, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
    try ctx.sync();

    const iters: u32 = 1000;
    try timer.begin(ctx.stream);
    for (0..iters) |_| try cuda.launch(fn_fused, .{ @intCast(m), 1, 1 }, .{ 256, 1, 1 }, 0, ctx.stream, &params);
    const fused_total_ms = try timer.endMs(ctx.stream);
    const fused_us = (fused_total_ms * 1000.0) / @as(f32, @floatFromInt(iters));
    std.debug.print(
        "Kernel speed : fused_add_rmsnorm_had128_f16 = {d:.2} us/call ({d:.2} ms across 128 norm+had sites/step)\n",
        .{ fused_us, (fused_us * 128.0) / 1000.0 },
    );

    if (info.free_vram >= 600 * 1024 * 1024) {
        const bw_bytes: usize = 256 * 1024 * 1024;
        const d_src = try cuda.Context.alloc(bw_bytes);
        defer cuda.Context.free(d_src);
        const d_dst = try cuda.Context.alloc(bw_bytes);
        defer cuda.Context.free(d_dst);
        try ctx.copyDtoDAsync(d_dst, d_src, bw_bytes);
        try ctx.sync();
        try timer.begin(ctx.stream);
        for (0..20) |_| try ctx.copyDtoDAsync(d_dst, d_src, bw_bytes);
        const d2d_ms = try timer.endMs(ctx.stream);
        const gb_s = (@as(f64, @floatFromInt(bw_bytes)) * 2.0 * 20.0) / (@as(f64, d2d_ms) * 1.0e6);
        std.debug.print("D2D VRAM BW  : {d:.1} GB/s (256 MiB x 20 via CUDA Driver API)\n", .{gb_s});
    } else {
        std.debug.print(
            "D2D VRAM BW  : skipped (only {d} MiB free VRAM — another GPU server is active)\n",
            .{info.free_vram / (1024 * 1024)},
        );
    }
}

fn toCString(allocator: std.mem.Allocator, s: []const u8) ![:0]u8 {
    const buf = try allocator.allocSentinel(u8, s.len, 0);
    @memcpy(buf, s);
    return buf;
}

fn runInfo(allocator: std.mem.Allocator, model_path: []const u8, drafter_path: []const u8) !void {
    std.debug.print("=== kintsugi-zig GGUF Zero-Copy Audit & Static VRAM Budget ===\n", .{});
    const model_z = try toCString(allocator, model_path);
    const drafter_z = try toCString(allocator, drafter_path);

    var model = try gguf.GgufFile.open(allocator, model_z);
    defer model.deinit(allocator);
    var drafter = try gguf.GgufFile.open(allocator, drafter_z);
    defer drafter.deinit(allocator);

    var trunk_bytes: u64 = 0;
    var mtp_bytes: u64 = 0;
    var embd_bytes: u64 = 0;
    var head_q8_bytes: u64 = 0;
    var escha_code_count: u32 = 0;

    for (model.tensors) |t| {
        const sz = t.byteSize();
        if (std.mem.eql(u8, t.name, "token_embd.weight")) {
            embd_bytes += sz;
        } else if (std.mem.eql(u8, t.name, "output.weight")) {
            head_q8_bytes += sz;
        } else if (std.mem.startsWith(u8, t.name, "blk.64.")) {
            mtp_bytes += sz;
        } else {
            trunk_bytes += sz;
            if (std.mem.endsWith(u8, t.name, ".escha_code")) escha_code_count += 1;
        }
    }

    var dflash_bytes: u64 = 0;
    for (drafter.tensors) |t| dflash_bytes += t.byteSize();

    std.debug.print(
        "Target GGUF  : arch={s}, v{d}, {d} tensors ({d} escha_code), file={d:.2} GiB\n",
        .{ model.architecture, model.version, model.n_tensors, escha_code_count, @as(f64, @floatFromInt(model.mapped.len)) / (1024.0 * 1024.0 * 1024.0) },
    );
    std.debug.print(
        "  - Trunk blk.0..63 + LUTs : {d:.1} MiB\n  - token_embd (host mmap) : {d:.1} MiB\n  - output.weight (Q8_0)   : {d:.1} MiB (stage-1 Q4_0 in VRAM = 682.0 MiB)\n  - blk.64 MTP (skipped)   : {d:.1} MiB\n",
        .{
            @as(f64, @floatFromInt(trunk_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(embd_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(head_q8_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(mtp_bytes)) / (1024.0 * 1024.0),
        },
    );
    std.debug.print(
        "DFlash2 GGUF : arch={s}, v{d}, {d} tensors, weights={d:.1} MiB\n",
        .{ drafter.architecture, drafter.version, drafter.n_tensors, @as(f64, @floatFromInt(dflash_bytes)) / (1024.0 * 1024.0) },
    );

    // Query live GPU VRAM to compute exact static plan (use clean-card usable VRAM if another server is running)
    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();
    const gpu = try ctx.queryInfo();
    const desktop_vram_bytes: u64 = 284 * 1024 * 1024;
    const clean_free_bytes: u64 = if (gpu.free_vram > 4 * 1024 * 1024 * 1024)
        gpu.free_vram
    else if (gpu.total_vram > desktop_vram_bytes)
        gpu.total_vram - desktop_vram_bytes
    else
        gpu.free_vram;
    const head_stage1_bytes: u64 = 715_161_600; // Q4_0 stage-1 LM head (248320 x 5120)
    const plan = config.VramPlan.compute(
        gpu.total_vram,
        clean_free_bytes,
        trunk_bytes,
        dflash_bytes,
        head_stage1_bytes,
        220, // 220 MiB desktop/driver headroom
    );

    std.debug.print("\n--- Static VRAM Budget Plan (vs llama.cpp 69,632 KV cells) ---\n", .{});
    std.debug.print("Card usable VRAM    : {d} MiB / {d} MiB (live free now: {d} MiB)\n", .{
        clean_free_bytes / (1024 * 1024),
        gpu.total_vram / (1024 * 1024),
        gpu.free_vram / (1024 * 1024),
    });
    std.debug.print("Fixed GPU residency : trunk {d} + head1 {d} + dflash {d} + GDN {d} + scratch {d} = {d} MiB\n", .{
        plan.trunk_weights_bytes / (1024 * 1024),
        plan.head_stage1_bytes / (1024 * 1024),
        plan.dflash_weights_bytes / (1024 * 1024),
        plan.gdn_state_bytes / (1024 * 1024),
        plan.scratch_bytes / (1024 * 1024),
        (plan.trunk_weights_bytes + plan.head_stage1_bytes + plan.dflash_weights_bytes + plan.gdn_state_bytes + plan.scratch_bytes) / (1024 * 1024),
    });
    std.debug.print(
        "VRAM KV headroom    : {d} MiB -> {d} VRAM KV cells (+{d} cells over llama.cpp 69,632!)\n",
        .{
            plan.available_for_kv_bytes / (1024 * 1024),
            plan.max_vram_kv_cells,
            if (plan.max_vram_kv_cells > 69632) plan.max_vram_kv_cells - 69632 else 0,
        },
    );
    std.debug.print(
        "256K tiered spill   : {d} MiB in pinned host RAM (full 262,144 ctx = {d} MiB)\n",
        .{ plan.host_spill_bytes / (1024 * 1024), plan.full_256k_kv_bytes / (1024 * 1024) },
    );
}

test {
    _ = config;
    _ = gguf;
}
