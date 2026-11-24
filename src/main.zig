const std = @import("std");
const config = @import("config.zig");
const cuda = @import("cuda.zig");
const gguf = @import("gguf.zig");
const escha_ref = @import("escha_ref.zig");
const escha_bench = @import("escha_bench.zig");
const ops_bench = @import("ops_bench.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const engine_mod = @import("engine.zig");

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
    } else if (std.mem.eql(u8, cmd, "bench-escha")) {
        const model_path = if (args.len >= 3) args[2] else default_model_path;
        try escha_bench.run(allocator, model_path);
    } else if (std.mem.eql(u8, cmd, "bench-ops")) {
        const model_path = if (args.len >= 3) args[2] else default_model_path;
        try ops_bench.run(allocator, model_path);
    } else if (std.mem.eql(u8, cmd, "verify-e2e")) {
        const model_path = if (args.len >= 3) args[2] else default_model_path;
        try runVerifyE2E(allocator, model_path);
    } else if (std.mem.eql(u8, cmd, "generate")) {
        try runGenerate(allocator, args[2..]);
    } else {
        std.debug.print(
            \\kintsugi-zig — dedicated sm_89 CUDA runtime for Kintsugi-27B + DFlash2
            \\
            \\Commands:
            \\  probe                         Verify RTX 4070 Ti CUDA Driver API, sm_89 CUBIN kernels & D2D bandwidth
            \\  info [model.gguf] [draft.gguf] Zero-copy mmap GGUF inspection & static VRAM KV budget plan
            \\  bench-escha [model.gguf]      Verify Escha 2/3-bit sm_89 GEMV against CPU ref & benchmark geometries
            \\  bench-ops [model.gguf]        Verify & benchmark all 14 GDN, FullAttn/Q4_0 KV, SwiGLU & LM-head kernels
            \\  verify-e2e [model.gguf]       Verify full 64-layer target engine + tokenizer exact match vs llama-server
            \\  generate [--prompt S] [--max-tokens N] [--model P]  Run greedy/sampled target text generation
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

pub fn nowMs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) / 1.0e6;
}

fn runVerifyE2E(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("=== kintsugi-zig Phase 1 Step 3: Full 64-Layer E2E Verification vs llama-server ===\n", .{});
    const t0 = nowMs();

    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();
    var ks = try engine_mod.Kernels.load();
    defer ks.deinit();

    var model = try model_mod.TargetModel.load(allocator, &ctx, ks.fn_q8_to_q4, model_path);
    defer model.deinit();
    const t_load = nowMs();

    var tok = try tokenizer_mod.Tokenizer.init(allocator, &model.gguf_file);
    defer tok.deinit();
    const t_tok = nowMs();

    var eng = try engine_mod.Engine.init(allocator, &ctx, &ks, &model, 1120); // reserve 1120 MiB for DFlash2
    defer eng.deinit();
    const gpu_after = try ctx.queryInfo();

    std.debug.print(
        "Loaded 27B   : trunk {d:.1} MiB + head1 {d:.1} MiB in {d:.0} ms | tokenizer {d:.0} ms ({d} vocab)\n",
        .{
            @as(f64, @floatFromInt(model.trunk_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(model.head_q4_bytes)) / (1024.0 * 1024.0),
            t_load - t0,
            t_tok - t_load,
            tok.meta.tokens.len,
        },
    );
    std.debug.print(
        "VRAM KV pool : {d} Q4_0 KV cells allocated (free VRAM remaining for DFlash2: {d} MiB)\n",
        .{ eng.max_kv_cells, gpu_after.free_vram / (1024 * 1024) },
    );

    const Case = struct {
        name: []const u8,
        prompt: []const u8,
        ref_prompt_toks: []const u32,
        ref_gen_prefix: []const u32,
    };

    const cases = [_]Case{
        .{
            .name = "France",
            .prompt = "The capital of France is",
            .ref_prompt_toks = &[_]u32{ 760, 6511, 314, 9338, 369 },
            .ref_gen_prefix = &[_]u32{ 11751, 13, 198, 760, 6511, 314, 9338, 369 },
        },
        .{
            .name = "Fibonacci",
            .prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
            .ref_prompt_toks = &[_]u32{ 727, 73111, 1393, 1590, 198, 262, 413, 307, 2564, 220, 16, 25, 198, 285, 460, 307, 198, 262, 460 },
            .ref_gen_prefix = &[_]u32{ 73111, 1393, 12, 16, 8, 478, 73111, 1393 },
        },
    };

    var all_pass = true;
    for (cases) |c| {
        std.debug.print("\n--- Case: {s} ---\n", .{c.name});
        const enc = try tok.encode(allocator, c.prompt);
        defer allocator.free(enc);
        const tok_match = std.mem.eql(u32, enc, c.ref_prompt_toks);
        std.debug.print("Tokenizer    : {d} toks -> {s}", .{ enc.len, if (tok_match) "EXACT MATCH" else "MISMATCH" });
        if (!tok_match) {
            std.debug.print("\n  got: {any}\n  ref: {any}", .{ enc, c.ref_prompt_toks });
            all_pass = false;
        }
        std.debug.print("\n", .{});

        // 1. Greedy M=1 generation (32 tokens)
        try eng.reset();
        const tp0 = nowMs();
        const pf = try eng.prefill(enc);
        const tp1 = nowMs();
        const last_idx = (enc.len - 1) % engine_mod.MAX_M;
        var cur_tok = pf.tokens[last_idx];

        var gen_m1: [32]u32 = undefined;
        gen_m1[0] = cur_tok;
        const td0 = nowMs();
        for (1..32) |step| {
            const out = try eng.forward(&[_]u32{cur_tok}, true, true);
            cur_tok = out.tokens[0];
            gen_m1[step] = cur_tok;
        }
        const td1 = nowMs();

        var prefix_match = true;
        for (c.ref_gen_prefix, 0..) |ref_id, i| {
            if (gen_m1[i] != ref_id) prefix_match = false;
        }
        if (!prefix_match) all_pass = false;

        const text = try tok.decode(allocator, &gen_m1);
        defer allocator.free(text);
        const decode_ms = td1 - td0;
        const ms_per_tok = decode_ms / 31.0;
        const tok_s = 1000.0 / ms_per_tok;

        std.debug.print("M=1 Greedy   : prefix8={s} | prefill={d:.1} ms | decode={d:.2} ms/tok ({d:.2} tok/s)\n", .{
            if (prefix_match) "EXACT MATCH" else "MISMATCH",
            tp1 - tp0,
            ms_per_tok,
            tok_s,
        });
        std.debug.print("  ids[0..8]  : {any}\n", .{gen_m1[0..8]});
        std.debug.print("  ref[0..8]  : {any}\n", .{c.ref_gen_prefix});
        std.debug.print("  text       : \"{s}\"\n", .{text});

        // 2. Verify M=8 speculative verify (read-only GDN) + partial commitSpeculative(5) matches M=1!
        try eng.reset();
        const pf2 = try eng.prefill(enc);
        const t0_id = pf2.tokens[last_idx];
        // Feed 8 tokens: [t0, gen_m1[1], gen_m1[2], gen_m1[3], gen_m1[4], WRONG, WRONG, WRONG]
        const spec_in = [8]u32{ t0_id, gen_m1[1], gen_m1[2], gen_m1[3], gen_m1[4], 100, 101, 102 };
        const tv0 = nowMs();
        const vout = try eng.forward(&spec_in, false, true);
        const tv1 = nowMs();
        var spec_ok = true;
        for (0..5) |k| {
            if (vout.tokens[k] != gen_m1[k + 1]) spec_ok = false;
        }
        // Commit only the 5 accepted tokens, then step M=1 to check GDN state fidelity!
        const tc0 = nowMs();
        try eng.commitSpeculative(5);
        try ctx.sync();
        const tc1 = nowMs();
        const next_after_commit = try eng.forward(&[_]u32{gen_m1[5]}, true, true);
        const commit_ok = (next_after_commit.tokens[0] == gen_m1[6]);
        if (!spec_ok or !commit_ok) all_pass = false;

        std.debug.print("M=8 SpecTest : verify5={s} ({d:.2} ms/step) | commit5={s} ({d:.2} ms) | post_commit_tok={d} (exp {d})\n", .{
            if (spec_ok) "EXACT MATCH" else "MISMATCH",
            tv1 - tv0,
            if (commit_ok) "EXACT MATCH" else "MISMATCH",
            tc1 - tc0,
            next_after_commit.tokens[0],
            gen_m1[6],
        });
    }

    if (!all_pass) return error.E2EVerificationFailed;
    std.debug.print("\n[ALL E2E CHECKS PASSED: Tokenizer + 64L Target M=1 + M=8 Speculative Verify/Commit == llama-server!]\n", .{});
}

fn runGenerate(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var model_path: []const u8 = default_model_path;
    var prompt: []const u8 = "The capital of France is";
    var max_tokens: usize = 64;
    var temp: f32 = 0.0;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--prompt") and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, args[i], "--max-tokens") and i + 1 < args.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--temp") and i + 1 < args.len) {
            i += 1;
            temp = try std.fmt.parseFloat(f32, args[i]);
        }
    }

    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();
    var ks = try engine_mod.Kernels.load();
    defer ks.deinit();
    var model = try model_mod.TargetModel.load(allocator, &ctx, ks.fn_q8_to_q4, model_path);
    defer model.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(allocator, &model.gguf_file);
    defer tok.deinit();
    var eng = try engine_mod.Engine.init(allocator, &ctx, &ks, &model, 1120);
    defer eng.deinit();

    const prompt_toks = try tok.encode(allocator, prompt);
    defer allocator.free(prompt_toks);

    var rng = std.Random.DefaultPrng.init(42);
    const tp0 = nowMs();
    const pf = try eng.prefill(prompt_toks);
    const tp1 = nowMs();

    const last_idx = (prompt_toks.len - 1) % engine_mod.MAX_M;
    var cur = engine_mod.sampleFromCandidates(&pf.cands[last_idx], temp, 64, 0.95, &rng);
    var gen: std.ArrayListUnmanaged(u32) = .empty;
    defer gen.deinit(allocator);

    const td0 = nowMs();
    var piece_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer piece_buf.deinit(allocator);

    for (0..max_tokens) |step| {
        if (tok.isEos(cur)) break;
        try gen.append(allocator, cur);
        piece_buf.clearRetainingCapacity();
        try tok.decodeTokenTo(cur, &piece_buf, allocator);
        std.debug.print("{s}", .{piece_buf.items});
        if (step + 1 == max_tokens) break;
        const out = try eng.forward(&[_]u32{cur}, true, true);
        cur = engine_mod.sampleFromCandidates(&out.cands[0], temp, 64, 0.95, &rng);
    }
    const td1 = nowMs();
    const dec_toks = if (gen.items.len > 1) gen.items.len - 1 else 1;
    std.debug.print(
        "\n\n[prompt: {d} toks in {d:.1} ms | generated: {d} toks in {d:.1} ms ({d:.2} tok/s) | KV pool: {d} cells]\n",
        .{
            prompt_toks.len,
            tp1 - tp0,
            gen.items.len,
            td1 - td0,
            @as(f64, @floatFromInt(dec_toks)) * 1000.0 / (td1 - td0),
            eng.max_kv_cells,
        },
    );
}

test {
    _ = config;
    _ = gguf;
    _ = escha_ref;
    _ = tokenizer_mod;
}
