const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const sm_arch = b.option([]const u8, "arch", "CUDA SM architecture (default: sm_89)") orelse "sm_89";

    // Step 1: Build self-contained Zig runtime embedding all native sm_89 CUBINs
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root_mod.linkSystemLibrary("cuda", .{});

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.linkSystemLibrary("cuda", .{});

    // Compile every kernels/*.cu -> sm_89 native CUBIN via CUDA 12.8 NVRTC
    const kernel_specs = [_]struct { src: []const u8, name: []const u8 }{
        .{ .src = "kernels/sm89_core.cu", .name = "sm89_core.cubin" },
        .{ .src = "kernels/escha_sm89.cu", .name = "escha_sm89.cubin" },
        .{ .src = "kernels/gdn_sm89.cu", .name = "gdn_sm89.cubin" },
        .{ .src = "kernels/attn_sm89.cu", .name = "attn_sm89.cubin" },
        .{ .src = "kernels/lmhead_sm89.cu", .name = "lmhead_sm89.cubin" },
        .{ .src = "kernels/dflash_sm89.cu", .name = "dflash_sm89.cubin" },
    };
    for (kernel_specs) |ks| {
        const cmd = b.addSystemCommand(&.{"python3"});
        cmd.addFileArg(b.path("tools/compile_cubin.py"));
        cmd.addFileArg(b.path(ks.src));
        const cubin_out = cmd.addOutputFileArg(ks.name);
        cmd.addArg(sm_arch);
        root_mod.addAnonymousImport(ks.name, .{ .root_source_file = cubin_out });
        test_mod.addAnonymousImport(ks.name, .{ .root_source_file = cubin_out });
    }

    const exe = b.addExecutable(.{
        .name = "kintsugi-zig",
        .root_module = root_mod,
    });
    b.installArtifact(exe);

    // `zig build run -- <args>`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run kintsugi-zig");
    run_step.dependOn(&run_cmd.step);

    // `zig build test` (CPU unit tests — no GPU needed)
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
