const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const sm_arch = b.option([]const u8, "arch", "CUDA SM architecture (default: sm_89)") orelse "sm_89";

    // Step 1: Compile kernels/sm89_core.cu -> sm_89 native CUBIN via CUDA 12.8 NVRTC
    const compile_cubin = b.addSystemCommand(&.{ "python3", "tools/compile_cubin.py" });
    compile_cubin.addFileArg(b.path("kernels/sm89_core.cu"));
    const cubin_out = compile_cubin.addOutputFileArg("sm89_core.cubin");
    compile_cubin.addArg(sm_arch);

    // Step 2: Build self-contained Zig runtime embedding sm89_core.cubin
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root_mod.addAnonymousImport("sm89_core.cubin", .{
        .root_source_file = cubin_out,
    });
    root_mod.linkSystemLibrary("cuda", .{});

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
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addAnonymousImport("sm89_core.cubin", .{
        .root_source_file = cubin_out,
    });
    test_mod.linkSystemLibrary("cuda", .{});

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
