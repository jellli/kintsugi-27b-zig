const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const sm_arch = b.option([]const u8, "arch", "CUDA SM architecture (default: sm_89)") orelse "sm_89";

    // Step 1: Compile CUDA kernels -> sm_89 native CUBIN via CUDA 12.8 NVRTC
    const compile_core = b.addSystemCommand(&.{ "python3", "tools/compile_cubin.py" });
    compile_core.addFileArg(b.path("kernels/sm89_core.cu"));
    const core_cubin_out = compile_core.addOutputFileArg("sm89_core.cubin");
    compile_core.addArg(sm_arch);

    const compile_escha = b.addSystemCommand(&.{ "python3", "tools/compile_cubin.py" });
    compile_escha.addFileArg(b.path("kernels/escha_sm89.cu"));
    const escha_cubin_out = compile_escha.addOutputFileArg("escha_sm89.cubin");
    compile_escha.addArg(sm_arch);

    // Step 2: Build self-contained Zig runtime embedding native sm_89 CUBINs
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root_mod.addAnonymousImport("sm89_core.cubin", .{
        .root_source_file = core_cubin_out,
    });
    root_mod.addAnonymousImport("escha_sm89.cubin", .{
        .root_source_file = escha_cubin_out,
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
        .root_source_file = core_cubin_out,
    });
    test_mod.addAnonymousImport("escha_sm89.cubin", .{
        .root_source_file = escha_cubin_out,
    });
    test_mod.linkSystemLibrary("cuda", .{});

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
