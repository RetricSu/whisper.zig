const std = @import("std");

/// Whisper library artifact with module - for use by downstream dependencies
pub const WhisperLib = struct {
    artifact: *std.Build.Step.Compile,
    module: *std.Build.Module,
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build whisper and get the linkable artifact
    const whisper_lib = buildWhisperLib(b, target, optimize);

    // =============== Main executable ====================
    const exe = b.addExecutable(.{
        .name = "whisper.zig",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("src/main.zig"),
        }),
    });

    // Link whisper
    exe.root_module.addImport("whisper", whisper_lib.module);
    exe.linkLibrary(whisper_lib.artifact);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // =============== Test ====================
    const whisper_dep = b.dependency("whisper", .{});
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("src/root.zig"),
        }),
    });

    // Add include paths for tests
    tests.addIncludePath(whisper_dep.path("include"));
    tests.addIncludePath(whisper_dep.path("ggml/include"));
    tests.addIncludePath(whisper_dep.path("src"));

    tests.linkLibrary(whisper_lib.artifact);

    const test_step = b.step("test", "Run unit tests");
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);

    // =============== Static library (for local install) ====================
    const lib_install = b.addInstallArtifact(whisper_lib.artifact, .{});
    const lib_step = b.step("lib", "Build the static library");
    lib_step.dependOn(&lib_install.step);

    // Also install library by default so downstream can access via artifact("whisper")
    b.installArtifact(whisper_lib.artifact);
}

/// Build whisper library - can be called by downstream dependencies
/// Usage in downstream build.zig:
///   const whisper_dep = b.dependency("whisper_zig", .{ .target = target, .optimize = optimize });
///   const whisper_lib = @import("whisper_zig").buildWhisperLib(b, target, optimize);
///   exe.root_module.addImport("whisper", whisper_lib.module);
///   exe.linkLibrary(whisper_lib.artifact);
pub fn buildWhisperLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) WhisperLib {
    // Get whisper.cpp source from dependency
    const whisper_dep = b.dependency("whisper", .{});
    const whisper_path = whisper_dep.path("").getPath(b);

    // Get the cache root path (this is the downstream project's .zig-cache)
    const cache_root = b.cache_root.path orelse ".zig-cache";
    const whisper_build_dir = b.fmt("{s}/whisper_build", .{cache_root});

    // Run CMake to build whisper.cpp
    const cmake_build = runCMakeBuild(b, target, optimize, whisper_path, whisper_build_dir);

    // Create the Zig library that wraps whisper
    const lib = b.addLibrary(.{
        .name = "whisper",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("src/root.zig"),
        }),
    });

    // Add whisper include paths
    lib.addIncludePath(whisper_dep.path("include"));
    lib.addIncludePath(whisper_dep.path("ggml/include"));
    lib.addIncludePath(whisper_dep.path("src"));

    // Add CMake build output paths - link the static libraries directly (use absolute paths)
    lib.addObjectFile(.{ .cwd_relative = b.fmt("{s}/src/libwhisper.a", .{whisper_build_dir}) });
    lib.addObjectFile(.{ .cwd_relative = b.fmt("{s}/ggml/src/libggml.a", .{whisper_build_dir}) });
    lib.addObjectFile(.{ .cwd_relative = b.fmt("{s}/ggml/src/libggml-base.a", .{whisper_build_dir}) });
    lib.addObjectFile(.{ .cwd_relative = b.fmt("{s}/ggml/src/libggml-cpu.a", .{whisper_build_dir}) });

    // macOS-specific static libraries
    if (target.result.os.tag.isDarwin()) {
        lib.addObjectFile(.{ .cwd_relative = b.fmt("{s}/ggml/src/ggml-metal/libggml-metal.a", .{whisper_build_dir}) });
        lib.addObjectFile(.{ .cwd_relative = b.fmt("{s}/ggml/src/ggml-blas/libggml-blas.a", .{whisper_build_dir}) });
    }

    // Link platform-specific dependencies
    if (target.result.os.tag.isDarwin()) {
        lib.linkFramework("Foundation");
        lib.linkFramework("Accelerate");
        lib.linkFramework("Metal");
    }

    lib.root_module.addCMacro("_GNU_SOURCE", "");
    lib.linkLibCpp();

    // Ensure CMake build runs first
    lib.step.dependOn(&cmake_build.step);

    // Create the module for importing
    const module = b.addModule("whisper", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    module.addIncludePath(whisper_dep.path("include"));
    module.addIncludePath(whisper_dep.path("ggml/include"));
    module.addIncludePath(whisper_dep.path("src"));
    module.link_libc = true;

    return .{
        .artifact = lib,
        .module = module,
    };
}

fn runCMakeBuild(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    whisper_path: []const u8,
    build_dir: []const u8,
) *std.Build.Step.Run {
    const cmake_configure = b.addSystemCommand(&.{
        "cmake",
        "-G",
        "Ninja",
        "-B",
        build_dir,
        "-S",
        whisper_path,
        b.fmt("-DCMAKE_BUILD_TYPE={s}", .{switch (optimize) {
            .Debug => "Debug",
            .ReleaseFast => "Release",
            .ReleaseSafe => "RelWithDebInfo",
            .ReleaseSmall => "MinSizeRel",
        }}),
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_OPENMP=OFF",
        "-DWHISPER_BUILD_EXAMPLES=OFF",
        "-DWHISPER_BUILD_TESTS=OFF",
    });

    if (target.result.os.tag.isDarwin()) {
        cmake_configure.addArgs(&.{
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_METAL=ON",
        });
    } else {
        cmake_configure.addArgs(&.{
            "-DGGML_METAL_EMBED_LIBRARY=OFF",
            "-DGGML_METAL=OFF",
        });
    }

    const cmake_build = b.addSystemCommand(&.{
        "cmake",
        "--build",
        build_dir,
    });
    cmake_build.step.dependOn(&cmake_configure.step);

    return cmake_build;
}
