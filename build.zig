const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // rootpath from zig-package (zig GLOBAL_CACHE_DIR)
    const whisperLazyPath = b.dependency("whisper", .{}).path("");

    // =============== Whisper library (built via CMake) ====================
    const whisper_build = buildWhisper(b, .{
        .target = target,
        .optimize = optimize,
        .dep_path = whisperLazyPath,
    });

    // =============== Library module ====================
    const whisper_module = b.addModule("whisper", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add include paths to the module
    whisper_module.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "include" }),
    });
    whisper_module.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "ggml", "include" }),
    });
    whisper_module.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "src" }),
    });

    whisper_module.link_libc = true;

    // =============== Main executable ====================
    const exe = b.addExecutable(.{
        .name = "whisper.zig",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("src/main.zig"),
        }),
    });

    // Add whisper include path (abspath)
    exe.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "include" }),
    });
    exe.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "ggml", "include" }),
    });
    exe.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "src" }),
    });

    // cmake build path
    exe.addLibraryPath(b.path(".zig-cache/whisper_build/src"));
    exe.addLibraryPath(b.path(".zig-cache/whisper_build/ggml/src"));

    if (exe.rootModuleTarget().os.tag.isDarwin()) {
        exe.linkFramework("Foundation");
        exe.linkFramework("Accelerate");
        exe.linkFramework("Metal");
    }

    exe.step.dependOn(&whisper_build.step);

    exe.linkSystemLibrary("whisper");
    exe.linkSystemLibrary("ggml");

    exe.root_module.addCMacro("_GNU_SOURCE", "");
    exe.linkLibCpp();

    // Add whisper module import
    exe.root_module.addImport("whisper", whisper_module);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // =============== Test executable ====================
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("src/root.zig"),
        }),
    });

    // Add include paths
    tests.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "include" }),
    });
    tests.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "ggml", "include" }),
    });
    tests.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ whisperLazyPath.getPath(b), "src" }),
    });

    // Add library paths
    tests.addLibraryPath(b.path(".zig-cache/whisper_build/src"));
    tests.addLibraryPath(b.path(".zig-cache/whisper_build/ggml/src"));

    // Link frameworks on macOS
    if (tests.rootModuleTarget().os.tag.isDarwin()) {
        tests.linkFramework("Foundation");
        tests.linkFramework("Accelerate");
        tests.linkFramework("Metal");
    }

    tests.step.dependOn(&whisper_build.step);

    // Link libraries
    tests.linkSystemLibrary("whisper");
    tests.linkSystemLibrary("ggml");

    tests.root_module.addCMacro("_GNU_SOURCE", "");
    tests.linkLibCpp();

    const test_step = b.step("test", "Run unit tests");
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
}

fn buildWhisper(b: *std.Build, args: struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    dep_path: std.Build.LazyPath,
}) *std.Build.Step.Run {
    const whisper_path = args.dep_path.getPath(b);
    const whisper_configure = b.addSystemCommand(&.{
        "cmake",
        "-G",
        "Ninja",
        "-B",
        ".zig-cache/whisper_build",
        "-S",
        whisper_path,
        b.fmt("-DCMAKE_BUILD_TYPE={s}", .{switch (args.optimize) {
            .Debug => "Debug",
            .ReleaseFast => "Release",
            .ReleaseSafe => "RelWithDebInfo",
            .ReleaseSmall => "MinSizeRel",
        }}),
        "-DGGML_OPENMP=OFF",
        "-DWHISPER_BUILD_EXAMPLES=OFF",
        "-DWHISPER_BUILD_TESTS=OFF",
    });

    if (args.target.result.os.tag.isDarwin())
        whisper_configure.addArgs(&.{
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_METAL=ON",
        })
    else
        whisper_configure.addArgs(&.{
            "-DGGML_METAL_EMBED_LIBRARY=OFF",
            "-DGGML_METAL=OFF",
        });

    const whisper_build = b.addSystemCommand(&.{
        "cmake",
        "--build",
        ".zig-cache/whisper_build",
    });
    whisper_build.step.dependOn(&whisper_configure.step);
    return whisper_build;
}
