const std = @import("std");
// 引入我们在 build.zig 中定义的 "whisper" 模块
// 这会自动链接 root.zig 和 C++ 库
const whisper = @import("whisper");

pub fn main() !void {
    // 1. 初始化内存分配器
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 2. 准备文件路径
    // 注意：请确保这些文件在你的运行目录下存在
    // model 可以在 https://huggingface.co/ggerganov/whisper.cpp/tree/main 下载
    const model_path = "ggml-large-v3-turbo.bin";
    const wav_path = "short.wav"; // 必须是 16kHz, 16-bit Mono

    // 3. 检查文件是否存在 (可选，但为了友好的报错)
    try checkFileExists(wav_path);
    try checkFileExists(model_path);

    std.debug.print("Loading model: {s}\n", .{model_path});
    std.debug.print("Processing audio: {s}\n", .{wav_path});

    // 4. 配置参数
    const params = whisper.GetTextParams{
        .allocator = allocator,
        .file_path = wav_path,
        .model_path = model_path,
        .n_threads = 4, // M1/M2/M3 通常 4-8 线程最快
        .use_gpu = true, // macOS 会自动使用 Metal
    };

    // 5. 调用核心功能
    // 这里内部使用了我们手写的 Zig WAV 解析器，完全不依赖 libsndfile
    var result = whisper.get_text_from_wav(params) catch |err| {
        std.debug.print("Error during transcription: {}\n", .{err});
        if (err == error.InvalidSampleRate) {
            std.debug.print("Hint: Please use ffmpeg to convert audio: ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le short.wav\n", .{});
        }
        return err;
    };

    // 6. 清理内存
    defer {
        for (result.items) |str| allocator.free(str);
        result.deinit(allocator);
    }

    // 7. 输出结果
    std.debug.print("\n---------------- Transcription Result ----------------\n", .{});
    for (result.items) |segment| {
        std.debug.print("{s}\n", .{segment});
    }
    std.debug.print("------------------------------------------------------\n", .{});
}

// 辅助函数：检查文件是否存在
fn checkFileExists(path: []const u8) !void {
    std.fs.cwd().access(path, .{}) catch {
        std.debug.print("Error: File not found: {s}\n", .{path});
        return error.FileNotFound;
    };
}
