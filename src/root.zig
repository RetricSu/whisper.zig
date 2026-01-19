const std = @import("std");

// 1. 只引入 whisper.h，不再引入 sndfile.h
pub const c = @cImport({
    @cInclude("whisper.h");
});

pub const GetTextParams = struct {
    allocator: std.mem.Allocator,
    file_path: []const u8,
    model_path: []const u8,
    n_threads: u32 = 4,
    use_gpu: bool = true,
};

// 2. 你的核心函数 (API 保持不变)
pub fn get_text_from_wav(params: GetTextParams) !std.ArrayList([]const u8) {
    const allocator = params.allocator;

    // 读取并解析 WAV 文件
    const wav_data = try internal_read_wav(allocator, params.file_path);
    defer allocator.free(wav_data);

    // wav_data 已经是 []f32 了，直接传给处理函数
    return get_text_from_samples(allocator, params.model_path, wav_data, params.n_threads, params.use_gpu);
}

pub fn get_text_from_samples(allocator: std.mem.Allocator, model_path: []const u8, samples: []f32, n_threads: u32, use_gpu: bool) !std.ArrayList([]const u8) {
    const model_path_z = try allocator.dupeZ(u8, model_path);
    defer allocator.free(model_path_z);

    // Configure Context
    var cparams: c.whisper_context_params = c.whisper_context_default_params();
    cparams.use_gpu = use_gpu;

    const ctx = c.whisper_init_from_file_with_params(model_path_z, cparams);
    if (ctx == null) {
        std.debug.print("Failed to create whisper context\n", .{});
        return error.WhisperInitFailed;
    }
    defer c.whisper_free(ctx);

    var fparams = c.whisper_full_default_params(c.WHISPER_SAMPLING_GREEDY);
    fparams.n_threads = @intCast(n_threads);
    fparams.print_realtime = false;
    fparams.print_progress = false;
    fparams.no_timestamps = true;

    const n_samples: usize = samples.len;

    // Run
    if (c.whisper_full(ctx, fparams, samples.ptr, @intCast(n_samples)) != 0) {
        std.debug.print("failed to call whisper_full\n", .{});
        return error.WhisperRunFailed;
    }

    var string_list = try std.ArrayList([]const u8).initCapacity(allocator, 8);
    errdefer {
        for (string_list.items) |str| allocator.free(str);
        string_list.deinit(allocator);
    }

    const n_segments: i32 = c.whisper_full_n_segments(ctx);
    for (0..@intCast(n_segments)) |i| {
        const raw_text: [*c]const u8 = c.whisper_full_get_segment_text(ctx, @intCast(i));
        // std.debug.print("{s}\n", .{raw_text}); // 可选：打印日志
        const text_slice = std.mem.span(raw_text);
        const owned_text = try allocator.dupe(u8, text_slice);
        try string_list.append(allocator, owned_text);
    }

    return string_list;
}

// ============================================================
// 3. 私有辅助函数：纯 Zig 实现 WAV 解析
// ============================================================
fn internal_read_wav(allocator: std.mem.Allocator, file_path: []const u8) ![]f32 {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // WAV Header Structure (44 bytes)
    const WavHeader = extern struct {
        riff_header: [4]u8,
        wav_size: u32,
        wave_header: [4]u8,
        fmt_header: [4]u8,
        fmt_chunk_size: u32,
        audio_format: u16,
        num_channels: u16,
        sample_rate: u32,
        byte_rate: u32,
        sample_alignment: u16,
        bit_depth: u16,
        data_header: [4]u8,
        data_bytes: u32,
    };

    var header: WavHeader = undefined;
    const header_bytes = std.mem.asBytes(&header);
    const bytes_read = try file.read(header_bytes);
    if (bytes_read != @sizeOf(WavHeader)) {
        return error.InvalidWavFile;
    }

    // 校验 WAV 格式
    if (!std.mem.eql(u8, &header.riff_header, "RIFF") or !std.mem.eql(u8, &header.wave_header, "WAVE")) {
        return error.InvalidWavFile;
    }
    // 校验采样率 (Whisper 必须 16kHz)
    if (header.sample_rate != 16000) {
        std.debug.print("Error: Input file must be 16kHz. Current: {d}Hz\n", .{header.sample_rate});
        return error.InvalidSampleRate;
    }
    // 校验位深 (只支持 16-bit PCM)
    if (header.audio_format != 1 or header.bit_depth != 16) {
        return error.UnsupportedAudioFormat; // 只支持 16-bit PCM
    }
    // 校验声道 (只支持单声道)
    if (header.num_channels != 1) {
        return error.UnsupportedChannels; // 只支持 Mono
    }

    // 读取数据并转换
    const num_samples = header.data_bytes / 2; // 16bit = 2 bytes
    var samples = try allocator.alloc(f32, num_samples);
    errdefer allocator.free(samples);

    for (0..num_samples) |i| {
        // 读取 16位 整数 (Little Endian)
        var sample_bytes: [2]u8 = undefined;
        const read_count = try file.read(&sample_bytes);
        if (read_count != 2) {
            return error.UnexpectedEndOfFile;
        }
        const sample_i16 = std.mem.readInt(i16, &sample_bytes, .little);
        // 归一化到 [-1.0, 1.0]
        samples[i] = @as(f32, @floatFromInt(sample_i16)) / 32768.0;
    }

    return samples;
}

// ============================================================
// 4. 测试代码
// ============================================================
test "get_text_from_wav" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 注意：运行 zig build test 时，确保这些文件存在，或者把这个 test 注释掉
    // 如果没有文件，这个 test 会失败
    const params = GetTextParams{
        .allocator = allocator,
        .file_path = "short.wav",
        .model_path = "ggml-tiny.en.bin",
        .n_threads = 4,
        .use_gpu = true, // Metal embedded library now works
    };

    // 仅仅为了编译通过，我们可以先捕获错误
    // 实际使用时请确保文件存在
    var result = get_text_from_wav(params) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Test skipped: files not found\n", .{});
            return;
        }
        return err;
    };

    defer {
        for (result.items) |str| allocator.free(str);
        result.deinit(allocator);
    }

    // Audio file might be too short, which is OK for basic testing
    // Just verify the transcription ran without crashing
    std.debug.print("\nTranscribed {d} segments:\n", .{result.items.len});
    for (result.items) |segment| {
        std.debug.print("  {s}\n", .{segment});
    }
}
