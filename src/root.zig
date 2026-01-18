const std = @import("std");

pub const whisper = @cImport({
    @cInclude("whisper.h");
});
pub const libsndfile = @cImport({
    @cInclude("sndfile.h");
});

const GetTextParams = struct {
    allocator: std.mem.Allocator,
    file_path: []const u8,
    model_path: []const u8,
    n_threads: u32, // 4 threads is usually the sweet spot for M1
    use_gpu: bool,
};

pub fn get_text_from_wav(params: GetTextParams) !std.ArrayList([]const u8) {
    const allocator = params.allocator;
    const file_path = params.file_path;
    const model_path = params.model_path;
    const use_gpu = params.use_gpu;
    const n_threads = params.n_threads;

    var sfInfo: libsndfile.SF_INFO = .{};
    const file = libsndfile.sf_open(file_path, libsndfile.SFM_READ, &sfInfo);
    if (file == null) {
        std.debug.print("Could not open wav file\n", .{});
        return error.FileOpenFailed;
    }

    // ---------------------------------------------------------
    // CRITICAL SPEED CHECK:
    // Whisper REQUIRES 16000Hz audio.
    // If your wav is 44100Hz, the output will be garbage.
    // ---------------------------------------------------------
    if (sfInfo.samplerate != 16000) {
        std.debug.print("Error: Input file must be 16kHz. Current: {d}Hz\n", .{sfInfo.samplerate});
        // For a real app, you must implement resampling here.
        // For testing, just convert your file: ffmpeg -i input.wav -ar 16000 short.wav
        return error.InvalidSampleRate;
    }

    const n_samples: usize = @intCast(sfInfo.frames * sfInfo.channels);
    var samples: []f32 = undefined;
    samples = try std.heap.page_allocator.alloc(f32, n_samples);
    defer std.heap.page_allocator.free(samples); // Good practice to free

    const read_samples = libsndfile.sf_read_float(file, samples.ptr, @intCast(n_samples));
    if (read_samples != n_samples) {
        return error.FileReadFailed;
    }

    // Configure Context
    var cparams: whisper.whisper_context_params = whisper.whisper_context_default_params();
    cparams.use_gpu = use_gpu; // <--- FORCE METAL GPU USAGE

    const ctx = whisper.whisper_init_from_file_with_params(model_path, cparams);
    if (ctx == null) {
        std.debug.print("Failed to create whisper context\n", .{});
        std.process.exit(1);
    }
    defer whisper.whisper_free(ctx);

    var fparams = whisper.whisper_full_default_params(whisper.WHISPER_SAMPLING_GREEDY);
    fparams.n_threads = n_threads;
    fparams.print_realtime = false;
    fparams.print_progress = false;
    fparams.no_timestamps = true;

    // Run
    if (whisper.whisper_full(ctx, fparams, samples.ptr, @intCast(n_samples)) != 0) {
        std.debug.print("failed to call whisper_full\n", .{});
    }

    var string_list = std.ArrayList([]const u8).init(allocator);
    // If we fail partway through, we need to decide if we free what we have
    // or return the error. Usually, on error, we clean up.
    errdefer {
        for (string_list.items) |str| allocator.free(str);
        string_list.deinit();
    }

    const n_segments: i32 = whisper.whisper_full_n_segments(ctx);
    for (0..@intCast(n_segments)) |i| {
        const raw_text: [*c]const u8 = whisper.whisper_full_get_segment_text(ctx, @intCast(i));
        std.debug.print("{s}\n", .{raw_text});
        const text_slice = std.mem.span(raw_text);
        const owned_text = try allocator.dupe(u8, text_slice);
        try string_list.append(owned_text);
    }

    return string_list;
}
