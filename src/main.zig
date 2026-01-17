const std = @import("std");

const whisper = @cImport({
    @cInclude("whisper.h");
});
const libsndfile = @cImport({
    @cInclude("sndfile.h");
});

pub fn main() !void {
    const model_path = "ggml-large-v3-turbo.bin";
    // IMPORTANT: Make sure this WAV file is 16000Hz!
    // Whisper fails or runs slow on 44.1kHz/48kHz without resampling.
    const wav_path = "short.wav";

    // Read wav
    var sfInfo: libsndfile.SF_INFO = undefined;
    const file = libsndfile.sf_open(wav_path, libsndfile.SFM_READ, &sfInfo);
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

    // --- REMOVED THE LOG SILENCER HERE ---

    // Configure Context
    var cparams: whisper.whisper_context_params = whisper.whisper_context_default_params();
    cparams.use_gpu = true; // <--- FORCE METAL GPU USAGE

    const ctx = whisper.whisper_init_from_file_with_params(model_path, cparams);
    if (ctx == null) {
        std.debug.print("Failed to create whisper context\n", .{});
        std.process.exit(1);
    }
    defer whisper.whisper_free(ctx);

    var fparams = whisper.whisper_full_default_params(whisper.WHISPER_SAMPLING_GREEDY);

    // Optimization: 4 threads is usually the sweet spot for M1
    fparams.n_threads = 4;

    fparams.print_realtime = false;
    fparams.print_progress = false;
    fparams.no_timestamps = true;

    // Run
    if (whisper.whisper_full(ctx, fparams, samples.ptr, @intCast(n_samples)) != 0) {
        std.debug.print("failed to call whisper_full\n", .{});
    }

    const n_segments: i32 = whisper.whisper_full_n_segments(ctx);
    for (0..@intCast(n_segments)) |i| {
        const text: [*c]const u8 = whisper.whisper_full_get_segment_text(ctx, @intCast(i));
        std.debug.print("{s}\n", .{text});
    }
}
