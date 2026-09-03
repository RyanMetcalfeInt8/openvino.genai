// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/debug_dump.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ov::genai {

namespace {

const std::string& debug_dump_dir() {
    static const std::string dir = [] {
        const char* env = std::getenv("OPENVINO_GENAI_ASR_DEBUG_DIR");
        return env ? std::string(env) : std::string();
    }();
    return dir;
}

void write_u32(std::ofstream& out, uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_u16(std::ofstream& out, uint16_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Minimal canonical mono PCM16 WAV writer — no external dependency needed for a debug-only dump.
void write_wav_mono_pcm16(const std::filesystem::path& path, const std::vector<float>& audio, uint32_t sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("asr_debug_dump_chunk: failed to open " + path.string() + " for writing");
    }

    constexpr uint16_t bits_per_sample = 16;
    constexpr uint16_t num_channels = 1;
    const uint32_t num_samples = static_cast<uint32_t>(audio.size());
    const uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const uint16_t block_align = num_channels * bits_per_sample / 8;
    const uint32_t data_bytes = num_samples * block_align;

    out.write("RIFF", 4);
    write_u32(out, 36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(out, 16);  // fmt chunk size
    write_u16(out, 1);   // PCM
    write_u16(out, num_channels);
    write_u32(out, sample_rate);
    write_u32(out, byte_rate);
    write_u16(out, block_align);
    write_u16(out, bits_per_sample);
    out.write("data", 4);
    write_u32(out, data_bytes);

    for (const float sample : audio) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        out.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
    }
}

std::string join_words_with_space(const std::vector<std::string>& words) {
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            out += ' ';
        }
        out += words[i];
    }
    return out;
}

// Renders like Python's repr() of a list of str, e.g. "['Okay,', 'so', 'today']" -- matches the
// style run_whisper_streaming_reference.py's --debug-trace already prints word lists in, so the
// two trace files can be diffed token-for-token instead of just eyeballed.
std::string format_word_list_python_style(const std::vector<std::string>& words) {
    std::string out = "[";
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += '\'';
        out += words[i];
        out += '\'';
    }
    out += ']';
    return out;
}

}  // namespace

bool asr_debug_dump_enabled() {
    return !debug_dump_dir().empty();
}

void asr_debug_dump_chunk(const std::string& model_tag,
                          size_t chunk_index,
                          const std::vector<float>& audio,
                          size_t sample_rate,
                          const std::string& decoder_prefix_text) {
    const std::string& dir = debug_dump_dir();
    if (dir.empty()) {
        return;
    }
    std::filesystem::create_directories(dir);

    char stem[96];
    std::snprintf(stem, sizeof(stem), "%s_chunk_%04zu", model_tag.c_str(), chunk_index);

    write_wav_mono_pcm16(std::filesystem::path(dir) / (std::string(stem) + "_audio.wav"),
                         audio,
                         static_cast<uint32_t>(sample_rate));

    std::ofstream prefix_out(std::filesystem::path(dir) / (std::string(stem) + "_prefix.txt"), std::ios::binary);
    prefix_out << decoder_prefix_text;
}

void asr_debug_dump_agreement_trace(size_t chunk_index,
                                    const std::string& prompt_text,
                                    float window_start_sec,
                                    float window_duration_sec,
                                    const std::vector<std::string>& history_words,
                                    const std::vector<std::string>& new_words,
                                    size_t overlap,
                                    const std::vector<std::string>& prev_tail_words,
                                    size_t agreed,
                                    const std::vector<std::string>& tail_words) {
    const std::string& dir = debug_dump_dir();
    if (dir.empty()) {
        return;
    }
    std::filesystem::create_directories(dir);

    // Truncate on this process's first write so each run starts a fresh log instead of appending
    // after whatever an earlier run left behind; append for every write after that within the run.
    static bool first_write_this_process = true;
    const std::ios::openmode mode =
        std::ios::binary | (first_write_this_process ? std::ios::trunc : std::ios::app);
    first_write_this_process = false;

    std::ofstream out(std::filesystem::path(dir) / "agreement_trace.log", mode);

    const std::vector<std::string> fresh_words(new_words.begin() + overlap, new_words.end());
    const std::vector<std::string> committed_words(fresh_words.begin(), fresh_words.begin() + agreed);

    char transcribing_line[96];
    std::snprintf(transcribing_line, sizeof(transcribing_line), "transcribing %.2f seconds from %.2f",
                  window_duration_sec, window_start_sec);

    out << "=== chunk " << chunk_index << " ===\n"
        << "DEBUG\tPROMPT: " << prompt_text << "\n"
        << "DEBUG\tCONTEXT: " << join_words_with_space(history_words) << "\n"
        << "DEBUG\t" << transcribing_line << "\n"
        << "[trace] raw new words (decode) (" << new_words.size()
        << "): " << format_word_list_python_style(new_words) << "\n"
        << "[trace] fresh_words after overlap strip, overlap=" << overlap << " (" << fresh_words.size()
        << "): " << format_word_list_python_style(fresh_words) << "\n"
        << "[trace] prev_tail_words before agreement (" << prev_tail_words.size()
        << "): " << format_word_list_python_style(prev_tail_words) << "\n"
        << "[trace] committed this pass, agreed=" << agreed << " (" << committed_words.size()
        << "): " << format_word_list_python_style(committed_words) << "\n"
        << "[trace] tail_words carried forward (" << tail_words.size()
        << "): " << format_word_list_python_style(tail_words) << "\n\n";
}

}  // namespace ov::genai
