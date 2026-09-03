// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ov::genai {

/// Whether OPENVINO_GENAI_ASR_DEBUG_DIR is set. Checked once and cached; set the env var before
/// the process starts (it is not re-read after the first call).
bool asr_debug_dump_enabled();

/// Writes the exact encoder input and decoder text prefix used for one decode pass, so a
/// streaming session's chunk-by-chunk behavior can be inspected offline. No-op if
/// asr_debug_dump_enabled() is false. Writes, under OPENVINO_GENAI_ASR_DEBUG_DIR:
///   <model_tag>_chunk_<chunk_index>_audio.wav   — mono PCM16 @ sample_rate, the encoder input
///   <model_tag>_chunk_<chunk_index>_prefix.txt  — decoder_prefix_text verbatim (may be empty)
void asr_debug_dump_chunk(const std::string& model_tag,
                          size_t chunk_index,
                          const std::vector<float>& audio,
                          size_t sample_rate,
                          const std::string& decoder_prefix_text);

/// Appends one block describing a single Agreement-policy decode pass's word-alignment state to
/// <OPENVINO_GENAI_ASR_DEBUG_DIR>/agreement_trace.log, so the overlap/agreement computation in
/// decode_current_accum_agreement() can be inspected chunk-by-chunk offline. No-op if
/// asr_debug_dump_enabled() is false. Deliberately mirrors run_whisper_streaming_reference.py's
/// --debug-trace output (PROMPT / CONTEXT / "transcribing N seconds from M" lines) so the two logs
/// can be diffed line-by-line: prompt_text is the reference's PROMPT (evicted, prompt-only text),
/// history_words is the reference's CONTEXT (still-in-window committed text, re-decoded and
/// stripped rather than prompted), and window_start_sec/window_duration_sec are the reference's
/// buffer_time_offset/len(audio_buffer)/SAMPLING_RATE.
void asr_debug_dump_agreement_trace(size_t chunk_index,
                                    const std::string& prompt_text,
                                    float window_start_sec,
                                    float window_duration_sec,
                                    const std::vector<std::string>& history_words,
                                    const std::vector<std::string>& new_words,
                                    size_t overlap,
                                    const std::vector<std::string>& prev_tail_words,
                                    size_t agreed,
                                    const std::vector<std::string>& tail_words);

}  // namespace ov::genai
