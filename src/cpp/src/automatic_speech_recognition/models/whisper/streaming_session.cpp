// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "streaming_session.hpp"

#include <algorithm>
#include <cctype>

#include "automatic_speech_recognition/debug_dump.hpp"
#include "automatic_speech_recognition/sliding_window.hpp"
#include "pipeline.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

namespace ov::genai {

namespace {

// Whisper encoder accepts at most 30 seconds of 16 kHz audio.
static constexpr size_t WHISPER_MAX_SAMPLES = 480000;

// UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER — signals a corrupted decode boundary.
static constexpr const char* REPLACEMENT_CHAR_UTF8 = "\xef\xbf\xbd";

// Splits on runs of ASCII whitespace, dropping empty tokens. Safe on UTF-8 text: continuation
// bytes never equal the ASCII space/tab/newline byte values being split on.
std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        size_t start = i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        if (i > start) {
            words.emplace_back(text.substr(start, i - start));
        }
    }
    return words;
}

std::string join_words(const std::vector<std::string>& words) {
    std::string joined;
    for (const auto& word : words) {
        if (!joined.empty()) {
            joined += ' ';
        }
        joined += word;
    }
    return joined;
}

}  // namespace

WhisperASRStreamingSessionImpl::WhisperASRStreamingSessionImpl(WhisperASRPipelineAdapter* pipeline,
                                                               const ASRStreamingConfig& streaming_config,
                                                               const ASRGenerationConfig& generation_config)
    : m_pipeline{pipeline},
      m_streaming_config{streaming_config},
      m_generation_config{generation_config},
      m_chunk_size_samples{static_cast<size_t>(
          std::max(1.0f, streaming_config.chunk_size_sec) * 16000.0f)} {
    OPENVINO_ASSERT(m_pipeline != nullptr, "WhisperASRStreamingSessionImpl: pipeline pointer must not be null");
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

// Trims context_rollback_tokens off the tail of raw (tokenizing it fresh), retrying with a larger
// rollback on a UTF-8 boundary corruption. Same policy as Qwen3-ASR's identically-named helper.
std::string WhisperASRStreamingSessionImpl::trim_rollback(const std::string& raw) const {
    if (raw.empty() || m_chunk_count < m_streaming_config.warmup_chunks) {
        return "";
    }

    const TokenizedInputs encoded = m_pipeline->m_tokenizer.encode(raw);
    const ov::Tensor& ids_tensor = encoded.input_ids;
    const size_t n_tokens = ids_tensor.get_shape()[1];

    size_t rollback = m_streaming_config.context_rollback_tokens;

    while (true) {
        const size_t keep = (n_tokens > rollback) ? n_tokens - rollback : 0;
        if (keep == 0) {
            return "";
        }
        const int64_t* data = ids_tensor.data<const int64_t>();
        const std::vector<int64_t> kept_ids(data, data + keep);
        const std::string trimmed = m_pipeline->m_tokenizer.decode(kept_ids);
        if (trimmed.find(REPLACEMENT_CHAR_UTF8) == std::string::npos) {
            return trimmed;
        }
        ++rollback;
        if (rollback >= n_tokens) {
            return "";
        }
    }
}

std::string WhisperASRStreamingSessionImpl::history_text() const {
    std::string text;
    for (const auto& rec : m_commit_history) {
        if (!text.empty() && !rec.text_delta.empty()) {
            text += ' ';
        }
        text += rec.text_delta;
    }
    return text;
}

void WhisperASRStreamingSessionImpl::decode_current_accum() {
    // Evict commit_history entries whose grounding audio no longer exists in m_audio_accum, tied
    // directly to how far the sliding window has actually rolled (m_total_dropped_samples) rather
    // than a fixed chunk-count margin -- same mechanism as
    // Qwen3ASRStreamingSessionImpl::decode_current_accum().
    if (!m_streaming_config.unbounded_prefix) {
        while (!m_commit_history.empty() &&
               (m_commit_history.front().chunk_index + 1) * m_chunk_size_samples <= m_total_dropped_samples) {
            m_commit_history.pop_front();
        }
    }

    // Captured before this pass runs: the exact tag-free text this pass's prefix was built from,
    // used below as the split point for this pass's own delta. Using this (rather than the full
    // historical m_current_committed_text) is what makes delta computation immune to how
    // aggressively m_commit_history has just been evicted above.
    const std::string prefix_text = history_text();

    // Inject the bounded prefix as forced decoder tokens so the model is anchored to stable prior
    // output. With no prior committed text (warmup or empty history) the prefix is left unset.
    const bool inject_prefix = m_chunk_count >= m_streaming_config.warmup_chunks && !prefix_text.empty();
    ASRGenerationConfig config_for_pass = m_generation_config;
    if (inject_prefix) {
        config_for_pass.prefix = prefix_text;
    }

    if (asr_debug_dump_enabled()) {
        asr_debug_dump_chunk("whisper",
                             m_chunk_count,
                             m_audio_accum,
                             /*sample_rate=*/16000,
                             inject_prefix ? prefix_text : std::string());
    }

    const ASRDecodedResults results = m_pipeline->generate(m_audio_accum, config_for_pass, nullptr);
    OPENVINO_ASSERT(!results.texts.empty(), "WhisperASRStreamingSessionImpl: generate returned empty results");

    m_current_language = results.languages.empty() ? "" : results.languages[0];

    // The decoder only generates the continuation beyond the forced prefix tokens; reconstruct the
    // full transcript by prepending prefix_text so trim_rollback() can operate on it. prefix_text is
    // reproduced verbatim by construction (forced tokens), so candidate texts below always begin
    // with it.
    m_current_text = (inject_prefix ? prefix_text : "") + results.texts[0];
    const size_t this_chunk_index = m_chunk_count;
    ++m_chunk_count;

    const std::string candidate_committed = trim_rollback(m_current_text);

    // This pass's own contribution, relative to the prefix it actually saw -- not a length
    // comparison against the historical total, which broke once eviction made m_commit_history's
    // reconstructed prefix diverge from the ever-growing committed text (a shorter, differently
    // trimmed candidate could still out-length the old total and silently clobber it). Appending a
    // prefix-relative delta is safe by construction: it can only grow.
    std::string delta;
    size_t committed_len_this_pass = prefix_text.size();
    if (candidate_committed.size() > prefix_text.size()) {
        delta = candidate_committed.substr(prefix_text.size());
        committed_len_this_pass = candidate_committed.size();
    }
    if (!delta.empty()) {
        m_commit_history.push_back({this_chunk_index, delta});
        m_current_committed_text += delta;
    }

    // Local to this pass -- independent of m_current_committed_text's (unbounded) length, which
    // can legitimately outgrow m_current_text once older commit_history entries are evicted.
    m_current_partial_text = m_current_text.size() >= committed_len_this_pass
                                 ? m_current_text.substr(committed_len_this_pass)
                                 : m_current_text;
    m_current_new_committed_text = delta;
}

void WhisperASRStreamingSessionImpl::decode_current_accum_agreement() {
    // Evict commit_history entries whose grounding audio has scrolled out of m_audio_accum, same
    // trigger as the Rollback path -- but here the evicted text is preserved (appended to
    // m_evicted_prompt_text) rather than discarded, since Agreement needs it as plain prompt
    // context instead of a forced prefix. unbounded_prefix is intentionally not consulted here:
    // it has no faithful analog once there is no forced prefix to keep unbounded, so Agreement
    // always evicts on the normal schedule.
    while (!m_commit_history.empty() &&
           (m_commit_history.front().chunk_index + 1) * m_chunk_size_samples <= m_total_dropped_samples) {
        if (!m_evicted_prompt_text.empty() && !m_commit_history.front().text_delta.empty()) {
            m_evicted_prompt_text += ' ';
        }
        m_evicted_prompt_text += m_commit_history.front().text_delta;
        m_commit_history.pop_front();
    }

    // Still-in-window committed text (identical to Rollback's prefix_text) -- the sliding window
    // may not yet have scrolled past all of it, so this pass's unprompted re-decode can legitimately
    // reproduce some of it at the front of its output before reaching genuinely new content. Used
    // below to find exactly how much, by content rather than a word count (the window can drop only
    // *part* of a given pass's audio, so only part of that pass's committed words may still be
    // audible -- a count-based skip would over- or under-skip).
    const std::vector<std::string> history_words = split_words(history_text());

    ASRGenerationConfig config_for_pass = m_generation_config;
    if (!m_evicted_prompt_text.empty()) {
        config_for_pass.initial_prompt = m_evicted_prompt_text;
    }
    // No forced prefix under Agreement: every pass re-decodes the window from scratch and lets
    // the two-pass word comparison below decide what is stable.

    if (asr_debug_dump_enabled()) {
        asr_debug_dump_chunk("whisper", m_chunk_count, m_audio_accum, /*sample_rate=*/16000, m_evicted_prompt_text);
    }

    const ASRDecodedResults results = m_pipeline->generate(m_audio_accum, config_for_pass, nullptr);
    OPENVINO_ASSERT(!results.texts.empty(), "WhisperASRStreamingSessionImpl: generate returned empty results");

    m_current_language = results.languages.empty() ? "" : results.languages[0];
    m_current_text = results.texts[0];
    const std::vector<std::string> new_words = split_words(m_current_text);
    const size_t this_chunk_index = m_chunk_count;
    ++m_chunk_count;

    if (this_chunk_index < m_streaming_config.warmup_chunks) {
        // Cold-start: nothing to agree against yet, so nothing commits this pass.
        m_prev_hypothesis_tail_words = new_words;
        m_current_partial_text = m_current_text;
        m_current_new_committed_text = "";
        return;
    }

    // Locate the reproduced-history prefix by content: the longest suffix of history_words that
    // exactly matches a prefix of new_words. Everything up through that overlap is just this
    // pass re-transcribing audio whose text was already committed, not new hypothesis content --
    // comparing it against m_prev_hypothesis_tail_words (which starts *after* that same point,
    // by construction, since it's assigned from the previous pass's own post-overlap tail below)
    // would compare two unrelated spans of text and spuriously reset agreement to 0.
    size_t overlap = std::min(history_words.size(), new_words.size());
    for (; overlap > 0; --overlap) {
        if (std::equal(history_words.end() - overlap, history_words.end(), new_words.begin())) {
            break;
        }
    }
    const std::vector<std::string> fresh_words(new_words.begin() + overlap, new_words.end());

    // Longest word-for-word agreement between this pass's genuinely new hypothesis and the
    // previous pass's still-provisional tail, from the front -- mirrors whisper_streaming's
    // HypothesisBuffer.flush(). Everything up to the first mismatch (or the shorter list's end)
    // is now confirmed by two independent decode passes and safe to commit.
    size_t agreed = 0;
    while (agreed < fresh_words.size() && agreed < m_prev_hypothesis_tail_words.size() &&
           fresh_words[agreed] == m_prev_hypothesis_tail_words[agreed]) {
        ++agreed;
    }

    const std::vector<std::string> committed_words(fresh_words.begin(), fresh_words.begin() + agreed);
    std::vector<std::string> tail_words(fresh_words.begin() + agreed, fresh_words.end());

    if (asr_debug_dump_enabled()) {
        asr_debug_dump_agreement_trace(this_chunk_index, m_evicted_prompt_text,
                                       static_cast<float>(m_total_dropped_samples) / 16000.0f,
                                       static_cast<float>(m_audio_accum.size()) / 16000.0f, history_words,
                                       new_words, overlap, m_prev_hypothesis_tail_words, agreed, tail_words);
    }

    const std::string delta = join_words(committed_words);
    // The leading separator, when present, is part of what's newly appended -- included in
    // new_committed_text so `previous committed_text + new_committed_text` always reconstructs
    // the updated committed_text exactly, as ASRPartialResult's contract promises.
    std::string delta_with_separator = delta;
    if (!delta.empty()) {
        if (!m_current_committed_text.empty()) {
            delta_with_separator = ' ' + delta;
        }
        m_commit_history.push_back({this_chunk_index, delta});
        m_current_committed_text += delta_with_separator;
    }

    // Same separator logic as above: partial_text is meant to be displayed appended directly
    // after committed_text, so it needs its own leading space when both are non-empty.
    std::string partial_text = join_words(tail_words);
    if (!partial_text.empty() && !m_current_committed_text.empty()) {
        partial_text = ' ' + partial_text;
    }
    m_current_partial_text = std::move(partial_text);
    m_current_new_committed_text = delta_with_separator;
    m_prev_hypothesis_tail_words = std::move(tail_words);
}

std::optional<ASRPartialResult> WhisperASRStreamingSessionImpl::push_chunk(const std::vector<float>& pcm16k) {
    if (m_window_full) {
        // The Whisper 30-second window is exhausted; callers should invoke finish().
        return std::nullopt;
    }

    m_buffer.insert(m_buffer.end(), pcm16k.begin(), pcm16k.end());

    if (m_buffer.size() < m_chunk_size_samples) {
        return std::nullopt;
    }

    // m_audio_accum has been fully decoded by the end of every prior push_chunk()/finish()
    // call, so its size right before this drain is exactly how much is safe to drop from.
    const size_t already_inferred_samples = m_audio_accum.size();

    // Drain the buffer, capping at the Whisper maximum window.
    const size_t remaining_capacity = WHISPER_MAX_SAMPLES - m_audio_accum.size();
    const size_t drain = std::min(m_buffer.size(), remaining_capacity);
    m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.begin() + drain);
    m_buffer.erase(m_buffer.begin(), m_buffer.begin() + drain);

    // Drop already-decoded audio outside the sliding window now that the drain is done. This
    // frees capacity against WHISPER_MAX_SAMPLES for the *next* call, so a window configured
    // to stay under 30s means m_window_full is rarely (if ever) reached and streaming is
    // effectively unbounded, not just single-utterance.
    m_total_dropped_samples += apply_sliding_window_drop(m_audio_accum,
                                                         already_inferred_samples,
                                                         m_chunk_size_samples,
                                                         m_streaming_config.window_chunk_num,
                                                         m_streaming_config.window_rollback_chunk_num);

    if (m_audio_accum.size() >= WHISPER_MAX_SAMPLES) {
        m_window_full = true;
        m_buffer.clear();  // discard audio that would exceed the window
    }

    if (m_streaming_config.commit_policy == ASRCommitPolicy::Agreement) {
        decode_current_accum_agreement();
    } else {
        decode_current_accum();
    }
    return ASRPartialResult{m_current_language, m_current_committed_text,
                            m_current_new_committed_text, m_current_partial_text};
}

ASRPartialResult WhisperASRStreamingSessionImpl::finish() {
    m_current_new_committed_text = "";

    if (!m_buffer.empty() && !m_window_full) {
        const size_t already_inferred_samples = m_audio_accum.size();

        const size_t remaining_capacity = WHISPER_MAX_SAMPLES - m_audio_accum.size();
        const size_t drain = std::min(m_buffer.size(), remaining_capacity);
        m_audio_accum.insert(m_audio_accum.end(), m_buffer.begin(), m_buffer.begin() + drain);
        m_buffer.clear();

        m_total_dropped_samples += apply_sliding_window_drop(m_audio_accum,
                                                             already_inferred_samples,
                                                             m_chunk_size_samples,
                                                             m_streaming_config.window_chunk_num,
                                                             m_streaming_config.window_rollback_chunk_num);

        if (!m_audio_accum.empty()) {
            if (m_streaming_config.commit_policy == ASRCommitPolicy::Agreement) {
                decode_current_accum_agreement();
            } else {
                decode_current_accum();
            }
        }
    }

    // Commit any remaining partial tail; final result always has partial_text == "".
    m_current_committed_text += m_current_partial_text;
    m_current_new_committed_text += std::move(m_current_partial_text);
    m_current_partial_text = "";

    return {m_current_language, m_current_committed_text,
            m_current_new_committed_text, m_current_partial_text};
}

}  // namespace ov::genai
