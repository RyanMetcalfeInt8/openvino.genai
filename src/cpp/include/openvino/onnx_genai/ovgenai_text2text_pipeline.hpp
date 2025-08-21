// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <stdexcept>

#include <pipelines/text2text_pipeline.hpp>
#include <pipelines/pipeline_factory.hpp>
#include <openvino/genai/llm_pipeline.hpp>

using namespace onnx::genai;
using namespace onnx::genai::Text2Text;

namespace ov::genai {

class OVGenAIText2TextPipeline : public Pipeline {
public:
    OVGenAIText2TextPipeline(const std::filesystem::path& models_path, const std::vector<Device> devices) {
        switch (devices.size()) {
        case 0:
            throw std::invalid_argument("No devices provided for OV GenAI Text2Text pipeline initialization.");
            break;
        case 1:
            break;
        default:
            throw std::invalid_argument("OVGenAI text2text pipeline does not have multidevice fallback support.");
            break;
        }
        m_llm_pipeline = std::make_shared<ov::genai::LLMPipeline>(models_path, devices[0].identifier);
        //TODO: This should be removed here when there is some API that can better control state
        m_llm_pipeline->start_chat();
    }

    ~OVGenAIText2TextPipeline() {
        // TODO: This should be removed here when there is some API that can better control state
        m_llm_pipeline->finish_chat();
    }

    GenerationResult operator()(const GenerationInput& input) override {
        GenerationResult result;
        result.text = m_llm_pipeline->generate(input.text);
        return result;
    }

    onnx::genai::Text2Text::GenerationConfig get_generation_config() const override {
        auto ov_config = m_llm_pipeline->get_generation_config();
        onnx::genai::Text2Text::GenerationConfig config;

        config.max_length = ov_config.max_length;
        config.max_new_tokens = ov_config.max_new_tokens;
        config.min_new_tokens = ov_config.min_new_tokens;
        config.eos_token_ids = ov_config.stop_token_ids;
        config.stop_strings = ov_config.stop_strings;

        config.sampling_config.do_sample = ov_config.do_sample;
        config.sampling_config.rng_seed = ov_config.rng_seed;
        config.sampling_config.temperature = ov_config.temperature;
        config.sampling_config.top_k = ov_config.top_k;
        config.sampling_config.top_p = ov_config.top_p;
        config.sampling_config.repetition_penalty = ov_config.repetition_penalty;

        config.beam_search_config.num_beams = ov_config.num_beams;
        config.beam_search_config.num_beam_groups = ov_config.num_beam_groups;
        config.beam_search_config.diversity_penalty = ov_config.diversity_penalty;
        config.beam_search_config.length_penalty = ov_config.length_penalty;
        config.beam_search_config.num_return_sequences = ov_config.num_return_sequences;
        config.beam_search_config.no_repeat_ngram_size = ov_config.no_repeat_ngram_size;

        switch (ov_config.stop_criteria) {
            case StopCriteria::EARLY:
            config.beam_search_config.stop_criteria = onnx::genai::Text2Text::GenerationConfig::BeamSearchConfig::StopCriteria::EARLY;
                break;
            case StopCriteria::HEURISTIC:
                config.beam_search_config.stop_criteria = onnx::genai::Text2Text::GenerationConfig::BeamSearchConfig::StopCriteria::HEURISTIC;
                break;
            case StopCriteria::NEVER:
                config.beam_search_config.stop_criteria = onnx::genai::Text2Text::GenerationConfig::BeamSearchConfig::StopCriteria::NEVER;
                break;
        }

        return config;
    }

    void set_generation_config(const onnx::genai::Text2Text::GenerationConfig& config) override {
        // Initialize the 'base' config as what's already set in llm_pipeline.
        auto ov_config = m_llm_pipeline->get_generation_config();

        ov_config.max_length = config.max_length;
        ov_config.max_new_tokens = config.max_new_tokens;
        ov_config.min_new_tokens = config.min_new_tokens;
        ov_config.stop_token_ids = config.eos_token_ids;
        ov_config.stop_strings = config.stop_strings;

        ov_config.do_sample = config.sampling_config.do_sample;
        ov_config.rng_seed = config.sampling_config.rng_seed;
        ov_config.temperature = config.sampling_config.temperature;
        ov_config.top_k = config.sampling_config.top_k;
        ov_config.top_p = config.sampling_config.top_p;
        ov_config.repetition_penalty = config.sampling_config.repetition_penalty;

        ov_config.num_beams = config.beam_search_config.num_beams;
        ov_config.num_beam_groups = config.beam_search_config.num_beam_groups;
        ov_config.diversity_penalty = config.beam_search_config.diversity_penalty;
        ov_config.length_penalty = config.beam_search_config.length_penalty;
        ov_config.num_return_sequences = config.beam_search_config.num_return_sequences;
        ov_config.no_repeat_ngram_size = config.beam_search_config.no_repeat_ngram_size;

        switch (config.beam_search_config.stop_criteria) {
        case onnx::genai::Text2Text::GenerationConfig::BeamSearchConfig::StopCriteria::EARLY:
            ov_config.stop_criteria = StopCriteria::EARLY;
            break;
        case onnx::genai::Text2Text::GenerationConfig::BeamSearchConfig::StopCriteria::HEURISTIC:
            ov_config.stop_criteria = StopCriteria::HEURISTIC;
            break;
        case onnx::genai::Text2Text::GenerationConfig::BeamSearchConfig::StopCriteria::NEVER:
            ov_config.stop_criteria = StopCriteria::NEVER;
            break;
        }

        m_llm_pipeline->set_generation_config(ov_config);
    }

private:
	std::shared_ptr<ov::genai::LLMPipeline> m_llm_pipeline;
};

}  // namespace onnx::genai::Text2Text


namespace {
    struct Registrar {
        Registrar() {
            Text2TextPipelineFactory::GetInstance().Register(
                "openvino.genai",
                [](const std::filesystem::path& models_path, const std::vector<onnx::genai::Device> devices) {
                    return std::make_shared<ov::genai::OVGenAIText2TextPipeline>(models_path, devices);
                }
            );

            DeviceFactory::GetInstance().Register(
                "openvino.genai",
                []() {
                    std::vector<Device> devices(1);
                    devices[0].identifier = "CPU";
                    return devices;
                }
            );
        }
    };
    static Registrar registrar;
}
