// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/onnx_genai/text2textpipeline.hpp>
#include <openvino/genai/llm_pipeline.hpp>

namespace ov {
namespace genai {

class OpenVINOText2TextPipeline : public onnx::genai::Text2TextPipeline {
public:

    OpenVINOText2TextPipeline(const std::filesystem::path& models_path, const std::string& device);
    virtual ~OpenVINOText2TextPipeline();

	onnx::genai::GenerationResult operator()(const std::string& input) override;

    onnx::genai::GenerationConfig get_generation_config() const override;

    void set_generation_config(const onnx::genai::GenerationConfig& config) override;

private:

	std::shared_ptr<LLMPipeline> m_llm_pipeline;
};

OpenVINOText2TextPipeline::OpenVINOText2TextPipeline(const std::filesystem::path& models_path,
                                                     const std::string& device)
    : m_llm_pipeline(std::make_shared<LLMPipeline>(models_path, device)) {

    //TODO: This should be removed here when there is some API that can better control state
    m_llm_pipeline->start_chat();
}

OpenVINOText2TextPipeline::~OpenVINOText2TextPipeline() {

    // TODO: This should be removed here when there is some API that can better control state
    m_llm_pipeline->finish_chat();
}

onnx::genai::GenerationResult OpenVINOText2TextPipeline::operator()(const std::string& input) {
    auto decoded_results = m_llm_pipeline->generate(input);

    onnx::genai::GenerationResult result;
    result.text = decoded_results;

    return result;
}

onnx::genai::GenerationConfig OpenVINOText2TextPipeline::get_generation_config() const {
    auto ov_config = m_llm_pipeline->get_generation_config();

    onnx::genai::GenerationConfig config;
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
    switch (ov_config.stop_criteria)
    {
        case StopCriteria::EARLY:
            config.beam_search_config.stop_criteria =
                onnx::genai::GenerationConfig::BeamSearchConfig::StopCriteria::EARLY;
            break;

        case StopCriteria::HEURISTIC:
            config.beam_search_config.stop_criteria =
                onnx::genai::GenerationConfig::BeamSearchConfig::StopCriteria::HEURISTIC;
            break;

        case StopCriteria::NEVER:
            config.beam_search_config.stop_criteria =
                onnx::genai::GenerationConfig::BeamSearchConfig::StopCriteria::NEVER;
            break;
    }

    return config;
}

void OpenVINOText2TextPipeline::set_generation_config(const onnx::genai::GenerationConfig& config) {

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
    case onnx::genai::GenerationConfig::BeamSearchConfig::StopCriteria::EARLY:
        ov_config.stop_criteria = StopCriteria::EARLY;
        break;

    case onnx::genai::GenerationConfig::BeamSearchConfig::StopCriteria::HEURISTIC:
        ov_config.stop_criteria = StopCriteria::HEURISTIC;
        break;

    case onnx::genai::GenerationConfig::BeamSearchConfig::StopCriteria::NEVER:
        ov_config.stop_criteria = StopCriteria::NEVER;
        break;
    }

    m_llm_pipeline->set_generation_config(ov_config);
}

std::shared_ptr<onnx::genai::Text2TextPipeline> create_text2text_pipeline(const std::filesystem::path& models_path,
    const std::string& device) {
    return std::make_shared<OpenVINOText2TextPipeline>(models_path, device);
}

}  // namespace genai
}  // namespace ov

// Automatically register the OpenVINO backend with the factory.
namespace {
    struct Registrar {
        Registrar() {
            PipelineFactory::GetInstance().Register("openvino",
                [](const std::filesystem::path& path, const std::string& device) {
                    return std::make_shared<ov::genai::OpenVINOText2TextPipeline>(path, device);
                }
            );
        }
    };
    static Registrar registrar;
}

// This dummy function forces the linker to include this file,
// ensuring the static Registrar object above is initialized.
void link_openvino_backend() {
    // This function can be empty.
}

