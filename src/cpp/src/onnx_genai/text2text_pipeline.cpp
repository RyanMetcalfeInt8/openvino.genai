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

std::shared_ptr<onnx::genai::Text2TextPipeline> create_text2text_pipeline(const std::filesystem::path& models_path,
    const std::string& device) {
    return std::make_shared<OpenVINOText2TextPipeline>(models_path, device);
}

}  // namespace genai
}  // namespace ov