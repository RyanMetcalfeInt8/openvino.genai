// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <stdexcept>

#include <pipelines/text2image_pipeline.hpp>
#include <pipelines/pipeline_factory.hpp>
#include "openvino/genai/image_generation/text2image_pipeline.hpp"

using namespace onnx::genai;
using namespace onnx::genai::Text2Image;

namespace ov::genai {

class OVGenAIText2ImagePipeline : public Pipeline {
public:
    OVGenAIText2ImagePipeline(const std::filesystem::path& models_path, const std::vector<Device> devices) {
        switch (devices.size()) {
        case 0:
            throw std::invalid_argument("No devices provided for OVGenAI Text2Image pipeline initialization.");
            break;
        case 1:
            break;
        default:
            throw std::invalid_argument("OVGenAI Text2Image pipeline does not have multidevice fallback support.");
            break;
        }
        m_pipeline = std::make_shared<ov::genai::Text2ImagePipeline>(models_path, devices[0].identifier);
    }

    ~OVGenAIText2ImagePipeline() {}

    GenerationResult operator()(const GenerationInput& input) override {
        GenerationResult result;
        ov::Tensor ov_t = m_pipeline->generate(input.text);
        result.tensor = {ov_t.data(), ov_t.get_element_type().size(), ov_t.get_shape()};
        return result;
    }

    onnx::genai::Text2Image::GenerationConfig get_generation_config() const override {
        auto ov_config = m_pipeline->get_generation_config();
        onnx::genai::Text2Image::GenerationConfig config;

        config.image_width = ov_config.width;
        config.image_height = ov_config.height;
        config.num_inference_steps = ov_config.num_inference_steps;
        config.num_images_per_prompt = ov_config.num_images_per_prompt;

        return config;
    }

    void set_generation_config(const onnx::genai::Text2Image::GenerationConfig& config) override {
        auto ov_config = m_pipeline->get_generation_config();

        ov_config.width = config.image_width;
        ov_config.height = config.image_height;
        ov_config.num_inference_steps = config.num_inference_steps;
        ov_config.num_images_per_prompt = config.num_images_per_prompt;

        m_pipeline->set_generation_config(ov_config);
    }


private:
	std::shared_ptr<ov::genai::Text2ImagePipeline> m_pipeline;
};

}  // namespace onnx::genai::Text2Image


namespace {
    struct Registrar {
        Registrar() {
            Text2ImagePipelineFactory::GetInstance().Register(
                "openvino.genai",
                [](const std::filesystem::path& models_path, const std::vector<onnx::genai::Device> devices) {
                    return std::make_shared<ov::genai::OVGenAIText2ImagePipeline>(models_path, devices);
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
