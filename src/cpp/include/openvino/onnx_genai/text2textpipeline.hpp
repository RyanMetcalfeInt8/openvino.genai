// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <filesystem>
#include <string>

#include <pipelines/text2text_pipeline.hpp>
#include <openvino/genai/visibility.hpp>

namespace ov {
namespace genai {

OPENVINO_GENAI_EXPORTS 
std::shared_ptr<onnx::genai::Text2TextPipeline> create_text2text_pipeline(const std::filesystem::path& models_path,
	                                                                      const std::string& device);

}  // namespace genai
}  // namespace ov
