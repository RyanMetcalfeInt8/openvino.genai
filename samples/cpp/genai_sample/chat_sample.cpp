// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#if defined(USE_ONNXRUNTIME_GENAI)
#include <ortgenai_text2text_pipeline.hpp>
namespace genai = ort::genai;
#elif defined(USE_OPENVINO_GENAI)
#include <openvino/onnx_genai/text2textpipeline.hpp>
namespace genai = ov::genai;
#endif

int main(int argc, char* argv[]) try {
    if (argc != 2) {
    throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    auto text2text_pipeline = genai::create_text2text_pipeline(models_path);

    auto generation_config = text2text_pipeline->get_generation_config();
    generation_config.sampling_config.do_sample = true;
    generation_config.sampling_config.temperature = 0.7;
    generation_config.sampling_config.top_k = 40;
    generation_config.sampling_config.rng_seed = 42;
    text2text_pipeline->set_generation_config(generation_config);

    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        if (prompt == "quit()")
            break;

        auto results = (*text2text_pipeline)(prompt);
        std::cout << results.text << std::endl;
        std::cout << "\n----------\n"
                        "question:\n";
    }
}
catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    }
    catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    }
    catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}