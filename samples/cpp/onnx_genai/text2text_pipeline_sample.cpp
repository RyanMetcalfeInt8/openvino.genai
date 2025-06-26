#include <iostream>
#include <openvino/onnx_genai/text2textpipeline.hpp>

int main(int argc, char* argv[]) try {
    if (argc < 2 || argc > 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    // Default device is CPU; can be overridden by the second argument
    std::string device = (argc == 3) ? argv[2] : "CPU";  // GPU, NPU can be used as well

    auto text2text_pipeline = ov::genai::create_text2text_pipeline(models_path, device);

    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        auto results = (*text2text_pipeline)(prompt);
        std::cout << results.text << std::endl;
        std::cout << "\n----------\n"
                     "question:\n";
    }

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}