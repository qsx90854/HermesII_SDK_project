#include "model_runner.h"
#include <iostream>

namespace VisionSDK {

bool ModelRunner::Init(const std::string& model_path) {
    std::cout << "[AI] Loading model from: " << model_path << std::endl;
    // Simulate loading
    return true;
}

bool ModelRunner::Run(const Image& img, std::vector<DetectionResult>& results) {
    std::cout << "[AI] Running inference on image (" << img.width << "x" << img.height << ")" << std::endl;
    // Simulate detection
    results.push_back({1, 0.95f, 10, 10, 100, 100});
    return true;
}

}
