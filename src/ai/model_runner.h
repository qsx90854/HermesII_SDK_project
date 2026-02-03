#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include "HermesII_sdk.h"
#include <string>
#include <vector>

namespace VisionSDK {

class ModelRunner {
public:
    bool Init(const std::string& model_path);
    bool Run(const Image& img, std::vector<DetectionResult>& results);
};

}

#endif
