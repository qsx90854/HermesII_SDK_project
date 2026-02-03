#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <vector>
#include <string>
#include <cstdint>
#include "HermesII_sdk.h"

// Forward declaration of NPU types to avoid exposing them in public header if possible,
// but for simplicity in this SDK structure, we might need some internal handles if the Impl pattern isn't strictly enforced for everything.
// However, looking at FallDetector, it uses PIMPL. Let's try to keep dependencies minimal or use PIMPL if complex.
// The user request simply said "Create face_detector.h and .cpp".
// I'll stick to a simple class first, referencing HermesII_sdk types.

namespace VisionSDK {

struct FaceROI {
    float x1, y1, x2, y2;
    float score;
};

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    // Initialize the model
    StatusCode Init(const std::string& model_path);

    // Detect faces in the input image (assumes YUV NV12 or similar if direct NPU usage)
    // For this specific task, we'll follow the pattern in ai_ex/main.cpp but adapted.
    // However, FallDetector receives a VisionSDK::Image (likely RGB or Grayscale).
    // The user said: "fall_detecor.cpp流程內加上呼叫此臉部偵測功能" and "目前範例是載入128x128的圖片".
    // We need a Resize function.
    
    // Main detection method
    // Returns 0 on success
    int Detect(const Image& img, std::vector<FaceROI>& faces);

    // Resize function (placeholder as requested)
    // Takes input image and fills output (128x128 pre-allocated internally or passed in?)
    // User said: "function內留空, 先預設會回傳128x128的圖片"
    bool Resize(const Image& src, Image& dst);

private:
   class Impl;
   std::shared_ptr<Impl> pImpl;
};

} // namespace VisionSDK

#endif // FACE_DETECTOR_H
