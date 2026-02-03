#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include "HermesII_sdk.h"
#include "fhhcp/npu.h"
#include "fhhcp/sys.h"
#include "types/vmm_api.h"
#include <memory>

namespace VisionSDK {

class ImageProcess {
public:
    ImageProcess();
    ~ImageProcess();

    // Resize generic Image (RGB/Gray) to target Width/Height
    // Default output format is same as input? Or generic?
    // User requested "generic resize function".
    // For FaceDetector usage, we need RGB -> Gray(YUV400) or RGB->RGB.
    // Let's support formatting hint or default.
    // For now, let's keep it simple: generic resize.
    // But TY_CV_CvtResize needs format info. 
    // We assume input is RGB (from demo) or NV12.
    // Let's add an explicit method: ResizeRGB to Gray, or generic Resize with options.
    // Given the task, let's make it robust.
    
    // Resize src (assumed RGB_888_PLANAR or INTERLEAVED?)
    // Note: CvtResize takes Planar usually. SW conversion needed if interleaved.
    // Let's assume input is standard VisionSDK::Image (which is usually INTERLEAVED RGB/BGR in memory).
    // Returns status.
    // Convert & Resize (with optional V-Flip)
    bool Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip = false);

private:
   class Impl;
   std::shared_ptr<Impl> pImpl;
};

} // namespace VisionSDK

#endif // IMAGE_PROCESS_H
