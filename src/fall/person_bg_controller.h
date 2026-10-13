#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include "HermesII_sdk.h"
#include "fall_safety.h"
namespace hermes_person {
// Platform-independent policy in intent: scheduling, delayed-result safeguards,
// person protection, BG alpha updates and history for fall confirmation must survive
// Linux migration. Current .cpp still constructs the Windows backend; see porting guide.
class PersonBgController {
public:
 PersonBgController();~PersonBgController();
 bool enabled() const;
 // Invalid background is represented by current image, never by a synthetic dark floor.
 void MaskUnknown(uint8_t* bg,const uint8_t* now,int w,int h);
// Called per SDK frame on the SDK thread; stamp is source time in milliseconds.
// now/bg are packed full-frame grayscale buffers. BG updates happen only when a
// successful fresh inference result is consumed, not on every call.
 void Step(uint64_t stamp,const uint8_t* now,uint8_t* bg,int w,int h,int cols,int rows,const std::vector<VisionSDK::MotionObject>& objects);
// History boxes are full SDK-frame SOURCE-image coordinates, before motion-bridge
// translation/expansion used for BG protection. Preserve this distinction on Linux.
 const std::deque<hermes_safety::YoloSample>& History() const;
 void Finish();
private:struct Impl;std::unique_ptr<Impl> impl_;
};
}
