#include "event_recorder.h"
#include <stdexcept>
namespace VisionSDK {
EventRecorder::EventRecorder() { enabled_=false; }
EventRecorder::~EventRecorder() = default;
void EventRecorder::Configure(bool enable,int,int,bool) { if(enable) throw std::runtime_error("Recording unsupported in evaluation build"); enabled_=false; }
void EventRecorder::PushFrame(const Image&) {}
void EventRecorder::OnEvent(const VisionSDKEvent&,std::vector<uint8_t>&&) {}
void EventRecorder::TriggerSelfTest(std::vector<uint8_t>&&) {}
void EventRecorder::RecordProcessTime(uint64_t) {}
void EventRecorder::RecordFrameAnalysis(FrameAnalysis&&) {}
void EventRecorder::SetBedRegion(const std::vector<std::pair<int,int>>&) {}
void EventRecorder::SetMotionConfig(const MotionEstimation_v1&) {}
void EventRecorder::SetObjectConfig(const ObjectExtraction_v1&) {}
void EventRecorder::SetFallConfig(const FallDetection_v3&) {}
void EventRecorder::SetImageConfig(const ImageRelated_v1&) {}
void EventRecorder::Shutdown() {}
void EventRecorder::SetPcOutputBase(const char*) {}
}
