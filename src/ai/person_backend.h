#pragma once
#include "person_detector.h"
#include <memory>
#include <stdexcept>
#if defined(HERMES_ENABLE_ONNX_PERSON) && defined(HERMES_ENABLE_NPU_PERSON)
#error Select one person backend per build
#endif
#ifdef HERMES_ENABLE_NPU_PERSON
#include "person_detector_npu.h"
#else
#include "person_detector_onnx.h"
#endif
namespace hermes_person {
inline std::unique_ptr<Detector> createDetector(const Config& c){
#ifdef HERMES_ENABLE_NPU_PERSON
 return std::unique_ptr<Detector>(new PersonDetectorNpu(c));
#else
 return std::unique_ptr<Detector>(new PersonDetectorOnnx(c));
#endif
}
// Offline synchronous inference is deliberately kept outside the common contract.
inline Result runOffline(Detector& detector,const uint8_t*p,int w,int h,uint64_t ts,uint64_t id){
#ifdef HERMES_ENABLE_NPU_PERSON
 throw std::runtime_error("NPU backend requires HERMES_PERSON_MODE=async");
#else
 return static_cast<PersonDetectorOnnx&>(detector).RunSync(p,w,h,ts,id);
#endif
}
}
