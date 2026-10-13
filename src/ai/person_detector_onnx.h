#pragma once
#include "person_detector.h"
#include <memory>
namespace hermes_person {
// Windows backend only. Public headers intentionally contain no ONNX/Windows types.
// Replace this implementation for Linux/NPU; retain the Detector data contract.
// RunSync is an offline test extension, not part of Detector's asynchronous API.
// The controller still names this concrete class; replacing the header alone is insufficient.
class PersonDetectorOnnx final : public Detector {
public:
 explicit PersonDetectorOnnx(const Config&);
 ~PersonDetectorOnnx() override;
 bool TrySubmit(const uint8_t*,int,int,uint64_t,uint64_t) override;
 bool TryGetResult(Result&) override;
 void Shutdown() override;
 Result RunSync(const uint8_t*,int,int,uint64_t,uint64_t); // Offline deterministic mode only.
private:
 struct Impl;std::unique_ptr<Impl> impl_;
};
}
