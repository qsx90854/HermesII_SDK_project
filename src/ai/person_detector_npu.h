#pragma once
#include "person_detector.h"
#include <memory>
namespace hermes_person {
class PersonDetectorNpu final:public Detector {
public:
 explicit PersonDetectorNpu(const Config&);
 ~PersonDetectorNpu() override;
 bool TrySubmit(const uint8_t*,int,int,uint64_t,uint64_t) override;
 bool TryGetResult(Result&) override;
 void Shutdown() override;
private:
 struct Impl;std::unique_ptr<Impl> impl_;
};
}
