#pragma once
#include <cstdint>
#include <map>
#include <limits>
namespace hermes_npu {
// Single SDK-thread admission. Only deadlines are queued, never camera buffers.
// A client must cancel its deadline when disabled, reset or destroyed.
class Admission {
 std::map<const void*,uint64_t> due_;
 const void* active_=nullptr;
public:
 void demand(const void* owner,uint64_t due) { due_.emplace(owner,due); }
 void cancel(const void* owner) { due_.erase(owner); }
 bool eligible(const void* owner) const {
  if(active_ || !due_.count(owner)) return false;
  for(const auto& d:due_) if(d.second<due_.at(owner)) return false;
  return true;
 }
 bool claim(const void* owner) { if(!eligible(owner)) return false;active_=owner;due_.erase(owner);return true; }
 void complete(const void* owner) { if(active_==owner) active_=nullptr; }
 bool busy()const{return active_!=nullptr;}
};
Admission& admission();
// Shared process lifetime for SDK clients. Set HERMES_NPU_EXTERNAL_RUNTIME=1
// before SDK construction if the embedding application owns SysInit/SysExit.
void acquire();
void release();
}
