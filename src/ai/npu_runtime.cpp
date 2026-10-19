#include "npu_runtime.h"
#include <fhhcp/npu.h>
#include <cstdlib>
#include <stdexcept>
#include <cstdio>
#include <thread>
namespace hermes_npu {
namespace { unsigned users=0; bool external=false; std::thread::id owner_thread; Admission gate; }
Admission& admission(){return gate;}
void acquire(){
 if(users){if(owner_thread!=std::this_thread::get_id())throw std::runtime_error("NPU clients must share one SDK thread");++users;return;}
 const char* value=std::getenv("HERMES_NPU_EXTERNAL_RUNTIME");external=value&&std::string(value)=="1";
 if(!external){int rc=TY_NPU_SysInit();if(rc)throw std::runtime_error("TY_NPU_SysInit failed: "+std::to_string(rc));}
 owner_thread=std::this_thread::get_id();users=1;
}
void release(){
 if(!users)return;
 if(--users==0&&!external){
  // An uncertain in-flight task must never be freed by SysExit.
  if(gate.busy()){users=1;std::fprintf(stderr,"NPU retained: outstanding task not drained\n");return;}
  int rc=TY_NPU_SysExit();if(rc)std::fprintf(stderr,"TY_NPU_SysExit failed: %d\n",rc);
 }
}
}
