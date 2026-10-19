#include "person_detector_npu.h"
#include "person_postprocess.h"
#include "npu_runtime.h"
#include <fhhcp/npu.h>
#include <fhhcp/sys.h>
#include <types/vmm_api.h>
#include <chrono>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cstdio>
namespace hermes_person {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void check(int rc,const char* what){if(rc)throw std::runtime_error(std::string(what)+": "+std::to_string(rc));}
void flush(T_TY_Mem& m){check(FH_SYS_VmmFlushCache64(m.phyAddr,(void*)m.virAddr,m.size),"NPU cache flush");}
void freeMem(T_TY_Mem& m){if(m.phyAddr)FH_SYS_VmmFreeOne64(m.phyAddr);m={};}
void alloc(T_TY_Mem& m,uint32_t size,E_TY_MemAllocType type=E_TY_MEM_VMM_CACHED){
 FH_UINT64 physical=0;void* virtual_address=nullptr;
 int rc=type==E_TY_MEM_VMM_NO_CACHED?FH_SYS_VmmAllocEx64(&physical,&virtual_address,"NPU","anonymous",size,128):FH_SYS_VmmAllocEx_Cached64(&physical,&virtual_address,"NPU_CACHED","anonymous",size,128);
 m.phyAddr=physical;m.virAddr=reinterpret_cast<uintptr_t>(virtual_address);
 check(rc,"NPU allocation");m.size=size;
 if(!m.phyAddr||!m.virAddr)throw std::runtime_error("Invalid MMZ allocation");
}
void allocSegments(T_TY_MemSegmentInfo& s){for(int i=0;i<s.segNum;++i)alloc(s.memInfo[i].mem,s.memInfo[i].allocInfo.size,s.memInfo[i].allocInfo.allocType);}
void freeSegments(T_TY_MemSegmentInfo& s){for(int i=0;i<s.segNum;++i)freeMem(s.memInfo[i].mem);s={};}
}
struct PersonDetectorNpu::Impl {
 Config config;T_TY_MemSegmentInfo model_mem{},task_mem{};T_TY_ModelDesc desc{};
 TY_NPU_MODEL_HANDLE model=nullptr;TY_NPU_TASK_HANDLE task=nullptr;TY_NPU_NN_HANDLE nn{};
 T_TY_TaskInput input{};T_TY_TaskOutput output{};
 bool runtime=false,pending=false,stopped=false;int width=0,height=0;Result result;
 Clock::time_point started,submitted;
 explicit Impl(const Config& c):config(c){try{
  hermes_npu::acquire();runtime=true;
  std::ifstream file(c.model_path,std::ios::binary|std::ios::ate);
  if(!file)throw std::runtime_error("Cannot open person .ty: "+c.model_path);
  auto size=file.tellg();if(size<=0||size>128*1024*1024)throw std::runtime_error("Invalid person model size");
  model_mem.segNum=1;auto& a=model_mem.memInfo[0].allocInfo;a.alignByteSize=128;a.allocType=E_TY_MEM_VMM_CACHED;a.shareType=E_MEM_EXCLUSIVED;a.size=uint32_t(size);
  allocSegments(model_mem);file.seekg(0);if(!file.read((char*)model_mem.memInfo[0].mem.virAddr,size))throw std::runtime_error("Short person model read");flush(model_mem.memInfo[0].mem);
  T_TY_ModelCfgParam mc{};check(TY_NPU_CreateModelFromPhyMem(&model_mem,&mc,nullptr,&desc,&model),"Create person model");
  if(desc.ioDesc.inputNum!=1||desc.ioDesc.outputNum!=1||desc.ioDesc.in[0].type!=E_TY_BLOB_IMAGE_WITH_PRE_PROC)throw std::runtime_error("Expected one AIPP image input and one output");
  const auto& out=desc.ioDesc.out[0];const auto& t=out.tensor;
  if(out.type!=E_TY_BLOB_DATA||t.dataType!=E_TY_FLOAT||t.numDims!=3||t.dims[0]!=1||t.dims[1]!=6300||t.dims[2]!=7)throw std::runtime_error("Expected float32 [1,6300,7] people output");
  T_TY_TaskCfgParam tc{};check(TY_NPU_GetTaskMemSize(model,&tc,&task_mem),"Person task memory");allocSegments(task_mem);
  check(TY_NPU_CreateTask(model,&tc,&task_mem,&task),"Create person task");alloc(output.dataOut,6300*7*sizeof(float));
 }catch(...){cleanup();throw;}}
 bool drain(){if(!pending)return true;E_TY_Bool done=E_TY_FALSE;int rc=TY_NPU_Query(E_TY_NPU_ID_0,nn,&done,1);if(rc||done!=E_TY_TRUE){std::fprintf(stderr,"Person NPU drain failed (%d); retaining live resources\n",rc);return false;}pending=false;hermes_npu::admission().complete(this);return true;}
 void cleanup(){hermes_npu::admission().cancel(this);if(!drain())return;
  if(task)TY_NPU_ReleaseTask(task);
  task=nullptr;
  if(model)TY_NPU_ReleaseModel(model);
  model=nullptr;
  freeMem(input.dataIn);freeMem(output.dataOut);freeSegments(task_mem);freeSegments(model_mem);if(runtime){hermes_npu::release();runtime=false;}}
 ~Impl(){cleanup();}
 bool submit(const uint8_t* gray,int w,int h,uint64_t ts,uint64_t id){
  if(stopped||pending)return false;
  if(!gray||w<=0||h<=0||w>4096||h>4096)throw std::runtime_error("Invalid NPU gray input");
  auto& gate=hermes_npu::admission();gate.demand(this,ts);if(!gate.claim(this))return false;
  started=Clock::now();result={};result.request_id=id;result.source_timestamp_ms=ts;
  try{
   if(w!=width||h!=height){freeMem(input.dataIn);width=height=0;alloc(input.dataIn,uint32_t(w*h*3));width=w;height=h;}
   input.descIn={};input.descIn.type=E_TY_BLOB_IMAGE;auto& im=input.descIn.img;
   im.picFormat=E_TY_PIXEL_FORMAT_RGB_888_PLANAR;im.picWidth=im.picWidthStride=w;im.picHeight=im.picHeightStride=h;im.roi.x=im.roi.y=0;im.roi.width=w;im.roi.height=h;
   size_t plane=size_t(w)*h;for(int c=0;c<3;++c)std::memcpy((uint8_t*)input.dataIn.virAddr+c*plane,gray,plane);flush(input.dataIn);
   submitted=Clock::now();check(TY_NPU_ForwardAsync(task,E_TY_NPU_ID_0,1,&input,1,&output,&nn),"Submit person NPU");pending=true;return true;
  }catch(...){gate.complete(this);throw;}
 }
 bool poll(Result& out){if(!pending)return false;E_TY_Bool done=E_TY_FALSE;int rc=TY_NPU_Query(E_TY_NPU_ID_0,nn,&done,0);
  // Query errors do not prove that DMA has stopped. Fail closed, keeping ownership.
  if(rc)throw std::runtime_error("Person NPU query failed; explicit shutdown required: "+std::to_string(rc));
  if(done!=E_TY_TRUE)return false;
  pending=false;hermes_npu::admission().complete(this);result.inference_ms=elapsed(submitted);
  try{flush(output.dataOut);result.persons=decodePeople((const float*)output.dataOut.virAddr,width,height,config.confidence,config.nms_iou);result.success=true;}
  catch(const std::exception& e){result.success=false;result.error=e.what();}
  result.total_ms=elapsed(started);out=std::move(result);return true;
 }
};
PersonDetectorNpu::PersonDetectorNpu(const Config&c):impl_(new Impl(c)){}
PersonDetectorNpu::~PersonDetectorNpu()=default;
bool PersonDetectorNpu::TrySubmit(const uint8_t*p,int w,int h,uint64_t ts,uint64_t id){return impl_->submit(p,w,h,ts,id);}
bool PersonDetectorNpu::TryGetResult(Result&r){return impl_->poll(r);}
void PersonDetectorNpu::Shutdown(){impl_->stopped=true;hermes_npu::admission().cancel(impl_.get());if(!impl_->drain())throw std::runtime_error("Cannot drain person NPU");impl_->cleanup();}
}
