// Host-only driver double. Verifies ownership/lifecycle; does NOT emulate NPU accuracy.
#include "ai/person_detector_npu.h"
#include "ai/npu_runtime.h"
#include "ai/face_detector.h"
#include "ai/blaze_face_detect_parser.h"
#include "image_process/image_process.h"
#include <fhhcp/cv.h>
#include <fhhcp/npu.h>
#include <types/vmm_api.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
static int inits=0,exits=0,drains=0,submit_error=0,query_error=0;
static bool inflight=false,complete=false,bad_shape=false,face_model=false;
static const T_TY_TaskInput* saved_input;
static const T_TY_TaskOutput* saved_output;
static std::set<void*> allocations;
namespace VisionSDK {
bool g_cv_sys_initialized=false;
ImageProcess::ImageProcess()=default;
ImageProcess::~ImageProcess()=default;
bool ImageProcess::Resize(const Image&,Image&,int,int,bool,int,int,int,int){return false;}
}
extern "C" {
int32_t TY_CV_SysInit(){return 0;}
int blaze_face_parse(float*,float*,FaceDetection* r,int* n,int,int){*n=1;r[0]={};r[0].x1=10;r[0].y1=20;r[0].x2=30;r[0].y2=40;r[0].score=.9;return 0;}

int32_t TY_NPU_SysInit(){++inits;return 0;}
int32_t TY_NPU_SysExit(){assert(!inflight);++exits;return 0;}
FH_SINT32 FH_SYS_VmmAllocEx64(FH_UINT64*p,FH_VOID**v,const FH_CHAR*,const FH_CHAR*,FH_UINT32 n,FH_UINT32){*v=std::calloc(1,n);*p=(uintptr_t)*v;allocations.insert(*v);return 0;}
FH_SINT32 FH_SYS_VmmAllocEx_Cached64(FH_UINT64*p,FH_VOID**v,const FH_CHAR*a,const FH_CHAR*b,FH_UINT32 n,FH_UINT32 align){return FH_SYS_VmmAllocEx64(p,v,a,b,n,align);}
FH_SINT32 FH_SYS_VmmFreeOne64(FH_UINT64 p){assert(!inflight);assert(allocations.erase((void*)(uintptr_t)p)==1);std::free((void*)(uintptr_t)p);return 0;}
FH_SINT32 FH_SYS_VmmFlushCache64(FH_UINT64,FH_VOID*,FH_UINT32 n){assert(n>0);return 0;}
int32_t TY_NPU_CreateModelFromPhyMem(const T_TY_MemSegmentInfo*,const T_TY_ModelCfgParam*,const T_TY_MemSegmentInfo*,T_TY_ModelDesc*d,TY_NPU_MODEL_HANDLE*h){
 *d={};d->ioDesc.inputNum=d->ioDesc.outputNum=1;d->ioDesc.in[0].type=E_TY_BLOB_IMAGE_WITH_PRE_PROC;
 auto& o=d->ioDesc.out[0];o.type=E_TY_BLOB_DATA;o.tensor.dataType=E_TY_FLOAT;o.tensor.numDims=3;o.tensor.dims[0]=1;o.tensor.dims[1]=bad_shape?1:6300;o.tensor.dims[2]=7;if(face_model){d->ioDesc.outputNum=2;for(int i=0;i<2;++i){auto& t=d->ioDesc.out[i];t.type=E_TY_BLOB_DATA;t.tensor.dataType=E_TY_FLOAT;t.tensor.numDims=1;t.tensor.dims[0]=16;}}
 *h=(void*)1;return 0;
}
int32_t TY_NPU_GetTaskMemSize(TY_NPU_MODEL_HANDLE,const T_TY_TaskCfgParam*,T_TY_MemSegmentInfo*m){*m={};return 0;}
int32_t TY_NPU_CreateTask(TY_NPU_MODEL_HANDLE,const T_TY_TaskCfgParam*,const T_TY_MemSegmentInfo*,TY_NPU_TASK_HANDLE*h){*h=(void*)2;return 0;}
int32_t TY_NPU_ReleaseTask(TY_NPU_TASK_HANDLE){assert(!inflight);return 0;}
int32_t TY_NPU_ReleaseModel(TY_NPU_TASK_HANDLE){assert(!inflight);return 0;}
int32_t TY_NPU_ForwardAsync(TY_NPU_TASK_HANDLE,E_TY_NpuID,uint32_t,const T_TY_TaskInput*i,uint32_t,const T_TY_TaskOutput*o,TY_NPU_NN_HANDLE*h){
 if(submit_error)return submit_error;assert(!inflight);inflight=true;complete=false;saved_input=i;saved_output=o;*h=1;return 0;
}
int32_t TY_NPU_Query(E_TY_NpuID,TY_NPU_NN_HANDLE,E_TY_Bool*s,int32_t block){
 assert(inflight);if(query_error)return query_error;if(block){++drains;complete=true;}
 *s=complete?E_TY_TRUE:E_TY_FALSE;if(complete)inflight=false;return 0;
}
}
int main(int argc,char** argv){
 assert(argc==2);std::ofstream(argv[1],std::ios::binary)<<"test model";
 hermes_person::Config cfg;cfg.model_path=argv[1];
 {
  hermes_person::PersonDetectorNpu detector(cfg);hermes_person::Result result;
  std::vector<uint8_t> gray(800*422,37);
  assert(detector.TrySubmit(gray.data(),800,422,1600,7));
  std::fill(gray.begin(),gray.end(),99);
  auto* input=(uint8_t*)saved_input->dataIn.virAddr;
  assert(input[0]==37&&input[800*422]==37&&input[800*422*3-1]==37);
  assert(saved_input->descIn.img.picHeight==422);
  assert(!detector.TrySubmit(gray.data(),800,422,1733,8));assert(!detector.TryGetResult(result));
  float* out=(float*)saved_output->dataOut.virAddr;float box[]={120,68,80,40,.8,.9,.1};std::memcpy(out,box,sizeof(box));
  complete=true;assert(detector.TryGetResult(result));assert(result.success&&result.request_id==7&&result.source_timestamp_ms==1600&&result.persons.size()==1);
  assert(result.persons[0].y1==120);assert(!detector.TryGetResult(result));
  assert(detector.TrySubmit(gray.data(),800,422,3200,9));std::memset((void*)saved_output->dataOut.virAddr,0,saved_output->dataOut.size);
  complete=true;assert(detector.TryGetResult(result));assert(result.success&&result.persons.empty());
  submit_error=17;bool threw=false;try{detector.TrySubmit(gray.data(),800,422,4800,10);}catch(...){threw=true;}assert(threw&&!hermes_npu::admission().busy());submit_error=0;
  assert(detector.TrySubmit(gray.data(),800,422,4800,11));query_error=18;threw=false;try{detector.TryGetResult(result);}catch(...){threw=true;}assert(threw&&hermes_npu::admission().busy());query_error=0;
  detector.Shutdown();assert(drains==1);assert(!detector.TrySubmit(gray.data(),800,422,6400,12));
 }
 assert(inits==1&&exits==1&&allocations.empty());
 bad_shape=true;bool threw=false;try{hermes_person::PersonDetectorNpu invalid(cfg);}catch(...){threw=true;}assert(threw&&allocations.empty()&&inits==2&&exits==2);
 // Models share one process runtime, only the last release exits it.
 bad_shape=false;{hermes_person::PersonDetectorNpu a(cfg),b(cfg);assert(inits==3&&exits==2);}assert(exits==3&&allocations.empty());
 {
  face_model=true;VisionSDK::FaceDetector face;assert(face.Init(argv[1])==VisionSDK::StatusCode::OK);face_model=false;
  hermes_person::PersonDetectorNpu people(cfg);std::vector<uint8_t> pixels(128*128,42);VisionSDK::Image image;image.data=pixels.data();image.width=image.height=128;image.channels=1;
  face.Demand(100);assert(face.CanSubmit());assert(face.TrySubmit(image,133));
  assert(!people.TrySubmit(pixels.data(),128,128,140,20));
  std::fill(pixels.begin(),pixels.end(),0);assert(((uint8_t*)saved_input->dataIn.virAddr)[0]==42);assert(((uint8_t*)saved_input->dataIn.virAddr)[128*128]==128);
  VisionSDK::FaceDetector::Result fr;assert(!face.TryGetResult(fr));complete=true;assert(face.TryGetResult(fr));assert(fr.success&&fr.timestamp==133&&fr.faces.size()==1);assert(fr.faces[0].y1==88&&fr.faces[0].y2==108);
  face.Demand(200);assert(!face.CanSubmit()); // older waiting YOLO deadline wins
  assert(people.TrySubmit(pixels.data(),128,128,266,21));people.Shutdown();
  assert(face.CanSubmit());assert(face.TrySubmit(image,400));face.ResetAsync();assert(!face.TryGetResult(fr));face.Release();
 }
 assert(allocations.empty()&&!hermes_npu::admission().busy());
 std::cout<<"NPU backend ownership, results, failure and drain tests passed\n";
}
