// WINDOWS IMPLEMENTATION ONLY: DLL lifetime, ORT session and worker live here.
// Do not port windows.h/LoadLibraryExW to the edge algorithm layer.
// Pre/postprocessing below documents the Windows reference behavior, not a required
// NPU tensor layout. Linux must adapt its model then return the same Box/Result contract.
#define NOMINMAX
#include <windows.h>
#include "person_detector_onnx.h"
#include <onnxruntime_c_api.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
namespace hermes_person {
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static float overlap(const Box&a,const Box&b){float v=std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1))*std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));return v/std::max(1e-6f,(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-v);}
struct PersonDetectorOnnx::Impl {
 std::string input_name,output_name;
 int input_w=640,input_h=640;
 Config cfg;HMODULE dll=nullptr;const OrtApi* api=nullptr;OrtEnv* env=nullptr;OrtSession* session=nullptr;OrtMemoryInfo* memory=nullptr;
 std::thread worker;std::mutex mutex;std::condition_variable cv;bool stopping=false,busy=false,queued=false,ready=false;
 std::vector<uint8_t> pixels;int width=0,height=0;uint64_t stamp=0,id=0;Result result;
 void check(OrtStatus*s){if(s){std::string error=api->GetErrorMessage(s);api->ReleaseStatus(s);throw std::runtime_error(error);}}
 void release(){if(api){if(memory)api->ReleaseMemoryInfo(memory);if(session)api->ReleaseSession(session);if(env)api->ReleaseEnv(env);}if(dll)FreeLibrary(dll);memory=nullptr;session=nullptr;env=nullptr;dll=nullptr;}
 explicit Impl(const Config&c):cfg(c){
  OrtSessionOptions* options=nullptr;
  try{
   auto path=std::filesystem::u8path(cfg.runtime_path);dll=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
   if(!dll)throw std::runtime_error("Cannot load ONNX Runtime DLL: "+cfg.runtime_path);
   auto entry=reinterpret_cast<const OrtApiBase*(ORT_API_CALL*)()>(GetProcAddress(dll,"OrtGetApiBase"));
   if(!entry || !(api=entry()->GetApi(ORT_API_VERSION)))throw std::runtime_error("ONNX Runtime API version mismatch");
   check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"hermes_person",&env));check(api->CreateSessionOptions(&options));
   check(api->SetIntraOpNumThreads(options,cfg.threads));check(api->SetSessionGraphOptimizationLevel(options,ORT_ENABLE_ALL));
   check(api->CreateSession(env,std::filesystem::u8path(cfg.model_path).c_str(),options,&session));api->ReleaseSessionOptions(options);options=nullptr;
   OrtTypeInfo* type=nullptr;check(api->SessionGetInputTypeInfo(session,0,&type));
   try {
    const OrtTensorTypeAndShapeInfo* info=nullptr;check(api->CastTypeInfoToTensorInfo(type,&info));
    size_t rank=0;check(api->GetDimensionsCount(info,&rank));std::vector<int64_t> dims(rank);check(api->GetDimensions(info,dims.data(),rank));
    ONNXTensorElementDataType element;check(api->GetTensorElementType(info,&element));
    if(rank!=4||dims[0]!=1||dims[1]!=3||dims[2]<=0||dims[3]<=0||dims[2]>4096||dims[3]>4096||element!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)throw std::runtime_error("Expected static float [1,3,H,W] model input");
    input_h=int(dims[2]);input_w=int(dims[3]);
   }catch(...){api->ReleaseTypeInfo(type);throw;}
   api->ReleaseTypeInfo(type);
   check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
   OrtAllocator* allocator=nullptr;check(api->GetAllocatorWithDefaultOptions(&allocator));
   char* name=nullptr;check(api->SessionGetInputName(session,0,allocator,&name));input_name=name;allocator->Free(allocator,name);
   check(api->SessionGetOutputName(session,0,allocator,&name));output_name=name;allocator->Free(allocator,name);

  }catch(...){if(options)api->ReleaseSessionOptions(options);release();throw;}
  worker=std::thread([this]{loop();});
 }
 ~Impl(){stop();release();}
 void stop(){ {std::lock_guard<std::mutex>lock(mutex);stopping=true;}cv.notify_all();if(worker.joinable())worker.join();}
 Result run(const uint8_t* gray,int w,int h,uint64_t ts,uint64_t rid){
  Result r;r.request_id=rid;r.source_timestamp_ms=ts;auto begin=Clock::now();OrtValue* input=nullptr;OrtValue* output=nullptr;OrtTensorTypeAndShapeInfo* shape=nullptr;
  try{
   if(!gray||w<=0||h<=0)throw std::runtime_error("Invalid gray input");
// Packed grayscale is replicated into three equal channels; this is not original
// RGB video. Letterbox fill=114, float32 NCHW /255; input H/W come from the model.
   float ratio=std::min(float(input_w)/w,float(input_h)/h);int nw=std::lround(w*ratio),nh=std::lround(h*ratio),left=(input_w-nw)/2,top=(input_h-nh)/2;
   std::vector<float> tensor(size_t(3)*input_w*input_h,114.f/255);
   // Bilinear resize, same half-pixel coordinate convention as OpenCV.
   for(int y=0;y<nh;++y){float sy=(y+.5f)*h/nh-.5f;int iy=int(std::floor(sy));float fy=sy-iy;int y0=std::clamp(iy,0,h-1),y1=std::clamp(iy+1,0,h-1);
    for(int x=0;x<nw;++x){float sx=(x+.5f)*w/nw-.5f;int ix=int(std::floor(sx));float fx=sx-ix;int x0=std::clamp(ix,0,w-1),x1=std::clamp(ix+1,0,w-1);
     float v=((1-fy)*((1-fx)*gray[y0*w+x0]+fx*gray[y0*w+x1])+fy*((1-fx)*gray[y1*w+x0]+fx*gray[y1*w+x1]));v=std::round(v)/255.f;
     for(int c=0;c<3;++c)tensor[size_t(c)*input_w*input_h+(top+y)*input_w+left+x]=v;
   }}
   int64_t dims[]={1,3,input_h,input_w};check(api->CreateTensorWithDataAsOrtValue(memory,tensor.data(),tensor.size()*sizeof(float),dims,4,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&input));
   const char* names[]={input_name.c_str()};const char* outputs[]={output_name.c_str()};auto infer=Clock::now();check(api->Run(session,nullptr,names,&input,1,outputs,1,&output));r.inference_ms=ms(infer);
   check(api->GetTensorTypeAndShape(output,&shape));size_t rank=0;check(api->GetDimensionsCount(shape,&rank));
   std::vector<int64_t> outdims(rank);check(api->GetDimensions(shape,outdims.data(),rank));
// Supported SDK decoders: [1,6,N] = xywh + two class scores (no objectness),
// and [1,25200,85] = COCO xywh/objectness/classes, retaining person class 0.
// The 20261013 [1,6300,7] model is currently supported ONLY by the Python viewer;
// this C++ decoder intentionally rejects it. Do not use the 6-channel score rule for it.
   bool modern=rank==3&&outdims[0]==1&&outdims[1]==6&&outdims[2]>0,legacy=outdims==std::vector<int64_t>{1,25200,85};
   if(!modern&&!legacy)throw std::runtime_error("Unsupported person model output; expected [1,6,N] or [1,25200,85]");
   float* p=nullptr;check(api->GetTensorMutableData(output,reinterpret_cast<void**>(&p)));std::vector<Box> candidates;
   if(modern){
    const size_t count=size_t(outdims[2]);
    for(size_t i=0;i<count;++i)for(int cls=0;cls<2;++cls){float score=p[(4+cls)*count+i];if(score<cfg.confidence||!std::isfinite(score))continue;
     float cx=p[i],cy=p[count+i],bw=p[2*count+i],bh=p[3*count+i];if(!std::isfinite(cx+cy+bw+bh)||bw<=0||bh<=0)continue;
     candidates.push_back({(cx-bw/2-left)/ratio,(cy-bh/2-top)/ratio,(cx+bw/2-left)/ratio,(cy+bh/2-top)/ratio,score,cls});}
   }else for(size_t i=0;i<25200;++i){auto q=p+i*85;float score=q[4]*q[5];if(score<cfg.confidence||!std::isfinite(score))continue;if(std::max_element(q+5,q+85)!=q+5)continue;
    candidates.push_back({(q[0]-q[2]/2-left)/ratio,(q[1]-q[3]/2-top)/ratio,(q[0]+q[2]/2-left)/ratio,(q[1]+q[3]/2-top)/ratio,score,0});}
// Class-aware NMS; boxes were restored from letterbox to submitted-image pixels.
// Clamping follows NMS. Keep this ordering when checking backend equivalence.
   std::stable_sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.confidence>b.confidence;});
   std::vector<Box> kept;for(auto b:candidates){bool suppress=false;for(auto& k:kept)if(b.class_id==k.class_id&&overlap(b,k)>cfg.nms_iou){suppress=true;break;}if(!suppress){kept.push_back(b);if(kept.size()>=300)break;}}
   for(auto b:kept){b.x1=std::clamp(b.x1,0.f,float(w));b.x2=std::clamp(b.x2,0.f,float(w));b.y1=std::clamp(b.y1,0.f,float(h));b.y2=std::clamp(b.y2,0.f,float(h));if(b.x2>b.x1&&b.y2>b.y1)r.persons.push_back(b);}r.success=true;
  }catch(const std::exception&e){r.error=e.what();}
  if(shape)api->ReleaseTensorTypeAndShapeInfo(shape);if(input)api->ReleaseValue(input);if(output)api->ReleaseValue(output);r.total_ms=ms(begin);return r;
 }
 void loop(){for(;;){std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[this]{return stopping||queued;});if(stopping)return;queued=false;lock.unlock();auto r=run(pixels.data(),width,height,stamp,id);lock.lock();result=std::move(r);ready=true;}}
};
PersonDetectorOnnx::PersonDetectorOnnx(const Config&c):impl_(new Impl(c)){}
PersonDetectorOnnx::~PersonDetectorOnnx()=default;
void PersonDetectorOnnx::Shutdown(){impl_->stop();}
bool PersonDetectorOnnx::TrySubmit(const uint8_t*p,int w,int h,uint64_t t,uint64_t id){std::lock_guard<std::mutex>lock(impl_->mutex);if(impl_->busy||impl_->stopping)return false;impl_->pixels.assign(p,p+size_t(w)*h);impl_->width=w;impl_->height=h;impl_->stamp=t;impl_->id=id;impl_->busy=impl_->queued=true;impl_->cv.notify_one();return true;}
bool PersonDetectorOnnx::TryGetResult(Result&r){std::lock_guard<std::mutex>lock(impl_->mutex);if(!impl_->ready)return false;r=std::move(impl_->result);impl_->ready=impl_->busy=false;return true;}
Result PersonDetectorOnnx::RunSync(const uint8_t*p,int w,int h,uint64_t t,uint64_t id){std::lock_guard<std::mutex>lock(impl_->mutex);if(impl_->busy)throw std::runtime_error("Cannot mix synchronous and asynchronous inference");return impl_->run(p,w,h,t,id);}
}
