// CPU API/lifecycle test. The model double never performs hardware inference.
#include "HermesII_sdk.h"
#include "ai/person_detector_onnx.h"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace hermes_person {
struct PersonDetectorOnnx::Impl {};
PersonDetectorOnnx::PersonDetectorOnnx(const Config& c):impl_(new Impl){
 if(c.model_path=="missing.ty")throw std::runtime_error("test: missing model");
}
PersonDetectorOnnx::~PersonDetectorOnnx()=default;
bool PersonDetectorOnnx::TrySubmit(const uint8_t*,int,int,uint64_t,uint64_t){return false;}
bool PersonDetectorOnnx::TryGetResult(Result&){return false;}
void PersonDetectorOnnx::Shutdown(){}
Result PersonDetectorOnnx::RunSync(const uint8_t*,int,int,uint64_t,uint64_t){return {};}
}
using namespace VisionSDK;
void configure(VisionSDK::VisionSDK& sdk,BackgroundControl_v1 bg){
 FallDetection_v3 fall;fall.header={ConfigType::FallDetection_v3,2};
 fall.enable_face_detection=false;
 assert(sdk.SetConfig(&fall)==StatusCode::OK);
 assert(sdk.SetConfig(&bg)==StatusCode::OK);
}
BackgroundStatus_v1 status(VisionSDK::VisionSDK& sdk){BackgroundStatus_v1 s;assert(sdk.GetBackgroundStatus(s)==StatusCode::OK);return s;}
int main(){
 for(auto key:{"HERMES_FILE_OUTPUT","HERMES_STAGE_TIMING","HERMES_TRACK_REPLAY","HERMES_PERSON_TRACE","HERMES_PERSON_FG_REPLAY","HERMES_FALL_REASON_TRACE"})setenv(key,"1",1);
 setenv("HERMES_BG_STABILITY_SECONDS","invalid",1);setenv("HERMES_BG_STABILITY","1",1);
 setenv("HERMES_IDENTITY_A","1",1);setenv("HERMES_FG_GAP_MS","invalid",1);
 std::ofstream("sdk.ini")<<"save_txt=1\nonly_save_raw=1\nevent_record_self_test=1\n";
 {
 VisionSDK::VisionSDK a,b;BackgroundControl_v1 bg;bg.enable_bg_stability=1;bg.bg_stability_window_ms=400;
 configure(a,bg);bg.enable_bg_stability=0;configure(b,bg);
 auto bad=bg;bad.enable_yolo=2;assert(a.SetConfig(&bad)==StatusCode::ERROR_INVALID_INPUT);
 bad=bg;bad.struct_size--;assert(a.SetConfig(&bad)==StatusCode::ERROR_INVALID_INPUT);
 bad=bg;std::memset(bad.yolo_model_path,'x',sizeof(bad.yolo_model_path));assert(a.SetConfig(&bad)==StatusCode::ERROR_INVALID_INPUT);
 bad=bg;bad.enable_yolo=1;assert(a.SetConfig(&bad)==StatusCode::ERROR_INVALID_INPUT);
 assert(a.Init("")==StatusCode::OK);assert(b.Init("")==StatusCode::OK);
 assert(a.SetConfig(&bg)==StatusCode::ERROR_INVALID_INPUT);
 assert(status(a).state==BackgroundState::WaitingInitialBackground);
 assert(status(b).state==BackgroundState::Disabled);
 std::vector<unsigned char> pixels(800*450,80);
 a.SetBedRegion({{100,100},{700,100},{700,400},{100,400}});
 auto feed=[&](VisionSDK::VisionSDK& sdk,uint64_t t){assert(sdk.SetInputMemory(pixels.data(),800,450,1,t)==StatusCode::OK);assert(sdk.ProcessNextFrame()==StatusCode::OK);};
 for(int i=0;i<10;i++){feed(a,i*134);feed(b,i*134);}
 assert(status(a).state==BackgroundState::Stable);assert(status(b).state==BackgroundState::Disabled);
 const auto first=status(a);feed(a,1340);assert(status(a).stable_frame_index==first.stable_frame_index);
 feed(a,0);assert(status(a).state==BackgroundState::Observing);
 a.SetBedRegion({{120,120},{680,120},{680,390},{120,390}});assert(status(a).stable_frame_index==UINT64_MAX);
 assert(a.Release()==StatusCode::OK);assert(!status(a).has_processed_frame);
 assert(a.Init("")==StatusCode::OK);assert(status(a).state==BackgroundState::WaitingInitialBackground);
 for(int i=0;i<10;i++)feed(a,i*134);
 assert(status(a).state==BackgroundState::Stable);a.Release();b.Release();
 VisionSDK::VisionSDK failed;bg.enable_yolo=1;std::strcpy(bg.yolo_model_path,"missing.ty");configure(failed,bg);
 assert(failed.Init("")==StatusCode::ERROR_INIT_FAILED);
 bg.enable_yolo=0;assert(failed.SetConfig(&bg)==StatusCode::OK);assert(failed.Init("")==StatusCode::OK);failed.Release();
 }
 for(const auto& entry:std::filesystem::directory_iterator("."))assert(entry.path().filename()=="sdk.ini");
 std::cout<<"PASS: config validation, instance isolation, explicit environment override, stable latch/reset/reinit, failed model retry, no SDK output files\n";
}
