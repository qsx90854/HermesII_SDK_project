#include "person_detector_onnx.h"
#include "person_bg_controller.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
int main(int argc,char**argv){try{
 if(argc<4)throw std::runtime_error("probe model runtime shared_input.bin");
 hermes_person::Config c;c.model_path=argv[1];c.runtime_path=argv[2];hermes_person::PersonDetectorOnnx detector(c);
 std::ifstream f(argv[3],std::ios::binary);f.seekg(30*(800*450+16));uint64_t frame,stamp;f.read((char*)&frame,8);f.read((char*)&stamp,8);std::vector<uint8_t> gray(800*450);f.read((char*)gray.data(),gray.size());if(!f)throw std::runtime_error("input read failed");
 auto r=detector.RunSync(gray.data(),800,450,stamp,1);if(!r.success)throw std::runtime_error(r.error);
 std::ofstream out("probe_boxes.csv");out<<"x1,y1,x2,y2,confidence\n";for(auto b:r.persons)out<<b.x1<<','<<b.y1<<','<<b.x2<<','<<b.y2<<','<<b.confidence<<'\n';out.close();
 if(!detector.TrySubmit(gray.data(),800,450,stamp,2)||detector.TrySubmit(gray.data(),800,450,stamp,3))throw std::runtime_error("single request invariant failed");
 auto start=std::chrono::steady_clock::now();while(!detector.TryGetResult(r)){if(std::chrono::steady_clock::now()-start>std::chrono::seconds(10))throw std::runtime_error("async timeout");std::this_thread::sleep_for(std::chrono::milliseconds(5));}
 if(!r.success||r.request_id!=2)throw std::runtime_error("async result failed");
 if(!detector.TrySubmit(gray.data(),800,450,stamp,4))throw std::runtime_error("submit failed");detector.Shutdown();
 std::cout<<"PASS sync/async/single-inflight/shutdown; source_frame="<<frame<<" boxes="<<r.persons.size()<<'\n';
 // Uniform input contains no persons. Verify block weights through real controller logs.
 hermes_person::PersonBgController controller;std::fill(gray.begin(),gray.end(),80);std::vector<uint8_t> bg(gray.size(),0);std::vector<VisionSDK::MotionObject> objects;
 for(uint64_t t=1;t<=30001;t+=100){controller.MaskUnknown(bg.data(),gray.data(),800,450);controller.Step(t,gray.data(),bg.data(),800,450,8,5,objects);}
 controller.Finish();std::cout<<"PASS controller synthetic sequence\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
