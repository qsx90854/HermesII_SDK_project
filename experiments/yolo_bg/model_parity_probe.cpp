#include "person_detector_onnx.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){try{if(argc!=7)throw std::runtime_error("model runtime gray width height output");hermes_person::Config c;c.model_path=argv[1];c.runtime_path=argv[2];c.nms_iou=.7f;hermes_person::PersonDetectorOnnx detector(c);int w=std::stoi(argv[4]),h=std::stoi(argv[5]);std::vector<uint8_t> gray(w*h);std::ifstream in(argv[3],std::ios::binary);in.read((char*)gray.data(),gray.size());if(!in)throw std::runtime_error("gray read failed");auto r=detector.RunSync(gray.data(),w,h,1,1);if(!r.success)throw std::runtime_error(r.error);std::ofstream out(argv[6]);out<<"x1,y1,x2,y2,confidence,class_id\n";for(auto b:r.persons)out<<b.x1<<','<<b.y1<<','<<b.x2<<','<<b.y2<<','<<b.confidence<<','<<b.class_id<<'\n';std::cout<<"OK boxes="<<r.persons.size()<<'\n';}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
