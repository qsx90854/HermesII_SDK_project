#pragma once
#include <cstdlib>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace hermes_person {
class FgReplayWriter {
 bool enabled_=false;std::ofstream index_;
public:
 FgReplayWriter(){auto s=std::getenv("HERMES_PERSON_FG_REPLAY");enabled_=s&&std::string(s)=="1";}
 void Write(int frame,uint64_t stamp,const uint8_t* bg,const uint8_t* mask,int w,int h){
  if(!enabled_)return;
  if(!index_.is_open()){std::filesystem::create_directory("fg_replay");index_.open("fg_replay/index.csv");if(!index_)throw std::runtime_error("Cannot open FG replay index");index_<<"sdk_frame,timestamp_ms,bg,fg,width,height\n";}
  std::string prefix="fg_replay/"+std::to_string(frame),bp=prefix+"_bg.png",fp=prefix+"_fg.png";
  if(!stbi_write_png(bp.c_str(),w,h,1,bg,w)||!stbi_write_png(fp.c_str(),w,h,1,mask,w))throw std::runtime_error("Cannot write BG/FG replay PNG");
  index_<<frame<<','<<stamp<<','<<frame<<"_bg.png,"<<frame<<"_fg.png,"<<w<<','<<h<<'\n';if(!index_)throw std::runtime_error("FG replay index write failed");
 }
 void Finish(){if(index_.is_open()){index_.flush();if(!index_)throw std::runtime_error("FG replay index flush failed");index_.close();}}
};
}
