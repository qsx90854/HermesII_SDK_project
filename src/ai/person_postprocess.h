#pragma once
#include "person_detector.h"
#include <algorithm>
#include <cmath>
namespace hermes_person {
// This .ty embeds LEFT_TOP letterbox, fill=128, size=320. Input boxes are
// model pixels. Both output classes are person evidence; no fall is emitted.
inline std::vector<Box> decodePeople(const float* output,int w,int h,float confidence,float nms) {
 std::vector<Box> candidates,kept;
 const float scale=std::min(320.f/w,320.f/h);
 for(int i=0;i<6300;++i){const float* p=output+7*i;
  bool finite=true;for(int k=0;k<7;++k)finite=finite&&std::isfinite(p[k]);
  if(!finite||p[2]<=0||p[3]<=0)continue;
  int cls=p[6]>p[5]?1:0;float score=p[4]*p[5+cls];if(score<confidence)continue;
  candidates.push_back({p[0]-p[2]/2,p[1]-p[3]/2,p[0]+p[2]/2,p[1]+p[3]/2,score,cls});
 }
 std::stable_sort(candidates.begin(),candidates.end(),[](const Box&a,const Box&b){return a.confidence>b.confidence;});
 std::vector<Box> selected;
 for(auto b:candidates){bool suppressed=false;for(auto a:selected){if(a.class_id!=b.class_id)continue;
  float intersection=std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1))*std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
  float total=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection;
  if(intersection/std::max(total,1e-6f)>nms){suppressed=true;break;}
 }if(!suppressed){selected.push_back(b);if(selected.size()==300)break;}}
 for(auto b:selected){b.x1=std::clamp(b.x1/scale,0.f,float(w));b.x2=std::clamp(b.x2/scale,0.f,float(w));b.y1=std::clamp(b.y1/scale,0.f,float(h));b.y2=std::clamp(b.y2/scale,0.f,float(h));if(b.x2>b.x1&&b.y2>b.y1)kept.push_back(b);}
 return kept;
}
}
