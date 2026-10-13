#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace hermes_bg_stability {
struct Row{uint64_t stamp,elapsed;size_t changed,total;bool ready;const char* reason;};
struct Model{
 bool enabled=false,ready=false,started=false;uint64_t window=6000,start=0,last=0,activation=0;int delta=12,edge=40,width=0,height=0;
 std::vector<uint8_t> previous;std::vector<uint64_t> changed_at;std::vector<uint8_t> valid;std::vector<Row> rows;
 static double parameter(const char* key,double def,double lo,double hi){const char* s=std::getenv(key);if(!s)return def;char* e=nullptr;double x=std::strtod(s,&e);if(e==s||*e||!std::isfinite(x)||x<lo||x>hi)throw std::runtime_error(std::string("Invalid ")+key);return x;}
 Model(){const char* s=std::getenv("HERMES_BG_STABILITY");enabled=s&&std::string(s)=="1";if(enabled){window=uint64_t(parameter("HERMES_BG_STABILITY_SECONDS",6,.1,600)*1000);delta=int(parameter("HERMES_BG_STABILITY_DELTA",12,0,255));}}
 void reset(){ready=started=false;start=last=activation=0;width=height=0;previous.clear();changed_at.clear();valid.clear();rows.clear();}
 bool update(uint64_t stamp,const uint8_t* im,const uint8_t* bed,int w,int h,bool bg_available,int excluded_top=0){
  if(!enabled)return true;
  if(!bg_available){rows.push_back({stamp,0,0,0,false,"waiting_initial_bg"});return false;}
  if(!im||!bed||w<=2*edge||h<=2*edge)throw std::runtime_error("BG stability requires valid bed mask and geometry");
  if(!started||w!=width||h!=height||stamp<=last||stamp-last>1000){
   // Timing discontinuity or geometry change requires a new continuous window.
   ready=false;started=true;width=w;height=h;start=last=stamp;activation=0;previous.assign(im,im+size_t(w)*h);changed_at.assign(size_t(w)*h,0);valid.assign(size_t(w)*h,0);
   for(int y=std::max(edge,excluded_top);y<h-edge;++y)for(int x=edge;x<w-edge;++x)valid[y*w+x]=bed[y*w+x]!=0;
  }
  size_t total=0,changed=0;
  if(!ready){for(size_t p=0;p<valid.size();++p)if(valid[p]){
   ++total;if(std::abs(int(im[p])-int(previous[p]))>delta)changed_at[p]=stamp;
   if(changed_at[p]&&stamp-changed_at[p]<window)++changed;
  }
  }else total=std::count(valid.begin(),valid.end(),uint8_t(1));
  uint64_t elapsed=stamp-start;
  if(!ready&&elapsed>=window&&total&&changed*4<total){ready=true;activation=stamp;}
  rows.push_back({stamp,elapsed,changed,total,ready,ready?(activation==stamp?"ready":"ready_latched"):!total?"empty_region":elapsed<window?"warming_up":"too_much_change"});
  if(!ready)std::copy(im,im+previous.size(),previous.begin());last=stamp;return ready;
 }
 void finish(){if(!enabled||rows.empty())return;std::ofstream f("bg_stability.csv"),j("bg_stability_summary.json");if(!f||!j)throw std::runtime_error("Cannot write BG stability log");
  f<<"timestamp_ms,elapsed_ms,changed_pixels,valid_pixels,changed_ratio,ready,reason\n";for(const auto& r:rows)f<<r.stamp<<','<<r.elapsed<<','<<r.changed<<','<<r.total<<','<<(std::string(r.reason)=="ready_latched"?-1:r.total?double(r.changed)/r.total:-1)<<','<<r.ready<<','<<r.reason<<'\n';
  j<<"{\"ready\":"<<(ready?"true":"false")<<",\"activation_timestamp_ms\":"<<activation<<",\"window_ms\":"<<window<<",\"gray_delta\":"<<delta<<",\"edge_px\":"<<edge<<",\"max_changed_ratio_strict\":0.25,\"metric\":\"union of observed adjacent-image changes in trailing window; after ready measurement stops\"}";
  f.close();j.close();if(!f||!j)throw std::runtime_error("BG stability write failed");
 }
};
}
