#pragma once
// Optional, per-instance wall-clock profiling on the SDK thread. No file IO in
// measured frames; bounded storage is written explicitly during Release/Finish.
#include <array>
#include "../file_output_policy.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>
namespace hermes_timing {
class Recorder {
public:
 static constexpr size_t capacity=32;
 struct Row { uint64_t index=0,stamp=0; double total=0; std::array<double,capacity> ms{}; std::array<unsigned,capacity> calls{}; };
 bool enabled=false;
 std::vector<const char*> names;
 std::vector<Row> rows;
 using Counts=std::array<uint64_t,24>;
 std::vector<const char*> count_names;
 std::vector<Counts> counts;
 size_t limit=4096; uint64_t dropped=0;
 explicit Recorder(std::initializer_list<const char*> labels, std::initializer_list<const char*> counters={}):names(labels),count_names(counters){
  const char* e=std::getenv("HERMES_STAGE_TIMING");enabled=hermes_io::enabled()&&e&&std::strcmp(e,"1")==0;
  if(count_names.size()>24)throw std::runtime_error("Too many workload counters");
  if(names.size()>capacity)throw std::runtime_error("Too many timing stages");
  if(enabled){if(const char*s=std::getenv("HERMES_STAGE_TIMING_MAX_FRAMES")){char*end=nullptr;long n=std::strtol(s,&end,10);if(end==s||*end||n<1||n>100000)throw std::runtime_error("Invalid HERMES_STAGE_TIMING_MAX_FRAMES");limit=size_t(n);}rows.reserve(limit);if(!count_names.empty())counts.reserve(limit);}
 }
 void finish(const char* path){
  if(!enabled||rows.empty())return;
  FILE*f=std::fopen(path,"w");if(!f)throw std::runtime_error("Cannot open stage timing CSV");
  std::fprintf(f,"frame_index,timestamp_ms,total_ms,dropped_frames");
  for(auto n:names)std::fprintf(f,",%s_ms,%s_calls",n,n);
  for(auto n:count_names)std::fprintf(f,",%s",n);
  std::fputc('\n',f);
  for(size_t row_index=0;row_index<rows.size();++row_index){const auto&r=rows[row_index];std::fprintf(f,"%llu,%llu,%.6f,%llu",(unsigned long long)r.index,(unsigned long long)r.stamp,r.total,(unsigned long long)dropped);for(size_t i=0;i<names.size();++i)std::fprintf(f,",%.6f,%u",r.ms[i],r.calls[i]);for(size_t i=0;i<count_names.size();++i)std::fprintf(f,",%llu",(unsigned long long)counts[row_index][i]);std::fputc('\n',f);}
  bool bad=std::ferror(f)!=0;int rc=std::fclose(f);if(bad||rc)throw std::runtime_error("Cannot write stage timing CSV");rows.clear();counts.clear();dropped=0;
 }
};
class Frame {
 using Clock=std::chrono::steady_clock;
 Recorder* recorder=nullptr;Recorder::Row row;Recorder::Counts counts{};Clock::time_point begin,last;
 int current=0;
public:
 Frame(Recorder&r,uint64_t index,uint64_t stamp){if(!r.enabled)return;if(r.rows.size()>=r.limit){++r.dropped;return;}recorder=&r;row.index=index;row.stamp=stamp;row.calls[0]=1;begin=last=Clock::now();}
 bool active()const{return recorder!=nullptr;}
 void count(size_t index,uint64_t value){if(recorder)counts[index]=value;}
 int stage()const{return current;}
 void next(int stage){if(!recorder)return;auto now=Clock::now();row.ms[current]+=std::chrono::duration<double,std::milli>(now-last).count();last=now;current=stage;++row.calls[current];}
 ~Frame(){if(!recorder)return;auto now=Clock::now();row.ms[current]+=std::chrono::duration<double,std::milli>(now-last).count();row.total=std::chrono::duration<double,std::milli>(now-begin).count();recorder->rows.push_back(row);if(!recorder->count_names.empty())recorder->counts.push_back(counts);}
};
// Nested scopes exclude their time from the enclosing stage.
class Scope {Frame& frame;int previous;public:Scope(Frame&f,int stage):frame(f),previous(f.stage()){frame.next(stage);}~Scope(){frame.next(previous);}};
}
