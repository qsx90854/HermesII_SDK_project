#include "fall/stage_timing.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
int main(){
 unsetenv("HERMES_STAGE_TIMING");
 hermes_timing::Recorder off{"a","b"};{hermes_timing::Frame f(off,0,0);f.next(1);}assert(off.rows.empty());
 setenv("HERMES_STAGE_TIMING","1",1);setenv("HERMES_STAGE_TIMING_MAX_FRAMES","1",1);
 hermes_timing::Recorder r{"outer","inner"};
 {hermes_timing::Frame f(r,7,933);{hermes_timing::Scope s(f,1);volatile int n=0;for(int i=0;i<10000;++i)n+=i;}}
 {hermes_timing::Frame f(r,8,1067);f.next(1);}
 assert(r.rows.size()==1&&r.dropped==1);const auto&row=r.rows.front();assert(row.index==7&&row.stamp==933&&row.calls[1]==1&&row.calls[0]==2);assert(std::abs(row.ms[0]+row.ms[1]-row.total)<1e-8);
 r.finish("stage_test.csv");assert(r.rows.empty()&&r.dropped==0);std::ifstream in("stage_test.csv");std::string line;std::getline(in,line);assert(line=="frame_index,timestamp_ms,total_ms,dropped_frames,outer_ms,outer_calls,inner_ms,inner_calls");std::getline(in,line);assert(line.find("7,933,")==0);assert(!std::getline(in,line));
 {hermes_timing::Frame f(r,9,1200);}assert(r.rows.size()==1);std::remove("stage_test.csv");
 hermes_timing::Recorder counted({"work"},{"pixels","objects"});
 {hermes_timing::Frame f(counted,42,5600);assert(f.active());f.count(0,90000);f.count(1,3);}
 {hermes_timing::Frame f(counted,43,5733);assert(!f.active());f.count(0,999);}
 assert(counted.counts.size()==1&&counted.counts[0][0]==90000&&counted.counts[0][1]==3&&counted.dropped==1);
 counted.finish("counter_test.csv");assert(counted.counts.empty());
 std::ifstream ci("counter_test.csv");std::getline(ci,line);assert(line=="frame_index,timestamp_ms,total_ms,dropped_frames,work_ms,work_calls,pixels,objects");
 std::getline(ci,line);assert(line.substr(line.size()-8)==",90000,3");std::remove("counter_test.csv");
 std::cout<<"PASS: disabled, exclusive nested timing, bounded storage, dropped count, export/reset\n";
}
