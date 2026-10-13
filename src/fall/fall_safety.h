#pragma once
#include "../ai/person_detector.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <map>
#include <set>
#include <fstream>
#include <string>
#include <vector>
namespace hermes_safety {
inline bool flag(const char* key){auto p=std::getenv(key);return p&&std::string(p)=="1";}
using Box=hermes_person::Box;
inline float area(Box b){return std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);}
inline float overlap(Box a,Box b){return std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1))*std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));}
inline float shift(Box a,Box b){return std::hypot((a.x1+a.x2-b.x1-b.x2)*.5f,(a.y1+a.y2-b.y1-b.y2)*.5f);}
inline bool person_matches(Box candidate,Box person){return area(candidate)>0&&area(person)>0&&overlap(candidate,person)/area(candidate)>=.5f&&overlap(candidate,person)/area(person)>=.15f;}
struct YoloSample{uint64_t source=0,applied=0,id=0;bool success=false;std::vector<Box> persons;};
struct TrackState {
 Box box,anchor,stationary_box,fg_box;uint64_t fg_stamp=0;bool measured=false,quiet=false,departed=false;int quiet_run=0,quiet_count=0,valid_count=0,fg=-1;uint64_t last_measured=0,stationary_since=0,observation_since=0,born=0,last_seen=0;
 std::vector<int> support;std::vector<uint8_t> appearance;std::string evidence="unknown",confirmation="none";
 uint64_t pending_since=0,evidence_source=0;Box pending_box;bool accepted=false,rejected=false;
};
class Guard {
public:
 bool history_two=flag("HERMES_PERSON_HISTORY_TWO");
 bool coast=flag("HERMES_COAST_VALIDITY"),confirm=flag("HERMES_FALL_PERSON_CONFIRM"),exclude=flag("HERMES_EXCLUDE_TOP_ROW"),trace=flag("HERMES_TRACK_REPLAY");
 std::map<int,TrackState> tracks;std::set<int> ghosts;std::map<int,int> associations;std::ofstream objects,fg_log,events,checks;
 std::vector<uint8_t> previous;uint64_t last=0,stamp=0;int width=0,height=0,top=0;
 void finish(){for(auto*f:{&objects,&fg_log,&events,&checks})if(f->is_open()){f->flush();if(!*f)throw std::runtime_error("Safety trace write failed");f->close();}}
 void reset(){finish();tracks.clear();ghosts.clear();associations.clear();previous.clear();last=stamp=0;}
 void event(int id,const std::string& status){if(!trace)return;if(!events.is_open()){events.open("fall_confirmation.csv");events<<"timestamp_ms,track_id,status,yolo_source_ms,candidate_x1,candidate_y1,candidate_x2,candidate_y2\n";}auto& q=tracks[id];events<<stamp<<','<<id<<','<<status<<','<<q.evidence_source<<','<<q.pending_box.x1<<','<<q.pending_box.y1<<','<<q.pending_box.x2<<','<<q.pending_box.y2<<'\n';}
 template<class O> static Box bounds(const O&o,int w,int h,int cols,int rows){Box b{float(w),float(h),0,0,0};for(int i:o.blocks){int c=i%cols,r=i/cols;b.x1=std::min(b.x1,float(c*(w/cols)));b.y1=std::min(b.y1,float(r*(h/rows)));b.x2=std::max(b.x2,float(c==cols-1?w:(c+1)*(w/cols)));b.y2=std::max(b.y2,float(r==rows-1?h:(r+1)*(h/rows)));}return b;}
 template<class Objects,class Fgs> void prepare(uint64_t ts,const uint8_t* image,const std::vector<uint8_t>&mask,int w,int h,int cols,int rows,const Objects& os,const Fgs&fgs){
  if(!coast&&!confirm&&!trace)return;
  if(width!=w||height!=h||ts<=last||(last&&ts-last>1000)){tracks.clear();previous.clear();}
  width=w;height=h;top=exclude?h/rows:0;stamp=ts;associations.clear();
  for(auto&o:os){auto&s=tracks[o.id];s.last_seen=ts;s.box=bounds(o,w,h,cols,rows);s.measured=!o.is_coasting&&!ghosts.count(o.id);s.quiet=false;s.departed=false;s.fg=-1;
   if(s.measured){
    if(s.last_measured&&shift(s.box,s.stationary_box)>std::max(40.f,.5f*std::hypot(s.stationary_box.x2-s.stationary_box.x1,s.stationary_box.y2-s.stationary_box.y1))){s.stationary_since=ts;s.stationary_box=s.box;s.support.clear();s.quiet_run=0;s.departed=true;if(!history_two)s.accepted=false;}
    if(!s.stationary_since){s.stationary_since=ts;s.stationary_box=s.box;}
    s.last_measured=ts;s.anchor=s.box;s.support.clear();s.appearance.clear();
    if(mask.size()==size_t(w*h))for(int y=std::max(top,int(s.box.y1));y<std::min(h,int(s.box.y2));y+=4)for(int x=std::max(0,int(s.box.x1));x<std::min(w,int(s.box.x2));x+=4){int i=y*w+x;if(mask[i]){s.support.push_back(i);s.appearance.push_back(image[i]);}}
    s.evidence="measured";
   }else{
    int supported=0,same=0,changed=0;for(size_t k=0;k<s.support.size();++k){int i=s.support[k];if(mask.size()==size_t(w*h)&&mask[i])++supported;if(std::abs(int(image[i])-int(s.appearance[k]))<=12)++same;if(previous.size()==size_t(w*h)&&std::abs(int(image[i])-int(previous[i]))>12)++changed;}
    float n=float(s.support.size());bool valid=n>=16&&s.last_measured&&ts-s.last_measured<=12000&&previous.size()==size_t(w*h);
    bool exists=valid&&supported>=.5f*n&&same>=.65f*n;
    if(valid&&!exists){s.departed=true;s.stationary_since=ts;if(!history_two)s.accepted=false;}
    bool competing=false;for(auto&other:os)if(other.id!=o.id&&!other.is_coasting&&!ghosts.count(other.id)&&overlap(s.anchor,bounds(other,w,h,cols,rows))>.25f*std::max(1.f,area(s.anchor)))competing=true;
    bool stable=exists&&changed<=.2f*n&&!competing;
    s.quiet_run=stable?s.quiet_run+1:0;s.quiet=s.quiet_run>=3;
    s.evidence=competing?"ownership_conflict":!valid?"unknown":!exists?"support_lost":s.quiet?"image_quiet":"image_changing";
   }
  }
  for(auto it=tracks.begin();it!=tracks.end();)if(ts-it->second.last_seen>30000)it=tracks.erase(it);else ++it;
  previous.assign(image,image+size_t(w)*h);last=ts;
 }
 bool eligible(int id){auto it=tracks.find(id);return !coast||(it!=tracks.end()&&(it->second.measured||it->second.quiet));}
 bool measured(int id){auto it=tracks.find(id);return !coast||(it!=tracks.end()&&it->second.measured);}
 bool decelerated(int id,float momentum){return !coast?momentum<4:tracks[id].measured?momentum<4:tracks[id].quiet;}
 // Latest valid result wins. Failed inference is unknown, never a negative observation.
 static int personEvidence(const std::deque<YoloSample>& history,Box b,uint64_t earliest,uint64_t now,uint64_t& source){
  for(auto it=history.rbegin();it!=history.rend();++it){if(!it->success||it->source<earliest||it->source>now||now-it->source>10000)continue;source=it->source;for(auto p:it->persons)if(person_matches(b,p))return 1;return 0;}return -1;
 }
 // Historical evidence is evaluated only at candidate creation. No appearance-time cutoff.
 bool checkPerson(int id,const YoloSample& y,const char* stage){
  auto&s=tracks[id];bool matched=false;
  if(trace&&!checks.is_open()){checks.open("person_confirmation_checks.csv");checks<<"timestamp_ms,track_id,stage,request_id,source_ms,applied_ms,persons,class_id,confidence,candidate_overlap,person_overlap,matched\n";}
  if(y.persons.empty()&&trace)checks<<stamp<<','<<id<<','<<stage<<','<<y.id<<','<<y.source<<','<<y.applied<<",0,-1,0,0,0,0\n";
  for(auto b:y.persons){float inter=overlap(s.pending_box,b);bool ok=person_matches(s.pending_box,b);matched|=ok;
   if(trace)checks<<stamp<<','<<id<<','<<stage<<','<<y.id<<','<<y.source<<','<<y.applied<<','<<y.persons.size()<<','<<b.class_id<<','<<b.confidence<<','<<inter/std::max(1.f,area(s.pending_box))<<','<<inter/std::max(1.f,area(b))<<','<<ok<<'\n';}
  return matched;
 }
 template<class Objects> void filterHistoryTwo(uint64_t ts,Objects&os,bool&fall,const std::deque<YoloSample>&history){
  stamp=ts;fall=false;std::set<int> seen;
  for(auto&o:os){seen.insert(o.id);auto&s=tracks[o.id];bool raw=o.is_fall_this_frame;o.is_fall_this_frame=false;
   if(!raw&&!s.pending_since){if(s.accepted){s.accepted=false;s.confirmation="none";}continue;}
   if(s.rejected){s.confirmation="rejected_wait_new_trigger";continue;}
   bool created=!s.pending_since&&!s.accepted;
   if(created){s.pending_since=ts;s.evidence_source=0;s.pending_box=(s.fg_stamp&&ts>=s.fg_stamp&&ts-s.fg_stamp<=1000&&area(s.fg_box)>0)?s.fg_box:s.box;
    if(area(s.pending_box)<=0){s.pending_since=0;s.rejected=true;s.confirmation="cancel_invalid_candidate";event(o.id,s.confirmation);continue;}
    s.confirmation="pending_person";event(o.id,s.confirmation);int count=0;
    // Evaluate both even when the newest is negative; record both for replay.
    for(auto it=history.rbegin();it!=history.rend()&&count<2;++it){if(!it->success||it->applied>ts||it->source>ts)continue;++count;
     if(checkPerson(o.id,*it,"history")){s.accepted=true;if(!s.evidence_source)s.evidence_source=it->source;}}
    if(s.accepted){s.pending_since=0;s.confirmation="person_confirmed_history";event(o.id,s.confirmation);}
   }
   if(!created&&s.pending_since){
    // First successful result captured after the candidate, not an older in-flight image.
    for(auto& y:history){if(!y.success||y.applied>ts||y.source<s.pending_since||y.source>ts)continue;
     s.evidence_source=y.source;if(checkPerson(o.id,y,"next")){s.accepted=true;s.confirmation="person_confirmed_next";}
     else{s.rejected=true;s.confirmation="cancel_yolo_no_person";}s.pending_since=0;event(o.id,s.confirmation);break;}
    if(s.pending_since&&ts-s.pending_since>=10000){s.pending_since=0;s.rejected=true;s.confirmation="cancel_yolo_timeout";event(o.id,s.confirmation);}
   }
   if(s.accepted){o.is_fall_this_frame=true;fall=true;}
  }
  for(auto&[id,s]:tracks)if(s.pending_since&&!seen.count(id)){s.pending_since=0;s.rejected=true;s.confirmation="cancel_track_lost";event(id,s.confirmation);}
 }
 template<class Objects> void filter(uint64_t ts,Objects&os,bool&fall,const std::deque<YoloSample>&history){
  if(!confirm)return;if(history_two){filterHistoryTwo(ts,os,fall,history);return;}stamp=ts;fall=false;
  std::set<int> seen;
  for(auto&o:os){seen.insert(o.id);auto&s=tracks[o.id];bool raw=o.is_fall_this_frame;o.is_fall_this_frame=false;
   if(!raw&&!s.pending_since){if(s.accepted){s.accepted=false;s.confirmation="none";}continue;}
   if(s.rejected){s.confirmation="rejected_wait_new_trigger";continue;}
   if(s.departed||area(s.box)<=0){s.pending_since=0;s.accepted=false;s.rejected=true;s.confirmation="cancel_support_or_departure";event(o.id,s.confirmation);continue;}
   if(!s.pending_since&&!s.accepted){s.pending_since=ts;s.pending_box=(s.fg_stamp&&ts-s.fg_stamp<=1000&&area(s.fg_box)>0)?s.fg_box:s.box;s.confirmation="pending_person";event(o.id,s.confirmation);}
   if(s.pending_since&&shift(s.pending_box,s.box)>std::max(40.f,.5f*std::hypot(s.pending_box.x2-s.pending_box.x1,s.pending_box.y2-s.pending_box.y1))){s.rejected=true;s.pending_since=0;s.confirmation="cancel_displacement";event(o.id,s.confirmation);continue;}
   if(!s.accepted){uint64_t source=0;int ev=personEvidence(history,s.pending_box,std::max(s.stationary_since,s.observation_since),ts,source);s.evidence_source=source;
    if(ev==1){s.accepted=true;s.pending_since=0;s.confirmation="person_confirmed";event(o.id,s.confirmation);}
    else if((ev==0&&source>=s.pending_since)||ts-s.pending_since>=10000){s.rejected=true;s.pending_since=0;s.confirmation=ev==0?"cancel_yolo_no_person":"cancel_yolo_timeout";event(o.id,s.confirmation);}
   }
   if(s.accepted){o.is_fall_this_frame=true;fall=true;}
  }
  for(auto&[id,s]:tracks)if(s.pending_since&&!seen.count(id)){s.pending_since=0;s.rejected=true;s.confirmation="cancel_track_lost";event(id,s.confirmation);}
 }
 template<class Objects,class Fgs,class States> void write(int frame,const Objects&os,const Fgs&fgs,const States&states,int cols,int rows,int observation_target=45){
  if(!trace||width<=0||height<=0)return;
  if(!objects.is_open()){objects.open("track_replay.csv");objects<<"sdk_frame,timestamp_ms,track_id,x1,y1,x2,y2,origin,momentum_valid,strength,fg_id,fg_checked,evidence,phase,observed,measured_samples,quiet_samples,confirmation,fall,blocks,observation_target\n";fg_log.open("fg_objects.csv");fg_log<<"sdk_frame,fg_id,x1,y1,x2,y2,cx,cy,pixels\n";}
  for(auto&f:fgs){int x1=width,y1=height,x2=0,y2=0;for(int p:f.pixels){x1=std::min(x1,p%width);y1=std::min(y1,p/width);x2=std::max(x2,p%width+1);y2=std::max(y2,p/width+1);}fg_log<<frame<<','<<f.id<<','<<x1<<','<<y1<<','<<x2<<','<<y2<<','<<f.cx<<','<<f.cy<<','<<f.pixels.size()<<'\n';}
  for(auto&o:os){auto&s=tracks[o.id];auto it=states.find(o.id);int observed=0,n=0;const char* phase="idle";if(it!=states.end()){auto&v=it->second;observed=v.frames_observed;n=int(v.momentum_samples.size());phase=v.waiting_for_deceleration?"waiting":v.is_active?"observing":"idle";}
   auto b=bounds(o,width,height,cols,rows);objects<<frame<<','<<stamp<<','<<o.id<<','<<b.x1<<','<<b.y1<<','<<b.x2<<','<<b.y2<<','<<(ghosts.count(o.id)?"ghost":o.is_coasting?"coasting":"motion")<<','<<s.measured<<','<<o.strength<<','<<(associations.count(o.id)?associations.at(o.id):-1)<<','<<associations.count(o.id)<<','<<s.evidence<<','<<phase<<','<<observed<<','<<n<<','<<s.quiet_count<<','<<s.confirmation<<','<<o.is_fall_this_frame<<',';
   for(int b:o.blocks)objects<<b<<';';objects<<','<<observation_target<<'\n';
  }
 }
};
}
