#pragma once
// Opt-in, instance-local tracking and FG ownership. Raw BGs are never updated here.
#include "../tracking/hungarian.h"
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace hermes_identity_a {
inline bool flag(const char*k){auto v=std::getenv(k);return v&&std::string(v)=="1";}
inline bool diagnostics(){return flag("HERMES_TRACK_REPLAY");}
enum class Mode { off, shadow, track, fall, long_track };
inline Mode mode(){return flag("HERMES_IDENTITY_A")?Mode::long_track:Mode::off;}
struct Params {
 uint64_t gap_ms=1000,prediction_ms=700,hold_ms=30000;
 double gate_blocks=3.0,max_speed_blocks_s=8.0,min_overlap=.15,margin=.10;
 int min_pixels=100,pad_blocks=1;
 static double env(const char* name,double value,double lo,double hi){const char* s=std::getenv(name);if(!s)return value;char* end=nullptr;double x=std::strtod(s,&end);if(end==s||*end||!std::isfinite(x)||x<lo||x>hi)throw std::runtime_error(std::string("Invalid ")+name);return x;}
 void load(){gap_ms=uint64_t(env("HERMES_FG_GAP_MS",gap_ms,1,10000));prediction_ms=uint64_t(env("HERMES_FG_PREDICTION_MS",prediction_ms,1,10000));hold_ms=uint64_t(env("HERMES_FG_HOLD_MS",hold_ms,1,120000));gate_blocks=env("HERMES_FG_GATE_BLOCKS",gate_blocks,.1,10);min_overlap=env("HERMES_FG_MIN_OVERLAP",min_overlap,.01,1);margin=env("HERMES_FG_MARGIN",margin,.01,1);}
};
struct Event {std::string type;int id=-1,other=-1;};
struct Candidate {int id,component;double cost,distance,overlap;std::string reason;};
struct Link {std::string state="predicted_only";std::vector<int> long_ids,short_ids;int assisted_id=-1,added=0,pixels=0;bool ambiguous=false;uint64_t missing_ms=0,motion_age_ms=0;};
template<class Obj,class FG>struct Model {
 struct Track {Obj last{};uint64_t measured_at=0,motion_at=0,supported_at=0;double vx=0,vy=0;std::vector<int> footprint;bool outside=false,long_support=false;
  bool mature=false,position_valid=false;double person_x=0,person_y=0;
  uint64_t born_at=0,independent_since=0,position_at=0;int confirmation_samples=0;};
 bool identity_a=true;
 Mode selected=mode();Params params;int w=0,h=0,cols=0,rows=0;uint64_t now=0,clock=0;
 std::map<int,Track> tracks;std::map<int,Link> links;std::vector<FG> assisted;std::vector<Event> events;std::vector<Candidate> candidates;
 std::set<int> measured;std::vector<int> expired;
 void record_event(Event e){if(diagnostics())events.push_back(std::move(e));}
 Model(){if(enabled())params.load();}
 bool enabled()const{return selected!=Mode::off;}bool active()const{return selected==Mode::track||selected==Mode::fall||selected==Mode::long_track;}
 bool use_fall()const{return selected==Mode::fall||selected==Mode::long_track;}
 void reset(){tracks.clear();links.clear();assisted.clear();events.clear();candidates.clear();measured.clear();expired.clear();clock=now=0;w=h=cols=rows=0;conflicts.clear();finish();}
 int block(int p)const{return std::min(p/w/(h/rows),rows-1)*cols+std::min(p%w/(w/cols),cols-1);}
 static void unique(std::vector<int>& v){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());}
 static double overlap(const std::vector<int>& a,const std::vector<int>& b){size_t i=0,j=0,n=0;while(i<a.size()&&j<b.size()){if(a[i]==b[j]){++n;++i;++j;}else if(a[i]<b[j])++i;else ++j;}return double(n)/std::max(size_t(1),std::min(a.size(),b.size()));}
 void begin(uint64_t stamp,int width,int height,int gc,int gr){
  events.clear();candidates.clear();links.clear();assisted.clear();measured.clear();expired.clear();
  if(clock&&(stamp<=clock||stamp-clock>params.gap_ms||width!=w||height!=h||gc!=cols||gr!=rows)){
   for(const auto& kv:tracks){expired.push_back(kv.first);record_event({"FG_CLOCK_RESET",kv.first});}tracks.clear();conflicts.clear();
  }now=clock=stamp;w=width;h=height;cols=gc;rows=gr;
 }
 std::vector<int> mask_blocks(const std::vector<int>& ps)const{std::vector<int> bs;for(int p:ps)if(p>=0&&p<w*h)bs.push_back(block(p));unique(bs);return bs;}
 template<class FGs>std::vector<int> supported_pixels(const Obj& o,const FGs& fgs)const{
  std::set<int> bs(o.blocks.begin(),o.blocks.end());std::vector<int> out;
  for(const auto& f:fgs)for(int p:f.pixels)if(bs.count(block(p)))out.push_back(p);unique(out);return out;
 }
 template<class Vectors>bool real_motion(const Obj& o,const Vectors& vectors,const std::vector<bool>& changed)const{
  for(int b:o.blocks)if(b>=0&&b<int(vectors.size())&&b<int(changed.size())&&changed[b]&&(vectors[b].dx!=0||vectors[b].dy!=0))return true;return false;
 }
 // Association uses previous-frame footprints, never the mask reconstructed later in this frame.
 template<class FGs,class Vectors>void associate(std::vector<Obj>& detections,const FGs& long_fg,const Vectors& vectors,const std::vector<bool>& changed,int& next_id){
  std::vector<int> ids;for(const auto& kv:tracks)ids.push_back(kv.first);
  const size_t nt=ids.size(),nd=detections.size();std::vector<std::vector<int>> pixels(nd),blocks(nd);
  for(size_t j=0;j<nd;++j){pixels[j]=supported_pixels(detections[j],long_fg);blocks[j]=detections[j].blocks;unique(blocks[j]);}
  // Prior whole-person support is checked before single-component assignment.
  std::vector<int> preferred(nd,-1);std::vector<bool> prior_conflict(nd,false);
  if(identity_a)for(size_t j=0;j<nd;++j){
   std::vector<int> eligible;
   for(int id:ids){const auto& t=tracks.at(id);if(!t.mature||t.footprint.empty()||now-t.motion_at>params.hold_ms)continue;
    for(const auto& f:long_fg){size_t local=0;for(int p:f.pixels)local+=std::binary_search(blocks[j].begin(),blocks[j].end(),block(p));if(local<20)continue;
     auto fp=f.pixels;unique(fp);size_t shared=0;for(int p:fp)shared+=std::binary_search(t.footprint.begin(),t.footprint.end(),p);
     if(shared<100||double(shared)/std::max(size_t(1),std::min(fp.size(),t.footprint.size()))<.35)continue;
     auto tb=mask_blocks(t.footprint);size_t covered=0;for(int b:blocks[j]){bool near=false;for(int old:tb)if(std::abs(b%cols-old%cols)<=1&&std::abs(b/cols-old/cols)<=1){near=true;break;}covered+=near;}
     if(double(covered)/std::max(size_t(1),blocks[j].size())>=.5){eligible.push_back(id);break;}
    }
   }
   if(eligible.size()==1){preferred[j]=eligible.front();record_event({"A_FRAGMENT_PRIOR_OWNER",preferred[j],int(j)});}
   else if(eligible.size()>1){prior_conflict[j]=true;record_event({"A_MATURE_OWNER_CONFLICT",-1,int(j)});}
  }
  std::vector<std::vector<float>> costs(nt,std::vector<float>(nd+nt,.85f));
  for(size_t i=0;i<nt;++i){const auto& t=tracks.at(ids[i]);double dt=double(now-t.measured_at)/1000.0,advance=std::min(dt,double(params.prediction_ms)/1000.0);
   double px=(identity_a&&t.position_valid?t.person_x:t.last.centerX)+t.vx*advance,py=(identity_a&&t.position_valid?t.person_y:t.last.centerY)+t.vy*advance;
   auto tb=t.last.blocks;unique(tb);
   for(size_t j=0;j<nd;++j){double dx=detections[j].centerX-px,dy=detections[j].centerY-py,d=std::hypot(dx,dy),ov=std::max(overlap(pixels[j],t.footprint),overlap(blocks[j],tb));
    double gate=params.gate_blocks+std::min(2.0,params.max_speed_blocks_s*advance);double cost=.6*std::min(1.0,d/gate)+.4*(1-ov);
    std::string reason="candidate";if((d>gate && !(selected==Mode::long_track && overlap(pixels[j],t.footprint)>=.5))||(dt>double(params.prediction_ms)/1000.0&&ov<params.min_overlap)){cost=999;reason="gate_rejected";}
    if(identity_a&&preferred[j]>=0){cost=preferred[j]==ids[i]?0:999;reason=preferred[j]==ids[i]?"whole_person_prior":"owned_by_other_prior";}
    if(identity_a&&prior_conflict[j]){cost=999;reason="mature_owner_conflict";}
    costs[i][j]=float(cost);if(diagnostics())candidates.push_back({ids[i],int(j),cost,d,ov,reason});
   }
  }
  // A near tie between different people remains unknown rather than an arbitrary ownership.
  std::vector<bool> ambiguous=prior_conflict;
  for(size_t j=0;j<nd;++j){double best=999,second=999;for(size_t i=0;i<nt;++i){double c=costs[i][j];if(c<best){second=best;best=c;}else second=std::min(second,c);}if(best<.85&&second-best<params.margin){ambiguous[j]=true;for(size_t i=0;i<nt;++i)costs[i][j]=999;record_event({"FG_ASSOCIATION_AMBIGUOUS",-1,int(j)});}}
  auto assigned=VisionSDK::HungarianAlgorithm::Solve(costs);std::vector<int> owner(nd,-1);std::set<int> used;
  for(size_t i=0;i<nt;++i){int j=assigned[i];if(j>=0&&size_t(j)<nd&&costs[i][j]<.85){owner[j]=ids[i];used.insert(ids[i]);}}
  if(identity_a)for(size_t j=0;j<nd;++j)if(preferred[j]>=0&&!ambiguous[j]){owner[j]=preferred[j];used.insert(preferred[j]);}
  // Attach an extra motion fragment only when the old footprint uniquely supports it.
  for(size_t j=0;j<nd;++j)if(owner[j]<0&&!ambiguous[j]){int best=-1;double score=0,second=0;for(int id:ids){if(selected!=Mode::long_track&&!used.count(id))continue;double ov=overlap(pixels[j],tracks.at(id).footprint);if(ov>score){second=score;score=ov;best=id;}else second=std::max(second,ov);}if(score>=.5&&score-second>=params.margin){owner[j]=best;record_event({"FG_MOTION_SPLIT_GROUPED",best,int(j)});}}
  std::map<int,Obj> grouped;
  for(size_t j=0;j<nd;++j){if(ambiguous[j])continue;int id=owner[j];if(id<0){id=next_id++;record_event({"FG_NEW_TRACK",id,int(j)});}auto o=detections[j];o.id=id;o.is_coasting=false;
   auto found=grouped.find(id);if(found==grouped.end())grouped.emplace(id,o);else {auto& g=found->second;g.blocks.insert(g.blocks.end(),o.blocks.begin(),o.blocks.end());g.block_motion_vectors.clear();g.block_groups.clear();g.strength=std::max(g.strength,o.strength);unique(g.blocks);double sx=0,sy=0;for(int b:g.blocks){sx+=b%cols;sy+=b/cols;}g.centerX=float(sx/g.blocks.size());g.centerY=float(sy/g.blocks.size());}
  }
  detections.clear();for(auto& kv:grouped){int id=kv.first;auto& o=kv.second;auto it=tracks.find(id);if(it!=tracks.end()){auto& t=it->second;double dt=double(now-t.measured_at)/1000.0;
    if(dt>0&&!identity_a){t.vx=std::clamp((o.centerX-t.last.centerX)/dt,-params.max_speed_blocks_s,params.max_speed_blocks_s);t.vy=std::clamp((o.centerY-t.last.centerY)/dt,-params.max_speed_blocks_s,params.max_speed_blocks_s);}
    if(t.last.is_coasting)record_event({"FG_TRACK_REACQUIRED",id});o.trajectory=t.last.trajectory;o.acceleration=o.strength-t.last.strength;
   }
   o.trajectory.emplace_back(int(o.centerX),int(o.centerY));if(o.trajectory.size()>120)o.trajectory.erase(o.trajectory.begin());
   auto& t=tracks[id];if(!t.born_at)t.born_at=now;if(!t.measured_at)t.motion_at=now;t.measured_at=now;if(real_motion(o,vectors,changed))t.motion_at=now;t.last=o;measured.insert(id);detections.push_back(o);
  }
  for(auto& kv:tracks)if(!measured.count(kv.first)){auto o=kv.second.last;if(!o.is_coasting)record_event({"FG_COAST_BEGIN",kv.first});o.is_coasting=true;o.strength=0;o.acceleration=0;o.avgDx=o.avgDy=0;o.matched_fg_obj_id=-1;o.is_fall_this_frame=false;kv.second.last=o;detections.push_back(o);}
 }
 // Position updates require current whole-FG geometry. Changing motion fragments never generate velocity.
 void update_person_position(int id,const std::vector<int>& ps){
  if(!identity_a||ps.empty())return;auto& t=tracks.at(id);
  double x=0,y=0;for(int p:ps){x+=p%w;y+=p/w;}x=x/ps.size()/double(w/cols)-.5;y=y/ps.size()/double(h/rows)-.5;
  double ratio=t.footprint.empty()?1:double(ps.size())/t.footprint.size();
  bool adequate=!t.position_valid||(ratio>=.6&&ratio<=1.7&&std::hypot(x-t.person_x,y-t.person_y)<=2.0);
  if(!adequate){t.vx=t.vy=0;record_event({"A_PERSON_POSITION_UNCERTAIN",id});return;}
  if(t.position_valid&&now>t.position_at&&now-t.position_at<=params.gap_ms){double dt=double(now-t.position_at)/1000.;t.vx=std::clamp((x-t.person_x)/dt,-params.max_speed_blocks_s,params.max_speed_blocks_s);t.vy=std::clamp((y-t.person_y)/dt,-params.max_speed_blocks_s,params.max_speed_blocks_s);}
  else t.vx=t.vy=0;
  t.person_x=x;t.person_y=y;t.position_at=now;t.position_valid=true;
 }
 static FG make_fg(int id,const std::vector<int>& ps,int width){FG f{};f.id=id;f.pixels=ps;f.area=int(ps.size());f.min_x=f.min_y=INT32_MAX;double sx=0,sy=0;for(int p:ps){int x=p%width,y=p/width;f.min_x=std::min(f.min_x,x);f.max_x=std::max(f.max_x,x);f.min_y=std::min(f.min_y,y);f.max_y=std::max(f.max_y,y);sx+=x;sy+=y;}if(!ps.empty()){f.cx=float(sx/ps.size());f.cy=float(sy/ps.size());}return f;}
 // A port: current YOLO-assisted BG foreground replaces the old long-FG input.
 // Grid-sized support masks avoid one full-frame bitmap for every track.
 std::map<int,uint64_t> conflicts;
 std::ofstream trace_log;
 void finish(){if(trace_log.is_open()){trace_log.flush();if(!trace_log)throw std::runtime_error("Identity A trace failed");trace_log.close();}}
 bool eligible(int id)const{auto t=tracks.find(id);auto l=links.find(id);return t!=tracks.end()&&t->second.mature&&l!=links.end()&&!l->second.ambiguous;}
 template<class FGs>void reconstruct(std::vector<Obj>& objects,const FGs& fgs){
  std::map<int,std::vector<uint8_t>> allowed;std::map<int,std::vector<int>> masks;
  for(auto&o:objects){auto&t=tracks.at(o.id);auto&a=allowed[o.id];a.assign(cols*rows,0);
   for(int b:mask_blocks(t.footprint))a[b]=1;
   if(measured.count(o.id))for(int b:o.blocks)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int x=b%cols+dx,y=b/cols+dy;if(x>=0&&x<cols&&y>=0&&y<rows)a[y*cols+x]=1;}
  }
  for(auto&fg:fgs){std::vector<int> owners;std::vector<int> counts(cols*rows,0);for(int p:fg.pixels)++counts[block(p)];
   for(auto&kv:allowed){int n=0;for(int b=0;b<cols*rows;++b)if(kv.second[b])n+=counts[b];if(n>=params.min_pixels&&double(n)/std::max(size_t(1),fg.pixels.size())>=params.min_overlap)owners.push_back(kv.first);}
   bool mature=false;for(int id:owners)mature|=tracks.at(id).mature;
   if(mature)owners.erase(std::remove_if(owners.begin(),owners.end(),[&](int id){if(tracks.at(id).mature)return false;record_event({"A_TENTATIVE_FG_DEFERRED",id});return true;}),owners.end());
   if(owners.size()>1){for(int id:owners)links[id].ambiguous=true;record_event({"A_REGION_OWNERSHIP_CONFLICT",owners[0],owners[1]});continue;}
   if(owners.empty())continue;int id=owners[0];links[id].long_ids.push_back(fg.id);auto&ps=masks[id];
   for(int p:fg.pixels)if(allowed[id][block(p)])ps.push_back(p);
  }
  for(auto&o:objects){auto&t=tracks.at(o.id);auto&l=links[o.id];auto&ps=masks[o.id];unique(ps);l.missing_ms=now-t.measured_at;l.motion_age_ms=now-t.motion_at;l.state=measured.count(o.id)?"motion_measured":"predicted_only";
   if(l.ambiguous){ps.clear();l.long_ids.clear();l.state="ambiguous";if(!conflicts[o.id])conflicts[o.id]=now;if(now-conflicts[o.id]>=2000)t.footprint.clear();record_event({"A_CONFLICT_PAUSE",o.id});}
   else conflicts.erase(o.id);
   if(!measured.count(o.id)){std::vector<int> clipped;std::set_intersection(t.footprint.begin(),t.footprint.end(),ps.begin(),ps.end(),std::back_inserter(clipped));ps=std::move(clipped);}
   if(ps.size()>=size_t(params.min_pixels)){
    update_person_position(o.id,ps);bool independent=true;auto cells=mask_blocks(ps);
    for(auto&other:tracks)if(other.first!=o.id&&other.second.mature&&overlap(cells,mask_blocks(other.second.footprint))>=.2)independent=false;
    if(measured.count(o.id)&&!t.mature&&independent){if(!t.independent_since)t.independent_since=now;++t.confirmation_samples;if(now-t.independent_since>=600&&t.confirmation_samples>=3){t.mature=true;record_event({"A_TRACK_CONFIRMED",o.id});}}
    else if(!t.mature){t.independent_since=0;t.confirmation_samples=0;}
    t.supported_at=now;l.pixels=int(ps.size());l.assisted_id=int(assisted.size());assisted.push_back(make_fg(l.assisted_id,ps,w));if(!measured.count(o.id))l.state="fg_supported";
    // The footprint follows current evidence. Coasting can shrink but never grow.
    if(measured.count(o.id))t.footprint=ps;
    else {std::vector<int> kept;std::set_intersection(t.footprint.begin(),t.footprint.end(),ps.begin(),ps.end(),std::back_inserter(kept));t.footprint=std::move(kept);}
   }else{if(!t.mature){t.independent_since=0;t.confirmation_samples=0;}if(!l.ambiguous&&now-t.supported_at>params.prediction_ms)t.footprint.clear();}
  }
  for(auto it=objects.begin();it!=objects.end();){auto&t=tracks.at(it->id);auto&l=links[it->id];bool grace=conflicts.count(it->id)&&now-conflicts[it->id]<2000;
   bool remove=!measured.count(it->id)&&(now-t.motion_at>params.hold_ms||(!grace&&l.state!="fg_supported"&&now-t.measured_at>params.prediction_ms));
   if(remove){int id=it->id;expired.push_back(id);record_event({"FG_TRACK_EXPIRED",id});tracks.erase(id);links.erase(id);conflicts.erase(id);it=objects.erase(it);}else ++it;
  }
  std::vector<FG> kept;for(auto&kv:links)if(kv.second.assisted_id>=0){auto f=assisted[kv.second.assisted_id];f.id=int(kept.size());kv.second.assisted_id=f.id;kept.push_back(std::move(f));}assisted=std::move(kept);
 }
 void write(int frame){if(!diagnostics())return;if(!trace_log.is_open())trace_log.open("identity_a.jsonl");trace_log<<"{\"sdk_frame\":"<<frame<<",\"timestamp_ms\":"<<now<<",\"association\":"<<json()<<"}\n";}
 int matched(int id,const std::vector<FG>& fgs)const{auto it=links.find(id);if(it==links.end()||it->second.ambiguous)return -1;const auto& l=it->second;int best=-1;size_t area=0;for(size_t i=0;i<fgs.size();++i){bool valid=use_fall()?fgs[i].id==l.assisted_id:std::find(l.long_ids.begin(),l.long_ids.end(),fgs[i].id)!=l.long_ids.end();if(valid&&fgs[i].pixels.size()>area){area=fgs[i].pixels.size();best=int(i);}}return best;}
 static void ints(std::ostream& s,const std::vector<int>& v){s<<'[';for(size_t i=0;i<v.size();++i){if(i)s<<',';s<<v[i];}s<<']';}
 std::string json()const{std::ostringstream s;s<<"{\"schema\":1,\"mode\":\""<<(selected==Mode::shadow?"shadow":use_fall()?"fall":"track")<<"\",\"parameters\":{\"prediction_ms\":"<<params.prediction_ms<<",\"hold_ms\":"<<params.hold_ms<<",\"gap_ms\":"<<params.gap_ms<<",\"gate_blocks\":"<<params.gate_blocks<<",\"max_speed_blocks_s\":"<<params.max_speed_blocks_s<<",\"min_overlap\":"<<params.min_overlap<<",\"margin\":"<<params.margin<<",\"min_pixels\":"<<params.min_pixels<<",\"pad_blocks\":"<<params.pad_blocks<<"},\"tracks\":[";bool c=false;
  for(const auto& kv:links){if(c)s<<',';c=true;const auto& l=kv.second;s<<"{\"id\":"<<kv.first<<",\"state\":\""<<l.state<<"\",\"missing_ms\":"<<l.missing_ms<<",\"motion_age_ms\":"<<l.motion_age_ms<<",\"long_ids\":";ints(s,l.long_ids);s<<",\"short_ids\":";ints(s,l.short_ids);s<<",\"assisted_id\":"<<l.assisted_id<<",\"added_pixels\":"<<l.added<<",\"pixels\":"<<l.pixels<<",\"ambiguous\":"<<(l.ambiguous?"true":"false")<<'}';}
  s<<"],\"identity_a\":"<<(identity_a?"true":"false")<<",\"person_tracks\":[";c=false;
  if(identity_a)for(const auto& kv:tracks){if(c)s<<',';c=true;const auto& t=kv.second;s<<"{\"id\":"<<kv.first<<",\"mature\":"<<(t.mature?"true":"false")<<",\"position_valid\":"<<(t.position_valid?"true":"false")<<",\"person_x\":"<<t.person_x<<",\"person_y\":"<<t.person_y<<",\"position_age_ms\":"<<(t.position_at?now-t.position_at:0)<<",\"confirmation_samples\":"<<t.confirmation_samples<<'}';}
  s<<"],\"candidates\":[";c=false;for(const auto& x:candidates){if(c)s<<',';c=true;s<<"{\"track\":"<<x.id<<",\"component\":"<<x.component<<",\"cost\":"<<x.cost<<",\"distance_blocks\":"<<x.distance<<",\"overlap\":"<<x.overlap<<",\"reason\":\""<<x.reason<<"\"}";}
  s<<"],\"events\":[";c=false;for(const auto& e:events){if(c)s<<',';c=true;s<<"{\"type\":\""<<e.type<<"\",\"id\":"<<e.id<<",\"other\":"<<e.other<<'}';}s<<"]}";return s.str();
 }
};
}
