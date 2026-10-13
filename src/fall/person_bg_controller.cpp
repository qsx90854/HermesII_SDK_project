#include "person_bg_controller.h"
// LINUX PORT: this include, concrete pointer and BOTH construction sites must be
// replaced via a backend factory/injection. Keep the controller policy below.
// HERMES_ENABLE_ONNX_PERSON also gates shared SDK integration; do not simply disable
// that macro and assume person-assisted BG / fall confirmation remain enabled.
#include "../ai/person_detector_onnx.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <chrono>
namespace hermes_person {
static std::string env(const char*k,const std::string&v=""){auto s=std::getenv(k);return s?s:v;}
static double number(const char*k,double def,double lo,double hi){auto s=env(k);if(s.empty())return def;size_t n=0;double v=std::stod(s,&n);if(n!=s.size()||!std::isfinite(v)||v<lo||v>hi)throw std::runtime_error(std::string("Invalid ")+k);return v;}
struct Track{int id;Box box;float x,y;};
static float area(Box b){return std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);}
static float intersection(Box a,Box b){return std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1))*std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));}
struct PersonBgController::Impl {
 bool exclude=hermes_safety::flag("HERMES_EXCLUDE_TOP_ROW");
 int crop_top=0;std::deque<hermes_safety::YoloSample> history;
 bool protection_v2=env("HERMES_PERSON_PROTECTION_V2")=="1";
 bool on=env("HERMES_PERSON_BG")=="1",offline=env("HERMES_PERSON_MODE","async")=="offline",trace=env("HERMES_PERSON_TRACE")=="1";
 uint64_t interval=4000,delay=2000,hold=14000,last=0,next=0,serial=0,request_time=0;int w=0,h=0,cols=0,rows=0;
 float base=.08f,increment=.04f,maximum=.32f,margin=.10f;bool pending=false,started=false,finished=false;Result delayed;
 std::unique_ptr<PersonDetectorOnnx> detector;
 std::vector<uint8_t> valid,source,moved;std::vector<unsigned> misses;std::vector<float> alpha;
 std::vector<Track> captured;std::map<int,Track> previous;
 struct Bridge{float dx=0,dy=0;bool continuous=true;};std::map<int,Bridge> bridges;
 struct Protection{int id=-1;Box box;uint64_t confirmed=0;std::vector<uint8_t> reference;uint64_t unsupported_since=0;const char* state="tracked";};std::vector<Protection> protections;
 std::ofstream events,blocks,boxes_log,protection_log;uint64_t submitted=0,success=0,failed=0,updates=0,stale=0;double infer_ms=0,control_ms=0;
 Impl(){if(!on)return;auto mode=env("HERMES_PERSON_MODE","async");if(mode!="async"&&mode!="offline")throw std::runtime_error("Invalid HERMES_PERSON_MODE");
  interval=uint64_t(number("HERMES_PERSON_INTERVAL_MS",4000,100,60000));delay=uint64_t(number("HERMES_PERSON_DELAY_MS",2000,0,60000));hold=uint64_t(number("HERMES_PERSON_HOLD_MS",14000,0,120000));
  base=float(number("HERMES_PERSON_BASE_ALPHA",.08,0,1));increment=float(number("HERMES_PERSON_ALPHA_STEP",.04,0,1));maximum=float(number("HERMES_PERSON_MAX_ALPHA",.32,base,1));margin=float(number("HERMES_PERSON_MARGIN",.10,0,1));
  Config c;c.model_path=env("HERMES_PERSON_MODEL");c.runtime_path=env("HERMES_PERSON_RUNTIME");c.confidence=float(number("HERMES_PERSON_CONF",.25,.001,1));c.nms_iou=float(number("HERMES_PERSON_NMS",.45,.001,1));
  detector=std::make_unique<PersonDetectorOnnx>(c);
  if(trace){protection_log.open("person_bg_protection.csv");protection_log<<"timestamp_ms,track_id,confirmed_ms,action,appearance_support,x1,y1,x2,y2\n";}
  events.open("person_bg_events.csv");if(!events)throw std::runtime_error("Cannot write person events");events<<"request_id,source_timestamp_ms,applied_timestamp_ms,success,persons,inference_ms,total_ms,valid_blocks,protected_blocks,status\n";
  if(trace){boxes_log.open("person_bg_boxes.csv");if(!boxes_log)throw std::runtime_error("Cannot write person boxes");boxes_log<<"request_id,source_timestamp_ms,applied_timestamp_ms,track_id,kind,x1,y1,x2,y2,confidence,class_id\n";blocks.open("person_bg_blocks.csv");if(!blocks)throw std::runtime_error("Cannot write person blocks");blocks<<"timestamp_ms,block,alpha,nonperson_count,valid,reason\n";}
 }
 std::vector<Track> tracks(const std::vector<VisionSDK::MotionObject>& objects){std::vector<Track> out;int bw=w/cols,bh=h/rows;for(auto&o:objects){if(o.is_coasting||o.blocks.empty())continue;Box b{float(w),float(h),0,0,0};for(int i:o.blocks)if(i>=0&&i<cols*rows){int c=i%cols,r=i/cols;b.x1=std::min(b.x1,float(c*bw));b.y1=std::min(b.y1,float(r*bh));b.x2=std::max(b.x2,float(c==cols-1?w:(c+1)*bw));b.y2=std::max(b.y2,float(r==rows-1?h:(r+1)*bh));}if(area(b)>0)out.push_back({o.id,b,(b.x1+b.x2)/2,(b.y1+b.y2)/2});}return out;}
 Box block(int i){int bw=w/cols,bh=h/rows,c=i%cols,r=i/cols;return {float(c*bw),float(r*bh),float(c==cols-1?w:(c+1)*bw),float(r==rows-1?h:(r+1)*bh),0};}
 Box expand(Box b){float dx=(b.x2-b.x1)*margin,dy=(b.y2-b.y1)*margin;b.x1=std::max(0.f,b.x1-dx);b.x2=std::min(float(w),b.x2+dx);b.y1=std::max(float(crop_top),b.y1-dy);b.y2=std::min(float(h),b.y2+dy);return b;}
 std::vector<uint8_t> sample(const uint8_t* image,Box b){std::vector<uint8_t> out;for(int y=0;y<8;++y)for(int x=0;x<8;++x){int px=int(b.x1+(b.x2-b.x1)*(x+.5f)/8),py=int(b.y1+(b.y2-b.y1)*(y+.5f)/8);if(px>=0&&px<w&&py>=0&&py<h)out.push_back(image[py*w+px]);else out.push_back(0);}return out;}
 float support(const uint8_t* image,const Protection&p){auto values=sample(image,p.box);if(p.reference.size()!=values.size())return 0;int same=0;for(size_t i=0;i<values.size();++i)if(std::abs(int(values[i])-int(p.reference[i]))<=12)++same;return float(same)/values.size();}
 void log_protection(uint64_t ts,const Protection&p,const char*action,float score){if(trace)protection_log<<ts<<','<<p.id<<','<<p.confirmed<<','<<action<<','<<score<<','<<p.box.x1<<','<<p.box.y1<<','<<p.box.x2<<','<<p.box.y2<<'\n';}
 void mark(std::vector<uint8_t>&mask,Box b){for(int i=0;i<cols*rows;++i)if(intersection(block(i),b)>0)mask[i]=1;}
 void mask(uint8_t* bg,const uint8_t* now,int iw,int ih){if(!on||!bg)return;if(iw!=w||ih!=h||valid.empty()){std::copy(now,now+size_t(iw)*ih,bg);return;}for(int i=0;i<cols*rows;++i)if(!valid[i]){auto b=block(i);for(int y=int(b.y1);y<int(b.y2);++y)std::copy(now+y*w+int(b.x1),now+y*w+int(b.x2),bg+y*w+int(b.x1));}}
 void step(uint64_t ts,const uint8_t* now,uint8_t* bg,int iw,int ih,int gc,int gr,const std::vector<VisionSDK::MotionObject>&objects){
  if(!on)return;auto begin=std::chrono::steady_clock::now();if(iw<=0||ih<=0||gc<=0||gr<=0||gc>iw||gr>ih)throw std::runtime_error("Invalid BG grid");
  if(!started||iw!=w||ih!=h||gc!=cols||gr!=rows||ts<=last||ts-last>1000){
   // Drain old asynchronous work before accepting a new stream/geometry.
   if(pending&&!offline){detector->Shutdown();Result discard;detector->TryGetResult(discard);Config c;c.model_path=env("HERMES_PERSON_MODEL");c.runtime_path=env("HERMES_PERSON_RUNTIME");c.confidence=float(number("HERMES_PERSON_CONF",.25,.001,1));c.nms_iou=float(number("HERMES_PERSON_NMS",.45,.001,1));detector=std::make_unique<PersonDetectorOnnx>(c);}
   started=true;w=iw;h=ih;cols=gc;rows=gr;valid.assign(gc*gr,0);misses.assign(gc*gr,0);alpha.assign(gc*gr,0);protections.clear();previous.clear();pending=false;next=ts;history.clear();crop_top=exclude?h/rows:0;
  }
  auto current=tracks(objects);std::map<int,Track> curr;for(auto t:current)curr[t.id]=t;
  for(auto& [id,b]:bridges){auto a=previous.find(id),c=curr.find(id);if(a==previous.end()||c==curr.end()){b.continuous=false;continue;}float dx=c->second.x-a->second.x,dy=c->second.y-a->second.y;if(std::hypot(dx,dy)>std::max(w,h)*.25f){b.continuous=false;continue;}b.dx+=dx;b.dy+=dy;}
  for(auto&p:protections){if(p.id>=0){auto a=previous.find(p.id),c=curr.find(p.id);if(a!=previous.end()&&c!=curr.end()&&std::hypot(c->second.x-a->second.x,c->second.y-a->second.y)<=std::max(w,h)*.25f){float dx=c->second.x-a->second.x,dy=c->second.y-a->second.y;p.box.x1+=dx;p.box.x2+=dx;p.box.y1+=dy;p.box.y2+=dy;if(protection_v2){p.reference=sample(now,p.box);p.unsupported_since=0;p.state="tracked";}}else {p.id=-1;if(protection_v2)log_protection(ts,p,"track_lost",support(now,p));}}
   if(protection_v2&&p.id<0){float score=support(now,p);const char* state=score>=.65f?"stationary_supported":"unmatched_grace";if(std::string(state)!=p.state)log_protection(ts,p,state,score);p.state=state;if(score>=.65f)p.unsupported_since=0;else if(!p.unsupported_since)p.unsupported_since=ts;}
  }
  protections.erase(std::remove_if(protections.begin(),protections.end(),[&](auto&p){bool expired=ts-p.confirmed>hold;bool departed=protection_v2&&p.id<0&&p.unsupported_since&&ts-p.unsupported_since>=1000;if(expired||departed){if(protection_v2)log_protection(ts,p,expired?"expired":"appearance_changed_release",support(now,p));return true;}return false;}),protections.end());
  // New activity anywhere since the sampled image invalidates a negative claim for that block.
  if(pending)for(int i=exclude?cols:0;i<cols*rows;++i){auto b=block(i);int n=0,total=int(area(b));for(int y=int(b.y1);y<int(b.y2);++y)for(int x=int(b.x1);x<int(b.x2);++x)if(std::abs(int(now[y*w+x])-int(source[y*w+x]))>12)++n;if(n*5>total)moved[i]=1;}
  Result result;bool received=false;
// Offline computes synchronously at submission but exposes the result only after
// source time + delay. Async polls actual completion and does NOT add simulated sleep.
// DELAY_MS still participates in the freshness limit below, even in async mode.
  if(pending){if(offline){if(ts>=request_time+delay){result=std::move(delayed);received=true;}}else received=detector->TryGetResult(result);}
// Backend boxes refer to the cropped submitted frame: restore crop_top once.
// History records source geometry; BG protection may subsequently follow a motion bridge.
// Failed/stale results must not accelerate background absorption.
  if(received){for(auto&b:result.persons){b.y1+=crop_top;b.y2+=crop_top;}pending=false;bool fresh=result.source_timestamp_ms<=ts&&ts-result.source_timestamp_ms<=interval+delay+1000;int protected_count=0;
   if(fresh&&(!hermes_safety::flag("HERMES_PERSON_HISTORY_TWO")||result.success)){history.push_back({result.source_timestamp_ms,ts,result.request_id,result.success,result.persons});while(history.size()>2)history.pop_front();}
   if(result.success&&fresh){++success;infer_ms+=result.inference_ms;
    for(auto box:result.persons){if(trace)boxes_log<<result.request_id<<','<<result.source_timestamp_ms<<','<<ts<<",-1,source,"<<box.x1<<','<<box.y1<<','<<box.x2<<','<<box.y2<<','<<box.confidence<<','<<box.class_id<<'\n';int match=-1;float best=0;bool conflict=false;for(auto&t:captured){float v=intersection(t.box,box)/std::max(1.f,area(t.box));if(v>.5f){if(match>=0)conflict=true;if(v>best){best=v;match=t.id;}}}
     if(conflict)match=-1;
     if(match>=0&&bridges[match].continuous&&curr.count(match)){auto b=bridges[match];box.x1+=b.dx;box.x2+=b.dx;box.y1+=b.dy;box.y2+=b.dy;protections.erase(std::remove_if(protections.begin(),protections.end(),[&](auto&p){return p.id==match;}),protections.end());}
     else match=-1;
     // Unmatched source boxes are conservative temporary protection, never evidence of a present track.
     protections.erase(std::remove_if(protections.begin(),protections.end(),[&](auto&p){return intersection(p.box,box)/std::max(1.f,std::min(area(p.box),area(box)))>.5f;}),protections.end());
     Protection added{match,expand(box),result.source_timestamp_ms};
     if(protection_v2){added.reference=sample(match>=0?now:source.data(),added.box);float score=support(now,added);added.state=match>=0?"tracked":score>=.65f?"stationary_supported":"unmatched_grace";if(match<0&&score<.65f)added.unsupported_since=ts;log_protection(ts,added,"created",score);}
     protections.push_back(std::move(added));
    }
    if(trace)for(auto&p:protections)boxes_log<<result.request_id<<','<<p.confirmed<<','<<ts<<','<<p.id<<",protection,"<<p.box.x1<<','<<p.box.y1<<','<<p.box.x2<<','<<p.box.y2<<','<<p.box.confidence<<','<<p.box.class_id<<'\n';
    std::vector<uint8_t> protect(cols*rows,0),motion(cols*rows,0);for(auto&p:protections)mark(protect,p.box);for(auto&t:current)mark(motion,t.box);
    for(int i=0;i<cols*rows;++i){const char* reason="nonperson";if(exclude&&i<cols){alpha[i]=0;misses[i]=0;valid[i]=0;reason="excluded";}else if(protect[i]){alpha[i]=0;misses[i]=0;reason="person_hold";++protected_count;}else if(motion[i]||moved[i]){alpha[i]=0;reason=motion[i]?"motion":"changed_since_request";}else{misses[i]=std::min(misses[i]+1,100000u);alpha[i]=std::min(maximum,base+increment*misses[i]);auto b=block(i);for(int y=int(b.y1);y<int(b.y2);++y)for(int x=int(b.x1);x<int(b.x2);++x){int p=y*w+x;bg[p]=valid[i]?uint8_t(std::lround((1-alpha[i])*bg[p]+alpha[i]*now[p])):now[p];}valid[i]=1;}
     if(trace)blocks<<ts<<','<<i<<','<<alpha[i]<<','<<misses[i]<<','<<int(valid[i])<<','<<reason<<'\n';
    }++updates;
   }else{if(!fresh)++stale;else ++failed;}
   events<<result.request_id<<','<<result.source_timestamp_ms<<','<<ts<<','<<result.success<<','<<result.persons.size()<<','<<result.inference_ms<<','<<result.total_ms<<','<<std::count(valid.begin(),valid.end(),uint8_t(1))<<','<<protected_count<<','<<(fresh?(result.success?"applied":"inference_failed"):"stale")<<'\n';
   if(!result.success){std::ofstream errors("person_bg_errors.log",std::ios::app);errors<<result.error<<'\n';}
  }
// Submission cadence uses source time, not completion time + interval. Busy slots
// are skipped (no queue); next advances even when submission cannot happen.
// Only the top grid row is cropped, preserving packed pitch=w. Backend must not add
// this crop offset. The source copy supports changes detected while inference is pending.
  if(ts>=next){next=ts+interval;if(!pending){++serial;bool accepted=true;if(offline)delayed=detector->RunSync(now+crop_top*w,w,h-crop_top,ts,serial);else accepted=detector->TrySubmit(now+crop_top*w,w,h-crop_top,ts,serial);if(accepted){pending=true;request_time=ts;++submitted;source.assign(now,now+size_t(w)*h);moved.assign(cols*rows,0);captured=current;bridges.clear();for(auto&t:current)bridges[t.id]={};}}}
  previous=std::move(curr);last=ts;control_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 }
 void finish(){if(!on||finished)return;finished=true;detector->Shutdown();if(events.is_open())events.flush();if(blocks.is_open())blocks.flush();if(boxes_log.is_open())boxes_log.flush();if(protection_log.is_open())protection_log.flush();std::ofstream f("person_bg_summary.json");f<<"{\"width\":"<<w<<",\"height\":"<<h<<",\"grid_cols\":"<<cols<<",\"grid_rows\":"<<rows<<",\"submitted\":"<<submitted<<",\"successful_applied\":"<<success<<",\"failed\":"<<failed<<",\"stale\":"<<stale<<",\"updates\":"<<updates<<",\"pending_at_end\":"<<pending<<",\"inference_ms\":"<<infer_ms<<",\"controller_ms_includes_sync_inference\":"<<control_ms<<",\"interval_ms\":"<<interval<<",\"delay_ms\":"<<delay<<",\"offline\":"<<offline<<"}";if(!f||!events||(trace&&(!blocks||!boxes_log)))throw std::runtime_error("Person BG log write failed");}
};
PersonBgController::PersonBgController():impl_(new Impl){}
PersonBgController::~PersonBgController()=default;
bool PersonBgController::enabled()const{return impl_->on;}
void PersonBgController::MaskUnknown(uint8_t*b,const uint8_t*n,int w,int h){if(impl_->finished)impl_.reset(new Impl);impl_->mask(b,n,w,h);}
void PersonBgController::Step(uint64_t t,const uint8_t*n,uint8_t*b,int w,int h,int c,int r,const std::vector<VisionSDK::MotionObject>&o){if(impl_->finished)impl_.reset(new Impl);impl_->step(t,n,b,w,h,c,r,o);}
const std::deque<hermes_safety::YoloSample>& PersonBgController::History()const{return impl_->history;}
void PersonBgController::Finish(){impl_->finish();}
}
