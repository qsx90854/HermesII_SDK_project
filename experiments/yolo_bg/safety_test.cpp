#include "../../src/fall/fall_safety.h"
#include "HermesII_sdk.h"
#include <stdexcept>
#include <iostream>
using namespace hermes_safety;
void check(bool v,const char*m){if(!v)throw std::runtime_error(m);}
struct F{int id;float cx,cy;std::vector<int> pixels;};
int main(){
 Guard g;g.coast=true;g.confirm=false;std::vector<uint8_t> im(10000,100),mask(10000,255);std::vector<VisionSDK::MotionObject> os(1);auto&o=os[0];o.id=7;o.blocks={22,23,32,33};o.strength=8;std::vector<F> f;
 g.prepare(1,im.data(),mask,100,100,10,10,os,f);check(g.measured(7),"motion must be measured");
 o.is_coasting=true;o.strength=0;
 g.prepare(101,im.data(),mask,100,100,10,10,os,f);check(!g.measured(7)&&!g.decelerated(7,0),"first coasting zero cannot decelerate");
 g.prepare(201,im.data(),mask,100,100,10,10,os,f);g.prepare(301,im.data(),mask,100,100,10,10,os,f);check(g.eligible(7)&&g.tracks[7].quiet,"supported stationary subject can accumulate");
 std::fill(im.begin(),im.end(),200);g.prepare(401,im.data(),mask,100,100,10,10,os,f);check(!g.eligible(7)&&g.tracks[7].departed,"appearance departure must invalidate");
 g.prepare(13001,im.data(),mask,100,100,10,10,os,f);check(!g.eligible(7),"expired support cannot persist");
 Guard q;q.confirm=true;q.trace=false;auto&s=q.tracks[7];s.box={20,20,40,40,1};s.stationary_since=s.observation_since=2000;
 std::deque<YoloSample> history={{1000,1500,1,true,{{20,20,40,40,1}}}};bool fall=true;o.is_fall_this_frame=true;
 q.filter(5000,os,fall,history);check(!fall&&s.pending_since==5000,"old chair occupancy must not confirm");
 history.push_back({6000,8000,2,true,{}});q.filter(8000,os,fall,history);check(!fall&&s.rejected,"new negative cancels pending");
 Guard pos;pos.confirm=true;auto&sp=pos.tracks[7];sp.box={20,20,40,40,1};sp.stationary_since=sp.observation_since=2000;history.clear();o.is_fall_this_frame=true;pos.filter(5000,os,fall,history);
 history.push_back({6000,8000,2,true,{{19,19,41,41,1}}});pos.filter(8000,os,fall,history);check(fall&&o.is_fall_this_frame&&sp.accepted,"fresh supported person must confirm");
 Guard err;err.confirm=true;auto&se=err.tracks[7];se.box={20,20,40,40,1};se.stationary_since=se.observation_since=2000;history={{6000,8000,2,false,{}}};o.is_fall_this_frame=true;err.filter(5000,os,fall,{});err.filter(8000,os,fall,history);check(!se.rejected&&!fall,"inference error is not negative");err.filter(15000,os,fall,history);check(se.rejected,"timeout must terminate pending");
 check(!person_matches({20,20,40,40,1},{0,0,25,25,1}),"tiny overlap cannot confirm");
 Guard shrink;shrink.confirm=true;auto&ss=shrink.tracks[7];ss.box={20,20,24,24,1};ss.fg_box={19,19,41,41,1};ss.fg_stamp=5000;ss.stationary_since=ss.observation_since=2000;history={{4000,5000,3,true,{{19,19,41,41,1}}}};o.is_fall_this_frame=true;shrink.filter(5000,os,fall,history);check(fall,"FG extent must confirm shrunken motion fragment");
 std::cout<<"PASS: unknown momentum, supported quiet, departure, expiry, old/new YOLO, errors, timeout, overlap\n";
}