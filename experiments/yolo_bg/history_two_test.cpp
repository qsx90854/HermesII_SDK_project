#include "../../src/fall/fall_safety.h"
#include <iostream>
#include <stdexcept>
using namespace hermes_safety;
struct Obj {int id=7;bool is_fall_this_frame=true;bool is_coasting=false;std::vector<int> blocks{22,23,32,33};};
void require(bool x,const char*m){if(!x)throw std::runtime_error(m);}
Guard guard(){Guard g;g.confirm=true;g.history_two=true;g.trace=false;auto&s=g.tracks[7];s.box={0,0,4,4,1};s.fg_box={20,20,40,40,1};s.fg_stamp=5000;s.departed=true;s.stationary_since=s.observation_since=4900;return g;}
int main(){std::vector<Obj> o(1);bool fall=true;Box person{19,19,41,41,.9f,1};
 auto g=guard();std::deque<YoloSample> h={{1000,2000,1,true,{person}},{3000,4000,2,true,{}}};g.filter(5000,o,fall,h);
 require(fall&&g.tracks[7].confirmation=="person_confirmed_history","older positive must win despite latest negative, time cutoff and fragment displacement");
 auto n=guard();o[0].is_fall_this_frame=true;h={{1000,2000,1,true,{}},{3000,4000,2,true,{}}};n.filter(5000,o,fall,h);require(!fall&&n.tracks[7].pending_since==5000,"two negatives must wait");
 h={{3000,4000,2,true,{}},{4500,6000,3,true,{person}}};n.filter(6000,o,fall,h);require(!fall&&!n.tracks[7].rejected,"pre-candidate in-flight image cannot be the next capture");
 h={{4500,6000,3,true,{person}},{6100,8100,4,false,{}}};n.filter(8100,o,fall,h);require(!fall&&!n.tracks[7].rejected,"failed inference must not count negative");
 h={{7000,9000,5,true,{person}}};n.filter(9000,o,fall,h);require(fall&&n.tracks[7].confirmation=="person_confirmed_next","next fresh positive confirms");
 auto neg=guard();o[0].is_fall_this_frame=true;neg.filter(5000,o,fall,{});neg.filter(8000,o,fall,{{6000,8000,1,true,{}}});require(!fall&&neg.tracks[7].rejected,"next negative rejects");
 auto timeout=guard();o[0].is_fall_this_frame=true;timeout.filter(5000,o,fall,{});timeout.filter(15000,o,fall,{});require(timeout.tracks[7].confirmation=="cancel_yolo_timeout","timeout must stop pending");
 auto future=guard();o[0].is_fall_this_frame=true;future.filter(5000,o,fall,{{4000,6000,1,true,{person}}});require(!fall,"unapplied result must not leak from future");
 auto lost=guard();o[0].is_fall_this_frame=true;lost.filter(5000,o,fall,{});o.clear();lost.filter(5100,o,fall,{});require(lost.tracks[7].confirmation=="cancel_track_lost","lost pending track cancels");
 auto retained=guard();retained.coast=false;retained.tracks.clear();o={Obj{}};std::vector<uint8_t> im(10000,100),mask(10000,255);std::vector<int> fg;
 retained.prepare(100,im.data(),mask,100,100,10,10,o,fg);retained.tracks[7].accepted=true;o[0].is_coasting=true;std::fill(im.begin(),im.end(),200);retained.prepare(200,im.data(),mask,100,100,10,10,o,fg);require(retained.tracks[7].accepted,"legacy appearance loss must not reset confirmed history-mode alarm");
 require(!person_matches({0,0,10,10,1},person),"disjoint person cannot confirm");
 std::cout<<"PASS history2/negative/failed/future/capture/timeout/track-lost/FG-fragment\n";
}
