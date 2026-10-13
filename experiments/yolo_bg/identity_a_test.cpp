#include "../../include/HermesII_sdk.h"
#include "../../src/fall/identity_a.h"
#include <iostream>
using Model=hermes_identity_a::Model<VisionSDK::MotionObject,VisionSDK::ObjectFeatures>;
void require(bool x,const char*message){if(!x)throw std::runtime_error(message);}
VisionSDK::MotionObject obj(std::vector<int> blocks){VisionSDK::MotionObject o;o.blocks=blocks;for(int b:blocks){o.centerX+=b%10;o.centerY+=b/10;}o.centerX/=blocks.size();o.centerY/=blocks.size();o.strength=3;return o;}
std::vector<int> pixels(std::vector<int> bs){std::vector<int> p;for(int b:bs)for(int y=b/10*20;y<(b/10+1)*20;++y)for(int x=b%10*20;x<(b%10+1)*20;++x)p.push_back(y*200+x);Model::unique(p);return p;}
int main(){
 _putenv_s("HERMES_IDENTITY_A","1");Model a;int next=1000;std::vector<VisionSDK::MotionVector> vectors(100);for(auto&v:vectors)v.dx=1;std::vector<bool> changed(100,true);
 auto fg=Model::make_fg(0,pixels({44,45,46}),200);std::vector<VisionSDK::ObjectFeatures> fgs{fg};std::vector<VisionSDK::MotionObject> os;
 auto tick=[&](uint64_t ts,std::vector<VisionSDK::MotionObject> ds){a.begin(ts,200,200,10,10);os=ds;a.associate(os,fgs,vectors,changed,next);a.reconstruct(os,fgs);};
 for(int t:{1,201,401,601})tick(t,{obj({44,45,46})});require(a.tracks.at(1000).mature,"track must mature");
 tick(801,{obj({44}),obj({46})});require(os.size()==1&&os[0].id==1000&&os[0].blocks.size()==2,"two fragments must retain original ID");
 auto old=a.tracks.at(1000).footprint;fgs={Model::make_fg(0,pixels({43,44,45,46,47}),200)};tick(1001,{});require(os.size()==1&&os[0].is_coasting,"supported coast");require(a.assisted[0].pixels==old,"coasting must not enlarge footprint");
 fgs.clear();for(int t:{1201,1401,1601,1801})tick(t,{});require(os.empty()&&a.tracks.empty(),"unsupported coast must expire");
 // Two established owners sharing a connected FG must stay ambiguous.
 a.begin(2001,200,200,10,10);Model::Track x;x.mature=true;x.motion_at=x.measured_at=2001;x.footprint=pixels({44});x.last=obj({44});x.last.id=20;a.tracks[20]=x;x.last.id=21;x.last.blocks={46};x.footprint=pixels({46});a.tracks[21]=x;a.measured={20,21};os={a.tracks[20].last,a.tracks[21].last};fgs={fg};a.reconstruct(os,fgs);require(!a.eligible(20)&&!a.eligible(21)&&a.assisted.empty(),"mature conflict must pause");
 // A tentative rival must not displace an established owner.
 a.links.clear();a.assisted.clear();a.tracks.at(21).mature=false;a.reconstruct(os,fgs);
 require(a.matched(20,a.assisted)>=0&&a.matched(21,a.assisted)<0,"tentative rival must not steal FG");
 a.begin(1901,200,200,10,10);require(a.tracks.empty()&&a.expired.size()==2,"clock rollback must invalidate tracking");
 std::cout<<"identity_a: maturity, fragmentation, bounded coasting, conflict, clock reset passed\n";
}
