#include "fall/pixel_geometry.h"
#include <cassert>
#include <cstdio>
#include <numeric>
#include <random>
using hermes_identity_a::PixelGeometry;
static void check(const std::vector<int>& ps,int width){
 PixelGeometry ref;
 for(int p:ps){int x=p%width,y=p/width;ref.sx+=x;ref.sy+=y;ref.min_x=std::min(ref.min_x,x);ref.min_y=std::min(ref.min_y,y);ref.max_x=std::max(ref.max_x,x);ref.max_y=std::max(ref.max_y,y);}
 auto got=hermes_identity_a::pixel_geometry(ps,width);
 assert(ref.sx==got.sx&&ref.sy==got.sy&&ref.min_x==got.min_x&&ref.min_y==got.min_y&&ref.max_x==got.max_x&&ref.max_y==got.max_y);
 if(!ps.empty()){assert(float(ref.sx/ps.size())==float(got.sx/ps.size()));assert(ref.sx/ps.size()/double(width/12)-.5==got.sx/ps.size()/double(width/12)-.5);}
}
int main(){
 for(uint32_t p=0;p<360000;++p)assert(((uint64_t(p)*1374389535u)>>40)==p/800);
 std::vector<int> all(360000);std::iota(all.begin(),all.end(),0);check(all,800);
 // Every pixel through vector lanes, with all remainder sizes exercised.
 for(int p=0;p<360000;p+=4)check({p,p+1,p+2,p+3},800);
 check({},800);for(int n=1;n<12;++n)check(std::vector<int>(all.end()-n,all.end()),800);
 std::mt19937 rng(20261020);std::shuffle(all.begin(),all.end(),rng);check(all,800);
 for(int n:{1,3,4,5,7908,72207,75400,360000}){std::vector<int> ps(n,359999);check(ps,800);for(auto& p:ps)p=int(rng()%360000);check(ps,800);}
 check(std::vector<int>(360001,359999),800);
 for(int width:{800,801,200}){check({-1,0,799,800,359999,360000},width);check({0,1,2,3,-1},width);check({0,1,2,3,360000},width);}
#if defined(__ARM_NEON) && !defined(HERMES_DISABLE_GEOMETRY_NEON)
 std::puts("PASS: NEON geometry equals original double scalar; all 360000 indices, sums/bbox, shuffled/duplicate pixels, tails and fallbacks");
#else
 std::puts("PASS: scalar geometry and exact division proof; NEON execution requires ARM test binary");
#endif
}
