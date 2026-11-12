#include "HermesII_sdk.h"
#include "fall/identity_a.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
using Model=hermes_identity_a::Model<VisionSDK::MotionObject,VisionSDK::ObjectFeatures>;
static int original_block(const Model&m,int p){return std::min(p/m.w/(m.h/m.rows),m.rows-1)*m.cols+std::min(p%m.w/(m.w/m.cols),m.cols-1);}
static std::vector<int> original_mask(const Model&m,const std::vector<int>& ps){std::vector<int> b;for(int p:ps)if(p>=0&&p<m.w*m.h)b.push_back(original_block(m,p));Model::unique(b);return b;}
static std::vector<int> original_supported(const Model&m,const VisionSDK::MotionObject&o,const std::vector<VisionSDK::ObjectFeatures>&fs){std::set<int>b(o.blocks.begin(),o.blocks.end());std::vector<int>out;for(auto&f:fs)for(int p:f.pixels)if(b.count(original_block(m,p)))out.push_back(p);Model::unique(out);return out;}
int main(){
 Model m;std::mt19937 rng(401);size_t comparisons=0;
 const int geometries[][4]={{800,450,12,16},{801,451,12,16},{200,200,10,10},{32,9,32,9}};
 for(const auto&g:geometries){m.begin(133,g[0],g[1],g[2],g[3]);
  for(int p=-1000;p<m.w*m.h+1000;++p){assert(m.block(p)==original_block(m,p));++comparisons;}
  std::vector<int> pixels(m.w*m.h);std::iota(pixels.begin(),pixels.end(),0);std::shuffle(pixels.begin(),pixels.end(),rng);
  pixels.insert(pixels.end(),{0,0,-1,m.w*m.h,m.w*m.h-1});
  assert(m.mask_blocks(pixels)==original_mask(m,pixels));assert(m.mask_blocks({}).empty());
  for(int n:{0,1,800,70000}){
   std::vector<VisionSDK::ObjectFeatures>fs(2);fs[0].pixels.assign(pixels.begin(),pixels.begin()+std::min(size_t(n),pixels.size()));fs[1].pixels={-1000,-1,0,0,m.w*m.h-1,m.w*m.h};
   VisionSDK::MotionObject o;o.blocks={-100,-1,0,0,3,m.cols*m.rows-1,m.cols*m.rows};
   assert(m.supported_pixels(o,fs)==original_supported(m,o,fs));
  }
  // Same geometry / backward timestamp retain the same mapping.
  m.begin(1,g[0],g[1],g[2],g[3]);assert(m.block(m.w*m.h-1)==original_block(m,m.w*m.h-1));
  // Public geometry changed without begin must fall back, not use stale cache.
  ++m.w;for(int p:{0,37,m.w*m.h-1})assert(m.block(p)==original_block(m,p));
  m.reset();assert(m.pixel_blocks.empty());
 }
 std::cout<<"PASS: "<<comparisons<<" original block mappings; dense/shuffled/duplicate masks, membership, geometry/reset\n";
}
