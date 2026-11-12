#include "HermesII_sdk.h"
#include "fall/identity_a.h"
#include <cassert>
#include <numeric>
#include <random>
#include <iostream>
using Model=hermes_identity_a::Model<VisionSDK::MotionObject,VisionSDK::ObjectFeatures>;
static void check(Model& m,const std::vector<int>& pixels){
 std::vector<int> expanded,blocks,expected_blocks;std::vector<size_t> counts(size_t(m.cols)*m.rows,0),ref(counts.size(),0);
 for(int p:pixels){int b=m.block(p);expected_blocks.push_back(b);if(b>=0&&size_t(b)<ref.size())++ref[b];}
 m.visit_pixel_spans(pixels,[&](size_t first,size_t last,int b){assert(first<last);if(b>=0&&size_t(b)<counts.size())counts[b]+=last-first;for(size_t i=first;i<last;++i){expanded.push_back(pixels[i]);blocks.push_back(b);}});
 assert(expanded==pixels&&blocks==expected_blocks&&counts==ref);
 for(int mod:{2,3,7}){auto got=m.select_pixels(pixels,[&](int b){return b%mod==0;});std::vector<int> want;for(int p:pixels)if(m.block(p)%mod==0)want.push_back(p);assert(got==want);}
 std::vector<int> expected_mask;for(int p:pixels)if(p>=0&&p<m.w*m.h)expected_mask.push_back(m.block(p));Model::unique(expected_mask);assert(m.mask_blocks(pixels)==expected_mask);
}
int main(){Model m;std::mt19937 rng(8123);
 for(auto g:std::vector<std::vector<int>>{{800,450,12,16},{801,451,12,16},{200,200,10,10},{32,9,32,9}}){m.begin(1,g[0],g[1],g[2],g[3]);std::vector<int> ps(m.w*m.h);std::iota(ps.begin(),ps.end(),0);check(m,ps);check(m,{});
  for(int count:{1,19,20,64,75000,150000,300000,360000}){std::vector<int> part(ps.begin(),ps.begin()+std::min(ps.size(),size_t(count)));check(m,part);}
  std::shuffle(ps.begin(),ps.end(),rng);check(m,ps);ps.insert(ps.end(),{0,0,0,-1,-1000,m.w*m.h,m.w*m.h+100});check(m,ps);
  ++m.w;check(m,ps); // stale geometry cache must preserve original fallback
 }
 std::cout<<"PASS: span expansion, block counts, selected pixels and masks match per-pixel reference; up to 360000 FG points, shuffled/duplicates/out-of-range and geometry changes\n";
}
