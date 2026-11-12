#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>
#if defined(__ARM_NEON) && !defined(HERMES_DISABLE_GEOMETRY_NEON)
#include <arm_neon.h>
#endif
namespace hermes_identity_a {
struct PixelGeometry {
 double sx=0,sy=0;
 int min_x=INT_MAX,min_y=INT_MAX,max_x=0,max_y=0;
};
// Scalar fallback retains the original order and double arithmetic for arbitrary
// dimensions/indices. The bounded 800x450 path sums integers exactly, then uses
// the same double divisions as the caller. No approximate reciprocal or FP SIMD.
inline PixelGeometry pixel_geometry(const std::vector<int>& ps,int width) {
#if defined(__ARM_NEON) && !defined(HERMES_DISABLE_GEOMETRY_NEON)
 if(width==800 && !ps.empty() && ps.size()<=360000) {
  uint32x4_t sx=vdupq_n_u32(0),sy=sx;
  uint32x4_t minx=vdupq_n_u32(799),miny=vdupq_n_u32(449),maxx=sx,maxy=sx;
  size_t i=0;bool valid=true;
  for(;i+4<=ps.size();i+=4) {
   const uint32x4_t p=vreinterpretq_u32_s32(vld1q_s32(ps.data()+i));
   const uint32x4_t bad=vcgeq_u32(p,vdupq_n_u32(360000));
   const uint32x2_t bad2=vorr_u32(vget_low_u32(bad),vget_high_u32(bad));
   if(vget_lane_u32(bad2,0)||vget_lane_u32(bad2,1)){valid=false;break;}
   // floor(p/800) == (uint64_t(p)*1374389535)>>40 in this range.
   const uint32x2_t magic=vdup_n_u32(1374389535u);
   const uint32x4_t hi=vcombine_u32(vshrn_n_u64(vmull_u32(vget_low_u32(p),magic),32),vshrn_n_u64(vmull_u32(vget_high_u32(p),magic),32));
   const uint32x4_t y=vshrq_n_u32(hi,8),x=vsubq_u32(p,vmulq_n_u32(y,800));
   sx=vaddq_u32(sx,x);sy=vaddq_u32(sy,y);
   minx=vminq_u32(minx,x);miny=vminq_u32(miny,y);maxx=vmaxq_u32(maxx,x);maxy=vmaxq_u32(maxy,y);
  }
  if(valid) {
   PixelGeometry g;
   const uint64x2_t sumx=vpaddlq_u32(sx),sumy=vpaddlq_u32(sy);
   uint64_t totalx=vgetq_lane_u64(sumx,0)+vgetq_lane_u64(sumx,1),totaly=vgetq_lane_u64(sumy,0)+vgetq_lane_u64(sumy,1);
   if(i){uint32_t lo_x[4],lo_y[4],hi_x[4],hi_y[4];vst1q_u32(lo_x,minx);vst1q_u32(lo_y,miny);vst1q_u32(hi_x,maxx);vst1q_u32(hi_y,maxy);
    for(int k=0;k<4;++k){g.min_x=std::min(g.min_x,int(lo_x[k]));g.min_y=std::min(g.min_y,int(lo_y[k]));g.max_x=std::max(g.max_x,int(hi_x[k]));g.max_y=std::max(g.max_y,int(hi_y[k]));}}
   for(;i<ps.size();++i){int p=ps[i];if(p<0||p>=360000){valid=false;break;}int y=p/800,x=p-y*800;totalx+=x;totaly+=y;g.min_x=std::min(g.min_x,x);g.min_y=std::min(g.min_y,y);g.max_x=std::max(g.max_x,x);g.max_y=std::max(g.max_y,y);}
   if(valid){g.sx=double(totalx);g.sy=double(totaly);return g;}
  }
 }
#endif
 PixelGeometry g;
 for(int p:ps){const int x=p%width,y=p/width;g.sx+=x;g.sy+=y;g.min_x=std::min(g.min_x,x);g.min_y=std::min(g.min_y,y);g.max_x=std::max(g.max_x,x);g.max_y=std::max(g.max_y,y);}
 return g;
}
}
