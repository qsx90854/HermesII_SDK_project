#include "fall/bg_pixel_update.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// Original scalar kernel, kept separately to compare exactly the old arithmetic.
__attribute__((noinline)) static void original(uint8_t* bg,const uint8_t* now,int w,
        int x1,int y1,int x2,int y2,bool valid,float alpha) {
    for(int y=y1;y<y2;++y)for(int x=x1;x<x2;++x){int p=y*w+x;
        bg[p]=valid?uint8_t(std::lround((1-alpha)*bg[p]+alpha*now[p])):now[p];
    }
}
int main(){
    std::vector<float> weights{0.f,1.f,.00001f,.5f,.99999f,.13f,.33333334f};
    for(unsigned misses=1;misses<=8;++misses)weights.push_back(std::min(.32f,.08f+.04f*misses));
    std::vector<uint8_t> initial(65536),src(65536);
    for(int p=0;p<65536;++p){initial[p]=p/256;src[p]=p%256;}
    size_t checked=0;
    for(float a:weights){auto expected=initial,actual=initial;
        original(expected.data(),src.data(),256,0,0,256,256,true,a);
        hermes_person::updateBackgroundPixels(actual.data(),src.data(),256,0,0,256,256,true,a);
        if(actual!=expected){std::cerr<<"Pixel mismatch alpha="<<a<<"\n";return 1;}checked+=actual.size();
    }
    // Nonzero origin, odd tail widths, untouched outside block, initialization,
    // exact-buffer aliasing and empty blocks.
    for(bool valid:{false,true})for(bool alias:{false,true})for(int x2:{3,16,19}){
        std::vector<uint8_t> expected(initial.begin(),initial.begin()+19*11),actual=expected;
        const uint8_t* input=src.data();
        original(expected.data(),alias?expected.data():input,19,3,2,x2,10,valid,.24f);
        hermes_person::updateBackgroundPixels(actual.data(),alias?actual.data():input,19,3,2,x2,10,valid,.24f);
        assert(actual==expected);
    }
    std::cout<<"PASS: "<<checked<<" blend comparisons; rectangular boundaries, copy, alias and empty blocks\n";
}
