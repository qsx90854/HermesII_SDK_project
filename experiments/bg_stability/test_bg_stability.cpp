#include "../../src/fall/bg_stability.h"
#include <iostream>
static void require(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
int main(){
 using hermes_bg_stability::Model;
 const int w=100,h=100;std::vector<uint8_t> image(w*h,10),bed(w*h,255);
 Model a;a.enabled=true;a.window=6000;a.delta=12;
 require(!a.update(1,image.data(),bed.data(),w,h,true),"early ready");
 for(uint64_t t=1001;t<=5001;t+=1000)require(!a.update(t,image.data(),bed.data(),w,h,true),"short window");
 for(int y=40;y<45;++y)for(int x=40;x<60;++x)image[y*w+x]=30; // exactly100/400=25%
 require(!a.update(6001,image.data(),bed.data(),w,h,true),"25% must fail");
 for(uint64_t t=7001;t<=11001;t+=1000)require(!a.update(t,image.data(),bed.data(),w,h,true),"union must persist");
 require(a.update(12001,image.data(),bed.data(),w,h,true),"old change must expire");
 std::fill(image.begin(),image.end(),200);require(a.update(13001,image.data(),bed.data(),w,h,true),"ready must latch");
 Model b;b.enabled=true;b.window=6000;std::fill(image.begin(),image.end(),10);b.update(1,image.data(),bed.data(),w,h,true);
 for(uint64_t t=1001;t<=5001;t+=1000)b.update(t,image.data(),bed.data(),w,h,true);
 for(int i=0;i<10000;++i)if(i%w<40)image[i]=100;
 require(b.update(6001,image.data(),bed.data(),w,h,true),"edge changes must not count");
 Model below;below.enabled=true;below.window=6000;std::fill(image.begin(),image.end(),10);below.update(1,image.data(),bed.data(),w,h,true);
 for(uint64_t t=1001;t<=5001;t+=1000)below.update(t,image.data(),bed.data(),w,h,true);
 for(int i=0;i<99;++i)image[(40+i/20)*w+40+i%20]=30;
 require(below.update(6001,image.data(),bed.data(),w,h,true),"24.75% must pass");
 Model off;off.enabled=false;require(off.update(1,nullptr,nullptr,0,0,false),"OFF must pass");
 Model c;c.enabled=true;c.window=6000;std::fill(bed.begin(),bed.end(),0);c.update(1,image.data(),bed.data(),w,h,true);
 for(uint64_t t=1001;t<=6001;t+=1000)require(!c.update(t,image.data(),bed.data(),w,h,true),"empty mask must not pass");
 std::cout<<"PASS: 6s boundary, strict25%, sliding expiry, latch, edge mask, empty mask, OFF\n";
}
