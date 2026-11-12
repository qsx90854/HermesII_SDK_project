#pragma once
#include <cstdint>
#include <cstddef>
namespace hermes_fall_policy {
struct Evidence {
    uint64_t stamp=0;
    float momentum=0;
    int pixels=0,area=0;
    bool valid(uint64_t now) const { return now>=stamp && now-stamp<=4000; }
};
struct Occupancy {
    size_t bottom=0,total=0;
    bool majority() const {return bottom>total/2;}
    float ratio() const {return total?float(bottom)/float(total):0.0f;}
};
template<class Pixels> Occupancy bottomOccupancy(const Pixels& pixels,int width,int height,int band) {
    Occupancy result;
    if(width<=0 || height<=0 || band<=0)return result;
    // Exactly the last `band` rows; compare flattened offsets, no per-pixel division.
    const int64_t end=int64_t(width)*height;
    const int64_t begin=int64_t(width)*(height>band?height-band:0);
    for(auto pixel:pixels) if(pixel>=0 && int64_t(pixel)<end) {
        ++result.total;
        if(int64_t(pixel)>=begin)++result.bottom;
    }
    return result;
}
}
