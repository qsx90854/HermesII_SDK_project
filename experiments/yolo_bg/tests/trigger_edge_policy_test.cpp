#include "fall/trigger_edge_policy.h"
#include <cassert>
#include <vector>
int main(){
 using namespace hermes_fall_policy;
 Evidence e{100,25,7500,7500};
 assert(e.valid(100)&&e.valid(4100)&&!e.valid(4101)&&!e.valid(99));
 auto check=[](std::vector<int> p){return bottomOccupancy(p,800,450,40);};
 assert(!check({}).majority());
 assert(!check({800*150,800*200,800*410}).majority());
 assert(!check({800*409,800*410}).majority());
 assert(check({800*200,800*410,800*449}).majority());
 assert(check({800*410}).ratio()==1.f);
 auto invalid=check({-1,800*450,800*409});assert(invalid.total==1&&invalid.bottom==0);
}
