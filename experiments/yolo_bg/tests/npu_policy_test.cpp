#include "ai/npu_runtime.h"
#include "ai/person_postprocess.h"
#include <cassert>
#include <limits>
#include <iostream>
int main(){
 hermes_npu::Admission gate;int face,yolo;
 gate.demand(&face,100);gate.demand(&yolo,80);
 assert(!gate.claim(&face));assert(gate.claim(&yolo));assert(!gate.claim(&face));
 gate.complete(&yolo);assert(gate.claim(&face));gate.complete(&face);
 gate.demand(&face,500);gate.demand(&face,700);gate.demand(&yolo,600);
 assert(gate.claim(&face));gate.complete(&face);gate.cancel(&yolo);assert(!gate.claim(&yolo));
 std::vector<float> data(6300*7,0);
 auto set=[&](int i,float x,float y,float w,float h,float obj,float a,float b){float row[]={x,y,w,h,obj,a,b};std::copy(row,row+7,data.begin()+i*7);};
 // LEFT_TOP 800x450 -> 320x180, scale=.4, no vertical subtraction.
 set(0,120,68,80,40,.8,.9,.1);
 set(1,120,68,80,40,.7,.9,.1); // same class suppressed
 set(2,120,68,80,40,.9,.1,.9); // other person class retained
 set(3,120,290,80,20,.9,.9,.1); // padding-only box removed
 set(4,std::numeric_limits<float>::quiet_NaN(),10,10,10,1,1,0);
 auto boxes=hermes_person::decodePeople(data.data(),800,450,.25,.45);
 assert(boxes.size()==2);assert(boxes[0].class_id==1);assert(boxes[1].class_id==0);
 assert(std::abs(boxes[0].x1-200)<.001 && std::abs(boxes[0].y1-120)<.001);
 assert(std::abs(boxes[1].confidence-.72)<.001);
 // Cropped height must not get the controller's top offset here.
 auto cropped=hermes_person::decodePeople(data.data(),800,422,.25,.45);
 assert(cropped.size()==2 && std::abs(cropped[0].y1-120)<.001);
 std::fill(data.begin(),data.end(),0);
 assert(hermes_person::decodePeople(data.data(),800,450,.25,.45).empty());
 std::cout<<"NPU admission and people decoding tests passed\n";
}
