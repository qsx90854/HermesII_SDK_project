#include "../file_output_policy.h"
#include "abi_diagnostic_impl.h"
extern "C" void hermes_abi_snapshot(HermesAbiSnapshot* out) { *out=abi_local_snapshot(); }
extern "C" int hermes_abi_io_probe(const char* path,int kind,HermesIoProbe* out) {
    if(!hermes_io::enabled()) return -1;
    return abi_local_io_probe(path,kind,out);
}
extern "C" int hermes_abi_string_probe(void* text,void* config) {
    auto& value=*static_cast<std::string*>(text);
    auto& image=*static_cast<VisionSDK::ImageRelated_v1*>(config);
    if(value!="short" && value!=std::string(256,'L'))return 1;
    if(image.save_image_path!="/tmp/hermes_abi_path" || image.expected_frame_interval_ms!=133)return 2;
    value+="/sdk";image.save_image_path+="/sdk";image.expected_frame_interval_ms=134;
    return 0;
}
