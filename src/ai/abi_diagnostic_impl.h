#pragma once
#include "abi_diagnostic.h"
#include "HermesII_sdk.h"
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
// Internal linkage prevents runner/DSO helper implementations being interposed.
static HermesAbiSnapshot abi_local_snapshot() {
    VisionSDK::ImageRelated_v1 image;
    HermesAbiSnapshot s{};
    s.pointer_size=sizeof(void*); s.string_size=sizeof(std::string);
    s.string_align=alignof(std::string); s.image_config_size=sizeof(image);
    s.image_path_offset=reinterpret_cast<const char*>(&image.save_image_path)-reinterpret_cast<const char*>(&image);
    s.motion_config_size=sizeof(VisionSDK::MotionEstimation_v1);
    s.fall_config_size=sizeof(VisionSDK::FallDetection_v3);
    s.ofstream_size=sizeof(std::ofstream); s.ostringstream_size=sizeof(std::ostringstream);
#ifdef _GLIBCXX_USE_CXX11_ABI
    s.cxx11_abi=_GLIBCXX_USE_CXX11_ABI;
#else
    s.cxx11_abi=0xffffffffu;
#endif
#ifdef __GLIBCXX__
    s.glibcxx_date=__GLIBCXX__;
#endif
    s.cplusplus=__cplusplus;
    return s;
}
static void abi_write_numbers(std::ostream& out, HermesIoProbe& r) {
    r.num_put_available=std::has_facet<std::num_put<char>>(out.getloc())?1:0;
    r.initial_state=out.rdstate();
    out << "text="; r.text_state=out.rdstate();
    out << 800; r.int_state=out.rdstate();
    out << ",u64=" << uint64_t(4294967297ULL); r.uint64_state=out.rdstate();
    out << ",double=" << 0.25 << '\n'; r.double_state=out.rdstate();
    out.flush(); r.flush_state=out.rdstate();
}
static int abi_local_io_probe(const char* path, int kind, HermesIoProbe* out) {
    *out={}; errno=0;
    try {
        if(kind==3) {
            FILE* file=std::fopen(path,"wb");
            if(!file){out->error_number=errno;return 1;}
            bool ok=std::fprintf(file,"text=%d,u64=%llu,double=%.2f\n",800,4294967297ULL,0.25)>0;
            if(std::fclose(file)!=0)ok=false;
            out->error_number=errno;return ok?0:1;
        }
        if(kind==2) {
            std::ostringstream stream; abi_write_numbers(stream,*out);
            const auto data=stream.str(); FILE* file=std::fopen(path,"wb");
            if(!file){out->error_number=errno;return 1;}
            bool ok=std::fwrite(data.data(),1,data.size(),file)==data.size();
            if(std::fclose(file)!=0)ok=false;
            out->error_number=errno;return (!ok||!stream)?1:0;
        }
        std::ofstream stream;
        if(kind==1)stream.imbue(std::locale::classic());
        stream.open(path,std::ios::out|std::ios::trunc);
        abi_write_numbers(stream,*out);
        stream.close(); out->close_state=stream.rdstate();out->error_number=errno;
        return stream?0:1;
    } catch(...) {out->exception=1;out->error_number=errno;return 1;}
}
