#include "ai/abi_diagnostic_impl.h"
#include <dlfcn.h>
#include <cstring>
#include <sys/stat.h>
static bool compare(const HermesAbiSnapshot& a,const HermesAbiSnapshot& b) {
    bool equal=true;
#define FIELD(name) do { \
    const bool ok=a.name==b.name; \
    std::printf("%-24s runner=%u sdk=%u %s\n",#name,a.name,b.name,ok?"OK":"MISMATCH"); \
    if(!ok)equal=false; \
} while(0)
    FIELD(pointer_size);FIELD(string_size);FIELD(string_align);FIELD(image_config_size);
    FIELD(image_path_offset);FIELD(motion_config_size);FIELD(fall_config_size);
    FIELD(ofstream_size);FIELD(ostringstream_size);FIELD(cxx11_abi);FIELD(glibcxx_date);FIELD(cplusplus);
#undef FIELD
    return equal;
}
static void print_runtime() {
    Dl_info info{};
    if(dladdr(dlsym(RTLD_DEFAULT,"hermes_abi_snapshot"),&info))std::printf("SDK loaded: %s\n",info.dli_fname);
    FILE* maps=std::fopen("/proc/self/maps","r");if(!maps)return;
    char line[1024];
    while(std::fgets(line,sizeof(line),maps))
        if(std::strstr(line,"libstdc++")||std::strstr(line,"libgcc_s")||std::strstr(line,"libc.so"))std::printf("MAP %s",line);
    std::fclose(maps);
}
static bool check_file(const char* path) {
    char data[128]={};FILE* f=std::fopen(path,"rb");if(!f)return false;
    const auto count=std::fread(data,1,sizeof(data)-1,f);
    const bool read_ok=!std::ferror(f);std::fclose(f);
    const char* expected="text=800,u64=4294967297,double=0.25\n";
    const bool ok=read_ok && count==std::strlen(expected) && std::memcmp(data,expected,count)==0;
    std::printf("  bytes=%zu content=[%s] %s\n",count,data,ok?"OK":"BAD_CONTENT");return ok;
}
int main(int argc,char** argv) {
    if(argc!=2){std::fprintf(stderr,"Usage: %s NEW_OUTPUT_DIRECTORY\n",argv[0]);return 2;}
    if(::mkdir(argv[1],0755)!=0){std::perror("create output directory");return 2;}
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::printf("ABI diagnostic: no SDK instance, no NPU init, no video input\n");print_runtime();
    std::printf("Stream flags: good=%u bad=%u eof=%u fail=%u\n",unsigned(std::ios::goodbit),unsigned(std::ios::badbit),unsigned(std::ios::eofbit),unsigned(std::ios::failbit));
    const auto local=abi_local_snapshot();HermesAbiSnapshot sdk{};hermes_abi_snapshot(&sdk);
    const bool compatible=compare(local,sdk);int failures=compatible?0:1;
    if(compatible) {
        for(int long_text=0;long_text<2;++long_text) {
            std::string text=long_text?std::string(256,'L'):std::string("short");
            const std::string expected=text+"/sdk";
            VisionSDK::ImageRelated_v1 config;
            config.save_image_path="/tmp/hermes_abi_path";config.expected_frame_interval_ms=133;
            int rc=hermes_abi_string_probe(&text,&config);
            bool ok=rc==0 && text==expected && config.save_image_path=="/tmp/hermes_abi_path/sdk" && config.expected_frame_interval_ms==134;
            std::printf("string/config roundtrip (%s): %s rc=%d returned_size=%zu\n",long_text?"long":"short",ok?"PASS":"FAIL",rc,text.size());
            if(!ok)++failures;
        }
    } else std::printf("SKIP string roundtrip: incompatible snapshots; no STL objects passed to SDK\n");
    for(int side=0;side<2;++side)for(int kind=0;kind<4;++kind) {
        const char* mode=kind==0?"ofstream":kind==1?"classic":kind==2?"ostringstream":"stdio";
        char path[4096];
        if(std::snprintf(path,sizeof(path),"%s/%s_%s.txt",argv[1],side?"sdk":"runner",mode)>=int(sizeof(path)))return 2;
        HermesIoProbe r{};
        int rc=side?hermes_abi_io_probe(path,kind,&r):abi_local_io_probe(path,kind,&r);
        std::printf("IO %s/%s rc=%d states(initial,text,int,u64,double,flush,close)=%u,%u,%u,%u,%u,%u,%u errno=%d exception=%d num_put=%d\n",
            side?"sdk":"runner",mode,rc,r.initial_state,r.text_state,r.int_state,r.uint64_state,r.double_state,r.flush_state,r.close_state,r.error_number,r.exception,r.num_put_available);
        const bool content=check_file(path);if(rc||!content)++failures;
    }
    std::printf("RESULT=%s failures=%d\n",failures?"FAIL":"PASS",failures);
    std::printf("PASS covers this process before SDK/NPU initialization; runtime corruption requires a separate test.\n");
    return failures?1:0;
}
