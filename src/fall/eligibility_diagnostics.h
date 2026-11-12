#pragma once
// Read-only snapshots of pre-trigger gates. Never replays/accepts candidates.
#include "../file_output_policy.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint>
namespace hermes_diagnostics {
struct History {
    size_t frames=0;
    int samples=0, recent_samples=0;
    float recent_momentum=0;
    bool valid() const {return frames>=5 && samples>=10;}
};
template<class Frames> History examine_history(const Frames& frames,int id) {
    History result;result.frames=frames.size();
    // Match the live path: last 20 frames, first object of this ID per frame,
    // then the last three matching samples; preserve float addition order.
    std::vector<float> strengths;strengths.reserve(20);
    const size_t first=frames.size()>20?frames.size()-20:0;
    for(size_t i=first;i<frames.size();++i)for(const auto& object:frames[i])
        if(object.id==id){strengths.push_back(object.strength);break;}
    result.samples=int(strengths.size());
    result.recent_samples=std::min(3,result.samples);
    if(result.recent_samples){
        float sum=0;
        for(int i=result.samples-result.recent_samples;i<result.samples;++i)sum+=strengths[i];
        result.recent_momentum=sum/result.recent_samples;
    }
    return result;
}
struct EligibilityRow {
    uint64_t frame=0,stamp=0;
    int id=0;
    bool bg_ready=false,identity_enabled=false,mature=false,ambiguous=false,identity_eligible=true;
    History history;
    float threshold=20;
    int pixels=0,fg_area=0,area_metric=0,area_threshold=3000;
    bool measured=false,active=false,waiting=false;
    int post_fall=0,confirmation_samples=0;
    float x1=0,y1=0,x2=0,y2=0;
    bool momentum_pass() const {return history.recent_samples && history.recent_momentum>=threshold;}
    bool area_pass() const {return area_metric>=area_threshold;}
    bool fresh_allowed() const {return !active&&!waiting&&post_fall==0;}
    bool candidate_without_gates() const {return history.valid()&&momentum_pass()&&area_pass()&&measured&&fresh_allowed();}
    bool blocked() const {return candidate_without_gates()&&(!bg_ready||!identity_eligible);}
};
class EligibilityRecorder {
    bool enabled_=false;
    std::vector<EligibilityRow> rows_;
    size_t dropped_=0;
public:
    EligibilityRecorder(){
        const char* value=std::getenv("HERMES_ELIGIBILITY_TRACE");
        enabled_=hermes_io::enabled()&&value&&std::strcmp(value,"1")==0;
    }
    bool enabled() const {return enabled_;}
    void record(const EligibilityRow& row){
        if(!enabled_)return;
        if(rows_.size()>=262144){++dropped_;return;}
        if(rows_.empty())rows_.reserve(4096);
        rows_.push_back(row);
    }
    void finish(){
        if(!enabled_||rows_.empty())return;
        FILE* file=std::fopen("eligibility_diagnostics.csv","w");
        if(!file)throw std::runtime_error("Cannot open eligibility diagnostics");
        std::fputs("gray_index,timestamp_ms,track_id,bg_ready,identity_enabled,mature,ambiguous,identity_eligible,history_frames,history_samples,recent_samples,recent_momentum,threshold,momentum_pass,pixel_count,fg_area,area_metric,area_threshold,area_pass,measured,active,waiting,post_fall_counter,fresh_allowed,candidate_without_gates,candidate_blocked,block_bg,block_identity,confirmation_samples,x1,y1,x2,y2\n",file);
        for(const auto& r:rows_){
            std::fprintf(file,"%llu,%llu,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%.9g,%.9g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g\n",
                (unsigned long long)r.frame,(unsigned long long)r.stamp,r.id,int(r.bg_ready),int(r.identity_enabled),int(r.mature),int(r.ambiguous),int(r.identity_eligible),
                r.history.frames,r.history.samples,r.history.recent_samples,double(r.history.recent_momentum),double(r.threshold),int(r.momentum_pass()),
                r.pixels,r.fg_area,r.area_metric,r.area_threshold,int(r.area_pass()),int(r.measured),int(r.active),int(r.waiting),r.post_fall,int(r.fresh_allowed()),
                int(r.candidate_without_gates()),int(r.blocked()),int(!r.bg_ready),int(!r.identity_eligible),r.confirmation_samples,
                double(r.x1),double(r.y1),double(r.x2),double(r.y2));
        }
        bool bad=std::ferror(file)!=0;if(std::fclose(file))bad=true;
        if(bad)throw std::runtime_error("Eligibility diagnostic write failed");
        FILE* summary=std::fopen("eligibility_diagnostics_summary.csv","w");
        if(!summary)throw std::runtime_error("Cannot open eligibility diagnostic summary");
        const int rc=std::fprintf(summary,"rows,dropped\n%zu,%zu\n",rows_.size(),dropped_);
        const int closed=std::fclose(summary);
        if(rc<0||closed)throw std::runtime_error("Eligibility diagnostic summary write failed");
        rows_.clear();dropped_=0;
    }
};
}
