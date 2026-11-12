#include "fall/eligibility_diagnostics.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
struct Object {int id;float strength;};
int main(){
    using namespace hermes_diagnostics;
    std::vector<std::vector<Object>> history(25);
    for(int i=0;i<25;++i)history[i]={{7,float(i)},{8,999},{7,999}};
    const auto original=history;
    const auto h=examine_history(history,7);
    assert(h.frames==25&&h.samples==20&&h.recent_samples==3&&h.recent_momentum==23&&h.valid());
    for(size_t i=0;i<history.size();++i)assert(history[i][0].strength==original[i][0].strength);
    assert(!examine_history(history,99).valid());
    EligibilityRow row;row.history=h;row.bg_ready=true;row.identity_enabled=true;row.identity_eligible=false;
    row.mature=false;row.ambiguous=true;row.area_metric=3000;row.measured=true;
    assert(row.candidate_without_gates()&&row.blocked());
    row.waiting=true;assert(!row.blocked());row.waiting=false;
    row.area_metric=2999;assert(!row.blocked());row.area_metric=3000;
    row.history.samples=9;assert(!row.blocked());row.history=h;
    row.measured=false;assert(!row.blocked());row.measured=true;
    setenv("HERMES_FILE_OUTPUT","1",1);setenv("HERMES_ELIGIBILITY_TRACE","1",1);
    EligibilityRecorder recorder;
#if HERMES_ENABLE_FILE_OUTPUT
    assert(recorder.enabled());recorder.record(row);recorder.finish();
    std::ifstream file("eligibility_diagnostics.csv");std::string header,line;
    assert(std::getline(file,header)&&std::getline(file,line));
    auto columns=[](const std::string& s){return std::count(s.begin(),s.end(),',')+1;};
    assert(columns(header)==33&&columns(line)==33);
#else
    assert(!recorder.enabled());recorder.record(row);recorder.finish();
    assert(!std::ifstream("eligibility_diagnostics.csv"));
#endif
}
