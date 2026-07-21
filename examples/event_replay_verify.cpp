// event_replay_verify.cpp
//
// Re-validates a SDK-generated event recording (see src/event_recorder.cpp):
// feeds the exact frames from an evt_*.raw file back into a FRESH SDK
// instance (same detection config as parameter.ini, same bed region and
// background the SDK had at event time, both read from the recording's own
// .meta.json / _bg.raw), and checks whether the SDK still detects the same
// event type near the same position in the window.
//
// Usage:
//   ./event_replay_verify <path/to/evt_..._fN.raw> [--ini=parameter.ini] [--tolerance=15]
//
// Exit code: 0 = re-detected near the original event position (or a
// self-test recording, nothing to verify), 1 = usage/file error,
// 2 = re-validation FAILED (SDK no longer detects the event on this build).
//
// IMPORTANT CAVEAT: a real fall/bed-exit event window has usually accumulated
// object-tracking/observation state over MANY frames before entering the 300
// pre-frames captured in the recording. Replaying only those 601 frames from
// a brand-new SDK instance cannot reproduce that history, so a mid-window
// trigger that fired in the original run because of context outside the
// window may legitimately fail to reproduce here. What SHOULD reproduce is
// the FIRST trigger (the one that opened the recording, always located at
// file position == pre_frames) once the background is seeded — that is what
// this tool treats as the pass/fail criterion. All other original triggers
// are reported for reference only.

#include "HermesII_sdk.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace VisionSDK;

// ==========================================================================
// Minimal INI loader (duplicated from examples/sdk_gray_test.cpp so this
// example stays a single self-contained .cpp, consistent with the other
// examples in this directory).
// ==========================================================================
class SimpleConfig {
    std::map<std::string, std::string> settings;
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line, section;
        while (std::getline(f, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == ';' || line[first] == '#') continue;
            std::string trimmed = line.substr(first);
            if (trimmed[0] == '[') {
                size_t end = trimmed.find(']');
                if (end != std::string::npos && end > 1) section = trimmed.substr(1, end - 1);
            } else {
                size_t eq = trimmed.find('=');
                if (eq != std::string::npos) {
                    std::string key = trimmed.substr(0, eq);
                    std::string val = trimmed.substr(eq + 1);
                    size_t k_last = key.find_last_not_of(" \t\r\n");
                    if (k_last != std::string::npos) key.erase(k_last + 1);
                    size_t v_first = val.find_first_not_of(" \t\r\n");
                    if (v_first != std::string::npos) val.erase(0, v_first);
                    size_t v_last = val.find_last_not_of(" \t\r\n");
                    if (v_last != std::string::npos) val.erase(v_last + 1);
                    settings[section.empty() ? key : (section + "." + key)] = val;
                }
            }
        }
        return true;
    }
    std::string getStr(const std::string& key, const std::string& def) {
        auto it = settings.find(key);
        return it != settings.end() ? it->second : def;
    }
    int getInt(const std::string& key, int def) {
        auto it = settings.find(key);
        return it != settings.end() ? std::atoi(it->second.c_str()) : def;
    }
    float getFloat(const std::string& key, float def) {
        auto it = settings.find(key);
        return it != settings.end() ? (float)std::atof(it->second.c_str()) : def;
    }
};

// ==========================================================================
// Tiny hand-rolled reader for THIS SDK's own meta.json schema
// (src/event_recorder.cpp writes a known, fixed, mostly-flat layout — this
// is not a general JSON parser, it only needs to survive that one writer).
// ==========================================================================
namespace metajson {

bool ReadWholeFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool FindValuePos(const std::string& js, const std::string& key, size_t from, size_t& value_pos) {
    std::string needle = "\"" + key + "\"";
    size_t p = js.find(needle, from);
    if (p == std::string::npos) return false;
    size_t colon = js.find(':', p + needle.size());
    if (colon == std::string::npos) return false;
    value_pos = colon + 1;
    while (value_pos < js.size() && (js[value_pos] == ' ' || js[value_pos] == '\t' || js[value_pos] == '\n'))
        value_pos++;
    return true;
}

bool GetLongLong(const std::string& js, const std::string& key, long long& out, size_t from = 0) {
    size_t vp;
    if (!FindValuePos(js, key, from, vp)) return false;
    if (js.compare(vp, 4, "null") == 0) return false;
    out = std::atoll(js.c_str() + vp);
    return true;
}

bool GetDouble(const std::string& js, const std::string& key, double& out, size_t from = 0) {
    size_t vp;
    if (!FindValuePos(js, key, from, vp)) return false;
    if (js.compare(vp, 4, "null") == 0) return false;
    out = std::atof(js.c_str() + vp);
    return true;
}

bool GetBool(const std::string& js, const std::string& key, bool& out, size_t from = 0) {
    size_t vp;
    if (!FindValuePos(js, key, from, vp)) return false;
    out = (js.compare(vp, 4, "true") == 0);
    return true;
}

bool GetString(const std::string& js, const std::string& key, std::string& out, size_t from = 0) {
    size_t vp;
    if (!FindValuePos(js, key, from, vp)) return false;
    if (vp >= js.size() || js[vp] != '\"') return false; // null or non-string value
    size_t start = vp + 1;
    size_t end = js.find('\"', start);
    if (end == std::string::npos) return false;
    out = js.substr(start, end - start);
    return true;
}

// "bed_region": [[120,80],[680,90],...]
std::vector<std::pair<int, int>> GetBedRegion(const std::string& js) {
    std::vector<std::pair<int, int>> pts;
    size_t vp;
    if (!FindValuePos(js, "bed_region", 0, vp) || vp >= js.size() || js[vp] != '[') return pts;
    size_t end = js.find(']', vp + 1); // outer array closes right after the last "]" pair; find via bracket count
    // bed_region has one level of nested [x,y] pairs, so scan bracket depth to find the true end.
    int depth = 0;
    size_t i = vp;
    for (; i < js.size(); ++i) {
        if (js[i] == '[') depth++;
        else if (js[i] == ']') { depth--; if (depth == 0) break; }
    }
    end = i;
    std::string body = js.substr(vp + 1, end > vp ? end - vp - 1 : 0);
    std::vector<long> nums;
    std::string cur;
    for (char c : body) {
        if (c == ',' || c == '[' || c == ']') {
            if (!cur.empty()) { nums.push_back(std::atol(cur.c_str())); cur.clear(); }
        } else if (std::isdigit((unsigned char)c) || c == '-') {
            cur += c;
        }
    }
    if (!cur.empty()) nums.push_back(std::atol(cur.c_str()));
    for (size_t k = 0; k + 1 < nums.size(); k += 2) pts.push_back({(int)nums[k], (int)nums[k + 1]});
    return pts;
}

struct Trigger {
    long long frame_index = -1;
    long long seq = -1;
    long long timestamp_ms = 0;
    std::string type;
    double confidence = 0.0;
};

std::vector<Trigger> GetTriggers(const std::string& js) {
    std::vector<Trigger> out;
    size_t vp;
    if (!FindValuePos(js, "triggers", 0, vp) || vp >= js.size() || js[vp] != '[') return out;
    size_t arr_end = js.find(']', vp); // trigger objects contain no nested [ ]
    if (arr_end == std::string::npos) return out;
    size_t pos = vp;
    while (true) {
        size_t obj_start = js.find('{', pos);
        if (obj_start == std::string::npos || obj_start > arr_end) break;
        size_t obj_end = js.find('}', obj_start);
        if (obj_end == std::string::npos || obj_end > arr_end) break;
        std::string obj = js.substr(obj_start, obj_end - obj_start + 1);
        Trigger t;
        long long v;
        double d;
        std::string s;
        if (GetLongLong(obj, "frame_index", v)) t.frame_index = v;
        if (GetLongLong(obj, "seq", v)) t.seq = v;
        if (GetLongLong(obj, "timestamp_ms", v)) t.timestamp_ms = v;
        if (GetString(obj, "type", s)) t.type = s;
        if (GetDouble(obj, "confidence", d)) t.confidence = d;
        out.push_back(t);
        pos = obj_end + 1;
    }
    return out;
}

} // namespace metajson

// ==========================================================================
// Helpers
// ==========================================================================

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string DirOf(const std::string& path) {
    size_t p = path.find_last_of('/');
    return (p == std::string::npos) ? std::string(".") : path.substr(0, p);
}

// event_type "fall+bed_exit" -> the set of required detected sub-types
std::vector<std::string> SplitEventType(const std::string& type) {
    std::vector<std::string> parts;
    if (type.find("fall") != std::string::npos) parts.push_back("fall");
    if (type.find("bed_exit") != std::string::npos) parts.push_back("bed_exit");
    return parts;
}

struct ReplayEvent {
    long long frame_index; // == 0-based feed position, since a fresh SDK's
                            // internal frame_idx starts at 0 and this tool
                            // calls ProcessNextFrame exactly once per frame.
    std::string type;      // "fall" or "bed_exit" (an event can carry both)
    float confidence;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <path/to/evt_..._fN.raw> [--ini=parameter.ini] [--tolerance=15]\n"
                     "\n"
                     "Replays a SDK EventRecorder recording through a fresh SDK instance\n"
                     "(same parameter.ini config, same bed region / background seeded from\n"
                     "the recording) and checks whether the SDK still detects the event that\n"
                     "was originally recorded. Use this to re-validate old event recordings\n"
                     "against a new SDK build (e.g. after a detection logic fix).\n";
        return 1;
    }

    std::string raw_path = argv[1];
    std::string ini_path = "parameter.ini";
    int tolerance = 15;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--ini=", 0) == 0) ini_path = a.substr(6);
        else if (a.rfind("--tolerance=", 0) == 0) tolerance = std::atoi(a.c_str() + 12);
    }

    std::string meta_path = raw_path;
    if (EndsWith(meta_path, ".raw")) meta_path = meta_path.substr(0, meta_path.size() - 4);
    meta_path += ".meta.json";

    std::string meta_json;
    if (!metajson::ReadWholeFile(meta_path, meta_json)) {
        std::cerr << "Error: cannot open meta file: " << meta_path << std::endl;
        return 1;
    }

    long long width = 0, height = 0, channels = 0, pre_frames = 0, post_frames = 0, total_frames = 0;
    metajson::GetLongLong(meta_json, "width", width);
    metajson::GetLongLong(meta_json, "height", height);
    metajson::GetLongLong(meta_json, "channels", channels);
    metajson::GetLongLong(meta_json, "pre_frames", pre_frames);
    metajson::GetLongLong(meta_json, "post_frames", post_frames);
    metajson::GetLongLong(meta_json, "total_frames", total_frames);
    std::string event_type, bg_file;
    metajson::GetString(meta_json, "event_type", event_type);
    bool has_bg = metajson::GetString(meta_json, "bg_file", bg_file);
    std::vector<std::pair<int, int>> bed_region = metajson::GetBedRegion(meta_json);
    std::vector<metajson::Trigger> triggers = metajson::GetTriggers(meta_json);

    if (width <= 0 || height <= 0 || channels <= 0 || total_frames <= 0) {
        std::cerr << "Error: meta file missing/invalid width/height/channels/total_frames: "
                  << meta_path << std::endl;
        return 1;
    }

    std::cout << "=========================================\n";
    std::cout << "SDK Event Replay Verification\n";
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << "\n";
    std::cout << "=========================================\n";
    std::cout << "Recording        : " << raw_path << "\n";
    std::cout << "Meta             : " << meta_path << "\n";
    std::cout << "Geometry         : " << width << "x" << height << " x" << channels << "ch\n";
    std::cout << "Frames in file   : " << total_frames << " (pre=" << pre_frames
              << " post=" << post_frames << ")\n";
    std::cout << "Original event   : " << event_type << " (" << triggers.size() << " trigger(s) recorded)\n";
    std::cout << "Bed region       : " << (bed_region.empty() ? "(none recorded)" :
                  (std::to_string(bed_region.size()) + " point(s)")) << "\n";
    std::cout << "Background       : " << (has_bg ? bg_file : std::string("(not recorded)")) << "\n";

    if (event_type == "self_test") {
        std::cout << "\nThis is a self-test recording (synthetic forced trigger, not a real\n"
                     "detection) -- there is nothing to re-validate. Exiting.\n";
        return 0;
    }
    if (triggers.empty()) {
        std::cerr << "Error: meta has no triggers to verify against.\n";
        return 1;
    }

    // The FIRST trigger opened the capture window: its file position is
    // always exactly pre_frames (first_seq = event_seq - pre_frames, and
    // event_seq == triggers[0].seq), independent of anything else in meta.
    long long anchor_pos = pre_frames;
    std::string anchor_type = triggers.front().type;
    std::cout << "Anchor (original): file position " << anchor_pos << ", type=" << anchor_type
              << ", confidence=" << triggers.front().confidence << "\n";

    // --- Build detection config exactly like examples/sdk_gray_test.cpp ---
    SimpleConfig cfg;
    if (cfg.load(ini_path)) {
        std::cout << "Loaded " << ini_path << " for detection config.\n";
    } else {
        std::cerr << "Warning: cannot open " << ini_path << ", using SDK defaults "
                     "(may not match the config that produced this recording).\n";
    }

    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = cfg.getInt("Motion.Grid_Cols", 12);
    motionCfg.grid_rows = cfg.getInt("Motion.Grid_Rows", 16);
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = cfg.getInt("Motion.Diff_Check_Range", 5);
    motionCfg.block_change_threshold = cfg.getFloat("Motion.Block_Difference_Ratio_Threshold", 0.03f);
    motionCfg.search_mode = cfg.getInt("Motion.Search_Mode", 1);
    motionCfg.enable_block_decay = (cfg.getInt("Motion.Enable_Block_Decay", 1) != 0);
    motionCfg.block_decay_frames = cfg.getInt("Motion.Block_Decay_Frames", 3);
    motionCfg.enable_block_dilation = (cfg.getInt("Motion.Enable_Block_Dilation", 1) != 0);
    motionCfg.block_dilation_threshold = cfg.getInt("Motion.Block_Dilation_Threshold", 2);

    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
    objCfg.foreground_merge_radius = cfg.getInt("Object.Foreground_Merge_Range", 1);
    objCfg.object_extraction_threshold = 2.0f;
    objCfg.tracking_overlap_threshold = cfg.getFloat("Tracking.Tracking_Overlap_Threshold", 0.5f);
    objCfg.tracking_mode = cfg.getInt("Tracking.Tracking_Mode", 1);
    objCfg.tracking_ttl = cfg.getInt("Tracking.Tracking_TTL", 60);

    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 3.0f);
    fallCfg.fall_strong_threshold = cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 8.0f);
    fallCfg.fall_acceleration_threshold = cfg.getFloat("FallDetect.Fall_Detect_Acceleration_Threshold", 5.0f);
    fallCfg.fall_acceleration_upper_threshold = cfg.getFloat("FallDetect.Fall_Detect_Accel_Upper_Threshold", 6.0f);
    fallCfg.fall_acceleration_lower_threshold = cfg.getFloat("FallDetect.Fall_Detect_Accel_Lower_Threshold", -4.0f);
    fallCfg.bed_pixel_ratio_threshold = cfg.getFloat("FallDetect.Fall_Detect_Bed_Pixel_Ratio_Threshold", 0.15f);
    fallCfg.safe_area_ratio_threshold = cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5f);
    fallCfg.fall_window_size = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 30);
    fallCfg.fall_duration = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 5);
    fallCfg.post_fall_distance_threshold = cfg.getFloat("FallDetect.Fall_Detect_Post_Fall_Distance_Threshold", 4.0f);
    fallCfg.post_fall_check_frames = cfg.getInt("FallDetect.Fall_Detect_Post_Fall_Check_Frames", 5);
    fallCfg.momentum_calc_type = cfg.getInt("FallDetect.Fall_Detect_Momentum_Calc_Type", 1);
    fallCfg.enable_face_detection = false; // Disable to avoid NPU crash or other issues
    fallCfg.face_detect_interval_frames = cfg.getInt("FallDetect.Face_Detect_Interval_Frames", 8);
    fallCfg.enable_edge_drop_filter = (cfg.getInt("FallDetect.Enable_Edge_Drop_Filter", 1) != 0);
    fallCfg.enable_bed_exit_verification = true;
    fallCfg.enable_block_shrink_verification = true;
    fallCfg.enable_save_bg_mask = (cfg.getInt("FallDetect.Enable_Save_BG_Mask", 1) != 0);
    fallCfg.bg_init_start_frame = cfg.getInt("FallDetect.BG_Init_Start_Frame", 2);
    fallCfg.bg_init_end_frame = cfg.getInt("FallDetect.BG_Init_End_Frame", 5);
    fallCfg.bg_diff_threshold = cfg.getInt("FallDetect.BG_Diff_Threshold", 18);
    fallCfg.bg_update_interval_frames = cfg.getInt("FallDetect.BG_Update_Interval", 12);
    fallCfg.bg_update_alpha = cfg.getFloat("FallDetect.BG_Update_Alpha", 0.08f);
    fallCfg.bed_update_alpha_multiplier = cfg.getFloat("FallDetect.Bed_Update_Alpha_Multiplier", 8.0f);
    fallCfg.enable_post_bed_exit_threshold = (cfg.getInt("FallDetect.Enable_Post_BedExit_Threshold", 1) != 0);
    fallCfg.post_bed_exit_threshold_multiplier = cfg.getFloat("FallDetect.Post_BedExit_Threshold_Multiplier", 0.7f);
    fallCfg.projection_use_foreground = (cfg.getInt("FallDetect.Projection_Use_Foreground", 0) != 0);
    fallCfg.opt_flow_frame_distance = cfg.getInt("OpticalFlow.CompareFrameDistance", 2);
    fallCfg.perspective_point_x = cfg.getInt("OpticalFlow.PerspectivePointX", 416);
    fallCfg.perspective_point_y = cfg.getInt("OpticalFlow.PerspectivePointY", 474);
    fallCfg.min_trigger_area = cfg.getInt("OpticalFlow.MinTriggerArea", 2000);
    fallCfg.enable_fall_and_bed_exit = true;

    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    imgCfg.enable_save_images = false;
    imgCfg.enable_draw_bg_noise = false;
    imgCfg.expected_frame_interval_ms = cfg.getInt("Validation.Expected_Frame_Interval", 33);
    imgCfg.frame_interval_tolerance_ms = cfg.getInt("Validation.Frame_Interval_Tolerance", 10);

    VisionSDK::VisionSDK sdk;
    if (sdk.Init("models/blaze_face_detect_nnp310_128x128.ty", 4) != VisionSDK::StatusCode::OK) {
        std::cerr << "Error: SDK Init failed.\n";
        return 1;
    }

    // This tool must not itself spawn a nested recording while replaying one.
    VisionSDK::EventRecording_v1 recCfg;
    recCfg.header.type = VisionSDK::ConfigType::EventRecording_v1;
    recCfg.header.version = 1;
    recCfg.enable = false;
    sdk.SetConfig(&recCfg);

    sdk.SetConfig(&motionCfg);
    sdk.SetConfig(&objCfg);
    sdk.SetConfig(&fallCfg);
    sdk.SetConfig(&imgCfg);

    if (!bed_region.empty()) {
        sdk.SetBedRegion(bed_region);
    } else {
        std::cout << "Warning: no bed region in meta; bed-exit re-detection may differ "
                     "from the original run.\n";
    }

    // Seed the background exactly as it was at event time, instead of letting
    // the fresh SDK "learn" a background from the first few frames of this
    // 601-frame window (which show the room mid-event, not an empty room).
    if (has_bg) {
        std::string bg_path = DirOf(raw_path) + "/" + bg_file;
        FILE* bf = fopen(bg_path.c_str(), "rb");
        if (bf) {
            std::vector<uint8_t> bg_data((size_t)(width * height)); // recorder always saves bg as 1-channel
            size_t got = fread(bg_data.data(), 1, bg_data.size(), bf);
            fclose(bf);
            if (got == bg_data.size()) {
                sdk.SetBackground(bg_data.data(), (int)width, (int)height, 1);
                std::cout << "Seeded background from " << bg_path << "\n";
            } else {
                std::cerr << "Warning: " << bg_path << " size mismatch (" << got << "/"
                          << bg_data.size() << " bytes), background not seeded.\n";
            }
        } else {
            std::cerr << "Warning: cannot open background file " << bg_path << "\n";
        }
    } else {
        std::cout << "Warning: no background recorded; the SDK will build one from the first "
                     "few replay frames, which likely does not reflect the real empty-room "
                     "background since this window starts mid-event.\n";
    }

    std::vector<ReplayEvent> replay_events;
    sdk.RegisterVisionSDKCallback([&](const VisionSDKEvent& e) {
        if (e.is_fall_detected) {
            replay_events.push_back({e.frame_index, "fall", e.confidence});
            std::cout << "[Replay Callback] Fall at frame " << e.frame_index
                      << " (confidence " << e.confidence << ")\n";
        }
        if (e.is_bed_exit) {
            replay_events.push_back({e.frame_index, "bed_exit", e.confidence});
            std::cout << "[Replay Callback] BedExit at frame " << e.frame_index
                      << " (confidence " << e.confidence << ")\n";
        }
    });

    FILE* rf = fopen(raw_path.c_str(), "rb");
    if (!rf) {
        std::cerr << "Error: cannot open " << raw_path << std::endl;
        return 1;
    }
    size_t frame_size = (size_t)(width * height * channels);
    std::vector<uint8_t> buf(frame_size);
    long long fed = 0;
    std::cout << "\nFeeding " << total_frames << " frames...\n";
    for (long long i = 0; i < total_frames; ++i) {
        size_t got = fread(buf.data(), 1, frame_size, rf);
        if (got != frame_size) {
            std::cerr << "Warning: short read at frame " << i << " (" << got << "/" << frame_size
                      << " bytes); stopping early.\n";
            break;
        }
        uint64_t ts = (uint64_t)(i * 33); // replay pacing metadata only; detection doesn't depend on it
        sdk.SetInputMemory(buf.data(), (int)width, (int)height, (int)channels, ts);
        sdk.ProcessNextFrame();
        fed++;
        if (fed % 100 == 0) std::cout << "  fed " << fed << "/" << total_frames << "\n";
    }
    fclose(rf);
    std::cout << "Fed " << fed << " frames total.\n";
    sdk.Release();

    // --- Anchor check: this is the pass/fail criterion ---
    std::vector<std::string> needed = SplitEventType(anchor_type);
    bool anchor_match = !needed.empty();
    long long worst_diff = 0;
    for (const std::string& sub : needed) {
        long long best_diff = -1;
        for (const auto& re : replay_events) {
            if (re.type != sub) continue;
            long long diff = std::llabs(re.frame_index - anchor_pos);
            if (best_diff == -1 || diff < best_diff) best_diff = diff;
        }
        if (best_diff == -1 || best_diff > tolerance) {
            anchor_match = false;
        }
        if (best_diff > worst_diff) worst_diff = best_diff;
    }

    // --- Informational: how each individually-recorded trigger fared ---
    std::cout << "\n--- Original triggers (informational; mid-window triggers may not\n"
                 "    reproduce without the original run's full tracking history) ---\n";
    long long base_seq = triggers.front().seq - pre_frames;
    for (const auto& t : triggers) {
        long long pos = t.seq - base_seq;
        std::vector<std::string> sub_needed = SplitEventType(t.type);
        bool ok = !sub_needed.empty();
        for (const std::string& sub : sub_needed) {
            bool found = false;
            for (const auto& re : replay_events) {
                if (re.type == sub && std::llabs(re.frame_index - pos) <= tolerance) { found = true; break; }
            }
            if (!found) ok = false;
        }
        std::cout << "  file_pos=" << pos << " type=" << t.type << " confidence=" << t.confidence
                   << "  -> " << (ok ? "reproduced" : "not reproduced") << "\n";
    }

    std::cout << "\n=========================================\n";
    if (anchor_match) {
        std::cout << "RESULT: PASS -- SDK re-detected \"" << anchor_type
                  << "\" within " << worst_diff << " frame(s) of the original anchor "
                  << "(position " << anchor_pos << ", tolerance " << tolerance << ").\n";
        std::cout << "=========================================\n";
        return 0;
    } else {
        std::cout << "RESULT: FAIL -- SDK did NOT re-detect \"" << anchor_type
                  << "\" within " << tolerance << " frame(s) of the original anchor "
                  << "(position " << anchor_pos << ").\n";
        std::cout << "This means the current SDK build behaves differently on this recording\n"
                     "than the build that originally produced it.\n";
        std::cout << "=========================================\n";
        return 2;
    }
}
