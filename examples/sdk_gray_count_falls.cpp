// sdk_gray_count_falls.cpp
//
// Minimal standalone tool: feed a single raw grayscale (.gray) video into the
// SDK and print how many fall / bed-exit events it detects. No errorlog_,
// no mp4/ffmpeg fallback, no mock data -- just "read this .gray, tell me what
// the SDK sees". Useful for quickly checking a clip someone hands you.
//
// Usage:
//   ./sdk_gray_count_falls <path.gray> [--width=800] [--height=450] [--channels=1]
//                           [--interval=1] [--ini=sdk_gray_count_falls.ini]
//                           [--bed=x1,y1,x2,y2,x3,y3,x4,y4]
//
// --interval N: only send every Nth frame to the SDK (matches how
//               examples/sdk_gray_test.cpp samples video at frame_interval=8;
//               default here is 1, i.e. every frame, since a bare .gray file
//               has no known original frame rate).
//
// Config file: sdk_gray_count_falls.ini (own file, NOT the shared
// parameter.ini used by other examples -- keeps this tool's settings
// independent). Same [Motion]/[Object]/[FallDetect]/[Validation] sections as
// parameter.ini, plus an optional bed region:
//   [Bed]
//   Points=312,108,470,108,541,449,233,449   ; x1,y1,x2,y2,x3,y3,x4,y4, SDK-resolution pixels
// --bed=... on the command line overrides this if both are given.

#include "HermesII_sdk.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
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

struct DetectedEvent {
    int frame_index;
    std::string type;
    float confidence;
};

// "x1,y1,x2,y2,x3,y3,x4,y4" -> 4 points, in SDK-resolution pixel coords
// (same format as --bed and the [Bed] Points= ini key).
std::vector<std::pair<int, int>> ParseBedPoints(const std::string& s) {
    std::vector<int> nums;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',') { if (!cur.empty()) { nums.push_back(std::atoi(cur.c_str())); cur.clear(); } }
        else cur += c;
    }
    std::vector<std::pair<int, int>> pts;
    for (size_t i = 0; i + 1 < nums.size(); i += 2) pts.push_back({nums[i], nums[i + 1]});
    return pts;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <path.gray> [--width=800] [--height=450] [--channels=1]\n"
                     "                    [--interval=1] [--ini=sdk_gray_count_falls.ini]\n"
                     "                    [--bed=x1,y1,x2,y2,x3,y3,x4,y4]\n";
        return 1;
    }

    std::string gray_path = argv[1];
    int width = 800, height = 450, channels = 1, interval = 1;
    std::string ini_path = "sdk_gray_count_falls.ini";
    std::string bed_arg;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--width=", 0) == 0) width = std::atoi(a.c_str() + 8);
        else if (a.rfind("--height=", 0) == 0) height = std::atoi(a.c_str() + 9);
        else if (a.rfind("--channels=", 0) == 0) channels = std::atoi(a.c_str() + 11);
        else if (a.rfind("--interval=", 0) == 0) interval = std::max(1, std::atoi(a.c_str() + 11));
        else if (a.rfind("--ini=", 0) == 0) ini_path = a.substr(6);
        else if (a.rfind("--bed=", 0) == 0) bed_arg = a.substr(6);
        else std::cerr << "Warning: unknown argument \"" << a << "\", ignored.\n";
    }

    FILE* fp = fopen(gray_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: cannot open " << gray_path << std::endl;
        return 1;
    }

    std::cout << "=========================================\n";
    std::cout << "SDK .gray Fall/BedExit Counter\n";
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << "\n";
    std::cout << "=========================================\n";
    std::cout << "File     : " << gray_path << "\n";
    std::cout << "Geometry : " << width << "x" << height << " x" << channels << "ch\n";
    std::cout << "Interval : every " << interval << " frame(s) sent to SDK\n";

    SimpleConfig cfg;
    if (cfg.load(ini_path)) {
        std::cout << "Loaded " << ini_path << " for detection config.\n";
    } else {
        std::cerr << "Warning: cannot open " << ini_path << ", using SDK defaults.\n";
    }

    // --- Bed region: read + parse BEFORE sdk.Init()/SetConfig() run at all. ---
    // Diagnostic checkpoint: this proves whether the parsed value is already
    // wrong right after loading (a parsing bug) or only becomes garbled later
    // once the SDK/NPU init path has run (memory corruption from that path).
    // --bed on the command line takes priority; otherwise [Bed] Points= in the ini.
    std::string bed_source = "--bed";
    std::string bed_str = bed_arg;
    if (bed_str.empty()) {
        bed_str = cfg.getStr("Bed.Points", "");
        bed_source = ini_path + " [Bed] Points";
    }
    std::vector<std::pair<int, int>> bed_pts;
    if (!bed_str.empty()) {
        bed_pts = ParseBedPoints(bed_str);
        std::cout << "[pre-Init] Bed region source: " << bed_source << "\n";
        std::cout << "[pre-Init] Bed raw string: \"" << bed_str << "\" (" << bed_str.size() << " bytes)\n";
        std::cout << "[pre-Init] Parsed " << bed_pts.size() << " point(s):";
        for (const auto& p : bed_pts) std::cout << " (" << p.first << "," << p.second << ")";
        std::cout << "\n";
    } else {
        std::cout << "[pre-Init] No bed region source found (use --bed=... or [Bed] Points= in "
                  << ini_path << ").\n";
    }

    // Same config field list/defaults as examples/sdk_gray_test.cpp, so
    // results are directly comparable between the two tools.
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
        fclose(fp);
        return 1;
    }

    // This tool is just a counter; it must not create its own event recording.
    VisionSDK::EventRecording_v1 recCfg;
    recCfg.header.type = VisionSDK::ConfigType::EventRecording_v1;
    recCfg.header.version = 1;
    recCfg.enable = false;
    sdk.SetConfig(&recCfg);

    sdk.SetConfig(&motionCfg);
    sdk.SetConfig(&objCfg);
    sdk.SetConfig(&fallCfg);
    sdk.SetConfig(&imgCfg);

    // --- Re-check the SAME bed_str/bed_pts computed pre-Init, now AFTER
    // sdk.Init() + all SetConfig() calls have run. If this printout differs
    // from the "[pre-Init]" one above, the value was corrupted somewhere
    // during Init()/SetConfig() (e.g. the NPU init path) -- not during parsing. ---
    if (!bed_str.empty()) {
        std::cout << "[post-Init] Bed raw string: \"" << bed_str << "\" (" << bed_str.size() << " bytes)\n";
        std::cout << "[post-Init] Parsed (pre-Init) point count: " << bed_pts.size() << "\n";
        if (bed_pts.size() == 4) {
            sdk.SetBedRegion(bed_pts);
            std::cout << "Bed region set from " << bed_source << " (" << bed_pts.size() << " points).\n";
        } else {
            std::cerr << "Warning: bed region from " << bed_source
                      << " needs exactly 4 x,y pairs, got " << bed_pts.size()
                      << "; ignored. Raw value at this point: \"" << bed_str << "\"\n";
        }
    } else {
        std::cout << "No bed region set (use --bed=... or [Bed] Points= in " << ini_path << ").\n";
    }

    std::vector<DetectedEvent> events;
    sdk.RegisterVisionSDKCallback([&](const VisionSDKEvent& e) {
        if (e.is_fall_detected) {
            events.push_back({e.frame_index, "Fall", e.confidence});
            std::cout << "[Callback] Fall detected at frame " << e.frame_index
                      << " (confidence " << e.confidence << ")\n";
        }
        if (e.is_bed_exit) {
            events.push_back({e.frame_index, "BedExit", e.confidence});
            std::cout << "[Callback] BedExit detected at frame " << e.frame_index
                      << " (confidence " << e.confidence << ")\n";
        }
    });

    size_t frame_size = (size_t)(width * height * channels);
    std::vector<uint8_t> buf(frame_size);
    long total_read = 0, total_fed = 0;
    std::cout << "\nReading frames...\n";
    while (fread(buf.data(), 1, frame_size, fp) == frame_size) {
        if (total_read % interval == 0) {
            uint64_t ts = (uint64_t)(total_read * 33);
            sdk.SetInputMemory(buf.data(), width, height, channels, ts);
            sdk.ProcessNextFrame();
            total_fed++;
        }
        total_read++;
        if (total_read % 1000 == 0) std::cout << "  read " << total_read << " frames...\n";
    }
    fclose(fp);
    sdk.Release();

    int fall_count = 0, bedexit_count = 0;
    for (const auto& ev : events) {
        if (ev.type == "Fall") fall_count++;
        else if (ev.type == "BedExit") bedexit_count++;
    }

    std::cout << "\n=========================================\n";
    std::cout << "Frames read from file : " << total_read << "\n";
    std::cout << "Frames sent to SDK    : " << total_fed << "\n";
    std::cout << "Fall events           : " << fall_count << "\n";
    std::cout << "BedExit events        : " << bedexit_count << "\n";
    if (!events.empty()) {
        std::cout << "Detected at frames    :";
        for (const auto& ev : events) std::cout << " " << ev.frame_index << "(" << ev.type << ")";
        std::cout << "\n";
    }
    std::cout << "=========================================\n";
    // Machine-parseable summary line for scripting.
    std::cout << "FALL_COUNT=" << fall_count << " BEDEXIT_COUNT=" << bedexit_count << "\n";

    return 0;
}
