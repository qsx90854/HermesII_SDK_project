#include "HermesII_sdk.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <io.h>
#include <fcntl.h>
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
    const std::map<std::string, std::string>& all() const { return settings; }
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


int main(int argc,char** argv) { try {
 if(argc!=6) throw std::runtime_error("Usage: runner config.ini bed_scaled.txt output.csv width height; stdin: uint64 source, uint64 timestamp, gray bytes; UINT64_MAX source ends stream");
 SimpleConfig cfg; if(!cfg.load(argv[1])) throw std::runtime_error("Cannot load INI");
 int width=std::stoi(argv[4]), height=std::stoi(argv[5]);
 if(width<=0 || height<=0 || width>4096 || height>4096) throw std::runtime_error("Invalid dimensions");
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
    objCfg.header.version = 2;   // v2: carries appended Object.* new fields
    objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
    objCfg.foreground_merge_radius = cfg.getInt("Object.Foreground_Merge_Range", 1);
    objCfg.object_extraction_threshold = 2.0f;
    objCfg.tracking_overlap_threshold = cfg.getFloat("Tracking.Tracking_Overlap_Threshold", 0.5f);
    objCfg.tracking_mode = cfg.getInt("Tracking.Tracking_Mode", 1);
    objCfg.tracking_ttl = cfg.getInt("Tracking.Tracking_TTL", 60);
    objCfg.merge_overlapping_enable = (cfg.getInt("Object.Merge_Overlapping_Enable", 0) != 0);
    objCfg.merge_overlapping_iou = cfg.getFloat("Object.Merge_Overlapping_IoU", 0.3f);
    objCfg.merge_tracked_enable = (cfg.getInt("Object.Merge_Tracked_Enable", 0) != 0);
    objCfg.merge_tracked_overlap = cfg.getFloat("Object.Merge_Tracked_Overlap", 0.3f);
    objCfg.merge_tracked_max_dist = cfg.getFloat("Object.Merge_Tracked_Max_Dist", 0.0f);
    objCfg.enable_kalman_predict = (cfg.getInt("Tracking.Enable_Kalman_Predict", 0) != 0);
    objCfg.use_fg_area = (cfg.getInt("Object.Use_FG_Area", 0) != 0);
    objCfg.min_trigger_fg_area = cfg.getInt("Object.Min_Trigger_FG_Area", 12000);
    objCfg.still_lying_fg_area = cfg.getInt("Object.Still_Lying_FG_Area", 3000);
    objCfg.use_fg_area_trigger = (cfg.getInt("Object.Use_FG_Area_Trigger", 0) != 0);

    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 2;  // v2: carries appended BG_Protect_* fields
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
    fallCfg.enable_bed_exit_verification = cfg.getInt("FallDetect.Enable_Bed_Exit_Verification", 0) != 0;
    fallCfg.enable_block_shrink_verification = cfg.getInt("FallDetect.Enable_Block_Shrink_Verification", 0) != 0;
    fallCfg.enable_save_bg_mask = false;
    fallCfg.bg_init_start_frame = cfg.getInt("FallDetect.BG_Init_Start_Frame", 2);
    fallCfg.bg_init_end_frame = cfg.getInt("FallDetect.BG_Init_End_Frame", 5);
    fallCfg.bg_diff_threshold = cfg.getInt("FallDetect.BG_Diff_Threshold", 18);
    fallCfg.bg_update_interval_frames = cfg.getInt("FallDetect.BG_Update_Interval", 12);
    fallCfg.bg_update_alpha = cfg.getFloat("FallDetect.BG_Update_Alpha", 0.08f);
    fallCfg.bg_protect_max_frames = cfg.getInt("FallDetect.BG_Protect_Max_Frames", 0);
    fallCfg.bg_protect_min_fg = cfg.getInt("FallDetect.BG_Protect_Min_FG", 20);
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
 if(sdk.Init()!=VisionSDK::StatusCode::OK) throw std::runtime_error("Init failed");
 VisionSDK::EventRecording_v1 rec{}; rec.header={VisionSDK::ConfigType::EventRecording_v1,2}; rec.enable=false;
 sdk.SetConfig(&rec);
 for(auto config : std::vector<const void*>{&motionCfg,&objCfg,&fallCfg,&imgCfg})
   if(sdk.SetConfig(config)!=VisionSDK::StatusCode::OK) throw std::runtime_error("Config failed");
 std::ifstream bed(argv[2]); std::vector<std::pair<int,int>> points; int x,y; char comma;
 while(bed>>x>>comma>>y) { if(comma!=',') throw std::runtime_error("Bed format"); points.emplace_back(x,y); }
 if(points.size()!=4) throw std::runtime_error("Need four bed points"); sdk.SetBedRegion(points);
 std::ofstream output(argv[3]); if(!output) throw std::runtime_error("Cannot open output");
 output<<"source_frame,sdk_frame,timestamp_ms,is_fall,callback_count,process_us,fall_object_ids\n";
 int callback_count=0, sdk_frame=-1; bool fall=false;
 sdk.RegisterVisionSDKCallback([&](const VisionSDK::VisionSDKEvent& e){++callback_count;sdk_frame=e.frame_index;fall=e.is_fall_detected;});
 _setmode(_fileno(stdin),_O_BINARY);
 std::vector<unsigned char> image(static_cast<size_t>(width)*height);
 auto read_exact=[](void* p,size_t size){if(fread(p,1,size,stdin)!=size) throw std::runtime_error("Truncated input stream");};
 uint64_t last=0; bool first=true;
 while(true) {
  uint64_t source,ts; read_exact(&source,sizeof(source)); if(source==UINT64_MAX) break;
  read_exact(&ts,sizeof(ts)); read_exact(image.data(),image.size());
  if(!first && source<=last) throw std::runtime_error("Nonmonotonic source frame"); first=false;last=source;
  callback_count=0;fall=false;sdk_frame=-1;
  if(sdk.SetInputMemory(image.data(),width,height,1,ts)!=VisionSDK::StatusCode::OK) throw std::runtime_error("Input failed");
  auto begin=std::chrono::steady_clock::now();
  if(sdk.ProcessNextFrame()!=VisionSDK::StatusCode::OK) throw std::runtime_error("Detect failed");
  auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();
  if(callback_count!=1) throw std::runtime_error("Unexpected callback count");
  output<<source<<','<<sdk_frame<<','<<ts<<','<<fall<<','<<callback_count<<','<<us<<',';
  if(fall) { std::vector<VisionSDK::MotionObject> objects; sdk.GetMotionObjects(objects); for(const auto& obj:objects) if(obj.is_fall_this_frame) output<<obj.id<<';'; }
  output<<'\n'; if(!output) throw std::runtime_error("Output write failed");
 }
 sdk.Release(); output.flush(); if(!output) throw std::runtime_error("Output flush failed"); return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
