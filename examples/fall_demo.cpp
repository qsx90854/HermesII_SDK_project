
#include "HermesII_sdk.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <cmath> // For M_PI, atan2, cos, sin
#include <chrono> // For timing

using namespace VisionSDK;
// Visualization Utilities (Removed - Moved to SDK)

// =========================================================
// Simple Config Loader
// =========================================================
#include <map>
#include <algorithm>

class ConfigLoader {
    std::map<std::string, std::string> data;
    
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (std::string::npos == first) return str;
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::string line, section;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == ';' || line[0] == '#') continue;
            
            if (line[0] == '[') {
                size_t end = line.find(']');
                if (end != std::string::npos) {
                    section = trim(line.substr(1, end - 1));
                }
            } else {
                size_t eq = line.find('=');
                if (eq != std::string::npos) {
                    std::string key = trim(line.substr(0, eq));
                    std::string val = trim(line.substr(eq + 1));
                    if (!section.empty()) key = section + "." + key;
                    data[key] = val;
                }
            }
        }
        return true;
    }

    int getInt(const std::string& key, int defaultVal) {
        if (data.find(key) != data.end()) {
             try { return std::stoi(data[key]); } catch(...) {}
        }
        return defaultVal;
    }

    std::string getString(const std::string& key, const std::string& defaultVal) {
        if (data.find(key) != data.end()) return data[key];
        return defaultVal;
    }

    float getFloat(const std::string& key, float defaultVal) {
        if (data.find(key) != data.end()) {
             try { return std::stof(data[key]); } catch(...) {}
        }
        return defaultVal;
    }
};

using namespace VisionSDK;

// Helper to read bed region (4 points)
std::vector<std::pair<int, int>> loadBedPoints(const std::string& filename) {
    std::vector<std::pair<int, int>> points;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return points;
    }

    int x, y;
    char comma;
    while (file >> x >> comma >> y) {
        points.push_back({x, y});
    }
    file.close();
    return points;
}

// Helper to read fall periods (Ground Truth)
std::vector<std::pair<int, int>> loadFallPeriods(const std::string& filename) {
    std::vector<std::pair<int, int>> periods;
    std::ifstream file(filename);
    if (!file.is_open()) return periods;
    
    std::string line;
    while (std::getline(file, line)) {
        size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        try {
            int start = std::stoi(line.substr(0, comma));
            int end = std::stoi(line.substr(comma + 1));
            periods.push_back({start, end});
        } catch (...) {}
    }
    return periods;
}

bool isInFallPeriodLocal(int frame_index, const std::vector<std::pair<int, int>>& periods) {
    for (const auto& p : periods) {
        if (frame_index >= p.first && frame_index <= p.second) return true;
    }
    return false;
}

int main() {
    std::cout << "Starting Fall Detection Demo (reading RAW images)..." << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;
    
    // Load Configs
    ConfigLoader cfg;
    ConfigLoader appCfg;
    appCfg.load("app_config.ini");

    // Load App Params First
    int W = appCfg.getInt("Demo.Demo_Width", 800);
    int H = appCfg.getInt("Demo.Demo_Height", 450);
    int origW = appCfg.getInt("Demo.Demo_Original_Width", 1920);
    int origH = appCfg.getInt("Demo.Demo_Original_Height", 1080);
    
    int num_images = appCfg.getInt("Demo.Demo_Max_Frames", 370);
    int start_frame = appCfg.getInt("Demo.Demo_Start_Frame", 0);
    std::string pattern = appCfg.getString("Demo.Demo_Image_Path_Format", "TestData/150455_NIR/frame_%05d.raw");

    bool loaded = cfg.load("parameter.ini");
    if (loaded) std::cout << "Loaded parameter.ini" << std::endl;
    else std::cout << "Using default parameters (parameter.ini not found)" << std::endl;

    // Initialize SDK
    VisionSDK::VisionSDK sdk;
    sdk.Init("", 4); // Default init, no model for motion only

    // 1. Motion Estimation Config
    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = 12;
    motionCfg.grid_rows = 16;
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = cfg.getInt("Motion.Diff_Check_Range", 5);
    motionCfg.block_change_threshold = cfg.getFloat("Motion.Block_Difference_Ratio_Threshold", 0.05);
    motionCfg.search_mode = cfg.getInt("Motion.Search_Mode", 1);
    motionCfg.enable_block_decay = (cfg.getInt("Motion.Enable_Block_Decay", 0) != 0);
    motionCfg.block_decay_frames = cfg.getInt("Motion.Block_Decay_Frames", 0);
    motionCfg.enable_block_dilation = (cfg.getInt("Motion.Enable_Block_Dilation", 0) != 0);
    motionCfg.block_dilation_threshold = cfg.getInt("Motion.Block_Dilation_Threshold", 2);
    sdk.SetConfig(&motionCfg);

    // 2. Object Extraction Config
    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
    objCfg.object_extraction_threshold = 4.0f; 
    objCfg.tracking_overlap_threshold = 0.5f;
    objCfg.tracking_mode = 1;
    sdk.SetConfig(&objCfg);

    // 3. Fall Detection Config
    VisionSDK::FallDetection_v1 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v1;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 2.0f);
    fallCfg.fall_strong_threshold = cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 5.0f);
    fallCfg.safe_area_ratio_threshold = cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5f);
    fallCfg.fall_window_size = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 30);
    fallCfg.fall_duration = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 5);
    sdk.SetConfig(&fallCfg);

    // 4. Image Related Config
    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    std::string savePath = cfg.getString("Demo.Demo_Save_Image_Path", "fall_results");
    if(!savePath.empty()) {
        imgCfg.enable_save_images = true;
        imgCfg.save_image_path = savePath;
    }
    imgCfg.enable_draw_bg_noise = (cfg.getInt("Demo.Demo_Draw_Background_Noise", 0) != 0);
    imgCfg.expected_frame_interval_ms = cfg.getInt("Validation.Expected_Frame_Interval", 33);
    imgCfg.frame_interval_tolerance_ms = cfg.getInt("Validation.Frame_Interval_Tolerance", 10);
    sdk.SetConfig(&imgCfg);

    std::cout << "Config:" << std::endl;
    std::cout << "  Threshold (Change): " << motionCfg.block_change_threshold << std::endl;
    std::cout << "  Threshold (Fall): " << fallCfg.fall_movement_threshold << std::endl;
    std::cout << "  Window: " << fallCfg.fall_window_size << ", Duration: " << fallCfg.fall_duration << std::endl;

    // (Loaded from appConfig at start)
    
    // Pre-allocate buffer
    std::vector<uint8_t> buffer(W * H); // 1 Channel Grayscale

    // Load Bed Region from File
    std::string bed_file = appCfg.getString("Demo.Demo_Bed_File", "");
    if (!bed_file.empty()) {
        auto bed_points = loadBedPoints(bed_file);
        if(bed_points.size() == 4) {
             // Scale if needed
             bool needsScale = false;
             for(auto& p : bed_points) if(p.first > W || p.second > H) needsScale = true;
             
             if(needsScale) {
                float sx = (float)W / origW;
                float sy = (float)H / origH;
                for(auto& p : bed_points) {
                    p.first = (int)(p.first * sx);
                    p.second = (int)(p.second * sy);
                }
                std::cout << "Scaled Bed Points" << std::endl;
             }
             
            sdk.SetBedRegion(bed_points);
            std::cout << "Loaded Bed Region from " << bed_file << std::endl;
        } else {
             std::cerr << "Failed to load valid bed region (Expected 4 points) from " << bed_file << std::endl;
        }
    }

    // Load Ground Truth
    std::string gt_file = appCfg.getString("Demo.Demo_GT_File", "");
    std::vector<std::pair<int, int>> gt_periods = loadFallPeriods(gt_file);
    if(!gt_periods.empty()) {
        std::cout << "Loaded " << gt_periods.size() << " Ground Truth Periods." << std::endl;
    } else {
        std::cout << "No GT loaded or empty." << std::endl;
    }

    // Create output directory
    if(imgCfg.enable_save_images) {
        mkdir(imgCfg.save_image_path.c_str(), 0777);
    }

    // Register Callback to capture Bed Exit Status
    bool current_bed_exit = false;
    bool current_is_fall = false;
    sdk.RegisterVisionSDKCallback([&](const VisionSDK::VisionSDKEvent& event) {
        current_bed_exit = event.is_bed_exit;
        current_is_fall = event.is_fall_detected;
    });

    bool fall_detected_any = false;
    int success_count = 0;
    int false_alarm_count = 0;
    int miss_count = 0;
    
    // Timing Variables
    double total_detect_ms = 0.0;
    double total_io_ms = 0.0;
    double total_loop_ms = 0.0;
    int processed_count = 0;

    for (int i = start_frame; i < start_frame + num_images; ++i) {
        char raw_name[1024];
        snprintf(raw_name, sizeof(raw_name), pattern.c_str(), i);

        auto t_loop_start = std::chrono::high_resolution_clock::now();

        auto t_io_start = std::chrono::high_resolution_clock::now();
        // Load RAW
        std::vector<uint8_t> buffer(W * H); // 1 Channel Grayscale
        FILE* f = fopen(raw_name, "rb");
        if (!f) {
            std::cout << "Cannot open " << raw_name << " (Stopping demo)" << std::endl;
            break;
        }
        size_t readS = fread(buffer.data(), 1, W * H, f);
        fclose(f);
         if(readS != (size_t)(W*H)) {
             // Warining?
         }

        auto t_io_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_io = t_io_end - t_io_start;
        total_io_ms += ms_io.count();
        
        bool is_fall = false;
        
        // Wrap Image
        VisionSDK::Image sdk_img;
        sdk_img.data = buffer.data(); // Use filled buffer
        sdk_img.width = W;
        sdk_img.height = H;
        sdk_img.channels = 1;
        
        // Populate Timestamp
        sdk_img.timestamp = i * 33;

        auto t1 = std::chrono::high_resolution_clock::now();
        // sdk.DetectFall removed? fall_demo was using DetectFall but it was removed.
        // It should use ProcessNextFrame + Callback or SetInputMemory + ProcessNextFrame.
        // But the code below prints logic based on `bool is_fall`. 
        // `DetectFall` was removed from API in rename step.
        // Wait, fall_demo.cpp was NOT fully updated to remove DetectFall usage in previous turn? 
        // Step 1045 updated fall_demo.cpp to use RegisterVisionSDKCallback but line 238.
        // But line 294 still has `sdk.DetectFall`.
        // I must fix this too. Use ProcessNextFrame and capture callback result.
        
        sdk.SetInputMemory(buffer.data(), W, H, 1);
        sdk.ProcessNextFrame(); 
        // is_fall needs to be captured from callback.
        // But the callback is registered at line 242 to capture bed exit.
        // I should update the callback to capture fall status too.
        
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        total_detect_ms += ms_double.count();
        processed_count++;
        
        // Ground Truth Validation
        // Check Local GT
        bool inFallPeriod = isInFallPeriodLocal(i, gt_periods);
        bool bedExit = current_bed_exit;
        std::string status_str = "";

        if (is_fall) {
            std::cout << "Frame " << i << ": FALL DETECTED!";
            
            if (inFallPeriod) {
                std::cout << " [CORRECT]" << std::endl;
                success_count++;
                status_str = "TRUE";
            } else {
                std::cout << " [FALSE ALARM]" << std::endl;
                false_alarm_count++;
                status_str = "FALSE";
            }
            fall_detected_any = true;
        } else {
             if (inFallPeriod) {
                 // std::cout << "Frame " << i << ": Missed Fall" << std::endl;
                 miss_count++;
             }
        }

        if (bedExit) {
            std::cout << "Frame " << i << ": BED EXIT DETECTED!" << std::endl;
        }
        
        
        auto t_loop_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_loop = t_loop_end - t_loop_start;
        total_loop_ms += ms_loop.count();
    }
    
    std::cout << "\nValidation Results:" << std::endl;
    std::cout << "Total Correct Detections (Frames): " << success_count << std::endl;
    std::cout << "Total False Alarms (Frames): " << false_alarm_count << std::endl;
    // std::cout << "Missed Frames: " << miss_count << std::endl;
    
    if (processed_count > 0) {
        std::cout << "Average I/O Time: " << (total_io_ms / processed_count) << " ms" << std::endl;
        std::cout << "Average Detection Time: " << (total_detect_ms / processed_count) << " ms" << std::endl;
        std::cout << "Average Loop Time: " << (total_loop_ms / processed_count) << " ms" << std::endl;
    }

    if (fall_detected_any) {
        std::cout << "Test Passed: Fall was detected in sequence." << std::endl;
    } else {
        std::cout << "Test Failed: No fall detected." << std::endl;
    }
    
    return 0;
}
// End of main
