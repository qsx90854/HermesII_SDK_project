#include "HermesII_sdk.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <cstring>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace VisionSDK;

// ==========================================
// Config Loader
// ==========================================
class ConfigLoader {
    std::map<std::string, std::string> data;
    std::vector<std::string> keys; // Keep order

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
                    keys.push_back(key);
                }
            }
        }
        return true;
    }

    std::string getString(const std::string& key, const std::string& defaultVal) {
        if (data.find(key) != data.end()) return data[key];
        return defaultVal;
    }
    
    // Get all keys starting with prefix
    std::vector<std::string> getKeys(const std::string& prefix) {
        std::vector<std::string> result;
        for(const auto& k : keys) {
            if(k.find(prefix) == 0 && k.find(prefix + ".") == 0) {
                 result.push_back(k);
            }
        }
        return result;
    }

    int getInt(const std::string& key, int defaultVal) {
        if (data.find(key) != data.end()) {
            try { return std::stoi(data[key]); } catch(...) {}
        }
        return defaultVal;
    }

    float getFloat(const std::string& key, float defaultVal) {
        if (data.find(key) != data.end()) {
            try { return std::stof(data[key]); } catch(...) {}
        }
        return defaultVal;
    }
};

// ==========================================
// Helpers
// ==========================================
std::vector<std::pair<int, int>> loadBedPoints(const std::string& filename) {
    std::vector<std::pair<int, int>> points;
    std::ifstream file(filename);
    if (!file.is_open()) return points;
    int x, y;
    char comma;
    while (file >> x >> comma >> y) {
        points.push_back({x, y});
    }
    return points;
}

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

bool isFrameInPeriods(int frame, const std::vector<std::pair<int, int>>& periods) {
    for (auto& p : periods) {
        if (frame >= p.first && frame <= p.second) return true;
    }
    return false;
}

// Global collector for callback
std::vector<int> current_detected_frames;
void onFallDetected(const VisionSDKEvent& event) {
    if (event.is_fall_detected) {
        current_detected_frames.push_back(event.frame_index);
    }
}

// Params
struct TestParams {
    std::string name;
    std::string pattern;
    std::string bedFile;
    std::string gtFile;
};

// Helper to run a single test case
void run_test_case(const TestParams& ds, int mode, const std::string& modeName, ConfigLoader& cfg, std::ofstream& csv) {
    std::cout << "Testing " << ds.name << " Mode: " << modeName << "..." << std::flush;
    
    // Reset Global
    current_detected_frames.clear();
    
    // Init SDK inside function -> Clean Scope
    VisionSDK::VisionSDK sdk;
    sdk.Init("", 4);

     // 1. Motion Estimation Config
    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = cfg.getInt("Motion.Grid_Cols", 12);
    motionCfg.grid_rows = cfg.getInt("Motion.Grid_Rows", 16);
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = cfg.getInt("Motion.Diff_Check_Range", 5);
    motionCfg.block_change_threshold = cfg.getFloat("Motion.Block_Difference_Ratio_Threshold", 0.03f);
    motionCfg.search_mode = mode; // 1 or 0
    motionCfg.enable_block_decay = (cfg.getInt("Motion.Enable_Block_Decay", 1) != 0);
    motionCfg.block_decay_frames = cfg.getInt("Motion.Block_Decay_Frames", 5);
    motionCfg.enable_block_dilation = (cfg.getInt("Motion.Enable_Block_Dilation", 1) != 0);
    motionCfg.block_dilation_threshold = cfg.getInt("Motion.Block_Dilation_Threshold", 2);
    sdk.SetConfig(&motionCfg);

    // 2. Object Extraction Config
    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_extraction_threshold = cfg.getFloat("Object.Block_Extraction_Threshold", 2.0f);
    objCfg.tracking_overlap_threshold = cfg.getFloat("Tracking.Tracking_Overlap_Threshold", 0.5f);
    objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
    objCfg.tracking_mode = 1;
    sdk.SetConfig(&objCfg);

    // 3. Fall Detection Config
    // 3. Fall Detection Config
    VisionSDK::FallDetection_v2 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v2;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 3.0f);
    fallCfg.fall_strong_threshold = cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 8.0f);
    fallCfg.safe_area_ratio_threshold = cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5f);
    fallCfg.fall_acceleration_threshold = 5.0f; // Default
    fallCfg.fall_window_size = 30;
    fallCfg.fall_duration = 5;
    fallCfg.enable_face_detection = (cfg.getInt("FallDetect.Enable_Face_Detection", 1) != 0);
    sdk.SetConfig(&fallCfg);

    // 4. Image Related Config (Disable Saving)
    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    imgCfg.enable_save_images = false; 
    sdk.SetConfig(&imgCfg);

    sdk.RegisterVisionSDKCallback(onFallDetected);

    // Load Resources
    int W = 800; // Assume 800x450 for now. Could be read from INI if varied.
    int H = 450; 
    int origW = 1920; 
    int origH = 1080;
    
    auto bed_points = loadBedPoints(ds.bedFile);
    if (bed_points.size() == 4) {
        float sx = (float)W / origW;
        float sy = (float)H / origH;
        for(auto& p : bed_points) {
            p.first = (int)(p.first * sx);
            p.second = (int)(p.second * sy);
        }
        sdk.SetBedRegion(bed_points);
    }
    
    auto gt_periods = loadFallPeriods(ds.gtFile);
    
    // Run Loop
    size_t frame_size_gray = W * H;
    size_t frame_size_rgb = W * H * 3;
    std::vector<uint8_t> buffer(frame_size_rgb); // Max size

    int processedFrames = 0;
    for (int i = 0; ; ++i) {
        char path[512];
        snprintf(path, sizeof(path), ds.pattern.c_str(), i);
        std::ifstream f(path, std::ios::binary | std::ios::ate); // Open at end to get size
        if (!f) {
            processedFrames = i; 
            if (i > 0) break; 
            continue; 
        }
        
        std::streamsize size = f.tellg();
        f.seekg(0, std::ios::beg);
        
        int channels = 1;
        if (size == frame_size_rgb) {
            channels = 3;
            f.read((char*)buffer.data(), frame_size_rgb);
        } else if (size == frame_size_gray) {
            channels = 1;
            f.read((char*)buffer.data(), frame_size_gray);
        } else {
             // Unknown size, assume Gray or warning?
             // Just read what we can
             f.read((char*)buffer.data(), size);
        }

        sdk.SetInputMemory(buffer.data(), W, H, channels);
        sdk.ProcessNextFrame();
    }
    
    // Evaluate
    bool firstDetected = false;
    int totalFalls = gt_periods.size();
    int detectedCount = 0;
    int fpFrames = 0;
    
    // Check First Fall
    if (totalFalls > 0) {
            for(int f : current_detected_frames) {
                if (f >= gt_periods[0].first && f <= gt_periods[0].second) {
                    firstDetected = true;
                    break;
                }
            }
    }
    
    // Check All Falls
    for(const auto& p : gt_periods) {
        bool hit = false;
        for(int f : current_detected_frames) {
            if (f >= p.first && f <= p.second) {
                hit = true; break;
            }
        }
        if(hit) detectedCount++;
    }
    
    // Check FP
    for(int f : current_detected_frames) {
        if(!isFrameInPeriods(f, gt_periods)) fpFrames++;
    }
    
    // Write CSV
    csv << ds.name << "," << modeName << "," 
        << (firstDetected ? "Yes" : "No") << ","
        << totalFalls << "," 
        << detectedCount << "," 
        << fpFrames << ","
        << processedFrames << "\n";
        
    std::cout << " Done. (Det:" << detectedCount << "/" << totalFalls << " FP:" << fpFrames << " Frames:" << processedFrames << ")" << std::endl;
}

// ==========================================
// Main
// ==========================================
int main() {
    std::cout << "Starting Use Case: SDK_DemoCode_v1 Accuracy Test" << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;

    ConfigLoader cfg;
    if (!cfg.load("test_datasets.ini")) {
        std::cerr << "Error: Cannot load test_datasets.ini" << std::endl;
        return 1;
    }
    cfg.load("parameter.ini"); // Load additional parameters

    std::string resultCSV = cfg.getString("General.ResultCSV", "accuracy_results.csv");
    std::ofstream csv(resultCSV);
    csv << "Dataset,Mode,FirstFallDetected,TotalFalls,DetectedFalls,FalseAlarmFrames,TotalFrames\n";

    // Parse Datasets
    std::vector<TestParams> datasets;
    auto keys = cfg.getKeys("Datasets");
    
    for (const auto& key : keys) {
        std::string val = cfg.getString(key, "");
        if (val.empty()) continue;
        
        // Split val by comma
        std::vector<std::string> parts;
        std::stringstream ss(val);
        std::string item;
        while (std::getline(ss, item, ',')) {
            parts.push_back(item);
        }
        
        if (parts.size() >= 3) {
            TestParams tp;
            tp.name = key.substr(9); // Remove "Datasets."
            tp.pattern = parts[0];
            tp.bedFile = parts[1];
            tp.gtFile = parts[2];
            datasets.push_back(tp);
        }
    }
    
    std::cout << "Found " << datasets.size() << " datasets." << std::endl;

    // Modes: 0 (Low), 1 (High)
    int modes[] = {1, 0}; 
    std::string modeNames[] = {"High", "Low"};

    for (const auto& ds : datasets) {
        for (int m = 0; m < 2; ++m) {
            run_test_case(ds, modes[m], modeNames[m], cfg, csv);
        }
    }
    
    csv.close();
    std::cout << "Results saved to " << resultCSV << std::endl;
    return 0;
}
