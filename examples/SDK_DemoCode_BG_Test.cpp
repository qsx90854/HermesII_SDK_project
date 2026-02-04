
#include "HermesII_sdk.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <cstring>

using namespace VisionSDK;

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Simple BMP Saver
bool saveBMP_RGB(const std::string& filename, const uint8_t* rgbData, int width, int height) {
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) return false;

    int filesize = 54 + 3 * width * height;
    uint8_t header[54] = {
        0x42, 0x4D, 0,0,0,0, 0,0,0,0, 54,0,0,0, 40,0,0,0,
        0,0,0,0, 0,0,0,0, 1,0, 24,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
    };

    header[2] = (uint8_t)(filesize);
    header[3] = (uint8_t)(filesize >> 8);
    header[4] = (uint8_t)(filesize >> 16);
    header[5] = (uint8_t)(filesize >> 24);
    header[18] = (uint8_t)(width);
    header[19] = (uint8_t)(width >> 8);
    header[20] = (uint8_t)(width >> 16);
    header[21] = (uint8_t)(width >> 24);
    header[22] = (uint8_t)(height);
    header[23] = (uint8_t)(height >> 8);
    header[24] = (uint8_t)(height >> 16);
    header[25] = (uint8_t)(height >> 24);

    fwrite(header, 1, 54, f);

    int padSize = (4 - (width * 3) % 4) % 4;
    uint8_t pad[3] = {0, 0, 0};

    // BMP is bottom-to-top
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            // BMP expects BGR usually, but let's check input format.
            // Input is RGB Interleaved.
            uint8_t r = rgbData[idx];
            uint8_t g = rgbData[idx + 1];
            uint8_t b = rgbData[idx + 2];
            uint8_t bgr[3] = {b, g, r}; 
            fwrite(bgr, 1, 3, f);
        }
        fwrite(pad, 1, padSize, f);
    }
    fclose(f);
    return true;
}

// Drawing Helper
void drawRectRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, int rw, int rh, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0) x = 0; if (y < 0) y = 0;
    
    // Top & Bottom
    for(int cx = x; cx < x + rw && cx < w; cx++) {
        if(y >= 0 && y < h) {
             int idx = (y * w + cx) * 3;
             img[idx] = r; img[idx+1] = g; img[idx+2] = b;
        }
        if(y + rh - 1 >= 0 && y + rh - 1 < h) {
             int idx = ((y + rh - 1) * w + cx) * 3;
             img[idx] = r; img[idx+1] = g; img[idx+2] = b;
        }
    }
    // Left & Right
    for(int cy = y; cy < y + rh && cy < h; cy++) {
        if(x >= 0 && x < w) {
             int idx = (cy * w + x) * 3;
             img[idx] = r; img[idx+1] = g; img[idx+2] = b;
        }
        if(x + rw - 1 >= 0 && x + rw - 1 < w) {
             int idx = (cy * w + (x + rw - 1)) * 3;
             img[idx] = r; img[idx+1] = g; img[idx+2] = b;
        }
    }
}



// ==========================================
// Config Loader (Simple INI Parser)
// ==========================================
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

    std::string getString(const std::string& key, const std::string& defaultVal) {
        if (data.find(key) != data.end()) return data[key];
        return defaultVal;
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

// Global Stats
std::vector<int> detected_frames;
std::vector<uint8_t> current_frame_rgb; // Keep copy for drawing
int current_frame_idx = -1;
int pW = 800;
int pH = 450;
bool enable_save_face_images = false;


// Callback function
void onFallDetected(const VisionSDK::VisionSDKEvent& event) {
    if (event.is_fall_detected) {
        detected_frames.push_back(event.frame_index);
        std::cout << "\n>>> [CALLBACK] Event Detected! Frame: " << event.frame_index << " <<<" << std::endl;
        std::cout << "    Confidence: " << event.confidence << std::endl;
        std::cout << "    Type: " << (event.is_strong ? "Strong Fall" : "Normal Fall") << std::endl;
    }
    
    if (event.is_face && !current_frame_rgb.empty()) {
        std::cout << "[CALLBACK] Face Detected: " << event.face_x << "," << event.face_y << " " << event.face_w << "x" << event.face_h << std::endl;
        
        float scale_x = (float)pW / 128.0f;
        float scale_y = (float)pH / 128.0f;
        
        int draw_x = (int)(event.face_x * scale_x);
        int draw_y = (int)(event.face_y * scale_y);
        int draw_w = (int)(event.face_w * scale_x);
        int draw_h = (int)(event.face_h * scale_y);
        
        // Draw Rect (Red)
        drawRectRGB(current_frame_rgb, pW, pH, draw_x, draw_y, draw_w, draw_h, 255, 0, 0);
        
        // Save Image
        char filename[256];
        if (enable_save_face_images) {
            mkdir("FaceResult", 0777); 
            snprintf(filename, sizeof(filename), "FaceResult/face_det_%05d.bmp", current_frame_idx); 
            saveBMP_RGB(filename, current_frame_rgb.data(), pW, pH);
            std::cout << "Saved Face Image: " << filename << std::endl;
        } else {
             // std::cout << "[CALLBACK] Face Detected (Image Save Skipped)" << std::endl;
        }
    }
}

// Helper to read bed region
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

int main() {
    std::cout << "Starting Fall Callback Demo (No GT)..." << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;
    // 1. Load Configs
    ConfigLoader cfg;
    ConfigLoader appCfg;
    
    appCfg.load("app_config.ini");
    
    // Load App Params First (Dimensions, Files)
    int W = appCfg.getInt("Demo.Demo_Width", 800);
    int H = appCfg.getInt("Demo.Demo_Height", 450);
    int orgW = appCfg.getInt("Demo.Demo_Original_Width", 1920);
    int orgH = appCfg.getInt("Demo.Demo_Original_Height", 1080);
    
    std::string imgFormat = "TestData/images_150455_800x450_rgb_new/frame_%05d.raw"; // Hardcoded specific path
    std::string bedFile = appCfg.getString("Demo.Demo_Bed_File", ""); // Ignored for this test as per user request (focus on image loading and drawing)
    
    // int start_frame = appCfg.getInt("Demo.Demo_Start_Frame", 0);
    int start_frame = 0;
    int num_images = appCfg.getInt("Demo.Demo_Max_Frames", 500); // Increased default
    // int num_images = 100;
    
    // Config: Enable/Disable Face Saving
    enable_save_face_images = (appCfg.getInt("Demo.Demo_Save_Face_Images", 0) != 0);
    std::string pattern = imgFormat;
    
    // Override W/H
    pW = 800;
    pH = 450;
    W = 800;
    H = 450;

    // BG Mask Controls
    bool enable_save_bg_mask = false;
    int bg_init_start = 0;
    int bg_init_end = 0;
    int bg_diff_threshold = 30;

    if (cfg.load("parameter.ini")) {
        std::cout << "Loaded parameter.ini" << std::endl;
        enable_save_bg_mask = (cfg.getInt("FallDetect.Enable_Save_BG_Mask", 0) != 0);
        bg_init_start = cfg.getInt("FallDetect.BG_Init_Start_Frame", 10);
        bg_init_end = cfg.getInt("FallDetect.BG_Init_End_Frame", 20);
        bg_diff_threshold = cfg.getInt("FallDetect.BG_Diff_Threshold", 30);
    } else {
        std::cerr << "Warning: parameter.ini not found, using defaults." << std::endl;
    }
    
    // Config: Enable/Disable Face Saving (Loaded from app_config, but we care more about param.ini above)
    // enable_save_face_images = (appCfg.getInt("Demo.Demo_Save_Face_Images", 0) != 0);

    // 2. Initialize SDK
    VisionSDK::VisionSDK sdk;
    sdk.Init("", 4); // Default Init

    // 1. Motion Estimation Config
    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = cfg.getInt("Motion.Grid_Cols", 12);
    motionCfg.grid_rows = cfg.getInt("Motion.Grid_Rows", 16);
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = cfg.getInt("Motion.Diff_Check_Range", 5);
    motionCfg.block_change_threshold = cfg.getFloat("Motion.Block_Difference_Ratio_Threshold", 0.03);
    motionCfg.search_mode = cfg.getInt("Motion.Search_Mode", 1);
    motionCfg.enable_block_decay = (cfg.getInt("Motion.Enable_Block_Decay", 1) != 0);
    motionCfg.block_decay_frames = cfg.getInt("Motion.Block_Decay_Frames", 3);
    motionCfg.enable_block_dilation = (cfg.getInt("Motion.Enable_Block_Dilation", 1) != 0);
    motionCfg.block_dilation_threshold = cfg.getInt("Motion.Block_Dilation_Threshold", 2);
    sdk.SetConfig(&motionCfg);

    // 2. Object Extraction Config
    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
    objCfg.object_extraction_threshold = 2.0f; 
    objCfg.tracking_overlap_threshold = cfg.getFloat("Tracking.Tracking_Overlap_Threshold", 0.5f);
    objCfg.tracking_mode = 1;
    sdk.SetConfig(&objCfg);

    // 3. Fall Detection Config
    // 3. Fall Detection Config
    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 3.0);
    fallCfg.fall_strong_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 8.0);
    fallCfg.fall_acceleration_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Acceleration_Threshold", 5.0f);
    fallCfg.safe_area_ratio_threshold = (float)cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5);
    fallCfg.fall_window_size = 30;
    fallCfg.fall_duration = 5;
    fallCfg.fall_duration = 5;
    fallCfg.enable_face_detection = (cfg.getInt("FallDetect.Enable_Face_Detection", 1) != 0);
    
    // BG Params
    fallCfg.enable_save_bg_mask = enable_save_bg_mask;
    fallCfg.bg_init_start_frame = bg_init_start;
    fallCfg.bg_init_end_frame = bg_init_end;
    fallCfg.bg_diff_threshold = bg_diff_threshold;
    // Set Update Rate from Ini too (previously hardcoded or default?)
    fallCfg.bg_update_interval_frames = cfg.getInt("FallDetect.BG_Update_Interval", 10);
    fallCfg.bg_update_alpha = cfg.getFloat("FallDetect.BG_Update_Alpha", 0.01f);
    
    sdk.SetConfig(&fallCfg);

    // 4. Image Related Config
    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    std::string savePath = cfg.getString("Demo.Demo_Save_Image_Path", "fall_results_cb_no_gt");
    if(!savePath.empty()) {
        imgCfg.enable_save_images = true;
        imgCfg.save_image_path = savePath;
        // Create dir if needed
        mkdir(savePath.c_str(), 0777);
    }
    imgCfg.enable_draw_bg_noise = (appCfg.getInt("Demo.Demo_Draw_Background_Noise", 0) != 0);
    imgCfg.expected_frame_interval_ms = cfg.getInt("Validation.Expected_Frame_Interval", 33);
    imgCfg.frame_interval_tolerance_ms = cfg.getInt("Validation.Frame_Interval_Tolerance", 10);
    sdk.SetConfig(&imgCfg);


    sdk.RegisterVisionSDKCallback(onFallDetected);

    // 3. Load Demo Resources
    
    // Bed Region
    if (!bedFile.empty()) {
        auto bed_points = loadBedPoints(bedFile);
        if (bed_points.size() == 4) {
             // Basic scaling assumption
             bool needsScale = false;
             for(auto& p : bed_points) if(p.first > W || p.second > H) needsScale = true;
             
             if(needsScale) {
                float scale_x = (float)W / orgW;
                float scale_y = (float)H / orgH;
                std::cout << "Scaling Bed Points: " << scale_x << "x" << scale_y << std::endl;
                for (auto& p : bed_points) {
                    p.first = (int)(p.first * scale_x);
                    p.second = (int)(p.second * scale_y);
                }
             }
            sdk.SetBedRegion(bed_points);
            std::cout << "Bed region loaded from " << bedFile << std::endl;
        }
    }

    // 4. Processing Loop
    // size_t frame_size = W * H; // Gray
    size_t frame_size_rgb = W * H * 3; // RGB
    
    std::vector<uint8_t> file_buffer(frame_size_rgb);
    
    sdk.SetInputMemory(file_buffer.data(), W, H, 3);
    
    // Keep reference for callback drawing
    current_frame_rgb = std::vector<uint8_t>(frame_size_rgb);

    for (int i = start_frame; i < start_frame + num_images; ++i) {
        char raw_name[256];
        snprintf(raw_name, sizeof(raw_name), pattern.c_str(), i);
        
        std::ifstream file(raw_name, std::ios::binary);
        if (!file) {
            if (i > 0) break; 
            continue;
        }
        file.read(reinterpret_cast<char*>(file_buffer.data()), frame_size_rgb);
        file.close();

        // Copy for drawing
        memcpy(current_frame_rgb.data(), file_buffer.data(), frame_size_rgb);
        current_frame_idx = i;

        auto t1 = std::chrono::steady_clock::now();
        sdk.SetInputMemory(file_buffer.data(), W, H, 3);
        sdk.ProcessNextFrame();
        auto t2 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (i % 50 == 0) {
            std::cout << "Frame " << i << " Total Process Time: " << ms << " ms (" << (1000.0/ms) << " FPS)" << std::endl;
        }
    }

    // 5. Evaluation Removed
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing Complete." << std::endl;
    std::cout << "Detected " << detected_frames.size() << " events." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
