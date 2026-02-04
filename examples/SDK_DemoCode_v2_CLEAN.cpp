
#include "HermesII_sdk.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <cstring>
#include <thread>
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace VisionSDK;

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Drawing Helper
void drawRectRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, int rw, int rh, uint8_t r, uint8_t g, uint8_t b, int thickness=2) {
    if (x < 0) x = 0; if (y < 0) y = 0;
    
    // Top & Bottom
    for(int cx = x; cx < x + rw && cx < w; cx++) {
        for(int t=0; t<thickness; ++t) {
            if(y+t >= 0 && y+t < h) {
                 int idx = ((y+t) * w + cx) * 3;
                 img[idx] = r; img[idx+1] = g; img[idx+2] = b;
            }
            if(y + rh - 1 - t >= 0 && y + rh - 1 - t < h) {
                 int idx = ((y + rh - 1 - t) * w + cx) * 3;
                 img[idx] = r; img[idx+1] = g; img[idx+2] = b;
            }
        }
    }
    // Left & Right
    for(int cy = y; cy < y + rh && cy < h; cy++) {
         for(int t=0; t<thickness; ++t) {
            if(x+t >= 0 && x+t < w) {
                 int idx = (cy * w + (x+t)) * 3;
                 img[idx] = r; img[idx+1] = g; img[idx+2] = b;
            }
            if(x + rw - 1 - t >= 0 && x + rw - 1 - t < w) {
                 int idx = (cy * w + (x + rw - 1 - t)) * 3;
                 img[idx] = r; img[idx+1] = g; img[idx+2] = b;
            }
         }
    }
}

void drawLine(std::vector<uint8_t>& img, int w, int h, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness=2) {
    int dx = std::abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    int dy = -std::abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    int err = dx + dy, e2;
    
    // Safety check
    if (x1 == x2 && y1 == y2) return;

    for (;;) {
        for(int t=-thickness/2; t<=thickness/2; ++t) {
            for(int k=-thickness/2; k<=thickness/2; ++k) {
                int px = x1 + t;
                int py = y1 + k;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    int idx = (py * w + px) * 3;
                    img[idx] = r; img[idx+1] = g; img[idx+2] = b;
                }
            }
        }
        
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x1 += sx; }
        if (e2 <= dx) { err += dx; y1 += sy; }
    }
}

void drawArrow(std::vector<uint8_t>& img, int w, int h, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness=2) {
    // 1. Draw Shaft
    drawLine(img, w, h, x1, y1, x2, y2, r, g, b, thickness);

    // 2. Draw Arrowhead (Based on C_V2_EDGE.cpp drawArrow2)
    // IMPORTANT: Use Forward Vector (x1 -> x2)
    float dx = (float)(x2 - x1);
    float dy = (float)(y2 - y1);
    
    // 1. Calculate length
    float len = std::sqrt(dx*dx + dy*dy);
    if (len < 1.0f) return;
    
    // 2. Head Specs
    // Length: 20% of arrow length, min 3.0f
    float head_len = std::max(3.0f, len * 0.2f);
    // Angle: 30 degrees
    float angle = 30.0f * 3.14159265f / 180.0f;
    
    float ux = dx / len;
    float uy = dy / len;

    float sin_a = std::sin(angle);
    float cos_a = std::cos(angle);

    // 3. Calculate Wing Vectors (Rotated Forward Vector)
    
    float lx = cos_a * ux - sin_a * uy;
    float ly = sin_a * ux + cos_a * uy;
    float rx = cos_a * ux + sin_a * uy;
    float ry = -sin_a * ux + cos_a * uy;

    // 4. Calculate End Points
    // From Tip (x2, y2) backwards
    int p1x = (int)(x2 - head_len * lx);
    int p1y = (int)(y2 - head_len * ly);
    int p2x = (int)(x2 - head_len * rx);
    int p2y = (int)(y2 - head_len * ry);

    // Draw Wings
    drawLine(img, w, h, x2, y2, p1x, p1y, r, g, b, thickness);
    drawLine(img, w, h, x2, y2, p2x, p2y, r, g, b, thickness);
}

// Minimal 5x7 bitmap font helper (for text)
const uint8_t font5x7[] = {
    // 0-9
    0x1F,0x11,0x1F, 0x00,0x1F,0x00, 0x1D,0x15,0x17, 0x15,0x15,0x1F, 0x07,0x04,0x1F,
    0x17,0x15,0x1D, 0x1F,0x15,0x1D, 0x01,0x01,0x1F, 0x1F,0x15,0x1F, 0x17,0x15,0x1F,
    // .
    0x10,0x00,0x00,
    // F (11)
    0x1F,0x05,0x00,
    // A (12)
    0x1F,0x05,0x1F, 
    // L (13)
    0x1F,0x10,0x10,
    // T (14)
    0x01,0x1F,0x01,
    // R (15)
    0x1F,0x05,0x1A,
    // U (16)
    0x1F,0x10,0x1F,
    // E (17)
    0x1F,0x15,0x11,
    // S (18)
    0x1D,0x15,0x17,
     // M (19)
    0x1F,0x02,0x1F,
    // I (20)
    0x00,0x1F,0x00,
    // B (21)
    0x1F,0x15,0x0A,
    // D (22)
    0x1F,0x11,0x0E,
    // X (23)
    0x11,0x04,0x11,
    // Space (24)
    0x00,0x00,0x00,
    // Y (25)
    0x03,0x1C,0x03, 
    // O (26)
    0x1F,0x11,0x1F,
    // N (27)
    0x1F,0x02,0x1F,
    // G (28)
    0x1F,0x11,0x1D,
};

void drawChar(std::vector<uint8_t>& img, int w, int h, int cx, int cy, int charArgs, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
    if (charArgs < 0 || charArgs > 28) return;
    const uint8_t* ptr = font5x7 + charArgs * 3;
    for (int col = 0; col < 3; col++) {
        uint8_t colData = ptr[col];
        for (int row = 0; row < 5; row++) {
            if ((colData >> row) & 1) { 
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = cx + (col * 2) * scale + sx;
                        int py = cy + (row * 2) * scale + sy;
                        if(px>=0 && px<w && py>=0 && py<h) {
                             int idx = (py * w + px) * 3;
                             img[idx] = r; img[idx+1] = g; img[idx+2] = b;
                        }
                    }
                }
            }
        }
    }
}

void drawString(std::vector<uint8_t>& img, int w, int h, int x, int y, const std::string& s, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
    int cx = x;
    for (char c : s) {
        int idx = -1;
        if (c >= '0' && c <= '9') idx = c - '0';
        else if (c == '.') idx = 10;
        else if (c == 'F') idx = 11;
        else if (c == 'A') idx = 12; // L:13, T:14, R:15, U:16, E:17, S:18, M:19, I:20, B:21, D:22, X:23
        else if (c == 'L') idx = 13;
        else if (c == 'T') idx = 14;
        else if (c == 'R') idx = 15;
        else if (c == 'U') idx = 16;
        else if (c == 'E') idx = 17;
        else if (c == 'S') idx = 18;
        else if (c == 'M') idx = 19;
        else if (c == 'I') idx = 20;
        else if (c == 'B') idx = 21;
        else if (c == 'D') idx = 22;
        else if (c == 'X') idx = 23;
        else if (c == ' ') idx = 24;
        else if (c == 'Y') idx = 25;
        else if (c == 'O') idx = 26;
        else if (c == 'N') idx = 27;
        else if (c == 'G') idx = 28;
        
        if (idx != -1) {
            drawChar(img, w, h, cx, y, idx, r, g, b, scale);
            cx += 8 * scale; 
        } else {
             cx += 6 * scale; 
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
std::vector<uint8_t> clean_frame_rgb; // New: For saving original crops
std::vector<uint8_t> current_frame_rgb_onlyBlock; // Keep copy for drawing
int current_frame_idx = -1;
int pW = 800;
int pH = 450;
bool enable_save_face_images = false;
bool is_fall_in_current_frame = false;
bool is_strong_fall = false;
bool is_bed_exit_in_current_frame = false;
int global_fall_event_id = 0;
int fall_red_box_countdown = 0;

// Body Part Tracking State
struct BodyPartState {
    float upper_x, upper_y; // Last known Upper Body Centroid
    float lower_x, lower_y; // Last known Lower Body Centroid
    bool initialized;
};
std::map<int, BodyPartState> object_part_states;


// Callback function
void onFallDetected(const VisionSDK::VisionSDKEvent& event) {
    is_fall_in_current_frame = event.is_fall_detected;
    is_strong_fall = event.is_strong;
    is_bed_exit_in_current_frame = event.is_bed_exit;

    if (event.is_fall_detected) {
        // Warning Logic
        fall_red_box_countdown = 20;
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


// Delayed Decision Buffer
struct FrameContext {
    int frame_idx;
    // std::vector<uint8_t> rgb_data; // REMOVED to save memory
    int w, h;
    bool is_fall_detected; // SDK result
    std::vector<int> fall_obj_ids; 
    
    struct ObjSnapshot {
        int id;
        double angle; // PCA angle
    };
    std::vector<ObjSnapshot> objects;
};

// Global Buffer
std::deque<FrameContext> frame_buffer;
const int DELAY_FRAMES = 30;

int main() {
    std::cout << "Starting Fall Callback Demo v2 SAVE (30FPS Sim)..." << std::endl;
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
    
    std::string imgFormat = appCfg.getString("Demo.Demo_Image_Path_Format", "TestData/images_150455_800x450_rgb_new/frame_%05d.raw");
    std::string bedFile = appCfg.getString("Demo.Demo_Bed_File", ""); 
    std::string save_dir = appCfg.getString("Demo.Demo_Output_Dir", "0125_bgtest");
    
    int start_frame = appCfg.getInt("Demo.Demo_Start_Frame", 0);
    int num_images = appCfg.getInt("Demo.Demo_Max_Frames", 1920);
    int frame_step = appCfg.getInt("Demo.Demo_Frame_Step", 1);
    if (frame_step < 1) frame_step = 1;
    
    // Config: Enable/Disable Face Saving
    enable_save_face_images = (appCfg.getInt("Demo.Demo_Save_Face_Images", 0) != 0);
    std::string pattern = imgFormat;
    
    pW = W;
    pH = H;

    if (cfg.load("parameter.ini")) {
        std::cout << "Loaded parameter.ini" << std::endl;
        std::cout << "DEBUG: Object.Block_Merge_Range = " << cfg.getInt("Object.Block_Merge_Range", -1) << std::endl;
        std::cout << "DEBUG: Motion.Block_Dilation_Threshold = " << cfg.getInt("Motion.Block_Dilation_Threshold", -1) << std::endl;
    } else {
        std::cerr << "Warning: parameter.ini not found, using defaults." << std::endl;
    }

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
    objCfg.tracking_mode = cfg.getInt("Tracking.Tracking_Mode", 1);
    sdk.SetConfig(&objCfg);

    // 3. Fall Detection Config
    // 3. Fall Detection Config
    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 3.0);
    fallCfg.fall_strong_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 8.0);
    fallCfg.fall_acceleration_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Acceleration_Threshold", 5.0f);
    fallCfg.fall_acceleration_upper_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Accel_Upper_Threshold", 2.0f);
    fallCfg.fall_acceleration_lower_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Accel_Lower_Threshold", -2.0f);
    fallCfg.safe_area_ratio_threshold = (float)cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5);
    fallCfg.fall_window_size = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 30);
    fallCfg.fall_duration = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 5);
    fallCfg.post_fall_distance_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Post_Fall_Distance_Threshold", 10.0f);
    fallCfg.post_fall_check_frames = cfg.getInt("FallDetect.Fall_Detect_Post_Fall_Check_Frames", 5);
    std::cout << "DEBUG: Loaded Window Size: " << fallCfg.fall_window_size << std::endl;
    std::cout << "DEBUG: Loaded Duration: " << fallCfg.fall_duration << std::endl;
    std::cout << "DEBUG: Loaded Duration: " << fallCfg.fall_duration << std::endl;
    fallCfg.enable_face_detection = (cfg.getInt("FallDetect.Enable_Face_Detection", 1) != 0);
    // Load Verification Flags
    fallCfg.enable_bed_exit_verification = (cfg.getInt("FallDetect.Enable_Bed_Exit_Verification", 1) != 0);
    fallCfg.enable_block_shrink_verification = (cfg.getInt("FallDetect.Enable_Block_Shrink_Verification", 1) != 0);
    
    // Background Method Config
    fallCfg.enable_save_bg_mask = (cfg.getInt("FallDetect.Enable_Save_BG_Mask", 0) != 0);
    fallCfg.bg_init_start_frame = cfg.getInt("FallDetect.BG_Init_Start_Frame", 10);
    fallCfg.bg_init_end_frame = cfg.getInt("FallDetect.BG_Init_End_Frame", 20);
    fallCfg.bg_diff_threshold = cfg.getInt("FallDetect.BG_Diff_Threshold", 30);
    fallCfg.bg_update_interval_frames = cfg.getInt("FallDetect.BG_Update_Interval", 10);
    fallCfg.bg_update_alpha = cfg.getFloat("FallDetect.BG_Update_Alpha", 0.01f);

    sdk.SetConfig(&fallCfg);

    // 4. Image Related Config
    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    std::string savePath = cfg.getString("Demo.Demo_Save_Image_Path", "fall_results_cb_no_gt");
    imgCfg.save_image_path = savePath;
    imgCfg.enable_save_images = false; // We do manual saving here
    imgCfg.enable_draw_bg_noise = (appCfg.getInt("Demo.Demo_Draw_Background_Noise", 0) != 0);
    imgCfg.expected_frame_interval_ms = cfg.getInt("Validation.Expected_Frame_Interval", 33);
    imgCfg.frame_interval_tolerance_ms = cfg.getInt("Validation.Frame_Interval_Tolerance", 10);
    sdk.SetConfig(&imgCfg);

    sdk.RegisterVisionSDKCallback(onFallDetected);

    // 3. Load Demo Resources
    std::vector<std::pair<int, int>> bed_points_sorted;
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
                for (auto& p : bed_points) {
                    p.first = (int)(p.first * scale_x);
                    p.second = (int)(p.second * scale_y);
                }
             }
            sdk.SetBedRegion(bed_points);
            bed_points_sorted = bed_points; // Store for drawing
            std::cout << "Bed region loaded from " << bedFile << std::endl;
        }
    }

    // 4. Processing Loop
    size_t frame_size_rgb = W * H * 3; 
    std::vector<uint8_t> file_buffer(frame_size_rgb);
    sdk.SetInputMemory(file_buffer.data(), W, H, 3);
    current_frame_rgb = std::vector<uint8_t>(frame_size_rgb);
    clean_frame_rgb = std::vector<uint8_t>(frame_size_rgb);
    current_frame_rgb_onlyBlock = std::vector<uint8_t>(frame_size_rgb);

    mkdir(save_dir.c_str(), 0777);
    mkdir("v2_object_crops", 0777);

    // Background Accumulation Variables
    int bg_init_start_frame = fallCfg.bg_init_start_frame;
    int bg_init_end_frame = fallCfg.bg_init_end_frame;
    std::vector<uint32_t> bg_accumulator;
    std::vector<uint8_t> bg_reference; // Persistent BG for PCA
    std::map<int, double> obj_smooth_thetas; // Prev angles for smoothing
    int bg_frames_count = 0;
    bool bg_saved_flag = false;

    for (int i = start_frame; i < start_frame + num_images; i += frame_step) {
        
        auto t_read_start = std::chrono::steady_clock::now();

        char raw_name[256];
        snprintf(raw_name, sizeof(raw_name), pattern.c_str(), i);
        
        std::ifstream file(raw_name, std::ios::binary);
        if (!file) {
            if (i > 0) break; 
            continue;
        }
        file.read(reinterpret_cast<char*>(file_buffer.data()), frame_size_rgb);
        file.close();

        // Timer End for Read
        auto t_read_end = std::chrono::steady_clock::now();
        double read_duration_ms = std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();

        // Simulate 30FPS Camera (Ensure Read+Wait = 33ms)
        if (read_duration_ms < 33.0) {
            int sleep_ms = (int)(33.0 - read_duration_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }

        // Copy for drawing
        memcpy(current_frame_rgb.data(), file_buffer.data(), frame_size_rgb);
        memcpy(clean_frame_rgb.data(), file_buffer.data(), frame_size_rgb);
        memset(current_frame_rgb_onlyBlock.data(), 0, frame_size_rgb);
        current_frame_idx = i;
        is_fall_in_current_frame = false; // Reset
        is_bed_exit_in_current_frame = false;

        // Background Averaging Logic
        if (i >= bg_init_start_frame && i <= bg_init_end_frame) {
            if (bg_accumulator.empty()) bg_accumulator.resize(frame_size_rgb, 0); // Re-size if needed
            for(size_t k=0; k<frame_size_rgb; ++k) {
                bg_accumulator[k] += current_frame_rgb[k];
            }
            bg_frames_count++;
        }
        else if (i > bg_init_end_frame && bg_frames_count > 0 && !bg_saved_flag) {
            // Save BG
            bg_reference.resize(frame_size_rgb); // Store for PCA
            for(size_t k=0; k<frame_size_rgb; ++k) {
                bg_reference[k] = (uint8_t)(bg_accumulator[k] / bg_frames_count);
            }
            char bg_filename[256];
            snprintf(bg_filename, sizeof(bg_filename), "%s/background_init.jpg", save_dir.c_str());
            stbi_write_jpg(bg_filename, W, H, 3, bg_reference.data(), 90);
            std::cout << "[Demo] Saved averaged background to " << bg_filename << " (Frames " 
                      << bg_init_start_frame << "-" << bg_init_end_frame << ")" << std::endl;
            
            // Clean up accumulator but KEEP bg_reference
            bg_accumulator.clear(); 
            bg_accumulator.shrink_to_fit();
            bg_saved_flag = true;
        }
        // --- DYNAMIC BACKGROUND UPDATE (User Request) ---
        else if (bg_saved_flag && !bg_reference.empty()) {
            int interval = fallCfg.bg_update_interval_frames;
            float alpha = fallCfg.bg_update_alpha;
            
            if (interval > 0 && (i % interval == 0)) {
                float one_minus_alpha = 1.0f - alpha;
                for(size_t k=0; k<frame_size_rgb; ++k) {
                    float old_val = (float)bg_reference[k];
                    float curr_val = (float)current_frame_rgb[k];
                    bg_reference[k] = (uint8_t)(one_minus_alpha * old_val + alpha * curr_val);
                }
                // Optional: Print status periodically
                if (i % (interval*100) == 0) {
                     std::cout << "[Demo] Updated Background at Frame " << i << " (Alpha=" << alpha << ")" << std::endl;
                }
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        sdk.SetInputMemory(file_buffer.data(), W, H, 3);
        sdk.ProcessNextFrame();
        auto t2 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (i % 50 == 0) {
            std::cout << "Frame " << i << " Total Process Time: " << ms << " ms (" << (1000.0/ms) << " FPS)" << std::endl;
        }

        // ==========================================
        std::cout << "Frame " << i << " Total Process Time: " << ms << " ms (" << (1000.0/ms) << " FPS)" << std::endl;
        printf("[Debug] Step 0: Loop Start\n");

        // 1. Get Objects (Deep Copy)
        std::vector<MotionObject> objects = sdk.GetMotionObjects();
        std::vector<uint8_t> changed_blocks = sdk.GetChangedBlocks(); // Mask
        std::vector<MotionVector> vectors = sdk.GetMotionVectors();
        printf("[Debug] Step 1: SDK Getters OK. NumObjs=%zu\n", objects.size());

        // (Stripped BG Mask Gen)
        std::vector<uint8_t> bg_mask_img; // Keep variable empty to satisfy existing checks


        // 2. Draw Bed (Blue)
        if (bed_points_sorted.size() == 4) {
            for(int k=0; k<4; k++) {
            for(int k=0; k<4; k++) {
                drawLine(current_frame_rgb, W, H, 
                         bed_points_sorted[k].first, bed_points_sorted[k].second, 
                         bed_points_sorted[(k+1)%4].first, bed_points_sorted[(k+1)%4].second, 
                         0, 0, 255);
            }
            }
        }
        
        // Draw Red Box on Fall (20 frames, thickness 3)
        if (fall_red_box_countdown > 0) {
             drawRectRGB(current_frame_rgb, W, H, 0, 0, W, H, 255, 0, 0, 3);
             fall_red_box_countdown--;
        }
        
        printf("[Debug] Step 4: Draw Objects\n");
        
        // 3. Draw Changed Blocks (Green)
        int cols = motionCfg.grid_cols;
        int rows = motionCfg.grid_rows;
        if ((int)changed_blocks.size() == cols * rows) {
             float bw = (float)W / cols;
             float bh = (float)H / rows;
             for(int r=0; r<rows; r++) {
                 for(int c=0; c<cols; c++) {
                     int idx = r * cols + c;
                     if(changed_blocks[idx]) {
                         // Draw Green Rect
                         int x = (int)(c * bw);
                         int y = (int)(r * bh);
                         drawRectRGB(current_frame_rgb, W, H, x, y, (int)bw, (int)bh, 0, 255, 0, 1);
                     }
                 }
             }
        }

        // 4. Draw Objects (Blocks + Arrows)
        // Find object with max strength to highlight as "Fall Object" if fall detected
        int max_strength_id = -1;
        float max_str = -1.0f;
        if (is_fall_in_current_frame) {
            for(const auto& obj : objects) {
                if(obj.strength > max_str) {
                    max_str = obj.strength;
                    max_strength_id = obj.id;
                }
            }
        }

        // REMOVED PER USER REQUEST (Crash Debugging)
        // [Hull Filter Block]
        /*
        if (!bg_mask_img.empty() && !objects.empty()) {
             // ...
        }
        */

        for(const auto& obj : objects) 
        {
            printf("[Debug] Inside Obj Loop. ID=%d Blocks=%zu MVs=%zu. &obj=%p\n", obj.id, obj.blocks.size(), obj.block_motion_vectors.size(), &obj);
            bool is_fall_obj = (is_fall_in_current_frame && obj.id == max_strength_id);
            
            // Random Color for Object (based on ID)
            srand(obj.id * 123 + 456); 
            uint8_t objR = rand() % 200 + 55; // Avoid too dark
            uint8_t objG = rand() % 200 + 55;
            uint8_t objB = rand() % 200 + 55;

            float bw = (float)W / cols;
            float bh = (float)H / rows;

            // A. Draw Object Blocks 
            // User requested: "block area of each object... different for each object"
            // Reverting to Filled Blocks for Debug visualization of "What is in the object?"
            for(size_t i=0; i<obj.blocks.size(); ++i) 
            {
                int blkIdx = obj.blocks[i];
                int r = blkIdx / cols;
                int c = blkIdx % cols;
                int x = (int)(c * bw);
                int y = (int)(r * bh);
                
                // Draw Filled Rect with Object Color
                // for(int fy=y; fy<y+(int)bh; ++fy) {
                //     for(int fx=x; fx<x+(int)bw; ++fx) {
                //         if(fx>=0 && fx<W && fy>=0 && fy<H) {
                //             int idx = (fy * W + fx) * 3;
                //             current_frame_rgb[idx] = objR;
                //             current_frame_rgb[idx+1] = objG;
                //             current_frame_rgb[idx+2] = objB; 
                //         }
                //     }
                // }
                //Draw Filled Rect with Object Color
                // for(int fy=y; fy<y+(int)bh; ++fy) 
                // {
                //     for(int fx=x; fx<x+(int)bw; ++fx) 
                //     {
                //         if(fx>=0 && fx<W && fy>=0 && fy<H) 
                //         {
                //             int idx = (fy * W + fx) * 3;
                //             current_frame_rgb_onlyBlock[idx] = 0;
                //             current_frame_rgb_onlyBlock[idx+1] = 255;
                //             current_frame_rgb_onlyBlock[idx+2] = 0; 
                //         }
                //     }
                // }
                
                // Draw Motion Arrow
                if (i < obj.block_motion_vectors.size()) {
                    MotionVector mv = obj.block_motion_vectors[i];
                    printf("[Debug] blk=%zu MV dx=%d dy=%d\n", i, mv.dx, mv.dy);
                }
            }
            printf("[Debug] End Block Loop for Object %d\n", obj.id);
            printf("[Debug] End Block Loop\n");
            
            // -------------------------------------------------------------
            // NEW: Filter bg_mask_img using Convex Hull (Match SDK Logic)
            // -------------------------------------------------------------
            // (Hull Filter Moved Before Loop)
            
            // -------------------------------------------------------------
            // NEW: Axis Calculation (PCA) using Background Reference
            // -------------------------------------------------------------
            if (bg_saved_flag && !bg_reference.empty()) {
                printf("[Debug] Entering PCA\n");
                double bg_diff_thr = (double)fallCfg.bg_diff_threshold; 
                // Fallback if config not loaded/zero? default 30
                if (bg_diff_thr < 1.0) bg_diff_thr = 30.0;

                int blockW = W / cols;
                int blockH = H / rows;

                // Structure Tensor Moments (Gradient)
                double sxx = 0, syy = 0, sxy = 0;
                // Shape PCA Moments
                double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m02 = 0, m11 = 0;

                for (int blkIdx : obj.blocks) {
                    int r = blkIdx / cols;
                    int c = blkIdx % cols;
                    int x0 = c * blockW;
                     int y0 = r * blockH;
                     int x1 = std::min(x0 + blockW, W);
                     int y1 = std::min(y0 + blockH, H);

                     for (int y = y0; y < y1; ++y) {
                         for (int x = x0; x < x1; ++x) {
                             int idx = (y * W + x) * 3;

                             // USE FILTERED MASK (Fix Misalignment)
                             if (bg_mask_img.empty()) continue; 
                             
                             // bg_mask_img is now CLEANED (Eroded/Dilated)
                             // Use its value as weight directly (it contains weighted diff)
                             uint8_t val = bg_mask_img[idx]; // R channel
                             
                             if (val > 0) {
                                 // Weight = Severity of difference (from mask)
                                 float weight = (float)val; 
                                 
                                 m00 += weight;
                                 m10 += x * weight;
                                 m01 += y * weight;
                                 m20 += x*x * weight;
                                 m02 += y*y * weight;
                                 m11 += x*y * weight;

                                 // --- Structure Tensor (Gradient) ---
                                 // Compute Gradient Gx, Gy (Central Difference) ON MASK (Cleaner)
                                 // Check bounds
                                 if (x > 0 && x < W-1 && y > 0 && y < H-1) {
                                     // Use Mask Gradient (Shape Tensor) or Image Gradient (Texture Tensor)?
                                     // User wanted "Gradient within block" -> Texture.
                                     // So we stick to Image Gradient, BUT assume pixels are valid only if in Mask.
                                     // Okay, let's keep Image Gradient for Texture Axis.
                                     int idxL = (y * W + (x-1)) * 3;
                                     int idxR = (y * W + (x+1)) * 3;
                                     int idxU = ((y-1) * W + x) * 3;
                                     int idxD = ((y+1) * W + x) * 3;
                                     
                                     // Convert to grayscale roughly for gradient
                                     auto getGray = [&](int i) { return (current_frame_rgb[i] + current_frame_rgb[i+1]*2 + current_frame_rgb[i+2])/4; };
                                     
                                     int gx = getGray(idxR) - getGray(idxL);
                                     int gy = getGray(idxD) - getGray(idxU);
                                     
                                     sxx += gx * gx;
                                     syy += gy * gy;
                                     sxy += gx * gy;
                                 }
                             }
                         }
                     }
                 }

                 if (m00 > 10.0) { // Require some pixels
                     double cx = m10 / m00;
                     double cy = m01 / m00;
                     double mu20 = m20/m00 - cx*cx;
                     double mu02 = m02/m00 - cy*cy;
                     double mu11 = m11/m00 - cx*cy;

                     // 1. Shape PCA Angle
                     double delta = std::sqrt(4*mu11*mu11 + (mu20-mu02)*(mu20-mu02));
                     double l1 = (mu20 + mu02 + delta) / 2;
                     double l2 = (mu20 + mu02 - delta) / 2;
                     double theta_shape = 0.5 * std::atan2(2*mu11, mu20-mu02);

                     /*
                     // 2. Texture Gradient Angle
                     // Structure Tensor Eigen
                     double s_delta = std::sqrt(4*sxy*sxy + (sxx-syy)*(sxx-syy));
                     // ...
                     drawArrow(bg_mask_img, W, H, (int)cx, (int)cy, x_min, y_min, 0, 0, 255, 2); 
                     */
                     }
                // } // Removed inner obj loop brace
            }

            // A. Draw Hollow Bounding Box for Object (Not Filled Blocks)
            // "Object ROI based on constituent blocks"
            int minC = 9999, maxC = -1;
            int minR = 9999, maxR = -1;
            
            for(int blkIdx : obj.blocks) {
                int r = blkIdx / cols;
                int c = blkIdx % cols;
                if(c < minC) minC = c;
                if(c > maxC) maxC = c;
                if(r < minR) minR = r;
                if(r > maxR) maxR = r;
            }
            
            if (minC <= maxC && minR <= maxR) {
                 int x1 = (int)(minC * bw);
                 int y1 = (int)(minR * bh);
                 int x2 = (int)((maxC + 1) * bw);
                 int y2 = (int)((maxR + 1) * bh);
                 int rw = x2 - x1;
                 int rh = y2 - y1;
                 
                 // Draw Hollow Rect (Thickness 2)
                 // Use slightly different color? Or same? Same.
                 drawRectRGB(current_frame_rgb, W, H, x1, y1, rw, rh, 255, 255, 255, 2); // White border for contrast
                 
                 // SAVE SQUARE CROP (Object White Outline Part)
                 // 1. Determine Square Box
                 int side = std::max(rw, rh);
                 int cx = x1 + rw / 2;
                 int cy = y1 + rh / 2;
                 
                 int sq_x1 = cx - side / 2;
                 int sq_y1 = cy - side / 2;
                 
                 // 2. Prepare Buffer for Crop (Initialize with Black)
                 std::vector<uint8_t> crop_buf(side * side * 3, 0); 
                 
                 // 3. Copy Pixels
                 // Target: (0,0) to (side, side)
                 // Source: (sq_x1, sq_y1) to ...
                 
                 for(int r = 0; r < side; ++r) {
                     for(int c = 0; c < side; ++c) {
                         int src_y = sq_y1 + r;
                         int src_x = sq_x1 + c;
                         
                         if(src_x >= 0 && src_x < W && src_y >= 0 && src_y < H) {
                             int src_idx = (src_y * W + src_x) * 3;
                             int dst_idx = (r * side + c) * 3;
                             crop_buf[dst_idx] = clean_frame_rgb[src_idx]; // Use clean frame
                             crop_buf[dst_idx+1] = clean_frame_rgb[src_idx+1];
                             crop_buf[dst_idx+2] = clean_frame_rgb[src_idx+2];
                         }
                     }
                 }
                 
                 // 4. Save BMP
                 char crop_filename[256];
                 snprintf(crop_filename, sizeof(crop_filename), "v2_object_crops/frame_%05d_obj_%d.bmp", i, obj.id);
                 stbi_write_bmp(crop_filename, side, side, 3, crop_buf.data());
            }

            // C. Draw Momentum Arrow
            // "Momentum" -> usually Strength * Vector or just Vector.
            // visual: Arrow length = Vector * Scale
            // Fix: Arrow length was avgDx * 4. If avgDx is small (e.g. 0.5), it becomes length 2.
            // Try scaling up more or enforcing min length.
            if(std::abs(obj.avgDx) > 0.01 || std::abs(obj.avgDy) > 0.01) {
                // Determine Center in PIXELS (Scale from Block Coords)
                // obj.centerX/Y are in Block Index Units (e.g. 5.5) -> Valid range 0 to cols
                // To get pixel center: (Index + 0.5) * bw
                int pixelCenterX = (int)((obj.centerX + 0.5f) * bw);
                int pixelCenterY = (int)((obj.centerY + 0.5f) * bh);
            
                // Ensure visible length
                float vecX = -1*obj.avgDx;
                float vecY = -1*obj.avgDy;
                // Minimum visual magnitude
                float mag = std::sqrt(vecX*vecX + vecY*vecY);
                float scale = 4.0f;
                // Make arrow roughly proportional but visible
                // Vector 1.0 (1 block/frame) -> 64 pixels? block_size*4=64.
                // dx is in pixels (block search range). Wait, block matching output dx/dy is in PIXELS.
                // MotionVector dx/dy is pixels.
                // So if avgDx is 5 pixels, visual length 20 pixels is reasonable.
                
                if (mag * scale < 15.0f && mag > 0.001f) {
                     scale = 15.0f / mag; // Force at least 15px length
                }
                
                int ex = pixelCenterX + (int)(vecX * scale); 
                int ey = pixelCenterY + (int)(vecY * scale);
                
                // Arrow Color: Red if fall, Yellow otherwise
                // "Falling object ... different color (Red)"
                uint8_t ar = is_fall_obj ? 255 : 255;
                uint8_t ag = is_fall_obj ? 0 : 255;
                uint8_t ab = 0;
                
                // Draw thicker arrow
                drawArrow(current_frame_rgb, W, H, pixelCenterX, pixelCenterY, ex, ey, ar, ag, ab, 3); 
                //printf("[Demo] Frame %d, obj.id : %d, pixelCenterX %d, pixelCenterY %d\n", i, obj.id, pixelCenterX, pixelCenterY);
                
                // Draw Stats for Falling Object
                if (is_fall_obj) {
                    char msg[64];
                    // Display Acc and Str
                    snprintf(msg, sizeof(msg), "A:%.1f S:%.1f", obj.acceleration, obj.strength);
                    drawString(current_frame_rgb, W, H, pixelCenterX, pixelCenterY - 20, msg, 255, 0, 0, 2);
                    std::cout << "[FALL STATS] Frame " << i << " ObjID " << obj.id << " Acc: " << obj.acceleration << " Str: " << obj.strength << std::endl;
                } 
            }
        }
        
        // 5. Draw All Vectors (Optional, user asked for "each object vector")
        // "各個物件...還有各物件的向量" -> Handled above.
        
        // 6. Fall Status
        if (is_fall_in_current_frame) {
            std::string msg = "FALL DETECTED ";
            if (is_strong_fall) msg += "(STRONG)";
            else msg += "(NORMAL)";
            drawString(current_frame_rgb, W, H, 10, 10, msg, 255, 0, 0, 3);
        } else {
             drawString(current_frame_rgb, W, H, 10, 10, "NORMAL", 0, 255, 0, 3);
        }
        
        // SAVE BG MASK (Post-Loop)
        if (!bg_mask_img.empty()) {
             char mask_filename[256];
             // Use save path "0125_bgtest" (Hardcoded in loop usually, I need check where save happens)
             // I'll assume current directory or same pattern
             // Line 1058 saves to `save_dir`. I should use `save_dir`.
             // I need to finding `save_dir` variable name. It was `save_folder` in Step 409?
             // Line 1058: `snprintf(filename, ... "%s/frame_%05d.jpg", save_dir.c_str(), i);`
             // `save_dir` is defined at start of main. 
             // Is it visible here? Yes if single function.
             // Wait, I am inside `while`.
             // But I need to verify `save_dir` variable name.
             // I will use `bg_mask_frame_%05d.jpg` in current dir if fail, or `save_dir`.
             // I'll check `save_dir` in view_file 409?
             // Step 409: `std::string save_dir = "0125_bgtest";`.
             // So I can use `save_dir`.
             
             snprintf(mask_filename, sizeof(mask_filename), "%s/bg_mask_frame_%05d.jpg", save_dir.c_str(), i);
             stbi_write_jpg(mask_filename, W, H, 3, bg_mask_img.data(), 90);
        }

        // 7. Bed Exit Status (Yellow Border)
        if (is_bed_exit_in_current_frame) {
            drawRectRGB(current_frame_rgb, W, H, 0, 0, W, H, 255, 255, 0, 2);
            // Optionally print text
             drawString(current_frame_rgb, W, H, W/2 - 50, 10, "BED EXIT", 255, 255, 0, 2);
        }

        // (Stripped Delayed Feedback)

        // Save Current Frame IMMEDIATELY
        char out_filename[256];
        snprintf(out_filename, sizeof(out_filename), "%s/frame_%05d.jpg", save_dir.c_str(), i);
        stbi_write_jpg(out_filename, W, H, 3, current_frame_rgb.data(), 90);
        
    } // End Main Loop

    // (No flush needed for images, just clear buffer)
    frame_buffer.clear();

    std::cout << "Done. Saved to " << save_dir << "/" << std::endl;
    return 0;
}
