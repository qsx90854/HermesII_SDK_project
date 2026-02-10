
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
#include <sstream>
#include <set>

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

bool isPointInConvexQuad(const std::vector<std::pair<float, float>>& poly, float px, float py) {
    if (poly.size() < 3) return false;
    bool positive = false;
    bool negative = false;
    for (size_t i = 0; i < poly.size(); ++i) {
        float x1 = poly[i].first;
        float y1 = poly[i].second;
        float x2 = poly[(i + 1) % poly.size()].first;
        float y2 = poly[(i + 1) % poly.size()].second;
        float cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
        if (cross_product > 0) positive = true;
        if (cross_product < 0) negative = true;
    }
    return !(positive && negative);
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
bool custom_fall_signal = false;
std::string current_frame_reasons = "";
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

// Helper to load Ground Truth Intervals
std::vector<std::pair<int, int>> loadGroundTruth(const std::string& filename) {
    std::vector<std::pair<int, int>> intervals;
    std::ifstream file(filename);
    if (!file.is_open()) return intervals; // Return empty if not found (or optionally warn)

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Remove trailing comma if present
        if (line.back() == ',') line.pop_back(); 
        
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        int start, end;
        if (ss >> start >> end) {
            intervals.push_back({start, end});
        }
    }
    file.close();
    return intervals;
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

// Custom Fall Logic History
struct ObjStatsHistory {
    // Short-term history (30 frames)
    std::deque<int> area_hist;
    std::deque<int> red_hist;
    std::deque<float> lk_strength_hist; 
    std::deque<float> sad_strength_hist; // for sliding peak
    std::deque<float> sad_accel_hist;    // for sliding peak
    
    // NEW: Long-term history (120 frames) for peak-valley detection
    std::deque<int> red_hist_long;
    std::deque<int> area_hist_long;
    
    // NEW: Peak-valley tracking
    int red_peak_value = 0;           // Recorded peak value
    int red_peak_frame_offset = -1;   // Frames since peak
    bool in_decline_phase = false;    // Whether in decline phase
    int decline_start_frame = -1;     // Frame offset when decline started
    int low_value_counter = 0;        // Frames maintaining low value
    
    // Original fields
    int inconsistent_count = 0;
    bool red_down_trend_triggered = false;
    int frames_since_red_trigger = 0;
    float current_lk_strength = 0.0f;
    float current_lk_accel = 0.0f;
    int last_update_frame = -1; // Prevent duplicate updates
};
std::map<int, ObjStatsHistory> obj_histories;

int main(int argc, char** argv) {
    std::cout << "Starting Fall Callback Demo v2 SAVE (30FPS Sim)..." << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;
    // 1. Load Configs
    ConfigLoader cfg;
    ConfigLoader appCfg;
    
    std::string app_config_path = "app_config.ini";
    if (argc > 1) app_config_path = argv[1];
    std::cout << "Loading app config from: " << app_config_path << std::endl;
    appCfg.load(app_config_path);
    
    // Load App Params First (Dimensions, Files)
    int W = appCfg.getInt("Demo.Demo_Width", 800);
    int H = appCfg.getInt("Demo.Demo_Height", 450);
    int orgW = appCfg.getInt("Demo.Demo_Original_Width", 1920);
    int orgH = appCfg.getInt("Demo.Demo_Original_Height", 1080);
    
    std::string imgFormat = appCfg.getString("Demo.Demo_Image_Path_Format", "TestData/images_150455_800x450_rgb_new/frame_%05d.raw");
    std::string bedFile = appCfg.getString("Demo.Demo_Bed_File", ""); 
    std::string gtFile = appCfg.getString("Demo.Demo_GT_File", ""); // Load GT File Path
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
    float bed_pixel_ratio_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Bed_Pixel_Ratio_Threshold", 0.3f);
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
    
    // Optical Flow Params
    fallCfg.opt_flow_frame_distance = cfg.getInt("OpticalFlow.CompareFrameDistance", 3);
    fallCfg.perspective_point_x = cfg.getInt("OpticalFlow.PerspectivePointX", 416);
    fallCfg.perspective_point_y = cfg.getInt("OpticalFlow.PerspectivePointY", 474);
    float opt_flow_vel_threshold = cfg.getFloat("OpticalFlow.VelocityThreshold", 1.0f);

    sdk.SetConfig(&fallCfg);

    // 4. Image Related Config
    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    std::string savePath = save_dir; // Use the directory created by AppConfig
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

    // Dynamic adjustment if start_frame is later than init window
    if (bg_init_start_frame < start_frame) {
         int duration = bg_init_end_frame - bg_init_start_frame;
         bg_init_start_frame = start_frame + 2; 
         bg_init_end_frame = bg_init_start_frame + duration;
         std::cout << "[Demo] Adjusted BG Init Window to Frames " << bg_init_start_frame << "-" << bg_init_end_frame << std::endl;
    }
    std::vector<uint32_t> bg_accumulator;
    std::vector<uint8_t> bg_reference; // Persistent BG for PCA
    std::map<int, double> obj_smooth_thetas; // Prev angles for smoothing
    int bg_frames_count = 0;
    bool bg_saved_flag = false;

    // ------------------------------------------------------------------
    // NEW: Data Logging (User Request)
    // ------------------------------------------------------------------
    // Parse Dataset Name from Pattern
    // Pattern: "Path/To/Folder/frame_%05d.raw" -> "Folder"
    std::string dataset_name = "unknown";
    size_t last_slash = pattern.find_last_of('/');
    if (last_slash != std::string::npos) {
        size_t second_last_slash = pattern.find_last_of('/', last_slash - 1);
        if (second_last_slash != std::string::npos) {
            dataset_name = pattern.substr(second_last_slash + 1, last_slash - second_last_slash - 1);
        } else {
             dataset_name = pattern.substr(0, last_slash);
        }
    }
    
    std::string log_dir_path = save_dir + "/" + dataset_name + "_dir_var.txt";
    std::string log_mag_path = save_dir + "/" + dataset_name + "_mag_var.txt";
    std::string log_fg_path = save_dir + "/" + dataset_name + "_fg_count.txt";
    // NEW METRICS
    std::string log_accel_path = save_dir + "/" + dataset_name + "_accel.txt";
    std::string log_speed_path = save_dir + "/" + dataset_name + "_speed.txt";
    std::string log_bri_path   = save_dir + "/" + dataset_name + "_bri.txt";
    
    std::ofstream f_log_dir(log_dir_path);
    std::ofstream f_log_mag(log_mag_path);
    std::ofstream f_log_fg(log_fg_path);
    std::ofstream f_log_accel(log_accel_path);
    std::ofstream f_log_speed(log_speed_path);
    std::ofstream f_log_bri(log_bri_path);
    
    if(!f_log_dir.is_open()) std::cerr << "Failed to open " << log_dir_path << std::endl;
    // ------------------------------------------------------------------

    // NEW: Fall Interval Logging
    std::string f_interval_name = save_dir + "/detected_fall_intervals.txt";
    std::ofstream f_interval(f_interval_name);
    
    bool is_currently_falling = false;
    int fall_start_frame = -1;
    int total_fall_events = 0; // NEW: Counter

    
    
    // Stored Intervals for Verification
    struct FallInterval {
        int start;
        int end;
        std::string reasons;
    };
    std::vector<FallInterval> detected_intervals_vec;

    // Main processing loop
    int total_frames = start_frame + num_images; // Define total_frames based on existing variables
    for (int i = start_frame; i < total_frames; i += frame_step) {
        
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
            // 6. Save Image
            printf("[Debug] Step 7: Save Image (Frame %d)\n", i);
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
        is_fall_in_current_frame = false; // Reset for custom logic
        sdk.ProcessNextFrame();

        auto t2 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (i % 50 == 0) {
            std::cout << "Frame " << i << " Total Process Time: " << ms << " ms (" << (1000.0/ms) << " FPS)" << std::endl;
        }
        

        // 1. Get Objects (Deep Copy)
        std::vector<MotionObject> objects = sdk.GetMotionObjects();
        std::vector<uint8_t> changed_blocks = sdk.GetChangedBlocks(); // Mask
        std::vector<MotionVector> vectors = sdk.GetMotionVectors();
        

        // --- PREPARE BG MASK IMG (User Request) ---
        std::vector<uint8_t> bg_mask_img;
        if (bg_saved_flag && !bg_reference.empty()) {
             printf("[Debug] Step 2: Entering BG Mask Gen (Frame %d)\n", i);
             bg_mask_img.resize(W*H*3, 0); // Black
             double bg_diff_thr = (double)fallCfg.bg_diff_threshold; 
             if (bg_diff_thr < 1.0) bg_diff_thr = 30.0;
             
             // Compute Diff for Object Bounding Boxes (to capture holes)
             int grid_cols = cfg.getInt("Motion.Grid_Cols", 80);
             int grid_rows = cfg.getInt("Motion.Grid_Rows", 45);
             int bw = W / grid_cols;
             int bh = H / grid_rows;

             printf("[Debug] BBox Scan Init. W=%d H=%d bw=%d bh=%d. bg_ref_sz=%zu\n", W, H, bw, bh, bg_reference.size());

             // Compute Global Diff (Moved OUTSIDE object loop)
             int startX = 0;
             int startY = 0;
             int endX = W;
             int endY = H;
             
             // If Bed Region is defined, maybe restrict? 
             // User wants "bg_mask frame", likely full frame or at least same as before.
             // Previous code used object bbox. Let's do FULL FRAME for now to be safe.
             
             for(int y=startY; y<endY; ++y) {
                 for(int x=startX; x<endX; ++x) {
                     int idx = (y * W + x) * 3;
                     
                     int diff = std::abs((int)current_frame_rgb[idx] - (int)bg_reference[idx]) +
                                std::abs((int)current_frame_rgb[idx+1] - (int)bg_reference[idx+1]) +
                                std::abs((int)current_frame_rgb[idx+2] - (int)bg_reference[idx+2]);
                     
                     float diffVal = (float)diff / 3.0f;
                     if (diffVal > bg_diff_thr) {
                         uint8_t val = (uint8_t)std::min(255.0f, diffVal);
                         bg_mask_img[idx] = val;   
                         bg_mask_img[idx+1] = val;
                         bg_mask_img[idx+2] = val;
                     }
                 }
             }

             for (const auto& obj : objects) {
                 // 1. Find Bounding Box of Blocks (Keep logic for Hull/Stats if needed, but not for mask filling)
                 int min_c = grid_cols, max_c = 0;
                 int min_r = grid_rows, max_r = 0;
                 if (obj.blocks.empty()) continue;

                 for (int blkIdx : obj.blocks) {
                     int r = blkIdx / grid_cols;
                     int c = blkIdx % grid_cols;
                     if(c < min_c) min_c = c; if(c > max_c) max_c = c;
                     if(r < min_r) min_r = r; if(r > max_r) max_r = r;
                 }
                 
                 // 2. Scan Bounding Box - REMOVED MASK FILLING FROM HERE
             }
 
             // --- MORPHOLOGY (Erosion -> Dilation) [User Request: Fix Noise] ---
             // DISABLED to match SDK internal logic (which uses raw diff count inside hull)
             /*
             // Simple 3x3 kernel. Pass 1: Erode
             std::vector<uint8_t> temp_mask = bg_mask_img;
             for(int y=1; y<H-1; ++y) {
                 for(int x=1; x<W-1; ++x) {
                     int idx = (y*W+x)*3;
                     if(bg_mask_img[idx] > 0) {
                         // Check neighbors
                         bool keep = true;
                         if (bg_mask_img[((y-1)*W+x)*3] == 0) keep = false;
                         else if (bg_mask_img[((y+1)*W+x)*3] == 0) keep = false;
                         else if (bg_mask_img[(y*W+(x-1))*3] == 0) keep = false;
                         else if (bg_mask_img[(y*W+(x+1))*3] == 0) keep = false;
                         
                         if(!keep) {
                             temp_mask[idx] = 0; temp_mask[idx+1] = 0; temp_mask[idx+2] = 0;
                         }
                     }
                 }
             }
             // Pass 2: Dilate
             bg_mask_img = temp_mask; // Apply Erode result
             std::vector<uint8_t> result_mask = temp_mask;
             for(int y=1; y<H-1; ++y) {
                 for(int x=1; x<W-1; ++x) {
                     int idx = (y*W+x)*3;
                     if(temp_mask[idx] > 0) {
                         // Dilate to neighbors
                         auto setW = [&](int i, uint8_t v) { 
                             if (v > result_mask[i]) { result_mask[i]=v; result_mask[i+1]=v; result_mask[i+2]=v; } 
                         };
                         uint8_t v = temp_mask[idx];
                         setW(((y-1)*W+x)*3, v);
                         setW(((y+1)*W+x)*3, v);
                         setW((y*W+(x-1))*3, v);
                         setW((y*W+(x+1))*3, v);
                     }
                 }
             }
             bg_mask_img = result_mask; // Final Morph Result
             */
         }
         //printf("[Debug] Step 3: Morphology Disabled\n");
        // ------------------------------------------

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
             //drawRectRGB(current_frame_rgb, W, H, 0, 0, W, H, 255, 0, 0, 3);
             fall_red_box_countdown--;
        }

        // ------------------------------------------
        // NEW: Full Frame Group Visualization
        // ------------------------------------------
        std::vector<ObjectFeatures> ff_objs = sdk.GetFullFrameObjects();
        std::vector<uint8_t> ff_viz_img(W * H * 3, 0); // Start with black
        
        // Sort to get top 2
        std::sort(ff_objs.begin(), ff_objs.end(), [](const ObjectFeatures& a, const ObjectFeatures& b) {
            return a.area > b.area;
        });

        bool custom_fall_signal = false;
        int perspective_x = fallCfg.perspective_point_x;
        int perspective_y = fallCfg.perspective_point_y;
        
        int largest_obj_pixel_count = 0;
        int largest_obj_red_count = 0;

        // Process top 3 objects for custom logic
        for (int i_obj = 0; i_obj < std::min(3, (int)ff_objs.size()); ++i_obj) {
            const auto& f_obj = ff_objs[i_obj];
            int red_count = 0;
            
            float lk_sum_speed = 0.0f;
            int lk_pix_count = 0;
            
            for (size_t i_pix = 0; i_pix < f_obj.pixels.size(); ++i_pix) {
                int pix_idx = f_obj.pixels[i_pix];
                if (pix_idx >= 0 && pix_idx < W * H) {
                    int py = pix_idx / W;
                    int px = pix_idx % W;

                    uint8_t r, g, b;
                    if (i_obj == 0) {
                        r = 100; g = 100; b = 255; // Light Blue for largest group base
                    } else {
                        r = 120; g = 120; b = 120; // Gray for others
                    }

                    if (i_pix < f_obj.pixel_dx.size()) {
                        float dx = f_obj.pixel_dx[i_pix];
                        float dy = f_obj.pixel_dy[i_pix];
                        float angle = f_obj.pixel_dir[i_pix]; // Radians
                        float speed = std::sqrt(dx*dx + dy*dy);
                        
                        // NEW: Filter small motions for average
                        if (speed > 1.0f) {
                            lk_sum_speed += speed;
                            lk_pix_count++;
                        }

                        // Downward is PI/2 (90 deg). PI/2 +/- 40 deg => [50 deg, 130 deg]
                        // 50 deg = 0.8726 rad, 130 deg = 2.2689 rad
                        if (speed > opt_flow_vel_threshold && angle >= 0.8726f && angle <= 2.2689f) {
                            if (i_obj == 0) {
                                r = 255; g = 255; b = 0; // Yellow for largest group "red" points
                            } else {
                                r = 255; g = 0; b = 0; // Red for others
                            }
                            red_count++;
                        }
                    }

                    ff_viz_img[pix_idx * 3] = r;
                    ff_viz_img[pix_idx * 3 + 1] = g;
                    ff_viz_img[pix_idx * 3 + 2] = b;
                }
            }
            if (i_obj == 0) {
                largest_obj_pixel_count = f_obj.area;
                largest_obj_red_count = red_count;
            }

            // --- Custom Fall Logic Tracking ---
            // Match Foreground blob to persistent tracked SAD object to use persistent ID for trends
            int persistent_id = f_obj.id; 
            float min_dist = 1e9;
            float block_w = (float)W / motionCfg.grid_cols;
            float block_h = (float)H / motionCfg.grid_rows;

            for (const auto& s_obj : objects) {
                float sx = (s_obj.centerX + 0.5f) * block_w;
                float sy = (s_obj.centerY + 0.5f) * block_h;
                float dx = sx - f_obj.cx;
                float dy = sy - f_obj.cy;
                float dist = std::sqrt(dx*dx + dy*dy);
                if (dist < min_dist) {
                    min_dist = dist;
                    persistent_id = s_obj.id;
                }
            }

            // Strict match: 80 pixels (Confirmed scale is correct)
            if (min_dist > 80.0f) persistent_id = (f_obj.id + 5000); 

            int obj_id = persistent_id;
            auto& hist = obj_histories[obj_id];
            
            if (hist.last_update_frame != (int)i) {
                // Short-term history (30 frames)
                hist.area_hist.push_back(f_obj.area);
                hist.red_hist.push_back(red_count);
                
                // NEW: Long-term history (120 frames)
                hist.red_hist_long.push_back(red_count);
                hist.area_hist_long.push_back(f_obj.area);
                
                float current_lk_s = (lk_pix_count > 0) ? (lk_sum_speed / lk_pix_count) : 0.0f;
                hist.current_lk_accel = 0.0f;
                if (!hist.lk_strength_hist.empty()) {
                    hist.current_lk_accel = current_lk_s - hist.lk_strength_hist.back();
                }
                hist.current_lk_strength = current_lk_s;
                hist.lk_strength_hist.push_back(current_lk_s);

                // Maintain short-term buffer size (30 frames)
                if (hist.area_hist.size() > 30) hist.area_hist.pop_front();
                if (hist.red_hist.size() > 30) hist.red_hist.pop_front();
                if (hist.lk_strength_hist.size() > 30) hist.lk_strength_hist.pop_front();
                
                // NEW: Maintain long-term buffer size (120 frames)
                if (hist.red_hist_long.size() > 120) hist.red_hist_long.pop_front();
                if (hist.area_hist_long.size() > 120) hist.area_hist_long.pop_front();
                
                hist.last_update_frame = i;
            } else {
                // If matched multiple times, pick largest
                if (f_obj.area > hist.area_hist.back()) {
                    hist.area_hist.back() = f_obj.area;
                    hist.red_hist.back() = red_count;
                }
            }


            // ========== NEW: Peak-Valley Detection Algorithm ==========
            bool fall_detected = false;
            int peak_red = 0;
            int peak_idx = -1;
            float decline_ratio = 0.0f;
            float recent_avg_red = 0.0f;
            bool area_declined = false;
            
            // Need at least 60 frames of history for peak detection
            if (hist.red_hist_long.size() >= 60) {
                
                // Step 1: Find peak in past 60-120 frames
                int history_size = hist.red_hist_long.size();
                
                // Search from at least 30 frames ago, up to 90 frames ago
                int search_start = std::max(30, history_size - 90);
                int search_end = history_size - 5; // Leave 5-frame buffer
                
                for (int idx = search_start; idx < search_end; ++idx) {
                    int val = hist.red_hist_long[idx];
                    
                    // Check if this is a local peak (higher than +/- 10 frames around it)
                    bool is_peak = true;
                    for (int offset = -10; offset <= 10; ++offset) {
                        if (offset == 0) continue;
                        int check_idx = idx + offset;
                        if (check_idx >= 0 && check_idx < history_size) {
                            if (hist.red_hist_long[check_idx] > val) {
                                is_peak = false;
                                break;
                            }
                        }
                    }
                    
                    if (is_peak && val > peak_red) {
                        peak_red = val;
                        peak_idx = idx;
                    }
                }
                
                // Step 2: Check for significant decline from peak  
                // FILTER: 提高peak閾值,過濾低峰值誤判(FP1: peak=372)
                if (peak_red > 400) { // Raised from 300 to 400
                    int current_red = hist.red_hist_long.back();
                    decline_ratio = (float)(peak_red - current_red) / peak_red;
                    
                    // Decline must be > 60%
                    if (decline_ratio > 0.6f) {
                        
                        // Step 3: Verify sustained low value
                        // Check average of recent 30 frames
                        int recent_sum = 0;
                        int recent_count = std::min(30, (int)hist.red_hist_long.size());
                        for (int j = history_size - recent_count; j < history_size; ++j) {
                            recent_sum += hist.red_hist_long[j];
                        }
                        recent_avg_red = (float)recent_sum / recent_count;
                        
                        // NEW FILTER: RecentAvg下限 - 過濾物體消失的情況
                        // Data4 FP: RecentAvg=0-43 (物體離開視野)
                        // Real falls: RecentAvg=187-345 (人倒在地上仍可見)
                        if (recent_avg_red < 80.0f) {
                            // Recent avg太低,可能是物體已離開視野或太小
                            fall_detected = false;
                            break; // Exit the peak-valley check
                        }
                        
                        // Recent average must be < 30% of peak (strictened from 40%)
                        // Real falls: <6.3%, FP1: 32-40%
                        if (recent_avg_red < peak_red * 0.30f) {
                            
                            // Step 4: Check foreground area decline
                            if (hist.area_hist_long.size() >= 60 && peak_idx >= 0) {
                                int peak_area = hist.area_hist_long[peak_idx];
                                int current_area = hist.area_hist_long.back();
                                
                                // Area decline > 40%
                                if (peak_area > 500 && current_area < peak_area * 0.6f) {
                                    area_declined = true;
                                }
                            }
                            
                            // Mark as fall detected if area declined
                            fall_detected = area_declined;
                        }
                    }
                }
            }

            // Axial Inconsistency Check (unchanged)
            float dx_p = f_obj.cx - (float)perspective_x;
            float dy_p = f_obj.cy - (float)perspective_y;
            float p_angle = std::atan2(dy_p, dx_p);
            float diff = std::abs(f_obj.angle - p_angle);
            while (diff > M_PI/2.0f) diff = std::abs(diff - (float)M_PI);
            bool inconsistent = (diff > 0.785f); // > 45 deg

            // Complete the fall detection by combining with angle inconsistency
            if (fall_detected || (inconsistent && hist.red_hist_long.size() >= 60 && peak_red > 400)) {
                fall_detected = true;
            } else {
                fall_detected = false;
            }

            // Final Custom Signal Decision
            // Logic: Peak-Valley Decline detected AND (Area declined OR Angle inconsistent)
            if (fall_detected) {
                
                // Human-size guard: Real falls are usually > 300 pixels
                bool size_ok = (f_obj.area > 300 && f_obj.area < 30000);

                // --- Bed Region Filtering ---
                int in_bed_count = 0;
                if (!bed_points_sorted.empty()) {
                    std::vector<std::pair<float, float>> poly;
                    for(auto& p : bed_points_sorted) poly.push_back({(float)p.first, (float)p.second});
                    for(int pix_idx : f_obj.pixels) {
                        int py = pix_idx / W;
                        int px = pix_idx % W;
                        if (isPointInConvexQuad(poly, (float)px, (float)py)) {
                            in_bed_count++;
                        }
                    }
                }
                float bed_ratio = (f_obj.pixels.size() > 0) ? (float)in_bed_count / f_obj.pixels.size() : 0.0f;
                bool not_in_bed = (bed_ratio < bed_pixel_ratio_threshold);

                if (size_ok && not_in_bed) {
                    custom_fall_signal = true;
                    
                    // Detailed Trigger Logging: Peak-Valley Detection
                    current_frame_reasons += "PeakDecline;";
                    
                    std::cout << "[CUSTOM_FALL] Frame:" << i << " ID:" << obj_id 
                              << " Trigger:PeakDecline"
                              << " Peak:" << peak_red << " Current:" << hist.red_hist_long.back()
                              << " Decline:" << (int)(decline_ratio*100) << "%"
                              << " RecentAvg:" << (int)recent_avg_red
                              << " BedRatio:" << bed_ratio
                              << " Area:" << f_obj.area 
                              << " Inconsistent:" << inconsistent << std::endl;
                }
            }

            // Draw Major Axis for top 2
            float major = f_obj.major;
            float angle_pca = f_obj.angle;
            float cx = f_obj.cx;
            float cy = f_obj.cy;
            float cos_a = std::cos(angle_pca);
            float sin_a = std::sin(angle_pca);
            int maj_x1 = (int)(cx - (major/2.0f) * cos_a);
            int maj_y1 = (int)(cy - (major/2.0f) * sin_a);
            int maj_x2 = (int)(cx + (major/2.0f) * cos_a);
            int maj_y2 = (int)(cy + (major/2.0f) * sin_a);
            drawArrow(ff_viz_img, W, H, maj_x1, maj_y1, maj_x2, maj_y2, 255, 255, 0, 2);
        }

        // Override SDK fall signal
        static int custom_hold_frames = 0;
        if (custom_fall_signal) custom_hold_frames = 30; // Hold for 1 second at 30fps
        
        // Reset original flag and use custom one
        is_fall_in_current_frame = (custom_hold_frames > 0);
        if (custom_hold_frames > 0) custom_hold_frames--;

        // Interval Tracking Logic (Moved here)
        static std::set<std::string> current_interval_reasons_set;

        if (is_fall_in_current_frame) {
            if (!is_currently_falling) {
                is_currently_falling = true;
                fall_start_frame = i;
                current_interval_reasons_set.clear();
            }
            if (!current_frame_reasons.empty()) {
                // Parse "Area;Angle;" into set
                std::stringstream ss(current_frame_reasons);
                std::string segment;
                while(std::getline(ss, segment, ';')) {
                    if(!segment.empty()) current_interval_reasons_set.insert(segment);
                }
            }
        } else {
            if (is_currently_falling) {
                is_currently_falling = false;
                if (f_interval.is_open()) {
                     f_interval << fall_start_frame << "," << (i - 1) << "\n";
                     f_interval.flush(); // Ensure written
                     
                     std::string combined_reasons = "";
                     for(const auto& r : current_interval_reasons_set) combined_reasons += r + " ";
                     
                     detected_intervals_vec.push_back({fall_start_frame, (int)(i - 1), combined_reasons});
                     total_fall_events++;
                }
            }
        }
        
        char fg_groups_filename[256];
        snprintf(fg_groups_filename, sizeof(fg_groups_filename), "%s/fg_groups_frame_%05d.jpg", save_dir.c_str(), i);
        stbi_write_jpg(fg_groups_filename, W, H, 3, ff_viz_img.data(), 90);
        
        
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

        // -------------------------------------------------------------
        // NEW: Filter bg_mask_img using Convex Hull (Match SDK Logic)
        // MOVED HERE (Before Object Loop) to avoid N^2 complexity
        // -------------------------------------------------------------
        printf("[Debug] Step 5: Start Hull/Draw Loop (Frame %d)\n", i); fflush(stdout);
        if (true && !bg_mask_img.empty() && !objects.empty()) { // RE-ENABLED
            
            std::vector<uint8_t> final_mask(W*H*3, 0);
            
            struct Point { double x, y; }; // Changed to double for 0.5 logic
            // Use local vars if needed, verify cols
            if (cols == 0) { printf("[ERROR] cols is 0!\n"); cols=1; }
            float bw = (float)W / cols;
            float bh = (float)H / rows;

            for(const auto& obj : objects) {
                 
                // 1. Collect Block Points (All 4 Corners)
                std::vector<Point> blockPts;
                int min_c = cols, max_c = -1;
                int min_r = rows, max_r = -1;
                
                for (int blkIdx : obj.blocks) {
                    int r = blkIdx / cols;
                    int c = blkIdx % cols;
                    if(c < min_c) min_c = c; if(c > max_c) max_c = c;
                    if(r < min_r) min_r = r; if(r > max_r) max_r = r;
                    
                    // Push 4 Corners
                    blockPts.push_back({(double)c, (double)r});
                    blockPts.push_back({(double)c+1, (double)r});
                    blockPts.push_back({(double)c, (double)r+1});
                    blockPts.push_back({(double)c+1, (double)r+1});
                }
                 

                // 2. Compute Hull
                 if (blockPts.size() > 2) {
                     std::sort(blockPts.begin(), blockPts.end(), [](const Point& a, const Point& b){
                         return a.x < b.x || (a.x == b.x && a.y < b.y);
                     });
                     std::vector<Point> hull;
                     for(const auto& p : blockPts) {
                         while(hull.size() >= 2) {
                             const Point& o = hull[hull.size()-2];
                             const Point& a = hull.back();
                             double cp = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
                             if (cp <= 0) hull.pop_back(); else break;
                         }
                         hull.push_back(p);
                     }
                     int lower_len = hull.size();
                     for(int i=(int)blockPts.size()-2; i>=0; --i) {
                         const Point& p = blockPts[i];
                         while(hull.size() > lower_len) {
                             const Point& o = hull[hull.size()-2];
                             const Point& a = hull.back();
                             double cp = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
                             if (cp <= 0) hull.pop_back(); else break;
                         }
                         hull.push_back(p);
                     }
                     hull.pop_back();
                     blockPts = hull;
                }
                
                // 3. Scan Bbox and Fill Final Mask
                int startX = std::max(0, (int)(min_c * bw));
                int startY = std::max(0, (int)(min_r * bh));
                int endX = std::min(W, (int)((max_c + 1) * bw));
                int endY = std::min(H, (int)((max_r + 1) * bh));
                // printf("[Debug] Obj %d Hull Bounds: X[%d-%d] Y[%d-%d] (grid: %d-%d, %d-%d)\n", obj.id, startX, endX, startY, endY, min_c, max_c, min_r, max_r);
                
                for(int y=startY; y<endY; ++y) {
                    int gr = y / bh;
                    for(int x=startX; x<endX; ++x) {
                        int gc = x / bw;
                        
                        // Point In Hull (Check Center of Grid Cell)
                        double testX = gc + 0.5;
                        double testY = gr + 0.5;
                        
                        bool inside = false;
                        size_t n = blockPts.size();
                        if(n < 3) inside = true; // Fallback
                        else {
                            for(size_t i=0, j=n-1; i<n; j=i++) {
                                if (((blockPts[i].y > testY) != (blockPts[j].y > testY)) &&
                                    (testX < (blockPts[j].x - blockPts[i].x) * (testY - blockPts[i].y) / (blockPts[j].y - blockPts[i].y) + blockPts[i].x)) {
                                    inside = !inside;
                                }
                            }
                        }
                        
                        if(inside) {
                            int idx = (y * W + x) * 3;
                            if (idx < 0 || idx >= (int)final_mask.size() - 2) {
                                printf("[ERROR] OOB Write Detected! idx=%d Size=%zu (y=%d x=%d W=%d)\n", idx, final_mask.size(), y, x, W);
                                continue;
                            }
                            if (bg_mask_img[idx] > 0) {
                                 final_mask[idx] = bg_mask_img[idx];   
                                 final_mask[idx+1] = bg_mask_img[idx+1];   
                                 final_mask[idx+2] = bg_mask_img[idx+2];   
                            }
                        }
                    }
                }
                
            }
            bg_mask_img = final_mask; // Update Global Mask
            //printf("[Debug] Skipping Hull Filter (Before Loop). MaskSz=%zu ObjSz=%zu\n", bg_mask_img.size(), objects.size());
        }
        

        for(const auto& obj : objects) 
        {
            
            

            if (std::isnan(obj.centerX) || std::isnan(obj.centerY)) {
                printf("[ERROR] Obj %d has NaN coordinates!\n", obj.id);
                continue;
            }
            
            // ---------------------------------------------------------
            // NEW: Data Logging
            // ---------------------------------------------------------
            // Format: FrameIndex ObjID Value
            if (f_log_dir.is_open()) f_log_dir << i << " " << obj.id << " " << obj.direction_variance << "\n";
            if (f_log_mag.is_open()) f_log_mag << i << " " << obj.id << " " << obj.magnitude_variance << "\n";
            // Use obj.pixel_count (Assuming it's populated now by SDK)
            // SDK might populate pixel_count if Convex Hull counting is enabled? 
            // The SDK change (inserting variance) was in fall_detector.
            // Oh, I need to ensure pixel_count is actually populated in SDK! 
            // In extractMotionObjects, obj.pixel_count wasn't explicitly set in the snippet I saw.
            // Wait, I must check fall_detector.cpp again to be sure pixel_count is set.
            // Assuming it is, or use blocks.size() * 16*16 equivalent?
            // Actually, for now, let's log object strength or block count? 
            // User asked for "Foreground Points" (pixel count). 
            // In SDK, I added `pixel_count` member, but I didn't see where it's assigned in extractMotionObjects.
            // It IS assigned in `detectFallPixelStats`! 
            // BUT `GetMotionObjects()` returns specific objects. 
            // The `objects` here come from `sdk.GetMotionObjects()`.
            // Does `detectFallPixelStats` update the SAME list? Yes, `pImpl->current_objects`.
            // So pixel_count should be valid if `detectFallPixelStats` runs.
            // f_log_fg moved out of loop to show ONLY largest object count
            
            // NEW METRICS
            if (f_log_accel.is_open()) f_log_accel << i << " " << obj.id << " " << obj.acceleration << "\n";
            
            if (obj_histories.count(obj.id)) {
                auto& hist = obj_histories[obj.id];
                hist.sad_strength_hist.push_back(obj.strength);
                hist.sad_accel_hist.push_back(obj.acceleration);
                if (hist.sad_strength_hist.size() > 30) hist.sad_strength_hist.pop_front();
                if (hist.sad_accel_hist.size() > 30) hist.sad_accel_hist.pop_front();
            }
            
            // Re-map speed log to LK metrics for the primary tracked objects
            // Format: Frame ObjID LK_Strength LK_Acceleration SAD_Strength
            float lk_s = 0, lk_a = 0;
            if (obj_histories.count(obj.id)) {
                lk_s = obj_histories[obj.id].current_lk_strength;
                lk_a = obj_histories[obj.id].current_lk_accel;
            }
            if (f_log_speed.is_open()) {
                f_log_speed << i << " " << obj.id << " " << lk_s << " " << lk_a << " " << obj.strength << "\n";
            }
            // f_log_bri moved out of loop to show ONLY largest object red count
            // ---------------------------------------------------------
            //printf("[Debug] Inside Obj Loop. ID=%d Blocks=%zu MVs=%zu. &obj=%p\n", obj.id, obj.blocks.size(), obj.block_motion_vectors.size(), &obj);
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
                    int cx = x + (int)bw / 2;
                    int cy = y + (int)bh / 2;
                    int ex = cx + (-1*mv.dx);
                    int ey = cy + (-1*mv.dy);
                    // Draw if moving
                    if (mv.dx != 0 || mv.dy != 0) {
                        drawArrow(current_frame_rgb, W, H, cx, cy, ex, ey, 0, 255, 0, 1);
                    }
                }
            }
            printf("[Debug] Drawn Blocks for Obj %d (Frame %d)\n", obj.id, i);
            //printf("[Debug] End Block Loop for Object %d\n", obj.id);
            //printf("[Debug] End Block Loop\n");
            
            // -------------------------------------------------------------
            /*
            // -------------------------------------------------------------
            // NEW: Axis Calculation (PCA) using Background Reference
            // -------------------------------------------------------------
            if (bg_saved_flag && !bg_reference.empty()) {
                // ... PCA LOGIC DISABLED ...
            }
            */

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
                 //stbi_write_bmp(crop_filename, side, side, 3, crop_buf.data());
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
        } // End of objects loop

        // Log Largest Object Stats (ID 0)
        if (f_log_fg.is_open())  f_log_fg  << i << " 0 " << largest_obj_pixel_count << "\n";
        if (f_log_bri.is_open()) f_log_bri << i << " 0 " << largest_obj_red_count << "\n";
        
        // 5. Draw All Vectors (Optional, user asked for "each object vector")
        // "各個物件...還有各物件的向量" -> Handled above.
        
        // 5. Draw Fall Box
        if (is_fall_in_current_frame) {
             printf("[Debug] Frame %d Fall Detected! Drawing Box.\n", i);
            std::string msg = "FALL DETECTED ";
            if (is_strong_fall) msg += "(STRONG)";
            else msg += "(NORMAL)";
            drawString(current_frame_rgb, W, H, 10, 10, msg, 255, 0, 0, 3);
        } else {
             drawString(current_frame_rgb, W, H, 10, 10, "NORMAL", 0, 255, 0, 3);
        }
        
        // SAVE BG MASK (Post-Loop)
        // Ensure bg_mask_img is initialized if empty but we want to save
        if (bg_mask_img.empty() && fallCfg.enable_save_bg_mask) {
             bg_mask_img.resize(W*H*3, 0); // Black mask if no object
        }

        if (fallCfg.enable_save_bg_mask) {
             char mask_filename[256];
             snprintf(mask_filename, sizeof(mask_filename), "%s/bg_mask_frame_%05d.jpg", save_dir.c_str(), i);
             
             if (!bg_mask_img.empty()) {
                  stbi_write_jpg(mask_filename, W, H, 3, bg_mask_img.data(), 90);
             } 
        }

        // 7. Bed Exit Status (Yellow Border)
        if (is_bed_exit_in_current_frame) {
            drawRectRGB(current_frame_rgb, W, H, 0, 0, W, H, 255, 255, 0, 2);
            // Optionally print text
             drawString(current_frame_rgb, W, H, W/2 - 50, 10, "BED EXIT", 255, 255, 0, 2);
        }

        // -------------------------------------------------------------
        // DELAYED BUFFERING LOGIC (METADATA ONLY)
        // -------------------------------------------------------------
        FrameContext ctx;
        ctx.frame_idx = i;
        ctx.w = W; 
        ctx.h = H;
        // ctx.rgb_data = current_frame_rgb; // REMOVED
        ctx.is_fall_detected = is_fall_in_current_frame;
        
        // Snapshot Object Angles
        for(const auto& pair : obj_smooth_thetas) {
            bool present = false;
            for(const auto& o : objects) if(o.id == pair.first) present = true;
            if(present) {
                FrameContext::ObjSnapshot snap;
                snap.id = pair.first;
                snap.angle = pair.second;
                ctx.objects.push_back(snap);
            }
        }
        
        frame_buffer.push_back(ctx);

        // PROCESS DELAYED FRAME (Verification)
        std::string delayed_feedback = "";
        
        if (frame_buffer.size() > (size_t)DELAY_FRAMES) {
            FrameContext& target = frame_buffer.front(); // Frame T (30 frames ago)
            
            // VERIFICATION LOGIC (Same as before, using metadata)
            bool confirmed_fall = false;

            if (target.is_fall_detected) {
                std::vector<int> suspect_ids;
                for(const auto& obj : target.objects) {
                    // Check if angle is "Horizontal"
                    if (std::abs(std::sin(obj.angle)) < 0.75) { 
                        suspect_ids.push_back(obj.id);
                    }
                }

                if (!suspect_ids.empty()) {
                    // Verify Suspects in FUTURE frames (which are in the buffer)
                    for(int sid : suspect_ids) {
                        int horizontal_count = 0;
                        int found_count = 0;
                        
                        // Check buffer (future frames from T's perspective)
                        for(auto it = frame_buffer.begin() + 1; it != frame_buffer.end(); ++it) {
                            for(const auto& future_obj : it->objects) {
                                if(future_obj.id == sid) {
                                    found_count++;
                                    if (std::abs(std::sin(future_obj.angle)) < 0.75) {
                                        horizontal_count++;
                                    }
                                    break;
                                }
                            }
                        }
                        
                        // Decision
                        if (found_count > 5) {
                            float ratio = (float)horizontal_count / found_count;
                            if (ratio > 0.8f) {
                                confirmed_fall = true;
                                std::cout << "[Verification] Frame " << target.frame_idx << " Fall Confirmed (ID " << sid << ")" << std::endl;
                                delayed_feedback = "VERIFIED FALL (Frame " + std::to_string(target.frame_idx) + ")";
                                break;
                            }
                        }
                    }
                }
            }
            frame_buffer.pop_front();
        }
        
        // DRAW DELAYED FEEDBACK ON CURRENT FRAME
        if (!delayed_feedback.empty()) {
             // Draw prominent alert on current frame
             drawRectRGB(current_frame_rgb, W, H, 100, H-60, W-200, 50, 255, 0, 0, -1); // Filed Bar
             drawString(current_frame_rgb, W, H, 120, H-50, delayed_feedback, 255, 255, 255, 3);
        }

        // Save Current Frame IMMEDIATELY
        char out_filename[256];
        snprintf(out_filename, sizeof(out_filename), "%s/frame_%05d.jpg", save_dir.c_str(), i);
        stbi_write_jpg(out_filename, W, H, 3, current_frame_rgb.data(), 90);
        
    } // End of loop
    
    // Close interval if still falling at end
    if (is_currently_falling && f_interval.is_open()) {
        f_interval << fall_start_frame << "," << (total_frames - 1) << "\n";
        
        std::string combined_reasons = "";
        // reuse static set? No, it's outside main loop now. 
        // We'll just put "Unknown/AtEnd" or similar if we didn't capture.
        // But better to just close it.
        detected_intervals_vec.push_back({fall_start_frame, (int)(total_frames - 1), "AtEnd"});
        total_fall_events++;
    }
    
    // NEW: Ground Truth Verification
    int tp = 0;
    int fp = 0;
    int fn = 0;
    std::vector<std::pair<int, int>> gt_intervals;
    std::vector<bool> gt_found;
    
    if (!gtFile.empty()) {
        gt_intervals = loadGroundTruth(gtFile);
        gt_found.resize(gt_intervals.size(), false);
        
        // Check Detects vs GT
        int tolerance = 30; // +/- 30 frames (1 sec) tolerance
        
        for(const auto& det : detected_intervals_vec) {
            bool matched = false;
            for(size_t k=0; k<gt_intervals.size(); ++k) {
                // Expanded Overlap Check
                int det_start = std::max(0, det.start - tolerance);
                int det_end = det.end + tolerance;
                
                int overlap_start = std::max(det_start, gt_intervals[k].first);
                int overlap_end = std::min(det_end, gt_intervals[k].second);
                
                if (overlap_start <= overlap_end) {
                    matched = true;
                    gt_found[k] = true;
                    // Note: Do NOT increment TP here to avoid double counting multiple detections for one GT.
                }
            }
            if(!matched) fp++; // If this detection matched NO GT, it's a False Positive.
        }
        
        // Count TP (Unique GTs found) and FN (GTs missed)
        for(bool f : gt_found) {
            if(f) tp++;
            else fn++;
        }
        
    } else {
        // No GT file provided, fallback to "Detected Count = False Positive" if strict, or unknown.
        // We can't verify Timing without GT.
        // But we can report Detected Count as is.
        // If dataset is known negative (Data 1), all detected are FP.
        // If dataset is known positive, we can't distinguish TP/FP easily without GT.
        // We'll leave TP=0, FP=total, FN=0 for now, but batch_run script handles the "Expectation" logic based on count.
        // Update: Let's assume everything detected is "Unverified Positive" if no GT.
        // BUT, batch_run uses the counts.
        tp = 0; fp = total_fall_events; fn = 0; // Default fail-safe?
    }

    // Save Verification Report
    std::string f_ver_name = save_dir + "/verification_report.txt";
    std::ofstream report(f_ver_name);
    if(report.is_open()) {
        report << "TP=" << tp << "\n";
        report << "FP=" << fp << "\n";
        report << "FN=" << fn << "\n";
        
        report << "--- Debug Info ---\n";
        std::cout << "--- GT Verification Debug ---\n";
        report << "Loaded " << gt_intervals.size() << " GT Intervals:\n";
        std::cout << "Loaded " << gt_intervals.size() << " GT Intervals:\n";
        for(size_t k=0; k<gt_intervals.size(); ++k) {
            std::string status = (gt_found[k] ? " (FOUND)" : " (MISSED)");
            report << "  [" << k << "] " << gt_intervals[k].first << "-" << gt_intervals[k].second << status << "\n";
            std::cout << "  [" << k << "] " << gt_intervals[k].first << "-" << gt_intervals[k].second << status << "\n";
        }
        report.close();
    }

    // Append Detailed Detections to Report
    std::ofstream report_app(f_ver_name, std::ios::app);
    if(report_app.is_open()) {
        report_app << "\n--- Detected Intervals ---\n";
        for(const auto& det : detected_intervals_vec) {
            report_app << "Start: " << det.start << " End: " << det.end << " Reason: " << det.reasons << "\n";
        }
        report_app.close();
    }
    
    // NEW: Save Count File (User Request)
    std::string f_count_name = save_dir + "/fall_count.txt";
    std::ofstream f_count(f_count_name);
    if(f_count.is_open()) {
        f_count << total_fall_events << "\n";
        f_count.close();
    }
    
    if (f_interval.is_open()) f_interval.close();

    // (No flush needed for images, just clear buffer)
    frame_buffer.clear();

    std::cout << "Done. Saved to " << save_dir << "/" << std::endl;
    return 0;
}
