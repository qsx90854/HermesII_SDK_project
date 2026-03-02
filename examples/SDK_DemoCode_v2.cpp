
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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//#define USE_FFMPEG_READER 1
#include "ffmpeg_precoss.h" // 加入這行
#include <memory> // For std::unique_ptr


using namespace VisionSDK;

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// TOGGLE: 1 = Use SDK Internal Logic, 0 = Use Demo Custom Logic (Peak-Valley)
#define USE_SDK_FALL_RESULT 1



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




// ==========================================
// Homography & Warping Utilities (Manual)
// ==========================================

bool solveLinearSystem(int N, const float* A, const float* b, float* x) {
    std::vector<float> M(N * (N + 1));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) M[i * (N + 1) + j] = A[i * N + j];
        M[i * (N + 1) + N] = b[i];
    }
    for (int i = 0; i < N; ++i) {
        int pivot = i;
        for (int j = i + 1; j < N; ++j) {
            if (std::abs(M[j * (N + 1) + i]) > std::abs(M[pivot * (N + 1) + i])) pivot = j;
        }
        if (std::abs(M[pivot * (N + 1) + i]) < 1e-6) return false;
        if (pivot != i) {
            for (int k = i; k <= N; ++k) std::swap(M[i * (N + 1) + k], M[pivot * (N + 1) + k]);
        }
        float div = M[i * (N + 1) + i];
        for (int k = i; k <= N; ++k) M[i * (N + 1) + k] /= div;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                float mul = M[j * (N + 1) + i];
                for (int k = i; k <= N; ++k) M[j * (N + 1) + k] -= mul * M[i * (N + 1) + k];
            }
        }
    }
    for (int i = 0; i < N; ++i) x[i] = M[i * (N + 1) + N];
    return true;
}

// Compute Homography Matrix H (3x3) using 4 point correspondences
bool computeHomography(const std::vector<std::pair<int, int>>& src_points, 
                        const std::vector<std::pair<float, float>>& dst_points, 
                        float H[9]) {
    if (src_points.size() != 4 || dst_points.size() != 4) return false;
    float A[64] = {0};
    float b[8] = {0};
    for (int i = 0; i < 4; ++i) {
        float sx = (float)src_points[i].first;
        float sy = (float)src_points[i].second;
        float dx = dst_points[i].first;
        float dy = dst_points[i].second;
        int r1 = 2 * i;
        A[r1*8+0]=sx; A[r1*8+1]=sy; A[r1*8+2]=1.0f; A[r1*8+6]=-sx*dx; A[r1*8+7]=-sy*dx; b[r1]=dx;
        int r2 = 2 * i + 1;
        A[r2*8+3]=sx; A[r2*8+4]=sy; A[r2*8+5]=1.0f; A[r2*8+6]=-sx*dy; A[r2*8+7]=-sy*dy; b[r2]=dy;
    }
    float x[8];
    if (!solveLinearSystem(8, A, b, x)) return false;
    H[0]=x[0]; H[1]=x[1]; H[2]=x[2]; H[3]=x[3]; H[4]=x[4]; H[5]=x[5]; H[6]=x[6]; H[7]=x[7]; H[8]=1.0f;
    return true;
}

// Wrapper for simple rectangle target
bool computeHomographyRect(const std::vector<std::pair<int, int>>& src_points, float target_w, float target_h, float H[9]) {
    std::vector<std::pair<float, float>> dst = {{0,0}, {target_w, 0}, {target_w, target_h}, {0, target_h}};
    return computeHomography(src_points, dst, H);
}

void projectPoint(float u, float v, const float H[9], float& x, float& y) {
    float z = H[6] * u + H[7] * v + H[8];
    if (std::abs(z) > 1e-4) {
        x = (H[0] * u + H[1] * v + H[2]) / z;
        y = (H[3] * u + H[4] * v + H[5]) / z;
    } else { x = 0; y = 0; }
}

// Inverse Warping for efficiency without gaps
void warpPerspectiveDemo(const std::vector<uint8_t>& src, int sw, int sh, 
                        std::vector<uint8_t>& dst, int dw, int dh, 
                        const float H_inv[9]) {
    dst.assign(dw * dh * 3, 0);
    for (int dy = 0; dy < dh; ++dy) {
        for (int dx = 0; dx < dw; ++dx) {
            float sx, sy;
            projectPoint((float)dx, (float)dy, H_inv, sx, sy);
            
            int isx = (int)(sx + 0.5f);
            int isy = (int)(sy + 0.5f);
            
            if (isx >= 0 && isx < sw && isy >= 0 && isy < sh) {
                int src_idx = (isy * sw + isx) * 3;
                int dst_idx = (dy * dw + dx) * 3;
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
            }
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
// current_frame_rgb_onlyBlock removed (unused)
int current_frame_idx = -1;
int pW = 800;
int pH = 450;
bool enable_save_face_images = false;
bool is_fall_in_current_frame = false;
bool is_strong_fall = false;
bool custom_fall_signal = false;
std::string current_frame_reasons = "";
bool is_bed_exit_in_current_frame = false;
bool is_face_in_current_frame = false;
float current_face_x = 0, current_face_y = 0, current_face_w = 0, current_face_h = 0;
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
    is_bed_exit_in_current_frame = event.is_bed_exit;
    is_face_in_current_frame = event.is_face;
    current_face_x = event.face_x;
    current_face_y = event.face_y;
    current_face_w = event.face_w;
    current_face_h = event.face_h;

    if (is_fall_in_current_frame) {
        std::cout << "[Event] Fall detected at frame " << current_frame_idx << std::endl;
    }
    if (is_bed_exit_in_current_frame) {
        std::cout << "[Event] Bed exit detected at frame " << current_frame_idx << std::endl;
    }
    if (is_face_in_current_frame) {
        std::cout << "[Event] Face detected at frame " << current_frame_idx << std::endl;
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
    objCfg.foreground_merge_radius = cfg.getInt("Object.Foreground_Merge_Range", 1); // Default 1 pixel
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
    fallCfg.bed_pixel_ratio_threshold = bed_pixel_ratio_threshold; // NEW
    fallCfg.safe_area_ratio_threshold = (float)cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5);
    fallCfg.fall_window_size = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 30);
    fallCfg.fall_duration = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 5);
    fallCfg.post_fall_distance_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Post_Fall_Distance_Threshold", 10.0f);
    fallCfg.post_fall_check_frames = cfg.getInt("FallDetect.Fall_Detect_Post_Fall_Check_Frames", 5);
    fallCfg.momentum_calc_type = cfg.getInt("FallDetect.Fall_Detect_Momentum_Calc_Type", 0);
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
    fallCfg.bed_update_alpha_multiplier = cfg.getFloat("FallDetect.Bed_Update_Alpha_Multiplier", 4.0f);
    fallCfg.enable_post_bed_exit_threshold = (cfg.getInt("FallDetect.Enable_Post_BedExit_Threshold", 0) != 0);
    fallCfg.post_bed_exit_threshold_multiplier = cfg.getFloat("FallDetect.Post_BedExit_Threshold_Multiplier", 0.7f);
    fallCfg.projection_use_foreground = (cfg.getInt("FallDetect.Projection_Use_Foreground", 0) != 0);
    
    // Optical Flow Params
    fallCfg.opt_flow_frame_distance = cfg.getInt("OpticalFlow.CompareFrameDistance", 3);
    fallCfg.perspective_point_x = cfg.getInt("OpticalFlow.PerspectivePointX", 416);
    fallCfg.perspective_point_y = cfg.getInt("OpticalFlow.PerspectivePointY", 474);
    fallCfg.min_trigger_area = cfg.getInt("OpticalFlow.MinTriggerArea", 2000);
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


    // Background Accumulation Variables
    int bg_init_start_frame = fallCfg.bg_init_start_frame;
    int bg_init_end_frame = fallCfg.bg_init_end_frame;


    std::vector<uint8_t> bg_reference; // Persistent BG for PCA
    int bg_frames_count = 0;
    bool bg_saved_flag = false;

    // Load Initial Background if specified
    std::string bg_file_path = appCfg.getString("Demo.Demo_Background_Image_Path", "");
    if (!bg_file_path.empty()) {
        bool loaded = false;
        printf("[Demo] Loading Background: %s\n", bg_file_path.c_str());
        
        // Try STB Loading for JPG/PNG/BMP
        int iw, ih, ic;
        unsigned char* data = stbi_load(bg_file_path.c_str(), &iw, &ih, &ic, 3);
        if (data) {
             printf("[Demo] STB Loaded Image: %dx%d channels: %d\n", iw, ih, ic);
             if (iw == W && ih == H) {
                 bg_reference.assign(data, data + W*H*3);
                 sdk.SetBackground(bg_reference.data(), W, H, 3);
                 bg_saved_flag = true;
                 loaded = true;
                 printf("[Demo] Success: Set Background from Image.\n");
             } else {
                 printf("[Demo] Warning: Resizing Image %dx%d -> %dx%d\n", iw, ih, W, H);
                 bg_reference.resize(W * H * 3);
                 for (int y = 0; y < H; ++y) {
                     for (int x = 0; x < W; ++x) {
                         int src_x = x * iw / W;
                         int src_y = y * ih / H;
                         int src_idx = (src_y * iw + src_x) * 3; 
                         int dst_idx = (y * W + x) * 3;
                         bg_reference[dst_idx] = data[src_idx];
                         bg_reference[dst_idx+1] = data[src_idx+1];
                         bg_reference[dst_idx+2] = data[src_idx+2];
                     }
                 }
                 sdk.SetBackground(bg_reference.data(), W, H, 3);
                 bg_saved_flag = true;
                 loaded = true;
                 printf("[Demo] Success: Set Resized Background from Image.\n");
             }
             stbi_image_free(data);
        }
        
        if (!loaded) {
            std::ifstream fbg(bg_file_path, std::ios::binary | std::ios::ate);
            if (fbg) {
                std::streamsize size = fbg.tellg();
                fbg.seekg(0, std::ios::beg);
                if (size == W * H * 3) {
                    bg_reference.resize(W * H * 3);
                    if (fbg.read((char*)bg_reference.data(), size)) {
                        printf("[Demo] Loaded RAW Background Image: %s\n", bg_file_path.c_str());
                        sdk.SetBackground(bg_reference.data(), W, H, 3);
                        bg_saved_flag = true;
                        
                        // NEW: Save loaded background for verification (as requested)
                        char bg_save_name[256];
                        snprintf(bg_save_name, sizeof(bg_save_name), "%s/background_init_raw.jpg", save_dir.c_str());
                        stbi_write_jpg(bg_save_name, W, H, 3, bg_reference.data(), 90);
                        printf("[Demo] Saved raw background as %s\n", bg_save_name);
                    }
                } else {
                    // Start of Manual Resize Logic for RAW
                    int origW = appCfg.getInt("Demo.Demo_Original_Width", 1920);
                    int origH = appCfg.getInt("Demo.Demo_Original_Height", 1080);
                    long expected_orig_size = (long)origW * origH * 3;
                    
                    if (size == expected_orig_size) {
                        printf("[Demo] RAW Image Size Mismatch (%ld vs %d). Attempting Resize %dx%d -> %dx%d\n", (long)size, W*H*3, origW, origH, W, H);
                        std::vector<uint8_t> raw_orig(size);
                        if (fbg.read((char*)raw_orig.data(), size)) {
                            bg_reference.resize(W * H * 3);
                            for (int y = 0; y < H; ++y) {
                                for (int x = 0; x < W; ++x) {
                                    int src_x = x * origW / W;
                                    int src_y = y * origH / H;
                                    int src_idx = (src_y * origW + src_x) * 3; 
                                    int dst_idx = (y * W + x) * 3;
                                    bg_reference[dst_idx] = raw_orig[src_idx];
                                    bg_reference[dst_idx+1] = raw_orig[src_idx+1];
                                    bg_reference[dst_idx+2] = raw_orig[src_idx+2];
                                }
                            }
                            sdk.SetBackground(bg_reference.data(), W, H, 3);
                            bg_saved_flag = true;
                            printf("[Demo] Success: Set Resized Background from RAW.\n");
                        }
                    } else {
                         printf("[Demo] Warning: Background Image size mismatch! Expected %d or %ld, Got %ld. Ignoring.\n", W*H*3, (long)expected_orig_size, (long)size);
                    }
                }
            } else {
                 printf("[Demo] Warning: Background Image path provided but file not found: %s\n", bg_file_path.c_str());
            }
        }
    }

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


    // Check if input is MP4
    std::unique_ptr<VideoReader> videoReader = nullptr;
    if (imgFormat.size() > 4 && imgFormat.substr(imgFormat.size() - 4) == ".mp4") {
        std::cout << "[Demo] MP4 Mode Detected: " << imgFormat << std::endl;
        try {
            videoReader = std::make_unique<VideoReader>(imgFormat, W, H);
        } catch (const std::exception& e) {
            std::cerr << "Error opening MP4: " << e.what() << std::endl;
            return -1;
        }
    } else {
        std::cout << "[Demo] Image Sequence Mode: " << imgFormat << std::endl;
    }

    // Main processing loop
    int total_frames = start_frame + num_images; // Define total_frames based on existing variables
    // Variables for Average Time Calculation
    double total_process_time_ms = 0.0;
    long frame_count_time = 0;

    for (int i = start_frame; i < total_frames; i += frame_step) 
    {
        auto t_read_start = std::chrono::steady_clock::now();

        if (videoReader) {
            // Apply Frame Step Skipping for MP4
            // Since MP4 reader is sequential, we must consume and discard frames if step > 1
            if (i > start_frame) { // Don't skip before the very first frame
                for (int s = 0; s < frame_step - 1; ++s) {
                    // Read into dummy buffer or same buffer to discard
                    if (!videoReader->readFrame(file_buffer)) {
                        break; // End of video during skip
                    }
                }
            }

             // 讀取 MP4 的下一幀到 file_buffer 中，並自動 Resize 轉 RGB
            if (!videoReader->readFrame(file_buffer)) {
                std::cout << "影片讀取完畢或發生錯誤，結束迴圈。" << std::endl;
                break; // 影片結束就跳出迴圈
            }
        } else {
            char raw_name[256];
            snprintf(raw_name, sizeof(raw_name), pattern.c_str(), i);
            
            std::ifstream file(raw_name, std::ios::binary);
            if (!file) {
                if (i > 0) break; 
                continue;
            }
            file.read(reinterpret_cast<char*>(file_buffer.data()), frame_size_rgb);
            file.close();
        }
        // Timer End for Read
        auto t_read_end = std::chrono::steady_clock::now();
        double read_duration_ms = std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();

        // Simulate 30FPS Camera (Ensure Read+Wait = 33ms)
        if (read_duration_ms < 33.0) {
            int sleep_ms = (int)(33.0 - read_duration_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }



        // (current_frame_rgb_onlyBlock removed)
        current_frame_idx = i;
        is_fall_in_current_frame = false; // Reset
        is_bed_exit_in_current_frame = false;
        is_face_in_current_frame = false;

        auto t1 = std::chrono::steady_clock::now();
        sdk.SetInputMemory(file_buffer.data(), W, H, 3);
        
        is_fall_in_current_frame = false; // Reset for custom logic
        sdk.ProcessNextFrame();

        auto t2 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (i % 50 == 0) 
        {
            std::cout << "Frame " << i << " Total Process Time: " << ms << " ms (" << (1000.0/ms) << " FPS)" << std::endl;
        }
        
        // Accumulate Average Time
        total_process_time_ms += ms;
        frame_count_time++;

        bool custom_fall_signal = false;



        
        // Override SDK fall signal
        static int custom_hold_frames = 0;
        if (custom_fall_signal) custom_hold_frames = 30; // Hold for 1 second at 30fps
        
        // Reset original flag and use custom one
        #if USE_SDK_FALL_RESULT
            // SDK Logic: Do nothing (keep result from onFallDetected)
            // But we still countdown for debug visualization if needed
        #else
            // Demo Logic: Override SDK result
            is_fall_in_current_frame = (custom_hold_frames > 0);
        #endif
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
                     
                     FallInterval new_interval;
                     new_interval.start = fall_start_frame;
                     new_interval.end = i - 1;
                     new_interval.reasons = combined_reasons;
                     detected_intervals_vec.push_back(new_interval);
                     total_fall_events++;
                }
            }
        }



        

        
    } // End of loop
    
    // Close interval if still falling at end
    if (is_currently_falling && f_interval.is_open()) {
        f_interval << fall_start_frame << "," << (total_frames - 1) << "\n";
        
        std::string combined_reasons = "";
        // reuse static set? No, it's outside main loop now. 
        // We'll just put "Unknown/AtEnd" or similar if we didn't capture.
        // But better to just close it.
        FallInterval new_interval;
        new_interval.start = fall_start_frame;
        new_interval.end = (int)(total_frames - 1);
        new_interval.reasons = "AtEnd";
        detected_intervals_vec.push_back(new_interval);
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
        
        // Save Average Time
        if (frame_count_time > 0) {
            double avg_ms = total_process_time_ms / frame_count_time;
            double avg_fps = 1000.0 / avg_ms;
            report << "AvgTime=" << avg_ms << "\n";
            std::cout << "[Demo] Average Process Time: " << avg_ms << " ms (" << avg_fps << " FPS)" << std::endl;
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



    std::cout << "Done. Saved to " << save_dir << "/" << std::endl;
    return 0;
}
