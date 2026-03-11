
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

//#define USE_FFMPEG_READER 1
#include "ffmpeg_precoss.h" // 加入這行
#include <memory> // For std::unique_ptr

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace VisionSDK;

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// TOGGLE: 1 = Use SDK Internal Logic, 0 = Use Demo Custom Logic (Peak-Valley)
#define USE_SDK_FALL_RESULT 1



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




int main(int argc, char** argv) {
    std::cout << "Starting Fall Callback Demo v2 SAVE (30FPS Sim)..." << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;
    // 1. Load Configs
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

    // 2. Initialize SDK
    VisionSDK::VisionSDK sdk;
    sdk.Init("", 4); // Default Init

    // 1. Motion Estimation Config
    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = 10;
    motionCfg.grid_rows = 6;
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = 5;
    motionCfg.block_change_threshold = 0.03;
    motionCfg.search_mode = 1;
    motionCfg.enable_block_decay = false;
    motionCfg.block_decay_frames = 6;
    motionCfg.enable_block_dilation = false;
    motionCfg.block_dilation_threshold = 4;
    sdk.SetConfig(&motionCfg);

    // 2. Object Extraction Config
    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_merge_radius = 1;
    objCfg.foreground_merge_radius = 6; 
    objCfg.object_extraction_threshold = 2.0f; 
    objCfg.tracking_overlap_threshold = 0.5f;
    objCfg.tracking_mode = 4;
    sdk.SetConfig(&objCfg);

    // 3. Fall Detection Config
    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = 2.5f;
    fallCfg.fall_strong_threshold = 6.0f;
    fallCfg.fall_acceleration_threshold = -4.0f;
    fallCfg.fall_acceleration_upper_threshold = 6.0f;
    fallCfg.fall_acceleration_lower_threshold = -4.0f;
    fallCfg.bed_pixel_ratio_threshold = 0.15f; 
    fallCfg.safe_area_ratio_threshold = 0.5f;
    fallCfg.fall_window_size = 30;
    fallCfg.fall_duration = 5;
    fallCfg.post_fall_distance_threshold = 4.0f;
    fallCfg.post_fall_check_frames = 5;
    fallCfg.momentum_calc_type = 1;

    fallCfg.enable_face_detection = true;
    fallCfg.face_detect_interval_frames = 8;
    // Load Verification Flags
    fallCfg.enable_bed_exit_verification = false;
    fallCfg.enable_block_shrink_verification = false;
    
    // Background Method Config
    fallCfg.enable_save_bg_mask = true;
    fallCfg.bg_init_start_frame = 2;
    fallCfg.bg_init_end_frame = 5;
    fallCfg.bg_diff_threshold = 18;
    fallCfg.bg_update_interval_frames = 8;
    fallCfg.bg_update_alpha = 0.1f;
    fallCfg.bed_update_alpha_multiplier = 8.0f;
    fallCfg.enable_post_bed_exit_threshold = true;
    fallCfg.post_bed_exit_threshold_multiplier = 0.7f;
    fallCfg.projection_use_foreground = false;

    sdk.SetConfig(&fallCfg);

    // 4. Image Related Config
    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    std::string savePath = save_dir; // Use the directory created by AppConfig
    imgCfg.save_image_path = savePath;
    imgCfg.enable_save_images = false; // We do manual saving here
    imgCfg.enable_draw_bg_noise = (appCfg.getInt("Demo.Demo_Draw_Background_Noise", 0) != 0);
    imgCfg.expected_frame_interval_ms = 33;
    imgCfg.frame_interval_tolerance_ms = 10;
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



    std::vector<uint8_t> bg_reference; // Persistent BG for PCA


    // Load Initial Background if specified
    std::string bg_file_path = appCfg.getString("Demo.Demo_Background_Image_Path", "");
    if (!bg_file_path.empty()) 
    {
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
                 loaded = true;
                 printf("[Demo] Success: Set Resized Background from Image.\n");
             }
             stbi_image_free(data);
        }
        //Load Background by raw
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
    else
    {
        printf("[Demo] Warning: No Background Image path provided. Using default background.");
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

        
        // Accumulate Average Time
        total_process_time_ms += ms;
        frame_count_time++;

    } // End of loop
    

    std::cout << "Done. Saved to " << save_dir << "/" << std::endl;
    return 0;
}
