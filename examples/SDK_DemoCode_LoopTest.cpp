#include "HermesII_sdk.h"
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <fstream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

// Read config loader from demo utility
class ConfigLoader {
    std::string ini_file;
public:
    int getInt(const std::string& key, int defaultVal) { return defaultVal; }
    float getFloat(const std::string& key, float defaultVal) { return defaultVal; }
    bool load(const std::string& file) { return false; }
};

int main(int argc, char** argv) {
    std::cout << "Starting Edge SDK Loop Test..." << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;

    // Initialize SDK
    VisionSDK::VisionSDK sdk;
    sdk.Init("", 4);

    // Provide default configs (Motion, ObjectExtraction, FallDetection) to prevent default errors
    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = 10;
    motionCfg.grid_rows = 6;
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = 5;
    motionCfg.block_change_threshold = 0.03f;
    motionCfg.search_mode = 1;
    motionCfg.enable_block_decay = false;
    motionCfg.enable_block_dilation = false;
    sdk.SetConfig(&motionCfg);

    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_merge_radius = 1;
    objCfg.foreground_merge_radius = 6;
    objCfg.object_extraction_threshold = 2.0f;
    objCfg.tracking_overlap_threshold = 0.5f;
    objCfg.tracking_mode = 4;
    sdk.SetConfig(&objCfg);

    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = 2.5f;
    fallCfg.fall_strong_threshold = 6.0f;
    fallCfg.fall_acceleration_threshold = -4.0f;
    fallCfg.fall_acceleration_upper_threshold = 6.0f;
    fallCfg.fall_acceleration_lower_threshold = -4.0f;
    fallCfg.bed_pixel_ratio_threshold = 0.3f; // Default
    fallCfg.safe_area_ratio_threshold = 0.5f;
    fallCfg.fall_window_size = 30;
    fallCfg.fall_duration = 5;
    fallCfg.post_fall_distance_threshold = 4.0f;
    fallCfg.post_fall_check_frames = 5;
    fallCfg.momentum_calc_type = 1;
    fallCfg.enable_face_detection = true;
    fallCfg.face_detect_interval_frames = 8;
    fallCfg.enable_edge_drop_filter = true;
    sdk.SetConfig(&fallCfg);

    std::vector<std::string> image_files = {
        "test00001.bmp",
        "test00002.bmp",
        "test00003.bmp",
        "test00004.bmp",
        "test00005.bmp",
        "test00006.bmp",
        "test00007.bmp",
        "test00008.bmp"
    };

    std::vector<std::vector<uint8_t>> image_data(8);
    int width = 0, height = 0;
    bool all_files_loaded = true;

    for (size_t i = 0; i < image_files.size(); ++i) {
        int channels = 0;
        unsigned char* data = stbi_load(image_files[i].c_str(), &width, &height, &channels, 3);
        if (!data) {
            std::cerr << "Failed to load " << image_files[i] << " - Creating dummy 800x450 rgb black frame." << std::endl;
            all_files_loaded = false;
            width = 800;
            height = 450;
            image_data[i].resize(width * height * 3, 0); 
        } else {
            image_data[i].assign(data, data + width * height * 3);
            stbi_image_free(data);
            std::cout << "Loaded " << image_files[i] << ": " << width << "x" << height << std::endl;
        }
    }

    if (!all_files_loaded) {
        std::cerr << "Warning: Some or all test BMP files were not found. Please ensure test1.bmp, test2.bmp, test3.bmp exist in the run directory." << std::endl;
    }

    long long frame_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        int idx = (frame_count) % 8;
        
        sdk.SetInputMemory(image_data[idx].data(), width, height, 3);

        VisionSDK::StatusCode status = sdk.ProcessNextFrame();
        if (status != VisionSDK::StatusCode::OK) {
            std::cerr << "ProcessNextFrame returned error." << std::endl;
        }

        // Just silently looping. We can print progress every 1000 frames.
        if (frame_count % 1000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            std::cout << "[SDK_DemoCode_LoopTest] Processed " << frame_count << " frames. "
                      << "Total Time (since start): " << duration << " ms, "
                      << "FPS ~: " << (frame_count / ((duration > 0 ? duration : 1) / 1000.0f)) << std::endl;
        }
        
        frame_count++;
        // Very minimal sleep to yield CPU just slightly (1ms), optional but good practice for edge device endless loops to not entirely throttle hardware if empty dummy processing.
        // usleep(1000); 
    }

    return 0;
}
