#ifndef FALL_DETECTOR_H
#define FALL_DETECTOR_H

#include "Image.h"
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <memory>
#include <functional>

#include "HermesII_sdk.h"
#include "../ai/face_detector.h" // Include FaceDetector

namespace VisionSDK {

// Structs moved from HermesII_sdk.h
struct BlockPosition {
    int i = 0, j = 0;
    int x_start = 0, x_end = 0;
    int y_start = 0, y_end = 0;
};

// Internal Configuration Structure (aggregates all parameters)
struct InternalConfig {
    // General
    std::string model_path;
    int num_threads = 4;
    float confidence_threshold = 0.5f;

    // Image Related
    int expected_frame_interval_ms = 0; 
    int frame_interval_tolerance_ms = 33; 
    bool enable_draw_bg_noise = false; 
    bool enable_save_images = false;
    std::string save_image_path = "fall_results";

    // Motion Estimation
    int grid_cols = 12;
    int grid_rows = 16;
    int block_size = 16;
    int search_range = 24;
    int history_size = 5;
    int search_mode = 1;
    double block_change_threshold = 0.05;
    bool enable_block_decay = false;
    int block_decay_frames = 0;
    bool enable_block_dilation = false;
    int block_dilation_threshold = 2;

    // Object Extraction
    float object_extraction_threshold = 2.0f;
    int object_merge_radius = 3;
    float tracking_overlap_threshold = 0.5f;
    int tracking_mode = 1; 

    // Fall Detection
    float fall_movement_threshold = 2.0f;
    float fall_strong_threshold = 5.0f;
    float safe_area_ratio_threshold = 0.5f;
    float fall_acceleration_threshold = 5.0f; 
    float fall_acceleration_upper_threshold = 2.0f;
    float fall_acceleration_lower_threshold = -2.0f;
    float post_fall_distance_threshold = 10.0f;
    int post_fall_check_frames = 5;
    int fall_window_size = 30;
    int fall_duration = 5;
    bool enable_face_detection = true; // Default True
    
    // Background Update
    int bg_update_interval_frames = 0;
    float bg_update_alpha = 0.0f;
    bool enable_save_bg_mask = false;
    int bg_init_start_frame = 0;
    int bg_init_end_frame = 0;
    int bg_diff_threshold = 30;
    
    // Bed Exit Detection
    int bed_exit_history_len =60;          
    float bed_exit_min_inside_ratio = 0.6f; 
    float bed_exit_min_outside_ratio = 0.6f;
    bool enable_bed_exit_verification = true; // NEW
    bool enable_block_shrink_verification = true; // NEW

    // Optical Flow
    int opt_flow_frame_distance = 3;
    int perspective_point_x = 416;
    int perspective_point_y = 474;
};

class FallDetector {
public:
    FallDetector();
    ~FallDetector();

    // Configure algorithm parameters
    void SetConfig(const InternalConfig& config);

    // Set Bed Region directly with 4 points
    void SetBedRegion(const std::vector<std::pair<int, int>>& points);

    // Set the number of previous frames to check for block changes (default 5)
    // Set the number of previous frames to check for block changes (default 5)
    void SetHistorySize(int n);

    // Register a callback for fall events
    void RegisterCallback(VisionSDKCallback cb);

    // Main Detection Logic
    // Returns status code (0 = success)
    StatusCode Detect(const Image& frame, bool& is_fall);
    
    // Check if Bed Exit was detected in the last frame


    // Get the bed region points (if loaded)
    std::vector<std::pair<int, int>> GetBedRegion() const;

    // Get current motion objects
    // Get current motion objects
    const std::vector<MotionObject>& GetMotionObjects() const;
    
    // Get full-frame foreground objects
    std::vector<ObjectFeatures> GetFullFrameObjects() const;

    // Internal Getters
    std::vector<uint8_t> GetChangedBlocks() const;
    std::vector<MotionVector> GetMotionVectors() const;

private:
   class Impl;
   std::shared_ptr<Impl> pImpl;
};

}

#endif
