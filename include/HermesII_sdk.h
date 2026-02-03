#ifndef HERMESII_SDK_H
#define HERMESII_SDK_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include "Image.h"

namespace VisionSDK {

// Merged Configuration
// Config Types
enum class ConfigType {
    MotionEstimation_v1,
    ObjectExtraction_v1,
    FallDetection_v1,
    FallDetection_v2,
    FallDetection_v3,
    BedExitDetection_v1,
    ImageRelated_v1
};

struct ConfigHeader {
    ConfigType type;
    int version;
};

// Versioned Structs

struct MotionEstimation_v1 {
    ConfigHeader header;
    int grid_cols = 0;
    int grid_rows = 0;
    int block_size = 0;
    int search_range = 0;
    int history_size = 0;
    int search_mode = 0;
    double block_change_threshold = 0.0;
    bool enable_block_decay = false;
    int block_decay_frames = 0;
    bool enable_block_dilation = false;
    int block_dilation_threshold = 0;
};

struct ObjectExtraction_v1 {
    ConfigHeader header;
    float object_extraction_threshold = 0.0f;
    int object_merge_radius = 0;
    float tracking_overlap_threshold = 0.0f;
    int tracking_mode = 0; 
};

struct FallDetection_v1 {
    ConfigHeader header;
    float fall_movement_threshold = 0.0f;
    float fall_strong_threshold = 0.0f;
    float safe_area_ratio_threshold = 0.0f;
    float fall_acceleration_threshold = 0.0f; 
    int fall_window_size = 0;
    int fall_duration = 0;
};

struct FallDetection_v2 {
    ConfigHeader header;
    float fall_movement_threshold = 0.0f;
    float fall_strong_threshold = 0.0f;
    float safe_area_ratio_threshold = 0.0f;
    float fall_acceleration_threshold = 0.0f; 
    int fall_window_size = 0;
    int fall_duration = 0;
    bool enable_face_detection = true; // New Face Detection Control
    float fall_acceleration_upper_threshold = 2.0f;
    float fall_acceleration_lower_threshold = -2.0f;
};

struct FallDetection_v3 {
    ConfigHeader header;
    float fall_movement_threshold = 0.0f;
    float fall_strong_threshold = 0.0f;
    float safe_area_ratio_threshold = 0.0f;
    float fall_acceleration_threshold = 0.0f; 
    int fall_window_size = 0;
    int fall_duration = 0;
    bool enable_face_detection = true;
    bool enable_save_bg_mask = false;
    int bg_init_start_frame = 0;
    int bg_init_end_frame = 0;
    int bg_diff_threshold = 30; // Default
    
    // Background Update Params
    int bg_update_interval_frames = 0;
    float bg_update_alpha = 0.0f;
    float fall_acceleration_upper_threshold = 2.0f;
    float fall_acceleration_lower_threshold = -2.0f;
    float post_fall_distance_threshold = 10.0f;
    int post_fall_check_frames = 5;
    bool enable_bed_exit_verification = true; // NEW
    bool enable_block_shrink_verification = true; // NEW
};

struct BedExitDetection_v1 {
    ConfigHeader header;
    int bed_exit_history_len = 0;          
    float bed_exit_min_inside_ratio = 0.0f; 
    float bed_exit_min_outside_ratio = 0.0f;
};

struct ImageRelated_v1 {
    ConfigHeader header;
    int expected_frame_interval_ms = 0; 
    int frame_interval_tolerance_ms = 0; 
    bool enable_draw_bg_noise = false; 
    bool enable_save_images = false;
    std::string save_image_path = ""; 
};

struct Image {
    unsigned char* data;
    int width;
    int height;
    int channels;
    uint64_t timestamp; // Timestamp in milliseconds
};

struct DetectionResult {
    int class_id;
    float confidence;
    float x, y, w, h;
};

// Fusion V2 Structures
struct FusionIntrinsics {
    float fx, fy, cx, cy;
};

struct FusionDistCoeffs {
    float k1, k2, p1, p2, k3;
};

struct FusionExtrinsics {
    float R[9]; // Row-major 3x3
    float T[3]; // x, y, z
};

struct FusionParams {
    FusionIntrinsics K_ir, K_th;
    FusionDistCoeffs D_ir, D_th;
    FusionExtrinsics Extrinsics;
    float assumed_distance_mm;
};


// ==========================================
// Fall Detection Types
// ==========================================

struct MotionVector {
    int dx = 0;
    int dy = 0;
    MotionVector() {}
    MotionVector(int _dx, int _dy) : dx(_dx), dy(_dy) {}
};

struct MotionObject {
    int id = -1;
    float centerX = 0.0f, centerY = 0.0f;
    float avgDx = 0.0f, avgDy = 0.0f;
    std::vector<int> blocks;
    std::vector<MotionVector> block_motion_vectors; // NEW: per-block momentum
    std::vector<int> block_groups; // NEW: K-means group ID (0 or 1)
    float strength = 0.0f;
    float acceleration = 0.0f; // NEW: Change in strength/momentum
    int lifetime = 0;
    float safe_area_ratio = 0.0f;
    std::vector<std::pair<int, int>> trajectory; // Center point history
    
    // NEW for Pixel-Based Fall Detection
    int pixel_count = 0;
    float avg_brightness = 0.0f;
    
    // NEW for Stability Analysis
    float direction_variance = 0.0f; // Std Dev of Block Angles (Radians)
    float magnitude_variance = 0.0f; // Std Dev of Block Speeds

    // NEW: Global Frame Stats
    int total_frame_pixel_count = 0;
};


struct VisionSDKEvent {
    int frame_index;
    float confidence;
    bool is_fall_detected; // NEW: Explicit flag
    bool is_strong;
    bool is_bed_exit;
    bool is_face;
    float face_x, face_y, face_w, face_h; // Face ROI (Union)
};

using VisionSDKCallback = std::function<void(const VisionSDKEvent&)>;

enum class StatusCode {
    OK,
    ERROR_INIT_FAILED,
    ERROR_INFERENCE_FAILED,
    ERROR_INVALID_INPUT,
    ERROR_FUSION_FAILED,
    ERROR_TIMESTAMP_DISCONTINUITY
};

// ==========================================
// Camera Fusion (Extrinsics)
// ==========================================
struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
    float distortion[5]; // k1, k2, p1, p2, k3
};

struct CameraExtrinsics {
    float rotation[9]; // 3x3 Row-Major
    float translation[3];
};

class VisionSDK {
public:

    VisionSDK();
    ~VisionSDK();

    static const char* GetVersion();

    // Initialize SDK (Optional model loading)
    StatusCode Init(const std::string& model_path = "", int num_threads = 4);
    
    // Set Configuration
    StatusCode SetConfig(const void* config);

    // AI Inference
    StatusCode RunInference(const Image& img, std::vector<DetectionResult>& results);

    // Image Fusion
    StatusCode FuseImages(const Image& img1, const Image& img2, Image& output);

    /**
     * @brief Fuses/Overlays Image A onto Image B using camera parameters.
     * Assumes a planar scene (ground plane z=0).
     * 
     * @param imgA Source Image
     * @param camA Intrinsics/Extrinsics for Camera A
     * @param imgB Target Image (Base)
     * @param camB Intrinsics/Extrinsics for Camera B
     * @param output Output Image (same size as B)
     * @return StatusCode 
     */
    StatusCode FuseImages3D(const Image& imgA, const CameraIntrinsics& camA, const CameraExtrinsics& extA,
                            const Image& imgB, const CameraIntrinsics& camB, const CameraExtrinsics& extB,
                            Image& output);

    // Fusion V2
    StatusCode FuseImagesV2(const Image& img_ir, const Image& img_th, const FusionParams& params, Image& output_fused);
    
    // Coordinate Mapping V2
    StatusCode MapPointV2(float ir_x, float ir_y, const FusionParams& params, float& th_x, float& th_y);




    // Unified Fall Detection API
    void RegisterVisionSDKCallback(VisionSDKCallback callback);

    /**
     * @brief Set the Input Memory buffer for zero-copy processing.
     * 
     * @param buffer Pointer to the image data.
     * @param width Image width.
     * @param height Image height.
     * @param channels Image channels (e.g., 3 for RGB).
     * @return StatusCode 
     */
    StatusCode SetInputMemory(unsigned char* buffer, int width, int height, int channels);

    /**
     * @brief Trigger the SDK to process the current frame in the shared buffer.
     * The result will be reported via the registered FallCallback.
     * 
     * @return StatusCode 
     */
    StatusCode ProcessNextFrame();
    
    // Set Bed Region directly with 4 points
    void SetBedRegion(const std::vector<std::pair<int, int>>& points);
    
    // Get current bed region points
    std::vector<std::pair<int, int>> GetBedRegion();

    // Get internal state for visualization
    std::vector<MotionObject> GetMotionObjects();
    std::vector<uint8_t> GetChangedBlocks(); // Returns mask data same size as grid (cols*rows)
    std::vector<MotionVector> GetMotionVectors(); // Returns vector map same size as grid




private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace VisionSDK

#endif // HERMESII_SDK_H
