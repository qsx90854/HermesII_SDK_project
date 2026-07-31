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
    ImageRelated_v1,
    EventRecording_v1
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
    int object_merge_radius = 3;
    int foreground_merge_radius = 1; // NEW: Default 1 pixel merge
    float tracking_overlap_threshold = 0.0f;
    int tracking_mode = 0;
    int tracking_ttl = 60; // NEW
    // NEW: In-frame overlapping-detection merge. When enabled, two motion
    // detections in the SAME frame whose block bounding boxes overlap by at
    // least merge_overlapping_iou (IoU) are fused into one BEFORE tracking, so
    // a single person that fragments into 2+ blobs does not spawn 2+ IDs.
    bool merge_overlapping_enable = false;
    float merge_overlapping_iou = 0.3f;
    // NEW (方案4): post-tracking merge of overlapping TRACKED objects. After
    // association+coasting, two objects are fused (older ID kept) when they still
    // share >= merge_tracked_overlap of the smaller one's blocks, OR both fall
    // inside the same foreground blob. Fixes a person that fragments into 2 IDs
    // (old coasting ID left in place + new ID) stealing momentum from each other.
    bool merge_tracked_enable = false;
    float merge_tracked_overlap = 0.3f;
    // NEW: max centroid distance (grid units) for the same-foreground (criterion b)
    // merge. Two objects in the same FG blob are fused only if also within this
    // distance -- stops a big connected blob from merging things that are far
    // apart (e.g. a bed-head caregiver + something across the frame). 0 = no limit.
    float merge_tracked_max_dist = 0.0f;
    // NEW: use whole-object foreground area (fg_area) instead of block-local
    // pixel_count for the Case5 trigger-area gate and the still-lying persistence
    // gate. fg_area is much larger (whole blob vs moving blocks), so its own
    // thresholds are needed -- tune these. Default OFF = keep block-local behavior.
    bool use_fg_area = true;           // fg_area for the STILL-LYING persistence gate (default ON)
    int min_trigger_fg_area = 12000;   // trigger-gate threshold when use_fg_area_trigger
    int still_lying_fg_area = 2000;    // replaces the 1000 block-local still-lying gate
    // The TRIGGER gate ("is the moving region big enough") is separate: it defaults to
    // block-local pixel_count so a tiny 2-block remnant whose fg_area got inflated by a
    // nearby blob can't open an observation. Turn this on to use fg_area there too.
    bool use_fg_area_trigger = false;
    // Restore the Kalman Predict() step that a refactor dropped from mode-3/4 tracking
    // (+ anchor the coasting centroid to its blocks). Without it the filter's covariance
    // collapses, its gain -> 0, and the "predicted" track position freezes/lags behind a
    // moving object -> association fails and a continuing object spawns a new ID (data14
    // id-split). Default ON: validated on data4~17 (95.8%/FP1 vs old 91.7%/FP2). Set
    // false to revert to the old (pre-fix) tracking.
    bool enable_kalman_predict = true;
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
    int face_detect_interval_frames = 1; // NEW: Control face detection frequency
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
    float bed_update_alpha_multiplier = 4.0f; // NEW: Fast Bed Update Multiplier


    // Optical Flow Params
    int opt_flow_frame_distance = 3;
    int perspective_point_x = 416;
    int perspective_point_y = 474;

    // Area Override Filter
    int min_trigger_area = 2000; // Base min area (pixels) for mid-frame zone; scaled by Y position
    float bed_pixel_ratio_threshold = 0.3f; // NEW
    int momentum_calc_type = 0; // 0: Average (Default), 1: Max Block

    // Post-Bed-Exit Threshold Adjustment
    bool enable_post_bed_exit_threshold = false;
    float post_bed_exit_threshold_multiplier = 0.7f;
    
    // Projection Point Selection
    bool projection_use_foreground = false;
    
    // NEW: Edge Drop Filter
    bool enable_edge_drop_filter = false;
    
    // NEW: Control fall & bed exit detection logic
    bool enable_fall_and_bed_exit = true;
    // NEW: anti-ghost BG protection. APPENDED at the struct end (not inserted mid-
    // struct) so existing field offsets stay ABI-compatible with older callers. A
    // grid block covered by foreground is protected from BG absorption for up to
    // bg_protect_max_frames consecutive frames; 0 = disabled. Only read from
    // header.version >= 2 callers (see SetConfig).
    int bg_protect_max_frames = 0;
    int bg_protect_min_fg = 20;
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

// Event-triggered raw frame recording (mmap ring buffer on SD card).
// When enabled, every frame fed via SetInputMemory/ProcessNextFrame is spooled
// to an on-disk ring; on a fall / bed-exit event the SDK saves
// [event - pre_frames, event + post_frames] as a standalone .raw file, plus a
// background snapshot, a .meta.json and an append-only event_record.jsonl log.
struct EventRecording_v1 {
    ConfigHeader header;
    bool enable = false;
    int pre_frames = 300;    // frames kept before the event
    int post_frames = 300;   // frames recorded after the event
    // PC analysis-sidecar mode (only honored by builds compiled with
    // -DEVENT_RECORDER_PC_ANALYSIS=1, i.e. makefile2 / the PC .so). When set,
    // the whole-session per-frame analysis is written to
    // <pc_output_base>.analysis.json (+ <pc_output_base>.meta.json), next to
    // the input .gray, instead of recording per-event clips to the SD card.
    // Ignored entirely by the edge/SD build (makefile2_edge). Not owned by the
    // SDK: the caller must keep the string alive across SetConfig().
    const char* pc_output_base = nullptr;
};

struct Image {
    unsigned char* data;
    int width;
    int height;
    int channels;
    uint64_t timestamp; // Timestamp in milliseconds (ms since epoch or monotonic)
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

struct ObjectFeatures {
    int id = -1;
    int area;          // 面積 (pixel count)
    float cx, cy;      // 重心
    float angle;       // 角度
    float major, minor;// 長短軸
    int min_x = 0, min_y = 0, max_x = 0, max_y = 0;  // NEW: pixel bbox (precomputed at blob detection)
    std::vector<int> pixels; // 像素索引
    
    // NEW for Optical Flow
    std::vector<float> pixel_dx;
    std::vector<float> pixel_dy;
    std::vector<float> pixel_dir;
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
    
    // NEW: Projection Points for Visualization 
    float proj_top_x = 0.0f;
    float proj_top_y = 0.0f;
    float proj_bot_x = 0.0f;
    float proj_bot_y = 0.0f;
    bool has_projection = false;    
    
    // NEW: Source Image Points for Projection Validation
    float img_top_x = 0.0f;
    float img_top_y = 0.0f;
    float img_bot_x = 0.0f;
    float img_bot_y = 0.0f;
    
    // NEW for Pixel-Based Fall Detection
    int pixel_count = 0;
    float avg_brightness = 0.0f;
    
    // NEW for Stability Analysis
    float direction_variance = 0.0f; // Std Dev of Block Angles (Radians)
    float magnitude_variance = 0.0f; // Std Dev of Block Speeds

    // NEW: Global Frame Stats
    int total_frame_pixel_count = 0;
    
    // NEW: Debug info for visualization
    int matched_fg_obj_id = -1; // ID of the FullFrameObject used for perspective check
    float matched_fg_dist = -1.0f; // NEW: Squared distance to matched FG object
    bool is_in_observation_mode = false; // NEW: True if currently in Case 5 observation (or waiting)
    bool is_fall_this_frame = false;     // NEW: True if THIS object is being reported as a fall this frame
    int fg_area = 0;                     // NEW: actual pixel count of the matched foreground blob
                                         // (whole object), vs pixel_count = block-local FG only
    bool is_coasting = false;            // NEW: this object is a coasting/predicted remnant this
                                         // frame (kept alive by persistence, not a fresh detection)
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
    
    // Release SDK resources explicitly
    StatusCode Release();
    
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

    /**
     * @brief Store IR/Thermal camera intrinsics and extrinsics in the SDK
     *        for use with MapROI(). Call once before using MapROI().
     */
    void SetFusionCameraParams(const FusionParams& params);

    /**
     * @brief Map an IR ROI (x, y, w, h) to the corresponding bounding box
     *        in the thermal image, using params stored by SetFusionCameraParams().
     *
     * @param ir_x   ROI left edge in IR image (pixels)
     * @param ir_y   ROI top edge  in IR image (pixels)
     * @param ir_w   ROI width  in IR image (pixels)
     * @param ir_h   ROI height in IR image (pixels)
     * @param out_x  Output: left edge in thermal image
     * @param out_y  Output: top  edge in thermal image
     * @param out_w  Output: width  in thermal image
     * @param out_h  Output: height in thermal image
     * @return StatusCode::OK, or ERROR_INVALID_INPUT if params not set yet.
     */
    StatusCode MapROI(float ir_x, float ir_y, float ir_w, float ir_h,
                      float& out_x, float& out_y, float& out_w, float& out_h);


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
    StatusCode SetInputMemory(unsigned char* buffer, int width, int height, int channels, uint64_t timestamp = 0);

    /**
     * @brief Set the Background Image explicitly.
     * Takes a deep copy of the buffer.
     * 
     * @param buffer Pointer to the image data.
     * @param width Image width.
     * @param height Image height.
     * @param channels Image channels.
     * @return StatusCode 
     */
    StatusCode SetBackground(const unsigned char* buffer, int width, int height, int channels);

    /**
     * @brief Get the internal background image as a flattened buffer.
     * The output will be a 1-channel grayscale image data array (width * height).
     */
    void GetBackgroundImage(std::vector<uint8_t>& out_bg) const;

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
    // std::vector<MotionObject> GetMotionObjects(); // Deprecated: Return by value causes ABI issues
    void GetMotionObjects(std::vector<MotionObject>& out_objects); // NEW: Pass by ref
    std::vector<uint8_t> GetChangedBlocks(); // Returns mask data same size as grid (cols*rows)
    std::vector<MotionVector> GetMotionVectors(); // Returns vector map same size as grid

    // Get full-frame foreground objects
    std::vector<ObjectFeatures> GetFullFrameObjects();




private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace VisionSDK

#endif // HERMESII_SDK_H
