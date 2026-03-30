#include "HermesII_sdk.h"
#include "ai/model_runner.h"
#include "fusion/image_fusion.h"
#include "fall/fall_detector.h"
#include "Image.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <thread>

// Note: Do not wrap entire file in namespace VisionSDK
// to avoid "VisionSDK::VisionSDK::" confusion if using prefix.

namespace VisionSDK {
    // Define Impl inside namespace
    class VisionSDK::Impl {
    public:
        ModelRunner model_runner;
        ImageFusion image_fusion;
        FallDetector fall_detector;

        InternalConfig config;
        
        // Shared Memory Input
        unsigned char* input_buffer = nullptr;
        int input_width = 0;
        int input_height = 0;
        int input_channels = 0;
        uint64_t input_timestamp = 0;

        // Stored fusion params (for MapROI)
        FusionParams stored_fusion_params;
        bool has_stored_fusion_params = false;
    };
}

using namespace VisionSDK;

// Constructor/Destructor
VisionSDK::VisionSDK::VisionSDK() : pImpl(std::unique_ptr<Impl>(new Impl())) {}
VisionSDK::VisionSDK::~VisionSDK() = default;

#define VISION_SDK_VERSION_INTERNAL "2.0.2"

const char* VisionSDK::VisionSDK::GetVersion() {
    return VISION_SDK_VERSION_INTERNAL;
}

StatusCode VisionSDK::VisionSDK::Init(const std::string& model_path, int num_threads) {
    // ---- [sdk.ini] Read debug logging config ----
    {
        bool save_txt = false;
        std::ifstream ini_file("sdk.ini");
        if (ini_file.is_open()) {
            std::string line;
            while (std::getline(ini_file, line)) {
                // Strip spaces around '='
                auto pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key = line.substr(0, pos);
                    std::string val = line.substr(pos + 1);
                    // Trim whitespace
                    key.erase(0, key.find_first_not_of(" \t\r\n"));
                    key.erase(key.find_last_not_of(" \t\r\n") + 1);
                    val.erase(0, val.find_first_not_of(" \t\r\n"));
                    val.erase(val.find_last_not_of(" \t\r\n") + 1);
                    if (key == "save_txt" && val == "1") {
                        save_txt = true;
                    }
                    if (key == "only_save_raw" && val == "1") {
                        pImpl->config.only_save_raw = true;
                    }
                }
            }
            ini_file.close();
            std::cout << "[SDK] sdk.ini loaded. save_txt=" << (int)save_txt 
                      << " only_save_raw=" << (int)pImpl->config.only_save_raw << std::endl;
        } else {
            std::cout << "[SDK] sdk.ini not found, debug logging disabled." << std::endl;
        }

        if (save_txt) {
            // Build timestamped filename: sdk_fall_debug_YYYYMMDD_HHMMSS.txt
            std::time_t now = std::time(nullptr);
            char timebuf[32];
            std::strftime(timebuf, sizeof(timebuf), "%Y%m%d_%H%M%S", std::localtime(&now));
            std::string log_path = std::string("sdk_fall_debug_") + timebuf + ".txt";
            pImpl->fall_detector.EnableDebugLog(log_path);
        }
    }
    // ---- [sdk.ini] end ----

    pImpl->config.model_path = model_path;
    pImpl->config.num_threads = num_threads;
    
    if (!pImpl->model_runner.Init(model_path)) {
       // Warning
    }
    
    // Pass merged config to FallDetector
    pImpl->fall_detector.SetConfig(pImpl->config);

    std::cout << "VisionSDK Initialized." << std::endl;
    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::SetConfig(const void* config) {
    if (!config) return StatusCode::ERROR_INVALID_INPUT;

    // Use ConfigHeader to identify type and version
    const ConfigHeader* header = static_cast<const ConfigHeader*>(config);
    
    if (header->version != 1) {
        std::cerr << "Error: Unsupported config version: " << header->version << std::endl;
        return StatusCode::ERROR_INVALID_INPUT;
    }

    switch (header->type) {
        case ConfigType::MotionEstimation_v1: {
            const auto* c = static_cast<const MotionEstimation_v1*>(config);
            pImpl->config.grid_cols = c->grid_cols;
            pImpl->config.grid_rows = c->grid_rows;
            pImpl->config.block_size = c->block_size;
            pImpl->config.search_range = c->search_range;
            pImpl->config.history_size = 2;//c->history_size;
            pImpl->config.search_mode = c->search_mode;
            pImpl->config.block_change_threshold = c->block_change_threshold;
            pImpl->config.enable_block_decay = c->enable_block_decay;
            pImpl->config.block_decay_frames = c->block_decay_frames;
            pImpl->config.enable_block_dilation = c->enable_block_dilation;
            pImpl->config.block_dilation_threshold = c->block_dilation_threshold;
            break;
        }
        case ConfigType::ObjectExtraction_v1: {
            const auto* c = static_cast<const ObjectExtraction_v1*>(config);
            pImpl->config.object_extraction_threshold = c->object_extraction_threshold;
            pImpl->config.object_merge_radius = c->object_merge_radius;
            pImpl->config.foreground_merge_radius = c->foreground_merge_radius; // NEW
            pImpl->config.tracking_overlap_threshold = c->tracking_overlap_threshold;
            pImpl->config.tracking_mode = c->tracking_mode;
            pImpl->config.tracking_ttl = c->tracking_ttl; // NEW
            if (pImpl->config.tracking_ttl <= 0) pImpl->config.tracking_ttl = 1000; // Force default if user passed 0
            break;
        }
        case ConfigType::FallDetection_v1: {
            const auto* c = static_cast<const FallDetection_v1*>(config);
            pImpl->config.fall_movement_threshold = c->fall_movement_threshold;
            pImpl->config.fall_strong_threshold = c->fall_strong_threshold;
            pImpl->config.safe_area_ratio_threshold = c->safe_area_ratio_threshold;
            pImpl->config.fall_acceleration_threshold = c->fall_acceleration_threshold;
            pImpl->config.fall_window_size = c->fall_window_size;
            pImpl->config.fall_duration = c->fall_duration;
            pImpl->config.enable_face_detection = true; // V1 defaults to true
            break;
        }
        case ConfigType::FallDetection_v2: {
            const auto* c = static_cast<const FallDetection_v2*>(config);
            pImpl->config.fall_movement_threshold = c->fall_movement_threshold;
            pImpl->config.fall_strong_threshold = c->fall_strong_threshold;
            pImpl->config.safe_area_ratio_threshold = c->safe_area_ratio_threshold;
            pImpl->config.fall_acceleration_threshold = c->fall_acceleration_threshold;
            pImpl->config.fall_window_size = c->fall_window_size;
            pImpl->config.fall_duration = c->fall_duration;
            pImpl->config.enable_face_detection = c->enable_face_detection;
            break;
        }
        case ConfigType::FallDetection_v3: {
            const auto* c = static_cast<const FallDetection_v3*>(config);
            pImpl->config.fall_movement_threshold = c->fall_movement_threshold;
            pImpl->config.fall_strong_threshold = c->fall_strong_threshold;
            pImpl->config.safe_area_ratio_threshold = c->safe_area_ratio_threshold;
            pImpl->config.fall_acceleration_threshold = c->fall_acceleration_threshold;
            pImpl->config.fall_window_size = c->fall_window_size;
            pImpl->config.fall_duration = c->fall_duration;
            pImpl->config.enable_face_detection = c->enable_face_detection;
            pImpl->config.face_detect_interval_frames = 30;//c->face_detect_interval_frames;
            pImpl->config.bg_update_interval_frames = 12;//c->bg_update_interval_frames; //orig is 8
            pImpl->config.bg_update_alpha = 0.08;//c->bg_update_alpha; // orig is 0.1
            pImpl->config.enable_save_bg_mask = c->enable_save_bg_mask;
            pImpl->config.bg_init_start_frame = c->bg_init_start_frame;
            pImpl->config.bg_init_end_frame = c->bg_init_end_frame;
            pImpl->config.bg_diff_threshold = 18;//c->bg_diff_threshold; //18 up
            pImpl->config.fall_acceleration_upper_threshold = c->fall_acceleration_upper_threshold;
            pImpl->config.fall_acceleration_lower_threshold = c->fall_acceleration_lower_threshold;
            pImpl->config.post_fall_distance_threshold = c->post_fall_distance_threshold;
            pImpl->config.post_fall_check_frames = c->post_fall_check_frames;
            pImpl->config.enable_bed_exit_verification = c->enable_bed_exit_verification; // NEW
            pImpl->config.enable_block_shrink_verification = c->enable_block_shrink_verification; // NEW
            pImpl->config.opt_flow_frame_distance = c->opt_flow_frame_distance;
            pImpl->config.perspective_point_x = c->perspective_point_x;
            pImpl->config.perspective_point_y = c->perspective_point_y;
            pImpl->config.min_trigger_area = c->min_trigger_area;
            pImpl->config.bed_update_alpha_multiplier = c->bed_update_alpha_multiplier; // NEW
            pImpl->config.bed_pixel_ratio_threshold = c->bed_pixel_ratio_threshold; // NEW
            pImpl->config.momentum_calc_type = c->momentum_calc_type;
            pImpl->config.enable_post_bed_exit_threshold = c->enable_post_bed_exit_threshold;
            pImpl->config.post_bed_exit_threshold_multiplier = c->post_bed_exit_threshold_multiplier;
            pImpl->config.projection_use_foreground = c->projection_use_foreground;
            pImpl->config.enable_edge_drop_filter = c->enable_edge_drop_filter; // NEW
            break;
        }
        case ConfigType::BedExitDetection_v1: {
            const auto* c = static_cast<const BedExitDetection_v1*>(config);
            pImpl->config.bed_exit_history_len = c->bed_exit_history_len;
            pImpl->config.bed_exit_min_inside_ratio = c->bed_exit_min_inside_ratio;
            pImpl->config.bed_exit_min_outside_ratio = c->bed_exit_min_outside_ratio;
            break;
        }
        case ConfigType::ImageRelated_v1: {
            const auto* c = static_cast<const ImageRelated_v1*>(config);
            pImpl->config.expected_frame_interval_ms = c->expected_frame_interval_ms;
            pImpl->config.frame_interval_tolerance_ms = c->frame_interval_tolerance_ms;
            pImpl->config.enable_draw_bg_noise = c->enable_draw_bg_noise;
            pImpl->config.enable_save_images = c->enable_save_images;
            pImpl->config.save_image_path = c->save_image_path;
            break;
        }
        default:
            return StatusCode::ERROR_INVALID_INPUT;
    }

    // Update FallDetector with new merged config
    pImpl->fall_detector.SetConfig(pImpl->config);

    // ---- [PARAM-DUMP] Print all config after each SetConfig call ----
    const InternalConfig& cfg = pImpl->config;
    printf("[PARAM-DUMP] ===== InternalConfig (type=%d) =====\n", (int)header->type);
    // General
    printf("[PARAM-DUMP]  model_path                        = %s\n", cfg.model_path.c_str());
    printf("[PARAM-DUMP]  num_threads                       = %d\n", cfg.num_threads);
    printf("[PARAM-DUMP]  confidence_threshold              = %.4f\n", cfg.confidence_threshold);
    // Image Related
    printf("[PARAM-DUMP]  expected_frame_interval_ms        = %d\n", cfg.expected_frame_interval_ms);
    printf("[PARAM-DUMP]  frame_interval_tolerance_ms       = %d\n", cfg.frame_interval_tolerance_ms);
    printf("[PARAM-DUMP]  enable_draw_bg_noise              = %d\n", (int)cfg.enable_draw_bg_noise);
    printf("[PARAM-DUMP]  enable_save_images                = %d\n", (int)cfg.enable_save_images);
    printf("[PARAM-DUMP]  save_image_path                   = %s\n", cfg.save_image_path.c_str());
    // Motion Estimation
    printf("[PARAM-DUMP]  grid_cols                         = %d\n", cfg.grid_cols);
    printf("[PARAM-DUMP]  grid_rows                         = %d\n", cfg.grid_rows);
    printf("[PARAM-DUMP]  block_size                        = %d\n", cfg.block_size);
    printf("[PARAM-DUMP]  search_range                      = %d\n", cfg.search_range);
    printf("[PARAM-DUMP]  history_size                      = %d\n", cfg.history_size);
    printf("[PARAM-DUMP]  search_mode                       = %d\n", cfg.search_mode);
    printf("[PARAM-DUMP]  block_change_threshold            = %.4f\n", cfg.block_change_threshold);
    printf("[PARAM-DUMP]  enable_block_decay                = %d\n", (int)cfg.enable_block_decay);
    printf("[PARAM-DUMP]  block_decay_frames                = %d\n", cfg.block_decay_frames);
    printf("[PARAM-DUMP]  enable_block_dilation             = %d\n", (int)cfg.enable_block_dilation);
    printf("[PARAM-DUMP]  block_dilation_threshold          = %d\n", cfg.block_dilation_threshold);
    // Object Extraction
    printf("[PARAM-DUMP]  object_extraction_threshold       = %.4f\n", cfg.object_extraction_threshold);
    printf("[PARAM-DUMP]  object_merge_radius               = %d\n", cfg.object_merge_radius);
    printf("[PARAM-DUMP]  foreground_merge_radius           = %d\n", cfg.foreground_merge_radius);
    printf("[PARAM-DUMP]  tracking_overlap_threshold        = %.4f\n", cfg.tracking_overlap_threshold);
    printf("[PARAM-DUMP]  tracking_mode                     = %d\n", cfg.tracking_mode);
    printf("[PARAM-DUMP]  tracking_ttl                      = %d\n", cfg.tracking_ttl);
    // Fall Detection
    printf("[PARAM-DUMP]  fall_movement_threshold           = %.4f\n", cfg.fall_movement_threshold);
    printf("[PARAM-DUMP]  fall_strong_threshold             = %.4f\n", cfg.fall_strong_threshold);
    printf("[PARAM-DUMP]  safe_area_ratio_threshold         = %.4f\n", cfg.safe_area_ratio_threshold);
    printf("[PARAM-DUMP]  fall_acceleration_threshold       = %.4f\n", cfg.fall_acceleration_threshold);
    printf("[PARAM-DUMP]  fall_acceleration_upper_threshold = %.4f\n", cfg.fall_acceleration_upper_threshold);
    printf("[PARAM-DUMP]  fall_acceleration_lower_threshold = %.4f\n", cfg.fall_acceleration_lower_threshold);
    printf("[PARAM-DUMP]  post_fall_distance_threshold      = %.4f\n", cfg.post_fall_distance_threshold);
    printf("[PARAM-DUMP]  post_fall_check_frames            = %d\n", cfg.post_fall_check_frames);
    printf("[PARAM-DUMP]  fall_window_size                  = %d\n", cfg.fall_window_size);
    printf("[PARAM-DUMP]  fall_duration                     = %d\n", cfg.fall_duration);
    printf("[PARAM-DUMP]  enable_face_detection             = %d\n", (int)cfg.enable_face_detection);
    // Background Update
    printf("[PARAM-DUMP]  bg_update_interval_frames         = %d\n", cfg.bg_update_interval_frames);
    printf("[PARAM-DUMP]  bg_update_alpha                   = %.4f\n", cfg.bg_update_alpha);
    printf("[PARAM-DUMP]  enable_save_bg_mask               = %d\n", (int)cfg.enable_save_bg_mask);
    printf("[PARAM-DUMP]  bg_init_start_frame               = %d\n", cfg.bg_init_start_frame);
    printf("[PARAM-DUMP]  bg_init_end_frame                 = %d\n", cfg.bg_init_end_frame);
    printf("[PARAM-DUMP]  bg_diff_threshold                 = %d\n", cfg.bg_diff_threshold);
    // Bed Exit Detection
    printf("[PARAM-DUMP]  bed_exit_history_len              = %d\n", cfg.bed_exit_history_len);
    printf("[PARAM-DUMP]  bed_exit_min_inside_ratio         = %.4f\n", cfg.bed_exit_min_inside_ratio);
    printf("[PARAM-DUMP]  bed_exit_min_outside_ratio        = %.4f\n", cfg.bed_exit_min_outside_ratio);
    printf("[PARAM-DUMP]  enable_bed_exit_verification      = %d\n", (int)cfg.enable_bed_exit_verification);
    printf("[PARAM-DUMP]  enable_block_shrink_verification  = %d\n", (int)cfg.enable_block_shrink_verification);
    printf("[PARAM-DUMP]  bed_update_alpha_multiplier       = %.4f\n", cfg.bed_update_alpha_multiplier);
    printf("[PARAM-DUMP]  bed_pixel_ratio_threshold         = %.4f\n", cfg.bed_pixel_ratio_threshold);
    // Post-Bed-Exit Threshold Adjustment
    printf("[PARAM-DUMP]  enable_post_bed_exit_threshold    = %d\n", (int)cfg.enable_post_bed_exit_threshold);
    printf("[PARAM-DUMP]  post_bed_exit_threshold_multiplier= %.4f\n", cfg.post_bed_exit_threshold_multiplier);
    printf("[PARAM-DUMP]  post_bed_exit_window_frames       = %d\n", cfg.post_bed_exit_window_frames);
    printf("[PARAM-DUMP]  projection_use_foreground         = %d\n", (int)cfg.projection_use_foreground);
    // Optical Flow / Perspective
    printf("[PARAM-DUMP]  opt_flow_frame_distance           = %d\n", cfg.opt_flow_frame_distance);
    printf("[PARAM-DUMP]  perspective_point_x               = %d\n", cfg.perspective_point_x);
    printf("[PARAM-DUMP]  perspective_point_y               = %d\n", cfg.perspective_point_y);
    // Area / Momentum
    printf("[PARAM-DUMP]  min_trigger_area                  = %d\n", cfg.min_trigger_area);
    printf("[PARAM-DUMP]  momentum_calc_type                = %d\n", cfg.momentum_calc_type);
    printf("[PARAM-DUMP] ==========================================\n");
    // ---- [PARAM-DUMP] end ----

    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::RunInference(const Image& img, std::vector<DetectionResult>& results) {
    if (!pImpl->model_runner.Run(img, results)) {
        return StatusCode::ERROR_INFERENCE_FAILED;
    }
    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::FuseImages(const Image& img1, const Image& img2, Image& output) {
    if (!pImpl->image_fusion.Fuse(img1, img2, output)) {
        return StatusCode::ERROR_FUSION_FAILED;
    }
    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::FuseImages3D(const Image& imgA, const CameraIntrinsics& camA, const CameraExtrinsics& extA,
                        const Image& imgB, const CameraIntrinsics& camB, const CameraExtrinsics& extB,
                        Image& output) {
    if (imgA.width <= 0 || imgB.width <= 0) return StatusCode::ERROR_INVALID_INPUT;
    
    // 1. Compute H that maps B pixel -> A pixel
    ImageFusion::Matrix3x3 H = ImageFusion::ComputeHomographyFromParams(camA, extA, camB, extB);
    
    // 2. Build LUT for warping into B resolution (Scanning B pixels)
    pImpl->image_fusion.BuildWarpLUT(H, imgB.width, imgB.height);
    
    // 3. Alloc output
    output.width = imgB.width;
    output.height = imgB.height;
    output.channels = imgB.channels;
    
    if (!output.data) return StatusCode::ERROR_INVALID_INPUT;
    
    // 4. Warp A -> Output (Aligned with B)
    if (!pImpl->image_fusion.Warp(imgA, output)) {
        return StatusCode::ERROR_FUSION_FAILED;
    }
    
    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::SetInputMemory(unsigned char* buffer, int width, int height, int channels, uint64_t timestamp) {
    if (!buffer) return StatusCode::ERROR_INVALID_INPUT;
    
    pImpl->input_buffer = buffer;
    pImpl->input_width = width;
    pImpl->input_height = height;
    pImpl->input_channels = channels;
    pImpl->input_timestamp = timestamp;

    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::SetBackground(const unsigned char* buffer, int width, int height, int channels) {
    if (!buffer) return StatusCode::ERROR_INVALID_INPUT;

    // Wrap buffer in Image struct
    Image frame; 
    frame.data = (unsigned char*)buffer; // Casting const away for struct compatibility, but SetBackground should treat as read-only/copy
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    frame.timestamp = 0; 

    pImpl->fall_detector.SetBackground(frame);
    return StatusCode::OK;
}

void VisionSDK::VisionSDK::GetBackgroundImage(std::vector<uint8_t>& out_bg) const {
    pImpl->fall_detector.GetBackgroundImage(out_bg);
}

StatusCode VisionSDK::VisionSDK::ProcessNextFrame() {
    if (!pImpl->input_buffer) return StatusCode::ERROR_INVALID_INPUT;

    // Use VisionSDK::Image wrapper (struct)
    Image internal_img; 
    internal_img.data = pImpl->input_buffer;
    internal_img.width = pImpl->input_width;
    internal_img.height = pImpl->input_height;
    internal_img.channels = pImpl->input_channels;
    internal_img.timestamp = pImpl->input_timestamp;
    
    bool is_fall = false;
    
    // Performance Profiling
    auto t0 = std::chrono::high_resolution_clock::now();
    StatusCode ret = pImpl->fall_detector.Detect(internal_img, is_fall);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    long long duration_ms = duration / 1000;
    if (duration_ms < 110) {
        std::this_thread::sleep_for(std::chrono::milliseconds(110 - duration_ms));
    }
    
    //std::cout << "[SDK] Detect() Execution Time: " << duration << " us" << std::endl;

    if (ret != StatusCode::OK) return StatusCode::ERROR_INVALID_INPUT;
    
    return StatusCode::OK;
}

void VisionSDK::VisionSDK::RegisterVisionSDKCallback(VisionSDKCallback callback) {
    pImpl->fall_detector.RegisterCallback(callback);
}

void VisionSDK::VisionSDK::SetBedRegion(const std::vector<std::pair<int, int>>& points) {
    pImpl->fall_detector.SetBedRegion(points);
}

std::vector<std::pair<int, int>> VisionSDK::VisionSDK::GetBedRegion() {
    return pImpl->fall_detector.GetBedRegion();
}

StatusCode VisionSDK::VisionSDK::FuseImagesV2(const Image& img_ir, const Image& img_th, const FusionParams& params, Image& output_fused) {
    if (img_ir.width <= 0 || img_th.width <= 0) return StatusCode::ERROR_INVALID_INPUT;
    
    // Ensure output is allocated if needed, or check size
    // For now assuming caller allocated buffer for output_fused.data
    // If not, we can't allocate inside easily without an allocator interface
    if (!output_fused.data) return StatusCode::ERROR_INVALID_INPUT;
    
    if (!pImpl->image_fusion.FuseV2(img_ir, img_th, params, output_fused)) {
        return StatusCode::ERROR_FUSION_FAILED;
    }
    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::MapPointV2(float ir_x, float ir_y, const FusionParams& params, float& th_x, float& th_y) {
    pImpl->image_fusion.TransformPointV2(ir_x, ir_y, params, th_x, th_y);
    return StatusCode::OK;
}

void VisionSDK::VisionSDK::SetFusionCameraParams(const FusionParams& params) {
    pImpl->stored_fusion_params = params;
    pImpl->has_stored_fusion_params = true;
}

StatusCode VisionSDK::VisionSDK::MapROI(
        float ir_x, float ir_y, float ir_w, float ir_h,
        float& out_x, float& out_y, float& out_w, float& out_h) {
    if (!pImpl->has_stored_fusion_params) {
        std::cerr << "[SDK] MapROI: fusion params not set. Call SetFusionCameraParams() first." << std::endl;
        return StatusCode::ERROR_INVALID_INPUT;
    }
    const FusionParams& p = pImpl->stored_fusion_params;

    // Map 4 corners of the IR ROI to thermal coordinates
    float corners_ir[4][2] = {
        {ir_x,          ir_y         },  // TL
        {ir_x + ir_w,   ir_y         },  // TR
        {ir_x + ir_w,   ir_y + ir_h  },  // BR
        {ir_x,          ir_y + ir_h  }   // BL
    };

    float tx_min =  1e9f, ty_min =  1e9f;
    float tx_max = -1e9f, ty_max = -1e9f;

    for (int i = 0; i < 4; ++i) {
        float tx, ty;
        pImpl->image_fusion.TransformPointV2(corners_ir[i][0], corners_ir[i][1], p, tx, ty);
        if (tx < tx_min) tx_min = tx;
        if (tx > tx_max) tx_max = tx;
        if (ty < ty_min) ty_min = ty;
        if (ty > ty_max) ty_max = ty;
    }

    out_x = tx_min;
    out_y = ty_min;
    out_w = tx_max - tx_min;
    out_h = ty_max - ty_min;
    return StatusCode::OK;
}


// std::vector<MotionObject> VisionSDK::VisionSDK::GetMotionObjects() {
//     return pImpl->fall_detector.GetMotionObjects();
// }

void VisionSDK::VisionSDK::GetMotionObjects(std::vector<MotionObject>& out_objects) {
    out_objects.clear();
    const auto& objs = pImpl->fall_detector.GetMotionObjects();
    out_objects.reserve(objs.size());
    
    bool log_it = false;
    for(const auto& o : objs) {
        if(o.id == 1006) log_it = true;
    }

    if (log_it) {
         printf("[SDK-Wrapper-Ref] Filling Source Size:%zu IDs:", objs.size());
         for(const auto& o : objs) printf(" %d", o.id);
         printf("\n");
    }

    for (const auto& o : objs) {
        out_objects.push_back(o);
        // Keep the mutation for verification? 
        // Let's remove mutation now to see if REAL data flows.
        // If we see 1006 with correct flags, we are good.
        // if (out_objects.back().id == 1006) ...
    }
}

std::vector<uint8_t> VisionSDK::VisionSDK::GetChangedBlocks() {
    return pImpl->fall_detector.GetChangedBlocks();
}

std::vector<MotionVector> VisionSDK::VisionSDK::GetMotionVectors() {
    return pImpl->fall_detector.GetMotionVectors();
}

std::vector<ObjectFeatures> VisionSDK::VisionSDK::GetFullFrameObjects() {
    return pImpl->fall_detector.GetFullFrameObjects();
}
