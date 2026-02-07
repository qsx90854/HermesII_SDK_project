#include "HermesII_sdk.h"
#include "ai/model_runner.h"
#include "fusion/image_fusion.h"
#include "fall/fall_detector.h"
#include "Image.h"
#include <iostream>

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
    };
}

using namespace VisionSDK;

// Constructor/Destructor
VisionSDK::VisionSDK::VisionSDK() : pImpl(std::unique_ptr<Impl>(new Impl())) {}
VisionSDK::VisionSDK::~VisionSDK() = default;

#define VISION_SDK_VERSION_INTERNAL "1.0.1a"

const char* VisionSDK::VisionSDK::GetVersion() {
    return VISION_SDK_VERSION_INTERNAL;
}

StatusCode VisionSDK::VisionSDK::Init(const std::string& model_path, int num_threads) {
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
            pImpl->config.history_size = c->history_size;
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
            pImpl->config.tracking_overlap_threshold = c->tracking_overlap_threshold;
            pImpl->config.tracking_mode = c->tracking_mode;
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
            pImpl->config.bg_update_interval_frames = c->bg_update_interval_frames;
            pImpl->config.bg_update_alpha = c->bg_update_alpha;
            pImpl->config.enable_save_bg_mask = c->enable_save_bg_mask;
            pImpl->config.bg_init_start_frame = c->bg_init_start_frame;
            pImpl->config.bg_init_end_frame = c->bg_init_end_frame;
            pImpl->config.bg_diff_threshold = c->bg_diff_threshold;
            pImpl->config.fall_acceleration_upper_threshold = c->fall_acceleration_upper_threshold;
            pImpl->config.fall_acceleration_lower_threshold = c->fall_acceleration_lower_threshold;
            pImpl->config.post_fall_distance_threshold = c->post_fall_distance_threshold;
            pImpl->config.post_fall_check_frames = c->post_fall_check_frames;
            pImpl->config.enable_bed_exit_verification = c->enable_bed_exit_verification; // NEW
            pImpl->config.enable_block_shrink_verification = c->enable_block_shrink_verification; // NEW
            pImpl->config.opt_flow_frame_distance = c->opt_flow_frame_distance;
            pImpl->config.perspective_point_x = c->perspective_point_x;
            pImpl->config.perspective_point_y = c->perspective_point_y;
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

StatusCode VisionSDK::VisionSDK::SetInputMemory(unsigned char* buffer, int width, int height, int channels) {
    if (!buffer) return StatusCode::ERROR_INVALID_INPUT;
    
    pImpl->input_buffer = buffer;
    pImpl->input_width = width;
    pImpl->input_height = height;
    pImpl->input_channels = channels;

    // Wrap buffer in Image struct
    Image frame; 
    frame.data = pImpl->input_buffer;
    frame.width = pImpl->input_width;
    frame.height = pImpl->input_height;
    frame.channels = pImpl->input_channels;
    frame.timestamp = 0; 
    
    // Debug Print
    // printf("[VisionSDK::SetInputMemory] Input buffer=%p, w=%d, h=%d, c=%d\n", buffer, width, height, channels);

    return StatusCode::OK;
}

StatusCode VisionSDK::VisionSDK::ProcessNextFrame() {
    if (!pImpl->input_buffer) return StatusCode::ERROR_INVALID_INPUT;

    // Use VisionSDK::Image wrapper (struct)
    Image internal_img; 
    internal_img.data = pImpl->input_buffer;
    internal_img.width = pImpl->input_width;
    internal_img.height = pImpl->input_height;
    internal_img.channels = pImpl->input_channels;
    internal_img.timestamp = 0;
    
    bool is_fall = false;
    StatusCode ret = pImpl->fall_detector.Detect(internal_img, is_fall);
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


std::vector<MotionObject> VisionSDK::VisionSDK::GetMotionObjects() {
    return pImpl->fall_detector.GetMotionObjects();
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
