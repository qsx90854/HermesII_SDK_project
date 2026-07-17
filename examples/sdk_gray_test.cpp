#include "HermesII_sdk.h"
// #include "ffmpeg_precoss.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <map>
#include <cmath>
#include <unistd.h>
#include <cstdio>
#include <cstring>

#ifndef F_OK
#define F_OK 0
#endif
#ifdef __cplusplus
extern "C" {
#endif
int access(const char *pathname, int mode);
#ifdef __cplusplus
}
#endif

// Threadless, atomicless Simple Video Reader supporting .gray raw files, popen-ffmpeg fallback, and random mock generation
class VideoReaderSimple {
private:
    FILE* file;
    FILE* pipe_cmd;
    size_t frameSize;
    bool is_raw_file;
    bool is_pipe;
    int mock_total_frames;
    int mock_current_frame;

public:
    VideoReaderSimple(const std::string& videoPath, int width, int height, const std::string& pix_fmt) 
        : file(nullptr), pipe_cmd(nullptr), frameSize(0), is_raw_file(false), is_pipe(false), mock_total_frames(0), mock_current_frame(0) {
        
        int channels = (pix_fmt == "gray") ? 1 : 3;
        frameSize = width * height * channels;
        
        // 1. Try to open pre-converted .gray raw file
        std::string rawPath = videoPath;
        size_t dot_pos = rawPath.find_last_of('.');
        if (dot_pos != std::string::npos) {
            rawPath = rawPath.substr(0, dot_pos) + ".gray";
        } else {
            rawPath += ".gray";
        }
        
        if (access(rawPath.c_str(), F_OK) == 0) {
            file = fopen(rawPath.c_str(), "rb");
            if (file) {
                std::cout << "[VideoReader] Found raw file: " << rawPath << ", reading frames directly." << std::endl;
                is_raw_file = true;
                return;
            }
        }
        
        // 2. Try to run ffmpeg if available
        bool has_ffmpeg = (system("which ffmpeg >/dev/null 2>&1") == 0);
        bool mp4_exists = (access(videoPath.c_str(), F_OK) == 0);
        
        if (mp4_exists && has_ffmpeg) {
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "ffmpeg -i \"%s\" -f image2pipe -pix_fmt %s -s %dx%d -vcodec rawvideo - 2>/dev/null",
                     videoPath.c_str(), pix_fmt.c_str(), width, height);
            std::cout << "[VideoReader] Calling popen for: " << cmd << std::endl;
            pipe_cmd = popen(cmd, "r");
            if (pipe_cmd) {
                is_pipe = true;
                return;
            }
        }
        
        // 3. Fallback to random mock generation
        std::cout << "[VideoReader] Raw file or ffmpeg not available. Entering Mock random frame mode." << std::endl;
        mock_total_frames = 800; // Mock 800 frames
        mock_current_frame = 0;
    }

    ~VideoReaderSimple() {
        if (file) {
            fclose(file);
            file = nullptr;
        }
        if (pipe_cmd) {
            pclose(pipe_cmd);
            pipe_cmd = nullptr;
        }
    }

    bool readFrame(std::vector<uint8_t>& frameData) {
        if (frameData.size() != frameSize) {
            frameData.resize(frameSize);
        }
        
        if (is_raw_file && file) {
            size_t bytesRead = fread(frameData.data(), 1, frameSize, file);
            return bytesRead == frameSize;
        } else if (is_pipe && pipe_cmd) {
            size_t bytesRead = fread(frameData.data(), 1, frameSize, pipe_cmd);
            return bytesRead == frameSize;
        } else {
            if (mock_current_frame >= mock_total_frames) {
                return false;
            }
            // Generate random image pixels
            for (size_t i = 0; i < frameSize; i++) {
                frameData[i] = rand() % 256;
            }
            mock_current_frame++;
            return true;
        }
    }
};

struct TestVideo {
    std::string file_path;
    std::string trigger_type;
    int expected_frame = -1;
};

// Simple Config Loader
class SimpleConfig {
    std::map<std::string, std::string> settings;
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line;
        std::string section = "";
        while (std::getline(f, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == ';' || line[first] == '#') continue;
            std::string trimmed = line.substr(first);
            if (trimmed[0] == '[') {
                size_t end = trimmed.find(']');
                if (end != std::string::npos && end > 1) {
                    section = trimmed.substr(1, end - 1);
                }
            } else {
                size_t eq = trimmed.find('=');
                if (eq != std::string::npos) {
                    std::string key = trimmed.substr(0, eq);
                    std::string val = trimmed.substr(eq + 1);
                    // trim key & val
                    size_t k_last = key.find_last_not_of(" \t\r\n");
                    if (k_last != std::string::npos) {
                        key.erase(k_last + 1);
                    }
                    size_t v_first = val.find_first_not_of(" \t\r\n");
                    if (v_first != std::string::npos) {
                        val.erase(0, v_first);
                    }
                    size_t v_last = val.find_last_not_of(" \t\r\n");
                    if (v_last != std::string::npos) {
                        val.erase(v_last + 1);
                    }
                    if (!section.empty()) {
                        settings[section + "." + key] = val;
                    } else {
                        settings[key] = val;
                    }
                }
            }
        }
        return true;
    }
    std::string getStr(const std::string& key, const std::string& def) {
        return (settings.find(key) != settings.end()) ? settings[key] : def;
    }
    int getInt(const std::string& key, int def) {
        if (settings.find(key) != settings.end()) {
            return std::atoi(settings[key].c_str());
        }
        return def;
    }
    float getFloat(const std::string& key, float def) {
        if (settings.find(key) != settings.end()) {
            return (float)std::atof(settings[key].c_str());
        }
        return def;
    }
};

// Helper to clean quotes and commas
std::string cleanStr(const std::string& s) {
    std::string res = "";
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\"' && c != ',' && c != ':') {
            res += c;
        }
    }
    return res;
}

// Robust errorlog_ parser for standard JSON format
std::vector<TestVideo> parseErrorLogSimple(const std::string& filepath, 
                                          std::vector<std::pair<int, int>>& out_bed_points,
                                          int& out_org_w, int& out_org_h,
                                          int& out_sdk_w, int& out_sdk_h) {
    std::ifstream infile(filepath);
    std::vector<TestVideo> videos;
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open error log: " << filepath << std::endl;
        return videos;
    }

    std::string line;
    TestVideo cur_video;

    while (std::getline(infile, line)) {
        if (line.find("\"Bed_Region_point\"") != std::string::npos) {
            size_t open_bracket = line.find('[');
            size_t close_bracket = line.find(']');
            if (open_bracket != std::string::npos && close_bracket != std::string::npos && close_bracket > open_bracket) {
                std::string pts_str = line.substr(open_bracket + 1, close_bracket - open_bracket - 1);
                std::replace(pts_str.begin(), pts_str.end(), ',', ' ');
                std::stringstream ss(pts_str);
                std::vector<int> vals;
                int val;
                while (ss >> val) {
                    vals.push_back(val);
                }
                out_bed_points.clear();
                for (size_t i = 0; i + 1 < vals.size(); i += 2) {
                    out_bed_points.push_back({vals[i], vals[i+1]});
                }
            }
        }
        else if (line.find("\"Video_Width\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                out_org_w = std::atoi(line.substr(colon + 1).c_str());
            }
        }
        else if (line.find("\"Video_Height\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                out_org_h = std::atoi(line.substr(colon + 1).c_str());
            }
        }
        else if (line.find("\"SDK_Width\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                out_sdk_w = std::atoi(line.substr(colon + 1).c_str());
            }
        }
        else if (line.find("\"SDK_Height\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                out_sdk_h = std::atoi(line.substr(colon + 1).c_str());
            }
        }
        else if (line.find("\"file\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                size_t first_quote = line.find('\"', colon);
                if (first_quote != std::string::npos) {
                    size_t second_quote = line.find('\"', first_quote + 1);
                    if (second_quote != std::string::npos && second_quote > first_quote) {
                        std::string f_path = line.substr(first_quote + 1, second_quote - first_quote - 1);
                        if (!f_path.empty() && f_path[0] == '/') {
                            cur_video.file_path = f_path.substr(1);
                        } else {
                            cur_video.file_path = f_path;
                        }
                    }
                }
            }
        }
        else if (line.find("\"trigger\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                size_t first_quote = line.find('\"', colon);
                if (first_quote != std::string::npos) {
                    size_t second_quote = line.find('\"', first_quote + 1);
                    if (second_quote != std::string::npos && second_quote > first_quote) {
                        cur_video.trigger_type = line.substr(first_quote + 1, second_quote - first_quote - 1);
                    }
                }
            }
        }
        else if (line.find("\"trigger_frame\"") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                cur_video.expected_frame = std::atoi(line.substr(colon + 1).c_str());
                videos.push_back(cur_video);
                cur_video = TestVideo();
            }
        }
    }
    infile.close();
    return videos;
}

// Save 1-channel Grayscale Image as standard BMP (bottom-to-top format)
bool saveGrayBMP(const std::string& filename, const uint8_t* data, int width, int height) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout.is_open()) return false;

    // BMP File Header (14 bytes)
    uint8_t fileHeader[14] = {
        'B', 'M', // Signature
        0, 0, 0, 0, // Image file size in bytes
        0, 0, 0, 0, // Reserved
        54, 4, 0, 0 // Start of pixel data (54 header + 1024 palette)
    };

    // BMP Info Header (40 bytes)
    uint8_t infoHeader[40] = {
        40, 0, 0, 0, // Header size
        0, 0, 0, 0, // Width
        0, 0, 0, 0, // Height
        1, 0,       // Planes
        8, 0,       // Bits per pixel (8-bit grayscale)
        0, 0, 0, 0, // Compression (0 = BI_RGB)
        0, 0, 0, 0, // Image size
        0, 0, 0, 0, // X pixels per meter
        0, 0, 0, 0, // Y pixels per meter
        0, 0, 0, 0, // Colors in palette
        0, 0, 0, 0  // Important colors
    };

    int rowStride = (width + 3) & ~3;
    int imageSize = rowStride * height;
    int fileSize = 54 + 1024 + imageSize;

    // Fill fileHeader fields
    fileHeader[2] = (uint8_t)(fileSize);
    fileHeader[3] = (uint8_t)(fileSize >> 8);
    fileHeader[4] = (uint8_t)(fileSize >> 16);
    fileHeader[5] = (uint8_t)(fileSize >> 24);

    fileHeader[10] = (uint8_t)(54 + 1024);
    fileHeader[11] = (uint8_t)((54 + 1024) >> 8);

    // Fill infoHeader fields
    infoHeader[4] = (uint8_t)(width);
    infoHeader[5] = (uint8_t)(width >> 8);
    infoHeader[6] = (uint8_t)(width >> 16);
    infoHeader[7] = (uint8_t)(width >> 24);

    infoHeader[8] = (uint8_t)(height);
    infoHeader[9] = (uint8_t)(height >> 8);
    infoHeader[10] = (uint8_t)(height >> 16);
    infoHeader[11] = (uint8_t)(height >> 24);

    infoHeader[20] = (uint8_t)(imageSize);
    infoHeader[21] = (uint8_t)(imageSize >> 8);
    infoHeader[22] = (uint8_t)(imageSize >> 16);
    infoHeader[23] = (uint8_t)(imageSize >> 24);

    // Write Headers
    fout.write((char*)fileHeader, 14);
    fout.write((char*)infoHeader, 40);

    // Write Grayscale Palette (256 colors * 4 bytes each: B, G, R, A)
    for (int i = 0; i < 256; ++i) {
        uint8_t color[4] = { (uint8_t)i, (uint8_t)i, (uint8_t)i, 0 };
        fout.write((char*)color, 4);
    }

    // Write pixel rows bottom-to-top (as required by BMP)
    std::vector<uint8_t> rowBuffer(rowStride, 0);
    for (int y = height - 1; y >= 0; --y) {
        std::memcpy(rowBuffer.data(), data + y * width, width);
        fout.write((char*)rowBuffer.data(), rowStride);
    }

    fout.close();
    return true;
}

struct CallbackEvent {
    int frame_index;
    float confidence;
    std::string event_type;
};
std::vector<CallbackEvent> g_callback_events;

void onFallDetected(const VisionSDK::VisionSDKEvent& event) {
    if (event.is_fall_detected) {
        g_callback_events.push_back({event.frame_index, event.confidence, "Fall"});
        std::cout << "[Callback] Fall Detected at Frame: " << event.frame_index 
                  << ", Confidence: " << event.confidence << std::endl;
    }
    if (event.is_bed_exit) {
        g_callback_events.push_back({event.frame_index, event.confidence, "BedExit"});
        std::cout << "[Callback] BedExit Detected at Frame: " << event.frame_index 
                  << ", Confidence: " << event.confidence << std::endl;
    }
}

int main() {

    std::cout << "=========================================" << std::endl;
    std::cout << "SDK Single Channel Log Verification Test Start" << std::endl;
    std::cout << "SDK Version: " << VisionSDK::VisionSDK::GetVersion() << std::endl;
    std::cout << "=========================================" << std::endl;

    std::cout << "[DEBUG] Configured global test environments..." << std::endl;

    // Load Configurations from parameter.ini
    SimpleConfig cfg;
    if (cfg.load("parameter.ini")) {
        std::cout << "Loaded parameter.ini successfully." << std::endl;
    } else {
        std::cerr << "Warning: Cannot open parameter.ini, using SDK defaults." << std::endl;
    }

    std::cout << "[DEBUG] Preparing motion/obj/fall/img Config structures..." << std::endl;
    // Configure configurations
    VisionSDK::MotionEstimation_v1 motionCfg;
    motionCfg.header.type = VisionSDK::ConfigType::MotionEstimation_v1;
    motionCfg.header.version = 1;
    motionCfg.grid_cols = cfg.getInt("Motion.Grid_Cols", 12);
    motionCfg.grid_rows = cfg.getInt("Motion.Grid_Rows", 16);
    motionCfg.block_size = 16;
    motionCfg.search_range = 24;
    motionCfg.history_size = cfg.getInt("Motion.Diff_Check_Range", 5);
    motionCfg.block_change_threshold = cfg.getFloat("Motion.Block_Difference_Ratio_Threshold", 0.03f);
    motionCfg.search_mode = cfg.getInt("Motion.Search_Mode", 1);
    motionCfg.enable_block_decay = (cfg.getInt("Motion.Enable_Block_Decay", 1) != 0);
    motionCfg.block_decay_frames = cfg.getInt("Motion.Block_Decay_Frames", 3);
    motionCfg.enable_block_dilation = (cfg.getInt("Motion.Enable_Block_Dilation", 1) != 0);
    motionCfg.block_dilation_threshold = cfg.getInt("Motion.Block_Dilation_Threshold", 2);

    VisionSDK::ObjectExtraction_v1 objCfg;
    objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
    objCfg.header.version = 1;
    objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
    objCfg.foreground_merge_radius = cfg.getInt("Object.Foreground_Merge_Range", 1);
    objCfg.object_extraction_threshold = 2.0f;
    objCfg.tracking_overlap_threshold = cfg.getFloat("Tracking.Tracking_Overlap_Threshold", 0.5f);
    objCfg.tracking_mode = cfg.getInt("Tracking.Tracking_Mode", 1);
    objCfg.tracking_ttl = cfg.getInt("Tracking.Tracking_TTL", 60);

    VisionSDK::FallDetection_v3 fallCfg;
    fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
    fallCfg.header.version = 1;
    fallCfg.fall_movement_threshold = cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 3.0f);
    fallCfg.fall_strong_threshold = cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 8.0f);
    fallCfg.fall_acceleration_threshold = cfg.getFloat("FallDetect.Fall_Detect_Acceleration_Threshold", 5.0f);
    fallCfg.fall_acceleration_upper_threshold = cfg.getFloat("FallDetect.Fall_Detect_Accel_Upper_Threshold", 6.0f);
    fallCfg.fall_acceleration_lower_threshold = cfg.getFloat("FallDetect.Fall_Detect_Accel_Lower_Threshold", -4.0f);
    fallCfg.bed_pixel_ratio_threshold = cfg.getFloat("FallDetect.Fall_Detect_Bed_Pixel_Ratio_Threshold", 0.15f);
    fallCfg.safe_area_ratio_threshold = cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5f);
    fallCfg.fall_window_size = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 30);
    fallCfg.fall_duration = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 5);
    fallCfg.post_fall_distance_threshold = cfg.getFloat("FallDetect.Fall_Detect_Post_Fall_Distance_Threshold", 4.0f);
    fallCfg.post_fall_check_frames = cfg.getInt("FallDetect.Fall_Detect_Post_Fall_Check_Frames", 5);
    fallCfg.momentum_calc_type = cfg.getInt("FallDetect.Fall_Detect_Momentum_Calc_Type", 1);
    fallCfg.enable_face_detection = false; // Disable to avoid NPU crash or other issues
    fallCfg.face_detect_interval_frames = cfg.getInt("FallDetect.Face_Detect_Interval_Frames", 8);
    fallCfg.enable_edge_drop_filter = (cfg.getInt("FallDetect.Enable_Edge_Drop_Filter", 1) != 0);
    fallCfg.enable_bed_exit_verification = true;
    fallCfg.enable_block_shrink_verification = true;
    fallCfg.enable_save_bg_mask = (cfg.getInt("FallDetect.Enable_Save_BG_Mask", 1) != 0);
    fallCfg.bg_init_start_frame = cfg.getInt("FallDetect.BG_Init_Start_Frame", 2);
    fallCfg.bg_init_end_frame = cfg.getInt("FallDetect.BG_Init_End_Frame", 5);
    fallCfg.bg_diff_threshold = cfg.getInt("FallDetect.BG_Diff_Threshold", 18);
    fallCfg.bg_update_interval_frames = cfg.getInt("FallDetect.BG_Update_Interval", 12);
    fallCfg.bg_update_alpha = cfg.getFloat("FallDetect.BG_Update_Alpha", 0.08f);
    fallCfg.bed_update_alpha_multiplier = cfg.getFloat("FallDetect.Bed_Update_Alpha_Multiplier", 8.0f);
    fallCfg.enable_post_bed_exit_threshold = (cfg.getInt("FallDetect.Enable_Post_BedExit_Threshold", 1) != 0);
    fallCfg.post_bed_exit_threshold_multiplier = cfg.getFloat("FallDetect.Post_BedExit_Threshold_Multiplier", 0.7f);
    fallCfg.projection_use_foreground = (cfg.getInt("FallDetect.Projection_Use_Foreground", 0) != 0);
    fallCfg.opt_flow_frame_distance = cfg.getInt("OpticalFlow.CompareFrameDistance", 2);
    fallCfg.perspective_point_x = cfg.getInt("OpticalFlow.PerspectivePointX", 416);
    fallCfg.perspective_point_y = cfg.getInt("OpticalFlow.PerspectivePointY", 474);
    fallCfg.min_trigger_area = cfg.getInt("OpticalFlow.MinTriggerArea", 2000);
    fallCfg.enable_fall_and_bed_exit = true;

    VisionSDK::ImageRelated_v1 imgCfg;
    imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
    imgCfg.header.version = 1;
    imgCfg.enable_save_images = false;
    imgCfg.enable_draw_bg_noise = false;
    imgCfg.expected_frame_interval_ms = cfg.getInt("Validation.Expected_Frame_Interval", 33);
    imgCfg.frame_interval_tolerance_ms = cfg.getInt("Validation.Frame_Interval_Tolerance", 10);

    // Event recording: on fall/bed-exit the SDK saves pre/post raw frames to
    // the SD card ring (see EventRecording_v1 in HermesII_sdk.h).
    VisionSDK::EventRecording_v1 recCfg;
    recCfg.header.type = VisionSDK::ConfigType::EventRecording_v1;
    recCfg.header.version = 1;
    recCfg.enable = (cfg.getInt("EventRecorder.Enable_Event_Record", 1) != 0);
    recCfg.pre_frames = cfg.getInt("EventRecorder.Pre_Frames", 300);
    recCfg.post_frames = cfg.getInt("EventRecorder.Post_Frames", 300);

    std::cout << "[DEBUG] Parsing errorlog_ simple..." << std::endl;
    // Locate and parse errorlog_
    std::string errorlog_path = "Bug_Video/OneDrive_2_2026-7-3/errorlog_";
    std::vector<std::pair<int, int>> bed_points;
    int orgW = 1920, orgH = 1080;
    int W = 800, H = 450;

    std::vector<TestVideo> test_videos = parseErrorLogSimple(errorlog_path, bed_points, orgW, orgH, W, H);
    if (test_videos.empty()) {
        std::cerr << "Error: No video to test or errorlog_ not found!" << std::endl;
        return -1;
    }

    std::cout << "Successfully parsed " << test_videos.size() << " videos from log." << std::endl;
    std::cout << "Resolution: Original " << orgW << "x" << orgH << " -> SDK Target " << W << "x" << H << std::endl;

    VisionSDK::VisionSDK sdk;
    VisionSDK::StatusCode init_status = sdk.Init("models/blaze_face_detect_nnp310_128x128.ty", 4);
    if (init_status != VisionSDK::StatusCode::OK) {
        std::cerr << "Global SDK Initialization Failed!" << std::endl;
        return -1;
    }

    std::cout << "[DEBUG] Opening sdk_verification_report.txt..." << std::endl;
    // Open output report file
    std::ofstream report("sdk_verification_report.txt");
    if (!report.is_open()) {
        std::cerr << "Error: Cannot open report output file!" << std::endl;
        return -1;
    }

    report << "========================================================\n";
    report << "            HermesII SDK Verification Report            \n";
    report << "========================================================\n\n";

    int total_processed = 0;
    int total_matched = 0;

    sdk.RegisterVisionSDKCallback(onFallDetected);

        // Set bed region directly without scaling as json contains target 800x450 points
        if (!bed_points.empty()) {
            sdk.SetBedRegion(bed_points);
        }

        // Apply Configurations (Resetting frame index inside SDK implementation)
        sdk.SetConfig(&motionCfg);
        sdk.SetConfig(&objCfg);
        sdk.SetConfig(&fallCfg);
        sdk.SetConfig(&imgCfg);
        sdk.SetConfig(&recCfg);

    for (const auto& tv : test_videos) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Testing Video: " << tv.file_path << std::endl;
        std::cout << "Expected Frame: " << tv.expected_frame << std::endl;

        report << "Video Path (Log): " << tv.file_path << "\n";
        report << "Expected Event  : " << tv.trigger_type << " at frame " << tv.expected_frame << "\n";
        report.flush(); // Flush before processing so we know which video was in-flight if it crashes mid-loop.

        bool file_exists = (access(tv.file_path.c_str(), F_OK) == 0);
        std::string raw_path = tv.file_path;
        size_t dot_pos = raw_path.find_last_of('.');
        if (dot_pos != std::string::npos) {
            raw_path = raw_path.substr(0, dot_pos) + ".gray";
        } else {
            raw_path += ".gray";
        }
        bool raw_exists = (access(raw_path.c_str(), F_OK) == 0);

        if (!file_exists && !raw_exists) {
            std::cout << "[Verification] Video file " << tv.file_path << " and raw file " << raw_path 
                      << " not found. Mock frame simulation will be used." << std::endl;
        }

        total_processed++;

        

        // Clear Callback State
        g_callback_events.clear();
        std::cout << "[DEBUG-STEP] g_callback_events cleared." << std::endl;
        // Process frames
        bool run_success = true;
        try {
            std::cout << "[DEBUG-STEP] Declaring sdk_buffer size=" << (W * H * 1) << "..." << std::endl;
            std::vector<uint8_t> sdk_buffer(W * H * 1);
            
            std::cout << "[DEBUG-STEP] Creating VideoReader for: " << tv.file_path << "..." << std::endl;
            VideoReaderSimple reader(tv.file_path, W, H, "gray");
            
            int frame_count = 0;
            int frame_interval = 8;
            std::cout << "[DEBUG-STEP] Entering readFrame loop (frame_interval = " << frame_interval << ")..." << std::endl;
            while (reader.readFrame(sdk_buffer)) 
            {
                if (frame_count % frame_interval == 0) {
                    // Set input memory for every sampled frame to update buffer pointer and timestamp (33ms interval)
                    uint64_t timestamp = (uint64_t)frame_count * 33;
                    sdk.SetInputMemory(sdk_buffer.data(), W, H, 1, timestamp);
                    
                    VisionSDK::StatusCode frame_status = sdk.ProcessNextFrame();
                    if (frame_status != VisionSDK::StatusCode::OK) {
                        std::cout << "[Warning] Frame " << frame_count 
                                  << " ProcessNextFrame returned error code: " << (int)frame_status << std::endl;
                    }
                }
                frame_count++;
                if (frame_count % 300 == 0) {
                    std::cout << "[DEBUG-STEP] Frame " << frame_count << " read." << std::endl;
                }

                // Retrieve and save background BMP every 100 processed frames
                // if (frame_count % 100 == 0) 
                // {
                //     std::vector<uint8_t> bg_img;
                //     sdk.GetBackgroundImage(bg_img);
                //     if (!bg_img.empty() && bg_img.size() == (size_t)(W * H)) {
                //         char bmp_name[256];
                //         std::string video_name = tv.file_path;
                //         size_t slash_pos = video_name.find_last_of('/');
                //         if (slash_pos != std::string::npos) {
                //             video_name = video_name.substr(slash_pos + 1);
                //         }
                //         size_t dot_pos = video_name.find_last_of('.');
                //         if (dot_pos != std::string::npos) {
                //             video_name = video_name.substr(0, dot_pos);
                //         }
                //         snprintf(bmp_name, sizeof(bmp_name), "bg_%s_frame_%d.bmp", video_name.c_str(), frame_count);
                //         if (saveGrayBMP(bmp_name, bg_img.data(), W, H)) {
                //             std::cout << "[Verification] Saved background BMP: " << bmp_name << std::endl;
                //         } else {
                //             std::cerr << "[Error] Failed to save BMP: " << bmp_name << std::endl;
                //         }
                //     }
                // }
            }
            std::cout << "[DEBUG-STEP] Loop finished. Processed " << frame_count << " frames." << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Exception reading video: " << e.what() << std::endl;
            sdk.Release();
            run_success = false;
        }

        // Compare callback events
        bool match_success = false;
        int closest_frame_diff = 99999;
        int matched_sdk_frame = -1;
        float matched_confidence = 0.0f;

        report << "SDK Callbacks   :\n";
        if (run_success) {
            if (g_callback_events.empty()) {
                report << "  - (No events detected by SDK)\n";
            } else {
                for (const auto& ev : g_callback_events) {
                    report << "  - Frame " << ev.frame_index << " | Event: " << ev.event_type 
                           << " (Confidence: " << ev.confidence << ")\n";
                    
                    // Normalize types for comparison (e.g. "Fall"/"fall", "BedExit"/"bed_exit")
                    std::string norm_ev = ev.event_type;
                    std::string norm_tr = tv.trigger_type;
                    std::transform(norm_ev.begin(), norm_ev.end(), norm_ev.begin(), ::tolower);
                    std::transform(norm_tr.begin(), norm_tr.end(), norm_tr.begin(), ::tolower);
                    if (norm_ev.find("exit") != std::string::npos) norm_ev = "bed_exit";
                    if (norm_tr.find("exit") != std::string::npos) norm_tr = "bed_exit";
                    if (norm_ev.find("fall") != std::string::npos) norm_ev = "fall";
                    if (norm_tr.find("fall") != std::string::npos) norm_tr = "fall";
                    
                    if (norm_ev == norm_tr) {
                        int diff = std::abs(ev.frame_index - tv.expected_frame);
                        if (diff < closest_frame_diff) {
                            closest_frame_diff = diff;
                            matched_sdk_frame = ev.frame_index;
                            matched_confidence = ev.confidence;
                        }
                    }
                }
                if (closest_frame_diff <= 50) {
                    match_success = true;
                }
            }
        } else {
            report << "  - (SDK run failed during video decoding)\n";
        }

        if (match_success) {
            total_matched++;
            std::cout << "Match Status: SUCCESS (Closest matched frame: " << matched_sdk_frame 
                      << ", Diff: " << closest_frame_diff << " frames)" << std::endl;
            report << "Closest Match   : SDK Frame " << matched_sdk_frame 
                   << " (Diff: " << closest_frame_diff << " frames, Confidence: " << matched_confidence << ")\n";
            report << "Status          : SUCCESS (Matched within 50 frames)\n";
        } else {
            std::cout << "Match Status: FAILED" << std::endl;
            if (matched_sdk_frame != -1) {
                report << "Closest Match   : SDK Frame " << matched_sdk_frame 
                       << " (Diff: " << closest_frame_diff << " frames, Confidence: " << matched_confidence << ")\n";
            }
            report << "Status          : FAILED\n";
        }
        report << "----------------------------------------\n\n";
        report.flush(); // Flush per video so partial results survive a mid-run crash/OOM.
    }

    std::cout << "\n========================================\n";
    std::cout << "Verification Completed." << std::endl;
    std::cout << "Total Videos Checked : " << total_processed << std::endl;
    std::cout << "Successfully Matched : " << total_matched << std::endl;
    std::cout << "Results saved in sdk_verification_report.txt" << std::endl;
    std::cout << "========================================\n" << std::endl;

    report << "========================================================\n";
    report << "Summary:\n";
    report << "  Total Videos Checked : " << total_processed << "\n";
    report << "  Successfully Matched : " << total_matched << "\n";
    report << "  Match Accuracy       : " 
           << (total_processed > 0 ? (float)total_matched / total_processed * 100.0f : 0.0f) << "%\n";
    report << "========================================================\n";

    report.close();
    return 0;
}
