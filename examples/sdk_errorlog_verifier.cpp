#include "HermesII_sdk.h"
#include "ffmpeg_precoss.h"
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
#include <map>
#include <sstream>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cctype>

using namespace VisionSDK;

// ==========================================
// Config Loader (Simple INI Parser)
// ==========================================
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
        std::cout << "DEBUG: ConfigLoader::load opening " << filename << std::endl;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "DEBUG: ConfigLoader::load failed to open " << filename << std::endl;
            return false;
        }
        std::cout << "DEBUG: ConfigLoader::load opened " << filename << " successfully." << std::endl;
        
        std::string line, section;
        int line_num = 0;
        while (std::getline(file, line)) {
            line_num++;
            std::cout << "DEBUG: ConfigLoader::load reading line " << line_num << " (raw size=" << line.size() << ")" << std::endl;
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
                    std::string key = "";
                    if (eq > 0) {
                        key = trim(line.substr(0, eq));
                    }
                    std::string val = "";
                    if (eq + 1 < line.size()) {
                        val = trim(line.substr(eq + 1));
                    }
                    if (!key.empty()) {
                        if (!section.empty()) key = section + "." + key;
                        data[key] = val;
                    }
                }
            }
        }
        std::cout << "DEBUG: ConfigLoader::load finished reading " << filename << std::endl;
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

// ==========================================
// Callback for gathering fall detection events
// ==========================================
struct CallbackEvent {
    int frame_index;
    float confidence;
};
std::vector<CallbackEvent> g_callback_events;

void onFallDetected(const VisionSDK::VisionSDKEvent& event) {
    if (event.is_fall_detected) {
        g_callback_events.push_back({event.frame_index, event.confidence});
    }
}

// Global parsed configurations from errorlog_
std::vector<std::pair<int, int>> g_bed_points;
int g_video_width = 1920;
int g_video_height = 1080;
int g_sdk_width = 850;
int g_sdk_height = 450;

// ==========================================
// Helper functions for parsing errorlog_
// ==========================================
struct VideoVerifyInfo {
    std::string file_path;
    std::string trigger_type;
    int errorlog_frame = -1;
    bool has_open = false;
    bool has_closed = false;
};

std::string extractStringValue(const std::string& line, const std::string& key) {
    size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) return "";
    size_t colon_pos = line.find(":", key_pos);
    if (colon_pos == std::string::npos) {
        colon_pos = line.find("=", key_pos);
    }
    if (colon_pos == std::string::npos) return "";
    
    size_t first_quote = line.find("\"", colon_pos);
    if (first_quote == std::string::npos) return "";
    size_t second_quote = line.find("\"", first_quote + 1);
    if (second_quote == std::string::npos) return "";
    return line.substr(first_quote + 1, second_quote - first_quote - 1);
}

int extractIntValue(const std::string& line, const std::string& key) {
    size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) return -1;
    
    size_t sep_pos = line.find(":", key_pos);
    if (sep_pos == std::string::npos) {
        sep_pos = line.find("=", key_pos);
    }
    if (sep_pos == std::string::npos) return -1;
    
    std::string num_str = "";
    for (size_t i = sep_pos + 1; i < line.size(); ++i) {
        if (std::isdigit(line[i])) {
            num_str += line[i];
        } else if (!num_str.empty()) {
            break;
        }
    }
    if (!num_str.empty()) {
        try {
            return std::stoi(num_str);
        } catch (...) {}
    }
    return -1;
}

std::map<std::string, VideoVerifyInfo> parseErrorLog(const std::string& filepath) {
    std::ifstream infile(filepath);
    std::map<std::string, VideoVerifyInfo> video_map;
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open error log file: " << filepath << std::endl;
        return video_map;
    }
    
    std::string line;
    VideoVerifyInfo cur_info;
    bool in_block = false;
    
    while (std::getline(infile, line)) {
        // Parse Bed Region from errorlog_ if present
        if (line.find("Bed_Region_point") != std::string::npos) {
            size_t open_brace = line.find("{");
            size_t close_brace = line.find("}");
            if (open_brace != std::string::npos && close_brace != std::string::npos) {
                std::string pts_str = line.substr(open_brace + 1, close_brace - open_brace - 1);
                std::stringstream ss(pts_str);
                std::vector<int> vals;
                int val;
                while (ss >> val) {
                    vals.push_back(val);
                    if (ss.peek() == ',' || ss.peek() == ' ') {
                        ss.ignore();
                    }
                }
                if (vals.size() == 8) {
                    g_bed_points.clear();
                    for (size_t i = 0; i < 8; i += 2) {
                        g_bed_points.push_back({vals[i], vals[i+1]});
                    }
                    std::cout << "Parsed Bed Region from errorlog: ";
                    for (const auto& pt : g_bed_points) {
                        std::cout << "(" << pt.first << ", " << pt.second << ") ";
                    }
                    std::cout << std::endl;
                }
            }
            continue;
        }

        // Parse resolutions
        if (line.find("Video_Width") != std::string::npos) {
            int val = extractIntValue(line, "Video_Width");
            if (val != -1) g_video_width = val;
            continue;
        }
        if (line.find("Video_Height") != std::string::npos) {
            int val = extractIntValue(line, "Video_Height");
            if (val != -1) g_video_height = val;
            continue;
        }
        if (line.find("SDK_Width") != std::string::npos) {
            int val = extractIntValue(line, "SDK_Width");
            if (val != -1) g_sdk_width = val;
            continue;
        }
        if (line.find("SDK_Height") != std::string::npos) {
            int val = extractIntValue(line, "SDK_Height");
            if (val != -1) g_sdk_height = val;
            continue;
        }

        if (line.find("{") != std::string::npos && line.find(":") == std::string::npos && line.find("\"") == std::string::npos) {
            cur_info = VideoVerifyInfo();
            in_block = true;
            continue;
        }
        if (!in_block) continue;
        
        if (line.find("}") != std::string::npos && line.find(":") == std::string::npos && line.find("\"") == std::string::npos) {
            in_block = false;
            if (!cur_info.file_path.empty()) {
                auto& info = video_map[cur_info.file_path];
                info.file_path = cur_info.file_path;
                if (cur_info.has_open) info.has_open = true;
                if (cur_info.has_closed) info.has_closed = true;
                if (!cur_info.trigger_type.empty()) info.trigger_type = cur_info.trigger_type;
                if (cur_info.errorlog_frame != -1) info.errorlog_frame = cur_info.errorlog_frame;
            }
            continue;
        }
        
        std::string status = extractStringValue(line, "\"status\"");
        if (!status.empty()) {
            if (status == "open") cur_info.has_open = true;
            else if (status == "closed" || status == "close") cur_info.has_closed = true;
        }
        
        std::string file = extractStringValue(line, "\"file\"");
        if (!file.empty()) {
            cur_info.file_path = file;
        }
        
        std::string trigger = extractStringValue(line, "\"trigger\"");
        if (!trigger.empty()) {
            cur_info.trigger_type = trigger;
        }
        
        int frame1 = extractIntValue(line, "\"file_frame_index\"");
        if (frame1 != -1) cur_info.errorlog_frame = frame1;
        
        int frame2 = extractIntValue(line, "\"trigger_file_frame_index\"");
        if (frame2 != -1) cur_info.errorlog_frame = frame2;
    }
    infile.close();
    return video_map;
}

std::vector<std::pair<int, int>> loadBedPoints(const std::string& filename) {
    std::vector<std::pair<int, int>> points;
    std::ifstream file(filename);
    if (!file.is_open()) return points;
    int x, y;
    char comma;
    while (file >> x >> comma >> y) {
        points.push_back({x, y});
    }
    file.close();
    return points;
}

// ==========================================
// Robust File Search
// ==========================================
std::string findVideoFile(const std::string& file_path) {
    char cwd[1024];
    std::string abs_cwd = "";
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        abs_cwd = std::string(cwd);
    }
    
    std::string clean_file_path = file_path;
    if (!clean_file_path.empty() && clean_file_path[0] == '/') {
        clean_file_path = clean_file_path.substr(1);
    }
    
    std::vector<std::string> candidates;
    
    // Construct candidates
    candidates.push_back("Bug_Video/OneDrive_2_2026-7-3/" + clean_file_path);
    candidates.push_back("../Bug_Video/OneDrive_2_2026-7-3/" + clean_file_path);
    candidates.push_back(file_path);
    candidates.push_back(clean_file_path);
    
    size_t last_slash = file_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        std::string filename = file_path.substr(last_slash + 1);
        candidates.push_back("Bug_Video/OneDrive_2_2026-7-3/event/" + filename);
        candidates.push_back("../Bug_Video/OneDrive_2_2026-7-3/event/" + filename);
        candidates.push_back("event/" + filename);
        candidates.push_back("Bug_Video/OneDrive_2_2026-7-3/" + filename);
    }
    
    for (const auto& rel_path : candidates) {
        std::string abs_path = rel_path;
        if (rel_path[0] != '/' && !abs_cwd.empty()) {
            abs_path = abs_cwd + "/" + rel_path;
        }
        
        struct stat st;
        if (stat(abs_path.c_str(), &st) == 0) {
            return abs_path; 
        }
    }
    return "";
}

std::string findErrorLog(const std::string& filename) {
    std::cout << "DEBUG: findErrorLog - allocating cwd buffer" << std::endl;
    char cwd[1024];
    std::cout << "DEBUG: findErrorLog - calling getcwd" << std::endl;
    std::string abs_cwd = "";
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "DEBUG: findErrorLog - getcwd returned: " << cwd << std::endl;
        abs_cwd = std::string(cwd);
    } else {
        std::cout << "DEBUG: findErrorLog - getcwd returned nullptr" << std::endl;
    }
    
    std::cout << "DEBUG: findErrorLog - building candidates vector" << std::endl;
    std::vector<std::string> candidates;
    
    std::cout << "DEBUG: findErrorLog - push_back 1 (filename)" << std::endl;
    candidates.push_back(filename);
    
    std::cout << "DEBUG: findErrorLog - push_back 2" << std::endl;
    std::string p2 = "Bug_Video/OneDrive_2_2026-7-3/";
    p2 += filename;
    candidates.push_back(p2);
    
    std::cout << "DEBUG: findErrorLog - push_back 3" << std::endl;
    std::string p3 = "../Bug_Video/OneDrive_2_2026-7-3/";
    p3 += filename;
    candidates.push_back(p3);
    
    std::cout << "DEBUG: findErrorLog - push_back 4" << std::endl;
    std::string p4 = "/home/mark/nn_release/HermesII_SDK/Bug_Video/OneDrive_2_2026-7-3/";
    p4 += filename;
    candidates.push_back(p4);
    
    std::cout << "DEBUG: findErrorLog - push_back 5" << std::endl;
    std::string p5 = "/nfs/HermesII_SDK/Bug_Video/OneDrive_2_2026-7-3/";
    p5 += filename;
    candidates.push_back(p5);
    
    std::cout << "DEBUG: findErrorLog - candidates vector built. Count = " << candidates.size() << std::endl;
    
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& rel_path = candidates[idx];
        std::cout << "DEBUG: findErrorLog - checking candidate index " << idx << ": " << rel_path << std::endl;
        std::string abs_path = rel_path;
        if (rel_path[0] != '/' && !abs_cwd.empty()) {
            abs_path = abs_cwd + "/";
            abs_path += rel_path;
        }
        std::cout << "DEBUG: findErrorLog - calling stat on: " << abs_path << std::endl;
        struct stat st;
        if (stat(abs_path.c_str(), &st) == 0) {
            std::cout << "DEBUG: findErrorLog - stat SUCCESS: " << abs_path << std::endl;
            return abs_path;
        }
        std::cout << "DEBUG: findErrorLog - stat failed for: " << abs_path << std::endl;
    }
    std::cout << "DEBUG: findErrorLog - finished, no file found." << std::endl;
    return "";
}

std::string findModelFile(const std::string& model_name) {
    char cwd[1024];
    std::string abs_cwd = "";
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        abs_cwd = std::string(cwd);
    }
    
    std::vector<std::string> candidates;
    candidates.push_back(model_name);
    
    std::string p2 = "res/";
    p2 += model_name;
    candidates.push_back(p2);
    
    std::string p3 = "../res/";
    p3 += model_name;
    candidates.push_back(p3);
    
    std::string p4 = "/home/mark/nn_release/HermesII_SDK/res/";
    p4 += model_name;
    candidates.push_back(p4);
    
    std::string p5 = "/nfs/HermesII_SDK/res/";
    p5 += model_name;
    candidates.push_back(p5);
    
    for (const auto& rel_path : candidates) {
        std::string abs_path = rel_path;
        if (rel_path[0] != '/' && !abs_cwd.empty()) {
            abs_path = abs_cwd + "/";
            abs_path += rel_path;
        }
        struct stat st;
        if (stat(abs_path.c_str(), &st) == 0) {
            return abs_path;
        }
    }
    return "";
}

// ==========================================
// Video Resolution Reader via ffprobe
// ==========================================
std::pair<int, int> getVideoResolution(const std::string& videoPath) {
    std::string cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \""+ videoPath + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {0, 0};
    char buffer[128];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }
    pclose(pipe);
    
    size_t x_pos = result.find('x');
    if (x_pos != std::string::npos) {
        try {
            int w = std::stoi(result.substr(0, x_pos));
            int h = std::stoi(result.substr(x_pos + 1));
            return {w, h};
        } catch (...) {}
    }
    return {0, 0};
}

// ==========================================
// Bilinear / Nearest-Neighbor Resizer & Grayscale Converter
// ==========================================
void resizeAndGray(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH) {
    for (int y = 0; y < dstH; ++y) {
        int srcY = y * srcH / dstH;
        if (srcY >= srcH) srcY = srcH - 1;
        for (int x = 0; x < dstW; ++x) {
            int srcX = x * srcW / dstW;
            if (srcX >= srcW) srcX = srcW - 1;
            
            int dst_idx = y * dstW + x;
            int src_idx = (srcY * srcW + srcX) * 3;
            
            uint8_t r = src[src_idx];
            uint8_t g = src[src_idx + 1];
            uint8_t b = src[src_idx + 2];
            
            dst[dst_idx] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
        }
    }
}

void rgbToGray(const uint8_t* src, uint8_t* dst, int width, int height) {
    int size = width * height;
    for (int i = 0; i < size; ++i) {
        uint8_t r = src[i * 3];
        uint8_t g = src[i * 3 + 1];
        uint8_t b = src[i * 3 + 2];
        dst[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
    }
}

int main(int argc, char** argv) {
    std::cout << "=== HermesII SDK Errorlog Verifier ===" << std::endl;
    
    // Command line options
    bool disable_face = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--disable-face") == 0) {
            disable_face = true;
        }
    }

    // 1. Load Configurations
    ConfigLoader cfg;
    ConfigLoader appCfg;
    
    if (cfg.load("parameter.ini")) {
        std::cout << "Loaded parameter.ini" << std::endl;
    } else {
        std::cerr << "Warning: parameter.ini not found, using SDK defaults." << std::endl;
    }
    
    std::cout << "DEBUG: Loading app_config.ini..." << std::endl;
    appCfg.load("app_config.ini");
    std::cout << "DEBUG: app_config.ini load finished." << std::endl;
    std::string bedFile = appCfg.getString("Demo.Demo_Bed_File", "");
    std::cout << "DEBUG: Demo_Bed_File read: " << bedFile << std::endl;
    
    // 2. Locate and parse errorlog_ file
    std::cout << "DEBUG: Finding error log file..." << std::endl;
    std::string log_file_path = findErrorLog("errorlog_");
    std::cout << "DEBUG: findErrorLog finished. Path: " << log_file_path << std::endl;
    if (log_file_path.empty()) {
        std::cerr << "Error: Cannot locate errorlog_ file!" << std::endl;
        return 1;
    }
    std::cout << "Parsing errorlog from: " << log_file_path << std::endl;
    std::cout << "DEBUG: Parsing error log..." << std::endl;
    auto video_map = parseErrorLog(log_file_path);
    std::cout << "DEBUG: parseErrorLog finished." << std::endl;
    
    // Config values set by parsing errorlog_
    int W = g_sdk_width;
    int H = g_sdk_height;
    int orgW = g_video_width;
    int orgH = g_video_height;
    
    std::cout << "Resolution Parameters parsed from errorlog:\n"
              << "  - Video resolution: " << orgW << "x" << orgH << "\n"
              << "  - SDK target resolution: " << W << "x" << H << std::endl;
    
    // Load Bed points: Prefer errorlog parsed ones, fall back to app_config.ini
    std::vector<std::pair<int, int>> bed_points = g_bed_points;
    if (bed_points.empty()) {
        if (!bedFile.empty()) {
            bed_points = loadBedPoints(bedFile);
            if (!bed_points.empty()) {
                std::cout << "Loaded bed region points from app_config: " << bedFile << std::endl;
            }
        }
    } else {
        std::cout << "Using Bed Region points parsed from errorlog_." << std::endl;
    }
    
    // Find absolute path of face detection model
    std::string model_init_path = "";
    if (disable_face) {
        std::cout << "Face detection is explicitly DISABLED via parameter." << std::endl;
        model_init_path = "disable_face_detection_to_prevent_npu_crash.ty";
    } else {
        std::string abs_model_path = "models/blaze_face_detect_nnp310_128x128.ty";//findModelFile("blaze_face_detect_nnp310_128x128.ty");
        if (abs_model_path.empty()) {
            std::cerr << "Warning: Cannot locate blaze_face_detect_nnp310_128x128.ty model file!" << std::endl;
        } else {
            std::cout << "Using face detection model: " << abs_model_path << std::endl;
            // Diagnostic check
            std::ifstream test_f(abs_model_path, std::ios::binary | std::ios::ate);
            if (test_f.is_open()) {
                std::cout << "DEBUG SUCCESS: C++ std::ifstream successfully opened model. Size: " << test_f.tellg() << " bytes." << std::endl;
                test_f.close();
            }
            model_init_path = abs_model_path;
        }
    }

    // ==========================================
    // Unified Single Process Initialization
    // ==========================================
    // Initializing VisionSDK exactly ONCE outside the loop to follow Fuhan NPU driver binding rules 
    // and avoid memory mapping conflicts and Segmentation Fault (Signal 11) issues.
    std::cout << "Initializing VisionSDK exactly once..." << std::endl;
    VisionSDK::VisionSDK sdk;
    sdk.Init(model_init_path, 4);

    // 3. Open output report file
    std::ofstream report("sdk_verification_report.txt");
    if (!report.is_open()) {
        std::cerr << "Error: Cannot open report output file: sdk_verification_report.txt" << std::endl;
        return 1;
    }
    
    report << "========================================================\n";
    report << "            HermesII SDK Verification Report            \n";
    report << "========================================================\n\n";
    
    int total_processed = 0;
    int total_matched = 0;
    
    for (const auto& pair : video_map) {
        const auto& info = pair.second;
        
        // Check if the video has both open and closed status
        if (!info.has_open || !info.has_closed) {
            std::cout << "Skipping video: " << info.file_path 
                      << " (Does not have both open & closed blocks)" << std::endl;
            continue;
        }
        
        // Robust video file location check
        std::string video_abs_path = findVideoFile(info.file_path);
        
        std::cout << "\n----------------------------------------\n";
        std::cout << "Video Path in Log: " << info.file_path << std::endl;
        if (!video_abs_path.empty()) {
            std::cout << "Resolved Path    : " << video_abs_path << std::endl;
        }
        std::cout << "Expected trigger : " << info.trigger_type 
                  << " at frame: " << info.errorlog_frame << std::endl;
                  
        report << "Video Path (Log): " << info.file_path << "\n";
        if (!video_abs_path.empty()) {
            report << "Resolved Path   : " << video_abs_path << "\n";
        }
        report << "Expected Event  : " << info.trigger_type << " at frame " << info.errorlog_frame << "\n";
        
        if (video_abs_path.empty()) {
            std::cerr << "Error: Video file not found: " << info.file_path << std::endl;
            report << "Status          : FAILED (Video file not found)\n";
            report << "----------------------------------------\n\n";
            continue;
        }
        
        // Check resolution of the video stream
        std::pair<int, int> real_res = getVideoResolution(video_abs_path);
        
        if (real_res.first == 0 || real_res.second == 0) {
            std::cout << "WARNING: Cannot query video resolution via ffprobe for: " << video_abs_path << std::endl;
            std::cout << "  -> Fallback to assuming the video resolution matches the Video_Width parameter (" << orgW << "x" << orgH << ")." << std::endl;
            report << "Real Resolution : Unknown (ffprobe query failed)\n";
        } else {
            std::cout << "Video real resolution: " << real_res.first << "x" << real_res.second << std::endl;
            report << "Real Resolution : " << real_res.first << "x" << real_res.second << "\n";
            
            if (real_res.first == orgW && real_res.second == orgH) {
                std::cout << "Resolution matches video resolution parameter (" << orgW << "x" << orgH << "). Resize to SDK target resolution is required." << std::endl;
            } else if (real_res.first == W && real_res.second == H) {
                std::cout << "Resolution matches SDK target resolution (" << W << "x" << H << "). Directly piping stream to SDK." << std::endl;
            } else {
                std::cerr << "ERROR: Video resolution (" << real_res.first << "x" << real_res.second 
                          << ") matches neither the video resolution parameter (" << orgW << "x" << orgH 
                          << ") nor the SDK target resolution (" << W << "x" << H << ")!" << std::endl;
                report << "Status          : FAILED (Resolution mismatch: " << real_res.first << "x" << real_res.second << ")\n";
                report << "----------------------------------------\n\n";
                continue;
            }
        }
        
        total_processed++;
        
        // =======================================================
        // RE-CONFIGURE SDK FOR THE CURRENT VIDEO
        // =======================================================
        // Calling SetConfig resets absolute_frame_count to 0 inside SDK implementation.
        // This ensures the background update process resets perfectly for each video.
        
        // Configure SDK using parameter.ini
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
        
        VisionSDK::ObjectExtraction_v1 objCfg;
        objCfg.header.type = VisionSDK::ConfigType::ObjectExtraction_v1;
        objCfg.header.version = 1;
        objCfg.object_merge_radius = cfg.getInt("Object.Block_Merge_Range", 3);
        objCfg.foreground_merge_radius = cfg.getInt("Object.Foreground_Merge_Range", 1);
        objCfg.object_extraction_threshold = 2.0f;
        objCfg.tracking_overlap_threshold = cfg.getFloat("Tracking.Tracking_Overlap_Threshold", 0.5f);
        objCfg.tracking_mode = cfg.getInt("Tracking.Tracking_Mode", 1);
        objCfg.tracking_ttl = cfg.getInt("Tracking.Tracking_TTL", 60);
        sdk.SetConfig(&objCfg);
        
        VisionSDK::FallDetection_v3 fallCfg;
        fallCfg.header.type = VisionSDK::ConfigType::FallDetection_v3;
        fallCfg.header.version = 1;
        fallCfg.fall_movement_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 3.0);
        fallCfg.fall_strong_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Strong_Strength", 8.0);
        fallCfg.fall_acceleration_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Acceleration_Threshold", 5.0f);
        fallCfg.fall_acceleration_upper_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Accel_Upper_Threshold", 6.0f);
        fallCfg.fall_acceleration_lower_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Accel_Lower_Threshold", -4.0f);
        fallCfg.bed_pixel_ratio_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Bed_Pixel_Ratio_Threshold", 0.15f);
        fallCfg.safe_area_ratio_threshold = (float)cfg.getFloat("FallDetect.Safe_Area_Ratio_Threshold", 0.5);
        fallCfg.fall_window_size = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 30);
        fallCfg.fall_duration = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 5);
        fallCfg.post_fall_distance_threshold = (float)cfg.getFloat("FallDetect.Fall_Detect_Post_Fall_Distance_Threshold", 4.0f);
        fallCfg.post_fall_check_frames = cfg.getInt("FallDetect.Fall_Detect_Post_Fall_Check_Frames", 5);
        fallCfg.momentum_calc_type = cfg.getInt("FallDetect.Fall_Detect_Momentum_Calc_Type", 1);
        
        fallCfg.enable_face_detection = false;//(cfg.getInt("FallDetect.Enable_Face_Detection", 1) != 0);
        fallCfg.face_detect_interval_frames = cfg.getInt("FallDetect.Face_Detect_Interval_Frames", 8);
        fallCfg.enable_edge_drop_filter = (cfg.getInt("FallDetect.Enable_Edge_Drop_Filter", 1) != 0);
        fallCfg.enable_bed_exit_verification = (cfg.getInt("FallDetect.Enable_Bed_Exit_Verification", 0) != 0);
        fallCfg.enable_block_shrink_verification = (cfg.getInt("FallDetect.Enable_Block_Shrink_Verification", 0) != 0);
        
        fallCfg.enable_save_bg_mask = (cfg.getInt("FallDetect.Enable_Save_BG_Mask", 1) != 0);
        fallCfg.bg_init_start_frame = cfg.getInt("FallDetect.BG_Init_Start_Frame", 2);
        fallCfg.bg_init_end_frame = cfg.getInt("FallDetect.BG_Init_End_Frame", 5);
        fallCfg.bg_diff_threshold = cfg.getInt("FallDetect.BG_Diff_Threshold", 18);
        fallCfg.bg_update_interval_frames = cfg.getInt("FallDetect.BG_Update_Interval", 8);
        fallCfg.bg_update_alpha = cfg.getFloat("FallDetect.BG_Update_Alpha", 0.1f);
        fallCfg.bed_update_alpha_multiplier = cfg.getFloat("FallDetect.Bed_Update_Alpha_Multiplier", 8.0f);
        fallCfg.enable_post_bed_exit_threshold = (cfg.getInt("FallDetect.Enable_Post_BedExit_Threshold", 1) != 0);
        fallCfg.post_bed_exit_threshold_multiplier = cfg.getFloat("FallDetect.Post_BedExit_Threshold_Multiplier", 0.7f);
        fallCfg.projection_use_foreground = (cfg.getInt("FallDetect.Projection_Use_Foreground", 0) != 0);

        // Missing parameters in verifier:
        fallCfg.opt_flow_frame_distance = cfg.getInt("OpticalFlow.CompareFrameDistance", 2);
        fallCfg.perspective_point_x = cfg.getInt("OpticalFlow.PerspectivePointX", 416);
        fallCfg.perspective_point_y = cfg.getInt("OpticalFlow.PerspectivePointY", 474);
        fallCfg.min_trigger_area = cfg.getInt("OpticalFlow.MinTriggerArea", 2000);
        fallCfg.enable_fall_and_bed_exit = true;
        
        sdk.SetConfig(&fallCfg);
        
        VisionSDK::ImageRelated_v1 imgCfg;
        imgCfg.header.type = VisionSDK::ConfigType::ImageRelated_v1;
        imgCfg.header.version = 1;
        imgCfg.enable_save_images = false;
        imgCfg.enable_draw_bg_noise = (appCfg.getInt("Demo.Demo_Draw_Background_Noise", 0) != 0);
        imgCfg.expected_frame_interval_ms = cfg.getInt("Validation.Expected_Frame_Interval", 33);
        imgCfg.frame_interval_tolerance_ms = cfg.getInt("Validation.Frame_Interval_Tolerance", 10);
        sdk.SetConfig(&imgCfg);
        
        // Load Bed region scaled if points are in original coordinate system
        if (!bed_points.empty()) {
            auto scaled_bed_points = bed_points;
            bool needsScale = false;
            for(auto& p : scaled_bed_points) {
                if (p.first >= W || p.second >= H) {
                    needsScale = true;
                    break;
                }
            }
            if (needsScale) {
                float scale_x = (float)W / orgW;
                float scale_y = (float)H / orgH;
                std::cout << "Scaling Bed region points: " << scale_x << "x" << scale_y << std::endl;
                for (auto& p : scaled_bed_points) {
                    p.first = (int)(p.first * scale_x);
                    p.second = (int)(p.second * scale_y);
                }
            }
            sdk.SetBedRegion(scaled_bed_points);
        }
        
        // Register Callback & Reset lists
        g_callback_events.clear();
        sdk.RegisterVisionSDKCallback(onFallDetected);
        
        // Open video stream & process in the SAME process
        bool run_success = true;
        try {
            std::vector<uint8_t> sdk_buffer(W * H * 1);
            sdk.SetInputMemory(sdk_buffer.data(), W, H, 1);
            
            // We tell VideoReader to open the video with target size WxH and output format "gray" directly.
            // FFmpeg will handle both the resizing and the grayscale conversion on the fly.
            VideoReader reader(video_abs_path, W, H, "gray");
            int frame_count = 0;
            while (reader.readFrame(sdk_buffer)) {
                sdk.ProcessNextFrame();
                frame_count++;
            }
            std::cout << "Processed " << frame_count << " frames (directly scaled to gray via FFmpeg)." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception reading video: " << e.what() << std::endl;
            run_success = false;
        }
        
        // Process results
        bool match_success = false;
        int closest_frame_diff = 99999;
        int matched_sdk_frame = -1;
        float matched_confidence = 0.0f;
        
        report << "Real Resolution : " << (real_res.first == 0 ? "Unknown" : std::to_string(real_res.first) + "x" + std::to_string(real_res.second)) << "\n";
        report << "SDK Callbacks   :\n";
        
        if (run_success) {
            if (g_callback_events.empty()) {
                report << "  - (No events detected by SDK)\n";
            } else {
                for (const auto& ev : g_callback_events) {
                    report << "  - Frame " << ev.frame_index << " (Confidence: " << ev.confidence << ")\n";
                    
                    int diff = std::abs(ev.frame_index - info.errorlog_frame);
                    if (diff < closest_frame_diff) {
                        closest_frame_diff = diff;
                        matched_sdk_frame = ev.frame_index;
                        matched_confidence = ev.confidence;
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
